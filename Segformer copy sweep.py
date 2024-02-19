from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image
import numpy as np
import wandb
import sys
import yaml

with open('./segformer_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)


DATA_DIR = '/home/mresham/fruitQuality/exports_seeds/Images'
MASK_DIR = '/home/mresham/fruitQuality/exports_seeds/Masks'
SEED_DATA = '/home/mresham/fruitQuality/exports_seeds/'
wandb_logger = WandbLogger(log_model=True, experiment=run)
MODEL_BASE = wandb.config.backbone
EPOCHS = wandb.config.epochs

data_ids = [mask_id.split(".png")[0]
            for mask_id in os.listdir(SEED_DATA + "Masks")]
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)  # Also Test size
CLASSES = list(reversed(['seed', 'pulp', 'albedo', 'flavedo']))

batch_size = 8
num_workers = 4


class SemanticSegmentationDataset(Dataset):
    CLASSES = ['unlabelled', 'seed', 'pulp', 'albedo', 'flavedo']

    def __init__(
            self,
            ids,
            images_dir,
            masks_dir,
            classes=None,
            feature_extractor=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = ids
        # self.images_fps = [os.path.join(images_dir, image_id.replace(".png", ""))
#                           for image_id in self.ids]
        self.images_fps = [os.path.join(
            SEED_DATA, "Images", f"{id_}.png") for id_ in ids]
        # self.masks_fps = [os.path.join(
        #    masks_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(
            SEED_DATA, "Masks", f"{id_}.png") for id_ in ids]
        self.meta_fps = [os.path.join(
            SEED_DATA, "Images", f"{id_}_meta.json") for id_ in ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]

        self.id2label = {}
        for i in range(len(classes)):
            self.id2label[self.class_values[i]] = classes[i]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.feature_extractor = feature_extractor

    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        segmentation_map = Image.open(self.masks_fps[i])
        # with open(self.meta_fps[i]) as f:
        #     meta_data = json.load(f)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt")
        encoded_inputs['labels'] //= 51
        encoded_inputs['labels'] -= 1
        # encoded_inputs['seeds'] = torch.tensor([meta_data['seedCount']])

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs

    def __len__(self):
        return len(self.ids)

    def get_image(self, i):
        return Image.open(self.images_fps[i])

    def get_mask(self, i):
        return Image.open(self.masks_fps[i])

    # def get_nseeds(self, i):
    #     with open(self.meta_fps[i]) as f:
    #         meta_data = json.load(f)
    #     return meta_data['seedCount']


# class SegformerForSemanticSegmentationAndImageClassification(SegformerPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.segformer = SegformerModel(config)

#         self.decode_head = SegformerDecodeHead(config)


#         self.num_labels = config.num_labels
#         # Classifier head
#         self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)


#         # Initialize weights and apply final processing
#         self.post_init()


#     def forward(
#         self,
#         pixel_values: torch.FloatTensor,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SemanticSegmenterOutput, SegFormerImageClassifierOutput]:

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )


#         outputs = self.segformer(
#             pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=True,  # we need the intermediate hidden states
#             return_dict=return_dict,
#         )


#         encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]


#         logits = self.decode_head(encoder_hidden_states)


#         # Image classification stuff
#         sequence_output = outputs[0]
#         # convert last hidden states to (batch_size, height*width, hidden_size)
#         batch_size = sequence_output.shape[0]


#         if self.config.reshape_last_stage:
#             # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
#             sequence_output = sequence_output.permute(0, 2, 3, 1)
#         sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])
#          # global average pooling
#         sequence_output = sequence_output.mean(dim=1)
#         cl_logits = self.classifier(sequence_output)
#         cl_loss = None
#         # Image classification stuff end


#         loss = None
#         if labels is not None:
#             # upsample logits to the images' original size
#             upsampled_logits = nn.functional.interpolate(
#                 logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
#             )
#             if self.config.num_labels > 1:
#                 loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
#                 loss = loss_fct(upsampled_logits, labels)
#             elif self.config.num_labels == 1:
#                 valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
#                 loss_fct = BCEWithLogitsLoss(reduction="none")
#                 loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
#                 loss = (loss * valid_mask).mean()
#             else:
#                 raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")


#             # Classification stuff
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"


#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)


#         if not return_dict:
#             if output_hidden_states:
#                 output = (logits,) + outputs[1:]
#             else:
#                 output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output


#         # Classification Stuff
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output


#         return SemanticSegmenterOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states if output_hidden_states else None,
#             attentions=outputs.attentions,
#         )
#         return SegFormerImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        id2label[0] = 'Unlabeled'
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        self.save_hyperparameters()

        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_BASE,
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return (outputs)

    def training_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            metrics = {
                'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

            for k, v in metrics.items():
                self.log(k, v)

            return (metrics)
        else:
            return ({'loss': loss})

    def validation_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        self.validation_step_outputs.append(loss)

        return ({'val_loss': loss})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_val_loss = torch.stack(outputs).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        val_per_category_iou = metrics['per_category_iou']

        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou,
                   "val_mean_accuracy": val_mean_accuracy}
        for i in self.id2label.keys():
            metrics[f"val_{self.id2label[i]}_iou"] = val_per_category_iou[i]

        for k, v in metrics.items():
            self.log(k, v)

        self.validation_step_outputs.clear()

        return metrics

    def test_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        self.test_step_outputs.append(loss)

        return ({'test_loss': loss})

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_test_loss = torch.stack(outputs).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]
        test_per_category_iou = metrics['per_category_iou']

        metrics = {"test_loss": avg_test_loss, "test_mean_iou": test_mean_iou,
                   "test_mean_accuracy": test_mean_accuracy}
        for i in self.id2label.keys():
            metrics[f"test_{self.id2label[i]}_iou"] = test_per_category_iou[i]

        for k, v in metrics.items():
            self.log(k, v)

        self.test_step_outputs.clear()

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    MODEL_BASE)
feature_extractor.reduce_labels = False
feature_extractor.size = 128


train_dataset = SemanticSegmentationDataset(
    data_ids[:TRAIN_SIZE],
    DATA_DIR,
    MASK_DIR,
    classes=CLASSES,
    feature_extractor=feature_extractor,
)
val_dataset = SemanticSegmentationDataset(
    data_ids[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE],
    DATA_DIR,
    MASK_DIR,
    classes=CLASSES,
    feature_extractor=feature_extractor,
)
test_dataset = SemanticSegmentationDataset(
    data_ids[TRAIN_SIZE+VAL_SIZE:],
    DATA_DIR,
    MASK_DIR,
    classes=CLASSES,
    feature_extractor=feature_extractor,
)


train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers)

segformer_finetuner = SegformerFinetuner(
    train_dataset.id2label,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    metrics_interval=10,
)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

trainer = pl.Trainer(
    accelerator='gpu',
    callbacks=[checkpoint_callback],
    max_epochs=EPOCHS,
    logger=wandb_logger
)
wandb_logger.experiment.config.update({"model": MODEL_BASE})

trainer.fit(segformer_finetuner)


res = trainer.test(ckpt_path="best")

dataset_image = test_dataset.get_image(0)
encoded_inputs = test_dataset[0]
images, masks = encoded_inputs['pixel_values'], encoded_inputs['labels']
outputs = segformer_finetuner.model(images.unsqueeze(0), masks.unsqueeze(0))

loss, logits = outputs[0], outputs[1]

# First, rescale logits to original image size
upsampled_logits = nn.functional.interpolate(logits,
                                             # (height, width)
                                             size=dataset_image.size[::-1],
                                             mode='bilinear',
                                             align_corners=False)

# Second, apply argmax on the class dimension
seg = upsampled_logits.argmax(dim=1).cpu().numpy()[0]

# wandb_image = wandb.Image(dataset_image, caption="Input image")
wandb_logger.log_image(
    "example_image", [dataset_image], caption=["Input image"])


# Change pixel order to be seed, pulp, albedo, flavedo, background
ground_truth_mask = (np.array(test_dataset.get_mask(0)) // 51)
tmp_gt_mask = ground_truth_mask + 10
# Dont change order here
tmp_gt_mask = np.where(tmp_gt_mask == 10, 4, tmp_gt_mask)
tmp_gt_mask = np.where(tmp_gt_mask == 14, 0, tmp_gt_mask)
tmp_gt_mask = np.where(tmp_gt_mask == 13, 1, tmp_gt_mask)
tmp_gt_mask = np.where(tmp_gt_mask == 12, 2, tmp_gt_mask)
tmp_gt_mask = np.where(tmp_gt_mask == 11, 3, tmp_gt_mask)
ground_truth_mask = tmp_gt_mask
wandb_logger.log_image("example_image", [dataset_image], caption=["True Masks"],
                       masks=[{
                           "ground_truth": {
                               "mask_data": ground_truth_mask.squeeze(),
                               "class_labels": {4: "Background",
                                                0: "Seed",
                                                1: "Pulp",
                                                2: "Albedo",
                                                3: "Flavedo"
                                                }
                           }}
])

# Change pixel order to be Background, seed, pulp, albedo, flavedo
# tmp_seg = seg + 1
# seg = np.where(tmp_seg == 5, 0, tmp_seg)
wandb_logger.log_image("example_image", [dataset_image], caption=["Predicted Masks"],
                       masks=[{
                           "predicted_mask": {
                               "mask_data": seg.squeeze(),
                               "class_labels": {
                                   0: "Seed",
                                   1: "Pulp",
                                   2: "Albedo",
                                   3: "Flavedo",
                                   4: "Background",
                               }
                           }}
])

# color_map = {
#     0: (0, 0, 0),
#     1: (255, 0, 0),
#     2: (0, 255, 0),
#     3: (0, 0, 255),
#     4: (120, 120, 0),
# }


# def prediction_to_vis(prediction):
#     vis_shape = prediction.shape + (3,)
#     vis = np.zeros(vis_shape)
#     for i, c in color_map.items():
#         vis[prediction == i] = color_map[i]
#     return Image.fromarray(vis.astype(np.uint8))


# for batch in test_dataloader:
#     images, masks = batch['pixel_values'], batch['labels']
#     outputs = segformer_finetuner.model(images, masks)

#     loss, logits = outputs[0], outputs[1]

#     upsampled_logits = nn.functional.interpolate(
#         logits,
#         size=masks.shape[-2:],
#         mode="bilinear",
#         align_corners=False
#     )
#     predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
#     masks = masks.cpu().numpy()


# n_plots = 4
# f, axarr = plt.subplots(n_plots, 2)
# f.set_figheight(15)
# f.set_figwidth(15)
# for i in range(n_plots):
#     axarr[i, 0].imshow(prediction_to_vis(predicted_mask[i, :, :]))
#     axarr[i, 1].imshow(prediction_to_vis(masks[i, :, :]))
