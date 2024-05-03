from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import transformers
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
from PIL import Image
import numpy as np
import wandb
import sys
import random
import yaml
from collections import defaultdict

from transformers.models.segformer.modeling_segformer import BCEWithLogitsLoss, CrossEntropyLoss, SegformerDecodeHead, SegformerModel, SegformerPreTrainedModel, SemanticSegmenterOutput, SegFormerImageClassifierOutput, MSELoss
from typing import Optional, Tuple, Union
from evaluate import load

with open('./mtl_segformer_weighted_normalized.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)


def normalize_array(arr):
    arr_sum = np.sum(arr)
    normalized_arr = arr / arr_sum
    return normalized_arr

DATA_DIR = '/home/mresham/fruitQuality/exports_seeds/Images'
MASK_DIR = '/home/mresham/fruitQuality/exports_seeds/Masks'
SEED_DATA = '/home/mresham/fruitQuality/exports_seeds/'
wandb_logger = WandbLogger(log_model=False, experiment=run)
MODEL_BASE = wandb.config.backbone 
EPOCHS = wandb.config.epochs
WEIGHT = wandb.config.weight
MASK_TYPE = wandb.config.mask_type
if WEIGHT == "None" or MASK_TYPE == "None":
    WEIGHTS = None
elif wandb.config.weights != "None":
    WEIGHTS = wandb.config.weights
else:
    MASK_ORDER = ['seed', 'pulp', 'albedo', 'flavedo']
    idx = MASK_ORDER.index(MASK_TYPE) + 1
    WEIGHTS = np.array([1] * 5)
    WEIGHTS[idx] = WEIGHT
    WEIGHTS = normalize_array(WEIGHTS)
    WEIGHTS = torch.tensor(WEIGHTS, dtype=torch.float32).to(
        device="cuda") if WEIGHTS != "None" else None
LOSS = wandb.config.loss_fct
PREPROCESS_SIZE = wandb.config.preprocess_size

data_ids = ([mask_id.split(".png")[0]
            for mask_id in os.listdir(SEED_DATA + "Masks")])
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)  # Also Test size
CLASSES = list(reversed(['seed', 'pulp', 'albedo', 'flavedo']))

batch_size = 8
num_workers = 4
# Unsorted Image IDs
image_ids = [
    "147",
    "141",
    "165",
    "229",
    "234",
    "255",
    "325"
]
# # Sorted Image IDs
# image_ids = [
#     "281",
#     "285",
#     "291",
#     "313",
#     "322",
#     "345",
#     "347"
# ]


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
        self.ids: list = ids
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
        with open(self.meta_fps[i]) as f:
            meta_data = json.load(f)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt")
        encoded_inputs['labels'] //= 51
        encoded_inputs['labels'] -= 1

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        encoded_inputs['seedCount'] = torch.tensor([meta_data['seedCount']])

        return encoded_inputs

    def __len__(self):
        return len(self.ids)
    
    def get_idx_of_img_id(self, img_id):
        try:
            return self.ids.index(img_id)
        except ValueError:
            return -1

    def get_image(self, i):
        return Image.open(self.images_fps[i])

    def get_mask(self, i):
        return Image.open(self.masks_fps[i])

    def get_nseeds(self, i):
        with open(self.meta_fps[i]) as f:
            meta_data = json.load(f)
        return meta_data['seedCount']


class FocalLoss_MulticlassDiceLoss(nn.Module):
    """Multi-class Focal loss implementation.
    Args:
        gamma (float): The larger the gamma, the smaller
            the loss weight of easier samples.
        weight (float): A manual rescaling weight given to each
            class.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, num_classes, softmax_dim=None, gamma=2, weight=None, ignore_index=-100, loss="multi"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

        self.num_classes = num_classes
        self.softmax_dim = softmax_dim
        self.loss_fct = loss

    def forward(self, input, target, reduction='mean', smooth=1e-6):
        if self.loss_fct == "multi" or self.loss_fct == "focal":
            # Focal Loss
            logit = F.log_softmax(input, dim=1)
            pt = torch.exp(logit)
            logit = (1 - pt)**self.gamma * logit
            focal_loss = F.nll_loss(
                logit, target, self.weight, ignore_index=self.ignore_index)

        if self.loss_fct == "multi" or self.loss_fct == "dice":
            targets_one_hot = F.one_hot(
                target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            true_1_hot = targets_one_hot
            probas = F.softmax(input, dim=1)
            true_1_hot = true_1_hot.type(input.type())
            dims = (0,) + tuple(range(2, target.ndimension()))
            intersection = torch.sum(probas * true_1_hot, dims)
            cardinality = torch.sum(probas + true_1_hot, dims)
            dice_coefficient = (2. * intersection /
                                (cardinality + smooth)).mean()
            dice_loss = -dice_coefficient.log()

        total_loss = 0

        if self.loss_fct == "multi" or self.loss_fct == "focal":
            total_loss = focal_loss
        if self.loss_fct == "multi" or self.loss_fct == "dice":
            total_loss += dice_loss

        return total_loss


class SegformerWithDiceLossForSemanticSegmentationAndImageClassification(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        self.num_labels = config.num_labels

        # Classifier head
        self.clf_num_labels = 1
        self.classifier = nn.Sequential(nn.Linear(config.hidden_sizes[-1], 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.clf_num_labels))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        seed_count=None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput, SegFormerImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        # Image classification stuff
        sequence_output = outputs[0]
        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]

        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(
            batch_size, -1, self.config.hidden_sizes[-1])
        # global average pooling
        sequence_output = sequence_output.mean(dim=1)
        cl_logits = self.classifier(sequence_output)
        cl_loss = None
        # Image classification stuff end

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                if LOSS == 'cross-entropy':
                    loss_fct = CrossEntropyLoss(
                        ignore_index=self.config.semantic_loss_ignore_index)
                else:
                    loss_fct = FocalLoss_MulticlassDiceLoss(
                        num_classes=5, weight=WEIGHTS, loss=LOSS)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (
                    labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
            else:
                raise ValueError(
                    f"Number of labels should be >=0: {self.config.num_labels}")
            
            # Classification stuff
            if self.config.problem_type is None:
                if self.clf_num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.clf_num_labels > 1 and (seed_count.dtype == torch.long or seed_count.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.clf_num_labels == 1:
                    cl_loss = loss_fct(cl_logits.squeeze(),
                                       seed_count.squeeze().float())
                else:
                    cl_loss = loss_fct(cl_logits, seed_count)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cl_loss = loss_fct(
                    cl_logits.view(-1, self.clf_num_labels), seed_count.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                cl_loss = loss_fct(cl_logits, seed_count)
            # Image classification stuff end

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            seg_return = ((loss,) + output) if loss is not None else output

            # Classification Stuff
            cl_output = (cl_logits,) + outputs[1:]
            cl_return = (
                (cl_loss,) + cl_output) if loss is not None else cl_output

            return (seg_return, cl_return)

        return (SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        ), SegFormerImageClassifierOutput(
            loss=cl_loss,
            logits=cl_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ))


class MTLSegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(MTLSegformerFinetuner, self).__init__()
        id2label[0] = 'Unlabeled'
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        self.save_hyperparameters()

        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = SegformerWithDiceLossForSemanticSegmentationAndImageClassification.from_pretrained(
            MODEL_BASE,
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load("mean_iou")
        self.val_mean_iou = load("mean_iou")
        self.test_mean_iou = load("mean_iou")

        self.train_mse = load("mse")
        self.val_mse = load("mse")
        self.test_mse = load("mse")

        self.test_mae = load("mae")
        self.test_corr = load("pearsonr")

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.gt_pixel_density = defaultdict(lambda: 0)
        self.pred_pixel_density = defaultdict(lambda: 0)

    def forward(self, images, masks, seed_count):
        seg_outputs, clf_outputs = self.model(
            pixel_values=images, labels=masks, seed_count=seed_count)
        return (seg_outputs, clf_outputs)

    def training_step(self, batch, batch_nb):

        images, masks, seed_count = batch['pixel_values'], batch['labels'], batch['seedCount']

        seg_outputs, clf_outputs = self(images, masks, seed_count)      

        loss, logits = seg_outputs[0], seg_outputs[1]
        clf_loss, clf_logits = clf_outputs[0], clf_outputs[1]
        total_loss = torch.add(loss, clf_loss)

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
        self.train_mse.add_batch(
            predictions=clf_logits.detach().cpu().numpy(),
            references=seed_count.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )
            mse = self.train_mse.compute()['mse']

            metrics = {'loss': total_loss,
                       "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"], "mse": mse}

            for k, v in metrics.items():
                self.log(k, v, prog_bar=True)

            return (metrics)
        else:
            return ({'loss': total_loss})

    def validation_step(self, batch, batch_nb):

        images, masks, seed_count = batch['pixel_values'], batch['labels'], batch['seedCount']

        seg_outputs, clf_outputs = self(images, masks, seed_count)

        loss, logits = seg_outputs[0], seg_outputs[1]
        clf_loss, clf_logits = clf_outputs[0], clf_outputs[1]
        total_loss = torch.add(loss, clf_loss)

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
        self.val_mse.add_batch(
            predictions=clf_logits.detach().cpu().numpy(),
            references=seed_count.detach().cpu().numpy()
        )

        self.validation_step_outputs.append(total_loss)

        return ({'val_loss': total_loss})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )
        mse = self.val_mse.compute()['mse']

        avg_val_loss = torch.stack(outputs).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        val_per_category_iou = metrics['per_category_iou']

        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou,
                   "val_mean_accuracy": val_mean_accuracy, "val_mean_mse": mse}
        for i in self.id2label.keys():
            metrics[f"val_{self.id2label[i]}_iou"] = val_per_category_iou[i]

        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        self.validation_step_outputs.clear()

        return metrics

    def test_step(self, batch, batch_nb):

        images, masks, seed_count = batch['pixel_values'], batch['labels'], batch['seedCount']

        seg_outputs, clf_outputs = self(images, masks, seed_count)

        loss, logits = seg_outputs[0], seg_outputs[1]
        clf_loss, clf_logits = clf_outputs[0], clf_outputs[1]
        total_loss = torch.add(loss, clf_loss)

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
        self.test_mse.add_batch(
            predictions=clf_logits.detach().cpu().numpy(),
            references=seed_count.detach().cpu().numpy()
        )
        self.test_mae.add_batch(
            predictions=clf_logits.detach().cpu().numpy(),
            references=seed_count.detach().cpu().numpy()
        )
        self.test_corr.add_batch(
            predictions=clf_logits.detach().cpu().numpy(),
            references=seed_count.detach().cpu().numpy()
        )
        # Ground truth pixel density
        pixel_labels = masks.detach().cpu().numpy()
        unique, counts = np.unique(pixel_labels, return_counts=True)
        for pixel_val, count in zip(unique, counts):
            self.gt_pixel_density[pixel_val] += count
        # Predicted pixel density
        pixel_labels = predicted.detach().cpu().numpy()
        unique, counts = np.unique(pixel_labels, return_counts=True)
        for pixel_val, count in zip(unique, counts):
            self.pred_pixel_density[pixel_val] += count

        self.test_step_outputs.append(total_loss)

        return ({'test_loss': total_loss})

    def get_seed_to_fruit_ratio(self, pixel_density: dict, idx: int):
        try:
            return pixel_density.get(idx) / sum(pixel_density.values())
        except TypeError as e:
            print(e)
            print(pixel_density, idx)
            return None

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
        mse = self.test_mse.compute()['mse']
        mae = self.test_mae.compute()['mae']
        corr = self.test_corr.compute()['pearsonr']
        # GT {0: Background, 4: Seed, 3: Pulp, 2: Albedo, 1: Flavedo}
        gt_seed_to_fruit_ratio = self.get_seed_to_fruit_ratio(
            self.gt_pixel_density, 4)
        # Pred {4: Background, 0: Seed, 1: Pulp, 2: Albedo, 3: Flavedo}
        pred_seed_to_fruit_ratio = self.get_seed_to_fruit_ratio(
            self.pred_pixel_density, 0)

        metrics = {"test_loss": avg_test_loss, "test_mean_iou": test_mean_iou,
                   "test_mean_accuracy": test_mean_accuracy, "test_mean_mse": mse,
                   "test_mean_mae": mae, "test_mean_corr": corr,
                   "test_gt_seed_to_fruit_ratio": gt_seed_to_fruit_ratio,
                   "test_pred_seed_to_fruit_ratio": pred_seed_to_fruit_ratio}
        for i in self.id2label.keys():
            metrics[f"test_{self.id2label[i]}_iou"] = test_per_category_iou[i]

        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        self.test_step_outputs.clear()
        self.gt_pixel_density.clear()
        self.pred_pixel_density.clear()

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
feature_extractor.size = PREPROCESS_SIZE


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

mtl_segformer_finetuner = MTLSegformerFinetuner(
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
    logger=wandb_logger,
    val_check_interval=len(train_dataloader),
    log_every_n_steps=1
)
wandb_logger.experiment.config.update({"model": MODEL_BASE})

trainer.fit(mtl_segformer_finetuner)


res = trainer.test(ckpt_path="best")
mtl_segformer_finetuner = mtl_segformer_finetuner.to(device="cuda")

def run_sample_output(img_id):
    idx_in_test_set = test_dataset.get_idx_of_img_id(img_id)
    if idx_in_test_set == -1:
        return None
    dataset_image = test_dataset.get_image(idx_in_test_set)
    encoded_inputs = test_dataset[idx_in_test_set]
    images, masks, seed_count = encoded_inputs['pixel_values'], encoded_inputs['labels'], encoded_inputs['seedCount']
    # outputs = segformer_finetuner.model(images.unsqueeze(0), masks.unsqueeze(0))
    images, masks, seed_count = images.to(device="cuda"), masks.to(device="cuda"), seed_count.to(device="cuda")
    seg_outputs, clf_outputs = mtl_segformer_finetuner.model(
        images.unsqueeze(0), masks.unsqueeze(0), seed_count.unsqueeze(0))

    _, logits = seg_outputs[0], seg_outputs[1]
    _, clf_logits = clf_outputs[0], clf_outputs[1]

    # Seed Count outputs 
    my_data = [img_id, 
               str(seed_count.detach().cpu().numpy()),
               str(clf_logits.detach().cpu().numpy())]

    # Mask outptus
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
        img_id, [dataset_image], caption=["Input image"])


    # Change pixel order to be seed, pulp, albedo, flavedo, background
    ground_truth_mask = (np.array(test_dataset.get_mask(idx_in_test_set)) // 51)
    tmp_gt_mask = ground_truth_mask + 10
    # Dont change order here
    tmp_gt_mask = np.where(tmp_gt_mask == 10, 4, tmp_gt_mask)
    tmp_gt_mask = np.where(tmp_gt_mask == 14, 0, tmp_gt_mask)
    tmp_gt_mask = np.where(tmp_gt_mask == 13, 1, tmp_gt_mask)
    tmp_gt_mask = np.where(tmp_gt_mask == 12, 2, tmp_gt_mask)
    tmp_gt_mask = np.where(tmp_gt_mask == 11, 3, tmp_gt_mask)
    ground_truth_mask = tmp_gt_mask
    wandb_logger.log_image(img_id, [dataset_image], caption=["True Masks"],
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
    wandb_logger.log_image(img_id, [dataset_image], caption=["Predicted Masks"],
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

    return my_data

columns = ["image id", "seed label", "seed prediction"]
seed_count_data = []
for img_id in image_ids:
    seed_data = run_sample_output(img_id)
    if seed_data:
        seed_count_data.append(seed_data)

# using columns and data
wandb_logger.log_text(key="Seed Count", columns=columns, data=seed_count_data)
