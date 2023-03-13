import os
import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
import segmentation_models as sm
import albumentations as A

wandb.init(project="my-awesome-project")
# wandb.init(project="fruit-quality")


DATA_DIR = './Images/'
MASK_DIR = './Masks/'


def visualize(rows=1, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(rows, n//rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def image_to_same_shape(image, height, width):
    if len(image.shape) == 2:
        old_image_height, old_image_width = image.shape
    else:
        old_image_height, old_image_width, channels = image.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = width
    new_image_height = height
    color = (0)
    if len(image.shape) == 2:
        result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)
    else:
        result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
          x_center:x_center+old_image_width] = image
    
    return result



class Dataset:
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unlabelled', 'seed', 'pulp', 'albedo', 'flavedo']
    
    def __init__(
            self, 
            ids,
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = ids
        # self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id)+'.png' for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        image = image_to_same_shape(image, 1024, 1024)
        mask = image_to_same_shape(mask, 1024, 1024)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v*(255//4)) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    


class_labels = {0: "Seed",
                1: "Pulp",
                2: "Albedo",
                3: "Flavedo",
                4: "Background"
               }


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


BACKBONE = 'efficientnetb0'
BATCH_SIZE = 1
CLASSES = ['seed', 'pulp', 'albedo', 'flavedo']
LR = 0.0001
EPOCHS = 2
ARCHITECTURE = sm.Linknet

preprocess_input = sm.get_preprocessing(BACKBONE)


wandb.config.update({"epochs": EPOCHS, "lr": LR, "backbone": BACKBONE, "architecture": "Linknet", "activation": "softmax"})


# define network parameters
n_classes = len(CLASSES) + 1
activation = 'softmax'

#create model
model = ARCHITECTURE(BACKBONE, classes=n_classes, activation=activation)


# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)


from wandb.keras import WandbEvalCallback, WandbMetricsLogger


class PlotSampleImages(keras.callbacks.Callback):
    def __init__(self):
        super(PlotSampleImages, self).__init__()
    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(np.expand_dims(dataset_image, axis=0))
        
        image = wandb.Image(dataset_image, 
                    masks={"predicted_mask": {
                            "mask_data": np.argmax(train_pred, axis=-1).squeeze(),
                            "class_labels": class_labels
                    }}
                    , caption="Masks")
        wandb.log({"progress_image": image})

# We list the ids from the mask_dir to make sure that we have data in dataset that have been labelled
data_ids = [image_id.replace(".png", "") for image_id in os.listdir(MASK_DIR)]
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)

# Dataset for train images
# train_dataset = dataset[:TRAIN_SIZE]
train_dataset = Dataset(
    data_ids[:TRAIN_SIZE],
    DATA_DIR, 
    MASK_DIR, 
    classes=CLASSES, 
)

# Dataset for validation images
# valid_dataset = dataset[TRAIN_SIZE:VAL_SIZE]
valid_dataset = Dataset(
    data_ids[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE],
    DATA_DIR, 
    MASK_DIR,
    classes=CLASSES, 
)

# Dataset for test images
# test_dataset = dataset[VAL_SIZE:]
test_dataset = Dataset(
    data_ids[TRAIN_SIZE+VAL_SIZE:],
    DATA_DIR, 
    MASK_DIR,
    classes=CLASSES, 
)

dataset_image, mask = test_dataset[0] # get some sample


image = wandb.Image(dataset_image, caption="Input image")
wandb.log({"example_image": image})


image = wandb.Image(dataset_image, 
                    masks={"ground_truth": {
                            "mask_data": np.argmax(mask, axis=-1).squeeze(),
                            "class_labels": class_labels
                    }}
                    , caption="Masks")
wandb.log({"example_image": image})

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    wandb.keras.WandbModelCheckpoint('./keras_checkpoints/best_model_{epoch:02d}', save_weights_only=True, save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(),
    wandb.keras.WandbMetricsLogger(),
    PlotSampleImages()
]


# train model
history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)


# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# test_dataset = Dataset(
#     DATA_DIR, 
#     MASK_DIR, 
#     classes=CLASSES, 
# )

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
# load best weights
# model.load_weights('best_model.h5') 
scores = model.evaluate(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
wandb.run.summary["loss"] = scores[0]
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))
    wandb.run.summary["test_"+metric.__name__] = value


image = wandb.Image(dataset_image, 
                    masks={"predicted_mask": {
                            "mask_data": np.argmax(model.predict(np.expand_dims(dataset_image, axis=0)), axis=-1).squeeze(),
                            "class_labels": class_labels
                    }}
                    , caption="Masks Predicted")
wandb.log({"example_image": image})


# n = 5
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)

# for i in ids:
    
#     image, gt_mask = test_dataset[i]
#     image = np.expand_dims(image, axis=0)
#     pr_mask = model.predict(image)
    
#     max_channel_mask = np.argmax(pr_mask, axis=-1)
#     sharper_seed_mask = np.zeros_like(max_channel_mask)
#     sharper_pulp_mask = np.zeros_like(max_channel_mask)
#     sharper_albedo_mask = np.zeros_like(max_channel_mask)
#     sharper_flavedo_mask = np.zeros_like(max_channel_mask)
#     sharper_background_mask = np.zeros_like(max_channel_mask)
#     sharper_seed_mask[max_channel_mask == 0] = 1
#     sharper_pulp_mask[max_channel_mask == 1] = 1
#     sharper_albedo_mask[max_channel_mask == 2] = 1
#     sharper_flavedo_mask[max_channel_mask == 3] = 1
#     sharper_background_mask[max_channel_mask == 4] = 1

#     visualize(
#         3,
#         image=image.squeeze(),
#         seed_mask=gt_mask[..., 0].squeeze(),
#         pulp_mask=gt_mask[..., 1].squeeze(),
#         albedo_mask=gt_mask[..., 2].squeeze(),
#         flavedo_mask=gt_mask[..., 3].squeeze(),
#         background_mask=gt_mask[..., 4].squeeze(),
#         image_dup=image.squeeze(),
#         pr_seed_mask=pr_mask[..., 0].squeeze(),
#         pr_pulp_mask=pr_mask[..., 1].squeeze(),
#         pr_albedo_mask=pr_mask[..., 2].squeeze(),
#         pr_flavedo_mask=pr_mask[..., 3].squeeze(),
#         pr_background_mask=pr_mask[..., 4].squeeze(),
#         image_dup_dup=image.squeeze(),
#         pr_seed_mask_dup=sharper_seed_mask.squeeze(),
#         pr_pulp_mask_dup=sharper_pulp_mask.squeeze(),
#         pr_albedo_mask_dup=sharper_albedo_mask.squeeze(),
#         pr_flavedo_mask_dup=sharper_flavedo_mask.squeeze(),
#         pr_background_mask_dup=sharper_background_mask.squeeze(),
#     )


wandb.finish()





