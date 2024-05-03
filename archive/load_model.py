import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import wandb
import segmentation_models as sm
import tensorflow as tf
import numpy as np
import keras
import cv2


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
        result = np.full((new_image_height, new_image_width),
                         color, dtype=np.uint8)
    else:
        result = np.full((new_image_height, new_image_width,
                         channels), color, dtype=np.uint8)

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
        self.images_fps = [os.path.join(images_dir, image_id)
                           for image_id in self.ids]
        self.masks_fps = [os.path.join(
            masks_dir, image_id)+'.png' for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]

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


api = wandb.Api()
artifact_path = 'mresham/fruit-quality/likely-sweep-41_model:v0'
artifact = api.artifact(artifact_path)
artifact.download()
path = "artifacts/" + artifact_path.split("/")[-1]
model = tf.keras.saving.load_model(path, compile=False)

DATA_DIR = './Images/'
MASK_DIR = './Masks/'
CLASSES = ['seed', 'pulp', 'albedo', 'flavedo']

data_ids = [image_id.replace(".png", "") for image_id in os.listdir(MASK_DIR)]
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)
test_dataset = Dataset(
    data_ids[TRAIN_SIZE+VAL_SIZE:],
    DATA_DIR,
    MASK_DIR,
    classes=CLASSES,
)
test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

optim = keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)
scores = model.evaluate(test_dataloader)
print(scores)
