import os  # nopep8
os.environ['SM_FRAMEWORK'] = 'tf.keras'  # nopep8
from label_studio_ml.model import LabelStudioMLBase
import cv2
import keras
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import wandb
from label_studio_ml.utils import get_image_local_path
from label_studio_converter import brush
import random
import string
from uuid import uuid4


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
    x_center = abs(new_image_width - old_image_width) // 2
    y_center = abs(new_image_height - old_image_height) // 2

    # Return cropped image if orginal image was larger
    if old_image_height > new_image_height and old_image_width > new_image_width:
        result = image[old_image_height//2-height//2: old_image_height//2 +
                       height//2, old_image_width//2-width//2: old_image_width//2+width//2]
        return result

    # copy img image into center of result image
    result[y_center:y_center+old_image_height,
           x_center:x_center+old_image_width] = image

    return result


def image_to_original_shape(image, original_height, original_width):
    if len(image.shape) == 2:
        new_image_height, new_image_width = image.shape
    else:
        new_image_height, new_image_width, channels = image.shape

    color = (0)
    if len(image.shape) == 2:
        result = np.full((original_height, original_width),
                         color, dtype=np.uint8)
    else:
        result = np.full((original_height, original_width,
                         channels), color, dtype=np.uint8)

    # compute center offset
    x_center = abs(new_image_width - original_width) // 2
    y_center = abs(new_image_height - original_height) // 2

    # Return cropped image mask if orginal image was smaller
    if original_height < new_image_height and original_width < new_image_width:
        result = image[old_image_height//2-height//2: old_image_height//2 +
                       height//2, old_image_width//2-width//2: old_image_width//2+width//2]
        return result

    # copy img image into center of result image
    result[y_center:y_center+old_image_height,
           x_center:x_center+old_image_width] = image

    return result


def load_model():
    api = wandb.Api()
    artifact_path = 'mresham/fruit-quality/likely-sweep-41_model:v0'
    artifact = api.artifact(artifact_path)
    path = artifact.download()
    model = tf.keras.models.load_model(path, compile=False)
    optim = keras.optimizers.Adam(0.0001)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, total_loss, metrics)
    return model


class ImageSegmentationAPI(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ImageSegmentationAPI, self).__init__(**kwargs)

        self.model = load_model()
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]

    def predict(self, tasks, **kwargs):
        results = []
        predictions = []

        print(f"the kwargs are {kwargs}")
        print(f"the tasks are {tasks}")

        task = tasks[0]
        img_path = task["data"]["image"]

        image_path = get_image_local_path(img_path)
        print(image_path)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        image = tf.image.resize_with_crop_or_pad(image, 1024, 1024)

        preds = self.model.predict(image[tf.newaxis, ...])
        pred_mask = tf.math.argmax(preds, axis=-1)[0]
        labels = ['seed', 'pulp',  'albedo', 'flavedo', 'unlabelled']
        pred_mask = pred_mask.numpy()
        pred_mask = np.stack([np.where(pred_mask == v,
                                       np.ones((1024, 1024), dtype=np.int8),
                                       np.zeros((1024, 1024), dtype=np.int8)
                                       ) for v in range(5)],
                             axis=-1)
        # print(np.unique(pred_mask))
        pred_mask = np.multiply(pred_mask, 255)
        # print(np.unique(pred_mask))
        # pred_mask = np.stack([pred_mask[pred_mask == v] for v in range(5)],
        #                      axis=-1)
        # print(pred_mask.shape)

        pred_mask = tf.image.resize_with_crop_or_pad(pred_mask,
                                                     height,
                                                     width).numpy().astype('int')
        # print(pred_mask.shape)

        for i in range(len(labels)):
            mask = pred_mask[:, :, i]
            label = labels[i]
            if label == "unlabelled":
                continue
            # creates a random ID for your label everytime so no chance for errors
            label_id = ''.join(random.SystemRandom().choice(
                string.ascii_uppercase + string.ascii_lowercase + string.digits)),
            label_id = uuid4().hex
            # mask = tf.image.resize_with_crop_or_pad(np.expand_dims(mask),
            #                                         height,
            #                                         width)
            # print(mask.shape)
            # cv2.imwrite(f"tmp/outputs/dump_{label}.png", mask)
            rle = brush.mask2rle(mask.astype('int'))
            # tmp_img = brush.decode_rle(rle)
            # print(tmp_img.shape)
            # cv2.imshow(label, np.reshape(tmp_img, [height, width, 4])[:, :, 3])
            # cv2.waitKey(0)
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [label],
                },
                "type": "brushlabels",
                "id": label_id,
                "readonly": False,
            })

        predictions.append({"result": results})

        return predictions

    def fit(self, completions, **kwargs):
        pass
