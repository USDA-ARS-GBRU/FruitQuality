import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
import wandb

path = "keras_checkpoints/best_model_111"
# path = "test_model/test_model/best_model_108"
BACKBONE = 'efficientnetb6'

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Linknet(BACKBONE, input_shape=(1024,1024,3),
                   encoder_weights=None, 
                   classes=5, activation='softmax')
optim = keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)

# print(model.summary())
model.load_weights(f"{path}")


model.save("test_model/full_model", by_name=True)


