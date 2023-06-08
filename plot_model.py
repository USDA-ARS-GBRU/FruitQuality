import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import wandb
import segmentation_models as sm
import tensorflow as tf
import keras


api = wandb.Api()
artifact = api.artifact('mresham/my-awesome-project/clean-cosmos-7_model:v0')
artifact.download()
path = "artifacts/clean-cosmos-7_model-v0"
model = tf.keras.models.load_model(path, compile=False)


optim = keras.optimizers.Adam(0.0001)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)
tf.keras.utils.plot_model(model, to_file="model_img.png", show_shapes=True)
