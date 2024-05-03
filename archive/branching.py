import shutil  # nopep8
import os  # nopep8
os.environ['SM_FRAMEWORK'] = 'tf.keras'  # nopep8

import yaml
import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
import wandb

from keras_applications import get_submodules_from_kwargs

from segmentation_models.models._common_blocks import Conv2dBn
from segmentation_models.models._utils import freeze_model
from segmentation_models.backbones.backbones_factory import Backbones
from segmentation_models.models.linknet import Conv3x3BnReLU, Conv1x1BnReLU


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}


def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }
