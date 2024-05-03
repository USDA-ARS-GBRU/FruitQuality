from sklearn.model_selection import KFold
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

DATA_DIR = '/home/mresham/fruitQuality/exports_seeds/Images'
MASK_DIR = '/home/mresham/fruitQuality/exports_seeds/Masks'
SEED_DATA = '/home/mresham/fruitQuality/exports_seeds/'

MODEL_BASE = "nvidia/segformer-b0-finetuned-ade-512-512"
PREPROCESS_SIZE = 128


data_ids = sorted([mask_id.split(".png")[0]
            for mask_id in os.listdir(SEED_DATA + "Masks")])
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)  # Also Test size
CLASSES = list(reversed(['seed', 'pulp', 'albedo', 'flavedo']))


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


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    MODEL_BASE)
feature_extractor.reduce_labels = False
feature_extractor.size = PREPROCESS_SIZE

full_dataset = SemanticSegmentationDataset(
    data_ids,
    DATA_DIR,
    MASK_DIR,
    classes=CLASSES,
    feature_extractor=feature_extractor,
)

folds = KFold(n_splits=2)

splits = folds.split(np.arange(len(data_ids)), data_ids)

for i, (train_set, test_set) in enumerate(splits):
    print(np.array([data_ids[x] for x in train_set]))
    print(np.array([data_ids[x] for x in test_set]))
    print()
