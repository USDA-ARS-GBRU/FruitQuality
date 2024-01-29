# Import necessary libraries
import cv2
import json
import os
import numpy as np
from plantcv import plantcv as pcv
import pandas as pd
from label_studio_ml.utils import get_image_local_path
from pathlib import Path
import shutil
from label_studio_converter import brush


DEBUG = False
LABEL_EXPORT_DATA = "label exports\project-1-at-2023-08-14-17-00-bd7619d2.json"
IMAGES_DIR = "label exports\Images"
MASKS_DIR = "label exports\Masks"

with open(LABEL_EXPORT_DATA) as f:
    data = json.load(f)

for img_data in data[:]:
    if not img_data.get("annotation_id"):
        continue

    image_id = int(img_data.get("id"))
    image_url = img_data.get('image')
    seed_count = img_data.get("seedCount")[0].get("number")
    tags = img_data.get("tag")

    filepath = Path(get_image_local_path(image_url))
    filename = filepath.name
    print(f"Processing ID:{image_id} {filename}...")

    # Copy Images
    shutil.copyfile(filepath, os.path.join(IMAGES_DIR, f"{image_id:03d}.png"))


    # Make Masks
    width = None
    height = None
    masks = []
    for tag in tags:
        if tag.get('format') == "rle":
            width = tag.get("original_width")
            height = tag.get("original_height")
            label = tag.get("brushlabels")[0]
            rle = tag.get("rle")

            tmp_img = brush.decode_rle(rle)
            tmp_img = np.reshape(tmp_img, [height, width, 4])[:, :, 3]
            tmp_img[tmp_img >= 255//2] = 255
            tmp_img[tmp_img < 255//2] = 0

            if DEBUG:
                cv2.imshow(label, tmp_img)
                cv2.waitKey(0)
            masks.append((tmp_img, label))

    if width and height:
        mask_img = masks[0][0]
        base_color = 255 // 5
        color = {
            'seed': 1*base_color,
            'pulp': 2*base_color,
            'albedo': 3* base_color,
            'flavedo': 4*base_color
        }
        for mask in masks:
            placeholder = (mask[0] == 255)
            mask_img[placeholder] = mask[0][placeholder] * color[mask[1]]
            if DEBUG:
                cv2.imshow("mask", mask_img)
                cv2.waitKey(0)

        # Copy meta data
        with open(os.path.join(IMAGES_DIR, f"{image_id:03d}_meta.json"), "w") as f:
            json.dump({
                "id": image_id,
                "filename": filename,
                "width": width,
                "height": height,
                "seedCount": seed_count
            }, f)

        # Copy Mask
        cv2.imwrite(os.path.join(MASKS_DIR, f"{image_id:03d}.png"), mask_img)

        


