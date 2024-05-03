import os
import json
import numpy as np
from collections import Counter, defaultdict
import torch
import time

BASE_DIR = "exports_seeds/Images"
base_count = defaultdict(lambda: 0)
for file in os.listdir(BASE_DIR):
    if file.endswith(".json"):
        with open(f'{BASE_DIR}/{file}') as f:
            meta_data = json.load(f)
        # print(meta_data)
        base_count[meta_data['seedCount']] += 1

        seed_count = meta_data['seedCount']
        print(seed_count)
        one_hot = torch.nn.functional.one_hot(
            torch.Tensor([seed_count]).long(), num_classes=30)
        print(one_hot)
        print(one_hot.argmax())
        time.sleep(1)
    # img = cv2.imread(f'{BASE_DIR}/{file}', cv2.IMREAD_GRAYSCALE)
    # array_img = np.array(img)
    # unique, counts = np.unique(array_img, return_counts=True)
    # for pixel_val, count in zip(unique, counts):
    #     base_count[pixel_val] += count
# base_count.sort()
print(base_count)
print(sorted(base_count.keys()))

# total_pixels = sum(base_count.values())

# print(total_pixels)

# proportions = [(x / total_pixels) for x in base_count.values()]

# print(proportions)

# weights = [1-x for x in proportions]
# print(weights)


# def normalize_array(arr):
#     arr_sum = np.sum(arr)
#     normalized_arr = arr / arr_sum
#     return normalized_arr

# print(normalize_array(weights))
# weights[0] = 0.2
# print(normalize_array(weights))
