import os
import json
import numpy as np
from collections import Counter, defaultdict
import torch.nn as nn


BASE_DIR = "exports_seeds/Images"
base_count = defaultdict(lambda: 0)
for file in os.listdir(BASE_DIR):
    if file.endswith(".json"):
        with open(f'{BASE_DIR}/{file}') as f:
            meta_data = json.load(f)
        # print(meta_data)
        base_count[meta_data['seedCount']] += 1
    # img = cv2.imread(f'{BASE_DIR}/{file}', cv2.IMREAD_GRAYSCALE)
    # array_img = np.array(img)
    # unique, counts = np.unique(array_img, return_counts=True)
    # for pixel_val, count in zip(unique, counts):
    #     base_count[pixel_val] += count
# base_count.sort()
print(base_count)
print(sorted(base_count.keys()))
total_count = sum(base_count.values())
print(total_count)
probab = {}
for k,v in base_count.items():
    probab[k] = 1 - (v/total_count)

print(probab)


# total_pixels = sum(base_count.values())

# print(total_pixels)

# proportions = [(x / total_pixels) for x in base_count.values()]

# print(proportions)

# weights = [1-x for x in proportions]
# print(weights)


def normalize_array(arr):
    arr_sum = np.sum(arr)
    normalized_arr = arr / arr_sum
    return normalized_arr

weights = np.ones(30) * 0.5
print(weights)
for k,v in probab.items():
    weights[k] = v

print(weights)
print(normalize_array(weights))
# weights[0] = 0.2
# print(normalize_array(weights))
