import os
import cv2
import numpy as np
from collections import Counter, defaultdict


BASE_DIR = "exports_seeds/Masks"
base_count = defaultdict(lambda: 0)
for file in os.listdir(BASE_DIR):
    img = cv2.imread(f'{BASE_DIR}/{file}', cv2.IMREAD_GRAYSCALE)
    array_img = np.array(img)
    unique, counts = np.unique(array_img, return_counts=True)
    for pixel_val, count in zip(unique, counts):
        base_count[pixel_val] += count

print(base_count)

total_pixels = sum(base_count.values())

print(total_pixels)

proportions = [(x / total_pixels) for x in base_count.values()]

print(proportions)

weights = [1-x for x in proportions]
print(weights)


def normalize_array(arr):
    arr_sum = np.sum(arr)
    normalized_arr = arr / arr_sum
    return normalized_arr

print(normalize_array(weights))
weights[0] = 0.2
print(normalize_array(weights))
