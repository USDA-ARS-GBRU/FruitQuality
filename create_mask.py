# Import necessary libraries
import cv2
import json
import os
import numpy as np
from plantcv import plantcv as pcv
import pandas as pd

DEBUG = False
LABEL_EXPORT_DATA = "label exports\\project-1-at-2023-03-07-09-36-2e5228b0.json"

with open(LABEL_EXPORT_DATA) as f:
    data = json.load(f)

seed_data = []

for img_data in data:
    filename = img_data.get('file_upload')
    ftp_idx = filename.find("FTP")
    filename = filename[ftp_idx:]
    filename = filename.replace("Fruit_Quality_Fruit","Fruit Quality Fruit")
    print(f"Processing {filename}...")
    

    annotations = img_data.get('annotations')
    mask_points = {
        'seed':[],
        'pulp':[],
        'albedo':[],
        'flavedo':[]
    }
    for annotation in annotations[0].get("result"):
        w = annotation.get("original_width")
        h = annotation.get("original_height")
        points = annotation.get("value").get("points")
        label = annotation.get("value").get("polygonlabels")[0]
       
        points = points * np.float32([w/100, h/100])
      
        mask_points[label].append(points)

    # Seed Count data
    seed_data.append((filename, len(mask_points['seed'])))

    # Read an image
    img_path = f"Images/{filename}"
    img = cv2.imread(img_path)

    height, width, _ = img.shape
    
    # Initialise masks
    seed_mask = np.zeros((height, width), dtype=np.int32)
    pulp_mask = np.zeros((height, width), dtype=np.int32)
    albedo_mask = np.zeros((height, width), dtype=np.int32)
    flavedo_mask = np.zeros((height, width), dtype=np.int32)

    # Create masks
    for sm in mask_points['seed']:
        cv2.fillPoly(seed_mask, pts=np.int32([sm]), color=(255, 255, 255))
    for pm in mask_points['pulp']:
        cv2.fillPoly(pulp_mask, pts=np.int32([pm]), color=(255, 255, 255))
    for am in mask_points['albedo']:
        cv2.fillPoly(albedo_mask, pts=np.int32([am]), color=(255, 255, 255))
    for fm in mask_points['flavedo']:
        cv2.fillPoly(flavedo_mask, pts=np.int32([fm]), color=(255, 255, 255))

    # Removing seed mask from pulp mask
    pulp_mask = pcv.image_subtract(pulp_mask, seed_mask)
    # Removing pulp from albedo mask
    albedo_mask = pcv.image_subtract(pcv.image_subtract(albedo_mask, seed_mask), pulp_mask)
    # Removing albedo from pulp mask
    flavedo_mask = pcv.image_subtract(pcv.image_subtract(pcv.image_subtract(flavedo_mask, seed_mask), pulp_mask), albedo_mask)

    # Apply masks on image
    seed_masked = pcv.apply_mask(img=img, mask=seed_mask, mask_color='black')
    pulp_masked = pcv.apply_mask(img=img, mask=pulp_mask, mask_color='black')
    albedo_masked = pcv.apply_mask(img=img, mask=albedo_mask, mask_color='black')
    flavedo_masked = pcv.apply_mask(img=img, mask=flavedo_mask, mask_color='black')

    # Create combined mask
    combined_masks = [seed_mask, pulp_mask, albedo_mask, flavedo_mask]
    mask_image = np.zeros((height, width), dtype=np.uint16)
    color1 = np.array((1), dtype=np.uint8)
    color2 = np.array((2), dtype=np.uint8) 
    color3 = np.array((3), dtype=np.uint8) 
    color4 = np.array((4), dtype=np.uint8) 
    colors = [color1, color2, color3, color4]
    for i, mask in enumerate(combined_masks):
        mask_image[mask==255] = colors[i]

    mask_image = mask_image.astype(np.uint8)*(255//len(colors))

    if DEBUG:
        # Displaying the image
        cv2.imshow(filename, img)
        cv2.imshow("Seed", seed_masked)
        cv2.imshow("Pulp", pulp_masked)
        cv2.imshow("Albedo", albedo_masked)
        cv2.imshow("Flavedo", flavedo_masked)
        cv2.imshow("Mask", mask_image)

        # wait for the user to press any key to
        # exit window
        cv2.waitKey(0)

        # Closing all open windows
        cv2.destroyAllWindows()

    # Saving segmented images 
    out_path = "Images_Processed\\"
    cv2.imwrite(out_path+f"seed\\{filename}", seed_masked)
    cv2.imwrite(out_path+f"pulp\\{filename}", pulp_masked)
    cv2.imwrite(out_path+f"albedo\\{filename}", albedo_masked)
    cv2.imwrite(out_path+f"flavedo\\{filename}", flavedo_masked)

    # Saving combined segment images
    mask_path = "Masks\\"
    cv2.imwrite(mask_path+filename+'.png', mask_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Export seed count data
pd.DataFrame(seed_data, columns=("Filename", "Count")).to_csv("seed_data.csv", index=False)
