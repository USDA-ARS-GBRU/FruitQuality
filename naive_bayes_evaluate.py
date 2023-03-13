from naive_bayes import naive_bayes
import cv2
import os
from evaluate import dice_score, f_score, iou_score
import pandas as pd

IMAGE_FOLDER = "Images"
IMAGE_PROCESSED_FOLDER = "Images_Processed"

def run(image_filename):
    if not os.path.exists(f"{IMAGE_PROCESSED_FOLDER}\\flavedo\\"+image_filename):
        return 0
    print(f"Evaluating {image_filename}...")
    flavedo_ground_truth = cv2.imread(f"{IMAGE_PROCESSED_FOLDER}\\flavedo\\"+image_filename)
    albedo_ground_truth  = cv2.imread(f"{IMAGE_PROCESSED_FOLDER}\\albedo\\"+image_filename)
    pulp_ground_truth  = cv2.imread(f"{IMAGE_PROCESSED_FOLDER}\\pulp\\"+image_filename)

    filename, flavedo, albedo, pulp = naive_bayes(IMAGE_FOLDER+"\\"+image_filename, "out.json")

    scores = []

    for x in [
                ("flavedo", flavedo, flavedo_ground_truth),
                ("albedo", albedo, albedo_ground_truth),
                ("pulp", pulp, pulp_ground_truth)]:
        name, predicted, ground_truth = x
        scores.append([
                        image_filename,
                        name, 
                        dice_score(predicted, ground_truth), 
                        iou_score(predicted, ground_truth), 
                        f_score(predicted, ground_truth)])

    return scores

if __name__ == '__main__':
    out_scores = []
    for image in os.listdir(IMAGE_FOLDER):
        scores = run(image)
        if scores == 0:
            continue
        out_scores += scores

    df= pd.DataFrame(out_scores, columns=['Image', 'Name', 'Dice Score', 'IOU score', 'F-score'])
    print(df.tail())
    df.to_csv("naive_bayes_baseline.csv", index=False)