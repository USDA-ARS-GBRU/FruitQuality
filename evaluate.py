import numpy as np

def iou_score(img1, img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def f_score(img1, img2):
    true_positive = np.sum(np.logical_and(img1, img2))
    false_positive = np.sum(np.logical_and(img1, np.logical_not(img2)))
    false_negative = np.sum(np.logical_and(np.logical_not(img1), img2))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score

def dice_score(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())

if __name__ == '__main__':
    img1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    img2 = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]])

    print(iou_score(img1, img2))
    print(f_score(img1, img2))
    print(dice_score(img1, img2))
