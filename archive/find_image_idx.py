import os

SEED_DATA = '/home/mresham/fruitQuality/exports_seeds/'

image_ids = [
    "281",
    "285",
    "291",
    "313",
    "322",
    "345",
    "347"
]
image_ids = [
    "147",
    "141",
    "165",
    "229",
    "234",
    "255",
    "325"
]

data_ids = ([mask_id.split(".png")[0]
            for mask_id in os.listdir(SEED_DATA + "Masks")])
SIZE = len(data_ids)
TRAIN_SIZE = int(0.6 * SIZE)
VAL_SIZE = int(0.2 * SIZE)  # Also Test size

test_ids = data_ids[TRAIN_SIZE+VAL_SIZE:]

for image_id in image_ids:
    print(image_id in test_ids)
    try:
        print(test_ids.index(image_id))
    except ValueError:
        print(-1)
print(sorted(test_ids))
