## Naive Bayes Pipeline
- create_mask
  - Takes the label studio output file and segments flavedo, albedo and pulp into their separate images
- test_UP_run_script
  - This is the baseline Naive bayes model that runs on one image
  - Uses `pulp_naive_bayes_pdfs.txt` as the parameters for inference | It was generated uses `pcv-train.py` from `pcv` module previously, not present here
- naive_bayes
  - This file is a copy of `test_UP_run_script`. The modification is that it takes an image as an input and outputs the masked photos with the 20 pixel border removed
- naive_bayes_evaluate
  - This is the naive bayes pipeline that uses the scoring functions from `evaluate.py` to calculate metrics and then stores it in `naive_bayes_baseline.csv`
- evaluate
  - Calculates dice, iou and f score for 2 images

## Neural Network models
- train.py
  - Takes Raw Images from Images/ folder
  - Takes Masks from Masks/ folder
  - Runs a training pipeline using Raw images and Masks to create a model. Tracks all the checkpoint weights and training history logs on wandb
