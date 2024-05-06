# Baseline Algorithm for the PANORAMA challenge: PDAC detection in contrast enhanced CT scan
This repository contains the baseline algorithm for the PANORAMA challenge. Model weights were uploaded to zenodo and can be downloaded using this link:
The algorithm is based on the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) and consists of a two-step approach for pancreatic ductal adenocarcinoma detection on contrast-enhanced CT scans (CECT). First, a low-resolution nnU-Net model with Dice loss is trained to segment the pancreas. Based on this automatic segmentation a volume of interest (VOI) around the pancreas region is cropped from the original input CECT. The cropped VOIs are used to train a high-resolution nnU-Net algorithm for PDAC detection using the cross-entropy loss. This process is summarized in Figure 1.

## Model training
Training was performed according to the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) instructions.
Both nnU-Net algorithms were trained using 5-fold cross-validation. The data split per fold can be found
### Low resolution pancreas segmentation network
