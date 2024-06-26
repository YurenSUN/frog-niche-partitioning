# frog-niche-partitioning

This repo includes codes for ML for frog detection in the niche partitioning paper.

The link to the model that we used for the paper will be included here after paper acceptance.

## Citation
To add after paper acceptance.

## Contents
`audio` folder: the CSV label files on the training audio created with Raven Pro. These are less useful as the original audio files are confidential, and we provide them here for records.

Python scripts: the utility files with functions to support the jupyter notebooks
- `classification_utils.py`: the util functions for training and classification.
- `dataset_process_utils.py`: the util functions to process the data.

Jupyter notebooks: the pipelines for training and detections.
- `dataset_creation.ipynb`: the pipeline to pre-process the audio data and create the training data set
- `cnn_training.ipynb`: the pipeline to train the CNN model
- `detection_pipeline.ipynb`: the pipeline to load the Jasper Ridge audio and detect the species in audio.
The jupyter notebooks could be run with Google Colab, which has free GPU quota.

## Audio

The audio files are retrieved from California Herps, AmphibiaWeb, FonoZoo, and Jasper Ridge.
Due to copyright and data usage policies, only labels are included in the audio folder,
and the related audios are not included in this repo. If needed, users need to follow the
copyright and data usage policies from those sites to download the necessary data.
