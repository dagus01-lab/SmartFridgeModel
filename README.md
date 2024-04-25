# AI Chef Model

## Overview
This repository contains the programs used to train the machine 
learning models for our final project of  Digital Systems at the 
Computer Engineering Master's Degree program  of the University 
of Bologna. The project consists of an Android application aimed 
at reducing food waste by utilizing AI to suggest recipes based 
on the contents of users' refrigerators or pantries. 
You can find the repository of the android application
[here](https://github.com/bryanber102/SmartFridge)


## About the project

The primary objective of this project is to reduce food waste by 
leveraging artificial intelligence to recommend recipes based on 
the ingredients available in users' refrigerators or pantries. 
To achieve this goal, two models have been developed:
1. A food detection model: This model identifies and categorizes 
the food items present in the user's inventory.
2. A recipe recommendation model: Using the ingredients detected 
by the food detection model, this model suggests recipes that users 
can prepare with the available ingredients.

## Table of Contents

- [Food Detector](#FoodDetector)
    - [Dataset](#FoodDetectorDataset)
    - [Model](#FoodDetectorModel)
    - [Training](#FoodDetectorTraining)
- [Recipes Generator](#RecipesGenerator)
    - [Dataset](#RecipesGeneratorDataset)
    - [Model](#RecipesGeneratorModel)
    - [Training](#RecipesGeneratorTraining)
- [Usage](#Usage)

## FoodDetector

The food detector model was obtained by fine-tuning YOLOv8

### FoodDetectorDataset

The [initial dataset](https://universe.roboflow.com/karel-cornelis-q2qqg/aicook-lcv4d?ref=blog.roboflow.com)
was extended with some photos taken from our fridges and pantries
that we annotated using roboflow. You can find the final dataset
[here](https://app.roboflow.com/fridge-detection/smart-fridge-2uqsi).
Some augmentations were applied to the training input, namely: 
- random noise added to the images
- random placed black squares
- random rotation
- random horizontal/vertical flip
- random adjustments to brightness

### FoodDetectorModel

After some attempts we found out that the model that achieved the highest
performance was the nano version of YOLOv8. So we used the code from the 
[original repository](https://github.com/ultralytics/ultralytics)
for training and validation of the detection model, but applied some refactors 
to it, and added the first two transformations mentioned above in the
data preprocessing logic, which allowed us to achieve even better model
performance on new images.

### FoodDetectorTraining

Optimal results were achieved by using the following 
[configuration](https://github.com/dagus01-lab/AIChefModel/detector/cfg/default.yaml).
The model is the finally exported in tflite format.

## RecipesGenerator

The recipes generator model was obtained by fine-tuning GPT-2

### RecipesGeneratorDataset

The final dataset was obtained by combining two datasets: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
and https://eightportions.com/datasets/Recipes/.
The recipe texts underwent preprocessing, after which they were consolidated 
into a unified dataset. Each recipe entry comprises: a prompt (containing the list of 
ingredients available), a title, the necessary ingredients with quantities,
number of servings, instructions, and preparation time.

### RecipesGeneratorModel

The best model for our needs is GPT-2. We imported it using the 
[huggingface library](https://github.com/huggingface/transformers), 
albeit with a custom training function.
We opted for the distilled version of GPT-3 to ensure compatibility 
with Android mobile devices, which possess significant limitations 
in handling large models.

### RecipesGeneratorTraining
The model was trained using the loss function provided by huggingface.
Due to GPU memory constraints, a batch size of 8 was utilized, generating 
recipes with a maximum of 700 tokens.
Initially, we experimented with training the full GPT model and then 
applying knowledge distillation to DistilGPT-2. However, superior results 
were achieved by directly training DistilGPT-2 on the dataset. 
The model was ultimately exported in .tflite format, employing 
float16 quantization to further reduce its size.

## Usage

1. Clone the repository to your local machine.
2. Install software requirements:
`pip install requirements.txt`
3. Download the datasets for the detection model and for the recipe generator
to the folders of you choice.
4. You can find the Jupyter notebooks in the root directory of the project,
the code of the detector in the detector directory and the code of the 
recipe generator in the recipe generation directory
5. Follow the instructions provided in the notebooks to run the code, 
preprocess the data, train the models, and evaluate their performance.
