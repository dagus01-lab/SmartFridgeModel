import argparse
import csv
import glob
import math
import sys
import time

import cv2
import yaml
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import os
from tqdm import tqdm
MAX_PRODUCTS = 50
MAX_PRODUCTS_PER_TRAY = 20
MIN_PROD_HEIGHT = 50
MIN_PROD_WIDTH = 50
MAX_PROD_HEIGHT = 150
MAX_OVERLAP_RETRIES = 2

class IllegalBboxException(Exception):
    def __init__(self, message="Illegal value"):
        self.message = message
        super().__init__(self.message)


def not_overlaps(bbox, bboxes, overlap_degree:float=.8) -> bool:
    """Verifies that the found bounding box does not overlap the bounding boxes of other ingredients in the image"""
    for b in bboxes:
        x_min1 = bbox[0]
        x_max1 = bbox[1]
        y_min1 = bbox[2]
        y_max1 = bbox[3]
        area2 = b[2]*b[3]
        x_min2 = (b[0] * 2 - b[2]) // 2
        x_max2 = (b[0] * 2 + b[2]) // 2
        y_min2 = (b[1] * 2 - b[3]) // 2
        y_max2 = (b[1] * 2 + b[3]) // 2
        #if the squares completely overlap, return false
        if x_min1<=x_min2 and x_max1>=x_max2 and y_min1<=y_min2 and y_max1>=y_max2:
            return False
        #if the two squares intersect, check the overlapping area
        if (((x_min1 < x_min2 < x_max1 < x_max2) or (x_min2 < x_min1 < x_max2 < x_max1))
                and ((y_min1 < y_min2 < y_max1 < y_max2) or (y_min2 < y_min1 < y_max2 < y_max1))):
            #the two bboxes intersect on a corner
            hr = y_max1 - y_min2 if y_max2 > y_max1 else y_max2 - y_min1
            wr = x_max1 - x_min2 if x_max2 > x_max1 else x_max2 - x_min1

        elif y_min1<x_min2<x_max2<y_max1 and x_min2<x_min1<x_max1<x_max2:
            hr = y_max2 - y_min2
            wr = x_max1 - x_min1
        elif y_min2<y_min1<y_max1<y_max2 and x_min1<x_min2<x_max2<x_max1:
            hr = y_max1 - y_min1
            wr = x_max2 - x_min2
        # the two bboxes intersect on the left or right side
        elif x_min1 < x_min2 < x_max1 < x_max2 and y_min1 <= y_min2 <= y_max2 <= y_max1:
            hr = y_max2 - y_min2
            wr = x_max1 - x_min2
        elif x_min1 < x_min2 < x_max1 < x_max2 and y_min2 <= y_min1 <= y_max1 <= y_max2:
            hr = y_max1 - y_min1
            wr = x_max1 - x_min2
        elif x_min2 < x_min1 < x_max2 < x_max1 and y_min1 <= y_min2 <= y_max2 <= y_max1:
            hr = y_max2 - y_min2
            wr = x_max2 - x_min1
        elif x_min2 < x_min1 < x_max2 < x_max1 and y_min2 <= y_min1 <= y_max1 <= y_max2:
            hr = y_max1 - y_min1
            wr = x_max2 - x_min1
        # the two bboxes intersect on the upper or lower side
        elif y_min1 < y_min2 < y_max1 < y_max2 and x_min1 <= x_min2 <= x_max2 <= x_max1:
            hr = y_max1 - y_min2
            wr = x_max2 - x_min2
        elif y_min1 < y_min2 < y_max1 < y_max2 and x_min2 <= x_min1 <= x_max1 <= x_max2:
            hr = y_max1 - y_min2
            wr = x_max1 - x_min1
        elif y_min2 < y_min1 < y_max2 < y_max1 and x_min1 <= x_min2 <= x_max2 <= x_max1:
            hr = y_max2 - y_min1
            wr = x_max2 - x_min2
        elif y_min2 < y_min1 < y_max2 < y_max1 and x_min2 <= x_min1 <= x_max1 <= x_max2:
            hr = y_max2 - y_min1
            wr = x_max1 - x_min1
        else:
            continue
        overlapping_area = hr * wr
        # return false if the area of the already existing product image is being covered by the new image
        if area2 * overlap_degree < overlapping_area:
            return False
    return True


def find_product_bbox(image:Image) -> tuple[int, int, int, int]:
    """Gets the bounding box of the ingredient from the passed image"""
    # Convert the image to numpy array
    image_array = np.array(image)

    # Find non-white pixels
    non_white_indices = np.where(image_array != 0)

    # Get the minimum and maximum coordinates of non-white pixels
    min_x = np.min(non_white_indices[1])
    max_x = np.max(non_white_indices[1])
    min_y = np.min(non_white_indices[0])
    max_y = np.max(non_white_indices[0])

    # Return the bounding box coordinates in xyxy format
    return min_x, max_x, min_y, max_y


def remove_background(image_path: str | Path) -> tuple[Image, tuple:int]:
    """
    This function is aimed at adding to the alpha channel of the ingredient image
    :param image_path: path to ingredient image
    :return: the modified image and the bounding boxes coordinates in the xywh format
    """
    # Open the image
    image = Image.open(image_path)

    # Convert to grayscale
    grayscale_image = image.convert("L")

    # Apply edge detection
    edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
    # Apply median filter blur to highlight background
    blurred_image = grayscale_image.filter(ImageFilter.MedianFilter(size=7))
    # Combine the two images to highlight both edges and background
    highlighted_image = Image.blend(edge_image, blurred_image, alpha=0.5)

    # Convert to binary image using a threshold
    threshold = 125
    binary_image = highlighted_image.point(lambda p: 255 if p > threshold else 0)

    # Invert the binary image
    inverted_image = Image.eval(binary_image, lambda p: 255 - p)

    # Apply the inverted binary image as a mask
    image.putalpha(inverted_image)
    min_x, max_x, min_y, max_y = find_product_bbox(image)

    return image, min_x, max_x, min_y, max_y


def paste_ingredient(initial_image: Image, ingredient_image_path: str | Path, valid_range: tuple, bboxes: list[tuple[float]], attached_to_bottom=True) -> Image:
    """
    This function pastes an ingredient to a container image. It makes the ingredient image background invisible by
    pasting the ingredient image to the container image
    :param initial_image: initial image
    :param ingredient_image_path: path to ingredient image to paste
    :param valid_range: range of valid position where the ingredient can be pasted
    :param bboxes: list of bounding boxes
    :param attached_to_bottom: if true, the ingredient will be pasted directly on the vertical lowest point, otherwise it will be pasted in a random vertical position
    in the valid range
    :return: the initial image with the pasted intgredient
    """
    # Remove background from the ingredient image using edge detection
    l, r = valid_range[0]
    u, d = valid_range[1]
    ingredient_image, min_x, max_x, min_y, max_y = remove_background(ingredient_image_path)
    ah = MIN_PROD_HEIGHT
    bh = min(MAX_PROD_HEIGHT, d-u)
    ah, bh = (ah, bh) if ah < bh else (bh, ah)
    aw = MIN_PROD_WIDTH
    bw = (r-l)//3
    aw, bw = (aw, bw) if aw < bw else (bw, aw)
    h = random.randint(ah, bh)
    w = random.randint(aw, bw)
    min_x = (min_x * w) // ingredient_image.size[0]
    max_x = (max_x * w) // ingredient_image.size[0]
    min_y = (min_y * h) // ingredient_image.size[1]
    max_y = (max_y * h) // ingredient_image.size[1]
    ingredient_image = ingredient_image.resize((w, h))

    if attached_to_bottom:
        x = random.randint(l, r-ingredient_image.size[0]) #if l<r-ingredient_image.size[0] else l
        y = d-ingredient_image.size[1]
    else:
        x = random.randint(l, r - ingredient_image.size[0])
        y = random.randint(u, d - ingredient_image.size[1])

    if not not_overlaps((min_x, max_x, min_y, max_y), bboxes):
        raise IllegalBboxException("Bboxes cannot overlap")
    # Paste the ingredient image onto the initial image
    initial_image.paste(ingredient_image, (x, y), ingredient_image)

    return initial_image, min_x+x, max_x+x, min_y+y, max_y+y


def read_dimensions(file_path: str | Path) -> list[int]:
    """
    This function reads the legal dimensions range in container images for pasting ingredients.
    The following format is used: [trays_number, min_x, min_y, max_x, max_y]
    :param file_path: dimensions file
    :return: legal dimensions in a list
    """
    numbers = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming each row contains four numbers separated by a comma
            for number_str in row:
                numbers.append(int(number_str.strip()))  # Convert string to float
    return numbers


def generate_synthetic_image(base_containers_path: str | Path, base_ingredients_path: str | Path, draw_bboxes= False) -> Image:
    """
    This function generates a synthetic image, by taking some random ingredients from the available ones and
    pasting them to a container image
    :param base_containers_path: directory of containers images
    :param base_ingredients_path: directory of ingredients images
    :param draw_bboxes: if true saves the image with bboxes. Default value is False
    :return: a synthetic image
    """
    ingredients = os.listdir(base_ingredients_path)
    labels = []
    bboxes = []
    containers = []
    containers.extend(glob.glob(os.path.join(base_containers_path, '*.jpg')))
    containers.extend(glob.glob(os.path.join(base_containers_path, '*.png')))
    container = random.sample(containers, 1)[0]
    container_img = Image.open(container)
    container_annotation_path = container.split(".")[0]+".txt"
    trays, l, u, r, d = read_dimensions(container_annotation_path)
    ingredients_number = random.randint(0, MAX_PRODUCTS_PER_TRAY)
    if trays == 0:
        ingredients_tray = random.sample(ingredients, ingredients_number)
        for ingredient in ingredients_tray:
            ingre_path = random.sample(os.listdir(base_ingredients_path+ingredient+"/"), 1)[0]
            valid_range = ((l, r), (u, d))
            for _ in range(MAX_OVERLAP_RETRIES):
                try:
                    container_img, min_x, max_x, min_y, max_y = (
                                        paste_ingredient(
                                            container_img, base_ingredients_path+ingredient+"/"+ingre_path,
                                            valid_range, bboxes, attached_to_bottom=False
                                        )
                    )
                    bbox = ((min_x+max_x)//2, (min_y+max_y)//2, max_x-min_x, max_y-min_y)
                    bboxes.append(bbox)
                    labels.append(classes.index(ingredient))
                    break
                except IllegalBboxException as e:
                    pass

    for i in range(trays):
        height0, height1 = u+((d-u)//trays)*i, u+((d-u)//trays)*(i+1)
        valid_range = ((l, r), (height0, height1))
        max_ingre_per_tray = min(MAX_PRODUCTS_PER_TRAY, ingredients_number)
        ingre_per_tray = random.randint(0, max_ingre_per_tray)
        ingredients_tray = random.sample(ingredients, ingre_per_tray)
        for ingredient in ingredients_tray:
            for _ in range(MAX_OVERLAP_RETRIES):
                try:
                    ingre_path = random.sample(os.listdir(base_ingredients_path+ingredient+"/"), 1)[0]
                    container_img, min_x, max_x, min_y, max_y = paste_ingredient(container_img, base_ingredients_path+ingredient+"/"+ingre_path, valid_range, bboxes)
                    bbox = ((min_x + max_x) // 2, (min_y + max_y) // 2, max_x - min_x, max_y - min_y)
                    bboxes.append(bbox)
                    labels.append(classes.index(ingredient))
                    ingredients_number-=1
                    break
                except IllegalBboxException as e:
                    pass

    draw = ImageDraw.Draw(container_img)
    if draw_bboxes:
        for bbox in bboxes:
            x_min = (bbox[0]*2 - bbox[2])//2
            x_max = (bbox[0]*2 + bbox[2])//2
            y_min = (bbox[1]*2 - bbox[3])//2
            y_max = (bbox[1]*2 + bbox[3])//2
            draw.rectangle((x_min, x_max, y_min, y_max), outline="red")
    bboxes = [ (bbox[0]/container_img.size[0], bbox[1]/container_img.size[1], bbox[2]/container_img.size[0], bbox[3]/container_img.size[1]) for bbox in bboxes]
    return container_img, labels, bboxes


def save_labels(labels: list[str], bboxes: list[list[float] | tuple[float]], labels_file_path: str | Path) -> None:
    """
    This function saves the labels of an image to the specified location
    :param labels: list of labels in the image
    :param bboxes: list of bounding boxes
    :param labels_file_path: labels file directory
    """
    with open(labels_file_path, 'w') as file:
        for i in range(len(bboxes)):
            file.write(f"{labels[i]} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}\n")


def save_config_file(synthetic_dataset_path: str | Path, classes: list[str], file_path: str | Path) -> None:
    """
    This function saves the synthetic dataset configuration
    :param synthetic_dataset_path: path to synthetic dataset
    :param classes: list of classes names
    :param file_path: path to synthetic dataset configuration file
    """
    data= {
        "train": synthetic_dataset_path+"train/images",
        "test": synthetic_dataset_path+"test/images",
        "val": synthetic_dataset_path+"valid/images",
        "nc": len(classes),
        "names": classes
    }
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, default_style="'")

if __name__ == "__main__":
    """
        This script is aimed at creating a synthetic dataset by pasting some products images
        to images that represent usual containers in the kitchen (eg: tables, refrigerators, 
        pantries, etc).
        When invoking the script users should pass the configuration file as a parameter.
        The configuration file should contain:
        - base_container_path: path to the containers images
        - base_products_path: path to the ingredients images
        - perc: ratio of synthetic images over authentic images in the dataset
        - synthetic_dataset_path: path to synthetic dataset
        - original_dataset_path: path to original dataset
        - original_classes_path: original dataset configuration file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", type=str)

    args = parser.parse_args()
    config_file = args.config_file
    try:
        with open(config_file, 'r') as file:
            synthetic_data = yaml.safe_load(file)
            base_containers_path = synthetic_data["base_container_path"]
            base_ingredients_path = synthetic_data["base_products_path"]
            perc = synthetic_data["perc"]
            synthetic_dataset_path = synthetic_data["synthetic_dataset_path"]
            original_dataset_path = synthetic_data["original_dataset_path"]
            original_classes_path = synthetic_data["original_classes_path"]
    except OSError as e:
        print(f"Configuration file {config_file} not found")
        sys.exit(1)
    try:
        with open(original_classes_path, 'r') as file:
            data = yaml.safe_load(file)
            classes = data['names']
            for ingre in os.listdir(base_ingredients_path):
                classes.append(ingre)
    except OSError as e:
        print(f"Configuration file {config_file} not found")
        sys.exit(1)

    splits = ["train", "valid", "test"]
    for s in splits:
        bar_desc = "Processing "+s+" dataset files"
        original_images = os.listdir(original_dataset_path+s+"/images")
        synthetic_images_nr = math.floor(len(original_images) * perc)
        num_files = len(original_images)+synthetic_images_nr
        progress_bar = tqdm(total=num_files, desc=bar_desc, unit='file')
        if s not in os.listdir(synthetic_dataset_path):
            os.mkdir(synthetic_dataset_path+s)
        if "labels" not in os.listdir(synthetic_dataset_path+s):
            os.mkdir(synthetic_dataset_path+s+"/labels")
        if "images" not in os.listdir(synthetic_dataset_path+s):
            os.mkdir(synthetic_dataset_path+s+"/images")
        for image in original_images:
            shutil.copy(original_dataset_path+s+"/images/"+image, synthetic_dataset_path+s+"/images")
            progress_bar.update()
        original_labels = os.listdir(original_dataset_path+s+"/labels")
        for label in original_labels:
            shutil.copy(original_dataset_path+s+"/labels/"+label, synthetic_dataset_path+s+"/labels")
        if not os.path.exists(synthetic_dataset_path+s):
            os.mkdir(synthetic_dataset_path+s)
        base_image_path = synthetic_dataset_path+s+"/images"
        if not os.path.exists(base_image_path):
            os.mkdir(base_image_path)
        base_label_path = synthetic_dataset_path+s+"/labels"
        if not os.path.exists(base_label_path):
            os.mkdir(base_label_path)
        for i in range(synthetic_images_nr):
            final_img, labels, bboxes = generate_synthetic_image(base_containers_path=base_containers_path, base_ingredients_path=base_ingredients_path)
            final_img.save(f"{base_image_path}/img0000000{i}.jpg", "JPEG")
            save_labels(labels, bboxes, f"{base_label_path}/img0000000{i}.txt")
            progress_bar.update()

        progress_bar.close()

    save_config_file(synthetic_dataset_path, classes, "synthetic_data.yaml")
