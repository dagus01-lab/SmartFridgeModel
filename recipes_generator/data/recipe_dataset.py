import csv
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
import json
import re
import pandas as pd
import html


def read_ingredients_file(file:str | Path):
    """
    Reads the CSV file and returns a pandas dataframe containing: ingredient,recipient_singular,recipient_plural.
    """
    ingredients = {}
    df = pd.read_csv(file, dtype=str)
    for index, row in df.iterrows():
        ingredients[row['ingredient'].lower().strip()] = {'recipient_singular': row['recipient_singular'].lower().strip(), 'recipient_plural': row['recipient_plural'].lower().strip()}
    return ingredients

def get_ingre_with_measure_unit(ingredient:str,ingredients_with_unit:dict, quantity: str | None):
    lower_ingredient = ingredient.strip().lower()
    match = [i for i in ingredients_with_unit.keys() if i in lower_ingredient]
    if quantity is not None:
        #many ingredients quantities are a range of values. The model learns better when it is given just a value. Here we give the first one
        if isinstance(quantity,str):
            quantity = quantity.split("-")[0].strip()
            first_quantity_nr = quantity.split(" ")[0].strip().replace('"','')
        else:
            first_quantity_nr = str(quantity)
        if first_quantity_nr.isnumeric() and int(first_quantity_nr) >= 10:
                #if the quantity is > 10 it is likely that the unit is grams
                recipient = "g "
        elif len(match) == 0:
            recipient = ""
        elif first_quantity_nr.isnumeric() and int(first_quantity_nr)> 1 and int(first_quantity_nr)<10:
                #if the quantity is > 1 and <10, take the plural recipient name
                recipient = f"{ingredients_with_unit[match[0]]['recipient_plural']} "
        else:
            recipient = f"{ingredients_with_unit[match[0]]['recipient_singular']} "

        if len(recipient) > 1 and recipient[:-1] in lower_ingredient:
            recipient = ""
        updated_ingredient = f"{quantity} {recipient}{ingredient}"

    else:
        updated_ingredient = ingredient
    return updated_ingredient

def is_attr_not_empty(attr: str):
    return pd.notna(attr) and attr.strip() != "NA" and attr.strip() != "N/A" and attr.strip().lower() != "nan" and not attr.isspace()

def load_preprocess_raw_csv_data(file, max_chars=2500):
    '''
        This method is aimed at preprocessing recipes
        from the following dataset:
        https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
        take raw recipe data and preprocess it,
        return a list of recipe instances

        parameter: raw data

        return: recipe instances list
    '''
    ingredients_with_measures = read_ingredients_file('recipes_generator/data/food_containers.csv')
    recipes = []
    df = pd.read_csv(file, dtype=str)
    for index, row in df.iterrows():
        title = row['Name']
        quantities = row['RecipeIngredientQuantities'][2:-1].split(",") if pd.notna(row["RecipeIngredientQuantities"]) else row["RecipeIngredientQuantities"]
        if isinstance(row['RecipeIngredientParts'], str) and is_attr_not_empty(row['RecipeIngredientParts']):
            ingredient_classes = row['RecipeIngredientParts'][2:-1]
        else:
            continue
        ingredients = ingredient_classes.split(",")
        if isinstance(row['RecipeInstructions'], str) and is_attr_not_empty(row['RecipeInstructions']):
            instructions = row['RecipeInstructions'][2:-1].split(",")
        else:
            continue
        instructions = "".join(instructions)
        ingredients_list = "\n"
        if not isinstance(ingredients, list):
            ingredients = [ingredients]
        if not isinstance(quantities, list):
            quantities = [quantities]

        #remove duplicate ingredients
        ingredients = list(set(ingredients))

        n_ingre = min(len(quantities), len(ingredients))
        for i in range(n_ingre):
            if not is_attr_not_empty(quantities[i]) and is_attr_not_empty(ingredients[i]):
                ingredients_list += f"- {get_ingre_with_measure_unit(ingredients[i], ingredients_with_measures, None)}\n"
            else:
                ingredients_list += f"- {get_ingre_with_measure_unit(ingredients[i].strip(), ingredients_with_measures, quantities[i])}\n"


        servings = row['RecipeServings'] if is_attr_not_empty(str(row['RecipeServings'])) \
            else row['RecipeYield'] if is_attr_not_empty(str(row['RecipeYield'])) else 1
        cook_time = row['CookTime'].replace("PT","").replace("H"," h ").replace("M"," min ") if is_attr_not_empty(str(row['CookTime'])) else ""
        preparation_time = row['PrepTime'].replace("PT","").replace("H", " h ").replace("M"," min ") if is_attr_not_empty(str(row['PrepTime'])) else ""
        total_time = row['TotalTime'].replace("PT","").replace("H"," h ").replace("M"," min ") if is_attr_not_empty(str(row['TotalTime'])) else ""

        ingredient_classes = ingredient_classes.replace('"', "")
        title = title.replace('"', "")
        ingredients_list = ingredients_list.replace('"', "")
        instructions = instructions.replace('"', "")

        ingredient_classes = ingredient_classes.replace("\\", "")
        title = title.replace("\\", "")
        ingredients_list = ingredients_list.replace("\\", "")
        instructions = instructions.replace("\\", "")

        recipe_instance = f"<|startoftext|>Prompt: {html.unescape(ingredient_classes.strip())}" \
                f"\nTitle: {html.unescape(title)}" \
                f"\nIngredients: \n{html.unescape(ingredients_list.strip())}" \
                f"\nServings: {str(servings)}" \
                f"\nInstructions: {html.unescape(instructions.strip())}"
        if cook_time != "":
            recipe_instance += f"\nCook time: {html.unescape(str(cook_time))}"
        if preparation_time != "":
            recipe_instance += f"\nPreparation time: {html.unescape(str(preparation_time))}"
        if total_time != "":
            recipe_instance += f"\nTotal time: {html.unescape(str(total_time))}"
        recipe_instance += "<|endoftext|>"

        if len(recipe_instance) <= max_chars and len(instructions.strip())>10:
            recipes.append(recipe_instance)
    return recipes

def load_preprocess_raw_json_data(raw_data, max_chars=2500):
    '''
    This method is aimed at preprocessing recipes
    from the following dataset: https://eightportions.com/datasets/Recipes/
    take raw recipe data and preprocess it,
    return a list of recipe instances

    parameter: raw data

    return: recipe instance list

    '''
    with open(raw_data, 'r') as f:
        raw_dict = json.load(f)
    f.close()

    raw_list = []
    for recipe in raw_dict.values():
        # try/except will filter out recipes that don't have title, ingredients or instructions
        try:
            title = recipe['title'].replace("ADVERTISEMENT", "")
            ingredient_list = recipe['ingredients']
            ingredients = ""
            ingredient_classes = ""
            for ingredient in ingredient_list:
                ingredient = ingredient.replace("ADVERTISEMENT", "")
                if ingredient != "":
                    ingredients += ingredient + ", "
                    ingredient = ingredient.split('/')[0].strip()
                    ingredient = ingredient.split(')')
                    if len(ingredient) == 1:
                        ingredient = ingredient[0].strip()
                    else:
                        ingredient = ingredient[1].strip()
                    # Use re.sub() to replace the number with an empty string
                    ingredient = re.sub(r'\d+', '', ingredient).strip()

                    if ingredient != "":
                        ingredient_classes += ingredient + ", "

            instructions = recipe['instructions'].replace("ADVERTISEMENT", "")
            recipe_instance = f"<|startoftext|>Prompt: {ingredient_classes.strip()}" \
                              f"Title: {title.strip()}" \
                              f"Ingredients: \n{ingredient.strip()}" \
                              f"Instructions: {instructions.strip()}<|endoftext|>"
            if len(recipe_instance) <= max_chars:
                raw_list.append(recipe_instance)

        except:
            continue
    return raw_list

#after downloading a recipe dataset, make sure to parse recipes from the training file
class RecipeDataset(Dataset):
    def __init__(self, txt_list: list, tokenizer, max_length=700):
        self.tokenizer = tokenizer
        self.txt_list = txt_list#np.array(txt_list) #if isinstance(txt_list, list) else txt_list
        self.max_length = max_length

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        encodings_dict = self.tokenizer(self.txt_list[idx], truncation=True, max_length=self.max_length, padding="max_length")
        return torch.tensor(encodings_dict['input_ids']), torch.tensor(encodings_dict['attention_mask'])