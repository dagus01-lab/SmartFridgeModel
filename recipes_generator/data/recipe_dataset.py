import numpy as np
import torch
from torch.utils.data import Dataset
import json
import re
import pandas as pd
import html

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

        return: recipe instance list
    '''
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
        ingredients_list = ""
        if not isinstance(ingredients, list):
            ingredients = [ingredients]
        if not isinstance(quantities, list):
            quantities = [quantities]
        n_ingre = min(len(quantities), len(ingredients))
        for i in range(n_ingre):
            #if not is_attr_not_empty(ingredients[i]):
            #    continue
            if not is_attr_not_empty(quantities[i]) and is_attr_not_empty(ingredients[i]):
                ingredients_list += ingredients[i]
            else:
                ingredients_list += str(quantities[i]) + " " + ingredients[i].strip()

            if i != n_ingre - 1:
                ingredients_list += ", "

        servings = row['RecipeServings'] if is_attr_not_empty(str(row['RecipeServings'])) else 1
        cook_time = row['CookTime'] if is_attr_not_empty(str(row['CookTime'])) else "PT0M"
        preparation_time = row['PrepTime'] if is_attr_not_empty(str(row['PrepTime'])) else "PT0M"
        total_time = row['TotalTime'] if is_attr_not_empty(str(row['TotalTime'])) else "PT0M"

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
                f"\nIngredients: {html.unescape(ingredients_list.strip())}" \
                f"\nServings: {str(servings)}" \
                f"\nInstructions: {html.unescape(instructions.strip())}" \
                f"\nCook time: {html.unescape(str(cook_time))}" \
                f"\nPreparation time: {html.unescape(str(preparation_time))}" \
                f"\nTotal time: {html.unescape(str(total_time))}<|endoftext|>"
        if len(recipe_instance) <= max_chars:
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
                              f"Ingredients: {ingredient.strip()}" \
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