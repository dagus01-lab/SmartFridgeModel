import re

from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

from recipes_generator.model.recipe_model import generate_recipe
app = Flask(__name__)

# Define a function to parse the generated text
def parse_recipe(text):
    recipe_parts = {
        "title": "",
        "ingredients": "",
        "instructions": "",
        "servings": "",
        "cooking time": "",
        "preparation time": "",
        "total time": ""
    }

    # Regular expressions to match each part of the recipe
    regex_patterns = {
        "title": r"Title:\s*(.*)",
        "ingredients": r"Ingredients:\s*((?:.|\n)*?)(?=\n[A-Z])",
        "instructions": r"Instructions:\s*((?:.|\n)*?)(?=\n[A-Z])",
        "servings": r"Servings:\s*(\d+|.*)",
        "cooking time": r"Cooking time:\s*(.*)",
        "preparation time": r"Preparation time:\s*(.*)",
        "total time": r"Total time:\s*(.*)"
    }

    for key, pattern in regex_patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            recipe_parts[key] = match.group(1).strip()

    return recipe_parts

@app.route('/generate_recipe', methods=['POST'])
def process_data():
    # Get JSON data from request
    ingredients = request.data.decode('utf-8').strip().replace('"', '')

    recipe = generate_recipe(ingredients, model, tokenizer, device)
    print(recipe)
    parsed_recipe = parse_recipe(recipe)
    print(parsed_recipe)

    # Return the response as JSON
    return jsonify(parsed_recipe)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("../recipes_generator/model/recipes_generation_model", ignore_mismatched_sizes=True).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("../recipes_generator/model/recipes_generation_model")
    app.run(debug=False,  host='0.0.0.0', port=8088) #debug=True for debugging mode