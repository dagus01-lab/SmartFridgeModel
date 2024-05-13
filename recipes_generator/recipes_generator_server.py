from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

from recipes_generator.model.recipe_model import infer
app = Flask(__name__)

@app.route('/generate_recipe', methods=['POST'])
def process_data():
    # Get JSON data from request
    ingredients = request.data.decode('utf-8').strip()

    recipe = infer(ingredients, model, tokenizer, device)
    print(recipe)
    recipe_parts = recipe.split("\n")
    response = {}
    for part in recipe_parts:
        try:
            key, val = part.split(":")
        except Exception as e:
            key = part.split(":")[0]
            val = ""
        response[key.lower()] = val

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("../recipes_generation_model", ignore_mismatched_sizes=True).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("../tokenizer")
    app.run(debug=True,  host='0.0.0.0', port=8088)