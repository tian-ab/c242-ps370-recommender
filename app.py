from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

data = pd.read_csv('cleaned_recipes.csv') # dummy data for testing

nutritional_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                       'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']

def last_order(order_name):
    match = data[data['Name'] == order_name]
    if not match.empty:
        return match[nutritional_columns].iloc[0]
    else:
        print(f"'{order_name}' not found in the data.")
        return None

def food_recommender(input_food, user_allergens=[]):
    input_nutritional_data = last_order(input_food)
    if input_nutritional_data is None:
        return None

    scaler = MinMaxScaler()
    input_food_scaled = scaler.fit_transform([input_nutritional_data.values])  # Scale input food data

    if user_allergens:
        filtered_data = data[~data['Allergens'].str.contains('|'.join(user_allergens), na=False)]
    else:
        filtered_data = data

    filtered_food_data = scaler.transform(filtered_data[nutritional_columns])
    similarity_scores = cosine_similarity(input_food_scaled, filtered_food_data)
    top_indices = similarity_scores[0].argsort()[::-1][:10]
    recommendations = filtered_data.iloc[top_indices]

    relevant_columns = ['Name', 'Ingredients', 'Allergens'] + nutritional_columns
    return recommendations[relevant_columns].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API Endpoint to recommend food.
    Expects JSON input with 'food_name' and optional 'allergens'.
    """
    try:
        data = request.json
        food_name = data.get('food_name')
        allergens = data.get('allergens', [])

        if not food_name:
            return jsonify({"error": "Missing 'food_name' in request."}), 400

        recommendations = food_recommender(food_name, allergens)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)