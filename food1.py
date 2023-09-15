import pickle
import spacy
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

with open('indian_food_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('spacy_nlp.pkl', 'rb') as f:
    nlp = pickle.load(f)

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Define the Flask route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Define the Flask route for the prediction API
@app.route('/predict', methods=['POST'])

def predict():
    
    df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")
    # Get the input text from the POST reques

    cuisine_type = request.form['cuisine_type']
    recipe_time = request.form['recipe_time']
    input_text = preprocess_text(cuisine_type + " " + recipe_time)

    # Vectorize the input text using the loaded CountVectorizer
    X_input = vectorizer.transform([input_text])

    # Make the prediction using the loaded model
    probabilities = model.predict_proba(X_input)[0]
    top_five_indices = probabilities.argsort()[-5:][::-1]
    top_five_dish_names = model.classes_[top_five_indices]
    top_five_dishes_dict = df[df['TranslatedRecipeName'].isin(top_five_dish_names)].to_dict('records')
    top_five_dishes = pd.DataFrame(top_five_dishes_dict)

    # Render the results HTML template with the predicted dish names
    return render_template('result.html', dishes=top_five_dishes)


if __name__ == '__main__':
    app.run(debug=True)
