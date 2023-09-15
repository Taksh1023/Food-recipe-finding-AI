import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy


df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):

    doc = nlp(text)

    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    

    return " ".join(tokens)


df["preprocessed_text"] = (df["Cuisine"] + " " + df["TotalTimeInMins"].astype(str)).apply(preprocess_text)

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["preprocessed_text"])


model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(X, df["TranslatedRecipeName"])

# Save the trained model as a pickle file
with open('indian_food_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the CountVectorizer as a pickle file
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the spaCy nlp object as a pickle file
with open('spacy_nlp.pkl', 'wb') as f:
    pickle.dump(nlp, f)
