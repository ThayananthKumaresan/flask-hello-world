from flask import Flask, request, jsonify, render_template

# Data Analysis
import pandas as pd
import numpy as np


# Text Processing
import nltk
import re


# Machine Learning packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

#Sentiment packaages
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

from scipy.sparse import hstack

def cleaning_text(text):
    print('Cleaning text...')
    # Remove links
    text = re.sub(r'https?:\/\/.*?[\s+]', '', text.replace("|", " ") + " ")

    # Emoticon conversion
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    text = re.sub(r"\b{}{}[)pD]+\b|\b[(d]+{}{}\b".format(eyes, nose, nose, eyes), " SMILEFACE ", text)
    text = re.sub(r"\b{}{}p+\b".format(eyes, nose), " LOLFACE ", text)
    text = re.sub(r"\b{}{}\(+\b|\b\)+{}{}\b".format(eyes, nose, nose, eyes), " SADFACE ", text)
    text = re.sub(r"\b{}{}[|]\b".format(eyes, nose), " NEUTRALFACE ", text)

    # Number conversion
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " NUMBER123 ", text)

    # Remove multiple full stops
    text = re.sub(r'[^\w\s]', '', text)

    # Remove Non-words
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Removing words that are 1 to 2 characters long
    text = re.sub(r'\b\w{1,2}\b', '', text)

    # Remove very short or long words
    text = re.sub(r'(\b\w{0,3})?\b', '', text)
    text = re.sub(r'(\b\w{30,1000})?\b', '', text)

    # Remove punctuations
    text = re.sub(re.compile(r"[^a-z\s]"), " ", text)

    return text


def extract_emotion(post):
    print('extract_emotion...')
    
    # Create an NRCLex object for the post
    text_object = NRCLex(post)
    
    # Get emotion counts
    emotion_counts = text_object.affect_frequencies
    
    # Convert emotion counts to a DataFrame
    emotion_df = pd.DataFrame(emotion_counts, index=[0])
    
    return emotion_df

import pickle

with open('word_tf_vectorizer.pkl', 'rb') as file:
    word_tf_vectorizer = pickle.load(file)
with open('word_tfidf_vectorizer.pkl', 'rb') as file:
    word_tfidf_vectorizer = pickle.load(file)


def extract_features_from_text(user_input):
    print('extract_features_from_text...')
    
    # 1. Word-level TF
    word_tf= word_tf_vectorizer.transform([user_input])

    # 2. Word-level TFIDF    
    word_tfidf=  word_tfidf_vectorizer.transform([user_input])

    # 3. Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(user_input)['compound']

    # 4. Emotion
    text_object = NRCLex(user_input)
    emotion_counts = text_object.affect_frequencies
    emotion_df = pd.DataFrame(emotion_counts, index=[0])

    # Combine all features into a single DataFrame
    features_df = pd.concat([pd.DataFrame(word_tf.toarray(), index=[0], columns=word_tf_vectorizer.get_feature_names_out()),
                            pd.DataFrame(word_tfidf.toarray(), index=[0], columns=word_tfidf_vectorizer.get_feature_names_out()),
                            pd.DataFrame({'sentiment': [sentiment_score]}),
                            emotion_df], axis=1)

    return features_df



def prep (text):
    print('prep...')

    text_cleaned= cleaning_text(text)
    extracted_features = extract_features_from_text(text_cleaned)
    return extracted_features



def trace_back(combined):
    type_list = [
        {"0": "I", "1": "E"},
        {"0": "N", "1": "S"},
        {"0": "F", "1": "T"},
        {"0": "P", "1": "J"},
    ]
    result = []
    for num in combined:
        s = ""
        for i in range(len(num)):
            s += type_list[i][num[i]]
        result.append(s)
    return result


def combine_classes(y_pred1, y_pred2, y_pred3, y_pred4):
    combined = []
    for i in range(len(y_pred1)):
        combined.append(
            str(y_pred1[i]) + str(y_pred2[i]) + str(y_pred3[i]) + str(y_pred4[i])
        )
    result = trace_back(combined)
    return result[0]



import joblib

personality_types = ['E','N', 'T', 'J']

def predict(text):
    print('Predicting...')

    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    sentiment = {
        "pos_sentiment": score["pos"],
        "neg_sentiment": score["neg"],
        "neu_sentiment": score["neu"],
    }

    emotion_df = extract_emotion(text)

    pred=[]
    features = prep(text)
    print('Going into model...')
    for personality_type in personality_types:
    # # Load individual models
    #     loaded_gb_model = joblib.load(f'{personality_type}_gb_model.joblib')
    #     loaded_xgb_model = joblib.load(f'{personality_type}_xgb_model.joblib')
    #     loaded_lgbm_model = joblib.load(f'{personality_type}_lgbm_model.joblib')

    #     # Load meta-model
    #     loaded_meta_model = joblib.load(f'{personality_type}_meta_model.joblib')
        with open(f'{personality_type}_gb_model.pkl', 'rb') as f:
            loaded_gb_model = pickle.load(f)

        # Load XGBoost model
        with open(f'{personality_type}_xgb_model.pkl', 'rb') as f:
            loaded_xgb_model = pickle.load(f)

        # Load LGBM model
        with open(f'{personality_type}_lgbm_model.pkl', 'rb') as f:
            loaded_lgbm_model = pickle.load(f)

        # Load meta-model
        with open(f'{personality_type}_meta_model.pkl', 'rb') as f:
            loaded_meta_model = pickle.load(f)
        print("gb_sample_pred")
        gb_sample_pred = loaded_gb_model.predict(features)
        print("xgb_sample_pred")
        xgb_sample_pred = loaded_xgb_model.predict(features)
        print("lgbm_sample_pred")
        lgbm_sample_pred = loaded_lgbm_model.predict(features)
        # Stack predictions
        stacked_sample_predictions = np.column_stack((gb_sample_pred, xgb_sample_pred, lgbm_sample_pred))

        # Make prediction using meta-model
        print("meta_model_sample_pred")       
        meta_model_sample_pred = loaded_meta_model.predict(stacked_sample_predictions)
        pred.append(meta_model_sample_pred)
        # print(f"Predicted Personality Trait: {meta_model_sample_pred}")

    print("Before combining")
    print("Result :",result)
    print("sentiment :",sentiment)
    print("emotion_df :",emotion_df)
    result = combine_classes(pred[0], pred[1], pred[2], pred[3])
    print("Before returning")

    return {"prediction": result, "sentiment": sentiment, "emotion": emotion_df}






app = Flask(__name__)

###############################################################################
#                       SETTING UP APP ROUTES                                 #
###############################################################################

#app.use("/static", express.static('./static/'));

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/response", methods=["GET", "POST"])
def response():
    snippet = ""  # Initialize snippet with an empty string
    prediction =""
    if request.method == "POST":
        print(" B E F O R E   C A L L I N G   P R E D I C T")
        snippet = request.form["fsnippet"]
        try:
            prediction= predict(snippet)
        except:
            print("Predict contains : ",prediction)
            print("HANST RETURNED SUCCESFULLY")

        # Convert emotion index to list before passing it to the template
        emotion_labels = prediction['emotion'].columns.tolist()
        emotion_values = prediction['emotion'].values[0].tolist()
        emotion_data = {'labels': emotion_labels, 'values': emotion_values}

    return render_template("response.html", result=prediction,  emotion_data=emotion_data, string = snippet)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/explore")
def explore():
    wordcloud_data = [
        {'text': 'word1', 'weight': 10},
        {'text': 'word2', 'weight': 5},
        {'text': 'word3', 'weight': 8},
        # Add more words as needed
    ]
    eda_data = {
        'wordcloud_data': wordcloud_data,
        'chart_labels': ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ'],
        'chart_values': [1832, 1470, 1304, 1091, 685, 675, 337, 271, 231, 205, 190, 166, 89, 48, 42, 39]
    }
    
    return render_template("explore.html", eda_data=eda_data)
    

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")




###############################################################################
#                                   MAIN                                      #
###############################################################################

if __name__ == "__main__":
    app.run(debug=True)
