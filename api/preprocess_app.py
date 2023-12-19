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
from scipy.sparse import hstack

def cleaning_text(text):
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
    for personality_type in personality_types:
    # Load individual models
        loaded_gb_model = joblib.load(f'models\{personality_type}_gb_model.joblib')
        loaded_xgb_model = joblib.load(f'models\{personality_type}_xgb_model.joblib')
        loaded_lgbm_model = joblib.load(f'models\{personality_type}_lgbm_model.joblib')

        # Load meta-model
        loaded_meta_model = joblib.load(f'models\{personality_type}_meta_model.joblib')

        gb_sample_pred = loaded_gb_model.predict(features)
        xgb_sample_pred = loaded_xgb_model.predict(features)
        lgbm_sample_pred = loaded_lgbm_model.predict(features)
        
        # Stack predictions
        stacked_sample_predictions = np.column_stack((gb_sample_pred, xgb_sample_pred, lgbm_sample_pred))

        # Make prediction using meta-model
        meta_model_sample_pred = loaded_meta_model.predict(stacked_sample_predictions)
        pred.append(meta_model_sample_pred)
        print(f"Predicted Personality Trait: {meta_model_sample_pred}")


    result = combine_classes(pred[0], pred[1], pred[2], pred[3])

    return {"prediction": result, "sentiment": sentiment, "emotion": emotion_df}






# user_input_text = "This is unlike any kind of adventure movie my eyes have ever seen in such a long time, the characters, the musical score for every scene, the story, the beauty of the landscapes of Pandora, the rich variety and uniqueness of the flora and fauna of Pandora, the ways and cultures and language of the natives of Pandora, everything about this movie I am beyond impressed and truly captivated by. Sam Worthington is by far my favorite actor in this movie along with his character Jake Sulley, just as he was a very inspiring actor in The Shack Sam Worthington once again makes an unbelievable mark in one of the greatest and most captivating movies you'll ever see."
# text= "Mainly being anxious. I’m kinda scared to raise a kid because I’m afraid of accidentally smothering them and they end up not wanting to spend time with me. Also I’d probably be overprotective. Not like a helicopter parent because I want my child to have the freedom to pursue their own interests."
# # Extract features from user-input text
# user_features = predict(text)