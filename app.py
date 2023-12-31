from flask import Flask, request, jsonify, render_template

# Data Analysis
import pandas as pd
import numpy as np


# Text Processing
import nltk
import re

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

    # Replace very short or long words with a placeholder
    text = re.sub(r'(\b\w{0,3})?\b', 'SHORTWORD', text)
    text = re.sub(r'(\b\w{30,1000})?\b', 'LONGWORD', text)
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
    print("Number of word_tf in input data:", word_tf.shape[1])

    # 2. Word-level TFIDF    
    word_tfidf=  word_tfidf_vectorizer.transform([user_input])
    print("Number of word_tfidf in input data:", word_tfidf.shape[1])

    # 3. Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(user_input)['compound']
    print("Number of sSENTIMENT FEATURES in input data:", sentiment_score)

    # 4. Emotion
    text_object = NRCLex(user_input)
    emotion_counts = text_object.affect_frequencies
    emotion_df = pd.DataFrame(emotion_counts, index=[0])
    
    if emotion_df.shape[1] < 11:
        fake_emotion = {'fake_emotion': 0}
        emotion_df = emotion_df.join(pd.DataFrame(fake_emotion, index=[0]))
    

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



def generate_highlighted_circles(personality_type):
    labels = {
        "E": "Mind",
        "I": "Mind",
        "S": "Information",
        "N": "Information",
        "T": "Decision",
        "F": "Decision",
        "J": "Lifestyle",
        "P": "Lifestyle",
    }

    # Get the letters for the predicted personality type
    predicted_letters = list(personality_type)

    # Generate HTML for a table with highlighted circles, labels, and arrows
    table_html = '<table class="center">'
    for row in [["E", "Mind", "I"], ["S", "Information", "N"], ["T", "Decision", "F"], ["J", "Lifestyle", "P"]]:
        table_html += "<tr>"
        for index, element in enumerate(row):
            label = labels.get(element, "")
            table_html += "<td>"
            if index == 1:
                # Display the label between dichotomies
                table_html += f'<span class="label" title="{label}">{element}</span>'
            elif element in predicted_letters:
                # Add an arrow for the predicted letter
                table_html += f'<span class="dot highlighted" title="{label}">{element} </span>'
            else:
                table_html += f'<span class="dot" title="{label}">{element}</span>'
            table_html += "</td>"
        table_html += "</tr>"
    table_html += "</table>"

    return table_html


# Add this function to your app.py file


# Add this function to your app.py file
def get_personality_explanation(personality_type):
    explanations = {
        "ISTJ": "A Logistician (ISTJ) is someone with the Introverted, Observant, Thinking, and Judging personality traits. These people tend to be reserved yet willful, with a rational outlook on life. They compose their actions carefully and carry them out with methodical purpose.Logisticians pride themselves on their integrity. People with this personality type mean what they say, and when they commit to doing something, they make sure to follow through.This personality type makes up a good portion of the overall population, and while Logisticians may not be particularly flashy or attention-seeking, they do more than their share to keep society on a sturdy, stable foundation. In their families and their communities, Logisticians often earn respect for their reliability, their practicality, and their ability to stay grounded and logical, even in the most stressful situations.In a world where many people shirk their responsibilities or say what they think others want to hear, Logisticians stand out as dedicated, responsible, and honest.",
        "ISFJ": "A Defender (ISFJ) is someone with the Introverted, Observant, Feeling, and Judging personality traits. These people tend to be warm and unassuming in their own steady way. They’re efficient and responsible, giving careful attention to practical details in their daily lives. n their unassuming, understated way, Defenders help make the world go round. Hardworking and devoted, people with this personality type feel a deep sense of responsibility to those around them. Defenders can be counted on to meet deadlines, remember birthdays and special occasions, uphold traditions, and shower their loved ones with gestures of care and support. But they rarely demand recognition for all that they do, preferring instead to operate behind the scenes. This is a capable, can-do personality type, with a wealth of versatile gifts. Though sensitive and caring, Defenders also have excellent analytical abilities and an eye for detail. And despite their reserve, they tend to have well-developed people skills and robust social relationships. Defenders are truly more than the sum of their parts, and their varied strengths shine in even the most ordinary aspects of their daily lives.",
        "INFJ": "An Advocate (INFJ) is someone with the Introverted, Intuitive, Feeling, and Judging personality traits. They tend to approach life with deep thoughtfulness and imagination. Their inner vision, personal values, and a quiet, principled version of humanism guide them in all things.Advocates (INFJs) may be the rarest personality type of all, but they certainly leave their mark on the world. Idealistic and principled, they aren’t content to coast through life – they want to stand up and make a difference. For Advocate personalities, success doesn’t come from money or status but from seeking fulfillment, helping others, and being a force for good in the world.While they have lofty goals and ambitions, Advocates shouldn’t be mistaken for idle dreamers. People with this personality type care about integrity, and they’re rarely satisfied until they’ve done what they know to be right. Conscientious to the core, they move through life with a clear sense of their values, and they aim never to lose sight of what truly matters – not according to other people or society at large, but according to their own wisdom and intuition.",
        "INTJ": "An Architect (INTJ) is a person with the Introverted, Intuitive, Thinking, and Judging personality traits. These thoughtful tacticians love perfecting the details of life, applying creativity and rationality to everything they do. Their inner world is often a private, complex one. It can be lonely at the top. As one of the rarest personality types – and one of the most capable – Architects (INTJs) know this all too well. Rational and quick-witted, Architects pride themselves on their ability to think for themselves, not to mention their uncanny knack for seeing right through phoniness and hypocrisy. But because their minds are never at rest, Architects may struggle to find people who can keep up with their nonstop analysis of everything around them.",
        "ISTP": "A Virtuoso (ISTP) is someone with the Introverted, Observant, Thinking, and Prospecting personality traits. They tend to have an individualistic mindset, pursuing goals without needing much external connection. They engage in life with inquisitiveness and personal skill, varying their approach as needed.Virtuosos love to explore with their hands and their eyes, touching and examining the world around them with cool rationalism and spirited curiosity. People with this personality type are natural Makers, moving from project to project, building the useful and the superfluous for the fun of it, and learning from their environment as they go. Often mechanics and engineers, Virtuosos find no greater joy than in getting their hands dirty pulling things apart and putting them back together, just a little bit better than they were before.",
        "ISFP": "An Adventurer (ISFP) is a person with the Introverted, Observant, Feeling, and Prospecting personality traits. They tend to have open minds, approaching life, new experiences, and people with grounded warmth. Their ability to stay in the moment helps them uncover exciting potentials.Adventurers are true artists – although not necessarily in the conventional sense. For this personality type, life itself is a canvas for self-expression. From what they wear to how they spend their free time, Adventurers act in ways that vividly reflect who they are as unique individuals. And every Adventurer is certainly unique. Driven by curiosity and eager to try new things, people with this personality often have a fascinating array of passions and interests. With their exploratory spirits and their ability to find joy in everyday life, Adventurers can be among the most interesting people you’ll ever meet. The only irony? Unassuming and humble, Adventurers tend to see themselves as “just doing their own thing,” so they may not even realize how remarkable they really are. ",
        "INFP": "A Mediator (INFP) is someone who possesses the Introverted, Intuitive, Feeling, and Prospecting personality traits. These rare personality types tend to be quiet, open-minded, and imaginative, and they apply a caring and creative approach to everything they do. Although they may seem quiet or unassuming, Mediators (INFPs) have vibrant, passionate inner lives. Creative and imaginative, they happily lose themselves in daydreams, inventing all sorts of stories and conversations in their minds. These personalities are known for their sensitivity – Mediators can have profound emotional responses to music, art, nature, and the people around them. Idealistic and empathetic, Mediators long for deep, soulful relationships, and they feel called to help others. But because this personality type makes up such a small portion of the population, Mediators may sometimes feel lonely or invisible, adrift in a world that doesn’t seem to appreciate the traits that make them unique.",
        "INTP": "A Logician (INTP) is someone with the Introverted, Intuitive, Thinking, and Prospecting personality traits. These flexible thinkers enjoy taking an unconventional approach to many aspects of life. They often seek out unlikely paths, mixing willingness to experiment with personal creativity. Logicians pride themselves on their unique perspectives and vigorous intellect. They can’t help but puzzle over the mysteries of the universe – which may explain why some of the most influential philosophers and scientists of all time have been Logicians. This personality type is fairly rare, but with their creativity and inventiveness, Logicians aren’t afraid to stand out from the crowd.",
        "ESTP": "An Entrepreneur (ESTP) is someone with the Extraverted, Observant, Thinking, and Prospecting personality traits. They tend to be energetic and action-oriented, deftly navigating whatever is in front of them. They love uncovering life’s opportunities, whether socializing with others or in more personal pursuits.Entrepreneurs always have an impact on their immediate surroundings – the best way to spot them at a party is to look for the whirling eddy of people flitting about them as they move from group to group. Laughing and entertaining with a blunt and earthy humor, Entrepreneur personalities love to be the center of attention. If an audience member is asked to come on stage, Entrepreneurs volunteer – or volunteer a shy friend. Theory, abstract concepts and plodding discussions about global issues and their implications don’t keep Entrepreneurs interested for long. Entrepreneurs keep their conversation energetic, with a good dose of intelligence, but they like to talk about what is – or better yet, to just go out and do it. Entrepreneurs leap before they look, fixing their mistakes as they go, rather than sitting idle, preparing contingencies and escape clauses.",
        "ESFP": "An Entertainer (ESFP) is a person with the Extraverted, Observant, Feeling, and Prospecting personality traits. These people love vibrant experiences, engaging in life eagerly and taking pleasure in discovering the unknown. They can be very social, often encouraging others into shared activities. If anyone is to be found spontaneously breaking into song and dance, it is the Entertainer personality type. Entertainers get caught up in the excitement of the moment, and want everyone else to feel that way, too. No other personality type is as generous with their time and energy as Entertainers when it comes to encouraging others, and no other personality type does it with such irresistible style.",
        "ENFP": "A Campaigner (ENFP) is someone with the Extraverted, Intuitive, Feeling, and Prospecting personality traits. These people tend to embrace big ideas and actions that reflect their sense of hope and goodwill toward others. Their vibrant energy can flow in many directions.Campaigners (ENFPs) are true free spirits – outgoing, openhearted, and open-minded. With their lively, upbeat approach to life, they stand out in any crowd. But even though they can be the life of the party, Campaigners don’t just care about having a good time. These personality types run deep – as does their longing for meaningful, emotional connections with other people.",
        "ENTP": "A Debater (ENTP) is a person with the Extraverted, Intuitive, Thinking, and Prospecting personality traits. They tend to be bold and creative, deconstructing and rebuilding ideas with great mental agility. They pursue their goals vigorously despite any resistance they might encounter. Quick-witted and audacious, Debaters aren’t afraid to disagree with the status quo. In fact, they’re not afraid to disagree with pretty much anything or anyone. Few things light up people with this personality type more than a bit of verbal sparring – and if the conversation veers into controversial terrain, so much the better. It would be a mistake, though, to think of Debaters as disagreeable or mean-spirited. Instead, people with this personality type are knowledgeable and curious, with a playful sense of humor, and they can be incredibly entertaining. They simply have an offbeat, contrarian idea of fun – one that involves a healthy dose of spirited debate",
        "ENTJ": "A Commander (ENTJ) is someone with the Extraverted, Intuitive, Thinking, and Judging personality traits. They are decisive people who love momentum and accomplishment. They gather information to construct their creative visions but rarely hesitate for long before acting on them.Commanders are natural-born leaders. People with this personality type embody the gifts of charisma and confidence, and project authority in a way that draws crowds together behind a common goal. However, Commanders are also characterized by an often ruthless level of rationality, using their drive, determination and sharp minds to achieve whatever end they’ve set for themselves. Perhaps it is best that they make up only three percent of the population, lest they overwhelm the more timid and sensitive personality types that make up much of the rest of the world – but we have Commanders to thank for many of the businesses and institutions we take for granted every day.",
        "ESFJ": "A Consul (ESFJ) is a person with the Extraverted, Observant, Feeling, and Judging personality traits. They are attentive and people-focused, and they enjoy taking part in their social community. Their achievements are guided by decisive values, and they willingly offer guidance to others.For Consuls, life is sweetest when it’s shared with others. People with this personality type form the bedrock of many communities, opening their homes – and their hearts – to friends, loved ones, and neighbors.This doesn’t mean that Consuls like everyone, or that they’re saints. But Consuls do believe in the power of hospitality and good manners, and they tend to feel a sense of duty to those around them. Generous and reliable, people with this personality type often take it upon themselves – in ways both large and small – to hold their families and their communities together.",
        "ENFJ": "A Protagonist (ENFJ) is a person with the Extraverted, Intuitive, Feeling, and Judging personality traits. These warm, forthright types love helping others, and they tend to have strong ideas and values. They back their perspective with the creative energy to achieve their goals. Protagonists (ENFJs) feel called to serve a greater purpose in life. Thoughtful and idealistic, these personality types strive to have a positive impact on other people and the world around them. They rarely shy away from an opportunity to do the right thing, even when doing so is far from easy.Protagonists are born leaders, which explains why these personalities can be found among many notable politicians, coaches, and teachers. Their passion and charisma allow them to inspire others not just in their careers but in every arena of their lives, including their relationships. Few things bring Protagonists a deeper sense of joy and fulfillment than guiding friends and loved ones to grow into their best selves.",
        "ESTJ": "An Executive (ESTJ) is someone with the Extraverted, Observant, Thinking, and Judging personality traits. They possess great fortitude, emphatically following their own sensible judgment. They often serve as a stabilizing force among others, able to offer solid direction amid adversity. Executives are representatives of tradition and order, utilizing their understanding of what is right, wrong and socially acceptable to bring families and communities together. Embracing the values of honesty, dedication and dignity, people with the Executive personality type are valued for their clear advice and guidance, and they happily lead the way on difficult paths. Taking pride in bringing people together, Executives often take on roles as community organizers, working hard to bring everyone together in celebration of cherished local events, or in defense of the traditional values that hold families and communities together.",
    }

    return explanations.get(personality_type, "Explanation not available.")




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
    print("Number of features in input data:", features.shape[1])
    print('Going into model...')
    for personality_type in personality_types:

        with open(f'{personality_type}_svm_PREFINAL_model.pkl', 'rb') as f:
            loaded_svm_model = pickle.load(f)

        with open(f'{personality_type}_lr_PREFINAL_model.pkl', 'rb') as f:
            loaded_lr_model = pickle.load(f)
            
        with open(f'{personality_type}_gb_PREFINAL_model.pkl', 'rb') as f:
            loaded_gb_model = pickle.load(f)

        # Load XGBoost model
        with open(f'{personality_type}_xgb_PREFINAL_model.pkl', 'rb') as f:
            loaded_xgb_model = pickle.load(f)

        # Load LGBM model
        with open(f'{personality_type}_lgbm_PREFINAL_model.pkl', 'rb') as f:
            loaded_lgbm_model = pickle.load(f)

        # Load LGBM model
        with open(f'{personality_type}_cb_PREFINAL_model.pkl', 'rb') as f:
            loaded_cb_model = pickle.load(f)

        # Load LGBM model
        with open(f'{personality_type}_ab_PREFINAL_model.pkl', 'rb') as f:
            loaded_ab_model = pickle.load(f)

        # Load LGBM model
        with open(f'{personality_type}_bag_PREFINAL_model.pkl', 'rb') as f:
            loaded_bag_model = pickle.load(f)
           
            

        # with open(f'{personality_type}_gb_model.pkl', 'rb') as f:
        #     loaded_gb_model = pickle.load(f)

        # # Load XGBoost model
        # with open(f'{personality_type}_xgb_model.pkl', 'rb') as f:
        #     loaded_xgb_model = pickle.load(f)

        # # Load LGBM model
        # with open(f'{personality_type}_lgbm_model.pkl', 'rb') as f:
        #     loaded_lgbm_model = pickle.load(f)



        # Load meta-model
        with open(f'{personality_type}_PREFINAL_meta_model.pkl', 'rb') as f:
            loaded_meta_model = pickle.load(f)
        
        
        loaded_svm_pred= loaded_svm_model.predict(features)
        loaded_lr_pred= loaded_lr_model.predict(features)
        gb_sample_pred = loaded_gb_model.predict(features)
        xgb_sample_pred = loaded_xgb_model.predict(features)
        lgbm_sample_pred = loaded_lgbm_model.predict(features)
       # loaded_cb_pred= loaded_cb_model.predict(features)
        loaded_cb_pred = np.array([0])  # You can replace 0 with any fake prediction value for your testing
        loaded_ab_pred = loaded_ab_model.predict(features)
        loaded_bag_pred= loaded_bag_model.predict(features)
        
        
        
        # Stack predictions
        stacked_sample_predictions = np.column_stack((loaded_cb_pred,loaded_svm_pred,loaded_lr_pred, gb_sample_pred, xgb_sample_pred, lgbm_sample_pred,loaded_ab_pred,loaded_bag_pred ))

        # Make prediction using meta-model
        print("meta_model_sample_pred")       
        meta_model_sample_pred = loaded_meta_model.predict(stacked_sample_predictions)
        pred.append(meta_model_sample_pred)
        # print(f"Predicted Personality Trait: {meta_model_sample_pred}")

    print("Before combining")
    result = combine_classes(pred[0], pred[1], pred[2], pred[3])
    print("After combining")
    print("Result :",result)
    print("sentiment :",sentiment)
    print("emotion_df :",emotion_df)
    print("Before returning")

    return {"prediction": result, "sentiment": sentiment, "emotion": emotion_df}

import pickle

with open('result_df.pkl', 'rb') as file:
    result_df = pickle.load(file)




app = Flask(__name__)

###############################################################################
#                       SETTING UP APP ROUTES                                 #
###############################################################################

#app.use("/static", express.static('./static/'));

@app.route("/")
def index():
    print("AT INDEX PAGEEE")
    return render_template("index.html")


@app.route("/response", methods=["GET", "POST"])
def response():
    snippet = ""  # Initialize snippet with an empty string
    emotion_data=[]
    sentiment_data = {'pos_sentiment': 0, 'neg_sentiment': 0, 'neu_sentiment': 0}
    output =""
    prediction ={}
    highlighted_circless=""
    personality_explanation=""
    if request.method == "POST":
        print(" B E F O R E   C A L L I N G   P R E D I C T")
        snippet = request.form["fsnippet"]
        print("AT RESPONSE FINCTION RECEIVED TEXT ",snippet)
        prediction= predict(snippet)
        output = prediction['prediction']

        # Convert sentiment_data index to list before passing it to the template
        sentiment_data = prediction.get('sentiment', {'pos_sentiment': 0, 'neg_sentiment': 0, 'neu_sentiment': 0})
        # Convert emotion index to list before passing it to the template
        emotion_labels = prediction['emotion'].columns.tolist()
        emotion_values = prediction['emotion'].values[0].tolist()
        emotion_data = {'labels': emotion_labels, 'values': emotion_values}
        print("sentiment_data :",sentiment_data)
        highlighted_circless = generate_highlighted_circles(output)
        personality_explanation = get_personality_explanation(output)

    return render_template("response.html", predicted=output, emotion_data=emotion_data, result=prediction, string=snippet,highlighted_circles=highlighted_circless,personality_explanation=personality_explanation)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/explore", methods=["GET", "POST"])
def explore():
  

    eda_data = {
        'chart_labels': ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ'],
        'chart_values': [1832, 1470, 1304, 1091, 685, 675, 337, 271, 231, 205, 190, 166, 89, 48, 42, 39]
    }
  
  # Replace this logic with your actual data extraction
    column_names = result_df.columns.tolist()
    eda2_data = []

    for _, row in result_df.iterrows():
        row_data = {}
        for col_name in column_names:
            row_data[col_name] = row[col_name]
        eda2_data.append(row_data)

    return render_template("explore.html", eda_data=eda_data,eda2_data=eda2_data)
    

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


###############################################################################
#                                   MAIN                                      #
###############################################################################

if __name__ == "__main__":
    app.run(debug=True)
