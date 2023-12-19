from flask import Flask, request, jsonify, render_template
from preprocess_app import predict

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

    if request.method == "POST":
        snippet = request.form["fsnippet"]
        prediction= predict(snippet)

        # Convert emotion index to list before passing it to the template
        emotion_labels = prediction['emotion'].columns.tolist()
        emotion_values = prediction['emotion'].values[0].tolist()
        emotion_data = {'labels': emotion_labels, 'values': emotion_values}

    return render_template("response.html", result=predict(snippet),  emotion_data=emotion_data, string = snippet)


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
