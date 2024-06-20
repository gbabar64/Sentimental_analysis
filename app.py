import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from process import pre_process, pro

model = load_model("model_lstm.h5")

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        text = request.form['tweet']
        lis = pre_process(text)
        pred = model.predict(lis)
        a = pred[0][0]
        b = pred[0][1]
        c = pred[0][2]
        mx = 0
        if a > b and a > c:
            mx = a
        elif b > a and b > c:
            mx = b
        elif c > a and c > b:
            mx = c
        if (mx == b):
            return render_template('index.html', prediction_textb='Positive' + ' ' + str(b))
        elif (mx == a):
            return render_template('index.html', prediction_texta='Neutral' + ' ' + str(a))
        elif (mx == c):
            return render_template('index.html', prediction_textc='Negative' + ' ' + str(c))

if __name__ == "__main__":
    app.run(debug=True)
