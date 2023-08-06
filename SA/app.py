# app.py
from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

# load the sentiment analysis model
model = joblib.load("model/sentiment_analysis_model.pkl")
cv = joblib.load("model/cv.pkl")


# define the routes
@app.route('/')
def index():
    return render_template('index.html')

def lowercase(text):
    x = ''
    for i in text:
        if i == ' ':
            x = x + ' '
        else:
            x = x + i.lower()
    return x

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stemming(text):
    x = ''
    for i in text.split():
        if(x == ''):
            x = x + ps.stem(i)
        else:
            x = x + " " + ps.stem(i)
    return x



@app.route('/predict', methods=['POST'])
def predict():
    para = request.form['text']
    lc_text = lowercase(para)
    stem_text = stemming(lc_text)
    list_stem_text = [stem_text]
    vec_text = cv.transform(list_stem_text).toarray()
    prediction = model.predict(vec_text)[0]
    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    return render_template('index.html', prediction_text = "The sentiment is " + sentiment + "."
)

if __name__ == '__main__':
    app.run(debug=True)
