from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        transformed_sms = vectorizer.transform([sms_text])
        prediction = model.predict(transformed_sms)
        
        if prediction[0] == 1:
            result = 'Spam'
        else:
            result = 'Not Spam'
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
