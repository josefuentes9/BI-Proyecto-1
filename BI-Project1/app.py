import os
import pickle
import re
import sqlite3

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from nltk.stem import porter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from wtforms import Form, TextAreaField, validators
from dotenv import load_dotenv

# import HashingVectorizer from local dir
app = Flask(__name__)


# Preparing the Classifier
cur_dir = os.path.dirname(__file__)

stop = pickle.load(open(os.path.join(
    cur_dir, 'pkl_objects', 'stopwords.pkl'),
    'rb'))
count = CountVectorizer()


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
           + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def tokenizer_porter(text):
    return ' '.join([porter.stem(word) for word in text.split()])


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
reviews_df = pd.read_csv("./training/reviews/tipo1_entrenamiento_estudiantes.csv")
X_train = reviews_df.loc[:7875, 'Review'].values
y_train = reviews_df.loc[:7875, 'Class'].values
clf.fit(X_train, y_train)


def classify(document):
    y = clf.predict([document])
    return  y


def train(document, y):
    X = count.fit([document])
    np.append(X_train, [document])
    np.append(y_train, [y])
    clf.fit(X_train, y_train)


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, class, date)" \
              " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


# Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                 validators.length(min=15)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y = classify(review)
        return render_template('results.html',
                               content=review,
                               prediction=y,
                               probability=80)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback1 = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    y = prediction
    if feedback1 == 'Incorrect':
        y = int(not (y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')


if __name__ == '__main__':
    app.run(debug=True)
