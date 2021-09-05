import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import export_graphviz

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import graphviz
from graphviz import Source
from matplotlib import pyplot as plt

import re
import copy

import nltk
# from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet


# case 1
def preprocessing_1(x):
    pd_data = x.copy(deep=True)
    return pd_data


# case 2
def preprocessing_2(x):
    pd_data = x.copy(deep=True)
    for i in range(len(pd_data['verified_reviews'])):
        comments = pd_data['verified_reviews'][i]
        # remove Url
        # comments = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', comments)
        comments = re.sub(r'https?\S+', ' ', comments)
        # remove except upper,lowercase, number, {’, #, @, $}
        new_comments = re.sub(r"[^a-zA-Z0-9\#\@\$\'\’\s]+", ' ', comments)
        pd_data.loc[i, 'verified_reviews'] = new_comments
    return pd_data


# case 3
def preprocessing_3(x):
    pd_data_3 = x.copy(deep=True)
    for i in range(len(pd_data_3['verified_reviews'])):
        comments = pd_data_3['verified_reviews'][i]
        new_comments = comments.lower()
        pd_data_3.loc[i, 'verified_reviews'] = new_comments
    return pd_data_3


# case 4
def preprocessing_4(x):
    pd_data_4 = x.copy(deep=True)
    for i in range(len(pd_data_4['verified_reviews'])):
        comments = pd_data_4['verified_reviews'][i]
        new_comments = " ".join([word for word in comments.split(" ") if not word in set(stopwords.words('english'))])
        pd_data_4.loc[i, 'verified_reviews'] = new_comments
    return pd_data_4


def limit_attributes(the_data, n_attributes):
    pd_data = the_data.copy(deep=True)
    vec = CountVectorizer(analyzer=lambda x: x.split(), max_features=n_attributes)
    dense_x = vec.fit_transform(pd_data['verified_reviews'])
    sparse_x = pd.DataFrame.sparse.from_spmatrix(dense_x, columns=vec.get_feature_names())
    return sparse_x


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    if word:
        # if word is not space
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    else:
        # if word is space, treat it as noun.
        return "n"


lemmatizer = WordNetLemmatizer()


# First I drop the row with no comments (verified_reviews is empty), this is because
# Then I change the value of the feedback {1: "nonegative", 0: "negative"}
# Finally I add it at the end of the reviews and lemmatize the all words in reviews.

def feature_engineer_for_feedback(the_dataframe):
    # drop the row with no comments
    data_5_2 = the_dataframe.copy(deep=True)
    # get empty reviews index
    drop_index_list = [i for i in range(len(data_5_2)) if data_5_2['verified_reviews'][i] == ' ']
    # remove empty common
    data_5_2.drop(drop_index_list, inplace=True)
    data_5_2 = data_5_2.reset_index(drop=True)

    # add new feature for reviews
    for i in range(len(data_5_2['verified_reviews'])):
        comments = data_5_2['verified_reviews'][i]
        if data_5_2['feedback'][i] == 0:
            new_comments = " ".join(
                [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in comments.split(" ")] + ["negative"])
        else:
            new_comments = " ".join(
                [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in comments.split(" ")] + ["nonnegative"])
        data_5_2.loc[i, 'verified_reviews'] = new_comments
    return data_5_2


data = pd.read_csv('amazon_alexa.tsv', sep='\t')
a_map = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
data["rating"].replace(a_map, inplace=True)

vectorizer = CountVectorizer(analyzer=lambda x: x.split(), max_features=1000)
x_dense = vectorizer.fit_transform(data['verified_reviews'])
x_sparse = pd.DataFrame.sparse.from_spmatrix(x_dense, columns=vectorizer.get_feature_names())

pre_1_data = preprocessing_1(data)
pre_2_data = preprocessing_2(data)
pre_3_data = preprocessing_3(pre_2_data)
pre_4_data = preprocessing_4(pre_3_data)

print(pre_4_data)
pre_4_data.to_csv("pre_4_amazon.csv")
