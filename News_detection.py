import pandas as pd

df = pd.read_json("E:/Projects/Machine learning/News Detection/news.json" , lines=True)
df = df[df.category.isin(["POLITICS","ENTERTAINMENT" ,"COMEDY"])]
corpus = df["headline"]
target = df["category"]

corpus.value_counts()

#Data Cleaning

import regex as re
from nltk.corpus import stopwords
sw = stopwords.words("English")

def text_cleaning(doc):
    doc=doc.lower()
    doc=re.sub('[^a-z ]','',doc)
    tokens=doc.split()
    newdoc=""
    for token in tokens:
        if token not in sw:
            newdoc=newdoc+token+" "
    return newdoc.strip()
final_corpus=list(map(text_cleaning,corpus))
final_corpus
train_set = []
for i in zip(final_corpus,target):
    train_set.append(i)

print(train_set)
from textblob.classifiers import NaiveBayesClassifier

model = NaiveBayesClassifier(train_set)

sample = input("Enter your review : ")
clean_sample = text_cleaning(sample)
print(model.classify(clean_sample))