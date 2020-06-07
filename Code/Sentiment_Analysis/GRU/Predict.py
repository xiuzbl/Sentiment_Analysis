from os import listdir
from os.path import isfile, join
import os
import jieba
import codecs
# from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pickle
import random
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pandas as pd
import pycantonese as pc
import re
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import warnings
warnings.filterwarnings('ignore')

corpus = pc.hkcancor()
freq = corpus.word_frequency()


def save(file_path, init_words_path, tagged_words):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(init_words_path, 'r') as t:
        lines = t.readlines()
        with open(file_path, 'w') as f:
            for word in tagged_words:
                word_freq = freq[word[0]] if word[0] in freq else None
                word_tag = word[1].lower()
                word_tag_matched = bool(re.match('^[a-z]+$', word_tag))
                word_line = word[0]
                if word_freq is not None:
                    word_line = word_line + ' ' + str(word_freq)
                if word_tag_matched is True:
                    word_line = word_line + ' ' + str(word_tag)
                f.write(word_line + '\n')

            for line in lines:
                f.write(line)

save('cantonese-corpus/data/dict.txt', 'cantonese-corpus/data/init_dict.txt', corpus.tagged_words())
jieba.load_userdict("cantonese-corpus/data/dict.txt")

def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff
model = None
sentiment_tag = None
maxLength = None
# Model = 'sentiment_chinese_model.HDF5'
def loadModel():
    global model, sentiment_tag, maxLength
    metaData = __loadStuff("meta_sentiment_chinese.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    embedding_dim = 512
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(GRU(64, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        # model.add(Dense(output_dimen, activation='softmax'))
        model.add(Dense(output_dimen, activation='sigmoid'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        print('*'*100)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('sentiment_chinese_model.HDF5')
        model.summary()
    print("Model weights loaded!")

stop_words = pc.stop_words()
def findFeatures(text):
    text = [w for w in text if not w in stop_words]
    text = ''.join(text).strip()
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = list(seg_list)
    text = " ".join(seg_list)
    text = Converter('zh-hans').convert(text)
    textArray = [text]
    input_tokenizer_load = __loadStuff("input_tokenizer_chinese.p")
    textArray = np.array(pad_sequences(input_tokenizer_load.texts_to_sequences(textArray), maxlen=maxLength))
    return textArray
def predictResult(text):
    if model is None:
        return None
    features = findFeatures(text)
    predicted = model.predict(features)[0] # we have only one sentence to predict, so take index 0
    predicted = np.array(predicted)
    # print(predicted,predicted.shape)
    # print(predicted.argmax())
    # if predicted.argmax()==0:
    #     return -(2*predicted.max()-1)
    # else:
    return 2*predicted[1]-1
    # probab = predicted.max()*predicted.argmax()
    # # predition = sentiment_tag[predicted.argmax()]

loadModel()

# filename = open('test.csv','r')
df_test = pd.read_csv('test.csv')
gru_res = open('gru_res.txt','w')
result = pd.DataFrame(columns=['review','label','score'])
true_label = []
pred_label = []
for idx,row in df_test.iterrows():
    text = row['review'].strip('\n')
    label = row['label']
    true_label.append(label)
    score = predictResult(text)
    pred = 0 if score<0 else 1
    pred_label.append(pred)
    result.loc[idx] = [text,label, score]
    print(idx)
true_label = np.array(true_label)
pred_label = np.array(pred_label)
print('AUC Score is %.4f'%roc_auc_score(true_label,pred_label),file=gru_res)
print('F1 Score is %.4f'%f1_score(true_label,pred_label),file=gru_res)

result.to_csv('GRU_result.csv',index=False)
class_names = ['neg','pos']
print('Classification Report:\n',classification_report(true_label, pred_label, target_names=class_names),file=gru_res)
gru_res.close()
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig('gru_cm.png')
cm = confusion_matrix(true_label, pred_label)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)






