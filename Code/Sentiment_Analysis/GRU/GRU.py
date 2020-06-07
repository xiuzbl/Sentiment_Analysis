#!/usr/bin/env python
# coding: utf-8

# In[1]:

from os import listdir
from os.path import isfile, join
import jieba
import codecs
# from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pickle
import random
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import warnings
from keras.callbacks import History

from keras.utils import plot_model
warnings.filterwarnings('ignore')
# from gensim.models import CoherenceModel
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# import scipy.stats as stats
# import pylab as pl

def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff
def measure(y_true,y_pred):
    y1_pred = []
    for i in y_pred:
        if i>0:
            y1_pred.append(1)
        else:
            y1_pred.append(0)
    y1_pred = np.array(y1_pred)
    y_true  = np.array(y_true)
    return  np.mean(y_true==y1_pred)
    # return accuracy_score(y_true,y_pred)

posi_file = 'pos_cut_stw.txt'
neg_file = 'neg_cut_stw.txt'
document = []
for line in open(posi_file,'r'):
    line = line.strip('\n')
    document.append([line,1])
for line in open(neg_file,'r'):
    line = line.strip('\n')
    document.append([line,0])


# In[5]:
random.shuffle(document)
Train = document



# In[6]:

X = []
Y = []
for doc in Train:
    X.append(doc[0])
    Y.append(doc[1])


# In[7]:
Y = to_categorical(Y,num_classes=2)
# Y_test = to_categorical(Y_test,num_classes = 2)
output_dimen = Y.shape[1]


h = sorted([len(sentence) for sentence in X])
maxLength = h[int(len(h) * 0.8)]
print('maxLength = %d'%maxLength)
input_tokenizer = Tokenizer(30000) # Initial vocab size
input_tokenizer.fit_on_texts(X)
vocab_size = len(input_tokenizer.word_index) + 1
print("input vocab_size:",vocab_size)
X = np.array(pad_sequences(input_tokenizer.texts_to_sequences(X), maxlen=maxLength))
# X_test = np.array(pad_sequences(input_tokenizer.texts_to_sequences(X_test), maxlen=maxLength))
__pickleStuff("input_tokenizer_chinese.p", input_tokenizer)


# In[10]:
metaData = {"maxLength":maxLength,"vocab_size":vocab_size,"output_dimen":output_dimen}
__pickleStuff("meta_sentiment_chinese.p", metaData)

print(len(document))


embedding_dim = 512
history = History()

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length = maxLength))
# Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
# All the intermediate outputs are collected and then passed on to the second GRU layer.
model.add(GRU(256, dropout=0.9, return_sequences=True))
# Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
model.add(GRU(64, dropout=0.9))
# The output is then sent to a fully connected layer that would give us our final output_dim classes
model.add(Dense(output_dimen, activation='sigmoid'))
# We use the adam optimizer instead of standard SGD since it converges much faster
# tbCallBack = TensorBoard(log_dir='./Graph/sentiment_chinese', histogram_freq=0,
#                             write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.load_weights('sentiment_chinese_model.HDF5')
model.summary()
model.fit(X, Y, batch_size=32, epochs=20,validation_split=0.1, verbose=1, callbacks=[history])
model.save_weights('sentiment_chinese_model.HDF5')
# plot_model(model, to_file='gru_model.png')


# 绘制训练 & 验证的准确率值
history_dict = history.history
print(history_dict.keys())
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim([0,20])

plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('gru_tr_acc.png')

# 绘制训练 & 验证的损失值
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim([0,20])
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('gru_tr_loss.png')
print("Saved model!")
# loadModel('sentiment_chinese_model.HDF5')

# y_pred = []
# for text in X_test:
#     yy = model.predict(text.reshape((1,len(text))))[0]
#     y_pred.append(2*yy[1]-1)
# y_re = open('y_re.txt','w')
# print(y_pred,Y_test,file=y_re)
# print('*'*100)
# print('Evaluation on test dataset',measure(Y_test,y_pred))