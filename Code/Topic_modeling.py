import re
import numpy as np
import pandas as pd
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel
# import spacy
import jieba

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud,ImageColorGenerator
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pycantonese as pc
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
stop_words = pc.stop_words()

df = pd.read_csv('test.csv')
# data = np.array(df['review'].values).tolist()
text = open('test_text.txt','w')
for idx,row in df.iterrows():
    print(row['review'],file=text)

n_topics = 5
n_top_words = 15
filename = open('text_cut_stw.txt','r')
data  = []
for line in filename:
    data.append(line.strip())
# tf_vectorizer = CountVectorizer(input='filename',
#                                 stop_words='english')
tf_vectorizer = TfidfVectorizer()
tf = tf_vectorizer.fit_transform(data)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
# print(tf_feature_names)
font = 'simheittf/simhei.ttf'
lda_res = open('lda_res.txt','w')
for i in range (0,n_topics):
    termsInTopic = lda.components_[i].argsort()[:-50-1:-1]
    termsAndCounts = {}
    for term in termsInTopic:
        termsAndCounts[str(tf_feature_names[term].encode().decode())] \
            = math.ceil(lda.components_[i][term]*1000)
    print('Topics'+str(i),':\n',termsAndCounts,file=lda_res)
    cloud = WordCloud(background_color="white",font_path=font)
    cloud.generate_from_frequencies(termsAndCounts)
    # image_color = ImageColorGenerator(graph)

    plt.imshow(cloud)
    plt.axis("off")
    plt.savefig(str(i))
    plt.show()
    # break
    # plt.savefig('')