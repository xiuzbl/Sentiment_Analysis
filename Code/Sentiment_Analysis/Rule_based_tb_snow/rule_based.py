from bixin import predict
import pandas as pd
import requests
import json
import time
from textblob import TextBlob
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# text ="可是3年後cut單，是虧損的。沒意思"
# print(predict(text))
# filename = open('sen_2.txt','r')
# result = pd.DataFrame(columns=('content', 'pro_score', 'comp_score'))
df_test = pd.read_csv('sentence_vader_result.csv')
analyzer = SentimentIntensityAnalyzer()

# def toEnglish(sentence,from_lang):
#     to_lang = "en"
#
#     api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(sentence, from_lang,
#                                                                                 to_lang)
#     # requests.get(link, headers={'User-agent': 'your bot 0.1'})
#
#     hdrs = {
#         'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
#         'Accept-Encoding': 'none',
#         'Accept-Language': 'en-US,en;q=0.8',
#         'Connection': 'keep-alive'}
#     response = requests.get(api_url, headers=hdrs)
#     response_json = json.loads(response.text)
#     translation = response_json["responseData"]["translatedText"]
#     print(translation)
#     vs = analyzer.polarity_scores(translation)
#     return vs

# vader_blob_res = pd.DataFrame(columns=['review','Eng','vader_score','vader_compound','textblob_score'])
# translator = Translator()
#
# for idx,row in df_test.iterrows():
#     print(idx)
#     if idx>=345:
#         review = row['content']
#         # label = row['label']
#         trans = translator.translate(review, src='zh-tw', dest='en').text
#         va_score = analyzer.polarity_scores(trans)
#         tx_score = TextBlob(trans).sentiment.polarity
#         va_com = va_score['compound']
#         vader_blob_res.loc[idx] = [review,trans,va_score,va_com,tx_score]
#         vader_blob_res.to_csv('insurance_vader_blob_res_2.csv',index=False)
in1 = pd.read_csv('insurance_vader_blob_res_1.csv')
in2 = pd.read_csv('insurance_vader_blob_res_2.csv')
in_all = pd.concat((in1,in2))
in_all.to_csv('insurance_res_vader.csv',index=False)


# for sentence in filename:
#     # time.sleep(5)
#     sentence = sentence.strip()
#     if len(sentence)==2:
#         sentence+=' '
#     elif len(sentence)<2:
#         score = {'compound':0}
#         continue
#     if sentence.isalnum():
#         sentence1 = sentence
#     else:
#         sentence1 = translator.translate(sentence,src='zh-tw',dest='en').text
#     # print(sentence1)
#     score = analyzer.polarity_scores(sentence1)
#
#     print(i)
#     result.loc[i] = [sentence,score,score['compound']]
#     i+=1
#     result.to_csv('sentence_vader_18.csv',index=False)
# import pandas as pd
# filename = 'sentence_vader_'
# df0 = pd.read_csv('sentence_vader_1.csv')
# for i in range(2,19):
#     file = filename + str(i)+'.csv'
#     df = pd.read_csv(file)
#     df0 = pd.concat((df0,df))
#
# df0.to_csv('sentence_vader_result.csv',index=False)
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,f1_score
# # df1 = pd.read_csv('vader_blob_res_1.csv')
# # df2 = pd.read_csv('vader_blob_res_2.csv')
# # df = pd.concat((df1,df2))
# # df.to_csv('vader_blob_res.csv',index=False)
# #
# # true_label = df['label'].values
# # va_pred = []
# # tb_pred = []
# # for idx,row in df.iterrows():
# #     va_score = 0 if row['vader_compound']<=0 else 1
# #     tb_score = 0 if row['textblob_score']<=0 else 1
# #     va_pred.append(va_score)
# #     tb_pred.append(tb_score)
# # va_pred = np.array(va_pred)
# # tb_pred = np.array(tb_pred)
# # res = open('vd_tb_res.txt','w')
# # class_names=['neg','pos']
# # print('AUC score for Vader is %.4f, for textblob is %.4f'%(roc_auc_score(true_label,va_pred),roc_auc_score(true_label,tb_pred)),file=res)
# # print('F1 score for Vader is %.4f, for textblob is %.4f'%(f1_score(true_label,va_pred),f1_score(true_label,tb_pred)),file=
# #       res)
# # print('Classification Report of Vader:\n',classification_report(true_label, va_pred, target_names=class_names),file=res)
# # print('Classification Report of Textblob:\n',classification_report(true_label, tb_pred, target_names=class_names),file=res)
# # def show_confusion_matrix(confusion_matrix):
# #     hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
# #     hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
# #     hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
# #     plt.ylabel('True sentiment')
# #     plt.xlabel('Predicted sentiment')
# #     plt.savefig('vader_cm.png')
# # cm = confusion_matrix(true_label, va_pred)
# # df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
# # show_confusion_matrix(df_cm)
# #
# # # def show_confusion_matrix(confusion_matrix):
# # #     plt.figure()
# # #     hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
# # #     hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
# # #     hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
# # #     plt.ylabel('True sentiment')
# # #     plt.xlabel('Predicted sentiment')
# # #     plt.savefig('textblob_cm.png')
# # # cm = confusion_matrix(true_label, tb_pred)
# # # df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
# # # show_confusion_matrix(df_cm)
# '''SnowNLP'''
# # from snownlp import SnowNLP
# # df_test =  pd.read_csv('test.csv')
# # snow_res = pd.DataFrame(columns=['review','label','score'])
# #
# # for idx,row in df_test.iterrows():
# #     review = row['review']
# #     label = row['label']
# #     score = 2*SnowNLP(review).sentiments-1
# #     # print(label,score)
# #     snow_res.loc[idx] = [review,label,score]
# #
# # snow_res.to_csv('snow_res.csv',index=False)
# snow_res = pd.read_csv('snow_res.csv')
# true = snow_res['label'].values
# print(true)
# pred = []
# for idx,row in snow_res.iterrows():
#     sc = row['score']
#     score = 0 if sc<0 else 1
#     pred.append(score)
# true = np.array(true)
# pred = np.array(pred)
# print('AUC is %.4f:'%roc_auc_score(true,pred))
# print('F1 is %.4f:'%f1_score(true,pred))


