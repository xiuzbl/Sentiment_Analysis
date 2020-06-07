# import torch
# from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import warnings
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import confusion_matrix, classification_report
# from collections import defaultdict
# from textwrap import wrap
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# warnings.filterwarnings('ignore')
# # from Bert_sentiment_analysis import create_data_loader
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# class Sentiment(nn.Module):
#     def __init__(self,bert_model):
#         super(Sentiment, self).__init__()
#         self.bert = bert_model
#         self.drop = nn.Dropout(p=0.25)
#         self.out = nn.Linear(self.bert.config.hidden_size, 2)
#     def forward(self, input_ids, attention_mask):
#         _, pooled_output = self.bert(
#           input_ids=input_ids,
#           attention_mask=attention_mask
#         )
#         output = self.drop(pooled_output)
#         return self.out(output)
#
# bert_model = BertModel.from_pretrained('bert-base-chinese')
# best_model = Sentiment(bert_model).to(device)
# best_model.load_state_dict(torch.load('best_model.bin'))
# # best_model.eval()
#
# class GPReviewDataset(Dataset):
#     def __init__(self, reviews, targets, tokenizer, max_len):
#         self.reviews = reviews
#         self.targets = targets
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.reviews)
#
#     def __getitem__(self, item):
#         review = str(self.reviews[item])
#         target = self.targets[item]
#         encoding = self.tokenizer.encode_plus(
#             review,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             return_token_type_ids=False,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#         return {
#             'review': review,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'targets': torch.tensor(target, dtype=torch.long)
#         }
#
#
#
# # Create Data Loaders
#
# def create_data_loader(df, tokenizer, max_len, batch_size):
#     ds = GPReviewDataset(
#         reviews=df.review.to_numpy(),
#         targets=df.label.to_numpy(),
#         tokenizer=tokenizer,
#         max_len=max_len)
#
#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         num_workers=4)
#
#
# def get_predictions(model, data_loader):
#     model = model.eval()
#     review_texts = []
#     predictions = []
#     prediction_probs = []
#     real_values = []
#     pred_scores = []
#     with torch.no_grad():
#         for d in data_loader:
#             texts = d["review"]
#             input_ids = d["input_ids"].to(device)
#             attention_mask = d["attention_mask"].to(device)
#             targets = d["targets"].to(device)
#             outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#             )
#
#             outputs = F.softmax(outputs, dim=1)
#             max_preds, idx_preds = torch.max(outputs, dim=1)
#             scores = 2 * (outputs[:,1]) - 1
#             pred_scores.extend(scores)
#             # _, preds = torch.max(outputs, dim=1)
#             review_texts.extend(texts)
#             predictions.extend(idx_preds)
#             prediction_probs.extend(outputs)
#             real_values.extend(targets)
#     predictions = torch.stack(predictions).cpu()
#     prediction_probs = torch.stack(prediction_probs).cpu()
#     real_values = torch.stack(real_values).cpu()
#     # predictions = torch.stack(predictions)
#     # prediction_probs = torch.stack(prediction_probs)
#     # real_values = torch.stack(real_values)
#     pred_scores = torch.stack(pred_scores).cpu()
#     return review_texts, predictions, prediction_probs, real_values, pred_scores
#
# MAX_LEN = 256
# BATCH_SIZE = 32
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# df_test = pd.read_csv('test.csv')
# test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
#
# y_review_texts, y_pred, y_pred_probs, y_test,scores = get_predictions(
#   best_model,
#   test_data_loader
# )
# class_names = ['neg','pos']
# res_df = pd.DataFrame(columns=['review','pred','score','label'])
# res_df['review'] = y_review_texts
# res_df['pred'] = y_pred
# res_df['label'] = y_test
# res_df['score'] = scores
# print(res_df)
# res_df.to_csv('bert_result.csv',index=False)
# bert_pre = open('bert_pred_insurance.txt','w')
# print('Classification Report:',classification_report(y_test, y_pred, target_names=class_names),file=bert_pre)
#
# def show_confusion_matrix(confusion_matrix):
#     hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
#     hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#     hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#     plt.ylabel('True sentiment')
#     plt.xlabel('Predicted sentiment')
#     plt.savefig('cm.png')
# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
# show_confusion_matrix(df_cm)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,f1_score
import pandas as pd
import numpy as np
res = pd.read_csv('bert_result.csv')
true = res['label']
pred = []
be_res = open('be_res.txt','w')
for idx,row in res.iterrows():
    score = row['score']
    pre = 0 if score<0 else 1
    pred.append(pre)
true = np.array(true)
pred = np.array(pred)
print('AUC is %.4f'%roc_auc_score(true,pred),file=be_res)
print('F1 is %.4f'%f1_score(true,pred),file=be_res)

