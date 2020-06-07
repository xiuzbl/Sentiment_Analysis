#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
warnings.filterwarnings('ignore')


# In[2]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(len(tokenizer.vocab))


# In[4]:


tokens = tokenizer.tokenize('你在干什么')
indexes = tokenizer.convert_tokens_to_ids(tokens)
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)


# In[5]:


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


# In[6]:


df = pd.read_csv('train.csv')
df.head()


# In[7]:


token_lens = []
for txt in df.review:
    tokens = tokenizer.encode(str(txt), max_length=512)
    token_lens.append(len(tokens))
sns.distplot(token_lens)
plt.xlim([0, 512])
plt.xlabel('Token count')


# In[8]:


MAX_LEN = 256
BATCH_SIZE = 32


# ## Create dataset

# In[9]:


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


# In[10]:


df_train, df_val = train_test_split(df,test_size=0.1,random_state=SEED)


# ### Create Data Loaders

# In[11]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.review.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4)


# In[12]:


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)


# In[14]:


bert_model = BertModel.from_pretrained('bert-base-chinese')


# In[15]:


sample_txt = '这家酒店太拥挤'
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)
last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask'])


# ## Build BERT model

# In[16]:


class Sentiment(nn.Module):
    def __init__(self):
        super(Sentiment, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# In[17]:


model = Sentiment()
model = model.to(device)
data = next(iter(train_data_loader))
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[18]:


optimizer = AdamW(model.parameters(),lr=2e-5,correct_bias=False)
criterion = nn.CrossEntropyLoss().to(device)
EPOCHS=20
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
# # def binary_accuracy(preds, y):
# #     """
# #     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
# #     """

# #     #round predictions to the closest integer
# #     rounded_preds = torch.round(torch.sigmoid(preds[:,1]))
# #     correct = (rounded_preds == y).float() #convert into float for division 
# #     acc = correct.sum() / len(correct)
#     return acc


# In[19]:


def train_epoch(model,
                data_loader,
                criterion,
                optimizer,
                device,
                scheduler,
                n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        preds = F.softmax(outputs,dim=1)
        _, preds = torch.max(preds, dim=1)
        # print(outputs.shape,targets.shape)

        loss = criterion(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
#         acc = binary_accuracy(outputs,targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return correct_predictions.double() / n_examples, np.mean(losses)


# In[20]:


def eval_model(model,data_loader, criterion,device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
    #         _, preds = torch.max(outputs, dim=1)
            preds = F.softmax(outputs,dim=1)
            _, preds = torch.max(preds, dim=1)
            # print(outputs.shape,targets.shape)
#             loss = criterion(outputs[:][1], targets.reshape((32,1)))
            loss = criterion(outputs,targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# In[21]:


history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        criterion,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        criterion,
        device,
        len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}','\n')
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_acc


# In[ ]:
process = open('train_acc.txt','w')
val_process = open('val_acc.txt','w')
print(history['train_acc'],file=process)
print(history['val_acc'],file=val_process)

# fig = plt.figure()
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.xlim([0,20])
plt.ylim([0.80, 1])
plt.savefig('bert_tr_ev.png')


# In[ ]:




