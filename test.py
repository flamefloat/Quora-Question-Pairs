import math
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
#import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import DataLoader, TensorDataset
#import matplotlib.pyplot as plt
from GloVeEmb import root_path, token2vector, gloveTensor_path, voca_path
from GloVeEmb import train_question1_path, train_question2_path, train_seqlen1_Set_path, train_seqlen2_Set_path, train_label_Set_path
from trainModel import QQmodel 

MODEL_PATH = root_path + "/Quora-Question-Pairs/latent_task_model/model22.pth"
test_sentiment_path = root_path + "/Quora-Question-Pairs/multi_task22_result.csv"
#criterion = nn.CrossEntropyLoss()

print('loading data and model...')
test_data = pd.read_csv(root_path + '/Data/quora-question-pairs/test.csv', sep=',')
voca = torch.load(voca_path)
gloveTensor = torch.load(gloveTensor_path)
model_param = torch.load(MODEL_PATH)
trainedModel = QQmodel(100, 256, 3, 2)
trainedModel.load_state_dict(model_param['net'])
trainedModel.eval()

""""
question_x_set = torch.load(train_question1_path)
question_y_set = torch.load(train_question2_path)
seqlen_x_set = torch.load(train_seqlen1_Set_path)
seqlen_y_set = torch.load(train_seqlen2_Set_path)
print('******')
label = torch.load(train_label_Set_path)
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    predict, latent_label1, latent_label2 = trainedModel(question_x_set[0:5000], question_y_set[0:5000], seqlen_x_set[0:5000], seqlen_y_set[0:5000])
    train_loss = criterion(predict, label[0:5000])
print("train loss is:", train_loss)
"""

print('start to inference...')
question_len = len(test_data['test_id'])
print('test data length is ', question_len)
result = []
batch = 100000
for i in range(0, question_len, batch):
    batch_start = time.time()
    mini_test_data = test_data[i:i+batch]
    TEST_ID = mini_test_data['test_id'].tolist()
    question1, seqlen1, question2, seqlen2, nulllabel = token2vector(mini_test_data, gloveTensor, voca, isTrian_data=False)
    with torch.no_grad():
        predict, _, _ = trainedModel(question1, question2, seqlen1, seqlen2)
        predict = F.softmax(predict, dim=1)
        predict = predict[:, 1]
    for j in range(len(mini_test_data)):  # 最后一组数据长度可能不够一个batch
        temp = {}
        temp['test_id'] = TEST_ID[j]  # test_id,is_duplicate
        temp['is_duplicate'] = predict[j].item()
        result.append(temp)
    batch_end = time.time()
    print("%d Question-Pairs/s"%(batch/(batch_end-batch_start)))
print('saved data length is ', len(result))
df = pd.DataFrame(result, columns=['test_id', 'is_duplicate'])
df.to_csv(test_sentiment_path, index=False)
"""

Result = pd.read_csv(root_path + "/Quora-Question-Pairs/multi_task_result.csv", sep=',')
ids = Result['test_id']
is_d = Result['is_duplicate']
result = []
for j in range(len(ids)):  # 最后一组数据长度可能不够一个batch
    temp = {}
    temp['test_id'] = ids[j]  # test_id,is_duplicate
    temp['is_duplicate'] = round(is_d[j], 4)
    result.append(temp)
    #print("%d Question-Pairs/s"%(batch/(batch_end-batch_start)))
print('saved data length is ', len(result))
df = pd.DataFrame(result, columns=['test_id', 'is_duplicate'])
df.to_csv(test_sentiment_path, index=False)
"""