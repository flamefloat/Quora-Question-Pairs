import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
"""
root_path = 'G:/MHwork/MHdata/NLP'
test_data = pd.read_csv(root_path + '/Data/quora-question-pairs/test.csv',sep=',', nrows=20)
print(test_data, "********\n",len(test_data['test_id']))
a = [0,1,2,3,4,5,6,7,8,9,10]
for i in range(0,10,3):
    print(a[i:i+3])

predict = torch.randn(4,2)
print(predict)
predict = F.softmax(predict, dim=1)
print(predict)
predict = predict[:,1]
print(predict)

root_path = 'G:/MHwork/MHdata/NLP'
for epoch in range(10):
    a = torch.randn(2,3)
    MODEL_PATH = root_path + "/Quora-Question-Pairs/model/model" + str(epoch) + ".pth"
    torch.save(a,MODEL_PATH)

optimizer = optim.Adam(, lr=0.1)
a = optimizer.learning_rate
print(a)

root_path = 'G:/MHwork/MHdata/NLP'
test_data = pd.read_csv(root_path + '/Data/quora-question-pairs/11.csv',sep='\t', nrows=10)
a = test_data[0:5]['question1'].tolist()
print(a[1].split(' '))
"""
a = torch.randn(6,3)
print(a, a[-4:-1,:])