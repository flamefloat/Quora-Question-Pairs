import numpy as np
import torch
import pandas as pd

glove_path = 'G:/MHwork/MHdata/NLP/glove'
gloveTensor_path = glove_path + '/gloveTensor.100d.pth'
voca_path = glove_path + '/voca.100d.pth'
voca = torch.load(voca_path)
num = len(voca)
print(num, voca[0:5])
"""
glove = pd.read_csv( glove_path + '/glove.6B.100d.txt', sep='\ ', header = None, index_col = 0,  engine='python')
#print(glove)
#print(glove[1].tolist())
gloveTensor = torch.zeros(num,100)
for i in range(100):
    gloveTensor[:,i] = torch.tensor(glove[i+1].tolist())
"""
gloveTensor = torch.load(gloveTensor_path)
print(gloveTensor.size())
#torch.save(gloveTensor, gloveTensor_path)
#print(glove.head(3))
#print(gloveTensor[1:3,:])
print(voca.index('what'))
print(str.isalpha('asdfDD-J'))