"""
采用GolVe预训练词向量对输入的词序列编码
"""
import cProfile
import re
import numpy as np
import torch
import pandas as pd
import time
root_path = 'G:/MHwork/MHdata/NLP'
glove_path = 'G:/MHwork/MHdata/NLP/glove'
voca_path = glove_path + '/voca.100d.pth'
gloveTensor_path = glove_path + '/gloveTensor.100d.pth'
leaf_path = 'embdata'
# training data
train_question1_path = root_path + '/Quora-Question-Pairs/'+leaf_path+'/train_question1.pth'
train_seqlen1_Set_path = root_path + '/Quora-Question-Pairs/'+leaf_path+'/train_seqlen1_Set.pth'
train_question2_path = root_path + '/Quora-Question-Pairs/'+leaf_path+'/train_question2.pth'
train_seqlen2_Set_path = root_path + '/Quora-Question-Pairs/'+leaf_path+'/train_seqlen2_Set.pth'
train_label_Set_path = root_path + '/Quora-Question-Pairs/'+leaf_path+'/train_label_Set.pth'
#test data
test_question1_path = root_path + '/Quora-Question-Pairs/embdata/test_question1.pth'
test_seqlen1_Set_path = root_path + '/Quora-Question-Pairs/embdata/test_seqlen1_Set.pth'
test_question2_path = root_path + '/Quora-Question-Pairs/embdata/test_question2.pth'
test_seqlen2_Set_path = root_path + '/Quora-Question-Pairs/embdata/test_seqlen2_Set.pth'


def token2vector(data, gloveTensor, voca, isTrian_data=True):
    #UNK = torch.zeros(100) # unknown word, OOV
    voca_index = {} # 将出现过的词-索引存储在字典里，提高查询速度
    remove_chars = '''[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》?"".''！[\\]^_`{|}~]+'''
    question1_Set = data['question1'].tolist() #
    seqlen1_Set = []
    question2_Set = data['question2'].tolist()
    seqlen2_Set = []
    label_Set = []
    batch = len(question1_Set)
    #print("$$$$$$$$$$$$",question1_Set)
    #batch2 = len(question2_Set)
    #label_Set = data['is_duplicate']
    #batch3 = len(label_Set)
    #print('########',batch1,batch2, batch3)
    for i in range(batch):
        #print(i)
        try:
            seqlen1_Set.append(len(question1_Set[i].split(' ')))
        except Exception as e: #question1 is null
            seqlen1_Set.append(1) # null 长度记为1， 便于使用函数pack_padded_sequence
        #if question2_Set[i]:
        try:
            seqlen2_Set.append(len(question2_Set[i].split(' ')))
        #else:
        except Exception as e:
            seqlen2_Set.append(1)
    #print('********seqlen', seqlen1_Set)   
    max_len1 = max(seqlen1_Set)
    max_len2 = max(seqlen2_Set)
    max_len = max(max_len1, max_len2)
    data_tensor1 = torch.zeros(batch, max_len, 100)
    data_tensor2 = torch.zeros(batch, max_len, 100)
    for i in range(batch): # batch
        if seqlen1_Set[i] > 1:
            temp_question1_Set = re.sub(remove_chars, '', question1_Set[i])
            list_Phrase1_Set = temp_question1_Set.split(' ')
            for j in range(seqlen1_Set[i]): # seq_len
                #print('*********', list_Phrase1_Set[j])
                if list_Phrase1_Set[j] in voca_index.keys():
                    Index = voca_index[list_Phrase1_Set[j]]
                    data_tensor1[i,j,:] = gloveTensor[Index, :]
                else:
                    if list_Phrase1_Set[j] in voca:
                        Index = voca.index(list_Phrase1_Set[j])
                        voca_index[list_Phrase1_Set[j]] = Index
                        data_tensor1[i,j,:] = gloveTensor[Index, :]
                    elif str.isalpha(list_Phrase1_Set[j]):  # 全英文字母字符串
                        if list_Phrase1_Set[j].lower() in voca:
                            Index = voca.index(list_Phrase1_Set[j].lower())
                            voca_index[list_Phrase1_Set[j]] = Index
                            data_tensor1[i,j,:] = gloveTensor[Index, :]
                    else:
                        pass  #OOV

        if seqlen2_Set[i] > 1:
            temp_question2_Set = re.sub(remove_chars, '', question2_Set[i])
            list_Phrase2_Set = temp_question2_Set.split(' ')
            for k in range(seqlen2_Set[i]): # seq_len
                if list_Phrase2_Set[k] in voca_index.keys():
                    Index = voca_index[list_Phrase2_Set[k]]
                    data_tensor2[i,k,:] = gloveTensor[Index, :]
                else:
                    if list_Phrase2_Set[k] in voca:
                        Index = voca.index(list_Phrase2_Set[k])
                        voca_index[list_Phrase2_Set[k]] = Index
                        data_tensor2[i,k,:] = gloveTensor[Index, :]
                    elif str.isalpha(list_Phrase2_Set[k]):  # 全英文字母字符串
                        if list_Phrase2_Set[k].lower() in voca:
                            Index = voca.index(list_Phrase2_Set[k].lower())
                            voca_index[list_Phrase2_Set[k]] = Index
                            data_tensor2[i,k,:] = gloveTensor[Index, :]
                    else:
                        pass  #OOV
    #print(seqlen1_Set, seqlen2_Set)
    seqlen1_Set = torch.tensor(seqlen1_Set)
    seqlen2_Set = torch.tensor(seqlen2_Set)
    if isTrian_data:
        label_Set = data['is_duplicate']
        label_Set = torch.tensor(label_Set)
    return data_tensor1, seqlen1_Set, data_tensor2, seqlen2_Set, label_Set  # [batch, max_len, dim], [bath], [bath]
    

if __name__ == '__main__':
    time_start = time.time()
    train_data = pd.read_csv(root_path + '/Data/quora-question-pairs/train.csv', sep=',', nrows=5000)
    #print(train_data)
    
    #test_data = pd.read_csv(root_path + '/Data/quora-question-pairs/test.csv', sep=',')
    gloveTensor = torch.load(gloveTensor_path)
    #glove = pd.read_csv( glove_path + '/glove.6B.100d.txt', sep='\ ', header = None, index_col = 0,  engine='python')
    voca = torch.load(voca_path)
    time_end = time.time()
    print('load time cost:%fs'%(time_end-time_start))

    print('start embedding...')
    time_start = time.time()

    # train_data embedding
    #cProfile.run('token2vector(train_data, gloveTensor, voca, isTrian_data = True)')
    
    train_question1, train_seqlen1_Set, train_question2, train_seqlen2_Set, train_label_Set = token2vector(train_data, gloveTensor, voca, isTrian_data=True)
    torch.save(train_question1, train_question1_path)
    print(train_question1.size(), train_question1[0,:])
    torch.save(train_seqlen1_Set, train_seqlen1_Set_path)
    torch.save(train_question2, train_question2_path)
    torch.save(train_seqlen2_Set, train_seqlen2_Set_path)
    torch.save(train_label_Set, train_label_Set_path)

    time_end1 = time.time()
    print('train data embedding time cost:%fs'%(time_end1-time_start))
    #print(train_question1[0,:,:],train_question1.size())
    """
    # test_data embedding
    test_question1, test_seqlen1_Set, test_question2, test_seqlen2_Set, test_label_Set = token2vector(train_data, gloveTensor, voca, isTrian_data = False)
    torch.save(test_question1, test_question1_path)
    torch.save(test_seqlen1_Set, test_seqlen1_Set_path)
    torch.save(test_question2, test_question2_path)
    torch.save(test_seqlen2_Set, test_seqlen2_Set_path)
    print('end embedding.')

    time_end2 = time.time()
    print('test data embedding time cost:%fs'%(time_end2-time_end1))
    """
    
    

    
