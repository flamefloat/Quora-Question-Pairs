import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from GloVeEmb import root_path, train_question1_path, train_question2_path, train_seqlen1_Set_path, train_seqlen2_Set_path, train_label_Set_path
"""
TODO list
1、多卡训练 
2、学习率 warm-up
3、dropout
4、训练不收敛：
    数据整理有误？重新嵌入小数据集训练 YES
5、数据过拟合：
    epoch过大
    引入输入dropout
    
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
question_x_set_path = root_path + "/Quora-Question-Pairs/embdata/train_question1.pth"
question_y_set_path = root_path + "/Quora-Question-Pairs/embdata/train_question2.pth"
question_x_len_set_path = root_path + "/Quora-Question-Pairs/embdata/train_seqlen1_Set.pth"
question_y_len_set_path = root_path + "/Quora-Question-Pairs/embdata/train_seqlen2_Set.pth"
label_path = root_path + "/Quora-Question-Pairs/embdata/train_label_Set.pth"
"""
MODEL_PATH = root_path + "/Quora-Question-Pairs/latent_task_model/model.pth"

#model parameters
input_size = 100 #embedding_size
hidden_size = 256
num_layers = 3
n_class = 2 #is or not
epochs = 30

class QQmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_class):
        super(QQmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_class = n_class
        self.dropout = nn.Dropout(p=0.3)
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=True, batch_first=True, dropout=0.3)
        self.reduce_dim1 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size)) # reduce dim for multi-head attention
        self.reduce_dim2 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size))
        self.reduce_dim3 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size))
        self.hidden_out = nn.Linear(self.hidden_size * 3, self.n_class)
        self.out = nn.Linear(self.n_class * 2, self.n_class)

    def get_att_weight(self, encoder1_hidden_states, encoder2_hidden_state): # softmax(Q * K^T)
        #n_batch = encoder1_hidden_states.size(0)
        #seq_len = encoder1_hidden_states.size(1) #[batch, seq_len, hidden_size]
        d_k = encoder1_hidden_states.size(-1)
        #attn_scores = torch.zeros(n_batch, seq_len)
        #[batch, 1, hidden_size] * [batch, hidden_size, seq_len] = [batch, 1, seq_len]
        attn_scores = torch.matmul(encoder2_hidden_state.unsqueeze(1), encoder1_hidden_states.transpose(1, 2)) \
            /math.sqrt(d_k) # Scaled Dot Product Attention
        attn_scores = F.softmax(attn_scores, dim=2) #[batch, 1, seq_len]
        return attn_scores

    
    def get_att_context(self, encoder1_hidden_states, encoder2_hidden_state, reduce_dim):
        encoder1_states = reduce_dim(encoder1_hidden_states)
        encoder2_state = reduce_dim(encoder2_hidden_state)
        attn_scores = self.get_att_weight(encoder1_states, encoder2_state)
        att_context = torch.matmul(attn_scores, encoder1_states)
        att_context = att_context.squeeze(1) #[batch, mini_hidden_size]
        #att_context = torch.cat((att_context, encoder2_state), dim=1)
        return att_context, encoder2_state

    def get_full_att_context(self, encoder1_hidden_states, encoder2_hidden_state, reduce_dim1, reduce_dim2, reduce_dim3):
        """
        concat multi-head attention
        """
        att_context1, encoder2_state1 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, reduce_dim1)
        att_context2, encoder2_state2 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, reduce_dim2)
        att_context3, encoder2_state3 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, reduce_dim3)
        #print(att_context1.size(),encoder2_hidden_state.size())
        full_att_context = torch.cat((att_context1, att_context2, att_context3, encoder2_state1, encoder2_state2, encoder2_state3), dim=1)
        return full_att_context

    def forward(self, x, y, seqlen_x, seqlen_y):
        x = nn.utils.rnn.pack_padded_sequence(x, seqlen_x, batch_first=True, enforce_sorted=False)
        y = nn.utils.rnn.pack_padded_sequence(y, seqlen_y, batch_first=True, enforce_sorted=False)
        encoder1, (h_nx, c_nx) = self.encoder(x)
        encoder2, (h_ny, c_ny) = self.encoder(y)
        #output.view(seq_len, batch, num_directions, hidden_size)
        #encoder1 = encoder1.view(encoder1.size(0), encoder1.size(1), 2, self.hidden_size)
        #encoder2 = encoder2.view(encoder2.size(0), encoder2.size(1), 2, self.hidden_size) 
        encoder1 = nn.utils.rnn.pad_packed_sequence(encoder1, batch_first=True, padding_value=0.0)[0]
        encoder2 = nn.utils.rnn.pad_packed_sequence(encoder2, batch_first=True, padding_value=0.0)[0]
        #encoder1 = torch.cat((encoder1[:,:,0,:], encoder1[:,:,1,:]), dim=2)
        #encoder2 = torch.cat((encoder2[:,:,0,:], encoder2[:,:,1,:]), dim=2)
        #[num_layers * num_directions, batch, hidden_size]
        h_nx = h_nx.view(self.num_layers, 2, h_nx.size(1),self.hidden_size)
        h_nx = torch.cat((h_nx[2,0,:,:], h_nx[2,1,:,:]), dim=1)
        h_ny = h_ny.view(self.num_layers, 2, h_ny.size(1),self.hidden_size)
        h_ny = torch.cat((h_ny[2,0,:,:], h_ny[2,1,:,:]), dim=1)
        full_att_context_x = self.get_full_att_context(encoder2, h_nx, self.reduce_dim1, self.reduce_dim2, self.reduce_dim3)
        full_att_context_y = self.get_full_att_context(encoder1, h_ny, self.reduce_dim1, self.reduce_dim2, self.reduce_dim3)
        output1 = self.dropout(full_att_context_x)
        output2 = self.dropout(full_att_context_y)
        latent_output1 = self.hidden_out(output1)
        latent_output2 = self.hidden_out(output2)
        #real_output1 = F.relu(latent_output1)
        #real_output2 = F.relu(latent_output2)
        output = self.out(torch.cat((latent_output1, latent_output2), dim=1))
        return output, latent_output1, latent_output2

def trainModel(model, dataSet_loader, verify_question_x_set, verify_question_y_set, verify_seqlen_x_set, verify_seqlen_y_set, verify_label):
    step = 0
    warm_step = 2000
    warm_up = math.pow(warm_step, -1.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=warm_up)
    #lambda1 = lambda step: 0.2*min(math.pow(step, -0.5), step*warm_up)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    time_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        for i, train_data in enumerate(dataSet_loader):
            step += 1
            optimizer.param_groups[0]["lr"] = 0.2*min(math.pow(step, -0.5), step*warm_up)
            #optimizer.set_learning_rate(learningRate)
            question_x_set, question_y_set, seqlen_x_set, seqlen_y_set, label = train_data
            question_x_set = question_x_set.to(DEVICE)
            question_y_set = question_y_set.to(DEVICE)
            seqlen_x_set = seqlen_x_set.to(DEVICE)
            seqlen_y_set = seqlen_y_set.to(DEVICE)
            label = label.to(DEVICE)
            predict, latent_label1, latent_label2 = model(question_x_set, question_y_set, seqlen_x_set, seqlen_y_set)
            train_loss = criterion(predict, label)
            latent_loss1 = criterion(latent_label1, label)
            latent_loss2 = criterion(latent_label2, label)
            multi_loss = train_loss + latent_loss1 + latent_loss2
            if (i+1)%200 == 0:
                time_end = time.time()
                print('Epoch:', '%04d' % (epoch + 1), ', batch:', '%04d' % (i + 1), ', learningRate=%f'%optimizer.param_groups[0]["lr"], ', loss={:.4f}, loss1={:.4f},loss2={:.4f}'.format(train_loss, latent_loss1, latent_loss2),', %d Question-Pairs/s'%((200*256)/(time_end-time_start)))
                time_start = time.time()
            optimizer.zero_grad()
            multi_loss.backward()
            optimizer.step()
            #scheduler.step()
        epoch_end = time.time()
        print('training speed:%ds/epoch'%(epoch_end-epoch_start))
        with torch.no_grad():
            verify_predict, _, _ = model(verify_question_x_set, verify_question_y_set, verify_seqlen_x_set, verify_seqlen_y_set)
            verify_loss = criterion(verify_predict, verify_label)
            print('verify_loss is ', verify_loss)
        if epoch > 10 and (epoch%2) == 0:
            trained_model = {'net': model.state_dict()}
            MODEL_PATH = root_path + "/Quora-Question-Pairs/latent_task_model/model" + str(epoch) + ".pth"
            torch.save(trained_model, MODEL_PATH)
            print('Trained neural network model has been saved\n')
    
    print('end of training\n')

if __name__ == "__main__":
    model = QQmodel(input_size, hidden_size, num_layers, n_class).to(DEVICE)
    #model = nn.DataParallel(model, device_ids=[0,1])
    model.train()
    print('loading data')
    time_start = time.time()
    question_x_set = torch.load(train_question1_path)   
    question_y_set = torch.load(train_question2_path)
    seqlen_x_set = torch.load(train_seqlen1_Set_path)
    seqlen_y_set = torch.load(train_seqlen2_Set_path)
    label = torch.load(train_label_Set_path)
    time_end = time.time()
    print('load data cost %ds'%(time_end - time_start))

    verify_question_x_set = question_x_set[-50000:-1,:]
    verify_question_y_set = question_y_set[-50000:-1,:]
    verify_seqlen_x_set = seqlen_x_set[-50000:-1,:]
    verify_seqlen_y_set = seqlen_y_set[-50000:-1,:]
    verify_label = label[-50000:-1,:]
    

    dataSet = TensorDataset(question_x_set[0:-50000,:], question_y_set[0:-50000,:], seqlen_x_set[0:-50000,:], seqlen_y_set[0:-50000,:], label[0:-50000,:])
    dataSet_loader = DataLoader(dataSet, batch_size=256, shuffle=True)
    trainModel(model, dataSet_loader, verify_question_x_set, verify_question_y_set, verify_seqlen_x_set, verify_seqlen_y_set, verify_label)
    

