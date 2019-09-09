import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
"""
todo list
1、多卡训练
2、
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_x_set_path = root_path + "/traindataSet/question_x_set.pth"
question_y_set_path = root_path + "/traindataSet/question_y_set.pth"
question_x_len_set_path = root_path + "/traindataSet/question_x_len_set.pth"
question_y_len_set_path = root_path + "/traindataSet/question_y_len_set.pth"
label_path = root_path + "/traindataSet/label.pth"

#model parameters
input_size = 100 #embedding_size
hidden_size = 128
num_layers = 3
n_class = 2 #is or not

class QQmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_class):
        super(QQmodel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_class = n_class
        self.dropout = nn.Dropout(p=0.5)
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirection = True, batch_first = True, dropout = 0.5)
        self.reduce_dim1 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size)) # reduce dim for multi-head attention
        self.reduce_dim2 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size))
        self.reduce_dim3 = nn.Linear(hidden_size * 2, int(0.5 * hidden_size))
        self.out = nn.Linear(self.hidden_size * 4, self.n_class)

    def get_att_weight(self, encoder1_hidden_states, encoder2_hidden_state): # softmax(Q * K^T)
        n_batch = encoder_hidden_states.size(0)
        seq_len = encoder_hidden_states.size(1) #[batch, seq_len, hidden_size]
        d_k = encoder_hidden_states.size(-1)
        attn_scores = torch.zeros(n_batch, seq_len)
        #[batch, 1, hidden_size] * [batch, hidden_size, seq_len] = [batch, 1, seq_len]
        attn_scores = torch.matmul(encoder2_hidden_state, encoder1_hidden_state.transpose(1, 2)) \
            /math.sqrt(d_k) # Scaled Dot Product Attention
        return F.softmax(attn_scores, dim = 1) #[batch, 1, seq_len]
    
    def get_att_context(self, encoder1_hidden_states, encoder2_hidden_state, reduce_dim):
        encoder1_states = reduce_dim(encoder1_hidden_states)
        encoder2_state = reduce_dim(encoder2_hidden_state)
        attn_scores = self.get_att_weight(encoder1_states, encoder2_state)
        att_context = torch.matmul(attn_scores, encoder1_states)
        att_context = att_context.squeeze(1) #[batch, mini_hidden_size]
        #att_context = torch.cat((att_context, encoder2_state), dim=1)
        return att_context

    def get_full_att_context(self, encoder1_hidden_states, encoder2_hidden_state, self.reduce_dim1, self.reduce_dim2, self.reduce_dim3):
        """
        concat multi-head attention
        """
        att_context1 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, self.reduce_dim1)
        att_context2 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, self.reduce_dim2)
        att_context3 = self.get_att_context(encoder1_hidden_states, encoder2_hidden_state, self.reduce_dim3)
        full_att_context = torch.cat((att_context1, att_context2, att_context3, encoder2_hidden_state), dim=1)
        return full_att_context

    def forward(self, x, y, seqlen_x, seqlen_y):
        x = nn.utils.rnn.pack_padded_sequence(x, seqlen_x, batch_first = True, enforce_sorted = False)
        y = nn.utils.rnn.pack_padded_sequence(x, seqlen_y, batch_first = True, enforce_sorted = False)
        encoder1, (h_nx, c_nx)= self.encoder(x)
        encoder2, (h_ny, c_ny)= self.encoder(y)
        h_nx = h_nx.view(self.num_layers, 2, h_nx.size(1),self.hidden_size)
        h_nx = torch.cat((h_nx[2,0,:,:], h_nx[2,1,:,:]), dim=1)
        full_att_context_x = self.get_full_att_context(encoder2, h_nx, self.reduce_dim1, self.reduce_dim2, self.reduce_dim3)
        full_att_context_y = self.get_full_att_context(encoder1, h_ny, self.reduce_dim1, self.reduce_dim2, self.reduce_dim3)
        output = self.dropout(torch.cat((full_att_context_x, full_att_context_y), dim=1))
        output = self.out(output)
        return output

def trainModel(model, dataSet_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    time_start = time.time()
    for epoch in range(epochs):
        for i, train_data in enumerate(dataSet_loader):
            train_data = train_data.to(DEVICE)
            question_x_set, question_y_set, seqlen_x_set, seqlen_y_set, label = train_data
            predict = model(question_x_set, question_y_set, seqlen_x_set, seqlen_y_set)
            train_loss = criterion(predict, label)
            if (i+1)%200 == 0:
                time_end = time.time()
                print('Epoch:', '%04d' % (epoch + 1), 'batch:', '%04d' % (i + 1),'loss =', '{:.6f}'.format(train_loss),'%d Phrases/s'%((200*256)/(time_end-time_start)))
                time_start = time.time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    trained_model = {'net': model.state_dict()}
    print('end of training\n')
    torch.save(trained_model, MODEL_PATH)
    print('Trained neural network model has been saved\n')

if __name__ == "__main__":
    model = QQmodel(input_size, hidden_size, num_layers, n_class).to(DEVICE)
    model.train()
    question_x_set = torch.load(question_x_set_path)
    question_y_set = torch.load(question_y_set_path)
    seqlen_x_set = torch.load(question_x_len_set_path)
    seqlen_y_set = torch.load(question_y_len_set_path)
    label = torch.load(label_path)
    dataSet = TensorDataset(question_x_set, question_y_set, seqlen_x_set, seqlen_y_set, label)
    dataSet_loader = DataLoader(dataSet, batch_size=256, shuffle=True)
    trainModel(model, dataSet_loader)


