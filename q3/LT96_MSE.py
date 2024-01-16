import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
import warnings
from torch.utils.data import DataLoader, TensorDataset
import math
import torch.optim as optim
from datetime import datetime
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子

setup_seed(60)
warnings.filterwarnings("ignore")
file_path = 'ETTh1.csv'
device=torch.device('cuda:2')
date = pd.read_csv(file_path,usecols=["date"])
hufl = pd.read_csv(file_path,usecols=["HUFL"])
hull = pd.read_csv(file_path,usecols=["HULL"])
mufl = pd.read_csv(file_path,usecols=["MUFL"])
mull = pd.read_csv(file_path,usecols=["MULL"])
lufl = pd.read_csv(file_path,usecols=["LUFL"])
lull = pd.read_csv(file_path,usecols=["LULL"])
ot = pd.read_csv(file_path,usecols=["OT"])
#标签

scaler = MinMaxScaler()

#日期数据
def hours_of_year(date_string,hours):
    # 将字符串解析为datetime对象
    dt_object = datetime.strptime(date_string, "%m-%d")
    # 提取月份和日
    month = dt_object.month
    day = dt_object.day
    # 定义每个月的天数
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # 如果是闰年，二月份天数加一
    if dt_object.year % 4 == 0 and (dt_object.year % 100 != 0 or dt_object.year % 400 == 0):
        days_in_month[2] += 1
    # 计算是该年中的第几天
    day_number = sum(days_in_month[:month]) + day
    hours=day_number*24+hours
    if hours==0:
        hours=0.5
    return hours
date=pd.to_datetime(date.date, infer_datetime_format=True)
m_date=date.dt.strftime('%m-%d %H:%M:%S')
date_list=[]
hour_list=[]
for i in m_date:
    date_list.append(hours_of_year(i[:5],int(i[6:8])))
    hour_list.append(int(i[6:8]))

data = pd.concat([pd.Series(date_list),pd.Series(hour_list),hufl,hull,mufl,mull,lufl,lull,ot], axis=1)
data.columns = data.columns.astype(str)
data_normalized = scaler.fit_transform(data)
# 对整个 DataFrame 进行归一化
data_normalized = scaler.fit_transform(data)

# 将归一化后的数据转为 DataFrame
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)
# 制作时间窗口
def create_sequences(data,seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length*2):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length*2]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
seq_length=96
data_normalized,target_normalized=create_sequences(data_normalized,seq_length)
#print(data_normalized.shape,target_normalized.shape)

# 分割数据集为训练和测试
train_size = int(len(list(data_normalized)) * 0.6)
test_size = len(list(data_normalized)) - train_size
val_size = int(test_size*0.5)
test_size = test_size-val_size
train_data, test_data = data_normalized[0:train_size,:], data_normalized[train_size:train_size+val_size,:]
train_target, test_target = target_normalized[0:train_size,:], target_normalized[train_size:train_size+val_size,:]
val_data=data_normalized[train_size+val_size:,:]
val_target=target_normalized[train_size+val_size:,:]

# 构建PyTorch数据集
train_tensor = torch.FloatTensor(train_data).to(device)
train_target_tensor = torch.FloatTensor(train_target).to(device)

test_tensor = torch.FloatTensor(test_data).to(device)
test_target_tensor = torch.FloatTensor(test_target).to(device)

val_tensor = torch.FloatTensor(val_data).to(device)
val_target_tensor = torch.FloatTensor(val_target).to(device)


train_dataset = TensorDataset(train_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_dataset = TensorDataset(val_tensor, val_target_tensor)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self,  input_dim_L, hidden_dim_L, num_layers_L, output_dim_L,input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.lstm=nn.LSTM(input_dim_L, hidden_dim_L, num_layers_L, batch_first=True)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(d_model, output_dim)
    def forward(self, src):
        src, (hn, cn) =self.lstm(src)
        #print(src.shape)
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = self.linear(output)
        #print(output.shape)
        return output

input_dim_L = 9  # 输入的特征数量
hidden_dim_L = 4  # 隐藏层的数量
num_layers_L = 1  # LSTM层数
output_dim_L = 2  # 输出的维度


# 实例化模型
input_dim = hidden_dim_L  # 输入维度
output_dim = 9  # 输出维度
d_model = 128  # 模型维度
nhead = 4  # 头数
num_layers = 4  # 层数
dim_feedforward = 216  # 前馈网络维度
dropout = 0.3  # dropout比例
model_path = "LT_model96MSE.pth"

model = TransformerModel(input_dim_L,hidden_dim_L,num_layers_L,output_dim_L,input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward,dropout=0.1).to(device)
# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='mean').to(device)

# 训练模型
def train_model(model, optimizer, criterion, train_loader, val_loader, test_tensor,test_target_tensor,epochs):
    if not os.path.exists(model_path):
        for epoch in range(epochs):
            model.train()
            running_loss=0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/ len(train_loader)}')
        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss / len(val_loader)}')
        model.cpu()
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            running_loss=0.0
            val_loss=0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            for i, data in enumerate(val_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            predictions = model(test_tensor).to(device)
            print('训练集损失：',running_loss/ len(train_loader),'验证集损失：',val_loss/ len(val_loader),'测试集损失：'
                ,criterion(test_target_tensor,predictions).item())
            test_date=date[train_size+val_size:train_size+val_size+2*seq_length]
            
            s=200 #预测哨兵
            a1=test_tensor[s].reshape((96,9)).cpu()
            b1=test_tensor[s+97].reshape((96,9)).cpu()
            a2=predictions[s].cpu()
            variable=8 #展示预测的是哪个变量,8-ot
            c1=list(a1[:,variable])+list(b1[:,variable])
            c2=list(a1[:,variable])+list(a2[:,variable].cpu().detach().numpy())
            plt.switch_backend('agg')
            plt.plot(test_date.index-train_size-val_size,c1 , label='Original', alpha=0.7)
            plt.plot(test_date.index-train_size-val_size,c2 , label='Predictions', alpha=0.7)
            plt.savefig("LT-MSE-96.jpg")
            plt.show()

train_model(model, optimizer, criterion, train_loader, val_loader,test_tensor,test_target_tensor,epochs=300)
