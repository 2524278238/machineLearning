import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
file_path = 'ETTh1.csv'
device=torch.device('cuda')
date = pd.read_csv(file_path,usecols=["date"])
hufl = pd.read_csv(file_path,usecols=["HUFL"])
hull = pd.read_csv(file_path,usecols=["HULL"])
mufl = pd.read_csv(file_path,usecols=["MUFL"])
mull = pd.read_csv(file_path,usecols=["MULL"])
lufl = pd.read_csv(file_path,usecols=["LUFL"])
lull = pd.read_csv(file_path,usecols=["LULL"])
ot = pd.read_csv(file_path,usecols=["OT"])
#标签

date=pd.to_datetime(date.date, infer_datetime_format=True)
print(type(date))

scaler = MinMaxScaler()
data = pd.concat([hufl,hull,mufl,mull,lufl,lull,ot], axis=1)
data_normalized = scaler.fit_transform(data)
# 制作时间窗口
def create_sequences(data,seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length*2):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length*2]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
seq_length=336
data_normalized,target_normalized=create_sequences(data_normalized,seq_length)

# 分割数据集为训练和测试
train_size = int(len(list(data_normalized)) * 0.6)
test_size = len(list(data_normalized)) - train_size
val_size = int(test_size*0.5)
test_size = test_size-val_size
train_data, val_data = data_normalized[0:train_size,:], data_normalized[train_size:train_size+val_size,:]
train_target, val_target = target_normalized[0:train_size,:], target_normalized[train_size:train_size+val_size,:]
test_data=data_normalized[train_size+val_size:,:]
test_target=target_normalized[train_size+val_size:,:]

# 构建PyTorch数据集
train_tensor = torch.FloatTensor(train_data).to(device)
train_target_tensor = torch.FloatTensor(train_target).to(device)

test_tensor = torch.FloatTensor(test_data).to(device)
test_target_tensor = torch.FloatTensor(test_target).to(device)

val_tensor = torch.FloatTensor(val_data).to(device)
val_target_tensor = torch.FloatTensor(val_target).to(device)
model_path = "lstm5_model336h1.pth"

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*seq_length, output_dim*seq_length)

    def forward(self, x):

        out, (hn, cn) = self.lstm(x)
        out = self.linear(out.reshape(x.size(0),-1))
        out=out.reshape(x.shape)
        #print(out.shape)
        return out
def denormalize(normalized, min_value, max_value):
    return normalized * (max_value - min_value) + min_value

input_dim = 7  # 输入的特征数量
hidden_dim = 4  # 隐藏层的数量
num_layers = 2  # LSTM层数
output_dim = 7  # 输出的维度


model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)

if not os.path.exists(model_path):
    loss_function = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('////////////')
    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        outputs = model(train_tensor).to(device)
        optimizer.zero_grad()
        loss = loss_function(outputs, train_target_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


    
    predictions = model(test_tensor)
    print(loss_function(predictions,test_target_tensor))
    #print(denormalize(predictions, min(list(ot)), max(list(ot))))
    model.cpu()
    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))
    loss_function = torch.nn.MSELoss(reduction='mean').to(device)
    #loss_function2 = F.l1_loss().to(device)
    outputs1 = model(train_tensor).to(device)
    outputs2 = model(test_tensor).to(device)
    outputs3 = model(val_tensor).to(device)
    print('训练集损失(MSE)：',loss_function(outputs1, train_target_tensor).item(),'验证集损失(MSE)：',loss_function(outputs3, val_target_tensor).item(),'测试集损失(MSE)：'
          ,loss_function(outputs2, test_target_tensor).item())
    print('训练集损失(MAE)：',F.l1_loss(outputs1, train_target_tensor).item(),'验证集损失(MAE)：',F.l1_loss(outputs3, val_target_tensor).item(),'测试集损失(MAE)：'
          ,F.l1_loss(outputs2, test_target_tensor).item())
    model.to(device)
    model.eval()
    #确定日期
    #这个预测的只是验证集的第一段
    test_date=date[train_size+val_size:train_size+val_size+2*seq_length]
    predictions = model(test_tensor)
    s=0
    a1=test_tensor[s].reshape((seq_length,7)).cpu()
    b1=test_tensor[s+seq_length+1].reshape((seq_length,7)).cpu()
    a2=predictions[s].reshape((seq_length,7)).cpu()
    variable=6 #展示预测的是哪个变量,6-ot
    c1=list(a1[:,variable])+list(b1[:,variable])
    c2=list(a1[:,variable])+list(a2[:,variable].cpu().detach().numpy())
    plt.switch_backend('agg')
    plt.plot(test_date,c1 , label='Original', alpha=0.7)
    plt.plot(test_date,c2 , label='Predictions', alpha=0.7)
    plt.savefig("h1-96.jpg")
    plt.show()
