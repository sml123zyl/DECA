#!/usr/bin/env python
# coding: utf-8
import csv
import os

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as opt

from sklearn.preprocessing import MinMaxScaler

import models
import utils
from scipy.signal import savgol_filter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)
data_df = pd.read_csv('./all_six_datasets/ENERGY/KAG_energydata_complete.csv' , encoding='gbk' )
data_arr = np.array(data_df)
scaler = MinMaxScaler()
data_arr_scalered = scaler.fit_transform(data_arr[:,1:])
separate = pd.read_excel('./all_six_datasets/ENERGY/数据集循环分段.xlsx')
data_tensor = torch.tensor(data_arr_scalered , requires_grad=True).type(torch.float32).to(device)

data_name = 'ENERGY'
seq_len = 60
predict_len =120
win_shift = 1
epoch = 1
alpha = 0.5
learn_rate = 0.0001
n = 28

# 模型参数
output_size_vsn=n
input_size_att=n
num_head_att=1
input_size_transf=n
num_head_transf=2
output_size_transf=12
input_size_lstm=20
output_size_lstm=16
output_size=12
data_transf_dim=n


folder_name = (f"./result_models1/{data_name}-seq_len{seq_len}-predict_len{predict_len}"
                            f"-epoch{epoch}-winsh{win_shift}-learn_rate{learn_rate}-alpha{alpha}")

# 创建文件夹
os.makedirs(folder_name, exist_ok=True)
data = np.load('./all_six_datasets/ENERGY/causal.npy', allow_pickle=True)
objective_matrix,subjective_matrix =utils.getCausalMatrix(data)

mccn , losses = utils.train(data_tensor , separate , epoch , win_shift , alpha,
                            learn_rate, seq_len , predict_len,n,
                            output_size_vsn,input_size_att,num_head_att,input_size_transf,num_head_transf,
                            output_size_transf,input_size_lstm,output_size_lstm,output_size,data_transf_dim,
                            objective_matrix,subjective_matrix )

torch.save(mccn , '{}/{}-seq_len{}-predict_len{}-epoch{}-winsh{}-learn_rate{}-alpha{}'
           .format(folder_name,data_name, seq_len,predict_len,epoch,win_shift,learn_rate,alpha))

model = torch.load('{}/{}-seq_len{}-predict_len{}-epoch{}-winsh{}-learn_rate{}-alpha{}'
           .format(folder_name,data_name, seq_len,predict_len,epoch,win_shift,learn_rate,alpha))

start_sep = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_index = separate.iloc[start_sep , 0]
end_index = separate.iloc[start_sep , 1]
start = start_index
csloss = utils.Causal_Loss(alpha=alpha)
losses = []
y_hats = torch.zeros(seq_len , n)
y_segment = data_tensor[start_index : end_index , :]
y_real = torch.zeros(seq_len , n)
while(start+seq_len+predict_len <= end_index):
    with torch.no_grad():
        x_transf , x_vsn , y = utils.load_data(data_tensor , start , seq_len,predict_len,n)
        y_hat = model(x_transf , x_vsn , device)
        loss = csloss(y , y_hat,n)
        losses.append(loss.item())
        y_hats = torch.cat([y_hats , y_hat[: , :].cpu()])
        y_real = torch.cat([y_real , y[: , :].cpu()])
        start = start+1
y_hats = y_hats[seq_len: , :]
y_real = y_real[seq_len: , :]

# 预测值滤波
y_hats = np.array(y_hats.cpu().detach().numpy())
for i in range(y_hats.shape[1]):
    y_hats[: , i] = savgol_filter(y_hats[: , i], 101, 3, mode='nearest')

csv_file_path = f"{folder_name}/{data_name}{predict_len}_pred.csv"
np.savetxt(csv_file_path, y_hats, delimiter=',', fmt='%.6f', comments='')
csv_file_path1 = f"{folder_name}/{data_name}{predict_len}_real.csv"
np.savetxt(csv_file_path1, y_real, delimiter=',', fmt='%.6f', comments='')
df_pred = pd.DataFrame(y_hats)
df_real = pd.DataFrame(y_real)
mse_all = 0.0
mae_all = 0.0
rmse_all = 0.0
mape_all = 0.0
mspe_all = 0.0
for column in df_pred.columns:
    mae = utils.MAE(df_pred[column], df_real[column])
    mse = utils.MSE(df_pred[column],df_real[column])
    rmse = utils.RMSE(df_pred[column], df_real[column])
    mase = utils.RMSE(df_pred[column], df_real[column])
    mspe = utils.RMSE(df_pred[column], df_real[column])
    mae_all = mae_all + mae
    mse_all = mse_all + mse
    rmse_all = rmse_all + rmse
    mape_all = mape_all + mase
    mspe_all = mspe_all + mspe
# 保存到 CSV 文件
file_path = '{}/{}_{}_{}_metric.csv' .format(folder_name,data_name, seq_len,predict_len)
# 打开文件并写入数据
with open(file_path, 'w', newline='') as csvfile:
    # 创建 CSV writer 对象
    csv_writer = csv.writer(csvfile)
    # 写入一行数据
    csv_writer.writerow(['MAE', 'MSE','RMSE', 'MAPE','MPSE'])
    # 写入两个值到相应的列
    csv_writer.writerow([mae_all, mse_all,rmse_all,mape_all ,mspe_all])
print(f'The values have been saved to {file_path}.')