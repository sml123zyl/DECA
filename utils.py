import torch
import torch.nn as nn
import models
import torch.optim as opt
import numpy as np




class Causal_Loss(nn.Module):
    def __init__(self , alpha=0.5) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self , y , y_hat,n):
        # 初始化一个空的列表来存储结果
        inter_vars = []

        # 使用循环生成每个inter_var
        for i in range(0, n, 2):  # 从0开始，步长为2，直到7（包括）
            if i==n-1:
                 var = (torch.pow((y[:,i] - y_hat[:,n-1]), 2)) * self.alpha
                 inter_vars.append(var)
                 break
            else:
                var = torch.pow((y[:,i] - y_hat[:, i]), 2) + torch.pow((y[:,i + 1] - y_hat[:, i + 1]), 2)
                inter_vars.append(var)
        result = torch.mean(torch.stack(inter_vars))
        return result

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def MAPE(target, predict):
    return (abs((target-predict)/target)).mean() * 100

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def load_data(data : torch.Tensor , start , seq_len , predict_len,n):
# def load_data(data: torch.Tensor, start, seq_len, predict_len=24):

    # x_target = data[start : start+seq_len , -4:]
    x_target = data[start: start + seq_len, :]
    if seq_len<= predict_len:
        x_target0 = torch.zeros(predict_len-seq_len , n).to('cuda:0')
        x_target = torch.cat((x_target, x_target0), dim=0)
    # print(start,start+seq_len)
    # x_vsn = dict({
    #     '顶护盾缸有杆腔压力' : data[start : start+seq_len , 0].reshape(seq_len , 1) ,
    #     '主机皮带机驱动压力' : data[start : start+seq_len , 1].reshape(seq_len , 1) ,
    #     '撑靴撑紧力' : data[start : start+seq_len , 2].reshape(seq_len , 1) ,
    #     '推进速度' : data[start : start+seq_len , 3].reshape(seq_len , 1) ,
    #     '刀盘速度' : data[start : start+seq_len , 4].reshape(seq_len , 1) ,
    # })
    # x_vsn = dict({
    #     '% WEIGHTED ILI': data[start: start + seq_len, 0].reshape(seq_len, 1),
    #     '%UNWEIGHTED ILI': data[start: start + seq_len, 1].reshape(seq_len, 1),
    #     'AGE 0-4': data[start: start + seq_len, 2].reshape(seq_len, 1),
    #     'AGE 5-24': data[start: start + seq_len, 3].reshape(seq_len, 1),
    #     'ILITOTAL': data[start: start + seq_len, 4].reshape(seq_len, 1),
    #     'NUM OF PROVIDERS': data[start: start + seq_len, 5].reshape(seq_len, 1),
    #     'OT': data[start: start + seq_len, 6].reshape(seq_len, 1),
    # })
    x_vsn = {}
    for i in range(1, n + 1):
        key = str(i)
        if seq_len<= predict_len:
            value0 = data[start: start + seq_len, i - 1].reshape(seq_len, 1)
            value1 = torch.zeros(predict_len-seq_len , 1).to('cuda:0').reshape(predict_len-seq_len, 1)
            value = torch.cat((value0, value1),dim=0)
        else:
            value = data[start: start + seq_len, i-1].reshape(seq_len, 1) # 或者你想要的其他值
        x_vsn[key] = value
    # x_vsn = dict({
    #     '1': data[start: start + seq_len, 0].reshape(seq_len, 1),
    #     '2': data[start: start + seq_len, 1].reshape(seq_len, 1),
    #     '3': data[start: start + seq_len, 2].reshape(seq_len, 1),
    #     '4': data[start: start + seq_len, 3].reshape(seq_len, 1),
    #     '5': data[start: start + seq_len, 4].reshape(seq_len, 1),
    #     '6': data[start: start + seq_len, 5].reshape(seq_len, 1),
    #     '7': data[start: start + seq_len, 6].reshape(seq_len, 1),
    #  })

    y = data[start+seq_len : start+seq_len+predict_len , :]
    # print(start+seq_len,start+seq_len+predict_len)
    # y = data[start + seq_len: start + seq_len + predict_len, :]
    return x_target , x_vsn , y


def getCausalMatrix(data):
    val_matrix = data[0]
    p_matrix = data[1]
    adjacency_matrix = data[2]
    val_matrix = np.array(val_matrix)
    p_matrix = np.array(p_matrix)
    adjacency_matrix = np.array(adjacency_matrix)
    p_matrix[p_matrix < 0.1] = 0
    matrix = np.dot(val_matrix, p_matrix)
    in_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0))
    zero_indegree_columns = np.where(np.all(in_degree_matrix == 0, axis=0))[0]
    objective_matrix = np.copy(matrix)
    subjective_matrix = np.copy(matrix)
    objective_matrix[zero_indegree_columns,: ] = 0
    mask = np.zeros(matrix.shape[0], dtype=bool)
    mask[zero_indegree_columns] = True
    # 使用布尔索引将不包含指定行号的其余行的值置为零
    subjective_matrix = subjective_matrix * mask[:, None]
    return objective_matrix,subjective_matrix



def win_step_forward(data:torch.tensor , y_hat):
    data = torch.cat([data , y_hat] , axis=0)
    data = data[1: , :]
    return data


def train(data , separate  , epoch , win_shift , alpha,learn_rate , seq_len , predict_len,n,
          output_size_vsn,input_size_att,num_head_att,input_size_transf,num_head_transf,
                            output_size_transf,input_size_lstm,output_size_lstm,output_size,data_transf_dim,
          objective_matrix,subjective_matrix ):

    # 定义固定参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # input_sizes = dict({
    #     '顶护盾缸有杆腔压力' : 1,
    #     '主机皮带机驱动压力' : 1,
    #     '撑靴撑紧力' : 1,
    #     '推进速度' : 1 ,
    #     '刀盘速度' : 1 ,
    #
    # })
    # input_sizes = dict({
    #     '% WEIGHTED ILI': 1,
    #     '%UNWEIGHTED ILI': 1,
    #     'AGE 0-4': 1,
    #     'AGE 5-24': 1,
    #     'ILITOTAL': 1,
    #     'NUM OF PROVIDERS': 1,
    #     'OT': 1,
    # })
    input_sizes = {}
    for i in range(1, n + 1):
        key = str(i)
        value = 1  # 或者你想要的其他值
        input_sizes[key] = value
    # learn_rate = 0.001
    # learn_rate = 0.0001
    # seq_len = 200
    # seq_len = 720

    # 定义模型、损失函数、优化器
    mccn = models.MultiChannelCausalNetwork(input_sizes,output_size_vsn,input_size_att,num_head_att,input_size_transf,num_head_transf,
                            output_size_transf,input_size_lstm,output_size_lstm,output_size,data_transf_dim,
                                            objective_matrix,subjective_matrix,predict_len).to(device)
    csloss = Causal_Loss(alpha=alpha)
    optimiser = opt.SGD(mccn.parameters() , lr=learn_rate)
    losses = []

    for l in range(epoch):
        for i in range(len(separate)):
            

            j = separate.iloc[i , 0]
            while(j+seq_len +predict_len< separate.iloc[i , 1]):
                optimiser.zero_grad()
                # for k in range(0, predict_len):

                x_transf , x_vsn , y = load_data(data, j , seq_len, predict_len, n)
                y_hat = mccn(x_transf , x_vsn , device=device)
                loss = csloss(y, y_hat,n)


                # loss = csloss(y, y_hat[0:24,:])

                print('epoch=' , l ,'     i = ' , i ,  '       j =：' , j , '     loss = ' , loss)
                losses.append(loss.item())
                loss.backward()
                optimiser.step()

                j = j + win_shift

    return mccn , losses



def sin_train(data , separate  , epoch , win_shift , predict_len , alpha):

    # 定义固定参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_sizes = dict({
        '顶护盾缸有杆腔压力' : 1,
        '主机皮带机驱动压力' : 1,
        '撑靴撑紧力' : 1,
        '推进速度' : 1 ,
        '刀盘速度' : 1 ,

    })
    learn_rate = 0.001
    seq_len = 200

    # 定义模型、损失函数、优化器
    mccn = models.MultiChannelCausalNetwork(input_sizes).to(device)
    csloss = Causal_Loss(alpha=alpha)
    optimiser = opt.SGD(mccn.parameters() , lr=learn_rate)
    losses = []

    for l in range(epoch):
        for i in range(len(separate)):
            
            j = separate.iloc[i , 0]
            while(j+seq_len+predict_len < separate.iloc[i , 1]):

                x_transf , x_vsn , y = load_data(data, j , seq_len , predict_len)

                optimiser.zero_grad()
                y_hats = torch.zeros((1 , 4)).to(device)
                for k in range(predict_len):

                    y_hat = mccn(x_transf , x_vsn , device=device)
                    y_hats = torch.cat([y_hats , y_hat] , axis=0)
                    x_transf = win_step_forward(x_transf , y_hat)
                loss = csloss(y , y_hats[1: , :])
                print('epoch=' , l ,'     i = ' , i ,  '       j =：' , j , '     loss = ' , loss)
                losses.append(loss.item())
                loss.backward()
                optimiser.step()

                j = j + win_shift

    return mccn , losses
    


def predict(model , data , separate , start_sep , win_shift , alpha):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = 200
    start_index = separate.iloc[start_sep , 0]
    end_index = separate.iloc[start_sep , 1]
    start = start_index
    csloss = Causal_Loss(alpha=alpha)

    losses = []
    y_hats = torch.zeros(seq_len , 4)
    y_segment = data[start_index : end_index , -4:]
    

    while(start+2*seq_len <= end_index):
        with torch.no_grad():
            x_transf , x_vsn , y = load_data(data , start , seq_len)
            y_hat = model(x_transf , x_vsn , device)
            loss = csloss(y , y_hat)
            losses.append(loss.item())
            y_hats = torch.cat([y_hats , y_hat[:win_shift , :].cpu()])

            start = start+win_shift
            #print('start : ' , start)
    return y_hats[seq_len: , :] , y_segment , losses


def sin_predict(model , data , separate , start_sep , win_shift  , alpha):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = 200
    start_index = separate.iloc[start_sep , 0]
    end_index = separate.iloc[start_sep , 1]
    start = start_index
    csloss = Causal_Loss(alpha=alpha)

    losses = []
    y_hats = torch.zeros(seq_len , 4)
    y_segment = data[start_index : end_index , -4:]
    

    while(start+2*seq_len <= end_index):
        with torch.no_grad():
            x_transf , x_vsn , y = load_data(data , start , seq_len)
            for i in range(win_shift):

                y_hat = model(x_transf , x_vsn , device)
                x_transf = win_step_forward(x_transf , y_hat)
                loss = csloss(y , y_hat)
                losses.append(loss.item())
                y_hats = torch.cat([y_hats , y_hat.cpu()])

            start = start+win_shift
            #print('start : ' , start)
    return y_hats[seq_len: , :] , y_segment , losses


def sin_predict_2(model , data , separate , start_sep  , alpha):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = 200
    start_index = separate.iloc[start_sep , 0]
    end_index = separate.iloc[start_sep , 1]
    start = start_index
    csloss = Causal_Loss(alpha=alpha)

    losses = []
    y_hats = torch.zeros(seq_len , 4)
    y_segment = data[start_index : end_index , -4:]
    

    while(start+2*seq_len <= end_index):
        with torch.no_grad():
            x_transf , x_vsn , y = load_data(data , start , seq_len)
            

            y_hat = model(x_transf , x_vsn , device)
            x_transf = win_step_forward(x_transf , y_hat)
            loss = csloss(y , y_hat)
            losses.append(loss.item())
            y_hats = torch.cat([y_hats , y_hat.cpu()])

            start = start+1
            #print('start : ' , start)
    return y_hats[seq_len: , :] , y_segment , losses
# def MAE(pred, true):
#     return np.mean(np.abs(pred-true))
#
# def MSE(pred, true):
#     return np.mean((pred-true)**2)
#
# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))