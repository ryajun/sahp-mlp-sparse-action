import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

import numpy as np
import random

from models.sahp import SAHP
from utils import atten_optimizer
from utils import util

def make_model(nLayers=6, d_model=128, atten_heads=8, dropout=0.1, process_dim=10,
               device = 'cpu', pe='concat', max_sequence_length=4096):#修改, hidden_dim_list = [16,16,16,16], latent_dim = 16, input_dim = 16
    "helper: construct a models form hyper parameters"

    model = SAHP(nLayers, d_model, atten_heads, dropout=dropout, process_dim=process_dim, device = device,
                 max_sequence_length=max_sequence_length)#修改2,hidden_dim_list = hidden_dim_list, latent_dim = latent_dim, input_dim = input_dim

    # initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model#返回什么


def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')#返回矩阵的上三角 不包括对角线
    aaa = torch.from_numpy(mask) == 0#先产生一个上三角矩阵 然后把1变为0 把0变为1 变成一个包括对角线的下三角矩阵
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self,src,pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)#src就是原来的batch seq_type 返回一个

    @staticmethod#实例化 再调用
    def make_std_mask(tgt,pad,device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask  #在pytorch中的Variable就是一个存放会变化值的地理位置


def l1_loss(model):
    ## l1 loss
    l1 = 0
    for p in model.parameters():
        l1 = l1 + p.abs().sum()# abs返回参数的绝对值
    return l1

def eval_sahp(batch_size, loop_range, seq_lengths, seq_times, seq_types, seq_acts, model, device, lambda_l1=0):#添加
    model.eval()
    epoch_loss = 0
    for i_batch in loop_range:
        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, _,batch_seq_acts,seq_lengths = \
            util.get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, seq_acts, rnn=False)#修改部分
        batch_seq_types = batch_seq_types[:, 1:]
        batch_seq_acts = batch_seq_acts[:, 1:]#添加

        masked_seq_types = MaskBatch(batch_seq_types,pad=model.process_dim, device=device)# exclude the first added event
        masked_seq_acts = MaskBatch(batch_seq_acts, pad=model.process_dim, device=device)#添加
        # model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)

        model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask, masked_seq_acts.src)#修改
        nll = model.compute_loss(batch_seq_times, batch_onehot)

        loss = nll
        epoch_loss += loss.detach()
    event_num = torch.sum(seq_lengths).float()
    model.train()
    return event_num, epoch_loss


def train_eval_sahp(params):

    args, process_dim, device, tmax, \
    train_seq_times, train_seq_types, train_seq_lengths, train_seq_acts, \
    dev_seq_times, dev_seq_types, dev_seq_lengths, dev_seq_acts, \
    test_seq_times, test_seq_types, test_seq_lengths, test_seq_acts, \
    batch_size, epoch_num, use_cuda = params#修改部分

    ## sequence length
    train_seq_lengths, reorder_indices_train = train_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    train_seq_times = train_seq_times[reorder_indices_train]
    train_seq_types = train_seq_types[reorder_indices_train]
    train_seq_acts = train_seq_acts[reorder_indices_train]#修改部分
    #
    dev_seq_lengths, reorder_indices_dev = dev_seq_lengths.sort(descending=True)#从大到小进行排序
    # # Reorder by descending sequence length
    dev_seq_times = dev_seq_times[reorder_indices_dev]
    dev_seq_types = dev_seq_types[reorder_indices_dev]
    dev_seq_acts = dev_seq_acts[reorder_indices_dev]#修改部分

    test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    test_seq_times = test_seq_times[reorder_indices_test]
    test_seq_types = test_seq_types[reorder_indices_test]
    test_seq_acts = test_seq_acts[reorder_indices_test]#修改部分

    max_sequence_length = max(train_seq_lengths[0], dev_seq_lengths[0], test_seq_lengths[0])
    print('max_sequence_length: {}'.format(max_sequence_length))

    d_model = args.d_model
    atten_heads = args.atten_heads#d_model=128, atten_heads=8
    dropout = args.dropout

    model = make_model(nLayers=args.nLayers, d_model=d_model, atten_heads=atten_heads,
                    dropout=dropout, process_dim=process_dim, device=device, pe=args.pe,
                    max_sequence_length=max_sequence_length + 1).to(device)#修改,hidden_dim_list=[16,16,16,16], latent_dim = 16, input_dim=16

    print("the number of trainable parameters: " + str(util.count_parameters(model)))#还没看

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.lambda_l2)
    model_opt = atten_optimizer.NoamOpt(args.d_model, 1, 100, initial_lr=args.lr, optimizer=optimizer)


    ## Size of the traing dataset
    train_size = train_seq_times.size(0)
    dev_size = dev_seq_times.size(0)
    test_size = test_seq_times.size(0)
    tr_loop_range = list(range(0, train_size, batch_size))
    de_loop_range = list(range(0, dev_size, batch_size))
    test_loop_range = list(range(0, test_size, batch_size))

    last_dev_loss = 0.0
    epoch_dev_loss = 0.0
    epoch_test_loss = 0.0
    early_step = 0

    model.train()
    for epoch in range(epoch_num):
        epoch_train_loss = 0.0
        print('Epoch {} starts '.format(epoch))

        ## training
        random.shuffle(tr_loop_range)
        for i_batch in tr_loop_range:#train data

            model_opt.optimizer.zero_grad()#把模型中参数置0

            batch_onehot, batch_seq_times, batch_dt, batch_seq_types,_, _, _, _,batch_seq_acts, batch_seq_lengths = \
                util.get_batch(batch_size, i_batch, model, train_seq_lengths, train_seq_times, train_seq_types, train_seq_acts, rnn=False)#修改部分, _, _, _, batch_seq_lengths = \

            batch_seq_types = batch_seq_types[:, 1:]
            batch_seq_acts = batch_seq_acts[:, 1:]#添加

            masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim, device=device)# exclude the first added even
            masked_seq_acts = MaskBatch(batch_seq_acts, pad=model.process_dim, device=device)#添加
            model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask, masked_seq_acts.src)#修改
            nll = model.compute_loss(batch_seq_times, batch_onehot)#根据公式计算 很多细节

            loss = nll

            loss.backward() #后向传播 计算梯度
            model_opt.optimizer.step()#梯度计算好后 调用optimizer.step 更新所有参数

            if i_batch %50 == 0:
                batch_event_num = torch.sum(batch_seq_lengths).float()
                print('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
                      .format(epoch, i_batch, loss.item()/ batch_event_num))
            epoch_train_loss += loss.detach()

        if epoch_train_loss < 0:
            break
        train_event_num = torch.sum(train_seq_lengths).float()
        print('---\nEpoch.{} Training set\nTrain Negative Log-Likelihood per event: {:5f} nats\n' \
              .format(epoch, epoch_train_loss / train_event_num))

        ## dev
        for i_batch in de_loop_range:  # train data
            model.eval()
            # model_opt.optimizer.zero_grad()

            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, _, batch_seq_acts, batch_seq_lengths = \
                util.get_batch(batch_size, i_batch, model, dev_seq_lengths, dev_seq_times, dev_seq_types,
                               dev_seq_acts, rnn=False)  # 修改部分, _, _, _, batch_seq_lengths = \

            batch_seq_types = batch_seq_types[:, 1:]
            batch_seq_acts = batch_seq_acts[:, 1:]  # 添加

            masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                         device=device)  # exclude the first added even
            masked_seq_acts = MaskBatch(batch_seq_acts, pad=model.process_dim, device=device)  # 添加
            model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask, masked_seq_acts.src)  # 修改
            nll = model.compute_loss(batch_seq_times, batch_onehot)

            loss = nll

            # loss.backward()
            # model_opt.optimizer.step()

            # if i_batch % 50 == 0:
            #     batch_event_num = torch.sum(batch_seq_lengths).float()
            #     print('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
            #           .format(epoch, i_batch, loss.item() / batch_event_num))
            epoch_dev_loss += loss.detach()

        # if epoch_dev_loss < 0:
        #     break
        dev_event_num = torch.sum(dev_seq_lengths).float()
        model.train()
        print('---\nEpoch.{} dev set\ndev Negative Log-Likelihood per event: {:5f} nats\n' \
              .format(epoch, epoch_dev_loss / dev_event_num))

        ## test
        for i_batch in test_loop_range:  # train data
            model.eval()
            # model_opt.optimizer.zero_grad()

            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, _, batch_seq_acts, batch_seq_lengths = \
                util.get_batch(batch_size, i_batch, model, test_seq_lengths, test_seq_times, test_seq_types,
                               test_seq_acts, rnn=False)  # 修改部分, _, _, _, batch_seq_lengths = \

            batch_seq_types = batch_seq_types[:, 1:]
            batch_seq_acts = batch_seq_acts[:, 1:]  # 添加

            masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                         device=device)  # exclude the first added even
            masked_seq_acts = MaskBatch(batch_seq_acts, pad=model.process_dim, device=device)  # 添加
            model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask, masked_seq_acts.src)  # 修改
            nll = model.compute_loss(batch_seq_times, batch_onehot)

            loss = nll

            # loss.backward()
            # model_opt.optimizer.step()

            # if i_batch % 50 == 0:
            #     batch_event_num = torch.sum(batch_seq_lengths).float()
            #     print('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
            #           .format(epoch, i_batch, loss.item() / batch_event_num))
            epoch_test_loss += loss.detach()

            # if epoch_test_loss < 0:
            #     break
        test_event_num = torch.sum(test_seq_lengths).float()
        model.train()
        print('---\nEpoch.{} test set\ntest Negative Log-Likelihood per event: {:5f} nats\n' \
              .format(epoch, epoch_test_loss / test_event_num))

        ## early stopping
        gap = epoch_dev_loss / dev_event_num - last_dev_loss
        if abs(gap) < args.early_stop_threshold:
            early_step += 1
        last_dev_loss = epoch_dev_loss / dev_event_num


        if early_step >=3:
            print('Early Stopping')
            break

        # prediction
        avg_rmse, types_predict_score = \
            prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_seq_acts, test_size, tmax)

    return model


def prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_seq_acts, test_size, tmax):
    model.eval()#添加
    from utils import evaluation
    test_data = (test_seq_times, test_seq_types, test_seq_lengths, test_seq_acts )#添加
    incr_estimates, incr_errors, types_real, types_estimates = \
        evaluation.predict_test(model, *test_data, pad=model.process_dim, device=device,
                                hmax=tmax, use_jupyter=False, rnn=False)
    if device != 'cpu':
        incr_errors = [incr_err.item() for incr_err in incr_errors]
        types_real = [types_rl.item() for types_rl in types_real]
        types_estimates = [types_esti.item() for types_esti in types_estimates]

    avg_rmse = np.sqrt(np.mean(incr_errors), dtype=np.float64)
    print("rmse", avg_rmse)
    mse_var = np.var(incr_errors, dtype=np.float64)

    delta_meth_stderr = 1 / test_size * mse_var / (4 * avg_rmse)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score#混淆矩阵主要用于比较分类结果和实例的真实信息。矩阵中的每一行代表实例的预测类别，每一列代表实例的真实类别
    types_predict_score = f1_score(types_real, types_estimates, average='micro')# preferable in class imbalance

    print("Type prediction score:", types_predict_score)
    # print("Confusion matrix:\n", confusion_matrix(types_real, types_estimates))
    model.train()
    return avg_rmse, types_predict_score

if __name__ == "__main__":
    mode = 'train'

    if mode == 'train':
        with autograd.detect_anomaly():
            train_eval_sahp()

    else:
        pass
    print("Done!")



