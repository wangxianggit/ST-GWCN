import time
import logging
import os
import argparse
import configparser
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torchsummary import summary
from script import dataloader, utility, earlystopping
from model import models

def set_seed(seed):    # 随机种子
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    set_seed(worker_id)

##########################-----------------------------------------------参数获取

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN for road traffic prediction')
    parser.add_argument('--enable_cuda', type=bool, default='True',
                        help='enable CUDA, default as True')
    parser.add_argument('--n_pred', type=int, default=3,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs, default as 500')
    parser.add_argument('--dataset_config_path', type=str, default='./config/data/train/road_traffic/pems-bay.ini',
                        help='the path of dataset config file, pemsd7-m.ini for PeMSD7-M, \
                            metr-la.ini for METR-LA, and pems-bay.ini for PEMS-BAY,and pemsd04.ini for PeMSD04')

    parser.add_argument('--model_config_path', type=str, default='./config/model/graph_waveletsconv_sym_glu.ini',
                        help='the path of model config file,graph_waveletsconv_sym_glu.ini for STGWNN(graph_waveletsconv, Ks=3, Kt=3), \
                            and graph_fastwaveletsconv_sym_glu.ini for STGWNN(graph_fastwaveletsconv, Kt=3)')  #chebconv_sym_glu.ini for STGCN(ChebConv, Ks=3, Kt=3)
    parser.add_argument('--opt', type=str, default='AdamW',
                        help='optimizer, default as AdamW')

    parser.add_argument("--threshold",
                        type=float,
                        default=1e-4,
                        help="Sparsification parameter. Default is 1e-4.")
    parser.add_argument("--approximation-order",
                        type=int,
                        default=4,
                        help="Order of Chebyshev polynomial. Default is 4.")
    parser.add_argument("--scale",
                        type=float,
                        default=2.0,
                        help="Scaling parameter. Default is 1.0.")



    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    config = configparser.ConfigParser()

    def ConfigSectionMap(section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
                if dict1[option] == -1:
                    logging.debug("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_config_path = args.model_config_path
    dataset_config_path = args.dataset_config_path
    config.read(dataset_config_path, encoding="utf-8")

    dataset = ConfigSectionMap('data')['dataset']
    time_intvl = int(ConfigSectionMap('data')['time_intvl'])
    n_his = int(ConfigSectionMap('data')['n_his'])
    Kt = int(ConfigSectionMap('data')['kt'])
    stblock_num = int(ConfigSectionMap('data')['stblock_num'])
    if ((Kt - 1) * 2 * stblock_num > n_his) or ((Kt - 1) * 2 * stblock_num <= 0):
        raise ValueError(f'ERROR: {Kt} and {stblock_num} are unacceptable.')
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    drop_rate = float(ConfigSectionMap('data')['drop_rate'])
    batch_size = int(ConfigSectionMap('data')['batch_size'])
    learning_rate = float(ConfigSectionMap('data')['learning_rate'])
    weight_decay_rate = float(ConfigSectionMap('data')['weight_decay_rate'])
    step_size = int(ConfigSectionMap('data')['step_size'])
    gamma = float(ConfigSectionMap('data')['gamma'])
    data_path = ConfigSectionMap('data')['data_path']
    wam_path = ConfigSectionMap('data')['wam_path']
    model_save_path = ConfigSectionMap('data')['model_save_path']
    config.read(model_config_path, encoding="utf-8")
    gated_act_func = ConfigSectionMap('casualconv')['gated_act_func']
    graph_conv_type = ConfigSectionMap('graph_waveletsconv')['graph_conv_type']

    if (graph_conv_type != "graph_waveletsconv") and (graph_conv_type != "graph_fastwaveletsconv"):
        raise NotImplementedError(f'ERROR: {graph_conv_type} is not implemented.')
    else:
        graph_conv_type = graph_conv_type
    Ks = int(ConfigSectionMap('graph_waveletsconv')['ks'])
    if (graph_conv_type == 'graph_waveletsconv') and (Ks != 2):
        Ks = 2
    mat_type = ConfigSectionMap('graph_waveletsconv')['mat_type']

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []               #[[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]]
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    # print(len(blocks))
    # print(blocks[-3][-1])
    day_slot = int(24 * 60 / time_intvl)
    n_pred = args.n_pred
    time_pred = n_pred * time_intvl
    time_pred_str = str(time_pred) + '_mins'
    model_name = ConfigSectionMap('graph_waveletsconv')['model_name']

    print(model_name)
    model_save_path = model_save_path + model_name + '_' + dataset + '_' + time_pred_str + '.pth'
    adj_mat = dataloader.load_weighted_adjacency_matrix(wam_path)
    n_vertex_vel = pd.read_csv(data_path, header=None).shape[1]
    n_vertex_adj = pd.read_csv(wam_path, header=None).shape[1]
    if n_vertex_vel != n_vertex_adj:
        raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
    else:
        n_vertex = n_vertex_vel

    opt = args.opt
    epochs = args.epochs

    wavelets, wavelet_inv =utility.wavelet_basis(adj_mat, args.scale, args.threshold,mat_type)
    wavelets, wavelet_inv = (torch.from_numpy(wavelets.toarray())).float().to(device), (torch.from_numpy(wavelet_inv.toarray())).float().to(device)
    stgcn_wavelet_basis= models.STGWNN(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, wavelets, wavelet_inv, drop_rate).to(device)
    model = stgcn_wavelet_basis
    return device, n_his, n_pred, day_slot, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma

def data_preparate(data_path, device, n_his, n_pred, day_slot, batch_size):
    data_col = pd.read_csv(data_path, header=None).shape[0]                #'./data/train/road_traffic/metr-la/vel.csv';;;;data_col=34272
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(data_path, len_train, len_val)
    #normalization
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, n_his, n_pred, day_slot, device)
    x_val, y_val = dataloader.data_transform(val, n_his, n_pred, day_slot, device)
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, day_slot, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return zscore, train_iter, val_iter, test_iter



def main(learning_rate, weight_decay_rate, graph_conv_type, model_save_path, model, n_his, n_vertex, step_size, gamma, opt):
    loss = nn.MSELoss()
    learning_rate = learning_rate
    weight_decay_rate = weight_decay_rate
    early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)

    model_stats = summary(model, (1, n_his, n_vertex))

    if opt == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return loss, early_stopping, optimizer, scheduler


def train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter,zscore):
    min_val_loss = np.inf
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        val_loss = val(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        early_stopping(val_loss, model)
        # GPU memory usage
        # tiain_MAE, tiain_MAPE, tiain_RMSE = utility.evaluate_metric(model, train_iter, zscore)
        tiain_MAE, tiain_RMSE, tiain_WMAPE = utility.evaluate_metric(model, train_iter, zscore)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB '.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        # print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB | Tiain_MAE: {:.6f} | Tiain_MAPE: {:.6f} |Tiain_RMSE: {:.6f} |'.\
        #     format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc,tiain_MAE, tiain_RMSE, tiain_WMAPE))
        Tloss=l_sum / n
        vloss=val_loss
        # lr = optimizer.param_groups[0]['lr']
        list = [Tloss, vloss]
        data = pd.DataFrame([list])
        data.to_csv('data.csv',mode='a',header=False,index=False)

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')
    model.load_state_dict(best_state)






def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        s1 = time.time()
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        s2 = time.time()
        log = 'Inference Time: {:.4f} secs'
        print(log.format((s2-s1)))
        return l_sum / n


def test(zscore, loss, model, test_iter):
    best_model = model
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = utility.evaluate_model(best_model, loss, test_iter)
    print('Test loss {:.6f}'.format(test_MSE))
    test_MAE, test_MAPE, test_RMSE = utility.evaluate_metric1(best_model, test_iter, zscore)

    print(f'MAE {test_MAE:.6f} | MAPE {test_MAPE:.8f} | RMSE {test_RMSE:.6f}')
    # test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(best_model, test_iter, zscore)
    # print(f'MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')




if __name__ == "__main__":
    # For stable experiment results
    SEED=150882500 #150882500
    # SEED = [1708825600,1508825600,150882500,180882600,15025600,18825600,
    #         1208258600,508825600,308825600,130882500,150872500,140872500,16082500,156882500,153882500,150282500]    #1608825600
    # for i in SEED:
    #    set_seed(i)

       # For multi-threading dataloader
   # worker_init_fn(SEED)

       # Logging
       #logger = logging.getLogger('stgcn')
       #logging.basicConfig(filename='stgcn.log', level=logging.INFO)

    logging.basicConfig(level=logging.INFO)

    device, n_his, n_pred, day_slot, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma = get_parameters()
    zscore, train_iter, val_iter, test_iter = data_preparate(data_path, device, n_his, n_pred, day_slot, batch_size)

    loss, early_stopping, optimizer, scheduler = main(learning_rate, weight_decay_rate, graph_conv_type, model_save_path, model, n_his, n_vertex, step_size, gamma, opt)

       # Training
    train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter,zscore)

       # Testing
    test(zscore, loss, model, test_iter)
