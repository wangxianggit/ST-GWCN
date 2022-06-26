import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
import torch
from sklearn.preprocessing import normalize
import numpy as np
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.optimize import minimize
from scipy.integrate import quad
import sys
import math
import time

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.identity(n_vertex)

    # D_row
    deg_mat_row = np.diag(np.sum(adj_mat, axis=1))
    # D_com
    #deg_mat_col = np.diag(np.sum(adj_mat, axis=0))

    # D = D_row as default
    deg_mat = deg_mat_row

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)

        # Symmetric normalized Laplacian
        # For SpectraConv
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = np.matmul(np.matmul(deg_mat_inv_sqrt, com_lap_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        ev_max_sym = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / ev_max_sym - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):

        deg_mat_inv = fractional_matrix_power(deg_mat, -1)
        wid_deg_mat_inv = fractional_matrix_power(wid_deg_mat, -1)

        # Random Walk normalized Laplacian
        # For SpectraConv
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = np.matmul(deg_mat_inv, com_lap_mat)

        # For ChebConv
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        ev_max_rw = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / ev_max_rw - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])




def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()         #.tolist()将数组或者矩阵转换成列表
            mse += (d ** 2).tolist()

        MAE = np.array(mae).mean()
        mape=[x for x in mape if (x>-5 and x < 100)]
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, MAPE, RMSE
        # return MAE, RMSE, WMAPE

def evaluate_metric1(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()         #.tolist()将数组或者矩阵转换成列表
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        mape=[x for x in mape if (x>0 and x < 100)]
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, MAPE, RMSE


 ##############################################
    # eye(m[, n, k, dtype, format])：对角线为1的稀疏矩阵
    # identity(n[, dtype, format])：单位矩阵
    # diags(diagonals[, offsets, shape, format, dtype])：构造对角矩阵（含偏移量）
    # spdiags(data, diags, m, n[, format])：从矩阵中返回含偏移量的对角稀疏矩阵
    # hstack(blocks[, format, dtype]) Stack sparse matrices horizontally (column wise)
    # vstack(blocks[, format, dtype]) Stack sparse matrices vertically (row wise)
################################################

def weight_wavelet(s, lamb, U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.exp(-lamb[i] * s)
    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))      #np.dot矩阵乘法

    return Weight


def weight_wavelet_inverse(s, lamb, U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.exp(lamb[i] * s)

    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

    return Weight


def fourier(L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]                               #返回的是元素值从小到大排序后的索引值的数组
    if algo is 'eig':
        lamb, U = np.linalg.eig(np.asarray(L))                    # np.linalg.eig();;lamb特征值, U特征向量
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(np.asarray(L))
        lamb, U = sort(lamb, U)
    elif algo is 'eigs':
        lamb, U = sp.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = sp.linalg.eigsh(L, k=k, which='SM')

    return lamb, U


def wavelet_basis(adj, s, threshold,mat_type):
    L = scaled_Laplacian(adj)
    lamb, U = fourier(L)
    # lamb, U = fourier(adj)
    Weight = weight_wavelet(s, lamb, U)
    inverse_Weight = weight_wavelet_inverse(s, lamb, U)
    del U, lamb
    Weight[Weight < threshold] = 0.0
    inverse_Weight[inverse_Weight < threshold] = 0.0
    Weight = normalize(Weight, norm='l1', axis=1)
    inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)
    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    t_k = (Weight, inverse_Weight)
    return t_k

def largest_lamb(L):
    lamb, U = sp.linalg.eigsh(L, k=1, which='LM')
    lamb = lamb[0]
    # print(lamb)
    return lamb


def threshold_to_zero(mx, threshold):
    """Set value in a sparse matrix lower than
     threshold to zero.

    Return the 'coo' format sparse matrix.

    Parameters
    ----------
    mx : array_like
        Sparse matrix.
    threshold : float
        Threshold parameter.
    """
    high_values_indexes = set(zip(*((np.abs(mx) >= threshold).nonzero())))
    nonzero_indexes = zip(*(mx.nonzero()))

    if not sp.isspmatrix_lil(mx):
        mx = mx.tolil()

    for s in nonzero_indexes:
        if s not in high_values_indexes:
            mx[s] = 0.0
    # mx =  sp.coo_matrix(mx)
    mx=sp.csr_matrix(mx)
    mx.eliminate_zeros()
    return mx
def spectral_basis(adj,mat_type):
    # from weighting_func import laplacian,fourier,weight_wavelet,weight_wavelet_inverse
    L = calculate_laplacian_matrix(adj,mat_type)
    lamb, U = fourier(L)

    U = sp.csr_matrix(U)
    # U_transpose = sp.csr_matrix(np.transpose(U))
    # t_k = [U]
    return U