import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from models import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def metric(label, output):
    zs = torch.sigmoid(output).to('cpu').data.numpy()
    # print(zs.shape)
    ts = label.to('cpu').data.numpy()
    # print(ts.shape)
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))

    auc = roc_auc_score(ts, zs)
    aupr = average_precision_score(ts, zs)

    preds_list, t_list = [], []
    preds_list = np.append(preds_list, preds)
    t_list = np.append(t_list, ts)
    acc = accuracy_score(t_list, preds_list)
    precision = precision_score(t_list, preds_list)

    recall = recall_score(t_list, preds_list)
    mcc = matthews_corrcoef(t_list, preds_list)

    f1_scroe = (2 * precision * recall) / (recall + precision)
    # 为了计算SP，我们首先得到混淆矩阵
    tn, fp, fn, tp = confusion_matrix(t_list, preds_list).ravel()

    # 计算特异性
    specificity = tn / (tn + fp)

    # se=tp/(tp+fn)
    return auc, acc, precision, recall, f1_scroe, mcc, specificity
