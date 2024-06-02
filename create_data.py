from bondedgeconstruction import smiles_to_data
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.data import Data
for i in range(1,11):
    root=f'/home/dell/mxq/toxic_mol/model/MS2/datasets/Rabbit/kfold_splits_adj/fold_{i}'#data_rlm
    # root=f'/home/dell/mxq/toxic_mol/model/MS2/datasets/rabbit/split'#data_rlm
    list_train=pd.read_csv(root+'/train.csv')
    list_test=pd.read_csv(root+'/test_dataset.csv')
    list_valid=pd.read_csv(root+'/val.csv')
    list_external=pd.read_csv(root+'/Rabbit_external.csv')
    import os
    if not os.path.exists(root+'/train_resample.pth'):
        datasettrain=[]
        for idx, row in tqdm(list_train.iterrows()):
            print(idx)
            print(row)
            data1=smiles_to_data(row['SMILES'])
            data1.y = torch.tensor(row['Label'], dtype=torch.long)
            data_listtrain = data1
            datasettrain.append(data_listtrain)
        torch.save(datasettrain, root+'/train_resample.pth')
    if not os.path.exists(root+'/test.pth'):
        datasettest=[]
        for idx, row in tqdm(list_test.iterrows()):
            data2=smiles_to_data(row['Canonical_Smiles'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            data_listtest = data2
            datasettest.append(data_listtest)
        torch.save(datasettest, root+'/test.pth')

    if not os.path.exists(root+'/valid.pth'):
        datasettest1=[]
        for idx, row in tqdm(list_valid.iterrows()):
            data2=smiles_to_data(row['SMILES'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            data_listtest = data2
            datasettest1.append(data_listtest)
        torch.save(datasettest1, root+'/valid.pth')

    if not os.path.exists(root+'/external.pth'):
        datasettest1=[]             
        for idx, row in tqdm(list_external.iterrows()):
            data2=smiles_to_data(row['SMILES'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            data_listtest = data2
            datasettest1.append(data_listtest)
        torch.save(datasettest1, root+'/external.pth')