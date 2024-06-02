import sys
from models import *
# from utils import *
from evalution import *
import torch.nn.functional as F
from nt_xent import NT_Xent
from models.model import set_seed
from models.model import GraphADT
import argparse
from structuralremap_construction import *
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch,criterion):
    lossz = 0
    model.train()
    # print(train_loader)
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch1 = data.batch1.detach()
        edge = data.edge_index1.detach()
        xd = data.x1.detach()
        n = data.y.shape[0]  # batch
        optimizer.zero_grad()
        output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
        loss_1 = criterion(output, data.y.view(-1, 1).float())
        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
        criterion2=NT_Xent(output.shape[0], 0.1, 1)
        loss_2 = criterion(output1, data.y.view(-1, 1).float())
        cl_loss_node = criterion1(x_g, x_g1)
        cl_loss_graph=criterion2(output,output1)
        # loss = loss_2
        loss = loss_1+loss_2+(0.3*cl_loss_node)
        # loss = loss_2
        loss.backward()
        optimizer.step()    
        lossz = loss + lossz

    print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch1 = data.batch1.detach()
            edge = data.edge_index1.detach()
            xd = data.x1.detach()
            output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels, total_preds
def run_experiment(args,run,i,time):
    print(f'dataset:{i}',f'fold:{run}',f'time:{time}')
    set_seed(42)
    cuda_name = "cuda:1"
    print('cuda_name:', cuda_name)

    NUM_EPOCHS = 300
    LR = 0.001

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    root = f'/home/dell/mxq/toxic_mol/model/GraphADT/Dataset/{i}/kfold_splits/fold_{run}'  # data_External/data_rlm/data/noH
    processed_train = root + '/train.pth'
    processed_valid = root + '/external.pth'
    # processed_test = f'/home/dell/mxq/toxic_mol/model/GraphADT/datasets/{i}/kfold_splits/valid.pth'
    data_listtrain = torch.load(processed_train)
    data_listtest = torch.load(processed_valid)
    def custom_batching(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]
    best_results = []
    best_results_list = []
    batchestrain = list(custom_batching(data_listtrain, 256))
    batchestrain1 = list()
    for batch_idx, data in enumerate(batchestrain):
        data = collate_with_circle_index(data)
        data.edge_attr = None
        batchestrain1.append(data)
    batchestest = list(custom_batching(data_listtest, 1000))
    batchestest1 = list()
    for batch_idx, data in enumerate(batchestest):
        data = collate_with_circle_index(data)
        data.edge_attr = None
        batchestest1.append(data)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = GraphADT(args=args).to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    max_auc = 0
    max_acc=0
    max_recall=0
    model_file_name = f'/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result_n/{i}/external/fold_{k}/'+f'model_{i}_{run}_{time}' + '.pt'
    # result_file_name = 'result' + '.csv'
    for epoch in range(NUM_EPOCHS):
        train(model, device, batchestrain1, optimizer, epoch + 1,criterion)
        G, P = predicting(model, device, batchestest1)
        auc, acc, precision, recall, f1_scroe, mcc, specificity = metric(G, P)
        ret = [auc, acc, precision, recall, f1_scroe, mcc, specificity]
        if (acc+auc) > (max_acc+max_auc):
            max_acc=acc
            max_auc = auc
            max_recall=recall
            torch.save(model.state_dict(), model_file_name)
            # with open(result_file_name, 'w') as f:
            #     f.write(','.join(map(str, ret)))
            ret1 = [auc, acc, precision, recall, f1_scroe, mcc, specificity]

        print(
            'test---------------------------  auc:%.4f\t acc:%.4f\t precision:%.4f\t recall:%.4f\tf1_scroe:%.4f\t mcc:%.4f\t specificity:%.4f' % (
                auc, acc, precision, recall, f1_scroe, mcc, specificity))

    print('Maximum acc found. Model saved.')
    best_results.append({
        'auc, acc, precision, recall, f1_scroe, mcc, specificity': (
            ret1)

    })
    best_results_list.append(ret1)
    for i, result in enumerate(best_results, 1):
        print(f"Fold {run}:")
        print(f"Best auc, acc, precision, recall, f1_scroe, mcc,specificity: {result['auc, acc, precision, recall, f1_scroe, mcc, specificity']}")
    return ret1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--sample_neighbor', type=bool, default=False, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
    parser.add_argument('--hop_connection', type=bool, default=True, help='whether directly connect node within h-hops')
    parser.add_argument('--hop', type=int, default=3, help='h-hops')
    parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=2.0, help='trade-off parameter')
    parser.add_argument('--dataset', type=str, default='rabbit', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
    parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--experiment_root', type=str, default='/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result/rabbit/noramal/kfold_splits/fold_', help='patience for early stopping')
    args = parser.parse_args()
    # 运行实验30次
    for i in ['Rabbit','Rat']:
        for k in range(1,10):
            results = []
            results = []
            result_file_path = f'/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result_n/{i}/external/fold_{k}/results_external.csv'
            average_result_file_path = f'/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result_n/{i}/external/fold_{k}/results_external_with_average.csv'     
            # 检查文件是否存在
            import os
            # if os.path.exists(result_file_path) and os.path.exists(average_result_file_path):
            #     print(f"实验结果文件和平均值文件已存在于 {i}, fold {k}。跳过此实验。")
            #     continue  # 如果文件存在，跳过当前循环的其余部分
            for run in range(1):
                result = run_experiment(args,k,i,run)
                print(result)
                results.append(result)
            import pandas as pd
            # 将结果保存到CSV文件
            results_df = pd.DataFrame(results, columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'Specificity'])
            results_df.to_csv(f'/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result_n/{i}/external/fold_{k}/results_external.csv', index=False)
            print("实验完成，结果已保存。")
            # 在之前的代码基础上继续

            # 计算平均值
            average_results = results_df.mean()

            # 输出平均值
            print("平均结果：")
            print(average_results)

            # 如果需要，也可以将平均值保存到CSV文件中
            average_results_df = pd.DataFrame([average_results], index=['Average'])
            # final_results_df = pd.concat([average_results_df])
            average_results_df.to_csv(f'/home/dell/mxq/toxic_mol/model/GraphADT/experiment_result_n/{i}/external/fold_{k}/results_results_external_with_average.csv', index=False)
            print("实验完成，结果和平均值已保存。")