# from gincov import GINConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph
from structuralremap_construction import smiles_to_data, collate_with_circle_index
import copy
import random
import sys
sys.path.append('/home/dell/mxq/toxic_mol/model/GraphADT/GRAPHADT')
from .MVPool.models import MVPool
from torch_geometric.nn import SAGPooling, TopKPooling, GCNConv
# random.seed(42)
def set_seed(seed_value=42):
    random.seed(seed_value)
    # np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GraphADT(nn.Module):
    def __init__(self,args,num_features_xd=128,dropout=0,aug_ratio=0.4,weights=[0.6, 0.3, 0.1, 0.6, 0.3, 0.1]):
        super(GraphADT, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )

        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_g1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        self.fc_final1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        ######93
        self.conv1 = GINConv(nn.Linear(93, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd ))
        ######43
        self.conv3 = GINConv(nn.Linear(128, num_features_xd))
        self.conv4 = GINConv(nn.Linear(43, num_features_xd ))

        self.conv5 = GINConv(nn.Linear(num_features_xd, num_features_xd ))
        self.conv6 = GINConv(nn.Linear(num_features_xd, num_features_xd ))
        self.conv7 = GINConv(nn.Linear(num_features_xd, num_features_xd ))
        self.conv8 = GINConv(nn.Linear(num_features_xd, num_features_xd ))

        self.relu = nn.ReLU()
        self.aug_ratio = aug_ratio
        self.linear2=nn.Linear(300,256)

        r_prime=feature_num=embed_dim = 256
        self.max_walk_len = 3
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        ##
        self.W = torch.nn.Parameter(torch.randn(embed_dim, feature_num), requires_grad=True)
        self.Wv = torch.nn.Parameter(torch.randn(r_prime, embed_dim), requires_grad=True)
        self.Ww = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.linea1=nn.Linear(57,93)
        self.linea2=nn.Linear(43,93)
        self.fc1 = nn.Linear(43, 43)
        self.fc2 = nn.Linear(43, 43)
        self.fc3 = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.pool1 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool2 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool3 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool4 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool5 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool6 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool7 = MVPool(self.nhid, self.pooling_ratio, args)
        self.pool8 = MVPool(self.nhid, self.pooling_ratio, args)
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)
        self.weight_r4 = torch.nn.Parameter(torch.Tensor([weights[4]]), requires_grad=True)
        self.weight_r5 = torch.nn.Parameter(torch.Tensor([weights[5]]), requires_grad=True)
        self.weight_r6=torch.nn.Parameter(torch.Tensor([weights[5]]), requires_grad=True)
        self.weight_r7=torch.nn.Parameter(torch.Tensor([weights[5]]), requires_grad=True)
        self.pools1 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pools2 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pools3 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pools4 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pools5 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pools6 = TopKPooling(self.nhid, self.pooling_ratio)
    def forward(self, data,x,edge_index,batch,a,edge,c):
        ###bond graph
        ###1
        x_g = self.relu(self.conv1(x, edge_index))
        # x_g, edge_index, edge_attr, batch, _ = self.pool1(x_g, edge_index, data.edge_attr, batch)
        x_g, edge_index, _, batch, _, _ = self.pools1(x_g, edge_index, data.edge_attr, batch)
        x_g_1= torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        ####2
        x_g = self.relu(self.conv2(x_g, edge_index))
        # x_g, edge_index, edge_attr, batch, _ = self.pool2(x_g, edge_index, data.edge_attr, batch)
        x_g, edge_index, _, batch, _, _ = self.pools2(x_g, edge_index, data.edge_attr, batch)
        x_g_2 = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        # # # ###3
        x_g = self.relu(self.conv3(x_g, edge_index))
        # x_g, edge_index, edge_attr, batch, _ = self.pool3(x_g, edge_index, data.edge_attr, batch)
        x_g, edge_index, _, batch, _, _= self.pools3(x_g, edge_index, data.edge_attr, batch)
        
        x_g_3 = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        # # ####4
        # x_g = self.relu(self.conv7(x_g, edge_index))
        # x_g, edge_index, edge_attr, batch, _ = self.pool7(x_g, edge_index, data.edge_attr, batch)
        # x_g_4 = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        x_g_all=None
        # x_g_all = F.relu(x_g_1) + F.relu(x_g_2) + F.relu(x_g_3)
        # x_g_all = F.relu(x_g_1) + F.relu(x_g_2)
        # x_g_all = F.relu(x_g_2) 
        # x_g_all = self.weight_r0*F.relu(x_g_1) + self.weight_r1*F.relu(x_g_2)+self.weight_r2*F.relu(x_g_3)+self.weight_r6*F.relu(x_g_4)
        x_g_all = self.weight_r0*F.relu(x_g_1) + self.weight_r1*F.relu(x_g_2)+self.weight_r2*F.relu(x_g_3)
        # x_g_all = self.weight_r0*F.relu(x_g_1) + self.weight_r1*F.relu(x_g_2)
        # x_g_all = F.relu(x_g_1) 
        x_g_all = F.relu(x_g_1) + F.relu(x_g_2) 
        x_g_all = self.fc_g(x_g_all)
        z = self.fc_final((x_g_all))
        # a=self.linea2(a)
        ###graph
        ##1
        x_g_in = self.relu(self.conv4(a, edge))
        # x_g_in, edge, edge_attr, c, _ = self.pool4(x_g_in, edge, data.edge_attr, c)
        x_g_in, edge, edge_attr, c, _,_ = self.pools4(x_g_in, edge, data.edge_attr, c)
        x_g_in_1 = torch.cat([gmp(x_g_in, c), gap(x_g_in, c)], dim=1)
        # ##2
        x_g_in = self.relu(self.conv5(x_g_in, edge))
        # x_g_in, edge, edge_attr, c, _ = self.pool5(x_g_in, edge, data.edge_attr, c)
        x_g_in, edge, edge_attr, c, _,_ = self.pools5(x_g_in, edge, data.edge_attr, c)
        x_g_in_2 = torch.cat([gmp(x_g_in, c), gap(x_g_in, c)], dim=1)
        # ##3
        x_g_in = self.relu(self.conv6(x_g_in, edge))
        # x_g_in, edge, edge_attr, c, _ = self.pool6(x_g_in, edge, data.edge_attr, c)
        x_g_in, edge, edge_attr, c, _,_ = self.pools6(x_g_in, edge, data.edge_attr, c)
        x_g_in_3 = torch.cat([gmp(x_g_in, c), gap(x_g_in, c)], dim=1)
        # ##4
        # x_g_in = self.relu(self.conv8(x_g_in, edge))
        # x_g_in, edge, edge_attr, c, _ = self.pool8(x_g_in, edge, data.edge_attr, c)
        # x_g_in_4 = torch.cat([gmp(x_g_in, c), gap(x_g_in, c)], dim=1)
        x_g_in_all=None
        # x_g_in_all = self.weight_r3*F.relu(x_g_in_1) + self.weight_r4*F.relu(x_g_in_2)+self.weight_r5*F.relu(x_g_in_3)+self.weight_r7*F.relu(x_g_in_4)
        x_g_in_all = self.weight_r3*F.relu(x_g_in_1) + self.weight_r4*F.relu(x_g_in_2)+self.weight_r5*F.relu(x_g_in_3)
        # x_g_in_all = self.weight_r3*F.relu(x_g_in_1) + self.weight_r4*F.relu(x_g_in_2)
        # x_g_in_all = F.relu(x_g_in_3)
        # x_g_in_all = F.relu(x_g_in_2) 
        # x_g_in_all = F.relu(x_g_in_1)+F.relu(x_g_in_2)
        x_g1 = self.fc_g1(x_g_in_all)
        z1 = self.fc_final1((x_g1))
        return z,x_g_all,x_g1,z1
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def subiso(self, z):
        z = self.projection(z)
        cos_sim = self.sim(z, z)
        cos_sim = torch.unsqueeze(cos_sim, 2)
        cos_sim = self.fc3(cos_sim)
        return torch.squeeze(cos_sim, 2)
    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)







