from GM.evaluate import get_roc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from GM.data_unit.utils import normalize_graph
from GM.models import model_GM_homo
from torch_geometric.utils import degree
import os
import argparse
from termcolor import cprint
from GM.evaluate import mask_test_edges
from GM.datapro import read_data




def GM_get_args():
    parser = argparse.ArgumentParser(description='Parser for GCN and MLP Strategy')
    # Basics
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--dataset-name", default='rda')
    # Pretrain
    parser.add_argument("--pretrain", default=False, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--w_loss1', type=float, default=5, help='')
    parser.add_argument('--w_loss2', type=float, default=5, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.4, help='')
    parser.add_argument('--cfg', type=int, default=[128], help='')
    parser.add_argument('--NN', type=int, default=5, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')
    args = parser.parse_args()
    return args

def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])
def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))

def get_dataset(args,name,num, dataset_kwargs):

    data = read_data(name,num)
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    i = torch.LongTensor([data.edge_index[0].numpy(), data.edge_index[1].numpy()])
    v = torch.FloatTensor(torch.ones([data.num_edges]))
    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    # A_nomal = normalize_graph(A)
    A_I_nomal = normalize_graph(A_I)
    A_I_nomal = A_I_nomal.to_sparse()


    nb_feature = data.num_features

    nb_nodes = data.num_nodes
    data.x = torch.FloatTensor(data.x)
    eps = 2.2204e-16
    norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    data.x = data.x.div(norm.expand_as(data.x))
    adj_1 = csr_matrix(
        (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(data.num_nodes, data.num_nodes))


    return data, [A_I_nomal,adj_1,i], [data.x], [nb_feature, nb_nodes]


def GM_homo(args,name,num,gpu_id=None, **kwargs):
    # ===================================================#
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # ===================================================#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    cprint("## Loading Dataset ##", "yellow")
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, name,num, dataset_kwargs)
    nb_feature = nb_list[0]
    nb_nodes = nb_list[1]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    adj_1 = adj_list[1]
    edge = adj_list[2]
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj_1, test_frac=0.1, val_frac=0.05)
    cprint("## Done ##", "yellow")
    # ===================================================#
    model = model_GM_homo(nb_feature, cfg=args.cfg,
                       dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)
    # ===================================================#
    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    edge_index = A_I_nomal._indices()[1]
    # ===================================================#
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(running_device)
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
    idx_p_list = []
    for j in range(1, 101):
        random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
        idx_p = edge_index[random_list]
        idx_p_list.append(idx_p)
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        model.train()
        optimiser.zero_grad()
        
        h_a, h_p = model(feature_X, A_I_nomal)
        # h_a, h_p = model(feature_X, [A_I_nomal,edge])#测试使用其他GCN模型生成嵌入
        
        h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
            idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                     idx_p_list[(epoch + 8) % 100]]) / 5
        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_n_list = []
        
    # ===================================================# 测试使用提前生成好的负样本索引
        # idx = np.loadtxt('D:/bioinformatics/2_model/GCN_model/SUGRL/datasets/Cora/raw/idxlist.txt')
        # for i in range(idx.shape[1]):
        #     s_n = F.pairwise_distance(h_a, h_a[idx[:,i]])
        #     s_n_list.append(s_n)    
    # ===================================================#    
    
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes) #这样打乱顺序是否会产生实际有边连接的负样本对
            idx_list.append(idx_0)        
        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)

        loss_mar = 0
        loss_mar_1 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
        mask_margin_N = mask_margin_N / num_neg

        #---------------计算预测对比损失------------------#
        # edge = np.loadtxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge0.txt',dtype =int)
        # edge1 = np.loadtxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge1.txt',dtype =int)
        # posrna = h_p[edge[:,0]]
        # posdis = h_p[edge[:,1]]
        # negrna = h_p[edge1[:,0]]
        # negdis = h_p[edge1[:,1]]    
        # pospred = torch.sum(posrna * posdis, axis=-1) 
        # negpred = torch.sum(negrna * negdis, axis=-1)
        # preLoss = torch.max(lbl_z, 1.0 - (pospred - negpred)).sum()

        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3# + 0.01*preLoss
        loss.backward()
        optimiser.step()
        # string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||loss_4: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
        #                                                                       mask_margin_N.item(),preLoss.item())
        # if (epoch+1)%50 ==0:
        #     print(string_1)
        if args.pretrain:
            if os.path.exists(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth'):
                    load_params = torch.load(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth',map_location=torch.device('cpu'))
                    model_params = model.state_dict()
                    same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
                    model_params.update(same_parsms)
                    model.load_state_dict(model_params)
        if args.save_model:
            torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
        if epoch % args.epochs == 0 and epoch != 0:
            model.eval()
            h_a, h_p = model.embed(feature_X, A_I_nomal)
            # h_a, h_p = model.embed(feature_X, [A_I_nomal,edge])#测试使用其他GCN模型生成嵌入
            embs = h_p
            embs = embs / embs.norm(dim=1)[:, None]
            sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_1)
            print('AUC', sc_roc, 'AP', sc_ap)

    return embs.cpu().detach().numpy(),h_a.cpu().detach().numpy()


















