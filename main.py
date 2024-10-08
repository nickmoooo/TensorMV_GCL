import os
from sys import argv
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import DataLoader
from aug import TUDataset_aug as TUDataset_aug
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx

from util import *

from tqdm import trange
from models.tensorgcn import TenGCN
from model import *
from arguments import arg_parse


import collections
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

criterion = nn.CrossEntropyLoss()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def match_nodes(original_data, data_aug, device): 
    if args.aug in ['dnodes', 'subgraph', 'random2', 'random3', 'random4']:
        node_num, _ = original_data.x.size()
        edge_idx = data_aug.edge_index.numpy()
        _, edge_num = edge_idx.shape
        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

        node_num_aug = len(idx_not_missing)
        data_aug.x = data_aug.x[idx_not_missing]

        if original_data.batch is not None:
            data_aug.batch = original_data.batch[idx_not_missing]

        idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
        data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1).to(device)

        data_aug = data_aug.to(device)

    return data_aug


def pad_graphs(graphs, max_nodes):
    for graph in graphs:
        num_nodes = graph.x.size(0)
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            # Pad node features
            padding = torch.zeros(pad_size, graph.x.size(1)).to(graph.x.device)
            graph.x = torch.cat([graph.x, padding], dim=0)

    return graphs


def pass_data_iteratively(model, graphs, PIs, args, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    args.mode = 'eval'

    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        embeddings = model(batch_graph, None, batch_PI, None, args)
        output.append(embeddings.detach())
    return torch.cat(output, 0)


def pass_data_iteratively(model, graphs, PIs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))

    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])

        output.append(model(batch_graph, None, batch_PI, None, args).detach())
    return torch.cat(output, 0)


def eval(model, device, test_graphs_eval, test_PIs, args):
    model.eval()
    args.mode = 'eval'
    test_embeddings = pass_data_iteratively(model, test_graphs_eval, test_PIs)
    test_pred = test_embeddings.max(1, keepdim=True)[1]
    test_labels = torch.LongTensor([graph.y for graph in test_graphs_eval]).to(device)
    test_correct = test_pred.eq(test_labels.view_as(test_pred)).sum().cpu().item()
    acc_test = test_correct / float(len(test_graphs_eval))
    return acc_test
    
    
def furthur_train(model, device, graphs, PIs, optimizer, args):
    model.train()
    args.mode = 'fur_train'

    graphs_without_labels = remove_labels(graphs)
    total_iters = args.iters_per_epoch
    pbar = trange(total_iters, unit='batch')

    loss_accum = 0

    for pos in pbar:
        selected_idx = np.random.permutation(len(graphs))[:args.batch_size]

        graphs_with_labels = [graphs[idx] for idx in selected_idx]
        graphs_batch = [graphs_without_labels[idx] for idx in selected_idx]
        PIs_batch = torch.stack([PIs[idx] for idx in selected_idx])

        # Pad both sets of augmented graphs to the maximum number of nodes
        score = model(graphs_batch, None, PIs_batch, None, args)
        labels = torch.LongTensor([graph.y for graph in graphs_with_labels]).to(device)
        loss = criterion(score, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))
    return average_loss


def train(args, model, device, graphs_without_labels, PIs_aug1, PIs_aug2, dataset_aug1_without_labels, dataset_aug2_without_labels, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = trange(total_iters, unit='batch')

    accumulate_loss = 0.0

    for pos in pbar:
        selected_idx = np.random.permutation(len(graphs_without_labels))[:args.batch_size]

        # Prepare the mini-batch data
        batch_graph1 = [match_nodes(graphs_without_labels[idx], dataset_aug1_without_labels[idx], device) for idx in selected_idx]
        batch_graph2 = [match_nodes(graphs_without_labels[idx], dataset_aug2_without_labels[idx], device) for idx in selected_idx]
        batch_PIs1 = torch.stack([PIs_aug1[idx] for idx in selected_idx])
        batch_PIs2 = torch.stack([PIs_aug2[idx] for idx in selected_idx])

        # Get the maximum number of nodes in this batch
        max_nodes = max([graph.x.size(0) for graph in batch_graph1 + batch_graph2])

        # Pad both sets of augmented graphs to the maximum number of nodes
        batch_graph1 = pad_graphs(batch_graph1, max_nodes)
        batch_graph2 = pad_graphs(batch_graph2, max_nodes)

        optimizer.zero_grad()
        args.mode = 'train'
        loss = model(batch_graph1=batch_graph1, batch_graph2=batch_graph2, PIs_aug1=batch_PIs1, PIs_aug2=batch_PIs2, args=args)
        loss.backward()
        optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        accumulate_loss += loss

        pbar.set_description('Epoch: %d' % (epoch))

    average_loss = accumulate_loss / total_iters
    print("Training loss : %f " % (average_loss))
    return average_loss
    
    
def add_noise_to_PI(pi_tensor, noise_level=0.1):
    noise = torch.randn(pi_tensor.size()) * noise_level  
    noisy_pi_tensor = pi_tensor + noise
    return noisy_pi_tensor




if __name__ == '__main__':
    print('Starting')
    args = arg_parse()
    set_random_seed(42)
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")

    accuracies = {'val':[], 'test':[]}
    args.batch_size = 16
    batch_size = args.batch_size
    args.epochs = 150
    log_interval = 10
    lr = args.lr
    noise_level = 0.1
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets')
    
    # Two sets of different augmented views of graphs
    args.dataset = 'PTC_MR'
    dataset1 = TUDataset_aug(path, name=args.dataset, aug='dnodes')
    dataset2 = TUDataset_aug(path, name=args.dataset, aug='pedges')
    graphs_Train = TUDataset_aug(root=path, name=args.dataset, aug='none')
    # graphs_Eval = TUDataset_aug(root=path, name=args.dataset, aug='none').shuffle()

    num_classes = graphs_Train.num_classes
    try:
        input_dim = graphs_Train.get_num_feature()
    except:
        input_dim = 1
    # num_features = graphs_Train.get_num_feature()
    num_features = 0

    dataset_aug1 = [data_aug for data, data_aug in dataset1]
    dataset_aug2 = [data_aug for data, data_aug in dataset2]
    graphs_train = [data for data, data_aug in graphs_Train]
    # graphs_eval = [data for data, data_aug in graphs_Eval]
    

    # NOTE: compute graph PI tensor if necessary
    PIs = compute_PI_tensor(graphs_train, args.PI_dim, args.sublevel_filtration_methods) ##### When compute PIs during training, should the lable be removed immediately or after
    noisy_PIs = add_noise_to_PI(PIs, noise_level=noise_level)
    torch.save(noisy_PIs, '{}_{}_PI_noisy0.1.pt'.format(args.dataset, args.PI_dim))

    PIs_aug1 = compute_PI_tensor(dataset_aug1, args.PI_dim, args.sublevel_filtration_methods)
    noisy_PIs_aug1 = add_noise_to_PI(PIs_aug1, noise_level=noise_level)
    torch.save(noisy_PIs_aug1, '{}_{}_PI_dnodes_noisy0.1.pt'.format(args.dataset, args.PI_dim))

    PIs_aug2 = compute_PI_tensor(dataset_aug2, args.PI_dim, args.sublevel_filtration_methods)
    noisy_PIs_aug2 = add_noise_to_PI(PIs_aug2, noise_level=noise_level)
    torch.save(noisy_PIs_aug2, '{}_{}_PI_pedges_noisy0.1.pt'.format(args.dataset, args.PI_dim))

    noisy_ori_PIs = torch.load('{}_{}_PI_noisy.pt'.format(args.dataset, args.PI_dim)).to(device)
    noisy_aug1_PIs = torch.load('{}_{}_PI_dnodes_noisy0.1.pt'.format(args.dataset, args.PI_dim)).to(device)
    noisy_aug2_PIs = torch.load('{}_{}_PI_pedges_noisy0.1.pt'.format(args.dataset, args.PI_dim)).to(device)

    
    train_graphs = graphs_train
    graphs_without_labels = remove_labels(train_graphs)
    dataset_aug1_without_labels = remove_labels(dataset_aug1)
    dataset_aug2_without_labels = remove_labels(dataset_aug2)

    model = TenGCN(args.num_gc_layers, args.num_mlp_layers, input_dim, args.hidden_dim, num_classes, args.final_dropout, args.tensor_layer_type, 
                      args.node_pooling, args.PI_dim, args.sublevel_filtration_methods, device, args.prior, num_features
                      ).to(device)
    # model.load_state_dict(torch.load('trained_model_' + args.dataset + '_1_noisy.pth'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    max_acc = 0.0
    acc_val = 0.0

    print("Starting training...")
    accuracies = {'val': [], 'test': []}
    for epoch in range(1, args.epochs + 1):
        args.mode = 'train'
        avg_loss = train(args, model, device, graphs_without_labels, noisy_aug1_PIs, noisy_aug2_PIs, dataset_aug1_without_labels, dataset_aug2_without_labels, optimizer, epoch)
        scheduler.step()


    # for epoch in range(1, args.epochs + 1):
    #     train_graphs_eval, test_graphs_eval, train_PIs, test_PIs = train_test_split(graphs_eval, noisy_ori_PIs, test_size=0.1)
    #     avg_loss = furthur_train(model, device, train_graphs_eval, train_PIs, optimizer, args)
    #     scheduler.step()
    #     acc_test_1 = eval(model, device, test_graphs_eval, test_PIs, args)

    #     if epoch % 1 == 0:
    #         train_graphs_eval, test_graphs_eval, train_PIs, test_PIs = train_test_split(train_graphs_eval, train_PIs, test_size=0.1)
    #         avg_loss = furthur_train(model, device, train_graphs_eval, train_PIs, optimizer, args)
    #         scheduler.step()
    #         acc_test_2 = eval(model, device, test_graphs_eval, test_PIs, args)
    #         print(f"Epoch {epoch}: Test Accuracy: {acc_test_1:.4f} | Validation Accuracy: {acc_test_2:.4f}")
    #         with open('NEW_Result_' + args.dataset +'_2_noisy0.1.txt', 'a+') as f:
    #             f.write(f'Epoch {epoch}: Test Acc: {acc_test_1} | Val Acc: {acc_test_2}\n')
    
    torch.save(model.state_dict(), 'trained_model_' + args.dataset + '_1_noisy0.1' + '.pth') 