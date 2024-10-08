import networkx as nx
import numpy as np
import gudhi as gd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])

    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmOrd0 if p[0] == 0]) if len(
        dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmRel1 if p[0] == 1]) if len(
        dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt0 if p[0] == 0]) if len(
        dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt1 if p[0] == 1]) if len(
        dgmExt1) else np.empty([0, 2])
    final_dgm = np.concatenate([dgmOrd0, dgmExt0, dgmRel1, dgmExt1], axis=0)
    return final_dgm

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    if PXs.shape[0]==0 and PYs.shape[0]==0:
        norm_output = np.zeros((resolution)) + 1e-6
    else:
        xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
        x = np.linspace(xm, xM, resolution[0])
        y = np.linspace(ym, yM, resolution[1])
        X, Y = np.meshgrid(x, y)
        Zfinal = np.zeros(X.shape)
        X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

        # Compute persistence image
        P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
        weight = np.abs(P1 - P0)
        distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

        if return_raw:
            lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
            lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
        else:
            weight = weight ** power
            Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

        output = [lw, lsum] if return_raw else Zfinal

        if normalization:
            norm_output = (output - np.min(output)) / (np.max(output) - np.min(output))
        else:
            norm_output = output

        if np.sum(np.isnan(norm_output))>0:
            norm_output = np.zeros((resolution)) + 1e-6
    return norm_output


def sublevel_persistence_diagram(net, method):
    # for a given adjacency matrix A of a graph using a specified method for determining node importance (filtration values)
    assert method in ['degree','betweenness','communicability','eigenvector','closeness']

    if method == 'degree':
        return np.array(list(nx.degree_centrality(net).values()))
    elif method == 'betweenness':
        return np.array(list(nx.betweenness_centrality(net).values()))
    elif method == 'communicability':
        return np.array(list(nx.communicability_betweenness_centrality(net).values()))
    elif method == 'eigenvector':
        return np.array(list(nx.eigenvector_centrality(net, max_iter=30000, tol=1e-06).values()))
    elif method == 'closeness':
        return np.array(list(nx.closeness_centrality(net).values()))
    # compute persistence diagrams for graph data using various centrality measures as filtration values.


def compute_PI_tensor(graph_list,PI_dim,sublevel_filtration_methods=['degree','betweenness','communicability','eigenvector','closeness']):

    PI_list = [] # Initialize Persistence Image List
    for graph in graph_list:
        edge_index = ((graph.edge_index).numpy()).transpose()
        net = nx.from_edgelist(edge_index)
        adj = nx.adjacency_matrix(net).toarray()
        PI_list_i = [] # Initialize List for Current Graph's Persistence Images
        # PI tensor
        for j in range(len(sublevel_filtration_methods)): # Compute Persistence Diagrams and Images
            scores = sublevel_persistence_diagram(net, sublevel_filtration_methods[j])
            dgm = apply_graph_extended_persistence(A=adj, filtration_val=scores)
            pi = torch.FloatTensor(persistence_images(dgm, resolution=[PI_dim]*2))
            PI_list_i.append(pi)
        PI_tensor_i = torch.stack(PI_list_i) # Stack Current Graph's Persistence Images into Tensor
        PI_list.append(PI_tensor_i) # Add to Main List

    PI_concat = torch.stack(PI_list) # Concatenate All Graphs' Tensors
    return PI_concat
    # compute tensors of persistence images from graphs using various sublevel filtration methods

def remove_labels(graph_list):
    new_graph_list = []

    for item in graph_list:
        if isinstance(item, tuple):
            graph = item[0]
        else:
            graph = item

        new_graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr if 'edge_attr' in graph else None,
            pos=graph.pos if 'pos' in graph else None
        )
        new_graph_list.append((new_graph,) + item[1:]) if isinstance(item, tuple) else new_graph_list.append(new_graph)

    return new_graph_list


def loss_cal(self, x_aug1, x_aug2):
    T = 0.2
    batch_size = x_aug1.size(0)
    feature_size = x_aug1.size(1)
    
    x_abs = x_aug1.norm(dim=1)
    x_aug_abs = x_aug2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x_aug1, x_aug2) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = torch.diag(sim_matrix)

    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss