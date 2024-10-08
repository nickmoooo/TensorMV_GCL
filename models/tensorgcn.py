import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import tensorly as tl
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from tensorly.decomposition import tucker, parafac

from typing import Optional, Union
from torch_geometric.utils import scatter
import sys

sys.path.append("models/")
from models.gin import Encoder
from mlp import MLP, MLP_output
from cnn import CNN
from diagram import sum_diag_from_point_cloud
#pip install tensorly
from tltorch import TRL, TCL
from torch_geometric.data import Data
from util import loss_cal


def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = torch.cumsum(num_nodes, dim=0)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]


def maybe_num_nodes(edge_index, num_nodes=None):
    """
    Infers or validates the number of nodes in a graph.

    Args:
    - edge_index (Tensor): The edge indices of the graph.
    - num_nodes (Optional[int]): The number of nodes, if known.

    Returns:
    - int: The number of nodes in the graph.
    """

    max_index = edge_index.max().item()  # Find the maximum node index
    inferred_num_nodes = max_index + 1  # Assume node indices start at 0

    # If num_nodes is provided, validate it; otherwise, use the inferred number
    if num_nodes is not None:
        assert num_nodes >= inferred_num_nodes, \
            "Provided num_nodes is less than the number of nodes inferred from edge_index."
        return num_nodes
    else:
        return inferred_num_nodes


def filter_adj(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        node_index: Tensor,
        cluster_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if cluster_index is None:
        cluster_index = torch.arange(node_index.size(0),
                                     device=node_index.device)

    mask = node_index.new_full((num_nodes,), -1)
    mask[node_index] = cluster_index

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


def tucker_decompose_core(tensor, rank):
    tensor = tl.tensor(tensor)
    core, _ = tucker(tensor, rank=rank)
    return core

def cp_decompose_core(tensor, rank):
    tensor = tl.tensor(tensor)  # Convert to TensorLy tensor
    factors = parafac(tensor, rank=rank)  # Perform CP decomposition
    return tl.kruskal_to_tensor(factors)  # Reconstruct the tensor using the decomposed factors


class TenGCN(nn.Module): 
    def __init__(self, num_gc_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, tensor_layer_type, node_pooling, PI_dim, sublevel_filtration_methods, device, prior, num_features,
                 alpha=0.5, beta=1., gamma=.5):
        '''
            num_layers: number of GCN layers (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of for all hidden units
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            tensor_layer: Tensor layer type, TCL/TRL'
            PI_dim: int size of PI
            sublevel_filtration_methods: methods to generate PD
            device: which device to use
            num_features: number of features of each node
        '''

        super(TenGCN, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.device = device
        self.num_layers = num_gc_layers
        self.num_neighbors = 5
        self.hidden_dim = hidden_dim

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2) # GCN layer for scoring nodes based on their features, 
        self.node_pooling = node_pooling                                   # potentially for use in a node pooling mechanism. The 
                                                                           # layer outputs features of dimension self.num_neighbors * 2 
        self.dropout = nn.Dropout(p=final_dropout)
        self.init_emb()

        # GCN block = gcn + mlp 
        # initialize GCN and MLP layers
        self.GCNs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.GCNs.append(GCNConv(input_dim,hidden_dim))
            else:
                self.GCNs.append(GCNConv(hidden_dim**2,hidden_dim))  #  Squaring the dimensionality
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim**2))  # multiple mlp and gcn

        # tensor layer
        # Initializes a tensor layer (TCL or TRL) for the GCN block with specified input and hidden shapes.
        tensor_input_shape = (self.num_layers-1,hidden_dim,hidden_dim)
        tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim]
        if tensor_layer_type == 'TCL':
            self.GCN_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.GCN_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)

        # PI tensor block
        # CNN
        self.cnn = CNN(len(sublevel_filtration_methods),hidden_dim,kernel_size=2,stride=2)
        cnn_output_shape = self.cnn.cnn_output_dim(PI_dim)

        # tensor layer
        tensor_input_shape = (hidden_dim,cnn_output_shape,cnn_output_shape)
        tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim]
        if tensor_layer_type == 'TCL':
            self.PI_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.PI_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)
        # Initializes a CNN layer for the PI block to process persistence images.
        # Computes the output shape of the CNN.
        # Initializes a tensor layer for the PI block with the computed input and hidden shapes.
        

        # output block
        self.attend = nn.Linear(2*hidden_dim, 1)
        tensor_input_shape = (2*hidden_dim,hidden_dim,hidden_dim)
        tensor_hidden_shape = [2*hidden_dim,hidden_dim,hidden_dim]
        # self.attend = nn.Linear(hidden_dim, 1)
        # tensor_input_shape = (hidden_dim,hidden_dim,hidden_dim)
        # tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim]

        if tensor_layer_type == 'TCL':
            self.output_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.output_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)


        self.mlp = MLP(num_mlp_layers, (num_gc_layers - 1) * (hidden_dim ** 2), hidden_dim=64, output_dim=2)

        self.output = MLP_output(hidden_dim,output_dim,final_dropout) # Initializes the final MLP output layer with specified dimensions and dropout rate.

        self.transform_attention = nn.Linear(32 * 32, 2)


    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def compute_batch_feat(self, batch_graph):   # batch feature computation, focusing on graph data manipulation and pooling
        edge_attr = None
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []

        for i, graph in enumerate(batch_graph):
            x = graph.x.to(self.device)  # move Node features of the current graph to the specified device (CPU or GPU)
            edge_index = graph.edge_index.to(self.device)

            if self.node_pooling:
                # graph pooling based on node topological scores
                node_embeddings = self.score_node_layer(x, edge_index).to(self.device)
                node_point_clouds = node_embeddings.view(-1, self.num_neighbors, 2).to(self.device) # Reshape the embeddings to prepare for pooling,
                score_lifespan = torch.FloatTensor([sum_diag_from_point_cloud(node_point_clouds[i,...]) for i in range(node_point_clouds.size(0))]).to(self.device)
                # Computes a score for each node based on its topological significance
                batch = torch.LongTensor([0] * x.size(0)).to(self.device)
                perm = topk(score_lifespan, 0.5, batch) # Indices of the top scoring nodes based on score_lifespan, keeping 50% of the nodes.
                x = x[perm] # Updates x to only include features of nodes selected by perm.
                edge_index, _ = filter_adj(edge_index, edge_attr, perm, num_nodes=graph.x.size(0))
                # Adjusts edge_index to only include edges that connect the selected nodes.
            
            start_idx.append(start_idx[i] + x.size(0)) # Update start_idx for the next graph.
            edge_mat_list.append(edge_index + start_idx[i]) 
            pooled_x.append(x)
            pooled_graph_sizes.append(x.size(0))

        pooled_X_concat = torch.cat(pooled_x, 0).to(self.device)
        Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device) # Concatenate all pooled node features and edge matrices to form the batch inputs for subsequent layers.
        return Adj_block_idx, pooled_X_concat, pooled_graph_sizes # Returns the concatenated adjacency indices, pooled node features, and sizes of pooled graphs.

    # Defines a single GCN layer that applies a GCN convolution followed by an MLP.
    def GCN_layer(self, h, edge_index, layer):
        h = self.GCNs[layer](h, edge_index)
        h = self.mlps[layer](h)
        return h
            



    def forward(self, batch_graph1, batch_graph2, PIs_aug1, PIs_aug2, args):
        if (args.mode == 'train'):
            Adj_block_idx1, pooled_X_concat1, pooled_graph_sizes1 = self.compute_batch_feat(batch_graph1)
            Adj_block_idx2, pooled_X_concat2, pooled_graph_sizes2 = self.compute_batch_feat(batch_graph2)

            #NOTE Addtional layer of tensor decomposition for EPH


            # rank = args.rank
            # PIs_aug1 = PIs_aug1.cpu().numpy()
            # PIs_aug2 = PIs_aug2.cpu().numpy()
            # core_PIs_aug1 = []
            # core_PIs_aug2 = []

            # for tensor in PIs_aug1:
            #     core_slices = [cp_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug1.append(core_tensor)

            # for tensor in PIs_aug2:
            #     core_slices = [cp_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug2.append(core_tensor)
            
            # core_PIs_aug1 = torch.tensor(core_PIs_aug1).to(self.device)
            # core_PIs_aug2 = torch.tensor(core_PIs_aug2).to(self.device)

            # # Pass decomposed tensors to CNN
            # print(f"Shape of core_Pis: {core_PIs_aug1.shape}")  # Assuming this is the weight in the layer causing the issue
            # PI_emb1 = self.cnn(core_PIs_aug1)
            # PI_hidden1 = self.PI_tensor_layer(PI_emb1).to(self.device)
            # PI_emb2 = self.cnn(core_PIs_aug2)
            # PI_hidden2 = self.PI_tensor_layer(PI_emb2).to(self.device)



            # rank = [args.rank, args.rank]  # Example rank for Tucker decomposition, adjust based on your data
            # PIs_aug1 = PIs_aug1.cpu().numpy()  # Convert to numpy array for tensorly
            # PIs_aug2 = PIs_aug2.cpu().numpy()

            # core_PIs_aug1 = []
            # core_PIs_aug2 = []

            # for tensor in PIs_aug1:
            #     core_slices = [tucker_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug1.append(core_tensor)
            
            # for tensor in PIs_aug2:
            #     core_slices = [tucker_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug2.append(core_tensor)

            # core_PIs_aug1 = torch.tensor(core_PIs_aug1).to(self.device)
            # core_PIs_aug2 = torch.tensor(core_PIs_aug2).to(self.device)


            # PI_emb1 = self.cnn(core_PIs_aug1)
            # PI_hidden1 = self.PI_tensor_layer(PI_emb1).to(self.device)
            # PI_emb2 = self.cnn(core_PIs_aug2)
            # PI_hidden2 = self.PI_tensor_layer(PI_emb2).to(self.device)


            PI_emb1 = self.cnn(PIs_aug1)
            PI_hidden1 = self.PI_tensor_layer(PI_emb1).to(self.device)
            PI_emb2 = self.cnn(PIs_aug2)
            PI_hidden2 = self.PI_tensor_layer(PI_emb2).to(self.device)

            hidden_rep1 = []
            h1 = pooled_X_concat1
            edge_index1 = Adj_block_idx1

            hidden_rep2 = []
            h2 = pooled_X_concat2
            edge_index2 = Adj_block_idx2

            for layer in range(self.num_layers - 1):
                h1 = self.GCN_layer(h1, edge_index1, layer)
                hidden_rep1.append(h1)

                h2 = self.GCN_layer(h2, edge_index2, layer)
                hidden_rep2.append(h2)

            hidden_rep1 = torch.stack(hidden_rep1).transpose(0, 1)
            graph_sizes1 = pooled_graph_sizes1
            node_embeddings1 = torch.split(hidden_rep1, graph_sizes1, dim=0)

            hidden_rep2 = torch.stack(hidden_rep2).transpose(0, 1)
            graph_sizes2 = pooled_graph_sizes2
            node_embeddings2 = torch.split(hidden_rep2, graph_sizes2, dim=0)

            gcn_tensor1 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
            gcn_tensor2 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)

            pi_tensor1 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
            pi_tensor2 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)

            for g_i in range(len(graph_sizes1)):
                # GCN tensor
                cur_node_embeddings1 = node_embeddings1[g_i]
                cur_node_embeddings1 = cur_node_embeddings1.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
                cur_node_embeddings_hidden1 = self.GCN_tensor_layer(cur_node_embeddings1).to(self.device)
                cur_graph_tensor_hidden1 = torch.mean(cur_node_embeddings_hidden1, dim=0)

                cur_node_embeddings2 = node_embeddings2[g_i]
                cur_node_embeddings2 = cur_node_embeddings2.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
                cur_node_embeddings_hidden2 = self.GCN_tensor_layer(cur_node_embeddings2).to(self.device)
                cur_graph_tensor_hidden2 = torch.mean(cur_node_embeddings_hidden2, dim=0)

                gcn_tensor1[g_i] = cur_graph_tensor_hidden1
                gcn_tensor2[g_i] = cur_graph_tensor_hidden2


                # PI tensor
                cur_PI_tensor_hidden1 = PI_hidden1[g_i]
                cur_PI_tensor_hidden2 = PI_hidden2[g_i]
                pi_tensor1[g_i] = cur_PI_tensor_hidden1
                pi_tensor2[g_i] = cur_PI_tensor_hidden2
            

            gcn_tensor1_flat = gcn_tensor1.reshape(gcn_tensor1.size(0), -1)
            gcn_tensor2_flat = gcn_tensor2.reshape(gcn_tensor2.size(0), -1)

            gcn_tensor_loss = loss_cal(self, gcn_tensor1_flat, gcn_tensor2_flat)

            pi_tensor1_flat = pi_tensor1.reshape(pi_tensor1.size(0), -1)
            pi_tensor2_flat = pi_tensor2.reshape(pi_tensor2.size(0), -1)

            pi_tensor_loss = loss_cal(self, pi_tensor1_flat, pi_tensor2_flat) 

            loss = 1 * gcn_tensor_loss + 0.3 * pi_tensor_loss

            return loss

        
        # if (args.mode == 'tda'):
        #     Adj_block_idx1, pooled_X_concat1, pooled_graph_sizes1 = self.compute_batch_feat(batch_graph1)
        #     graph_sizes1 = pooled_graph_sizes1
        #     PI_emb1 = self.cnn(PIs_aug1)
        #     PI_hidden1 = self.PI_tensor_layer(PI_emb1).to(self.device)
        #     PI_emb2 = self.cnn(PIs_aug2)
        #     PI_hidden2 = self.PI_tensor_layer(PI_emb2).to(self.device)

        #     pi_tensor1 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
        #     pi_tensor2 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)

        #     for g_i in range(len(graph_sizes1)):
        #         cur_PI_tensor_hidden1 = PI_hidden1[g_i]
        #         cur_PI_tensor_hidden2 = PI_hidden2[g_i]
        #         pi_tensor1[g_i] = cur_PI_tensor_hidden1
        #         pi_tensor2[g_i] = cur_PI_tensor_hidden2
            
        #     pi_tensor1_flat = pi_tensor1.reshape(pi_tensor1.size(0), -1)
        #     pi_tensor2_flat = pi_tensor2.reshape(pi_tensor2.size(0), -1)

        #     pi_tensor_loss = loss_cal(self, pi_tensor1_flat, pi_tensor2_flat) 

        #     return pi_tensor_loss
        
        # elif (args.mode == 'gcn'):
        #     Adj_block_idx1, pooled_X_concat1, pooled_graph_sizes1 = self.compute_batch_feat(batch_graph1)
        #     Adj_block_idx2, pooled_X_concat2, pooled_graph_sizes2 = self.compute_batch_feat(batch_graph2)

        #     hidden_rep1 = []
        #     h1 = pooled_X_concat1
        #     edge_index1 = Adj_block_idx1

        #     hidden_rep2 = []
        #     h2 = pooled_X_concat2
        #     edge_index2 = Adj_block_idx2

        #     for layer in range(self.num_layers - 1):
        #         h1 = self.GCN_layer(h1, edge_index1, layer)
        #         hidden_rep1.append(h1)

        #         h2 = self.GCN_layer(h2, edge_index2, layer)
        #         hidden_rep2.append(h2)

        #     hidden_rep1 = torch.stack(hidden_rep1).transpose(0, 1)
        #     graph_sizes1 = pooled_graph_sizes1
        #     node_embeddings1 = torch.split(hidden_rep1, graph_sizes1, dim=0)

        #     hidden_rep2 = torch.stack(hidden_rep2).transpose(0, 1)
        #     graph_sizes2 = pooled_graph_sizes2
        #     node_embeddings2 = torch.split(hidden_rep2, graph_sizes2, dim=0)

        #     gcn_tensor1 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
        #     gcn_tensor2 = torch.zeros(len(graph_sizes1), self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)

        #     for g_i in range(len(graph_sizes1)):
        #         cur_node_embeddings1 = node_embeddings1[g_i]
        #         cur_node_embeddings1 = cur_node_embeddings1.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
        #         cur_node_embeddings_hidden1 = self.GCN_tensor_layer(cur_node_embeddings1).to(self.device)
        #         cur_graph_tensor_hidden1 = torch.mean(cur_node_embeddings_hidden1, dim=0)

        #         cur_node_embeddings2 = node_embeddings2[g_i]
        #         cur_node_embeddings2 = cur_node_embeddings2.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
        #         cur_node_embeddings_hidden2 = self.GCN_tensor_layer(cur_node_embeddings2).to(self.device)
        #         cur_graph_tensor_hidden2 = torch.mean(cur_node_embeddings_hidden2, dim=0)

        #         gcn_tensor1[g_i] = cur_graph_tensor_hidden1
        #         gcn_tensor2[g_i] = cur_graph_tensor_hidden2

        #     gcn_tensor1_flat = gcn_tensor1.reshape(gcn_tensor1.size(0), -1)
        #     gcn_tensor2_flat = gcn_tensor2.reshape(gcn_tensor2.size(0), -1)

        #     gcn_tensor_loss = loss_cal(self, gcn_tensor1_flat, gcn_tensor2_flat)

        #     return gcn_tensor_loss

        
        else:
            Adj_block_idx, pooled_X_concat, pooled_graph_sizes = self.compute_batch_feat(batch_graph1)
        
            # PI block
            # CNN
            PI_emb = self.cnn(PIs_aug1) # before cnn: [batch_size,5,PI_dim,PI_dim]; after cnn: [batch_size,hidden_dim,cnn_output_shape,cnn_output_shape]
            PI_emb_dim = PI_emb.shape
            # print(PI_emb_dim)
            # torch.save(PI_emb, '{}_{}_x_pit.pt'.format(args.dataset, PI_emb_dim))
            # tensor layer
            PI_hidden = self.PI_tensor_layer(PI_emb).to(self.device) # [batch_size,hidden_dim,hidden_dim,hidden_dim]
            PI_hidden_dim = PI_hidden.shape
            # print(PI_hidden_dim)
            torch.save(PI_hidden, '{}_{}_x_prime_pit.pt'.format(args.dataset, PI_hidden_dim)) 

            # #NOTE Addtional layer of tensor decomposition for EPH
            # rank = args.rank
            # PIs_aug = PIs_aug1.cpu().numpy()
            # core_PIs_aug = []

            # for tensor in PIs_aug:
            #     core_slices = [cp_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug.append(core_tensor)

            # core_PIs_aug = torch.tensor(core_PIs_aug).to(self.device)

            # # Pass decomposed tensors to CNN
            # PI_emb = self.cnn(core_PIs_aug)
            # PI_hidden = self.PI_tensor_layer(PI_emb).to(self.device)


            # rank = [args.rank, args.rank]  # Example rank for Tucker decomposition, adjust based on your data
            # PIs_aug = PIs_aug1.cpu().numpy()  # Convert to numpy array for tensorly

            # core_PIs_aug = []

            # for tensor in PIs_aug:
            #     core_slices = [tucker_decompose_core(tensor[i], rank) for i in range(tensor.shape[0])]
            #     core_tensor = np.stack(core_slices)  # Stack slices back to the original shape
            #     core_PIs_aug.append(core_tensor)

            # core_PIs_aug = torch.tensor(core_PIs_aug).to(self.device)


            # PI_emb = self.cnn(core_PIs_aug)
            # PI_hidden = self.PI_tensor_layer(PI_emb).to(self.device)


            ## GCN block
            hidden_rep = []
            h = pooled_X_concat
            edge_index = Adj_block_idx
            for layer in range(self.num_layers-1):
                h = self.GCN_layer(h, edge_index, layer) # shape: [start_idx[-1]=N,hidden_dim**2]
                hidden_rep.append(h)
            # batch GCN tensor
            hidden_rep = torch.stack(hidden_rep).transpose(0,1) # shape: [start_idx[-1]=N, self.num_layers-1, hidden_dim**2]

            ## graph tensor concat
            graph_sizes = pooled_graph_sizes
            batch_graph_tensor = torch.zeros(len(graph_sizes), 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim).to(self.device)
            node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

            gcn_tensors_before = []
            gcn_tensors_after = []

            for g_i in range(len(graph_sizes)):
                # current graph GCN tensor
                cur_node_embeddings = node_embeddings[g_i] # (n,self.num_layers-1,hidden_dim**2)
                cur_node_embeddings = cur_node_embeddings.view(-1,self.num_layers-1,self.hidden_dim,self.hidden_dim) # (n,self.num_layers-1,hidden_dim,hidden_dim)
                gcn_tensors_before.append(cur_node_embeddings.clone().detach())
                print(cur_node_embeddings.shape)
                cur_node_embeddings_hidden = self.GCN_tensor_layer(cur_node_embeddings).to(self.device) # (n,hidden_dim,hidden_dim,hidden_dim)
                cur_graph_tensor_hidden = torch.mean(cur_node_embeddings_hidden,dim=0) # (hidden_dim,hidden_dim,hidden_dim)
                gcn_tensors_after.append(cur_graph_tensor_hidden.clone().detach())
                # print(cur_graph_tensor_hidden.shape)
                # concat with PI tensor
                cur_PI_tensor_hidden = PI_hidden[g_i] # (hidden_dim,hidden_dim,hidden_dim)
                cur_tensor_hidden = torch.cat([cur_graph_tensor_hidden, cur_PI_tensor_hidden], dim=0) # (2*hidden_dim,hidden_dim,hidden_dim)
                # cur_tensor_hidden = (cur_graph_tensor_hidden + cur_PI_tensor_hidden)/2
                batch_graph_tensor[g_i] = cur_tensor_hidden

            # max_nodes = max([tensor.size(0) for tensor in gcn_tensors_before])
            # def pad_tensor(tensor, max_nodes):
            #     print(tensor.shape)
            #     # Current tensor size
            #     n, layers, hidden_dim, hidden_dim = tensor.size()
            #     # Padding along the first dimension (number of nodes)
            #     pad_size = (0, 0, 0, 0, 0, 0, 0, max_nodes - n)
            #     # Apply padding (on node dimension)
            #     return F.pad(tensor, pad_size, 'constant', 0)

            # # Pad each tensor to have the same number of nodes (max_nodes)
            # gcn_tensors_before_padded = [pad_tensor(tensor, max_nodes) for tensor in gcn_tensors_before]

            # Stack the padded tensors into a single tensor
            # gcn_tensors_before_stacked = torch.stack(gcn_tensors_before_padded)
            gcn_tensors_after_stacked = torch.stack(gcn_tensors_after)
            # # Save the list of tensors directly without stacking
            # torch.save(gcn_tensors_before_stacked, '{}_s_g_list.pt'.format(args.dataset))
            torch.save(gcn_tensors_after_stacked, '{}_x_prime_g.pt'.format(args.dataset))


            batch_graph_tensor = self.output_tensor_layer(batch_graph_tensor).transpose(1, 3)
            batch_graph_attn = self.attend(batch_graph_tensor).squeeze()
            score = self.output(batch_graph_attn)

            return score
        

