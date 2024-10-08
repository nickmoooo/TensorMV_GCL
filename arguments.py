import argparse

def arg_parse():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="DHFR",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less than 10.')
    parser.add_argument('--num_gc_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--tensor_layer_type', type=str, default="TCL", choices=["TCL","TRL"],
                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--node_pooling', action="store_false",
                        help='node pooling based on node scores')
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','eigenvector','closeness'],
                        help='Methods for sublevel filtration on PDs')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')

    return parser.parse_args()

