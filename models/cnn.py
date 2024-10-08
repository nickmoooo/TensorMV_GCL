import torch.nn as nn

# for PI block
class CNN(nn.Module):
    def __init__(self, dim_in, hidden_dim, kernel_size, stride):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.cnn = nn.Sequential(   # a sequential container 'self.cnn' contains a series of layers
            nn.Conv2d(dim_in, hidden_dim, kernel_size=kernel_size, stride=stride),  # a 2D convolutional layer with dim_in as input and hidden_dim as output
            nn.ReLU(inplace=True),  # a ReLU activation function
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)    # a 2D max pooling layer 
        )
    def forward(self, PI):
        feature = self.cnn(PI)
        return feature

    # compute output dim given the above kernel_size and stride
    def cnn_output_dim(self, PI_dim):
        tmp_dim = int((PI_dim-self.kernel_size)/self.stride)+1
        output_dim = int((tmp_dim-self.kernel_size)/self.stride)+1
        return output_dim
    
# CNN class represents a simple convolutional neural network block with a single convolutional layer 
# followed by a ReLU activation and a max pooling layer. It also includes a method to compute output dimension of CNN.