from nets.tensor.tensor import Tensor
from nets.tensor.utils import randn
from nets.nn.base import Module


class Linear(Module):
    '''
    The basic linear layer of logistic regression
    '''
    def __init__(self, in_features: int, out_features: int):
        '''
        Initialize the Linear class module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        ''' 
        self.w = randn(dimensions=(in_features, out_features))
        self.b = randn(dimensions=(out_features,))

    def __call__(self, x: Tensor) -> Tensor:
        '''
        The forward method of the layer

        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The output of the Linear layer.
        '''
        return x.dot(self.w) + self.b
