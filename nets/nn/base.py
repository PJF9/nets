from nets.tensor.tensor import Tensor


class Module:
    '''
    The base class containing common functionality accross all layers
    '''
    def zero_grad(self) -> None:
        '''
        Reseting layer's gradients
        '''
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> Tensor:
        '''
        The tensor containing all the parameters in the layer
        '''
        return Tensor()
