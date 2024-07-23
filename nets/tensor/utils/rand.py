from nets.tensor._element import _Element
from nets.tensor.tensor import Tensor

import random
from typing import Tuple


def randn(dimensions: Tuple[int], mode: str='gauss', seed: int=None) -> Tensor:
    '''
    The function that constructs a tensor and filling it with random elements.

    :param dimensions: The dimensions of the returing tensor.
    :param mode: The distribution of which the random values will be taken.
    :param seed: The seed of the random number generator.

    :return: The random initialized tensor.
    '''
    if seed is not None:
        random.seed(seed)

    def _randn(dims):
        if len(dims) == 0:
            return _Element(random.gauss(0, 1)) if mode == 'gauss' else _Element(random.uniform(-1, 1))
        else:
            return [_randn(dims[1:]) for _ in range(dims[0])]

    res_elements = _randn(dimensions)
    return Tensor.from_elements(res_elements)
