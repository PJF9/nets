from nets.tensor._element import _Element

from typing import Tuple, Union, List, Generator, Iterator


# Define the aliases that tensor uses
dtype = Union[int, float]
NestedList = Union[dtype, List['NestedList']]
NestedTensor = Union[_Element, List['NestedTensor']]


class Tensor:
    '''
    A class to represent tensors of any dimension in Python.
    '''
    def __init__(self, data: NestedList) -> None:
        '''
        Initializes a Tensor object with the specified dimensions.

        :param data: The content of the tensor
        '''
        self.data = self._initialize_data(data)
        self.shape = self._find_dimensions(data)

    @staticmethod
    def _initialize_data(value: NestedList) -> NestedTensor:
        '''
        Recursively initializes the data structure for the tensor.

        :param dimensions: The dimensions of the tensor
        :param value: The default value that the tensor will be filled with

        :return: The initialized data structure
        '''
        if isinstance(value, (int, float)):
            return _Element(value)
        
        return [Tensor._initialize_data(v) for v in value]

    @staticmethod
    def _find_dimensions(data: Union[NestedList, NestedTensor]) -> Tuple[int]:
        '''
        Recursively finds the dimensions of the tensor

        :param data: The input tensor

        :return: A tuple representing the dimensions of the nested list
        '''
        if isinstance(data, (int, float, _Element)) or not data:
            return ()  # Base case: for a single integer, return an empty tuple
        else:
            return (len(data),) + Tensor._find_dimensions(data[0])  # Recursively find dimensions of sub-lists
    
    @staticmethod
    def from_elements(elements: NestedTensor) -> 'Tensor':
        '''
        Initialize a Tensor object from a nested list of _Elements.

        Args:
            elements (NestedTensor): The elements of the the tensor.

        Returns:
            Tensor: The initialized tensor.
        '''
        res_obj = Tensor([])
        res_obj.data = elements
        res_obj.shape = Tensor._find_dimensions(elements)

        return res_obj

    @staticmethod
    def from_shape(shape: Tuple[int], value: dtype=0.0) -> 'Tensor':
        '''
        Initialize a Tensor from a given shape.

        Args:
            shape (Tuple[int]): The shape that the resulting tensor will have
            value (dtype): The value that will fill the tensor. Default is 0.0.
        
        Returns:
            Tensor: The new initialized tensor.
        '''
        def _from_shape(dims):
            if len(dims) == 0:
                return _Element(value)  # Scalars have no dims, so data is just a number
            else:
                return [_from_shape(dims[1:]) for _ in range(dims[0])]
        
        res_elements = _from_shape(shape)
        return Tensor.from_elements(res_elements)
    
    def _get_value(self, *indices: Tuple[int]) -> NestedTensor:
        '''
        Retrieves the value from the specified indices in the tensor.

        :param indices: Variable number of indices to access the _Element in the tensor

        :return: The value at the specified indices
        '''
        current = self.data
        for index in indices:
            current = current[index]
        return current
    
    def _assign_value(self, value: NestedTensor, *indices: Tuple[int]) -> None:
        '''
        Assigns a value to the specified indices in the tensor. Doesn't keep the
            gradient of the previous _Element, it overwrites the _Element completely.

        :param value: The value to be assigned
        :param indices: Variable number of indices to access the _Element in the tensor
        '''
        current = self.data
        for index in indices[:-1]:
            current = current[index]

        if isinstance(current[indices[-1]], _Element):
            current[indices[-1]] = _Element(value)
        else:
            current[indices[-1]] = Tensor._initialize_data(value)
    
    def _iterate_data(self) -> Generator[NestedTensor, None, None]:
        '''
        Recursively iterate over all _Elements of the tensor data

        :return: A generator yielding _Elements of type dtype
        '''
        if len(self.shape) == 0:
            raise ValueError('Scalar tensors are not iterable.')

        for i in range(self.shape[0]):
            yield self.data[i]
    
    def __str__(self) -> str:
        '''
        Convert the tensor to string
        '''
        return f'Tensor({str(self.data)})'
    
    def __repr__(self) -> str:
        '''
        Convert the tensor to string
        '''
        return str(self)
    
    def __len__(self) -> int:
        '''
        Return the length of the first dimension of the tensor

        :return: The length of the tensor
        '''
        return self.shape[0] if len(self.shape) != 0 else 0
    
    def __getitem__(self, indices: Union[int, Tuple[int]]) -> 'Tensor':
        '''
        Overloaded method to provide a more intuitive way of accessing _Elements in the tensor.

        :param indices: Indices to access the _Element in the tensor

        :return: The tesnor at the specified indices
        '''
        if isinstance(indices, tuple):
            return Tensor.from_elements(self._get_value(*indices))
        return Tensor.from_elements(self.data[indices])
    
    def __setitem__(self, indices: Union[int, Tuple[int]], value: dtype) -> None:
        '''
        Overloaded method to assign a value to the specified indices in the tensor.

        :param indices: Indices to access the _Element in the tensor
        :param value: The value to be assigned
        '''
        if isinstance(indices, tuple):
            self._assign_value(value, *indices)
        else:
            self._assign_value(value, indices)

    def __iter__(self) -> Iterator[dtype]:
        '''
        Iterate over all _Elements of the tensor

        :return: An iterator yielding _Elements of type dtype
        '''
        yield from self._iterate_data()

    def __add__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        The method that allows addition between two tensors.

        Args:
            sec_obj (Tensor): The tensor that is going to added to self.

        Returns:
            Tensor: The results of the addition.
        '''
        assert (self.shape == sec_obj.shape) or ((self.shape[-1],) == sec_obj.shape), 'Dimensions don\'t match for addition.'

        # Adding two tensors that have the same dimensions
        if self.shape == sec_obj.shape:
            def _add(value_1: NestedTensor, value_2: NestedTensor) -> NestedTensor:
                if isinstance(value_1, _Element) and isinstance(value_2, _Element):
                    return value_1 + value_2
                return [_add(v_1, v_2) for v_1, v_2 in zip(value_1, value_2)]

        # Adding two tensors only on the last dimension
        if (self.shape[-1],) == sec_obj.shape:
            j = 0
            def _add(value_1: NestedTensor, value_2: NestedTensor) -> NestedTensor:
                nonlocal j
                if isinstance(value_1, _Element):
                    res_el = value_1 + value_2[j]
                    j = (j + 1) % len(value_2)
                    return res_el
                return [_add(v, value_2) for v in value_1]

        res_elements = _add(self.data, sec_obj.data)
        return Tensor.from_elements(res_elements)
    
    def __neg__(self) -> 'Tensor':
        '''
        The method that allows negation of a tensor.

        Returns:
            Tensor: The result of the negation of self
        '''
        def _neg(value: NestedTensor) -> NestedTensor:
            if isinstance(value, _Element):
                return -value
            return [_neg(v) for v in value]
        
        res_elements = _neg(self.data)
        return Tensor.from_elements(res_elements)
    
    def __sub__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        The method that allows subtraction between two tensors.

        Args:
            sec_obj (Tensor): The tensor that is going to subtracted from self.

        Returns:
            Tensor: The results of the subtraction.
        '''
        def _sub(value_1: NestedTensor, value_2: NestedTensor) -> NestedTensor:
            if isinstance(value_1, _Element) and isinstance(value_2, _Element):
                return value_1 - value_2
            return [_sub(v_1, v_2) for v_1, v_2 in zip(value_1, value_2)]
        
        res_elements = _sub(self.data, sec_obj.data)
        return Tensor.from_elements(res_elements)
    
    def __mul__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        The method that allows multiplication between two tensors.

        Args:
            sec_obj (Tensor): The tensor that is going to be multiplied from self.

        Returns:
            Tensor: The results of the multiplication.
        '''
        def _mul(value_1: NestedTensor, value_2: NestedTensor) -> NestedTensor:
            if isinstance(value_1, _Element) and isinstance(value_2, _Element):
                return value_1 * value_2
            return [_mul(v_1, v_2) for v_1, v_2 in zip(value_1, value_2)]
        
        res_elements = _mul(self.data, sec_obj.data)
        return Tensor.from_elements(res_elements)
    
    def __pow__(self, exp: Union[int, float]) -> 'Tensor':
        '''
        The method that allows powering a tensor.
        
        Args:
            exp (int, float): The exponent of the power operation

        Returns:
            Tensor: The result of the power operation.
        '''
        def _pow(value: NestedTensor) -> NestedTensor:
            if isinstance(value, _Element):
                return value ** exp
            return [_pow(v) for v in value]
        
        res_elements = _pow(self.data)
        return Tensor.from_elements(res_elements)
    
    def __truediv__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        The method that allows devision between two tensors.

        Args:
            sec_obj (Tensor): The tensor that is going to be devided from self.

        Returns:
            Tensor: The results of the devision.
        '''
        return self * (sec_obj ** -1)
    
    def dot(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        This method allows dot product between two tensor.

        Args:
            sec_obj (Tensor): The object that is going to be multiplied to self.

        Returns:
            Tensor: The result of the dot product operation.
        '''
        # Dimensions conditions
        assert len(self.shape) >= 2, 'Must have dimensions.'
        assert len(sec_obj.shape) == 2, 'Other matrix should only be a 2d matrix.'
        assert self.shape[-1] == sec_obj.shape[-2], 'Last dimensions must match.'

        sec_obj_T = sec_obj.T()

        def _dot_op(A: NestedTensor, B_T: NestedTensor) -> NestedTensor:
            # Dimensions of the result matrix
            rows_A = len(A)
            cols_A = len(A[0])
            rows_B_T = len(B_T)  # which is equal to the original cols_B
            # Initialize the result matrix with zeros
            result = [[0] * rows_B_T for _ in range(rows_A)]
            # Perform the dot product
            for i in range(rows_A):
                for j in range(rows_B_T):
                    res_el = _Element(0)
                    for k in range(cols_A):
                        res_el += A[i][k] * B_T[j][k]
                    result[i][j] = res_el
            return result

        def _dot_funct(value: NestedTensor) -> NestedTensor:
            if isinstance(value[0][0], _Element):
                return _dot_op(value, sec_obj_T.data)
            return [_dot_funct(v) for v in value]

        result__Elements = _dot_funct(self.data)
        return Tensor.from_elements(result__Elements)

    def exp(self) -> 'Tensor':
        '''
        The method that allows tensors to use the e^X function.
        
        Returns:
            Tensor: The result of the exp function.
        '''
        def _exp(value: NestedTensor) -> NestedTensor:
            if isinstance(value, _Element):
                return value.exp()
            return [_exp(v) for v in value]
        
        res_elements = _exp(self.data)
        return Tensor.from_elements(res_elements)

    def tanh(self) -> 'Tensor':
        '''
        The method that allows tensors to use the tanh function.
        
        Returns:
            Tensor: The result of the tanh function.
        '''
        def _tanh(value: NestedTensor) -> NestedTensor:
            if isinstance(value, _Element):
                return value.tanh()
            return [_tanh(v) for v in value]
        
        res_elements = _tanh(self.data)
        return Tensor.from_elements(res_elements)
    
    def relu(self) -> 'Tensor':
        '''
        The method that allows tensors to use the relu function.
        
        Returns:
            Tensor: The result of the relu function.
        '''
        def _relu(value: NestedTensor) -> NestedTensor:
            if isinstance(value, _Element):
                return value.relu()
            return [_relu(v) for v in value]
        
        res_elements = _relu(self.data)
        return Tensor.from_elements(res_elements)

    def size(self, index: int) -> int:
        '''
        Return the dimensions of the tensor

        :param index: Return a specific dimension

        :return: The dimensions of the tensor
        '''
        return self.shape[index]

    def tolist(self) -> NestedList:
        '''
        Convert the tensor to a Python list

        :return: The list version of the tensor
        '''
        def _to_list(value: NestedTensor) -> NestedList:
            if isinstance(value, _Element):
                return value.data
            return [_to_list(v) for v in value]

        return _to_list(self.data)

    def item(self) -> dtype:
        '''
        Convert a scalar tensor to intager or float

        :return: The resulting intager or float if tensor is a scalar, otherwise error
        '''
        if len(self.shape) == 0:
            return self.data.data
        raise ValueError(f'The given tensor is not a scalar, it has dimensions of {self.dimensions}')
    
    def flatten(self) -> 'Tensor':
        '''
        The method that converts the tensor into 1d.

        Returns:
            Tensor: The flattened tensor.
        '''
        res_elements = []
        def _flatten(value: NestedTensor) -> None:
            if isinstance(value, _Element):
                res_elements.append(value)
            else:
                for v in value:
                    _flatten(v)

        _flatten(self.data)
        return Tensor.from_elements(res_elements)
    
    def T(self) -> 'Tensor':
        '''
        Transpose a 2d or a 3d tensor on the last dimensions

        Returns:
            Tensor: The transposed tensor
        '''
        assert len(self.shape) == 3 or len(self.shape) == 2, 'Can only transpose tensors of 2 or 3 dimensions.'

        if len(self.shape) == 3:
            transposed__Elements = []
            for tensor_el in self.data:
                transposed__Elements.append(list(map(list, zip(*tensor_el))))
        else:
            transposed__Elements = list(map(list, zip(*self.data)))

        return Tensor.from_elements(transposed__Elements)

    def zero_grad(self) -> None:
        '''
        Reset the gradient of the tensor
        '''
        def _zero_grad(value: NestedList):
            if isinstance(value, _Element):
                value.grad = 0.0
            else:
                for v in value:
                    _zero_grad(v)
        _zero_grad(self.data)

    def backward(self) -> None:
        '''
        Implement backpropagation for the tensor class
        '''
        def _backward(value: NestedTensor) -> None:
            if isinstance(value, _Element):
                value.backward()
            else:
                for v in value:
                    _backward(v)
        
        _backward(self.data)
