import math
from typing import Union, Tuple


# Define the aliases that tensor uses
dtype = Union[int, float]


class _Element:
    '''
    My replication of a the simplest data structure (scalar) that I have implemented
    backpropagation.
    '''
    def __init__(self,
            data: dtype,
            _children: Tuple['_Element'] = (),
        ) -> None:
        '''
        Initialize the class which would be a normal intager or float with the
        capability of backpropagetion

        :param data: The scalar that we will compute its gradients
        :param _children: The Tensor objects that produces this exact scalar
        '''
        self.data = data
        # We are storing in a set for efficiency, O(logn) retrieving and adding complexity
        self._prev = set(_children)
        # Initializing to empy function to handle the leaf nodes
        self._backward = lambda: None
        # Initializing to 0 because we don't want to affect the output
        self.grad = 0.0

    def __repr__(self):
        '''
        Implementation of the __repr__ method to provide the debug capability

        :return: The debug string representation of the string
        '''
        return f"_Element(data={self.data}, grad={self.grad})"
    
    def __add__(self, sec_obj: '_Element') -> '_Element':
        '''
        Implement the __add__ method to add two scalars

        :param sec_obj: The object that will be added to self

        :return: The result of the addition
        '''
        res_obj = _Element(self.data + sec_obj.data, _children=(self, sec_obj))

        # Updating `_children` total gradients by passing through derivative
        def _backward():
            self.grad += 1.0 * res_obj.grad
            sec_obj.grad += 1.0 * res_obj.grad

        res_obj._backward = _backward

        return res_obj
    
    def __sub__(self, sec_obj: '_Element') -> '_Element':
        '''
        Implement the __sub__ method to subtrack two scalars

        :param sec_obj: The object that will be added to self

        :return: The result of the subtraction
        '''
        res_obj = _Element(self.data - sec_obj.data, _children=(self, sec_obj))

        # Updating `_children` total gradients by passing through derivative
        def _backward():
            self.grad += 1.0 * res_obj.grad
            sec_obj.grad -= 1.0 * res_obj.grad

        res_obj._backward = _backward

        return res_obj

    def __neg__(self) -> '_Element':
        '''
        Implement the __neg__ method to negate an _Element

        :return: The result of the negation
        '''
        res_obj = _Element(-self.data, _children=(self,))

        # Updating `_children` total gradients by passing through derivative
        def _backward():
            self.grad -= 1.0 * res_obj.grad

        res_obj._backward = _backward

        return res_obj

    def __mul__(self, sec_obj: '_Element') -> '_Element':
        '''
        Implement the multiplication operator between two scalars

        :param sec_obj: The second operand of the multiplication

        :return: The result of the multiplication
        '''
        res_obj = _Element(self.data * sec_obj.data, _children=(self, sec_obj))

        # Updating `_children` total gradients by applying the rule we found res_obj before
        def _backward():
            self.grad += sec_obj.data * res_obj.grad
            sec_obj.grad += self.data * res_obj.grad

        res_obj._backward = _backward

        return res_obj
    
    def __pow__(self, exp: Union[int, float]) -> '_Element':
        '''
        Implement the self**exp operator for the scalar

        :parap exp: The exponent of the power operation

        :return: The result of the power operation
        '''
        assert isinstance(exp, (int, float)), "Only supporting int/float types"

        res_obj = _Element(self.data**exp, _children=(self,))

        def backward():
            self.grad += exp*(self.data**(exp - 1)) * res_obj.grad

        res_obj._backward = backward

        return res_obj
    
    def __truediv__(self, sec_obj: '_Element') -> '_Element':
        '''
        Implement the division operation between two sclars

        :param sec_obj: The second operand of the division

        :return: The result of the devision
        '''
        return self * (sec_obj ** -1)
    
    def exp(self) -> '_Element':
        '''
        Implement the e^X operation for the scalar

        :return: The result of the e^X function
        '''
        res_obj = _Element(math.exp(self.data), _children=(self,))

        def _backward():
            self.grad += res_obj.data * res_obj.grad
        
        res_obj._backward = _backward

        return res_obj

    def tanh(self) -> '_Element':
        '''
        Implement the tanh function for scalars

        :return: The result of the tanh function
        '''
        res_obj = _Element((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), _children=(self,))

        def _backward():
            self.grad += (1 - res_obj.data**2) * res_obj.grad

        res_obj._backward = _backward

        return res_obj
    
    def relu(self) -> '_Element':
        '''
        Implement the relu function for scalars

        :return: The result of the relu function
        '''
        res_obj = _Element(self.data * (self.data > 0), _children=(self,))

        def _backward():
            self.grad += (self.data > 0) * res_obj.grad
        
        res_obj._backward = _backward

        return res_obj
    
    def backward(self) -> None:
        '''
        Implement backpropagation for the _Element class
        '''
        topo = []
        visited = set()

        def build_topo(v: '_Element') -> None:
            '''
            Apply topological sorting to the children of a scalar

            :param v: The object that we want to apply topological sorting
                the its children
            '''
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set the initial grad 
        self.grad = 1.0

        # Apply topological sorted to compute the _backward method of every children
        for n in reversed(topo):
            n._backward()
