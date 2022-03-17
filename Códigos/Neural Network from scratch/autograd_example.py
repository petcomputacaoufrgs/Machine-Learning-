# class Tensor:
#     def __init__(self,
#                  data: Arrayable,
#                  requires_grad: bool = False,
#                  dependencies: List[Dependency] = None) -> None:
#         self.data = arrayable2array(data)
#         self.requires_grad = requires_grad  # Basically means that this tensor will have it's gradient calculated
#         self.dependencies = dependencies or []
#         self.shape = self.data.shape
#         self.grad: Optional[np.ndarray] = None
#
#         if self.requires_grad:
#             self.zero_grad()
#
#     def zero_grad(self) -> None:
#         self.grad = np.zeros_like(self.data, dt   ype=np.float64)
#
#     def __repr__(self) -> str:
#         return f"Tensor({self.data}, requires_grad={self.requires_grad})"
#
#     def backward(self, grad: Optional[np.ndarray] = None) -> None:
#         assert self.requires_grad, "called backward on non-requires-grad tensor"
#
#         if grad is None:
#             if self.shape == ():  # No shape
#                 grad = np.array(1.0)
#             else:
#                 raise RuntimeError("Grad must be specified for non-0 tensor")
#
#         self.grad += grad
#
#         for dependency in self.dependencies:
#             backward_grad = dependency.grad_fn(grad)
#             dependency.tensor.backward(backward_grad)
#
#     def sum(self, axis=None, keepdims=False) -> 'Tensor':
#         return _tensor_sum(self, axis, keepdims)
#
#     def __add__(self, other) -> 'Tensor':
#         return _add(self, tensorable2tensor(other))
#
#     def __radd__(self, other) -> 'Tensor':
#         return _add(tensorable2tensor(other), self)
#
#     def __iadd__(self, other) -> 'Tensor':
#         self.data = self.data + tensorable2tensor(other).data
#         return self
#
#     def __sub__(self, other) -> 'Tensor':
#         return _sub(self, tensorable2tensor(other))
#
#     def __rsub__(self, other) -> 'Tensor':
#         return _sub(tensorable2tensor(other), self)
#
#     def __isub__(self, other) -> 'Tensor':
#         self.data = self.data - tensorable2tensor(other).data
#         return self
#
#     def __mul__(self, other) -> 'Tensor':
#         return _mul(self, tensorable2tensor(other))
#
#     def __rmul__(self, other) -> 'Tensor':
#         return _mul(tensorable2tensor(other), self)
#
#     def __imul__(self, other) -> 'Tensor':
#         self.data = self.data * tensorable2tensor(other).data
#         return self
#
#     def __truediv__(self, other) -> 'Tensor':
#         return _div(self, tensorable2tensor(other))
#
#     def __idiv__(self, other) -> 'Tensor':
#         self.data = self.data / tensorable2tensor(other).data
#         return self
#
#     def __rdiv__(self, other) -> 'Tensor':
#         return tensorable2tensor(other) / self.data
#
#     def __neg__(self) -> 'Tensor':
#         return _neg(self)
#
#     def __pow__(self, other) -> 'Tensor':
#         return _pow(self, tensorable2tensor(other))
#
#     def __matmul__(self, other) -> 'Tensor':
#         return _matmul(self, tensorable2tensor(other))
#
#     def __getitem__(self, item) -> 'Tensor':
#         return _slice(self, item)
#
#
# # Returns the sum of two tensors
# def _add(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data + t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     dependencies: List[Dependency] = []
#
#     # Reasoning for grad_fn:
#     # If f = t1 + t2, df/t1 = 1, df/t2 = 1
#     # It also needs to consider numpy broadcasting; gradients relative to broadcast
#     # elements should be summed up and then applied to the original tensor
#
#     if t1.requires_grad:
#         def grad_fn1(grad: np.ndarray) -> np.ndarray:
#             return handle_broadcasting(grad, t1)
#
#         dependencies.append(Dependency(t1, grad_fn1))
#
#     if t2.requires_grad:
#         def grad_fn2(grad: np.ndarray) -> np.ndarray:
#             return handle_broadcasting(grad, t2)
#
#         dependencies.append(Dependency(t2, grad_fn2))
#
#     return Tensor(data, requires_grad, dependencies)
#
#
#
# # Multiplies two tensors
# def _mul(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data * t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     dependencies: List[Dependency] = []
#
#     # If f = a * b, df/da = b
#     # Similar to sum, except the gradient is multiplied by b instead of 1
#
#     if t1.requires_grad:
#         def grad_fn1(grad: np.ndarray) -> np.ndarray:
#             grad = grad * t2.data
#             return handle_broadcasting(grad, t1)
#
#         dependencies.append(Dependency(t1, grad_fn1))
#
#     if t2.requires_grad:
#         def grad_fn2(grad: np.ndarray) -> np.ndarray:
#             grad = grad * t1.data
#             return handle_broadcasting(grad, t2)
#
#         dependencies.append(Dependency(t2, grad_fn2))
#
#     return Tensor(data, requires_grad, dependencies)

# [3, 4, 5] + [1] -> Numpy -> [3, 4, 5] + [1, 1, 1] = [4, 5, 6] (soma)
# dalgo/dsoma = [x, y, z] = dy
# dy/[a, b, c] + [d]
# dy/a = x * 1
# dy/b = y * 1
# dy/c = z * 1
# dy/d = 1 * (x + y + z)

# Handles numpy broadcasting conflicting with gradients
def handle_broadcasting(grad: np.ndarray, t: Tensor) -> np.ndarray:
    # Handles dimensions being added "to the beginning"
    # [1, 2] -> broadcast -> [[1, 2], [1, 2]] -> handle_broadcasting = [2, 4]
    added_dimensions = grad.ndim - t.data.ndim
    for _ in range(added_dimensions):
        grad = np.sum(grad, axis=0)

    # Handles broadcasting in dimensions that are 1
    # Ex: (5 x 4 x 1) + (5 x 4 x 3) -> (5 x 4 x 3) + (5 x 4 x 3)
    for a, dim in enumerate(t.shape):
        # a = index = axis
        if dim == 1:
            grad = np.sum(grad, axis=a, keepdims=True)
    return grad
