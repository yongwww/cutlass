import numpy as np
import random

import cutlass

# This controls whether the C++ GEMM declaration will be printed at each step.
# Set to `False` to omit this information.
print_module = True

m = 128
n = m
k = m

dtype = np.float32
type_A = np.float32
type_B = np.float32
type_C = np.float32
type_D = np.float32

np.random.seed(1234)
random.seed(1234)
scope_min = -10
scope_max = 10
tensor_A = np.random.uniform(low=scope_min, high=scope_max, size=(m, k)).astype(type_A)
tensor_B = np.random.uniform(low=scope_min, high=scope_max, size=(k, n)).astype(type_B)
tensor_C = np.random.uniform(low=scope_min, high=scope_max, size=(m, n)).astype(type_C)

alpha = np.float32(1.0)
beta = np.float32(0.0)

tensor_D = np.zeros(tensor_C.shape).astype(type_D)

plan = cutlass.Gemm(
    element=dtype, layout=cutlass.LayoutType.RowMajor, element_accumulator=np.float32
)
# Use Simt instead of tf32. By default the tf32 will be used for Tensorcore
# plan.opclass = cutlass.OpcodeClass.Simt
plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module)

tensor_D_numpy = (alpha * (tensor_A @ tensor_B)) + (beta * tensor_C)
np.testing.assert_array_equal(tensor_D, tensor_D_numpy)

print(plan.opclass)
