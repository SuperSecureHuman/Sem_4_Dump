import numpy as np
from admm_parallel import ADMM as origadmm
from time import time, sleep


num_iterations = 10
N = 1000
D = 100


A = np.random.randn(N, D)
b = np.random.randn(N, 1)
# Y = np.random.rand(N, 1)

admmparallel = origadmm(A, b, parallel = True)

# Get times for each of the admm methods. Dont have to print anything else

# Sleep of 10 sec between each method for machine cooldown

# Parallel
start = time()
for i in range(0, 3):
    admmparallel.step()
end = time()

print(admmparallel.A.shape)
print(admmparallel.b.shape)
print(admmparallel.A.shape)
print(admmparallel.X.shape)
print(admmparallel.A.dot(admmparallel.X).shape)

