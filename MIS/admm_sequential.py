import numpy as np
from numpy.linalg import inv, norm

# This code implements the ADMM algorithm to solve the Lasso problem. The function step() takes a single step of the algorithm. The function LassoObjective() computes the objective function of the Lasso problem at the current point.

class ADMM:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.D = A.shape[1]
        self.N = A.shape[0]
        self.nu = np.zeros((self.D, 1))
        self.rho = 1
        self.X = np.random.randn(self.D, 1)
        self.Z = np.zeros((self.D, 1))
        self.A = A
        self.b = b
        self.alpha = 0.01

    def step(self):
        # Solve for X_t+1
        ATA = np.zeros((self.D, self.D))
        ATb = np.zeros((self.D, 1))

        for i in range(self.N):
            for j in range(self.D):
                for k in range(self.D):
                    ATA[j, k] += self.A[i, j] * self.A[i, k]
                ATb[j] += self.A[i, j] * self.b[i]

        ATA_rho = ATA + self.rho * np.eye(self.D)
        X_rho_Z_nu = self.rho * self.Z - self.nu

        X_inv = np.linalg.inv(ATA_rho)
        self.X = np.dot(X_inv, ATb + X_rho_Z_nu)

        # Solve for Z_t+1
        for i in range(self.D):
            self.Z[i] = self.X[i] + self.nu[i] / self.rho - (self.alpha / self.rho) * np.sign(self.Z[i])

        # Update nu
        for i in range(self.D):
            self.nu[i] = self.nu[i] + self.rho * (self.X[i] - self.Z[i])



    def LassoObjective(self):
        AX_b = self.A.dot(self.X) - self.b
        objective = 0.5 * norm(AX_b) ** 2 + self.alpha * norm(self.X, 1)
        return objective