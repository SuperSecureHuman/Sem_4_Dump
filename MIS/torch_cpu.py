import torch

class ADMM:
    def __init__(self, A, b):
        self.D = A.shape[1]
        self.N = A.shape[0]
        self.nu = torch.zeros((self.D, 1)).to(torch.float)
        self.rho = 1
        self.X = torch.randn(self.D, 1).to(torch.float)
        self.Z = torch.zeros((self.D, 1)).to(torch.float)
        self.A = torch.tensor(A).to(torch.float)
        self.b = torch.tensor(b).to(torch.float)
        self.alpha = 0.01

    def step(self):
        # Solve for X_t+1
        ATA = self.A.t().matmul(self.A)
        ATb = self.A.t().matmul(self.b)
        X_inv = torch.inverse(ATA + self.rho * torch.eye(self.D, dtype=torch.float))
        X_rho_Z_nu = self.rho * self.Z - self.nu
        self.X = X_inv.matmul(ATb + X_rho_Z_nu)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - \
            (self.alpha / self.rho) * self.Z.sign_()

        # Update nu
        self.nu += self.rho * (self.X - self.Z)

    def LassoObjective(self):
        AX_b = self.A.matmul(self.X) - self.b
        objective = 0.5 * torch.norm(AX_b) ** 2 + \
            self.alpha * torch.norm(self.X, 1)
        return objective.item()
