import torch
import torch.nn.functional as F

class ADMM:
    def __init__(self, A, b):
        self.D = A.shape[1]
        self.N = A.shape[0]
        self.nu = torch.zeros((self.D, 1), device='cuda').float()
        self.rho = 1.0
        self.X = torch.randn(self.D, 1, device='cuda').float()
        self.Z = torch.zeros((self.D, 1), device='cuda').float()
        self.A = torch.from_numpy(A).float().to('cuda')
        self.b = torch.from_numpy(b).float().to('cuda')
        self.alpha = 0.01

    def step(self):
        # Solve for X_t+1
        ATA = torch.matmul(self.A.t(), self.A)
        ATb = torch.matmul(self.A.t(), self.b)
        X_inv = torch.inverse(ATA + self.rho * torch.eye(self.D, dtype=torch.float, device='cuda'))
        X_rho_Z_nu = self.rho * self.Z - self.nu
        self.X.add_(torch.matmul(X_inv, ATb + X_rho_Z_nu))

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * torch.sign(self.Z)

        # Update nu
        self.nu.add_(self.rho * (self.X - self.Z))

    def LassoObjective(self):
        AX_b = torch.matmul(self.A, self.X) - self.b
        objective = 0.5 * torch.norm(AX_b) ** 2 + self.alpha * torch.norm(self.X, 1)
        return objective.item()
