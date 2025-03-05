import torch

class VPSDE(bmin=0.1, bmax=20):
    def __init__(self, bmin=0.1, bmax=20):
        self.bmin = bmin
        self.bmax = bmax

    def beta(self, t):
            return self.bmin + t * (self.bmax - self.bmin)

    def alpha(self, t):
        x = self.bmin * t + ((self.bmax - self.bmin) * t**2) / 2
        return torch.exp(-x / 2)

    def p(self, x, t):
        a = self.alpha(t).view(-1, 1)  # for proper broadcasting
        mu = x * a
        std = torch.sqrt(1 - a**2)
        return mu, std

    def f(self, x, t):
        """
        Drift function f(x, t). (Placeholder: adjust as needed.)
        """
        return -0.5 * self.beta(t) * x

    def g(self, t):
        """
        Diffusion coefficient g(t).
        """
        return torch.sqrt(self.beta(t))