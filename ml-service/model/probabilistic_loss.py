import torch
import torch.nn as nn

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    Used for Probabilistic Deep Learning where the model outputs Mean and Variance.
    
    The network outputs 6 values:
    - [0:3]: Mean (mu) for x, y, z
    - [3:6]: Log Variance (log_var) for x, y, z
    
    Why Log Variance? Because variance must be positive. We predict s = log(v), then v = exp(s).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        """
        output: [Batch, 6] -> [mu_x, mu_y, mu_z, logvar_x, logvar_y, logvar_z]
        target: [Batch, 3] -> [true_x, true_y, true_z]
        """
        # Split output into Mean and LogVariance
        mu = output[:, :3]
        log_var = output[:, 3:]
        
        # Calculate Variance (ensure stability)
        var = torch.exp(log_var)
        
        # GNLL Formula: 0.5 * (log(var) + (target - mu)^2 / var)
        # Sum over x, y, z dimensions, then mean over batch
        loss = 0.5 * (log_var + (target - mu)**2 / var)
        
        return loss.mean()
