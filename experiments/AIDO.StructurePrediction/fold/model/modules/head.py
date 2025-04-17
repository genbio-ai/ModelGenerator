

import torch
import torch.nn as nn
from torch.nn import Linear


# Adapted From openfold.model.heads
class DistogramHead(nn.Module):
    """Implements Algorithm 1 [Line17] in AF3
    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2)
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        """
        Args:
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            no_bins (int, optional): Number of distogram bins. Defaults to 64.
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(in_features=self.c_z, out_features=self.no_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [*, N, N, C_z]
        """
        Args:
            z (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, no_bins]
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
