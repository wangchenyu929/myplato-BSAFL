"""
The federated averaging algorithm for PyTorch.
"""
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """在pytorch中，torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数"""
        """Extract weights from the model."""
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=True)
