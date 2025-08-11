import torch
import torch.nn as nn

class PropertyEmbedder(nn.Module): 
    def __init__(self, input_dim: int = 1, 
                 embedding_dim: int = 128,
                 start: float = 0.0246,
                 stop: float = 0.6221,
                 n_gaussians: int = 5,  
                 use_activation: bool = True):
        """
        start (float): is min value of the property, default is for gap
        stop (float): is max value of the property, default is for gap 
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # First layer
        layers = [nn.Linear(n_gaussians, embedding_dim)]
        if use_activation:
            layers.append(nn.SiLU())
        layers.append(nn.Linear(embedding_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

        # Gaussian expansion layer
        self.gaussian_expansion = GaussianExpansion(start=start, stop=stop, n_gaussians=n_gaussians, trainable=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Property values tensor of shape (batch_size, 1)
        Returns:
            Embedded tensor of shape (batch_size, embedding_dim)
        """
        expanded_x = self.gaussian_expansion(x)
        return self.mlp(expanded_x)

class GaussianExpansion(nn.Module): 
    # GaussianExpansion class code reference: https://github.com/atomistic-machine-learning/cG-SchNet/blob/main/nn_classes.py#L616
    r"""Expansion layer using a set of Gaussian functions.

    Args:
        start (float): center of first Gaussian function, :math:`\mu_0`.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}`.
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`
            (default: 50).
        trainable (bool, optional): if True, widths and offset of Gaussian functions
            are adjusted during training process (default: False).
        widths (float, optional): width value of Gaussian functions (provide None to
            set the width to the distance between two centers :math:`\mu`, default:
            None).

    """

    def __init__(self, start, stop, n_gaussians=50, trainable=False,
                 width=None):
        super(GaussianExpansion, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) *
                                       torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, property):
        """Compute expanded gaussian property values.

        Args:
            property (torch.Tensor): property values of (N_b x 1) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_g) shape.

        """
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(self.widths, 2)[None, :]
        # Use advanced indexing to compute the individual components
        diff = property - self.offsets[None, :]
        # compute expanded property values
        return torch.exp(coeff * torch.pow(diff, 2))