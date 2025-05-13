import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import List, Callable, Optional
import numpy as np
from modelgenerator.adapters.base import (
    TokenAdapter,
    SequenceAdapter,
    ConditionalGenerationAdapter,
)


class MLPAdapter(nn.Sequential, TokenAdapter):
    """Multi-layer perceptron (MLP) adapter.

    Args:
        in_features (int): Number of features of the input
        out_features (int): Number of features of the output
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh.
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        layers = [nn.Dropout(dropout)]
        in_dim = in_features
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer())
            if dropout_in_middle:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_features, bias=bias))

        super().__init__(*layers)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)

        Returns:
            torch.Tensor: predictions (n, seq_len, out_features)
        """
        output = super().forward(hidden_states)
        return output

class MLPAdapterWithoutOutConcat(nn.Module, TokenAdapter):
    """Multi-layer perceptron (MLP) adapter without outer concatenate
    
    This class is generally used in PairwiseTokenClassification. The following two implementations are equivalent:
    1. hidden_states -> outer_concat -> MLPAdapter
    2. hidden_states -> MLPAdapterWithoutOutConcat
    MLPAdapterWithoutOutConcat avoids the large memory consumption of outer_concat
    
    Args:
        in_features (int): Number of features of the input
        out_features (int): Number of features of the output
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh.
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        super().__init__()
        self.in_dropout = nn.Dropout(dropout)
        assert len(hidden_sizes) > 0, f"len(hidden_sizes) must be > 0"
        
        min_value = -np.sqrt(1 / (in_features*2))
        max_value =  np.sqrt(1 / (in_features*2))
        init_values = torch.FloatTensor(hidden_sizes[0], in_features*2).uniform_(min_value, max_value)
        self.first_linear = torch.nn.Parameter( init_values, requires_grad=True )
        init_values = torch.FloatTensor(hidden_sizes[0]).uniform_(min_value, max_value)
        self.first_bias   = torch.nn.Parameter( init_values, requires_grad=True )
        
        in_dim = hidden_sizes[0]
        
        layers = [ activation_layer() ]
        if dropout_in_middle:
            layers.append(nn.Dropout(dropout))
        for hidden_dim in hidden_sizes[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer())
            if dropout_in_middle:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, out_features, bias=bias))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)

        Returns:
            torch.Tensor: predictions (n, seq_len, seq_len, out_features)
        """
        n, seq_len, in_features = hidden_states.shape
        
        hidden_states = self.in_dropout(hidden_states) # [B, L, D]
        
        left  = hidden_states @ self.first_linear[:, 0::2].T # [B, L, E]
        right = hidden_states @ self.first_linear[:, 1::2].T # [B, L, E]
        
        hidden_states = left[..., :, None, :] + right[..., None, :, :] + self.first_bias[None, None, None, :] # [B, L, L, D1]
        
        I, J = torch.tril_indices(seq_len, seq_len, -1)
        hidden_states[..., I, J, :] = hidden_states[..., J, I, :]
        
        return self.layers(hidden_states)

class MLPPoolAdapter(nn.Module, SequenceAdapter):
    """MLP adapter for a 2D embedding with pooling for the sequence length dimension

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        pooling (str): Sequence pooling method. Defaults to mean_pooling
        hidden_sizes (List[int], optional): List of the hidden feature dimensions. Defaults to [].
        activation_layer (Callable[..., torch.nn.Module]): Activation function. Defaults to torch.nn.Tanh
        bias (bool): Whether to use bias in the linear layer. Defaults to True
        dropout (float): The probability for the dropout layer. Defaults to 0.0
        dropout_in_middle (bool): Whether to use dropout in the middle layers. Defaults to True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pooling: str = "mean_pooling",
        hidden_sizes: List[int] = [],
        activation_layer: Callable[..., torch.nn.Module] = torch.nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_in_middle: bool = True,
    ):
        super().__init__()
        self.pooling = pooling
        self.mlp = MLPAdapter(
            in_features,
            out_features,
            hidden_sizes,
            activation_layer,
            bias,
            dropout,
            dropout_in_middle,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if self.pooling == "mean_pooling":
            if attention_mask is not None:
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states)
                )
                embeddings = torch.sum(
                    hidden_states * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                embeddings = hidden_states.mean(1)
        elif self.pooling == "cls_pooling":
            embeddings = hidden_states[:, 0]
        else:
            raise NotImplementedError
        output = self.mlp(embeddings)
        return output


class LinearAdapter(MLPAdapter):
    """Simple linear adapter for a 1D embedding

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, hidden_sizes=[])


class LinearCLSAdapter(nn.Module, SequenceAdapter):
    """Simple linear adapter for a 1D embedding

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        output = self.linear(hidden_states[:, 0])
        return output


class LinearMeanPoolAdapter(nn.Module, SequenceAdapter):
    """Mean pooling adapter for hidden_states of shape (n, seq_len, in_features)

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            embeddings = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        else:
            embeddings = torch.mean(hidden_states, dim=1)
        output = self.linear(embeddings)
        return output


class LinearMaxPoolAdapter(nn.Module, SequenceAdapter):
    """Simple max pooling adapter for [batch,seq_len,dim] embeddings

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            hidden_states_final = hidden_states.masked_fill(attention_mask == 0, -torch.inf)
        else:
            hidden_states_final = hidden_states
        embeddings = hidden_states_final.max(1)[0]
        return self.linear(embeddings)


class LinearTransformerAdapter(nn.Module, SequenceAdapter):
    """Transformer adapter

    Note: Support cls_pooling only.

    Args:
        embed_dim (int): Hidden size
        out_features (int): Number of output features
    """

    def __init__(self, embed_dim: int, out_features: int):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.linear = nn.Linear(embed_dim, out_features)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            hidden_states (torch.Tensor): of shape (n, seq_len, in_features)
            attention_mask (torch.Tensor): of shape (n, seq_len)

        Returns:
            torch.Tensor: predictions (n, out_features)
        """
        x = self.transformer(hidden_states)
        embeddings = x[:, 0]  # cls pooling
        output = self.linear(embeddings)
        return output


class ConditionalLMAdapter(nn.Module, ConditionalGenerationAdapter):
    """Conditional sequence adapter

    Args:
        in_features (int): Number of input features
        embed_dim (int): Hidden size
        seq_len (int): Sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        in_features: int,
        out_features: int,
        pretrained_decoder: nn.Module,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 2,
    ):
        super().__init__()
        self.label_encoder = nn.Linear(in_features, embed_dim)
        if pretrained_decoder is not None:
            self.lm_decoder = pretrained_decoder
        else:
            self.lm_decoder = nn.Linear(embed_dim, out_features)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Forward pass

        Args:
            embeddings (torch.Tensor): Input embeddings (n, seq_len, in_features)
            condition (torch.Tensor): Condition tensor (n, condition_features)

        Returns:
            torch.Tensor: Output embeddings (n, seq_len, out_features)
        """
        label_embeddings = self.label_encoder(labels)
        embeddings += label_embeddings.unsqueeze(1).tile(1, embeddings.shape[1], 1)
        transformed_embeddings_residual = self.transformer(embeddings)
        logits = self.lm_decoder(transformed_embeddings_residual + embeddings)
        return logits


class ResNet2DAdapter(nn.Module, SequenceAdapter):
    """Adapter that applies ResNet2DModule to input embeddings.

    Args:
        in_channels (int): Input matrix channels.
        num_res_blocks (int): Number of residual blocks in the ResNet2DModule.
        conv_channels (int): Intermediate convolution channels.
        kernel_size (int): Kernel size for convolutions.
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

    def __init__(
        self,
        in_channels: int,
        num_res_blocks: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_to_conv = nn.Linear(in_channels, conv_channels)
        self.resnet = self.ResNet2DModule(conv_channels, num_res_blocks, kernel_size)
        self.conv_to_output = nn.Conv2d(
            conv_channels, 1, kernel_size=kernel_size, padding="same"
        )

    def symmetrize_matrix(self, matrix: Tensor) -> Tensor:
        """
        Symmetrizes a square matrix by preserving the upper triangular part
        and mirroring it across the diagonal.

        Args:
            matrix (Tensor): Input tensor of shape (B, H, W), where H = W

        Returns:
            Tensor: Symmetrized tensor of shape (B, H, W).
        """
        upper_triangular = torch.triu(matrix, diagonal=1)
        return upper_triangular + upper_triangular.transpose(-1, -2)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (B, H, W, E).

        Returns:
            Tensor: Symmetrized output tensor of shape (B, H, W).
        """
        x = self.in_to_conv(x)
        x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W).

        x = self.resnet(x)
        x = self.conv_to_output(x)
        x = x.squeeze(1)  # Reduce to (B, H, W).

        x = self.symmetrize_matrix(x)

        return x

    class ResNet2DModule(nn.Module):
        """A stack of 2D residual blocks.

        Args:
            channels (int): Number of input/output channels.
            num_res_blocks (int): Number of residual blocks.
            kernel_size (int): Kernel size for each residual block.
            use_bias (bool): Whether to use bias in convolution layers.
        """

        # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

        def __init__(
            self,
            channels: int,
            num_res_blocks: int,
            kernel_size: int = 3,
            use_bias: bool = False,
        ):
            super().__init__()
            self.res2D_blocks = nn.ModuleList(
                [
                    self.Residual2DBlock(channels, kernel_size, use_bias)
                    for _ in range(num_res_blocks)
                ]
            )

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass

            Args:
                x (Tensor): Input tensor of shape (B, C, H, W).
            Returns:
                Tensor: Output tensor of the same shape as input.
            """
            for res_block in self.res2D_blocks:
                x = res_block(x)
            return x

        class Residual2DBlock(nn.Module):
            """A single residual block for a 2D ResNet.

            Args:
                channels (int): Number of input/output channels.
                kernel_size (int): Kernel size for the 2D convolution.
                use_bias (bool): Whether to use bias in convolution layers.
            """

            # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

            def __init__(self, channels, kernel_size=3, use_bias=False):
                super().__init__()
                self.residual_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias=use_bias,
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias=use_bias,
                    ),
                    nn.InstanceNorm2d(channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass

                Args:
                    x (Tensor): Input tensor of shape (B, C, H, W).
                Returns:
                    Tensor: Output tensor of the same shape as input.
                """
                return x + self.residual_block(x)


class ResNet1DAdapter(nn.Module, SequenceAdapter):
    """Adapter module that applies a ResNet1DModule to sequence data.

    Args:
        input_dim (int): Input feature dimension.
        channels (int): Number of channels for ResNet.
        num_blocks (int): Number of residual blocks in ResNet1DModule.
        dropout (float): Dropout rate.
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

    def __init__(
        self, input_channels, num_outputs=1, channels=256, num_blocks=9, dropout=0.2
    ):
        super().__init__()
        self.in_to_conv = nn.Linear(input_channels, channels)
        self.resnet = self.ResNet1DModule(channels, num_blocks)
        self.dropout = nn.Dropout(p=dropout)
        if num_outputs != 1:
            raise NotImplementedError("Please set num_outputs = 1.")
        self.conv_to_output = nn.Linear(channels, num_outputs)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (B, L, C).
            padding_mask (Tensor, optional): Padding mask of shape (B, L). Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B,).
        """
        # Project input to embedding dimension
        x = self.in_to_conv(x)

        # Reshape for convolutional processing
        x = x.permute(0, 2, 1)
        x = self.resnet(x)
        x = x.permute(0, 2, 1)

        # Global pooling across the sequence length
        if padding_mask is not None:
            x[padding_mask, :] = 0.0
            x = x.sum(dim=-2) / (~padding_mask).sum(dim=-1)[:, None]
        else:
            x = x.mean(dim=-2)

        # Apply dropout and project to output
        x = self.dropout(x)
        x = self.conv_to_output(x).squeeze(-1)

        return x

    class ResNet1DModule(nn.Module):
        """A stack of 1D residual blocks.

        Args:
            channels (int): Number of input/output channels.
            num_blocks (int): Number of residual blocks.
            kernel_size (int): Size of the convolution kernel.
            use_bias (bool): Whether to use bias in convolution layers.
        """

        # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

        def __init__(self, channels, num_blocks, kernel_size=3, use_bias=False):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    self.Residual1DBlock(channels, kernel_size, use_bias=use_bias)
                    for _ in range(num_blocks)
                ]
            )

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass

            Args:
                x (Tensor): Input tensor of shape (B, C, L).
            Returns:
                Tensor: Output tensor of the same shape as input.
            """
            for block in self.blocks:
                x = block(x)
            return x

        class Residual1DBlock(nn.Module):
            """A single residual block for a 1D ResNet.

            Args:
                channels (int): Number of input/output channels.
                kernel_size (int): Size of the convolution kernel.
                stride (int): Stride for the convolution.
                use_bias (bool): Whether to use bias in convolution layers.
            """

            # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py

            def __init__(self, channels, kernel_size=3, stride=1, use_bias=False):
                super().__init__()
                self.residual_block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm1d(channels),
                    nn.ELU(inplace=True),
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=use_bias,
                        padding="same",
                    ),
                    nn.InstanceNorm1d(channels),
                    nn.ELU(inplace=True),
                )

            def forward(self, x: Tensor) -> Tensor:
                """
                Forward pass for the residual block.
                Args:
                    x (Tensor): Input tensor of shape (B, C, L).
                Returns:
                    Tensor: Output tensor of the same shape as input.
                """
                return x + self.residual_block(x)
