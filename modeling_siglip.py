from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """
    Configuration class for SiglipVision model.

    This class holds the configuration parameters for the SiglipVision model, including
    architectural details such as hidden sizes, number of layers, and attention heads.

    Args:
        hidden_size (int): Size of the embedding vector of the vision transformer. Default is 768.
        intermediate_size (int): Size of the linear layer used in the feed-forward network. Default is 3072.
        num_hidden_layers (int): Number of layers in the vision transformer. Default is 12.
        num_attention_heads (int): Number of heads in the multi-head attention. Default is 12.
        num_channels (int): Number of channels in the input image (e.g., 3 for RGB). Default is 3.
        image_size (int): Size of the input image (assumes square images). Default is 224.
        patch_size (int): Size of each image patch. Default is 16.
        layer_norm_eps (float): Epsilon value for layer normalization. Default is 1e-6.
        attention_dropout (float): Dropout rate for attention layers. Default is 0.0.
        num_image_tokens (int, optional): Number of image token embeddings output by the visual transformer.

    Note:
        The SiglipVision model can be configured for different sizes, with each configuration
        affecting the model's capacity and performance characteristics.
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size 
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """
    Embeddings module for the SiglipVision model.

    This module is responsible for creating patch embeddings from input images and
    adding positional embeddings to these patches.

    Args:
        config (SiglipVisionConfig): Configuration object for the SiglipVision model.

    Attributes:
        config (SiglipVisionConfig): The configuration object.
        embed_dim (int): Dimension of the embedding vectors.
        image_size (int): Size of the input images.
        patch_size (int): Size of each image patch.
        patch_embedding (nn.Conv2d): Convolutional layer for creating patch embeddings.
        num_patches (int): Total number of patches for an image.
        num_positions (int): Number of position embeddings.
        position_embeddings (nn.Embedding): Learnable position embeddings.
        position_ids (torch.Tensor): Tensor of position IDs.
    """

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            strid=self.patch_size,
            padding="valid", # No padding is added
        ) 
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass of the SiglipVisionEmbeddings module.

        Args:
            pixel_values (torch.FloatTensor): Input tensor of pixel values.

        Returns:
            torch.Tensor: Tensor of patch embeddings with added positional information.
        """
        _, _, height, width = pixel_values.shape# [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1,2) 
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        # Contrary to the Vanilla transformer where we used sinusoidal embeddings, 
        # which were precalculated,here the positional embeddings are learned
        embeddings = embeddings + self.position_embeddings(self.position_ids) 
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class SiglipEncoderLayer(nn.Module):
    """
    Encoder layer for the SiglipVision model.

    This class represents a single layer in the encoder stack of the SiglipVision model.
    It includes self-attention and feed-forward neural network components.

    Args:
        config (SiglipVisionConfig): Configuration object for the SiglipVision model.

    Attributes:
        embed_dim (int): Dimension of the embedding vectors.
        self_attn (SiglipAttention): Self-attention module.
        layer_norm1 (nn.LayerNorm): Layer normalization for the first sub-layer.
        mlp (SiglipMLP): Multi-layer perceptron module.
        layer_norm2 (nn.LayerNorm): Layer normalization for the second sub-layer.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the SiglipEncoderLayer.

        Args:
            hidden_states (torch.Tensor): Input tensor of hidden states.

        Returns:
            torch.Tensor: Output tensor after passing through the encoder layer.
        """
        #residual: [Batch_size, num_patches, embed_dim]
        residual = hidden_states
        #[Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]        
        hidden_states = self.layer_norm1(hidden_states)
        #[Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]        
        hidden_states = self.self_attn(hidden_states=hidden_states)
        #[Batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states        
        #residual: [Batch_size, num_patches, embed_dim]        
        residual = hidden_states
        #[Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]        
        hidden_states = self.layer_norm2(hidden_states)
        #[Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        #[Batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        
        return hidden_states

class SiglipMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for the SiglipVision model.

    This class implements a simple feed-forward neural network used in the transformer architecture.

    Args:
        config (SiglipVisionConfig): Configuration object for the SiglipVision model.

    Attributes:
        config (SiglipVisionConfig): The configuration object.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SiglipMLP.

        Args:
            hidden_states (torch.Tensor): Input tensor of hidden states.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        #hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states

class SiglipEncoder(nn.Module):
    """
    Encoder module for the SiglipVision model.

    This class represents the full encoder stack, consisting of multiple SiglipEncoderLayers.

    Args:
        config (SiglipVisionConfig): Configuration object for the SiglipVision model.

    Attributes:
        config (SiglipVisionConfig): The configuration object.
        layers (nn.ModuleList): List of SiglipEncoderLayer modules.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the SiglipEncoder.

        Args:
            inputs_embeds (torch.Tensor): Input tensor of embedded patches.

        Returns:
            torch.Tensor: Output tensor after passing through all encoder layers.
        """
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipAttention(nn.Module):
    """
    Attention module for the SiglipVision model.

    This class implements the multi-head self-attention mechanism used in the transformer architecture.

    Args:
        config (SiglipVisionConfig): Configuration object for the SiglipVision model.

    Attributes:
        config (SiglipVisionConfig): The configuration object.
        embed_dim (int): Dimension of the embedding vectors.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        dropout (float): Dropout rate.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        q_proj (nn.Linear): Linear projection for queries.
        out_proj (nn.Linear): Output projection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the SiglipAttention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of hidden states.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple containing the attention output and attention weights.
        """
        #hidden_state: [Batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        #[Batch_Size, Num_Heads, Num_Patches, Head_Dim] * [Batch_Size, Num_Heads, Head_Dim, Num_patches]
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous() #Contiguous allows to save the attn_output in memmory contgiously so that the reshape is done just by changing the stride and have no computation overhead
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
        
        
class SiglipVisionTransformer(nn.Module):
    """
    SiglipVisionTransformer implements the vision transformer architecture for the SIGLIP model.

    This class processes input images through a series of transformer layers to produce
    image embeddings. It includes patch embedding, encoder layers, and post-processing.

    Args:
        config (SiglipVisionConfig): Configuration object containing model parameters.

    Attributes:
        config (SiglipVisionConfig): The model configuration.
        embeddings (SiglipVisionEmbeddings): Module for creating patch embeddings.
        enconder (SiglipEncoder): The transformer encoder layers.
        post_layernorm (nn.LayerNorm): Final layer normalization.

    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.enconder = SiglipEncoder(config) 
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process input images through the vision transformer.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [Batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [Batch_size, num_patches, embed_dim].

        """
        #pixel_values: [Batch_size, channels, height, width] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.enconder(inputs_embeds=hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
    

class SiglipVisionModel(nn.Module):
    """
    SiglipVisionModel is a wrapper class for the SiglipVisionTransformer.

    This class encapsulates the vision component of the SIGLIP (Sigmoid Loss for Language Image Pre-training) model.
    It processes input images through a vision transformer architecture to produce image embeddings.

    Args:
        config (SiglipVisionConfig): Configuration object containing model parameters.

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model parameters.
        vision_model (SiglipVisionTransformer): The underlying vision transformer model.

    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vison_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        Process input images and return their embeddings.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [Batch_size, channels, height, width].

        Returns:
            Tuple[torch.Tensor]: A tuple containing the output of the vision transformer,
                                 typically image embeddings of shape [Batch_size, num_patches, embed_dim].

        """
        # [Batch_size, channels, height, width] -> [Batch_size, num_patches, embed_dim]
        return self.vison_model(pixel_values=pixel_values)
