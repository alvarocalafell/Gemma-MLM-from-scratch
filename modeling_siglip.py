from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

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
        """
        Configuration class for SiglipVision model.

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
        embeddings = embeddings + self.position_embeddings(self.position_ids) # Contrary to the Vanilla transformer where we used sinusoidal embeddings which were precalculated,
                                                                              # here the positional embeddings are learned
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipVisionTramsformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.enconder = SiglipEncoder(config) 
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
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

    Attributes:
        config (SiglipVisionConfig): Configuration object containing model parameters.
        vision_model (SiglipVisionTransformer): The underlying vision transformer model.

    Methods:
        forward(pixel_values): Processes input images and returns their embeddings.

    Note:
        The forward method expects pixel values as input and returns a tuple containing
        the output of the vision transformer, typically image embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vison_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, channels, height, width] -> [Batch_size, num_patches, embed_dim]
        return self.vison_model(pixel_values=pixel_values)
    
        
