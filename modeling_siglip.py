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
        PaliGamma Model comes in different sizes so each model has it's own configuration,
        
        hidden_size: Size of the embedding vector of the vision transformer
        intermediate_size: Size of the linear layer used in the feed forward network
        num_hidden_layers: Number of layers in the vision transformer
        num_attention_heads: Number of heads in the Multi-head attention
        num_channels: Number of channels each image has, in this case RGB
        image_size: Paligamma comes in 3 sizes, in this case by default image size is 224x224 images
        patch_size: Size of each patch, each image will be divided into patches and each patch will be 16x16
        layer_norm_eps: Parameter for the layer normalization
        attention_dropout: Parameter for dropout, we will not be using it in this case
        num_image_tokens: Number of embeddings this visual transformer output, number of embeddings we will have for each image
        
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
