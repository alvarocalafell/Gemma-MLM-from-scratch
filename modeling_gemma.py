import torch
from torch import nn
from typing import Optional, Tuple, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    """
    A class to manage key-value caches for transformer layers.

    This cache stores and updates key and value states for each layer,
    allowing for efficient processing of sequential data.

    Attributes:
        key_cache (List[torch.Tensor]): List of cached key states for each layer.
        value_cache (List[torch.Tensor]): List of cached value states for each layer.
    """

    def __init__(self) -> None:
        """
        Initialize an empty KVCache.
        """
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        """
        Get the number of items (sequence length) in the cache.

        Returns:
            int: The number of items in the cache, or 0 if empty.
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states for a specific layer.

        Args:
            key_states (torch.Tensor): New key states to add to the cache.
            value_states (torch.Tensor): New value states to add to the cache.
            layer_idx (int): Index of the layer to update.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value states for the layer.
        """
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    """
    Configuration class for the Gemma model.

    This class holds various hyperparameters and settings for the Gemma model architecture.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Dimension of the hidden layers.
        intermediate_size (int): Dimension of the intermediate (feed-forward) layers.
        num_hidden_layers (int): Number of hidden layers in the model.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of key/value heads (can be different from num_attention_heads).
        head_dim (int): Dimension of each attention head.
        max_position_embeddings (int): Maximum sequence length the model can handle.
        rms_norm_eps (float): Epsilon value for layer normalization.
        rope_theta (float): Base value for rotary position embeddings.
        attention_bias (bool): Whether to use bias in attention calculations.
        attention_dropout (float): Dropout probability for attention weights.
        pad_token_id (int): ID of the padding token.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        """
        Initialize the GemmaConfig with model hyperparameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimension of the hidden layers.
            intermediate_size (int): Dimension of the intermediate layers.
            num_hidden_layers (int): Number of hidden layers in the transformer.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int): Number of key/value heads.
            head_dim (int, optional): Dimension of each attention head. Defaults to 256.
            max_position_embeddings (int, optional): Maximum sequence length. Defaults to 8192.
            rms_norm_eps (float, optional): Epsilon for layer normalization. Defaults to 1e-6.
            rope_theta (float, optional): Base value for rotary position embeddings. Defaults to 10000.0.
            attention_bias (bool, optional): Whether to use bias in attention. Defaults to False.
            attention_dropout (float, optional): Dropout rate for attention. Defaults to 0.0.
            pad_token_id (int, optional): ID of the padding token. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        

class PaliGemmaConfig():
    """
    Configuration class for the PaliGemma model.

    This class combines configurations for both vision and text components of the PaliGemma model.

    Attributes:
        vision_config (SiglipVisionConfig): Configuration for the vision component.
        text_config (GemmaConfig): Configuration for the text component.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Token index representing image placeholders.
        vocab_size (int): Size of the vocabulary.
        projection_dim (int): Dimension of the projection layer.
        hidden_size (int): Size of the hidden layers.
        pad_token_id (int): ID of the padding token.
        is_encoder_decoder (bool): Whether the model uses an encoder-decoder architecture.
    """

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        """
        Initialize the PaliGemmaConfig with vision and text configurations.

        Args:
            vision_config (dict, optional): Configuration for the visual encoder.
            text_config (dict, optional): Configuration for the text decoder (Gemma).
            ignore_index (int, optional): Index to ignore in loss calculation. Defaults to -100.
            image_token_index (int, optional): Token index for image placeholders. Defaults to 256000.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 257152.
            projection_dim (int, optional): Dimension of the projection layer. Defaults to 2048.
            hidden_size (int, optional): Size of the hidden layers. Defaults to 2048.
            pad_token_id (int, optional): ID of the padding token. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaModel(nn.Module):
    """
    The main Gemma model class.

    This class implements the core Gemma model architecture, including token embedding,
    transformer layers, and final normalization.

    Attributes:
        config (GemmaConfig): The model configuration.
        embed_tokens (nn.Embedding): Token embedding layer.
        layers (nn.ModuleList): List of GemmaDecoderLayer instances.
        norm (GemmaRMSNorm): Final layer normalization.
    """

    def __init__(self, config: GemmaConfig):
        """
        Initialize the GemmaModel.

        Args:
            config (GemmaConfig): Configuration object for the model.
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def get_input_embeddings(self):
        """
        Get the input embeddings layer of the model.

        Returns:
            nn.Embedding: The input embeddings layer.
        """
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass of the GemmaModel.

        Args:
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Indices of positions of each input sequence tokens. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Embedded representation of input. Defaults to None.
            kv_cache (Optional[KVCache], optional): Key-value cache for attention layers. Defaults to None.

        Returns:
            torch.FloatTensor: Output tensor of shape [Batch_Size, Seq_Len, Hidden_Size].
        """
        
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states
    

class GemmaDecoderLayer(nn.Module):
    """
    Represents a single decoder layer in the Gemma model.

    This layer applies self-attention followed by a feed-forward neural network,
    with layer normalization and residual connections.

    Args:
        config (GemmaConfig): Configuration object for the Gemma model.
        layer_idx (int): Index of this layer in the stack of decoder layers.

    Attributes:
        hidden_size (int): Dimensionality of the hidden states.
        self_attn (GemmaAttention): Self-attention mechanism.
        mlp (GemmaMLP): Feed-forward neural network.
        input_layernorm (GemmaRMSNorm): Layer normalization applied before self-attention.
        post_attention_layernorm (GemmaRMSNorm): Layer normalization applied after self-attention.
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the GemmaDecoderLayer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Seq_Len, Hidden_Size].
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Indices of positions of each input sequence tokens. Defaults to None.
            kv_cache (Optional[KVCache], optional): Key-value cache for attention layers. Defaults to None.

        Returns:
            torch.FloatTensor: Output tensor of shape [Batch_Size, Seq_Len, Hidden_Size].
        """
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaMLP(nn.Module):
    """
    Implements the Multi-Layer Perceptron (MLP) component of the Gemma model.

    This MLP uses a gated activation function and consists of three linear projections.

    Args:
        config (GemmaConfig): Configuration object for the Gemma model.

    Attributes:
        hidden_size (int): Dimensionality of the input and output.
        intermediate_size (int): Dimensionality of the intermediate layer.
        gate_proj (nn.Linear): Linear projection for the gate.
        up_proj (nn.Linear): Linear projection for the up-sampling.
        down_proj (nn.Linear): Linear projection for the down-sampling.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the GemmaMLP.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Seq_Len, Hidden_Size].

        Returns:
            torch.Tensor: Output tensor of shape [Batch_Size, Seq_Len, Hidden_Size].
        """
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)) # Gate projection is just adding parameters before the gelu activation function

class GemmaRMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.

    This normalization technique is used in the Gemma model instead of traditional Layer Normalization.

    Args:
        dim (int): The number of dimensions to normalize over.
        eps (float, optional): A small constant for numerical stability. Defaults to 1e-6.

    Attributes:
        eps (float): A small constant for numerical stability.
        weight (nn.Parameter): Learnable scale parameter.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Applies the RMS normalization to the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # We add the eps in the denominator to avoid division by 0

    def forward(self, x):
        """
        Forward pass of the GemmaRMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized and scaled tensor.
        """
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)



class GemmaForCausalLM(nn.Module):
    """
    Gemma model for causal language modeling.

    This class wraps the base GemmaModel and adds a language modeling head on top.

    Args:
        config (GemmaConfig): Configuration object for the Gemma model.

    Attributes:
        config (GemmaConfig): Model configuration.
        model (GemmaModel): The base Gemma model.
        vocab_size (int): Size of the vocabulary.
        lm_head (nn.Linear): Linear layer for language modeling predictions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_input_embeddings(self):
        """
        Get the input embeddings layer of the model.

        Returns:
            nn.Embedding: The input embeddings layer.
        """
        return self.model.embed_tokens
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Forward pass of the GemmaForCausalLM model.

        Args:
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Indices of positions of each input sequence tokens. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Embedded representation of input. Defaults to None.
            kv_cache (Optional[KVCache], optional): Key-value cache for attention layers. Defaults to None.

        Returns:
            Tuple: Contains the logits and optionally the updated kv_cache.
        """

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
        

class PaliGemmaMultiModalProjector(nn.Module):
    """
    A Linear layer that converts the size of the image features extracted
    from the Vision Encoder (vision_config.hidden_size) into the same size 
    of the embedding size used by the Language model (vision_config.projection_dim).
    It basically resizes the embeddings so that they can be concatenated with the text tokens.

    Args:
        config (PaliGemmaConfig): Configuration object for the PaliGemma model.

    Attributes:
        linear (nn.Linear): Linear layer for projection.
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        """
        Forward pass of the PaliGemmaMultiModalProjector.

        Args:
            image_features (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Projected tensor of shape [Batch_Size, Num_Patches, Projection_Dim].
        """
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    """
    PaliGemma model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    for tasks that involve both image and text inputs.

    Args:
        config (PaliGemmaConfig): Configuration object for the PaliGemma model.

    Attributes:
        config (PaliGemmaConfig): Model configuration.
        vision_tower (SiglipVisionModel): Vision model for processing image inputs.
        multi_modal_projector (PaliGemmaMultiModalProjector): Projector for aligning vision and language features.
        vocab_size (int): Size of the vocabulary.
        language_model (GemmaForCausalLM): Language model for text generation.
        pad_token_id (int): ID of the padding token.
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        """
        Tie the weights in the language model.
        """
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        """
        Merge input embeddings with image features.

        This method combines the embeddings of text tokens and image tokens,
        handling padding and creating the appropriate attention mask and position IDs.

        Args:
            image_features (torch.Tensor): Processed image features.
            inputs_embeds (torch.Tensor): Text token embeddings.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            kv_cache (Optional[KVCache], optional): Key-value cache for attention layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Combined embeddings
                - Updated attention mask
                - Position IDs
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        
        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
    
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value states for multi-query attention.

    This function repeats the key and value states to match the number of query heads
    in multi-query attention mechanisms.

    Args:
        hidden_states (torch.Tensor): Input tensor of shape [batch, num_key_value_heads, slen, head_dim].
        n_rep (int): Number of times to repeat each key and value state.

    Returns:
        torch.Tensor: Repeated tensor of shape [batch, num_key_value_heads * n_rep, slen, head_dim].
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        """
        Since Gemma is a decoder only model with many layers and each layer
        will have it's own KV Cache, so we need to provide the layer_idx to
        know which one to apply
        This module implements the attention mechanism for the Gemma model, including
        multi-query attention and rotary position embeddings.

        Args:
            config (GemmaConfig): Configuration object for the Gemma model.
            layer_idx (Optional[int]): Index of the current layer in the model stack.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx 

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # Ensure that the hidden size is divisible by the number of heads, since each ehad is watching a part of the embedding.
        assert self.hidden_size % self.num_heads == 0
        
        #In this case we don't have number of features/hidden_size as output, but numb_heads * head_dim
        # Hidden_size = 1024
        # Head_dim = 1024 / 8 = 128
        # Wq: [1024, 8 * 128] = [1024, 1024]
        # Wk: [1024, 4 * 128] = [1024, 512] 
        # Wv: [1024, 4 * 128] = [1024, 512] 
        # So the difference with grouped query attention is that we have less heads for the keys and values which result in smaller projections
        # Each 2 heads share a key value for example
        #Multi query only 1 head of k-v for all queries vs Grouped query multiple heads of k-v per query
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Perform the forward pass of the attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Attention mask tensor.
            position_ids (Optional[torch.LongTensor]): Position IDs for rotary embeddings.
            kv_cache (Optional[KVCache]): Key-value cache for efficient inference.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                - Output tensor after attention mechanism.
                - Attention weights (optional).
                - Updated key-value cache (optional).
        """
        
        bsz, q_len, _ = hidden_states.size() # [Batch_Size, Seq_Len, Hidden_Size]
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        # We need to use this function since me don't have a custon CUDA Kernel to leverage the non-repeating of the heads
        # If we used FlashAttention we could beenfit from the memory saving capabilities
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # In our case the Attention mask will always be full of 0's since we have no padding and as we said
        # the PaliGemma team decided not to mask the prompt text so now mask is needed
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask
        
        # Apply the softmax
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # Multiply by the values. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        """
        HuggingFace's implementation of Rotary Positional Encoding Paper
        """
        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute rotary positional encodings.

        Args:
            x (torch.Tensor): Input tensor.
            position_ids (torch.Tensor): Position IDs.
            seq_len (int, optional): Sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine components of the rotary encoding.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """
    Rotate half of the dimensions of a tensor.

    This function is used in applying rotary positional encodings.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with half of its last dimension rotated.
    """
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine component of rotary encoding.
        sin (torch.Tensor): Sine component of rotary encoding.
        unsqueeze_dim (int): Dimension to unsqueeze for broadcasting.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensors with rotary positional encodings applied.
    """
    cos = cos.unsqueeze(unsqueeze_dim)# Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)# Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed