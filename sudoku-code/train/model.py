# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model Architecture."""

import functools
from typing import Any, Callable

from flax import linen as nn
from flax import struct
from jax import numpy as jnp

@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int = 1
    dtype: Any = jnp.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    seq_len: int = 2048  # Maximum sequence length
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False

    
class TransformerBlock(nn.Module):
    config: Any = None

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.emb_dim = self.config.emb_dim
        self.num_layers = self.config.num_layers

    @nn.compact
    def __call__(self, inputs, causal_mask_inputs, training=True):
        """
        Transformer Block call function.

        Args:
            inputs: Input tensor.
            causal_mask_inputs: Causal mask for the inputs.
            training: Whether the model is in training mode.

        Returns:
            Transformed tensor after self-attention and MLP layers.
        """
        
        x = inputs + nn.SelfAttention(
            num_heads=self.config.num_heads, dtype=self.config.dtype,
            qkv_features=self.config.qkv_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6), 
            use_bias=False, broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout_rate, normalize_qk=True,
            deterministic=self.config.deterministic)(inputs, causal_mask_inputs)

        def mlp(x):
            """
            Multi-Layer Perceptron function.

            Args:
                x: Input tensor.

            Returns:
                Transformed tensor after applying MLP layers.
            """
            dense_with_init = functools.partial(
                nn.Dense,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6)
                )
            x = dense_with_init(features=self.config.mlp_dim)(x)
            x = nn.gelu(x)
            x = dense_with_init(features=self.config.emb_dim)(x)
            x = nn.Dropout(rate=self.config.dropout_rate,
                        deterministic=self.config.deterministic)(x)
            return x

        x = x + mlp(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: Any = None

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.emb_dim = self.config.emb_dim
        self.num_layers = self.config.num_layers

    @nn.compact
    def __call__(self, inputs, training=True):
        """
        Transformer LM Head call function.

        Args:
            inputs: Input tensor.
            training: Whether the model is in training mode.

        Returns:
            Transformed tensor after applying the Transformer layers and LM head.
        """
        batch_size, seq_size = inputs.shape

        causal_mask_x = nn.make_causal_mask(inputs, dtype=self.config.dtype)

        # Embed the input tensor using a learnable embedding matrix.
        embed_with_init = functools.partial(
            nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))
        token_embeddings = embed_with_init(
            num_embeddings=self.config.vocab_size,
            features=self.config.emb_dim,
        )(inputs)

        # Check the shape of the embedded tensor.
        assert token_embeddings.shape == (batch_size, seq_size,
                                      self.config.emb_dim)

        # Initialize the positional embedding variable.
        pos_embedding_variable = self.variable(
            "params",
            "position_embeddings",
            jnp.zeros,
            (self.config.seq_len, self.config.emb_dim),
        )

        # Slice the positional embedding array to the correct sequence length.
        pos_embeddings = pos_embedding_variable.value[:seq_size, :]

        # Check the shape of the positional embedding array.
        output_tuple = (pos_embeddings.shape, token_embeddings.shape[1:])
        assert pos_embeddings.shape == token_embeddings.shape[1:], output_tuple

        # Add the positional embeddings to the token embeddings.
        x = token_embeddings + pos_embeddings[None, :, :]

        # Apply dropout to the input.
        x = nn.Dropout(rate=self.config.dropout_rate,
                    deterministic=self.config.deterministic)(x)

        # Apply the Transformer layers.
        for i in range(self.num_layers):
            x = TransformerBlock(config=self.config)(
                    x, causal_mask_x, training=training)
      
            self.sow('intermediates', 'feature_' + str(i), x)

        # Apply the final layer normalization.
        x = nn.LayerNorm()(x)

        # Apply the LM head.
        logits = nn.Dense(features=self.config.vocab_size,
                        kernel_init=nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.normal(stddev=1e-6),
                        use_bias=False)(x)

        # Check the shape of the output tensor.
        assert logits.shape == (batch_size, seq_size, self.config.vocab_size)
        return logits
