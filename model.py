import torch 
import torch.nn as nn
import torch.nn.Functional as F 
import math 
from dataclasses import dataclass
from typing import Optional

# this can be thought of as a struct of all are models meta data 
@dataclass
class ModelArgs:
    dim: int = 4096 # embedding dimension 
    n_layers: int = 32 # stacked layers 
    n_heads: int = 32 # Q heads
    n_kv_heads: Optional[int] = None # GQA heads vary for KV
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256 # ffn layer multiple 
    ffn_dim_multiplier: Optional[float] = None # more params in ffn as heads are lower in KV
    norm_eps: float = 1e-5 #for num stability Needed for KV cache
    max_batch_size: int = 32 
    max_seq_len: int = 2048
    device: str = None

"""
In rotary positional embeddings (RoPE), the positional information is injected by rotating the query and key vectors in 2-dimensional subspaces. 
The hidden size of each attention head is divided into consecutive pairs of dimensions,
and is rotated by an angle that depends on the tokens absolute position and a frequency assigned to that pair. 
This way, instead of adding a position vector, RoPE applies a position-dependent phase rotation. 
When queries and keys are later compared in the dot product, the relative rotation between them naturally encodes the distance between positions,
which allows self-attention to capture relative positional relationships without storing them explicitly.
"""
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2) calcualte the thetas 
    theta_numerator = torch.arange(0, head_dim, 2).float() # we need one theta for 2 dimesnions as its polar grouped
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Shape: (Seq_Len)calculate the positions (the "m" parameter)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1 , "vocab_size not set"
        self.args = args
        #pre computing to be injected in forward call 
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
        #embedding layer
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        #stacked transfomers blocks
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        #after final block rms --> linear
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        #--> softmax --> sampling
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "inference mode not set"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        # RoPE is appplied per head the values remain the same but injected per head dim
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output