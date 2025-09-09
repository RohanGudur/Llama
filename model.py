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

def RMSnorm(nn.Module):
    def __init__(self , dim = int , eps = float ):
        super().__init_()
        self.eps = eps
        self.dim = dim

    def forward(x:torch.Tensor):
        weights = nn.Parameter(torch.ones(dim))
        return weights *  (x * torch.rsqrt(x.pow(2).mean(-1,keepdims = True)+ self.eps))

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1,  "Vocab size must be set"
        self.args = args
        
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
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output