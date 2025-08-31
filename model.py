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