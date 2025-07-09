import torch
from torch import nn
from torch.nn import Linear, ReLU, Sigmoid
import torch.nn.functional as F
import numpy as np
import random

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

def squareplus(x, b):
    return torch.mul(0.5, torch.add(x, torch.sqrt(torch.add(torch.square(x), b))))

class SquarePlus(nn.Module):
    def __init__(self, b=1.52382103):
        super().__init__()
        self.b = b

    def forward(self, x):
        return squareplus(x, self.b)
    
class mean(nn.Module):
    def __init__(self,  out_size, encoder, hidden_dim ):
        super(mean, self).__init__()
        self.encoder = encoder
        self.ln = nn.LayerNorm((20))
        

    def forward(self, x,i=None):
        encoding_result = self.encoder(x).float()
        if i != None:
            output = torch.mean(encoding_result[:,i].unsqueeze(-1),dim=-1)
        else:
            output = encoding_result
            output = self.ln(output)
        return output


class naf(nn.Module):
    def __init__(self,  out_size, encoder, hidden_dim):
        super(naf, self).__init__()

        self.encoder = encoder
        in_size = encoder.output_dim

        self.ln = nn.LayerNorm((in_size))

        self.linear1 = nn.Sequential(
            Linear(in_size, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            )
        self.linear2 = nn.Sequential(
            Linear(hidden_dim+in_size, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, out_size),
            SquarePlus()
            )

    def forward(self, x=0, mask=1, feature_vector=None):

        if feature_vector != None:
            encoding_result = feature_vector
        else:
            encoding_result = self.encoder(x).float()
        
        encoding_result = self.ln(encoding_result)*mask
        
        y = self.linear1(encoding_result)
        y = torch.cat([encoding_result, y], -1)
        
        output = self.linear2(y)
        
        return output
    