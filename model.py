import torch
from torch import nn
import numpy as np
device="cuda" if torch.cuda.is_available() else "cpu"
class EncoderBlock(nn.Module):
    def __init__(self,n_heads,embedding_dim,mlp_size,dropout_p):
        super(EncoderBlock,self).__init__()
        self.multihead=nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=n_heads,batch_first=True)
        self.drop1=nn.Dropout(dropout_p)
        self.drop2=nn.Dropout(dropout_p)
        self.norm1=nn.LayerNorm(embedding_dim)
        self.norm2=nn.LayerNorm(embedding_dim)
        self.linear=nn.Sequential(
            nn.Linear(embedding_dim,mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size,embedding_dim)
        )
    def forward(self,x):
        value,_=self.multihead(x,x,x)
        x1=self.drop1(self.norm1(value+x))
        x2=self.drop2(self.norm2(self.linear(x1)+x1))
        return x2



class positional_embedding(nn.Module):
    def __init__(self,max_len,embedding_dim):
        super(positional_embedding,self).__init__()
        self.embed_dim=embedding_dim
        self.pe=torch.zeros((max_len,embedding_dim)).to(device)
        for pos in range(max_len):
            for i in range(0,embedding_dim,2):
                self.pe[pos,i]=np.sin(pos/(10000**((2*i)/embedding_dim)))
                self.pe[pos,i+1]=np.cos(pos/(10000**((2*(i+1))/embedding_dim)))
        self.pe=self.pe.unsqueeze(dim=0)
    def forward(self,x):
        x=x*np.sqrt(self.embed_dim)
        seq_len=x.shape[1]
        x=x+self.pe[:,:seq_len]
        return x

class Encoder(nn.Module):
    def __init__(self,in_vocab_size,out_vocab_size,embedding_dim,n_heads,MLP_size,n_blocks,max_len,dropout_p=0):
        super(Encoder,self).__init__()
        self.embedding=nn.Embedding(in_vocab_size,embedding_dim=embedding_dim)
        self.pe=positional_embedding(max_len=max_len,embedding_dim=embedding_dim)
        self.layer=nn.ModuleList([EncoderBlock(n_heads=n_heads,
                                               embedding_dim=embedding_dim,
                                               mlp_size=MLP_size,
                                               dropout_p=dropout_p
                                               ) for i in range(n_blocks)])
        self.out=nn.Linear(embedding_dim,out_vocab_size)
        
    def forward(self,x):
        x=self.embedding(x)
        x=self.pe(x)
        for layer in self.layer:
            x=layer(x)
        return self.out(x).permute(0,2,1)
def test():
    x=torch.randint(0,200,(6,50)).to(device)
    ender=Encoder(in_vocab_size=200,
                  out_vocab_size=10,
                  embedding_dim=400,
                  n_heads=8,
                  MLP_size=1000,
                  n_blocks=20,
                  max_len=50,
                  dropout_p=0).to(device)
    print(ender(x).shape)    
test()