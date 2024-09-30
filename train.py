import torch
from torch import nn
from preprocess_data import preprocess
from model import Encoder
from engine import train
from utils import save_model

device="cuda" if torch.cuda.is_available() else "cpu"

epochs=5
train_loader,test_loader,vocab,label_vocab,max_len=preprocess()
model=Encoder(in_vocab_size=len(vocab),out_vocab_size=len(label_vocab),embedding_dim=400,n_heads=8,MLP_size=400*4,n_blocks=5,max_len=max_len,dropout_p=0.2)

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters())

result=train(model,train_loader,test_loader,optimizer=optimizer,loss_fn=loss_fn,epochs=epochs,device=device)
save_model(model=model,target_dir="save_model",model_name="NER_with_transfomer.pth")
