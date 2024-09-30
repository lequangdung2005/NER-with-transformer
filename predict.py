import torch
from pathlib import Path
import pickle
from model import Encoder
from utils import vectorized_sentence
from preprocess_data import preprocess
PATH=Path("save_model/NER_with_transfomer.pth")

device="cuda" if torch.cuda.is_available() else "cpu"

train_loader,test_loader,vocab,label_vocab,max_len=preprocess()


model=Encoder(vocab_size=len(vocab),embedding_dim=400,n_heads=8,MLP_size=400*4,n_blocks=5,max_len=max_len,dropout_p=0.2)

model.load_state_dict(torch.load(PATH,map_location=device))

with open('embed_dict.pkl', 'rb') as f:
    embed_dict = pickle.load(f)

with open("reverse_dict.pkl","rb") as d:
    output_dict=pickle.load(d)

sentence=["hello how are you Alex."]

with torch.inference_mode():
    vector_sentence=vectorized_sentence(embed_dict,sentence)
    input=torch.tensor(vector_sentence[0]).unsqueeze(dim=0).detach()
    yhat=model(input)
    output=yhat.argmax(dim=1)
    output=output.squeeze(dim=1)
    
    print(yhat.shape)
   
    

    