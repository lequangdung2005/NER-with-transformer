import torch
from torch.utils.data import Dataset
import os
def pad_tensor(x,max_len):
    z=[]
    for x_ in x:
        x_=torch.cat([x_,torch.zeros(max_len-x_.shape[0])])
        z.append(x_.type(torch.int32))
    return z

class PadCollate:
    def pad_collate(self,data):
        max_len=0
        x=[data[i][0] for i in range(len(data))]
        y=[data[i][1]for i in range(len(data))]
        for tensor in x:
            max_len= tensor.shape[0] if tensor.shape[0]>max_len else max_len
        x=pad_tensor(x,max_len)
        y=pad_tensor(y,max_len)
        x=torch.stack(x)
        y=torch.stack(y)
        return x,y
    def __call__(self,data):
        return self.pad_collate(data)

        

def load_data(file_path):
    with open(file_path,"r",encoding="utf8") as file:
        data=[ line.strip() for line in file.readlines()] 
    return data

def create_vocab(sentences,reverse=False):
    vocab={}
    vocab["<unk>"]=0
    i=1
    max_len=0
    for sentence in sentences:
        sentence=sentence.strip().split()
        max_len =len(sentence) if max_len<len(sentence) else max_len
        for word in sentence:
            if word not in vocab.keys():
                vocab[word]=i
                i+=1
    if reverse==True:
        reverse_vocab={k:v for v,k in vocab.items()}
        return vocab,max_len,reverse_vocab
    return vocab,max_len

def vectorized_sentence(vocab,sentences):
    vectorized_sentences=[]
    for i,sentence in enumerate(sentences):
        sentence=sentence.strip().split()
        vectorized=[vocab.get(word,0) for word in sentence]
        vectorized_sentences.append(torch.tensor(vectorized))
    return vectorized_sentences

class Data(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return (self.data[index],self.label[index])




import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
