import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
from utils import load_data,create_vocab,vectorized_sentence,Data,PadCollate

def preprocess():
    #load data
    train_sentence=load_data('data/large/train/sentences.txt')
    train_labels=load_data('data/large/train/labels.txt')
    
    test_sentence=load_data('data/large/test/sentences.txt')
    test_labels=load_data('data/large/test/labels.txt')

    
    #create vocab
    vocab,max1=create_vocab(train_sentence)
    label_vocab,max2,reverse_vocab=create_vocab(train_labels,reverse=True)
    max_len=max(max1,max2)
    #vectorize data
    train_data=vectorized_sentence(vocab,train_sentence)
    test_data=vectorized_sentence(vocab,test_sentence)
    train_label_data=vectorized_sentence(label_vocab,train_labels)
    test_label_data=vectorized_sentence(label_vocab,test_labels)

    #convert to Dataset
    train_dataset=Data(train_data,train_label_data)
    test_dataset=Data(test_data,test_label_data)
    
    colllate_fn=PadCollate()

    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,collate_fn=colllate_fn)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True,collate_fn=colllate_fn)

    

    with open('embed_dict.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open("reverse_dict.pkl","wb") as d:
        pickle.dump(reverse_vocab,d)
    return train_loader,test_loader,vocab,label_vocab,max_len
train_loader,test_loader,vocab,label_vocab,max_len=preprocess()
