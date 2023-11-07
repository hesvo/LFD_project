import csv
import random as python_random
import statistics
import json
import argparse
import numpy as np
import pickle

P_ADD = 0.05
P_DEL = 0.025
P_ALT = 0.05

RAND_CHARS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',]

np.random.seed(1234)


def sim_noise(op, docs):
    
    match op:
        case "add":
            docs = noise_add(docs)
        
        case "delete":
            docs = noise_delete(docs)

        case "alter":
            docs = noise_alter(docs)

        case "add_delete":
            docs = noise_add(docs)
            docs = noise_delete(docs)
        
        case "add_alter":
            docs = noise_add(docs)
            docs = noise_alter(docs)

        case "delete_alter":
            docs = noise_delete(docs)
            docs = noise_alter(docs)

        case "add_delete_alter":
            docs = noise_add(docs)
            docs = noise_delete(docs)
            docs = noise_alter(docs)
    return docs
    

def noise_add(docs):
    new_docs = []
    for d in docs:
        nd = []
        for w in d:
            if w.find("\u2066@USER") == -1 and w.find("@USER") and w.find("URL") == -1:
                for i, l in enumerate(w):
                    r = np.random.rand()
                    if r < P_ADD and l != "#":
                        r_char = np.random.randint(0, len(RAND_CHARS))
                        w = w[:i] + RAND_CHARS[r_char] + w[i:]
            nd.append(w)
        new_docs.append(nd)

    return new_docs
    
    


def noise_delete(docs):
    new_docs = []
    for d in docs:
        nd = []
        for w in d:
            if w.find("\u2066@USER") == -1 and w.find("@USER") and w.find("URL") == -1:
                for i, l in enumerate(w):
                    r = np.random.rand()
                    if r < P_ADD and l != "#":
                        r_char = np.random.randint(0, len(RAND_CHARS))
                        w = w[:i] + w[i+1:]
            nd.append(w)
        new_docs.append(nd)

    return new_docs

def noise_alter(docs):
    new_docs = []
    for d in docs:
        nd = []
        for w in d:
            if w.find("\u2066@USER") == -1 and w.find("@USER") and w.find("URL") == -1:
                for i, l in enumerate(w):
                    r = np.random.rand()
                    if r < P_ADD and l != "#":
                        r_char = np.random.randint(0, len(RAND_CHARS))
                        w = w[:i] + RAND_CHARS[r_char] + w[i+1:]
            nd.append(w)
        new_docs.append(nd)

    return new_docs

if __name__ == "__main__":
    
    train_f = "train"
    dev_f = "dev"
    test_f = "test"


    ops = ["add", "delete", "alter", "add_delete", "add_alter", "delete_alter", "add_delete_alter"]

    f_lst = [train_f, dev_f, test_f]

    # perform each operation for each data file
    for f in f_lst:
        documents = []
        labels = []

        with open("./data/"+f+".tsv", encoding='utf-8') as in_file:
            for line in in_file:
                tokens = line.strip().split()
                
                documents.append(tokens[:-1])
                labels.append(tokens[-1])
            
        for o in ops:
            with open("./data/"+f+"_"+o+".tsv", "w+", encoding='utf-8') as out_file:
                new_docs = sim_noise(o, documents.copy())
                for i, d in enumerate(new_docs):
                    for w in d:
                        out_file.write(w+" ")
                    out_file.write("\t")
                    out_file.write(labels[i])
                    out_file.write("\n")
                    
    
