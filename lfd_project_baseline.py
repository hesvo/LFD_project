#!/usr/bin/env python

'''
This Python script allows testing various classifier algorithms from
the sklearn library as well as specifying aspects of the feature vectors
used, through the command line arguments provided.
Run script with -h for more information.
'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import sys
from nltk import PorterStemmer, WordNetLemmatizer
import os

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.tsv', type=str,
                        help="Train file to learn from (default train.tsv)")
    parser.add_argument("-df", "--dev_file", default='dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.tsv)")
    parser.add_argument("-sf", "--test_file", default='test.tsv', type=str,
                        help="Dev file to evaluate on (default dev.tsv)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-b", "--base", action="store_true",
                        help="Use baseline version of model")
    parser.add_argument("-c", "--combine", action="store_true",
                        help="Combine CountVectorizer and TF-IDF vectorizer")
    parser.add_argument("-char", "--character", action="store_true",
                        help="Use character ngrams instead of word ngrams")
    parser.add_argument("-n", "--ngram", default="1,1", type=str,
                        help="Use specified ngram range (min,max) (default (1,1))")
    parser.add_argument("-f", "--freq", default="1,1.0", type=str,
                        help="Use specified document frequency range (min,max) (default (1,1.0))")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''
    The read_corpus() function is used to load data (training or testing data)
    from a file into separate variables that store the tokens from the document
    and the associated labels.
    '''
    documents = []
    labels = []

    with open("./data/"+corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()

            documents.append(tokens[:-1])
            labels.append(tokens[-1])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp

def run_model(args, op):
    # converts arguments for ngram range to appropriate variables
    ngrange = args.ngram
    ngrange = ngrange.split(',')
    n1 = int(ngrange[0])
    n2 = int(ngrange[1])

    # converts arguments for min/max document frequency to variable
    # requires int or float depending on input
    doc_freq = args.freq
    doc_freq = doc_freq.split(',')

    if doc_freq[0].isdigit():
        d1 = int(doc_freq[0])
    else:
        d1 = float(doc_freq[0])

    if doc_freq[1].isdigit():
        d2 = int(doc_freq[1])
    else:
        d2 = float(doc_freq[1])

    # X_train, Y_train = read_corpus("train_"+op+".tsv")
    # X_dev, Y_dev = read_corpus("dev_"+op+".tsv")
    # X_test, Y_test = read_corpus("test_"+op+".tsv")
    
    
    # reads the training and testing set from separate files
    # sentiment or topoic is used based on argument

    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    X_test, Y_test = read_corpus(args.test_file)


    if args.character:
        ngram_type = 'char'
    else:
        ngram_type = 'word'
    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec_tf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(n1,n2), min_df=d1, max_df=d2, analyzer=ngram_type)

    # Bag of Words vectorizer
    vec_cv = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(n1,n2), min_df=d1, max_df=d2, analyzer=ngram_type)


    select = SelectKBest(k=20)
    x1 = vec_cv.fit_transform(X_train)
    y1 = Y_train
    select.fit_transform(x1, y1)
    print("Highest score features:")
    print(vec_cv.get_feature_names_out()[select.get_support()])

    
    # create correct pipeline according to selected classifier and features from arguments
    if args.combine:
        union = FeatureUnion([("count", vec_cv), ("tf", vec_tf)])
        pipe = ('union', union)
    elif args.tfidf:
        pipe = ('vec', vec_tf)
    else:
        pipe = ('vec', vec_cv)

    if args.base:
        class_alg = MultinomialNB()
    else:
        class_alg = MultinomialNB(alpha=2.6)
    classifier = Pipeline(steps=[pipe, ('cls', class_alg)])



    # train specified classifier on the training data (X_train and Y_train)
    classifier.fit(X_train, Y_train)

    # get classifier predictions for testing data (X_test)
    Y_pred = classifier.predict(X_test)

    # compare predicted labels (Y_pred) to actual classes of training data (Y_test) to get classifier accuracy
    acc = accuracy_score(Y_test, Y_pred)
    print(f"Final accuracy: {acc}")

    prec, rec, f1, s = precision_recall_fscore_support(Y_test,Y_pred, labels=['NOT', 'OFF'])
    print(f"Final precision: {prec}")
    print(f"Final recall: {rec}")
    print(f"Final f1: {f1}")
    print("(Order: NOT, OFF)")

    # with open("./results/classic_"+"base"+"_res.txt", "w+", encoding="utf-8") as res_file:
    #     res_file.write(f"Final accuracy: {acc}\n")
    #     res_file.write(f"Final precision: {prec}\n")
    #     res_file.write(f"Final recall: {rec}\n")
    #     res_file.write(f"Final f1: {f1}\n")
    #     res_file.write("(Order: NOT, OFF)\n")



if __name__ == "__main__":
    args = create_arg_parser()

    run_model(args, "none")

    # ops = ["add", "delete", "alter", "add_delete", "add_alter", "delete_alter", "add_delete_alter"]

    
    # for o in ops:
    #     run_model(args, o)


    

