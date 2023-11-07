
# LFD Project - NLP and noise simulation
For each of the different type of model used (classic, LSTM, pretrained) there is a separate file. For the classic Naive Bayes model, run "lfd_project_baseline.py". For the LSTM model, run "lfd_project_lstm.py". For the pretrained language models (BERT and XLNet), use Google Colab to run "lfd_project_pretrained.ipynb". Further specific instructions per file follow.

  

  

## Running lfd_project_baseline.py
Dependencies: sklearn (install with "pip install sklearn"), numpy (install with "pip install numpy")

Ensure the data files (train.tsv, dev.tsv, test.tsv) are all placed in a folder called "data" which is placed in the directory containing the python file.

1. To run the fully default baseline version of the model, run:

  

```sh
lfd_project_baseline.py -b
```

  

<br>

  

2. To run the model with optimized parameters and feature sets, run:

  

```sh

lfd_project_baseline.py -f 6,0.8 -c

```

  

<br>

  

3. To run the model and evaluate with a specified test file, run:

```sh
lfd_project_baseline.py -f 6,0.8 -c -sf "TEST_FILE"
```

<br>

  

## Running lfd_project_lstm.py
Depdendencies: sklearn (install with "pip install sklearn"), tensorflow (install with "pip install tensorflow"), numpy (install with "pip install numpy")

Ensure the data files (train.tsv, dev.tsv, test.tsv) are all placed in a folder called "data" which is placed in the directory containing the python file.
  

1. To run the baseline version of the model, run:

  

```sh
lfd_project_lstm.py -b
```

  

<br>

  

2. To run the model with optimized parameters and architecture, run:

  

```sh

lfd_project_lstm.py

```

  

<br>

  

3. To run the model and evaluate with a specified test file, run:

```sh
lfd_project_lstm.py -t "TEST_FILE"
```

<br>

  

## Running lfd_project_pretrained.ipynb
Dependencies: none, dependencies are installed via the notebook onto Google Colab environment

Upload the notebook file to Google Colab and place it in Google Drive in a folder called "LFD_project". Place all data files inside a folder called "data" that is inside the "LFD_project" folder.
  
1. Switch to GPU runtime environment
2. Set to chosen model (BERT or XLNet)
 ```sh
lm  =  "bert-base-cased"
```
or
```sh
lm  =  "xlnet-base-cased"
```
3. Run all code cells (note: training has very long run time)

<br>

To run the model and use a specified test file for evaluation:

1. Upload test file to "LFD_project/data"
2. Rename file "test.tsv"
3. Run final code cell (or run all cells if model is not fitted yet)

 