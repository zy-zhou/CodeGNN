# CodeGNN

## Requirements
* Python 3.8
* PyTorch 1.10.1
* pyg 2.0.4
* torchtext 0.11.1
* cudatoolkit 11.3.1
* cudnn 8.2.1
* javalang 0.13
* tqdm 4.47
* numpy 1.18.5
* NLTK 3.5
* py-rouge 1.1

## Training & Inference
* Train on Java dataset: `python Main.py`
* Set batch size to *n*: `python Main.py -b n`
* Evaluate on test set: `python Main.py -t`
