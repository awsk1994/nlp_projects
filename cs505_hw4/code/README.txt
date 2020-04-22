# README

## Section 1
 - Run "python3 ner.py". Output will be 'results.txt'

## Section 2
 - Run "python3 hmm.py". Output will be 'results.txt'

## Section 3 (Tagger)
### Installation
 - Make sure Theanos is installed (pip3 install theano)

### Training
 - Run "./train.py --train ./data/train.txt --dev ./data/dev.txt --test ./data/test.txt"

### Inference

You can either grab it the temp files:
 	 - Output: ./evaluation/temp/<largest number>.output
 	 - Score: ./evaluation/temp/<largest number>.score

Or, run "./predict.py --test ./data/test.txt"