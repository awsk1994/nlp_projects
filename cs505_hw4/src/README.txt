# Section 1
 - Run "python3 ner.py". Output will be 'results.txt'

# Section 2
 - Run "python3 hmm.py". Output will be 'results.txt'

# Section 3 (Tagger)
## Installation
1. Make sure Theanos is installed (pip3 install theano)
2. Run: 'cd tagger'

## Training
3. Run: "./train.sh" (or "./train.py --train ./data/train.txt --dev ./data/dev.txt --test ./data/test.txt")
4. You can stop early by hitting "ctrl + c". (Evaluation Results are shown after each run, but with my current attempts, waiting until epoch=5 is completed yields good results.)

## Inference
4. Run: "./predict.sh" (or "./predict.py --train ./data/train.txt --test ./data/test.txt", or pick latest ./evaluation/temp/<any_file, recommend largest number>.output)

## Evaluation
5. python3 conlleval <output> (<output> from running predict.sh should be output.txt)
