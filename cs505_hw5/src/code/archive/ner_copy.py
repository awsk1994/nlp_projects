from nltk.corpus import conll2002
from hmmlearn import hmm
import numpy as np
# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

import random

def conputePar(sents, vocab, tag):
    m = len(vocab)
    n = len(tag)
    a_matrix = np.zeros((n, n))
    o_matrix = np.zeros((n, m))
    pi = np.zeros(n)

    for lst in sents:
        seq_len = len(lst)
        for i in range(seq_len - 1):
            current_tagid = tag[lst[i][-1]]
            next_tagid = tag[lst[i + 1][-1]]
            a_matrix[current_tagid][next_tagid] += 1
    a_matrix[a_matrix == 0.] = 1e-10
    a_matrix = a_matrix / a_matrix.sum(axis=1).reshape(-1, 1)
    print("done")

    for lst in sents:
        for i in range(len(lst)):
            tag_id = tag[lst[i][-1]]
            word_id = vocab[lst[i][0]]
            o_matrix[tag_id][word_id] += 1
    o_matrix[o_matrix == 0.] = 1e-10
    o_matrix = o_matrix / o_matrix.sum(axis=1).reshape(-1, 1)
    print("done | emission shape={}".format(o_matrix.shape))

    for lst in sents:
        init_tagid = tag[lst[0][-1]]
        pi[init_tagid] += 1
    pi[pi == 0.] = 1e-10
    pi= pi / pi.sum()
    print("done")

    return a_matrix, o_matrix, pi

def write_out(filename, datas):
    with open(filename, "w") as out:
        out.write("{}".format(datas))

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf) # uncomment to print entire matrix

    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))

    train_vocab={}
    train_tag={}

    for sent in train_sents:
        for i in range(len(sent)):
            if sent[i][0] not in train_vocab:
                train_vocab[sent[i][0]] = len(train_vocab)
            if sent[i][-1] not in train_tag:
                train_tag[sent[i][-1]] = len(train_tag)

    # train_vocab["UNK"] = len(train_vocab) # random.randint(0, len(train_vocab) - 1)
    # train_word_id=[]
    X=[]
    train_lengths = []
    for sent in train_sents:
        # train_word_id_s = []
        train_lengths.append(len(sent))
        for i in range(len(sent)):
            # train_word_id_s.append(train_vocab[sent[i][0]])
            X.append(train_vocab[sent[i][0]])
        # train_word_id.append(train_word_id_s)

    X=np.array(X).reshape(-1, 1)

    print("done1")
    a_train_matrix,o_train_matrix,pi_train=conputePar(train_sents, train_vocab, train_tag)
    # TODO: play with other models
    n_states=len(train_tag)
    print("n_states={}".format(n_states))
    model = hmm.MultinomialHMM(n_components=n_states)
    model.startprob_ = pi_train
    model.transmat_ = a_train_matrix
    model.emissionprob_ = o_train_matrix

    write_out("startprob.txt", pi_train)
    write_out("transmat.txt", a_train_matrix)
    write_out("emissionprob.txt", o_train_matrix)

    # model.fit(X, train_lengths)

    # model.fit(X)
    print("a")

    # switch to test_sents for your final results
    test_word_id = []
    X_dev=[]
    test_lengths = []
    for sent in test_sents:
        test_word_id_s = []
        test_lengths.append(len(sent))
        for i in range(len(sent)):
            if sent[i][0] not in train_vocab:
                # test_word_id_s.append(train_vocab["UNK"]-1)
                # X_dev.append(train_vocab["UNK"])
                # X_dev.append(train_vocab["O"])
                # X_dev.append(len(train_vocab)-1)
                X_dev.append(random.randint(0, len(train_vocab)-1))
            else:
                # test_word_id_s.append(train_vocab[sent[i][0]])
                X_dev.append(train_vocab[sent[i][0]])
        # test_word_id.append(test_word_id_s)

    X_dev = np.array(X_dev).reshape(-1, 1)
    print("X_dev shape={}".format(X_dev.shape))
    # model = model.fit(X_dev, test_lengths)
    _, y_pred_id = model.decode(X_dev,test_lengths)
    # print(y_pred_id)
    # logprob, y_pred_id = model.decode(X_dev,algorithm="viterbi")
    y_pred=[]
    tag_train = dict((id_, tag) for tag, id_ in train_tag.items())
    for i in range(len(y_pred_id)):
        y_pred.append(tag_train[y_pred_id[i]])

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
