from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def getfeats(word, o, isPER, isLOC, isORG, isMISC):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)

    features = [
        (o + "word", word),
        # (o + "word_isPER", str(isPER)),
        # (o + "word_isLOC", str(isLOC)),
        # (o + "word_isORG", str(isORG)),
        # (o + "word_isMISC", str(isMISC))
    ]
    # print(features)
    return features


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token

    isPER, isLOC, isORG, isMISC = False, False, False, False
    for o in [-1, 0, 1]:    # TODO: Add plus -2, 2
        if i + o >= 0 and i + o < len(sent):

            word = sent[i + o][0]

            if sent[i + o][-1] == "B-PER" or sent[i + o][-1] == "I-PER":
                isPER = True
                isLOC, isORG, isMISC = False, False, False
            if sent[i + o][-1] == "B-LOC" or sent[i + o][-1] == "I-LOC":
                isLOC = True
                isPER, isORG, isMISC = False, False, False
            if sent[i + o][-1] == "B-ORG" or sent[i + o][-1] == "I-ORG":
                isORG = True
                isLOC, isPER, isMISC = False, False, False
            if sent[i + o][-1] == "B-MISC" or sent[i + o][-1] == "I-MISC":
                isMISC = True
                isLOC, isPER, isORG = False, False, False

            featlist = getfeats(word, o, isPER, isLOC, isORG, isMISC)
            features.extend(featlist)

    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))     # [ [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...], ...]

    print(train_sents)
    
    train_feats = []
    train_labels = []

    for sent in train_sents:
        # print("sent:", sent, ", send_len:", len(sent))  # sent = [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...]
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    model = Perceptron(verbose=1)
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents: # dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in test_sents: # dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
