from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from hmmlearn.hmm import GaussianHMM
import numpy as np
# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, isPER, isLOC, isORG, isMISC):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)

    # print("word:", word, ", isMISC:", isMISC)
    features = [
        (o + "word", word),
        (o + "word_isPER", str(isPER)),
        (o + "word_isLOC", str(isLOC)),
        (o + "word_isORG", str(isORG)),
        (o + "word_isMISC", str(isMISC))
        # TODO: add more features here.
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
            if sent[i + o][-1] == "B-LOC" or sent[i + o][-1] == "I-LOC":
                isLOC = True
            if sent[i + o][-1] == "B-ORG" or sent[i + o][-1] == "I-ORG":
                isORG = True
            if sent[i + o][-1] == "B-MISC" or sent[i + o][-1] == "I-MISC":
                isMISC = True

            featlist = getfeats(word, o, isPER, isLOC, isORG, isMISC)
            features.extend(featlist)

    return dict(features)

def gen_trans_mat(data, n_components):
    pass

def get_start_prob_mat(data, n_components):
    pass

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))     # [ [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...], ...]


    # N_COMPONENTS = 9
    # start_prob_mat = get_start_prob_mat(train_sents, N_COMPONENTS)
    # trans_mat = gen_trans_mat(train_sents, N_COMPONENTS)  # array, shape (n_components, n_components)
    # model = hmm.GaussianHMM(n_components=N_COMPONENTS)
    # model.startprob_ = start_prob_mat
    # model.transmat_ = trans_mat
    
    state_count_hash = {}
    obs_count_hash = {}
    uniq_states = set()

    # TODO: start as transition prob?

    # Count number of state and number of observations
    for sent in train_sents:    # TODO: can do with set, no need hash
        # print("sent:", sent, ", send_len:", len(sent))  # sent = [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...]
        for word in sent:
            # word = ('Comisión', 'NC', 'B-ORG')
            if word[1] in state_count_hash:
                state_count_hash[word[1]] += 1
            else:
                state_count_hash[word[1]] = 1

            if word[2] in obs_count_hash:
                obs_count_hash[word[2]] += 1
            else:
                obs_count_hash[word[2]] = 1

    state_count_sum = sum(state_count_hash.values())
    n_state = len(list(state_count_hash.keys()))
    n_obs = len(list(obs_count_hash.keys()))

    # Generate state to state_id map # TODO: can do with set, no need map
    state_to_state_id_map = {}
    for i in range(len(state_count_hash.keys())):
        state_val = list(state_count_hash.keys())[i]
        state_to_state_id_map[state_val] = i

    # Generate obs to obs_id map # TODO: can do with set, no need map
    obs_to_obs_id_map = {}
    for i in range(len(obs_count_hash.keys())):
        obs_val = list(obs_to_obs_id_map.keys())[i]
        obs_to_obs_id_map[obs_val] = i

    # Generate start_prob and transition count matrix
    trans_count_mat = np.zeros((n_state, n_state))
    start_prob_count_mat = np.zeros((n_state))

    for sent in train_sents:
        # Aggregate trans_count_mat on first word (sent[0])
        state_val = sent[0][1]
        state_id = state_to_state_id_map[state_val]
        start_prob_count_mat[state_id] += 1

        prev_state_id = None

        for word in sent:
            state_val = word[1]
            state_id = state_to_state_id_map[state_val]

            if prev_state_id != None: # Skip first word for now. TODO: consider adding one more state '$' indicating start of phrase.
                trans_count_mat[prev_state_id][state_id] += 1
            
            prev_state_id = state_id

    # np.set_printoptions(threshold=np.inf) # uncomment to print entire matrix
    # print("trans_count_mat")
    # print(trans_count_mat)

    # print("state_to_state_id_map")
    # print(state_to_state_id_map)

    # Generate probability version of the count matrix
    start_prob_mat = start_prob_count_mat / np.sum(start_prob_count_mat) 

    trans_mat = np.zeros(trans_count_mat.shape)
    for row_id in range(trans_count_mat.shape[0]):
        trans_row = trans_count_mat[row_id]
        sum_per_row = np.sum(trans_row)
        trans_mat[row_id] = trans_row/np.sum(trans_row) if sum_per_row != 0 else np.zeros(trans_row.shape)

    print("start_prob_mat")
    print(start_prob_mat)

    print("trans_mat")
    print(trans_mat)

    # Generate emission mat
    emission_count_mat = np.zeros((n_state, n_obs))
    for sent in train_sents:
        for word in sent:
            state = word[1]
            obs = word[2]

            state_id = state_to_state_id_map[state]
            obs_id = obs_to_obs_id_map[obs]

            emission_count_mat[state_id][obs_id] += 1

    # Generate emission prob mat
    emission_mat = np.zeros(emission_count_mat.shape)
    for row_id in range(emission_mat.shape[0]):
        emis_row = emission_mat[row_id]
        sum_per_row = np.sum(emis_row)
        emission_mat[row_id] = emis_row/np.sum(emis_row) if sum_per_row != 0 else np.zeros(emis_row.shape)

    # Apply model
    model = hmm.MultinomialHMM(n_components=n_state)
    model.startprob = start_prob_mat
    model.transmat = trans_mat
    model.emissionprob = emission_mat

    # Generate sequence to predict
    for sent in train_sents:    # TODO: can do with set, no need hash
        # print("sent:", sent, ", send_len:", len(sent))  # sent = [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...]
        for word in sent:


    bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T   # bob_says -> observation

    model = model.fit(bob_says)
    logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")  # alice_hears -> state

    print("bob_says")
    print(bob_says)

    print("alice_hears")
    print(alice_hears)




    # start_prob_dict = {k: v / state_count_sum for (k, v) in state_count_hash.items()}

    # n_uniq_states = len(state_count_hash.keys())


    # model = GaussianHMM(n_components = n_uniq_states)


    '''
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # # TODO: play with other models
    # model = GaussianHMM(n_components = 9) # Perceptron(verbose=1)
    # print(X_train)

    # model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents: # test_sents: # :
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
        for sent in dev_sents: # test_sents: # dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
    '''
    