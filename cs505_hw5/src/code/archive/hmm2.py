from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
# from hmmlearn.hmm import GaussianHMM
import hmmlearn.hmm as hmm
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

def predict(model, seq, n_obs, hidden_state_id_to_hidden_state_map):
    # Generate fit X
    X = np.zeros((len(seq), n_obs), dtype=int)
    for i in range(len(seq)):
        word = seq[i]
        obs_id = obs_to_obs_id_map[word[1]]
        X[i][obs_id] += 1
    X = X.reshape((X.shape[0] * X.shape[1], 1))
    
    # Fit and Predict
    model = model.fit(X)
    logprob, raw_preds = model.decode(X, algorithm="viterbi")  # alice_hears -> hidden_state

    # Process Prediction
    print("len(seq)={}, n_obs={}".format(len(seq), n_obs))
    print("raw_preds shape (before split) = ", raw_preds.shape)

    raw_preds = np.split(raw_preds, len(seq))
    print(raw_preds)
    print("raw_preds shape. rows = {}, num_elem_per_row = {}, {} ".format(len(raw_preds), len(raw_preds[0]), len(raw_preds[-1])))
    
    pred_arr = []
    for raw_pred in raw_preds:
        hidden_state_id = np.argwhere(np.array(raw_pred) == 1)[0][0]

        print("hidden_state_id={}".format(hidden_state_id))

        hidden_state = hidden_state_id_to_hidden_state_map[hidden_state_id]
        pred_arr.append(hidden_state)

    return pred_arr

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
    
    hidden_state_count_hash = {}
    obs_count_hash = {}
    uniq_states = set()

    # TODO: start as transition prob?

    # Count number of hidden_state and number of observations
    for sent in train_sents:    # TODO: can do with set, no need hash
        # print("sent:", sent, ", send_len:", len(sent))  # sent = [('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...]
        for word in sent:
            # word = ('Comisión', 'NC', 'B-ORG')
            if word[2] in hidden_state_count_hash:
                hidden_state_count_hash[word[2]] += 1
            else:
                hidden_state_count_hash[word[2]] = 1

            if word[1] in obs_count_hash:
                obs_count_hash[word[1]] += 1
            else:
                obs_count_hash[word[1]] = 1

    hidden_state_count_sum = sum(hidden_state_count_hash.values())
    n_hidden_state = len(list(hidden_state_count_hash.keys()))
    n_obs = len(list(obs_count_hash.keys()))

    # Generate hidden_state to hidden_state_id map # TODO: can do with set, no need map
    hidden_state_to_hidden_state_id_map = {}
    for i in range(len(hidden_state_count_hash.keys())):
        hidden_state_val = list(hidden_state_count_hash.keys())[i]
        hidden_state_to_hidden_state_id_map[hidden_state_val] = i

    hidden_state_id_to_hidden_state_map = {}
    for i in range(len(hidden_state_count_hash.keys())):
        hidden_state_val = list(hidden_state_count_hash.keys())[i]
        hidden_state_id_to_hidden_state_map[i] = hidden_state_val

    # Generate obs to obs_id map # TODO: can do with set, no need map
    obs_to_obs_id_map = {}
    # print("len(obs_count_hash.keys()) = ", len(obs_count_hash.keys()))
    for i in range(len(obs_count_hash.keys())):
        obs_val = list(obs_count_hash.keys())[i]
        obs_to_obs_id_map[obs_val] = i

    # Generate start_prob and transition count matrix
    trans_count_mat = np.zeros((n_hidden_state, n_hidden_state))
    start_prob_count_mat = np.zeros((n_hidden_state))

    for sent in train_sents:
        # Aggregate trans_count_mat on first word (sent[0])
        hidden_state_val = sent[0][2]
        hidden_state_id = hidden_state_to_hidden_state_id_map[hidden_state_val]
        start_prob_count_mat[hidden_state_id] += 1

        prev_hidden_state_id = None

        for word in sent:
            hidden_state_val = word[2]
            hidden_state_id = hidden_state_to_hidden_state_id_map[hidden_state_val]

            if prev_hidden_state_id != None: # Skip first word for now. TODO: consider adding one more hidden_state '$' indicating start of phrase.
                trans_count_mat[prev_hidden_state_id][hidden_state_id] += 1
            
            prev_hidden_state_id = hidden_state_id

    # np.set_printoptions(threshold=np.inf) # uncomment to print entire matrix
    # print("trans_count_mat")
    # print(trans_count_mat)

    # print("hidden_state_to_hidden_state_id_map")
    # print(hidden_state_to_hidden_state_id_map)

    print("hidden_state_id_to_hidden_state_map shape={}".format(len(hidden_state_id_to_hidden_state_map.keys())))

    # Generate probability version of the count matrix
    start_prob_mat = start_prob_count_mat / np.sum(start_prob_count_mat) 

    trans_mat = np.zeros(trans_count_mat.shape)
    for row_id in range(trans_count_mat.shape[0]):
        trans_row = trans_count_mat[row_id]
        sum_per_row = np.sum(trans_row)
        trans_mat[row_id] = trans_row/np.sum(trans_row) if sum_per_row != 0 else np.zeros(trans_row.shape)

    # print("start_prob_mat")
    # print(start_prob_mat.shape)

    # print("trans_mat")
    # print(trans_mat.shape)

    # Generate emission mat
    emission_count_mat = np.zeros((n_hidden_state, n_obs))
    for sent in train_sents:
        for word in sent:
            hidden_state = word[2]
            obs = word[1]

            hidden_state_id = hidden_state_to_hidden_state_id_map[hidden_state]
            obs_id = obs_to_obs_id_map[obs]

            emission_count_mat[hidden_state_id][obs_id] += 1

    # Generate emission prob mat
    emission_mat = np.zeros(emission_count_mat.shape)
    for row_id in range(emission_mat.shape[0]):
        emis_row = emission_mat[row_id]
        sum_per_row = np.sum(emis_row)
        emission_mat[row_id] = emis_row/np.sum(emis_row) if sum_per_row != 0 else np.zeros(emis_row.shape)

    print("n_hidden_state={}".format(n_hidden_state))
    # Apply model
    model = hmm.GaussianHMM(n_components=n_hidden_state, covariance_type="diag", n_iter=1000)
    model.startprob = start_prob_mat
    model.transmat = trans_mat
    model.emissionprob = emission_mat


    for sent in train_sents[5:10]:
        y_pred = predict(model, sent, n_obs, hidden_state_id_to_hidden_state_map)
        ans = [w[2] for w in sent]
        print("y_pred={}\nans={}\n\n\n======".format(y_pred, ans))






    # print("pred_arr=", pred_arr)
    # print("ans_arr=", [w[2] for w in train_sents[0]])

    # bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T   # bob_says -> observation

    # model = model.fit(bob_says)
    # logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")  # alice_hears -> hidden_state

    # print("bob_says")
    # print(bob_says)

    # print("alice_hears")
    # print(alice_hears)




    # start_prob_dict = {k: v / hidden_state_count_sum for (k, v) in hidden_state_count_hash.items()}

    # n_uniq_states = len(hidden_state_count_hash.keys())


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
    