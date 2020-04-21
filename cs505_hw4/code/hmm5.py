from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
# from hmmlearn.hmm import GaussianHMM
import hmmlearn.hmm as hmm
import numpy as np
import random

# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

# def calc_score(lst_a, lst_b):
#     score = 0
#     total = 0
#     for i in range(len(lst_a)):
#         if lst_a[i] == lst_b[i]:
#             score += 1
#         total += 1
#     print("score/total = {}/{} = {}".format(score, total, score/total))
#     return score/total


# TODO: check if the transmat is really insreted into the model. Consider using: 
# startprob_prior (array, shape (n_components, ), optional) – Parameters of the Dirichlet prior distribution for startprob_.
# transmat_prior (array, shape (n_components, n_components), optional) – Parameters of the Dirichlet prior distribution for each row of the transition probabilities transmat_.


class CALC_N_HIDDEN_STATE_OBS:
    @staticmethod
    def gen_hidden_state_and_obs_ds(train_sents):
        print("Generating hidden state and observation data structures...")
        hidden_state_count_hash, obs_count_hash, n_hidden_state, n_obs = CALC_N_HIDDEN_STATE_OBS.count_hidden_state_and_obs(train_sents)
        hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map = CALC_N_HIDDEN_STATE_OBS.gen_obj_id_mapping(hidden_state_count_hash)
        obs_to_obs_id_map, obs_id_to_obs_map = CALC_N_HIDDEN_STATE_OBS.gen_obj_id_mapping(obs_count_hash)
        
        if DEBUG_DATA == 1:
            print("hidden_state_to_hidden_state_id_map:\n{}\nhidden_state_id_to_hidden_state_map:\n{}\n".format(hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map))
            # print("obs_to_obs_id_map:\n{}\nobs_id_to_obs_map:\n{}\n".format(obs_to_obs_id_map, obs_id_to_obs_map))
            print("first 6 obs = ", ["{}:{}".format(i, obs_id_to_obs_map[i]) for i in range(6)])

        return hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, obs_id_to_obs_map, n_hidden_state, n_obs

    @staticmethod
    def count_hidden_state_and_obs(train_sents):
        hidden_state_count_hash = {}
        obs_count_hash = {}

        for sent in train_sents:    # TODO: can do with set, no need hash        
            if ACCOUNT_FIRST_TRANSITION == 1:
                if '$' in hidden_state_count_hash:        
                    hidden_state_count_hash['$'] += 1
                    obs_count_hash['$'] += 1
                else:
                    hidden_state_count_hash['$'] = 1
                    obs_count_hash['$'] = 1

            for word in sent:
                if word[HIDDEN_STATE_IDX] in hidden_state_count_hash:
                    hidden_state_count_hash[word[HIDDEN_STATE_IDX]] += 1
                else:
                    hidden_state_count_hash[word[HIDDEN_STATE_IDX]] = 1

                if word[OBS_IDX] in obs_count_hash:
                    obs_count_hash[word[OBS_IDX]] += 1
                else:
                    obs_count_hash[word[OBS_IDX]] = 1


        n_hidden_state = len(list(hidden_state_count_hash.keys()))
        n_obs = len(list(obs_count_hash.keys()))

        if DEBUG_DATA == 1:
            print("Hidden states ={}, len = {}".format(hidden_state_count_hash.keys(), len(hidden_state_count_hash.keys())))
            print("Observations len = {}".format(len(obs_count_hash.keys())))

        return hidden_state_count_hash, obs_count_hash, n_hidden_state, n_obs

    @staticmethod
    def gen_obj_id_mapping(obj_hash):
        obj_to_obj_id_map = {}
        obj_id_to_obj_map = {}

        for i in range(len(obj_hash.keys())):
            obj = list(obj_hash.keys())[i]
            obj_to_obj_id_map[obj] = i
            obj_id_to_obj_map[i] = obj

        return obj_to_obj_id_map, obj_id_to_obj_map

class CALC_START_TRANS_MAT:
    @staticmethod
    def gen_start_and_transition_mat(train_sents, n_hidden_state, hidden_state_to_hidden_state_id_map):
        if DEBUG == 1:
            print("Generating start and trans mat...")
        start_prob_count_mat, trans_count_mat = CALC_START_TRANS_MAT.count_start_transition_prob_mat(train_sents, n_hidden_state, hidden_state_to_hidden_state_id_map)
        start_prob_mat, trans_mat = CALC_START_TRANS_MAT.calc_start_trans_prob(start_prob_count_mat, trans_count_mat)
        return start_prob_mat, trans_mat

    @staticmethod
    def count_start_transition_prob_mat(train_sents, n_hidden_state, hidden_state_to_hidden_state_id_map):
        if DEBUG == 1:
            print("Generate start_prob and transition count matrix")

        trans_count_mat = np.zeros((n_hidden_state, n_hidden_state))
        start_prob_count_mat = np.zeros((n_hidden_state))

        for sent in train_sents:
            prev = None
            prev_hidden_state_id = None

            # Aggregate start_prob (only on first word of the sentence/phrase)
            hidden_state_id = hidden_state_to_hidden_state_id_map[sent[0][HIDDEN_STATE_IDX]]
            start_prob_count_mat[hidden_state_id] += 1

            if ACCOUNT_FIRST_TRANSITION == 1:
                prev_hidden_state_id = hidden_state_to_hidden_state_id_map['$']
            else:
                prev_hidden_state_id = None

            for word in sent:
                hidden_state_id = hidden_state_to_hidden_state_id_map[word[HIDDEN_STATE_IDX]]

                if ACCOUNT_FIRST_TRANSITION == 1 or (ACCOUNT_FIRST_TRANSITION == 0 and prev_hidden_state_id != None):
                    trans_count_mat[prev_hidden_state_id][hidden_state_id] += 1

                prev_hidden_state_id = hidden_state_id
                prev = word

        if DEBUG_DATA == 1:
            print("start_prob_count_mat={}".format(start_prob_count_mat))
            print("trans_count_mat={}".format(trans_count_mat))

        # print("hidden_state_id_to_hidden_state_map shape={}".format(len(hidden_state_id_to_hidden_state_map.keys())))

        return start_prob_count_mat, trans_count_mat

    @staticmethod
    def calc_start_trans_prob(start_prob_count_mat, trans_count_mat):
        # Generate probability version of the count matrix
        if DEBUG == 1:
            print("Generate probability version of the count matrix")

        trans_mat = trans_count_mat/np.sum(trans_count_mat,axis=1).reshape(-1,1)
        trans_mat[trans_mat == 0.] = 1e-10

        start_prob_mat = start_prob_count_mat / np.sum(start_prob_count_mat) 
        start_prob_mat[start_prob_mat == 0.] = 1e-10

        if DEBUG_DATA == 1:
            print("start_prob_mat={}".format(start_prob_mat))
            print("trans_mat={}".format(trans_mat))

        return start_prob_mat, trans_mat

class CALC_EMISSION_MAT:
    @staticmethod
    def gen_emission_mat(train_sents, n_hidden_state, n_obs, hidden_state_to_hidden_state_id_map, obs_to_obs_id_map):
        if DEBUG == 1:
            print("Generating emission_mat")
        emission_count_mat = CALC_EMISSION_MAT.count_emission(train_sents, n_hidden_state, n_obs, hidden_state_to_hidden_state_id_map, obs_to_obs_id_map)
        emission_mat = CALC_EMISSION_MAT.calc_emission(emission_count_mat)

        return emission_mat

    @staticmethod
    def count_emission(train_sents, n_hidden_state, n_obs, hidden_state_to_hidden_state_id_map, obs_to_obs_id_map):
        emission_count_mat = np.zeros((n_hidden_state, n_obs))
        for sent in train_sents:
            for word in sent:
                hidden_state_id = hidden_state_to_hidden_state_id_map[word[HIDDEN_STATE_IDX]]
                obs_id = obs_to_obs_id_map[word[OBS_IDX]]
                emission_count_mat[hidden_state_id][obs_id] += 1
        if DEBUG_DATA == 1:
            print("emission_count_mat[:,:6]={}".format(emission_count_mat[:,:5]))
            print("c2 | emission_count_mat sum = ", np.sum(emission_count_mat, axis=1))

        return emission_count_mat

    @staticmethod
    def calc_emission(emission_count_mat):
        emission_mat = emission_count_mat/np.sum(emission_count_mat,axis=1).reshape(-1,1)
        emission_mat[emission_mat == 0.] = 1e-10
        return emission_mat

def get_pred_size(lst):
    count = 0
    for l in lst:
        count += len(l)
    return count

def iter_predict(data, model, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, HIDDEN_STATE_IDX, OBS_IDX, ACCOUNT_FIRST_TRANSITION):
    if DEBUG == 1:
        print("Iter predicting...")

    X = []
    # lengths = []

    unfound_obs_count = 0
    for i in range(len(data)):
        sent = data[i].copy()
        if ACCOUNT_FIRST_TRANSITION == 1:
            sent.append(('$', '$', '$'))

        seq = []
        for j in range(len(sent)):
            word = sent[j]

            if word[OBS_IDX] in obs_to_obs_id_map.keys():
                obs_id = obs_to_obs_id_map[word[OBS_IDX]]
            else:
                if ACCOUNT_UNK == 1:
                    obs_id = obs_to_obs_id_map['UNK']
                else:
                    obs_id = random.randint(0, len(obs_to_obs_id_map.keys())-1)
                unfound_obs_count += 1
            seq.append(obs_id)
        X.append(seq)
        # lengths.append(len(sent))

    if DEBUG == 1:
        print("Unseen observation from data_to_pred=", unfound_obs_count)

    fit_count = 0
    print(np.sum(model.emissionprob_, axis=1))
    for seq in X:
        seq = np.array([seq])
        # seq = seq.reshape((seq.shape[0], 1))
        model = model.fit(seq, [len(seq)])
        print("fit count={}".format(fit_count))
        print("model emission")
        print(np.sum(model.emissionprob_, axis=1))
        fit_count += 1

    logprob, hidden_state_id_preds = model.decode(X, algorithm="viterbi")  # alice_hears -> hidden_state

    # Process Prediction
    # print("len(seq)={}, n_obs={}".format(len(seq), n_obs))
    print("hidden_state_id_preds shape = ", hidden_state_id_preds.shape)

    # raw_preds = np.split(raw_preds, len(seq))
    # print(hidden_state_id_preds)
    # print("raw_preds shape. rows = {}, num_elem_per_row = {}, {} ".format(len(raw_preds), len(raw_preds[0]), len(raw_preds[-1])))
    
    pred_arr = []
    for hidden_state_id_pred in hidden_state_id_preds:
        hidden_state = hidden_state_id_to_hidden_state_map[hidden_state_id_pred]
        pred_arr.append(hidden_state)

    final_pred1 = []
    final_pred2 = []
    final_pred2_nosplit = []

    i = 0
    for l in lengths:
        pred_seq = pred_arr[i:i+l]
        final_pred1.append(pred_seq)
        final_pred2.append(pred_seq[1:])
        if ACCOUNT_FIRST_TRANSITION == 1:
            final_pred2_nosplit += pred_seq[1:]
        else:
            final_pred2_nosplit += pred_seq
        i = i + l

    return pred_arr, final_pred1, final_pred2, final_pred2_nosplit



def batch_predict(data, model, n_obs, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, HIDDEN_STATE_IDX, OBS_IDX, ACCOUNT_FIRST_TRANSITION):
    if DEBUG == 1:
        print("Batch predicting...")

    X = []
    lengths = []

    unfound_obs_count = 0
    for i in range(len(data)):
        sent = data[i].copy()
        if ACCOUNT_FIRST_TRANSITION == 1:
            sent.append(('$', '$', '$'))

        for j in range(len(sent)):
            word = sent[j]

            if word[OBS_IDX] in obs_to_obs_id_map.keys():
                obs_id = obs_to_obs_id_map[word[OBS_IDX]]
            else:
                if ACCOUNT_UNK == 1:
                    obs_id = obs_to_obs_id_map['UNK']
                else:
                    obs_id = random.randint(0, len(obs_to_obs_id_map.keys())-1)
                unfound_obs_count += 1
            X.append(obs_id)

        lengths.append(len(sent))

    if DEBUG == 1:
        print("Unseen observation from data_to_pred=", unfound_obs_count)

    X = np.array(X)
    X = X.reshape((X.shape[0], 1))

    print("X shape={}".format(X.shape))
    
    # Fit and Predict

    # print("before model.fit | transmat_ sum(axis=1)=", model.transmat_.sum(axis=1))
    # model = model.fit(X, lengths)
    # print("after model.fit | transmat_ sum(axis=1)=", model.transmat_.sum(axis=1))

    logprob, hidden_state_id_preds = model.decode(X, lengths)  # alice_hears -> hidden_state

    # Process Prediction
    # print("len(seq)={}, n_obs={}".format(len(seq), n_obs))
    print("hidden_state_id_preds shape = ", hidden_state_id_preds.shape)

    # raw_preds = np.split(raw_preds, len(seq))
    # print(hidden_state_id_preds)
    # print("raw_preds shape. rows = {}, num_elem_per_row = {}, {} ".format(len(raw_preds), len(raw_preds[0]), len(raw_preds[-1])))
    
    pred_arr = []
    for hidden_state_id_pred in hidden_state_id_preds:
        hidden_state = hidden_state_id_to_hidden_state_map[hidden_state_id_pred]
        pred_arr.append(hidden_state)

    final_pred1 = []
    final_pred2 = []
    final_pred2_nosplit = []

    i = 0
    for l in lengths:
        pred_seq = pred_arr[i:i+l]
        final_pred1.append(pred_seq)
        final_pred2.append(pred_seq[1:])
        if ACCOUNT_FIRST_TRANSITION == 1:
            final_pred2_nosplit += pred_seq[1:]
        else:
            final_pred2_nosplit += pred_seq
        i = i + l

    return pred_arr, final_pred1, final_pred2, final_pred2_nosplit

def get_model(start_prob_mat, trans_mat, emission_mat, n_hidden_state):
    '''
    params (string, optional) – Controls which parameters are updated in the training process. Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, and other characters for subclass-specific emission parameters. Defaults to all parameters.
    init_params (string, optional) – Controls which parameters are initialized prior to training. Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, and other characters for subclass-specific emission parameters. Defaults to all parameters.
    '''
    model = hmm.MultinomialHMM(n_components = n_hidden_state)
    model.startprob_ = start_prob_mat
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_mat
    return model

def write_to_results_txt(data_to_pred, preds):
    j = 0
    if DEBUG == 1:
        print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in data_to_pred:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = preds[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

def write_out(filename, datas):
    with open(filename, "w") as out:
        out.write("{}".format(datas))

def main():
    # A: Load the training data
    # [[('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...], ...]
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb")) 

    # B: Count number of hidden_state and observations, as well as generate mapping
    hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, obs_id_to_obs_map, n_hidden_state, n_obs = CALC_N_HIDDEN_STATE_OBS.gen_hidden_state_and_obs_ds(train_sents)
    print("n_obs={}, n_hidden={}".format(n_obs, n_hidden_state))

    # C: Generate start_prob and transition count matrix
    start_prob_mat, trans_mat = CALC_START_TRANS_MAT.gen_start_and_transition_mat(train_sents, n_hidden_state, hidden_state_to_hidden_state_id_map)

    # D: Generate emission mat
    emission_mat = CALC_EMISSION_MAT.gen_emission_mat(train_sents, n_hidden_state, n_obs, hidden_state_to_hidden_state_id_map, obs_to_obs_id_map)

    print("emission_mat shape = {}".format(emission_mat.shape))

    write_out("startprob_hmm5.txt", start_prob_mat)
    write_out("transmat_hmm5.txt", trans_mat)
    write_out("emissionprob_hmm5.txt", emission_mat)

    # E: train/instantiate model
    model = get_model(start_prob_mat, trans_mat, emission_mat, n_hidden_state)

    # F: Prediction
    data_to_pred = test_sents
    print("dev_sents shape (b4) = ", get_pred_size(data_to_pred))

    pred_arr, final_pred1, final_pred2, final_pred2_nosplit = batch_predict(data_to_pred, model, n_obs, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, HIDDEN_STATE_IDX, OBS_IDX, ACCOUNT_FIRST_TRANSITION)
    # pred_arr, final_pred1, final_pred2, final_pred2_nosplit = iter_predict(data_to_pred, model, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, HIDDEN_STATE_IDX, OBS_IDX, ACCOUNT_FIRST_TRANSITION)


    print("pred_arr shape = ", len(pred_arr))
    print("final_pred1 shape = ", get_pred_size(final_pred1))
    print("final_pred2 shape = ", get_pred_size(final_pred2))
    print("final_pred2_nosplit shape = ", len(final_pred2_nosplit))
    print("data_to_pred shape (after) = ", get_pred_size(data_to_pred))

    # G: write to result
    write_to_results_txt(data_to_pred, final_pred2_nosplit)

if __name__ == "__main__":
    HIDDEN_STATE_IDX = 2
    OBS_IDX = 0

    DEBUG_DATA = 0
    DEBUG = 1

    ACCOUNT_FIRST_TRANSITION = 0
    ACCOUNT_UNK = 0
    
    MODEL_INIT_PARAM = 'e'
    MODEL_PARAM = 'ste'
    # MAX_UNK = 2
    
    np.set_printoptions(threshold=np.inf) # uncomment to print entire matrix

    print("MODEL_INIT_PARAM={}, MODEL_PARAM={}, ACCOUNT_FIRST_TRANSITION={}, ACCOUNT_UNK={}".format(MODEL_INIT_PARAM, MODEL_PARAM, ACCOUNT_FIRST_TRANSITION, ACCOUNT_UNK))
    main()

