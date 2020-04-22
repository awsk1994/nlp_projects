from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
import hmmlearn.hmm as hmm
import numpy as np
import random

# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

class CALC_N_HIDDEN_STATE_OBS:
    @staticmethod
    def gen_hidden_state_and_obs_count_and_mapping(train_sents):
        if DEBUG == 1:
            print("Generating hidden state and observation data structures...")
        hidden_state_count_hash, obs_count_hash, n_hidden_state, n_obs = CALC_N_HIDDEN_STATE_OBS.count_hidden_state_and_obs(train_sents)
        hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map = CALC_N_HIDDEN_STATE_OBS.gen_obj_id_mapping(hidden_state_count_hash)
        obs_to_obs_id_map, obs_id_to_obs_map = CALC_N_HIDDEN_STATE_OBS.gen_obj_id_mapping(obs_count_hash)
        return hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, obs_id_to_obs_map, n_hidden_state, n_obs

    @staticmethod
    def count_hidden_state_and_obs(train_sents):
        hidden_state_count_hash = {}
        obs_count_hash = {}

        for sent in train_sents:    # TODO: can do with set, no need hash
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
            print("Hidden states len = {}, Observations len = {}".format(len(hidden_state_count_hash.keys()), len(obs_count_hash.keys())))

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
            print("Generating start_prob and transition count matrix...")

        trans_count_mat = np.zeros((n_hidden_state, n_hidden_state))
        start_prob_count_mat = np.zeros((n_hidden_state))

        for sent in train_sents:
            prev_hidden_state_id = None

            # Aggregate start_prob (only on first word of the sentence/phrase)
            first_hidden_state_id = hidden_state_to_hidden_state_id_map[sent[0][HIDDEN_STATE_IDX]]
            start_prob_count_mat[first_hidden_state_id] += 1

            for word in sent:
                hidden_state_id = hidden_state_to_hidden_state_id_map[word[HIDDEN_STATE_IDX]]
                if prev_hidden_state_id != None:
                    trans_count_mat[prev_hidden_state_id][hidden_state_id] += 1
                prev_hidden_state_id = hidden_state_id

        if DEBUG_DATA == 1:
            print("start_prob_count_mat={}".format(start_prob_count_mat))
            print("trans_count_mat={}".format(trans_count_mat))

        return start_prob_count_mat, trans_count_mat

    @staticmethod
    def calc_start_trans_prob(start_prob_count_mat, trans_count_mat):
        # Generate probability version of the count matrix
        if DEBUG == 1:
            print("Generating probability version of the count matrix...")

        start_prob_mat = start_prob_count_mat / np.sum(start_prob_count_mat) 
        start_prob_mat[start_prob_mat == 0.] = 1e-10

        trans_mat = trans_count_mat/np.sum(trans_count_mat,axis=1).reshape(-1,1)
        trans_mat[trans_mat == 0.] = 1e-10

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
            print("emission_count_mat sum = ", np.sum(emission_count_mat, axis=1))

        return emission_count_mat

    @staticmethod
    def calc_emission(emission_count_mat):
        emission_mat = emission_count_mat/np.sum(emission_count_mat,axis=1).reshape(-1,1)
        emission_mat[emission_mat == 0.] = 1e-10
        return emission_mat

class MODEL:
    @staticmethod
    def batch_predict(data, model, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map):
        if DEBUG == 1:
            print("Batch predicting...")

        X = []
        lengths = []

        # Preprocess test data
        unfound_obs_count = 0
        for sent in data:
            lengths.append(len(sent))
            for j in range(len(sent)):
                word = sent[j]
                if word[OBS_IDX] in obs_to_obs_id_map.keys():
                    obs_id = obs_to_obs_id_map[word[OBS_IDX]]
                else:
                    obs_id = random.randint(0, len(obs_to_obs_id_map.keys())-1)
                    unfound_obs_count += 1
                X.append(obs_id)
        X = np.array(X).reshape((len(X), 1))
        # X = X.reshape((X.shape[0], 1))

        if DEBUG == 1:
            print("Unseen observation from test_data=", unfound_obs_count)
        
        # Predict
        _, hidden_state_id_preds = model.decode(X, lengths)  # alice_hears -> hidden_state
        
        pred_arr = []
        for hidden_state_id_pred in hidden_state_id_preds:
            hidden_state = hidden_state_id_to_hidden_state_map[hidden_state_id_pred]
            pred_arr.append(hidden_state)

        return pred_arr

    @staticmethod
    def get_model(start_prob_mat, trans_mat, emission_mat, n_hidden_state):
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

def main():
    # A: Load the training data # [[('Comisión', 'NC', 'B-ORG'), ('Europea', 'AQ', 'I-ORG'), (',', 'Fc', 'O'), ...], ...]
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb")) 

    # B: Count number of hidden_state and observations, as well as generate mapping
    hidden_state_to_hidden_state_id_map, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map, obs_id_to_obs_map, n_hidden_state, n_obs = CALC_N_HIDDEN_STATE_OBS.gen_hidden_state_and_obs_count_and_mapping(train_sents)
    if DEBUG_DATA == 1:
        print("n_obs={}, n_hidden={}".format(n_obs, n_hidden_state))

    # C: Generate start_prob and transition count matrix
    start_prob_mat, trans_mat = CALC_START_TRANS_MAT.gen_start_and_transition_mat(train_sents, n_hidden_state, hidden_state_to_hidden_state_id_map)

    # D: Generate emission mat
    emission_mat = CALC_EMISSION_MAT.gen_emission_mat(train_sents, n_hidden_state, n_obs, hidden_state_to_hidden_state_id_map, obs_to_obs_id_map)

    # E: train/instantiate model
    model = MODEL.get_model(start_prob_mat, trans_mat, emission_mat, n_hidden_state)

    # F: Prediction
    data_to_pred = test_sents
    pred_arr = MODEL.batch_predict(data_to_pred, model, hidden_state_id_to_hidden_state_map, obs_to_obs_id_map)

    # G: write to result
    write_to_results_txt(data_to_pred, pred_arr)

if __name__ == "__main__":
    HIDDEN_STATE_IDX = 2
    OBS_IDX = 0
    DEBUG_DATA = 0
    DEBUG = 1
    # np.set_printoptions(threshold=np.inf) # uncomment to print entire matrix
    main()
