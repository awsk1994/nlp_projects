import numpy as np

import vec_spaces
from preprocessing import read_processed_data, DataLoader


def test_newsgroup_similarity(
    mat, id_to_newsgroups, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, D)
            Each column is "newsgroup" vector. 

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        sim_func: `function(array(M,), array(N,M))->array(N,)
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############

    newsgroup_len = len(id_to_newsgroups.keys())
    most_sims = np.zeros(newsgroup_len, dtype='int')
    newsgroup_token_mat = np.swapaxes(mat, 0, 1)  # to change from (V,D) to (D,V), since output of compute_cosine_similarity has to be array of D

    for newsgroup_idx in range(newsgroup_len):
        a = newsgroup_token_mat[newsgroup_idx]
        similarities = sim_func(a, newsgroup_token_mat)

        # Get second largest newsgroup
        largest, largest_idx = None, None
        second_largest, second_largest_idx = None, None
        for sim_idx in range(similarities.shape[0]):
            sim = similarities[sim_idx]
            if largest is None or sim > largest:
                largest = sim
                largest_idx = sim_idx
            elif second_largest is None or sim > second_largest:
                second_largest = sim
                second_largest_idx = sim_idx

        largest_sim, largest_sim_idx = second_largest, second_largest_idx # because, first largest is itself.
        # print("test_newsgroup_similarity | largest_sim={}, idx={}".format(largest_sim, largest_sim_idx))
        most_sims[newsgroup_idx] = largest_sim_idx
    print("test_newsgroup_similarity | newsgroup that yields the highest similarity score for each newsgroup = ", most_sims)
    ####### End of your code #############

def test_word_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############
    max_token_len = 100  # first 50 (TODO: improve this later)
    tokens_len = max_token_len # len(id_to_tokens.keys())
    most_sims = np.zeros(tokens_len, dtype='int')
    token_newsgroup_mat = mat[:max_token_len]

    for token_idx in range(tokens_len):
        a = token_newsgroup_mat[token_idx]
        similarities = sim_func(a, token_newsgroup_mat)

        # Get second largest newsgroup
        largest, largest_idx = None, None
        second_largest, second_largest_idx = None, None
        for sim_idx in range(similarities.shape[0]):
            sim = similarities[sim_idx]
            if largest is None or sim > largest:
                largest = sim
                largest_idx = sim_idx
            elif second_largest is None or sim > second_largest:
                second_largest = sim
                second_largest_idx = sim_idx

        largest_sim, largest_sim_idx = second_largest, second_largest_idx # because, first largest is itself.
        # print("test_newsgroup_similarity | largest_sim={}, idx={}".format(largest_sim, largest_sim_idx))
        most_sims[token_idx] = largest_sim_idx
    print("test_word_similarity | tokens that yields the highest similarity score for each tokens = ", most_sims)
    ####### End of your code #############

def test_word2vec_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """
    ####### Your code here ###############

    max_token_len = 100  # first 50 (TODO: improve this later)
    tokens_len = max_token_len # len(id_to_tokens.keys())
    mat = mat[:max_token_len]
    sim_hash = {}

    for token_id in range(mat.shape[0]):
        # sim = np.zeros((mat.shape[0]))
        similarities = sim_func(mat[token_id], mat)    # mat = (vocab x dimension), sims = (vocab)
        
        if token_id == 0:
            print("test_word2vec_similarity | token_id = {} ({}), similarities = {}".format(token_id, id_to_tokens[token_id], similarities))

        # Get second largest newsgroup
        largest, largest_idx = None, None
        second_largest, second_largest_idx = None, None
        for sim_idx in range(similarities.shape[0]):
            sim = id_to_tokens[sim_idx]
            if largest is None or sim > largest:
                largest = sim
                largest_idx = sim_idx
            elif second_largest is None or sim > second_largest:
                second_largest = sim
                second_largest_idx = sim_idx
        largest_sim, largest_sim_idx = second_largest, second_largest_idx # because, first largest is itself.

        curr_token = id_to_tokens[token_id]
        sim_hash[curr_token] = largest_sim
    print("test_word2vec_similarity | tokens that yields the highest similarity score for each tokens = ", sim_hash)

    ####### End of your code #############


def manual_debug_test_word2vec_similarity(
    mat, id_to_tokens
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """
    ####### Your code here ###############
    
    def cosine_sim(vec1, vec2):
        sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        return (sim)

    def get_id(target, word2vec_id_to_tokens):
        for key in word2vec_id_to_tokens.keys():
            if word2vec_id_to_tokens[key] == target:
                print("{}, {}".format(target, key))
                return key
        print("ERROR - cannot find target({})".format(target))

    def compare(a,b, word2vec_mat, word2vec_id_to_tokens):
        dist = cosine_sim(word2vec_mat[get_id(a, word2vec_id_to_tokens)], word2vec_mat[get_id(b, word2vec_id_to_tokens)])
        print("compare {} and {} => {}".format(a,b,dist))

    print("manual_debug_test_word2vec_similarity | Start")
    compare("talking", "about", mat, id_to_tokens)
    compare("talking", "bad", mat, id_to_tokens)
    compare("good", "about", mat, id_to_tokens)
    compare("bad", "good", mat, id_to_tokens)
    compare("good", "john", mat, id_to_tokens)
    compare("buffalo", "university", mat, id_to_tokens)
    print("manual_debug_test_word2vec_similarity | Done")

    ####### End of your code #############

def main():
    ####### Your code here ###############
    # Read Data
    data = read_processed_data()
    print("1. Data loaded.")

    newsgroup_and_token_ids_per_post = data['newsgroup_and_token_ids_per_post']
    id_to_tokens = data['id_to_tokens']
    id_to_newsgroups = data['id_to_newsgroups']

    # Create term_newsgroup_mat
    term_newsgroup_mat = vec_spaces.create_term_newsgroup_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups)
    print("2. Created term_newsgroup_mat")

    # Create term_context_mat
    term_context_mat = vec_spaces.create_term_context_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups, ppmi_weighing=False, window_size=5)
    print("3. Created term_context_mat")

    # Newsgroup similarities
    print("4. Test Newsgroup similarities.")
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_jaccard_similarity)        # Jaccard
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_dice_similarity)        # Dice
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine

    # Context similarities
    print("5. Test Context similarities.")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_jaccard_similarity)          # context similarity
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_dice_similarity)        # Dice
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine

    # word2vec
    print("6. Test Word2Vec")
    dataset = list(DataLoader())
    list_of_list = [data[0] for data in dataset]
    word2vec_mat, word2vec_id_to_tokens = vec_spaces.create_word2vec_matrix(list_of_list)
    manual_debug_test_word2vec_similarity(word2vec_mat, word2vec_id_to_tokens)
    test_word2vec_similarity(word2vec_mat, word2vec_id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity)

    print("Program Ended.")

    ####### End of your code #############


if __name__ == "__main__":
    main()

