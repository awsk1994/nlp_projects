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

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############

    newsgroup_len = len(id_to_newsgroups.keys())
    most_sims = np.zeros(newsgroup_len, dtype='int')
    newsgroup_token_mat = np.swapaxes(mat, 0, 1)  # to change from (V,D) to (D,V)

    for newsgroup_idx in range(newsgroup_len):
        a = newsgroup_token_mat[newsgroup_idx]
        similarities = vec_spaces.compute_cosine_similarity(a, newsgroup_token_mat)

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
    print("newsgroup that yields the highest similarity score for each newsgroup = ", most_sims)
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
    max_token_len = 50  # first 50 (TODO: improve this later)
    tokens_len = max_token_len # len(id_to_tokens.keys())
    most_sims = np.zeros(tokens_len, dtype='int')
    token_newsgroup_mat = mat[:max_token_len]

    for token_idx in range(tokens_len):
        a = token_newsgroup_mat[token_idx]
        similarities = vec_spaces.compute_cosine_similarity(a, token_newsgroup_mat)

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
    print("tokens that yields the highest similarity score for each tokens = ", most_sims)

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
    pass
    ####### End of your code #############


def main():
    ####### Your code here ###############
    # Read Data
    data = read_processed_data()
    newsgroup_and_token_ids_per_post = data['newsgroup_and_token_ids_per_post']
    id_to_tokens = data['id_to_tokens']
    id_to_newsgroups = data['id_to_newsgroups']

    # Create term_newsgroup_mat
    term_newsgroup_mat = vec_spaces.create_term_newsgroup_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups)

    # Test newsgroup similarities
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups)


    # Test word similarities
    test_word_similarity(term_newsgroup_mat, id_to_tokens)

    ####### End of your code #############


if __name__ == "__main__":
    main()

