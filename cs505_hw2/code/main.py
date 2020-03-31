import numpy as np

import vec_spaces
from preprocessing import read_processed_data, DataLoader

WORDS_OF_INTEREST = ["lethal", "handgun", "money", "engineer", "rich", "republican"]

# Helper Functions
def get_words_of_interest():
    return WORDS_OF_INTEREST

def get_second_largest_count(similarities):
    # Get second largest in array. Sort -> Get second largest.
    largest_count, largest_idx = None, None
    second_largest_count, second_largest_count_idx = None, None
    for sim_idx in range(similarities.shape[0]):
        sim = similarities[sim_idx]
        if largest_count is None or sim > largest_count:
            largest_count = sim
            largest_idx = sim_idx
        elif second_largest_count is None or sim > second_largest_count:
            second_largest_count = sim
            second_largest_count_idx = sim_idx

    return second_largest_count, second_largest_count_idx # because, first largest is itself.

def get_token_id(target, word2vec_id_to_tokens):
    for key in word2vec_id_to_tokens.keys():
        if word2vec_id_to_tokens[key] == target:
            return key
    print("ERROR - cannot find target({})".format(target))
    return None

def get_similarities_sorted(similarities, id_to_tokens):
    word_sim_val_hash = {}
    for sim_id in range(len(similarities)):
        sim_word = "{}(similarity={})".format(id_to_tokens[sim_id], similarities[sim_id])
        word_sim_val_hash[sim_word] = similarities[sim_id]

    similarities_sorted = [word for word in sorted(word_sim_val_hash, key=word_sim_val_hash.get, reverse=True)][1:] # remove first because it's going to be self.
    return similarities_sorted

def get_top_last_n_similarities(similarities_sorted, top_n=10):
    top_n_sim = similarities_sorted[:top_n]
    last_n_sim = list(reversed(similarities_sorted[(-1 * top_n):]))
    return top_n_sim, last_n_sim

# Test Functions
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

    '''
    Score how similar each newsgroup is to compared to other newsgroups. 
    Print out the newsgroup that yields the highest similarity score for each newsgroup.
    '''

    ####### Your code here ###############
    most_sims_hash = {}
    newsgroup_token_mat = np.swapaxes(mat, 0, 1)  # to change from (V,D) to (D,V), since output of compute_cosine_similarity has to be array of D
    top_n = 3

    for newsgroup_idx in range(len(id_to_newsgroups.keys())):
        similarities = sim_func(newsgroup_token_mat[newsgroup_idx], newsgroup_token_mat)    # apply similarity
        similarities_sorted = get_similarities_sorted(similarities, id_to_newsgroups)
        top_n_sim, last_n_sim = get_top_last_n_similarities(similarities_sorted, top_n = top_n)

        print("test_word_similarity | Top {} similar words for {} is {}".format(top_n, id_to_newsgroups[newsgroup_idx], top_n_sim))
        print("test_word_similarity | Last {} similar words for {} is {}".format(top_n, id_to_newsgroups[newsgroup_idx], last_n_sim))

        # largest_sim_value, largest_sim_idx = get_second_largest_count(similarities)         # get second largest (because first largest is self)

        # # Save results
        # newsgroup_name = id_to_newsgroups[newsgroup_idx]
        # most_sims_hash[newsgroup_name] = "{}(id={}, similarity={})".format(newsgroup_name, largest_sim_idx, largest_sim_value)

    # Print results
    # print("test_newsgroup_similarity | newsgroup that yields the highest similarity score for each newsgroup = ", most_sims_hash)
    
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
    top_n = 10
    words_of_interest = get_words_of_interest()
    for word_of_interest in words_of_interest:
        word_of_interest_id = get_token_id(word_of_interest, id_to_tokens)
        similarities = sim_func(mat[word_of_interest_id], mat)

        similarities_sorted = get_similarities_sorted(similarities, id_to_tokens)
        top_n_sim, last_n_sim = get_top_last_n_similarities(similarities_sorted)

        print("test_word_similarity | Top {} similar words for {} is {}".format(top_n, word_of_interest, top_n_sim))
        print("test_word_similarity | Last {} similar words for {} is {}".format(top_n, word_of_interest, last_n_sim))

        # largest_sim_value, largest_sim_idx = get_second_largest_count(similarities)
        # largest_sim_word = id_to_tokens[largest_sim_idx]

        # print("test_word_similarity | {}(id={}) is closest to {}(id={}) with similarity={})".format(
        #     word_of_interest, word_of_interest_id, largest_sim_word, largest_sim_idx, largest_sim_value))

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

    words_of_interest = get_words_of_interest()
    top_n = 10

    for word_of_interest in words_of_interest:
        word_of_interest_id = get_token_id(word_of_interest, id_to_tokens)
        similarities = sim_func(mat[word_of_interest_id], mat)    # mat = (vocab x dimension), sims = (vocab)

        similarities_sorted = get_similarities_sorted(similarities, id_to_tokens)
        top_n_sim, last_n_sim = get_top_last_n_similarities(similarities_sorted)

        print("test_word2vec_similarity | Top {} similar words for {} is {}".format(top_n, word_of_interest, top_n_sim))
        print("test_word2vec_similarity | Last {} similar words for {} is {}".format(top_n, word_of_interest, last_n_sim))

    # max_token_len = 300  # first 50 (TODO: improve this later)
    # # max_token_len = mat.shape[0]
    # print("test_word2vec_similarity | size = {}".format(max_token_len))
    # tokens_len = max_token_len # len(id_to_tokens.keys())
    # mat = mat[:max_token_len]
    # sim_hash = {}

    # for token_id in range(max_token_len):
    #     # sim = np.zeros((mat.shape[0]))
    #     similarities = sim_func(mat[token_id], mat)    # mat = (vocab x dimension), sims = (vocab)
        
    #     if token_id == 0:
    #         print("test_word2vec_similarity | token_id = {} ({}), similarities = {}".format(token_id, id_to_tokens[token_id], list(reversed(sorted(similarities)))[:10]))
    #         print("test_word2vec_similarity | check negative values | similaritity = {}".format(list(sorted(similarities))[:10]))
    #     if token_id % 100 == 0:
    #         print("token_id = {}".format(token_id))

    #     largest_sim_value, largest_sim_idx = get_second_largest_count(similarities)

    #     curr_token = id_to_tokens[token_id]
    #     sim_hash[curr_token] = largest_sim_value
    # print("test_word2vec_similarity | tokens that yields the highest similarity score for each tokens = ", sim_hash)

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
    print("\n2. Created term_newsgroup_mat")
    token_count_per_newsgroup = np.sum(term_newsgroup_mat, axis=0)
    for i in range(token_count_per_newsgroup.shape[0]):
        print("newsgroup({}) has count of {}.".format(id_to_newsgroups[i], token_count_per_newsgroup[i]))

    # Create term_context_mat
    term_context_mat = vec_spaces.create_term_context_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups, ppmi_weighing=False)
    print("\n3. Created term_context_mat")

    # Newsgroup similarities
    print("\n4. Test Newsgroup similarities (cosine)")
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups, sim_func = vec_spaces.compute_cosine_similarity)        

    # Context similarities
    print("\n5. Test Word similarities (cosine)")
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)

    # Context similarities
    print("\n6. Test Context similarities (cosine)")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)

    # word2vec
    print("\n7. Test Word2Vec")
    dataset = list(DataLoader())
    list_of_list = [data[0] for data in dataset]
    word2vec_mat, word2vec_id_to_tokens = vec_spaces.create_word2vec_matrix(list_of_list)
    test_word2vec_similarity(word2vec_mat, word2vec_id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity)

    print("\n8. Compare sim functions")
    print("\n8a. Test Newsgroup similarities.")
    print("\nJaccard")
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups, sim_func = vec_spaces.compute_jaccard_similarity)        # Jaccard
    print("\nCosine")
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine
    print("\nDice")
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups, sim_func = vec_spaces.compute_dice_similarity)        # Dice

    print("\n8a. Test Word similarities.")
    print("\nJaccard")
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_jaccard_similarity)        # Jaccard
    print("\nCosine")
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine
    print("\nDice")
    test_word_similarity(term_newsgroup_mat, id_to_tokens, sim_func = vec_spaces.compute_dice_similarity)        # Dice

    print("\n8c. Test Context similarities.")
    print("\nJaccard")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_jaccard_similarity)          # context similarity
    print("\nCosine")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine
    print("\nDice")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_dice_similarity)        # Dice

    print("\n9. Interesting Findings")
    print("\n9a. newsgroup counts")
    token_count_per_newsgroup = np.sum(term_newsgroup_mat, axis=0)
    for i in range(token_count_per_newsgroup.shape[0]):
        print("newsgroup({}) has count of {}.".format(id_to_newsgroups[i], token_count_per_newsgroup[i]))

    print("\n9b. non-tf-idf vs tf-idf (newsgroup)")
    print("\nnon-tf-idf")
    test_newsgroup_similarity(term_newsgroup_mat, id_to_newsgroups, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine
    print("\ntf-idf")
    term_newsgroup_mat_tfidf = vec_spaces.create_term_newsgroup_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups, tf_idf_weighing=True)
    test_newsgroup_similarity(term_newsgroup_mat_tfidf, id_to_newsgroups, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine

    print("\n9c. non-ppmi vs ppmi (word context)")
    print("\nnon-ppmi")
    test_word_similarity(term_context_mat, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine
    print("\nppmi")
    term_context_mat_ppmi = vec_spaces.create_term_context_matrix(newsgroup_and_token_ids_per_post, id_to_tokens, id_to_newsgroups, ppmi_weighing=True)
    test_word_similarity(term_context_mat_ppmi, id_to_tokens, sim_func = vec_spaces.compute_cosine_similarity)        # Cosine

    print("\n\n\nProgram Ended.")

    # dataset = list(DataLoader())
    # list_of_list = [data[0] for data in dataset]
    # word2vec_mat, word2vec_id_to_tokens = vec_spaces.create_word2vec_matrix(list_of_list)

    # target = "handgun"
    # sims = test_word2vec_similarity_taget(target, word2vec_mat, word2vec_id_to_tokens)

    ####### End of your code #############

if __name__ == "__main__":
    main()





# Archive


# def test_word_similarity_entire(
#     mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
# ):
#     """

#     Arguments
#     ---------
#         mat: `np.ndarray` (V, d)
#             Each row is a d-dimensional word vector.

#         id_to_tokens: `dict`
#             `int` token ids as keys and `str` tokens as values.
            
#                 { 0: "hi", 1: "hello"  ...}

#         sim_func: `function(array(M,), array(N,M))->array(N,)`
#             A function that returns the similarity of its first arg to every row
#             in it's second arg.
#     """

#     ####### Your code here ###############
#     max_token_len =300  # first 50 (TODO: improve this later)
#     # max_token_len = mat.shape[0]
#     print("test_word_similarity | size = {}".format(max_token_len))
#     tokens_len = max_token_len # len(id_to_tokens.keys())
#     most_sims = np.zeros(tokens_len, dtype='int')
#     most_sims_hash = {}
#     token_newsgroup_mat = mat[:max_token_len]

#     for token_idx in range(tokens_len):
#         a = token_newsgroup_mat[token_idx]
#         similarities = sim_func(a, token_newsgroup_mat)

#         if token_idx == 0:
#             print("test_word_similarity | first token({}) | similaritity = {}".format(id_to_tokens[token_idx], list(reversed(sorted(similarities)))[:10]))
#             print("test_word_similarity | check negative values | similaritity = {}".format(list(sorted(similarities))[:10]))
#         if token_idx % 100 == 0:
#             print("token_idx = {}".format(token_idx))

#         largest_sim, largest_sim_idx = get_second_largest_count(similarities)

#         most_sims[token_idx] = largest_sim_idx
#         most_sims_hash[id_to_tokens[token_idx]] = id_to_tokens[largest_sim_idx]
#     print("test_word_similarity | tokens that yields the highest similarity score for each tokens = ", most_sims)
#     print("test_word_similarity | tokens that yields the highest similarity score for each tokens = ", most_sims_hash)

#     ####### End of your code #############

# def test_word2vec_similarity_entire(
#     mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
# ):
#     """
#     Arguments
#     ---------
#         mat: `np.ndarray` (V, d)
#             Each row is a d-dimensional word vector.

#         id_to_tokens: `dict`
#             `int` token ids as keys and `str` tokens as values.
            
#                 { 0: "hi", 1: "hello"  ...}

#         sim_func: `function(array(M,), array(N,M))->array(N,)`
#             A function that returns the similarity of its first arg to every row
#             in it's second arg.
#     """
#     ####### Your code here ###############

#     max_token_len = 300  # first 50 (TODO: improve this later)
#     # max_token_len = mat.shape[0]
#     print("test_word2vec_similarity | size = {}".format(max_token_len))
#     tokens_len = max_token_len # len(id_to_tokens.keys())
#     mat = mat[:max_token_len]
#     sim_hash = {}

#     for token_id in range(max_token_len):
#         # sim = np.zeros((mat.shape[0]))
#         similarities = sim_func(mat[token_id], mat)    # mat = (vocab x dimension), sims = (vocab)
        
#         if token_id == 0:
#             print("test_word2vec_similarity | token_id = {} ({}), similarities = {}".format(token_id, id_to_tokens[token_id], list(reversed(sorted(similarities)))[:10]))
#             print("test_word2vec_similarity | check negative values | similaritity = {}".format(list(sorted(similarities))[:10]))
#         if token_id % 100 == 0:
#             print("token_id = {}".format(token_id))

#         largest_sim_value, largest_sim_idx = get_second_largest_count(similarities)

#         curr_token = id_to_tokens[token_id]
#         sim_hash[curr_token] = largest_sim_value
#     print("test_word2vec_similarity | tokens that yields the highest similarity score for each tokens = ", sim_hash)

# def test_word2vec_similarity_target(
#     target, mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
# ):
#     """
#     Arguments
#     ---------
#         mat: `np.ndarray` (V, d)
#             Each row is a d-dimensional word vector.

#         id_to_tokens: `dict`
#             `int` token ids as keys and `str` tokens as values.
            
#                 { 0: "hi", 1: "hello"  ...}

#         sim_func: `function(array(M,), array(N,M))->array(N,)`
#             A function that returns the similarity of its first arg to every row
#             in it's second arg.
#     """
#     ####### Your code here ###############

#     max_token_len = mat.shape[0]
#     print("test_word2vec_similarity_taget | size = {}".format(max_token_len))
#     tokens_len = max_token_len # len(id_to_tokens.keys())
#     mat = mat[:max_token_len]
#     sim_hash = {}

#     top_n = 100
#     target_id = get_token_id(target, id_to_tokens)
#     print("test_word2vec_similarity_taget | target={}, target_id = {}".format(target, target_id))

#     # Apply sim func (cosine)
#     similarities = sim_func(mat[target_id], mat)    # mat = (vocab x dimension), sims = (vocab)
    

#     for sim_id in range(len(similarities)):
#         sim_hash[id_to_tokens[sim_id]] = similarities[sim_id]

#     similarities_sorted = [w for w in sorted(sim_hash, key=sim_hash.get, reverse=True)]

#     print("test_word2vec_similarity_taget | Top {} similar words for {} => {}".format(top_n, target, similarities_sorted[1:(1+top_n)]))
#     print("test_word2vec_similarity_taget | Top {} similar word sim for {} => {}".format(top_n, target, list(reversed(sorted(similarities[1:(1+top_n)]))))) # todo: confirm same thing

#     return similarities_sorted
#     ####### End of your code #############


# def manual_debug_test_word2vec_similarity(
#     mat, id_to_tokens
# ):
#     """
#     Arguments
#     ---------
#         mat: `np.ndarray` (V, d)
#             Each row is a d-dimensional word vector.

#         id_to_tokens: `dict`
#             `int` token ids as keys and `str` tokens as values.
            
#                 { 0: "hi", 1: "hello"  ...}

#         sim_func: `function(array(M,), array(N,M))->array(N,)`
#             A function that returns the similarity of its first arg to every row
#             in it's second arg.
#     """
#     ####### Your code here ###############
    
#     def cosine_sim(vec1, vec2):
#         sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#         return (sim)

#     def compare(a,b, word2vec_mat, word2vec_id_to_tokens):
#         dist = cosine_sim(word2vec_mat[get_token_id(a, word2vec_id_to_tokens)], word2vec_mat[get_token_id(b, word2vec_id_to_tokens)])
#         print("compare {} and {} => {}".format(a,b,dist))

#     print("manual_debug_test_word2vec_similarity | Start")
#     compare("talking", "about", mat, id_to_tokens)
#     compare("talking", "bad", mat, id_to_tokens)
#     compare("good", "about", mat, id_to_tokens)
#     compare("bad", "good", mat, id_to_tokens)
#     compare("good", "john", mat, id_to_tokens)
#     compare("buffalo", "university", mat, id_to_tokens)
#     print("manual_debug_test_word2vec_similarity | Done")

#     ####### End of your code #############
