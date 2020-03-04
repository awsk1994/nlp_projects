import numpy as np
from gensim.models import Word2Vec


def create_word2vec_matrix(data_loader, min_count=20, size=100, window_size=3):
    """

    Arguments
    ---------
    `data_loader`: `list`-like
        A `list` (or `list` like object), each item of which is itself a list of tokens.
        For example:
                
                [
                    ['this', 'is', 'sentence', 'one'],
                    ['this', 'is', 'sentence', 'two.']
                ]

        Note that preprocesisng.DataLoader is exactly this kind of object.
    `min_count`: `int`
        The minimum count that a token has to have to be included in the vocabulary.
    `size`: `int`
        The dimensionality of the word vectors.
    `window_size`: `int`
        The window size. Read the assignment pdf if you don't know what that is.

    Returns
    -------
        `tuple(np.ndarray, dict)`:
            The first element will be an (V, `size`) matrix, where V is the
            resulting vocabulary size.
            
            The second element is a mapping from `int` to `str` of which word
            in the vocabulary corresponds to which index in the matrix. 
    """

    word2vec_mat = None
    word2vec_id_to_tokens = None
    # NOTE : Your code should not be longer than ~2 lines, excepting the part
    # where build the int to string token mapping(and even that can be a one-liner).
    ####### Your code here ###############

    ####### End of your code #############

    return word2vec_mat, word2vec_id_to_tokens


def create_term_newsgroup_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    tf_idf_weighing=False,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:
                
                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.
        
        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.
            
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        tf_idf_weighing: `bool`
            Whether to use TF IDF weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_newsgroups)).
            That is it will be a VxD matrix where V is vocabulary size and D is number of newsgroups.

            Note, you may choose to remove the row corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    D = len(id_to_newsgroups)

    mat = np.zeros((V, D), dtype='int')
    ####### Your code here ###############
    for newsgroup_and_token_ids in newsgroup_and_token_ids_per_post:
        token_ids, newsgroup_id = newsgroup_and_token_ids
        for token_id in token_ids:
            mat[token_id, newsgroup_id] += 1

    ####### End of your code #############

    return mat


def create_term_context_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    ppmi_weighing=False,
    window_size=5,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` is a post, contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:
                
                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.
        
        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.
            
                { 0: "hi", 1: "hello  ...}
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        ppmi_weighing: `bool`
            Whether to use PPMI weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_tokens)).
            That is it will be a VxV matrix where V is vocabulary size.

            Note, you may choose to remove the row/column corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    mat = None
    ####### Your code here ###############
    pass
    ####### End of your code #############
    return mat


def compute_cosine_similarity(a, B):
    """Cosine similarity.

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The cosine similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    
    def cosine_sim(a,b):
        dot = np.dot(a,b)
        normalize_a = np.linalg.norm(a)
        normalize_b = np.linalg.norm(b)
        sim = dot / (normalize_a * normalize_b)
        return sim

    ans = np.zeros((B.shape[0]))
    for b_idx in range(B.shape[0]):
        b = B[b_idx]    # len(b) = M ()
        ans[b_idx] = cosine_sim(a,b) # result shape = (N)

    return ans
    ####### End of your code #############


def compute_jaccard_similarity(a, B):
    """

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Jaccard similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    pass
    ####### End of your code #############


def compute_dice_similarity(a, B):
    """

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Dice similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    pass
    ####### End of your code #############

