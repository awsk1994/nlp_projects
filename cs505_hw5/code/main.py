'''
python3 main.py -r 0.1 -L word -f 2 -C 6 -W 2 -train data/train/english_train -test data/test/english_test -mapping data/mapping/english_mapping.txt



ap.add_argument("-r", "--regularizatio-constant", dest="C", type=float, default=1.0)
        ap.add_argument("-L", "--lowercase", dest="lowercase",
                default=None)

        ap.add_argument("-f", "--min-df", dest="min_df", type=int, default=1)

        ap.add_argument("-C", "--max-char-ng", dest="c_ngmax",
                type=int, default=1)

        ap.add_argument("-W", "--max-word-ng", dest="w_ngmax",
                type=int, default=1)

'''

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import os
import re

options = {
	'train_eng_text': './data/train/english_train.text',
	'train_eng_label': './data/train/english_train.labels',
	'test_eng_text': './data/test/english_test.text',
	'test_eng_label': './data/test/english_test.labels',
	'mapping_eng': './data/mapping/english_mapping.txt',

	'train_esp_text': './data/train/spanish_train.text',
	'train_esp_label': './data/train/spanish_train.labels',
	'test_esp_text': './data/test/spanish_test.text',
	'test_esp_label': './data/test/spanish_test.labels',
	'mapping_esp': './data/mapping/spanish_mapping.txt'
}

def identity(x):
    return x

def load_data(text_path, label_path):
	docs = []
	len_char = []
	labels = []

	if os.path.exists(text_path):
		f = open(text_path, "r")
		docs = [line.strip() for line in f]

	if os.path.exists(label_path):
		f = open(label_path, 'r')
		labels = [line.strip() for line in f.readlines()]

	return docs, labels

w_tokenizer = re.compile("\w+|[^ \t\n\r\f\v\w]+")

def get_ngrams(s, ngmin=1, ngmax=1,
                tokenizer=w_tokenizer.findall,
                separator="|",
                bos="<",
                eos=">",
                append="",
                flatten=True):
    """ Return all ngrams with in range ngmin-ngmax.
        The function specified by 'tokenizer' should return a list of
        tokens. By default, the tokenizer splits the string into word
        and non-word non-space characters.
    """
    # print("c1")
    # print(tokenizer)
    ngrams = [[] for x in range(1, ngmax + 1)]
    s = tokenizer(bos + s + eos)
    for i, ch in enumerate(s):
        for ngsize in range(ngmin, ngmax + 1):
            if (i + ngsize) <= len(s):
                ngrams[ngsize - 1].append(separator.join(s[i:i+ngsize]))
    if flatten:
        ngrams = [ng for nglist in ngrams for ng in nglist]
    return ngrams

def doc_to_ngrams2(docs):

    """ Return bag-of-n-grams features for the give document set
    """
    param = {
        'c_ngmax': 6, 
        'c_ngmin': 1, 
        'w_ngmax': 2, 
        'w_ngmin': 1,
        'min_df': 1,
        'sublinear': True,
        'norm': 'l2',
        'max_features': None,
        'input_name': None,
        'lowercase': 'word',
        'dim_reduce': None,
        'cache_dir': '.cache',
        'use_cached': True
    }
    # for k, v in kwargs.items(): param[k] = v

    # if param['input_name'] and use_cached or cache:
    #     os.makedirs(cache_dir, exist_ok=True)
    #     paramstr = ','.join([k + '=' + str(param[k]) for k in sorted(param)])
    #     cachefn = 'vectorizer-' + \
    #             hashlib.sha224(paramstr.encode('utf-8')).hexdigest() + '.z'
    #     cachefn = os.path.join(cache_dir, cachefn)
    # if use_cached and os.path.exists(cachefn):
    #     info('Using cached vectorizer: {}'.format(cachefn))
    #     fp = open(cachefn, 'r')
    #     v = joblib.load(cachefn)
    #     vectors = joblib.load(cachefn.replace('vectorizer-', 'vectors-'))
    #     fp.close()
    # else:
    features = get_features(docs, c_ngmin=param['c_ngmin'],
        c_ngmax=param['c_ngmax'], w_ngmin=param['w_ngmin'],
        w_ngmax=param['w_ngmax'], lowercase=param['lowercase'])

    # print("docs")
    # print(docs)

    # print("features")
    # print(features)

    v = TfidfVectorizer(analyzer=identity,
                        lowercase=(param['lowercase'] == 'all'),
                        sublinear_tf=param['sublinear'],
                        min_df=param['min_df'],
                        norm=param['norm'],
                        max_features=param['max_features'])
    vectors = v.fit_transform(features)
        # if cache and param['input_name']:
        #     info('Saving vectorizer: {}'.format(cachefn))
        #     joblib.dump(v, cachefn, compress=True)
        #     joblib.dump(vectors,
        #             cachefn.replace('vectorizer-', 'vectors-'),
        #             compress=True)

#     svd = None
#     if param['dim_reduce']:
#         info("reducing dimentionality {} -> {}".format(
#             len(v.vocabulary_), param['dim_reduce']))
#         svd = TruncatedSVD(n_components=param['dim_reduce'], n_iter=10)
# #        svd = TruncatedSVD(n_components=dim_reduce, #        algorithm="arpack")
#         svd.fit(vectors)
#         info("explained variance: {}".format(
#             svd.explained_variance_ratio_.sum()))
#         vectors = svd.transform(vectors)

    return vectors, v

def doc_to_ngrams3(docs, v):

    """ Return bag-of-n-grams features for the give document set
    """
    param = {
        'c_ngmax': 6, 
        'c_ngmin': 1, 
        'w_ngmax': 2, 
        'w_ngmin': 1,
        'min_df': 1,
        'sublinear': True,
        'norm': 'l2',
        'max_features': None,
        'input_name': None,
        'lowercase': 'word',
        'dim_reduce': None,
        'cache_dir': '.cache',
        'use_cached': True
    }

    features = get_features(docs, c_ngmin=param['c_ngmin'],
        c_ngmax=param['c_ngmax'], w_ngmin=param['w_ngmin'],
        w_ngmax=param['w_ngmax'], lowercase=param['lowercase'])

    vectors = v.transform(features)

    return vectors, v

def get_features(docs, c_ngmax=1, c_ngmin=1, w_ngmax=1, w_ngmin=1, lowercase=None):
    # convert docs to word/char ngrams with optional case normaliztion
    # this would ideally be tha anlyzer parameter of the
    # vectorizer, but requires lambda - which breaks saving 
    # convert docs to word/char ngrams with optional case normaliztion
    # this would ideally be tha anlyzer parameter of the
    # vectorizer, but requires lambda - which breaks saving 
    features = []
    for doc in docs:
        # character n-grams
        if lowercase == 'char':
            docfeat = get_ngrams(doc.lower(),
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        else:
            docfeat = get_ngrams(doc,
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        # word n-grams
        if lowercase == 'word':
            docfeat.extend(get_ngrams(doc.lower(),
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        else:
            docfeat.extend(get_ngrams(doc,
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        features.append(docfeat)
    return features

# def cache_vectors(vectors, v):
#     print('Saving vectorizer...')
#     vector_file_label = 'vector'		 # TODO: make in env var
#     vectorizer_file_label = 'vectorizer' # TODO: make in env var

#     vector_file_path = './cache/{}'.format(vector_file_name)
#     vectorizer_file_path = './cache/{}'.format(vectorizer_file_label)

#     joblib.dump(v, vectorizer_file_path, compress=True)
#     joblib.dump(vectors, vector_file_path, compress=True)

def convert_to_ngrams(datas):
	features = get_features(datas)

	print("features:")
	print(features)
	# TODO: make these in the options
	v = TfidfVectorizer(sublinear_tf=True)
						# analyzer=identity,
	                    # lowercase=True,
	                    # min_df=param['min_df'],
	                    # norm=param['norm'],
	                    # max_features=param['max_features'])
	vectors = v.fit_transform(features)
	# cache_vectors(vectors, v)
	return vectors, v

def main(opt):
	# Train.
	train_x, train_y = load_data(opt['train_eng_text'], opt['train_eng_label'])
	train_y = np.array(train_y)

	# top_n = 3
	# train_x, train_y = train_x[:top_n], train_y[:top_n]

	train_docs, v = doc_to_ngrams2(train_x)

	print("train_docs shape = {}".format(train_docs.shape))

	# train_docs, _ = convert_to_ngrams(train_x[:10])
	# test_docs, _ = convert_to_ngrams(test_x[:10])

	from sklearn.svm import LinearSVC
	m = LinearSVC(dual=True, C=1.0, verbose=0)

	m = OneVsRestClassifier(m, n_jobs=-1)
	skf = StratifiedKFold(n_splits=5)

	m.fit(train_docs, train_y)


	# Predict. 

	test_x, test_y = load_data(opt['test_eng_text'], opt['test_eng_label'])
	# test_x, test_y = test_x[:top_n], test_y[:top_n]
	# test_docs, v = doc_to_ngrams2(test_x)

	test_docs, v = doc_to_ngrams3(test_x, v)
	print("test_docs shape = {}".format(test_docs.shape))

	preds = m.predict(test_docs)

	with open('results.txt', 'w') as f:
		for pred in preds:
			f.write("{}\n".format(pred))

if __name__ == "__main__":
	main(options)