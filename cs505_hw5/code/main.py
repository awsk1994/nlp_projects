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
	'output_res_filename_eng': 'english_results.txt',

	'train_esp_text': './data/train/spanish_train.text',
	'train_esp_label': './data/train/spanish_train.labels',
	'test_esp_text': './data/test/spanish_test.text',
	'test_esp_label': './data/test/spanish_test.labels',
	'mapping_esp': './data/mapping/spanish_mapping.txt',
	'output_res_filename_esp': 'spanish_results.txt',

	'lang': 'spanish', # spanish or english
	'output_res_filename_default': 'results.txt',

	# config:
	'c_ngmax': 6, 
	'c_ngmin': 1, 
	'w_ngmax': 2, 
	'w_ngmin': 1,
	'lowercase': 'word',
	'min_df': 1,
	'norm': 'l2',
	'sublinear': True,
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

# TODO: modify
def get_ngrams(s, ngmin=1, ngmax=1,
				tokenizer=None,
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
	ngrams = [[] for x in range(1, ngmax + 1)]
	s = tokenizer(bos + s + eos)
	for i, ch in enumerate(s):
		for ngsize in range(ngmin, ngmax + 1):
			if (i + ngsize) <= len(s):
				ngrams[ngsize - 1].append(separator.join(s[i:i+ngsize]))
	if flatten:
		ngrams = [ng for nglist in ngrams for ng in nglist]
	return ngrams

def get_features(docs):
	# convert docs to word/char ngrams with optional case normaliztion
	# this would ideally be tha anlyzer parameter of the
	# vectorizer, but requires lambda - which breaks saving 
	# convert docs to word/char ngrams with optional case normaliztion
	# this would ideally be tha anlyzer parameter of the
	# vectorizer, but requires lambda - which breaks saving 
	features = []
	for doc in docs:
		docfeat = get_ngrams(doc,
			ngmax=options['c_ngmax'], ngmin=options['c_ngmin'],
			tokenizer=list)
		# word n-grams
		w_tokenizer = re.compile("\w+|[^ \t\n\r\f\v\w]+").findall
		docfeat.extend(get_ngrams(doc.lower(),
			ngmax=options['w_ngmax'], ngmin=options['w_ngmin'],
			append="W", tokenizer=w_tokenizer))
		features.append(docfeat)
	return features

def train_doc_to_ngrams(docs):
	""" Return bag-of-n-grams features for the give document set
	"""
	features = get_features(docs)
	v = TfidfVectorizer(analyzer=identity,
						lowercase=False,
						sublinear_tf=options['sublinear'],
						min_df=options['min_df'],
						norm=options['norm'],
						max_features=None)
	vectors = v.fit_transform(features)
	return vectors, v

def test_doc_to_ngrams(docs, v):
	""" Return bag-of-n-grams features for the give document set
	"""
	features = get_features(docs)
	vectors = v.transform(features)
	return vectors, v

def main(opt):
	# 1. Train.
	if opt['lang'] == 'english':
		train_x_file = opt['train_eng_text']
		train_y_file = opt['train_eng_label']
		test_x_file = opt['test_eng_text']
		test_y_file = opt['test_eng_label']
	elif opt['lang'] == 'spanish':
		train_x_file = opt['train_esp_text']
		train_y_file = opt['train_esp_label']
		test_x_file = opt['test_esp_text']
		test_y_file = opt['test_esp_label']
	else:
		print("ERROR | invalid opt['lang']")
		return

	train_x, train_y = load_data(train_x_file, train_y_file)
	train_y = np.array(train_y)

	train_docs, v = train_doc_to_ngrams(train_x)
	print("train_docs shape = {}".format(train_docs.shape))

	from sklearn.svm import LinearSVC
	m = LinearSVC(dual=True, C=1.0, verbose=0)
	m = OneVsRestClassifier(m, n_jobs=-1)
	skf = StratifiedKFold(n_splits=5)
	m.fit(train_docs, train_y)

	# 2. Predict. 
	test_x, test_y = load_data(test_x_file, test_y_file)
	test_docs, v = test_doc_to_ngrams(test_x, v)
	print("test_docs shape = {}".format(test_docs.shape))

	preds = m.predict(test_docs)

	# 3. Write to file
	if opt['lang'] == 'english':
		result_file_name = opt['output_res_filename_eng']
	elif opt['lang'] == 'spanish':
		result_file_name = opt['output_res_filename_esp']
	else:
		print("WARNING | write to file | invalid opt['lang']")
		result_file_name = opt['output_res_filename_default']

	with open(result_file_name, 'w') as f:
		for pred in preds:
			f.write("{}\n".format(pred))

	print("DONE.")

if __name__ == "__main__":
	main(options)
