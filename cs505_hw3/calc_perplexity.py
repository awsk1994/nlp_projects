import re

def get_train_data(data_path):
	data = open(data_path, 'r', encoding='utf-8').read() # should be simple plain text file
	return data

def get_count(text, target):
	target = target.replace("[", "\[")
	target = target.replace("]", "\]")
	target = target.replace("?", "\?")
	target = target.replace("+", "\+")
	target = target.replace("(", "\(")
	target = target.replace(")", "\)")

	target = re.findall(target, text) ## ['alice@google.com', 'bob@abc.com']
	return len(target)

def calculate_perplexity_per_chunk(train_data, val_data, DEBUG=True):
	'''
	P(i|SOT) = count(i)/vocab_size
	P(it will) = P(i|SOT) * P(t|i) * P(\s|it) * P(w|it ) * P(i|it w) * .... 
	perplexity = P(..)^(-1/n), where n = length of chunk
	'''
	if DEBUG:
		print("\tCalculating perplexity for \"{}\"".format(val_data))
	data_size = len(train_data)
	cond_prob_agg = 1
	for i in range(len(val_data)):
		count = get_count(train_data, val_data[:i+1])
		cond_prob_agg *= (count / data_size)	# P(word)
		# if DEBUG == True:
		# 	print("\t\tcond_prob of {} is {}/{} = {}".format(val_data[:i+1], count, data_size, (count / data_size)))

	perplexity = pow(cond_prob_agg, (-1/len(val_data))) if cond_prob_agg != 0 else 0
	if DEBUG == True:
		print("\tcond_prob_agg={}, perplexity={:.2f}\n".format(cond_prob_agg, perplexity))

	return perplexity

def calculate_perplexity(train_data, val_data, chunk_size=3, DEBUG=True):
	val_data_split_by_chunk = [val_data[i:i+chunk_size] for i in range(0, len(val_data), chunk_size)]

	print("val_data_split_by_chunk")
	print(val_data_split_by_chunk)

	perplexity_aggregate = 0
	for chunk_val_data in val_data_split_by_chunk:
		perplexity_aggregate += calculate_perplexity_per_chunk(train_data, chunk_val_data, DEBUG=DEBUG)

	ave_perplexity = perplexity_aggregate/len(val_data_split_by_chunk)

	if DEBUG:
		print("Average Perplexity for \"{}\" (using Chunk Size={}) = {}".format(val_data, chunk_size, ave_perplexity))
	return ave_perplexity

def main():
	train_data = get_train_data("./data/generative_data/trump_speeches.txt")
	val_data = get_train_data("./data/generative_data/trump_val_speech.txt")	
	calculate_perplexity(train_data, val_data, chunk_size=20, DEBUG=True)

if __name__ == "__main__":
	main()