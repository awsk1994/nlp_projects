"""Experiments with scraped wikipedia data and NLP tools
"""
import json
import spacy
from pathlib import Path

# The path that WikiExtractor.py will extract to
JSONL_FILES_DIR = Path("text/AA")


def yield_all_articles(path: Path = JSONL_FILES_DIR):
    """Go through all the extracted wikipedia files and "yield" one at a time.
    
    Read about generators here if necessary: https://wiki.python.org/moin/Generators
    """
    for one_file in path.iterdir():
        print("Going through", one_file.name)
        with open(one_file) as f:
            for line in f:
                text = json.loads(line)["text"]
                yield text
                break


def count_things():

    article_gen = yield_all_articles(JSONL_FILES_DIR)
    print("About to load spacy model")
    nlp = spacy.load("en")
    print("Finished loading spacy's English models")

    non_punct_non_space_token_count = 0
    lemma_count = 0
    ner_count_per_type = {}

    ###### Write below #########

    '''
    We only want unique lemmas, but repetitive tokens. At least that is what I got from David's response below.
    "Counting duplicate lemmas separately would be the same thing as counting the number of tokens." @15
    ~ An instructor (David Assefa Tofu) endorsed this answer  ~
    (https://piazza.com/class/k5o8wx7zkjk1zr?cid=26)
    '''

    import re
    import string

    def remove_punctuation_space_from_list(lst_word):
        nps_regex = re.compile(r"[^{}\s]+".format(string.punctuation))
        nps_list = list(filter(nps_regex.match, lst_word))
        return nps_list

    def get_unique_count_from_list(lst_word):
        nps_hash = {}
        for word in lst_word:
            if word in nps_hash.keys():
                nps_hash[word] += 1
            else:
                nps_hash[word] = 1
        return len(nps_hash.keys()), nps_hash

    # Collect data
    tokens = []
    lemmas = []
    entities = []
    for article in article_gen:
        doc = nlp(article)
        for token in doc:
            tokens.append(str(token))
            lemmas.append(str(token.lemma_))

        for ent in doc.ents:
            entities.append(ent)    # Interested value: ent.text, ent.label_

    # Process data
    ## Tokens
    nps_tokens = remove_punctuation_space_from_list(tokens)
    non_punct_non_space_token_count = len(nps_tokens)

    ## Lemmas
    nps_lemmas = remove_punctuation_space_from_list(lemmas)
    nps_uniq_lemmas_count, _ = get_unique_count_from_list(nps_lemmas)
    lemma_count = nps_uniq_lemmas_count

    ## Entities
    ent_hash = {}
    for ent in entities:
        if ent.label_ in ent_hash.keys():
            ent_hash[ent.label_] += 1
        else:
            ent_hash[ent.label_] = 1
    ner_count_per_type = ent_hash

    ###### End of your work #########

    print(
        "Non punctuation and non space token count: {}\nLemma count: {}".format(
            non_punct_non_space_token_count, lemma_count
        )
    )
    print("Named entity counts per type of named entity:")
    for ner_type, count in ner_count_per_type.items():
        print("{}: {}".format(ner_type, count))


if __name__ == "__main__":
    count_things()
