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
