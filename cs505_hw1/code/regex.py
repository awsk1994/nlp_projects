import re
from pathlib import Path

RAW_DUMP_XML = Path("raw_data/Wikipedia.xml")


def count_regexp():
    """Counts the occurences of the regular expressions you will write.
    """
    # Here's an example regular expression that roughly matches a valid email address.
    # The ones you write below should be shorter than this
    email = re.compile("[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z]{2,5}")

    ###### Write below #########
    subheading = re.compile("(?:==|===)([\w\s]+)(?:==|===)$", re.MULTILINE)
    link_to_subheading = re.compile("\[\[[\w\s]+\#([\w\s]+)\|[\w\s]+\]\]")
    doi_citation = re.compile("\{\{.*doi=(.+?)\|")
    ###### End of your work #########

    patterns = {
        "emails": email,
        "subheadings": subheading,
        "links to subheadings": link_to_subheading,
        "citations with DOI numbers": doi_citation,
    }

    with open(RAW_DUMP_XML) as f:
        dump_text = f.read()
        for name, pattern in patterns.items():
            if pattern is None:
                continue
            matches = pattern.findall(dump_text)
            count = len(matches)

            example_matches = [matches[i * (count // 5)] for i in range(5)]

            print("Found {} occurences of {}".format(count, name))
            print("Here are examples:")
            print("\n".join(example_matches))
            print("\n")


if __name__ == "__main__":
    count_regexp()
