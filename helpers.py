from io import open

from conllu import parse_incr
from typing import Generator

_BATCH_SIZE = 10


def read_conllu_file(filenames: str, batch_size=_BATCH_SIZE) -> Generator:
    """
    Reads all .conllu files and returns a list of lists of files,
    one per filename provided
    """
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as file:
            minibatch = []
            for tokenlist in parse_incr(file):
                minibatch.append(tokenlist)
                if len(minibatch) == batch_size:
                    yield minibatch
                    minibatch = []
            # Still anything to yield?
            if len(minibatch) != 0:
                yield minibatch


def read_txt_file(filename: str) -> Generator:
    """
    Reads txt files with one sentence per line
    """
    with open(filename, "r", encoding="utf-8") as file:
        while True:
            data = file.readline()
            if not data:
                break
            yield data


def get_structure(string: str) -> str:
    """
    Get the structure of the string, mapping lowercase to x, uppercase
    to X, numbers to d and other characters retained as original.
    """
    output = ''
    for i in string:
        if i.isnumeric():
            output += 'd'
        elif i.isupper():
            output += 'X'
        elif i.islower():
            output += 'x'
        else:
            output += i
    return output


def extract_features(sentence: list, i: int) -> dict:
    """
    Extract features from every word-sentence, by looking at parts of
    the word itself, as well as surrounding words, and the whole
    sentence.
    """
    features = {
        'word': sentence[i],
        'word.lower': sentence[i].lower(),
        'is_first': i == 0,
        'is_last': i == len(sentence)-1,
        'is_title': sentence[i].istitle(),
        'is_all_caps': sentence[i].isupper(),
        'is_all_lower': sentence[i].islower(),
        'is_alphanumeric': sentence[i].isalnum(),
        'is_only_alpha': sentence[i].isalpha(),
        'pre-1': sentence[i][0],
        'pre-2': sentence[i][:2],
        'pre-3': sentence[i][:3],
        'suf-1': sentence[i][-1],
        'suf-2': sentence[i][-2:],
        'suf-3': sentence[i][-3:],
        'has_hyphen': '-' in sentence[i],
        'is_numeric': sentence[i].isnumeric(),
        'capitals_inside': sentence[i][1:].lower() != sentence[i][1:],
        'structure': get_structure(sentence[i])
    }
    if i > 0:
        features.update({
            'prev.word': sentence[i-1],
            'prev.word.lower': sentence[i-1].lower(),
            'prev.word.is_title': sentence[i-1].istitle(),
            'prev.word.is_first': i-1 == 0
        })
    if i < len(sentence)-1:
        features.update({
            'next.word': sentence[i+1],
            'next.word.lower': sentence[i+1].lower(),
            'next.word.is_title': sentence[i+1].istitle(),
            'next.word.is_last': i+1 == len(sentence)-1
        })
    return features


def sent2features(sentence: list) -> list:
    """
    Extracts features from every word of a sentence iteratively
    """
    return [extract_features(sentence, i)
            for i in range(len(sentence))]
