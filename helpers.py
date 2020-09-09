from io import open

from conllu import parse_incr


def read_file(filename:str) -> list:
    """Reads all .conllu files and returns a list of lists of files,
    one per filename provided"""
    files_list = []
    with open(filename, "r", encoding="utf-8") as file:
        for tokenlist in parse_incr(file):
            files_list.append(tokenlist)
    return files_list


def files_to_treebank(files:list) -> list:
    """Translate a list of conllu TokenLists into a list of tuples,
    representing ([sentence], [tags])"""
    treebank = []
    for sentence in files:
        tokens = []
        tags = []
        for token in sentence:
            tokens.append(token['form'])
            tags.append(token['upostag'])
        treebank.append((tokens, tags))
    return treebank

def get_structure(string:str) -> str:
    """Get the structure of the string, mapping lowercase to x, uppercase to X, numbers to d and
    other characters retained as original."""
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


def extract_features(sentence:list, i:int) -> dict:
    """Extract features from every word-sentence, by looking at parts of the word itself,
    as well as surrounding words, and the whole sentence."""
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
      'pre-1':sentence[i][0],
      'pre-2':sentence[i][:2],
      'pre-3':sentence[i][:3],
      'suf-1':sentence[i][-1],
      'suf-2':sentence[i][-2:],
      'suf-3':sentence[i][-3:],
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


def sent2features(sentence:list)->list:
    """Extracts features from every word of a sentence iteratively"""
    return [extract_features(sentence, i) for i in range(len(sentence))]