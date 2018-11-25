import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implemented the recognizer
    for item in test_set.get_all_sequences():
        X, lengths = test_set.get_item_Xlengths(item)
        scores = {}
        best_word = None
        best_score = None
        for word, model in models.items():
            try:
                score = model.score(X, lengths)
            except:
                score = float("-inf")
            if best_score is None or best_score < score:
                best_score = score
                best_word = word
            scores[word] = score
        probabilities.append(scores)
        guesses.append(best_word)
    return (probabilities, guesses)