import math
import statistics
import warnings

import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from numpy import asarray

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Implemented model selection based on BIC scores
        #According to the slides provided above, "the lower the BIC value, the better the model"
        
        bic_value = float('inf')
        best_model = None
        
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                #fits model with n_components
                model = self.base_model(n) 
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.sequences)) # N is the number of data points
                #from https://discussions.udacity.com/t/parameter-in-bic-selector/394318
                # p is the number of parameters
                p = n ** 2 + 2 * n * model.n_features - 1
                #current BIC value:
                currentBIC = -2 * logL + p * logN
                if bic_value > currentBIC:
                    bic_value = currentBIC
                    best_model = model
            except:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implemented model selection based on DIC scores
        #based on https://discussions.udacity.com/t/selector-dic-taking-a-long-time-to-run/334060/5
        dic_value = float('-inf')
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                #fits model with n_components
                model = self.base_model(n)
                mscore = model.score(self.X,self.lengths)
                all_but_i = 0.0
                for w in self.words:
                    if w != self.this_word:
                        X, lengths = self.hwords[w]
                        all_but_i += model.score(X, lengths)
                currentDIC = mscore - (all_but_i / (len(self.words) - 1))
                if dic_value < currentDIC:
                    dic_value = currentDIC
                    best_model = model
            except:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implemented model selection using CV
        #How does CV work: http://scikit-learn.org/stable/modules/cross_validation.html
        #Example: #http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-oob-py
        cv_score = float('-inf')
        best_model = None
        
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                if (len(self.sequences) < 2):
                    currentModel = self.base_model(n)
                    currentCV = currentModel.score(self.X, self.lengths)
                else:
                    mscore = 0.0
                    count = 0
                    cv = KFold(n_splits=min(len(self.sequences), 3))
                    for train, test in cv.split(self.sequences):
                        # Kfold for train
                        X_train, lengths_train = combine_sequences(train, self.sequences)
                        # Kfold for test
                        X_test, lengths_test = combine_sequences(test, self.sequences)
                        #fits the model with train data
                        currentModel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        mscore += currentModel.score(X_test, lengths_test)
                        count += 1
                    currentCV = mscore/count
                if cv_score < currentCV:
                    cv_score = currentCV
                    best_model = currentModel
            except:
                pass
        return best_model
