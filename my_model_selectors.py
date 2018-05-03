import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
        #The lowest the BIC the better
        best_n, bestBic = self.n_constant,float("inf")

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:

                myModel=self.base_model(n)
                logLike=myModel.score(self.X,self.lengths)
                logN = math.log(len(self.sequences))
                number_features=myModel.n_features

# found in forums, based on means std (free parameters) etc.
                
                p = (n**2) + (2 * n *number_features) -1 
                currentBic = -2 * logLike + p * logN

                if bestBic is None or bestBic > currentBic:
                    bestBic, best_n = currentBic, n
            except:
                pass

        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):


        # TODO implement model selection based on DIC scores
        #now look the different initialization. Now bigger DIC better
        best_n, bestDic = self.n_constant,float("-inf")
        currentDic=0.0
        #We will need to train a model for each word

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                totalScore=0.0
                myModel=self.base_model(n)
                logLike = myModel.score(self.X, self.lengths)
                #Extracting info from the dictionary
                for currentWord, (currentX,currentLengths) in self.hwords.items():
                    if currentWord !=self.this_word:
                        totalScore+=myModel.score(currentX,currentLengths)

                currentDic= logLike - (totalScore/(len(self.words)-1))
                #Now the sign change since the bigger dic the better
                if bestDic is None or bestDic < currentDic:
                    bestDic, best_n = currentDic, n



            except:
                pass
        return self.base_model(best_n)




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        bestScore=None
        best_n=0
       
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                totalScore=0.0
                counter=0
                currentScore=0.0
                if (len(self.sequences))>=3:
                    n_folds=3
                    split_method = KFold(n_folds)
                    for trainIdx,testIdx in split_method.split(self.sequences):
                        xTrain,lenghTrain= combine_sequences(trainIdx, self.sequences)
                        xTest,lenghTest= combine_sequences(testIdx, self.sequences)
                        
                        myModel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(xTrain, lenghTrain)
                        totalScore+=myModel.score(xTest,lenghTest)
                        counter=counter+1
                        if counter ==0:
                            currentScore=totalScore/1.0
                        else:
                            currentScore=totalScore/counter
                else:
                    
                    myModel=GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    currentScore=myModel.score(self.X, self.lengths)
                if bestScore is None or bestScore < currentScore:
                    bestScore=currentScore
                    best_n = n
            except:
                pass
        model=GaussianHMM(n_components=best_n,covariance_type="diag",n_iter=1000,random_state=self.random_state,verbose=False).fit(self.X,self.lengths)
        return model



                   







        
