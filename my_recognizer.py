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
   
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    for testWord, (currentX,currentLen) in test_set.get_all_Xlengths().items():
        #Init
        probDic={}
        bestScore=float("-inf")
        bestWord=""
        
        for trainWord,model in models.items():
            try:
                currentScore=model.score(currentX,currentLen)
            except:
                currentScore=float("-inf")
            if currentScore > bestScore:
                bestScore=currentScore
                bestWord=trainWord
            probDic[trainWord]=currentScore
        probabilities.append(probDic)
        guesses.append(bestWord)
    return probabilities,guesses

        
        