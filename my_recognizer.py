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
    
    test_X_lengths = test_set.get_all_Xlengths()
    
   
    for test_word_id in test_X_lengths:
        word_probs = {}
        
        X, lengths = test_X_lengths[test_word_id]
       
        for word in models:
            try:
                model = models[word]

                word_probs[word] = model.score(X, lengths)
            except:
                word_probs[word] = float("-inf")
        probabilities.append(word_probs)
            

    for p in probabilities:
    	guesses.append(max(p, key=p.get))
    	    
    return probabilities, guesses