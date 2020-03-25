''' Important function: `text_to_realistic` 
Important parameters: AVG_NUM_REPEATS and STDEV (mean and std dev number of times a given letter is repeated),
'''

import numpy as np 

from finger_letter_mapping import letter_finger, finger_letter

AVG_NUM_REPEATS = 6
STDEV = 3

def text_to_finger(data): 
    '''Takes input either a list of sentences (list of strings) or one sentence (one string).
    Outputs the training input: for each sentence in the input, a list of letters in this 
    sentence filtered by occurence in the dictionary. 
    '''
    if type(data) == list or type(data) == np.ndarray: # if input is a list of sentences
        output = [] 
        for row in data:
            output.append(np.array([letter_finger[c] for c in row.lower() if c in letter_finger.keys()]))
        return np.array(output)
    elif type(data) == str:  # if input is one sentence 
        return np.array([letter_finger[c] for c in data.lower() if c in letter_finger.keys()])
    
def text_to_label(data): 
    '''Takes input either a list of sentences (list of strings) or one sentence (one string).
    Outputs the labels for testing: for each sentence in the input, a list of letters in this 
    sentence filtered by occurence in the dictionary. 
    '''
    if type(data) == list or type(data) == np.ndarray: # if input is a list of sentences    
        output = []
        for row in data: 
            output.append(np.array(list(filter(lambda c: c in letter_finger.keys(), row.lower()))))
        return np.array(output)
    elif type(data) == str: # if input is one sentence 
        return np.array(list(filter(lambda c: c in letter_finger.keys(), data.lower()))) 

def draw_int_normal(num, std):
    ''' Draw an integer from a normal distribution (num, std) mean standard dev, with a few 
    '''
    s = np.random.normal(num, std)
    if s <= 0: return 1 
    elif s <= num: return int(np.ceil(s))
    elif s > num*2: return int(num*2)
    else: return int(np.floor(s))
    
def add_repeats(sentence): 
    '''Input: sentence i.e. array of integers representing finger activation for each letter in sentence.
    Output: sentence with added repeats for each character: array of integers 
    '''
    with_repeats = []
    for entry in sentence: # each letter = finger in data
        num_repeats = draw_int_normal(AVG_NUM_REPEATS, STDEV)
        num_nosignal = draw_int_normal(AVG_NUM_REPEATS, STDEV)
        with_repeats += [entry]*num_repeats + [0]*num_nosignal #concatenate to output
    return with_repeats    
    
def finger_to_onehot(data): 
    '''Input: sentence (array of integers finger activations) or array of sentences.
    Output: if sentence input, array of one hot vectors. If array input, array of [ one hot vectors ] 
    for each sentence.
    '''
    NUM_VALS = 11    # labels from 0 to 10 
    if type(data[0]) == np.int32 or type(data[0]) == int:   
        # one sentence (if it were an array of sentences this type would be list)
        return np.eye(NUM_VALS)[data] 
    else: 
        return np.array([np.eye(NUM_VALS)[row] for row in data])
    
# not used 
def onehot_to_probdist(vector): 
    '''Input: 1 one_hot vector where len(vector)=11 for the one-hot encoding.
    Output: 1 vector of same length as input with added noise making it a probability vector.
    '''
    new_one_prob = np.random.uniform(low=0.3, high=1)  # 1 entry is now less 
    one_index = np.argmax(vector)
    removed = 1 - new_one_prob 
    output = np.random.multinomial(np.ceil(removed*1000), [1/(len(vector)-1)]*(len(vector)-1)) / 1000
    output = np.insert(output, one_index, new_one_prob)   
    return output

# used 
def random_normalize_rows(a): 
    '''Input: a sentence entry (nd array (x, 11)) 
    Output: sentence entry with added randomness and normalized. '''
    a = a + np.random.rand(*a.shape)/5
    return a / a.sum() if a.ndim == 1 else a / a.sum(axis=1, keepdims=True)

def text_to_realistic(data): 
    '''Input: text data, either sentence (one string) or array of sentences (array of strings). 
    Output (realistic siggy output): (one sentence case) array of probability vectors representing 
    probability of each finger. 
    '''
    if type(data) == str: # one sentence 
#         with_repeats = add_repeats(true_fingers) 
        one_hot = finger_to_onehot(add_repeats(text_to_finger(data)))  # array of one_hot vectors (finger) 
#         output = [onehot_to_probdist(vector) for vector in one_hot]  # old prob distribution 
        return random_normalize_rows(one_hot) 
    else: # array of sentences 
#         with_repeats = [add_repeats(sentence) for sentence in true_fingers] 
        true_fingers = text_to_finger(data) 
        one_hot = finger_to_onehot([add_repeats(sentence) for sentence in true_fingers]) # array of [ one_hot vectors ]
#         output = [] 
#         for sentence in one_hot: 
#             output.append([onehot_to_probdist(vector) for vector in sentence]) 
#         return output 
        return np.array([random_normalize_rows(sent) for sent in one_hot])

# testing 
if __name__ == "__main__": 
    print(text_to_finger('hi, this is a TEST'))
    print(text_to_label('hi, this is a TEST'))
    import nltk 
    from nltk.tag import hmm
    from nltk.probability import LaplaceProbDist
    
    nltk.download('abc')
    nltk.download('nps_chat')
    
    def wordsplit_to_sent(data) : 
        ''' Input array of sentences where sentence = array of words. 
        Output array of sentences where sentence = string.
        '''
        return [' '.join(sent)[:-2].lower() for sent in data]
    
    # get 9000 sentences each as array split by word, for training 
    sentences_wordsplit = nltk.corpus.abc.sents()[:9000]   
    # get 1000 sentences for testing 
    tester_wordsplit = nltk.corpus.abc.sents()[10000:11000]   
    
    # turn each sentence from a word array into a sentence string
    training = wordsplit_to_sent(sentences_wordsplit)  
    testing = wordsplit_to_sent(tester_wordsplit)
    
    print('example training sentences \n', training[:2])
    
    print('Letter space (hidden states):\n ', letter_finger.keys())
    print('Finger space (observed states): \n', finger_letter.keys())
    
    out = text_to_realistic(training[:1000]) 
    print('siggy output example: \n', out[0])