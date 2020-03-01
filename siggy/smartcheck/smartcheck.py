"""
Smartcheck: smart spellcheck in pure Python.

 FEATURES:
  - Norvig's autocorrect
  - 3-gram language model 

 TODO:
  - Combine Norvig + 3-gram approaches
  - Build better error model with errors from text
  - Save + pickle the trained 3-gram, language, error models
"""

from nltk import bigrams, word_tokenize
from collections import Counter, defaultdict
import re

class Smartcheck:
    """A smart spell checker.

    Uses a bigram language model.
    """

    def __init__(
            self, 
            dict_file = "dictionary.txt",
            model_file = "count_1w.txt", 
            bigram_file = "count_2w.txt"
    ):
        """Initializes language model with trigram probabilities."""
        self.dict_file = dict_file
        self.model_file = model_file
        self.bigram_file = bigram_file
        self.bigrams = defaultdict(lambda: defaultdict(lambda: 0))
        self.model = {} 
        self.pop_model()
        self.pop_bigrams()

    def process_file(self, filename):
        content = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                key, val = line.split("\t")
                content[key.lower()] = int(val)
        return content

    def sentences(self, text):
        """All sentences in a given text."""
        return re.findall(r'([A-Z][^\.!?]*[\.!?])', text)

    def words(self, text):
        """All words in a given text."""
        return re.findall(r'\w+', text)

    def pop_model(self):
        """Populate model with probability of word."""
        dict_words = set([line.strip().lower() for line in open(self.dict_file, "r").readlines()])
        word_counts = self.process_file(self.model_file) 
        N = sum(word_counts.values())
        for word in word_counts:
            if word in dict_words:
                self.model[word] = word_counts[word] / N

    def pop_bigrams(self):
        """Populate self.bigrams with probs of next words using Norvig"""
        bigram_counts = self.process_file(self.bigram_file)
        N = sum(bigram_counts.values())
        for bigram in bigram_counts:
            self.bigrams[bigram.lower()] = bigram_counts[bigram] / N
        
    def pop_bigrams_old(self):
        """Populate self.bigrams with probabilities of next words"""
        for sentence in self.sentences(self.corpus):
            for w1, w2 in bigrams(word_tokenize(sentence), pad_right=True, pad_left=True):
                self.bigrams[w1][w2] += 1

        # Convert trigrams to probabilities
        for wp in self.bigrams:
            total_count = float(sum(self.bigrams[wp].values()))
            for w2 in self.bigrams[wp]:
                self.bigrams[wp][w2] /= total_count

    def predict(self, sentence):
        """Predict the next words given the sentence."""
        prev_two_words = sentence.split()[-2:]
        options = dict(self.trigrams[tuple(prev_two_words)])
        return options

    def word_probability(self, word, prev):
        """Probability of a given word."""
        bg = "{} {}".format(prev, word)
        p_c = self.model[word] if word in self.model else 1e-10 
        p_cw = self.bigrams[bg] if bg in self.bigrams else 1e-10 
        p = p_c * p_cw if prev else p_c
        print(word, p_c, p_cw)
        return p

    def correction(self, word, prev):
        """Return the most probable correction."""
        # Case 1: word is in model
        if word in self.model:
            return word
        # Case 2: word is unknown
        return max(self.candidates(word), key=lambda w: self.word_probability(w, prev))

    def candidates(self, word):
        """Candidate list of possible correct words."""
        return (self.known([word]) or \
                self.known(self.edits1(word)) | \
                self.known(self.edits2(word)) or \
                set([word]))

    def known(self, words):
        return set(w for w in words if w in self.model)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

def test(test_file):
    sc = Smartcheck()
    correct = 0
    incorrect = 0
    with open(test_file, "r") as f:
        for line in f.readlines():
            wrong, real = line.split("\t")[:2]
            predict = sc.correction(wrong, "")
            if predict.strip() == real.strip():
                correct += 1
            else:
                incorrect += 1
                print(wrong, real, predict)
            print("Success rate:")
            print(correct / (correct + incorrect))
    print("Success rate:")
    print(correct / (correct + incorrect))


if __name__ == "__main__":
    # test("test2.txt")
    sc = Smartcheck()
    while 1:
        prev, word = input("").split()[:2]
        print(sc.correction(word, prev))

 
