# levenshtein with memoizaiton from https://www.python-course.eu/levenshtein_distance.php

def memoize(func):
    mem = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return memoizer

@memoize    
def levenshtein(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    
    res = min([levenshtein(s[:-1], t)+1,
               levenshtein(s, t[:-1])+1, 
               levenshtein(s[:-1], t[:-1]) + cost])

    return res

threshold = 5
words = []
def check(word):
    minimum = threshold
    cands = []

    for poss in words:
        d = levenshtein(poss, word)
        if d < minimum:
            minimum = d
            cands = [poss]
        elif d == minimum: 
            cands.append(poss)

    return cands

if __name__ == "__main__":
   words = [word.strip() for word in open("10k_dict.txt", "r").readlines()]
   while 1:
       print(check(input("")))

