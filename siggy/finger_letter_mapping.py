#!/usr/bin/env python
# coding: utf-8

# In[9]:


# author @miasya
# last modified mar 22, 2020

# DESC: This file defines 2 dicts with letter to finger AND finger to letter index mapping

# Our data will produce a vector of length 10, where index
# 0 - no signal
# 1 - right - thumb (SPACE)
# 2 - right - index 
# 3 - right - middle 
# 4 - right - ring 
# 5 - right - pinky 
# 6 - left - thumb
# 7 - left - index 
# 8 - left - middle 
# 9 - left - ring 
# 10 - left - pinky

letter_finger = {'q' : 10, 'a' : 10, 'z' : 10,
                'w' : 9, 's' : 9, 'x' : 9,
                'e' : 8, 'd' : 8, 'c' : 8,
                'r' : 7, 'f' : 7, 'v' : 7, 't' : 7, 'g' : 7, 'b' : 7,
                'p' : 5,
                'o' : 4, 'l' : 4,
                'i' : 3, 'k' : 3,
                'u' : 2, 'j' : 2, 'm' : 2, 'y' : 2, 'h' : 2, 'n' : 2, 
                ' ' : 1}

finger_letter = {10 : ('q', 'a', 'z'),
                9 : ('w', 's', 'x'),
                8 : ('e', 'd', 'c'),
                7 : ('r', 'f', 'v', 't', 'g', 'b'),
                1 : ' ',
                5 : ('p'),
                4 : ('o', 'l'),
                3 : ('i', 'k'),
                2 : ('u', 'j', 'm', 'y', 'h', 'n')}


"""
# Sample code
"""
if __name__ == "__main__": 
    my_sentence = "do you like pancakes"

    for c in my_sentence:
        if c in letter_finger.keys(): 
            print(letter_finger[c])

# In[ ]:




