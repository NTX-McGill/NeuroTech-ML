#!/usr/bin/env python
# coding: utf-8

# In[9]:


# author @miasya
# last modified feb 9, 2020

# DESC: This file defines 2 dicts with letter to finger AND finger to letter index mapping

# Our data will produce a vector of length 10, where index
# 0 - left - pinky
# 1 - left - ring
# 2 - left - middle
# 3 - left - index
# 4 - left - thumb
# 5 - right - pinky
# 6 - right - ring
# 7 - right - middle
# 8 - right - index
# 9 - right - thumb

letter_finger = {'q' : 0, 'a' : 0, 'z' : 0,
                'w' : 1, 's' : 1, 'x' : 1,
                'e' : 2, 'd' : 2, 'c' : 2,
                'r' : 3, 'f' : 3, 'v' : 3, 't' : 3, 'g' : 3, 'b' : 3,
                'p' : 5,
                'o' : 6, 'l' : 6,
                'i' : 7, 'k' : 7,
                'u' : 8, 'j' : 8, 'm' : 8, 'y' : 8, 'h' : 8, 'n' : 8}

finger_letter = {0 : ('q', 'a', 'z'),
                1 : ('w', 's', 'x'),
                2 : ('e', 'd', 'c'),
                3 : ('r', 'f', 'v', 't', 'g', 'b'),
                4 : None,
                5 : ('p'),
                6 : ('o', 'l'),
                7 : ('i', 'k'),
                8 : ('u', 'j', 'm', 'y', 'h', 'n'),
                9 : None}


"""
# Sample code
my_sentence = "do you like pancakes"

for c in my_sentence:
    if c != ' ':
        print(letter_finger[c])
    else:
        print(' ')
"""


# In[ ]:




