import json
import numpy as np
import pickle
from make_dic_tree import * # tree obj, supporting methods


# load the dictionary tree (needs to be global)
file_obj = open('dic_tree.obj','rb')
root = pickle.load(file_obj)
file_obj.close()

# map from number in 23456789 to set of letters abcdefghijklmnopqrstuvwxyz
ALPHABET_LIST = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
ALPHABET_TO_N = {'a':2,'b':2,'c':2,'d':3,'e':3,'f':3,'g':4,'h':4,'i':4,
                 'j':5,'k':5,'l':5,'m':6,'n':6,'o':6,'p':7,'q':7,'r':7,'s':7,
                 't':8,'u':8,'v':8,'w':9,'x':9,'y':9,'z':9,' ':1}
ntol = lambda n: ALPHABET_LIST[n-2]
lton = lambda l: ALPHABET_TO_N[l]

# returns a string of digits, somthing like "435313454136346145557"
def text_to_numbers_string(text): # assumes text is an english phrase
    n_list = [str(ALPHABET_TO_N[i]) for i in text]
    n_string=""
    for i in n_list: n_string = n_string + i
    return n_string

# takes a finger (n in 0123456789) and a set of nodes in the tree (1 is spacebar, 0 is delete)
# return string reps list and set of nodes in the tree, or None
def next_input(n,nodes=[root]):
    new_nodes,strings = [],[]
    if n==1:
        new_nodes = [node for node in nodes if node.data[1]==1]
        strings = [get_current_string(i) for i in new_nodes]
        new_nodes = [root]
    elif n==0:
        for node in nodes:
            new_nodes.append(node.parent)
        new_nodes = list(set(new_nodes))
        strings = [get_current_string(i) for i in new_nodes]
    elif n>=2 and n<= 9:
        for node in nodes:
            for l in ntol(n):
                child = get_child(node,l)
                if child!=None: new_nodes.append(child)
        if new_nodes==[]:
            print("there are no valid nodes here...")
            strings=[]
        else: strings=[get_current_string(i) for i in new_nodes]
    
    else: raise Exception("Input {} not valid, please enter a single (base 10) digit".format(n))
        
    return strings,new_nodes # might return empty 

if __name__=="__main__":    
    phrase = "this is a text"
    numbers_string = text_to_numbers_string(phrase)
    print('in phone digits, the phrase "{}" becomes {}\nwhere 1 is for space\n'.format(phrase,numbers_string))
    
    print("the methods output a list of possibilities for each word:\n")

    nodes = [root]
    sentence = []
    for i in numbers_string+"1":
        strings,nodes = next_input(int(i),nodes)
        if i=="1":
            sentence.append(strings)
            print(strings)
    
