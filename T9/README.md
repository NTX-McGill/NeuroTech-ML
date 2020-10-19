dic_tree.obj is an object of class Tree (defined at the top of make_dic_tree.py) 

this is the tree specified on this website [http://t9.nerevar.com/?q=details](http://t9.nerevar.com/?q=details) 

instructions on how to use and import it, at bottom of make_dic_tree.py

the file words_dictionary.json is the dictionary that I used, i got it off some other github repo

t9.py has methods that convert strings of text to strings of digits from 0-9 where 1 represents space and 0 is backspace

the main method next_input takes a single base ten digit n and a list of nodes in the dictionary-trie (Trie) and returns a list of possible string representations and the corresponding list of nodes in the Trie. 

this is what is output when you run t9.py 

![t9.py output](https://github.com/NTX-McGill/NeuroTech-ML/tree/master/T9/t9pyoutput.png)

pseudocode for text prediction flow:
user preses key
program travels down the Trie in all possible ways
program gets new text prediction from api based off of all the nodes we're at (the user can chose to autocomplete with fist clench(1), different fist clench to navigate though autocomplete suggestions)
the first suggestion should be the most likely autocomplete word, the next suggestions should be the most likely words which fit the profile exactly, then after more predictions fromt he api

