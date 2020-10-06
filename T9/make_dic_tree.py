import json
import pickle

class Tree:
    def __init__(self,data,parent):
        self.children = []
        self.parent = parent
        self.data = data


# %% some helpful methods that anyone can use import
def get_current_string(node):
    #takes node (tree boject), retuns string that it stands for"
    nd=node
    current_str=""
    while nd.data[0]!="root":
        current_str = nd.data[0]+current_str
        nd = nd.parent
    return current_str

def get_node_at(string,root):
    # takes a string and tells you if there is a node which represents that string
    s = string[:]
    node=root
    while s:
        if s[0] in [i.data[0] for i in node.children]:
            node = [i for i in node.children if i.data[0]==s[0]][0]
            s=s[1:]
        else: return None
    return node

# main reccursive tree building method
def build_tree(node,words_reduced):
    """
    for each letter in 'qwertyuiopasdfghjklzxcvbnm'
    append letter to node
    check if it's a valid word, if it is, set is_word=True
    check if there are valid words starting with this root
    if not, init as leaf iff is_word=True
    if yes, init new node proprely and call build_tree recursively
    """
    for l in 'abcdefghijklmnopqrstuvwxyz':
        current_str = get_current_string(node)
        new_str = current_str+l
        words_more_reduced = [w for w in words_reduced if new_str==w[:len(new_str)]]
        if words_more_reduced:
            is_word,is_not_last_node=0,0
            if new_str in words_reduced:is_word=1
            n_children = len(words_more_reduced)-is_word
            if n_children != 0:is_not_last_node=1
            # initialize the node
            new_node = Tree(data=[l,is_word,is_not_last_node,n_children],parent=node)
            node.children.append(new_node)
            if n_children != 0:
                build_tree(new_node,words_more_reduced)
    return

if __name__=="__main__":
    # import dictionary 
    words = [w for w in json.load(open('words_dictionary.json','r')).keys()]
    
    # build the tree using imported dictionary
    root =Tree(data=['root',0,1,len(words)],parent=None)
    build_tree(root,words)

    # pickle it - save it as binary file
    file_obj = open('dic_tree.obj','wb')
    pickle.dump(root,file_obj)
    file_obj.close()

    # %% example usage of importing and using methods
    
    # import the pickled tree (if you do this in another file you need to define the Tree class, see top of this script, it's only 5 lines of code!)
    file_obj = open('dic_tree.obj','rb')
    root = pickle.load(file_obj)
    file_obj.close()
    
    print("example usage of get_node_at method")   
    node = get_node_at('antithes',root)
    try:
        print(node.data)
        print("children : {}".format([i.data[0] for i in node.children]))
    except:
        print(node)
    
    print("bye!")
