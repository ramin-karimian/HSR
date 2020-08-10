class tNode(object):
    def __init__(self,idx=-1,word=None):
        self.left = None #node class object
        self.right = None #node class object 
        self.word = word # "string"
        self.size = 0 # int
        self.height = 1 # int
        self.parent = None #node class object
        self.label = None #int
        self.children = [] # list of nodes
        self.idx=idx #int
        self.span=None # list of words string ["I","am",..]

    def add_parent(self,parent):
        self.parent=parent
    def add_child(self,node):
        assert len(self.children) < 2
        self.children.append(node)
    def add_children(self,children):
        self.children.extend(children)

    def get_left(self):
        left = None
        if self.children:
            left=self.children[0]
        return left
    def get_right(self):
        right = None
        if len(self.children) == 2:
            right=self.children[1]
        return right

    @staticmethod
    def get_height(root):
        if root.children:
            root.height = max(root.get_left().height,root.get_right().height)+1
        else:
            root.height=1

    @staticmethod
    def get_size(root):
        if root.children:
            root.size = root.get_left().size+root.get_right().size+1
        else:
            root.size=1

    @staticmethod
    def get_spans(root):
        if root.children:
            root.span=root.get_left().span+root.get_right().span
        else:
            root.span=[root.word]

    @staticmethod
    def get_numleaves(self):
        if self.children:
            self.num_leaves=self.get_left().num_leaves+self.get_right().num_leaves
        else:
            self.num_leaves=1

    @staticmethod
    def postOrder(root,func=None,args=None):

        if root is None:
            return
        tNode.postOrder(root.get_left(),func,args)
        tNode.postOrder(root.get_right(),func,args)

        if args is not None:
            func(root,args)
        else:
            func(root)

    @staticmethod
    def encodetokens(root,func):
        if root is None:
            return
        if root.word is None:
            return
        else:
            root.word=func(root.word)



def processTree(root,funclist=None,argslist=None):

    if funclist is None:
        root.postOrder(root,root.get_height)
        root.postOrder(root,root.get_num_leaves)
        root.postOrder(root,root.get_size)
    else:
        for func,args in zip(funclist,argslist):
            root.postOrder(root,func,args)
    return root
