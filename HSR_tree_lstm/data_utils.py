from node import tNode,processTree
import numpy as np
import os
import pickle

class Vocab(object):

    def __init__(self,path,filename):
        self.words = []
        self.word2idx={}
        self.idx2word={}
        self.hashtag2idx={}
        self.idx2hashtag={}
        self.load(path,filename)

    def load(self,path,filename):
        with open(os.path.join(path,filename),'rb') as f:
            self.word2idx,self.idx2word,self.hashtag2idx=pickle.load(f)

            self.words=[w for w in self.word2idx.keys()]
            for h,idx in self.hashtag2idx.items():
                self.idx2hashtag[idx]=h


    def __len__(self):
        return len(self.words)

    def encode(self,word):
        if type(word)==bytes:
            word=word.decode("utf-8")
        if word not in self.words:
            word="unk"
        return self.word2idx[word]

    def decode(self,idx):
        assert idx >= len(self.words)
        return self.idx2word[idx]
    def size(self):
        return len(self.words)


def load_HS_treebank(data_dir,filename):
    voc=Vocab(os.path.join(data_dir),filename)

    split_paths={}
    for split in ['train','test','dev']:
        split_paths[split]=os.path.join(data_dir,"..\\data",split)

    fnlist=[tNode.encodetokens]
    arglist=[voc.encode]
    data={}
    for split,path in split_paths.items():
        sentencepath=os.path.join(path,'sents.txt')
        treepath=os.path.join(path,'parents.txt')
        labelspath=os.path.join(path,'labels.txt')
        trees=parse_trees(sentencepath,treepath,labelspath)
        trees = [(processTree(tree,fnlist,arglist),convert_to_one_hot_vec(tree.label,voc.hashtag2idx)) for tree in trees] #??
        data[split]=trees
    return data,voc

def load_embeding_matrix(embedding_path):
    with open(embedding_path, 'rb') as f:
        embedding=pickle.load(f)
    return embedding


def convert_to_one_hot_vec(Y,hashtag_to_index):
    Y=list(Y)
    vector=np.zeros(len(hashtag_to_index))
    for j in range(len(Y)):
        if Y[j] in hashtag_to_index.keys():
            vector[hashtag_to_index[Y[j]]]=1
    return vector

def parse_trees(sentencepath, treepath,labelspath):
    trees=[]
    with open(treepath,'rb') as ft, open(labelspath,'rb') as fl , open(
        sentencepath,'rb') as f:
        while True:
            parentidxs = ft.readline()
            sentence=f.readline()
            labels=fl.readline()
            labels=[l.decode("utf-8") for l in labels.strip().split()]
            if not parentidxs  or not sentence:
                break
            parentidxs=[int(p) for p in parentidxs.strip().split() ]
            tree=parse_tree(sentence,parentidxs)
            tree.label=labels
            trees.append(tree)
    return trees



def parse_tree(sentence,parents):
    nodes = {}
    parents = [p - 1 for p in parents]  #change to zero based
    sentence=[w.decode("utf-8") for w in sentence.strip().split()]
    for i in range(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx)  
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)
                nodes[idx] = node
                if idx < len(sentence):
                    node.word = sentence[idx]
                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break
                prev = node
                idx = parent
    return root

def BFStree(root):
    from collections import deque
    node=root
    leaves=[]
    inodes=[]
    queue=deque([node])
    func=lambda node:node.children==[]
    while queue:
        node=queue.popleft()
        if func(node):
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            queue.extend(node.children)

    return leaves,inodes

def extract_tree_data(tree,max_degree=2,only_leaves_have_vals=True):
    leaves,inodes=BFStree(tree)
    leaf_emb=[]
    tree_str=[]
    i=0
    for leaf in reversed(leaves):
        leaf.idx = i
        i+=1
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx=i
        c=[child.idx for child in node.children]
        tree_str.append(c)
        i+=1
    return (np.array(leaf_emb,dtype='int32'),
           np.array(tree_str,dtype='int32'))

def extract_tree_seq_data(tree,max_degree=2,only_leaves_have_vals=True):
    leaves,inodes=BFStree(tree)
    temp=[]
    temp2=[n.idx for n in leaves]
    for i in range(len(leaves)):
        id=np.where(np.array(temp2)==i)
        id=int(id[0])
        temp.append(leaves[id])
    leaves=temp
    leaf_emb=[]
    tree_str=[]
    i=0
    for leaf in leaves:
        leaf.idx = i
        i+=1
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx=i
        c=[child.idx for child in node.children]
        tree_str.append(c)
        i+=1
    return (np.array(leaf_emb,dtype='int32'),
           np.array(tree_str,dtype='int32'))

def extract_batch_tree_data(batchdata,num_for_hs,fillnum=40):
    dim1,dim2,dim3=len(batchdata),fillnum,num_for_hs
    leaf_emb_arr = np.empty([dim1,dim2],dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim1,dim2,2],dtype='int32')
    treestr_arr.fill(-1)
    labels_arr = np.empty([dim1,dim3],dtype=float)
    seqlngths=[]
    for i,(tree,label) in enumerate(batchdata):
        assert len(label)==dim3 , "error len(label)==dim3 "
        input_,treestr=extract_tree_seq_data(tree,
                                          max_degree=2,
                               only_leaves_have_vals=False)
        seqlngths.append(len(input_))
        leaf_emb_arr[i,0:len(input_)]=input_
        treestr_arr[i,0:len(treestr),0:2]=treestr
        labels_arr[i,0:dim3]=label
    return leaf_emb_arr,treestr_arr,labels_arr, seqlngths

def extract_seq_data(batch_data,num_for_hs,numsamples=0,fillnum=40):
    dim1, dim2, dim3=len(batch_data) , fillnum , num_for_hs
    labels_arr = np.empty([dim1,dim3],dtype='int32')
    seqarr=np.empty([dim1,dim2],dtype='int32')
    seqarr.fill(-1)
    seqlngths=[]
    for i, (tree,label) in enumerate(batch_data):
        seq=extract_seq_from_tree(tree,numsamples)
        seqlngths.append(len(seq))
        assert fillnum >=len(seq)
        seqarr[i,0:len(seq)]=seq
        labels_arr[i,0:dim3]=label
    return seqarr,labels_arr,seqlngths,fillnum

def extract_seq_from_tree(tree,numsamples=0):
    if tree.span is None:
        tree.postOrder(tree,tree.get_spans)
    seq=[]
    seq.extend(tree.span)
    if not numsamples:
        return seq
    return seq

def get_max_len_data(datadic):
    maxlen=0
    for data in datadic.values():
        for tree,_ in data:
            tree.postOrder(tree,tree.get_numleaves)
            assert tree.num_leaves > 1
            if tree.num_leaves > maxlen:
                maxlen=tree.num_leaves

    return maxlen

def get_max_node_size(datadic):
    maxsize=0
    for data in datadic.values():
        for tree,_ in data:
            tree.postOrder(tree,tree.get_size)
            assert tree.size > 1
            if tree.size > maxsize:
                maxsize=tree.size
    return maxsize
