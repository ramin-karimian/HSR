# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:40:32 2019

@author: ramin_karimian
"""

from stanfordnlp.server import CoreNLPClient
import os
import pickle 

# get data
package=""
version=""
# version="version_02(without_url_from_scratch)"
num_for_hs=30

datapath=f"HSR_tree_lstm\\pickles\\updated_preprocessed_dataset_{package}_(for_mon_4_to_10)_{version}_hastags{num_for_hs}.pkl"
with open(datapath,"rb") as f:
    data=pickle.load(f)


# set up the client
print('---')
print('starting up Java Stanford CoreNLP Server...')
os.environ["CORENLP_HOME"]="C:/Users/RAKA/Downloads/Compressed/stanford/stanford-corenlp-full-2018-10-05/stanford-corenlp-full-2018-10-05"


# set up the client
errList={"dot":[],
         "len":[]}
with CoreNLPClient(annotators=["tokenize","ssplit",'pos','parse'], timeout=100000,threads=10,output_format="json",max_char_length=1000000) as client:
    # submit the request to the server
    for i in range(len(data)):
        if "." in data[i]["tokens"]:
            errList["dot"].append((i,data[i]["tokens"]))
            continue
        if " ".join(data[i]["tokens"])[-1]!=".":
            text=" ".join(data[i]["tokens"])+" ."
            data[i]["tokens"].append(".")
        else:
            text=" ".join(data[i]["tokens"])
        if i%1000==0:
            print(f" for i: {i} before request")
        ann = client.annotate(text)
        if i%1000==0:
            print(f" for i: {i} after request")
        if len(ann["sentences"]) > 1:
            errList["len"].append((i,data[i]["tokens"]))
            continue
        else:
            data[i]["stanford_doc"]=ann["sentences"][0]["parse"]
    with open(f"HSR_tree_lstm\\pickles\\CoreNLPClient_updated_preprocessed_dataset_{package}_(for_mon_4_to_10)_{version}_hastags{num_for_hs}.pkl","wb") as f:
        pickle.dump(data,f) 
