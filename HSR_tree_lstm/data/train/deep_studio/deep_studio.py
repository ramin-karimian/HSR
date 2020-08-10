# -*- coding: utf-8 -*-

with open("sents.txt","r") as f,\
    open("labels.txt","r") as f2,\
    open("train.csv","w") as f1:
#        while
        lines=f.readlines()
        labels=f2.readlines()
        for  i in range(len(lines)):
            f1.write(lines[i][:-1]+","+labels[i].split()[0]+"\n")
    