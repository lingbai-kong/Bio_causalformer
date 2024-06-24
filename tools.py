import sys
# This code snippet is sourced from the TCDF project by M-Nauta.
# Original code: https://github.com/M-Nauta/TCDF
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0

import pandas as pd
import numpy as np
import networkx as nx
import copy
from sklearn.cluster import KMeans

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships."""
    gtdata = pd.read_csv(gtfile, header=None)

    readgt=dict()
    effects = gtdata[1]
    causes = gtdata[0]
    delays = gtdata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k]=[]
    for i in range(len(effects)):
        key=effects[i]
        value=causes[i]
        readgt[key].append(value)
        pairdelays[(key, value)]=delays[i]
        gtnrrelations+=1
    
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                #find extended delay
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    delay=0
                    for i in range(len(p)-1):
                        delay+=pairdelays[(p[i+1], p[i])]
                    extendedgtdelays[(effect, cause)].append(delay)
    return extendedgtdelays, readgt, extendedreadgt
def evaluate(gtfile, validatedcauses, columns):
    """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
    
    def serialization(data):
        return [f"{e[1]}->{e[0]}" for e in data]
    print(f"Total False Positives': {FP}")
    print(f"Total True Positives': {TP}")
    print(f"Total False Negatives: {FN}")
    print(f"Total Direct False Positives: {FPdirect}")
    print(f"Total Direct True Positives: {TPdirect}")
    print(f"TPs': {serialization(TPs)}")
    print(f"FPs': {serialization(FPs)}")
    print(f"TPs direct: {serialization(TPsdirect)}")
    print(f"FPs direct: {serialization(FPsdirect)}")
    print(f"FNs: {serialization(FNs)}")
    precision = recall = 0.

    print('(includes direct and indirect causal relationships)')
    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    print(f"Precision': {precision}")
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    print(f"Recall': {recall}")
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    print(f"F1' score: {F1}")

    print('(includes only direct causal relationships)')
    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    print(f"Precision: {precision}")
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    print(f"Recall: {recall}")
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    print(f"F1 score: {F1direct}")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct
def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
            else:
                next
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)

def mainEval(ans,gt,columns):
    # causal realtions (causal graph edge)
    print("===================Results===================")
    causalG=[]
    for e in ans:
        causalG.append((columns[e[0]],columns[e[1]],e[2]))
        print(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

    # evaluate
    allcauses={i:[] for i in range(len(columns))}
    alldelays={}
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]
    if gt:
        print("===================Evaluation===================")
        eval(gt, allcauses, alldelays, columns)
def eval(gt, allcauses, alldelays, columns):
    FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(gt, allcauses, columns)
    # evaluate delay discovery
    extendeddelays, readgt, extendedreadgt = getextendeddelays(gt, columns)
    percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1)*100
    print(f"Percentage of delays that are correctly discovered: {percentagecorrect}%")

def readData(data_dir):
    df_data = pd.read_csv(data_dir)
    data = df_data.values.astype('float32')
    columns = list(df_data.columns)
    return data,columns

# relA, 每一行都表示变量的causes (i,j)表示j→i
def cluster(relA, m=1, n=2):
    estimator = KMeans(n_clusters=n)
    ans = []
    # find causes of series i
    for i,relAi in enumerate(relA):
        if relAi.sum()==0.0: # all the weights to series i are zero
            continue
        data=np.array(relAi)
        estimator.fit(data.reshape(-1,1))
        cluster_labels = estimator.labels_
        cluster_centers = estimator.cluster_centers_
        cluster_centers = cluster_centers.reshape(-1)
        largest_m_clusters = np.argsort(cluster_centers)[-m:]
        for j in range(len(relAi)):
            if cluster_labels[j] in largest_m_clusters:
                ans.append((j,i,1))
    return ans

if __name__ == '__main__':
    ans=[[0,1,0],[1,1,0],[2,0,0]]
    gt='/remote-home/share/dmb_nas/konglingbai/causality/data/demo/demo_groundtruth.csv'
    columns=[0,1,2,3,4]
    mainEval(ans,gt,columns)
