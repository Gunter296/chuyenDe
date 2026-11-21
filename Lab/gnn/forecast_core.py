"""Graph construction for the isolated GNN forecast experiment."""
import numpy as np


def multiresource_graph(actual, containers, train_end, top_k=2):
    n=len(containers)
    similarity=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            pairs=[(row[containers[i]],row[containers[j]])
                   for row in actual[:train_end]
                   if containers[i] in row and containers[j] in row]
            if len(pairs)<3:continue
            arr=np.asarray(pairs,dtype=float)
            scores=[]
            for r in (0,1):
                a,b=arr[:,0,r],arr[:,1,r]
                if a.std()>0 and b.std()>0:
                    scores.append(max(0,float(np.corrcoef(a,b)[0,1])))
            if scores:similarity[i,j]=similarity[j,i]=sum(scores)/len(scores)
    adjacency=np.zeros_like(similarity)
    for i in range(n):
        for j in np.argsort(-similarity[i],kind="stable")[:min(top_k,n-1)]:
            if j!=i and similarity[i,j]>0:
                adjacency[i,j]=similarity[i,j]
    return adjacency/np.maximum(adjacency.sum(axis=1,keepdims=True),1)
