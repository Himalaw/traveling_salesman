#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[2]:


nmax = 130

def convertToDataframe(csv):
    n = len(csv)
    strNodes = str(n)

    df = pd.DataFrame(index=range(0,n), columns=['node','x', 'y'])
    for i in range(n):
        node = read[strNodes][i]
        splits = node.split()
        df['node'][i] = int(splits[0])
        df['x'][i] = float(splits[1])
        df['y'][i] = float(splits[2])
    return df

def randomDataframe(n):
    strN = str(n)
    df = pd.DataFrame(index=range(0,n), columns=['node','x', 'y'])
    for i in range(n):
        df['node'][i] = i + 1
        df['x'][i] = random.uniform(0, 50)
        df['y'][i] = random.uniform(0, 50)
    return df

def distanceEuclid(x1, x2, y1, y2):
    x = (x2 - x1)**2
    y = (y2 - y1)**2
    return (np.sqrt(x + y))

def matEuclid(df):
    l = len(df)
    mat = np.zeros((l,l),dtype = float)
    for i in range(l):
        for j in range(l):
            mat[i][j] = distanceEuclid(df['x'][i], df['x'][j], df['y'][i],df['y'][j])
    return mat


# In[3]:


def PPV(n, depot, mat):
    L = 0
    T = []
    done = []
    inf = sum(sum(line) for line in mat) + 1
    for i in range(n):
        done.append(False)
    T.append(depot)
    
    dmin = inf
    currentNode = depot
    while(done[currentNode] == False):
        for k in range(n):
            if (currentNode == k):
                continue
            if (done[k] == False and mat[currentNode][k] < dmin):
                dmin = mat[currentNode][k]
                imin = k
        if (dmin != inf):
            L = L + dmin
            T.append(imin)
        done[currentNode] = True
        currentNode = imin
        dmin = inf
    T.append(depot)
#     T = [x + 1 for x in T]
    L = L + mat[currentNode][depot]
    return { 'node': T, 'distance': L }


# In[4]:


def PLI(n, depot, mat):
    L = 0
    T = [depot, depot]
    
    for k in range((len(T) - 1), n):
        dmax = 0
        for visitedNode in T:
            for node in range(n):
                if (node not in T and mat[visitedNode][node] > dmax):
                    dmax = mat[visitedNode][node]
                    imax = node
        deltaMin = 100000
        for j in range(k):
            d1 = mat[T[j]][imax]
            d2 = mat[imax][T[j + 1]]
            d3 = mat[T[j]][T[j + 1]]
            delta = d1 + d2 - d3
            if (delta < deltaMin):
                deltaMin = delta
                jmin = j
        L = L + deltaMin
        T.insert(jmin + 1, imax)
#     T = [x + 1 for x in T]
    return { 'node': T, 'distance': L }


# In[5]:


def deltaVar(mat, n1, n2, n3, n4):
    return float(mat[n1][n3] + mat[n2][n4] - mat[n1][n2] - mat[n3][n4])


# In[6]:


def RL(n, depot, mat, initSol, version):
    T = initSol['node']
    L = initSol['distance']
    
#     T = [x - 1 for x in T]
    
    fi = False
    deltaMin = -1
    while deltaMin != 0:
        deltaMin = 0
        for i in range(2, n - 1):
            for j in range(i + 1, n):
                delta = deltaVar(mat, T[i - 1], T[i], T[j], T[j + 1])
                if delta < deltaMin:
                    deltaMin = delta
                    imin = i
                    jmin = j
                    if version == 'FI':
                        fi = True
                        break
            if fi == True:
                break
        if deltaMin < 0:
            while imin < jmin:
                tmp = T[jmin]
                T[jmin] = T[imin]
                T[imin] = tmp
                imin = imin + 1
                jmin = jmin - 1
            L = L + deltaMin
        fi = False
#     T = [x + 1 for x in T]
    return { 'node': T, 'distance': L }


# In[7]:


def getDistance(TP, mat):
    L = 0
    l = len(TP)
    for i in range(l - 1):
        L = L + mat[TP[i]][TP[i + 1]]
    return float(L)
        

def ILS(n, depot, mat, niter, nech):
    
    ppv = PPV(n, depot, mat)
    initSol = RL(n, depot, mat, ppv, 'BI')
    T = initSol['node']
    L = initSol['distance']

    for iteration in range(niter):
        TP = list(T)
        for ech in range(nech):
            i = 0
            j = 0
            while i == j:
                i = random.randint(1, n - 1)
                j = random.randint(1, n - 1)
            tmp = TP[i]
            TP[i] = TP[j]
            TP[j] = tmp
             
        LP = getDistance(TP, mat)

        newSol = { 'node': TP, 'distance': LP }
        res = RL(n, depot, mat, newSol, 'BI')

        Tprime = res['node']
        Lprime = res['distance']

        if Lprime < L:
            L = Lprime
            T = list(Tprime)

    T = [x + 1 for x in T]
    return { 'node': T, 'distance': L }


# In[8]:


read = pd.read_csv('./test-rd100.txt')
df = convertToDataframe(read)
df
mat = matEuclid(df)

n = len(read)

# pli = PPV(n, 0, mat)
# res = RL(n, 0, mat, pli, 'FI')
# print(res)
res = ILS(n, 0, mat, 200, 3)
print(res)
#df


# In[9]:


def generator(iteration):
    n = 50
    depot = 0
    niter = 200
    heuristics = ['PPV', 'PLI', 'MHI_MRL', 'ILS1', 'ILS2', 'ILS3']
    tab = pd.DataFrame(0, index=range(0, iteration + 2), columns=heuristics)
    for k in range(iteration):
        
        df = randomDataframe(n)
        mat = matEuclid(df)
        _PPV = PPV(n, depot, mat)
        _PLI = PLI(n, depot, mat)
        _MHI_MRL = RL(n, depot, mat, _PPV, 'BI')
        _ILS1 = ILS(n, depot, mat, niter, 1)
        _ILS2 = ILS(n, depot, mat, niter, 2)
        _ILS3 = ILS(n, depot, mat, niter, 3)
        
        ppv = tab['PPV'][k] = _PPV['distance']
        pli = tab['PLI'][k] = _PLI['distance']
        mhi_mrl = tab['MHI_MRL'][k] = _MHI_MRL['distance']
        ils1 = tab['ILS1'][k] = _ILS1['distance']
        ils2 = tab['ILS2'][k] = _ILS2['distance']
        ils3 = tab['ILS3'][k] = _ILS3['distance']
        
        distances = [ppv, pli, mhi_mrl, ils1, ils2, ils3]
        best = min(distances)
        bestIndex = distances.index(best)
        tab[heuristics[bestIndex]][iteration + 1] = tab[heuristics[bestIndex]][iteration + 1] + 1

    
    total = 0
    for i in range(iteration):
        total = total + tab['PPV'][i]
    tab['PPV'][iteration] = total / iteration
    
    total = 0
    for i in range(iteration):
        total = total + tab['PLI'][i]
    tab['PLI'][iteration] = total / iteration
    
    total = 0
    for i in range(iteration):
        total = total + tab['MHI_MRL'][i]
    tab['MHI_MRL'][iteration] = total / iteration
    
    total = 0
    for i in range(iteration):
        total = total + tab['ILS1'][i]
    tab['ILS1'][iteration] = total / iteration
    
    total = 0
    for i in range(iteration):
        total = total + tab['ILS2'][i]
    tab['ILS2'][iteration] = total / iteration
    
    total = 0
    for i in range(iteration):
        total = total + tab['ILS3'][i]
    tab['ILS3'][iteration] = total / iteration
    
    return tab


# In[11]:


res = generator(100)
print(res)


# In[ ]:




