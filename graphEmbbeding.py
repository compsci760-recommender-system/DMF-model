import scipy.sparse as sparse
import numpy as np
import random

def mini_batch_generator(graph,batch_size,row_length,col_length):
    i = random.randint(0, row_length-batch_size)
    j = random.randint(0,col_length-batch_size)
    return graph[i:i+batch_size,j:j+batch_size]

def negative_sampling(E,row_length,col_length,batch_size):
    S = sparse.dok_matrix((row_length+1,col_length+1)) #create empty negtive feedack graph
    for i in range(batch_size):
        a = random.randint(0, row_length)
        b = random.randint(0,col_length)
        randomEdge = (a,b) #randomly sample edges from the implicit feedback graph
        if (a,b) not in E:
            S[a,b]=1
    return S.tocoo()

def compute_e_ife(theta_u,theta_v,G_items,S_items):
    sumG = sumS = 0
    for (i,j) in G_items:
        currVal = np.dot(theta_u[i],np.transpose(theta_v[j]))
        h = 1/(1+np.exp(-currVal))
        sumG += np.log(h)
    for (i,j) in S_items:
        currVal = np.dot(theta_u[i],np.transpose(theta_v[j]))
        h = 1/(1+np.exp(-currVal))
        sumS += np.log(1-h)
    return sumG + sumS
            
def update_embeddings(theta_u,theta_v,G_items,S_items,a):
    sumGu = sumSu = sumGv = sumSv = 0
    for (i,j) in G_items:
        currVal = np.dot(theta_u[i],np.transpose(theta_v[j]))
        currVal = 1/(1+np.exp(-currVal))
        currVal = (1 - currVal)
        uVal = np.dot(currVal,theta_v[j])  
        vVal = np.dot(currVal,theta_u[i])
        sumGu += uVal
        sumGv += vVal
    for (i,j) in S_items:
        currVal = np.dot(theta_u[i], np.transpose(theta_v[j]))
        currVal = 1/(1+np.exp(-currVal))
        currVal = (-currVal)
        uVal = np.dot(currVal,theta_v[j])  
        vVal = np.dot(currVal,theta_u[i])
        sumSu += uVal
        sumSv += vVal
    theta_u = theta_u + (a*sumGu + a*sumSu) #update user embeddings
    theta_v = theta_v + (a*sumGv + a*sumSv) #update item embeddings
    
def implicit_feedback_embedding(G,S,b,a,k,row_length,col_length):
    #initiliase list of vector embeddings randomly
    theta_u = list()
    theta_v = list()
    for i in range(row_length+1):
        theta_u.append(np.random.rand(k))
    for i in range(col_length+1):
        theta_v.append(np.random.rand(k))   
    tolerance = 2 #convergence parameter epsilon
    convergence_difference = 5
    G_items = set(zip(G.row,G.col)) #get iterator for row,col pair with non-zero values
    S = sparse.dok_matrix((row_length+1,col_length+1)) #create empty negtive feedack graph
    #e_ife = compute_e_ife(theta_u,theta_v,G,S) #initialise e_ife
    G = G.tocsr()
    
    while abs(convergence_difference) > tolerance: #while embedding has not converged
        G_mini = mini_batch_generator(G,batch_size,row_length,col_length)
        S = negative_sampling(G_items,row_length,col_length,batch_size) #make negative feedback graph
        #print(G_mini.count_nonzero())
        #print(S.count_nonzero())
        G_mini = G.tocoo()
        #S.tocoo()
        S_items = set(zip(S.row,S.col))
        G_mini_items = set(zip(G_mini.row,G_mini.col))
        update_embeddings(theta_u,theta_v,G_mini_items,S_items,a)
        #new_ife = compute_e_ife(theta_u,theta_v,G,S)
        convergence_difference -= 2
     
#set parameters
batch_size = 20
learning_rate = 1
dimensionality = 5
#adjacency matrix representing the implicit Feedback Graph G
G = sparse.coo_matrix((data['click_count'].astype(float), (data['user_id'], data['item_id'])))
row_length = max(G.row)
col_length = max(G.col)
print("embedding...")
implicit_feedback_embedding(G,S,batch_size,learning_rate,dimensionality,row_length,col_length)
print("done")

