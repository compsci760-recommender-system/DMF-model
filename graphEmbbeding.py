#Author Manuel Aguero
import scipy.sparse as sparse
import numpy as np
import random
import mpmath

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
        sumG += np.log(1/(1+np.exp(np.dot(-theta_u[i],np.transpose(theta_v[j])))))
    for (i,j) in S_items:
        sumS += np.log(1 - (1/(1+np.exp(np.dot(-theta_u[i],np.transpose(theta_v[j]))))) )
    return sumG + sumS
            
def update_embeddings(theta_u,theta_v,G_mini_items,S_items,a,e_ife):
    sumGu = sumSu = sumGv = sumSv = 0
    for (i,j) in G_mini_items:
        currVal = (1 - 1/(1+np.exp(-np.dot(theta_u[i],np.transpose(theta_v[j])))))
        uVal = np.dot(currVal,theta_v[j])  
        vVal = np.dot(currVal,theta_u[i])
        sumGu += uVal
        sumGv += vVal
    for (i,j) in S_items:
        currVal = (-1/(1+np.exp(-np.dot(theta_u[i],np.transpose(theta_v[j])))))
        uVal = np.dot(currVal,theta_v[j])  
        vVal = np.dot(currVal,theta_u[i])
        sumSu += uVal
        sumSv += vVal
    theta_u = theta_u + (a*sumGu + a*sumSu) #update user embeddings
    theta_v = theta_v + (a*sumGv + a*sumSv) #update item embeddings
    new_eife = compute_e_ife(theta_u,theta_v,G_mini_items,S_items) #check S: expand S or use new mini S?
    difference = new_eife - e_ife
    if e_ife > new_eife:
        theta_u = theta_u - (a*sumGu + a*sumSu) #undo
        theta_v = theta_v - (a*sumGv + a*sumSv) #undo
    else:
        e_ife = new_eife
    return (difference,e_ife)
    
def implicit_feedback_embedding(G,b,a,k,row_length,col_length):
    #initiliase list of vector embeddings randomly
    theta_u = list()
    theta_v = list()
    for i in range(row_length+1):
        theta_u.append(np.random.rand(k))
    for i in range(col_length+1):
        theta_v.append(np.random.rand(k))   
    tolerance = 5 #convergence parameter epsilon
    convergence_difference = 100 #arbitrary to initialisation
    G_items = set(zip(G.row,G.col)) #get iterator for row,col pair with non-zero values
    e_ife = None
    G = G.tocsr()
    print(theta_u[0],theta_v[0])
    while abs(convergence_difference) > tolerance: #while embedding has not converged
        print("diff > tolerance?",abs(convergence_difference),tolerance)
        G_mini = mini_batch_generator(G,batch_size,row_length,col_length)
        S = negative_sampling(G_items,row_length,col_length,batch_size) #make negative feedback graph
        G_mini = G.tocoo()
        S_items = set(zip(S.row,S.col))
        G_mini_items = set(zip(G_mini.row,G_mini.col))
        if not e_ife:
            e_ife = compute_e_ife(theta_u,theta_v,G_mini_items,S_items) #initialise e_ife mini G or G?
        convergence_difference,e_ife = update_embeddings(theta_u,theta_v,G_mini_items,S_items,a,e_ife)
    print("diff > tolerance?",convergence_difference,tolerance)
    print(theta_u[0],theta_v[0])
     
#set parameters
batch_size = 2000
learning_rate = 0.00001
dimensionality = 5
#adjacency matrix representing the implicit Feedback Graph G
G = sparse.csr_matrix((data['click_count'].astype(float), (data['user_id'], data['item_id'])))#make user/item graph
G = G[:50000,:500000]
G = G.tocoo()
row_length = max(G.row)
col_length = max(G.col)
print("embedding...")
implicit_feedback_embedding(G,batch_size,learning_rate,dimensionality,row_length,col_length)
print("done")

