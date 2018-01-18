import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from math import log
import matplotlib.pyplot as plt


data = pd.read_csv("2015EE10466.csv")
data1 = pd.read_csv("2015EE10466.csv")
print(data.describe())
labels = list(data['pixel784'])
data = data.iloc[0:,:784]
data1 = np.array(data1.iloc[0:,:784])

pca = PCA(n_components=10)
data = pca.fit_transform(data)
data = np.array(data)
print('++++++++data shape+++++++')
print(data.shape)

clf = KMeans(n_clusters=10, n_init=10, max_iter=100)
clf.fit(data)
pred = clf.predict(data)
print(pred)

cluster_dict ={}

for i,val in enumerate(pred):
    if(val not in cluster_dict.keys()):
        cluster_dict[val]= []
    cluster_dict[val] = cluster_dict[val] + [data[i]]


mean = []
for i in clf.cluster_centers_:
    mean = mean+[i]
mean = np.array(mean)

sigma =[]
for i in cluster_dict.keys():
    a = np.array(cluster_dict[i])
    sigma = sigma+[np.cov(a.T)]
sigma = np.array(sigma)

p_cluster = []
for i in cluster_dict.keys():
    p_cluster = p_cluster + [len(cluster_dict[i])/3000]

print("P_cluster")
print(p_cluster)
prob_mat = []
Y_axis = []
iterations = 10
for _ in range(iterations):
    print('iteration:',_)
    prob_mat = []
    gm = []
    for i in range(10):
        gm = gm + [multivariate_normal(mean=mean[i], cov=sigma[i])]

    for i in data:
        prob = []
        for j in range(10):
            prob = prob + [gm[j].pdf(i) * p_cluster[j]]
        prob = [k / sum(prob) for k in prob]
        prob_mat = prob_mat + [prob]
    #3000 X 10 matrix with probabilities of each point in each cluster
    prob_mat = np.array(prob_mat)
    print('### W shape####')
    print(prob_mat.shape)

    p_cluster = (1.0 / 3000) * prob_mat.sum(axis=0)
    print('###sum of all prob ###')
    print(sum(p_cluster))

    mean = np.dot(prob_mat.T, data)
    # print(mean)

    for i in range(10):
        mean[i] = mean[i] / (3000 * p_cluster[i])
    print('##############mean  ####')
    print(mean.shape)
    # first row of mean corresponds mean of first cluster


    for j in range(10):

        sig = 0
        for i in range(3000):
            X = np.subtract(data[i], mean[j])

            X= np.reshape(X,(1,10))
            #print(X.shape)
            #print(np.dot(X.T, X))

            sig = sig + prob_mat[i][j] * np.dot(X.T, X)
        sig = (1/( (3000 * p_cluster[j])))*sig
        #print(sig)
        sigma[j] = sig
    print('##############sigma  ####')
    print(sigma.shape)
    #calculating P(X)

    likelihood = []

    for i in range(3000):
        l = 0
        for j in range(10):
            l = l + prob_mat[i][j]*p_cluster[j]
        likelihood = likelihood +[log(l,10)]
    Y_axis = Y_axis + [sum(likelihood)]

plt.scatter(range(iterations),Y_axis)
plt.plot(range(iterations),Y_axis)
#plt.show()

print(prob_mat[0])
print(prob_mat.sum(axis=1))
print(list(prob_mat[0]).index(max(prob_mat[0])))



cluster_dict ={}
for i in range(10):
    cluster_dict[i] = []

for i in range(3000):

    cluster_dict[list(prob_mat[i]).index(max(prob_mat[i]))] = cluster_dict[list(prob_mat[i]).index(max(prob_mat[i]))] + [i]

print(cluster_dict.keys())

sum1 = 0
cluster_id = {}
for j in cluster_dict.keys():

    lis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cluster_dict[j]:
        lis[labels[i]] = lis[labels[i]]+1
    print('cluster:', lis.index(max(lis)))
    cluster_id[j] =lis.index(max(lis))
    print(lis)
    #print(sum(lis),'sum')
   # print(max(lis))
    sum1 = sum1 + sum(lis) -max(lis)

    #print(sum1)
   # print(sum(lis)-max(lis))

print(sum1/3000,'error')

for i in range(prob_mat.shape[0]):

    s = 0
    title = []

    for j in range(10):
        if prob_mat[i][j] > 0.15:
            s= s+1
            title = title + [cluster_id[j]]


    if s>2:
        #print(i)

        #print(data1[i])
        img = data1[i]
        img = np.reshape(img,(28,28))
        plt.imshow(img)
        plt.title(title+['True Label:',labels[i]])
        plt.show()
