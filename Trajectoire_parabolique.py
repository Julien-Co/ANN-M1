import numpy as np
import numpy.random as r
from numpy import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
Ny = 95
Nx = 45
y= np.zeros((1000,2))
theta0 = np.zeros((1000,))
theta0_rel = np.zeros((1000,))
v0 = np.zeros((1000,))
v0_rel = np.zeros((1000,))
X_data = np.zeros((1000, 80))
def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches
def data():
    v = random.uniform(20,30) 
    Traj = np.zeros(80,)
    z = random.uniform(30,60) 
    g = 9.81
    theta =z/180.0*np.pi
    for x in range(0,80,1):
        y = -(g * x * x) / (2 * (v * np.cos(theta)) * (v * np.cos(theta))) + x * np.tan(theta)
        if y >=0:
            Traj[int(x)]= y
        if y<0:
            Traj[int(x)]= 0
    return v,z,Traj
for i in range(1000):
    v0[i],theta0[i], X_data[i] = data()
    v0_rel[i] = (v0[i]-19)/12
    theta0_rel[i] = ((theta0[i]/5)-6)/7
    y[i,0] = v0_rel[i]
    y[i,1] = theta0_rel[i]
X_scale = StandardScaler()
X = X_scale.fit_transform(X_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
nn_structure = [80, 50, 2]
def f(x):
    return 1 / (1 + np.exp(-x))
def f_prime(x): 
    return f(x) * (1 - f(x))
def setup_weights_bias(nn_structure):
    W= {}
    b= {}
    for l in range(1, len(nn_structure)): 
        W[l]= r.random_sample((nn_structure[l], nn_structure[l-1])) 
        b[l]= r.random_sample((nn_structure[l],)) 
    return W, b
def setup_Deltas(nn_structure):
    delta_W = {}
    delta_b = {}
    for l in range(1, len(nn_structure)): 
        delta_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        delta_b[l] = np.zeros((nn_structure[l],)) 
    return delta_W, delta_b
def calcul_nodes(X, W, b):
    h = {1: X}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            node_in = X
        else:
            node_in = h[l]
        z[l+1] = np.dot(W[l],node_in) + b[l]
        h[l+1] = f(z[l+1])
    return h, z
def run_nn(nn_structure, X, y, bs=100, iter_num=3000, alpha=0.25, beta=0.0001):
    W, b= setup_weights_bias(nn_structure)
    cnt=0
    m= len(y)
    avg_cost_func= []
    print("Lancement du programe pour {} itérations".format(iter_num))
    while cnt < iter_num:
        if cnt % 300 ==0:
            print("{} out of {}".format(cnt, iter_num))
        delta_W, delta_b = setup_Deltas(nn_structure)
        avg_cost = 0
        mini_batches = get_mini_batches(X, y, bs)
        for mb in mini_batches:
            X_mb = mb[0]
            y_mb = mb[1]
            for i in range(len(y_mb)):
                delta = {}
                h, z = calcul_nodes(X_mb[i, :], W, b) #feedforward
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        avg_cost += np.linalg.norm((y_mb[i,:] - h[l]))
                        delta[l] = -(y_mb[i,:]-h[l]) * f_prime(z[l])
                    else:
                        if l > 1:
                            delta[l] = np.dot(np.transpose(W[l]), delta[l+1]) * f_prime(z[l])
                        delta_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                        delta_b[l] += delta[l+1]
                for l in range(len(nn_structure) - 1, 0, -1):
                    W[l] += -alpha * (1/bs* delta_W[l] +  beta * W[l])
                    b[l] += -alpha * (1/bs *delta_b[l])
        avg_cost = (1.0 / m )* avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func
W, b, avg_cost_func = run_nn(nn_structure, X_train, y_train)
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,2))
    for i in range(m):
        h, z = calcul_nodes(X[i, :], W, b)
        y[i] = h[n_layers]  
    return y
y_pred = predict_y(W, b, X_test, 3)
a=0
for i in range(len(y_test)):
    a += (np.absolute((y_test[i,0]-y_pred[i,0])) + np.absolute((y_test[i,1]-y_pred[i,1])))/2
pourcent = (a/len(y_test))*100
print("{} %".format(pourcent))