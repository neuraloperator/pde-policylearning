import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from libs.utilities3 import LpLoss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
from functools import reduce
from functools import partial
import operator
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter
import deepxde as dde


ndata = 1000

X = 1
dx = 0.001
nx = int(round(X/dx))
spatial = np.linspace(dx, X, nx)
T = 2
dt = 0.0001
nt = int(round(T/dt))
temporal = np.linspace(0, T, nt)

# Parameters
epochs =500
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5
learning_rate = 0.001
step_size= 50
modes=12
width=32

X = 1
dx = 0.001
nx = int(round(X/dx))
grid = np.linspace(0, X, nx, dtype=np.float32).reshape(nx, 1)
grid = torch.from_numpy(np.array(grid)).cuda()

def solveThetaFunction(x, gamma):
    theta = np.zeros(nx)
    for idx, val in enumerate(x):
        theta[idx] = 5*math.cos(gamma*math.acos(val))
    return theta
        
def solveKernelFunction(theta):
    kappa = np.zeros(nx)
    for i in range(0, nx):
        kernelIntegral = 0
        for j in range(0, i):
            kernelIntegral += (kappa[i-j]*theta[j])*dx
        kappa[i] = kernelIntegral  - theta[i]
    return np.flip(kappa)
        
x = []
y = [] 
gammaArr= []
for i in range(ndata):
    gamma = np.random.uniform(2, 10)
    theta = solveThetaFunction(spatial, gamma)
    kappa = solveKernelFunction(theta)
    gammaArr.append(gamma)
    x.append(theta)
    y.append(kappa)
    
x= np.array(x, dtype=np.float32)
y= np.array(y, dtype=np.float32)
import pdb; pdb.set_trace()
np.savetxt("x.dat", x)
np.savetxt("y.dat", y)
np.savetxt("gamma.dat", gammaArr)
x = x.reshape(x.shape[0], x.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# Create train/test splits
x = np.loadtxt("data/b_to_k/Dataset/x.dat", dtype=np.float32)
y = np.loadtxt("data/b_to_k/Dataset/y.dat", dtype=np.float32)
gammaArr = np.loadtxt("data/b_to_k/Dataset/gamma.dat", dtype=np.float32)
y = y.reshape(y.shape[0], y.shape[1])
x = x.reshape(x.shape[0], x.shape[1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# Define a sequential torch network for batch and trunk. Can use COV2D which we will show later
dim_x = 1
m = 1000
model = dde.nn.DeepONetCartesianProd([m, 512, 256], [dim_x, 128, 256], "relu", "Glorot normal").cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss = LpLoss()
train_lossArr = []
test_lossArr = []
time_Arr = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model((x, grid))
        out = out.reshape((out.shape[0], out.shape[1]))        
        
        lp = loss(out.view(batch_size, -1), y.view(batch_size, -1))
        lp.backward()
        
        optimizer.step()
        train_loss += lp.item()
        
    scheduler.step()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in testData:
            x, y = x.cuda(), y.cuda()
            
            out = model((x, grid))
            out = out.reshape((out.shape[0], out.shape[1]))
            test_loss += loss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            
    train_loss /= len(trainData)
    test_loss /= len(testData)
    
    train_lossArr.append(train_loss)
    test_lossArr.append(test_loss)
    
    t2 = default_timer()
    time_Arr.append(t2-t1)
    if ep%50 == 0:
        print(ep, t2-t1, train_loss, test_loss)
        
        
# # Display Model Details
# plt.figure()
# plt.plot(train_lossArr, label="Train Loss")
# plt.plot(test_lossArr, label="Test Loss")
# plt.yscale("log")
# plt.legend()

testLoss = 0
trainLoss = 0
with torch.no_grad():
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        out = out.reshape((out.shape[0], out.shape[1]))
        trainLoss += loss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        
    for x, y in testData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        out = out.reshape((out.shape[0], out.shape[1]))
        testLoss += loss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
    
print("Avg Epoch Time:", sum(time_Arr)/len(time_Arr))
print("Final Testing Loss:", testLoss/len(testData))
print("Final Training Loss:", trainLoss/len(trainData))