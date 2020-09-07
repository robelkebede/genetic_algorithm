import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import pdb

inp = 4; hidden = 15; out=4
rate = 0.03

w1 = np.random.uniform(-1,1,(inp,hidden))
w2 = np.random.uniform(-1,1,(hidden,out))

iris = load_iris()

def one_hot(x,nb_classes):

    zeros = np.zeros(nb_classes)
    zeros[x] = 1
    return zeros


X = iris.data
#y = iris.target
y = [list(one_hot(i,4)) for i in iris.target]

#pdb.set_trace()
def sigmoid(x):
    return 1/(1+np.exp(-x))

def feedforward(x,w1,w2):

    z = np.tanh(x@w1)
    yh = sigmoid(z@w2)
    return yh

def mutate(w,rate):

    dw = np.random.uniform(-1,1,(w.shape)) * rate
    return dw + w


def train(x,y,w1,w2,rate,gen):

    err = []

    for i in range(gen):
        
        mw1 = mutate(w1,rate)
        mw2 = mutate(w2,rate)

        ff1 = feedforward(x,w1,w2) #parent
        ff2 = feedforward(x,mw1,mw2) #child

        error_parent = np.sum(np.abs(y - ff1)) 
        error_child = np.sum(np.abs(y - ff2)) 

        if error_child<error_parent:
            
            #update child error
            w1 = mw1 
            w2 = mw2
            
        print("Training Error ",error_child)  
        err.append(error_child)

    return w1,w2,err

def main():

    tw1,tw2,err = train(X,y,w1,w2,0.03,1000)

    plt.plot(err)
    plt.show()


if __name__ == "__main__":

    main()




