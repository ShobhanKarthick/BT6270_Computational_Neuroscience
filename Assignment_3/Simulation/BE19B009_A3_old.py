"""
Comment lines with imsave 
DONT FORGET!!!
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ball = np.array(np.sign(pd.read_csv("ball.txt", header=None)))
cat  = np.array(np.sign(pd.read_csv("cat.txt", header=None)))
mona = np.array(np.sign(pd.read_csv("mona.txt", header=None)))

ball_vect = ball.flatten()
cat_vect = cat.flatten()
mona_vect = mona.flatten()


# cut_ball = cut_image(ball)
# cut_cat = cut_image(cat)
# cut_mona = cut_image(mona)

class Hopfield:
    def __init__(self,niter):
        self.V = np.zeros((9000,1))
        self.U = np.zeros((9000,1))
        self.weights = np.zeros((9000,9000))
        self.U_d = np.zeros((9000,1))
        self.rmse = np.zeros((niter,1))
        self.flag = 0 
        
    def cut_image(self, image):
        new_image = np.zeros(image.shape)
        new_image[0:40, 25:65] = image[0:40, 25:65]

        return new_image

    def weight_matrix(self):
        if self.flag == 1:
            self.weights = np.matmul(mona_vect,mona_vect.T) + np.matmul(ball_vect,ball_vect.T) + np.matmul(cat_vect,cat_vect.T)
        if self.flag == 0:
            self.weights = np.matmul(ball_vect,ball_vect.T)
        

    def damage_weights(self, prob):
        damage = np.random.choice([0, 1], size=self.weights.flatten().shape[0], p=[prob, 1 - prob])
        damage = np.reshape(damage, self.weights.shape)
        return np.multiply(damage, self.weights)

def simulation(niter, lambdas, flag, p):
    dt=1/(100)
    Hop = Hopfield(niter)
    Hop.flag = flag
    Hop.weight_matrix()
    Hop.U = np.reshape(Hop.cut_image(ball),(9000,1))
    Hop.weights = Hop.damage_weights(p)/9000
    images_arr=[]
    for i in range(niter):
        print(Hop.weights)
        Hop.U_d = -Hop.U + np.matmul(Hop.weights,Hop.V)
        Hop.U = Hop.U + (Hop.U_d)*dt
        Hop.V = np.tanh(lambdas*Hop.U)
        Hop.rmse[i]=mean_squared_error(ball_vect,Hop.V)
        
        img = np.reshape(Hop.V,(90,100))
        images_arr.append(img)
    images_arr=np.array(images_arr)
    return images_arr,Hop.rmse
    
def show(images_arr,rmse,niter,p):
    images_arr = np.array(images_arr)
    for i in range(int(niter/10)):
        plt.imshow(images_arr[10*i,:,:],'Greys_r')
        # plt.title(f'Image after {10*i} iterations for {p*100}% of weight damage')
        plt.show()
        
    plt.plot(rmse)
    plt.title(f'Plot of RMSE for {p*100}% of weight damage')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()


images_arr, rmse = simulation(50, 10, 0, 0)
show(images_arr,rmse,niter,0)

plt.figure(1)
plt.imshow(ball, cmap='Greys_r',  interpolation='nearest')
plt.savefig("Images/Q1_1.png")

plt.figure(2)
plt.imshow(cat, cmap='Greys_r',  interpolation='nearest')
plt.savefig("Images/Q1_2.png")

plt.figure(3)
plt.imshow(mona, cmap='Greys_r',  interpolation='nearest')
plt.savefig("Images/Q1_3.png")

# plt.figure(4)
# plt.imshow(cut_ball, cmap='Greys_r',  interpolation='nearest')
# plt.savefig("Images/Q2_1.png")

# plt.figure(5)
# plt.imshow(cut_cat, cmap='Greys_r',  interpolation='nearest')
# plt.savefig("Images/Q2_2.png")

# plt.figure(6)
# plt.imshow(cut_mona, cmap='Greys_r',  interpolation='nearest')
# plt.savefig("Images/Q2_3.png")

plt.show()
