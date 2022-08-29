import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ball = np.array(np.sign(pd.read_csv("ball.txt", header=None)))
cat  = np.array(np.sign(pd.read_csv("cat.txt", header=None)))
mona = np.array(np.sign(pd.read_csv("mona.txt", header=None)))

size=ball.shape[0]*ball.shape[1]
ball_vect = np.reshape(ball,(size,1))
cat_vect = np.reshape(cat,(size,1))
mona_vect = np.reshape(mona,(size,1))


class Hopfield:
    def __init__(self,niter):
        self.V = np.zeros((9000,1))
        self.U = np.zeros((9000,1))
        self.weights = np.zeros((9000,9000))
        self.U_d = np.zeros((9000,1))
        self.rmse = np.zeros((niter,1))
        self.flag = 0 
        
    def weight_matrix(self):
        '''
        loads all images
        '''
        if self.flag==1:
            print('Loading all images')
            self.weights = np.matmul(mona_vect,mona_vect.T) + np.matmul(ball_vect,ball_vect.T) + np.matmul(cat_vect,cat_vect.T)
        if self.flag==0:
            print('Loading the image of the ball')
            self.weights = np.matmul(mona_vect,mona_vect.T)
        
    def cut_image(self,image):
        '''
        Loads patches of images
        '''
        new_image = np.zeros((90,100))
        new_image[0:40, 25:65] = image[0:40, 25:65]
        return new_image
        
    def damage_weights(self,p):
        '''
        Damages the weights of the network with probability p
        '''
        indices = np.random.randint(0,9000*9000-1,int(9000*9000*p))
        weights_damaged=np.copy(self.weights)
        weights_damaged=np.reshape(weights_damaged,(9000*9000,1))
        print('Damaging the weights')
        for i in (range(len(indices))):
            weights_damaged[indices[i]]=0
        weights_damaged = np.reshape(weights_damaged,(9000,9000))
        return weights_damaged
            
def simulation(niter,lambdas,flag,p):
    dt=0.01
    Hop=Hopfield(niter)
    Hop.flag=flag
    Hop.weight_matrix()
    Hop.U = np.reshape(Hop.cut_image(mona),(9000,1))
    Hop.weights=Hop.damage_weights(p)
    Hop.weights=Hop.weights/9000
    images_arr=[]
    for i in (range(niter)):
        Hop.U_d = -Hop.U + np.matmul(Hop.weights,Hop.V)
        Hop.U = Hop.U + (Hop.U_d)*dt
        Hop.V = np.tanh(lambdas*Hop.U)
        Hop.rmse[i]=mean_squared_error(mona_vect,Hop.V)
        
        img=np.reshape(Hop.V,(90,100))
        images_arr.append(img)
    images_arr=np.array(images_arr)
    return images_arr,Hop.rmse
    
def show(images_arr,rmse,niter,p, ques):
    images_arr=np.array(images_arr)
    for i in range(int(niter/10)):
        plt.imshow(images_arr[10*i,:,:],'Greys_r')
        plt.title(f'Image after {10*i} iterations for {p*100}% of weight damage')
        plt.savefig("mona" + ques + str(i))
        plt.show()
        
    plt.plot(rmse)
    plt.title(f'Plot of RMSE for {p*100}% of weight damage')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.grid()
    plt.savefig("monarmse" + str(i))
    plt.show()
    
niter=50
images_arr,rmse=simulation(niter,10,0,0)   
show(images_arr,rmse,niter,0, "Q2")

niter=100
images_arr,rmse=simulation(niter,10,1,0.25)
show(images_arr,rmse,niter,0.25, "Q3")

images_arr,rmse=simulation(niter,10,1,0.5)
show(images_arr,rmse,niter,0.5, "Q4")

images_arr,rmse=simulation(niter,10,1,0.8)
show(images_arr,rmse,niter,0.8, "Q5")


