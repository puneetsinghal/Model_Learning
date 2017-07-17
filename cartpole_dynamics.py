"""
The code is based on cartpole environment in openai: 
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

Editor: Puneet Singhal
"""

# import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
from copy import copy
import matplotlib.pyplot as plt
from sympy import *
from random import random
# from tempfile import TemporaryFile
import argparse

# logger = logging.getLogger(__name__)

class CartPoleEnv(object):

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.001  # seconds between state updates

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state[0]
        x, x_dot, theta, theta_dot = state

        # force = self.force_mag if action==1 else -self.force_mag
        force = action[0]

        # define the constants to be used multiple times. Just for easy coding
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta* temp)/ (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta/ self.total_mass

        # denom = 1.0/(self.masscart + self.masspole*sintheta*sintheta)
        
        # xacc = denom*(force + sintheta*(-self.polemass_length*theta_dot**2 + self.masspole*self.gravity*costheta))
        # thetaacc = denom/self.length*(force*costheta - self.polemass_length*costheta*sintheta*theta_dot**2 + self.total_mass*self.gravity*sintheta)
        
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array((x, x_dot, theta, theta_dot)).reshape(1, self.dim)

        return self.state

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        scale = 100
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state[0]
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _training(self, startPosition, startTime=0, runTime=4, debug=False):
        
        currentTime = np.array(startTime).reshape(1,1)
        trainingData = copy(startPosition)
        trainingData.shape = (1,5)
        self.state = copy(trainingData[0,0:4]).reshape(1,4)
        timeArray = copy(currentTime)
        applyForce = True
        forceMagnitude = 100
        while currentTime < runTime+startTime:
            if applyForce:
                action = forceMagnitude*np.ones((1,1)) if random() < 0.5 else -forceMagnitude*np.ones((1,1))
                applyForce = False
                count = 0
            else:   
                action = np.zeros((1,1))
            if self.state[0,0] > 1:
                action = -forceMagnitude*np.ones((1,1))
            elif self.state[0,0] < -1:
                action = forceMagnitude*np.ones((1,1))
            self._step(action)
            trainingData = np.vstack((trainingData, np.hstack((self.state, action))))
            currentTime = currentTime + self.tau
            timeArray = np.vstack((timeArray, currentTime))

            count += 1
            if(count == 10):
                applyForce=True

            if debug:
                # print(np.append(np.array(CP.state).reshape(1,4),action))
                self._render()
                print(currentTime, action, count, applyForce)
        if debug:
            plotting(trainingData, timeArray)

        return (trainingData, timeArray)

class koopmanMechanics(object):

    def __init__(self, psiArray, varArray):
        
        self.funArray = psiArray
        self.varArray = varArray
        self.tau = 0.001  # seconds between state updates
        self.dim = len(varArray) - 1

    def generateKoopmanOperator(self, trainingData):

        print("Learning G and A matrix")
        lengthFuncArray = len(self.funArray)

        G = np.matrix(np.zeros((lengthFuncArray, lengthFuncArray)))
        A = np.matrix(np.zeros((lengthFuncArray, lengthFuncArray)))

        numPoints = trainingData.shape[0]
        print("total {:d} number of training points".format(numPoints))

        for i in range (0, numPoints):
            currentValue = self.evaluateVectorValuedFunc(trainingData[i].tolist())
            G = G + currentValue.T*currentValue
            if i < numPoints -1:
                futureValue = self.evaluateVectorValuedFunc(trainingData[i+1].tolist())
                A = A + currentValue.T*futureValue
            if i%100  == 0:
                print("training point: ", i)
        
        G = G/numPoints
        A = A/numPoints

        np.savetxt("G.txt", G)
        np.savetxt("A.txt", A)
        G = np.matrix(np.loadtxt("G.txt"))
        A = np.matrix(np.loadtxt("A.txt"))
        
        print("Calculating Kooman Operator")
        self.K = np.matmul(np.linalg.pinv(G),A)
        # print(self.K)

        np.savetxt("koopmanOperatorCartPole.txt", self.K)

    def evaluateVectorValuedFunc(self, valArray):
        return np.matrix([f.subs(zip(self.varArray, valArray)).evalf(4) for f in self.funArray])

    def _step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = copy(self.state[0])

        # force = self.force_mag if action==1 else -self.force_mag
        force = [action[0]]

        KHat = self.K[:,0:self.dim]
        psiFunctionValue = self.evaluateVectorValuedFunc(state.tolist() + force)

        self.state = self.state + self.tau*np.array(KHat.T*psiFunctionValue.T).reshape(1, self.dim)

        return self.state

    def koopmanPlay(self, startPosition, startTime=0., runTime=4., debug=False):
        currentTime = np.array(startTime).reshape(1,1)
        data = copy(startPosition)
        data.shape = (1,5)
        self.state = copy(data[0,0:4]).reshape(1,4)
        timeArray = copy(currentTime)

        while currentTime < runTime+startTime:
            # if update == 0:
            action = np.zeros((1,1))
            self._step(action)
            data = np.vstack((data, np.hstack((self.state, action))))
            currentTime = currentTime + self.tau
            timeArray = np.vstack((timeArray, currentTime))

            if debug:
                # print(np.append(np.array(CP.state).reshape(1,4),action))
                # self._render()
                print(currentTime, action)

        return (data, timeArray)

def plotting(data,time):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(time, data[:,0], linewidth=3, label='x')
    plt.title('x')
    plt.legend()
    plt.ylabel('meter')
    plt.xlabel('time')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(time, data[:,1], linewidth=3, label='dx')
    plt.title('dx')
    plt.legend()
    plt.ylabel('m/s')
    plt.xlabel('time')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(time, data[:,2], linewidth=3, label='theta')
    plt.title('theta')
    plt.legend()
    plt.ylabel('radians')
    plt.xlabel('time')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(time, data[:,3], linewidth=3, label='dtheta')
    plt.title('dtheta')
    plt.legend()
    plt.ylabel('radians/sec')
    plt.xlabel('time')
    plt.grid()

if __name__ == '__main__':
    
    #Parsing inputs for plotting, trajectory generation, and saving options
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train",    type=int,  default = 0)
    parser.add_argument("-db", "--debug",    type=bool,  default = False)
    parser.add_argument("-s", "--save",     type=str,   default="no")
    parser.add_argument("-o", "--opt",      type=str,   default="no")

    args = parser.parse_args()
    training = args.train
    debugging = args.debug
    saving = (args.save.lower()=="yes")
    opt = (args.save.lower()=="yes")

     # Define symbolic variables and basis functions
    x, dx, th, dth, u = symbols('x dx th dth u')
    varArray = [x, dx, th, dth, u]
    
    if not training == 0:
        CP = CartPoleEnv()
        # plt.ion()   

        CP.dim = len(varArray) - 1

        startPosition = np.array((0., 0., math.pi, 0., 0.)) #[x, dx, theta, dtheta, input]
        startTime = 0.
        runtime = 20
        # Training Round 1
        print("training round 1...")
        trainingData, trainingTime = CP._training(startPosition, startTime, runtime, debug=debugging)

        # Training Round 2
        # print("training round 2...")
        # newData, newTime = CP._training(startPosition, trainingTime[-1,0], runtime, debug=False)
        # trainingData = np.vstack((trainingData,newData))
        # trainingTime = np.vstack((trainingTime, newTime))

        print("Saving training data to text file: cartPole_data.txt")
        np.savetxt("cartPole_data.txt", trainingData)
        
    else:
        trainingData = np.loadtxt("cartPole_data.txt")
        trainingTime = np.linspace(0,0.001*trainingData.shape[0], trainingData.shape[0])

    plotting(trainingData, trainingTime)
    plt.show(block=False)
    
    xLimit, dxLimit, thLimit, dthLimit, uLimit = np.max(np.fabs(trainingData[:,0:5]), axis=0)
    psiFunctions = [x-x+1, u/uLimit, x/xLimit*u/uLimit, dx/dxLimit*u/uLimit, th/thLimit*u/uLimit, dth/dthLimit*u/uLimit, x/xLimit*x/xLimit*u/uLimit, x/xLimit*dx/dxLimit*u/uLimit, x/xLimit*th/thLimit*u/uLimit, x/xLimit*dth/dthLimit*u/uLimit, dx/dxLimit*dx/dxLimit*u/uLimit, dx/dxLimit*th/thLimit*u/uLimit, dx/dxLimit*dth/dthLimit*u/uLimit, th/thLimit*th/thLimit*u/uLimit, th/thLimit*dth/dthLimit*u/uLimit, dth/dthLimit*dth/dthLimit*u/uLimit]
    psiArray = varArray + psiFunctions
    KOOPMAN = koopmanMechanics(psiArray, varArray)

    if not training == 0:
        # Learn Koopman Operator
        KOOPMAN.generateKoopmanOperator(trainingData)
    else:
        KOOPMAN.K = np.loadtxt("koopmanOperatorCartPole.txt")
    # Test the performance of Koopman Operator derived
    print("Testing Koopman Dynamics...")
    testData, testTime = KOOPMAN.koopmanPlay(startPosition = np.array((0., 0., 2.0, 0., 0.)), startTime=0., runTime=4., debug=False)
    plotting(testData, testTime)
    
    plt.show()
