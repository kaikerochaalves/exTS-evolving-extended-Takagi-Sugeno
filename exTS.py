# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:53:12 2022

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math

class exTS:
    def __init__(self, InitialOmega = 1000, mu_threshold = 1/3, epsilon = 0.01, rho = 1/2):
        self.hyperparameters = pd.DataFrame({'InitialOmega':[InitialOmega], 'mu_threshold':[mu_threshold], 'epsilon':[epsilon], 'rho':[rho]})
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'mu', 'Tau', 'Lambda', 'r', 'sigma', 'increment_center_x'])
        self.InitialPotential = 1.
        self.DataPotential = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
         
    def fit(self, X, y):
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        # Initialize the first rule
        self.parameters = self.parameters.append(self.Initialize_First_Cluster(x, y[0], z), ignore_index = True)
        # Update lambda of the first rule
        self.Update_Lambda(x)
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the potential for all rules
            for i in self.parameters.index:
                self.Update_Rule_Potential(z, i, k+1)
            # Compute the data potential
            self.Update_Data_Potential(z_prev, z, i, k+1)
            Greater_Zero = ((self.DataPotential.item() - self.parameters['Potential']) > 0).all()
            Lower_Zero = ((self.DataPotential - self.parameters['Potential']) < 0).all()
            # Verifying the needing to creating a new rule
            if Greater_Zero == True or Lower_Zero == True:
                self.mu(x)
                mu_onethird = 0
                for row in self.parameters.index:
                    if (self.parameters.loc[row, 'mu'] > self.hyperparameters.loc[0, 'mu_threshold']).all():
                        mu_onethird = 1
                if mu_onethird == 1:                            
                    # Update an existing rule
                    self.Rule_Update(x, z)
                else:
                    # Create a new rule
                    self.parameters = self.parameters.append(self.Initialize_Cluster(x, z, k+1, i), ignore_index = True)
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
            # Store the output in the array
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        return self.OutputTrainingPhase, self.rules
            
    def predict(self, X):
        for k in range(X.shape[0]):
            x = X[k,].reshape((1,-1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Update lambda of all rules
            self.Update_Lambda(x)
            # Verify if lambda is nan
            if math.isnan(self.parameters.loc[0, 'Lambda']) == True:
                l = 1/self.parameters.shape[0]
                for row in self.parameters.index:
                    self.parameters.loc[row, 'Lambda'] = l
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
            # Store the output in the array
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y, z):
        NewRow = {'Center_Z': z, 'Center_X': x, 'C': self.hyperparameters.loc[0, 'InitialOmega'] * np.eye(x.shape[0] + 1), 'Theta': np.zeros((x.shape[0] + 1, 1)), 'Potential': self.InitialPotential, 'TimeCreation': 1., 'NumPoints': 1., 'mu': np.zeros([x.shape[0], 1]), 'Tau': 1., 'r':np.ones([x.shape[0], 1]), 'sigma':np.ones([x.shape[0], 1]), 'increment_center_x':np.zeros([x.shape[0], 1])}
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
        return NewRow
    
    def Initialize_Cluster(self, x, z, k, i):
        Theta = np.zeros((x.shape[0] + 1, 1))
        # Update the lambda value for all rules
        self.Update_Lambda(x)
        sigma = np.zeros([x.shape[0], 1])
        for row in self.parameters.index:
            sigma = sigma + self.parameters.loc[row, 'sigma'] 
            Theta = Theta + self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'Theta']
        sigma = sigma / self.parameters.shape[0]
        NewRow = {'Center_Z': z, 'Center_X': x, 'C': self.hyperparameters.loc[0, 'InitialOmega'] * np.eye(x.shape[0] + 1), 'Theta': Theta, 'Potential': self.InitialPotential, 'TimeCreation': k, 'NumPoints': 1, 'mu': np.zeros([x.shape[0], 1]), 'Tau': 1., 'r':np.ones([x.shape[0], 1]), 'sigma':sigma, 'increment_center_x':np.zeros([x.shape[0], 1])}
        return NewRow
    
    def Update_Rule_Potential(self, z, i, k):
        self.parameters.at[i, 'Potential'] = ((k - 1) * self.parameters.loc[i, 'Potential']) / (k - 2 + self.parameters.loc[i, 'Potential'] + self.parameters.loc[i, 'Potential'] * self.Distance(z.T, self.parameters.loc[i, 'Center_Z'].T)**2)
        
    def Distance(self, p1, p2):
        distance = np.linalg.norm(p1 - p2)
        return distance
    
    def Update_Data_Potential(self, z_prev, z, i, k):
        self.Beta = self.Beta + z_prev
        self.Sigma = self.Sigma + sum(z_prev**2)
        vartheta = sum(z**2)
        upsilon = sum(z*self.Beta)
        self.DataPotential = ((k - 1)) / ((k - 1) * (vartheta + 1) + self.Sigma - 2*upsilon)
        
    def Minimum_Distance(self, z):
        dist = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
        return min(dist)
                           
    def Rule_Update(self, x, z):
        dist = []
        idx = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
            idx.append(row)
        index = idx.index(dist.index(min(dist)))
        self.parameters.at[index, 'NumPoints'] = self.parameters.loc[index, 'NumPoints'] + 1
        # Update the radius
        self.parameters.at[index, 'increment_center_x'] = self.parameters.loc[index, 'increment_center_x'] + ( self.parameters.at[index, 'Center_X'] - x )**2
        self.parameters.at[index, 'sigma'] = ( self.parameters.loc[index, 'increment_center_x'] / self.parameters.loc[index, 'NumPoints'] )**(1/2)
        self.parameters.at[index, 'r'] = self.hyperparameters.loc[0, 'rho'] * self.parameters.loc[index, 'r'] + ( 1 - self.hyperparameters.loc[0, 'rho'] ) * self.parameters.loc[index, 'sigma']
        # Update the rule
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
        self.parameters.at[index, 'Potential'] = self.DataPotential
            
    def Update_Lambda(self, x):
        # Computing lambda
        self.mu(x)
        Total_Tau = sum(self.parameters['Tau'])
        if Total_Tau == 0:
            self.parameters['Lambda'] = 1.0 / self.parameters.shape[0]
        else:
            for row in self.parameters.index:
                self.parameters.at[row, 'Lambda'] = self.parameters.loc[row, 'Tau'] / Total_Tau
    
    def mu(self, x):
        for row in self.parameters.index:
            mu = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                mu[j,0] = math.exp( - np.linalg.norm( x[j,0] - self.parameters.loc[row, 'Center_X'][j,0] )**2 / ( 2 * self.parameters.loc[row, 'r'][j,0] ** 2 ) )
            self.parameters.at[row, 'mu'] = mu
            self.parameters.at[row, 'Tau'] = np.prod(mu)
    
    def RLS(self, x, y, xe):
        self.Update_Lambda(x)
        for row in self.parameters.index:
            self.parameters.at[row, 'C'] = self.parameters.loc[row, 'C'] - ((self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'C'] @ xe @ xe.T @ self.parameters.loc[row, 'C'])/(1 + self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'C'] @ xe))
            self.parameters.at[row, 'Theta'] = self.parameters.loc[row, 'Theta'] + (self.parameters.loc[row, 'C'] @ xe * self.parameters.loc[row, 'Lambda'] * (y - xe.T @ self.parameters.loc[row, 'Theta']))
            
    def Remove_Rule(self, k):
        N_total = 0
        for i in self.parameters.index:
            N_total = N_total + self.parameters.loc[i, 'NumPoints']
        remove = []
        for i in self.parameters.index:
            if self.parameters.loc[i, 'NumPoints'] / N_total < self.hyperparameters.loc[0,'epsilon']:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)
