from sklearn.preprocessing import StandardScaler
import numpy as np


class LinearRegression(object):
    def __init__(self,normalize=False):

        self.coefficients = None  
        self.intercept = None 
        self.normalize = normalize 
        self.scaler = StandardScaler()

    def fit(self, X, y):
        num_nonzero_coefs, coef_norm = 0, 0 
        no_of_samples = X.shape[0]
        
        if self.normalize:
            scaled_X = self.scaler.fit(X)
            X = scaled_X.transform(X)

        X_centering = X - np.mean(X,axis=0)      
        self.coefficients = ((np.linalg.inv(np.transpose(X_centering) @ X_centering)) @ np.transpose(X_centering) @ y)
        self.intercept = 0

        for index in range(no_of_samples):
          self.intercept += y[index] - self.coefficients.T @ X[index]
        self.intercept = self.intercept / no_of_samples
       
        num_nonzero_coefs = np.count_nonzero(self.coefficients)
        coef_norm = np.linalg.norm(self.coefficients)     
    
        return num_nonzero_coefs, coef_norm

    def evaluate(self,test_x,test_y):
        root_mean_squared_error = 0
        if self.normalize:
            test_x = self.scaler.transform(test_x)
 
        root_mean_squared_error  = 0
        mean_square_error = 0
        for index in range(test_x.shape[0]):
            mean_square_error += np.square(test_y[index] - (np.transpose(self.coefficients) @ test_x[index] + self.intercept))
        root_mean_squared_error = np.sqrt(mean_square_error/ test_x.shape[0]) 

        return root_mean_squared_error

    def predict(self,text_x):
        y_predits = []

        for index in range(text_x.shape[0]):
            y_predict = self.coefficients * text_x[index] + self.intercept
            y_predits.insert(y_predict)
        return y_predits    


