from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
import numpy as np

class Lasso(object):
    def __init__(self, alpha, normalize=False):
        
        self.alpha = alpha  
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
            
        X_mean = X - np.mean(X,axis = 0)

        _, coef_path, _ = lasso_path(X_mean, y, alphas=[self.alpha])

        num_nonzero_coefs = coef_path.T[0]
        self.coefficients = num_nonzero_coefs

        self.intercept = 0
    
        for index in range(no_of_samples):
          self.intercept += y[index] - (np.transpose(self.coefficients) @ X[index])
        self.intercept = self.intercept / (no_of_samples)

        num_nonzero_coefs = np.count_nonzero(self.coefficients)
        coef_norm = np.linalg.norm(self.coefficients)   
        
        return num_nonzero_coefs, coef_norm

    def evaluate(self, X, y):
        
        root_mean_squared_error = 0

        if self.normalize:
            X = self.scaler.transform(X)
 
        root_mean_squared_error  = 0
        mean_square_error = 0
        for index in range(X.shape[0]):
            mean_square_error += np.square(y[index] - (np.transpose(self.coefficients) @ X[index]+ self.intercept))
        root_mean_squared_error = np.sqrt(mean_square_error/ X.shape[0]) 
        
        return root_mean_squared_error

    def predict(self,text_x):
        
        y_predits = []

        for index in range(text_x.shape[0]):
            y_predict = self.coefficients * text_x[index] + self.intercept
            y_predits.insert(y_predict)
        return y_predits     