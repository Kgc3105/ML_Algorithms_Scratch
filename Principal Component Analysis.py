import numpy as np
class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.components = None
        self.exp_varience = None
    
    def fit(self,data):
        self.mean = np.mean(data,axis=0)
        self.std = np.std(data,axis=0)
        data_std = (data - self.mean)/self.std

        covarience = (data_std.T @ data_std)/(data_std.shape[0] -1)
        eigenvalues, eigenvectors = np.linalg.eigh(covarience)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:,:self.n_components]
        self.exp_variance = eigenvalues[:self.n_components]
        self.total_variance = np.sum(eigenvalues)

    def transform(self, X):
        X_std = (X - self.mean) / self.std
        return np.dot(X_std, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def exp_variance_ratio(self):
        return self.exp_variance / self.total_variance
