import numpy as np
from sklearn.svm import SVC

import utils

class ClusterAdversarialClassifier():
    def __init__(self, model, input_transform=None):
        self.model = model
        self.input_transform = input_transform
        
    def fit(self, X, y):
        # Find RBF classification boundary in input space
        flat_X = X.reshape(X.shape[0], -1)
        if self.input_transform:
            self.input_cluster = SVC(kernel='rbf', C=1, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     self.input_transform.fit_transform(flat_X), y)
        else:
            self.input_cluster = SVC(kernel='rbf', C=1, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     flat_X, y)
        
        # Get model outputs
        train_dataloader = utils.create_dataloader(X, y)
        outputs = utils.get_network_outputs(self.model, train_dataloader)
        
        # Find RBF classification boundary in output space
        self.output_cluster = SVC(kernel='rbf', C=1, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                  outputs, y)
        
    def predict(self, X):
        # Get model outputs on test data
        test_dataloader = utils.create_dataloader(X, np.zeros(X.shape[0], dtype=np.uint8))
        outputs = utils.get_network_outputs(self.model, test_dataloader)
        # Get model preds from outputs
        preds = np.argmax(np.exp(outputs), axis=1)
        
        # Get input and output cluster predictions - if these disagree, we will consider the sample suspicious
        flat_X = X.reshape(X.shape[0], -1)
        if self.input_transform:
            input_cluster_preds = (self.input_cluster).predict(self.input_transform.transform(flat_X))
        else:
            input_cluster_preds = (self.input_cluster).predict(flat_X)
        output_cluster_preds = (self.output_cluster).predict(outputs)
        flagged_sample_indices = (input_cluster_preds != output_cluster_preds)
        
        self.proportion_flagged = (input_cluster_preds != output_cluster_preds).sum().item() / X.shape[0]
        
        # For samples flagged as likely to be adversarial by clusters, predict using instead the input cluster classification
        preds[flagged_sample_indices] = input_cluster_preds[flagged_sample_indices]
        
        return preds

    def score(self, X, y):
        # Get predictions
        preds = self.predict(X)
        
        # Return proportion that are correct
        return (preds == y).sum().item() / y.shape[0]