from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans	
from sklearn import metrics
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Conversion to float
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
# Normalization
x_train = x_train/255.0
x_test = x_test/255.0	


X_train = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)



total_clusters = len(np.unique(y_test))
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# Fitting the model to training set
kmeans.fit(X_train)


def retrieve_info(cluster_labels,y_train):
  '''
   Associates most probable label with each cluster in KMeans model
   returns: dictionary of clusters assigned to each label
  '''
  # Initializing
  reference_labels = {}
  # For loop to run through each label of cluster label
  for i in range(len(np.unique(kmeans.labels_))):
    index = np.where(cluster_labels == i,1,0)
    num = np.bincount(y_train[index==1]).argmax()
    reference_labels[i] = num
  
  return reference_labels


reference_labels = retrieve_info(kmeans.labels_,y_train)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]

print(accuracy_score(number_labels,y_train))

def calculate_metrics(model,output):
  print('Number of clusters is {}'.format(model.n_clusters))
  print('Inertia : {}'.format(model.inertia_))
  print('Homogeneity :       {}'.format(metrics.homogeneity_score(output,model.labels_)))



cluster_number = [10,16,36,64,144,256]
for i in cluster_number:
	total_clusters = len(np.unique(y_test))
	# Initialize the K-Means model
	kmeans = MiniBatchKMeans(n_clusters = i)
	# Fitting the model to training set
	kmeans.fit(X_train)
	# Calculating the metrics
 
calculate_metrics(kmeans,y_train)
# Calculating reference_labels
reference_labels = retrieve_info(kmeans.labels_,y_train)
# ‘number_labels’ is a list which denotes the number displayed in image
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
 
 number_labels[i] = reference_labels[kmeans.labels_[i]]
 
print('Accuracy score : {}'.format(accuracy_score(number_labels,y_train)))
print('\n')