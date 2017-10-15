# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1,2,3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                      levels = c('Female', 'Male'),
                                      labels = c(1,2)))

#splitting data into training and testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

#building the ANN model
#install.packages('h2o')
library(h2o)
#this will connect to only 1 thread in the remote server
h2o.init(nthreads = 1)
classififer = h2o.deeplearning(y=  'Exited',
                               training_frame = as.h2o(training_set),
                               activation = 'Rectifier',
                               #number of hidden layer, no of neurons in each layer
                               hidden = c(5,5),
                               epochs = 100,
                               #batch size
                               train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(classififer, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()