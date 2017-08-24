# importing libraries
#importing data set

dataset = read.csv('Data.csv')
dataset = dataset[,1:3]
#index starts from 1
#is.na--> if value in column age is missing
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age
                     )
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#factor function will transform them into factors, need not transform them into dummy variables
#press f1 for details
#create a vector of the categorical labels
dataset$Country = factor(dataset$Country, 
                         levels = c('France','Spain','Germany'), 
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased, 
                         levels = c('Yes','No'), 
                         labels = c(1,0))

#creating test and train set 
#we need to install caTools library
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) #this will return either true or false , true means in training, false means in testing
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#feature scaling
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])

