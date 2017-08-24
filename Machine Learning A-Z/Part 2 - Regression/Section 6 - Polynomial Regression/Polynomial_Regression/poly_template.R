
#building data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]


# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#building regression model
#create the regressor
regressor = lm(Salary ~ ., data = dataset)
summary(regressor)


#testing data for prediction
predict(regressor, data.frame(Level = 6.5, Level2 = 6.5^2,Level3 = 6.5^3,Level4 = 6.5^4))


library(ggplot2)
#plotting  regression graph
ggplot()+
  geom_point(aes(x =dataset$Level , y =dataset$Salary), color = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), color = 'blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

#smoother graph plot
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
newDataSet = data.frame(Level = x_grid, 
                        Level2 = x_grid^2,
                        Level3 = x_grid^3,
                        Level4 = x_grid^4)
ggplot()+
  geom_point(aes(x =dataset$Level , y =dataset$Salary), color = 'red')+
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = newDataSet)), color = 'blue')+
  ggtitle('Truth or Bluff (Smoother Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

