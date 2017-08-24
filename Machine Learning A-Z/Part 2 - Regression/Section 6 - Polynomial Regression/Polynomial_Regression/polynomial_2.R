dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

lin_regression = lm(Salary ~ Level, data = dataset)
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4


poly_lin_regression = lm(Salary ~ ., data = dataset)
summary(poly_lin_regression)

library(ggplot2)
#plotting linear regression graph
ggplot()+
  geom_point(aes(x =dataset$Level , y =dataset$Salary), color = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(lin_regression, newdata = dataset)), color = 'blue')+
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

#plotting poly linear regression graph
ggplot()+
  geom_point(aes(x =dataset$Level , y =dataset$Salary), color = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(poly_lin_regression, newdata = dataset)), color = 'blue')+
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
  geom_line(aes(x = x_grid, y = predict(poly_lin_regression, newdata = newDataSet)), color = 'blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

#testing data for prediction
predict(lin_regression, data.frame(Level=6.5))
predict(poly_lin_regression, data.frame(Level = 6.5, Level2 = 6.5^2,Level3 = 6.5^3,Level4 = 6.5^4))