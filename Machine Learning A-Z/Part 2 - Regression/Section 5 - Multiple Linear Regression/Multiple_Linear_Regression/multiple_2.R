dataset = read.csv('50_Startups.csv')

#data encoding state variable
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

#creating test and train data set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#using linear regression to create regressor and y_predict
#regressor = lm(Profit ~ R.D.Spend + Administration +Marketing.Spend + State, data = training_set)
regressor = lm(Profit ~ ., data = training_set)
y_predict = predict(regressor, newdata = test_set)

#after analysing the regressor, we come to know that R&D is main
regressor2 = lm(Profit ~ R.D.Spend, data = training_set)
y_predict2 = predict(regressor2, newdata = test_set)


#BackWard Elimination
regressor3 = lm(Profit ~ R.D.Spend + Administration +Marketing.Spend + State, data = dataset)
summary(regressor3)
#removing state3 because significance level is greater than 5%
regressor3 = lm(Profit ~ R.D.Spend + Administration +Marketing.Spend , data = dataset)
summary(regressor3)
#removing administration because significance level is greater than 5%
regressor3 = lm(Profit ~ R.D.Spend  +Marketing.Spend , data = dataset)
summary(regressor3)
#removing marketing because significance level is greater than 5%
regressor3 = lm(Profit ~ R.D.Spend , data = dataset)
summary(regressor3)