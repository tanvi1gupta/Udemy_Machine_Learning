dataset =read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10)
  wcss[i] = sum(kmeans(dataset, i)$withinss)

plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

#k = 5 is optimal
set.seed(29)
kmeans = kmeans(dataset, 5)
clusters  = kmeans$cluster

#install.packages('cluster')
library(cluster)
clusplot(dataset,
         clusters,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')