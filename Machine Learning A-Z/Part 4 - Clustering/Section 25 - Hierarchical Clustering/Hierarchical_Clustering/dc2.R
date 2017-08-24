dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#using dendograms to find the optimal number of clusters
dendogram = hclust( d = dist(X, method = 'euclidean'), method  = 'ward.D' )
plot(dendogram, 
     main = paste('dendogram'),
     xlab = 'Customers',
     ylab = 'Distance')

#no of clusters = 5
hc = hclust(d = dist(X, method = 'euclidean'), method = 'ward.D')
#this will cut the hc tree at no of clusters = 5
yc = cutree(hc, 5)


# Visualising the clusters
library(cluster)
clusplot(X,
         yc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')