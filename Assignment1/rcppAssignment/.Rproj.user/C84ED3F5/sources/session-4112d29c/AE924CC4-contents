#### R function ####
# Code for KNN starts here. You have to translate this code into C++ / Rcpp

my_knn_R = function(X, X0, y){
  # X data matrix with input attributes
  # y response variable values of instances in X  
  # X0 vector of input attributes for prediction (just one instance)
  
  nrows = nrow(X)
  ncols = ncol(X)
  
  distance = 0
  for(j in 1:ncols){
    difference = X[1,j]-X0[j]
    distance = distance + difference * difference
  }
  
  distance = sqrt(distance)
  
  closest_distance = distance
  closest_output = y[1]
  closest_neighbor = 1
  
  for(i in 2:nrows){
    
    distance = 0
    for(j in 1:ncols){
      difference = X[i,j]-X0[j]
      distance = distance + difference * difference
    }
    
    distance = sqrt(distance)
    
    if(distance < closest_distance){
      closest_distance = distance
      closest_output = y[i]
      closest_neighbor = i
    }
  }
  closest_output
}  
# Code for KNN ends here.

# Here, we test the function we just programmed
print(my_knn_R(X, X0, y))



#### Preparing data ####
set.seed(123) # for reproducibility 

# X contains the inputs as a matrix of real numbers
data("iris")

iris_shuffled <- iris[sample(nrow(iris)), ]

# X contains the input attributes (excluding the class)
X <- iris_shuffled[,-5]
# y contains the response variable (named medv, a numeric value)
y <- iris_shuffled[,5]

# From dataframe to matrix
X <- as.matrix(X)
# From factor to integer
y <- as.integer(y)

# This is the point we want to predict
X0 <- c(5.80, 3.00, 4.35, 1.30)



#### Task 1 ####

# Using my_knn_c and class:knn to predict point X0

# Testing R function
library(class)
print(class::knn(X, X0, y, k=1))

# Testing: Compile the C++ code with sourceCpp
library(Rcpp)
sourceCpp('C:/Users/gabri/OneDrive - Universidad Carlos III de Madrid/Escritorio/Advanced Programming/Task1/task1_my_knn_c.cpp')
print(my_knn_c(X, X0, y))


# Comparing time compile between c function and 
library(microbenchmark)
microbenchmark(my_knn_c(X, X0, y), knn(X, X0, y, k=1), my_knn_R(X, X0, y))



#### Task 2 ####
print(my_knn_c_task2(X, X0, y))

my_knn_c_euclidean(X[1,], X0)


#### Task 3 ####
print(my_knn_c_task3(X, X0, y, 2))
my_knn_c_minkowsky(X[1,], X0, 2)

print(my_knn_c_task3(X, X0, y, 0))

print(my_knn_c_task3(X, X0, y, -7))
print(my_knn_c_task3(X, X0, y, -1))
my_knn_c_minkowsky(X[1,], X0, -7)
my_knn_c_minkowsky(X[1,], X0, -1)


#### Task 4 ####
possible_p <- c(5,2,7,6)
possible_p <- c(2,7,5,6)

possible_p <- c(0.5,0.2,6)
possible_p <- c(0.5,6,-1)

print(my_knn_c_tuningp(X, X0, y, possible_p))
