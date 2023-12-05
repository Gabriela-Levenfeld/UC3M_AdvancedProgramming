#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double my_knn_c_euclidean(NumericVector X, NumericVector X0){
  int ncols = X.size();
  double distance = 0;
  
  for (int j=0; j<ncols; ++j){
    double difference = X(j)-X0[j];
    distance += difference * difference;
  }
  return sqrt(distance);
}


// [[Rcpp::export]]
int my_knn_c_task2(NumericMatrix X, NumericVector X0, IntegerVector y){
  int nrows = X.nrow();
  int ncols = X.ncol();
  
  double difference = 0;
  double closest_distance = my_knn_c_euclidean(X(0,_), X0);
  double closest_output = y[1];
  int closest_neighbor = 1;
  
  for (int i=1; i<nrows; ++i){
    double distance = my_knn_c_euclidean(X(i,_), X0);
    if(distance < closest_distance){
      closest_distance = distance;
      closest_output = y[i];
      closest_neighbor = i;
    }
  }
  return closest_output;
}


// [[Rcpp::export]]
double my_knn_c_minkowsky(NumericVector X, NumericVector X0, double p){
  int ncols = X.size();
  double distance = 0;
  double difference = 0;
  int j;
  if (p>0){
    for(int j=0; j<ncols; ++j){
      difference = abs(X[j] - X0[j]);
      distance += pow(difference,p);
    }
    distance = pow(distance, 1/p);
  } else if (p==0){
    Rcpp::stop("p can't be 0");
  } else{
    for(j=0; j<ncols; ++j){
      difference = abs(X[j] - X0[j]);
      distance = std::max(distance, difference);
    }
  }
  return distance;
}


// [[Rcpp::export]]
int my_knn_c_task3(NumericMatrix X, NumericVector X0, IntegerVector y, double p){
  
  int nrows = X.nrow();
  int ncols = X.ncol();
  double difference = 0;
  
  double closest_distance = my_knn_c_minkowsky(X(0,_), X0, p);
  double closest_output = y[1];
  int closest_neighbor = 1;
  for (int i=1; i<nrows; ++i){
    double distance = my_knn_c_minkowsky(X(i,_), X0, p);
    if(distance < closest_distance){
      closest_distance = distance;
      closest_output = y[i];
      closest_neighbor = i;
    }
  }
  return closest_output;
}
