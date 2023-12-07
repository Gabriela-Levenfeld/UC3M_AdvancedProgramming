#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
int my_knn_c(NumericMatrix X, NumericVector X0, IntegerVector y){
  
  int nrows = X.nrow();
  int ncols = X.ncol();

  double distance = 0;
  double difference = 0;
  int j;

  for (j=0; j<ncols; ++j){
    difference = X(1,j)-X0[j];
    distance = distance + difference * difference;
  }
  
  distance = sqrt(distance);
    
  double closest_distance = distance;
  double closest_output = y[1];
  int closest_neighbor = 1;
  
  for (int i=1; i<nrows; ++i){
    distance = 0;
    for (j=0; j<ncols; ++j){
      difference = X(i,j)-X0[j];
      distance = distance + difference * difference;
    }
    distance = sqrt(distance);
      if(distance < closest_distance){
        closest_distance = distance;
        closest_output = y[i];
        closest_neighbor = i;
      }
  }
  
  return closest_output;
}  
