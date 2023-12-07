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
  static bool warningDisplayed = false;

  if (p>0){
    for(int j=0; j<ncols; ++j){
      difference = abs(X[j] - X0[j]);
      distance += pow(difference,p);
    }
    distance = pow(distance, 1/p);
  } else if (p==0){
    if(!warningDisplayed){
      Rcpp::Rcerr << "Warning: p is 0; consider handling this case appropriately." << std::endl;
      warningDisplayed = true;
    }
    distance = NA_REAL;
  } else{
    for(int j=0; j<ncols; ++j){
      difference = abs(X[j] - X0[j]);
      distance = std::max(distance, difference);
    }
  }
  return distance;
}


// [[Rcpp::export]]
int my_knn_c_task3(NumericMatrix X, NumericVector X0, IntegerVector y, double p){
  int nrows = X.nrow();
  double closest_distance = R_PosInf;
  double closest_output = NA_INTEGER;
  int closest_neighbor = -1;
  
  for (int i=1; i<nrows; ++i){
    double distance = my_knn_c_minkowsky(X(i,_), X0, p);
    if (distance != NA_REAL){
      if(distance < closest_distance){
        closest_distance = distance;
        closest_output = y[i];
        closest_neighbor = i;
      } 
    }
  }
  return closest_output;
}


// [[Rcpp::export]]
List my_knn_c_tuningp(NumericMatrix X, NumericVector X0, IntegerVector y, NumericVector possible_p){
  
  int nrows = X.nrow();
  // Calculate the number of observations for the training set
  int train_size = floor((2.0 / 3.0) * nrows);
  
  // Split dataset into training (2/3) and validation (1/3)
  NumericMatrix X_train = X(Range(0, train_size-1), _);
  NumericMatrix X_val = X(Range(train_size, nrows-1), _);
  IntegerVector y_train = y[Range(0, train_size-1)];
  IntegerVector y_val = y[Range(train_size, nrows-1)];
  
  double best_accuracy = 0.0;
  double best_p = 0.0;
  
  // Iterate for all possible_p in order to find the best value
  for (int i=0; i<possible_p.size(); ++i){
    double p  = possible_p[i];
    int correct_predictions = 0;
    for (int j = 0; j < nrows - train_size; ++j) {
      // Reused code for task3
      int closest_output = my_knn_c_task3(X_train, X_val(j, _), y_train, p);
      // Check if prediction is correct
      if (closest_output == y_val[j]) {
        correct_predictions++;
      }
    }
    
    // Choose the best_p in relation to the accuracy
    double accuracy = static_cast<double>(correct_predictions)/(nrows-train_size);
    if (accuracy > best_accuracy) {
      best_accuracy = accuracy;
      best_p = p;
    }
  }
  
  // Return: final prediction, optimal value for p and the accuracy
  List final_params;
  final_params["best_p"] = best_p;
  // Use the best_p to make the final prediction for X0
  final_params["final_output"] = my_knn_c_task3(X, X0, y, best_p);
  final_params["best_accuracy"] = best_accuracy;
  return final_params;
}
