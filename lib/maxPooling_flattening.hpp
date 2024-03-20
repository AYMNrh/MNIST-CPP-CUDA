#ifndef MAX_POOLING_H
#define MAX_POOLING_H

#include <vector>   // Include vector for vector data structure
#include <algorithm> // Include algorithm for max function

using namespace std;

// Function to perform max pooling
vector<vector<double>> max_pooling(const vector<vector<double>>& input, int pool_size);

// Function for flattening max pooling outputs
vector<double> flatten(vector<vector<double>>& inputs);

#endif
