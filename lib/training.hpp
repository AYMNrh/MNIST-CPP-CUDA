#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include <list>
#include <iostream>
#include <cmath>
#include "dense.hpp"
#include "conv_layer.hpp" 

using namespace std;

// Function to perform backpropagation for convolutional layer
ConvolutionalLayer convolutional_back_Propagation(ConvolutionalLayer convolutional_layer, const vector<vector<double>> train_images, vector<vector<vector<double>>> conv_output, double learning_rate);

// Function to calculate crossentropy error
double crossentropy_error(uint8_t* training_label, const vector<double> predict_output);

// Function to perform backpropagation for dense layer
DenseLayer dense_back_Propagation(DenseLayer dense_layer, uint8_t* training_label, double learning_rate);

#endif
