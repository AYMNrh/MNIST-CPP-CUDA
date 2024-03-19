#include <vector>
#include <list>
#include <iostream>
#include <cmath>
#include "dense.h"
#include "conv_layer.h" 

using namespace std;

// TODO
ConvolutionalLayer convolutional_back_Propagation(ConvolutionalLayer convolutional_layer, const vector<vector<vector<double>>> train_images) {
    // TODO

    return convolutional_layer;
}

double crossentropy_error(const vector<int> training_label, const vector<double> predict_output) {
    // initialize the sum
    double sum = 0;
    // In our case equals to 10
    int n = predict_output.size();

    for (int i=0; i<n; i++) {
        sum += training_label[i] * log(predict_output[i]);
    }

    return -sum;
}

DenseLayer dense_back_Propagation(DenseLayer dense_layer, const vector<vector<vector<double>>> train_images, const vector<int> training_label, double learning_rate) {
    for (int i = 0; i < dense_layer.num_neurons; i++) {
        Neuron& neuron = dense_layer.neurons[i];
        for (int j = 0; j < neuron.num_inputs; j++) {
            neuron.weights[j] -= learning_rate * training_label[i] * dense_layer.derivative[i] * neuron.inputs[j];
        }
    }
    return dense_layer;
}




