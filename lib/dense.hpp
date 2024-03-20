#ifndef DENSE_H
#define DENSE_H

#include <vector> 
#include <cmath>
#include "neuron.hpp"

using namespace std;

class DenseLayer {
public:
    vector<Neuron> neurons; // Vector to store Neuron objects
    int num_neurons;         // Number of neurons in the layer
    vector<double> derivative; // Derivative of each neurons output for back propagation

public:

    // Parameterized constructor
    DenseLayer(int num_inputs, int num_neurons);

    // Softmax function
    vector<double> softmax(vector<double>& inputs);

    // Derivative function
    vector<double> calculate_derivative(vector<double>& outputs);

    // Function to compute the output of the layer given inputs
    vector<double> compute_output(vector<double>& inputs);
};

#endif // DENSE_H
