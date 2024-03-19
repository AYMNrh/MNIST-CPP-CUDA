#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include "neuron.h"

using namespace std;

class DenseLayer {
private:
    vector<Neuron> neurons; // Vector to store Neuron objects
    int num_neurons;         // Number of neurons in the layer

public:
    // Default constructor
    DenseLayer();

    // Parameterized constructor
    DenseLayer(int num_inputs, int num_neurons);

    // Function to compute the output of the layer given inputs
    vector<double> compute_output(vector<double>& inputs);
};

#endif // DENSE_H
