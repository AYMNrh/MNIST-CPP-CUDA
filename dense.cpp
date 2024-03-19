#include <vector> 
#include "neuron.h" // Include the Neuron header file
#include "dense.h"  // Include the header file for the DenseLayer class

using namespace std; // Use the std namespace

class DenseLayer {
private:
    vector<Neuron> neurons; // Vector to store Neuron objects
    int num_neurons;        // Number of neurons in the layer

public:
    // Default constructor
    DenseLayer() {
        num_neurons = 0;
    }

    // Parameterized constructor
    DenseLayer(int num_inputs, int num_neurons) : num_neurons(num_neurons) {
        // Resize the neurons vector and initialize neurons with specified number of inputs
        neurons.resize(num_neurons, Neuron(num_inputs));
    }

    // Function to compute the output of the layer given inputs
    vector<double> compute_output(vector<double>& inputs) {
        // Vector to store the output of each neuron
        vector<double> output(num_neurons);
        // Compute the output of each neuron in the layer
        for (int i = 0; i < num_neurons; ++i)
            output[i] = neurons[i].compute_output(inputs);
        return output; // Return the computed output
    }
};
