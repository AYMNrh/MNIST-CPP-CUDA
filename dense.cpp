#include <vector> 
#include <cmath>
#include "neuron.h" // Include the Neuron header file
#include "dense.h"  // Include the header file for the DenseLayer class

using namespace std; // Use the std namespace

class DenseLayer {
public:
    vector<Neuron> neurons; // Vector to store Neuron objects
    int num_neurons;        // Number of neurons in the layer
    vector<double> derivative; // Derivative of each neurons output for back propagation

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

    // Softmax function
    vector<double> softmax(const vector<double>& inputs) {
        vector<double> probabilities(inputs.size());
        double sum_exp = 0.0;

        // Compute the sum of exponentials of input values
        for (int i = 0; i < inputs.size(); ++i) {
            sum_exp += exp(inputs[i]);
        }

        // Compute probabilities using softmax formula
        for (int i = 0; i < inputs.size(); ++i) {
            probabilities[i] = exp(inputs[i]) / sum_exp;
        }

        return probabilities;
    }

    // Derivative function
    vector<double> calculate_derivative(vector<double> outputs) {
        vector<double> derivative(outputs.size());
        for (int i = 0; i < outputs.size(); ++i) {
            derivative[i] = outputs[i] * (1 - outputs[i]);   // Compute the derivative of the output
        }
        return derivative;
    }

    // Function to compute the output of the layer given inputs
    vector<double> compute_output(vector<double>& inputs) {
        // Vector to store the output of each neuron
        vector<double> outputs(num_neurons);
        // Compute the output of each neuron in the layer
        for (int i = 0; i < num_neurons; ++i){
            outputs[i] = neurons[i].compute_output(inputs);
        }

        // Apply softmax function
        outputs = softmax(outputs);

        // Apply derivative function
        derivative = calculate_derivative(outputs);

        return outputs; // Return the computed output
    }
};
