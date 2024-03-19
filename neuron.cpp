#include "neuron.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class Neuron {
private:
    vector<double> weights;    // Weights of the neuron
    int num_inputs;            // Number of inputs to the neuron
    double delta;              // Delta value for backpropagation
    double derivative;         // Derivative of the neuron's output
    vector<double> inputs;     // Inputs to the neuron

public:
    Neuron() {
        num_inputs = 0;
    }

    // Constructor with number of inputs specified
    Neuron(int num_inputs) : num_inputs(num_inputs) {
        weights.resize(num_inputs + 1);    // Resize weights vector to accommodate bias
        randomize_weights();               // Initialize weights randomly
    }

private:
    // Function to randomize weights
    void randomize_weights() {
        srand(time(0));                    // Seed the random number generator
        for (int i = 0; i <= num_inputs; i++)
            // Randomize each weight
            weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
    }

public:
    // Function to compute the output of the neuron given inputs
    double compute_output(vector<double>& inputs) {
        double fi = weights[0];               // Initialize with the bias weight
        for (int i = 1; i <= num_inputs; i++)
            fi += weights[i] * inputs[i - 1]; // Weighted sum of inputs and weights

        // Use the Rectified Linear Unit (ReLU) activation function
        double output = (fi > 0.0) ? 1.0 : 0.0;
        derivative = output * (1 - output);   // Compute the derivative of the output
        this->inputs = inputs;                // Store the inputs for later use
        return output;                        // Return the computed output
    }
};
