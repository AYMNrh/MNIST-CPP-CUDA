#include "neuron.hpp" // Include the header file for the Neuron class

// Constructor with number of inputs specified
Neuron::Neuron(int num_inputs) : num_inputs(num_inputs) {
    weights.resize(num_inputs + 1);    // Resize weights vector to accommodate bias
    randomize_weights();               // Initialize weights randomly
}

// Function to randomize weights
void Neuron::randomize_weights() {
    srand(time(0));                    // Seed the random number generator
    for (int i = 0; i <= num_inputs; i++) {
        // Randomize each weight between [-1,1]
        weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
    }
}

// Function to compute the output of the neuron given inputs
double Neuron::compute_output(vector<double>& inputs) {
    double output = weights[0];               // Initialize with the bias weight
    for (int i = 1; i <= num_inputs; i++) {
        output += weights[i] * inputs[i - 1]; // Weighted sum of inputs and weights
    }
    this->inputs = inputs;                // Store the inputs for later use
    return output;                        // Return the computed output
}
