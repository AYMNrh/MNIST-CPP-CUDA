#include "dense.hpp"  // Include the header file for the DenseLayer class

// Parameterized constructor
DenseLayer::DenseLayer(int num_inputs, int num_neurons) : num_neurons(num_neurons) {
    // Resize the neurons vector and initialize neurons with specified number of inputs
    neurons.resize(num_neurons, Neuron(num_inputs));
    for(int i = 0; i<num_neurons; i++){
        neurons[i].randomize_weights();
    }

}

// Softmax function
vector<double> DenseLayer::softmax(vector<double>& inputs) {
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
vector<double> DenseLayer::calculate_derivative(vector<double>& outputs) {
    vector<double> derivative(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
        //derivative[i] = outputs[i] * (1 - outputs[i]);   // Compute the derivative of the output
        if(outputs[i]>0){
            derivative[i] = 1;
        }
        derivative[i] = 0;
    }
    return derivative;
}

// Function to compute the output of the layer given inputs
vector<double> DenseLayer::compute_output(vector<double>& inputs) {
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
