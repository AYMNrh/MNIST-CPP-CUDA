#ifndef NEURON_H
#define NEURON_H

#include <vector>

using namespace std;

class Neuron {
private:
    vector<double> weights;    // Weights of the neuron
    int num_inputs;                  // Number of inputs to the neuron
    double delta;                    // Delta value for backpropagation
    double derivative;               // Derivative of the neuron's output
    vector<double> inputs;     // Inputs to the neuron

public:
    Neuron();                        // Default constructor
    Neuron(int num_inputs);          // Constructor with number of inputs specified

private:
    void randomize_weights();        // Function to randomize weights

public:
    double compute_output(vector<double>& inputs);  // Function to compute the output of the neuron given inputs
};

#endif
