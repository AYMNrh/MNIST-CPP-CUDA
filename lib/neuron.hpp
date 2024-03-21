#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class Neuron {
    public:
        int num_inputs;                  // Number of inputs to the neuron
        vector<double> weights;    // Weights of the neuron
        vector<double> inputs;     // Inputs to the neuron

    public:
        Neuron(int num_inputs);          // Constructor with number of inputs specified:
        void randomize_weights();        // Function to randomize weights

    public:
        double compute_output(vector<double>& inputs);  // Function to compute the output of the neuron given inputs
};

#endif
