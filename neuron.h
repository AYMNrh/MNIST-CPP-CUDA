#ifndef NEURON_H
#define NEURON_H

constexpr int NUM_CONV_FILTERS = 32;

struct Neuron {
    double weights[NUM_CONV_FILTERS];
    double bias;
    
    Neuron() {
        // Initialize weights and bias
        for (int i = 0; i < NUM_CONV_FILTERS; ++i) {
            weights[i] = 0.0; // Initialize weights randomly or with zeros
        }
        bias = 0.0; // Initialize bias
    }
};

void initialize_neuron(Neuron& neuron);
double neuron_forward(const double input[], const Neuron& neuron);

#endif /* NEURON_H */
