#ifndef CONV_LAYER_H
#define CONV_LAYER_H

constexpr int IMAGE_SIZE = 28;
constexpr int CONV_FILTER_SIZE = 3;
constexpr int POOL_SIZE = 2;

#include "neuron.h"

struct ConvOutput {
    double output[NUM_CONV_FILTERS][(IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE][(IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE];
};

class ConvLayer {
public:
    Neuron neurons[(IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE][(IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE];

    ConvLayer() {
        initialize();
    }

    void initialize();
    ConvOutput conv_forward(const double input_image[][IMAGE_SIZE]) const;
    ConvOutput max_pooling_forward(const ConvOutput& conv_output) const;
    void update_weights(const double input_image[][IMAGE_SIZE], const double target[], double learning_rate);

private:
    double convolution(const double input_patch[][CONV_FILTER_SIZE], const Neuron& neuron) const;
    double relu(double x) const;
};
#endif /* CONV_LAYER_H */
