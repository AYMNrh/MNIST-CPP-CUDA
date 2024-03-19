#include "conv_layer.h"
#include <cmath>


void ConvLayer::initialize() {
    // Initialize neurons in the convolutional layer
    for (int i = 0; i < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++i) {
        for (int j = 0; j < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++j) {
            initialize_neuron(neurons[i][j]);
        }
    }
}

ConvOutput ConvLayer::conv_forward(const double input_image[][IMAGE_SIZE]) const {
    ConvOutput conv_output;

    for (int i = 0; i < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++i) {
        for (int j = 0; j < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++j) {
            double input_patch[CONV_FILTER_SIZE][CONV_FILTER_SIZE];

            // Extract the input patch
            for (int di = 0; di < CONV_FILTER_SIZE; ++di) {
                for (int dj = 0; dj < CONV_FILTER_SIZE; ++dj) {
                    input_patch[di][dj] = input_image[i * POOL_SIZE + di][j * POOL_SIZE + dj];
                }
            }

            // Perform convolution and apply ReLU activation
            for (int k = 0; k < NUM_CONV_FILTERS; ++k) {
                conv_output.output[k][i][j] = relu(convolution(input_patch, neurons[i][j]));
            }
        }
    }

    return conv_output;
}

ConvOutput ConvLayer::max_pooling_forward(const ConvOutput& conv_output) const {
    ConvOutput pooled_output;

    for (int k = 0; k < NUM_CONV_FILTERS; ++k) {
        for (int i = 0; i < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++i) {
            for (int j = 0; j < (IMAGE_SIZE - CONV_FILTER_SIZE + 1) / POOL_SIZE; ++j) {
                double max_val = -INFINITY;

                // Find the maximum value in the pooling window
                for (int di = 0; di < POOL_SIZE; ++di) {
                    for (int dj = 0; dj < POOL_SIZE; ++dj) {
                        max_val = std::max(max_val, conv_output.output[k][i * POOL_SIZE + di][j * POOL_SIZE + dj]);
                    }
                }

                // Store the maximum value in the pooled output
                pooled_output.output[k][i][j] = max_val;
            }
        }
    }

    return pooled_output;
}

void ConvLayer::update_weights(const double input_image[][IMAGE_SIZE], const double target[], double learning_rate) {
    // Implement weight update using gradient descent
    // You can modify this function according to your training algorithm
}

double ConvLayer::convolution(const double input_patch[][CONV_FILTER_SIZE], const Neuron& neuron) const {
    double result = neuron.bias;
    for (int i = 0; i < CONV_FILTER_SIZE; ++i) {
        for (int j = 0; j < CONV_FILTER_SIZE; ++j) {
            result += input_patch[i][j] * neuron.weights[i * CONV_FILTER_SIZE + j];
        }
    }
    return result;
}

double ConvLayer::relu(double x) const {
    return std::max(0.0, x);
}