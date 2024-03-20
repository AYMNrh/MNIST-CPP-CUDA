#include "training.hpp"

ConvolutionalLayer convolutional_back_Propagation(ConvolutionalLayer convolutional_layer, const vector<vector<double>>& train_image, double learning_rate) {
    int num_filters = convolutional_layer.num_filters;
    int filter_size = convolutional_layer.filter_size;
    int stride = convolutional_layer.stride;

    int input_height = train_image.size();
    int input_width = train_image[0].size();

    // Apply convolution operation on the image
    vector<vector<vector<double>>> conv_output = convolutional_layer.apply_convolution(train_image);

    // Initialize gradients for filters
    vector<vector<vector<double>>> filter_gradients(num_filters, vector<vector<double>>(filter_size, vector<double>(filter_size, 0.0)));

    // Compute gradients for filters
    for (int k = 0; k < num_filters; ++k) {
        for (int u = 0; u < filter_size; ++u) {
            for (int v = 0; v < filter_size; ++v) {
                for (int i = 0; i < input_height; ++i) {
                    for (int j = 0; j < input_width; ++j) {
                        int input_i = i * stride + u;
                        int input_j = j * stride + v;
                        if (input_i >= 0 && input_i < input_height && input_j >= 0 && input_j < input_width) {
                            filter_gradients[k][u][v] += train_image[i][j] * conv_output[k][i][j];
                        }
                    }
                }
            }
        }
    }

    // Update filter weights
    for (int k = 0; k < num_filters; ++k) {
        for (int u = 0; u < filter_size; ++u) {
            for (int v = 0; v < filter_size; ++v) {
                convolutional_layer.filters[k][u][v] -= learning_rate * filter_gradients[k][u][v];
            }
        }
    }

    return convolutional_layer;
}

double crossentropy_error(const vector<int> training_label, const vector<double> predict_output) {
    // initialize the sum
    double sum = 0;
    // In our case equals to 10
    int n = predict_output.size();

    for (int i=0; i<n; i++) {
        sum += training_label[i] * log(predict_output[i]);
    }

    return -sum;
}

DenseLayer dense_back_Propagation(DenseLayer dense_layer, const vector<vector<vector<double>>> train_images, const vector<int> training_label, double learning_rate) {
    for (int i = 0; i < dense_layer.num_neurons; i++) {
        Neuron& neuron = dense_layer.neurons[i];
        for (int j = 0; j < neuron.num_inputs; j++) {
            neuron.weights[j] -= learning_rate * training_label[i] * dense_layer.derivative[i] * neuron.inputs[j];
        }
    }
    return dense_layer;
}