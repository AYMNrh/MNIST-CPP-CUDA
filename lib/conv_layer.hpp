#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class ConvolutionalLayer {
public:
    vector<vector<vector<double>>> filters; // Filters of the convolutional layer
    int num_filters; // Number of filters
    int filter_size; // Size of each filter (assuming square filters)
    int stride; // Stride of the convolution operation

    // Constructor
    ConvolutionalLayer(int num_filters, int filter_size, int stride);

    // Function to apply convolution operation
    vector<vector<double>> apply_convolution(const vector<vector<double>>& input);

private:
    // Function to randomize filter weights
    void randomize_filters();
};

#endif
