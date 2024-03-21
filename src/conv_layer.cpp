#include "conv_layer.hpp"

// Constructor implementation
ConvolutionalLayer::ConvolutionalLayer(int num_filters, int filter_size, int stride)
    : num_filters(num_filters), filter_size(filter_size), stride(stride) {
    // Initialize filters
    filters.resize(num_filters, vector<vector<double>>(filter_size, vector<double>(filter_size)));
    // Randomly initialize filter weights
    randomize_filters();
}

// Function to randomize filter weights
void ConvolutionalLayer::randomize_filters() {
    srand(66); // Seed the random number generator
    // Loop through each filter
    for (int k = 0; k < num_filters; ++k) {
        // Loop through each row of the filter
        for (int i = 0; i < filter_size; ++i) {
            // Loop through each column of the filter
            for (int j = 0; j < filter_size; ++j) {
                // Randomly initialize filter weights in the range [-1, 1]
                filters[k][i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            }
        }
    }
}

// Function to apply convolution operation
vector<vector<vector<double>>> ConvolutionalLayer::apply_convolution(const vector<vector<double>>& input) {
    // Number of rows and columns in input
    int input_height = input.size();
    int input_width = input[0].size();
    // Calculate output dimensions
    int output_height = (input_height - filter_size) / stride + 1;
    int output_width = (input_width - filter_size) / stride + 1;

    // Initialize the output matrix with zeros
    vector<vector<vector<double>>> output(num_filters, vector<vector<double>>(output_height, vector<double>(output_width, 0.0)));
    
    // Loop through each filter
    for (int k = 0; k < num_filters; ++k) {
        // Loop through each row of the output
        for (int i = 0; i < output_height; ++i) {
            // Loop through each column of the output
            for (int j = 0; j < output_width; ++j) {
                double sum = 0.0;
                // Loop through each row of the filter
                for (int u = 0; u < filter_size; ++u) {
                    // Loop through each column of the filter
                    for (int v = 0; v < filter_size; ++v) {
                        // Calculate the input indices based on the stride
                        int input_i = i * stride + u;
                        int input_j = j * stride + v;
                        // Check if the input indices are within the input dimensions
                        if (input_i >= 0 && input_i < input_height && input_j >= 0 && input_j < input_width) {
                            // Perform the convolution operation and accumulate the result
                            sum += input[input_i][input_j] * filters[k][u][v];
                        }
                    }
                }
                // Update the output matrix with the accumulated sum
                output[k][i][j] += sum;
            }
        }
    }

    return output;
}
