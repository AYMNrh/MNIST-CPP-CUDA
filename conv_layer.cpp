#include "conv_layer.h"  // Include the header file for the ConvolutionalLayer class
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
    ConvolutionalLayer(int num_filters, int filter_size, int stride) :
            num_filters(num_filters), filter_size(filter_size), stride(stride) {
        // Initialize filters
        filters.resize(num_filters, vector<vector<double>>(filter_size, vector<double>(filter_size)));
        // Randomly initialize filter weights
        randomize_filters();
    }

    // Function to randomize filter weights
    void randomize_filters() {
        srand(time(0)); // Seed the random number generator
        // Loop through each filter
        for (int k = 0; k < num_filters; ++k) {
            // Loop through each row of the filter
            for (int i = 0; i < filter_size; ++i) {
                // Loop through each column of the filter
                for (int j = 0; j < filter_size; ++j) {
                    // Randomly initialize filter weights in the range [-1, 1]
                    filters[k][i][j] = ((double) rand() / RAND_MAX - 0.5) * 2.0 ;
                }
            }
        }
    }

    // Function to apply convolution operation
    vector<vector<double>> apply_convolution(const vector<vector<double>>& input) {
        // number of rows
        int input_height = input.size();
        int output_height = (input_height - filter_size) / stride + 1;
        // number of columns
        int input_width = input[0].size();
        int output_width = (input_width - filter_size) / stride + 1;

        // Initialize the output matrix with zeros
        vector<vector<double>> output(output_height, vector<double>(output_width, 0.0));

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
                    output[i][j] += sum;
                }
            }
        }

        return output;
    }
};
