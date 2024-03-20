#include "maxPooling_flattening.hpp"  // Include the header file

using namespace std;

// Function to perform max pooling
vector<vector<double>> max_pooling(const vector<vector<double>>& input, int pool_size) {
    // Get the dimensions of the input feature map
    int input_height = input.size();    // Height of input feature map
    int input_width = input[0].size();  // Width of input feature map

    // Calculate the dimensions of the output feature map after pooling
    int output_height = input_height / pool_size;    // Height of output feature map
    int output_width = input_width / pool_size;      // Width of output feature map

    // Initialize the pooled output feature map with zeros
    vector<vector<double>> pooled_output(output_height, vector<double>(output_width, 0.0));

    // Iterate over each pooling window in the input feature map
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            // Initialize the maximum value with the top-left element of the pooling window
            double max_val = input[i * pool_size][j * pool_size];

            // Iterate over each element in the pooling window
            for (int u = 0; u < pool_size; ++u) {
                for (int v = 0; v < pool_size; ++v) {
                    // Update the maximum value if a greater value is found in the pooling window
                    max_val = max(max_val, input[i * pool_size + u][j * pool_size + v]);
                }
            }

            // Store the maximum value in the pooled output feature map
            pooled_output[i][j] = max_val;
        }
    }

    // Return the pooled output feature map
    return pooled_output;
}


// Function to flatten a 2D vector into a 1D vector
vector<double> flatten(vector<vector<double>>& inputs) {
    // Initialize the flattened vector with the appropriate size
    vector<double> flattened(inputs.size() * inputs[0].size());
    int index = 0; // Initialize index to keep track of position in the flattened vector

    // Iterate through each row of the inputs vector
    for (int i = 0; i < inputs.size(); ++i) {
        // Iterate through each element of the current row
        for (int j = 0; j < inputs[i].size(); ++j) {
            // Assign the current element to the next position in the flattened vector
            flattened[index++] = inputs[i][j];
        }
    }

    // Return the flattened vector
    return flattened;
}
