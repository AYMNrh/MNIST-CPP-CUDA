#include <iostream>
#include <fstream>
#include "neuron.hpp"
#include "dense.hpp"
#include "conv_layer.hpp"
#include "maxPooling_flattening.hpp" 

using namespace std;

int detect_class(vector<double> output){
    double max = 0;
    int output_class;
    for (int i=0; i<output.size(); i++) {
        if (output[i]>max){
            max = output[i];
            output_class = i;
        }
    }
    return output_class;
}

// Function to read MNIST images
vector<vector<vector<double>>> read_images(const string& file_path, int num_images, int num_rows, int num_cols) {
    ifstream file(file_path, ios::binary);

    vector<vector<vector<double>>> images(num_images, vector<vector<double>>(num_rows, vector<double>(num_cols)));

    // Read pixel values
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                unsigned char pixel_value;
                file.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
                images[i][r][c] = static_cast<double>(pixel_value) / 255.0; // Normalize pixel values
            }
        }
    }

    return images;
}

// Function to read MNIST labels
vector<int> read_labels(const string& file_path, int num_labels) {
    ifstream file(file_path, ios::binary);

    vector<int> labels(num_labels);

    // Read labels
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

int main() {

    // MNIST dataset paths
    const string train_images_path = "data/train-images-idx3-ubyte";
    const string train_labels_path = "data/train-labels-idx1-ubyte";
    const string test_images_path = "data/t10k-images-idx3-ubyte";
    const string test_labels_path = "data/t10k-labels-idx1-ubyte";


    vector<vector<vector<double>>> train_images = read_images(train_images_path, 60000, 28, 28);
    vector<int> train_labels = read_labels(train_labels_path, 60000);
    vector<vector<vector<double>>> test_images = read_images(test_images_path, 10000, 28, 28);
    vector<int> test_labels = read_labels(test_labels_path, 10000);

    // Transform labels in vectors which can be compared to the output
    vector<vector<int>>real_outputs(train_labels.size());
    
    // Loop over training labels
    for(int i=0; i<train_labels.size(); i++) {
        // Initialize vector
        vector<int> real_output(10, 0);
        int train_label = train_labels[i];
        // set 1 to the appropriate index of real output
        switch(train_label) {
            case 0:
                real_output[0]=1;
                break;
            case 1:
                real_output[1]=1;
                break;
            case 2:
                real_output[2]=1;
                break;
            case 3:
                real_output[3]=1;
                break;
            case 4:
                real_output[4]=1;
                break;
            case 5:
                real_output[5]=1;
                break;
            case 6:
                real_output[6]=1;
                break;
            case 7:
                real_output[7]=1;
                break;
            case 8:
                real_output[8]=1;
                break;
            case 9:
                real_output[9]=1;
                break;
            default:
                cout << "Invalid label value" << endl;
                break;
        }
        real_outputs[i] = real_output;
    }

    // Initialize convolutional layer
    ConvolutionalLayer conv_layer(32, 3, 1);

    // Initialize dense layer
    DenseLayer dense_layer(25088, 10);

    for (int i=0; i<train_images.size(); i++) {
        vector<vector<double>> convolution_output = conv_layer.apply_convolution(train_images[i]);
        vector<vector<double>> pooling_output = max_pooling(convolution_output, 2);
        vector<double> flattened_output = flatten(pooling_output);
        vector<double> output = dense_layer.compute_output(flattened_output);
        int output_class = detect_class(output);
        cout << "label value for predicted class : " << real_outputs[i][output_class] << "     predicted value : " << output_class << endl;
    }

    // Train the CNN


    return 0;
}
