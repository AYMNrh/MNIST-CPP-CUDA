#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include "neuron.hpp"
#include "dense.hpp"
#include "conv_layer.hpp"
#include "maxPooling_flattening.hpp"
#include "training.hpp"

using namespace std;

#define MNIST_IMAGE_SIZE 28 * 28
#define MNIST_NUM_IMAGES 60000
#define MNIST_NUM_LABELS 10
#define MNIST_NUM_IMAGES_TEST 9000

void read_mnist_images(const char* image_file_path, uint8_t** images, int num_images) {
    FILE* file = fopen(image_file_path, "rb");
    if (file == NULL) {
        printf("Error opening file: %s\n", image_file_path);
        return;
    }

    uint32_t magic_number, num_images_file;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_images_file, sizeof(uint32_t), 1, file);

    uint32_t rows, cols;
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);

    for (int i = 0; i < num_images; i++) {
        fread(images[i], sizeof(uint8_t), MNIST_IMAGE_SIZE, file);
    }

    fclose(file);
}

void read_mnist_labels(const char* label_file_path, uint8_t* labels, uint8_t** one_hot_labels, int num_labels) {
    FILE* file = fopen(label_file_path, "rb");
    if (file == NULL) {
        printf("Error opening file: %s\n", label_file_path);
        return;
    }

    uint32_t magic_number, num_labels_file;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&num_labels_file, sizeof(uint32_t), 1, file);

    for (int i = 0; i < num_labels; i++) {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, file);
        labels[i] = label;
        for (int j = 0; j < MNIST_NUM_LABELS; j++) {
            one_hot_labels[i][j] = (j == label) ? 1 : 0;
        }
    }

    fclose(file);
}

void printProgressBar(double progress) {
    const int barWidth = 70;

    int pos = barWidth * progress;

    printf("[");
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }
    printf("] %.2f%%", progress * 100);
}

int detect_class(vector<double> output) {
    double max = 0;
    int output_class;
    for (int i = 0; i < output.size(); i++) {
        if (output[i] > max) {
            max = output[i];
            output_class = i;
        }
    }
    return output_class;
}

int main() {

    // Initialize convolutional layer
    ConvolutionalLayer conv_layer(32, 3, 1);

    // Initialize dense layer
    DenseLayer dense_layer(100, 10);

    // Initialize accuracy
    double accuracy = 0.0;

    // Load training data
    uint8_t** images = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t** one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t* labels = (uint8_t*)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    printf("Loading training data...\n");
    read_mnist_images("../data/train-images.idx3-ubyte", images, MNIST_NUM_IMAGES_TEST);
    read_mnist_labels("../data/train-labels.idx1-ubyte", labels, one_hot_labels, MNIST_NUM_IMAGES_TEST);

    vector<vector<vector<double>>> training_images(MNIST_NUM_IMAGES, vector<vector<double>>(28, vector<double>(28)));
    for (int n=0; n<MNIST_NUM_IMAGES; ++n){
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                // Cast each uint8_t value to double and assign it to the corresponding position in the result vector
                training_images[n][i][j] = images[i][j];
                if (images[n][i] != 0){
                    cout << "training_images : " << training_images[n][i][j] << endl;
                    cout << "training_images : " << images[n][i] << endl;
                }  
            }
        }
    }

    // Loop over training labels
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        vector<vector<vector<double>>> convolution_output = conv_layer.apply_convolution(training_images[i]);
        vector<vector<vector<double>>> pooling_output = max_pooling(convolution_output, 2);
        vector<double> flattened_output = flatten(pooling_output);
        vector<double> output = dense_layer.compute_output(flattened_output);
        int output_class = detect_class(output);
        //cout << "predicted output : " << output_class << "   label : " << labels[i] << endl;
        if (output[output_class] == one_hot_labels[i][output_class]) {
            accuracy += 1;
        }
    }

    // Free allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(images[i]);
        free(one_hot_labels[i]);
    }
    
    free(images);
    free(one_hot_labels);
    free(labels);

    // Display accuracy
    cout << "Accuracy : " << accuracy / MNIST_IMAGE_SIZE;
};
