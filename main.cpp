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
#define MNIST_NUM_IMAGES 1000
#define MNIST_NUM_LABELS 100
#define MNIST_NUM_IMAGES_TEST 10

void writeLabelsToFile(const uint8_t* labels, int num_labels, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // Write the labels to the file
    for (int i = 0; i < num_labels; ++i) {
        file << static_cast<int>(labels[i]) << "\n"; // Convert uint8_t to int and write to file
    }

    file.close();
}

void printVector(const vector<double>& vec) {
    for (const auto& elem : vec) {
        cout << elem << " ";
    }
    cout << endl;
}

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
    int output_class = -1;
    for (int i = 0; i < output.size(); i++) {
        if (output[i] > max) {
            max = output[i];
            output_class = i;
        }
    }
    return output_class;
}

vector<vector<vector<double>>> convert_images(uint8_t** images, int num_images) {
    // Convert uint8_t* images into vector<vector<vector<double>>>
    vector<vector<vector<double>>> transformed_images(num_images, vector<vector<double>>(28, vector<double>(28)));
    for (int n = 0; n < num_images; ++n) {
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                // Calculate the index in the 1D array for the current pixel
                int index = i * 28 + j;
                // Cast each uint8_t value to double and assign it to the corresponding position in the result vector
                transformed_images[n][i][j] = static_cast<double>(images[n][index]) / 255.0;
            }
        }
    }
    return transformed_images;
}

int main() {
    // Initialize convolutional layer
    ConvolutionalLayer conv_layer(32, 3, 1);

    // Initialize dense layer
    DenseLayer dense_layer(5408, 10);

    // Initialize training accuracy
    double accuracy = 0.0;
    // Initialize precision for testing
    double precision = 0.0;

    // Load training data
    uint8_t** train_images = (uint8_t**)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        train_images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }

    uint8_t** train_one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        train_one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }

    uint8_t* train_labels = (uint8_t*)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t));


    printf("Loading training data...\n");
    read_mnist_images("../data/train-images.idx3-ubyte", train_images, MNIST_NUM_IMAGES);
    read_mnist_labels("../data/train-labels.idx1-ubyte", train_labels, train_one_hot_labels, MNIST_NUM_IMAGES);
    // Write labels to file
    writeLabelsToFile(train_labels, MNIST_NUM_IMAGES, "labels.txt");
    printf("Loading done \n");

    // Load testing data
    uint8_t** test_images = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        test_images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }

    uint8_t** test_one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        test_one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }

    uint8_t* test_labels = (uint8_t*)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    printf("Loading testing data...\n");
    read_mnist_images("../data/t10k-images.idx3-ubyte", test_images, MNIST_NUM_IMAGES_TEST);
    read_mnist_labels("../data/t10k-labels.idx1-ubyte", test_labels, test_one_hot_labels, MNIST_NUM_IMAGES_TEST);
    printf("Loading done \n");

    // Transform images
    vector<vector<vector<double>>> training_images = convert_images(train_images, MNIST_NUM_IMAGES);
    vector<vector<vector<double>>> testing_images = convert_images(test_images, MNIST_NUM_IMAGES_TEST);

    // Loop over training labprintf("output class : %d", output_class);els
    printf("Start Training \n");
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        vector<vector<vector<double>>> convolution_output = conv_layer.apply_convolution(training_images[i]);
        vector<vector<vector<double>>> pooling_output = max_pooling(convolution_output, 2);
        vector<double> flattened_output = flatten(pooling_output);
        
        vector<double> output = dense_layer.compute_output(flattened_output);

        // Train CNN
        //conv_layer = convolutional_back_Propagation(conv_layer, training_images[i], convolution_output, 0.1);
        //convolution_output = conv_layer.apply_convolution(training_images[i]);
        //pooling_output = max_pooling(convolution_output, 2);
        //flattened_output = flatten(pooling_output);
        //dense_layer = dense_back_Propagation(dense_layer, train_one_hot_labels[i], 0.1);
        //output = dense_layer.compute_output(flattened_output);
        int output_class = detect_class(output);

        if (output_class == train_labels[i]) {
            accuracy += 1;
        }
        printf("output class : %d\n", output_class);
    }

    // Free allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        free(train_images[i]);
        free(train_one_hot_labels[i]);
    }
    
    free(train_images);
    free(train_one_hot_labels);
    free(train_labels);

    printf("Training Done \n");

    double training_accuracy = accuracy / MNIST_NUM_IMAGES;

    // Display accuracy
    printf("Training Accuracy : %f \n", training_accuracy);

    for(int i=0; i<MNIST_NUM_IMAGES_TEST; i++) {
        vector<vector<vector<double>>> test_convolution_output = conv_layer.apply_convolution(testing_images[i]);
        vector<vector<vector<double>>> test_pooling_output = max_pooling(test_convolution_output, 2);
        vector<double> test_flattened_output = flatten(test_pooling_output);
        vector<double> test_output = dense_layer.compute_output(test_flattened_output);
        int test_output_class = detect_class(test_output);

        if (test_output_class == test_labels[i]) {
            precision += 1;
        }
    }


    // Free allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(test_images[i]);
        free(test_one_hot_labels[i]);
    }
    
    free(test_images);
    free(test_one_hot_labels);
    free(test_labels);

    precision /= MNIST_NUM_IMAGES_TEST;

    // Display precision
    printf("Testing precision : %f\n", precision);

};
