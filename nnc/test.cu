#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MNIST_IMAGE_SIZE 28 * 28
#define MNIST_NUM_IMAGES 60000
#define MNIST_NUM_LABELS 10
#define MNIST_NUM_IMAGES_TEST 9000

double relu_derivative(double x);
void printProgressBar(double progress);

__global__ void relu_kernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (x[idx] < 0) {
            x[idx] = 0;
        }
    }
}

double* relu_cuda(double* x, int size) {
    double* d_x;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMemcpyAsync(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize>>>(d_x, size);

    cudaMemcpyAsync(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    return x;
}

__global__ void softmax_kernel(double* x, int size, double max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = exp(x[idx] - max_val);
    }
}

__global__ void normalize_kernel(double* x, int size, double* x_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] /= *x_sum;
    }
}

double* softmax_cuda(double* x, int size) {
    double* d_x;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMemcpyAsync(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    softmax_kernel<<<numBlocks, blockSize>>>(d_x, size, max_val);

    double* d_x_sum;
    cudaMalloc(&d_x_sum, sizeof(double));
    cudaMemcpyAsync(d_x_sum, &x[0], sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 1; i < size; i++) {
        atomicAdd(d_x_sum, x[i]);
    }

    normalize_kernel<<<numBlocks, blockSize>>>(d_x, size, d_x_sum);

    cudaMemcpyAsync(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_x_sum);
    return x;
}

__global__ void categorical_cross_entropy_loss_kernel(double* x, double* target, int size, double* res) {
    __shared__ double shared_res;
    if (threadIdx.x == 0) {
        shared_res = 0.0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&shared_res, target[idx] * log(x[idx] + 0.000000000000001));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(res, shared_res);
    }
}

double categorical_cross_entropy_loss(double* x, double* target, int size) {
    double* d_x, *d_target, *d_res;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_target, size * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    cudaMemsetAsync(d_res, 0, sizeof(double));

    cudaMemcpyAsync(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_target, target, size * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    categorical_cross_entropy_loss_kernel<<<numBlocks, blockSize>>>(d_x, d_target, size, d_res);

    double res;
    cudaMemcpyAsync(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_target);
    cudaFree(d_res);

    return res;
}

void layer_builder(int size, int size_input, double** l_p, double** l_b) {
    *l_p = (double*)malloc(size * size_input * sizeof(double));
    *l_b = (double*)malloc(size * sizeof(double));

    for (int i = 0; i < size * size_input; i++) {
        (*l_p)[i] = rand() / (RAND_MAX + 1.0);
    }
    for (int i = 0; i < size; i++) {
        (*l_b)[i] = rand() / (RAND_MAX + 1.0);
    }

    printf("Number of params for layer: %d\n", size * size_input);
}

__global__ void layerInfKernel(double* l_p, double* l_b, int num_outputs, double* neurons, int num_inputs, double* output) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < num_outputs) {
        double sum = l_b[j];
        for (int i = 0; i < num_inputs; i++) {
            sum += neurons[i] * l_p[j * num_inputs + i];
        }
        output[j] = sum;
    }
}

double* layer_inf_cuda(double* l_p, double* l_b, int num_outputs, double* neurons, int num_inputs) {
    double* output = (double*)malloc(num_outputs * sizeof(double));
    double* d_l_p, * d_l_b, * d_neurons, * d_output;

    cudaMalloc(&d_l_p, num_outputs * num_inputs * sizeof(double));
    cudaMalloc(&d_l_b, num_outputs * sizeof(double));
    cudaMalloc(&d_neurons, num_inputs * sizeof(double));
    cudaMalloc(&d_output, num_outputs * sizeof(double));

    cudaMemcpyAsync(d_l_p, l_p, num_outputs * num_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_l_b, l_b, num_outputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_neurons, neurons, num_inputs * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_outputs + blockSize - 1) / blockSize;
    layerInfKernel <<<gridSize, blockSize >>> (d_l_p, d_l_b, num_outputs, d_neurons, num_inputs, d_output);

    cudaMemcpyAsync(output, d_output, num_outputs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_l_p);
    cudaFree(d_l_b);
    cudaFree(d_neurons);
    cudaFree(d_output);

    return output;
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

void custom_MLP_2l_build(int size_input, int size_l2, int size_output, double** l1p, double** l1b, double** l2p, double** l2b) {
    printf("Build layer 1\n");
    layer_builder(size_l2, size_input, l1p, l1b);
    printf("Build layer 2\n");
    layer_builder(size_output, size_l2, l2p, l2b);
}

double* custom_MLP_2l_inf(double* input_data, int size_input, int size_l2, int size_output, double* l1p, double* l1b, double* l2p, double* l2b) {
    double* n1 = (double*)malloc(size_l2 * sizeof(double));
    double* n2 = (double*)malloc(size_output * sizeof(double));

    double* d_input_data, *d_n1, *d_n2;
    cudaMalloc(&d_input_data, size_input * sizeof(double));
    cudaMalloc(&d_n1, size_l2 * sizeof(double));
    cudaMalloc(&d_n2, size_output * sizeof(double));

    cudaMemcpyAsync(d_input_data, input_data, size_input * sizeof(double), cudaMemcpyHostToDevice);

    n1 = layer_inf_cuda(l1p, l1b, size_l2, d_input_data, size_input);
    n1 = relu_cuda(n1, size_l2);
    n2 = layer_inf_cuda(l2p, l2b, size_output, n1, size_l2);
    n2 = softmax_cuda(n2, size_output);

    cudaMemcpyAsync(n1, d_n1, size_l2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(n2, d_n2, size_output * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_input_data);
    cudaFree(d_n1);
    cudaFree(d_n2);
    free(n1);

    return n2;
}

double evaluate_accuracy(double* l1p, double* l1b, double* l2p, double* l2b, int size_input, int size_l2, int size_output) {
    uint8_t** test_images = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST* sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        test_images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t** one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t* test_labels = (uint8_t*)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    read_mnist_images("train-images.idx3-ubyte", test_images, MNIST_NUM_IMAGES_TEST);
    read_mnist_labels("train-labels.idx1-ubyte", test_labels, one_hot_labels, MNIST_NUM_IMAGES_TEST);
    double correct_predictions = 0.0;
    double* input_data = (double*)malloc(MNIST_IMAGE_SIZE * sizeof(double));
    double* output;

    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            input_data[j] = (double)(test_images[i][j]) / 255.0;
        }
        output = custom_MLP_2l_inf(input_data, MNIST_IMAGE_SIZE, size_l2, size_output, l1p, l1b, l2p, l2b);
        int predicted_label = 0;
        double max_output = output[0];
        for (int j = 1; j < size_output; j++) {
            if (output[j] > max_output) {
                max_output = output[j];
                predicted_label = j;
            }
        }

        int true_label = 0;
        for (int j = 0; j < MNIST_NUM_LABELS; j++) {
            if (one_hot_labels[i][j] == 1) {
                true_label = j;
                break;
            }
        }

        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }
    free(input_data);
    free(test_images);
    free(test_labels);
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(one_hot_labels[i]);
    }
    free(one_hot_labels);

    return correct_predictions / MNIST_NUM_IMAGES_TEST;
}

void train_model(double* l1p, double* l1b, double* l2p, double* l2b, int size_input, int size_l2, int size_output, double learning_rate, int epochs) {
    uint8_t** train_images = (uint8_t**)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        train_images[i] = (uint8_t*)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t** one_hot_labels = (uint8_t**)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t*));
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        one_hot_labels[i] = (uint8_t*)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t* labels = (uint8_t*)malloc(MNIST_NUM_IMAGES * sizeof(uint8_t));

    printf("Loading training images...\n");
    read_mnist_images("train-images.idx3-ubyte", train_images, MNIST_NUM_IMAGES);
    printf("Loading training labels...\n");
    read_mnist_labels("train-labels.idx1-ubyte", labels, one_hot_labels, MNIST_NUM_IMAGES);

    double* d_l1p, *d_l1b, *d_l2p, *d_l2b;
    cudaMalloc(&d_l1p, size_input * size_l2 * sizeof(double));
    cudaMalloc(&d_l1b, size_l2 * sizeof(double));
    cudaMalloc(&d_l2p, size_l2 * size_output * sizeof(double));
    cudaMalloc(&d_l2b, size_output * sizeof(double));

    cudaMemcpyAsync(d_l1p, l1p, size_input * size_l2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_l1b, l1b, size_l2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_l2p, l2p, size_l2 * size_output * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_l2b, l2b, size_output * sizeof(double), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        printf("Epoch %d: ", epoch + 1);
        printProgressBar((double)(epoch + 1) / epochs);
        printf("\n");

        for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
            // Forward pass
            double* input_data = (double*)malloc(MNIST_IMAGE_SIZE * sizeof(double));
            double* output;

            for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
                input_data[j] = (double)(train_images[i][j]) / 255.0;
            }
            output = custom_MLP_2l_inf(input_data, MNIST_IMAGE_SIZE, size_l2, size_output, d_l1p, d_l1b, d_l2p, d_l2b);

            // Calculate loss
            double* target = (double*)malloc(size_output * sizeof(double));
            for (int j = 0; j < MNIST_NUM_LABELS; j++) {
                target[j] = one_hot_labels[i][j];
            }
            double loss = categorical_cross_entropy_loss(output, target, size_output);
            total_loss += loss;

            // Backpropagation
            // Update layer 2 weights and biases
            double* l2_deltas = (double*)malloc(size_output * sizeof(double));
            for (int j = 0; j < size_output; j++) {
                l2_deltas[j] = output[j] - target[j];
            }
            double* d_l2_deltas;
            cudaMalloc(&d_l2_deltas, size_output * sizeof(double));
            cudaMemcpyAsync(d_l2_deltas, l2_deltas, size_output * sizeof(double), cudaMemcpyHostToDevice);

            int blockSize = 256;
            int numBlocks = (size_output + blockSize - 1) / blockSize;
            update_layer2_kernel<<<numBlocks, blockSize>>>(d_l2p, d_l2b, d_l2_deltas, n1, size_l2, size_output, learning_rate);

            // Update layer 1 weights and biases
            double* l1_deltas = (double*)malloc(size_l2 * sizeof(double));
            for (int j = 0; j < size_l2; j++) {
                double sum = 0.0;
                for (int k = 0; k < size_output; k++) {
                    sum += l2_deltas[k] * d_l2p[k * size_l2 + j];
                }
                l1_deltas[j] = sum * relu_derivative(n1[j]);
            }
            double* d_l1_deltas;
            cudaMalloc(&d_l1_deltas, size_l2 * sizeof(double));
            cudaMemcpyAsync(d_l1_deltas, l1_deltas, size_l2 * sizeof(double), cudaMemcpyHostToDevice);

            numBlocks = (size_input + blockSize - 1) / blockSize;
            update_layer1_kernel<<<numBlocks, blockSize>>>(d_l1p, d_l1b, d_l1_deltas, input_data, size_input, size_l2, learning_rate);

            free(l2_deltas);
            free(l1_deltas);
            cudaFree(d_l2_deltas);
            cudaFree(d_l1_deltas);
            free(output);
            free(target);
            free(input_data);
        }

        printf("Epoch %d loss: %f\n", epoch + 1, total_loss / MNIST_NUM_IMAGES);
    }

    cudaMemcpyAsync(l1p, d_l1p, size_input * size_l2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(l1b, d_l1b, size_l2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(l2p, d_l2p, size_l2 * size_output * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(l2b, d_l2b, size_output * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_l1p);
    cudaFree(d_l1b);
    cudaFree(d_l2p);
    cudaFree(d_l2b);

    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        free(train_images[i]);
        free(one_hot_labels[i]);
    }
    free(train_images);
    free(one_hot_labels);
    free(labels);
}


int main() {
    srand(time(NULL));

    int size_input = MNIST_IMAGE_SIZE;
    int size_l2 = 256;
    int size_output = MNIST_NUM_LABELS;

    double* l1p, *l1b, *l2p, *l2b;
    custom_MLP_2l_build(size_input, size_l2, size_output, &l1p, &l1b, &l2p, &l2b);

    double learning_rate = 0.001;
    int epochs = 10;

    train_model(l1p, l1b, l2p, l2b, size_input, size_l2, size_output, learning_rate, epochs);

    double accuracy = evaluate_accuracy(l1p, l1b, l2p, l2b, size_input, size_l2, size_output);
    printf("Test accuracy: %.2f%%\n", accuracy * 100);

    free(l1p);
    free(l1b);
    free(l2p);
    free(l2b);

    return 0;
}