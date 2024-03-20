#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define MNIST_IMAGE_SIZE 28 * 28
#define MNIST_NUM_IMAGES 60000
#define MNIST_NUM_LABELS 10
#define MNIST_NUM_IMAGES_TEST 9000

#define BLOCK_SIZE 16 // Adjust block size according to your GPU architecture

__global__ void matrix_multiply(double *a, double *b, double *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__device__ double relu(double x) {
    return x < 0 ? 0 : x;
}

__device__ double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

__global__ void apply_relu(double *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = relu(x[idx]);
    }
}

__global__ void softmax(double *x, int size) {
    // Assuming x is a single row matrix (1 x size)
    double max_val = x[0];
    double x_sum = 0.0;

    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        x_sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= x_sum;
    }
}

__global__ void categorical_cross_entropy_loss(double *x, double *target, double *res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        res[idx] = -target[idx] * log(x[idx] + 0.00000000000000001); // Adding small epsilon to prevent log(0)
    }
}

void layer_builder(int size, int size_input, double *l_p, double *l_b) {
    for (int i = 0; i < size * size_input; i++) {
        l_p[i] = rand() / (RAND_MAX + 1.0);
    }
    for (int i = 0; i < size; i++) {
        l_b[i] = rand() / (RAND_MAX + 1.0);
    }

    printf("Number of params for layer: %d\n", size * size_input);
}

void custom_MLP_2l_build(int size_input, int size_l2, int size_output, double *l1p, double *l1b, double *l2p,
                          double *l2b) {
    printf("Build layer 1\n");
    layer_builder(size_l2, size_input, l1p, l1b);
    printf("Build layer 2\n");
    layer_builder(size_output, size_l2, l2p, l2b);
}

void custom_MLP_2l_inf(double *input_data, int size_input, int size_l2, int size_output, double *l1p, double *l1b,
                        double *l2p, double *l2b, double *output) {
    double *n1, *n2;
    cudaMalloc(&n1, size_l2 * sizeof(double));
    cudaMalloc(&n2, size_output * sizeof(double));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((size_l2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (size_output + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiply<<<numBlocks, threadsPerBlock>>>(l1p, input_data, n1, size_l2, size_input, 1);
    apply_relu<<<(size_l2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n1, size_l2);
    matrix_multiply<<<numBlocks, threadsPerBlock>>>(l2p, n1, n2, size_output, size_l2, 1);
    softmax<<<1, size_output>>>(n2, size_output);

    cudaMemcpy(output, n2, size_output * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(n1);
    cudaFree(n2);
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
            output = custom_MLP_2l_inf(input_data, MNIST_IMAGE_SIZE, size_l2, size_output, l1p, l1b, l2p, l2b);

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
            for (int j = 0; j < size_output; j++) {
                for (int k = 0; k < size_l2; k++) {
                    l2p[j * size_l2 + k] -= learning_rate * l2_deltas[j] * relu_derivative(l2p[j * size_l2 + k]);
                }
                l2b[j] -= learning_rate * l2_deltas[j];
            }
            free(l2_deltas);
            free(target);

            // Update layer 1 weights and biases
            double* l1_deltas = (double*)malloc(size_l2 * sizeof(double));
            for (int j = 0; j < size_l2; j++) {
                double error = 0.0;
                for (int k = 0; k < size_output; k++) {
                    error += l2p[k * size_l2 + j] * l2_deltas[k];
                }
                l1_deltas[j] = error * relu_derivative(l1b[j]);
            }
            for (int j = 0; j < size_l2; j++) {
                for (int k = 0; k < size_input; k++) {
                    l1p[j * size_input + k] -= learning_rate * l1_deltas[j] * input_data[k];
                }
                l1b[j] -= learning_rate * l1_deltas[j];
            }
            free(l1_deltas);
            free(input_data);
            free(output);
        }
        printf("Epoch %d - Loss: %f\n", epoch+1, total_loss / MNIST_NUM_IMAGES);
    }

    // Free allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        free(train_images[i]);
        free(one_hot_labels[i]);
    }
    free(train_images);
    free(one_hot_labels);
    free(labels);
}
int main() {
    srand(66); // Initialize random seed

    // Load training data
    uint8_t **images = (uint8_t **)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t *));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        images[i] = (uint8_t *)malloc(MNIST_IMAGE_SIZE * sizeof(uint8_t));
    }
    uint8_t **one_hot_labels = (uint8_t **)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t *));
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        one_hot_labels[i] = (uint8_t *)malloc(MNIST_NUM_LABELS * sizeof(uint8_t));
    }
    uint8_t *labels = (uint8_t *)malloc(MNIST_NUM_IMAGES_TEST * sizeof(uint8_t));

    printf("Loading training data...\n");
    read_mnist_images("train-images.idx3-ubyte", images, MNIST_NUM_IMAGES_TEST);
    read_mnist_labels("train-labels.idx1-ubyte", labels, one_hot_labels, MNIST_NUM_IMAGES_TEST);

    // Build model
    double *l1p, *l1b, *l2p, *l2b;
    cudaMallocManaged(&l1p, MNIST_IMAGE_SIZE * 128 * sizeof(double));
    cudaMallocManaged(&l1b, 128 * sizeof(double));
    cudaMallocManaged(&l2p, 128 * MNIST_NUM_LABELS * sizeof(double));
    cudaMallocManaged(&l2b, MNIST_NUM_LABELS * sizeof(double));

    custom_MLP_2l_build(MNIST_IMAGE_SIZE, 128, MNIST_NUM_LABELS, l1p, l1b, l2p, l2b);

    // Measure time taken for training
    clock_t start_time = clock(); // Record start time

    // Train the model
    printf("Training the model...\n");
    train_model(l1p, l1b, l2p, l2b, MNIST_IMAGE_SIZE, 128, MNIST_NUM_LABELS, 0.001, 15);

    clock_t end_time = clock(); // Record end time

    // Calculate elapsed time
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Training completed in %.2f seconds\n", elapsed_time);

    // Evaluate model
    printf("Evaluating the model...\n");
    double accuracy = evaluate_accuracy(l1p, l1b, l2p, l2b, MNIST_IMAGE_SIZE, 128, MNIST_NUM_LABELS);
    printf("Accuracy: %f\n", accuracy);

    // Free allocated memory
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        free(images[i]);
        free(one_hot_labels[i]);
    }
    free(images);
    free(one_hot_labels);
    free(labels);
    cudaFree(l1p);
    cudaFree(l1b);
    cudaFree(l2p);
    cudaFree(l2b);

    return 0;
}
