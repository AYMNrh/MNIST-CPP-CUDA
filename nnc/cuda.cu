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
        res[idx] = -target[idx] * log(x[idx] + 0.00000000000000001); 
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

double evaluate_accuracy(double *l1p, double *l1b, double *l2p, double *l2b, int size_input, int size_l2, int size_output) {
    // CUDA Kernel Variables
    double *input_data_gpu, *output_gpu;
    cudaMalloc(&input_data_gpu, MNIST_IMAGE_SIZE * sizeof(double));
    cudaMalloc(&output_gpu, MNIST_NUM_LABELS * sizeof(double));

    // Copy layer weights and biases to GPU
    double *l1p_gpu, *l1b_gpu, *l2p_gpu, *l2b_gpu;
    cudaMalloc(&l1p_gpu, size_l2 * size_input * sizeof(double));
    cudaMalloc(&l1b_gpu, size_l2 * sizeof(double));
    cudaMalloc(&l2p_gpu, size_output * size_l2 * sizeof(double));
    cudaMalloc(&l2b_gpu, size_output * sizeof(double));
    cudaMemcpy(l1p_gpu, l1p, size_l2 * size_input * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(l1b_gpu, l1b, size_l2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(l2p_gpu, l2p, size_output * size_l2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(l2b_gpu, l2b, size_output * sizeof(double), cudaMemcpyHostToDevice);

    // Load test images and labels
    // Assuming test_images, test_labels, one_hot_labels are loaded in a similar manner as in the training phase

    // Loop over test images and calculate accuracy
    int correct_predictions = 0;
    for (int i = 0; i < MNIST_NUM_IMAGES_TEST; i++) {
        // Copy test image to GPU
        cudaMemcpy(input_data_gpu, test_images[i], MNIST_IMAGE_SIZE * sizeof(double), cudaMemcpyHostToDevice);

        // Forward pass
        matrix_vector_add<<<(size_l2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(l1p_gpu, input_data_gpu, output_gpu, size_l2, size_input);
        apply_relu<<<(size_l2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(output_gpu, size_l2);
        matrix_vector_add<<<(size_output + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(l2p_gpu, output_gpu, output_gpu, size_output, size_l2);
        softmax<<<1, size_output>>>(output_gpu, size_output);

        // Copy output back to CPU
        double *output_cpu = (double *)malloc(size_output * sizeof(double));
        cudaMemcpy(output_cpu, output_gpu, size_output * sizeof(double), cudaMemcpyDeviceToHost);

        // Calculate predicted label
        int predicted_label = 0;
        double max_output = output_cpu[0];
        for (int j = 1; j < size_output; j++) {
            if (output_cpu[j] > max_output) {
                max_output = output_cpu[j];
                predicted_label = j;
            }
        }

        // Check against true label
        int true_label = 0; // Assuming the true label is known
        if (predicted_label == true_label) {
            correct_predictions++;
        }

        free(output_cpu);
    }

    // Free allocated memory on GPU
    cudaFree(input_data_gpu);
    cudaFree(output_gpu);
    cudaFree(l1p_gpu);
    cudaFree(l1b_gpu);
    cudaFree(l2p_gpu);
    cudaFree(l2b_gpu);

    // Calculate and return accuracy
    return (double)correct_predictions / MNIST_NUM_IMAGES_TEST;
}


void train_model(double *l1p, double *l1b, double *l2p, double *l2b, int size_input, int size_l2, int size_output,
                 double learning_rate, int epochs) {
    // Allocate device memory for training images, labels, and one-hot labels
    uint8_t **d_train_images;
    uint8_t **d_one_hot_labels;
    uint8_t *d_labels;

    cudaMalloc(&d_train_images, MNIST_NUM_IMAGES * sizeof(uint8_t *));
    cudaMalloc(&d_one_hot_labels, MNIST_NUM_IMAGES * sizeof(uint8_t *));
    cudaMalloc(&d_labels, MNIST_NUM_IMAGES * sizeof(uint8_t));

    // Copy data from host to device
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        uint8_t *d_image;
        cudaMalloc(&d_image, MNIST_IMAGE_SIZE * sizeof(uint8_t));
        cudaMemcpy(d_image, train_images[i], MNIST_IMAGE_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_train_images[i], &d_image, sizeof(uint8_t *), cudaMemcpyHostToDevice);

        uint8_t *d_one_hot_label;
        cudaMalloc(&d_one_hot_label, MNIST_NUM_LABELS * sizeof(uint8_t));
        cudaMemcpy(d_one_hot_label, one_hot_labels[i], MNIST_NUM_LABELS * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_one_hot_labels[i], &d_one_hot_label, sizeof(uint8_t *), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_labels, labels, MNIST_NUM_IMAGES * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Allocate device memory for intermediate results
    double *d_input_data, *d_output, *d_target, *d_l2_deltas, *d_l1_deltas;
    cudaMalloc(&d_input_data, MNIST_IMAGE_SIZE * sizeof(double));
    cudaMalloc(&d_output, size_output * sizeof(double));
    cudaMalloc(&d_target, size_output * sizeof(double));
    cudaMalloc(&d_l2_deltas, size_output * sizeof(double));
    cudaMalloc(&d_l1_deltas, size_l2 * sizeof(double));

    // Launch training epochs
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        printf("Epoch %d: ", epoch + 1);
        printProgressBar((double)(epoch + 1) / epochs);
        printf("\n");

        // Launch training iterations
        for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
            // Forward pass
            cudaMemcpy(d_input_data, train_images[i], MNIST_IMAGE_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            custom_MLP_2l_inf<<<1, 1>>>(d_input_data, MNIST_IMAGE_SIZE, size_l2, size_output, l1p, l1b, l2p, l2b);
            cudaDeviceSynchronize();

            // Calculate loss
            cudaMemcpy(d_target, d_one_hot_labels[i], size_output * sizeof(double), cudaMemcpyDeviceToDevice);
            categorical_cross_entropy_loss<<<1, 1>>>(d_output, d_target, size_output);
            cudaMemcpy(&total_loss, d_output, sizeof(double), cudaMemcpyDeviceToHost);

            // Backpropagation
            // Update layer 2 weights and biases
            for (int j = 0; j < size_output; j++) {
                // Update weights
                for (int k = 0; k < size_l2; k++) {
                    l2p[j * size_l2 + k] -= learning_rate * d_l2_deltas[j] * relu_derivative(l2p[j * size_l2 + k]);
                }
                // Update biases
                l2b[j] -= learning_rate * d_l2_deltas[j];
            }

            // Update layer 1 weights and biases
            for (int j = 0; j < size_l2; j++) {
                // Calculate layer 1 deltas
                double error = 0.0;
                for (int k = 0; k < size_output; k++) {
                    error += l2p[k * size_l2 + j] * d_l2_deltas[k];
                }
                d_l1_deltas[j] = error * relu_derivative(l1b[j]);

                // Update weights
                for (int k = 0; k < size_input; k++) {
                    l1p[j * size_input + k] -= learning_rate * d_l1_deltas[j] * d_input_data[k];
                }
                // Update biases
                l1b[j] -= learning_rate * d_l1_deltas[j];
            }
        }
        printf("Epoch %d - Loss: %f\n", epoch+1, total_loss / MNIST_NUM_IMAGES);
    }

    // Free device memory for training images, labels, and one-hot labels
    cudaFree(d_labels);
    for (int i = 0; i < MNIST_NUM_IMAGES; i++) {
        cudaFree(d_train_images[i]);
        cudaFree(d_one_hot_labels[i]);
    }
    cudaFree(d_train_images);
    cudaFree(d_one_hot_labels);

    // Free intermediate results
    cudaFree(d_input_data);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_l2_deltas);
    cudaFree(d_l1_deltas);
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
