cmake_minimum_required(VERSION 3.10)
project(MNIST-CPP-CUDA)

# Add the include directories
include_directories(lib)

# Add source files
set(SOURCE_FILES
    src/conv_layer.cpp
    src/dense.cpp
    src/maxPooling_flattening.cpp
    src/neuron.cpp
    src/training.cpp
)

# Add the executable
add_executable(main
    main.cpp
    ${SOURCE_FILES}
)
