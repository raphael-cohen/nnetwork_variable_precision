//
// Created by Milap Jhumkhawala on 10/22/18.
//

#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <math.h>
#include "mnist/mnist_reader.hpp"

class data_loader {

public:
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    data_loader(){
        std::cout << "MNIST data directory: " << "/Users/Milap/Desktop/IIT/SEM_3/Data Intensive Computing/udnn-team5/mnist_dataset" << std::endl;

        // Load MNIST data

        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/Users/Milap/Desktop/IIT/SEM_3/Data Intensive Computing/udnn-team5/mnist_dataset");

    }

    void test() {
            std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
            std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
            std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
            std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    }


};