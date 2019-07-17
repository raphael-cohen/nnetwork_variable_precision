//
// Created by Milap Jhumkhawala on 10/22/18.
//

#ifndef UDNN_TEAM5_TRAIN_H
#define UDNN_TEAM5_TRAIN_H

#endif //UDNN_TEAM5_TRAIN_H

void train(Network *nn, matrix_d data, vector_d labels, int batch, int epoch, double learning_rate);
double eval_nn(Network nn, vector_d labels, matrix_d data, int maximum);