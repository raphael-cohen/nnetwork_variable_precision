#include "Network.cpp"
#include "train.h"

typedef Eigen::MatrixXd matrix_d;
typedef Eigen::MatrixXf matrix_f;
typedef Eigen::VectorXd vector_d;
typedef Eigen::VectorXf vector_f;

// Maximum = batch size
double eval_nn(Network nn, vector_d labels, matrix_d data, int maximum)
{
    double correct_no = 0;
    double how_many = data.rows();

    for(int i = 0; i<how_many; i++)
    {

        vector_d out = nn.forward(data.row(i));
        vector_d::Index argmax;

        double y = out.maxCoeff(&argmax);

        double t = labels(i);

        //
        //cout << "\nmaxcoeff\n" << t << " vs " << argmax << '\n';

        if(t == double(argmax))
        {
            correct_no++;
        }

    }
    return correct_no/how_many;

}


// mettre data train & eval
void train(Network *nn, matrix_d data, vector_d labels, int batch, int epoch = 1, double learning_rate = 0.1)
{

    for(int j = 0; j<epoch; j++)
    {
        for(int i = 0; i<data.rows(); i++) {

            //cout << "\ninputs:\n" << data.row(i) << '\n';
            vector_d nn_output = nn->forward(data.row(i));

            vector_d expected_output(nn_output.size());
            expected_output.setZero();
            expected_output(labels(i)) = 1;
            vector_d error = nn_output - expected_output;

            nn->backward(data.row(i), error);

            nn->update_parameters(learning_rate);

            cout << "Accuracy: " << eval_nn(*nn, labels, data, 0)*100 << "%\n";
        }
    }

}
