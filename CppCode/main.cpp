#include <iostream>
#include "train.cpp"
#include "train.h"
#include "data_loader.cpp"
#include <eigen3/Eigen/Dense>
#include <typeinfo>

using namespace Eigen;
typedef Eigen::MatrixXd matrix_d;
typedef Eigen::MatrixXf matrix_f;
typedef Eigen::VectorXd vector_d;
typedef Eigen::VectorXf vector_f;



Eigen::MatrixXd iris_data(Eigen::MatrixXd*, Eigen::VectorXd*);

int main() {


    vector<int> lay_size{4,4,2};
    matrix_d data(4,2);
    data << 0,0,
            0,1,
            1,0,
            1,1;

    cout<< typeid(data).name();
    cout<<"\n";
    vector_d labels(4);
    labels << 0,0,0,1;

    cout<< typeid(labels).name();

    int batch_size = 0;
    int num_epochs = 20;
    double learning_rate = 0.1;
    data_loader dataLoad;
    vector<function<vector_d (vector_d, bool)>> arr_func = {identity,identity,identity,identity, logistic};
    Network nn(data.cols(), lay_size, arr_func);
    train(&nn, data, labels, batch_size, num_epochs, learning_rate);
    //iris_data(&data, &labels);

    dataLoad.test();

    return 0;
}

//Eigen::MatrixXd iris_data(Eigen::MatrixXd *data, Eigen::VectorXd *labels)
//{
//    Eigen::MatrixXd iris(150,5);
//    ifstream file ( "iris.data.csv" ); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
//    string value;
//    vector<string> lines;
//    while ( file.good() )
//    {
//        getline ( file, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
//        lines.push_back(value);
//
//        getline ( file, value, ',' );
//        lines.push_back(value);
//
//        getline ( file, value, ',' );
//        lines.push_back(value);
//
//        getline ( file, value, ',' );
//        lines.push_back(value);
//
//        getline ( file, value, '\n' );
//        lines.push_back(value);
//
//        //cout << string( value, 1, value.length()-2 ); // display value removing the first and the last character from it
//    }
//    for(int i = 0; i < lines.size(); i++){
//        if(i%4 == 0)
//        {
//
//        }
//    }
//    cout << "line:\n" << stod(lines[5])<<'\n';
//    return iris;
//}