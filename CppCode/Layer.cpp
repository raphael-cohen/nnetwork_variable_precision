#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <math.h>

typedef Eigen::MatrixXd matrix_d;
typedef Eigen::MatrixXf matrix_f;
typedef Eigen::VectorXd vector_d;
typedef Eigen::VectorXf vector_f;

using namespace std;
class Layer
{


  //Change public to private TODO
    // private:
  public:
      int num_input;
      int num_output;

      // vector< vector< double> > weights;

      matrix_d weights;
      vector_d biases;
      // vector<double> biases;

      vector_d outputs;
      matrix_d g_weights;
      vector_d g_outputs;
      vector_d g_biases;

      function <vector_d (vector_d, bool)> fn;

    // Member Functions()
    public:

    void init_wb(default_random_engine, normal_distribution<double>);
    vector_d forward(vector_d);
    vector_d backward(vector_d, vector_d);

    Layer(int n_in, int n_out, function<vector_d (vector_d, bool)> func)
    {
      num_input = n_in;
      num_output = n_out;
      fn = func;

      weights.resize(num_output, num_input);
      biases.resize(num_output);

      default_random_engine generator;
      double var = sqrt(2.0/(num_input+num_output));
      normal_distribution<double> distribution(0.0,var);
      init_wb(generator, distribution);

      outputs.setZero(num_output);
      g_weights.setZero(num_output, num_input);
      g_outputs.setZero(num_output);
      g_biases.setZero(num_output);

      //cout << "\ng_weights size " << g_weights.size()<<'\n';
      //cout << "\nweights size " << weights.size()<<'\n';

    }

};

void Layer::init_wb(default_random_engine g, normal_distribution<double> d)
{
  for(int i = 0; i< num_output; i++){
    // weights[i].resize(num_input);
    // biases[i] = d(g);
    biases(i) = d(g);
    for(int j = 0; j<num_input; j++){

      // weights[i][j] = d(g);
      weights(i,j) = d(g);
      //cout << weights(i,j) << " ";
    }
    //cout << '\n';
  }
}

vector_d Layer::forward(vector_d inputs){

  outputs = fn((weights*inputs) + biases, false);
  //cout << '\n' << "outputs:\n";
  //cout << outputs;
  return outputs;
}

vector_d Layer::backward(vector_d inputs, vector_d errors){


  this->g_outputs = fn(outputs, true);

  this->g_biases = errors.cwiseProduct(g_outputs);//errors * g_outputs;

  this->g_weights = (errors.cwiseProduct(g_outputs))*inputs.transpose();//.dot(inputs);//(errors.transpose() * g_outputs)*inputs.transpose();


  return weights.transpose()*g_biases;
}




vector_d identity(vector_d x, bool derivate = false)
{

    if(derivate)
    {
        x.setOnes();

    }
    return x;
}

vector_d logistic(vector_d x, bool derivate = false)
{
    if(derivate)
    {
        vector_d ones(x.size());
        ones.setOnes();
        return x.cwiseProduct(ones-x);
    }

    vector_d res(x.size());

    for(int i = 0; i < x.size(); i++)
    {
        res(i) = 1.0/(1.0+exp(-x(i)));
    }

    return res;
}

vector_d relu(vector_d x, bool derivate = false)
{
    vector_d res(x.size());

    if(derivate)
    {
        for(int i = 0; i < x.size(); i++)
        {
            if(x(i) < 0)
            {
                res(i) = 0;
            }
            else{
                res(i) = 1;
            }

        }

        return res;
    }

    for(int i = 0; i < x.size(); i++)
    {
        if(x(i) < 0)
        {
            res(i) = 0;
        }
        else{
            res(i) = x(i);
        }
    }

    return res;
}
