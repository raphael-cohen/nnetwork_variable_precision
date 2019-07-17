#include "Layer.cpp"
// #include <vector>

using namespace std;
typedef Eigen::MatrixXd matrix_d;
typedef Eigen::MatrixXf matrix_f;
typedef Eigen::VectorXd vector_d;
typedef Eigen::VectorXf vector_f;

class Network
{
public:
  int input_size;
  int layer_size;
  int activation_func;

  vector<Layer> layers;



public:

  vector_d forward(vector_d);
  void backward(vector_d, vector_d);
  void update_parameters(double);

  // TODO Change function eigen ... with an array of func
  Network(unsigned int inp_size, vector<int> lay_size, vector<function<vector_d (vector_d, bool)>> arr_func)
  {
    int last_size = inp_size;
    // size of the layers size list
    int number_layers = static_cast<int>(lay_size.size());
    // layers.resize(number_layers);

    for(int i = 0; i < number_layers; i++)
    {
      Layer l(last_size, lay_size[i], arr_func[i]);
      layers.push_back(l);
      // layers[i] = Layer(last_size, lay_size[i]); // Add act_func as a parameter
      last_size = lay_size[i];
    }
    //layers.push_back(Layer(last_size, lay_size[number_layers-1], identity));


  }
};

vector_d Network::forward(vector_d inputs)
{

    vector_d last_input = inputs;

    for(int i = 0; i<layers.size(); i++)
    {
      last_input = layers[i].forward(last_input);

    }
    // Last input = the output of the network at this point
    return last_input;
}

void Network::backward(vector_d inputs, vector_d output_error)
{
  //current error to update

  vector_d crt_error = output_error;

  for(int i=layers.size()-1; i>0; i--)
  {

    //avoid copying layers, instead, use the indexes
    //Layer crt_layer = layers[i];
    //Layer prev_layer = layers[i - 1];
    //crt_error = crt_layer.backward(prev_layer.outputs, crt_error);
    crt_error = layers[i].backward(layers[i-1].outputs, crt_error);

  }

  // Last backward outside of the loop --> we use inputs instead of outputs from previous layers !
  layers[0].backward(inputs, crt_error);


}


  // Update the weights of the layer according to the learning rate and
  // the gradient of the calculated error
  //Change float ? TODO
void Network::update_parameters(double learning_rate)
{
  for(int i=0; i<layers.size(); i++)
  {


    //cout << "\nweights\n" << layers[i].weights << '\n';
    matrix_d g_weights = layers[i].g_weights*learning_rate;
    //cout << "\ngradient weights\n" << g_weights << '\n';
    layers[i].weights = layers[i].weights - g_weights;

    //cout << "\nweights after\n" << layers[i].weights << '\n';

  }
  //return layers;
}
