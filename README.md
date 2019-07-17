# nnetwork_variable_precision
Test neural net performances with different variable precision &amp; GPU (C++ and Python)

Read report for full details

Deep learning models are a complicated network of layers of
neurons. The more complex a problem gets, deeper the model
needs to be in order to perform better, this also means more
amount of computation is required.

 Example of a dense deep learning network
The current deep learning frameworks are more like a black
box, they do not really tell how they work. They only expose a
few parameters we can tweak for our implementation. That is
why we need a simpler and more customizable, transparent
program to work with. The purpose of the project is to
evaluate how does using variable precision(8-bit, 16-bit, 32-bit
and 64-bit) on different kind of neural network model affect
the quality, the performance and the speed the of model.

II.PROBLEM STATEMENT
Most of popular the deep neural network frameworks used
usually focus on optimizing the model itself and not the
underlying driver functions, like changing the precision or
modifying the runtime environment or instruction set
architecture of the hardware. Since these make up for the
foundation of any processor based computation, they are kind
of left unexplored in terms of how it might affect the neural
network performance. Thus a question that arises is “Are we
computing more than required”?
