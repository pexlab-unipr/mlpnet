#ifndef MLPNET_H
#define MLPNET_H

typedef struct {
	int* size; // layer sizes, including input and output layers
	int nh; // number of hidden layers
	float eta; // learning rate
	float (*f)(float); // activation function
	float (*df)(float); // derivative of the activation function
	float** X; // vectors of inputs of all layers
	float** Y; // vectors of outputs of all layers
	float** W; // matrices of weights arranged in row-major order
	float** B; // vectors of biases
	float* work; // workspace memory
} mlpnet; // Multilayer perceptron (MLP) neural network

// Initialize an MLP network with nh hidden layers and 
// size = { number of inputs, 
//          number of neurons in hidden layer 1, 
//          number of neurons in hidden layer 2, 
//          ..., 
//          number of neurons in hidden layer nh, 
//          number of outputs }.
// Return 0 on success, -1 on allocation failure.
int mlpnet_init(mlpnet* net, int* size, int nh);
// Free the allocated memory.
void mlpnet_free(mlpnet* net);
// Evaluate the network at x. Return a pointer to
// the vector of network outputs.
float* mlpnet_eval(mlpnet* net, const float* x);
// Compute the gradient of the loss function w.r.t. the 
// network parameters by backpropagation at (x, y), then 
// update the parameters by stochastic gradient descent.
// Return the loss value computed before the update.
float mlpnet_update(mlpnet* net, const float* x, const float* y);
// Compute the Jacobian matrix J at x. The elements of 
// J are arranged in row-major order.
void mlpnet_jacobian(mlpnet* net, const float* x, float* J);

#endif