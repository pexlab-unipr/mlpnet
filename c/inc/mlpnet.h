#ifndef MLPNET_H
#define MLPNET_H

#include <stddef.h>

// Multilayer perceptron (MLP) neural network
typedef struct {
	size_t* size; // layer sizes, including input and output layers
	size_t nh; // number of hidden layers
	float eta; // learning rate
	float (*f)(float); // activation function
	float (*df)(float); // derivative of the activation function
	float** Y; // output vectors for each layer
	float** B; // vectors of biases
	float** W; // matrices of weights arranged in row-major order
	float* work; // workspace memory
} mlpnet;

#ifdef MLPNET_USE_VARIADIC
// Create and initialize an MLP network with the specified 
// layer sizes, including input and output layers. Example:
// mlpnet(net, ni, n1, n2, n3, no) instantiates an MLP net
// with ni inputs, no outputs and three hidden layers 
// containing n1, n2 and n3 neurons, respectively. 
#define mlpnet(net, ...) \
mlpnet net; \
size_t net##_size[] = {__VA_ARGS__}; \
mlpnet_init(&net, net##_size, sizeof(net##_size) / sizeof(size_t) - 2)
#endif

// Initialize an MLP network with nh hidden layers and 
// size = { number of inputs, 
//          number of neurons in hidden layer 1, 
//          number of neurons in hidden layer 2, 
//          ..., 
//          number of neurons in hidden layer nh, 
//          number of outputs }.
// Return 0 on success, -1 on allocation failure.
int mlpnet_init(mlpnet* net, size_t* size, size_t nh);

// Free the allocated memory.
void mlpnet_free(mlpnet* net);

// Evaluate the network at x and update Y. Return a 
// pointer to the vector of network outputs.
float* mlpnet_eval(mlpnet* net, const float* x);

// Compute the gradient of the loss function w.r.t. the 
// network parameters by backpropagation at (x, y), then 
// update W and B by stochastic gradient descent. Return
// the loss value computed before the update. The function
// also updates Y by evaluating the network at x.
float mlpnet_update(mlpnet* net, const float* x, const float* y);

// Compute the Jacobian matrix J at x and store its 
// elements in row-major order. The function also
// updates Y by evaluating the network at x.
void mlpnet_jacobian(mlpnet* net, const float* x, float* J);

#endif