#ifndef MLPNET_H
#define MLPNET_H

typedef struct {
	int nh; // number of hidden layers
	int* size; // layer sizes, including input and output layers
	float eta; // learning rate
	float (*f)(float); // activation function
	float (*df)(float); // derivative of the activation function
	float* J; // Jacobian matrix (arranged in row-major order)
	float** X; // vectors of inputs of all layers
	float** Y; // vectors of outputs of all layers
	float** W; // matrices of weights (arranged in row-major order)
	float** B; // vectors of biases
	float** work; // workspace memory
} mlpnet; // Multilayer perceptron (MLP) neural network

// Initialize an MLP network with nh hidden layers and 
// size = { number of inputs, 
//          number of neurons in hidden layer 1, 
//          number of neurons in hidden layer 2, 
//          ..., 
//          number of neurons in hidden layer nh, 
//          number of outputs }.
// Return 0 on success, -1 on allocation failure.
int mlpnet_init(mlpnet* net, int nh, int* size);
// Free the allocated memory.
void mlpnet_free(mlpnet* net);
// Evaluate the network at x. Return a pointer to
// the vector of network outputs.
float* mlpnet_eval(mlpnet* net, float* x);
// Compute the gradient of the loss function w.r.t. the 
// network parameters by backpropagation at (x, y), then 
// update the parameters by stochastic gradient descent.
// Return the loss value computed before the update.
float mlpnet_update(mlpnet* net, float* x, float* y);
// Compute the Jacobian matrix at x. Return a pointer
// to the Jacobian matrix. 
float* mlpnet_jaco(mlpnet* net, float* x);

#endif