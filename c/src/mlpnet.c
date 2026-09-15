#include <math.h>
#include <stdlib.h>
#include "mlpnet.h"

static float f_tanh(float x)
{
	return tanhf(x);
}

static float df_tanh(float x)
{
	float y = tanhf(x);

	return 1.0f - y * y;
}

static float randn(void)
{
	float u1 = (float)rand() / (float)RAND_MAX;
	float u2 = (float)rand() / (float)RAND_MAX;

	return sqrtf(-2.0f * logf(u1)) * cosf(6.2831855f * u2);
}

int mlpnet_init(mlpnet* net, int nh, int* size)
{
	int i, k;

	net->nh = nh;
	net->eta = 1e-3f;
	net->f = f_tanh;
	net->df = df_tanh;
	net->size = size;
	net->J = calloc((size_t)size[0] * (size_t)size[nh + 1], sizeof(float));
	net->X = calloc(((size_t)nh + 1), sizeof(float*));
	net->Y = calloc(((size_t)nh + 1), sizeof(float*));
	net->W = calloc(((size_t)nh + 1), sizeof(float*));
	net->B = calloc(((size_t)nh + 1), sizeof(float*));
	net->dW = calloc(((size_t)nh + 1), sizeof(float*));
	net->dB = calloc(((size_t)nh + 1), sizeof(float*));
	if (net->X == 0 || net->W == 0 || net->dW == 0 ||
		net->Y == 0 || net->B == 0 || net->dB == 0 || net->J == 0) {
		return -1;
	}
	for (k = 0; k <= nh; k++) {
		net->X[k] = calloc((size_t)size[k], sizeof(float));
		net->Y[k] = calloc((size_t)size[k + 1], sizeof(float));
		net->W[k] = calloc((size_t)size[k] * (size_t)size[k + 1], sizeof(float));
		net->B[k] = calloc((size_t)size[k + 1], sizeof(float));
		net->dW[k] = calloc((size_t)size[k] * (size_t)size[k + 1], sizeof(float));
		net->dB[k] = calloc((size_t)size[k + 1], sizeof(float));
		if (net->X[k] == 0 || net->W[k] == 0 || net->dW[k] == 0 ||
			net->Y[k] == 0 || net->B[k] == 0 || net->dB[k] == 0) {
			return -1;
		}
		for (i = 0; i < size[k] * size[k + 1]; i++) {
			net->W[k][i] = randn() / size[k];
		}
	}
	return 0;
}

void mlpnet_free(mlpnet* net)
{
	int k;

	for (k = 0; k <= net->nh; k++) {
		free(net->X[k]);
		free(net->Y[k]);
		free(net->W[k]);
		free(net->B[k]);
		free(net->dW[k]);
		free(net->dB[k]);
	}
	free(net->J);
	free(net->X);
	free(net->Y);
	free(net->W);
	free(net->B);
	free(net->dW);
	free(net->dB);
	net->J = 0;
	net->X = 0;
	net->Y = 0;
	net->W = 0;
	net->B = 0;
	net->dW = 0;
	net->dB = 0;
}

float* mlpnet_eval(mlpnet* net, float* x)
{
	int i, j, k;

	for (i = 0; i < net->size[0]; i++) {
		net->X[0][i] = x[i];
	}
	for (k = 0; k <= net->nh; k++) {
		for (j = 0; j < net->size[k + 1]; j++) {
			net->Y[k][j] = net->B[k][j];
		}
		for (i = 0; i < net->size[k]; i++) {
			for (j = 0; j < net->size[k + 1]; j++) {
				net->Y[k][j] += net->X[k][i] * net->W[k][i * net->size[k + 1] + j];
			}
		}
		if (k < net->nh) {
			for (i = 0; i < net->size[k + 1]; i++) {
				net->X[k + 1][i] = net->f(net->Y[k][i]);
			}
		}
	}
	return net->Y[net->nh];
}

float mlpnet_update(mlpnet* net, float* x, float* y)
{
	int i, j, k;
	float loss = 0, * yh = mlpnet_eval(net, x);

	for (i = 0; i < net->size[net->nh + 1]; i++) {
		net->dB[net->nh][i] = yh[i] - y[i];
		loss += net->dB[net->nh][i] * net->dB[net->nh][i];
	}
	loss *= 0.5f;
	for (k = net->nh; k >= 0; k--) {
		for (i = 0; i < net->size[k]; i++) {
			for (j = 0; j < net->size[k + 1]; j++) {
				net->dW[k][i * net->size[k + 1] + j] = net->dB[k][j] * net->X[k][i];
			}
		}
		if (k > 0) {
			for (i = 0; i < net->size[k]; i++) {
				for (net->dB[k - 1][i] = 0, j = 0; j < net->size[k + 1]; j++) {
					net->dB[k - 1][i] += net->dB[k][j] * net->W[k][i * net->size[k + 1] + j];
				}
				net->dB[k - 1][i] *= net->df(net->Y[k - 1][i]);
			}
		}
	}
	for (k = 0; k <= net->nh; k++) {
		for (i = 0; i < net->size[k] * net->size[k + 1]; i++) {
			net->W[k][i] -= net->eta * net->dW[k][i];
		}
		for (i = 0; i < net->size[k + 1]; i++) {
			net->B[k][i] -= net->eta * net->dB[k][i];
		}
	}
	return loss;
}

float* mlpnet_jaco(mlpnet* net, float* x)
{
	int i, j, k, o;

	mlpnet_eval(net, x);
	for (o = 0; o < net->size[net->nh + 1]; o++) {
		for (j = 0; j < net->size[net->nh + 1]; j++) {
			net->dB[net->nh][j] = (float)(j == o);
		}
		for (k = net->nh; k > 0; k--) {
			for (i = 0; i < net->size[k]; i++) {
				for (net->dB[k - 1][i] = 0, j = 0; j < net->size[k + 1]; j++) {
					net->dB[k - 1][i] += net->dB[k][j] * net->W[k][i * net->size[k + 1] + j];
				}
				net->dB[k - 1][i] *= net->df(net->Y[k - 1][i]);
			}
		}
		for (i = 0; i < net->size[0]; i++) {
			for (net->J[o * net->size[0] + i] = 0, j = 0; j < net->size[1]; j++) {
				net->J[o * net->size[0] + i] += net->dB[0][j] * net->W[0][i * net->size[1] + j];
			}
		}
	}
	return net->J;
}