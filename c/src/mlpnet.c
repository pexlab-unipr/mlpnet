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

	return sqrtf(-2.0f * logf(u1)) * cosf(6.2831855f * u2); // Box-Muller transform
}

int mlpnet_init(mlpnet* net, size_t* size, size_t nh)
{
	size_t i, k, work_len = size[nh + 1];

	net->nh = nh;
	net->eta = 1e-3f;
	net->f = f_tanh;
	net->df = df_tanh;
	net->size = size;
	net->Y = calloc(nh + 1, sizeof(float*));
	net->B = calloc(nh + 1, sizeof(float*));
	net->W = calloc(nh + 1, sizeof(float*));
	if (net->Y == 0 || net->B == 0 || net->W == 0) {
		return -1;
	}
	for (k = 0; k <= nh; k++) {
		net->Y[k] = calloc(size[k + 1], sizeof(float));
		net->B[k] = calloc(size[k + 1], sizeof(float));
		net->W[k] = calloc(size[k] * size[k + 1], sizeof(float));
		if (net->Y[k] == 0 || net->B[k] == 0 || net->W[k] == 0) {
			return -1;
		}
		for (i = 0; i < size[k] * size[k + 1]; i++) {
			net->W[k][i] = randn() / size[k];
		}
		if (k > 0) {
			if (size[k] + size[k + 1] > work_len) {
				work_len = size[k] + size[k + 1];
			}
		}
	}
	net->work = calloc(work_len, sizeof(float*));
	if (net->work == 0) {
		return -1;
	}
	return 0;
}

void mlpnet_free(mlpnet* net)
{
	size_t k;

	for (k = 0; k <= net->nh; k++) {
		free(net->Y[k]);
		free(net->B[k]);
		free(net->W[k]);
	}
	free(net->Y);
	free(net->B);
	free(net->W);
	free(net->work);
}

float* mlpnet_eval(mlpnet* net, const float* x)
{
	size_t i, j, k;

	for (k = 0; k <= net->nh; k++) {
		for (j = 0; j < net->size[k + 1]; j++) {
			net->Y[k][j] = net->B[k][j];
		}
		for (i = 0; i < net->size[k]; i++) {
			for (j = 0; j < net->size[k + 1]; j++) {
				net->Y[k][j] += (k ? net->f(net->Y[k - 1][i]) : x[i]) * net->W[k][i * net->size[k + 1] + j];
			}
		}
	}
	return net->Y[net->nh];
}

float mlpnet_update(mlpnet* net, const float* x, const float* y)
{
	size_t i, j, k;
	float loss = 0, * yh = mlpnet_eval(net, x);

	for (i = 0; i < net->size[net->nh + 1]; i++) {
		net->work[i] = yh[i] - y[i];
		loss += net->work[i] * net->work[i];
	}
	loss *= 0.5f;
	for (k = net->nh + 1; k-- > 0;) {
		if (k > 0) {
			for (i = 0; i < net->size[k]; i++) {
				net->work[i + net->size[k + 1]] = 0;
				for (j = 0; j < net->size[k + 1]; j++) {
					net->work[i + net->size[k + 1]] += net->work[j] * net->W[k][i * net->size[k + 1] + j];
				}
				net->work[i + net->size[k + 1]] *= net->df(net->Y[k - 1][i]);
			}
		}
		for (i = 0; i < net->size[k]; i++) {
			for (j = 0; j < net->size[k + 1]; j++) {
				net->W[k][i * net->size[k + 1] + j] -= net->eta * net->work[j] * (k ? net->f(net->Y[k - 1][i]) : x[i]);
			}
		}
		for (i = 0; i < net->size[k + 1]; i++) {
			net->B[k][i] -= net->eta * net->work[i];
		}
		if (k > 0) {
			for (i = 0; i < net->size[k]; i++) {
				net->work[i] = net->work[i + net->size[k + 1]];
			}
		}
	}
	return loss;
}

void mlpnet_jacobian(mlpnet* net, const float* x, float* J)
{
	size_t i, j, k, o;

	if (net->nh > 0) {
		mlpnet_eval(net, x);
		for (o = 0; o < net->size[net->nh + 1]; o++) {
			for (i = 0; i < net->size[net->nh]; i++) {
				net->work[i] = net->W[net->nh][i * net->size[net->nh + 1] + o] * net->df(net->Y[net->nh - 1][i]);
			}
			for (k = net->nh - 1; k > 0; k--) {
				for (i = 0; i < net->size[k]; i++) {
					net->work[i + net->size[k + 1]] = 0;
					for (j = 0; j < net->size[k + 1]; j++) {
						net->work[i + net->size[k + 1]] += net->work[j] * net->W[k][i * net->size[k + 1] + j];
					}
					net->work[i + net->size[k + 1]] *= net->df(net->Y[k - 1][i]);
				}
				for (i = 0; i < net->size[k]; i++) {
					net->work[i] = net->work[i + net->size[k + 1]];
				}
			}
			for (i = 0; i < net->size[0]; i++) {
				J[o * net->size[0] + i] = 0;
				for (j = 0; j < net->size[1]; j++) {
					J[o * net->size[0] + i] += net->work[j] * net->W[0][i * net->size[1] + j];
				}
			}
		}
	}
	else {
		for (o = 0; o < net->size[1]; o++) {
			for (i = 0; i < net->size[0]; i++) {
				J[o * net->size[0] + i] = net->W[0][i * net->size[1] + o];
			}
		}
	}
}