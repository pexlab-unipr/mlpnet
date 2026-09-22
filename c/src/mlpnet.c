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
	size_t i, j, k, s0, s1;
	float fy, * Y_, * Y, * B, * W;

	for (k = 0; k <= net->nh; k++) {
		s0 = net->size[k];
		s1 = net->size[k + 1];
		Y_ = net->Y[k - 1];
		Y = net->Y[k];
		B = net->B[k];
		W = net->W[k];
		for (j = 0; j < s1; j++) {
			Y[j] = B[j];
		}
		for (i = 0; i < s0; i++) {
			fy = k ? net->f(Y_[i]) : x[i];
			for (j = 0; j < s1; j++, W++) {
				Y[j] += fy * *W;
			}
		}
	}
	return net->Y[net->nh];
}

float mlpnet_update(mlpnet* net, const float* x, const float* y)
{
	size_t i, j, k, s0, s1;
	float fy, loss = 0, * Y_, * B, * W, * tmp0 = net->work, * tmp1, * yh;
	const float eta = net->eta;

	yh = mlpnet_eval(net, x);
	for (i = 0; i < net->size[net->nh + 1]; i++) {
		tmp0[i] = yh[i] - y[i];
		loss += tmp0[i] * tmp0[i];
	}
	for (k = net->nh + 1; k-- > 0;) {
		s0 = net->size[k];
		s1 = net->size[k + 1];
		Y_ = net->Y[k - 1];
		B = net->B[k];
		if (k > 0) {
			W = net->W[k];
			tmp1 = tmp0 + s1;
			for (i = 0; i < s0; i++, tmp1++) {
				*tmp1 = 0;
				for (j = 0; j < s1; j++, W++) {
					*tmp1 += tmp0[j] * *W;
				}
				*tmp1 *= net->df(Y_[i]);
			}
		}
		W = net->W[k];
		for (i = 0; i < s0; i++) {
			fy = k ? net->f(Y_[i]) : x[i];
			for (j = 0; j < s1; j++, W++) {
				*W -= eta * tmp0[j] * fy;
			}
		}
		for (i = 0; i < s1; i++) {
			B[i] -= eta * tmp0[i];
		}
		if (k > 0) {
			tmp1 = tmp0 + s1;
			for (i = 0; i < s0; i++) {
				tmp0[i] = tmp1[i];
			}
		}
	}
	return 0.5f * loss;
}

void mlpnet_jacobian(mlpnet* net, const float* x, float* J)
{
	size_t i, j, k, o, s0, s1;
	float* Y_, * W, * tmp0 = net->work, * tmp1, * pJ = J;

	if (net->nh > 0) {
		mlpnet_eval(net, x);
		for (o = 0; o < net->size[net->nh + 1]; o++) {
			s0 = net->size[net->nh];
			s1 = net->size[net->nh + 1];
			Y_ = net->Y[net->nh - 1];
			W = net->W[net->nh];
			for (i = 0; i < s0; i++) {
				tmp0[i] = W[i * s1 + o] * net->df(Y_[i]);
			}
			for (k = net->nh - 1; k > 0; k--) {
				s0 = net->size[k];
				s1 = net->size[k + 1];
				Y_ = net->Y[k - 1];
				W = net->W[k];
				tmp1 = tmp0 + s1;
				for (i = 0; i < s0; i++, tmp1++) {
					*tmp1 = 0;
					for (j = 0; j < s1; j++, W++) {
						*tmp1 += tmp0[j] * *W;
					}
					*tmp1 *= net->df(Y_[i]);
				}
				tmp1 = tmp0 + s1;
				for (i = 0; i < s0; i++) {
					tmp0[i] = tmp1[i];
				}
			}
			s0 = net->size[0];
			s1 = net->size[1];
			W = net->W[k];
			for (i = 0; i < s0; i++, pJ++) {
				*pJ = 0;
				for (j = 0; j < s1; j++, W++) {
					*pJ += tmp0[j] * *W;
				}
			}
		}
	}
	else {
		s0 = net->size[0];
		s1 = net->size[1];
		W = net->W[0];
		for (o = 0; o < s1; o++) {
			for (i = 0; i < s0; i++, pJ++) {
				*pJ = W[i * s1 + o];
			}
		}
	}
}