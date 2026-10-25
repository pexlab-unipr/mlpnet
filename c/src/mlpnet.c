#include "mlpnet.h"
#include "detectum.h"
#include <stdlib.h>
#include <math.h>

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

int mlpnet_init(mlpnet* net, size_t* sz, size_t nh)
{
	size_t i, k, ny, np, dy, dp, work_len = sz[nh + 1];
	float* y, * p;

	net->sz = sz;
	net->nh = nh;
	net->eta = 1e-3f;
	net->f = f_tanh;
	net->df = df_tanh;
	net->itermax = 1000;
	net->tol = 1e-2f;
	net->rho = 1e-3f;
	net->mu = 2.0f;
	np = 0;
	ny = 0;
	for (k = 0; k <= nh; k++) {
		ny += sz[k + 1];
		np += sz[k + 1] * (1 + sz[k]);
	}
	net->np = np;
	y = calloc(ny, sizeof(float));
	if (y == 0) {
		return -1;
	}
	p = calloc(np, sizeof(float));
	if (p == 0) {
		return -1;
	}
	net->Y = calloc(nh + 1, sizeof(float*));
	if (net->Y == 0) {
		return -1;
	}
	net->B = calloc(nh + 1, sizeof(float*));
	if (net->B == 0) {
		return -1;
	}
	net->W = calloc(nh + 1, sizeof(float*));
	if (net->W == 0) {
		return -1;
	}
	dy = 0;
	dp = 0;
	for (k = 0; k <= nh; k++) {
		net->Y[k] = y + dy;
		dy += sz[k + 1];
		net->B[k] = p + dp;
		dp += sz[k + 1];
		net->W[k] = p + dp;
		dp += sz[k] * sz[k + 1];
		for (i = 0; i < sz[k] * sz[k + 1]; i++) {
			net->W[k][i] = randn() / sz[k];
		}
		if (k > 0) {
			if (sz[k] + sz[k + 1] > work_len) {
				work_len = sz[k] + sz[k + 1];
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
	free(net->Y[0]);
	free(net->B[0]);
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
		s0 = net->sz[k];
		s1 = net->sz[k + 1];
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
	float fy, loss, t, * Y_, * B, * W, * tmp = net->work, * yh;
	const float eta = net->eta;

	yh = mlpnet_eval(net, x);
	loss = 0;
	for (i = 0; i < net->sz[net->nh + 1]; i++) {
		tmp[i] = yh[i] - y[i];
		loss += tmp[i] * tmp[i];
	}
	for (k = net->nh + 1; k-- > 0;) {
		s0 = net->sz[k];
		s1 = net->sz[k + 1];
		Y_ = net->Y[k - 1];
		B = net->B[k];
		if (k > 0) {
			W = net->W[k];
			for (i = 0; i < s0; i++) {
				t = 0;
				for (j = 0; j < s1; j++, W++) {
					t += tmp[j] * *W;
				}
				tmp[i + s1] = t * net->df(Y_[i]);
			}
		}
		W = net->W[k];
		for (i = 0; i < s0; i++) {
			fy = k ? net->f(Y_[i]) : x[i];
			for (j = 0; j < s1; j++, W++) {
				*W -= eta * tmp[j] * fy;
			}
		}
		for (i = 0; i < s1; i++) {
			B[i] -= eta * tmp[i];
		}
		if (k > 0) {
			for (i = 0; i < s0; i++) {
				tmp[i] = tmp[i + s1];
			}
		}
	}
	return 0.5f * loss;
}

void mlpnet_jacobian(mlpnet* net, const float* x, float* J)
{
	size_t i, j, k, out, s0, s1;
	float t, * Y_, * W, * tmp = net->work;

	if (net->nh > 0) {
		mlpnet_eval(net, x);
		for (out = 0; out < net->sz[net->nh + 1]; out++) {
			s0 = net->sz[net->nh];
			s1 = net->sz[net->nh + 1];
			Y_ = net->Y[net->nh - 1];
			W = net->W[net->nh];
			for (i = 0; i < s0; i++) {
				tmp[i] = W[i * s1 + out] * net->df(Y_[i]);
			}
			for (k = net->nh - 1; k > 0; k--) {
				s0 = net->sz[k];
				s1 = net->sz[k + 1];
				Y_ = net->Y[k - 1];
				W = net->W[k];
				for (i = 0; i < s0; i++) {
					t = 0;
					for (j = 0; j < s1; j++, W++) {
						t += tmp[j] * *W;
					}
					tmp[i + s1] = t * net->df(Y_[i]);
				}
				for (i = 0; i < s0; i++) {
					tmp[i] = tmp[i + s1];
				}
			}
			s0 = net->sz[0];
			s1 = net->sz[1];
			W = net->W[k];
			for (i = 0; i < s0; i++, J++) {
				t = 0;
				for (j = 0; j < s1; j++, W++) {
					t += tmp[j] * *W;
				}
				*J = t;
			}
		}
	}
	else {
		s0 = net->sz[0];
		s1 = net->sz[1];
		W = net->W[0];
		for (out = 0; out < s1; out++) {
			for (i = 0; i < s0; i++, J++) {
				*J = W[i * s1 + out];
			}
		}
	}
}

int mlpnet_train(mlpnet* net, size_t n, const float* X, const float* Y, float* work)
{
	size_t i, j, k, s0, s1, obs, out;
	const size_t np = net->np;
	const size_t m = net->sz[net->nh + 1];
	const size_t mn = m * n;
	float r_norm = INFINITY, r_norm_, g_norm, J_norm;
	float fy, t, rho = net->rho * net->mu;
	const float* x, * y;
	float* yh, * Y_, * W, * J, * p, * tmp = net->work;
	float* g_data, * r_data, * r_data_, * Jt_data, * A_data, * b_data;
	int iter = 0;
	Matrixf g, r, Jt, A, b;

	if (work) {
		g_data = work;
		r_data = g_data + np;
		r_data_ = r_data + mn;
		Jt_data = r_data_ + mn;
		A_data = Jt_data + mn * np;
		b_data = A_data + np * np;
	}
	else {
		g_data = calloc(np, sizeof(float));
		if (g_data == 0) {
			return -1;
		}
		r_data = calloc(mn, sizeof(float));
		if (r_data == 0) {
			return -1;
		}
		r_data_ = calloc(mn, sizeof(float));
		if (r_data_ == 0) {
			return -1;
		}
		Jt_data = calloc(mn * np, sizeof(float));
		if (Jt_data == 0) {
			return -1;
		}
		A_data = calloc(np * np, sizeof(float));
		if (A_data == 0) {
			return -1;
		}
		b_data = calloc(np, sizeof(float));
		if (b_data == 0) {
			return -1;
		}
	}
	p = net->B[0];
	matrixf_init(&g, np, 1, g_data, 0);
	matrixf_init(&r, mn, 1, r_data, 0);
	matrixf_init(&Jt, np, mn, Jt_data, 0);
	matrixf_init(&A, np, np, A_data, 0);
	matrixf_init(&b, np, 1, b_data, 0);
	while (iter < net->itermax) {
		r_norm_ = r_norm;
		for (i = 0, obs = 0; obs < n; obs++) {
			x = X + net->sz[0] * obs;
			y = Y + m * obs;
			yh = mlpnet_eval(net, x);
			for (out = 0; out < m; out++, i++) {
				r_data[i] = y[out] - yh[out];
			}
		}
		r_norm = normf(r_data, mn, 1);
		if (net->mu > 1.0f && r_norm > r_norm_) { // reject the update
			rho *= net->mu * net->mu;
			for (i = 0; i < np; i++) {
				p[i] -= b_data[i];
			}
			for (i = 0; i < mn; i++) {
				r_data[i] = r_data_[i];
			}
			r_norm = r_norm_;
		}
		else { // accept the update and compute the new Jacobian matrix
			rho /= net->mu;
			for (i = 0; i < mn; i++) {
				r_data_[i] = r_data[i];
			}
			J = Jt_data + mn * np - 1;
			for (obs = n; obs-- > 0;) {
				x = X + net->sz[0] * obs;
				mlpnet_eval(net, x);
				for (out = m; out-- > 0;) {
					for (i = 0; i < m; i++) {
						tmp[i] = 0;
					}
					tmp[out] = 1.0f;
					for (k = net->nh + 1; k-- > 0;) {
						s0 = net->sz[k];
						s1 = net->sz[k + 1];
						Y_ = net->Y[k - 1];
						if (k > 0) {
							W = net->W[k];
							for (i = 0; i < s0; i++) {
								t = 0;
								for (j = 0; j < s1; j++, W++) {
									t += tmp[j] * *W;
								}
								tmp[i + s1] = t * net->df(Y_[i]);
							}
						}
						for (i = s0; i-- > 0;) {
							fy = k ? net->f(Y_[i]) : x[i];
							for (j = s1; j-- > 0; J--) {
								*J = tmp[j] * fy;
							}
						}
						for (i = s1; i-- > 0; J--) {
							*J = tmp[i];
						}
						if (k > 0) {
							for (i = 0; i < s0; i++) {
								tmp[i] = tmp[i + s1];
							}
						}
					}
				}
			}
			matrixf_multiply(&Jt, &r, &g, 1.0f, 0.0f, 0, 0);
			g_norm = normf(g_data, np, 1);
			J_norm = normf(Jt_data, mn * np, 1);
			if (g_norm < net->tol * (1.0f + J_norm)) {
				break;
			}
		}
		matrixf_multiply(&Jt, &Jt, &A, 1.0f, 0.0f, 0, 1);
		for (i = 0; i < np; i++) {
			at(&A, i, i) += rho;
			b_data[i] = g_data[i];
		}
		matrixf_solve_chol(&A, &b);
		for (i = 0; i < np; i++) {
			p[i] += b_data[i];
		}
		iter++;
	}
	if (work == 0) {
		free(g_data);
		free(r_data);
		free(r_data_);
		free(Jt_data);
		free(A_data);
		free(b_data);
	}
	return iter;
}