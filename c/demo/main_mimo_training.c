// This demo is designed to work together with
// the MATLAB script 'demo_peaks_mimo.m'.
#define MLPNET_USE_VARIADIC
#include "mlpnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline int tick(struct timespec* t0) {
#ifdef _WIN32
	if (timespec_get(t0, TIME_UTC) == TIME_UTC) return 1;
#else
	if (clock_gettime(CLOCK_REALTIME, t0) == 0) return 1;
#endif
	return 0;
}
static inline double tock(struct timespec* t0) {
	struct timespec tf = { 0 }; tick(&tf);
	return ((tf.tv_sec - t0->tv_sec) + ((tf.tv_nsec - t0->tv_nsec) * 1e-9));
}

#define n 1000

int main()
{
	FILE* xt_file = fopen("../xt.bin", "rb");
	FILE* yt_file = fopen("../yt.bin", "rb");
	FILE* zt_file = fopen("../zt.bin", "rb");
	FILE* par_file = fopen("../par.bin", "wb");
	float* X = malloc(2 * n * sizeof(float));
	float* Y = malloc(3 * n * sizeof(float));
	float xti, yti, zti;
	int i, j, k, iter;
	struct timespec tim;

	for (i = 0, j = 0, k = 0; k < n; k++) {
		fread(&xti, sizeof(float), 1, xt_file);
		fread(&yti, sizeof(float), 1, yt_file);
		fread(&zti, sizeof(float), 1, zt_file);
		X[i++] = xti;
		X[i++] = yti;
		Y[j++] = +zti;
		Y[j++] = -zti;
		Y[j++] = +zti * 0.1f;
	}
	fclose(xt_file);
	fclose(yt_file);
	fclose(zt_file);
	mlpnet(net, 2, 10, 10, 10, 3);
	tick(&tim);
	iter = mlpnet_train(&net, n, X, Y, 0);
	printf("Elapsed time: %f s\n", tock(&tim));
	printf("# iterations: %d\n", iter);
	fwrite(net.B[0], sizeof(float), net.np, par_file);
	fclose(par_file);
	free(X);
	free(Y);
	return 0;
}