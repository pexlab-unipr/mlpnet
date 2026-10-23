// This demo is designed to work together with
// the MATLAB script 'demo_peaks_mimo.m'.
#define MLPNET_USE_VARIADIC
#include "mlpnet.h"
#include <stdio.h>
#include <stdlib.h>

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
	iter = mlpnet_train(&net, n, X, Y, 0);
	printf("# iterations: %d\n", iter);
	fwrite(net.B[0], sizeof(float), net.np, par_file);
	fclose(par_file);
	free(X);
	free(Y);
	return 0;
}