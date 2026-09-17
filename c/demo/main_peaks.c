/* This demo shows the fitting of the peaks function. You can use the
   following MATLAB/Octave code to show the results.

   fid = fopen("xt.bin", "rb"); xt = fread(fid, "single"); fclose(fid);
   fid = fopen("yt.bin", "rb"); yt = fread(fid, "single"); fclose(fid);
   fid = fopen("zt.bin", "rb"); zt = fread(fid, "single"); fclose(fid);
   fid = fopen("zh.bin", "rb"); zh = fread(fid, "single"); fclose(fid);
   n = sqrt(length(xt));
   xt = reshape(xt, n, n);
   yt = reshape(yt, n, n);
   zt = reshape(zt, n, n);
   zh = reshape(zh, n, n);
   figure
   surf(xt,yt,zt,'EdgeColor','red','FaceColor','none')
   grid on
   hold on
   surf(xt,yt,zh)
   legend('True','Fitting','Location','northeast')
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mlpnet.h"

static float peaks(float x, float y)
{
	return 3 * powf(1 - x, 2) * expf(-x * x - powf(y + 1, 2))
		- 10 * (x / 5 - powf(x, 3) - powf(x, 5)) * expf(-x * x - y * y)
		- expf(-powf(x + 1, 2) - y * y) / 3;
}

static void export(float* data, int len, char* filename)
{
	FILE* data_file = fopen(filename, "wb");

	fwrite(data, sizeof(float), len, data_file);
	fclose(data_file);
}

#define N 50
//#define USE_RELU

#ifdef USE_RELU
static float f_relu(float x)
{
	return x > 0 ? x : 0.0f;
}

static float df_relu(float x)
{
	return x > 0 ? 1.0f : 0.0f;
}
#endif

int main()
{
	int i, j, idx, iter;
	int net_size[] = { 2, 10, 10, 1 };
	mlpnet net;
	float alpha = 0.999f;
	float loss, input[2] = { 0 }, * output;
	float x_min, y_min, eta, z_, z, J[2];
	float* xt = malloc(N * N * sizeof(float));
	float* yt = malloc(N * N * sizeof(float));
	float* zt = malloc(N * N * sizeof(float));
	float* zh = malloc(N * N * sizeof(float));

	/* Generate the dataset */
	for (j = 0; j < N; j++) {
		for (i = 0; i < N; i++) {
			xt[i + N * j] = 6.0f * (float)j / (N - 1) - 3.0f;
			yt[i + N * j] = 6.0f * (float)i / (N - 1) - 3.0f;
			zt[i + N * j] = peaks(xt[i + N * j], yt[i + N * j]);
		}
	}

	/* Intialize the network */
	//srand(0);
	mlpnet_init(&net, net_size, sizeof(net_size) / sizeof(net_size[0]) - 2);
#ifdef USE_RELU
	net.f = f_relu;
	net.df = df_relu;
	net.eta = 1e-4f;
#endif

	/* Train the network */
	printf("Network training\n\n");
	iter = 0;
	loss = 1.0f;
	while (loss > 1e-2f && iter < 10000000) {
		idx = (int)((float)(N * N - 1) * (float)rand() / (float)RAND_MAX);
		input[0] = xt[idx] / 3.0f;
		input[1] = yt[idx] / 3.0f;
		output = &zt[idx];
		loss = alpha * loss + (1 - alpha) * mlpnet_update(&net, input, output);
		if (!(++iter % 10000)) {
			printf("Loss = %9.4e\n", loss);
		}
	}

	/* Evaluate the network */
	for (j = 0; j < N; j++) {
		for (i = 0; i < N; i++) {
			input[0] = xt[i + N * j] / 3.0f;
			input[1] = yt[i + N * j] / 3.0f;
			output = mlpnet_eval(&net, input);
			zh[i + N * j] = *output;
		}
	}

	/* Export data */
	export(xt, N * N, "../xt.bin");
	export(yt, N * N, "../yt.bin");
	export(zt, N * N, "../zt.bin");
	export(zh, N * N, "../zh.bin");

	/* Find the minimum of the peaks function */
	x_min = -0.7f; // starting point
	y_min = -0.9f; // starting point
	eta = 1e-5f;
	z_ = 1.0f;
	z = 0;
	iter = 0;
	printf("\nFunction minimization\n");
	printf("\n\tx\ty\tz");
	while (fabsf(z - z_) > 1e-8f && iter < 10000) {
		z_ = z;
		input[0] = x_min / 3.0f;
		input[1] = y_min / 3.0f;
		output = mlpnet_eval(&net, input);
		z = *output;
		if (!(++iter % 100)) {
			printf("\n%9.4g %9.4g %9.4g", x_min, y_min, z);
		}
		mlpnet_jacobian(&net, input, J);
		x_min -= eta * J[0];
		y_min -= eta * J[1];
	}
	printf("\n");

	/* Free the memory */
	mlpnet_free(&net);
	free(xt);
	free(yt);
	free(zt);
	free(zh);
	return 0;
}