/**
 * syr2k.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
// #define n 2048
// #define m 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



// void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
// {
// 	int i, j;
  
// 	for (i = 0; i < n; i++)
//     	{
//     		for (j = 0; j < n; j++)
// 		{
// 			C[i*n + j] = ((DATA_TYPE) i*j + 2) / n;
// 		}
      	
// 		for (j = 0; j < m; j++)
// 		{
// 	  		A[i*n + j] = ((DATA_TYPE) i*j) / n;
// 	  		B[i*n + j] = ((DATA_TYPE) i*j + 1) / n;
// 		}
//     	}
// }


// void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
// {
// 	int i, j, k;
		
//   	for (i = 0; i < n; i++)
// 	{
//    		for (j = 0; j < n; j++)
// 		{
//      			C[i*n + j] *= BETA;
// 		}
// 	}

//   	for (i = 0; i < n; i++)
// 	{
//    		for (j = 0; j < n; j++)
// 		{
//       			for (k = 0; k < m; k++)
// 			{
// 	  			C[i*n + j] += ALPHA * A[i*m + k] * B[j*m + k];
// 	 		 	C[i*n + j] += ALPHA * B[i*m + k] * A[j*m + k];
// 			}
// 		}
// 	}
// }


// void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu)
// {
// 	int i,j,fail;
// 	fail = 0;

// 	// Compare C with D
// 	for (i=0; i<n; i++)
// 	{
// 		for (j=0; j<n; j++)
// 		{
// 			if (percentDiff(C[i*n + j], C_outputFromGpu[i*n + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
// 			{ 
// 				fail++;
// 			}
// 		}
// 	}
	
// 	// print results
// 	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
// }


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void syr2k_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, int n, int m)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < n) && (j < n))
	{
		c[i * n + j] *= BETA;
		
		int k;
		for(k = 0; k < m; k++)
		{
			c[i * n + j] += ALPHA * a[i * m + k] * b[j * m + k] + ALPHA * b[i * m + k] * a[j * m + k];
		}
	}
}


void syr2kCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int n, int m) 
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * n * m);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * n * m);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * n * n);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * n * m, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)n) / ((float)DIM_THREAD_BLOCK_X) ), (size_t)(ceil( ((float)n) / ((float)DIM_THREAD_BLOCK_Y) )));
	
	t_start = rtclock();
	syr2k_kernel<<<grid,block>>>(A_gpu,B_gpu,C_gpu, n, m);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}


int main(int argc, char** argv)
{
	// double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

	int m, n;

	if(argc != 2){
		fprintf(stdout, "E.g.: exe size\n");
		return 1;
	}

	m = atoi(argv[1]);
	n = m;

	A = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));
	C_outputFromGpu = (DATA_TYPE*)malloc(n*m*sizeof(DATA_TYPE));

	// init_arrays(A, B, C);
    
	GPU_argv_init();
	syr2kCuda(A, B, C, C_outputFromGpu, n, m);
	
	// t_start = rtclock();
	// syr2k(A, B, C);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(C, C_outputFromGpu);

	free(A);
	free(B);
	free(C);
	free(C_outputFromGpu);

  	return 0;
}

