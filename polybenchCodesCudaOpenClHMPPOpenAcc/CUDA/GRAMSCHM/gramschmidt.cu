/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
// #define m 2048
// #define n 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



// void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
// {
// 	int i,j,k;
// 	DATA_TYPE nrm;
// 	for (k = 0; k < n; k++)
// 	{
// 		nrm = 0;
// 		for (i = 0; i < m; i++)
// 		{
// 			nrm += A[i*n + k] * A[i*n + k];
// 		}
		
// 		R[k*n + k] = sqrt(nrm);
// 		for (i = 0; i < m; i++)
// 		{
// 			Q[i*n + k] = A[i*n + k] / R[k*n + k];
// 		}
		
// 		for (j = k + 1; j < n; j++)
// 		{
// 			R[k*n + j] = 0;
// 			for (i = 0; i < m; i++)
// 			{
// 				R[k*n + j] += Q[i*n + k] * A[i*n + j];
// 			}
// 			for (i = 0; i < m; i++)
// 			{
// 				A[i*n + j] = A[i*n + j] - Q[i*n + k] * R[k*n + j];
// 			}
// 		}
// 	}
// }


// void init_array(DATA_TYPE* A)
// {
// 	int i, j;

// 	for (i = 0; i < m; i++)
// 	{
// 		for (j = 0; j < n; j++)
// 		{
// 			A[i*n + j] = ((DATA_TYPE) (i+1)*(j+1)) / (m+1);
// 		}
// 	}
// }


// void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
// {
// 	int i, j, fail;
// 	fail = 0;

// 	for (i=0; i < m; i++) 
// 	{
// 		for (j=0; j < n; j++) 
// 		{
// 			if (percentDiff(A[i*n + j], A_outputFromGpu[i*n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
// 			{				
// 				fail++;
// 				printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*n + j], A_outputFromGpu[i*n + j]);
// 			}
// 		}
// 	}
	
// 	// Print results
// 	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
// }


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );	
	return;
}


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k, int m, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < m; i++)
		{
			nrm += a[i * n + k] * a[i * n + k];
		}
      		r[k * n + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k, int m, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < m)
	{	
		q[i * n + k] = a[i * n + k] / r[k * n + k];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k, int m, int n)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < n))
	{
		r[k*n + j] = 0.0;

		int i;
		for (i = 0; i < m; i++)
		{
			r[k*n + j] += q[i*n + k] * a[i*n + j];
		}
		
		for (i = 0; i < m; i++)
		{
			a[i*n + j] -= q[i*n + k] * r[k*n + j];
		}
	}
}


void gramschmidtCuda(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, DATA_TYPE* A_outputFromGpu, int m, int n)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)n) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)n) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
	DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * m * n);
	cudaMalloc((void **)&R_gpu, sizeof(DATA_TYPE) * m * n);
	cudaMalloc((void **)&Q_gpu, sizeof(DATA_TYPE) * m * n);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * m * n, cudaMemcpyHostToDevice);
	
	t_start = rtclock();
	int k;
	for (k = 0; k < n; k++)
	{
		gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, k, m, n);
		cudaThreadSynchronize();
		gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, k, m, n);
		cudaThreadSynchronize();
		gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, k, m, n);
		cudaThreadSynchronize();
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * m * n, cudaMemcpyDeviceToHost);    

	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);
}


int main(int argc, char *argv[])
{
	// double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;
	DATA_TYPE* R;
	DATA_TYPE* Q;

	int m, n;

	if(argc != 2){
		fprintf(stdout, "E.g.: exe size\n");
		return 1;
	}

	m = atoi(argv[1]);
	n = m;
	
	A = (DATA_TYPE*)malloc(m*n*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(m*n*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(m*n*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(m*n*sizeof(DATA_TYPE));  
	
	// init_array(A);
	
	GPU_argv_init();
	gramschmidtCuda(A, R, Q, A_outputFromGpu, m, n);
	
	// t_start = rtclock();
	// gramschmidt(A, R, Q);
	// t_end = rtclock();

	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(A, A_outputFromGpu);
	
	free(A);
	free(A_outputFromGpu);
	free(R);
	free(Q);  

    	return 0;
}

