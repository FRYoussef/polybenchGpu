/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
// # define ni 2048
// # define nj 2048
// # define nk 2048
// # define nl 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, int ni, int nj, int nk, int nl)
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i*ni + j] = ((DATA_TYPE) i*j) / ni;
		}
	}

	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i*nk + j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}

	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i*nl + j] = ((DATA_TYPE) i*(j+3)) / nl;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i*nl + j] = ((DATA_TYPE) i*(j+2)) / nk;	
		}
	}
}


// void compareResults(DATA_TYPE *E, DATA_TYPE *E_outputFromGpu)
// {
// 	int i,j,fail;
// 	fail = 0;

// 	for (i=0; i < nl; i++)
// 	{
// 		for (j=0; j < ni; j++)
// 		{
// 			if (percentDiff(E[i*ni + j], E_outputFromGpu[i*ni + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


__global__ void mm2_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int ni, int nj, int nk, int nl)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nj))
	{ 
		int k;
		for (k = 0; k < nk; k++)
		{
			C[i * nj + j] += A[i * nk + k] * B[k * nj + j];
		}
	}
}


__global__ void mm2_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, int ni, int nj, int nk, int nl)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nl))
	{ 
		int k;
		for (k = 0; k < nj; k++)
		{
			E[i * nl + j] += C[i * nj + k] * D[k * nl + j];
		}
	}
}


// void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
// {
// 	int i, j, k;
	
//   	for (i = 0; i < ni; i++)
// 	{
// 		for (j = 0; j < nj; j++)
// 		{
// 			C[i*nj + j] = 0.0;
// 			for (k = 0; k < nk; ++k)
// 			{
// 				C[i*nj + j] += A[i*nk + k] * B[k*nj + j];
// 			}
// 		}
// 	}
	
// 	for (i = 0; i < ni; i++)
// 	{
// 		for (j = 0; j < nl; j++)
// 		{
// 			E[i*nl + j] = 0.0;
// 			for (k = 0; k < nj; ++k)
// 			{
// 				E[i*nl + j] += C[i*nj + k] * D[k*nl + j];
// 			}
// 		}
// 	}
// }


void mm2Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* E_outputFromGpu, 
	int ni, int nj, int nk, int nl, int local_size)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * ni * nk);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * nk * nj);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * ni * nj);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * nj * nl);
	cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * ni * nl);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * nj * nl, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * ni * nl, cudaMemcpyHostToDevice);	
		
	dim3 block(local_size, local_size);
	dim3 grid1((size_t)ceil( ((float)nj) / ((float)block.x) ), (size_t)ceil( ((float)ni) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)nl) / ((float)block.x) ), (size_t)ceil( ((float)ni) / ((float)block.y)) );
	t_start = rtclock();
	mm2_kernel1<<<grid1,block>>>(A_gpu, B_gpu, C_gpu, ni, nj, nk, nl);
	cudaThreadSynchronize();
	mm2_kernel2<<<grid2,block>>>(C_gpu, D_gpu, E_gpu, ni, nj, nk, nl);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(E_outputFromGpu, E_gpu, sizeof(DATA_TYPE) * ni * nl, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
}


int main(int argc, char** argv)
{
	// double t_start, t_end;
	int ni, nj, nk, nl, local_size;

	if(argc != 3){
		fprintf(stdout, "E.g.: exe size local_size\n");
		return 1;
	}

	ni = atoi(argv[1]);
	nj = ni;
	nk = ni;
	nl = ni;
	local_size = atoi(argv[2]);
	
	DATA_TYPE* C;
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* E_outputFromGpu;

	C = (DATA_TYPE*)malloc(ni*nj*sizeof(DATA_TYPE));
	A = (DATA_TYPE*)malloc(ni*nk*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(nk*nj*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(nj*nl*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(ni*nl*sizeof(DATA_TYPE));
	E_outputFromGpu = (DATA_TYPE*)malloc(ni*nl*sizeof(DATA_TYPE));

  	init_array(A, B, C, D, ni, nj, nk, nl);
	GPU_argv_init();

	mm2Cuda(A, B, C, D, E, E_outputFromGpu, ni, nj, nk, nl, local_size);

	// t_start = rtclock();
	// mm2_cpu(A, B, C, D, E);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	// compareResults(E, E_outputFromGpu);

	free(C);
	free(A);
	free(B);
	free(D);
	free(E);
	free(E_outputFromGpu);

  	return 0;
}

