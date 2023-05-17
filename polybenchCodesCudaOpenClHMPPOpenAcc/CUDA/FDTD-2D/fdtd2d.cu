/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 500
// #define nx 2048
// #define ny 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, int nx, int ny)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			ex[i*ny + j] = ((DATA_TYPE) i*(j+1) + 1) / nx;
			ey[i*ny + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / nx;
			hz[i*ny + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / nx;
		}
	}
}


// void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
// {
// 	int t, i, j;
	
// 	for (t=0; t < tmax; t++)  
// 	{
// 		for (j=0; j < ny; j++)
// 		{
// 			ey[0*ny + j] = _fict_[t];
// 		}
	
// 		for (i = 1; i < nx; i++)
// 		{
//        		for (j = 0; j < ny; j++)
// 			{
//        			ey[i*ny + j] = ey[i*ny + j] - 0.5*(hz[i*ny + j] - hz[(i-1)*ny + j]);
//         		}
// 		}

// 		for (i = 0; i < nx; i++)
// 		{
//        		for (j = 1; j < ny; j++)
// 			{
// 				ex[i*(ny+1) + j] = ex[i*(ny+1) + j] - 0.5*(hz[i*ny + j] - hz[i*ny + (j-1)]);
// 			}
// 		}

// 		for (i = 0; i < nx; i++)
// 		{
// 			for (j = 0; j < ny; j++)
// 			{
// 				hz[i*ny + j] = hz[i*ny + j] - 0.7*(ex[i*(ny+1) + (j+1)] - ex[i*(ny+1) + j] + ey[(i+1)*ny + j] - ey[i*ny + j]);
// 			}
// 		}
// 	}
// }


// void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
// {
// 	int i, j, fail;
// 	fail = 0;
	
// 	for (i=0; i < nx; i++) 
// 	{
// 		for (j=0; j < ny; j++) 
// 		{
// 			if (percentDiff(hz1[i*ny + j], hz2[i*ny + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
// 			{
// 				fail++;
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
}



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, int nx, int ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < nx) && (j < ny))
	{
		if (i == 0) 
		{
			ey[i * ny + j] = _fict_[t];
		}
		else
		{ 
			ey[i * ny + j] = ey[i * ny + j] - 0.5f*(hz[i * ny + j] - hz[(i-1) * ny + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, int nx, int ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < nx) && (j < ny) && (j > 0))
	{
		ex[i * (ny+1) + j] = ex[i * (ny+1) + j] - 0.5f*(hz[i * ny + j] - hz[i * ny + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t, int nx, int ny)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < nx) && (j < ny))
	{	
		hz[i * ny + j] = hz[i * ny + j] - 0.7f*(ex[i * (ny+1) + (j+1)] - ex[i * (ny+1) + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu, int nx, int ny)
{
	double t_start, t_end;

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * nx * (ny + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (nx + 1) * ny);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * nx * ny);

	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
	cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * nx * (ny + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (nx + 1) * ny, cudaMemcpyHostToDevice);
	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)ny) / ((float)block.x)), (size_t)ceil(((float)nx) / ((float)block.y)));

	t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t, nx, ny);
		cudaThreadSynchronize();
		fdtd_step2_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, nx, ny);
		cudaThreadSynchronize();
		fdtd_step3_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, nx, ny);
		cudaThreadSynchronize();
	}
	
	t_end = rtclock();
    	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * nx * ny, cudaMemcpyDeviceToHost);	
		
	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
}


int main(int argc, char** argv)
{
	// double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;

	int nx, ny;

	if(argc != 2){
		fprintf(stdout, "E.g.: exe size\n");
		return 1;
	}

	nx = atoi(argv[1]);
	ny = nx;

	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(nx*(ny+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((nx+1)*ny*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(nx*ny*sizeof(DATA_TYPE));

	init_arrays(_fict_, ex, ey, hz, nx, ny);

	GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu, nx, ny);

	// t_start = rtclock();
	// runFdtd(_fict_, ex, ey, hz);
	// t_end = rtclock();
	
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(hz, hz_outputFromGpu);

	free(_fict_);
	free(ex);
	free(ey);
	free(hz);
	free(hz_outputFromGpu);

	return 0;
}

