/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <MersenneTwister.h>

#define BLKWIDTH 32


//______________________________________________________________________
//
//
//  The following compares the random number generation on the CPU vs GPU
//
//
//______________________________________________________________________

//______________________________________________________________________
//
inline int RoundUp(double d)
{
  if(d>=0){
    if((d-(int)d) == 0){
      return (int)d;
    } else{
      return (int)(d+1);
    }
  } else {
    return (int)d;
  }
}
//______________________________________________________________________
//
void stopwatch( char message[], time_t start)
{    
  double secs;
  time_t stop;                 /* timing variables             */

  stop = time(nullptr);
  secs = difftime(stop, start);               
  fprintf(stdout,"    %.f [s] %s  \n",secs, message);       
}
//______________________________________________________________________
//  CPU based random number generations
void randCPU( double *M, int nRandNums)
{
  MTRand mTwister;
  for (int i = 0; i< nRandNums; i++){
    M[i] = mTwister.rand();
    // printf( "%i rand: %g \n",i, M[i]);
  }
}


//______________________________________________________________________
//  Determine device properties
void deviceProperties( int &maxThreadsPerBlock )
{  
  // Number of CUDA devices
  int devCount;
  cudaGetDeviceCount(&devCount);

  // Iterate through devices
  for (int deviceNum = 0; deviceNum < devCount; ++deviceNum){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNum);
   // printDevProp(deviceProp);

    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  }
}

//______________________________________________________________________
//  This is the host side random number generation using cuda
void randHostGPU( double *M, int nRandNums)
{
  int size = nRandNums* sizeof(double);
  double* Md;

  //__________________________________
  //  allocate device memory and copy memory to the device
  cudaMalloc( (void**)&Md, size);  
  
  cudaMemcpy( Md, M, size, cudaMemcpyHostToDevice );
  
  //__________________________________
  // Create pseudo-random number generator
  // set the seed 
  // generate the numbers
  curandGenerator_t randGen;
  
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(randGen, 1234ULL);

  curandGenerateUniformDouble(randGen, Md, nRandNums);
 
  
  //__________________________________
  //   copy from device memory and free device matrices
  cudaMemcpy( M, Md, size, cudaMemcpyDeviceToHost );
  cudaFree( Md );
  curandDestroyGenerator(randGen);
}

//______________________________________________________________________
//    Returns an random number
__device__ double randDevice(curandState* globalState, const int tid)
{
  curandState localState = globalState[tid];
  double val = curand(&localState);
  globalState[tid] = localState;
  return (double)val * (1.0/4294967295.0);
}


//______________________________________________________________________
//    Returns an random number  excluding 0 & 1.0.  See MersenneTwister.h
//
__device__ double randDblExcDevice(curandState* globalState, const int tid)
{
  curandState localState = globalState[tid];
  double val = curand(&localState);
  globalState[tid] = localState;
  return ( double(val) + 0.5 ) * (1.0/4294967296.0);
}

//______________________________________________________________________
//
__global__ void setup_kernel(curandState* randNumStates)
{
   int tID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
   /* Each thread gets same seed, a different sequence number, no offset */
   curand_init(1234, tID, 0, &randNumStates[tID]);
}
//______________________________________________________________________
//    Kernel:  
__global__ void randNumKernel( curandState* randNumStates, double* M, double* N, int nRandNums )
{
  int tID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
//  for (int k = 0; k < nRandNums; ++k){
    M[tID] = randDblExcDevice( randNumStates, tID);
    N[tID] = randDevice( randNumStates, tID );
//  }
}

//______________________________________________________________________
//  Device side random number generator
void randDeviceGPU( double *M, double *N,int nRandNums)
{
  int size = nRandNums* sizeof(double);
  double* Md;
  double* Nd;
  //__________________________________
  //  allocate device memory and copy memory to the device
  cudaMalloc( (void**)&Md, size);  
  cudaMalloc( (void**)&Nd, size);
  //__________________________________
  //  copy host memory -> device
  cudaMemcpy( Md, M, size, cudaMemcpyHostToDevice );
  cudaMemcpy( Nd, N, size, cudaMemcpyHostToDevice );  
  //__________________________________
  //
  int maxThreadsPerBlock = 0;
  deviceProperties( maxThreadsPerBlock );
  
  int xMaxThreadsPerBlock = BLKWIDTH;
  int yMaxThreadsPerBlock = BLKWIDTH;
  maxThreadsPerBlock = xMaxThreadsPerBlock * yMaxThreadsPerBlock;       // hardwired for now
  
  
  int threadsPerBlock = min(maxThreadsPerBlock, nRandNums);
  
  int xBlocks = 0;
  int yBlocks = 0;
  
  if( nRandNums > maxThreadsPerBlock){
    int nBlocks = RoundUp(  nRandNums/sqrt(maxThreadsPerBlock) );
    xBlocks = RoundUp(  nRandNums/xMaxThreadsPerBlock );
    yBlocks = RoundUp(  nRandNums/yMaxThreadsPerBlock );
  }else{
    xBlocks = 1;   // if matrix is smaller than 1 block
    yBlocks = 1;
  }
  
  int nBlocks = xBlocks = yBlocks;           // Assumption that
  int me = xBlocks * yBlocks * threadsPerBlock;
  
  fprintf(stdout, "    xBlocks: %d, yBlocks: %d, nRandNums: %d BLKWIDTH: %d, threadsPerBlock %d ",xBlocks, yBlocks, nRandNums, BLKWIDTH, threadsPerBlock);
  fprintf(stdout, "    number of threads: %d\n",me);
  
  //__________________________________
  //  Kernel invocation
  dim3 dimBlock(BLKWIDTH, BLKWIDTH, 1);
  dim3 dimGrid( xBlocks,  yBlocks,  1);
  
  // setup random number generator states on the device, 1 for each thread
  curandState* randNumStates;
  int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
  cudaMalloc((void**)&randNumStates, numStates * sizeof(curandState));

  //__________________________________
  //  Global Memory Kernel
  time_t start = time(nullptr);
  setup_kernel<<<dimGrid, dimBlock>>>( randNumStates );
  stopwatch("  randDeviceGPU setup_kernel: ", start);
  
  start = time(nullptr);
  randNumKernel<<<dimGrid, dimBlock>>>( randNumStates, Md, Nd, nRandNums );
  stopwatch("  randDeviceGPU randNumKernel: ", start);
  
  //__________________________________
  //   copy from device memory and free memory
  start = time(nullptr);
  cudaMemcpy( M, Md, size, cudaMemcpyDeviceToHost );
  cudaMemcpy( N, Nd, size, cudaMemcpyDeviceToHost );
  stopwatch(" randDeviceGPU memcopy: ", start);
  
  start = time(nullptr);
  cudaFree( Md );
  cudaFree( Nd );
  cudaFree(randNumStates) ;
  stopwatch("  randDeviceGPU free memory: ", start);
}


//______________________________________________________________________
int main( int argc, char** argv)
{  

//  for(int power = 4; power<8; ++power) { 
//    int nRandNums = pow(10,power);
    int nRandNums = 512;   
    fprintf(stdout,"__________________________________\n");
    fprintf(stdout," nRand %d  \n", nRandNums);
    
    //__________________________________
    //  allocate memory
    unsigned int size = nRandNums;
    unsigned int mem_size = sizeof(double) * size;
    double* rand_CPU       = (double*)malloc(mem_size); 
    double* rand_hostGPU   = (double*)malloc(mem_size);
    double* rand_devGPU_M  = (double*)malloc(mem_size);
    double* rand_devGPU_N  = (double*)malloc(mem_size); 
       
    time_t start;
    start = time(nullptr);
    //__________________________________
    //  Compute the random numbers
    randCPU( rand_CPU, nRandNums );
    stopwatch(" randCPU: ", start);
    
    start = time(nullptr);
    randHostGPU( rand_hostGPU, nRandNums);
    stopwatch(" randHostGPU: ", start);
     
    start = time(nullptr);    
    randDeviceGPU( rand_devGPU_M, rand_devGPU_N, nRandNums);
    stopwatch(" randDeviceGPU: ", start);
    
    //__________________________________
    //  Output data
    FILE *fp;
    fp = fopen("randomNumbers.dat", "w");
    fprintf( fp, "#CPU, hostCPU, GPU_dblExc, GPU_dblInc\n");
    for (int i = 0; i< nRandNums; i++){
      fprintf( fp, "%i, %16.15E, %16.15E, %16.15E,  %16.15E\n",i, rand_CPU[i], rand_hostGPU[i], rand_devGPU_M[i], rand_devGPU_N[i] );
      //printf(      "%i, %16.15E, %16.15E, %16.15E,  %16.15E\n",i, rand_CPU[i], rand_hostGPU[i], rand_devGPU_M[i], rand_devGPU_N[i] );
    }
    fclose(fp);
    
    //__________________________________
    //Free memory
    free( rand_CPU );
    free( rand_hostGPU );
    free( rand_devGPU_M );
    free( rand_devGPU_N );
//  }   // loop 
}




