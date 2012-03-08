/*

 The MIT License

 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
 University of Utah.

 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 */

//-----------------------------------------------------------------------------

/*
Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
University.  All rights reserved.
Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
University and University of Tokyo.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the Hiroshima University nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


//----- MersenneTwister.cu ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwisterGPU.cuh>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/DbgOutput.h>
#include <sci_defs/cuda_defs.h>
#include <stdint.h>

//--------------------------------------------------------------
//
using namespace Uintah;
using namespace std;
static DebugStream dbg("MERSENNE_TWISTER_GPU",       false);
static DebugStream dbg2("MERSENNE_TWISTER_GPU_DEBUG",false);
static DebugStream dbg_BC("MERSENNE_TWISTER_GPU_BC", false);

MTRandGPU::MTRandGPU()
{

}

MTRandGPU::~MTRandGPU()
{

}

/**
 * This function sets constants in device memory.
 * @param params input, MTGP64 parameters.
 */
void MTRandGPU::make_constant(const mtgp64_params_fast_t params[], int block_num)
{
  const int size1 = sizeof(uint32_t) * block_num;
  uint32_t *h_pos_tbl;
  uint32_t *h_sh1_tbl;
  uint32_t *h_sh2_tbl;
  uint32_t *h_mask;
  h_pos_tbl = (uint32_t *)malloc(size1);
  h_sh1_tbl = (uint32_t *)malloc(size1);
  h_sh2_tbl = (uint32_t *)malloc(size1);
  h_mask = (uint32_t *)malloc(sizeof(uint32_t) * 2);
  if (h_pos_tbl == NULL || h_sh1_tbl == NULL || h_sh2_tbl == NULL || h_mask == NULL) {
    printf("failure in allocating host memory for constant table.\n");
    exit(1);
  }
  h_mask[0] = params[0].mask >> 32;
  h_mask[1] = params[0].mask & 0xffffffffU;
  for (int i = 0; i < block_num; i++) {
    h_pos_tbl[i] = params[i].pos;
    h_sh1_tbl[i] = params[i].sh1;
    h_sh2_tbl[i] = params[i].sh2;
  }
  // copy from malloc area only
  cudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1);
  cudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1);
  cudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1);
  cudaMemcpyToSymbol(mask, h_mask, sizeof(uint32_t) * 2);
  free(h_pos_tbl);
  free(h_sh1_tbl);
  free(h_sh2_tbl);
  free(h_mask);
}

/**
 * This function sets constants in device memory.
 * @param params input, MTGP64 parameters.
 */
void MTRandGPU::make_texture(const mtgp64_params_fast_t params[],
                             uint32_t *d_texture_tbl[3],
                             int block_num)
{
  const int count = block_num * TBL_SIZE;
  const int size = sizeof(uint32_t) * count;
  uint32_t *h_texture_tbl[3];
  int i, j;
  for (i = 0; i < 3; i++) {
    h_texture_tbl[i] = (uint32_t *)malloc(size);
    if (h_texture_tbl[i] == NULL) {
      for (j = 0; j < i; j++) {
        free(h_texture_tbl[i]);
      }
      printf("failure in allocating host memory for constant table.\n");
      exit(1);
    }
  }
  for (int i = 0; i < block_num; i++) {
    for (int j = 0; j < TBL_SIZE; j++) {
      h_texture_tbl[0][i * TBL_SIZE + j] = params[i].tbl[j] >> 32;
      h_texture_tbl[1][i * TBL_SIZE + j] = params[i].tmp_tbl[j] >> 32;
      h_texture_tbl[2][i * TBL_SIZE + j] = params[i].dbl_tmp_tbl[j] >> 32;
    }
  }
  cudaMemcpy(d_texture_tbl[0], h_texture_tbl[0], size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_texture_tbl[1], h_texture_tbl[1], size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_texture_tbl[2], h_texture_tbl[2], size, cudaMemcpyHostToDevice);
  tex_param_ref.filterMode = cudaFilterModePoint;
  tex_temper_ref.filterMode = cudaFilterModePoint;
  tex_double_ref.filterMode = cudaFilterModePoint;
  cudaBindTexture(0, tex_param_ref, d_texture_tbl[0], size);
  cudaBindTexture(0, tex_temper_ref, d_texture_tbl[1], size);
  cudaBindTexture(0, tex_double_ref, d_texture_tbl[2], size);
  free(h_texture_tbl[0]);
  free(h_texture_tbl[1]);
  free(h_texture_tbl[2]);
}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP64 parameters. needed for the initialization.
 */
void MTRandGPU::make_kernel_data64(mtgp64_kernel_status_t *d_status,
                        mtgp64_params_fast_t params[],
                        int block_num)
{
  mtgp64_kernel_status_t* h_status = (mtgp64_kernel_status_t *)malloc(sizeof(mtgp64_kernel_status_t) * block_num);

  if (h_status == NULL) {
    printf("failure in allocating host memory for kernel I/O data.\n");
    exit(8);
  }
  for (int i = 0; i < block_num; i++) {
    mtgp64_init_state(&(h_status[i].status[0]), &params[i], i + 1);
  }
#ifdef __STDC_FORMAT_MACROS
  printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[0]);
  printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[1]);
  printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[2]);
  printf("h_status[0].status[0]:%016"PRIx64"\n", h_status[0].status[3]);
#endif
  cudaMemcpy(d_status, h_status, sizeof(mtgp64_kernel_status_t) * block_num, cudaMemcpyHostToDevice);
  free(h_status);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void MTRandGPU::make_uint64_random(mtgp64_kernel_status_t* d_status,
                        int num_data,
                        int block_num)
{
  uint64_t* d_data;
  uint64_t* h_data;
  cudaError_t e;
  float gputime;
  cudaEvent_t start;
  cudaEvent_t end;

  printf("generating 64-bit unsigned random numbers.\n");
  cudaMalloc((void**)&d_data, sizeof(uint64_t) * num_data);
  /* ccutCreateTimer(&timer); */
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  h_data = (uint64_t *)malloc(sizeof(uint64_t) * num_data);
  if (h_data == NULL) {
    printf("failure in allocating host memory for output data.\n");
    exit(1);
  }
  /* cutStartTimer(timer); */
  cudaEventRecord(start, 0);
  if (cudaGetLastError() != cudaSuccess) {
    printf("error has been occured before kernel call.\n");
    exit(1);
  }

  /* kernel call */
  mtgp64_uint64_kernel<<< block_num, THREAD_NUM>>>(d_status, d_data, num_data / block_num);
  cudaThreadSynchronize();

  e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
    exit(1);
  }
  /* cutStopTimer(timer); */
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaMemcpy(h_data, d_data, sizeof(uint64_t) * num_data, cudaMemcpyDeviceToHost);
  /* gputime = cutGetTimerValue(timer); */
  cudaEventElapsedTime(&gputime, start, end);
//  print_uint64_array(h_data, num_data, block_num);
  printf("generated numbers: %d\n", num_data);
  printf("Processing time: %f (ms)\n", gputime);
  printf("Samples per second: %E \n", num_data / (gputime * 0.001));
  /* cutDeleteTimer(timer); */
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  //free memories
  free(h_data);
  cudaFree(d_data);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void MTRandGPU::make_double01_random(mtgp64_kernel_status_t* d_status, int num_data, int block_num)
{
  double* d_data;
  double* h_data;
  cudaError_t e;
  float gputime;
  cudaEvent_t start;
  cudaEvent_t end;

  printf("generating double precision floating point random numbers.\n");
  cudaMalloc((void**)&d_data, sizeof(double) * num_data);
  /* CUT_SAFE_CALL(cutCreateTimer(&timer)); */
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  h_data = (double *)malloc(sizeof(double) * num_data);
  if (h_data == NULL) {
    printf("failure in allocating host memory for output data.\n");
    exit(1);
  }
  /* CUT_SAFE_CALL(cutStartTimer(timer)); */
  cudaEventRecord(start, 0);
  if (cudaGetLastError() != cudaSuccess) {
    printf("error has been occured before kernel call.\n");
    exit(1);
  }

  /* kernel call */
  mtgp64_double01_kernel<<< block_num, THREAD_NUM >>>(d_status, d_data, num_data / block_num);
  cudaThreadSynchronize();

  e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
    exit(1);
  }
  /* CUT_SAFE_CALL(cutStopTimer(timer)); */
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaMemcpy(h_data, d_data, sizeof(uint64_t) * num_data, cudaMemcpyDeviceToHost);
  /* gputime = cutGetTimerValue(timer); */
  cudaEventElapsedTime(&gputime, start, end);

//  print_double_array(h_data, num_data, block_num);

  printf("generated numbers: %d\n", num_data);
  printf("Processing time: %f (ms)\n", gputime);
  printf("Samples per second: %E \n", num_data / (gputime * 0.001));
  /* CUT_SAFE_CALL(cutDeleteTimer(timer)); */
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  //free memories
  free(h_data);
  cudaFree(d_data);
}

/**
 * This function initializes the internal state array with a 64-bit
 * integer seed. The allocated memory should be freed by calling
 * mtgp64_free(). \b para should be one of the elements in the
 * parameter table (mtgp64-param-ref.c).
 *
 * This function is call by cuda program, because cuda program uses
 * another structure and another allocation method.
 *
 * @param[out] array MTGP internal status vector.
 * @param[in] para parameter structure
 * @param[in] seed a 64-bit integer used as the seed.
 */
void MTRandGPU::mtgp64_init_state(uint64_t array[],
                       const mtgp64_params_fast_t *para,
                       uint64_t seed)
{
  int i;
  int size = para->mexp / 64 + 1;
  uint64_t hidden_seed;
  uint64_t tmp;
  hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
  tmp = hidden_seed >> 32;
  tmp += tmp >> 16;
  tmp += tmp >> 8;
  memset(array, tmp & 0xff, sizeof(uint64_t) * size);
  array[0] = seed;
  array[1] = hidden_seed;
  for (i = 1; i < size; i++) {
//    array[i] ^= UINT64_C(6364136223846793005) * (array[i - 1] ^ (array[i - 1] >> 62)) + i;
  }
}

/**
 * kernel function.
 * This function generates 64-bit unsigned integers in d_data
 *
 * @params d_status kernel I/O data
 * @params d_data output
 * @params size number of output data requested.
 */
__global__ void mtgp64_uint64_kernel(mtgp64_kernel_status_t* d_status,
                                     uint64_t* d_data,
                                     int size)
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  int pos = pos_tbl[bid];
  uint32_t YH;
  uint32_t YL;
  uint64_t o;

  // copy status data from global memory to shared memory.
  status_read(status, d_status, bid, tid);

  // main loop
  for (int i = 0; i < size; i += LARGE_SIZE) {

#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
      printf("status[0][LARGE_SIZE - N + tid]:%08x\n",
          status[0][LARGE_SIZE - N + tid]);
      printf("status[1][LARGE_SIZE - N + tid]:%08x\n",
          status[1][LARGE_SIZE - N + tid]);
      printf("status[0][LARGE_SIZE - N + tid + 1]:%08x\n",
          status[0][LARGE_SIZE - N + tid + 1]);
      printf("status[1][LARGE_SIZE - N + tid + 1]:%08x\n",
          status[1][LARGE_SIZE - N + tid + 1]);
      printf("status[0][LARGE_SIZE - N + tid + pos]:%08x\n",
          status[0][LARGE_SIZE - N + tid + pos]);
      printf("status[1][LARGE_SIZE - N + tid + pos]:%08x\n",
          status[1][LARGE_SIZE - N + tid + pos]);
      printf("sh1:%d\n", sh1_tbl[bid]);
      printf("sh2:%d\n", sh2_tbl[bid]);
      printf("high_mask:%08x\n", mask[0]);
      printf("low_mask:%08x\n", mask[1]);
      for (int j = 0; j < 16; j++) {
        printf("tbl[%d]:%08x\n", j, param_tbl[0][j]);
      }
    }
#endif
    para_rec(&YH, &YL, status[0][LARGE_SIZE - N + tid], status[1][LARGE_SIZE - N + tid], status[0][LARGE_SIZE - N + tid + 1],
             status[1][LARGE_SIZE - N + tid + 1], status[0][LARGE_SIZE - N + tid + pos], status[1][LARGE_SIZE - N + tid + pos],
             bid);
    status[0][tid] = YH;
    status[1][tid] = YL;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
      printf("status[0][tid]:%08x\n", status[0][tid]);
      printf("status[1][tid]:%08x\n", status[1][tid]);
    }
#endif
    o = temper(YH, YL, status[1][LARGE_SIZE - N + tid + pos - 1], bid);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
    if ((i == 0) && (bid == 0) && (tid <= 1)) {
      printf("o:%016" PRIx64 "\n", o);
    }
#endif
    d_data[size * bid + i + tid] = o;
    __syncthreads();

    para_rec(&YH, &YL, status[0][(4 * THREAD_NUM - N + tid) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
             status[0][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
             status[0][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
             bid);
    status[0][tid + THREAD_NUM] = YH;
    status[1][tid + THREAD_NUM] = YL;
    o = temper(YH, YL, status[1][(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE], bid);
    d_data[size * bid + THREAD_NUM + i + tid] = o;
    __syncthreads();

    para_rec(&YH, &YL, status[0][2 * THREAD_NUM - N + tid], status[1][2 * THREAD_NUM - N + tid],
             status[0][2 * THREAD_NUM - N + tid + 1], status[1][2 * THREAD_NUM - N + tid + 1],
             status[0][2 * THREAD_NUM - N + tid + pos], status[1][2 * THREAD_NUM - N + tid + pos], bid);
    status[0][tid + 2 * THREAD_NUM] = YH;
    status[1][tid + 2 * THREAD_NUM] = YL;
    o = temper(YH, YL, status[1][tid + pos - 1 + 2 * THREAD_NUM - N], bid);
    d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
    __syncthreads();
  }
  // write back status for next call
  status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates double precision floating point numbers in d_data.
 * Resulted outputs are distributed in the range [0, 1).
 *
 * @params d_status kernel I/O data
 * @params d_data output. IEEE double precision format.
 * @params size number of output data requested.
 */
__global__ void mtgp64_double01_kernel(mtgp64_kernel_status_t* d_status,
                                       double* d_data,
                                       int size)
{

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  int pos = pos_tbl[bid];
  uint32_t YH;
  uint32_t YL;
  double o;

  // copy status data from global memory to shared memory.
  status_read(status, d_status, bid, tid);

  // main loop
  for (int i = 0; i < size; i += LARGE_SIZE) {
    para_rec(&YH, &YL, status[0][LARGE_SIZE - N + tid], status[1][LARGE_SIZE - N + tid], status[0][LARGE_SIZE - N + tid + 1],
             status[1][LARGE_SIZE - N + tid + 1], status[0][LARGE_SIZE - N + tid + pos], status[1][LARGE_SIZE - N + tid + pos],
             bid);
    status[0][tid] = YH;
    status[1][tid] = YL;
    o = temper_double01(YH, YL, status[1][LARGE_SIZE - N + tid + pos - 1], bid);
    d_data[size * bid + i + tid] = o;
    __syncthreads();

    para_rec(&YH, &YL, status[0][(4 * THREAD_NUM - N + tid) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
             status[0][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
             status[0][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE], status[1][(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
             bid);
    status[0][tid + THREAD_NUM] = YH;
    status[1][tid + THREAD_NUM] = YL;
    o = temper_double01(YH, YL, status[1][(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE], bid);
    d_data[size * bid + THREAD_NUM + i + tid] = o;
    __syncthreads();

    para_rec(&YH, &YL, status[0][2 * THREAD_NUM - N + tid], status[1][2 * THREAD_NUM - N + tid],
             status[0][2 * THREAD_NUM - N + tid + 1], status[1][2 * THREAD_NUM - N + tid + 1],
             status[0][2 * THREAD_NUM - N + tid + pos], status[1][2 * THREAD_NUM - N + tid + pos], bid);
    status[0][tid + 2 * THREAD_NUM] = YH;
    status[1][tid + 2 * THREAD_NUM] = YL;
    o = temper_double01(YH, YL, status[1][tid + pos - 1 + 2 * THREAD_NUM - N], bid);
    d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
    __syncthreads();
  }
  // write back status for next call
  status_write(d_status, status, bid, tid);
}

/**
 * The tempering function.
 *
 * @param VH MSBs of the output value should be tempered.
 * @param VL LSBs of the output value should be tempered.
 * @param TL LSBs of the tempering helper value.
 * @param bid block id.
 * @return the tempered value.
 */
__device__ uint64_t temper(uint32_t VH,
                           uint32_t VL,
                           uint32_t TL,
                           int bid)
{
  uint32_t MAT;
  uint64_t r;
  TL ^= TL >> 16;
  TL ^= TL >> 8;
  MAT = tex1Dfetch(tex_temper_ref, bid * 16 + (TL & 0x0f));
  VH ^= MAT;
  r = ((uint64_t)VH << 32) | VL;
  return r;
}

/**
 * The tempering and converting function.
 * By using the presetted table, converting to IEEE format
 * and tempering are done simultaneously.
 * Resulted outputs are distributed in the range [0, 1).
 *
 * @param VH MSBs of the output value should be tempered.
 * @param VL LSBs of the output value should be tempered.
 * @param TL LSBs of the tempering helper value.
 * @param bid block id.
 * @return the tempered and converted value.
 */
__device__ double temper_double01(uint32_t VH,
                                  uint32_t VL,
                                  uint32_t TL,
                                  int bid)
{
  uint32_t MAT;
  uint64_t r;
  TL ^= TL >> 16;
  TL ^= TL >> 8;
  MAT = tex1Dfetch(tex_double_ref, bid * 16 + (TL & 0x0f));
  r = ((uint64_t)VH << 32) | VL;
  r = (r >> 12) ^ ((uint64_t)MAT << 32);
  return __longlong_as_double(r) - 1.0;
}

/**
 * The function of the recursion formula calculation.
 *
 * @param RH 32-bit MSBs of output
 * @param RL 32-bit LSBs of output
 * @param X1H MSBs of the farthest part of state array.
 * @param X1L LSBs of the farthest part of state array.
 * @param X2H MSBs of the second farthest part of state array.
 * @param X2L LSBs of the second farthest part of state array.
 * @param YH MSBs of a part of state array.
 * @param YL LSBs of a part of state array.
 * @param bid block id.
 */
__device__ void para_rec(uint32_t *RH,
                         uint32_t *RL,
                         uint32_t X1H,
                         uint32_t X1L,
                         uint32_t X2H,
                         uint32_t X2L,
                         uint32_t YH,
                         uint32_t YL,
                         int bid)
{
  uint32_t XH = (X1H & mask[0]) ^ X2H;
  uint32_t XL = (X1L & mask[1]) ^ X2L;
  uint32_t MAT;

  XH ^= XH << sh1_tbl[bid];
  XL ^= XL << sh1_tbl[bid];
  YH = XL ^ (YH >> sh2_tbl[bid]);
  YL = XH ^ (YL >> sh2_tbl[bid]);
  MAT = tex1Dfetch(tex_param_ref, bid * 16 + (YL & 0x0f));
  *RH = YH ^ MAT;
  *RL = YL;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param status shared memory.
 * @param d_status kernel I/O data
 * @param bid block id
 * @param tid thread id
 */
__device__ void status_read(uint32_t status[2][LARGE_SIZE],
                            const mtgp64_kernel_status_t *d_status,
                            int bid,
                            int tid)
{
  uint64_t x;

  x = d_status[bid].status[tid];
  status[0][LARGE_SIZE - N + tid] = x >> 32;
  status[1][LARGE_SIZE - N + tid] = x & 0xffffffff;
  if (tid < N - THREAD_NUM) {
    x = d_status[bid].status[THREAD_NUM + tid];
    status[0][LARGE_SIZE - N + THREAD_NUM + tid] = x >> 32;
    status[1][LARGE_SIZE - N + THREAD_NUM + tid] = x & 0xffffffff;
  }
  __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param status shared memory.
 * @param d_status kernel I/O data
 * @param bid block id
 * @param tid thread id
 */
__device__ void status_write(mtgp64_kernel_status_t *d_status,
                             const uint32_t status[2][LARGE_SIZE],
                             int bid,
                             int tid)
{
  uint64_t x;

  x = (uint64_t)status[0][LARGE_SIZE - N + tid] << 32;
  x = x | status[1][LARGE_SIZE - N + tid];
  d_status[bid].status[tid] = x;
  if (tid < N - THREAD_NUM) {
    x = (uint64_t)status[0][4 * THREAD_NUM - N + tid] << 32;
    x = x | status[1][4 * THREAD_NUM - N + tid];
    d_status[bid].status[THREAD_NUM + tid] = x;
  }
  __syncthreads();
}









