/*
 * Copyright (c) 2014 The University of Utah
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

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <spatialops/structured/BoundaryCellInfo.h>
#include "CudaStencilLib.h"
#include "ExternalStencilTest.h"

using namespace SpatialOps;
using namespace Point;
using namespace ema::cuda;

#define SMALL 130
#define HEIGHT 130
#define ITERATIONS 1

void stencil7_cpu( float* data_in, float* data_out, int nx, int ny, int nz ){
  const float dx = 1;//, dy = 1, dz = 1;
  const float dxsq = dx*dx;
  //float dysq = dy*dy;
  //float dzsq = dz*dz;

  for( int k = 1; k < HEIGHT-1; ++k)
   for( int j = 1; j < SMALL-1; ++j)
    for( int i = 1; i < SMALL-1; ++i){
      for( int data_reuse = 0; data_reuse < 1; ++data_reuse ){
        const float tijk = 2*data_in[ Index3D( nx, ny, i, j, k) ];
        data_out[Index3D (nx, ny, i, j, k)] =
        (
          // X direction
          (
              data_in[Index3D (nx, ny, i - 1, j, k)] +
              data_in[Index3D (nx, ny, i + 1, j, k)]
              -
              tijk
          )
          +
          // Y direction
          (
              data_in[Index3D (nx, ny, i, j - 1, k)] +
              data_in[Index3D (nx, ny, i, j + 1, k)]
              -
              tijk
          )
          +
          // Z direction
          (
              data_in[Index3D (nx, ny, i, j, k - 1)] +
              data_in[Index3D (nx, ny, i, j, k + 1)]
              -
              tijk
          )
        ) / dxsq
        ;
      }
    }
}

int main(int argc, char** argv){
  const IntVec sset(SMALL,SMALL,HEIGHT);
  const IntVec mset(64,64,64);
  const IntVec lset(128,128,128);

  float* svals = new float[SMALL*SMALL*HEIGHT];
  float* data_out = new float[SMALL*SMALL*HEIGHT];

  clock_t tick;
  clock_t tock;

  srand(0);

  for( int i = 1; i < SMALL-1; ++i)
    for( int j = 1; j < SMALL-1; ++j)
      for( int k = 1; k < HEIGHT-1; ++k)
         svals[Index3D(SMALL,SMALL, i,j,k)] = (float)rand() / (float)RAND_MAX;

  CUDADeviceInterface& CDI = CUDADeviceInterface::self();

  const MemoryWindow swin(sset);

  const GhostData ghost(0);
  const BoundaryCellInfo bc = BoundaryCellInfo::build<PointFloatField>(false,false,false);
  PointFloatField  pff(swin, bc, ghost, NULL,  InternalStorage, CPU_INDEX);
  PointFloatField spff(swin, bc, ghost, svals, InternalStorage, GPU_INDEX);

  CDI.print_device_info();

  double avgcpu = 0;
  for( int p = 0; p < ITERATIONS; ++p){
    tick = clock();
    stencil7_cpu(svals, data_out, SMALL, SMALL, HEIGHT);
    clock_t tock = clock() - tick;
    avgcpu += tock;
  }
  avgcpu /= ITERATIONS;

  std::cout << "Elapsed: " << (double)avgcpu / (double)CLOCKS_PER_SEC << std::endl;

  //Dump first computed plane in the z axis
  std::ofstream file;
  file.open("cpuout.txt");
  for(int i = 0; i < SMALL; ++i){
    for( int j = 0; j < SMALL; ++j){
      float q = data_out[Index3D(SMALL, SMALL, i, j, 1)];
      file.precision(8);
      file << std::setw(12) << q << " ";
    }
    file << std::endl;
  }
  file.close();

  double avggpu=0;
  for(int j = 0; j < ITERATIONS; ++j){
    tick = clock();
    divergence_float_gpu(&spff, 1, 1, 1);
    tock = clock() - tick;
    avggpu += tock;
  }
  avggpu /= ITERATIONS;

  std::cout << "Average elapsed: " << (double)avggpu / (double)CLOCKS_PER_SEC << std::endl;
  std::cout << "Speedup: " << avgcpu / avggpu << std::endl;

  pff = spff;

  delete[] svals;
  if( !system("diff cudaout.txt cpuout.txt &> /dev/null") ){
    std::cout << "Success\n";
  } else {
    std::cout << "Fail: output samples do not match.\n";
  }
}
