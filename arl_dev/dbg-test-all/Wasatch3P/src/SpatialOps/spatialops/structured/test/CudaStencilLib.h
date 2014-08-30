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

#ifndef CUDASTENCILLIB_H_
#define CUDASTENCILLIB_H_

#include <cuda.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <spatialops/structured/SpatialField.h>
#include <spatialops/structured/MemoryTypes.h>

// Used to avoid obnoxious error parsing in eclipse. __CDT_PARSER__ is an
// eclipse defined variable used by the syntax parser.
#ifdef __CDT_PARSER__
#define __host__
#define __global__
#define __device__
#define __shared__
#endif

#define BLOCK_DIM 16
#define INNER_BLOCK_DIM 14

#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))
#define Index3DG(_nx, _ny, _g, _i, _j, _k) (((_i)+(_g)) + _nx*((_j)+(_g) + _ny*((_k)+(_g)) ))

void __global__ _div_float_slow( float* global_data_in,
                                 float* global_data_out,
                                 float dx, float dy, float dz,
                                 int nx, int ny, int nz){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  //Temporarily -> don't deal with computing boundary points
  if( i>0 && j>0 && (i<nx-1) && (j<ny-1) )
  {
    for(int k=1;k<nz-1;k++)/* For each frame along the z-axis */
    {
      float tijk = 2*global_data_in[ Index3D( nx, ny, i,j,k) ];
      global_data_out[Index3D (nx, ny, i, j, k)] =
      // X direction
      (
          global_data_in[Index3D (nx, ny, i - 1, j, k)] +
          global_data_in[Index3D (nx, ny, i + 1, j, k)] -
          tijk
      ) / ( dx * dx )
      +
      // Y direction
      (
          global_data_in[Index3D (nx, ny, i, j - 1, k)] +
          global_data_in[Index3D (nx, ny, i, j + 1, k)] -
          tijk
      ) / ( dy * dy )
      +
      // Z direction
      (
          global_data_in[Index3D (nx, ny, i, j, k - 1)] +
          global_data_in[Index3D (nx, ny, i, j, k + 1)] -
          tijk
      ) / ( dz * dz );
    }
  }
}

void __global__ _div_float_opt1( float* global_data_in,
                                 float* global_data_out,
                                 float dx, float dy, float dz,
                                 int nx, int ny, int nz){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  //Temporarily -> don't deal with computing boundary points
  float zm1, zc, zp1;
  zc = global_data_in[Index3D( nx, ny, i, j, 0)];
  zp1 = global_data_in[Index3D( nx, ny, i, j, 1 )];

  float dxsq = dx*dx;
  //float dysq = dy*dy;
  //float dzsq = dz*dz;

  if( i > 0 && j > 0 && ( i < nx-1 ) && ( j < ny-1 ) ) {
    for(int k = 1; k < nz-1 ; ++k)/* For each frame along the z-axis */ {
      float x;
      for( int data_reuse = 0; data_reuse < 1; ++data_reuse ){
        zm1 = zc;
        zc = zp1;
        zp1 = global_data_in[Index3D( nx, ny, i, j, k + 1 )];
        float tijk = 2*zc;

        x =
        (// X direction
          (
              global_data_in[Index3D (nx, ny, i - 1, j, k)] +
              global_data_in[Index3D (nx, ny, i + 1, j, k)] -
              tijk
          )
          +
          // Y direction
          (
              global_data_in[Index3D (nx, ny, i, j - 1, k)] +
              global_data_in[Index3D (nx, ny, i, j + 1, k)] -
              tijk
          )
          +
          // Z direction
          (
              zm1 +
              zp1 -
              tijk
          )
        ) / ( dxsq );
      }
      __syncthreads();
      global_data_out[Index3D (nx, ny, i, j, k)] = x;
    }
  }
}

// There appears to be a negative effect in making the effort to stage into shared
// memory unless we can exhibit significant reuse in the shared data. Local reuse
// between stencil points does not overcome the additional complexity of staging.
void __global__ _div_float_opt2( float* global_data_in,
                                 float* global_data_out,
                                 float dx, float dy, float dz,
                                 int nx, int ny, int nz){

  int ghost = 1; // static, because shared memory allocation has to be fixed.

  /** Compute inner tile variables **/
  int inner_dim    = INNER_BLOCK_DIM;                     // Dimensions of the inner tile
  int inner_x      = ( blockIdx.x * inner_dim ) + ghost;// Inner tile absolute 'x' val
  int inner_y      = ( blockIdx.y * inner_dim ) + ghost;// Inner tile absolute 'y' val

  /** Compute outer tile variables **/
  //int outer_dim    = BLOCK_DIM;             // Dimensions of the outer tile
  int outer_x      = inner_x - ghost;       // Outer tile absolute 'x' val
  int outer_y      = inner_y - ghost;       // Outer tile absolute 'y' val

  // Cache the vertical stencil values zm1, z, zp1
  float zlow;
  float zcenter = global_data_in[ Index3D(nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, 0) ];
  float zhigh   = global_data_in[ Index3D(nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, 1) ];

  float dxsq = dx*dx;
  //float dysq = dy*dy;
  //float dzsq = dz*dz;

  __shared__ float local_flat[BLOCK_DIM*BLOCK_DIM];

  for( int k = 1; k < nz - 1; ++k ) { // Iterate frames along the Z axis
    int inner_offset = Index3D(nx, ny, inner_x, inner_y, k);
    int outer_offset = Index3D(nx, ny, outer_x, outer_y, k);
    float tijk;

    if( ( (outer_x + threadIdx.x) < nx ) && ( (outer_y + threadIdx.y) < ny ) ){
      local_flat[ Index3D(BLOCK_DIM, BLOCK_DIM, threadIdx.x, threadIdx.y, 0) ] =
          global_data_in[outer_offset + threadIdx.x + nx * threadIdx.y];
    }
    __syncthreads();

    /*
    if( k == 1 ){
      d_debug[threadIdx.x + nx * threadIdx.y] =
          local_flat[ Index3D( BLOCK_DIM, BLOCK_DIM, threadIdx.x, threadIdx.y, 0) ];
    }
    */

    //Entire outer memory tile is loaded at this point.

    if( (threadIdx.x < INNER_BLOCK_DIM )
        && (threadIdx.y < INNER_BLOCK_DIM )
        && ( inner_x + threadIdx.x < nx - 1 )
        && ( inner_y + threadIdx.y < ny - 1 )){
      int local_index_x = threadIdx.x + 1;
      int local_index_y = threadIdx.y + 1;

      zlow = zcenter;
      zcenter = zhigh;
      zhigh = global_data_in[ Index3D( nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, k + 1 ) ];
      tijk = 2 * zcenter;

      float x;

      for( int data_reuse=0; data_reuse < 1; ++data_reuse ){
       x =
        // X direction
       (
        (
            local_flat[ Index3D( BLOCK_DIM, BLOCK_DIM, local_index_x - 1, local_index_y, 0) ] +
            local_flat[ Index3D( BLOCK_DIM, BLOCK_DIM, local_index_x + 1, local_index_y, 0) ]
            -
            tijk
        )
        +
        // Y direction
        (
            local_flat[ Index3D( BLOCK_DIM, BLOCK_DIM, local_index_x, local_index_y - 1, 0) ] +
            local_flat[ Index3D( BLOCK_DIM, BLOCK_DIM, local_index_x, local_index_y + 1, 0) ]
            -
            tijk
        )
        +
        // Z direction
        (
            zlow +
            zhigh
            -
            tijk
        )
       ) / dxsq;
      }
      global_data_out[ inner_offset + threadIdx.x + nx * threadIdx.y ] = x;
    }
    __syncthreads();
  }
}

void __global__ _div_float_opt3( float* global_data_in,
                                 float* global_data_out,
                                 float dx, float dy, float dz,
                                 int nx, int ny, int nz){

  int ghost = 1; // static, because shared memory allocation has to be fixed.

  //16x16 blocks, each block loads all 4 of its XY plane stencil components -> overlap occurs, but we
  //get thread work-load-balancing.

  /** Compute inner tile variables **/
  int inner_dim    = blockDim.x;                        // Dimensions of the inner tile
  int inner_x      = ( blockIdx.x * inner_dim ) + ghost;// Inner tile absolute 'x' val
  int inner_y      = ( blockIdx.y * inner_dim ) + ghost;// Inner tile absolute 'y' val

  /** Compute outer tile variables **/
  //int outer_dim    = blockDim.x + 2 * ghost;// Dimensions of the outer tile
  //int outer_x      = inner_x - ghost;       // Outer tile absolute 'x' val
  //int outer_y      = inner_y - ghost;       // Outer tile absolute 'y' val

  float zlow;
  float zcenter = global_data_in[ Index3D(nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, 0) ];
  float zhigh   = global_data_in[ Index3D(nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, 1) ];

  float dxsq = dx*dx;

  __shared__ float local_flat[ (BLOCK_DIM+2)*(BLOCK_DIM+2) ];

  for( int k = 1; k < nz - 1; ++k ) { // Iterate frames along the Z axis
    int inner_offset = Index3D(nx, ny, inner_x, inner_y, k);
    //int outer_offset = Index3D(nx, ny, outer_x, outer_y, k);
    float tijk;

    int local_index_x = threadIdx.x + ghost;
    int local_index_y = threadIdx.y + ghost;

    zlow = zcenter;
    zcenter = zhigh;
    zhigh = global_data_in[ Index3D( nx, ny, inner_x + threadIdx.x, inner_y + threadIdx.y, k + 1 ) ];
    tijk = 2 * zcenter;

    local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x - 1, local_index_y, 0 ) ] =
            global_data_in[ Index3D( nx, ny, inner_x - 1, inner_y, k ) ];
    local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x + 1, local_index_y, 0 ) ] =
            global_data_in[ Index3D( nx, ny, inner_x + 1, inner_y, k ) ];
    local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x, local_index_y - 1, 0 ) ] =
            global_data_in[ Index3D( nx, ny, inner_x, inner_y - 1, k ) ];
    local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x, local_index_y + 1, 0 ) ] =
            global_data_in[ Index3D( nx, ny, inner_x, inner_y + 1, k ) ];
    __syncthreads();

    float x;

    for( int data_reuse=0; data_reuse < 1; ++data_reuse ){
     x =
      // X direction
     (
      (
          local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x - 1, local_index_y, 0) ] +
          local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x + 1, local_index_y, 0) ]
          -
          tijk
      )
      +
      // Y direction
      (
          local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x, local_index_y - 1, 0) ] +
          local_flat[ Index3D( blockDim.x + 2, blockDim.x + 2, local_index_x, local_index_y + 1, 0) ]
          -
          tijk
      )
      +
      // Z direction
      (
          zlow +
          zhigh
          -
          tijk
      )
     ) / dxsq;
    }
    global_data_out[ inner_offset + threadIdx.x + nx * threadIdx.y ] = x;

    __syncthreads();
  }
}

template< class FieldT >
void __host__ divergence_float_gpu(FieldT* f, float dx, float dy, float dz){
    /** Determine dimensions
     *          - grab x, y extents ( we should get the entire memory window ) ?
     *          - decide how many 16x16 blocks are needed to tile the base plane.
     *            This might be a value that is not a multiple of 16.
     **/
    using namespace SpatialOps;
    cudaError err;

    MemoryWindow window = f->window_with_ghost();
    const IntVec& extent = window.extent(); // Get the dimensions of this window.
    unsigned int blockx = ( extent[0] / BLOCK_DIM + ( extent[0] % BLOCK_DIM == 0 ? 0 : 1) );
    unsigned int blocky = ( extent[1] / BLOCK_DIM + ( extent[1] % BLOCK_DIM == 0 ? 0 : 1) );

    unsigned int blockx_op2 = ( ( extent[0] - 2) / (INNER_BLOCK_DIM) +
        ( ( extent[0] - 2 ) % (INNER_BLOCK_DIM) == 0 ? 0 : 1) );
    unsigned int blocky_op2 = ( ( extent[1] - 2) / (INNER_BLOCK_DIM) +
        ( ( extent[1] - 2 ) % (INNER_BLOCK_DIM) == 0 ? 0 : 1) );

    unsigned int blockx_op3 = ( ( extent[0] - 2) / (BLOCK_DIM) +
        ( ( extent[0] - 2 ) % (BLOCK_DIM) == 0 ? 0 : 1) );
    unsigned int blocky_op3 = ( ( extent[1] - 2) / (BLOCK_DIM) +
        ( ( extent[1] - 2 ) % (BLOCK_DIM) == 0 ? 0 : 1) );

    float* h_workspace = (float*)malloc(f->allocated_bytes());
    float* d_workspace;

    cudaMalloc((void**)&d_workspace, f->allocated_bytes() );
    cudaMemset(d_workspace, 0, f->allocated_bytes());
    cudaMemcpy((void*)d_workspace, (void*)f->field_values(GPU_INDEX), f->allocated_bytes(), cudaMemcpyDeviceToDevice );

    dim3 dimBlock( BLOCK_DIM, BLOCK_DIM );
    dim3 dimGrid( blockx, blocky );

    dim3 dimBlock_op2( BLOCK_DIM, BLOCK_DIM );
    dim3 dimGrid_op2( blockx_op2, blocky_op2 );

    dim3 dimBlock_op3( BLOCK_DIM, BLOCK_DIM );
    dim3 dimGrid_op3( blockx_op3, blocky_op3 );

    //std::cout << "Executing with " << blockx_op2 << " X " << blocky_op2 << " block grid" << std::endl;
    //std::cout << "               " << BLOCK_DIM << " X " << BLOCK_DIM << " block threads\n";

    //std::cout << "Extent: " << extent[0] << " " << extent[1] << " " << extent[2] << std::endl;

    if( cudaSuccess != ( err = cudaSetDevice( f->active_device_index() ) ) ){
      throw( std::runtime_error( cudaGetErrorString(err) ) );
    }

    //_div_float_opt1<<<dimGrid, dimBlock, 0, 0>>>( d_workspace,
    _div_float_opt2<<<dimGrid_op2, dimBlock_op2, 0, 0>>>( d_workspace,
    //_div_float_opt3<<<dimGrid_op3, dimBlock_op3, 0, 0>>>( d_workspace,
                                                f->field_values(GPU_INDEX),
                                                dx, dy, dz,
                                                extent[0], extent[1], extent[2] );

    cudaThreadSynchronize();
    cudaMemcpy(h_workspace, f->field_values( GPU_INDEX ), f->allocated_bytes(), cudaMemcpyDeviceToHost);
    cudaFree(d_workspace);

    std::ofstream file;
    file.open("cudaout.txt");
    for(int i = 0; i < extent[0]; ++i){
      for( int j = 0; j < extent[1]; ++j){
        float q = h_workspace[ Index3D(extent[0], extent[1], i, j, 1) ];
        file.precision(8);
        file << std::setw(12) << q << " ";
      }
      file << std::endl;
    }
    file.close();
}

template< class FieldT >
void __host__ divergence_double(FieldT* f, double dx, double dy, double dz ){
 throw;
}

#endif /* CUDASTENCILLIB_H_ */
