/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
/* GPU DataWarehouse device&host access*/

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#ifndef __CUDA_ARCH__
#include <string.h>
#endif
//#include <Core/Util/GPU.h>

// This belongs in GPU.h
__device__ bool isThread0_Blk0(){
  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  
  bool test = (blockID == 0 && threadID == 0);
  return test;
}

namespace Uintah {

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUGridVariableBase& var, char const* name, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(name, patchID, matlIndex);
  if (item){
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }else{    
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
    
    int i=threadID;
    while(i<d_numItems){
      printf( "   Available labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlIndex);
      assert(0);
    }

#else
    printf("\t ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlIndex);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, char const* name, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item=getItem(name, patchID, matlIndex);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }else{
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
    
    
    int i=threadID;
    while(i<d_numItems){
      printf( "  Available Labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlIndex);
      assert(0);
    }

#else
    printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlIndex);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::put(GPUGridVariableBase &var, char const* name, int patchID, int matlIndex, bool overWrite)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n",name);
#else
  
  //__________________________________
  //cpu code 
  if (d_numItems==MAX_ITEM) {
    printf("out of GPUDataWarehouse space");
    exit(-1);
  }
  
  int i=d_numItems;
  d_numItems++; 
  strncpy(d_varDB[i].label, name, MAX_NAME);
  d_varDB[i].domainID  = patchID;
  d_varDB[i].matlIndex = matlIndex;
  var.getArray3(d_varDB[i].var_offset, d_varDB[i].var_size, d_varDB[i].var_ptr);
  
  if (d_debug){
    printf("host put \"%s\" (patch: %d) loc %p into GPUDW %p on device %d, size [%d,%d,%d]\n", name, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
  }
  d_dirty=true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* name, int patchID, int matlIndex, int3 low, int3 high)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n",name);
#else
  //__________________________________
  //  cpu code
  cudaError_t retVal;
  int3 size   = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;
  void* addr  = NULL;
  
  var.setArray3(offset, size, addr);
  CUDA_RT_SAFE_CALL( retVal = cudaSetDevice(d_device_id) );
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc(&addr, var.getMemSize()) );
  
  if (d_debug && retVal == cudaSuccess) {
    printf("cudaMalloc for \"%s\", size %ld from (%d,%d,%d) to (%d,%d,%d) ", name, var.getMemSize(),
            low.x, low.y, low.z, high.x, high.y, high.z);
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  
  var.setArray3(offset, size, addr);
  put(var, name, patchID, matlIndex);

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUParticleVariableBase& var, char const* name, int patchID, int matlID)
{
  GPUDataWarehouse::dataItem* item = getItem(name, patchID, matlID);
  if (item){
    var.setData(item->num_elems, item->var_ptr);
  }else{
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

    int i=threadID;
    while(i<d_numItems){
      printf( "   Available labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlID);
      assert(0);
    }

#else
    printf("\t ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlID);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUParticleVariableBase& var, char const* name, int patchID, int matlID)
{
  GPUDataWarehouse::dataItem* item = getItem(name, patchID, -1 /* matlID */);
  if (item) {
    var.setData(item->num_elems, item->var_ptr);
  } else {
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;


    int i=threadID;
    while(i<d_numItems){
      printf( "  Available Labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlID);
      assert(0);
    }

#else
    printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlID);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUParticleVariableBase& var, char const* name, int patchID, int matlIndex, bool overWrite)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", name);
#else

  //__________________________________
  //cpu code
  if (d_numItems==MAX_ITEM) {
    printf("out of GPUDataWarehouse space");
    exit(-1);
  }

  int i=d_numItems;
  d_numItems++;
  strncpy(d_varDB[i].label, name, MAX_NAME);
  d_varDB[i].domainID  = patchID;
  d_varDB[i].matlIndex = -1; // matlIndex;

  var.getData(d_varDB[i].num_elems, d_varDB[i].var_ptr);

  if (d_debug){
    printf("host put \"%s\" (patch: %d) loc %p into GPUDW %p on device %d, size %lu\n", name, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].num_elems);
  }
  d_dirty=true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUParticleVariableBase& var, char const* name, int patchID, int matlID, size_t numElems)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",name);
#else
  //__________________________________
  //  cpu code
  cudaError_t retVal;
  size_t numVals = numElems;
  void* addr  = NULL;

  CUDA_RT_SAFE_CALL( retVal = cudaSetDevice(d_device_id) );
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc(&addr, var.getMemSize()) );

  if (d_debug && retVal == cudaSuccess) {
    printf("cudaMalloc for \"%s\", size %ld", name, var.getMemSize());
    printf(" at %p on device %d\n", addr, d_device_id);
  }

  var.setData(numVals, addr);
  put(var, name, patchID, matlID);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUReductionVariableBase& var, char const* name, int patchID, int matlID)
{
  GPUDataWarehouse::dataItem* item = getItem(name, patchID, matlID);
  if (item){
    var.setData(item->num_elems, item->var_ptr);
  }else{
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

    int i=threadID;
    while(i<d_numItems){
      printf( "   Available labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlID);
      assert(0);
    }

#else
    printf("\t ERROR: GPUDataWarehouse::get( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlID);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, char const* name, int patchID, int matlID)
{
  GPUDataWarehouse::dataItem* item = getItem(name, patchID, -1 /* matlID */);
  if (item) {
    var.setData(item->num_elems, item->var_ptr);
  } else {
#ifdef __CUDA_ARCH__
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;


    int i=threadID;
    while(i<d_numItems){
      printf( "  Available Labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", name, patchID, matlID);
      assert(0);
    }

#else
    printf("  ERROR: GPUDataWarehouse::getModifiable( \"%s\", patchID: %i, matl: %i )  unknown variable\n", name, patchID, matlID);
#endif
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUReductionVariableBase& var, char const* name, int patchID, int matlIndex, bool overWrite)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", name);
#else

  //__________________________________
  //cpu code
  if (d_numItems==MAX_ITEM) {
    printf("out of GPUDataWarehouse space");
    exit(-1);
  }

  int i=d_numItems;
  d_numItems++;
  strncpy(d_varDB[i].label, name, MAX_NAME);
  d_varDB[i].domainID  = patchID;
  d_varDB[i].matlIndex = -1; // matlIndex;

  var.getData(d_varDB[i].num_elems, d_varDB[i].var_ptr);

  if (d_debug){
    printf("host put \"%s\" (patch: %d) loc %p into GPUDW %p on device %d, size %lu\n", name, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].num_elems);
  }
  d_dirty=true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* name, int patchID, int matlID, int numElems)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",name);
#else
  //__________________________________
  //  cpu code
  cudaError_t retVal;
  size_t numVals = numElems;
  void* addr  = NULL;

  CUDA_RT_SAFE_CALL( retVal = cudaSetDevice(d_device_id) );
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc(&addr, var.getMemSize()) );

  if (d_debug && retVal == cudaSuccess) {
    printf("cudaMalloc for \"%s\", size %ld", name, var.getMemSize());
    printf(" at %p on device %d\n", addr, d_device_id);
  }

  var.setData(numVals, addr);
  put(var, name, patchID, matlID);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE GPUDataWarehouse::dataItem*
GPUDataWarehouse::getItem(char const* name, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  __shared__ int index;
  int numThreads = blockDim.x*blockDim.y*blockDim.z;
  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  int i=threadID;
  char const* s1 = name;
  __syncthreads();
  index = -1;

  if (d_debug && threadID == 0 && blockID==0) {
    printf("device getting item \"%s\" from GPUDW %p", name, this);
    printf("size (%d vars)\n Available labels:", d_numItems);
  }

  //sync before get
  __syncthreads();

  while(i<d_numItems){
    int strmatch=0;
    char* s2 = &(d_varDB[i].label[0]);
    while (!(strmatch = *(unsigned char *) s1 - *(unsigned char *) s2) && *s2) { //strcmp
      ++s1, ++s2;
    }

    if (strmatch==0 && d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==matlIndex){
      index = i;
    }
    i=i+numThreads;
  }
  //sync before return;
  __syncthreads();

  if (index==-1) {
    return NULL;
  } else {
    return &d_varDB[index];
  }
#else
  //__________________________________
  // cpu code
  int i= 0;
  while(i<d_numItems){
    if (!strncmp(d_varDB[i].label, name, MAX_NAME) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==matlIndex) {
      break;
    }
    i++;
  }

  if (i==d_numItems) {
    printf("ERROR:\nGPUDataWarehouse::get( %s ) host get unknown variable from GPUDataWarehouse", name);
    exit(-1);
  }

  if (d_debug){
    printf("host got \"%s\" loc %p from GPUDW %p on device %u\n", name, d_varDB[i].var_ptr, d_device_copy, d_device_id);
  }
  return &d_varDB[i];
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* name, int patchID, int matlID)
{
#ifdef __CUDA_ARCH__
  printf("exist() is only for framework code\n");
#else
  //__________________________________
  //  cpu code
  int i= 0;
  while(i<d_numItems){
    if (!strncmp(d_varDB[i].label, name, MAX_NAME) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==matlID) {
      return true;
    }
    i++;
  }
#endif 
return false;
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::remove(char const* name, int patchID, int matlID)
{
#ifdef __CUDA_ARCH__
  printf("remove() is only for framework code\n");
#else
  int i= 0;
  while(i<d_numItems){
    if (!strncmp(d_varDB[i].label, name, MAX_NAME) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==matlID) {
      cudaError_t retVal;
      CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));

      if (d_debug){
        printf("cuda Free for \"%s\" at %p on device %d\n" , d_varDB[i].label, d_varDB[i].var_ptr, d_device_id );
      }

      d_varDB[i].label[0] = 0; // leave a hole in the flat array, not deleted.
      d_dirty=true;
    }
    i++;
  }
#endif 
  return false;
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::init_device(int id)
{
#ifdef __CUDA_ARCH__
  // no meaning in device method
#else
  cudaError_t retVal;
  d_device_id = id;
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice( d_device_id ));
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc((void**)&d_device_copy, sizeof(GPUDataWarehouse)));
  
  if(d_debug){
    printf("Init GPUDW on-device copy %lu bytes to %p on device %d\n", sizeof(GPUDataWarehouse), d_device_copy, d_device_id);
  }
  
  d_dirty=true;
#endif 
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::syncto_device()
{
#ifdef __CUDA_ARCH__
  // no meaning in device method
#else
  if (!d_device_copy) {
    printf("ERROR:\nGPUDataWarehouse::syncto_device()\nNo device copy\n");
    exit(-1);
  }
  // TODO: only sync the difference
  if (d_dirty){
    cudaError_t retVal;
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice( d_device_id ));
    CUDA_RT_SAFE_CALL (retVal = cudaMemcpy( d_device_copy,this, sizeof(GPUDataWarehouse), cudaMemcpyHostToDevice));
    
    if (d_debug) {
      printf("sync GPUDW %p to device %d\n", d_device_copy, d_device_id);
    }
  }
  d_dirty=false;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::clear() 
{
#ifdef __CUDA_ARCH__
  // no meaning in device method
#else

  cudaError_t retVal;
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice( d_device_id ));
  for (int i=0; i<d_numItems; i++) {
    if (d_varDB[i].label[0] != 0){
      CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));
      
      if (d_debug){
        printf("cudaFree for \"%s\" at %p on device %d\n", d_varDB[i].label, d_varDB[i].var_ptr, d_device_id );
      }
    }
  }

  d_numItems=0;
  if ( d_device_copy ) {
    CUDA_RT_SAFE_CALL(retVal =  cudaFree( d_device_copy ));
    if(d_debug){
      printf("Delete GPUDW on-device copy at %p on device %d \n",  d_device_copy, d_device_id);
    }
  }
#endif
}
//______________________________________________________________________
//


}
