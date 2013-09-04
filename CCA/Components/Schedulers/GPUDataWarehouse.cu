/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

namespace Uintah {

HOST_DEVICE void
GPUDataWarehouse::get(GPUGridVariableBase &var, char const* name, int patchID, int maltIndex)
{
#ifdef __CUDA_ARCH__
__shared__ int3 offset;
__shared__ int3 size;
__shared__ void* ptr;
ptr=NULL;
__syncthreads();  //sync before get
int numThreads = blockDim.x*blockDim.y*blockDim.z;
int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
int i=threadID;
char const *s1 = name;
//if (d_debug && threadID == 0 && blockID==0) {
//  printf("device getting %s from DW 0x%x\n", name, (unsigned int)this);
//  printf("size (%d vars)\n", numItems);
//}
while(i<numItems){
  int strmatch=0;
  char *s2 = &(d_varDB[i].label[0]);
  while (!(strmatch = *(unsigned char *) s1 - *(unsigned char *) s2) && *s2) ++s1, ++s2; //strcmp
  if (strmatch==0 && d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==maltIndex){
    offset = d_varDB[i].var_offset;
    size= d_varDB[i].var_size;
    ptr = d_varDB[i].var_ptr;
  }
  i=i+numThreads;
}
//sync before return;
__syncthreads();
if (!ptr){
  printf("device get unknown variable %s on GPUDataWarehouse", name);
}
var.setArray3(offset, size, ptr);
if (d_debug && threadID == 1 && blockID==0) {
  printf("device got %s loc 0x%x ", name, ptr); // printf from GPU only support two variables...
  printf("from GPUDW 0x%x\n", this);
}

#else
// cpu code
if (d_debug) printf("host getting %s from GPUDW 0x%x\n", name, this);
int i= 0;
while(i<numItems){
  if (!strncmp(d_varDB[i].label, name, MAX_NAME) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==maltIndex) {
    var.setArray3(d_varDB[i].var_offset, d_varDB[i].var_size , d_varDB[i].var_ptr);
    break;
  }
  i++;
}
if (i==numItems) {
  printf("host get unknown variable on GPUDataWarehouse");
  exit(-1);
}
#endif
}

HOST_DEVICE void 
GPUDataWarehouse::put(GPUGridVariableBase &var, char const* name, int patchID, int maltIndex, bool overWrite)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  //cpu code 
  if (numItems==MAX_ITEM) {
    printf("out of GPUDataWarehouse space");
    exit(-1);
  }
  int i=numItems;
  numItems++; 
  strncpy(d_varDB[i].label, name, MAX_NAME);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndex = maltIndex;
  var.getArray3(d_varDB[i].var_offset, d_varDB[i].var_size, d_varDB[i].var_ptr);
  if (d_debug) printf("host put %s loc 0x%x into GPU dw 0x%x\n", name, d_varDB[i].var_ptr, device_copy);
  d_dirty=true;
#endif
}

HOST_DEVICE void 
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* name, int patchID, int maltIndex, int3 low, int3 high)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  cudaError_t retVal;
  int3 size=make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset=low;
  var.setArray3(offset, size, NULL);
  void* addr;
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc(&addr, var.getMemSize()) );
  if (d_debug) {
    printf("cuda Malloc for %s, size %ld from (%d,%d,%d) to (%d,%d,%d) at 0x%x\n" , name, var.getMemSize(), 
            low.x, low.y, low.z, high.x, high.y, high.z, addr );
  }
  var.setArray3(offset, size, addr);
  put(var, name, patchID, maltIndex);
#endif
}

HOST_DEVICE bool
GPUDataWarehouse::exist(char const* name, int patchID, int maltIndex)
{
#ifdef __CUDA_ARCH__
  printf("exist() is only for framework code\n");
#else
int i= 0;
while(i<numItems){
  if (!strncmp(d_varDB[i].label, name, MAX_NAME) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==maltIndex) {
    return true;
  }
  i++;
}
return false;
#endif 
}

HOST_DEVICE void
GPUDataWarehouse::init_device()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else
  cudaError_t retVal;
  CUDA_RT_SAFE_CALL( retVal = cudaMalloc((void**)&device_copy, sizeof(GPUDataWarehouse)));
   if(d_debug) printf("Init GW in-device copy %d bytes to 0x%x \n", sizeof(GPUDataWarehouse), device_copy);
  d_dirty=true;
#endif 
}
HOST_DEVICE void
GPUDataWarehouse::syncto_device()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else
  if (!device_copy) {
    printf("error: no device copy\n");
    exit(-1);
  }
  //TODO: only sync the difference
  if (d_dirty){
    cudaError_t retVal;
    CUDA_RT_SAFE_CALL (retVal = cudaMemcpy(device_copy,this, sizeof(GPUDataWarehouse), cudaMemcpyHostToDevice));
  }
  d_dirty=false;
#endif
}

HOST_DEVICE void
GPUDataWarehouse::clear() 
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else

cudaError_t retVal;
for (int i=0; i<numItems; i++) {
  CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));
  if (d_debug) printf("cuda Free for %s at 0x%x\n" , d_varDB[i].label, d_varDB[i].var_ptr );
}

numItems=0;
if (device_copy) {
  CUDA_RT_SAFE_CALL(retVal =  cudaFree(device_copy));
  if(d_debug) printf("Delete GW in-device copy at 0x%x \n",  device_copy);
}
#endif
}


}
