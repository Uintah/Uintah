/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

/* GPU DataWarehouse device & host access*/

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GPUPerPatch.h>

#include <sci_defs/cuda_defs.h>

#ifndef __CUDA_ARCH__
#  include <string.h>
#endif

#include <Core/Util/GPU.h>

namespace Uintah {

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getLevel(const GPUGridVariableBase& var, char const* label, int matlIndx, int levelIndx)
{
  GPUDataWarehouse::dataItem* item = getLevelItem(label, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetLevelError("levelDB GPUDataWarehouse::getLevel(GPUGridVariableBase& var, ...)", label, levelIndx, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, -1 /* matlIndx */, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::put(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */, bool overWrite /* = false */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", label);
#else
  //__________________________________
  // CPU code
  if (d_numItems == MAX_ITEM) {
    printf("Out of GPUDataWarehouse space.  You can try increasing src/CCA/Components/Schedulers/GPUDataWarehouse.h: #define MAX_ITEMS.");
    varDBLock.writeUnlock();  // writeLock() is called from allocateAndPut(). This is the escape clause if things go bad
    exit(-1);
  }

  int i = d_numItems;
  d_numItems++;

  strncpy(d_varDB[i].label, label, MAX_NAME);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndx   = matlIndx;
  d_varDB[i].levelIndx  = levelIndx;

  var.getArray3(d_varDB[i].var_offset, d_varDB[i].var_size, d_varDB[i].var_ptr);

  if (d_debug) {
    printf("GPUDW::put() host put \"%-15s\" (patch: %d) (level: %d) (loc %p) into GPUDW %p on device %d, size [%d,%d,%d]\n", label, patchID, levelIndx, d_varDB[i].var_ptr,
           d_device_copy, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
  }
  d_dirty = true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::putLevel(GPUGridVariableBase& var, char const* label, int matlIndx, int levelIndx, bool overWrite /* = false */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::putLevel( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", label);
#else
  //__________________________________
  // CPU code
  if (d_numLevelItems == MAX_LVITEM) {
    printf("Out of GPUDataWarehouse levelDB space.  You can try increasing src/CCA/Components/Schedulers/GPUDataWarehouse.h: #define MAX_LVITEM.");
    levelDBLock.writeUnlock();  // writeLock() is called from allocateAndPut(). This is the escape clause if things go bad
    exit(-1);
  }

  int i = d_numLevelItems;
  d_numLevelItems++;

  strncpy(d_levelDB[i].label, label, MAX_NAME);
  d_levelDB[i].domainID = -1;
  d_levelDB[i].matlIndx   = matlIndx;
  d_levelDB[i].levelIndx  = levelIndx;

  var.getArray3(d_levelDB[i].var_offset, d_levelDB[i].var_size, d_levelDB[i].var_ptr);

  if (d_debug) {
    printf("GPUDW::putLevel() host put level-var \"%-15s\" (level: %d) (matl: %i) (loc %p) into GPUDW %p on device %d, size [%d,%d,%d]\n", label,  levelIndx, matlIndx, d_levelDB[i].var_ptr,
           d_device_copy, d_device_id, d_levelDB[i].var_size.x, d_levelDB[i].var_size.y, d_levelDB[i].var_size.z);
  }
  d_dirty = true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */, bool overWrite /* = false */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", label);
#else
  //__________________________________
  // CPU code
  if (d_numItems == MAX_ITEM) {
    printf("Out of GPUDataWarehouse space.  You can try increasing src/CCA/Components/Schedulers/GPUDataWarehouse.h: #define MAX_ITEMS");
    exit(-1);
  }

  int i = d_numItems;
  d_numItems++;

  strncpy(d_varDB[i].label, label, MAX_NAME);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndx   = -1;  // matlIndx;
  d_varDB[i].levelIndx  = levelIndx;

  var.getData(d_varDB[i].var_ptr);

  if (d_debug) {
    printf("GPUDW::put(reduction) host put \"%s\" (patch: %d) (level: %d) (loc %p) into GPUDW %p on device %d\n", label, patchID, levelIndx, d_varDB[i].var_ptr, d_device_copy,
           d_device_id);
  }
  d_dirty = true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */, bool overWrite /* = false */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All device memory should be allocated on the CPU with cudaMalloc\n", label);
#else
  //__________________________________
  // CPU code
  if (d_numItems == MAX_ITEM) {
    printf("Out of GPUDataWarehouse space.  You can try increasing src/CCA/Components/Schedulers/GPUDataWarehouse.h: #define MAX_ITEMS");
    exit(-1);
  }

  int i = d_numItems;
  d_numItems++;

  strncpy(d_varDB[i].label, label, MAX_NAME);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndx   = matlIndx;
  d_varDB[i].levelIndx  = levelIndx;

  var.getData(d_varDB[i].var_ptr);

  if (d_debug) {
    printf("GPUDW::put(PerPatchBase) host put \"%s\" (patch: %d) (level: %d) (loc %p) into GPUDW %p on device %d\n", label, patchID, levelIndx, d_varDB[i].var_ptr, d_device_copy,
           d_device_id);
  }
  d_dirty = true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int3 low, int3 high, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device. All device memory should be allocated on the CPU with cudaMalloc\n", label);
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    int3 size = make_int3(high.x - low.x, high.y - low.y, high.z - low.z);
    int3 offset = low;
    void* addr = NULL;

    var.setArray3(offset, size, addr);

    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
    CUDA_RT_SAFE_CALL(retVal = cudaMalloc(&addr, var.getMemSize()));

    if (d_debug && retVal == cudaSuccess) {
      printf("GPUDW::allocateAndPut() cudaMalloc for \"%-15s\", patch: %i, mat: %i size %ld from (%d,%d,%d) to (%d,%d,%d) ", label,
             patchID, matlIndx, var.getMemSize(), low.x, low.y, low.z, high.x, high.y, high.z);
      printf(" at %p on device %d\n", addr, d_device_id);
    }

    var.setArray3(offset, size, addr);
    put(var, label, patchID, matlIndx, levelIndx);
  }
  varDBLock.writeUnlock();

  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase& var, char const* label, int matlIndx, int3 low, int3 high, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device. All device memory should be allocated on the CPU with cudaMalloc\n", label);
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  levelDBLock.writeLock();
  {
    int3 size = make_int3(high.x - low.x, high.y - low.y, high.z - low.z);
    int3 offset = low;
    void* addr = NULL;

    var.setArray3(offset, size, addr);

    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
    CUDA_RT_SAFE_CALL(retVal = cudaMalloc(&addr, var.getMemSize()));

    if (d_debug && retVal == cudaSuccess) {
      printf("GPUDW::allocateAndPut() cudaMalloc for level-var \"%-15s\", L-%i, matl: %i size %ld from (%d,%d,%d) to (%d,%d,%d) ",
             label, levelIndx, matlIndx, var.getMemSize(), low.x, low.y, low.z, high.x, high.y, high.z);
      printf(" at %p on device %d\n", addr, d_device_id);
    }

    var.setArray3(offset, size, addr);
    putLevel(var, label, matlIndx, levelIndx);
  }
  levelDBLock.writeUnlock();

  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    void* addr = NULL;

    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
    CUDA_RT_SAFE_CALL(retVal = cudaMalloc(&addr, var.getMemSize()));

    if (d_debug && retVal == cudaSuccess) {
      printf("GPUDW::allocateAndPut() cudaMalloc for \"%-15s\", L-%i, patch: %i, matl: %i size %ld", label, levelIndx, patchID,
             matlIndx, var.getMemSize());
      printf(" at %p on device %d\n", addr, d_device_id);
    }

    var.setData(addr);
    put(var, label, patchID, matlIndx, levelIndx);
  }
  varDBLock.writeUnlock();

  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    void* addr = NULL;

    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
    CUDA_RT_SAFE_CALL(retVal = cudaMalloc(&addr, var.getMemSize()));

    if (d_debug && retVal == cudaSuccess) {
      printf("GPUDW::allocateAndPut() cudaMalloc for \"%-15s\", L-%i, patch: %i, matl: %i, size %ld", label, levelIndx, patchID,
             matlIndx, var.getMemSize());
      printf(" at %p on device %d\n", addr, d_device_id);
    }

    var.setData(addr);
    put(var, label, patchID, matlIndx, levelIndx);
  }
  varDBLock.writeUnlock();

  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE GPUDataWarehouse::dataItem*
GPUDataWarehouse::getItem(char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__
  __shared__ int index;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadID = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  int i = threadID;
  char const* s1 = label;

  __syncthreads();
  index = -1;

  if (d_debug && threadID == 0 && blockID == 0) {
    printf("GPUDW::getItem() varDB item \"%-15s\" L-%i Patch: %i, Matl: %i from GPUDW %p, size (%d vars)", label, levelIndx, patchID, matlIndx, this, d_numItems);
    printf("  Available varDB labels: %d\n", MAX_ITEM - d_numItems);
  }

  //sync before get
  __syncthreads();

  while (i < d_numItems) {
    int strmatch = 0;
    char* s2 = &(d_varDB[i].label[0]);
    while (!(strmatch = *(unsigned char *)s1 - *(unsigned char *)s2) && *s2) {  //strcmp
      ++s1, ++s2;
    }

    if (strmatch == 0 && d_varDB[i].domainID == patchID && d_varDB[i].matlIndx == matlIndx && d_varDB[i].levelIndx == levelIndx) {
      index = i;
    }
    i = i + numThreads;
  }
  //sync before return;
  __syncthreads();

  if (index == -1) {
    return NULL;
  }
  else {
    return &d_varDB[index];
  }
#else
  //__________________________________
  //  CPU code
  int i = 0;
  varDBLock.readLock();
  {
    while (i < d_numItems) {
      if (!strncmp(d_varDB[i].label, label, MAX_NAME) && d_varDB[i].domainID == patchID && d_varDB[i].matlIndx == matlIndx
          && d_varDB[i].levelIndx == levelIndx) {
        break;
      }
      i++;
    }

    if (i == d_numItems) {
      printf("ERROR:\nGPUDataWarehouse::get( %s ) host get unknown variable from GPUDataWarehouse", label);
      varDBLock.readUnlock();
      exit(-1);
    }

    if (d_debug) {
      printf("GPUDW::getItem() host got \"%-15s\" loc %p from GPUDW %p on device %u\n", label, d_varDB[i].var_ptr, d_device_copy,
             d_device_id);
    }
  }
  varDBLock.readUnlock();

  return &d_varDB[i];
#endif
}

//______________________________________________________________________
//
HOST_DEVICE GPUDataWarehouse::dataItem*
GPUDataWarehouse::getLevelItem(char const* label, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  __shared__ int index;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadID = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  int i = threadID;
  char const* s1 = label;

  __syncthreads();
  index = -1;

  if (d_debug && threadID == 0 && blockID == 0) {
    printf("GPUDW::getLevelItem() \"%-13s\" L-%i from GPUDW %p, size (%d vars)", label, levelIndx, this, d_numLevelItems);
    printf("  Available levelDB labels: %d\n", MAX_LVITEM - d_numLevelItems);
  }

  //sync before get
  __syncthreads();

  while (i < d_numLevelItems) {
    int strmatch = 0;
    char* s2 = &(d_levelDB[i].label[0]);
    while (!(strmatch = *(unsigned char *)s1 - *(unsigned char *)s2) && *s2) {  //strcmp
      ++s1, ++s2;
    }

    if (strmatch == 0 &&  d_levelDB[i].levelIndx == levelIndx && d_levelDB[i].matlIndx == matlIndx) {
      index = i;
    }
    i = i + numThreads;
  }
  //sync before return;
  __syncthreads();

  if (index == -1) {
    return NULL;
  }
  else {
    return &d_levelDB[index];
  }
#else
  //__________________________________
  //  CPU code
  int i = 0;
  levelDBLock.readLock();
  {
    while (i < d_numLevelItems) {
      if (!strncmp(d_levelDB[i].label, label, MAX_NAME) && d_levelDB[i].levelIndx == levelIndx && d_levelDB[i].matlIndx == matlIndx) {
        break;
      }
      i++;
    }

    if (i == d_numLevelItems) {
      printf("ERROR:\nGPUDataWarehouse::getLevelItem( %s ) host get unknown variable from GPUDataWarehouse", label);
      levelDBLock.readUnlock();
      exit(-1);
    }

    if (d_debug) {
      printf("GPUDW::getLevelItem() host got \"%-15s\" L-%i matl: %i loc %p from GPUDW %p on device %u\n", label, levelIndx, matlIndx,
             d_levelDB[i].var_ptr, d_device_copy, d_device_id);
    }
  }
  levelDBLock.readUnlock();

  return &d_levelDB[i];
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
#ifdef __CUDA_ARCH__
  printf("exist() is only for framework code\n");
  return false;
#else
  //__________________________________
  //  CPU code
  varDBLock.readLock();
  {
    int i = 0;
    while (i < d_numItems) {
      if (!strncmp(d_varDB[i].label, label, MAX_NAME) && d_varDB[i].domainID == patchID && d_varDB[i].matlIndx == matlIndx
          && d_varDB[i].levelIndx == levelIndx) {
        varDBLock.readUnlock();
        return true;
      }
      i++;
    }
  }
  varDBLock.readUnlock();
  return false;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::remove(char const* label, int patchID, int matlIndx, int levelIndx /* = 0 */)
{
  // I think this method may be dicey in general. Really only the infrastructure should call it.

#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    int i = 0;
    while (i < d_numItems) {
      if (!strncmp(d_varDB[i].label, label, MAX_NAME) && d_varDB[i].domainID == patchID && d_varDB[i].matlIndx == matlIndx
          && d_varDB[i].levelIndx == levelIndx) {
        CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));

        if (d_debug && retVal == cudaSuccess) {
          printf("GPUDW::remove() cuda free for \"%-15s\" L-%i, patch: %i, matl: %i at %p on device %d\n", d_varDB[i].label,
                 levelIndx, patchID, matlIndx, d_varDB[i].var_ptr, d_device_id);
        }

        d_varDB[i].label[0] = 0;  // leave a hole in the flat array, not deleted.
        d_dirty = true;
      }
      i++;
    }
  }
  varDBLock.writeUnlock();
  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::init_device(int id)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  d_device_id = id;
  CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
  CUDA_RT_SAFE_CALL(retVal = cudaMalloc((void** )&d_device_copy, sizeof(GPUDataWarehouse)));

  if (d_debug && retVal == cudaSuccess) {
    printf("Init GPUDW on-device copy %lu bytes to %p on device %d\n", sizeof(GPUDataWarehouse), d_device_copy, d_device_id);
  }

  d_dirty = true;
  return retVal;
#endif 
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::syncto_device()
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
  return false;
#else
  //__________________________________
  //  CPU code
  if (!d_device_copy) {
    printf("ERROR:\nGPUDataWarehouse::syncto_device()\nNo device copy\n");
    exit(-1);
  }
  // TODO: only sync the difference
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    if (d_dirty) {
      CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
      CUDA_RT_SAFE_CALL(retVal = cudaMemcpy(d_device_copy, this, sizeof(GPUDataWarehouse), cudaMemcpyHostToDevice));

      if (d_debug && retVal == cudaSuccess) {
        printf("GPUDW::::syncto_device() sync GPUDW %p to device %d\n", d_device_copy, d_device_id);
      }
    }
    d_dirty = false;
  }
  varDBLock.writeUnlock();

  return retVal;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::clear()
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
  return false;
#else
  //__________________________________
  //  CPU code
  cudaError_t retVal;
  varDBLock.writeLock();
  {
    CUDA_RT_SAFE_CALL(retVal = cudaSetDevice(d_device_id));
    for (int i = 0; i < d_numItems; i++) {
      if (d_varDB[i].label[0] != 0) {
        CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));

        if (d_debug && retVal == cudaSuccess) {
          printf("GPUDW::clear() cudaFree for \"%-15s\" at %p on device %d\n", d_varDB[i].label, d_varDB[i].var_ptr, d_device_id);
        }
      }
    }

    d_numItems = 0;
    if (d_device_copy) {
      CUDA_RT_SAFE_CALL(retVal = cudaFree(d_device_copy));
      if (d_debug && retVal == cudaSuccess) {
        printf("GPUDW::clear() Delete GPUDW on-device copy at %p on device %d \n", d_device_copy, d_device_id);
      }
    }
  }
  varDBLock.writeUnlock();

  return retVal;
#endif
}
//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printGetLevelError(const char* msg, char const* label, int levelIndx, int matlIndx)
{
#ifdef __CUDA_ARCH__
  __syncthreads();
  if( isThread0() ){
    for (int i = 0; i < d_numLevelItems; i++) {
      printf("   Available labels(%i): \"%s\"\n", d_numLevelItems, d_levelDB[i].label);
    }
    __syncthreads();
    printf("  ERROR: %s( \"%s\", levelIndx: %i, matl: %i )  unknown variable\n\n", msg,  label, levelIndx, matlIndx);
    printThread();
    printBlock();
    assert(0);            // FIXME!!! the code doesn't exit clean when we hit this assert.  This could be costly on big runs that land here.
  }
#else
  //__________________________________
  //  CPU code
  printf("\t ERROR: %s( \"%s\", levelIndx: %i, matl: %i )  unknown variable\n", msg, label, levelIndx, matlIndx);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printGetError(const char* msg, char const* label, int levelIndx, int patchID, int matlIndx)
{
#ifdef __CUDA_ARCH__
  __syncthreads();
  if( isThread0() ){
    for (int i = 0; i < d_numItems; i++) {
      printf("   Available labels(%i): \"%s\"\n", d_numItems, d_varDB[i].label);
    }
    __syncthreads();
    printf("  ERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i )  unknown variable\n\n", msg,  label, levelIndx, patchID, matlIndx);
    printThread();
    printBlock();
    assert(0);            // FIXME!!! the code doesn't exit clean when we hit this assert.  This could be costly on big runs that land here.
  }
#else
  //__________________________________
  //  CPU code
  printf("\t ERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i )  unknown variable\n", msg, label, levelIndx, patchID, matlIndx);
#endif
}

} // end namespace Uintah
