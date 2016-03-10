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

/* GPU DataWarehouse device & host access*/

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GPUPerPatch.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <sci_defs/cuda_defs.h>

#include <Core/Thread/Thread.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>

#ifndef __CUDA_ARCH__
#  include <string.h>
#include <string>
using namespace std;
#endif

#include <Core/Util/GPU.h>

extern DebugStream gpu_stats;

extern SCIRun::Mutex cerrLock;



namespace Uintah {

std::multimap<GPUDataWarehouse::gpuMemoryPoolItem, GPUDataWarehouse::gpuMemoryData>* GPUDataWarehouse::gpuMemoryPool = new std::multimap<GPUDataWarehouse::gpuMemoryPoolItem, GPUDataWarehouse::gpuMemoryData>;
//TODO, should be deallocated?
SCIRun::CrowdMonitor * GPUDataWarehouse::gpuPoolLock = new SCIRun::CrowdMonitor("gpu pool lock");
//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setArray3(vp.device_offset, vp.device_size, vp.device_ptr);
  }
  else {
    printf("I'm GPUDW %s at %p \n", _internalName, this);
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::stagingVarExists(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
#ifdef __CUDA_ARCH__
  //device code
  printError("This method not defined for the device.", "stagingVarExists", label, patchID, matlIndx, levelIndx);
  return false;
#else
  //host code
  varLock->readLock();
  bool retval = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
    retval = (staging_it != it->second.stagingVars.end());
  }
  varLock->readUnlock();
  return retval;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getStagingVar(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
#ifdef __CUDA_ARCH__
  //device code
  printError("This method not defined for the device.", "getStagingVar", label, patchID, matlIndx, levelIndx);
#else
  //host code
  varLock->readLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
    if (staging_it != it->second.stagingVars.end()) {
      var.setArray3(offset, size, staging_it->second.device_ptr);

    } else {
      printf("GPUDataWarehouse::getStagingVar() - Didn't find a staging variable from the device for label %s patch %d matl %d level %d offset (%d, %d, %d) size (%d, %d, %d).",
          label, patchID, matlIndx, levelIndx,
          offset.x, offset.y, offset.z, size.x, size.y, size.z);
      exit(-1);
    }
  } else {
    printError("Didn't find a staging variable from the device.", "getStagingVar", label, patchID, matlIndx, levelIndx);
  }
  varLock->readUnlock();
#endif
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getLevel(const GPUGridVariableBase& var, char const* label, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  get(var, label, -99999999, matlIndx, levelIndx);
#else
  //host code
  get(var, label, -99999999, matlIndx, levelIndx);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setData(vp.device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setData(vp.device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setArray3(vp.device_offset, vp.device_size, vp.device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID,  matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setData(vp.device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  //host code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->operator[](lpml);
    var.setData(vp.device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
#endif
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUGridVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, bool staging,
                      GhostType gtype, int numGhostCells, void* host_ptr)
{

  varLock->writeLock();

  int3 var_offset;        // offset
  int3 var_size;          // dimensions of GPUGridVariable
  void* var_ptr;           // raw pointer to the memory

  var.getArray3(var_offset, var_size, var_ptr);

  //See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  //sanity checks
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side GPU DW without it first existing in the internal database.\n");
    exit(-1);
  } else if (staging) {
    stagingVar sv;
    sv.device_offset = var_offset;
    sv.device_size = var_size;
    staging_it = iter->second.stagingVars.find(sv);
    if (staging_it == iter->second.stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side GPU DW without this staging var first existing in the internal database.\n");
      exit(-1);
    }
  }

  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
        << " GPUDataWarehouse::put( " << label << " ) - "
        << " Attempting to put a variable in the host-side varPointers map for label " << label
        << " patch " << patchID
        << " matl " << matlIndx
        << " level " << levelIndx;
        if (staging) {
          gpu_stats << " staging: true";
        } else {
          gpu_stats << " staging: false";
        }
        gpu_stats << " at device address " << var_ptr
        << " with status codes ";
        if (!staging) {
          gpu_stats << getDisplayableStatusCodes(iter->second.atomicStatusInGpuMemory);
        } else {
          gpu_stats << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory);
        }
        gpu_stats << " datatype size " << sizeOfDataType
        << " on device " << d_device_id
        << " into GPUDW at " << std::hex << this << std::dec
        << " with description " << _internalName
        << " current varPointers size is: " << varPointers->size()
        << " low (" << var_offset.x << ", " << var_offset.y << ", " << var_offset.z << ") "
        << endl;
    }
    cerrLock.unlock();
  }

  if (staging == false) {

    iter->second.varDB_index = -1;
    iter->second.device_ptr = var_ptr;
    iter->second.device_offset =  var_offset;
    iter->second.device_size = var_size;
    iter->second.sizeOfDataType = sizeOfDataType;
    iter->second.gtype = gtype;
    iter->second.numGhostCells = numGhostCells;
    iter->second.host_contiguousArrayPtr = host_ptr;
    iter->second.atomicStatusInHostMemory = UNKNOWN;

    //previously set, do not set here


    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put a regular non-staging variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " at device address " << var_ptr
            << " with datatype size " << iter->second.sizeOfDataType
            << " with status codes " << getDisplayableStatusCodes(iter->second.atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << " current varPointers size is: " << varPointers->size()
            << endl;
      }
      cerrLock.unlock();
    }

  } else { // if (staging == true)


    staging_it->second.device_ptr = var_ptr;
    staging_it->second.host_contiguousArrayPtr = host_ptr;
    staging_it->second.varDB_index = -1;
    staging_it->second.atomicStatusInHostMemory = UNKNOWN;

    //Update the non-staging var's sizeOfDataType.  The staging var uses this number.
    //It's possible that a staging var can exist and an empty placeholder non-staging var also exist,
    //if so, then then empty placeholder non-staging var won't have correct data type size.
    //So we grab it here.
    iter->second.sizeOfDataType = sizeOfDataType;

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put a staging variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " with offset (" << var_offset.x << ", " << var_offset.y << ", " << var_offset.z << ")"
            << " and size (" << var_size.x << ", " << var_size.y << ", " << var_size.z << ")"
            << " at device address " << var_ptr
            << " with datatype size " << iter->second.sizeOfDataType
            << " with status codes " << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory)
            << " on device " << d_device_id

            << " into GPUDW at " << std::hex << this << std::dec
            << endl;
      }
      cerrLock.unlock();
    }

  }

  varLock->writeUnlock();

}


//______________________________________________________________________
//
//This method puts an empty placeholder entry into the GPUDW database and marks it as unallocated
__host__ void
GPUDataWarehouse::putUnallocatedIfNotExists(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 offset, int3 size)
{

  varLock->writeLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  //if (!staging) {
  //If it's a normal non-staging variable, check if doesn't exist.  If so, add an "unallocated" entry.
  //If it's a staging variable, then still check if the non-staging part exists.  A staging must exist within a non-staging variable.
  //A scenario where this can get a staging variable without a non-staging variable is receiving data from neighbor nodes.
  //For example, suppose node A has patch 0, and node B has patch 1, and A's patch 0 needs ghost cells from B's patch 1.  Node A will
  //receive those ghost cells, but they will be marked as belonging to patch 1.  Since A doesn't have the regular non-staging var
  //for patch 1, we make an empty placeholder for patch 1 so A can have a staging var to hold the ghost cell for patch 1.
  if ( it == varPointers->end()) {

    allVarPointersInfo vp;

    vp.varDB_index = -1;
    vp.device_ptr = NULL;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = UNALLOCATED;
    vp.host_contiguousArrayPtr = NULL;
    vp.sizeOfDataType = 0;

    std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator, bool> ret = varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    if (!ret.second) {
      printf("ERROR:\nGPUDataWarehouse::putUnallocatedIfNotExists( ) Failure inserting into varPointers map.\n");
      varLock->writeUnlock();
      exit(-1);
    }
    it = ret.first;
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::putUnallocatedIfNotExists( " << label << " ) - "
            << " Put an unallocated non-staging variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << endl;
      }
      cerrLock.unlock();
    }

  }
  //} else { //staging = true
  if (staging) {
    std::map<stagingVar, stagingVarInfo>::iterator staging_it;

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.stagingVars.find(sv);
    if (staging_it == it->second.stagingVars.end()){
      stagingVarInfo svi;
      svi.varDB_index = -1;
      svi.device_ptr = NULL;
      svi.host_contiguousArrayPtr = NULL;
      svi.atomicStatusInHostMemory = UNKNOWN;
      svi.atomicStatusInGpuMemory = UNALLOCATED;

      std::pair<stagingVar, stagingVarInfo> p = make_pair( sv, svi );

      it->second.stagingVars.insert( p );

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUDataWarehouse::putUnallocatedIfNotExists( " << label << " ) - "
              << " Put an unallocated staging variable in the host-side varPointers map for label " << label
              << " patch " << patchID
              << " matl " << matlIndx
              << " level " << levelIndx
              << " offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
              << " size (" << size.x << ", " << size.y << ", " << size.z << ")"
              << " on device " << d_device_id
              << " into GPUDW at " << std::hex << this << std::dec
              << " with description " << _internalName
              << endl;
        }
        cerrLock.unlock();
      }
    }
  }

  varLock->writeUnlock();

}

//______________________________________________________________________
//
__host__ void*
GPUDataWarehouse::allocateCudaSpaceFromPool(int device_id, size_t memSize) {

  gpuPoolLock->writeLock();

  void * addr = NULL;
  bool claimedAnItem = false;

  gpuMemoryPoolItem gpuItem(device_id, memSize);

  std::pair <std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator,
             std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator> ret;

  ret = gpuMemoryPool->equal_range(gpuItem);
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator gpuPoolIter = ret.first;
  while (!claimedAnItem && gpuPoolIter != ret.second) {
    if (gpuPoolIter->second.status == 0) {
      //claim this one
      addr = gpuPoolIter->second.ptr;
      gpuPoolIter->second.status = 1;
      claimedAnItem = true;
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUDataWarehouse::allocateCudaSpaceFromPool() -"
              << " reusing space starting at " << gpuPoolIter->second.ptr
              << " on device " << d_device_id
              << " with size " << memSize
              << " from the GPU memory pool"
              << endl;
        }
        cerrLock.unlock();
      }

    } else {
      ++gpuPoolIter;
    }
  }
  //No open spot in the pool, go ahead and allocate it.
  if (!claimedAnItem) {
    CUDA_RT_SAFE_CALL( cudaMalloc(&addr, memSize) );
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateCudaSpaceFromPool() -"
            << " allocated GPU space starting at " << addr
            << " on device " << d_device_id
            << " with size " << memSize
            << endl;
      }
      cerrLock.unlock();
    }
    gpuMemoryData gmd;
    gmd.status = 1;
    gmd.timestep = 99999999; //Fix me
    gmd.ptr = addr;
    gpuPoolIter = gpuMemoryPool->insert(std::pair<gpuMemoryPoolItem, gpuMemoryData>(gpuItem,gmd));
  }
  gpuPoolLock->writeUnlock();
  return addr;
}


//______________________________________________________________________
//
__host__ bool GPUDataWarehouse::freeCudaSpaceFromPool(int device_id, size_t memSize, void* addr){
  gpuPoolLock->writeLock();
  if (memSize == 0) {
    printf("ERROR:\nGPUDataWarehouse::freeCudaSpaceFromPool(), requesting to free from pool memory of size zero at address %p\n", addr);
    gpuPoolLock->writeUnlock();
    return false;
  }
  bool foundItem = false;


  /*//For debugging, shows everything in the pool
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator end;
  for (std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator it = gpuMemoryPool->begin();
       it !=  gpuMemoryPool->end();
       ++it) {
    gpu_stats << "device: " << it->first.device_id
              << " deviceSize: " << it->first.deviceSize << " - "
              << " status: " << it->second.status
              << " timestep: " << it->second.timestep
              << " ptr: " << it->second.ptr
              << endl;
  }
  gpu_stats << endl;
  */
  gpuMemoryPoolItem gpuItem(d_device_id, memSize);

  std::pair <std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator,
             std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator> ret;

  ret = gpuMemoryPool->equal_range(gpuItem);
  std::multimap<gpuMemoryPoolItem, gpuMemoryData>::iterator gpuPoolIter = ret.first;

  while (!foundItem && gpuPoolIter != ret.second) {
    if (gpuPoolIter->second.ptr == addr) {
      //Found it.
      //Mark it as reusable
      gpuPoolIter->second.status = 0;
      foundItem = true;

      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUDataWarehouse::freeCudaSpaceFromPool() -"
              << " space starting at " << addr
              << " on device " << d_device_id
              << " with size " << memSize
              << " marked for reuse in the GPU memory pool"
              << endl;
        }
        cerrLock.unlock();
      }

    } else {
      ++gpuPoolIter;
    }
  }
  gpuPoolLock->writeUnlock();
  if (foundItem == false) {
    cout << "break here" << endl;
  }
  return foundItem;

}
//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype, int numGhostCells)
{

//Allocate space on the GPU and declare a variable onto the GPU.
//This method does NOT stage everything in a big array.

  //Check if it exists prior to allocating memory for it.
  //If it has already been allocated, just use that.
  //If it hasn't, this is lock free and the first thread to request allocating gets to allocate
  //If another thread sees that allocating is in process, it loops and waits until the allocation complete.

  bool allocationNeeded = false;
  int3 size = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread() << " Calling putUnallocatedIfNotExists() for " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx;
          if (staging) {
            gpu_stats << " staging: true";
          } else {
            gpu_stats << " staging: false";
          }
          gpu_stats<< " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
          << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName << endl;
    }
    cerrLock.unlock();
  }
  //This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, staging, offset, size);

  varLock->readLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  if (staging) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.stagingVars.find(sv);
  }

  varLock->readUnlock();

  //Locking not needed here on out in this method.  STL maps ensure that iterators point to correct values
  //even if other threads add nodes.  We just can't remove values, but that shouldn't ever happen.

  //This prepares the var with the offset and size.  Any possible allocation will come later.
  //If it needs to go into the database, that will also come later
  void* addr = NULL;
  var.setArray3(offset, size, addr);

  //Now see if we allocate the variable or use a previous existing allocation.
  if (staging == false) {

    //See if someone has stated they are allocating it
    allocationNeeded = testAndSetAllocating(it->second.atomicStatusInGpuMemory);
    if (!allocationNeeded) {
    //Someone else is allocating it or it has already been allocated.
      if (it->second.device_offset.x == low.x
          && it->second.device_offset.y == low.y
          && it->second.device_offset.z == low.z
          && it->second.device_size.x == size.x
          && it->second.device_size.y == size.y
          && it->second.device_size.z == size.z) {

         //Space for this var already exists.  Use that and return.
         if (gpu_stats.active()) {
           cerrLock.lock();
           {
             gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
                << " This non-staging/regular variable already exists.  No need to allocate another.  GPUDW has a variable for label " << label
                << " patch " << patchID
                << " matl " << matlIndx
                << " level " << levelIndx
                << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
                << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
                << " on device " << d_device_id
                << " with data pointer " << it->second.device_ptr
                << " with status codes " << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory)
                << " into GPUDW at " << std::hex << this << std::dec
                << endl;
           }
           cerrLock.unlock();
         }
         //We need the pointer.  We can't move on until we get the pointer.
         //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
         //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
         bool allocated = false;
         while (!allocated) {

           allocated = checkAllocated(it->second.atomicStatusInGpuMemory);
         }
         //Have this var use the existing memory address.
         var.setArray3(it->second.device_offset, it->second.device_size, it->second.device_ptr);
      } else {
        printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  Variable in database but of the wrong size.  This shouldn't ever happen. this needs low (%d, %d, %d,) and size (%d, %d, %d), but in the database it is low (%d, %d, %d,) and size (%d, %d, %d)\n",
            label, low.x, low.y, low.z, size.x, size.y, size.z,
            it->second.device_offset.x, it->second.device_offset.y, it->second.device_offset.z,
            it->second.device_size.x, it->second.device_size.y, it->second.device_size.z);
        exit(-1);
      }
    }
  } else {

    //it's a staging variable
    if (staging_it != it->second.stagingVars.end()) {

      ////This variable exists in the database, no need to "put" it in again.
      //putNeeded = false;
      //See if someone has stated they are allocating it
      allocationNeeded = testAndSetAllocating(staging_it->second.atomicStatusInGpuMemory);

      if (!allocationNeeded) {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
                << " This staging variable already exists.  No need to allocate another.  For label " << label
                << " patch " << patchID
                << " matl " << matlIndx
                << " level " << levelIndx
                << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
                << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
                << " on device " << d_device_id
                << " with data pointer " << staging_it->second.device_ptr
                << " with status codes " << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory)
                << " into GPUDW at " << std::hex << this << std::dec
                << endl;
          }
          cerrLock.unlock();
        }
        //We need the pointer.  We can't move on until we get the pointer.
        //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
        //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
        bool allocated = false;
        while (!allocated) {
          allocated = checkAllocated(staging_it->second.atomicStatusInGpuMemory);
        }
        //Have this var use the existing memory address.
        var.setArray3(offset, size, staging_it->second.device_ptr);

      }
    }
  }


  //Now allocate it
  if (allocationNeeded) {

    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);

    unsigned int memSize = var.getMemSize();

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::allocateAndPut(), calling allocateCudaSpaceFromPool"
           << " for " << label
           << " patch " << patchID
           << " material " <<  matlIndx
           << " level " << levelIndx;
        if (staging) {
         gpu_stats << " staging: true";
        } else {
         gpu_stats << " staging: false";
        }
        gpu_stats << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
           << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
           << " at " << addr
           << " with status codes ";
        if (!staging) {
         gpu_stats << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory);
        } else {
         gpu_stats << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory);
        }
        gpu_stats << " on device " << d_device_id
           << " into GPUDW at " << std::hex << this << std::dec << endl;
      }
      cerrLock.unlock();
    }

    addr = allocateCudaSpaceFromPool(d_device_id, memSize);

    //Also update the var object itself
    var.setArray3(offset, size, addr);

    //Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, gtype, numGhostCells);

    //Now that the database knows of this and other threads can see the device pointer, update the status from allocating to allocated
    if (!staging) {
      testAndSetAllocate(it->second.atomicStatusInGpuMemory);
    } else {
      testAndSetAllocate(staging_it->second.atomicStatusInGpuMemory);
    }
  }
}

//______________________________________________________________________
//
//This method is meant to take an entry from the host side DW and copy it  into
//the task datawarehouse whose job is to  eventually live GPU side.
__host__  void
GPUDataWarehouse::copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW, char const* label,
                                       int patchID, int matlIndx, int levelIndx, bool staging,
                                       int3 offset, int3 size) {


  if (d_device_copy == NULL) {
    //sanity check
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This method should only be called from a task data warehouse.\n");
    exit(-1);
  }

  varLock->readLock();
  if (d_numVarDBItems==MAX_VARDB_ITEMS) {
    printf("ERROR:  Out of GPUDataWarehouse space");
    varLock->readUnlock();
    exit(-1);
  }
  varLock->readUnlock();


  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  stagingVar sv;
  sv.device_offset = offset;
  sv.device_size = size;

  //Get the iterator(s) from the host side GPUDW.
  hostSideGPUDW->varLock->readLock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator hostSideGPUDW_iter = hostSideGPUDW->varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator hostSideGPUDW_staging_iter;
  if (staging) {
    hostSideGPUDW_staging_iter = hostSideGPUDW_iter->second.stagingVars.find(sv);
  }
  hostSideGPUDW->varLock->readUnlock();

  varLock->writeLock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);
  //sanity check
  if (iter != varPointers->end() && !staging) {
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This task datawarehouse already had an entry for %s patch %d material %d level %d\n", label, patchID, matlIndx, levelIndx);
    varLock->writeUnlock();
    exit(-1);
  }


  //If it's staging, there should already be a non-staging var in the host-side GPUDW (even if it's just a placeholder)

  //Inserting into this task DW, it is a requirement that non-staging variables get inserted first
  //then any staging variables can come in later.  This won't handle any scenario where a staging variable is requested
  //into the task DW without a non-staging variable already existing here.

  int d_varDB_index=d_numVarDBItems;
  d_numVarDBItems++;

  int i = d_varDB_index;

  if (!staging) {

    //copy the item
    allVarPointersInfo vp = hostSideGPUDW_iter->second;
    //Clear out any staging vars it may have had
    vp.stagingVars.clear();

    //Give it a d_varDB index
    vp.varDB_index = d_varDB_index;

    //insert it in
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );


    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.gtype;
    d_varDB[i].varItem.numGhostCells = hostSideGPUDW_iter->second.numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    d_varDB[i].var_offset = hostSideGPUDW_iter->second.device_offset;
    d_varDB[i].var_size = hostSideGPUDW_iter->second.device_size;
    d_varDB[i].var_ptr = hostSideGPUDW_iter->second.device_ptr;

  } else {

    if (iter == varPointers->end()) {
      //A staging item was requested but there's no regular variable for it to piggy back in.
      //So create an empty placeholder regular variable.

      //Start by getting a copy of what the GPU DW already had for this non-staging var
      allVarPointersInfo vp = hostSideGPUDW_iter->second;

      //Clear out any staging vars it may have had
      vp.stagingVars.clear();

      //Empty placeholders won't be placed in the d_varDB array.
      vp.varDB_index = -1;

      //insert it in
      std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator, bool> ret = varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
      if (!ret.second) {
        printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) Failure inserting into varPointers map.\n");
        varLock->writeUnlock();
        exit(-1);
      }
      iter = ret.first;

    }

    //copy the item
    stagingVarInfo svi = hostSideGPUDW_staging_iter->second;

    //Give it a d_varDB index
    svi.varDB_index = d_varDB_index;

    //insert it in
    std::map<stagingVar, stagingVarInfo>::iterator staging_iter = iter->second.stagingVars.find(sv);
    if (staging_iter != iter->second.stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) This staging var already exists in this task DW\n");
    }
    std::pair<stagingVar, stagingVarInfo> p = make_pair( sv, svi );
    iter->second.stagingVars.insert( p );

    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.gtype;
    d_varDB[i].varItem.numGhostCells = hostSideGPUDW_iter->second.numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    d_varDB[i].var_offset = hostSideGPUDW_staging_iter->first.device_offset;
    d_varDB[i].var_size = hostSideGPUDW_staging_iter->first.device_size;
    d_varDB[i].var_ptr = hostSideGPUDW_staging_iter->second.device_ptr;

  }



  d_dirty=true;
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
     gpu_stats << UnifiedScheduler::myRankThread()
         << " GPUDataWarehouse::copyItemIntoTaskDW( " << label << " ) - "
         << " Put into d_varDB at index " << i
         << " of max index " << maxdVarDBItems - 1
         << " label " << label
         << " patch " << patchID
         << " matl " << matlIndx
         << " level " << levelIndx;
     if (staging) {
       gpu_stats << " staging: true";
     } else {
       gpu_stats << " staging: false";
     }
     gpu_stats  << " datatype size " <<d_varDB[i].sizeOfDataType
         << " into address " << d_varDB[i].var_ptr
         << " on device " << d_device_id
         << " into GPUDW at " << std::hex << this << std::dec
         << " size [" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << "]"
         << " offset [" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << "]"
         << endl;
    }
    cerrLock.unlock();

  }
  varLock->writeUnlock();

}



HOST_DEVICE void
GPUDataWarehouse::putContiguous(GPUGridVariableBase &var, const char* indexID, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else

  varLock->writeLock();

  //first check if this patch/var/matl is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //Space for this patch already exists.  Use that and return.
    if (d_debug){
      printf("GPUDataWarehouse::putContiguous( %s ). This gpudw database has a variable for label %s patch %d matl %d level %d staging %s on device %d.  Reusing it.\n",
          label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id);

    }
    var.setArray3(varPointers->operator[](lpml).device_offset, varPointers->operator[](lpml).device_size, varPointers->operator[](lpml).device_ptr);
    varLock->writeUnlock();
    return;
  }

  int3 size=make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset=low;
  void* device_ptr=NULL;
  var.setArray3(offset, size, device_ptr);
  allocateLock->readLock();
  contiguousArrayInfo *ca = &(contiguousArrays->operator[](indexID));
  allocateLock->readUnlock();
  if ( (ca->allocatedDeviceMemory == NULL
       || ca->sizeOfAllocatedMemory - ca->assignedOffset < var.getMemSize())
      && stageOnHost) {
    printf("ERROR: No room left on device to be assigned address space\n");
    if (ca->allocatedDeviceMemory != NULL) {
      printf("There was %lu bytes allocated, %lu has been assigned, and %lu more bytes were attempted to be assigned for %s patch %d matl %d level %d staging %s\n",
          ca->sizeOfAllocatedMemory,
          ca->assignedOffset,
          var.getMemSize(), label, patchID, matlIndx, levelIndx, staging ? "true" : "false");
    }
    varLock->writeUnlock();
    exit(-1);
  } else {


    //There is already pre-allocated contiguous memory chunks with room available on
    //both the device and the host.  Just assign pointers for both the device and host contiguous arrays.


    //This prepares the var with the offset and size.  The actual address will come next.

    void* host_contiguousArrayPtr = NULL;

    int varMemSize = var.getMemSize();

    device_ptr = (void*)((uint8_t*)ca->allocatedDeviceMemory + ca->assignedOffset);
    var.setArray3(offset, size, device_ptr);
    host_contiguousArrayPtr = (void*)((uint8_t*)ca->allocatedHostMemory + ca->assignedOffset);

    //We ran into cuda misaligned errors previously when mixing different data types.  We suspect the ints at 4 bytes
    //were the issue.  So the engine previously computes buffer room for each variable as a multiple of UnifiedScheduler::bufferPadding.
    //So the contiguous array has been sized with extra padding.  (For example, if a var holds 12 ints, then it would be 48 bytes in
    //size.  But if UnifiedScheduler::bufferPadding = 32, then it should add 16 bytes for padding, for a total of 64 bytes).
    int memSizePlusPadding = ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
    ca->assignedOffset += memSizePlusPadding;


    if (stageOnHost) {
      //Some GPU grid variable data doesn't need to be copied from the host
      //For example, computes vars are just uninitialized space.
      //Others grid vars need to be copied.  This copies the data into a contiguous
      //array on the host so that copyDataHostToDevice() can copy the contiguous
      //host array to the device.

      //Data listed as required.  Or compute data that was initialized as a copy of something else.
      ca->copiedOffset += memSizePlusPadding;

      memcpy(host_contiguousArrayPtr, gridVar->getBasePointer(), varMemSize);

    } //else {
      //printf("Setting aside space %s %d %d from host location %p host contiguous array %p\n", label, patchID, matlIndx, host_ptr, host_contiguousArrayPtr);
    //}

    varLock->writeUnlock();

    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, None, 0, host_contiguousArrayPtr);

    //printf("Allocating for %s at patch %d and matl %d size is %d host_ptr %p host_contiguousPtr %p device_ptr %p\n", label, patchID, matlIndx, varMemSize, host_ptr, host_contiguousArrayPtr, device_ptr);
  }


#endif
}

HOST_DEVICE void
GPUDataWarehouse::allocate(const char* indexID, size_t size)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  if (size == 0) {
    return;
  }

  //This method allocates one big chunk of memory so that little allocations do not have to occur for each grid variable.
  //This is needed because devices often have substantial overhead for each device malloc and device copy.  By putting it into one
  //chunk of memory, only one malloc and one copy to device should be needed.
  double *d_ptr = NULL;
  double *h_ptr = NULL;
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);

  printf("Allocated GPU buffer of size %lu \n", (unsigned long)size);

  CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, size) );
  //printf("In allocate(), cuda malloc for size %ld at %p on device %d\n", size, d_ptr, d_device_id);


  if (d_debug) {
    printf("In allocate(), cudaMalloc for size %ld at %p on device %d\n", size, d_ptr, d_device_id);
  }
  //Now allocate that much also on the host. We want to do this because it's easier to pool up all the data on the host side
  //and then move it over to the device side later in one shot.  It also allows for one copy doing a device to host later.
  //h_ptr = new double[size];


  h_ptr = (double*)malloc(size);

  //Registering memory seems good in theory, but bad in practice for our purposes.
  //On the k20 device on beast.sci.utah.edu, this single register call was taking 0.1 seconds!
  //On my home GTX580 device, it was taking 0.015 seconds, better, but still substantial enough
  //we should avoid it for now. (If you want to use it, then also uncomment the cudaHostUnregister call in clear()).
  //cudaHostRegister(h_ptr, size, cudaHostRegisterPortable);

  contiguousArrayInfo ca(d_ptr, h_ptr, size);
  allocateLock->writeLock();
  contiguousArrays->insert( std::map<const char *, contiguousArrayInfo>::value_type( indexID, ca ) );
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->begin(); it != contiguousArrays->end(); ++it)
  //  printf("%s\n", it->first.c_str());

  allocateLock->writeUnlock();
#endif
}

//HOST_DEVICE cudaError_t
//GPUDataWarehouse::copyDataHostToDevice(const char* indexID, void *cuda_stream) {
//#ifdef __CUDA_ARCH__
//  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
//  //return something so there is no compiler warning
//  cudaError_t retVal = cudaErrorUnknown;
//  return retVal;
//#else
//  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
//  allocateLock->readLock();
//  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->begin(); it != contiguousArrays->end(); ++it)
//  //  printf("*** displaying %s\n", it->first.c_str());
//  contiguousArrayInfo *ca = &(contiguousArrays->operator[](indexID));
//
//
//  allocateLock->readUnlock();
//  cudaError_t retVal;
//  //copy only the initialized data, not the whole thing.
//  //printf("Copying to device %p from host %p amount %d\n", ca->allocatedDeviceMemory, ca->allocatedHostMemory, ca->copiedOffset);
//  //cudaError_t retVal = cudaMemcpy(ca->allocatedDeviceMemory, ca->allocatedHostMemory,
//  //                                     ca->copiedOffset, cudaMemcpyHostToDevice);
//  CUDA_RT_SAFE_CALL ( retVal = cudaMemcpyAsync(ca->allocatedDeviceMemory, ca->allocatedHostMemory,
//                                       ca->copiedOffset, cudaMemcpyHostToDevice, *((cudaStream_t*)cuda_stream)) );
//
//  return retVal;
//
//#endif
//}

/*
HOST_DEVICE cudaError_t
GPUDataWarehouse::copyDataDeviceToHost(const char* indexID, void *cuda_stream) {
#ifdef __CUDA_ARCH__
  //Should not be called from device side.
  //return something so there is no compiler warning
  cudaError_t retval = cudaErrorUnknown;
  return retval;
#else
  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );

  //see if this datawarehouse has anything for this patchGroupID.
  allocateLock->writeLock();
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->begin(); it != contiguousArrays->end(); ++it)
  //  printf("%s\n", it->first.c_str());
  std::string sIndex = indexID;
  std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->find(sIndex);
  if ( it != contiguousArrays->end()) {
    sIndex += " used";  //mark it as used so it doesn't match in any future find() queries for the next iteration step.
    contiguousArrays->operator[](sIndex) = it->second; //put it back into the map with a new key.
    contiguousArrayInfo ca = it->second;
    contiguousArrays->erase(it);

    allocateLock->writeUnlock();
    if (ca.sizeOfAllocatedMemory - ca.copiedOffset > 0) {
      //Previously we only copied into the device variables that were already initialized with data
      //But now we need to copy the computes data back to the host.
      //printf("Copying to host %p from device %p amount %d\n", ca->allocatedHostMemory + ca->copiedOffset, ca->allocatedDeviceMemory + ca->copiedOffset, ca->assignedOffset - ca->copiedOffset);

      cudaError_t retVal = cudaMemcpyAsync((void*)((uint8_t*)ca.allocatedHostMemory + ca.copiedOffset),
                                            (void*)((uint8_t*)ca.allocatedDeviceMemory + ca.copiedOffset),
                                            ca.assignedOffset - ca.copiedOffset,
                                            cudaMemcpyDeviceToHost,
                                            *((cudaStream_t*)cuda_stream));
      //cudaError_t retVal = cudaMemcpy((void*)((uint8_t*)ca.allocatedHostMemory + ca.copiedOffset),
      //                                     (void*)((uint8_t*)ca.allocatedDeviceMemory + ca.copiedOffset),
      //                                     ca.assignedOffset - ca.copiedOffset,
      //                                     cudaMemcpyDeviceToHost);


      if (retVal !=  cudaErrorLaunchFailure) {
         //printf("Copying to host ptr %p (starts at %p) from device ptr %p (allocation starts at %p) amount %d with stream %p\n", ca.allocatedHostMemory + ca.copiedOffset, ca.allocatedHostMemory, ca.allocatedDeviceMemory + ca.copiedOffset, ca.allocatedDeviceMemory, ca.assignedOffset - ca.copiedOffset, cuda_stream);
         CUDA_RT_SAFE_CALL(retVal);
      }
      return retVal;
    }
  } else {
    allocateLock->writeUnlock();
  }

  cudaError_t retVal = cudaSuccess;
  return retVal;

#endif
}
*/
/*
void GPUDataWarehouse::copyDataHostToDevice(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high) {
  CUDA_RT_SAFE_CALL(
     cudaMemcpyAsync(device_ptr,
         dynamic_cast<GridVariableBase*>(it->second.var)->getBasePointer(),
         it->second.varMemSize, cudaMemcpyHostToDevice, *stream));
}
*/
HOST_DEVICE void
GPUDataWarehouse::copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndx, int levelIndx) {
#ifdef __CUDA_ARCH__
  //Should not called from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  //see if this datawarehouse has anything for this patchGroupID.
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo info = varPointers->operator[](lpml);

    device_var.setArray3(varPointers->operator[](lpml).device_offset, varPointers->operator[](lpml).device_offset, info.device_ptr);
    varLock->readUnlock();
   // size_t size = device_var.getMemSize();

    //TODO: Instead of doing a memcpy, I bet the original host grid variable could just have its pointers updated
    //to work with what we were sent back.  This would take some considerable work though to get all the details right
    //TODO: This needs to be a memcpy async
    memcpy(host_var->getBasePointer(), info.host_contiguousArrayPtr, device_var.getMemSize());
    //Since we've moved it back into the host, lets mark it as being used.
    //It's possible in the future there could be a scenario where we want to bring it
    //back to the host but still retain it in the GPU.  One scenario is
    //sending data to an output .ups file but not modifying it on the host.
    remove(label, patchID, matlIndx, levelIndx);

  } else {
    varLock->readUnlock();
    printf("ERROR: host copyHostContiguoustoHost unknown variable on GPUDataWarehouse");
    //for (std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it=varPointers->begin(); it!=varPointers->end(); ++it)
    //  printf("%s %d %d => %d \n", it->first.label, it->first.patchID, it->first.matlIndx, it->second.varDB_index);
    exit(-1);
  }
#endif

}

/*
HOST_DEVICE void*
GPUDataWarehouse::getPointer(char const* label, int patchID, int matlIndx)

//GPU implementations can be faster if you work with direct pointers, each thread having its own pointer
//and doing pointer arithmetic on it.  This is obviously a more low level and "you're on your own" approach.
//Though for some problems with many grid variables, each thread will need many pointers, overwhelming
//the amount of registers available to store them.  So this approach can't work for these problems,
//and instead a GPU shared variable exists which hold the pointer, and then werecalculate the x,y,z
//offset each time a thread requests data from the array.
{
#ifdef __CUDA_ARCH__
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx);
  if (item){
    return item->var_ptr;
  }else{
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

    int i=threadID;
    while(i<d_numVarDBItems){
      printf( "   Available labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::getPointer( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", label, patchID, matlIndx);
      assert(0);
    }
    //printf("\t ERROR: GPUDataWarehouse::getPointer( \"%s\", patchID: %i, matl: %i )  unknown variable\n", label, patchID, matlIndx);
    return NULL;
  }
#else
  varLock->readLock();
  labelPatchMatlLevel lpm(label, patchID, matlIndx);
  if (varPointers->find(lpm) != varPointers->end()) {
    int i = varPointers[lpm].varDB_index;
    varLock->readUnlock();
    return d_varDB[i].var_ptr;
  } else {
    printf("ERROR: host get unknown variable on GPUDataWarehouse");
    varLock->readUnlock();
    exit(-1);
  }

// cpu code
//int i= 0;
//while(i<numItems){
//  if (!strncmp(d_varDB[i].label, label, MAX_NAME_LENGTH) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndx==matlIndx) {
//    return d_varDB[i].var_ptr;
//  }
//  i++;
//}
//if (i==numItems) {
//  printf("host get unknown variable on GPUDataWarehouse");
//  exit(-1);
//}
//if (d_debug) printf("host got %s loc 0x%x from GPUDW 0x%x on device %d\n", label, d_varDB[i].var_ptr, device_copy, device_id);
#endif
}
*/


//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUReductionVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, void* host_ptr)
{

  varLock->writeLock();

  void* var_ptr;           // raw pointer to the memory
  var.getData(var_ptr);

  //See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);

  //sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side GPU DW without it first existing in the internal database.\n");
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.device_ptr = var_ptr;
  iter->second.sizeOfDataType = sizeOfDataType;
  iter->second.gtype = None;
  iter->second.numGhostCells = 0;
  iter->second.host_contiguousArrayPtr = host_ptr;
  iter->second.atomicStatusInHostMemory = UNKNOWN;
  int3 zeroValue;
  zeroValue.x = 0;
  zeroValue.y = 0;
  zeroValue.z = 0;
  iter->second.device_offset = zeroValue;
  iter->second.device_size = zeroValue;


  //previously set, do not set here
  //iter->second.atomicStatusInGputMemory =

  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " GPUDataWarehouse::put( " << label << " ) - "
          << " Put a reduction variable in the host-side varPointers map for label " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx
          << " at device address " << var_ptr
          << " with datatype size " << iter->second.sizeOfDataType
          << " with status codes " << getDisplayableStatusCodes(iter->second.atomicStatusInGpuMemory)
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName
          << " current varPointers size is: " << varPointers->size()
          << endl;
    }
    cerrLock.unlock();
  }

  varLock->writeUnlock();

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUPerPatchBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, void* host_ptr)
{

  varLock->writeLock();
  void* var_ptr;           // raw pointer to the memory
  var.getData(var_ptr);

  //See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);

  //sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side GPU DW without it first existing in the internal database.\n");
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.device_ptr = var_ptr;
  iter->second.sizeOfDataType = sizeOfDataType;
  iter->second.gtype = None;
  iter->second.numGhostCells = 0;
  iter->second.host_contiguousArrayPtr = host_ptr;
  iter->second.atomicStatusInHostMemory = UNKNOWN;
  int3 zeroValue;
  zeroValue.x = 0;
  zeroValue.y = 0;
  zeroValue.z = 0;
  iter->second.device_offset = zeroValue;
  iter->second.device_size = zeroValue;

  //previously set, do not set here
  //iter->second.atomicStatusInGputMemory =

  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " GPUDataWarehouse::put( " << label << " ) - "
          << " Put a patch variable in the host-side varPointers map for label " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx
          << " at device address " << var_ptr
          << " with datatype size " << iter->second.sizeOfDataType
          << " with status codes " << getDisplayableStatusCodes(iter->second.atomicStatusInGpuMemory)
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName
          << " current varPointers size is: " << varPointers->size()
          << endl;
    }
    cerrLock.unlock();
  }

  varLock->writeUnlock();

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType)
{

  //Allocate space on the GPU and declare a variable onto the GPU.
  //This method does NOT stage everything in a big array.

  //Check if it exists prior to allocating memory for it.
  //If it has already been allocated, just use that.
  //If it hasn't, this is lock free and the first thread to request allocating gets to allocate
  //If another thread sees that allocating is in process, it loops and waits until the allocation complete.

  bool allocationNeeded = false;
  int3 size = make_int3(0,0,0);
  int3 offset = make_int3(0,0,0);
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread() << " Calling putUnallocatedIfNotExists() for " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName << endl;
    }
    cerrLock.unlock();
  }
  //This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset, size);

  varLock->readLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  varLock->readUnlock();

  void* addr = NULL;

  //Now see if we allocate the variable or use a previous existing allocation.

  //See if someone has stated they are allocating it
  allocationNeeded = testAndSetAllocating(it->second.atomicStatusInGpuMemory);
  if (!allocationNeeded) {
    //Someone else is allocating it or it has already been allocated.
     //Space for this var already exists.  Use that and return.
     if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
            << " This reduction variable already exists.  No need to allocate another.  GPUDW has a variable for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " on device " << d_device_id
            << " with data pointer " << it->second.device_ptr
            << " with status codes " << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory)
            << " into GPUDW at " << std::hex << this << std::dec
            << endl;
       }
       cerrLock.unlock();
     }
     //We need the pointer.  We can't move on until we get the pointer.
     //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
     //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
     bool allocated = false;
     while (!allocated) {
       allocated = checkAllocated(it->second.atomicStatusInGpuMemory);
     }
     //Have this var use the existing memory address.
     var.setData(addr);
  } else {
    //We are the first task to request allocation.  Do it.
    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
    size_t memSize = var.getMemSize();

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateAndPut(), calling allocateCudaSpaceFromPool"
            << " for reduction variable " << label
            << " patch " << patchID
            << " material " <<  matlIndx
            << " level " << levelIndx
            << " size " << var.getMemSize()
            << " at " << addr
            << " with status codes " << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec << endl;
      }
      cerrLock.unlock();
    }

    addr = allocateCudaSpaceFromPool(d_device_id, memSize);

    //Also update the var object itself
    var.setData(addr);

    //Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);

    //Now that the database knows of this and other threads can see the device pointer, update the status from allocating to allocated
    testAndSetAllocate(it->second.atomicStatusInGpuMemory);
  }

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType)
{

  //Allocate space on the GPU and declare a variable onto the GPU.
  //This method does NOT stage everything in a big array.

  //Check if it exists prior to allocating memory for it.
  //If it has already been allocated, just use that.
  //If it hasn't, this is lock free and the first thread to request allocating gets to allocate
  //If another thread sees that allocating is in process, it loops and waits until the allocation complete.

  bool allocationNeeded = false;
  int3 size = make_int3(0,0,0);
  int3 offset = make_int3(0,0,0);
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread() << " Calling putUnallocatedIfNotExists() for " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName << endl;
    }
    cerrLock.unlock();
  }
  //This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset, size);

  varLock->readLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  varLock->readUnlock();

  void* addr = NULL;

  //Now see if we allocate the variable or use a previous existing allocation.

  //See if someone has stated they are allocating it
  allocationNeeded = testAndSetAllocating(it->second.atomicStatusInGpuMemory);
  if (!allocationNeeded) {
    //Someone else is allocating it or it has already been allocated.
     //Space for this var already exists.  Use that and return.
     if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
            << " This patch variable already exists.  No need to allocate another.  GPUDW has a variable for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " on device " << d_device_id
            << " with data pointer " << it->second.device_ptr
            << " with status codes " << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory)
            << " into GPUDW at " << std::hex << this << std::dec
            << endl;
       }
       cerrLock.unlock();
     }
     //We need the pointer.  We can't move on until we get the pointer.
     //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
     //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
     bool allocated = false;
     while (!allocated) {
       allocated = checkAllocated(it->second.atomicStatusInGpuMemory);
     }
     //Have this var use the existing memory address.
     var.setData(addr);
  } else {
    //We are the first task to request allocation.  Do it.
    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
    size_t memSize = var.getMemSize();

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
            gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateAndPut(), calling allocateCudaSpaceFromPool"
            << " for PerPatch variable " << label
            << " patch " << patchID
            << " material " <<  matlIndx
            << " level " << levelIndx
            << " size " << var.getMemSize()
            << " at " << addr
            << " with status codes " << getDisplayableStatusCodes(it->second.atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec << endl;
      }
      cerrLock.unlock();
    }

    addr = allocateCudaSpaceFromPool(d_device_id, memSize);

    //Also update the var object itself
    var.setData(addr);

    //Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);

    //Now that the database knows of this and other threads can see the device pointer, update the status from allocating to allocated
    testAndSetAllocate(it->second.atomicStatusInGpuMemory);
  }
}

//______________________________________________________________________
//
HOST_DEVICE GPUDataWarehouse::dataItem* 
GPUDataWarehouse::getItem(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__


  //This upcoming __syncthreads is needed.  I believe with CUDA function calls are inlined.
  // If you don't have it this upcoming __syncthreads here's what can happen:

  // * The correct index was found by one of the threads.
  // * The last __syncthreads is called, all threads met up there.
  // * Some threads in the block then make a second "function" call and reset index to -1
  // * Meanwhile, those other threads were still in the first "function" call and hadn't
  //   yet processed if (index == -1).  They now run that line.  And see index is now -1.  That's bad.

  // So to prevent this scenario, we have one more __syncthreads.
  __syncthreads();  //sync before get



  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  //int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block

  int i = threadID;
  __syncthreads();  //sync before get


  //if (d_debug && threadID == 0 && blockID == 0) {
  //  printf("device getting item \"%s\" from GPUDW %p", label, this);
  //  printf("size (%d vars)\n Available labels:", d_numVarDBItems);
  //}

  //Have every thread try to find the label/patchId/matlIndx is a match in
  //array.  This is a clever approach so that instead of doing a simple
  //sequential search with one thread, we can let every thread search for it.  Only the
  //winning thread gets to write to shared data.
  __shared__ int index;
  index = -1;
  while(i<d_numVarDBItems){
    int strmatch=0;
    char const *s1 = label; //reset s1 and s2 back to the start
    char const *s2 = &(d_varDB[i].label[0]);

    //a one-line strcmp.  This should keep branching down to a minimum.
    while (!(strmatch = *(unsigned char *) s1 - *(unsigned char *) s2) && *s1++ && *s2++);

    //only one thread will ever match this.
    //And nobody on the device side should ever access "staging" variables.
    if (strmatch == 0) {
    	if (patchID ==-99999999                            //Only getLevel calls should hit this
    	&& d_varDB[i].matlIndx == matlIndx
		&& d_varDB[i].levelIndx == levelIndx
		&& d_varDB[i].varItem.staging == false             /* we don't support staging/foregin vars for get() */
		&& d_varDB[i].ghostItem.dest_varDB_index == -1) {  /*don't let ghost cell copy data mix in with normal variables for get() */
    		index = i; //we found it.
    	}
    	else if(d_varDB[i].domainID == patchID
        && d_varDB[i].matlIndx == matlIndx
        && d_varDB[i].levelIndx == levelIndx
        && d_varDB[i].varItem.staging == false
        && d_varDB[i].ghostItem.dest_varDB_index == -1) {
    		index = i; //we found it.
      //printf("I'm thread %d In DW at %p, We found it for var %s patch %d matl %d level %d.  d_varDB has it at index %d var %s patch %d at its item address %p with var pointer %p\n",
      //              threadID, this, label, patchID, matlIndx, levelIndx, index, &(d_varDB[index].label[0]), d_varDB[index].domainID, &d_varDB[index], d_varDB[index].var_ptr);

    	}
    }
    i = i + numThreads; //Since every thread is involved in searching for the string, have this thread loop to the next possible item to check for.
  }

  //sync before return;
  __syncthreads();
  if (index == -1) {
    printf("ERROR:\nGPUDataWarehouse::getItem() didn't find anything for %s patch %d matl %d with threadID %d and numthreads %d\n", label, patchID, matlIndx, threadID, numThreads);
    return NULL;
  }
  return &d_varDB[index];
#else
  //__________________________________
  // cpu code
  /*labelPatchMatlLevel lpm(label, patchID, matlIndx);
  int i = 0;
  varLock->readLock();
  if (varPointers->find(lpm) != varPointers->end()) {
    i = varPointers[lpm].varDB_index;
    varLock->readUnlock();
  } else {
    varLock->readUnlock();
    printf("ERROR:\nGPUDataWarehouse::getItem( %s ) host get unknown variable from GPUDataWarehouse\n",label);
    exit(-1);
  }

  if (d_debug){
    printf("host got \"%s\" loc %p from GPUDW %p on device %u\n", label, d_varDB[i].var_ptr, d_device_copy, d_device_id);
  }
  //quick error check
  if (strcmp(d_varDB[i].label, label) != 0 || d_varDB[i].domainID != patchID || d_varDB[i].matlIndx != matlIndx) {
    printf("ERROR:\nGPUDataWarehouse::getItem( %s ), data does not match what was expected\n",label);
    exit(-1);
  }
  */
  printError("This method should only be called device side.", "getItem()", label, patchID, matlIndx, levelIndx );
  //printf("ERROR:\nGPUDataWarehouse::getItem() should only be called device side.\n",label);
  return &d_varDB[0];
#endif
}

/*
//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("exist() is not yet implemented for the device.\n");
  return false;
#else
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->readLock();
  bool retVal = varPointers->find(lpml) != varPointers->end();
  varLock->readUnlock();
  return retVal;
#endif 
}
*/
//______________________________________________________________________
//
/*
__host__ bool
GPUDataWarehouse::getAllocated(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset, bool skipContiguous, bool onlyContiguous )
{
#ifdef __CUDA_ARCH__
  printf("getAllocated() is not yet implemented for the device.\n");
  return false;
#else
  //check if we have matching label, patch, material, size and offsets.
  bool retVal = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->readLock();
  if (varPointers->find(lpml) != varPointers->end()) {
    int3 device_offset = varPointers->operator[](lpml).device_offset;
    int3 device_size = varPointers->operator[](lpml).device_size;
    if (device_offset.x == host_offset.x && device_offset.y == host_offset.y && device_offset.z == host_offset.z
        && device_size.x == host_size.x && device_size.y == host_size.y && device_size.z == host_size.z) {
      retVal = true;
      //There is need sometimes to see if the variable exists, but not as part of a contiguous array
      if (skipContiguous) {
        if (varPointers->operator[](lpml).host_contiguousArrayPtr != NULL) {
          //It exists as part of a contiguous array
          retVal = false;
        }
      } if (onlyContiguous) {
        if (varPointers->operator[](lpml).host_contiguousArrayPtr == NULL) {
          //It exists as part of a contiguous array
          retVal = false;
        }
      }
    }
  }
  varLock->readUnlock();
  return retVal;
#endif
}
*/

//______________________________________________________________________
//

HOST_DEVICE bool
GPUDataWarehouse::remove(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should not be called on device.\n");
  return false;
#else
  //It seems there are few scenarios for calling remove.  I think the only time it should
  //happen is removing staging variables.
  //Avoid calling this unless you are absolutely sure what you are doing.
  //Further, this doesn't erase any staging vars within a var.
  bool retVal = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->writeLock();

  if (varPointers->find(lpml) != varPointers->end()) {
    int i = varPointers->operator[](lpml).varDB_index;
    d_varDB[i].label[0] = '\0'; //leave a hole in the flat array, not deleted.
    varPointers->erase(lpml);  //TODO: GPU Memory leak?
    retVal = true;
    d_dirty=true;
  }
  if (d_debug){
    printf("GPUDataWarehouse::remove( %s ). Removed a variable for label %s patch %d matl %d level %d \n",
        label, label, patchID, matlIndx, levelIndx);
  }
  varLock->writeUnlock();
  return retVal;
#endif
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::init(int id, std::string internalName)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::init() should not be called on the device.\n");
#else


  d_device_id = id;
  //this->_internalName = new std::string(internalName);
  strncpy(_internalName, internalName.c_str(), sizeof(_internalName));
  objectSizeInBytes = 0;
  maxdVarDBItems = 0;
  //this->placementNewBuffer = placementNewBuffer;
  allocateLock = new SCIRun::CrowdMonitor("allocate lock");
  varLock = new SCIRun::CrowdMonitor("var lock");
  varPointers = new std::map<labelPatchMatlLevel, allVarPointersInfo>;
  contiguousArrays = new std::map<std::string, contiguousArrayInfo>;

  //other data members are initialized in the constructor
  d_numVarDBItems = 0;
  d_numMaterials = 0;
  d_debug = false;
  //d_numGhostCells = 0;
  d_device_copy = NULL;
  d_dirty = true;
  objectSizeInBytes = 0;
  //resetdVarDB();
  numGhostCellCopiesNeeded = 0;

#endif
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::cleanup()
{

  delete allocateLock;
  delete varLock;
  delete varPointers;
  delete contiguousArrays;

}


//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::init_device(size_t objectSizeInBytes, unsigned int maxdVarDBItems)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::init_device() should only be called by the framework\n");
#else

    this->objectSizeInBytes = objectSizeInBytes;
    this->maxdVarDBItems = maxdVarDBItems;
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    void* temp = NULL;
    //CUDA_RT_SAFE_CALL(cudaMalloc(&temp, objectSizeInBytes));
    temp = allocateCudaSpaceFromPool(d_device_id, objectSizeInBytes);
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
       gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::init_device() -"
           << " requested GPU space from allocateCudaSpaceFromPool for Task DW of size " << objectSizeInBytes
           << " bytes at " << temp
           << " on device " << d_device_id
           << endl;
      }
      cerrLock.unlock();
    }
    d_device_copy = (GPUDataWarehouse*)temp;
    //cudaHostRegister(this, sizeof(GPUDataWarehouse), cudaHostRegisterPortable);



    d_dirty = true;

#endif 
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::syncto_device(void *cuda_stream)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
#else

  if (!d_device_copy) {
    printf("ERROR:\nGPUDataWarehouse::syncto_device()\nNo device copy\n");
    exit(-1);
  }
  varLock->writeLock();

  if (d_dirty){
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    //Even though this is in a writeLock state on the CPU, the nature of multiple threads
    //each with their own stream copying to a GPU means that one stream might seemingly go outg
    //of order.  This is ok for two reasons. 1) Nothing should ever be *removed* from a gpu data warehouse
    //2) Therefore, it doesn't matter if streams go out of order, each thread will still ensure it copies
    //exactly what it needs.  Other streams may write additional data to the gpu data warehouse, but cpu
    //threads will only access their own data, not data copied in by other cpu threada via streams.

    //This approach does NOT require CUDA pinned memory.
    //unsigned int sizeToCopy = sizeof(GPUDataWarehouse);
    cudaStream_t* stream = (cudaStream_t*)(cuda_stream);

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::syncto_device() - cudaMemcpy -"
            << " sync GPUDW at " << d_device_copy
            << " with description " << _internalName
            << " to device " << d_device_id
            << " on stream " << stream
            << endl;
      }
      cerrLock.unlock();
    }

    CUDA_RT_SAFE_CALL (cudaMemcpyAsync( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice, *stream));
    //CUDA_RT_SAFE_CALL (cudaMemcpy( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice));

    //if (d_debug) {
    //printf("%s sync GPUDW %p to device %d on stream %p\n", UnifiedScheduler::myRankThread().c_str(), d_device_copy, d_device_id, stream);
    //}
    d_dirty=false;
  }

  varLock->writeUnlock();

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::clear() 
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else

  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );

  //delete any grid var that isn't part of a contiguous array
  varLock->writeLock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator varIter;
  for (varIter = varPointers->begin(); varIter != varPointers->end(); ++varIter) {
    if (varIter->second.host_contiguousArrayPtr == NULL) {
      //clear out all the staging vars, if any
      std::map<stagingVar, stagingVarInfo>::iterator stagingIter;
      for (stagingIter = varIter->second.stagingVars.begin(); stagingIter != varIter->second.stagingVars.end(); ++stagingIter) {

        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::clear() -"
                << " calling freeCudaSpaceFromPool() for staging var for " << varIter->first.label
                << " at device ptr " <<  stagingIter->second.device_ptr
                << " on device " << d_device_id
                << endl;
          }
          cerrLock.unlock();
        }


        //CUDA_RT_SAFE_CALL(cudaFree(stagingIter->second.device_ptr));
        //stagingIter->second.device_ptr == NULL;
        size_t memSize = stagingIter->first.device_size.x *
                          stagingIter->first.device_size.y *
                          stagingIter->first.device_size.z *
                          varIter->second.sizeOfDataType;
        if (freeCudaSpaceFromPool(d_device_id, memSize, stagingIter->second.device_ptr)) {
          stagingIter->second.device_ptr == NULL;
        } else {
          //No open spot in the pool, go ahead and allocate it.
          printf("ERROR:\nGPUDataWarehouse::clear(), for a staging variable, couldn't find in the GPU memory pool the space starting at address %p\n", stagingIter->second.device_ptr);
          varLock->writeUnlock();
          exit(-1);
        }

      }
      varIter->second.stagingVars.clear();

      //clear out the regular vars

      //See if it's a placeholder var for staging vars.  This happens if the non-staging var
      //had a device_ptr of NULL, and it was only in the varPointers map to only hold staging vars
      if (varIter->second.device_ptr) {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::clear() -"
                << " calling freeCudaSpaceFromPool() for non-staging var for " << varIter->first.label
                << " at device ptr " <<  varIter->second.device_ptr
                << " on device " << d_device_id
                << endl;
          }
          cerrLock.unlock();
        }
        size_t memSize = varIter->second.sizeOfDataType;
        if (varIter->second.device_size.x != 0) {
          memSize = memSize *
                    varIter->second.device_size.x *
                    varIter->second.device_size.y *
                    varIter->second.device_size.z;
        }
        if (freeCudaSpaceFromPool(d_device_id, memSize, varIter->second.device_ptr)) {
          varIter->second.device_ptr == NULL;
        } else {
          printf("ERROR:\nGPUDataWarehouse::clear(), for a non-staging variable, couldn't find in the GPU memory pool the space starting at address %p\n", varIter->second.device_ptr);
          varLock->writeUnlock();
          exit(-1);
        }
      }
    }
  }
  varPointers->clear();

  //delete all the contiguous arrays
  std::map<std::string, contiguousArrayInfo>::iterator iter;
  for (iter = contiguousArrays->begin(); iter != contiguousArrays->end(); ++iter) {
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::clear() -"
            << " cudaFree for contiguous array for " << iter->first.c_str()
            << " at device ptr " << iter->second.allocatedDeviceMemory
            << " and host free at host ptr " << iter->second.allocatedHostMemory
            << " on device " << d_device_id
            << endl;
      }
      cerrLock.unlock();
    }
    CUDA_RT_SAFE_CALL(cudaFree(iter->second.allocatedDeviceMemory));
    //cudaHostUnregister(iter->second.allocatedHostMemory);
    free(iter->second.allocatedHostMemory);

  }
  contiguousArrays->clear();

  varLock->writeUnlock();

  init(d_device_id, _internalName);

#endif
}


HOST_DEVICE void
GPUDataWarehouse::deleteSelfOnDevice()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else
  if ( d_device_copy ) {
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << "GPUDataWarehouse::deleteSelfOnDevice - calling freeCudaSpaceFromPool for Task DW at " << std::hex
           << d_device_copy << " on device " << std::dec << d_device_id << std::endl;
      }
      cerrLock.unlock();
    }

    //cudaHostUnregister(this);
    freeCudaSpaceFromPool(d_device_id, objectSizeInBytes, d_device_copy);
    //CUDA_RT_SAFE_CALL(cudaFree( d_device_copy ));

  }
#endif
}

HOST_DEVICE void
GPUDataWarehouse::resetdVarDB()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else

  if (d_device_copy != NULL) {
    //TODO: When TaskDWs are removed, this section shouldn't be needed as there won't be concurrency problems

    //This is designed to help stop tricky race scenarios.  One such scenario I encountered was as follows:
    //Thread A would call getItem() on the GPU, and look thruogh d_varDB for a matching label/patch/matl tuple
    //Thread B would have previously added a new item to the d_varDB, then called syncto_device.
    //Thread B would be partway through updating d_varDB on the GPU.  It would increase the number of items by one
    //And it would write the label.  But it wouldn't yet write the patch or matl part of the tuple. By coincidence
    //the old garbage data in the GPU would have exactly the patch and matl that matches thread A's query
    //For a very brief window, there would be 2 tuples matching that label/patch/matl pair in d_varDB because
    //thread B hasn't fully written in all of his data.
    //Thread A's getItem() would run exactly in this brief window, find the wrong match, and use the wrong
    //memory address, and the program would crash with an invalid address.
    //The answer is to initialize d_varDB to items that should never provide an accidental match.
    //This should also occur for all other arrays.

    //TODO: Should this be could be cleaned up to only reset as much as was used.
    for (int i = 0; i < MAX_VARDB_ITEMS; i++) {
      d_varDB[i].label[0] = '\0';
      d_varDB[i].domainID = -1;
      d_varDB[i].matlIndx = -1;
      //d_varDB[i].staging = false;
      d_varDB[i].var_ptr = NULL;
      d_varDB[i].ghostItem.dest_varDB_index = -1;

    }
    for (int i = 0; i < MAX_LEVELDB_ITEMS; i++) {
      d_levelDB[i].label[0] = '\0';
      d_levelDB[i].domainID = -1;
      d_levelDB[i].matlIndx = -1;
      //d_varDB[i].staging = false;
      d_levelDB[i].var_ptr = NULL;
    }
    for (int i = 0; i < MAX_MATERIALSDB_ITEMS; i++) {
      d_materialDB[i].simulationType[0] = '\0';
    }
  }
#endif
}

HOST_DEVICE void
GPUDataWarehouse::putMaterials( std::vector< std::string > materials)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side

#else
  //__________________________________
  //cpu code

  varLock->writeLock();
  //see if a thread has already supplied this datawarehouse with the material data
  int numMaterials = materials.size();

  if (d_numMaterials != numMaterials) {
    //nobody has given us this material data yet, so lets add it in from the beginning.

    if (numMaterials > MAX_MATERIALSDB_ITEMS) {
      printf("ERROR: out of GPUDataWarehouse space for materials");
      exit(-1);
    }
    for (int i = 0; i < numMaterials; i++) {
      if (strcmp(materials.at(i).c_str(), "ideal_gas") == 0) {
        d_materialDB[i].material = IDEAL_GAS;
      } else {
        printf("ERROR:  This material has not yet been coded for GPU support\n.");
        exit(-1);
      }
    }
    d_numMaterials = numMaterials;

  }

  varLock->writeUnlock();

#endif
}

HOST_DEVICE int
GPUDataWarehouse::getNumMaterials() const
{
#ifdef __CUDA_ARCH__
  return d_numMaterials;
#else
  //I don't know if it makes sense to write this for the host side, when it already exists elsewhere host side.
  return -1;
#endif
}

HOST_DEVICE materialType
GPUDataWarehouse::getMaterial(int i) const
{
#ifdef __CUDA_ARCH__
  if (i >= d_numMaterials) {
    printf("ERROR: Attempting to access material past bounds\n");
    assert(0);
  }
  return d_materialDB[i].material;
#else
  //I don't know if it makes sense to write this for the host side, when it already exists elsewhere host side.
  printf("getMaterial() is only implemented as a GPU function");
  return IDEAL_GAS; //returning something to prevent a compiler error

#endif
}

HOST_DEVICE void
GPUDataWarehouse::copyGpuGhostCellsToGpuVars() {
#ifndef __CUDA_ARCH__
  //Not for the host side
#else

  //Copy all ghost cells from their source to their destination.
   //The ghost cells could either be only the data that needs to be copied,
   //or it could be on an edge of a bigger grid var.

   //I believe the x,y,z coordinates of everything should match.

   //This could probably be made more efficient by using only perhaps one block,
   //copying float 4s, and doing it with instruction level parallelism.
   int numThreads = blockDim.x*blockDim.y*blockDim.z;
   int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
   int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block
   int totalThreads = numThreads * gridDim.x * gridDim.y * gridDim.z;
   int assignedCellID;

   //go through every ghost cell var we need
   for (int i = 0; i < d_numVarDBItems; i++) {
     //if (threadID == 0) {
     //  if (d_varDB[i].ghostItem.dest_varDB_index != -1) {
     //    printf("d_varDB[%d].label is %s\n", i, d_varDB[d_varDB[i].ghostItem.dest_varDB_index].label, d_numVarDBItems);
     //  } else {
     //    printf("d_varDB[%d].label is %s\n", i, d_varDB[i].label, d_numVarDBItems);
     //  }
     //}
     //some things in d_varDB are meta data for simulation variables
     //other things in d_varDB are meta data for how to copy ghost cells.
     //Make sure we're only dealing with ghost cells here
     if(d_varDB[i].ghostItem.dest_varDB_index != -1) {
       assignedCellID = blockID * numThreads + threadID;
       int destIndex = d_varDB[i].ghostItem.dest_varDB_index;


       int3 ghostCellSize;
       ghostCellSize.x = d_varDB[i].ghostItem.sharedHighCoordinates.x - d_varDB[i].ghostItem.sharedLowCoordinates.x;
       ghostCellSize.y = d_varDB[i].ghostItem.sharedHighCoordinates.y - d_varDB[i].ghostItem.sharedLowCoordinates.y;
       ghostCellSize.z = d_varDB[i].ghostItem.sharedHighCoordinates.z - d_varDB[i].ghostItem.sharedLowCoordinates.z;

       //while there's still work to do (this assigned ID is still within the ghost cell)
       while (assignedCellID < ghostCellSize.x * ghostCellSize.y * ghostCellSize.z ) {
         int z = assignedCellID / (ghostCellSize.x * ghostCellSize.y);
         int temp = assignedCellID % (ghostCellSize.x * ghostCellSize.y);
         int y = temp / ghostCellSize.x;
         int x = temp % ghostCellSize.x;

         assignedCellID += totalThreads;

         //if we're in a valid x,y,z space for the variable.  (It's unlikely every cell will perfectly map onto every available thread.)
         if (x < ghostCellSize.x && y < ghostCellSize.y && z < ghostCellSize.z) {

           //offset them to their true array coordinates, not relative simulation cell coordinates
           //When using virtual addresses, the virtual offset is always applied to the source, but the destination is correct.
           int x_source_real = x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[i].ghostItem.virtualOffset.x - d_varDB[i].var_offset.x;
           int y_source_real = y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[i].ghostItem.virtualOffset.y - d_varDB[i].var_offset.y;
           int z_source_real = z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[i].ghostItem.virtualOffset.z - d_varDB[i].var_offset.z;
           //count over array slots.
           int sourceOffset = x_source_real + d_varDB[i].var_size.x * (y_source_real  + z_source_real * d_varDB[i].var_size.y);

           int x_dest_real = x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[destIndex].var_offset.x;
           int y_dest_real = y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[destIndex].var_offset.y;
           int z_dest_real = z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[destIndex].var_offset.z;


           int destOffset = x_dest_real + d_varDB[destIndex].var_size.x * (y_dest_real + z_dest_real * d_varDB[destIndex].var_size.y);



           //if (threadID == 0) {
           /*   printf("Going to copy, between (%d, %d, %d) from offset %d to offset %d.  From starts at (%d, %d, %d) with size (%d, %d, %d) at index %d pointer %p.  To starts at (%d, %d, %d) with size (%d, %d, %d).\n",
                  d_varDB[i].ghostItem.sharedLowCoordinates.x,
                  d_varDB[i].ghostItem.sharedLowCoordinates.y,
                  d_varDB[i].ghostItem.sharedLowCoordinates.z,
                  sourceOffset,
                  destOffset,
                  d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
                  d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
                  i,
                  d_varDB[i].var_ptr,
                  d_varDB[destIndex].var_offset.x, d_varDB[destIndex].var_offset.y, d_varDB[destIndex].var_offset.z,
                  d_varDB[destIndex].var_size.x, d_varDB[destIndex].var_size.y, d_varDB[destIndex].var_size.z);
            */
            //}

           //copy all 8 bytes of a double in one shot
           if (d_varDB[i].sizeOfDataType == sizeof(double)) {
             *((double*)(d_varDB[destIndex].var_ptr) + destOffset) = *((double*)(d_varDB[i].var_ptr) + sourceOffset);

             //Note: Every now and then I've seen this printf statement get confused, a line will print with the wrong variables/offset variables...
             /*  printf("Thread %d - %s At (%d, %d, %d), real: (%d, %d, %d), copying within region between (%d, %d, %d) and (%d, %d, %d).  Source d_varDB index (%d, %d, %d) varSize (%d, %d, %d) virtualOffset(%d, %d, %d), varOffset(%d, %d, %d), sourceOffset %d actual pointer %p, value %e.   Dest d_varDB index %d ptr %p destOffset %d actual pointer. %p\n",
                   threadID, d_varDB[destIndex].label, x, y, z, x_source_real, y_source_real, z_source_real,
                   d_varDB[i].ghostItem.sharedLowCoordinates.x, d_varDB[i].ghostItem.sharedLowCoordinates.y, d_varDB[i].ghostItem.sharedLowCoordinates.z,
                   d_varDB[i].ghostItem.sharedHighCoordinates.x, d_varDB[i].ghostItem.sharedHighCoordinates.y, d_varDB[i].ghostItem.sharedHighCoordinates.z,
                   x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[i].ghostItem.virtualOffset.x,
                   y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[i].ghostItem.virtualOffset.y,
                   z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[i].ghostItem.virtualOffset.z,
                   d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
                   d_varDB[i].ghostItem.virtualOffset.x, d_varDB[i].ghostItem.virtualOffset.y, d_varDB[i].ghostItem.virtualOffset.z,
                   d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
                   sourceOffset, (double*)(d_varDB[i].var_ptr) + sourceOffset, *((double*)(d_varDB[i].var_ptr) + sourceOffset),
                   destIndex, d_varDB[destIndex].var_ptr,  destOffset, (double*)(d_varDB[destIndex].var_ptr) + destOffset);
             */

           }
           //or copy all 4 bytes of an int in one shot.
           else if (d_varDB[i].sizeOfDataType == sizeof(int)) {
            *(((int*)d_varDB[destIndex].var_ptr) + destOffset) = *((int*)(d_varDB[i].var_ptr) + sourceOffset);
           //Copy each byte until we've copied all for this data type.
           } else {

             for (int j = 0; j < d_varDB[i].sizeOfDataType; j++) {
               *(((char*)d_varDB[destIndex].var_ptr) + (destOffset * d_varDB[destIndex].sizeOfDataType + j))
                   = *(((char*)d_varDB[i].var_ptr) + (sourceOffset * d_varDB[i].sizeOfDataType + j));
             }
           }
         }
       }
     }
   }

#endif
}

 __global__ void copyGpuGhostCellsToGpuVarsKernel( GPUDataWarehouse *gpudw) {
   gpudw->copyGpuGhostCellsToGpuVars();
}

HOST_DEVICE void
GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else

  //see if this GPU datawarehouse has ghost cells in it.
  if (numGhostCellCopiesNeeded > 0) {
    //call a kernel which gets the copy process started.
    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
    const int BLOCKSIZE = 1;
    int xblocks = 32;
    int yblocks = 1;
    int zblocks = 1;
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(1, 1, 1);  //Give each ghost copying kernel 32 * 32 = 1024 threads to copy
    //printf("Launching copyGpuGhostCellsToGpuVarsKernel\n");
    //cudaDeviceSynchronize();



    /*
    //View a variable before and after the ghost cell copy
    {
    cudaDeviceSynchronize();
    //pull out phi01
    Uintah::GPUGridVariable<double> myDeviceVar;
    getModifiable( myDeviceVar, "phi1", 0, 0 );
    double * uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
    printf("Before the device pointer is %p\n", uintahDeviceFieldVar);
    double * hostSideVar = new double[myDeviceVar.getMemSize()/8];
    CUDA_RT_SAFE_CALL(cudaMemcpy((void*)hostSideVar, (void*)uintahDeviceFieldVar, myDeviceVar.getMemSize(), cudaMemcpyDeviceToHost));
    printf("Contents of phi1:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 12; j++) {
        printf("%1.3lf ", hostSideVar[i*12+j]);
      }
      printf("\n");
    }

    delete[] hostSideVar;
    }
    */
    if (gpu_stats.active()) {
     cerrLock.lock();
     {
       gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker() - "
           << " Launching ghost cell copies kernel"
           << " on device " << d_device_id
           << " at GPUDW at " << std::hex << this << std::dec
           << " with description " << _internalName
           << endl;
     }
     cerrLock.unlock();
    }
    copyGpuGhostCellsToGpuVarsKernel<<< dimGrid, dimBlock, 0, *stream >>>(this->d_device_copy);
    //copyGpuGhostCellsToGpuVarsKernel<<< dimGrid, dimBlock >>>(this->d_device_copy);

    //printf("Finished copyGpuGhostCellsToGpuVarsKernel\n");
    //
    /*
    {

    //pull out phi0
    Uintah::GPUGridVariable<double> myDeviceVar;
    getModifiable( myDeviceVar, "phi1", 0, 0 );
    double * uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
    printf("After the device pointer is %p\n", uintahDeviceFieldVar);
    double * hostSideVar = new double[myDeviceVar.getMemSize()/8];
    CUDA_RT_SAFE_CALL(cudaMemcpy((void*)hostSideVar, (void*)uintahDeviceFieldVar, myDeviceVar.getMemSize(), cudaMemcpyDeviceToHost));
    printf("Contents of phi1:\n");
    for (int i = 0; i < 12; i++) {
      for (int j = 0; j < 12; j++) {
        printf("%1.3lf ", hostSideVar[i*12+j]);
      }
      printf("\n");
    }

    delete[] hostSideVar;
    }
    */


  }

#endif
}



HOST_DEVICE bool
GPUDataWarehouse::ghostCellCopiesNeeded()
{
#ifdef __CUDA_ARCH__
  //Not implemented for the device side
  printError("This method not allowed on the device.", "ghostCellCopiesNeeded");
  return false;
#else

  //see if this GPU datawarehouse has ghost cells in it.
  return (numGhostCellCopiesNeeded > 0);

#endif
}

HOST_DEVICE void
GPUDataWarehouse::putGhostCell(char const* label, int sourcePatchID, int destPatchID, int matlIndx, int levelIndx,
                               bool sourceStaging, bool destStaging,
                               int3 varOffset, int3 varSize,
                               int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset) {
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::putGhostCell( %s )  Not implemented for GPU\n",label);
#else
  //Add information describing a ghost cell that needs to be copied internally from
  //one chunk of data to the destination.  This covers a GPU -> same GPU copy scenario.
  varLock->writeLock();
  int i = d_numVarDBItems;
  if (i > maxdVarDBItems) {
    printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Exceeded maximum d_varDB entries.  Index is %d and max items is %d\n", i, maxdVarDBItems);
    varLock->writeUnlock();
    exit(-1);
  }
  int index = -1;
  d_numVarDBItems++;
  numGhostCellCopiesNeeded++;
  d_varDB[i].ghostItem.sharedLowCoordinates = sharedLowCoordinates;
  d_varDB[i].ghostItem.sharedHighCoordinates = sharedHighCoordinates;
  d_varDB[i].ghostItem.virtualOffset = virtualOffset;

  //look up the source index and the destination index for these.
  //it may be an entire variable (in which case staging is false)
  //or it may be a staging variable.
  labelPatchMatlLevel lpml_source(label, sourcePatchID, matlIndx, levelIndx);
  if (!sourceStaging) {

    if (varPointers->find(lpml_source) != varPointers->end()) {
      index = varPointers->operator[](lpml_source).varDB_index;
    }
  } else {
    //Find the variable that contains the region in which our ghost cells exist.
    //Usually the sharedLowCoordinates and sharedHighCoordinates correspond
    //exactly to the size of the staging variable.  But sometimes the ghost data is found within
    //a larger staging variable.
    stagingVar sv;
    sv.device_offset = varOffset;
    sv.device_size = varSize;

    std::map<stagingVar, stagingVarInfo>::iterator staging_it = varPointers->operator[](lpml_source).stagingVars.find(sv);
    if (staging_it != varPointers->operator[](lpml_source).stagingVars.end()) {

      index = staging_it->second.varDB_index;

    } else {
      printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Number of staging vars for this var: %d, No staging variable found label %s patch %d matl %d level %d offset (%d, %d, %d) size (%d, %d, %d) on DW at %p.\n",
                    label, varPointers->operator[](lpml_source).stagingVars.size(), label, sourcePatchID, matlIndx, levelIndx,
                    sv.device_offset.x, sv.device_offset.y, sv.device_offset.z,
                    sv.device_size.x, sv.device_size.y, sv.device_size.z,
                    this);
      varLock->writeUnlock();
      exit(-1);
    }
    //Find the d_varDB entry for this specific one.


  }

  if (index < 0) {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label %s, source patch ID %d, matlIndx %d, levelIndex %d staging %s not found in GPU DW %p\n",
        label, sourcePatchID, matlIndx, levelIndx, sourceStaging ? "true" : "false", this);
    varLock->writeUnlock();
    exit(-1);
  }
  //printf("The found index %d for var %s patch %d matl %d\n", index, label, sourcePatchID, matlIndx);
  //if (d_varDB[index].varItem.validOnGPU == false) {
    //Steps prior to this point should have checked for this scenario.
    //This is just a failsafe.
  //  printf("ERROR:\nGPUDataWarehouse::putGhostCell, attempting to use: label %s, source patch ID %d, materialID %d, it exists but the data is not valid.\n", label, sourcePatchID, matlIndx);
  //  exit(-1);
  //}
  d_varDB[i].var_offset = d_varDB[index].var_offset;
  d_varDB[i].var_size = d_varDB[index].var_size;
  d_varDB[i].var_ptr = d_varDB[index].var_ptr;
  d_varDB[i].sizeOfDataType = d_varDB[index].sizeOfDataType;
  if (gpu_stats.active()) {
   cerrLock.lock();
   {
     gpu_stats << UnifiedScheduler::myRankThread()
         << " GPUDataWarehouse::putGhostCell() - "
         << " Placed into d_varDB at index " << i << " of max index " << maxdVarDBItems - 1
         << " from patch " << sourcePatchID << " staging " << sourceStaging << " to patch " << destPatchID << " staging " << destStaging
         << " has shared coordinates (" << sharedLowCoordinates.x << ", " << sharedLowCoordinates.y << ", " << sharedLowCoordinates.z << "),"
         << " (" << sharedHighCoordinates.x << ", " << sharedHighCoordinates.y << ", " << sharedHighCoordinates.z << "), "
         << " from low/offset (" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << ") "
         << " size (" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << ") "
         << " virtualOffset (" << d_varDB[i].ghostItem.virtualOffset.x << ", " << d_varDB[i].ghostItem.virtualOffset.y << ", " << d_varDB[i].ghostItem.virtualOffset.z << ") "
         << " datatype size " << d_varDB[i].sizeOfDataType
         << " on device " << d_device_id
         << " at GPUDW at " << std::hex << this<< std::dec
         << endl;
   }
   cerrLock.unlock();
  }

  //if (d_debug){
  //  printf("Placed into d_varDB at index %d from patch %d to patch %d has shared coordinates (%d, %d, %d), (%d, %d, %d), from low/offset (%d, %d, %d) size (%d, %d, %d)  virtualOffset(%d, %d, %d)\n",
  //      i, sourcePatchID, destPatchID, sharedLowCoordinates.x, sharedLowCoordinates.y,
  //      sharedLowCoordinates.z, sharedHighCoordinates.x, sharedHighCoordinates.y, sharedHighCoordinates.z,
  //      d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
  //      d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
  //      d_varDB[i].ghostItem.virtualOffset.x, d_varDB[i].ghostItem.virtualOffset.y, d_varDB[i].ghostItem.virtualOffset.z);
  //}



  //Find where we are sending the ghost cell data to
  labelPatchMatlLevel lpml_dest(label, destPatchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml_dest);
  if (it != varPointers->end()) {
    if (destStaging) {
      //TODO: Do the same thing as the source.
      //If the destination is staging, then the shared coordinates are also the ghost coordinates.
      stagingVar sv;
      sv.device_offset = sharedLowCoordinates;
      sv.device_size = make_int3(sharedHighCoordinates.x-sharedLowCoordinates.x,
                                 sharedHighCoordinates.y-sharedLowCoordinates.y,
                                 sharedHighCoordinates.z-sharedLowCoordinates.z);

      std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
      if (staging_it != it->second.stagingVars.end()) {
        d_varDB[i].ghostItem.dest_varDB_index = staging_it->second.varDB_index;
      } else {
        printf("\nERROR:\nGPUDataWarehouse::putGhostCell() didn't find a staging variable from the device for offset (%d, %d, %d) and size (%d, %d, %d).\n",
            sharedLowCoordinates.x, sharedLowCoordinates.y, sharedLowCoordinates.z,
            sv.device_size.x, sv.device_size.y, sv.device_size.z);
        varLock->writeUnlock();
        exit(-1);
      }

    } else {
      d_varDB[i].ghostItem.dest_varDB_index = it->second.varDB_index;
    }
    //if (d_debug){
    //  int destIndex = d_varDB[i].ghostItem.dest_varDB_index;
    //  printf("The destination ghost cell copy is at d_varDB at index %d with size (%d, %d, %d), offset (%d, %d, %d)\n",
    //      destIndex,
    //      d_varDB[destIndex].var_size.x, d_varDB[destIndex].var_size.y, d_varDB[destIndex].var_size.z,
    //      d_varDB[destIndex].var_offset.x, d_varDB[destIndex].var_offset.y, d_varDB[destIndex].var_offset.z);
    //}
  } else {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell(), label: %s destination patch ID %d, matlIndx %d, levelIndex %d, staging %s not found in GPU DW variable database\n",
        label, destPatchID, matlIndx, levelIndx, destStaging ? "true" : "false");
    varLock->writeUnlock();
    exit(-1);
  }

  d_dirty=true;
  varLock->writeUnlock();
#endif
}

HOST_DEVICE void
GPUDataWarehouse::getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells,
  char const* label, int patchID, int matlIndx, int levelIndx) {
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::getSizes()  Not implemented for GPU\n");
#else
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo info = varPointers->operator[](lpml);
    low = info.device_offset;
    high.x = info.device_size.x - info.device_offset.x;
    high.y = info.device_size.y - info.device_offset.y;
    high.z = info.device_size.z - info.device_offset.z;
    siz = info.device_size;
    gtype = info.gtype;
    numGhostCells = info.numGhostCells;
  }
  varLock->readUnlock();

#endif
}
/*
HOST_DEVICE void GPUDataWarehouse::getTempGhostCells(void * dtask, std::vector<tempGhostCellInfo>& temp) {
#ifdef __CUDA_ARCH__
    printf("ERROR:\nGPUDataWarehouse::getTempGhostCells not implemented for GPU\n");
    exit(-1);
#else
  varLock->readLock();


  for ( vector<tempGhostCellInfo>::iterator it = tempGhostCells.begin();
        it != tempGhostCells.end();
        ++it) {
    //only this task should process its own outgoing GPU->other destination ghost cell copies
    if (dtask == (*it).cpuDetailedTaskOwner) {
      temp.push_back( (*it) );
    }
  }
  varLock->readUnlock();
#endif
}*/

//Go through all staging vars for a var. See if they are all marked as valid.
__host__ bool GPUDataWarehouse::areAllStagingVarsValid(char const* label, int patchID, int matlIndx, int levelIndx) {
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    for (std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.begin();
         staging_it != it->second.stagingVars.end();
         ++staging_it) {
     if (!checkValid(staging_it->second.atomicStatusInGpuMemory)) {
       varLock->readUnlock();
       if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread() << " GPUDataWarehouse::areAllStagingVarsValid() -"
               // Task: " << dtask->getName()
               << " Not all staging vars were ready for "
               << label << " patch " << patchID
               << " material " << matlIndx << " level " << levelIndx
               << " offset (" << staging_it->first.device_offset.x
               << ", " << staging_it->first.device_offset.y
               << ", " << staging_it->first.device_offset.z
               << ") and size (" << staging_it->first.device_size.x
               << ", " << staging_it->first.device_size.y
               << ", " << staging_it->first.device_size.z
               << ") with status codes " << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory) <<endl;
         }
         cerrLock.unlock();
       }
       return false;
     }
   }
  }
  varLock->readUnlock();
  return true;
}


//Simply performs an atomic fetch on the status variable.
typedef int atomicDataStatus;
__host__ atomicDataStatus
GPUDataWarehouse::getStatus(atomicDataStatus& status) {
  return __sync_or_and_fetch(&(status), 0);
}

__host__ string
GPUDataWarehouse::getDisplayableStatusCodes(atomicDataStatus& status) {
  atomicDataStatus varStatus  = __sync_or_and_fetch(&(status), 0);
  string retval = "";
  if (varStatus == 0) {
    retval += "Unallocated ";
  } else {
    if ((varStatus & ALLOCATING) == ALLOCATING) {
      retval += "Allocating ";
    }
    if ((varStatus & ALLOCATED) == ALLOCATED) {
      retval += "Allocated ";
    }
    if ((varStatus & COPYING_IN) == COPYING_IN) {
      retval += "Copying-in ";
    }
    if ((varStatus & VALID) == VALID) {
      retval += "Valid ";
    }
    if ((varStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) {
      retval += "Awaiting-ghost-copy ";
    }
    if ((varStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) {
      retval += "Valid-with-ghosts ";
    }
    if ((varStatus & UNKNOWN) == UNKNOWN) {
      retval += "Unknown ";
    }
  }
  return retval;
}


//returns false if something else already allocated space and we don't have to.
//returns true if we are the ones to allocate the space.
//performs operations with atomic compare and swaps
__host__ bool
GPUDataWarehouse::testAndSetAllocating(atomicDataStatus& status)
{

  bool allocating = false;

  while (!allocating) {

    //get the value
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(&(status), 0);
    //if it's allocated, return true
    if (((oldVarStatus & ALLOCATING) == ALLOCATING) || ((oldVarStatus & ALLOCATED) == ALLOCATED)) {
      //Something else already allocated or is allocating it.  So this thread won't do do any allocation.
      return false;
    } else {
      //Attempt to claim we'll allocate it.  If not go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | ALLOCATING;
      allocating = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);

    }
  }
  return true;
}

//Sets the allocated flag on a variables atomicDataStatus
//This is called after an allocation completes.
__host__ bool
GPUDataWarehouse::testAndSetAllocate(atomicDataStatus& status)
{

  bool allocated = false;

  //get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&(status), 0);
  if ((oldVarStatus & ALLOCATING) == 0) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::testAndSetAllocate( )  Can't allocate a status if it wasn't previously marked as allocating.\n");
    exit(-1);
  } else if  ((oldVarStatus & ALLOCATED) == ALLOCATED) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::testAndSetAllocate( )  Can't allocate a status if it's already allocated\n");
    exit(-1);
  }
  else {
    //Attempt to claim we'll allocate it.  Create what we want the status to look like
    //by turning off allocating and turning on allocated.
    //Note: No need to turn off UNALLOCATED, it's defined as all zero bits.
    //But the below is kept in just for readability's sake.
    atomicDataStatus newVarStatus = oldVarStatus & ~UNALLOCATED;
    newVarStatus = newVarStatus & ~ALLOCATING;
    newVarStatus = newVarStatus | ALLOCATED;

    //If we succeeded in our attempt to claim to allocate, this returns true.
    //If we failed, thats a real problem, and we crash the problem below.
    allocated = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
  }
  if (!allocated) {
    //Another sanity check
    printf("ERROR:\nGPUDataWarehouse::testAndSetAllocate( )  Something wrongly modified the atomic status while setting the allocated flag\n");
    exit(-1);
  }
  return allocated;
}



//Simply determines if a variable has been marked as allocated.
__host__ bool
GPUDataWarehouse::checkAllocated(atomicDataStatus& status)
{

  return ((__sync_or_and_fetch(&(status), 0) & ALLOCATED) == ALLOCATED);
}

//Simply determines if a variable has been marked as valid.
__host__ bool
GPUDataWarehouse::checkValid(atomicDataStatus& status)
{

  return ((__sync_or_and_fetch(&(status), 0) & VALID) == VALID);
}
/*
//returns false if something else already is copying in or copied the data and we don't have to.
//returns true if we are the ones to copy in the data.
__host__ bool
GPUDataWarehouse::testAndSetCopying(char const* label, int patchID, int matlIndx, int levelIndx)
{

  atomicDataStatus *status;



  bool copying = false;

  while (!copying) {

    //get the address
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      status = &(varPointers->operator[](lpml).atomicStatusInGpuMemory);
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopying( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
    varLock->readUnlock();

    //get the value
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(&(status), 0);
    if (oldVarStatus != ALLOCATED) {
      //Sanity check.  Nobody should attempt a copy if any flag other than just ALLOCATED is set.
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopying( )  Attempting to copy with a variable whose status is set to something other than just allocated.\n");
        exit(-1);
    }
    //if it's allocated, return true
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) || ((oldVarStatus & VALID) == VALID)) {
      //Something else already copied or is copying it.  So this thread won't launch any copies.
      return false;
    } else {
      //Attempt to claim we'll copy it.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copying = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}
*/
__host__ bool
GPUDataWarehouse::isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), 0) & ALLOCATED) == ALLOCATED);
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
}

__host__ bool
GPUDataWarehouse::isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //cout << "In isAllocatedOnGPU - For patchID " << patchID << " for the status is " << getDisplayableStatusCodes(varPointers->operator[](lpml).atomicStatusInGpuMemory) << endl;
    bool retVal = ((__sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), 0) & ALLOCATED) == ALLOCATED);
    if (retVal) {
      //now check the sizes
      int3 device_offset = varPointers->operator[](lpml).device_offset;
      int3 device_size = varPointers->operator[](lpml).device_size;
      retVal = (device_offset.x == offset.x && device_offset.y == offset.y && device_offset.z == offset.z
                && device_size.x == size.x && device_size.y == size.y && device_size.z == size.z);
    }
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
}

__host__ bool
GPUDataWarehouse::isValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), 0) & VALID) == VALID);
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
}

__host__ void
GPUDataWarehouse::setValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    __sync_and_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), ~COPYING_IN);
    __sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), VALID);

    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnGPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
}

__host__ void
GPUDataWarehouse::setValidOnGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
  varLock->writeLock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;

    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
    if (staging_it != it->second.stagingVars.end()) {
      __sync_and_and_fetch(&(staging_it->second.atomicStatusInGpuMemory), ~COPYING_IN);
      __sync_or_and_fetch(&(staging_it->second.atomicStatusInGpuMemory), VALID);
    } else {
      varLock->writeUnlock();
      printf("ERROR:\nGPUDataWarehouse::setValidOnGPUStaging( )  Staging variable %s not found.\n", label);
      exit(-1);
    }
  } else {
    varLock->writeUnlock();
    printf("ERROR:\nGPUDataWarehouse::setValidOnGPUStaging( )  Variable %s not found.\n", label);
    exit(-1);
  }
  varLock->writeUnlock();
}


//We have an entry for this item in the GPU DW, and it's not unknown.  Therefore
//if this returns true it means this GPU DW specifically knows something about the
//state of this variable. (The reason for the unknown check is currently when a
//var is added to the GPUDW, we also need to state what we know about its data in
//host memory.  Since it doesn't know, it marks it as unknown, meaning, the host
//side DW is possibly managing the data.)
__host__ bool GPUDataWarehouse::dwEntryExistsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx) {

   varLock->readLock();
   bool retVal = false;
   labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
   std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
   if (it != varPointers->end()) {
     if  ((it->second.atomicStatusInHostMemory & UNKNOWN) != UNKNOWN) {

       retVal = true;
     }
   }
   varLock->readUnlock();
   return retVal;

}

__host__ bool
GPUDataWarehouse::isValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {


    bool retVal = ((__sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInHostMemory), 0) & VALID) == VALID);
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
}

__host__ void
GPUDataWarehouse::setValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    __sync_and_and_fetch(&(varPointers->operator[](lpml).atomicStatusInHostMemory), ~COPYING_IN);
    __sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInHostMemory), VALID);
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnCPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
}


__host__ void
GPUDataWarehouse::setAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    __sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), AWAITING_GHOST_COPY);
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setAwaitingGhostDataOnGPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
}


//returns false if something else already changed a valid variable to valid awaiting ghost data
//returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::testAndSetAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool allocating = false;

  atomicDataStatus *status;

  while (!allocating) {
    //get the adress
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      status = &(varPointers->operator[](lpml).atomicStatusInGpuMemory);
      varLock->readUnlock();
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetAwaitingGhostDataOnGPU( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }

    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (((oldVarStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) || ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | AWAITING_GHOST_COPY;
      allocating = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);

    }
  }

  return true;
}



//returns false if something else already claimed to copy or has copied data into the GPU.
//returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::testAndSetCopyingIntoGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool copyingin = false;

  atomicDataStatus *status;

  while (!copyingin) {
    //get the adress
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      status = &(varPointers->operator[](lpml).atomicStatusInGpuMemory);
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPU( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
    varLock->readUnlock();
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPU( )  Variable %s is unallocated.\n", label);
      exit(-1);
    }
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
        ((oldVarStatus & VALID) == VALID) ||
        ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);

    }
  }

  return true;
}


//returns false if something else already claimed to copy or has copied data into the CPU.
//returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::testAndSetCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool copyingin = false;

  atomicDataStatus *status;

  while (!copyingin) {
    //get the adress
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      status = &(varPointers->operator[](lpml).atomicStatusInHostMemory);
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoCPU( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
    varLock->readUnlock();
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    //We don't have good tracking of CPU vars at the moment.
    //if (oldVarStatus == UNALLOCATED) {
    //  printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoCPU( )  Variable %s is unallocated.\n", label);
    //  exit(-1);
    //}
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
        ((oldVarStatus & VALID) == VALID) ||
        ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);

    }
  }

  return true;
}


//returns false if something else already claimed to copy or has copied data into the GPU.
//returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::testAndSetCopyingIntoGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{

  bool copyingin = false;

  atomicDataStatus *status;

  while (!copyingin) {
    //get the address
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

    if (it != varPointers->end()) {

      stagingVar sv;
      sv.device_offset = offset;
      sv.device_size = size;

      std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
      if (staging_it != it->second.stagingVars.end()) {
        status = &(staging_it->second.atomicStatusInGpuMemory);
      } else {
        varLock->readUnlock();
        printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPUStaging( )  Staging variable %s not found.\n", label);
        exit(-1);
        return false;
      }
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPUStaging( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
    varLock->readUnlock();
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPUStaging( )  Variable %s is unallocated.\n", label);
      exit(-1);
    } else if ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) {
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopyingIntoGPUStaging( )  Variable %s is marked as valid with ghosts, that should never happen with staging vars.\n", label);
      exit(-1);
    } else if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
               ((oldVarStatus & VALID) == VALID)) {
      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }

  return true;
}

/*
//returns false if something else already is copying in or copied the data and we don't have to.
//returns true if we are the ones to copy in the data.
__host__ bool
GPUDataWarehouse::testAndSetCopying(char const* label, int patchID, int matlIndx, int levelIndx)
{

  atomicDataStatus *status;



  bool copying = false;

  while (!copying) {

    //get the address
    varLock->readLock();
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      status = &(varPointers->operator[](lpml).atomicStatusInGpuMemory);
    } else {
      varLock->readUnlock();
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopying( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
    varLock->readUnlock();

    //get the value
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(&(status), 0);
    if (oldVarStatus != ALLOCATED) {
      //Sanity check.  Nobody should attempt a copy if any flag other than just ALLOCATED is set.
      printf("ERROR:\nGPUDataWarehouse::testAndSetCopying( )  Attempting to copy with a variable whose status is set to something other than just allocated.\n");
        exit(-1);
    }
    //if it's allocated, return true
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) || ((oldVarStatus & VALID) == VALID)) {
      //Something else already copied or is copying it.  So this thread won't launch any copies.
      return false;
    } else {
      //Attempt to claim we'll copy it.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copying = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}
*/

__host__ bool
GPUDataWarehouse::isValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), 0) & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    varLock->readUnlock();
    return retVal;
  } else {
    varLock->readUnlock();
    return false;
  }
}

__host__ void
GPUDataWarehouse::setValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {

    //make sure the valid is still turned on
    __sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), VALID);
    //turn off AWAITING_GHOST_COPY
    __sync_and_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), ~AWAITING_GHOST_COPY);
    //turn on VALID_WITH_GHOSTS
    __sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), VALID_WITH_GHOSTS);
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    exit(-1);
  }
}





//______________________________________________________________________
//
__device__ void
GPUDataWarehouse::print()
{
#ifdef __CUDA_ARCH__
  __syncthreads();
  if( isThread0_Blk0() ){
    printf("\nVariables in GPUDataWarehouse\n");
    for (int i = 0; i < d_numVarDBItems; i++) {
      dataItem me = d_varDB[i];
      printf("    %-15s matl: %i, patchID: %i, L-%i, size:[%i,%i,%i] pointer: %p\n", me.label, me.matlIndx,
             me.domainID, me.levelIndx, me.var_size.x, me.var_size.y, me.var_size.z, me.var_ptr);
    }
    __syncthreads();

    printThread();
    printBlock();
    printf("\n");

  }
#endif
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printError(const char* msg, const char* methodName, char const* label, int patchID, int matlIndx, int levelIndx )
{
#ifdef __CUDA_ARCH__
  __syncthreads();
  if( isThread0() ){
    if (label[0] == '\0') {
      printf("  \nERROR GPU-side: GPUDataWarehouse::%s() - %s\n", methodName, msg );
    } else {
      printf("  \nERROR GPU-side: GPUDataWarehouse::%s(), label:  \"%s\", patch: %i, matlIndx: %i, levelIndx: %i - %s\n", methodName, label, patchID, matlIndx, levelIndx, msg);
    }
    //Should this just loop through the variable database and print out only items with a
    //levelIndx value greater than zero? -- Brad

    //for (int i = 0; i < d_numLevelItems; i++) {
    //  printf("   Available levelDB labels(%i): \"%-15s\" matl: %i, L-%i \n", d_numLevelItems, d_levelDB[i].label, d_levelDB[i].matlIndx, d_levelDB[i].levelIndx);
    // }
    __syncthreads();

    printThread();
    printBlock();

    // we know this is fatal and why, so just stop kernel execution
    __threadfence();
    asm("trap;");
  }
#else
  //__________________________________
  //  CPU code
  if (label[0] == '\0') {
    printf("  \nERROR host-side: GPUDataWarehouse::%s() - %s\n", methodName, msg );
  } else {
    printf("  \nERROR host-side: GPUDataWarehouse::%s(), label:  \"%s\", patch: %i, matlIndx: %i, levelIndx: %i - %s\n", methodName, label, patchID, matlIndx, levelIndx, msg);
  }//for (int i = 0; i < d_numLevelItems; i++) {
  //  printf("   Available levelDB labels(%i): \"%-15s\" matl: %i, L-%i \n", d_numLevelItems, d_levelDB[i].label, d_levelDB[i].matlIndx, d_levelDB[i].levelIndx);
  //}
  exit(-1);
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
    printf("  \nERROR: %s( \"%s\", levelIndx: %i, matl: %i)  unknown variable\n", msg,  label, levelIndx, matlIndx);
    //Should this just loop through the variable database and print out only items with a
    //levelIndx value greater than zero? -- Brad

    //for (int i = 0; i < d_numLevelItems; i++) {
    //  printf("   Available levelDB labels(%i): \"%-15s\" matl: %i, L-%i \n", d_numLevelItems, d_levelDB[i].label, d_levelDB[i].matlIndx, d_levelDB[i].levelIndx);
    // }
    __syncthreads();

    printThread();
    printBlock();

    // we know this is fatal and why, so just stop kernel execution
    __threadfence();
    asm("trap;");
  }
#else
  //__________________________________
  //  CPU code
  printf("  \nERROR: %s( \"%s\", levelIndx: %i, matl: %i)  unknown variable\n", msg, label, levelIndx, matlIndx);
  //for (int i = 0; i < d_numLevelItems; i++) {
  //  printf("   Available levelDB labels(%i): \"%-15s\" matl: %i, L-%i \n", d_numLevelItems, d_levelDB[i].label, d_levelDB[i].matlIndx, d_levelDB[i].levelIndx);
  //}
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printGetError(const char* msg, char const* label, int levelIndx, int patchID, int matlIndx)
{
#ifdef __CUDA_ARCH__
  __syncthreads();
  if( isThread0() ) {
    printf("  \nERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i)  unknown variable\n",
            msg,  label, levelIndx, patchID, matlIndx);

    for (int i = 0; i < d_numVarDBItems; i++) {
      printf("   Available varDB labels(%i of %i): \"%-15s\" matl: %i, patchID: %i, level: %i\n", i, d_numVarDBItems, d_varDB[i].label, d_varDB[i].matlIndx,
             d_varDB[i].domainID, d_varDB[i].levelIndx);
    }
    __syncthreads();

    printThread();
    printBlock();
    printf("\n");

    // we know this is fatal and why, so just stop kernel execution
    __threadfence();
    asm("trap;");
  }
#else
  //__________________________________
  //  CPU code
  printf("  \nERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i)  unknown variable in DW %s\n",
         msg, label, levelIndx, patchID, matlIndx, _internalName);
  for (int i = 0; i < d_numVarDBItems; i++) {
    printf("   Available varDB labels(%i): \"%-15s\" matl: %i, patchID: %i, level: %i\n", d_numVarDBItems, d_varDB[i].label, d_varDB[i].matlIndx,
           d_varDB[i].domainID, d_varDB[i].levelIndx);
  }
#endif
}


HOST_DEVICE void*
GPUDataWarehouse::getPlacementNewBuffer() 
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::getPlacementNewBuffer() not for device code\n");
  return NULL;
#else
  return placementNewBuffer;
#endif
}

} // end namespace Uintah
