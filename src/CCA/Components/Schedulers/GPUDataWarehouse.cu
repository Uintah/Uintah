/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Schedulers/GPUMemoryPool.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GPUPerPatch.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>


#include <sci_defs/cuda_defs.h>

#ifndef __CUDA_ARCH__
  #include <string.h>
  #include <string>
#endif

#include <map>

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream gpu_stats;
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUGridVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
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
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->at(lpml);
    var.setArray3(vp.var->device_offset, vp.var->device_size, vp.var->device_ptr);
  }
  else {
    printf("I'm GPUDW with name: \"%s\" at %p \n", _internalName, this);
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::stagingVarExists(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
#ifdef __CUDA_ARCH__
  // device code
  printError("This method not defined for the device.", "stagingVarExists", label, patchID, matlIndx, levelIndx);
  return false;
#else
  // host code
  varLock->lock();
  bool retval = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.find(sv);
    retval = (staging_it != it->second.var->stagingVars.end());
  }
  varLock->unlock();
  return retval;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getStagingVar(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
#ifdef __CUDA_ARCH__
  // device code
  printError("This method not defined for the device.", "getStagingVar", label, patchID, matlIndx, levelIndx);
#else
  // host code
  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.find(sv);
    if (staging_it != it->second.var->stagingVars.end()) {
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
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getLevel(const GPUGridVariableBase& var, char const* label, int8_t matlIndx, int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  get(var, label, -99999999, matlIndx, levelIndx);
#else
  // host code
  get(var, label, -99999999, matlIndx, levelIndx);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->at(lpml);
    var.setData(vp.var->device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUPerPatchBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->at(lpml);
    var.setData(vp.var->device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    var.setArray3(it->second.var->device_offset, it->second.var->device_size, it->second.var->device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID,  matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->at(lpml);
    var.setData(vp.var->device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{
#ifdef __CUDA_ARCH__
  // device code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndx, levelIndx);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
#else
  // host code
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo vp = varPointers->at(lpml);
    var.setData(vp.var->device_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->unlock();
#endif
}

//______________________________________________________________________
//This method assumes the base patch in a superpatch region has already been allocated.
//This is a shallow copy.  It copies all datawarehouse metadata entries (except the status)
//from that item into this patch's item in the GPU DW.
__host__ void
GPUDataWarehouse::copySuperPatchInfo(char const* label, int superPatchBaseID, int superPatchDestinationID, int matlIndx, int levelIndx) {

   if (superPatchBaseID == superPatchDestinationID) {
     //don't handle shallow copying itself
     return;
   }
   //Possible TODO: Add in offsets so the variable could be accessed in a non-superpatch manner.

   labelPatchMatlLevel lpml_source(label, superPatchBaseID, matlIndx, levelIndx);
   labelPatchMatlLevel lpml_dest(label, superPatchDestinationID, matlIndx, levelIndx);


   varLock->lock();
   std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator source_iter = varPointers->find(lpml_source);
   if (source_iter != varPointers->end()) {
     std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator dest_iter = varPointers->find(lpml_dest);
     if (dest_iter != varPointers->end()) {


       if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread()
               << " GPUDataWarehouse::copySuperPatchInfo() - "
               << " label " << label
               << " matl " << matlIndx
               << " level " << levelIndx
               << " Forming a superpatch by merging/shallowcopying metadata for patch " << superPatchDestinationID
               << " into patch " << superPatchBaseID
               << " with source status codes " << getDisplayableStatusCodes(source_iter->second.var->atomicStatusInGpuMemory)
               << " and dest status codes " << getDisplayableStatusCodes(dest_iter->second.var->atomicStatusInGpuMemory)
               << " on device " << d_device_id
               << " into GPUDW at " << std::hex << this << std::dec
               << " with description " << _internalName
               << std::endl;
         }
         cerrLock.unlock();
       }

       //They now share the variable.  The magic of this happens because the var is a C++ shared_ptr
       //TODO: They don't share the same offset.  When offsets are added in, this should be updated
       //to manage offsets.
       dest_iter->second.var = source_iter->second.var;

     } else {
       printf("ERROR: GPUDataWarehouse::copySuperPatchInfo() - Didn't find a the destination ID at %d to copy into label %s patch %d matl %d level %d\n",
           superPatchDestinationID, label, superPatchDestinationID, matlIndx, levelIndx);
       varLock->unlock();
       exit(-1);
     }
   } else {
     printf("ERROR: GPUDataWarehouse::copySuperPatchInfo() - Didn't find a base superPatch ID at %d to copy into label %s patch %d matl %d level %d\n",
         superPatchBaseID, label, superPatchDestinationID, matlIndx, levelIndx);
     varLock->unlock();
     exit(-1);
   }
   varLock->unlock();

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUGridVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, bool staging,
                      GhostType gtype, int numGhostCells, void* host_ptr)
{

  varLock->lock();

  int3 var_offset;        // offset
  int3 var_size;          // dimensions of GPUGridVariable
  void* var_ptr;           // raw pointer to the memory

  var.getArray3(var_offset, var_size, var_ptr);

  // See if it already exists.  Also see if we need to update this into d_varDB.
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
    staging_it = iter->second.var->stagingVars.find(sv);
    if (staging_it == iter->second.var->stagingVars.end()) {
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
          gpu_stats << getDisplayableStatusCodes(iter->second.var->atomicStatusInGpuMemory);
        } else {
          gpu_stats << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory);
        }
        gpu_stats << " datatype size " << sizeOfDataType
        << " on device " << d_device_id
        << " into GPUDW at " << std::hex << this << std::dec
        << " with description " << _internalName
        << " current varPointers size is: " << varPointers->size()
        << " low (" << var_offset.x << ", " << var_offset.y << ", " << var_offset.z << ") "
        << std::endl;
    }
    cerrLock.unlock();
  }

  if (staging == false) {

    iter->second.varDB_index = -1;
    iter->second.var->device_ptr = var_ptr;
    iter->second.var->device_offset =  var_offset;
    iter->second.var->device_size = var_size;
    iter->second.var->sizeOfDataType = sizeOfDataType;
    iter->second.var->gtype = gtype;
    iter->second.var->numGhostCells = numGhostCells;
    iter->second.var->host_contiguousArrayPtr = host_ptr;
    iter->second.var->atomicStatusInHostMemory = UNKNOWN;

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
            << " with datatype size " << iter->second.var->sizeOfDataType
            << " with status codes " << getDisplayableStatusCodes(iter->second.var->atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << " current varPointers size is: " << varPointers->size()
            << std::endl;
      }
      cerrLock.unlock();
    }

  } else { // if (staging == true)


    staging_it->second.device_ptr = var_ptr;
    staging_it->second.host_contiguousArrayPtr = host_ptr;
    staging_it->second.varDB_index = -1;
    staging_it->second.atomicStatusInHostMemory = UNKNOWN;

    // Update the non-staging var's sizeOfDataType.  The staging var uses this number.
    // It's possible that a staging var can exist and an empty placeholder non-staging var also exist,
    // if so, then then empty placeholder non-staging var won't have correct data type size.
    // So we grab it here.
    iter->second.var->sizeOfDataType = sizeOfDataType;

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
            << " with datatype size " << iter->second.var->sizeOfDataType
            << " with status codes " << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory)
            << " on device " << d_device_id

            << " into GPUDW at " << std::hex << this << std::dec
            << std::endl;
      }
      cerrLock.unlock();
    }

  }

  varLock->unlock();

}

//______________________________________________________________________
// This method puts an empty placeholder entry into the GPUDW database and marks it as unallocated
__host__ void
GPUDataWarehouse::putUnallocatedIfNotExists(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 offset, int3 size)
{

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  // If it's a normal non-staging variable, check if doesn't exist.  If so, add an "unallocated" entry.
  // If it's a staging variable, then still check if the non-staging part exists.  A staging must exist within a non-staging variable.
  // A scenario where this can get a staging variable without a non-staging variable is receiving data from neighbor nodes.
  // For example, suppose node A has patch 0, and node B has patch 1, and A's patch 0 needs ghost cells from B's patch 1.  Node A will
  // receive those ghost cells, but they will be marked as belonging to patch 1.  Since A doesn't have the regular non-staging var
  // for patch 1, we make an empty placeholder for patch 1 so A can have a staging var to hold the ghost cell for patch 1.

  if ( it == varPointers->end()) {
    // Do not place size information.  The Data Warehouse should not declare its current size until after the allocation is complete.
    // Further, no scheduler thread should attempt to determine an entry's size until the allocated flag has been marked as true.
    allVarPointersInfo vp;

    vp.varDB_index = -1;
    vp.var->device_ptr = nullptr;
    vp.var->atomicStatusInHostMemory = UNKNOWN;
    vp.var->atomicStatusInGpuMemory = UNALLOCATED;
    vp.var->host_contiguousArrayPtr = nullptr;
    vp.var->sizeOfDataType = 0;

    std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator, bool> ret
                   = varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    if (!ret.second) {
      printf("ERROR:\nGPUDataWarehouse::putUnallocatedIfNotExists( ) Failure inserting into varPointers map.\n");
      varLock->unlock();
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
            << std::endl;
      }
      cerrLock.unlock();
    }

  }

  if (staging) {
    std::map<stagingVar, stagingVarInfo>::iterator staging_it;

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.var->stagingVars.find(sv);
    if (staging_it == it->second.var->stagingVars.end()){
      stagingVarInfo svi;
      svi.varDB_index = -1;
      svi.device_ptr = nullptr;
      svi.host_contiguousArrayPtr = nullptr;
      svi.atomicStatusInHostMemory = UNKNOWN;
      svi.atomicStatusInGpuMemory = UNALLOCATED;

      std::pair<stagingVar, stagingVarInfo> p = std::make_pair( sv, svi );

      it->second.var->stagingVars.insert( p );

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
              << std::endl;
        }
        cerrLock.unlock();
      }
    }
  }
  varLock->unlock();
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype, int numGhostCells)
{

  // Allocate space on the GPU and declare a variable onto the GPU.

  // Check if it exists prior to allocating memory for it.
  // If it has already been allocated, just use that.
  // If it hasn't, this is lock free and the first thread to request allocating gets to allocate
  // If another thread sees that allocating is in process, it loops and waits until the allocation complete.

  bool allocationNeeded = false;
  int3 size = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread() << " Calling putUnallocatedIfNotExists() for " << label
          << " patch " << patchID
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging: " << std::boolalpha << staging
          << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
          << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName << std::endl;
    }
    cerrLock.unlock();
  }
  // This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, staging, offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;

  if (staging) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.var->stagingVars.find(sv);
  }

  varLock->unlock();

  // Locking not needed from here on in this method.  STL maps ensure that iterators point to correct values
  // even if other threads add nodes.  We just can't remove values, but that shouldn't ever happen.

  // This prepares the var with the offset and size.  Any possible allocation will come later.
  // If it needs to go into the database, that will also come later
  void* addr = nullptr;
  var.setArray3(offset, size, addr);

  // Now see if we allocate the variable or use a previous existing allocation.
  if (staging == false) {

    // See if someone has stated they are allocating it
    allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
           << " allocationNeeded is " << std::boolalpha << allocationNeeded
           << " for label " << label
           << " patch " << patchID
           << " matl " << matlIndx
           << " level " << levelIndx
           << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
           << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
           << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
           << std::endl;
      }
      cerrLock.unlock();
    }

    if (!allocationNeeded) {
      // Someone else is allocating it or it has already been allocated. Wait until they are done.
      bool allocated = false;
      while (!allocated) {
        allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
      }

      // Sanity check to ensure we have correct size information.
      varLock->lock();
      it = varPointers->find(lpml);
      varLock->unlock();

      if (it->second.var->device_offset.x == low.x
          && it->second.var->device_offset.y == low.y
          && it->second.var->device_offset.z == low.z
          && it->second.var->device_size.x == size.x
          && it->second.var->device_size.y == size.y
          && it->second.var->device_size.z == size.z) {

         // Space for this var already exists.  Use that and return.
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
                << " with data pointer " << it->second.var->device_ptr
                << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
                << " into GPUDW at " << std::hex << this << std::dec
                << std::endl;
           }
           cerrLock.unlock();
         }

         // Have this var use the existing memory address.
         var.setArray3(it->second.var->device_offset, it->second.var->device_size, it->second.var->device_ptr);
      } else if (it->second.var->device_offset.x <= low.x
          && it->second.var->device_offset.y <= low.y
          && it->second.var->device_offset.z <= low.z
          && it->second.var->device_size.x >= size.x
          && it->second.var->device_size.y >= size.y
          && it->second.var->device_size.z >= size.z) {
        //It fits inside.  Just use it.
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
               << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
               << " This non-staging/regular variable fits inside another variable that already exists.  No need to allocate another.  GPUDW has a variable for label " << label
               << " patch " << patchID
               << " matl " << matlIndx
               << " level " << levelIndx
               << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
               << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
               << " on device " << d_device_id
               << " with data pointer " << it->second.var->device_ptr
               << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
               << " into GPUDW at " << std::hex << this << std::dec
               << std::endl;
          }
          cerrLock.unlock();
        }

        var.setArray3(it->second.var->device_offset, it->second.var->device_size, it->second.var->device_ptr);
      } else {
        printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  Variable in database but of the wrong size.  This shouldn't ever happen. This needs low (%d, %d, %d) and size (%d, %d, %d), but in the database it is low (%d, %d, %d) and size (%d, %d, %d)\n",
            label, low.x, low.y, low.z, size.x, size.y, size.z,
            it->second.var->device_offset.x, it->second.var->device_offset.y, it->second.var->device_offset.z,
            it->second.var->device_size.x,   it->second.var->device_size.y,   it->second.var->device_size.z);
        exit(-1);
      }
    }
  } else {

    // it's a staging variable
    if (staging_it != it->second.var->stagingVars.end()) {

      // This variable exists in the database, no need to "put" it in again.
      // See if someone has stated they are allocating it
      allocationNeeded = compareAndSwapAllocating(staging_it->second.atomicStatusInGpuMemory);

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
                << std::endl;
          }
          cerrLock.unlock();
        }
        // We need the pointer.  We can't move on until we get the pointer.
        // Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
        // but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
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
           << " GPUDataWarehouse::allocateAndPut(), calling GPUMemoryPool::allocateCudaSpaceFromPool"
           << " for " << label
           << " patch " << patchID
           << " material " <<  matlIndx
           << " level " << levelIndx
           << " staging: " << std::boolalpha << staging
           << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
           << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
           << " at " << addr
           << " with status codes ";
        if (!staging) {
          gpu_stats << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory);
        } else {
          gpu_stats << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory);
        }
        gpu_stats << " on device " << d_device_id
           << " into GPUDW at " << std::hex << this << std::dec << std::endl;
      }
      cerrLock.unlock();
    }

    addr = GPUMemoryPool::allocateCudaSpaceFromPool(d_device_id, memSize);

    // Also update the var object itself
    var.setArray3(offset, size, addr);

    // Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, gtype, numGhostCells);

    // Now that we have the pointer and that it has been inserted into the database,
    // Update the status from allocating to allocated
    if (!staging) {
      compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
    } else {
      compareAndSwapAllocate(staging_it->second.atomicStatusInGpuMemory);
    }
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::allocateAndPut(), complete"
           << " for " << label
           << " patch " << patchID
           << " material " <<  matlIndx
           << " level " << levelIndx
           << " staging: " << std::boolalpha << staging
           << " with offset (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
           << " and size (" << size.x << ", " << size.y << ", " << size.z << ")"
           << " at " << addr
           << " with status codes ";
        if (!staging) {
          gpu_stats << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory);
        } else {
          gpu_stats << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory);
        }
        gpu_stats << " on device " << d_device_id
           << " into GPUDW at " << std::hex << this << std::dec << std::endl;
      }
      cerrLock.unlock();
    }
  }
}

//______________________________________________________________________
// This method is meant to take an entry from the host side DW and copy it  into
// the task datawarehouse whose job is to  eventually live GPU side.
__host__  void
GPUDataWarehouse::copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW, char const* label,
                                       int patchID, int matlIndx, int levelIndx, bool staging,
                                       int3 offset, int3 size) {


  if (d_device_copy == nullptr) {
    // sanity check
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This method should only be called from a task data warehouse.\n");
    exit(-1);
  }

  varLock->lock();
  if (d_numVarDBItems==MAX_VARDB_ITEMS) {
    printf("ERROR:  Out of GPUDataWarehouse space");
    varLock->unlock();
    exit(-1);
  }
  varLock->unlock();


  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  stagingVar sv;
  sv.device_offset = offset;
  sv.device_size = size;

  // Get the iterator(s) from the host side GPUDW.
  hostSideGPUDW->varLock->lock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator hostSideGPUDW_iter = hostSideGPUDW->varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator hostSideGPUDW_staging_iter;
  if (staging) {
    hostSideGPUDW_staging_iter = hostSideGPUDW_iter->second.var->stagingVars.find(sv);
    if (hostSideGPUDW_staging_iter == hostSideGPUDW_iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - No staging var was found for for %s patch %d material %d level %d offset (%d, %d, %d) size (%d, %d, %d) in the DW located at %p\n", label, patchID, matlIndx, levelIndx, offset.x, offset.y, offset.z, size.x, size.y, size.z, hostSideGPUDW);
      varLock->unlock();
      exit(-1);
    }
  }

  hostSideGPUDW->varLock->unlock();

  varLock->lock();

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);
  //sanity check
  if (iter != varPointers->end() && !staging) {
    printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW() - This task datawarehouse already had an entry for %s patch %d material %d level %d\n", label, patchID, matlIndx, levelIndx);
    varLock->unlock();
    exit(-1);
  }


  // If it's staging, there should already be a non-staging var in the host-side GPUDW (even if it's just a placeholder)

  // Inserting into this task DW, it is a requirement that non-staging variables get inserted first
  // then any staging variables can come in later.  This won't handle any scenario where a staging variable is requested
  // into the task DW without a non-staging variable already existing here.

  //TODO: Replace with an atomic counter.
  int d_varDB_index=d_numVarDBItems;
  d_numVarDBItems++;

  int i = d_varDB_index;

  if (!staging) {

    // Create a new allVarPointersInfo object, copying over the offset.
    allVarPointersInfo vp;
    vp.device_offset = hostSideGPUDW_iter->second.device_offset;

    // Give it a d_varDB index
    vp.varDB_index = d_varDB_index;

    // insert it in
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );

    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);

    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.var->sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.var->gtype;
    d_varDB[i].varItem.numGhostCells = hostSideGPUDW_iter->second.var->numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    d_varDB[i].var_offset = hostSideGPUDW_iter->second.var->device_offset;
    d_varDB[i].var_size = hostSideGPUDW_iter->second.var->device_size;
    d_varDB[i].var_ptr = hostSideGPUDW_iter->second.var->device_ptr;

  } else {

    if (iter == varPointers->end()) {
      // A staging item was requested but there's no regular variable for it to piggy back in.
      // So create an empty placeholder regular variable.

      // Create a new allVarPointersInfo object, copying over the offset.
      allVarPointersInfo vp;
      vp.device_offset = hostSideGPUDW_iter->second.device_offset;

      // Empty placeholders won't be placed in the d_varDB array.
      vp.varDB_index = -1;

      // insert it in
      std::pair<std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator, bool> ret
                        = varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
      if (!ret.second) {
        printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) Failure inserting into varPointers map.\n");
        varLock->unlock();
        exit(-1);
      }
      iter = ret.first;

    }

    //copy the item
    stagingVarInfo svi = hostSideGPUDW_staging_iter->second;

    //Give it a d_varDB index
    svi.varDB_index = d_varDB_index;

    //insert it in
    std::map<stagingVar, stagingVarInfo>::iterator staging_iter = iter->second.var->stagingVars.find(sv);
    if (staging_iter != iter->second.var->stagingVars.end()) {
      printf("ERROR:\nGPUDataWarehouse::copyItemIntoTaskDW( ) This staging var already exists in this task DW\n");
    }
    std::pair<stagingVar, stagingVarInfo> p = std::make_pair( sv, svi );
    iter->second.var->stagingVars.insert( p );

    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    d_varDB[i].sizeOfDataType = hostSideGPUDW_iter->second.var->sizeOfDataType;
    d_varDB[i].varItem.gtype = hostSideGPUDW_iter->second.var->gtype;
    d_varDB[i].varItem.numGhostCells = hostSideGPUDW_iter->second.var->numGhostCells;
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
         << " of max index " << d_maxdVarDBItems - 1
         << " label " << label
         << " patch " << d_varDB[i].domainID
         << " matl " << matlIndx
         << " level " << levelIndx
         << " staging: " << std::boolalpha << staging
         << " datatype size " <<d_varDB[i].sizeOfDataType
         << " into address " << d_varDB[i].var_ptr
         << " on device " << d_device_id
         << " into GPUDW at " << std::hex << this << std::dec
         << " size [" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << "]"
         << " offset [" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << "]"
         << std::endl;
    }
    cerrLock.unlock();

  }
  varLock->unlock();
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::putContiguous(GPUGridVariableBase &var, const char* indexID, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost)
{
/*
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else

  varLock->lock();

  //first check if this patch/var/matl is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //Space for this patch already exists.  Use that and return.
    if (d_debug){
      printf("GPUDataWarehouse::putContiguous( %s ). This gpudw database has a variable for label %s patch %d matl %d level %d staging %s on device %d.  Reusing it.\n",
          label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id);

    }
    var.setArray3(varPointers->at(lpml).device_offset, varPointers->at(lpml).device_size, varPointers->at(lpml).device_ptr);
    varLock->unlock();
    return;
  }

  int3 size=make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset=low;
  void* device_ptr=nullptr;
  var.setArray3(offset, size, device_ptr);
  allocateLock->lock();
  contiguousArrayInfo *ca = &(contiguousArrays->at(indexID));
  allocateLock->unlock();
  if ( (ca->allocatedDeviceMemory == nullptr
       || ca->sizeOfAllocatedMemory - ca->assignedOffset < var.getMemSize())
      && stageOnHost) {
    printf("ERROR: No room left on device to be assigned address space\n");
    if (ca->allocatedDeviceMemory != nullptr) {
      printf("There was %lu bytes allocated, %lu has been assigned, and %lu more bytes were attempted to be assigned for %s patch %d matl %d level %d staging %s\n",
          ca->sizeOfAllocatedMemory,
          ca->assignedOffset,
          var.getMemSize(), label, patchID, matlIndx, levelIndx, staging ? "true" : "false");
    }
    varLock->unlock();
    exit(-1);
  } else {


    //There is already pre-allocated contiguous memory chunks with room available on
    //both the device and the host.  Just assign pointers for both the device and host contiguous arrays.


    //This prepares the var with the offset and size.  The actual address will come next.

    void* host_contiguousArrayPtr = nullptr;

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

    }
    varLock->unlock();

    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, None, 0, host_contiguousArrayPtr);
  }

#endif
*/
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::allocate(const char* indexID, size_t size)
{
/*
#ifdef __CUDA_ARCH__
  // Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  if (size == 0) {
    return;
  }

  //This method allocates one big chunk of memory so that little allocations do not have to occur for each grid variable.
  //This is needed because devices often have substantial overhead for each device malloc and device copy.  By putting it into one
  //chunk of memory, only one malloc and one copy to device should be needed.
  double *d_ptr = nullptr;
  double *h_ptr = nullptr;
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
  allocateLock->lock();
  contiguousArrays->insert( std::map<const char *, contiguousArrayInfo>::value_type( indexID, ca ) );
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->begin(); it != contiguousArrays->end(); ++it)
  //  printf("%s\n", it->first.c_str());

  allocateLock->unlock();
#endif
*/
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndx, int levelIndx) {
/*
#ifdef __CUDA_ARCH__
  //Should not called from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  //see if this datawarehouse has anything for this patchGroupID.
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo info = varPointers->at(lpml);

    device_var.setArray3(varPointers->at(lpml).device_offset, varPointers->at(lpml).device_offset, info.device_ptr);
    varLock->unlock();
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
    varLock->unlock();
    printf("ERROR: host copyHostContiguoustoHost unknown variable on GPUDataWarehouse");
    //for (std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it=varPointers->begin(); it!=varPointers->end(); ++it)
    //  printf("%s %d %d => %d \n", it->first.label, it->first.patchID, it->first.matlIndx, it->second.varDB_index);
    exit(-1);
  }
#endif
*/
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUReductionVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, void* host_ptr)
{

  varLock->lock();

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
  iter->second.var->device_ptr = var_ptr;
  iter->second.var->sizeOfDataType = sizeOfDataType;
  iter->second.var->gtype = None;
  iter->second.var->numGhostCells = 0;
  iter->second.var->host_contiguousArrayPtr = host_ptr;
  iter->second.var->atomicStatusInHostMemory = UNKNOWN;
  int3 zeroValue;
  zeroValue.x = 0;
  zeroValue.y = 0;
  zeroValue.z = 0;
  iter->second.var->device_offset = zeroValue;
  iter->second.var->device_size = zeroValue;


  //previously set, do not set here
  //iter->second.var->atomicStatusInGpuMemory =

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
          << " with datatype size " << iter->second.var->sizeOfDataType
          << " with status codes " << getDisplayableStatusCodes(iter->second.var->atomicStatusInGpuMemory)
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName
          << " current varPointers size is: " << varPointers->size()
          << std::endl;
    }
    cerrLock.unlock();
  }

  varLock->unlock();

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::put(GPUPerPatchBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, void* host_ptr)
{

  varLock->lock();
  void* var_ptr;           // raw pointer to the memory
  var.getData(var_ptr);

  //See if it already exists.  Also see if we need to update this into d_varDB.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator iter = varPointers->find(lpml);

  //sanity check
  if (iter == varPointers->end()) {
    printf("ERROR:\nGPUDataWarehouse::put( )  Can't use put() for a host-side GPU DW without it first existing in the internal database for %s patch %d matl %d.\n", label, patchID, matlIndx);
    exit(-1);
  }

  iter->second.varDB_index = -1;
  iter->second.var->device_ptr = var_ptr;
  iter->second.var->sizeOfDataType = sizeOfDataType;
  iter->second.var->gtype = None;
  iter->second.var->numGhostCells = 0;
  iter->second.var->host_contiguousArrayPtr = host_ptr;
  iter->second.var->atomicStatusInHostMemory = UNKNOWN;
  int3 zeroValue;
  zeroValue.x = 0;
  zeroValue.y = 0;
  zeroValue.z = 0;
  iter->second.var->device_offset = zeroValue;
  iter->second.var->device_size = zeroValue;

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
          << " with datatype size " << iter->second.var->sizeOfDataType
          << " with status codes " << getDisplayableStatusCodes(iter->second.var->atomicStatusInGpuMemory)
          << " on device " << d_device_id
          << " into GPUDW at " << std::hex << this << std::dec
          << " with description " << _internalName
          << " current varPointers size is: " << varPointers->size()
          << std::endl;
    }
    cerrLock.unlock();
  }

  varLock->unlock();

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
          << " with description " << _internalName << std::endl;
    }
    cerrLock.unlock();
  }
  //This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  varLock->unlock();

  void* addr = nullptr;

  //Now see if we allocate the variable or use a previous existing allocation.
  //See if someone has stated they are allocating it
  allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
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
            << " with data pointer " << it->second.var->device_ptr
            << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
            << " into GPUDW at " << std::hex << this << std::dec
            << std::endl;
       }
       cerrLock.unlock();
     }

     //We need the pointer.  We can't move on until we get the pointer.
     //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
     //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
     bool allocated = false;
     while (!allocated) {
       allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
       addr = it->second.var->device_ptr;
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
            << " GPUDataWarehouse::allocateAndPut(), calling GPUMemoryPool::allocateCudaSpaceFromPool"
            << " for reduction variable " << label
            << " patch " << patchID
            << " material " <<  matlIndx
            << " level " << levelIndx
            << " size " << var.getMemSize()
            << " at " << addr
            << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec << std::endl;
      }
      cerrLock.unlock();
    }

    addr = GPUMemoryPool::allocateCudaSpaceFromPool(d_device_id, memSize);

    //Also update the var object itself
    var.setData(addr);

    //Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);

    //Now that the database knows of this and other threads can see the device pointer, update the status from allocating to allocated
    compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);
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
          << " with description " << _internalName << std::endl;
    }
    cerrLock.unlock();
  }
  //This variable may not yet exist.  But we want to declare we're allocating it.  So ensure there is an entry.
  putUnallocatedIfNotExists(label, patchID, matlIndx, levelIndx, false, offset, size);

  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  varLock->unlock();

  void* addr = nullptr;

  //Now see if we allocate the variable or use a previous existing allocation.

  //See if someone has stated they are allocating it
  allocationNeeded = compareAndSwapAllocating(it->second.var->atomicStatusInGpuMemory);
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
            << " with data pointer " << it->second.var->device_ptr
            << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
            << " into GPUDW at " << std::hex << this << std::dec
            << std::endl;
       }
       cerrLock.unlock();
     }
     //We need the pointer.  We can't move on until we get the pointer.
     //Ensure that it has been allocated (just not allocating). Another thread may have been assigned to allocate it
     //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
     bool allocated = false;
     while (!allocated) {
       allocated = checkAllocated(it->second.var->atomicStatusInGpuMemory);
       addr = it->second.var->device_ptr;
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
            << " GPUDataWarehouse::allocateAndPut(), calling GPUMemoryPool::allocateCudaSpaceFromPool"
            << " for PerPatch variable " << label
            << " patch " << patchID
            << " material " <<  matlIndx
            << " level " << levelIndx
            << " size " << var.getMemSize()
            << " at " << addr
            << " with status codes " << getDisplayableStatusCodes(it->second.var->atomicStatusInGpuMemory)
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec << std::endl;
      }
      cerrLock.unlock();
    }

    addr = GPUMemoryPool::allocateCudaSpaceFromPool(d_device_id, memSize);

    //Also update the var object itself
    var.setData(addr);

    //Put all remaining information about the variable into the the database.
    put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);
    //Now that the database knows of this and other threads can see the device pointer, update the status from allocating to allocated
    compareAndSwapAllocate(it->second.var->atomicStatusInGpuMemory);

  }
}

//______________________________________________________________________
//
__device__ GPUDataWarehouse::dataItem*
GPUDataWarehouse::getItem(char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx)
{

  //This upcoming __syncthreads is needed.  With CUDA function calls are inlined.
  // If you don't have it this upcoming __syncthreads here's what I think can happen:

  // * The correct index was found by one of the threads.
  // * The last __syncthreads is called, all threads met up there.
  // * Some threads in the block then make a second "function" call and reset index to -1
  // * Meanwhile, those other threads were still in the first "function" call and hadn't
  //   yet processed if (index == -1).  They now run that line.  And see index is now -1.  That's bad.

  // So to prevent this scenario, we have one more __syncthreads listed immediately below.
  __syncthreads();  //sync before get



  short numThreads = blockDim.x * blockDim.y * blockDim.z;
  //int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
  int i = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block
  //int threadID = i;

  //if (d_debug && threadID == 0 && blockID == 0) {
  //  printf("device getting item \"%s\" from GPUDW %p", label, this);
  //  printf("size (%d vars)\n Available labels:", d_numVarDBItems);
  //}

  //Have every thread try to find the label/patchId/matlIndx is a match in
  //array.  This is a parallel approach so that instead of doing a simple
  //sequential search with one thread, we can let every thread search for it.  Only the
  //winning thread gets to write to shared data.

  __shared__ int index;
  index = -1;

  __syncthreads();  //sync before get, making sure everyone set index to -1

  while(i<d_numVarDBItems){
    short strmatch=0;
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
                /*&& d_varDB[i].levelIndx == levelIndx*/  // No need for level lookups, label + patchID + matl is a unique tuple.
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
    printf("ERROR:\nGPUDataWarehouse::getItem() didn't find anything for %s patch %d matl %d\n", label, patchID, matlIndx);
    return nullptr;
  }
  return &d_varDB[index];

}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::remove(char const* label, int patchID, int matlIndx, int levelIndx)
{

  /*
  //This is more of a stub.  Remove hasn't been needed up until yet.  If removing is needed, it
  //would likely be best to deallocate things but leave an entry in the collection.
  bool retVal = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();

  if (varPointers->find(lpml) != varPointers->end()) {
    int i = varPointers->at(lpml).varDB_index;
    d_varDB[i].label[0] = '\0'; //leave a hole in the flat array, not deleted.
    varPointers->erase(lpml);  //TODO: GPU Memory leak?
    retVal = true;
    d_dirty=true;
  }
  if (d_debug){
    printf("GPUDataWarehouse::remove( %s ). Removed a variable for label %s patch %d matl %d level %d \n",
        label, label, patchID, matlIndx, levelIndx);
  }
  varLock->unlock();
  return retVal;
  */
  return false;
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::init(int id, std::string internalName)
{

  d_device_id = id;
  //this->_internalName = new std::string(internalName);
  strncpy(_internalName, internalName.c_str(), sizeof(_internalName));
  objectSizeInBytes = 0;
  d_maxdVarDBItems = 0;
  //this->placementNewBuffer = placementNewBuffer;

  allocateLock     = new Uintah::MasterLock{};
  varLock          = new Uintah::MasterLock{};
  varPointers      = new std::map<labelPatchMatlLevel, allVarPointersInfo>;
  contiguousArrays = new std::map<std::string, contiguousArrayInfo>;

  //other data members are initialized in the constructor
  d_numVarDBItems = 0;
  d_numMaterials = 0;
  d_debug = false;
  //d_numGhostCells = 0;
  d_device_copy = nullptr;
  d_dirty = true;
  objectSizeInBytes = 0;
  //resetdVarDB();
  numGhostCellCopiesNeeded = 0;

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
GPUDataWarehouse::init_device(size_t objectSizeInBytes, unsigned int d_maxdVarDBItems)
{
  this->objectSizeInBytes = objectSizeInBytes;
  this->d_maxdVarDBItems = d_maxdVarDBItems;
  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
  void* temp = nullptr;
  //CUDA_RT_SAFE_CALL(cudaMalloc(&temp, objectSizeInBytes));
  temp = GPUMemoryPool::allocateCudaSpaceFromPool(d_device_id, objectSizeInBytes);
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
     gpu_stats << UnifiedScheduler::myRankThread()
         << " GPUDataWarehouse::init_device() -"
         << " requested GPU space from GPUMemoryPool::allocateCudaSpaceFromPool for Task DW of size " << objectSizeInBytes
         << " bytes at " << temp
         << " on device " << d_device_id
         << " the host GPUDW is at " << this
         << std::endl;
    }
    cerrLock.unlock();
  }
  d_device_copy = (GPUDataWarehouse*)temp;
  //cudaHostRegister(this, sizeof(GPUDataWarehouse), cudaHostRegisterPortable);



  d_dirty = true;

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::syncto_device(void *cuda_stream)
{
  if (!d_device_copy) {
    printf("ERROR:\nGPUDataWarehouse::syncto_device()\nNo device copy\n");
    exit(-1);
  }
  varLock->lock();

  if (d_dirty){
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    //Even though this is in a writeLock state on the CPU, the nature of multiple threads
    //each with their own stream copying to a GPU means that one stream might seemingly go out
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
            << std::endl;
      }
      cerrLock.unlock();
    }
    CUDA_RT_SAFE_CALL (cudaMemcpyAsync( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice, *stream));

    d_dirty=false;
  }

  varLock->unlock();

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::clear()
{

  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );

  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator varIter;
  for (varIter = varPointers->begin(); varIter != varPointers->end(); ++varIter) {
    // clear out all the staging vars, if any
    std::map<stagingVar, stagingVarInfo>::iterator stagingIter;
    for (stagingIter = varIter->second.var->stagingVars.begin(); stagingIter != varIter->second.var->stagingVars.end(); ++stagingIter) {
      if (compareAndSwapDeallocating(stagingIter->second.atomicStatusInGpuMemory)) {
        //The counter hit zero, so lets deallocate the var.
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::clear() -"
                << " calling GPUMemoryPool::freeCudaSpaceFromPool() for staging var for " << varIter->first.label
                << " at device ptr " <<  stagingIter->second.device_ptr
                << " on device " << d_device_id
                << std::endl;
          }
          cerrLock.unlock();
        }

        if (GPUMemoryPool::freeCudaSpaceFromPool(d_device_id, stagingIter->second.device_ptr) ) {
          stagingIter->second.device_ptr = nullptr;
          compareAndSwapDeallocate(stagingIter->second.atomicStatusInGpuMemory);
        } else {
          printf("ERROR:\nGPUDataWarehouse::clear(), for a staging variable, couldn't find in the GPU memory pool the space starting at address %p\n", stagingIter->second.device_ptr);
          varLock->unlock();
          exit(-1);
        }
      }
    }

    varIter->second.var->stagingVars.clear();

    // clear out the regular vars

    // See if it's a placeholder var for staging vars.  This happens if the non-staging var
    // had a device_ptr of nullptr, and it was only in the varPointers map to only hold staging vars
    if (compareAndSwapDeallocating(varIter->second.var->atomicStatusInGpuMemory)) {
      if (varIter->second.var->device_ptr) {

        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " GPUDataWarehouse::clear() -"
                << " calling GPUMemoryPool::freeCudaSpaceFromPool() for non-staging var for " << varIter->first.label
                << " at device ptr " <<  varIter->second.var->device_ptr
                << " on device " << d_device_id
                << std::endl;
          }
          cerrLock.unlock();
        }
        if (GPUMemoryPool::freeCudaSpaceFromPool(d_device_id, varIter->second.var->device_ptr)) {
          varIter->second.var->device_ptr = nullptr;
          compareAndSwapDeallocate(varIter->second.var->atomicStatusInGpuMemory);
        } else {
          printf("ERROR:\nGPUDataWarehouse::clear(), for a non-staging variable, couldn't find in the GPU memory pool the space starting at address %p\n", varIter->second.var->device_ptr);
          varLock->unlock();
          exit(-1);
        }
      }
    }
  }
  varPointers->clear();

  varLock->unlock();

  init(d_device_id, _internalName);

}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::deleteSelfOnDevice()
{
  if ( d_device_copy ) {
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::deleteSelfOnDevice - calling GPUMemoryPool::freeCudaSpaceFromPool for Task DW at " << std::hex
           << d_device_copy << " on device " << std::dec << d_device_id << std::endl;
      }
      cerrLock.unlock();
    }

    GPUMemoryPool::freeCudaSpaceFromPool(d_device_id, d_device_copy);

  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::resetdVarDB()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else

  if (d_device_copy != nullptr) {
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
      d_varDB[i].var_ptr = nullptr;
      d_varDB[i].ghostItem.dest_varDB_index = -1;

    }
    for (int i = 0; i < MAX_LEVELDB_ITEMS; i++) {
      d_levelDB[i].label[0] = '\0';
      d_levelDB[i].domainID = -1;
      d_levelDB[i].matlIndx = -1;
      //d_varDB[i].staging = false;
      d_levelDB[i].var_ptr = nullptr;
    }
    for (int i = 0; i < MAX_MATERIALSDB_ITEMS; i++) {
      d_materialDB[i].simulationType[0] = '\0';
    }
  }
#endif
}

//______________________________________________________________________
//These material methods below needs more work.  They haven't been tested.
__host__ void
GPUDataWarehouse::putMaterials( std::vector< std::string > materials)
{


  varLock->lock();
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

  varLock->unlock();
}

//______________________________________________________________________
//
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

//______________________________________________________________________
//
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

//______________________________________________________________________
//TODO: This is too slow.  It needs work.
__device__ void
GPUDataWarehouse::copyGpuGhostCellsToGpuVars() {

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

//         if (threadID == 0) {
//           printf("Going to copy, between (%d, %d, %d) from offset %d to offset %d.  From starts at (%d, %d, %d) with size (%d, %d, %d) at index %d pointer %p.  To starts at (%d, %d, %d) with size (%d, %d, %d).\n",
//                  d_varDB[i].ghostItem.sharedLowCoordinates.x,
//                  d_varDB[i].ghostItem.sharedLowCoordinates.y,
//                  d_varDB[i].ghostItem.sharedLowCoordinates.z,
//                  sourceOffset,
//                  destOffset,
//                  d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
//                  d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
//                  i,
//                  d_varDB[i].var_ptr,
//                  d_varDB[destIndex].var_offset.x, d_varDB[destIndex].var_offset.y, d_varDB[destIndex].var_offset.z,
//                  d_varDB[destIndex].var_size.x, d_varDB[destIndex].var_size.y, d_varDB[destIndex].var_size.z);
//         }

         //copy all 8 bytes of a double in one shot
         if (d_varDB[i].sizeOfDataType == sizeof(double)) {
           *((double*)(d_varDB[destIndex].var_ptr) + destOffset) = *((double*)(d_varDB[i].var_ptr) + sourceOffset);

           //Note: Every now and then I've seen this printf statement get confused, a line will print with the wrong variables/offset variables...
//           printf("Thread %d - %s At (%d, %d, %d), real: (%d, %d, %d), copying within region between (%d, %d, %d) and (%d, %d, %d).  Source d_varDB index (%d, %d, %d) varSize (%d, %d, %d) virtualOffset(%d, %d, %d), varOffset(%d, %d, %d), sourceOffset %d actual pointer %p, value %e.   Dest d_varDB index %d ptr %p destOffset %d actual pointer. %p\n",
//                  threadID, d_varDB[destIndex].label, x, y, z, x_source_real, y_source_real, z_source_real,
//                  d_varDB[i].ghostItem.sharedLowCoordinates.x, d_varDB[i].ghostItem.sharedLowCoordinates.y, d_varDB[i].ghostItem.sharedLowCoordinates.z,
//                  d_varDB[i].ghostItem.sharedHighCoordinates.x, d_varDB[i].ghostItem.sharedHighCoordinates.y, d_varDB[i].ghostItem.sharedHighCoordinates.z,
//                  x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[i].ghostItem.virtualOffset.x,
//                  y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[i].ghostItem.virtualOffset.y,
//                  z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[i].ghostItem.virtualOffset.z,
//                  d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
//                  d_varDB[i].ghostItem.virtualOffset.x, d_varDB[i].ghostItem.virtualOffset.y, d_varDB[i].ghostItem.virtualOffset.z,
//                  d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
//                  sourceOffset, (double*)(d_varDB[i].var_ptr) + sourceOffset, *((double*)(d_varDB[i].var_ptr) + sourceOffset),
//                  destIndex, d_varDB[destIndex].var_ptr,  destOffset, (double*)(d_varDB[destIndex].var_ptr) + destOffset);
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
}

//______________________________________________________________________
//
 __global__ void copyGpuGhostCellsToGpuVarsKernel( GPUDataWarehouse *gpudw) {
   gpudw->copyGpuGhostCellsToGpuVars();
}

//______________________________________________________________________
//
 __host__ void
GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream)
{
  //see if this GPU datawarehouse has ghost cells in it.
  if (numGhostCellCopiesNeeded > 0) {
    //call a kernel which gets the copy process started.
    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);

#if 0               // compiler warnings
    const int BLOCKSIZE = 1;
    int xblocks = 32;
    int yblocks = 1;
    int zblocks = 1;
#endif

    dim3 dimBlock(32, 16, 1);
    dim3 dimGrid(1, 1, 1);  //Give each ghost copying kernel 32 * 16 = 512 threads to copy
                            //(32x32 was too much for a smaller laptop GPU and the Uintah build server in debug mode)
    if (gpu_stats.active()) {
     cerrLock.lock();
     {
       gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker() - "
           << " Launching ghost cell copies kernel"
           << " on device " << d_device_id
           << " at GPUDW at " << std::hex << this << std::dec
           << " with description " << _internalName
           << std::endl;
     }
     cerrLock.unlock();
    }
    copyGpuGhostCellsToGpuVarsKernel<<< dimGrid, dimBlock, 0, *stream >>>(this->d_device_copy);
  }
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::ghostCellCopiesNeeded()
{
  //see if this GPU datawarehouse has ghost cells in it.
  return (numGhostCellCopiesNeeded > 0);
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::putGhostCell(char const* label, int sourcePatchID, int destPatchID, int matlIndx, int levelIndx,
                               bool sourceStaging, bool destStaging,
                               int3 varOffset, int3 varSize,
                               int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset) {

  //Add information describing a ghost cell that needs to be copied internally from
  //one chunk of data to the destination.  This covers a GPU -> same GPU copy scenario.
  varLock->lock();
  unsigned int i = d_numVarDBItems;
  if (i > d_maxdVarDBItems) {
    printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Exceeded maximum d_varDB entries.  Index is %d and max items is %d\n", label, i, d_maxdVarDBItems);
    varLock->unlock();
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
      index = varPointers->at(lpml_source).varDB_index;
    }
  } else {
    //Find the variable that contains the region in which our ghost cells exist.
    //Usually the sharedLowCoordinates and sharedHighCoordinates correspond
    //exactly to the size of the staging variable.
    //(TODO ? But sometimes the ghost data is found within larger staging variable. Not sure if there is a use case for this yet)
    stagingVar sv;
    sv.device_offset = varOffset;
    sv.device_size = varSize;

    std::map<stagingVar, stagingVarInfo>::iterator staging_it = varPointers->at(lpml_source).var->stagingVars.find(sv);
    if (staging_it != varPointers->at(lpml_source).var->stagingVars.end()) {

      index = staging_it->second.varDB_index;

    } else {
      int nStageVars = varPointers->at(lpml_source).var->stagingVars.size();
      printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). Number of staging vars for this var: %d, No staging variable found exactly matching all of the following: label %s patch %d matl %d level %d offset (%d, %d, %d) size (%d, %d, %d) on DW at %p.\n",
                    label, nStageVars, label, sourcePatchID, matlIndx, levelIndx,
                    sv.device_offset.x, sv.device_offset.y, sv.device_offset.z,
                    sv.device_size.x, sv.device_size.y, sv.device_size.z,
                    this);
      varLock->unlock();
      exit(-1);
    }
    //Find the d_varDB entry for this specific one.


  }

  if (index < 0) {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label %s, source patch ID %d, matlIndx %d, levelIndex %d staging %s not found in GPU DW %p\n",
        label, sourcePatchID, matlIndx, levelIndx, sourceStaging ? "true" : "false", this);
    varLock->unlock();
    exit(-1);
  }

  d_varDB[i].var_offset = d_varDB[index].var_offset;
  d_varDB[i].var_size = d_varDB[index].var_size;
  d_varDB[i].var_ptr = d_varDB[index].var_ptr;
  d_varDB[i].sizeOfDataType = d_varDB[index].sizeOfDataType;
  if (gpu_stats.active()) {
   cerrLock.lock();
   {
     gpu_stats << UnifiedScheduler::myRankThread()
         << " GPUDataWarehouse::putGhostCell() - "
         << " Placed into d_varDB at index " << i << " of max index " << d_maxdVarDBItems - 1
         << " from patch " << sourcePatchID << " staging " << sourceStaging << " to patch " << destPatchID << " staging " << destStaging
         << " has shared coordinates (" << sharedLowCoordinates.x << ", " << sharedLowCoordinates.y << ", " << sharedLowCoordinates.z << "),"
         << " (" << sharedHighCoordinates.x << ", " << sharedHighCoordinates.y << ", " << sharedHighCoordinates.z << "), "
         << " from low/offset (" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << ") "
         << " size (" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << ") "
         << " virtualOffset (" << d_varDB[i].ghostItem.virtualOffset.x << ", " << d_varDB[i].ghostItem.virtualOffset.y << ", " << d_varDB[i].ghostItem.virtualOffset.z << ") "
         << " datatype size " << d_varDB[i].sizeOfDataType
         << " on device " << d_device_id
         << " at GPUDW at " << std::hex << this<< std::dec
         << std::endl;
   }
   cerrLock.unlock();
  }


  // Find where we are sending the ghost cell data to
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

      std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.find(sv);
      if (staging_it != it->second.var->stagingVars.end()) {
        d_varDB[i].ghostItem.dest_varDB_index = staging_it->second.varDB_index;
      } else {
        printf("\nERROR:\nGPUDataWarehouse::putGhostCell() didn't find a staging variable from the device for offset (%d, %d, %d) and size (%d, %d, %d).\n",
            sharedLowCoordinates.x, sharedLowCoordinates.y, sharedLowCoordinates.z,
            sv.device_size.x, sv.device_size.y, sv.device_size.z);
        varLock->unlock();
        exit(-1);
      }

    } else {
      d_varDB[i].ghostItem.dest_varDB_index = it->second.varDB_index;
    }
  } else {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell(), label: %s destination patch ID %d, matlIndx %d, levelIndex %d, staging %s not found in GPU DW variable database\n",
        label, destPatchID, matlIndx, levelIndx, destStaging ? "true" : "false");
    varLock->unlock();
    exit(-1);
  }

  d_dirty=true;
  varLock->unlock();
}

//______________________________________________________________________
//
__host__ void
GPUDataWarehouse::getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells,
  char const* label, int patchID, int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    allVarPointersInfo info = varPointers->at(lpml);
    low = info.device_offset;
    high.x = info.var->device_size.x + info.var->device_offset.x;
    high.y = info.var->device_size.y + info.var->device_offset.y;
    high.z = info.var->device_size.z + info.var->device_offset.z;
    siz = info.var->device_size;
    gtype = info.var->gtype;
    numGhostCells = info.var->numGhostCells;
  }
  varLock->unlock();

}
//______________________________________________________________________
//Deep copies (not shallow copies or moves) an entry from one data warehouse to another.
//(Note: A deep copy is a full copy of data from one variable's memory space to another variable's memory space
//A shallow copy is just a pointer copy and a ref counting
//A move is a true std::move() reseating.)
//RMCRT and Arches often keep a variable in the old data warehouse alive by copying it to the new data warehouse.
//It can't be a move (it may be needed to use data from the old and the new)
//It can't be a shallow copy (it may be needed to modify the new and also use the old)
//So it must be a deep copy.
//Both the source and destination variables must be in the GPU data warehouse,
//both must be listed as "allocated".  If these are not the case, the transferFrom doesn't proceed.
//Both must have the same variable sizes.  If this is not the case, the program will exit.
//If all above conditions are met, then it will do a device to device memcopy call.
//*Important*: For this to work, it needs a GPU stream.  GPU streams are stored per task, every Uintah task is assigned
//a possible stream to use.  To get the stream you have to request it from the detailedTask object.
//Normal CPU task callback functions do not have access to the detailedTask object, but it is possible to
//extend the callack function parameter list so that it does.  See UnifiedSchedulerTest::timeAdvanceUnified as an example.
//*Also important*: For this to work, the destination variable *MUST* be listed as a computes in the task that's
//calling transferFrom().  That allows for the computes data to have been preallocated ahead of time by the scheduler.
//Uintah's scheduler is fine if it is able to allocate the space, so that it can allow the task developer to write data
//into space it created.  If it was a computes, then this method can copy data into the computes memory, and
//when the task which called transferFrom is done, the scheduler will mark this computes variable as VALID.
//Note: A shallow copy method has been requested by the Arches team.  That hasn't been implemented yet.  It would require
//ref counting a variable, and perhaps some sanity checks to ensure a shallow copied variable is not called a computes and
//then later listed as a modifies.
__host__ bool
GPUDataWarehouse::transferFrom(cudaStream_t* stream, GPUGridVariableBase &var_source, GPUGridVariableBase &var_dest, GPUDataWarehouse * from, char const* label, int patchID, int matlIndx, int levelIndx){
  from->varLock->lock();
  this->varLock->lock();  //lock both data warehouses, no way to lock free this section,
                          //you could get the dining philosophers problem.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);

  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator source_it = from->varPointers->find(lpml);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator dest_it   = this->varPointers->find(lpml);
  int proceed = true;
  if (source_it == from->varPointers->end()) {
    //It may just be there wasn't any requires in the GPU to begin with, so don't bother attempting to copy.
    //printf("GPU source not found\n");
    proceed = false;
  } else if (dest_it == this->varPointers->end()) {
    //It may just be there wasn't any computes in the GPU to begin with, so don't bother attempting to copy.
    //printf("GPU dest not found in DW at %p for variable %s patch %d matl %d level %d\n", this, label, patchID, matlIndx, levelIndx);
    proceed = false;
  } else if (((__sync_fetch_and_or(&(source_it->second.var->atomicStatusInGpuMemory), 0) & ALLOCATED) != ALLOCATED)){
    //It may just be there wasn't any computes in the GPU to begin with, so don't bother attempting to copy.
    //printf("GPU source not allocated for variable %s patch %d matl %d level %d, it has status codes %s\n",  label, patchID, matlIndx, levelIndx, getDisplayableStatusCodes(source_it->second.atomicStatusInGpuMemory).c_str());
    proceed = false;

    //Is this  a problem?  We know of this variable in the data warehouse, but we have no space for it.
    //printf("Error: GPUDataWarehouse::transferFrom() - No source variable device space found.  Cannot proceed with deep copy.  Exiting...\n");
    //exit(-1);
  } else if (((__sync_fetch_and_or(&(dest_it->second.var->atomicStatusInGpuMemory), 0) & ALLOCATED) != ALLOCATED)){
    //printf("GPU destination not allocated for variable %s patch %d matl %d level %d\n",  label, patchID, matlIndx, levelIndx);
    //It may just be there wasn't any computes in the GPU to begin with, so don't bother attempting to copy.
    proceed = false;

    //Is a problem?  We know of this variable in the data warehouse, but we have no space for it.
    //printf("Error: GPUDataWarehouse::transferFrom() - No destination variable device space found.  Cannot proceed with deep copy.  Exiting...\n");
    //exit(-1);
  }
  if (!proceed) {
    from->varLock->unlock();
    this->varLock->unlock();
    return false;
  }

  if (!(    source_it->second.var->device_offset.x == dest_it->second.var->device_offset.x
         && source_it->second.var->device_offset.y == dest_it->second.var->device_offset.y
         && source_it->second.var->device_offset.z == dest_it->second.var->device_offset.z
         && source_it->second.var->device_size.x   == dest_it->second.var->device_size.x
         && source_it->second.var->device_size.y   == dest_it->second.var->device_size.y
         && source_it->second.var->device_size.z   == dest_it->second.var->device_size.z    )) {

    printf("Error: GPUDataWarehouse::transferFrom() - The source and destination variables exists for variable %s patch %d matl %d level %d, but the sizes don't match.  Cannot proceed with deep copy.  Exiting...\n", label, patchID, matlIndx, levelIndx);
    printf("The source size is (%d, %d, %d) with offset (%d, %d, %d) and device size is (%d, %d, %d) with offset (%d, %d, %d)\n",
            source_it->second.var->device_size.x, source_it->second.var->device_size.y, source_it->second.var->device_size.z,
            source_it->second.var->device_offset.x, source_it->second.var->device_offset.y, source_it->second.var->device_offset.z,
            dest_it->second.var->device_size.x, dest_it->second.var->device_size.y, dest_it->second.var->device_size.z,
            dest_it->second.var->device_offset.x, dest_it->second.var->device_offset.y, dest_it->second.var->device_offset.z);

    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  } else if (!(source_it->second.var->device_ptr)) {
    //A couple more santiy checks, this may be overkill...
    printf("Error: GPUDataWarehouse::transferFrom() - No source variable pointer found for variable %s patch %d matl %d level %d\n", label, patchID, matlIndx, levelIndx);
    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  } else if (!(dest_it->second.var->device_ptr)) {
    printf("Error: GPUDataWarehouse::transferFrom() - No destination variable pointer found for variable %s patch %d matl %d level %d\n", label, patchID, matlIndx, levelIndx);
    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);

  } else if (!stream) {
    printf("ERROR: No stream associated with the detailed task.  Cannot proceed with deep copy.  Exiting...\n");
    printf("If you get this message, the fix is not that rough.  You need to change your CPU callback function to having the full set of parameters common for a GPU task.  If you do that, the engine should pick up the rest of the details.\n");
    from->varLock->unlock();
    this->varLock->unlock();
    exit(-1);
  }

  //We shouldn't need to allocate space on either the source or the datination.  The source should have been listed as a requires,
  //and the  destination should have been listed as a computes for the task.
  //And this solves a mess of problems, mainly deailing with when it is listed as allocated and when it's listed as valid.

  var_source.setArray3(source_it->second.var->device_offset, source_it->second.var->device_size, source_it->second.var->device_ptr);

  var_source.setArray3(dest_it->second.var->device_offset, dest_it->second.var->device_size, dest_it->second.var->device_ptr);

  cudaMemcpyAsync(dest_it->second.var->device_ptr, source_it->second.var->device_ptr,
      source_it->second.var->device_size.x * source_it->second.var->device_size.y * source_it->second.var->device_size.z * source_it->second.var->sizeOfDataType,
      cudaMemcpyDeviceToDevice, *stream);

  from->varLock->unlock();
  this->varLock->unlock();

  //Let the caller know we found and transferred something.
  return true;
}

//______________________________________________________________________
// Go through all staging vars for a var. See if they are all marked as valid.
__host__ bool
GPUDataWarehouse::areAllStagingVarsValid(char const* label, int patchID, int matlIndx, int levelIndx) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    for (std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.begin();
         staging_it != it->second.var->stagingVars.end();
         ++staging_it) {
     if (!checkValid(staging_it->second.atomicStatusInGpuMemory)) {
       varLock->unlock();
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
               << ") with status codes " << getDisplayableStatusCodes(staging_it->second.atomicStatusInGpuMemory) << std::endl;
         }
         cerrLock.unlock();
       }
       return false;
     }
   }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// Simply performs an atomic fetch on the status variable.
//typedef int atomicDataStatus;
//__host__ atomicDataStatus
//GPUDataWarehouse::getStatus(atomicDataStatus& status) {
//  return __sync_or_and_fetch(&(status), 0);
//}

//______________________________________________________________________
//
__host__ std::string
GPUDataWarehouse::getDisplayableStatusCodes(atomicDataStatus& status) {
  atomicDataStatus varStatus  = __sync_or_and_fetch(&status, 0);
  std::string retval = "";
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
    if ((varStatus & DEALLOCATING) == DEALLOCATING) {
      retval += "Deallocating ";
    }
    if ((varStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH) {
      retval += "Forming-superpatch ";
    }
    if ((varStatus & SUPERPATCH) == SUPERPATCH) {
      retval += "Superpatch ";
    }
    if ((varStatus & UNKNOWN) == UNKNOWN) {
      retval += "Unknown ";
    }
  }
  //trim whitespace
  retval.erase(std::find_if(retval.rbegin(), retval.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), retval.end());
  return retval;
}

//______________________________________________________________________
//

__host__ void
GPUDataWarehouse::getStatusFlagsForVariableOnGPU(bool& correctSize, bool& allocating, bool& allocated, bool& copyingIn,
                               bool& validOnGPU, bool& gatheringGhostCells, bool& validWithGhostCellsOnGPU,
                               bool& deallocating, bool& formingSuperPatch, bool& superPatch,
                               char const* label, const int patchID, const int matlIndx, const int levelIndx,
                               const int3& offset, const int3& size) {
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);

  if (varPointers->find(lpml) != varPointers->end()) {
    //check the sizes
    allVarPointersInfo vp = varPointers->at(lpml);
    int3 device_offset = vp.var->device_offset;
    int3 device_size = vp.var->device_size;
    //DS 12132019: GPU Resize fix. Changed condition == to <= (and >=). If device variable is greater than host, its ok.
    correctSize = (device_offset.x <= offset.x && device_offset.y <= offset.y && device_offset.z <= offset.z
                   && device_size.x >= size.x && device_size.y >= size.y && device_size.z >= size.z);

    //get the value
    atomicDataStatus varStatus  = __sync_or_and_fetch(&(vp.var->atomicStatusInGpuMemory), 0);

    allocating               = ((varStatus & ALLOCATING) == ALLOCATING);
    allocated                = ((varStatus & ALLOCATED)  == ALLOCATED);
    copyingIn                = ((varStatus & COPYING_IN) == COPYING_IN);
    validOnGPU               = ((varStatus & VALID)      == VALID);
    gatheringGhostCells      = ((varStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY);
    validWithGhostCellsOnGPU = ((varStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    deallocating             = ((varStatus & DEALLOCATING) == DEALLOCATING);
    formingSuperPatch        = ((varStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH);
    superPatch               = ((varStatus & SUPERPATCH) == SUPERPATCH);

  } else {
    correctSize              = false;
    allocating               = false;
    allocated                = false;
    copyingIn                = false;
    validOnGPU               = false;
    gatheringGhostCells      = false;
    validWithGhostCellsOnGPU = false;
    formingSuperPatch        = false;
    superPatch               = false;
  }

  varLock->unlock();
}

//______________________________________________________________________
// returns false if something else already allocated space and we don't have to.
// returns true if we are the ones to allocate the space.
// performs operations with atomic compare and swaps
__host__ bool
GPUDataWarehouse::compareAndSwapAllocating(atomicDataStatus& status)
{

  bool allocating = false;

  while (!allocating) {

    //get the value
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(&status, 0);

    unsigned int refCounter = (oldVarStatus >> 16);

    //if it's allocated, return true
    if (refCounter >= 1 ) {
      //Something else already took care of it, and it has moved beyond the allocating state into something else.
      return false;
    } else if ((oldVarStatus & UNALLOCATED) != UNALLOCATED) {
      //Sanity check.  The ref counter was zero, but the variable isn't unallocated. We can't have this.
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Something wrongly modified the atomic status while setting the allocated flag\n");
      exit(-1);

    } else {
      //Attempt to claim we'll allocate it.  If not go back into our loop and recheck
      short refCounter = 1;
      atomicDataStatus newVarStatus = (refCounter << 16) | (oldVarStatus & 0xFFFF);  //Place in the reference counter and save the right 16 bits.
      newVarStatus = newVarStatus | ALLOCATING;                                       //It's possible to preserve a flag, such as copying in ghost cells.
      allocating = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);

    }
  }
  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after an allocating process completes.  *Only* the thread that got a true from
// compareAndSwapAllocating() should immediately call this.
__host__ bool
GPUDataWarehouse::compareAndSwapAllocate(atomicDataStatus& status)
{

  bool allocated = false;

  //get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);
  if ((oldVarStatus & ALLOCATING) == 0) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Can't allocate a status if it wasn't previously marked as allocating.\n");
    exit(-1);
  } else if  ((oldVarStatus & ALLOCATED) == ALLOCATED) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Can't allocate a status if it's already allocated\n");
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
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Something wrongly modified the atomic status while setting the allocated flag\n");
    exit(-1);
  }
  return allocated;
}

//______________________________________________________________________
// Simply determines if a variable has been marked as allocated.
__host__ bool
GPUDataWarehouse::checkAllocated(atomicDataStatus& status)
{

  return ((__sync_or_and_fetch(&status, 0) & ALLOCATED) == ALLOCATED);
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::compareAndSwapDeallocating(atomicDataStatus& status)
{

  bool deallocating = false;

  while (!deallocating) {

    //get the value
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(&status, 0);

    unsigned int refCounter = (oldVarStatus >> 16);
    if (refCounter == 0
        || ((oldVarStatus & DEALLOCATING) == DEALLOCATING)
        || ((oldVarStatus & 0xFFFF) == UNALLOCATED)
        || ((oldVarStatus & UNKNOWN) == UNKNOWN)) {
      //There's nothing to deallocate, or something else already deallocated it or is deallocating it.
      //So this thread won't do it.
      return false;
    } else if (refCounter == 1) {
        //Ref counter is 1, we can deallocate it.
        //Leave the refCounter at 1.
        atomicDataStatus newVarStatus = (refCounter << 16) | (oldVarStatus & 0xFFFF);  //Place in the reference counter and save the right 16 bits.
        newVarStatus = newVarStatus | DEALLOCATING;  //Set it to deallocating so nobody else can attempt to use it
        bool successfulUpdate = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
        if (successfulUpdate) {
          //Need to deallocate, let the caller know it.
          deallocating = true;
        }
    } else  if (refCounter > 1) {
      //Something else is using this variable, don't deallocate, just decrement the counter
      refCounter--;
      atomicDataStatus newVarStatus = (refCounter << 16) | (oldVarStatus & 0xFFFF);
      bool successfulUpdate = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
      if (successfulUpdate) {
        //No need to deallocate, let the caller know it.
        return false;
      }
    } else {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocating( )  This variable's ref counter was 0, but its status said it was in use.  This shouldn't happen\n");
      exit(-1);
    }
  }
  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after a deallocating process completes.  *Only* the thread that got a true from
//compareAndSwapDeallocating() should immediately call this.
__host__ bool
GPUDataWarehouse::compareAndSwapDeallocate(atomicDataStatus& status)
{

  bool allocated = false;

  //get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&status, 0);
  unsigned int refCounter = (oldVarStatus >> 16);

  if ((oldVarStatus & DEALLOCATING) == 0) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Can't deallocate a status if it wasn't previously marked as deallocating.\n");
    exit(-1);
  } else if  ((oldVarStatus & 0xFFFF) == UNALLOCATED) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Can't deallocate a status if it's already deallocated\n");
    exit(-1);
  } else if (refCounter != 1) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Attemping to deallocate a variable but the ref counter isn't the required value of 1\n");
    exit(-1);
  } else {
    //Attempt to claim we'll deallocate it.  Create what we want the status to look like
    //by turning off all status flags (indicating unallocated), it should also zero out the reference counter.
    atomicDataStatus newVarStatus = UNALLOCATED;

    //If we succeeded in our attempt to claim to deallocate, this returns true.
    //If we failed, thats a real problem, and we crash the problem below.
    allocated = __sync_bool_compare_and_swap(&status, oldVarStatus, newVarStatus);
  }
  if (!allocated) {
    //Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapDeallocate( )  Something wrongly modified the atomic status while trying set the status flags to unallocated\n");
    exit(-1);
  }
  return allocated;
}

//______________________________________________________________________
// Simply determines if a variable has been marked as valid.
__host__ bool
GPUDataWarehouse::checkValid(atomicDataStatus& status)
{

  return ((__sync_or_and_fetch(&status, 0) & VALID) == VALID);
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) & ALLOCATED) == ALLOCATED);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //cout << "In isAllocatedOnGPU - For patchID " << patchID << " for the status is " << getDisplayableStatusCodes(varPointers->at(lpml).atomicStatusInGpuMemory) << endl;
    bool retVal = ((__sync_fetch_and_or(&(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) & ALLOCATED) == ALLOCATED);
    if (retVal) {
      //now check the sizes
      int3 device_offset = varPointers->at(lpml).var->device_offset;
      int3 device_size = varPointers->at(lpml).var->device_size;
      retVal = (device_offset.x == offset.x && device_offset.y == offset.y && device_offset.z == offset.z
                && device_size.x == size.x && device_size.y == size.y && device_size.z == size.z);
    }
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) & VALID) == VALID);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
__host__ bool
GPUDataWarehouse::compareAndSwapSetValidOnGPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(it->second.var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) == VALID) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
        newVarStatus = newVarStatus | VALID;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR\nGPUDataWarehouse::compareAndSwapSetValidOnGPU() - Unknown variable %s on GPUDataWarehouse\n", label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
__host__ bool
GPUDataWarehouse::compareAndSwapSetInvalidOnGPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

    if (it != varPointers->end()) {

      atomicDataStatus *status = &(it->second.var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);

      // somehow COPYING_IN flag is not getting reset at some places while setting VALID (or getting set by mistake). Which causes race conditions and hangs
      // so reset COPYING_IN and VALID_WITH_GHOSTS flags here
      if ((oldVarStatus & VALID) != VALID && (oldVarStatus & COPYING_IN) != COPYING_IN && (oldVarStatus & VALID_WITH_GHOSTS) != VALID_WITH_GHOSTS ) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        // Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID;
        newVarStatus = newVarStatus & ~VALID_WITH_GHOSTS;
        newVarStatus = newVarStatus & ~COPYING_IN;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
        it->second.var->curGhostCells = -1;
      }
    } else {
      varLock->unlock();
      return false;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
__host__ bool
GPUDataWarehouse::compareAndSwapSetValidOnGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{
  varLock->lock();
  bool settingValidOnStaging = false;
  while (!settingValidOnStaging) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

    if (it != varPointers->end()) {

      stagingVar sv;
      sv.device_offset = offset;
      sv.device_size = size;

      std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.find(sv);

      if (staging_it != it->second.var->stagingVars.end()) {
        atomicDataStatus *status = &(staging_it->second.atomicStatusInGpuMemory);
        atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);

        if ((oldVarStatus & VALID) == VALID) {
          //Something else already took care of it.  So this task won't manage it.
          varLock->unlock();
          return false;
        } else {
          //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
          atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
          newVarStatus = newVarStatus | VALID;
          settingValidOnStaging = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
        }

      } else {
        varLock->unlock();
        printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetValidOnGPUStaging( )  Staging variable %s not found.\n", label);
        exit(-1);
      }
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetValidOnGPUStaging( )  Variable %s not found.\n", label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// We have an entry for this item in the GPU DW, and it's not unknown.  Therefore
// if this returns true it means this GPU DW specifically knows something about the
// state of this variable. (The reason for the unknown check is currently when a
// var is added to the GPUDW, we also need to state what we know about its data in
// host memory.  Since it doesn't know, it marks it as unknown, meaning, the host
// side DW is possibly managing the data.)
__host__ bool GPUDataWarehouse::dwEntryExistsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx) {

   varLock->lock();
   bool retVal = false;
   labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
   std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
   if (it != varPointers->end()) {
     if  ((it->second.var->atomicStatusInHostMemory & UNKNOWN) != UNKNOWN) {

       retVal = true;
     }
   }
   varLock->unlock();
   return retVal;

}

//______________________________________________________________________
// We have an entry for this item in the GPU DW. status does not matter.
__host__ bool GPUDataWarehouse::dwEntryExists(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  bool retVal = false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    retVal = true;
  }
  varLock->unlock();
  return retVal;
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isValidOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {


    bool retVal = ((__sync_fetch_and_or(&(varPointers->at(lpml).var->atomicStatusInHostMemory), 0) & VALID) == VALID);
    varLock->unlock();
    return retVal;

  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//TODO: This needs to be turned into a compare and swap operation
//______________________________________________________________________
__host__ bool
GPUDataWarehouse::compareAndSwapSetValidOnCPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(it->second.var->atomicStatusInHostMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) == VALID) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
        newVarStatus = newVarStatus | VALID;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR\nGPUDataWarehouse::compareAndSwapSetValidOnCPU() - Unknown variable %s on GPUDataWarehouse\n", label);
      exit(-1);
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::compareAndSwapSetInvalidOnCPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(it->second.var->atomicStatusInHostMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      //somehow COPYING_IN flag is not getting reset at some places while setting VALID (or getting set by mistake). Which causes race conditions and hangs
      //so reset COPYING_IN and VALID_WITH_GHOSTS flags here      
      if ((oldVarStatus & VALID) != VALID && (oldVarStatus & COPYING_IN) != COPYING_IN && (oldVarStatus & VALID_WITH_GHOSTS) != VALID_WITH_GHOSTS ) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID;
        newVarStatus = newVarStatus & ~VALID_WITH_GHOSTS;
        newVarStatus = newVarStatus & ~COPYING_IN;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      return false;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// returns false if something else already changed a valid variable to valid awaiting ghost data
// returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::compareAndSwapAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool allocating = false;

  varLock->lock();
  while (!allocating) {
    //get the address
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    if (varPointers->find(lpml) != varPointers->end()) {
      atomicDataStatus *status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if (((oldVarStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) || ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus | AWAITING_GHOST_COPY;
        allocating = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapAwaitingGhostDataOnGPU( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data into the GPU.
// returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::compareAndSwapCopyingIntoGPU(char const* label, int patchID, int matlIndx, int levelIndx, int numGhosts/*=0*/)
{

  atomicDataStatus* status = nullptr;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    status = &(it->second.var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPU( )  Variable %s not found.\n", label);
    exit(-1);
    return false;
  }

  bool copyingin = false;
  while (!copyingin) {
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
     varLock->unlock();
     printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPU( )  Variable %s is unallocated.\n", label);
     exit(-1);
    }
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
        ((oldVarStatus & VALID) == VALID) /*||
        ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)*/) {
      // Something else already took care of it.  So this task won't manage it.
      varLock->unlock();
      return false;
    } else {
      // Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      it->second.var->curGhostCells = numGhosts;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
// returns false if delayed copy not needed i.e. current ghost cell is less than or equal to incoming ghost cells.
// returns true otherwise
__host__ bool
GPUDataWarehouse::isDelayedCopyingNeededOnGPU(char const* const label, int patchID, int matlIndx, int levelIndx, int numGhosts)
{
  bool retval=false;
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    if(it->second.var->curGhostCells < numGhosts){
      it->second.var->curGhostCells = numGhosts;
      retval = true;
    }
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::isDelayedCopyingNeededOnGPU( )  Variable %s not found.\n", label);
    exit(-1);
  }
  varLock->unlock();
  return retval;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data into the CPU.
// returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::compareAndSwapCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  atomicDataStatus* status = nullptr;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(it->second.var->atomicStatusInHostMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoCPU( )  Variable %s not found.\n", label);
    exit(-1);
    return false;
  }
  varLock->unlock();

  bool copyingin = false;
  while (!copyingin) {
    // get the address
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
       ((oldVarStatus & VALID) == VALID) ||
       ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
        // Something else already took care of it.  So this task won't manage it.
        return false;
      } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      newVarStatus = newVarStatus & ~UNKNOWN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//______________________________________________________________________
// returns false if something else already claimed to copy or has copied data into the GPU.
// returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size)
{

  atomicDataStatus* status;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {

    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.var->stagingVars.find(sv);
    if (staging_it != it->second.var->stagingVars.end()) {
      status = &(staging_it->second.atomicStatusInGpuMemory);
    } else {
      varLock->unlock();
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( )  Staging variable %s not found.\n", label);
      exit(-1);
      return false;
    }
  } else {
   varLock->unlock();
   printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( )  Variable %s not found.\n", label);
   exit(-1);
   return false;
  }
  varLock->unlock();

  bool copyingin = false;

  while (!copyingin) {
    //get the address
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (oldVarStatus == UNALLOCATED) {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( )  Variable %s is unallocated.\n", label);
      exit(-1);
    } else if ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS) {
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapCopyingIntoGPUStaging( )  Variable %s is marked as valid with ghosts, that should never happen with staging vars.\n", label);
      exit(-1);
    } else if (((oldVarStatus & COPYING_IN) == COPYING_IN) /*||
               ((oldVarStatus & VALID) == VALID)*/) {
    	//DS 06032020: Commented "((oldVarStatus & VALID) == VALID)" condition as a temporary fix for the defect: When a variable is modified on GPU, ValidWithGhost status
    	//is reverted to allow gathering of ghost cells again for the next requires dependency. But as of now there is no mechanism to mark staging variables
    	//invalid. As a result, although the ghost cells on the main variable are invalidated, staging variables still have valid status and hold old values and
    	//because of valid status prepareDeviceVars does not issue fresh H2D copy for the staging variable.
    	//The permanent fix is to find out and inactivate staging variables of neighboring patches. But that needs more time. So this temporary fix to ignore
    	//valid status as of now which will cause few redundant h2d copies, but will make code work.
//      printf("compareAndSwapCopyingIntoGPUStaging: %s %d COPYING_IN: %d VALID: %d\n", label, patchID, (oldVarStatus & COPYING_IN) == COPYING_IN, (oldVarStatus & VALID) == VALID );

      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      newVarStatus = newVarStatus & ~VALID; //DS 06032020: temp fix
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }

  return true;
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    bool retVal = ((__sync_fetch_and_or(&(it->second.var->atomicStatusInGpuMemory), 0) & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    varLock->unlock();
    return retVal;
  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//TODO: This needs to be turned into a compare and swap operation
__host__ void
GPUDataWarehouse::setValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if (it != varPointers->end()) {
    //UNKNOWN
    //make sure the valid is still turned on
    //do not set VALID here because one thread can gather the main patch and other gathers ghost cells
    //if ghost cells is finished first, setting valid here causes task to start even though other thread 
    //copying the main patch is not completed. race condition. Removed VALID_WITH_GHOSTS with from compareAndSwapCopyingInto*
    //add extra condition to check valid AND valid with ghost both in UnifiedSchedular::allGPUProcessingVarsReady
    //__sync_or_and_fetch(&(it->second.var->atomicStatusInGpuMemory), VALID);

    //turn off AWAITING_GHOST_COPY
    __sync_and_and_fetch(&(it->second.var->atomicStatusInGpuMemory), ~AWAITING_GHOST_COPY);

    //turn on VALID_WITH_GHOSTS
    __sync_or_and_fetch(&(it->second.var->atomicStatusInGpuMemory), VALID_WITH_GHOSTS);

    varLock->unlock();
  } else {
    varLock->unlock();
    exit(-1);
  }
}

//______________________________________________________________________
// returns false if something else already changed a valid variable to valid awaiting ghost data
// returns true if we are the ones to manage this variable's ghost data.
__host__ bool
GPUDataWarehouse::compareAndSwapSetInvalidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool allocating = false;

  varLock->lock();
  while (!allocating) {
    //get the address
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    auto it = varPointers->find(lpml);
    if (it != varPointers->end()) {
      atomicDataStatus *status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID_WITH_GHOSTS) == 0) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID_WITH_GHOSTS;
        allocating = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
        it->second.var->curGhostCells = -1;
      }
    } else {
      varLock->unlock();
      /* DS 11052019: Commented error. Even if variable is not found in the gpu dw, consider it to be invalid (at least in principle)
       * This is needed to mark variable in GPU dw invalid if it is modified on CPU. Should not throw error at this time if the variable is not existing on GPU
       */
      /*printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetInvalidWithGhostsOnGPU( )  Variable %s not found.\n", label);
      exit(-1);*/
      return false;
    }
  }
  varLock->unlock();
  return true;
}


//______________________________________________________________________
// returns true if successful if marking a variable as a superpatch.  False otherwise.
// Can only turn an unallocated variable into a superpatch.
__host__ bool
GPUDataWarehouse::compareAndSwapFormASuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool compareAndSwapSucceeded = false;

  //get the status
  atomicDataStatus* status = nullptr;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapFormASuperPatchGPU( )  Variable %s patch %d material %d levelIndx %d not found.\n", label, patchID, matlIndx, levelIndx);
    exit(-1);
    return false;
  }
  varLock->unlock();

  while (!compareAndSwapSucceeded) {

    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::compareAndSwapFormASuperPatchGPU() - "
           << " Attempting to set a superpatch flag for label " << label
           << " patch " << patchID
           << " matl " << matlIndx
           << " level " << levelIndx
           << " with status codes " << getDisplayableStatusCodes(oldVarStatus)
           << std::endl;
       }
       cerrLock.unlock();
     }

    if ( (oldVarStatus & FORMING_SUPERPATCH) == FORMING_SUPERPATCH
       || ((oldVarStatus & SUPERPATCH) == SUPERPATCH)) {
      //Something else already took care of it.  So this task won't manage it.
      return false;
    } else if (((oldVarStatus & ALLOCATING) == ALLOCATING)
              || ((oldVarStatus & ALLOCATED) == ALLOCATED)
              || ((oldVarStatus & ALLOCATING) == ALLOCATING)
              || ((oldVarStatus & COPYING_IN) == COPYING_IN)
              || ((oldVarStatus & VALID) == VALID)
              || ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)
              || ((oldVarStatus & DEALLOCATING) == DEALLOCATING)) {
              //Note, we DO allow a variable to be set as AWAITING_GHOST_COPY before anything else.

      //At the time of implementation this scenario shouldn't ever happen.  If so it means
      //Someone is requesting to take a variable already in memory that's not a superpatch
      //and turn it into a superpatch.  It would require some kind of special deep copy mechanism
      printf("ERROR:\nGPUDataWarehouse::compareAndSwapFormASuperPatchGPU( )  Variable %s cannot be turned into a superpatch, it's in use already with status %s.\n", label, getDisplayableStatusCodes(oldVarStatus).c_str());
      exit(-1);
      return false;
    } else {
      atomicDataStatus newVarStatus = oldVarStatus | FORMING_SUPERPATCH;
      compareAndSwapSucceeded = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }

  atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
  if (gpu_stats.active()) {
     cerrLock.lock();
     {
       gpu_stats << UnifiedScheduler::myRankThread()
         << " GPUDataWarehouse::compareAndSwapFormASuperPatchGPU() - "
         << " Success for label " << label
         << " patch " << patchID
         << " matl " << matlIndx
         << " level " << levelIndx
         << " with status codes " << getDisplayableStatusCodes(oldVarStatus)
         << std::endl;
     }
     cerrLock.unlock();
   }

  return true;
}

//______________________________________________________________________
// Sets the allocated flag on a variables atomicDataStatus
// This is called after a forming a superpatch process completes.  *Only* the thread that got to set FORMING_SUPERPATCH can
// set SUPERPATCH.  Further, no other thread should modify the atomic status
//compareAndSwapFormASuperPatchGPU() should immediately call this.
__host__ bool
GPUDataWarehouse::compareAndSwapSetSuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  bool superpatched = false;

  //get the status
  atomicDataStatus* status = nullptr;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    status = &(varPointers->at(lpml).var->atomicStatusInGpuMemory);
  } else {
    varLock->unlock();
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  Variable %s patch %d material %d levelIndx %d not found.\n", label, patchID, matlIndx, levelIndx);
    exit(-1);
    return false;
  }
  varLock->unlock();

  const atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
  if ((oldVarStatus & FORMING_SUPERPATCH) == 0) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  Can't set a superpatch status if it wasn't previously marked as forming a superpatch.\n");
    exit(-1);
  } else {
    //Attempt to claim forming it into a superpatch.
    atomicDataStatus newVarStatus = oldVarStatus;
    newVarStatus = newVarStatus & ~FORMING_SUPERPATCH;
    newVarStatus = newVarStatus | SUPERPATCH;

    //If we succeeded in our attempt to claim to deallocate, this returns true.
    //If we failed, thats a real problem, and we crash below.
    //printf("current status is %s oldVarStatus is %s newVarStatus is %s\n", getDisplayableStatusCodes(status)
    superpatched = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
  }
  if (!superpatched) {
    //Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapSetSuperPatchGPU( )  Something modified the atomic status between the phases of forming a superpatch and setting a superpatch.  This shouldn't happen\n");
    exit(-1);
  }
  return superpatched;
}

//______________________________________________________________________
//
__host__ bool
GPUDataWarehouse::isSuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  bool retVal = false;
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    retVal = ((__sync_fetch_and_or(&(varPointers->at(lpml).var->atomicStatusInGpuMemory), 0) & SUPERPATCH) == SUPERPATCH);
  }
  varLock->unlock();
  return retVal;
}

//______________________________________________________________________
//
__host__ void GPUDataWarehouse::setSuperPatchLowAndSize(char const* const label, const int patchID, const int matlIndx, const int levelIndx,
                                                        const int3& low, const int3& size){
  varLock->lock();

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  if ( it == varPointers->end()) {
    printf("ERROR: GPUDataWarehouse::setSuperPatchLowAndSize - Didn't find a variable for label %s patch %d matl %d level %d\n",
          label, patchID, matlIndx, levelIndx);
   varLock->unlock();
   exit(-1);
  }
  it->second.var->device_offset = low;
  it->second.var->device_size = size;
  varLock->unlock();

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
GPUDataWarehouse::printError(const char* msg, const char* methodName, char const* label, const int patchID, int8_t matlIndx, int8_t levelIndx )
{
#ifdef __CUDA_ARCH__
  __syncthreads();

  if ( isThread0() ) {
    if (label[0] == '\0') {
      printf("  \nERROR GPU-side: GPUDataWarehouse::%s() - %s\n", methodName, msg );
    }
    else {
      printf("  \nERROR GPU-side: GPUDataWarehouse::%s(), label:  \"%s\", patch: %i, matlIndx: %i, levelIndx: %i - %s\n", methodName, label, patchID, matlIndx, levelIndx, msg);
    }
    //Should this just loop through the variable database and print out only items with a
    //levelIndx value greater than zero? -- Brad

    //for ( int i = 0; i < d_numLevelItems; i++ ) {
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
  if ( label[0] == '\0' ) {
    printf("  \nERROR host-side: GPUDataWarehouse::%s() - %s\n", methodName, msg );
  }
  else {
    printf("  \nERROR host-side: GPUDataWarehouse::%s(), label:  \"%s\", patch: %i, matlIndx: %i, levelIndx: %i - %s\n", methodName, label, patchID, matlIndx, levelIndx, msg);
  }
  exit(-1);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printGetLevelError(const char* msg, char const* label, int8_t levelIndx, int8_t matlIndx)
{
#ifdef __CUDA_ARCH__
  __syncthreads();

  if ( isThread0() ) {
    printf("  \nERROR: %s( \"%s\", levelIndx: %i, matl: %i)  unknown variable\n", msg,  label, levelIndx, matlIndx);
    //Should this just loop through the variable database and print out only items with a
    //levelIndx value greater than zero? -- Brad

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
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::printGetError(const char* msg, char const* label, int8_t levelIndx, const int patchID, int8_t matlIndx)
{
#ifdef __CUDA_ARCH__
  __syncthreads();

  if ( isThread0() ) {
    printf("  \nERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i)  unknown variable\n", msg,  label, levelIndx, patchID, matlIndx);

    if ( d_numVarDBItems == 0 ) {
      printf("\tEmpty GPU-DW\n");
    }
    else {
      for ( int i = 0; i < d_numVarDBItems; i++ ) {
        printf("   Available varDB labels(%i of %i): \"%-15s\" matl: %i, patchID: %i, level: %i\n", i, d_numVarDBItems, d_varDB[i].label, d_varDB[i].matlIndx, d_varDB[i].domainID, d_varDB[i].levelIndx);
      }
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
  printf("  \nERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i)  unknown variable in DW %s\n", msg, label, levelIndx, patchID, matlIndx, _internalName);

  if ( d_numVarDBItems == 0 ) {
    printf("\tEmpty GPU-DW\n");
  }
  else {
    for ( int i = 0; i < d_numVarDBItems; i++ ) {
      printf("   Available varDB labels(%i): \"%-15s\" matl: %i, patchID: %i, level: %i\n", d_numVarDBItems, d_varDB[i].label, d_varDB[i].matlIndx, d_varDB[i].domainID, d_varDB[i].levelIndx);
    }
  }
#endif
}


//______________________________________________________________________
//
__host__ void*
GPUDataWarehouse::getPlacementNewBuffer()
{
  return placementNewBuffer;
}

//______________________________________________________________________
//  Returns true if threadID and blockID are 0.
//  Useful in conditional statements for limiting output.
//
__device__ bool
GPUDataWarehouse::isThread0_Blk0()
{
  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

  bool test (blockID == 0 && threadID == 0);
  return test;
}

//______________________________________________________________________
//  Returns true if threadID = 0 for this block
//  Useful in conditional statements for limiting output.
//
__device__ bool
GPUDataWarehouse::isThread0()
{
  int threadID = threadIdx.x +  threadIdx.y +  threadIdx.z;
  bool test (threadID == 0 );
  return test;
}

//______________________________________________________________________
// Output the threadID
//
__device__ void
GPUDataWarehouse::printThread()
{
  int threadID = threadIdx.x +  threadIdx.y +  threadIdx.z;
  printf( "Thread [%i,%i,%i], ID: %i\n", threadIdx.x,threadIdx.y,threadIdx.z, threadID);
}

//______________________________________________________________________
// Output the blockID
//
__device__ void
GPUDataWarehouse::printBlock()
{
  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  printf( "Block  [%i,%i,%i], ID: %i\n", blockIdx.x,blockIdx.y,blockIdx.z, blockID);
}
