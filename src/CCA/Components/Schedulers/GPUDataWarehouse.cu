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
HOST_DEVICE void 
GPUDataWarehouse::put(GPUGridVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, bool staging,
                      GhostType gtype, int numGhostCells, void* host_ptr)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printError("This method not allowed on the device.  All memory should be allocated host side", "put", label, patchID, matlIndx, levelIndx);
#else

  varLock->writeLock();
  if (d_numVarDBItems==MAX_VARDB_ITEMS) {
    printf("ERROR:  Out of GPUDataWarehouse space");
    varLock->writeUnlock();
    exit(-1);
  }

  int3 var_offset;        // offset
  int3 var_size;          // dimensions of GPUGridVariable
  void* var_ptr;           // raw pointer to the memory
  var.getArray3(var_offset, var_size, var_ptr);

  bool insertIntodVarDB = false;
  int d_varDB_index = -1;  // If it doesn't have a match in d_varDB, it should stay -1
  if (d_device_copy != NULL ) {
    //If this GPU DW is a Task GPU DW, then it will be copied to the GPU, so fill the d_varDB
    insertIntodVarDB = true;
    d_varDB_index=d_numVarDBItems;
    d_numVarDBItems++;
  }

  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);


  if (staging == false) {

    allVarPointersInfo vp;

    vp.varDB_index = d_varDB_index;
    vp.device_ptr = var_ptr;
    vp.device_offset =  var_offset;
    vp.device_size = var_size;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = gtype;
    vp.numGhostCells = numGhostCells;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = ALLOCATED;
    vp.host_contiguousArrayPtr = host_ptr;
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
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << " current varPointers size is: " << varPointers->size()
            << endl;
      }

      cerrLock.unlock();
    }

    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );

  } else { // if (staging == true)

    //see if there's a regular/non-staging var we can piggyback on.
    //Find this item.
    std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

    if (it == varPointers->end()) {
      //We need to input a staging variable, but there isn't a regular non-staging variable at the same label/patch/matl/level.
      //So go ahead and make an empty placeholder regular non-staging variable entry.
      //non staging vars create a new varPointers entry
      //I believe this scenario happens when GPU ghost cells are sent to this MPI rank from another rank.
      //This rank has no regular/non-staging var for those ghost cells, just the staging var.
      allVarPointersInfo vp;

      vp.varDB_index = -1; //no need for a d_varDB array entry.
      vp.device_ptr = NULL;
      int3 temp;
      temp.x = 0;
      temp.y = 0;
      temp.z = 0;
      vp.device_offset = temp;
      vp.device_size = temp;
      vp.sizeOfDataType = 0;
      vp.gtype = None;
      vp.numGhostCells = 0;
      vp.atomicStatusInHostMemory = UNKNOWN;
      vp.atomicStatusInGpuMemory = ALLOCATED;

      vp.host_contiguousArrayPtr = NULL;
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUDataWarehouse::put( " << label << " ) - "
              << " Put an EMPTY PLACEHOLDER regular non-staging variable in the host-side varPointers map for label " << label
              << " patch " << patchID
              << " matl " << matlIndx
              << " level " << levelIndx
              << " on device " << d_device_id
              << " into GPUDW at " << std::hex << this << std::dec
              << endl;
        }
        cerrLock.unlock();
      }
      //insert it and get a new iterator back.
      varPointers->insert(std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
      it = varPointers->find(lpml);
    }

    stagingVar sv;
    sv.device_offset = var_offset;
    sv.device_size = var_size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);

    stagingVarInfo svi;
    svi.device_ptr = var_ptr;
    svi.host_contiguousArrayPtr = host_ptr;
    svi.varDB_index = d_varDB_index;


    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put a staging variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " at device address " << var_ptr
            << " with offset (" << var_offset.x << ", " << var_offset.y << ", " << var_offset.z << ")"
            << " and size (" << var_size.x << ", " << var_size.y << ", " << var_size.z << ")"
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << endl;
      }
      cerrLock.unlock();
    }

    std::pair<stagingVar, stagingVarInfo> p = make_pair( sv, svi );
    //std::map<stagingVar, stagingVarInfo>::value_type( sv, svi )

    it->second.stagingVars.insert( p  );

  }

  if (insertIntodVarDB) {

    int i = d_varDB_index;
    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    d_varDB[i].sizeOfDataType = sizeOfDataType;
    d_varDB[i].varItem.gtype = gtype;
    d_varDB[i].varItem.numGhostCells = numGhostCells;
    d_varDB[i].varItem.staging = staging;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    d_varDB[i].var_offset = var_offset;
    d_varDB[i].var_size = var_size;
    d_varDB[i].var_ptr = var_ptr;

    d_dirty=true;
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put into d_varDB label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx;
        if (staging) {
          gpu_stats << " staging: true";
        } else {
          gpu_stats << " staging: false";
        }
        gpu_stats  << " into address " << d_varDB[i].var_ptr
            << " into  d_varDB index " << i
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " size [" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << "]"
            << " offset [" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << "]"
            << endl;
      }
      cerrLock.unlock();
    }
  }
  varLock->writeUnlock();

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype, int numGhostCells)
{

//Allocate space on the GPU and declare a variable onto the GPU.
//This method does NOT stage everything in a big array.
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n",label);
#else
  //__________________________________
  //  cpu code
  //check if it exists prior to allocating memory for it.
  //One scenario in which we need to handle is when two patches need ghost cells from the
  //same neighbor patch.  Suppose thread A's GPU patch 1 needs data from CPU patch 0, and
  //thread B's GPU patch 2 needs data from CPU patch 0.  This is a messy scenario, but the cleanest
  //is to have the first arriver between A and B to allocate memory on the GPU,
  //then the second arriver will use the same memory. The simplest approach is to have
  //threads A and B both copy the entire patch 0 into this GPU memory.  This should be ok as ghost cell
  //data cannot be modified, so in case of a race condition, nothing will be modified
  //if a memory chunk is being both read and written to simultaneously.
  //So allow duplicates if the data is read only, otherwise don't allow duplicates.

  //BUG: A race condition exists through the following:
  //Thread A for Patch A allocates space for a region X, writes into shared GPUDW
  //Thread B for Patch B sees this has already been allocated for region X at memory address for A.
  //Thread B launches a task for Patch B, using the memory address for A, using garbage data.
  //Thread A copies its border region X data host to device


  bool putNeeded = true;
  int3 size = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;

  varLock->writeLock();

  //First check if this variable is in the database already (it may not be allocated, but we have a database slot for it)
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);
  std::map<stagingVar, stagingVarInfo>::iterator staging_it;
  if (it != varPointers->end() && staging) {
    stagingVar sv;
    sv.device_offset = offset;
    sv.device_size = size;
    staging_it = it->second.stagingVars.find(sv);
  } //Note: If no regular var exists and a staging var is needed, the put() method will properly take care of that.


  //This prepares the var with the offset and size.  Any possible allocation will come later.
  //If it needs to go into the database, that will also come later
  void* addr = NULL;
  var.setArray3(offset, size, addr);

  //Now see if we allocate the variable or use a previous existing allocation.
  bool allocationNeeded = true;
  if (it != varPointers->end()) {
    if (staging == false) {

      //This variable exists in the database, no need to "put" it in again.
      putNeeded = false;

      allocationNeeded = testAndSetAllocating(it->second.atomicStatusInGpuMemory);

      if (!allocationNeeded) {
        if (it->second.device_offset.y == low.y
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
                  << " into GPUDW at " << std::hex << this << std::dec
                  << endl;
             }
             cerrLock.unlock();
           }
           //We need the pointer.
           //Ensure that it has been allocated.  Another thread may have been assigned to allocate it
           //but not completed that action.  If that's the case, wait until it's done so we can get the pointer.
           bool allocated = false;
           while (!allocated) {
             allocated = checkAllocated(it->second.atomicStatusInGpuMemory);
           }
           //Have this var use the existing memory address.
           var.setArray3(it->second.device_offset, it->second.device_size, it->second.device_ptr);
        } else {
          printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  Variable in database but of the wrong size.  This shouldn't ever happen.\n",label);
          exit(-1);
        }
      }
    } else {
      //it's a staging variable

      if (staging_it != it->second.stagingVars.end()) {

        //This variable exists in the database, no need to "put" it in again.
        putNeeded = false;

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
                  << " into GPUDW at " << std::hex << this << std::dec
                  << endl;
            }
            cerrLock.unlock();
          }
          //We need the pointer.
          //Ensure that it has been allocated.  Another thread may have been assigned to allocate it
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
  }

  //Now allocate it
  if (allocationNeeded) {

    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);

    CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::allocateAndPut(), cudaMalloc"
            << " for " << label
            << " patch " << patchID
            << " material " <<  matlIndx
            << " level " << levelIndx;
        if (staging) {
          gpu_stats << " staging: true";
        } else {
          gpu_stats << " staging: false";
        }
        gpu_stats << " size " << var.getMemSize()
            << " from (" << low.x << "," << low.y << "," << low.z << ")"
            << " to (" << high.x << "," << high.y << "," << high.z << ")"
            << " at " << addr
            << " on device " << d_device_id << endl;
      }
      cerrLock.unlock();
    }

    //We have the pointer!  Add it in.
    var.setArray3(offset, size, addr);

    varLock->writeUnlock();


    if (putNeeded) {
      //put() performs its own locking
      //Create a brand new var entry into the database. This also sets the proper atomicDataStatus flags.
      put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, gtype, numGhostCells);
    } else {
      //The variables were already in the database.  But we allocated memory for it.
      //So set the proper atomicDataStatus flags.
      //Locking not needed here.  STL maps ensure that iterators point to correct values
      //even if other threads add nodes.
      if (!staging) {
        testAndSetAllocate(it->second.atomicStatusInGpuMemory);
      } else {
        testAndSetAllocate(staging_it->second.atomicStatusInGpuMemory);
      }
    }
  } else {
    varLock->writeUnlock();
  }


#endif
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
    printf("In allocate(), cuda malloc for size %ld at %p on device %d\n", size, d_ptr, d_device_id);
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

HOST_DEVICE cudaError_t
GPUDataWarehouse::copyDataHostToDevice(const char* indexID, void *cuda_stream) {
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
  //return something so there is no compiler warning
  cudaError_t retVal = cudaErrorUnknown;
  return retVal;
#else
  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
  allocateLock->readLock();
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays->begin(); it != contiguousArrays->end(); ++it)
  //  printf("*** displaying %s\n", it->first.c_str());
  contiguousArrayInfo *ca = &(contiguousArrays->operator[](indexID));


  allocateLock->readUnlock();
  //copy only the initialized data, not the whole thing.
  //printf("Copying to device %p from host %p amount %d\n", ca->allocatedDeviceMemory, ca->allocatedHostMemory, ca->copiedOffset);
  //cudaError_t retVal = cudaMemcpy(ca->allocatedDeviceMemory, ca->allocatedHostMemory,
  //                                     ca->copiedOffset, cudaMemcpyHostToDevice);
  cudaError_t retVal;
  CUDA_RT_SAFE_CALL ( retVal = cudaMemcpyAsync(ca->allocatedDeviceMemory, ca->allocatedHostMemory,
                                       ca->copiedOffset, cudaMemcpyHostToDevice, *((cudaStream_t*)cuda_stream)) );

  return retVal;

#endif
}

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
HOST_DEVICE void
GPUDataWarehouse::put(GPUReductionVariableBase &var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx,
                      bool staging, void* host_ptr)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", label);
#else
  varLock->writeLock();
  if (d_numVarDBItems==MAX_VARDB_ITEMS) {
    printf("ERROR:  Out of GPUDataWarehouse space");
    varLock->writeUnlock();  // writeLock() is called from allocateAndPut(). This is the escape clause if things go bad
    exit(-1);
  }

  //first check if this patch/var/matl/level is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  allVarPointersInfo vp;
  vp.host_contiguousArrayPtr = host_ptr;

  if (d_device_copy != NULL ) {
    int i=d_numVarDBItems;
    d_numVarDBItems++;
    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx = levelIndx;
    //d_varDB[i].staging = staging;
    d_varDB[i].sizeOfDataType = sizeOfDataType;
    d_varDB[i].varItem.gtype = None;  //PerPatch has no ghost cells
    d_varDB[i].varItem.numGhostCells = 0;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    var.getData(d_varDB[i].var_ptr);

    vp.device_ptr = d_varDB[i].var_ptr;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = ALLOCATED;
    vp.varDB_index = i;

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put a reduction variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " at device address " << d_varDB[i].var_ptr
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << " current varPointers size is: " << varPointers->size()
            << endl;
      }

      cerrLock.unlock();
    }

    d_dirty=true;
  } else {
    var.getData(vp.device_ptr);
    vp.varDB_index = -1;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = ALLOCATED;
  }

  if (varPointers->find(lpml) == varPointers->end()) {
    if (d_debug){
      printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d level %d on device %d in this DW %p\n",
              label, label, patchID, matlIndx, levelIndx, d_device_id, this);
    }
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    varLock->writeUnlock();
  } else {
    printf("ERROR:\nGPUDataWarehouse::put( %s )  This gpudw database already has a variable for label %s patch %d matl %d level %d on device %d in GPUDW at %p\n",
            label, label, patchID, matlIndx, levelIndx, d_device_id, d_device_copy);
    varLock->writeUnlock();
    exit(-1);
  }

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUPerPatchBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx, void* host_ptr)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All device memory should be allocated on the CPU with cudaMalloc\n", label);
#else

  varLock->writeLock();
  if (d_numVarDBItems==MAX_VARDB_ITEMS) {
    printf("ERROR:  out of GPUDataWarehouse space");
    varLock->writeUnlock();

    exit(-1);
  }

  //first check if this patch/var/matl is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  //for (std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->begin(); it != varPointers->end();  ++it) {
  //    printf("For label %s patch %d matl %d on device %d   I have an entry\n", (*it).first.label.c_str(), d_device_id, (*it).first.patchID, (*it).first.matlIndx);
  //}

  allVarPointersInfo vp;
  vp.host_contiguousArrayPtr = host_ptr;

  bool insertIntodVarDB = false;
  int d_varDB_index = -1;  // If it doesn't have a match in d_varDB, it should stay -1
  void * device_ptr = NULL;
  if (d_device_copy != NULL ) {
    //If this GPU DW is a Task GPU DW, then it will be copied to the GPU, so fill the d_varDB
    insertIntodVarDB = true;
    d_varDB_index=d_numVarDBItems;
    d_numVarDBItems++;
  }

  if (d_device_copy != NULL ) {

    var.getData(device_ptr);
    vp.device_ptr = device_ptr;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = ALLOCATED;
    vp.varDB_index = d_varDB_index;
    //if (cmpstr(label, "dt") == 0) {
    //  double tempd = 1234.5678;
    //  void* temp_ptr = (void*)&tempd;
      //void* d_temp_ptr;
      //CUDA_RT_SAFE_CALL( cudaMalloc(&d_temp_ptr, 8 ));
    //  CUDA_RT_SAFE_CALL(retVal = cudaMemcpy(temp_ptr, device_ptr, 8, cudaMemcpyDeviceToHost));

    //}
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put a PerPatch variable in the host-side varPointers map for label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " at device address " << device_ptr
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " with description " << _internalName
            << " current varPointers size is: " << varPointers->size()
            << endl;
      }

      cerrLock.unlock();
    }
    d_dirty=true;
  } else {
    var.getData(vp.device_ptr);
    vp.varDB_index = -1;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.atomicStatusInHostMemory = UNKNOWN;
    vp.atomicStatusInGpuMemory = ALLOCATED;

  }

  varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );


  if (insertIntodVarDB) {

    int i=d_varDB_index;
    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx = levelIndx;
    //d_varDB[i].staging = false;
    d_varDB[i].sizeOfDataType = sizeOfDataType;
    d_varDB[i].varItem.gtype = None;  //PerPatch has no ghost cells
    d_varDB[i].varItem.numGhostCells = 0;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    d_varDB[i].var_ptr = device_ptr;

    d_dirty=true;
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::put( " << label << " ) - "
            << " Put into d_varDB PerPatch variable label " << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " into address " << d_varDB[i].var_ptr
            << " into  d_varDB index " << i
            << " on device " << d_device_id
            << " into GPUDW at " << std::hex << this << std::dec
            << " size [" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << "]"
            << " offset [" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << "]"
            << endl;
      }
      cerrLock.unlock();
    }
  }

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
#else
  varLock->writeLock();

  //first check if this patch/var/matl is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //Space for this patch already exists.  Use that and return.
    if (d_debug){
      printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a reduction variable for label %s patch %d matl %d level %d on device %d.  Reusing it.\n",
          label, label, patchID, matlIndx, levelIndx, d_device_id);
    }
    var.setData(varPointers->operator[](lpml).device_ptr);
    varLock->writeUnlock();
    return;
  }

  //It's not in the database, so create it.

  void* addr  = NULL;
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );


  if (d_debug) {
    printf("In allocateAndPut(), cudaMalloc for reduction variable %s patch %d material %d level %d size %ld on thread %d ",
            label, patchID, matlIndx, levelIndx, var.getMemSize(), SCIRun::Thread::self()->myid());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  varLock->writeUnlock();

  put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);


#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType)
{

#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
#else

  varLock->writeLock();

  //first check if we've already got this exact version of this variable already.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (it != varPointers->end()) {

     //BUG: A race condition exists through the following:
     //Thread A for Patch A allocates space for a region X, writes into shared GPUDW
     //Thread B for Patch B sees this has already been allocated for region X at memory address for A.
     //Thread B launches a task for Patch B, using the memory address for A, using garbage data.
     //Thread A copies its border region X data host to device

    //Space for this var already exists.  Use that and return.
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
           << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
           << " This non-staging/regular patch variable already exists.  No need to allocate another.  GPUDW has a variable for label " << label
           << " patch " << patchID
           << " matl " << matlIndx
           << " level " << levelIndx
           << " on device " << d_device_id
           << " into GPUDW at " << std::hex << this << std::dec
           << endl;
      }
      cerrLock.unlock();
    }

    //Have this var use the existng memory address.
    var.setData(varPointers->operator[](lpml).device_ptr);
    varLock->writeUnlock();
    return;
  }

  //It's not in the database, so create it.
  void* addr  = NULL;

  //This prepares the var with the offset and size.  The actual address will come next.
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );
  var.setData(addr);

  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " GPUDataWarehouse::allocateAndPut(), cudaMalloc"
          << " for PerPatch variable " << label
          << " patch " << patchID
          << " material " <<  matlIndx
          << " level " << levelIndx
          << " size " << var.getMemSize()
          << " at " << addr
          << " on device " << d_device_id << endl;
    }
    cerrLock.unlock();
  }

  varLock->writeUnlock();

  //put performs its own locking
  put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);

  /*
  varLock->writeLock();
  //first check if this patch/var/matl is in the process of loading in.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    //Space for this patch already exists.  Use that and return.
    if (d_debug){
      printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a variable for label %s patch %d matl %d level %d staging %s on device %d.  Reusing it.\n",
          label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id);
    }
   var.setData(varPointers->operator[](lpml).device_ptr);
   varLock->writeUnlock();
   return;
  }
  //It's not in the database, so create it.
  void* addr  = NULL;

  //This prepares the var with the offset and size.  The actual address will come next.
  //var.setData(addr);

  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );
  var.setData(addr);

  if (d_debug) {
    printf("In allocateAndPut(), cudaMalloc for \"%s\" patch %d size %ld\n", label, patchID, var.getMemSize());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  varLock->writeUnlock();
  put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);
  */

#endif
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
    if (strmatch == 0
        && d_varDB[i].domainID == patchID
        && d_varDB[i].matlIndx == matlIndx
        && d_varDB[i].levelIndx == levelIndx
        && d_varDB[i].varItem.staging == false             /* we don't support staging/foregin vars for get() */
        && d_varDB[i].ghostItem.dest_varDB_index == -1) {  /*don't let ghost cell copy data mix in with normal variables for get() */
      index = i; //we found it.
      //printf("I'm thread %d In DW at %p, We found it for var %s patch %d matl %d level %d.  d_varDB has it at index %d var %s patch %d at its item address %p with var pointer %p\n",
      //              threadID, this, label, patchID, matlIndx, levelIndx, index, &(d_varDB[index].label[0]), d_varDB[index].domainID, &d_varDB[index], d_varDB[index].var_ptr);

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

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset, bool skipContiguous, bool onlyContiguous )
{
#ifdef __CUDA_ARCH__
  printf("exist() is not yet implemented for the device.\n");
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
    CUDA_RT_SAFE_CALL(cudaMalloc(&temp, objectSizeInBytes));
    d_device_copy = (GPUDataWarehouse*)temp;
    //cudaHostRegister(this, sizeof(GPUDataWarehouse), cudaHostRegisterPortable);

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::init_device() -"
            << " allocated " << objectSizeInBytes
            << " bytes at " <<  d_device_copy
            << " on device " << d_device_id
            << endl;
      }
      cerrLock.unlock();
    }

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
    cudaStream_t* ptr = (cudaStream_t*)(cuda_stream);

    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " GPUDataWarehouse::syncto_device() -"
            << " sync GPUDW at " << d_device_copy
            << " with description " << _internalName
            << " to device " << d_device_id
            << " on stream " << ptr
            << endl;
      }
      cerrLock.unlock();
    }

    CUDA_RT_SAFE_CALL (cudaMemcpyAsync( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice, *ptr));
    //CUDA_RT_SAFE_CALL (cudaMemcpy( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice));


    //if (d_debug) {
    //  printf("sync GPUDW %p to device %d on stream %p\n", d_device_copy, d_device_id, ptr);
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
                << " GPUDataWarehouse::uintahSetCudaDevice() -"
                << " cudaFree for staging var for " << varIter->first.label
                << " at device ptr " <<  stagingIter->second.device_ptr
                << " on device " << d_device_id
                << endl;
          }
          cerrLock.unlock();
        }


        CUDA_RT_SAFE_CALL(cudaFree(stagingIter->second.device_ptr));
        //stagingIter->second.device_ptr == NULL;
      }
      varIter->second.stagingVars.clear();

      //clear out the regular vars
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " GPUDataWarehouse::uintahSetCudaDevice() -"
              << " cudaFree for non-staging var for " << varIter->first.label
              << " at device ptr " <<  varIter->second.device_ptr
              << " on device " << d_device_id
              << endl;
        }
        cerrLock.unlock();
      }

      CUDA_RT_SAFE_CALL(cudaFree(varIter->second.device_ptr));
      //varIter->second.device_ptr == NULL;
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
            << " GPUDataWarehouse::uintahSetCudaDevice() -"
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
        gpu_stats << UnifiedScheduler::myRankThread() << " Delete GPUDW on-device copy at " << std::hex
           << d_device_copy << " on device " << std::dec << d_device_id << std::endl;
      }
      cerrLock.unlock();
    }

    //cudaHostUnregister(this);
    CUDA_RT_SAFE_CALL(cudaFree( d_device_copy ));

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
                               bool sourceStaging, bool deststaging,
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
    //sv.device_offset = sharedLowCoordinates;
    //sv.device_size = make_int3(sharedHighCoordinates.x-sharedLowCoordinates.x,
    //                           sharedHighCoordinates.y-sharedLowCoordinates.y,
    //                           sharedHighCoordinates.z-sharedLowCoordinates.z);
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

  if (index == -1) {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label %s, source patch ID %d, matlIndx %d, levelIndex %d staging %s not found in GPU DW %p\n",
        label, sourcePatchID, matlIndx, levelIndx, "false", this);
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
         << " from patch " << sourcePatchID << " to patch " << destPatchID
         << " has shared coordinates (" << sharedLowCoordinates.x << ", " << sharedLowCoordinates.y << ", " << sharedLowCoordinates.z << "),"
         << " (" << sharedHighCoordinates.x << ", " << sharedHighCoordinates.y << ", " << sharedHighCoordinates.z << "), "
         << " from low/offset (" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << ") "
         << " size (" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << ") "
         << " virtualOffset (" << d_varDB[i].ghostItem.virtualOffset.x << ", " << d_varDB[i].ghostItem.virtualOffset.y << ", " << d_varDB[i].ghostItem.virtualOffset.z << ") "
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
    if (deststaging) {
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
        label, destPatchID, matlIndx, levelIndx, deststaging ? "true" : "false");
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
    if ((oldVarStatus & ALLOCATING == ALLOCATING) || (oldVarStatus & ALLOCATED == ALLOCATED)) {
      //Something else already allocated or is allocating it.  So this thread won't do do any allocation.
      return false;
    } else {
      //attempt to claim we'll allocate it
      atomicDataStatus newVarStatus = oldVarStatus | ALLOCATING;
      allocating = __sync_val_compare_and_swap(&status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//Sets the allocate flag on a variables atomicDataStatus
__host__ bool
GPUDataWarehouse::testAndSetAllocate(atomicDataStatus& status)
{

  bool allocated = false;

  //get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch(&(status), 0);
  //if it's allocated, return true
  if (oldVarStatus & ALLOCATED == ALLOCATED) {
    //already allocated, use it.
    printf("ERROR:\nGPUDataWarehouse::testAndSetAllocate(  )  Can't allocate a status it if it's already allocated\n");
    exit(-1);
  } else {
    //attempt to claim we'll allocate it
    atomicDataStatus newVarStatus = oldVarStatus | ALLOCATED;
    //turn off any allocating flag if it existed
    newVarStatus = newVarStatus & ~ALLOCATING;
    allocated = __sync_val_compare_and_swap(&status, oldVarStatus, newVarStatus);
  }
  if (!allocated) {
    printf("ERROR:\nGPUDataWarehouse::testAndSetAllocate( )  Something wrongly modified the atomic status while setting the allocated flag\n");
    exit(-1);
  }
  return allocated;
}

//Simply determines if a variable has been marked as allocated.
__host__ bool
GPUDataWarehouse::checkAllocated(atomicDataStatus& status)
{

  return (__sync_or_and_fetch(&(status), 0) & ALLOCATED == ALLOCATED);
}


__host__ bool
GPUDataWarehouse::getValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = (__sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), 0) & VALID == VALID);
      //varPointers->operator[](lpml).validOnGPU;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //bool retVal = d_varDB[i].varItem.validOnGPU;
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
    __sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInGpuMemory), VALID);

    //varPointers->operator[](lpml).varInGpuMemory.currentStatus = VALID;

    //varPointers->operator[](lpml).validOnGPU = true;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //d_varDB[i].varItem.validOnGPU = true;
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnGPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
}

__host__ bool
GPUDataWarehouse::getValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {


    bool retVal = (__sync_or_and_fetch(&(varPointers->operator[](lpml).atomicStatusInHostMemory), 0) & VALID == VALID);
    //varPointers->operator[](lpml).validOnGPU;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //bool retVal = d_varDB[i].varItem.validOnCPU;
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
    __sync_fetch_and_or(&(varPointers->operator[](lpml).atomicStatusInHostMemory), VALID);
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //d_varDB[i].varItem.validOnCPU = true;
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnCPU unknown variable %s on GPUDataWarehouse\n", label);
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
      printf("   Available varDB labels(%i): \"%-15s\" matl: %i, patchID: %i, level: %i\n", d_numVarDBItems, d_varDB[i].label, d_varDB[i].matlIndx,
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
