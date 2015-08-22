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
    printf("I'm at GPUDW %p\n", this);
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, levelIndx, patchID, matlIndx);
  }
  varLock->readUnlock();
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
    varLock->writeUnlock();  // writeLock() is called from allocateAndPut(). This is the escape clause if things go bad
    exit(-1);
  }

  //Find this item.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  if (staging == true && it == varPointers->end()) {
    //We need to input a staging variable, but there isn't a regular non-staging variable at the same label/patch/matl/level.
    //So go ahead and make an empty placeholder regular non-staging variable entry.
    //non staging vars create a new varPointers entry
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
    vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    vp.validOnCPU = false; //We don't know here if the data is in host memory.
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
            << " into GPUDW at " << this
            << endl;
      }
      cerrLock.unlock();
    }
    //insert it and get a new iterator back.
    varPointers->insert(std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    it = varPointers->find(lpml);

/*
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
            << " ERROR: GPUDataWarehouse::put( " << label << " ) - "
            << "This gpudw database is trying to input a staging variable without a non-staging variable at the same label/patch/matl/level for label "
            << label
            << " patch " << patchID
            << " matl " << matlIndx
            << " level " << levelIndx
            << " on device " << d_device_id
            << " into GPUDW at " << this
            << endl;
      }
      cerrLock.unlock();
    }


    //printf("ERROR: GPUDataWarehouse::put( %s ). This gpudw database is trying to input a staging variable without a non-staging variable at the same label/patch/matl/level for label %s patch %d matl %d level %d on device %d at %p.\n",
    //     label, label, patchID, matlIndx, levelIndx, d_device_id, it->second.device_ptr);
    varLock->writeUnlock();
    exit(-1);
    */
  }


  int3 var_offset;        // offset
  int3 var_size;          // dimensions of GPUGridVariable
  void* var_ptr;           // raw pointer to the memory
  int d_varDB_index = -1;         // If it doesn't have a match in d_varDB, it should stay -1
  var.getArray3(var_offset, var_size, var_ptr);
  if (d_device_copy != NULL ) {
    //If this GPU DW (a Task GPU DW) will be copied to the GPU, then fill the d_varDB
    int i=d_numVarDBItems;
    d_varDB_index = i;
    d_numVarDBItems++;
    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx  = levelIndx;
    //d_varDB[i].staging = staging;
    d_varDB[i].sizeOfDataType = sizeOfDataType;
    d_varDB[i].varItem.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    d_varDB[i].varItem.validOnCPU = false; //We don't know here if the data is on the CPU.
    d_varDB[i].varItem.gtype = gtype;
    d_varDB[i].varItem.numGhostCells = numGhostCells;
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
            << " level " << levelIndx
            << " into address " << d_varDB[i].var_ptr
            << " into  d_varDB index " << i
            << " on device " << d_device_id
            << " into GPUDW at " << this
            << " size [" << d_varDB[i].var_size.x << ", " << d_varDB[i].var_size.y << ", " << d_varDB[i].var_size.z << "]"
            << " offset [" << d_varDB[i].var_offset.x << ", " << d_varDB[i].var_offset.y << ", " << d_varDB[i].var_offset.z << "]"
            << endl;
      }
      cerrLock.unlock();
    }
    //if (d_debug){
    //  printf("host put \"%s\" (patch: %d) material %d, location %p into GPUDW %p into d_varDB index %d on device %d, size [%d,%d,%d]\n", label, patchID, matlIndx, d_varDB[i].var_ptr, d_device_copy, i, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
    //}

  }

  //Now fill the host-side varPointers collection
  if (staging) {
    //staging variables go inside the existing varPointers entry

    stagingVar sv;
    sv.device_offset = var_offset;
    sv.device_size = var_size;
    std::map<stagingVar, stagingVarInfo>::iterator staging_it = it->second.stagingVars.find(sv);
    //Often foreign vars are grouped together so that instead of sending a two variables from
    //one host to another, it will group them together into one variable if they're contiguous
    //so on this host, we'll see a similar thing, the needed neighbor for two dependencies
    //being the same foreign variable.
    if (staging_it != it->second.stagingVars.end()) {
      if (gpu_stats.active()) {
         cerrLock.lock();
         {
           gpu_stats << UnifiedScheduler::myRankThread()
               << " GPUDataWarehouse::put( " << label << " ) - "
               << " This staging variable already exists.  No need to add another.  For label " << label
               << " patch " << patchID
               << " matl " << matlIndx
               << " level " << levelIndx
               << " with offset (" << var_offset.x << ", " << var_offset.y << ", " << var_offset.z << ")"
               << " and size (" << var_size.x << ", " << var_size.y << ", " << var_size.z << ")"
               << " on device " << d_device_id
               << " into GPUDW at " << this
               << endl;
         }
         cerrLock.unlock();
       }

      varLock->writeUnlock();

      return;

    }
    stagingVarInfo svi;
    svi.device_ptr = var_ptr;
    svi.host_contiguousArrayPtr = host_ptr;
    svi.varDB_index = d_varDB_index;

    //if (d_debug){
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
              << " on device " << d_device_id
              << " into GPUDW at " << this
              << endl;
        }
        cerrLock.unlock();
      }

      //printf("GPUDataWarehouse::put( %s ) Put a staging variable for label %s patch %d matl %d level %d with offset (%d, %d, %d) and size (%d, %d, %d) on device %d in this DW %p\n",
      //    label, label, patchID, matlIndx, levelIndx,
      //    var_offset.x, var_offset.y, var_offset.z,
      //    var_size.x, var_size.y, var_size.z, d_device_id, this);
    //}

    std::pair<stagingVar, stagingVarInfo> p = make_pair( sv, svi );
    //std::map<stagingVar, stagingVarInfo>::value_type( sv, svi )

    it->second.stagingVars.insert( p  );

  } else {
    //non staging vars create a new varPointers entry
    allVarPointersInfo vp;

    vp.varDB_index = d_varDB_index;
    vp.device_ptr = var_ptr;
    vp.device_offset =  var_offset;
    vp.device_size = var_size;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = gtype;
    vp.numGhostCells = numGhostCells;
    vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    vp.validOnCPU = false; //We don't know here if the data is in host memory.
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
            << " on device " << d_device_id
            << " into GPUDW at " << this
            << endl;
      }
      cerrLock.unlock();
    }
    //if (d_debug){
    //  printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d level %d on device %d in this GPUDW at %p\n",
    //      label, label, patchID, matlIndx, levelIndx, d_device_id, this);
    //}
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
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


  varLock->writeLock();

  //first check if we've already got this exact version of this variable already.
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml);

  /*if (it == varPointers->end() && staging == true) {
    printf("ERROR: GPUDataWarehouse::allocateAndPut( %s ). This gpudw database is trying to input a staging variable without a non-staging variable at the same label/patch/matl/level for label %s patch %d matl %d level %d on device %d at %p.\n",
         label, label, patchID, matlIndx, levelIndx, d_device_id, it->second.device_ptr);
    varLock->writeUnlock();
    exit(-1);
  } else */
  if (it != varPointers->end() && it->second.device_offset.x == low.x
      && it->second.device_offset.y == low.y
      && it->second.device_offset.z == low.z
      && it->second.device_size.x == (high.x-low.x)
      && it->second.device_size.y == (high.y-low.y)
      && it->second.device_size.z == (high.z-low.z)
      && it->second.device_ptr != NULL ) {
    //Space for this var already exists.  Use that and return.
    if (d_debug){
      printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a variable for label %s patch %d matl %d level %d staging %s on device %d.  Reusing it.  It's at offset (%d, %d, %d) size (%d, %d, %d)\n",
          label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id,
          it->second.device_offset.x, it->second.device_offset.y, it->second.device_offset.z,
          it->second.device_size.x, it->second.device_size.y, it->second.device_size.z);
    }
    var.setArray3(varPointers->operator[](lpml).device_offset, varPointers->operator[](lpml).device_size, varPointers->operator[](lpml).device_ptr);
    varLock->writeUnlock();

    return;
  }  //else if it's a staging variable, then the sizes should be different from
     //what we have, so we'll allocate space and put it in this database.

  //It's not in the database, so create it.
  int3 size   = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;
  void* addr  = NULL;

  //This prepares the var with the offset and size.  The actual address will come next.
  var.setArray3(offset, size, addr);
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );
  
  if (gpu_stats.active()) {
       cerrLock.lock();
       {
         gpu_stats << UnifiedScheduler::myRankThread()
             << " GPUDataWarehouse::allocateAndPut( " << label << " ) - "
             << " cudaMalloc for " << label
             << " patch " << patchID
             << " matl " << matlIndx
             << " level " << levelIndx
             << " staging " << staging
             << " memsize " << var.getMemSize()
             << " from low (" << low.x << ", " << low.y << ", " << low.z << ")"
             << " and high (" << high.x << ", " << high.y << ", " << high.z << ")"
             << endl;
       }
       cerrLock.unlock();
     }

  if (d_debug) {
    printf("In allocateAndPut(), cudaMalloc for %s patch %d material %d level %d staging %s, size %ld from (%d,%d,%d) to (%d,%d,%d) on thread %d ",
            label, patchID, matlIndx, levelIndx, staging ? "true" : "false", var.getMemSize(),
            low.x, low.y, low.z, high.x, high.y, high.z, SCIRun::Thread::self()->myid());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  var.setArray3(offset, size, addr);
  varLock->writeUnlock();

  //put performs its own locking
  put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx, staging, gtype, numGhostCells);


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
      d_varDB[i].varItem.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
      d_varDB[i].varItem.validOnCPU = false; //We don't know here if the data is on the CPU.
      d_varDB[i].varItem.gtype = None;  //PerPatch has no ghost cells
      d_varDB[i].varItem.numGhostCells = 0;
      d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
      var.getData(d_varDB[i].var_ptr);

      vp.device_ptr = d_varDB[i].var_ptr;
      vp.sizeOfDataType = sizeOfDataType;
      vp.gtype = None;
      vp.numGhostCells = 0;
      vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
      vp.validOnCPU = false; //We don't know here if the data is on th
      vp.varDB_index = i;
      if (d_debug){
        printf("host put perPatch \"%s\" (patch: %d) material %d, level %d, staging %s, location %p into GPUDW %p into d_varDB index %d on device %d, size [%d,%d,%d]\n",
            label, patchID, matlIndx, levelIndx, staging ? "true" : "false",
               d_varDB[i].var_ptr, d_device_copy, i, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
      }
      d_dirty=true;
    } else {
      var.getData(vp.device_ptr);
      vp.varDB_index = -1;
      vp.sizeOfDataType = sizeOfDataType;
      vp.gtype = None;
      vp.numGhostCells = 0;
      vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
      vp.validOnCPU = false; //We don't know here if the data is on th
    }

  if (varPointers->find(lpml) == varPointers->end()) {
    if (d_debug){
      printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d level %d staging %s on device %d in this DW %p\n",
              label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id, this);
    }
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    varLock->writeUnlock();
  } else {
    printf("ERROR:\nGPUDataWarehouse::put( %s )  This gpudw database already has a variable for label %s patch %d matl %d level %d staging %s on device %d in GPUDW at %p\n",
            label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id, d_device_copy);
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

  if (d_device_copy != NULL ) {
    int i=d_numVarDBItems;
    d_numVarDBItems++;
    strncpy(d_varDB[i].label, label, MAX_NAME_LENGTH);
    d_varDB[i].domainID = patchID;
    d_varDB[i].matlIndx = matlIndx;
    d_varDB[i].levelIndx = levelIndx;
    //d_varDB[i].staging = false;
    d_varDB[i].sizeOfDataType = sizeOfDataType;
    d_varDB[i].varItem.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    d_varDB[i].varItem.validOnCPU = false; //We don't know here if the data is on the CPU.
    d_varDB[i].varItem.gtype = None;  //PerPatch has no ghost cells
    d_varDB[i].varItem.numGhostCells = 0;
    d_varDB[i].ghostItem.dest_varDB_index = -1; //Signify that this d_varDB item is NOT meta data to copy a ghost cell.
    var.getData(d_varDB[i].var_ptr);

    vp.device_ptr = d_varDB[i].var_ptr;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    vp.validOnCPU = false; //We don't know here if the data is on the CPU
    vp.varDB_index = i;
    if (d_debug){
      printf("host put perPatch \"%s\" (patch: %d) material %d, level %d, location %p into GPUDW %p into d_varDB index %d on device %d, size [%d,%d,%d]\n",
             label, patchID, matlIndx, levelIndx,
             d_varDB[i].var_ptr, d_device_copy, i, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
    }
    d_dirty=true;
  } else {
    var.getData(vp.device_ptr);
    vp.varDB_index = -1;
    vp.sizeOfDataType = sizeOfDataType;
    vp.gtype = None;
    vp.numGhostCells = 0;
    vp.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
    vp.validOnCPU = false; //We don't know here if the data is on the CPU
  }


  if (varPointers->find(lpml) == varPointers->end()) {
    if (d_debug){
      printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d level %d on device %d\n",
          label, label, patchID, matlIndx, levelIndx, d_device_id);
    }
    varPointers->insert( std::map<labelPatchMatlLevel, allVarPointersInfo>::value_type( lpml, vp ) );
    varLock->writeUnlock();
  } else {
    printf("ERROR:\nGPUDataWarehouse::put( %s )  This gpudw database already has a variable for label %s patch %d matl %d level %d on device %d in GPUDW at %p\n",
        label, label, patchID, matlIndx, levelIndx, d_device_id, d_device_copy);
    varLock->writeUnlock();
    exit(-1);
  }

  //if (strcmp(label, "sp_vol_CC") == 0) {
  //  printf("host put %s (patch: %d) loc %p into GPUDW %p on device %d, size [%d,%d,%d] with data %1.15e\n", label, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z, *((double *)vp.host_contiguousArrayPtr));
  //}

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, size_t sizeOfDataType)
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
      printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a variable for label %s patch %d matl %d level %d staging %s on device %d.  Reusing it.\n",
          label, label, patchID, matlIndx, levelIndx, staging ? "true" : "false", d_device_id);
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
    printf("In allocateAndPut(), cudaMalloc for %s patch %d material %d level %d staging %s size %ld on thread %d ",
            label, patchID, matlIndx, levelIndx, staging ? "true" : "false", var.getMemSize(), SCIRun::Thread::self()->myid());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  varLock->writeUnlock();

  put(var, sizeOfDataType, label, patchID, matlIndx, levelIndx);


#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, size_t sizeOfDataType)
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

  __shared__ int index;

  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  //int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block

  int i = threadID;
  index = -1;
  __syncthreads();  //sync before get


  //if (d_debug && threadID == 0 && blockID == 0) {
  //  printf("device getting item \"%s\" from GPUDW %p", label, this);
  //  printf("size (%d vars)\n Available labels:", d_numVarDBItems);
  //}

  //Have every thread try to find the label/patchId/matlIndx is a match in
  //array.  This is a clever approach so that instead of doing a simple
  //sequential search with one thread, we can let every thread search for it.  Only the
  //winning thread gets to write to shared data.

  while(i<d_numVarDBItems){
    int strmatch=0;
    char const *s1 = label; //reset s1 and s2 back to the start
    char const *s2 = &(d_varDB[i].label[0]);

    //a one-line strcmp.  This should keep branching down to a minimum.
    while (!(strmatch = *(unsigned char *) s1 - *(unsigned char *) s2) && *s1++ && *s2++);

    //only one thread will ever match this.
    //And nobody on the device side should ever access "staging" variables.
    if (strmatch == 0 && d_varDB[i].domainID == patchID && d_varDB[i].matlIndx == matlIndx && d_varDB[i].levelIndx == levelIndx) {
      index = i; //we found it.
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
GPUDataWarehouse::init(int id)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::init() should not be called on the device.\n");
#else

  d_device_id = id;
  objectSizeInBytes = 0;
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
HOST_DEVICE void
GPUDataWarehouse::init_device(size_t objectSizeInBytes)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::init_device() should only be called by the framework\n");
#else

    this->objectSizeInBytes = objectSizeInBytes;
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

    CUDA_RT_SAFE_CALL (cudaMemcpyAsync( d_device_copy, this, objectSizeInBytes, cudaMemcpyHostToDevice, *ptr));


    if (d_debug) {
      printf("sync GPUDW %p to device %d on stream %p\n", d_device_copy, d_device_id, cuda_stream);
    }
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

  //the old way
  //for (int i=0; i<d_numVarDBItems; i++) {
  //  if (d_varDB[i].label[0] != 0){
  //    CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));
  //    
  //    if (d_debug){
  //      printf("cudaFree for \"%s\" at %p on device %d\n", d_varDB[i].label, d_varDB[i].var_ptr, d_device_id );
  //    }
  //  }


  //delete any grid var that isn't part of a contiguous array
  varLock->writeLock();
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator varIter;
  for (varIter = varPointers->begin(); varIter != varPointers->end(); ++varIter) {
    if (varIter->second.host_contiguousArrayPtr == NULL) {
      //clear out all the staging vars, if any
      std::map<stagingVar, stagingVarInfo>::iterator stagingIter;
      for (stagingIter = varIter->second.stagingVars.begin(); stagingIter != varIter->second.stagingVars.end(); ++stagingIter) {
        if (d_debug){
          printf("cuda Free for staging var for %s at device ptr %p on device %d\n" , varIter->first.label.c_str(), stagingIter->second.device_ptr, d_device_id );
        }
        CUDA_RT_SAFE_CALL(cudaFree(stagingIter->second.device_ptr));
        //stagingIter->second.device_ptr == NULL;
      }
      varIter->second.stagingVars.clear();

      //clear out the regular vars
      if (d_debug){
        printf("cuda Free for %s at device ptr %p on device %d\n" , varIter->first.label.c_str(), varIter->second.device_ptr, d_device_id );
      }
      CUDA_RT_SAFE_CALL(cudaFree(varIter->second.device_ptr));
      //varIter->second.device_ptr == NULL;
    }
  }

  varPointers->clear();
  if (d_debug){
    printf("Freeing zero or more tempGhostCells on device %d, cpu thread %d\n",d_device_id, SCIRun::Thread::self()->myid());
  }

  /*
  std::vector<tempGhostCellInfo>::iterator tempGhostCellsIter;
  for (tempGhostCellsIter = tempGhostCells.begin(); tempGhostCellsIter != tempGhostCells.end(); ++tempGhostCellsIter) {
      CUDA_RT_SAFE_CALL(cudaFree(tempGhostCellsIter->device_ptr));
      tempGhostCellsIter->device_ptr == NULL;
  }

  tempGhostCells.clear();
*/

  //delete all the contiguous arrays
  std::map<std::string, contiguousArrayInfo>::iterator iter;
  for (iter = contiguousArrays->begin(); iter != contiguousArrays->end(); ++iter) {
    if (d_debug){
      printf("cuda Free for %s at device ptr %p on device %d\n" , iter->first.c_str(), iter->second.allocatedDeviceMemory, d_device_id );
      printf("delete[] for %s at host ptr %p on device %d\n" , iter->first.c_str(), iter->second.allocatedHostMemory, d_device_id );
    }
    CUDA_RT_SAFE_CALL(cudaFree(iter->second.allocatedDeviceMemory));
    //cudaHostUnregister(iter->second.allocatedHostMemory);
    free(iter->second.allocatedHostMemory);

  }
  contiguousArrays->clear();

  varLock->writeUnlock();

  init(d_device_id);

#endif
}


HOST_DEVICE void
GPUDataWarehouse::deleteSelfOnDevice()
{
#ifdef __CUDA_ARCH__
  //no meaning in device method
#else
  if ( d_device_copy ) {
    if(d_debug){
      printf("Delete GPUDW on-device copy at %p on device %d \n",  d_device_copy, d_device_id);
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
/*
__global__ void prepareGpuGhostCellIntoGpuArrayKernel(GPUDataWarehouse *gpudw, void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh) {
  gpudw->copyGhostCellsToArray(d_ghostCellData, index, ghostCellLow, ghostCellHigh);
}

HOST_DEVICE void
GPUDataWarehouse::prepareGpuGhostCellIntoGpuArray(void* cpuDetailedTaskOwner,
                                                    int3 ghostCellLow, int3 ghostCellHigh,
                                                    int sizeOfDataType,
                                                    char const* label, int matlIndx, int levelIndx, bool staging,
                                                    int fromPatchID, int toPatchID,
                                                    int fromDeviceIndex, int toDeviceIndex,
                                                    int fromresource, int toresource )
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  labelPatchMatlLevel lpml(label, fromPatchID, matlIndx, levelIndx);
  allVarPointersInfo vp;
  varLock->writeLock();
  if (varPointers->find(lpml) != varPointers->end()) {
    vp = varPointers[lpml];
  } else {
    varLock->writeUnlock();

    printf("ERROR:\nGPUDataWarehouse::prepareGhostCellForCopyingInvoker, label: %s source patch ID %d, materialID %d level %d staging %s not found in variable database",
        label, fromPatchID, matlIndx, levelIndx, staging ? "true" : "false");
    exit(-1);
  }
  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * sizeOfDataType;

  tempGhostCellInfo tgci;
  tgci.label = label;
  tgci.patchID = fromPatchID;
  tgci.matlIndx = matlIndx;
  tgci.levelIndx = levelIndx;
  tgci.staging = staging;
  tgci.cpuDetailedTaskOwner = cpuDetailedTaskOwner;
  //tgci->copied = false;
  tgci.memSize = ghostCellSize;
  tgci.ghostCellLow = ghostCellLow;
  tgci.ghostCellHigh = ghostCellHigh;
  tgci.sizeOfDataType = sizeOfDataType;
  tgci.toPatchID = toPatchID;
  tgci.fromDeviceIndex = fromDeviceIndex;
  tgci.toDeviceIndex = toDeviceIndex;

  if (vp.device_offset.x == ghostCellLow.x &&
      vp.device_offset.y == ghostCellLow.y &&
      vp.device_offset.z == ghostCellLow.z &&
      vp.device_offset.x+vp.device_size.x == ghostCellHigh.x &&
      vp.device_offset.y+vp.device_size.y == ghostCellHigh.y &&
      vp.device_offset.z+vp.device_size.z == ghostCellHigh.z) {
    //The entire source is the ghost cell, so use that.
    tgci.device_ptr = vp.device_ptr;
  } else {
    void *d_ghostCellData = NULL;

    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
    CUDA_RT_SAFE_CALL( cudaMalloc(&d_ghostCellData, ghostCellSize) );
    //call a kernel which gets the copy process started.
    const int BLOCKSIZE = 24;
    int xblocks = 32;
    int yblocks = 1;
    int zblocks = 1;
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);  //hopefully 24 *24 * 32 threads is enough for us.
    dim3 dimGrid(xblocks, yblocks, zblocks);

    //TODO, stream this?
    //prepareGpuGhostCellIntoGpuArrayKernel<<< dimGrid, dimBlock>>>(this->d_device_copy, d_ghostCellData, index, ghostCellLow, ghostCellHigh);

    //This copied ghost cell is only temporary.  Don't store it in the varDB, it should be moved and deleted very quickly.
    //Another reason not to store it in this object is the varPointers collection is a map, not a multimap.  We
    //do not want two entries for the same label/patch/matl tuple.
    tgci.device_ptr = d_ghostCellData;

  }

  if (d_debug) {
    printf("GPUDataWarehouse::prepareGhostCellForCopyingInvoker, Creating a tempGhostCellInfo from %d to %d starting at (%d, %d, %d) from patch %d to patch %d with address %p and size %d on threadID %d\n", fromDeviceIndex, toDeviceIndex, ghostCellLow.x, ghostCellLow.y, ghostCellLow.z, fromPatchID, toPatchID, tgci.device_ptr,  ghostCellSize, SCIRun::Thread::self()->myid());
  }
  tempGhostCells.push_back( tgci );
  varLock->writeUnlock();
#endif
}
*/
/*

HOST_DEVICE void
GPUDataWarehouse::prepareGpuToGpuGhostCellDestination(void* cpuDetailedTaskOwner,
                                                    int3 ghostCellLow, int3 ghostCellHigh,
                                                    int sizeOfDataType,
                                                    char const* label, int matlIndx, int levelIndx,
                                                    int fromPatchID, int toPatchID,
                                                    int fromDeviceIndex, int toDeviceIndex,
                                                    void * &data_ptr)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else



  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * sizeOfDataType;
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&data_ptr, ghostCellSize) );
  //call a kernel which gets the copy process started.

  //This ghost cell is only temporary.  Don't store it in the varDB, it should be moved and deleted very quickly.
  //Another reason not to store it in this object is the varPointers collection is a map, not a multimap.  We
  //do not want two entries for the same lebel/patch/matl tuple.
  varLock->writeLock();
  tempGhostCellInfo tgci;
  tgci.label = label;
  tgci.patchID = fromPatchID;
  tgci.matlIndx = matlIndx;
  tgci.levelIndx = levelIndx;
  tgci.cpuDetailedTaskOwner = cpuDetailedTaskOwner;
  //tgci->copied = false;
  tgci.device_ptr = data_ptr;
  tgci.memSize = ghostCellSize;
  tgci.ghostCellLow = ghostCellLow;
  tgci.ghostCellHigh = ghostCellHigh;
  tgci.sizeOfDataType = sizeOfDataType;
  tgci.toPatchID = toPatchID;
  tgci.fromDeviceIndex = fromDeviceIndex;
  tgci.toDeviceIndex = toDeviceIndex;
  if (d_debug){
    printf("GPUDataWarehouse::prepareGpuToGpuGhostCellDestination Creating a tempGhostCellInfo from %d to %d starting at (%d, %d, %d) with address %p and size %d on threadID %d\n", fromDeviceIndex, toDeviceIndex, ghostCellLow.x, ghostCellLow.y, ghostCellLow.z, data_ptr, ghostCellSize, SCIRun::Thread::self()->myid());
  }
  tempGhostCells.push_back( tgci );
  varLock->writeUnlock();
#endif
}*/
/*
HOST_DEVICE void
GPUDataWarehouse::copyGhostCellsToArray(void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh) {
#ifndef __CUDA_ARCH__
  //Not for the host side
#else
  int numThreads = blockDim.x*blockDim.y*blockDim.z;
  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block
  int totalThreads = numThreads * gridDim.x * gridDim.y * gridDim.z;
  int assignedCellID;

  int3 ghostCellSize;
  ghostCellSize.x = ghostCellHigh.x - ghostCellLow.x;
  ghostCellSize.y = ghostCellHigh.y - ghostCellLow.y;
  ghostCellSize.z = ghostCellHigh.z - ghostCellLow.z;

  assignedCellID = blockID * numThreads + threadID;
  while (assignedCellID < ghostCellSize.x * ghostCellSize.y * ghostCellSize.z ) {
    int z = assignedCellID / (ghostCellSize.x * ghostCellSize.y);
    int temp = assignedCellID % (ghostCellSize.x * ghostCellSize.y);
    int y = temp / ghostCellSize.x;
    int x = temp % ghostCellSize.x;

    //if we're in a valid x,y,z space for the variable.  (It's unlikely every cell will perfectly map onto every available thread.)
    if (x < ghostCellSize.x && y < ghostCellSize.y && z < ghostCellSize.z) {

      //offset them to their true array coordinates, not relative simulation cell coordinates
      x = x + ghostCellLow.x - d_varDB[index].var_offset.x;
      y = y + ghostCellLow.y - d_varDB[index].var_offset.y;
      z = z + ghostCellLow.z - d_varDB[index].var_offset.z;

      int varOffset = (x + y * d_varDB[index].var_size.x + z * d_varDB[index].var_size.x * d_varDB[index].var_size.y);
      //if (x == 1 && y == 1 && z == 9) {
      //  printf("here: from (%d, %d, %d) address is %p and offset it becomes %p and the value is %1.6lf\n", x, y, z, d_varDB[index].var_ptr, (double*)(d_varDB[index].var_ptr) + varOffset, *((double*)(d_varDB[index].var_ptr) + varOffset));
      //}
      //copy all 8 bytes of a double in one shot
      if (d_varDB[index].sizeOfDataType == sizeof(double)) {
        *(((double*)d_ghostCellData) + assignedCellID) = *(((double*)d_varDB[index].var_ptr) + varOffset);
      }

      //or copy all 4 bytes of an int in one shot.
      else if (d_varDB[index].sizeOfDataType == sizeof(int)) {
        *(((int*)d_ghostCellData) + assignedCellID) = *(((int*)d_varDB[index].var_ptr) + varOffset);
      }
      //Copy each byte until we've copied all for this data type.
      else {
        //varOffset = varOffset * d_varDB[index].sizeOfDataType;
        for (int j = 0; j < d_varDB[index].sizeOfDataType; j++) {
          *(((char*)d_ghostCellData) + (assignedCellID *  d_varDB[index].sizeOfDataType + j))
              = *(((char*)d_varDB[index].var_ptr) + (varOffset * d_varDB[index].sizeOfDataType + j));

        }
      }
    }
    assignedCellID += totalThreads;
  }
#endif
}
*/

//__global__ void copyGhostCellsToHostVarKernel(GPUDataWarehouse *gpudw, void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh) {
//  gpudw->copyGhostCellsToArray(d_ghostCellData, index, ghostCellLow, ghostCellHigh);
//}

/*
HOST_DEVICE void
GPUDataWarehouse::copyGhostCellsToHostVarInvoker(void* hostVarPointer, int3 ghostCellLow, int3 ghostCellHigh, char const* label, int patchID, int matlIndx)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  int index = 0;
  labelPatchMatlLevel lpm_source(label, patchID, matlIndx);
  varLock->readLock();
  if (varPointers->find(lpm_source) != varPointers->end()) {
    index = varPointers[lpm_source].varDB_index;

  } else {
    printf("ERROR:\nGPUDataWarehouse::copyGhostCellsToHostVar, label: %s source patch ID %d, materialID %d not found in variable database", label, patchID, matlIndx);
    exit(-1);
  }
  varLock->readUnlock();

  void *d_ghostCellData = NULL;

  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * d_varDB[index].sizeOfDataType;

  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&d_ghostCellData, ghostCellSize) );
  //call a kernel which gets the copy process started.
  const int BLOCKSIZE = 24;
  int xblocks = 32;
  int yblocks = 1;
  int zblocks = 1;
  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);  //hopefully 24 *24 * 32 threads is enough for us.
  dim3 dimGrid(xblocks, yblocks, zblocks);

  //TODO, stream this?
  copyGhostCellsToHostVarKernel<<< dimGrid, dimBlock>>>(this->d_device_copy, d_ghostCellData, index, ghostCellLow, ghostCellHigh);
  //cudaDeviceSynchronize();
  CUDA_RT_SAFE_CALL( cudaMemcpy(hostVarPointer, d_ghostCellData, ghostCellSize, cudaMemcpyDeviceToHost) );
  //printf("***Immediately received ghost cell, value is: %1.6lf\n", (*((double*)(hostVarPointer+22*8))));
  CUDA_RT_SAFE_CALL( cudaFree(d_ghostCellData) );
  //Copy the data out to the host at varPointer

#endif
}*/

/*
HOST_DEVICE void
GPUDataWarehouse::copyTempGhostCellsToHostVar(void* hostVarPointer, int3 ghostCellLow, int3 ghostCellHigh, char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  labelPatchMatlLevel lpml_source(label, patchID, matlIndx, levelIndx);
  varLock->readLock();
  tempGhostCellInfo tgci;
  bool found = false;
  //Check to make sure we have *exactly* the ghost cell we want.  (Other
  //CPU threads running other tasks can add their own data to the
  //tempGhostCells collection.
  //TODO: Make sure we only copy out external dependencies.  tempGhostCells can have both
  //internal (GPU -> another GPU same node) and external (GPU -> another node) dependencies.
  for (vector<tempGhostCellInfo>::iterator it = tempGhostCells.begin();
         it != tempGhostCells.end();
         ++it)
  {
    if (((*it).ghostCellLow.x == ghostCellLow.x) &&
        ((*it).ghostCellLow.y == ghostCellLow.y) &&
        ((*it).ghostCellLow.z == ghostCellLow.z) &&
        ((*it).ghostCellHigh.x == ghostCellHigh.x) &&
        ((*it).ghostCellHigh.y == ghostCellHigh.y) &&
        ((*it).ghostCellHigh.z == ghostCellHigh.z)) {
      tgci = (*it);
      found = true;
    }
  }

  if (found == false) {
    printf("ERROR:\nGPUDataWarehouse::copyTempGhostCellsToHostVar, label: %s source patch ID %d, materialID %d not found in temporary ghost cell collection", label, patchID, matlIndx);
    exit(-1);
  }
  varLock->readUnlock();

  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  //call a kernel which gets the copy process started.
  //printf("Preparing to copy into %p from device %p with memsize %d on threadID %d\n", hostVarPointer, tgci.device_ptr, tgci.memSize, SCIRun::Thread::self()->myid());
  //TODO: Stream this?
  CUDA_RT_SAFE_CALL( cudaMemcpy(hostVarPointer, tgci.device_ptr, tgci.memSize, cudaMemcpyDeviceToHost) );
  //printf("***Immediately received ghost cell, value is: %1.6lf\n", (*((double*)(hostVarPointer+22*8))));
  //CUDA_RT_SAFE_CALL( cudaFree(tgci.device_ptr) );
  //Copy the data out to the host at varPointer

#endif
}
*/
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
              /*printf("Going to copy, between (%d, %d, %d) from offset %d to offset %d.  From starts at (%d, %d, %d) with size (%d, %d, %d) pointer %p.  To starts at (%d, %d, %d) with size (%d, %d, %d).\n",
                  d_varDB[i].ghostItem.sharedLowCoordinates.x,
                  d_varDB[i].ghostItem.sharedLowCoordinates.y,
                  d_varDB[i].ghostItem.sharedLowCoordinates.z,
                  sourceOffset,
                  destOffset,
                  d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
                  d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
                  d_varDB[i].var_ptr,
                  d_varDB[destIndex].var_offset.x, d_varDB[destIndex].var_offset.y, d_varDB[destIndex].var_offset.z,
                  d_varDB[destIndex].var_size.x, d_varDB[destIndex].var_size.y, d_varDB[destIndex].var_size.z);
           */ //}

           //copy all 8 bytes of a double in one shot
           if (d_varDB[i].sizeOfDataType == sizeof(double)) {
             *((double*)(d_varDB[destIndex].var_ptr) + destOffset) = *((double*)(d_varDB[i].var_ptr) + sourceOffset);
             //if (threadID == 0) {
             //  if (d_varDB[i].ghostItem.sharedLowCoordinates.x == 4
             //      && d_varDB[i].ghostItem.sharedLowCoordinates.y == -1
             //       && d_varDB[i].ghostItem.sharedLowCoordinates.z == -1
             //      && d_varDB[i].ghostItem.sharedHighCoordinates.x == 5
             //      && d_varDB[i].ghostItem.sharedHighCoordinates.y == 0
             //      && d_varDB[i].ghostItem.sharedHighCoordinates.z == 0 ) {
             //    printf(" Copying this data for %s %1.16lf for 4, -1, -1 to 5, 0, 0 to %p\n",  d_varDB[destIndex].label, *((double*)(d_varDB[i].var_ptr) + sourceOffset), (double*)(d_varDB[destIndex].var_ptr) + destOffset);
             //  }
             //}

             //if (threadID == 0) {
             //  printf("Thread %d - At (%d, %d, %d), real: (%d, %d, %d), copying within region between (%d, %d, %d) and (%d, %d, %d).  Source d_varDB index %d (%d, %d, %d) varSize (%d, %d, %d) virtualOffset(%d, %d, %d), varOffset(%d, %d, %d), sourceOffset %d actual pointer %p, value %e.   Dest d_varDB index %d ptr %p destOffset %d actual pointer %p.\n",
             //      threadID, x, y, z, x_source_real, y_source_real, z_source_real,
             //      d_varDB[i].ghostItem.sharedLowCoordinates.x, d_varDB[i].ghostItem.sharedLowCoordinates.y, d_varDB[i].ghostItem.sharedLowCoordinates.z,
             //      d_varDB[i].ghostItem.sharedHighCoordinates.x, d_varDB[i].ghostItem.sharedHighCoordinates.y, d_varDB[i].ghostItem.sharedHighCoordinates.z,
             //      i,
             //      x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[i].ghostItem.virtualOffset.x,
             //      y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[i].ghostItem.virtualOffset.y,
             //      z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[i].ghostItem.virtualOffset.z,
             //      d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
             //      d_varDB[i].ghostItem.virtualOffset.x, d_varDB[i].ghostItem.virtualOffset.y, d_varDB[i].ghostItem.virtualOffset.z,
             //      d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
             //      sourceOffset, (double*)(d_varDB[i].var_ptr) + sourceOffset, *((double*)(d_varDB[i].var_ptr) + sourceOffset),
             //      destIndex, d_varDB[destIndex].var_ptr, destOffset, (double*)(d_varDB[destIndex].var_ptr) + destOffset);
             //}
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

    copyGpuGhostCellsToGpuVarsKernel<<< dimGrid, dimBlock, 0, *stream >>>(this->d_device_copy);

    /*
    {
    cudaDeviceSynchronize();
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
  varLock->readLock();
  int i = d_numVarDBItems;
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
    stagingVar sv;
    sv.device_offset = varOffset;
    sv.device_size = varSize;

    std::map<stagingVar, stagingVarInfo>::iterator staging_it = varPointers->operator[](lpml_source).stagingVars.find(sv);
    if (staging_it != varPointers->operator[](lpml_source).stagingVars.end()) {
      index = staging_it->second.varDB_index;

    } else {
      printf("ERROR: GPUDataWarehouse::putGhostCell( %s ). No staging variable found label %s patch %d matl %d level %d offset (%d, %d, %d) size (%d, %d, %d).\n",
                    label, label, sourcePatchID, matlIndx, levelIndx,
                    sv.device_offset.x, sv.device_offset.y, sv.device_offset.z,
                    sv.device_size.x, sv.device_size.y, sv.device_size.z);
      varLock->writeUnlock();
      exit(-1);
    }
    //Find the d_varDB entry for this specific one.


  }

  if (index == -1) {
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label %s, source patch ID %d, matlIndx %d, levelIndex %d staging %s not found in GPU DW %p\n",
        label, sourcePatchID, matlIndx, levelIndx, "false", this);
    varLock->readUnlock();
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
  if (d_debug){
    printf("Placed into d_varDB at index %d from patch %d to patch %d has shared coordinates (%d, %d, %d), (%d, %d, %d), from low/offset (%d, %d, %d) size (%d, %d, %d)  virtualOffset(%d, %d, %d)\n",
        i, sourcePatchID, destPatchID, sharedLowCoordinates.x, sharedLowCoordinates.y,
        sharedLowCoordinates.z, sharedHighCoordinates.x, sharedHighCoordinates.y, sharedHighCoordinates.z,
        d_varDB[i].var_offset.x, d_varDB[i].var_offset.y, d_varDB[i].var_offset.z,
        d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z,
        d_varDB[i].ghostItem.virtualOffset.x, d_varDB[i].ghostItem.virtualOffset.y, d_varDB[i].ghostItem.virtualOffset.z);
  }


  //Find where we are sending the ghost cell data to
  labelPatchMatlLevel lpml_dest(label, destPatchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, allVarPointersInfo>::iterator it = varPointers->find(lpml_dest);
  if (it != varPointers->end()) {
    if (deststaging) {
      //find the staging var inside this var.
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
        varLock->readUnlock();
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
    varLock->readUnlock();
    exit(-1);
  }
  d_dirty=true;
  varLock->readUnlock();
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


HOST_DEVICE bool
GPUDataWarehouse::getValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::getValidOnGPU( %s )  Not implemented for GPU\n",label);
  return false;
#else
  //__________________________________
  //  cpu code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = varPointers->operator[](lpml).validOnGPU;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //bool retVal = d_varDB[i].varItem.validOnGPU;
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
#endif
}

HOST_DEVICE void
GPUDataWarehouse::setValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::setValidOnGPU( %s )  Not implemented for GPU\n",label);
#else
  //__________________________________
  //  cpu code
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    varPointers->operator[](lpml).validOnGPU = true;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //d_varDB[i].varItem.validOnGPU = true;
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnGPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
#endif
}

HOST_DEVICE bool
GPUDataWarehouse::getValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::getValidOnCPU( %s )  Not implemented for GPU\n",label);
  return false;
#else
  //__________________________________
  //  CPU code
  varLock->readLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    bool retVal = varPointers->operator[](lpml).validOnCPU;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //bool retVal = d_varDB[i].varItem.validOnCPU;
    varLock->readUnlock();
    return retVal;

  } else {
    varLock->readUnlock();
    return false;
  }
#endif
}
HOST_DEVICE void
GPUDataWarehouse::setValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::setValidOnCPU( %s )  Not implemented for GPU\n",label);
#else
  //__________________________________
  //  CPU code
  varLock->writeLock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (varPointers->find(lpml) != varPointers->end()) {
    varPointers->operator[](lpml).validOnCPU = true;
    //allVarPointersInfo info = varPointers[lpm];
    //int i = info.varDB_index;
    //d_varDB[i].varItem.validOnCPU = true;
    varLock->writeUnlock();
  } else {
    varLock->writeUnlock();
    printf("host setValidOnCPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
#endif
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
  printf("  \nERROR: %s( \"%s\", levelIndx: %i, patchID: %i, matl: %i)  unknown variable\n",
         msg, label, levelIndx, patchID, matlIndx);
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
