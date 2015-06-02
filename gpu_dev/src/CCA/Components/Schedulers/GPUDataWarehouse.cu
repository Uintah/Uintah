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

namespace Uintah {

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUGridVariableBase& var, ...)", label, patchID, matlIndex);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUReductionVariableBase& var, ...)", label, patchID, matlIndex);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::get(const GPUPerPatchBase& var, char const* label, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::get(GPUPerPatchBase& var, ...)", label, patchID, matlIndex);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex)
{
  //Both the CPU and GPU share this code
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item) {
    var.setArray3(item->var_offset, item->var_size, item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUGridVariableBase& var, ...)", label, patchID, matlIndex);
  }
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, -1 /* matlIndex */);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUReductionVariableBase& var, ...)", label, patchID, matlIndex);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex)
{
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item) {
    var.setData(item->var_ptr);
  }
  else {
    printGetError("GPUDataWarehouse::getModifiable(GPUPerPatchBase& var, ...)", label, patchID, matlIndex);
  }
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::put(GPUGridVariableBase &var, char const* label, int patchID, int matlIndex, size_t xstride, GhostType gtype, int numGhostCells, GridVariableBase* gridVar, void* host_ptr)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n",label);
#else
  //NOTE!! varLock's writeLock() needs to be turned on prior to calling this function.
  //__________________________________
  //cpu code 
  if (d_numItems==MAX_ITEM) {
    printf("ERROR:  out of GPUDataWarehouse space");
    varLock.writeUnlock();

    exit(-1);
  }

  //first check if this patch/var/matl is in the process of loading in.
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  //for (std::map<charlabelPatchMatl, allVarPointersInfo>::iterator it = varPointers.begin(); it != varPointers.end();  ++it) {
  //    printf("For label %s patch %d matl %d on device %d I have an entry\n", (*it).first.label.c_str(), d_device_id, (*it).first.patchID, (*it).first.matlIndex);
  //}

  int i=d_numItems;
  d_numItems++;
  strncpy(d_varDB[i].label, label, MAX_LABEL);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndex = matlIndex;
  d_varDB[i].xstride = xstride;
  d_varDB[i].varItem.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
  d_varDB[i].varItem.queueingOnGPU = true; //Assume that because we created space for this variable, data will soon be arriving.
  d_varDB[i].varItem.validOnCPU = false; //We don't know here if the data is on the CPU.
  d_varDB[i].varItem.gtype = gtype;
  d_varDB[i].varItem.numGhostCells = numGhostCells;
  var.getArray3(d_varDB[i].var_offset, d_varDB[i].var_size, d_varDB[i].var_ptr);

  //Now store them in a map for easier lookups on the host
  //Without this map, we would have to loop through hundreds of array
  //The above is a good idea when we start implementing modified instead of just required and compute
  //elements looking for the right key.  With a map, we can find
  //our matching data faster, and we can store far more information
  //that the host should know and the device doesn't need to know about.
  allVarPointersInfo vp;
  vp.gridVar = gridVar;
  vp.host_contiguousArrayPtr = host_ptr;
  vp.device_ptr = d_varDB[i].var_ptr;
  vp.device_offset = d_varDB[i].var_offset;
  vp.device_size = d_varDB[i].var_size;
  vp.varDB_index = i;


  if (varPointers.find(lpm) == varPointers.end()) {
    //printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d on device %d\n",label, label, patchID, matlIndex, d_device_id);
    varPointers.insert( std::map<charlabelPatchMatl, allVarPointersInfo>::value_type( lpm, vp ) );
  } else {
    printf("ERROR:\nGPUDataWarehouse::put( %s )  This gpudw database already has a variable for label %s patch %d matl %d on device %d in GPUDW at %p\n",label, label, patchID, matlIndex, d_device_id, d_device_copy);
    varLock.writeUnlock();
    exit(-1);
  }

  //if (strcmp(label, "sp_vol_CC") == 0) {
  //  printf("host put %s (patch: %d) loc %p into GPUDW %p on device %d, size [%d,%d,%d] with data %1.15e\n", label, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z, *((double *)vp.host_contiguousArrayPtr));
  //}
  if (d_debug){
    printf("host put \"%s\" (patch: %d) material %d, location %p into GPUDW %p into d_varDB index %d on device %d, size [%d,%d,%d]\n", label, patchID, matlIndex, d_varDB[i].var_ptr, d_device_copy, i, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
  }
  d_dirty=true;

#endif
}

//______________________________________________________________________
//
HOST_DEVICE void 
GPUDataWarehouse::allocateAndPut(GPUGridVariableBase &var, char const* label, int patchID, int matlIndex, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype, int numGhostCells)
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
  //threads A and B both copy patch 0 into this GPU memory.  This should be ok as ghost cell
  //data cannot be modified, so in case of a race condition, nothing will be modified
  //if a memory chunk is being both read and written to simultaneously.
  //So allow duplicates if the data is read only, otherwise don't allow duplicates.


  varLock.writeLock();

  //first check if this patch/var/matl is in the process of loading in.
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    int index = varPointers[lpm].varDB_index;
    if (d_varDB[index].varItem.queueingOnGPU == true) {
      //It's loading up, use that and return.
      if (d_debug){
        printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a variable for label %s patch %d matl %d on device %d.  Reusing it.\n", label, label, patchID, matlIndex, d_device_id);

      }
      var.setArray3(d_varDB[index].var_offset, d_varDB[index].var_size, d_varDB[index].var_ptr);
      varLock.writeUnlock();
      return;
    }
  }

  //It's not in the database, so create it.
  int3 size   = make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset = low;
  void* addr  = NULL;

  //This prepares the var with the offset and size.  The actual address will come next.
  var.setArray3(offset, size, addr);
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );
  
  if (d_debug) {
    printf("In allocateAndPut(), cudaMalloc for \"%s\ patch %d size %ld from (%d,%d,%d) to (%d,%d,%d) on thread %d ", label, patchID, var.getMemSize(),
            low.x, low.y, low.z, high.x, high.y, high.z, SCIRun::Thread::self()->myid());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  var.setArray3(offset, size, addr);
  put(var, label, patchID, matlIndex, sizeOfDataType, gtype, numGhostCells);

  varLock.writeUnlock();
#endif
}


HOST_DEVICE void
GPUDataWarehouse::putContiguous(GPUGridVariableBase &var, const char* indexID, char const* label, int patchID, int matlIndex, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost)
{
#ifdef __CUDA_ARCH__
  //Should not put from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else

    varLock.writeLock();

    //first check if this patch/var/matl is in the process of loading in.
    charlabelPatchMatl lpm(label, patchID, matlIndex);
    if (varPointers.find(lpm) != varPointers.end()) {
      int index = varPointers[lpm].varDB_index;
      if (d_varDB[index].varItem.queueingOnGPU == true) {
        //It's loading up, use that and return.
        if (d_debug){
          printf("GPUDataWarehouse::putContiguous( %s ). This gpudw database has a variable for label %s patch %d matl %d on device %d.  Reusing it.\n", label, label, patchID, matlIndex, d_device_id);

        }
        var.setArray3(d_varDB[index].var_offset, d_varDB[index].var_size, d_varDB[index].var_ptr);
        varLock.writeUnlock();
        return;
      }
    }

  int3 size=make_int3(high.x-low.x, high.y-low.y, high.z-low.z);
  int3 offset=low;
  void* device_ptr=NULL;
  var.setArray3(offset, size, device_ptr);
  allocateLock.readLock();
  contiguousArrayInfo *ca = &(contiguousArrays[indexID]);
  allocateLock.readUnlock();
  //if (strcmp(indexID, "ICE::computeEquilibrationPressureUnifiedGPU (Patches: 1) (Matls: 0 - 1)") == 0) {
  //    printf("In allocateAndPut(), id is %s\n", indexID);
  //}
  if ( (ca->allocatedDeviceMemory == NULL
       || ca->sizeOfAllocatedMemory - ca->assignedOffset < var.getMemSize())
      && stageOnHost) {
    printf("ERROR: No room left on device to be assigned address space\n");
    if (ca->allocatedDeviceMemory != NULL) {
      printf("There was %lu bytes allocated, %lu has been assigned, and %lu more bytes were attempted to be assigned for %s patch %d matl %d\n",
          ca->sizeOfAllocatedMemory,
          ca->assignedOffset,
          var.getMemSize(), label, patchID, matlIndex);
    }
    varLock.writeUnlock();
    exit(-1);
  } else {


    //There is already pre-allocated contiguous memory chunks with room available on
    //both the device and the host.  Just assign pointers for both the device and host contiguous arrays.


    //This prepares the var with the offset and size.  The actual address will come next.

    void* host_contiguousArrayPtr = NULL;

    int varMemSize = var.getMemSize();


    device_ptr = ca->allocatedDeviceMemory + ca->assignedOffset;
    var.setArray3(offset, size, device_ptr);
    host_contiguousArrayPtr = ca->allocatedHostMemory + ca->assignedOffset;

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
      //if (strcmp(label, "sp_vol_CC") == 0) {
      //  printf("To   - copying %s %d %d from host location %p to host contiguous array at %p (starting at %p) to device array at %p (starting at %p) and used %d bytes with data %1.15e\n", label, patchID, matlIndex, host_ptr, host_contiguousArrayPtr, ca->allocatedHostMemory, device_ptr, ca->allocatedDeviceMemory, ca->copiedOffset, *((double *)host_contiguousArrayPtr));
      //}

    } //else {
      //printf("Setting aside space %s %d %d from host location %p host contiguous array %p\n", label, patchID, matlIndex, host_ptr, host_contiguousArrayPtr);
    //}


    put(var, label, patchID, matlIndex, sizeOfDataType, None, 0, gridVar, host_contiguousArrayPtr);

    //printf("Allocating for %s at patch %d and matl %d size is %d host_ptr %p host_contiguousPtr %p device_ptr %p\n", label, patchID, matlIndex, varMemSize, host_ptr, host_contiguousArrayPtr, device_ptr);
  }
  varLock.writeUnlock();

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

  printf("Allocated GPU buffer of size %d \n", size);

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
  allocateLock.writeLock();
  contiguousArrays.insert( std::map<const char *, contiguousArrayInfo>::value_type( indexID, ca ) );
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays.begin(); it != contiguousArrays.end(); ++it)
  //  printf("%s\n", it->first.c_str());

  allocateLock.writeUnlock();
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
  allocateLock.readLock();
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays.begin(); it != contiguousArrays.end(); ++it)
  //  printf("*** displaying %s\n", it->first.c_str());
  contiguousArrayInfo *ca = &(contiguousArrays[indexID]);


  allocateLock.readUnlock();
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
  allocateLock.writeLock();
  //for (std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays.begin(); it != contiguousArrays.end(); ++it)
  //  printf("%s\n", it->first.c_str());
  std::string sIndex = indexID;
  std::map<std::string, contiguousArrayInfo>::iterator it = contiguousArrays.find(sIndex);
  if ( it != contiguousArrays.end()) {
    sIndex += " used";  //mark it as used so it doesn't match in any future find() queries for the next iteration step.
    contiguousArrays[sIndex] = it->second; //put it back into the map with a new key.
    contiguousArrayInfo ca = it->second;
    contiguousArrays.erase(it);

    allocateLock.writeUnlock();
    if (ca.sizeOfAllocatedMemory - ca.copiedOffset > 0) {
      //Previously we only copied into the device variables that were already initialized with data
      //But now we need to copy the computes data back to the host.
      //printf("Copying to host %p from device %p amount %d\n", ca->allocatedHostMemory + ca->copiedOffset, ca->allocatedDeviceMemory + ca->copiedOffset, ca->assignedOffset - ca->copiedOffset);

      cudaError_t retVal = cudaMemcpyAsync(ca.allocatedHostMemory + ca.copiedOffset,
                                           ca.allocatedDeviceMemory + ca.copiedOffset,
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
    allocateLock.writeUnlock();
  }

  cudaError_t retVal = cudaSuccess;
  return retVal;

#endif
}

HOST_DEVICE void
GPUDataWarehouse::copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndex) {
#ifdef __CUDA_ARCH__
  //Should not called from device side as all memory allocation should be done on CPU side through CUDAMalloc()
#else
  //see if this datawarehouse has anything for this patchGroupID.
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    allVarPointersInfo info = varPointers[lpm];
    varLock.readUnlock();
    int i = info.varDB_index;
    device_var.setArray3(d_varDB[i].var_offset, d_varDB[i].var_size, info.device_ptr);
   // size_t size = device_var.getMemSize();
    //printf("Copying %s %d %d from host address %p to host address %p\n", label, patchID, matlIndex, info.host_contiguousArrayPtr, info.gridVar->getBasePointer());


    //TODO: Instead of doing a memcpy, I bet the original host grid variable could just have its pointers updated
    //to work with what we were sent back.  This would take some considerable work though to get all the details right
    //TODO: This needs to be a memcpy async
    memcpy(host_var->getBasePointer(), info.host_contiguousArrayPtr, device_var.getMemSize());
    //Since we've moved it back into the host, lets mark it as being used.
    //It's possible in the future there could be a scenario where we want to bring it
    //back to the host but still retain it in the GPU.  One scenario is
    //sending data to an output .ups file but not modifying it on the host.
    remove(label, patchID, matlIndex);


    //We're going to pass the host variable back out so it can be removed

    //host_var = info.gridVar;
    //printf("Host_var's data pointer is %p\n", host_var->getBasePointer());


    //We may not need to remember grid var on the device.  If the host_var
    //is going to be removed after this function is called, info.gridVar's pointer
    //could be NULL.  At this time, I'm not sure if we should remove it out of varPointers.
    //varLock.writeLock();
    //varPointers.remove(lpm);
    //varLock.writeUnlock();
    //Set the gridVar location to NULL.  We're going to be
    //info.gridVar = NULL;


  } else {
    varLock.readUnlock();
    printf("ERROR: host copyHostContiguoustoHost unknown variable on GPUDataWarehouse");
    //for (std::map<charlabelPatchMatl, allVarPointersInfo>::iterator it=varPointers.begin(); it!=varPointers.end(); ++it)
    //  printf("%s %d %d => %d \n", it->first.label, it->first.patchID, it->first.matlIndex, it->second.varDB_index);
    exit(-1);
  }
#endif

}

/*
HOST_DEVICE void*
GPUDataWarehouse::getPointer(char const* label, int patchID, int matlIndex)

//GPU implementations can be faster if you work with direct pointers, each thread having its own pointer
//and doing pointer arithmetic on it.  This is obviously a more low level and "you're on your own" approach.
//Though for some problems with many grid variables, each thread will need many pointers, overwhelming
//the amount of registers available to store them.  So this approach can't work for these problems,
//and instead a GPU shared variable exists which hold the pointer, and then werecalculate the x,y,z
//offset each time a thread requests data from the array.
{
#ifdef __CUDA_ARCH__
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  if (item){
    return item->var_ptr;
  }else{
    int numThreads = blockDim.x*blockDim.y*blockDim.z;
    int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

    int i=threadID;
    while(i<d_numItems){
      printf( "   Available labels: \"%s\"\n", d_varDB[i].label );
      i=i+numThreads;
    }
    if( isThread0_Blk0() ) {
      printf("  ERROR: GPUDataWarehouse::getPointer( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", label, patchID, matlIndex);
      assert(0);
    }
    //printf("\t ERROR: GPUDataWarehouse::getPointer( \"%s\", patchID: %i, matl: %i )  unknown variable\n", label, patchID, matlIndex);
    return NULL;
  }
#else
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    int i = varPointers[lpm].varDB_index;
    varLock.readUnlock();
    return d_varDB[i].var_ptr;
  } else {
    printf("ERROR: host get unknown variable on GPUDataWarehouse");
    varLock.readUnlock();
    exit(-1);
  }

// cpu code
//int i= 0;
//while(i<numItems){
//  if (!strncmp(d_varDB[i].label, label, MAX_LABEL) &&  d_varDB[i].domainID==patchID && d_varDB[i].matlIndex==matlIndex) {
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
GPUDataWarehouse::put(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc\n", label);
#else

  //__________________________________
  //cpu code
  if (d_numItems == MAX_ITEM) {
    printf("out of GPUDataWarehouse space.  You can try increasing GPUDataWarehouse.h: #define MAX_ITEMS");
    exit(-1);
  }

  int i = d_numItems;
  d_numItems++;
  strncpy(d_varDB[i].label, label, MAX_LABEL);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndex = -1;  // matlIndex;

  var.getData(d_varDB[i].var_ptr);

  if (d_debug) {
    printf("host put \"%s\" (patch: %d) loc %p into GPUDW %p on device %d\n", label, patchID, d_varDB[i].var_ptr, d_device_copy,
           d_device_id);
  }
  d_dirty = true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::put(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex, size_t xstride, GPUPerPatchBase* originalVar, void* host_ptr)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::put( %s )  You cannot use this on the device.  All device memory should be allocated on the CPU with cudaMalloc\n", label);
#else

  //NOTE!! varLock's writeLock() needs to be turned on prior to calling this function.
  //__________________________________
  //cpu code
  if (d_numItems==MAX_ITEM) {
    printf("ERROR:  out of GPUDataWarehouse space");
    varLock.writeUnlock();

    exit(-1);
  }

  //first check if this patch/var/matl is in the process of loading in.
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  //for (std::map<charlabelPatchMatl, allVarPointersInfo>::iterator it = varPointers.begin(); it != varPointers.end();  ++it) {
  //    printf("For label %s patch %d matl %d on device %d I have an entry\n", (*it).first.label.c_str(), d_device_id, (*it).first.patchID, (*it).first.matlIndex);
  //}

  int i=d_numItems;
  d_numItems++;
  strncpy(d_varDB[i].label, label, MAX_LABEL);
  d_varDB[i].domainID = patchID;
  d_varDB[i].matlIndex = matlIndex;
  d_varDB[i].xstride = xstride;
  d_varDB[i].varItem.validOnGPU = false; //We've created a variable, but we haven't yet put data into it.
  d_varDB[i].varItem.queueingOnGPU = true; //Assume that because we created space for this variable, data will soon be arriving.
  d_varDB[i].varItem.validOnCPU = false; //We don't know here if the data is on the CPU.
  d_varDB[i].varItem.gtype = None;  //PerPatch has no ghost cells
  d_varDB[i].varItem.numGhostCells = 0;
  var.getData(d_varDB[i].var_ptr);

  //Now store them in a map for easier lookups on the host
  //Without this map, we would have to loop through hundreds of array
  //The above is a good idea when we start implementing modified instead of just required and compute
  //elements looking for the right key.  With a map, we can find
  //our matching data faster, and we can store far more information
  //that the host should know and the device doesn't need to know about.
  allVarPointersInfo vp;
  vp.gridVar = NULL; //TODO: Is this even needed?
  vp.host_contiguousArrayPtr = host_ptr;
  vp.device_ptr = d_varDB[i].var_ptr;
  vp.device_offset = d_varDB[i].var_offset;
  vp.device_size = d_varDB[i].var_size;
  vp.varDB_index = i;


  if (varPointers.find(lpm) == varPointers.end()) {
    //printf("GPUDataWarehouse::put( %s ) Put a variable for label %s patch %d matl %d on device %d\n",label, label, patchID, matlIndex, d_device_id);
    varPointers.insert( std::map<charlabelPatchMatl, allVarPointersInfo>::value_type( lpm, vp ) );
  } else {
    printf("ERROR:\nGPUDataWarehouse::put( %s )  This gpudw database already has a variable for label %s patch %d matl %d on device %d in GPUDW at %p\n",label, label, patchID, matlIndex, d_device_id, d_device_copy);
    varLock.writeUnlock();
    exit(-1);
  }

  //if (strcmp(label, "sp_vol_CC") == 0) {
  //  printf("host put %s (patch: %d) loc %p into GPUDW %p on device %d, size [%d,%d,%d] with data %1.15e\n", label, patchID, d_varDB[i].var_ptr, d_device_copy, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z, *((double *)vp.host_contiguousArrayPtr));
  //}
  if (d_debug){
    printf("host put perPatch \"%s\" (patch: %d) material %d, location %p into GPUDW %p into d_varDB index %d on device %d, size [%d,%d,%d]\n", label, patchID, matlIndex, d_varDB[i].var_ptr, d_device_copy, i, d_device_id, d_varDB[i].var_size.x, d_varDB[i].var_size.y, d_varDB[i].var_size.z);
  }
  d_dirty=true;
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
#else
  //__________________________________
  //  cpu code
  cudaError_t retVal;
  void* addr = NULL;

  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
  CUDA_RT_SAFE_CALL(retVal = cudaMalloc(&addr, var.getMemSize()));

  if (d_debug && retVal == cudaSuccess) {
    printf("cudaMalloc for \"%s\", size %ld", label, var.getMemSize());
    printf(" at %p on device %d\n", addr, d_device_id);
  }

  var.setData(addr);
  put(var, label, patchID, matlIndex);
#endif
}

//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex, size_t sizeOfDataType)
{
#ifdef __CUDA_ARCH__  // need to limit output
  printf("ERROR:\nGPUDataWarehouse::allocateAndPut( %s )  You cannot use this on the device.  All memory should be allocated on the CPU with cudaMalloc()\n",label);
#else
  //__________________________________
  //  cpu code
  //check if it exists prior to allocating memory for it.
  //One scenario in which we need to handle is when two patches need ghost cells from the
  //same neighbor patch.  Suppose thread A's GPU patch 1 needs data from CPU patch 0, and
  //thread B's GPU patch 2 needs data from CPU patch 0.  This is a messy scenario, but the cleanest
  //is to have the first arriver between A and B to allocate memory on the GPU,
  //then the second arriver will use the same memory. The simplest approach is to have
  //threads A and B both copy patch 0 into this GPU memory.  This should be ok as ghost cell
  //data cannot be modified, so in case of a race condition, nothing will be modified
  //if a memory chunk is being both read and written to simultaneously.
  //So allow duplicates if the data is read only, otherwise don't allow duplicates.
  varLock.writeLock();
  //first check if this patch/var/matl is in the process of loading in.
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    int index = varPointers[lpm].varDB_index;
    if (d_varDB[index].varItem.queueingOnGPU == true) {
     //It's loading up, use that and return.
     if (d_debug){
       printf("GPUDataWarehouse::allocateAndPut( %s ). This gpudw database has a variable for label %s patch %d matl %d on device %d.  Reusing it.\n", label, label, patchID, matlIndex, d_device_id);
      }
     var.setData(d_varDB[index].var_ptr);

     varLock.writeUnlock();
     return;
   }
  }
  //It's not in the database, so create it.
  void* addr  = NULL;

  //This prepares the var with the offset and size.  The actual address will come next.
  //var.setData(addr);

  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&addr, var.getMemSize()) );

  if (d_debug) {
    printf("In allocateAndPut(), cudaMalloc for \"%s\ patch %d size %ld on thread %d ", label, patchID, var.getMemSize(), SCIRun::Thread::self()->myid());
    printf(" at %p on device %d\n", addr, d_device_id);
  }
  var.setData(addr);
  put(var, label, patchID, matlIndex, sizeOfDataType);
  varLock.writeUnlock();
#endif
}

//______________________________________________________________________
//
HOST_DEVICE GPUDataWarehouse::dataItem* 
GPUDataWarehouse::getItem(char const* label, int patchID, int matlIndex)
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
  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; //blockID on the grid
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;  //threadID in the block

  int i = threadID;
  index = -1;
  __syncthreads();  //sync before get


  //if (d_debug && threadID == 0 && blockID == 0) {
  //  printf("device getting item \"%s\" from GPUDW %p", label, this);
  //  printf("size (%d vars)\n Available labels:", d_numItems);
  //}

  //Have every thread try to find the label/patchId/matlIndex is a match in
  //array.  This is a clever approach so that instead of doing a simple
  //sequential search with one thread, we can let every thread search for it.  Only the
  //winning thread gets to write to shared data.

  while(i<d_numItems){
    int strmatch=0;
    char const *s1 = label; //reset s1 and s2 back to the start
    char const *s2 = &(d_varDB[i].label[0]);

    //a one-line strcmp.  This should keep branching down to a minimum.
    while (!(strmatch = *(unsigned char *) s1 - *(unsigned char *) s2) && *s1++ && *s2++);

    //only one thread will ever match this.
    if (strmatch == 0 && d_varDB[i].domainID == patchID && d_varDB[i].matlIndex == matlIndex) {
      index = i; //we found it.
    }
    i = i + numThreads; //Since every thread is involved in searching for the string, have this thread loop to the next possible item to check for.
  }

  //sync before return;
  __syncthreads();
  if (index == -1) {
    printf("ERROR:\nGPUDataWarehouse::getItem() didn't find anything for %s patch %d matl %d with threadID %d and numthreads %d\n", label, patchID, matlIndex, threadID, numThreads);
    return NULL;
  }
  return &d_varDB[index];
#else
  //__________________________________
  // cpu code
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  int i = 0;
  varLock.readLock();
  if (varPointers.find(lpm) != varPointers.end()) {
    i = varPointers[lpm].varDB_index;
    varLock.readUnlock();
  } else {
    varLock.readUnlock();
    printf("ERROR:\nGPUDataWarehouse::getItem( %s ) host get unknown variable from GPUDataWarehouse\n",label);
    exit(-1);
  }

  if (d_debug){
    printf("host got \"%s\" loc %p from GPUDW %p on device %u\n", label, d_varDB[i].var_ptr, d_device_copy, d_device_id);
  }
  //quick error check
  if (strcmp(d_varDB[i].label, label) != 0 || d_varDB[i].domainID != patchID || d_varDB[i].matlIndex != matlIndex) {
    printf("ERROR:\nGPUDataWarehouse::getItem( %s ), data does not match what was expected\n",label);
    exit(-1);
  }
  return &d_varDB[i];
#endif
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("exist() is only for framework code\n");
  return false;
#else
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  varLock.readLock();
  bool retVal = varPointers.find(lpm) != varPointers.end();
  varLock.readUnlock();
  return retVal;
#endif 
}

//______________________________________________________________________
//
HOST_DEVICE bool
GPUDataWarehouse::exist(char const* label, int patchID, int matlIndex, int3 host_size, int3 host_offset, bool skipContiguous, bool onlyContiguous )
{
#ifdef __CUDA_ARCH__
  printf("exist() is only for framework code\n");
  return false;
#else
  //check if we have matching label, patch, material, size and offsets.
  bool retVal = false;
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  varLock.readLock();
  if (varPointers.find(lpm) != varPointers.end()) {
    int3 device_offset = varPointers[lpm].device_offset;
    int3 device_size = varPointers[lpm].device_size;
    if (device_offset.x == host_offset.x && device_offset.y == host_offset.y && device_offset.z == host_offset.z
        && device_size.x == host_size.x && device_size.y == host_size.y && device_size.z == host_size.z) {
      retVal = true;
      //There is need sometimes to see if the variable exists, but not as part of a contiguous array
      if (skipContiguous) {
        if (varPointers[lpm].host_contiguousArrayPtr != NULL) {
          //It exists as part of a contiguous array
          retVal = false;
        }
      } if (onlyContiguous) {
        if (varPointers[lpm].host_contiguousArrayPtr == NULL) {
          //It exists as part of a contiguous array
          retVal = false;
        }
      }
    }
  }
  varLock.readUnlock();
  return retVal;
#endif
}


//______________________________________________________________________
//

HOST_DEVICE bool
GPUDataWarehouse::remove(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
  return false;
#else
  //Remove may be a bad idea entirely, avoid calling this unless you are
  //absolutely sure what you are doing
  bool retVal = false;
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  varLock.writeLock();
  if (varPointers.find(lpm) != varPointers.end()) {
    int i = varPointers[lpm].varDB_index;
    d_varDB[i].label[0] = '\0'; //leave a hole in the flat array, not deleted.
    varPointers.erase(lpm);
    retVal = true;
    d_dirty=true;
  }

  varLock.writeUnlock();
  return retVal;
#endif
}


//______________________________________________________________________
//
HOST_DEVICE void
GPUDataWarehouse::init_device(int id)
{
#ifdef __CUDA_ARCH__
  printf("GPUDataWarehouse::remove() should only be called by the framework\n");
#else

  d_device_id = id;
  OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
  void* temp = NULL;
  CUDA_RT_SAFE_CALL(cudaMalloc(&temp, sizeof(GPUDataWarehouse)));
  d_device_copy = (GPUDataWarehouse*)temp;
  //cudaHostRegister(this, sizeof(GPUDataWarehouse), cudaHostRegisterPortable);
  if (d_debug) {
    printf("Init GPUDW on-device copy %lu bytes to %p on device %u\n", sizeof(GPUDataWarehouse), d_device_copy, d_device_id);
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
  varLock.writeLock();

  if (d_dirty){
    OnDemandDataWarehouse::uintahSetCudaDevice( d_device_id );
    //Even though this is in a writeLock state on the CPU, the nature of multiple threads
    //each with their own stream copying to a GPU means that one stream might seemingly go out
    //of order.  This is ok for two reasons. 1) Nothing should ever be *removed* from a gpu data warehouse
    //2) Therefore, it doesn't matter if streams go out of order, each thread will still ensure it copies
    //exactly what it needs.  Other streams may write additional data to the gpu data warehouse, but cpu
    //threads will only access their own data, not data copied in by other cpu threada via streams.

    //This approach does NOT require CUDA pinned memory.

    CUDA_RT_SAFE_CALL (cudaMemcpyAsync( d_device_copy, this, sizeof(GPUDataWarehouse), cudaMemcpyHostToDevice, *((cudaStream_t*)cuda_stream)));


    if (d_debug) {
      printf("sync GPUDW %p to device %d on stream %p\n", d_device_copy, d_device_id, cuda_stream);
    }
    d_dirty=false;
  }

  varLock.writeUnlock();

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
  //for (int i=0; i<d_numItems; i++) {
  //  if (d_varDB[i].label[0] != 0){
  //    CUDA_RT_SAFE_CALL(retVal = cudaFree(d_varDB[i].var_ptr));
  //    
  //    if (d_debug){
  //      printf("cudaFree for \"%s\" at %p on device %d\n", d_varDB[i].label, d_varDB[i].var_ptr, d_device_id );
  //    }
  //  }


  //delete any grid var that isn't part of a contiguous array
  varLock.writeLock();
  std::map<charlabelPatchMatl, allVarPointersInfo>::iterator varIter;
  for (varIter = varPointers.begin(); varIter != varPointers.end(); ++varIter) {
    if (varIter->second.host_contiguousArrayPtr == NULL) {
      if (d_debug){
        printf("cuda Free for %s at device ptr %p on device %d\n" , d_varDB[varIter->second.varDB_index].label, d_varDB[varIter->second.varDB_index].var_ptr, d_device_id );

      }
      CUDA_RT_SAFE_CALL(cudaFree(d_varDB[varIter->second.varDB_index].var_ptr));
      d_varDB[varIter->second.varDB_index].var_ptr == NULL;
    }
  }

  varPointers.clear();
  if (d_debug){
    printf("Freeing zero or more tempGhostCells on device %d, cpu thread %d\n",d_device_id, SCIRun::Thread::self()->myid());
  }

  std::vector<tempGhostCellInfo>::iterator tempGhostCellsIter;
  for (tempGhostCellsIter = tempGhostCells.begin(); tempGhostCellsIter != tempGhostCells.end(); ++tempGhostCellsIter) {
      CUDA_RT_SAFE_CALL(cudaFree(tempGhostCellsIter->device_ptr));
      tempGhostCellsIter->device_ptr == NULL;
  }

  tempGhostCells.clear();

  //delete all the contiguous arrays
  std::map<std::string, contiguousArrayInfo>::iterator iter;
  for (iter = contiguousArrays.begin(); iter != contiguousArrays.end(); ++iter) {
    if (d_debug){
      printf("cuda Free for %s at device ptr %p on device %d\n" , iter->first.c_str(), iter->second.allocatedDeviceMemory, d_device_id );
      printf("delete[] for %s at host ptr %p on device %d\n" , iter->first.c_str(), iter->second.allocatedHostMemory, d_device_id );
    }
    CUDA_RT_SAFE_CALL(cudaFree(iter->second.allocatedDeviceMemory));
    //cudaHostUnregister(iter->second.allocatedHostMemory);
    free(iter->second.allocatedHostMemory);

  }
  contiguousArrays.clear();

  d_numItems=0;
  ghostCellsExist = false;
  resetdVarDB();
  varLock.writeUnlock();
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
  for (int i = 0; i < MAX_ITEM; i++) {
    d_varDB[i].label[0] = '\0';
    d_varDB[i].domainID = -1;
    d_varDB[i].matlIndex = -1;
    d_varDB[i].var_ptr = NULL;
    d_varDB[i].ghostItem.cpuDetailedTaskOwner = NULL;
    d_varDB[i].ghostItem.source_varDB_index = -1;
    d_varDB[i].ghostItem.dest_varDB_index = -1;

  }
  for (int i = 0; i < MAX_LVITEM; i++) {
    d_levelDB[i].label[0] = '\0';
    d_levelDB[i].domainID = -1;
    d_levelDB[i].matlIndex = -1;
    d_levelDB[i].var_ptr = NULL;
  }
  for (int i = 0; i < MAX_MATERIALS; i++) {
    d_materialDB[i].simulationType[0] = '\0';
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

  varLock.writeLock();
  //see if a thread has already supplied this datawarehouse with the material data
  int numMaterials = materials.size();

  if (d_numMaterials != numMaterials) {
    //nobody has given us this material data yet, so lets add it in from the beginning.

    if (numMaterials > MAX_MATERIALS) {
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

  varLock.writeUnlock();

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

__global__ void prepareGpuGhostCellIntoGpuArrayKernel(GPUDataWarehouse *gpudw, void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh) {
  gpudw->copyGhostCellsToArray(d_ghostCellData, index, ghostCellLow, ghostCellHigh);
}

HOST_DEVICE void
GPUDataWarehouse::prepareGpuGhostCellIntoGpuArray(void* cpuDetailedTaskOwner, void* toDetailedTask,
                                                    int3 ghostCellLow, int3 ghostCellHigh,
                                                    int xstride,
                                                    char const* label, int matlIndex,
                                                    int fromPatchID, int toPatchID,
                                                    int fromDeviceIndex, int toDeviceIndex,
                                                    int fromresource, int toresource )
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  int index = 0;
  charlabelPatchMatl lpm(label, fromPatchID, matlIndex);
  varLock.writeLock();
  if (varPointers.find(lpm) != varPointers.end()) {
    index = varPointers[lpm].varDB_index;

  } else {
    varLock.writeUnlock();

    printf("ERROR:\nGPUDataWarehouse::prepareGhostCellForCopyingInvoker, label: %s source patch ID %d, materialID %d not found in variable database", label, fromPatchID, matlIndex);
    exit(-1);
  }
  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * d_varDB[index].xstride;

  tempGhostCellInfo tgci;
  tgci.label = label;
  tgci.patchID = fromPatchID;
  tgci.matlIndex = matlIndex;
  tgci.cpuDetailedTaskOwner = cpuDetailedTaskOwner;
  tgci.toDetailedTask = toDetailedTask;
  //tgci->copied = false;
  tgci.memSize = ghostCellSize;
  tgci.ghostCellLow = ghostCellLow;
  tgci.ghostCellHigh = ghostCellHigh;
  tgci.xstride = xstride;
  tgci.toPatchID = toPatchID;
  tgci.fromDeviceIndex = fromDeviceIndex;
  tgci.toDeviceIndex = toDeviceIndex;

  if (d_varDB[index].var_offset.x == ghostCellLow.x &&
      d_varDB[index].var_offset.x == ghostCellLow.x &&
      d_varDB[index].var_offset.x == ghostCellLow.x &&
      d_varDB[index].var_offset.x+d_varDB[index].var_size.x == ghostCellHigh.x &&
      d_varDB[index].var_offset.y+d_varDB[index].var_size.y == ghostCellHigh.y &&
      d_varDB[index].var_offset.z+d_varDB[index].var_size.z == ghostCellHigh.y) {
    //The entire source is the ghost cell, so use that.
    tgci.device_ptr = d_varDB[index].var_ptr;
    tgci.usingVarDBData = true;
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
    prepareGpuGhostCellIntoGpuArrayKernel<<< dimGrid, dimBlock>>>(this->d_device_copy, d_ghostCellData, index, ghostCellLow, ghostCellHigh);

    //This copied ghost cell is only temporary.  Don't store it in the varDB, it should be moved and deleted very quickly.
    //Another reason not to store it in this object is the varPointers collection is a map, not a multimap.  We
    //do not want two entries for the same label/patch/matl tuple.
    tgci.device_ptr = d_ghostCellData;
    tgci.usingVarDBData = false;

  }

  if (d_debug) {
    printf("GPUDataWarehouse::prepareGhostCellForCopyingInvoker, Creating a tempGhostCellInfo from %d to %d starting at (%d, %d, %d) from patch %d to patch %d with address %p and size %d on threadID %d\n", fromDeviceIndex, toDeviceIndex, ghostCellLow.x, ghostCellLow.y, ghostCellLow.z, fromPatchID, toPatchID, tgci.device_ptr,  ghostCellSize, SCIRun::Thread::self()->myid());
  }
  tempGhostCells.push_back( tgci );
  varLock.writeUnlock();
#endif
}

HOST_DEVICE void
GPUDataWarehouse::prepareGpuToGpuGhostCellDestination(void* cpuDetailedTaskOwner, void* toDetailedTask,
                                                    int3 ghostCellLow, int3 ghostCellHigh,
                                                    int xstride,
                                                    char const* label, int matlIndex,
                                                    int fromPatchID, int toPatchID,
                                                    int fromDeviceIndex, int toDeviceIndex,
                                                    void * &data_ptr)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else



  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * xstride;
  OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
  CUDA_RT_SAFE_CALL( cudaMalloc(&data_ptr, ghostCellSize) );
  //call a kernel which gets the copy process started.

  //This ghost cell is only temporary.  Don't store it in the varDB, it should be moved and deleted very quickly.
  //Another reason not to store it in this object is the varPointers collection is a map, not a multimap.  We
  //do not want two entries for the same lebel/patch/matl tuple.
  varLock.writeLock();
  tempGhostCellInfo tgci;
  tgci.label = label;
  tgci.patchID = fromPatchID;
  tgci.matlIndex = matlIndex;
  tgci.cpuDetailedTaskOwner = cpuDetailedTaskOwner;
  tgci.toDetailedTask = toDetailedTask;
  //tgci->copied = false;
  tgci.device_ptr = data_ptr;
  tgci.memSize = ghostCellSize;
  tgci.ghostCellLow = ghostCellLow;
  tgci.ghostCellHigh = ghostCellHigh;
  tgci.xstride = xstride;
  tgci.toPatchID = toPatchID;
  tgci.fromDeviceIndex = fromDeviceIndex;
  tgci.toDeviceIndex = toDeviceIndex;
  if (d_debug){
    printf("GPUDataWarehouse::prepareGpuToGpuGhostCellDestination Creating a tempGhostCellInfo from %d to %d starting at (%d, %d, %d) with address %p and size %d on threadID %d\n", fromDeviceIndex, toDeviceIndex, ghostCellLow.x, ghostCellLow.y, ghostCellLow.z, data_ptr, ghostCellSize, SCIRun::Thread::self()->myid());
  }
  tempGhostCells.push_back( tgci );
  varLock.writeUnlock();
#endif
}

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
      if (d_varDB[index].xstride == sizeof(double)) {
        *(((double*)d_ghostCellData) + assignedCellID) = *(((double*)d_varDB[index].var_ptr) + varOffset);
      }

      //or copy all 4 bytes of an int in one shot.
      else if (d_varDB[index].xstride == sizeof(int)) {
        *(((int*)d_ghostCellData) + assignedCellID) = *(((int*)d_varDB[index].var_ptr) + varOffset);
      }
      //Copy each byte until we've copied all for this data type.
      else {
        //varOffset = varOffset * d_varDB[index].xstride;
        for (int j = 0; j < d_varDB[index].xstride; j++) {
          *(((char*)d_ghostCellData) + (assignedCellID *  d_varDB[index].xstride + j))
              = *(((char*)d_varDB[index].var_ptr) + (varOffset * d_varDB[index].xstride + j));

        }
      }
    }
    assignedCellID += totalThreads;
  }
#endif
}

//__global__ void copyGhostCellsToHostVarKernel(GPUDataWarehouse *gpudw, void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh) {
//  gpudw->copyGhostCellsToArray(d_ghostCellData, index, ghostCellLow, ghostCellHigh);
//}

/*
HOST_DEVICE void
GPUDataWarehouse::copyGhostCellsToHostVarInvoker(void* hostVarPointer, int3 ghostCellLow, int3 ghostCellHigh, char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  int index = 0;
  charlabelPatchMatl lpm_source(label, patchID, matlIndex);
  varLock.readLock();
  if (varPointers.find(lpm_source) != varPointers.end()) {
    index = varPointers[lpm_source].varDB_index;

  } else {
    printf("ERROR:\nGPUDataWarehouse::copyGhostCellsToHostVar, label: %s source patch ID %d, materialID %d not found in variable database", label, patchID, matlIndex);
    exit(-1);
  }
  varLock.readUnlock();

  void *d_ghostCellData = NULL;

  int ghostCellSize = (ghostCellHigh.x-ghostCellLow.x) * (ghostCellHigh.y-ghostCellLow.y) * (ghostCellHigh.z-ghostCellLow.z) * d_varDB[index].xstride;

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

HOST_DEVICE void
GPUDataWarehouse::copyTempGhostCellsToHostVar(void* hostVarPointer, int3 ghostCellLow, int3 ghostCellHigh, char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else


  charlabelPatchMatl lpm_source(label, patchID, matlIndex);
  varLock.readLock();
  tempGhostCellInfo tgci;
  bool found = false;
  //Check to make sure we have *exactly* the ghost cell we want.  (Other
  //CPU threads running other tasks can add their own data to the
  //tempGhostCells collection.
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
    printf("ERROR:\nGPUDataWarehouse::copyTempGhostCellsToHostVar, label: %s source patch ID %d, materialID %d not found in temporary ghost cell collection", label, patchID, matlIndex);
    exit(-1);
  }
  varLock.readUnlock();

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

HOST_DEVICE void
GPUDataWarehouse::copyGpuGhostCellsToGpuVars(void* taskID) {
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
   for (int i = 0; i < d_numItems; i++) {
     //don't copy somebody else's ghost cell, only our own, and only do it once.
     if (d_varDB[i].ghostItem.cpuDetailedTaskOwner == taskID && d_varDB[i].ghostItem.copied == false) {
       assignedCellID = blockID * numThreads + threadID;
       int sourceIndex = d_varDB[i].ghostItem.source_varDB_index;
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
           int x_source_real = x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[i].ghostItem.virtualOffset.x - d_varDB[i].var_offset.x;
           int y_source_real = y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[i].ghostItem.virtualOffset.y - d_varDB[i].var_offset.y;
           int z_source_real = z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[i].ghostItem.virtualOffset.z - d_varDB[i].var_offset.z;
           //count over array slots.
           int sourceOffset = x_source_real + d_varDB[i].var_size.x * (y_source_real  + z_source_real * d_varDB[i].var_size.y);

           int x_dest_real = x + d_varDB[i].ghostItem.sharedLowCoordinates.x - d_varDB[destIndex].var_offset.x;
           int y_dest_real = y + d_varDB[i].ghostItem.sharedLowCoordinates.y - d_varDB[destIndex].var_offset.y;
           int z_dest_real = z + d_varDB[i].ghostItem.sharedLowCoordinates.z - d_varDB[destIndex].var_offset.z;

           int destOffset = x_dest_real + d_varDB[destIndex].var_size.x * (y_dest_real + z_dest_real * d_varDB[destIndex].var_size.y);

           //printf("id is %d\n", assignedCellID);


           //copy all 8 bytes of a double in one shot
           if (d_varDB[i].xstride == sizeof(double)) {
             //if (destOffset == 134) {
             //if (d_ghostCellDB[i].sharedLowCoordinates.x == 0 && d_ghostCellDB[i].sharedLowCoordinates.y == 1 && d_ghostCellDB[i].sharedLowCoordinates.z == 1) {
               //printf("here: from %d (%d, %d, %d) to %d (%d, %d, %d) address is %p and offset it becomes %p and the value is %1.6lf\n",  sourceIndex,x_source_real,y_source_real,z_source_real, destIndex, x_dest_real,y_dest_real,z_dest_real, d_varDB[sourceIndex].var_ptr, (double*)(d_varDB[sourceIndex].var_ptr) + sourceOffset, *((double*)(d_varDB[sourceIndex].var_ptr) + sourceOffset));
             //  printf("copyGpuGhostCellsToGpuVars: copying between (%d, %d, %d) to (%d, %d, %d) address is %p and offset it becomes %p and the value is %1.6lf\n",  d_ghostCellDB[i].sharedLowCoordinates.x, d_ghostCellDB[i].sharedLowCoordinates.y, d_ghostCellDB[i].sharedLowCoordinates.z, d_ghostCellDB[i].sharedHighCoordinates.x, d_ghostCellDB[i].sharedHighCoordinates.y, d_ghostCellDB[i].sharedHighCoordinates.z, d_ghostCellDB[i].source_ptr, (double*)(d_ghostCellDB[i].source_ptr) + sourceOffset, *((double*)(d_ghostCellDB[i].source_ptr) + sourceOffset));
               //printf("copyGpuGhostCellsToGpuVars: copying from %d to %d - (%d, %d, %d) address is %p and offset it becomes %p and the value is %1.6lf\n", d_varDB[sourceIndex].domainID, d_varDB[destIndex].domainID, x, y, d_ghostCellDB[i].sharedLowCoordinates.z, d_varDB[sourceIndex].var_ptr, (double*)(d_varDB[sourceIndex].var_ptr) + sourceOffset, *((double*)(d_varDB[sourceIndex].var_ptr) + sourceOffset));
             //}
             *((double*)(d_varDB[destIndex].var_ptr) + destOffset) = *((double*)(d_varDB[i].var_ptr) + sourceOffset);
           }
           //or copy all 4 bytes of an int in one shot.
           else if (d_varDB[i].xstride == sizeof(int)) {
             *(((int*)d_varDB[destIndex].var_ptr) + destOffset) = *((int*)(d_varDB[i].var_ptr) + sourceOffset);
           //Copy each byte until we've copied all for this data type.
           } else {
             for (int j = 0; j < d_varDB[i].xstride; j++) {
               *(((char*)d_varDB[destIndex].var_ptr) + (destOffset * d_varDB[destIndex].xstride + j))
                   = *(((char*)d_varDB[i].var_ptr) + (sourceOffset * d_varDB[i].xstride + j));
             }
           }
         }
       }
     }
   }

#endif
}

 __global__ void copyGpuGhostCellsToGpuVarsKernel( GPUDataWarehouse *gpudw, void* taskID) {
   gpudw->copyGpuGhostCellsToGpuVars(taskID);
}

HOST_DEVICE void
GPUDataWarehouse::copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream, void* taskID)
{
#ifdef __CUDA_ARCH__
  //Not for the device side
#else

  bool copy_ghostCellsExist;
  //see if this GPU datawarehouse has ghost cells in it.
  varLock.readLock();
  //numGhostCells = d_numGhostCells;
  copy_ghostCellsExist = ghostCellsExist;
  varLock.readUnlock();
  if (copy_ghostCellsExist > 0) {
    //call a kernel which gets the copy process started.
    OnDemandDataWarehouse::uintahSetCudaDevice(d_device_id);
    const int BLOCKSIZE = 24;
    int xblocks = 32;
    int yblocks = 1;
    int zblocks = 1;
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);  //hopefully 24 *24 * 32 threads is enough for us.
    dim3 dimGrid(xblocks, yblocks, zblocks);
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

    copyGpuGhostCellsToGpuVarsKernel<<< dimGrid, dimBlock, 0, *stream >>>(this->d_device_copy, taskID);

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


HOST_DEVICE void
GPUDataWarehouse::putGhostCell(void* dtask, char const* label, int sourcePatchID, int destPatchID, int matlIndex,
                               int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset,
                               bool sourceIsInTempGhostCells, void * data_ptr, int3 var_offset, int3 var_size, int xstride) {
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::putGhostCell( %s )  Not implemented for GPU\n",label);
#else
  //Add information describing a ghost cell that needs to be copied internally from
  //one chunk of data to the destination.  This covers a GPU -> same GPU copy scenario.
  varLock.writeLock();
  int i = d_numItems;
  d_numItems++;
  ghostCellsExist = true;
  d_varDB[i].ghostItem.cpuDetailedTaskOwner = dtask;
  d_varDB[i].ghostItem.copied = false;
  d_varDB[i].ghostItem.sharedLowCoordinates = sharedLowCoordinates;
  d_varDB[i].ghostItem.sharedHighCoordinates = sharedHighCoordinates;
  d_varDB[i].ghostItem.virtualOffset = virtualOffset;
  if (d_debug){
    printf("Placed into the ghostCellDB in index %d from %d to %d has shared coordinates (%d, %d, %d), (%d, %d, %d)\n", i, sourcePatchID, destPatchID, sharedLowCoordinates.x, sharedLowCoordinates.y, sharedLowCoordinates.z, sharedHighCoordinates.x, sharedHighCoordinates.y, sharedHighCoordinates.z);
  }
  if (sourceIsInTempGhostCells) {
    d_varDB[i].ghostItem.source_varDB_index = -1;
    d_varDB[i].var_offset = var_offset;
    d_varDB[i].var_size = var_size;
    d_varDB[i].var_ptr = data_ptr;
    d_varDB[i].xstride = xstride;
  } else {
    //look up the source index and the destination index for these
    charlabelPatchMatl lpm_source(label, sourcePatchID, matlIndex);
    if (varPointers.find(lpm_source) != varPointers.end()) {
      int index = varPointers[lpm_source].varDB_index;
      if (d_varDB[index].varItem.validOnGPU == false && d_varDB[index].varItem.queueingOnGPU == false) {
        //Steps prior to this point should have checked for this scenario.
        //This is just a failsafe.
        printf("ERROR:\nGPUDataWarehouse::putGhostCell, attempting to use label: %s source patch ID %d, materialID %d, it exists but the data is not valid and not going to be valid on this GPU.", label, sourcePatchID, matlIndex);
        exit(-1);
      }
      d_varDB[i].ghostItem.source_varDB_index = index;
      d_varDB[i].var_offset = d_varDB[index].var_offset;
      d_varDB[i].var_size = d_varDB[index].var_size;
      d_varDB[i].var_ptr = d_varDB[index].var_ptr;
      d_varDB[i].xstride = d_varDB[index].xstride;
    } else {
      printf("ERROR:\nGPUDataWarehouse::putGhostCell, label: %s source patch ID %d, materialID %d not found in variable database\n", label, sourcePatchID, matlIndex);
      exit(-1);
    }
  }

  charlabelPatchMatl lpm_dest(label, destPatchID, matlIndex);
  if (varPointers.find(lpm_dest) != varPointers.end()) {
    d_varDB[i].ghostItem.dest_varDB_index = varPointers[lpm_dest].varDB_index;
  } else {;
    printf("ERROR:\nGPUDataWarehouse::putGhostCell, label: %s destination patch ID %d, materialID %d not found in variable database", label, destPatchID, matlIndex);
    exit(-1);
  }
  d_dirty=true;
  varLock.writeUnlock();
#endif
}

HOST_DEVICE int
GPUDataWarehouse::getNumGhostCells() {
#ifdef __CUDA_ARCH__

  //return d_numGhostCells;
  return -1;
#else
  //return d_numGhostCells;
  return -1;
#endif
}

HOST_DEVICE void GPUDataWarehouse::markGhostCellsCopied(void* taskID) {
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::markGhostCellsCopied()  Not implemented for GPU\n");
#else
  int numItems = d_numItems;
  for (int i = 0; i < numItems; i++) {
    if (d_varDB[i].ghostItem.cpuDetailedTaskOwner == taskID && d_varDB[i].ghostItem.copied == false) {
      d_varDB[i].ghostItem.copied = true;
    }
  }
#endif
}


HOST_DEVICE void
GPUDataWarehouse::getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells,
    char const* label, int patchID, int matlIndex) {
  GPUDataWarehouse::dataItem* item = getItem(label, patchID, matlIndex);
  low = item->var_offset;
  high.x = item->var_size.x - item->var_offset.x;
  high.y = item->var_size.y - item->var_offset.y;
  high.z = item->var_size.z - item->var_offset.z;
  siz = item->var_size;
  gtype = item->varItem.gtype;
  numGhostCells = item->varItem.numGhostCells;
}

HOST_DEVICE void GPUDataWarehouse::getTempGhostCells(void * dtask, std::vector<tempGhostCellInfo>& temp) {
#ifdef __CUDA_ARCH__
    printf("ERROR:\nGPUDataWarehouse::getTempGhostCells not implemented for GPU\n");
    exit(-1);
#else
  varLock.readLock();


  for ( vector<tempGhostCellInfo>::iterator it = tempGhostCells.begin();
        it != tempGhostCells.end();
        ++it) {
    //only this task should process its own outgoing GPU->other destination ghost cell copies
    if (dtask == (*it).cpuDetailedTaskOwner) {
      temp.push_back( (*it) );
    }
  }
  varLock.readUnlock();
#endif
}


HOST_DEVICE bool
GPUDataWarehouse::getValidOnGPU(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::getValidOnGPU( %s )  Not implemented yet for GPU\n",label);
  return false;
#else
  //__________________________________
  //  cpu code
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    allVarPointersInfo info = varPointers[lpm];
    int i = info.varDB_index;
    bool retVal = d_varDB[i].varItem.validOnGPU;
    varLock.readUnlock();
    return retVal;

  } else {
    varLock.readUnlock();
    return false;
  }
#endif
}

HOST_DEVICE void
GPUDataWarehouse::setValidOnGPU(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::setValidOnGPU( %s )  Not implemented yet for GPU\n",label);
#else
  //__________________________________
  //  cpu code
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    allVarPointersInfo info = varPointers[lpm];
    int i = info.varDB_index;
    d_varDB[i].varItem.validOnGPU = true;
    varLock.readUnlock();
  } else {
    varLock.readUnlock();
    printf("host setValidOnGPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
#endif
}

HOST_DEVICE bool
GPUDataWarehouse::getValidOnCPU(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::getValidOnCPU( %s )  Not implemented yet for GPU\n",label);
  return false;
#else
  //__________________________________
  //  cpu code
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    allVarPointersInfo info = varPointers[lpm];
    int i = info.varDB_index;
    bool retVal = d_varDB[i].varItem.validOnCPU;
    varLock.readUnlock();
    return retVal;

  } else {
    varLock.readUnlock();
    return false;
  }
#endif
}
HOST_DEVICE void
GPUDataWarehouse::setValidOnCPU(char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  printf("ERROR:\nGPUDataWarehouse::setValidOnCPU( %s )  Not implemented yet for GPU\n",label);
#else
  //__________________________________
  //  cpu code
  varLock.readLock();
  charlabelPatchMatl lpm(label, patchID, matlIndex);
  if (varPointers.find(lpm) != varPointers.end()) {
    allVarPointersInfo info = varPointers[lpm];
    int i = info.varDB_index;
    d_varDB[i].varItem.validOnCPU = true;
    varLock.readUnlock();
  } else {
    varLock.readUnlock();
    printf("host setValidOnCPU unknown variable %s on GPUDataWarehouse\n", label);
    exit(-1);
  }
#endif
}

HOST_DEVICE void
GPUDataWarehouse::printGetError(const char* msg, char const* label, int patchID, int matlIndex)
{
#ifdef __CUDA_ARCH__
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  int threadID = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

  int i = threadID;
  while (i < d_numItems) {
    printf("   Error - Available labels: \"%s\"\n", d_varDB[i].label);
    i = i + numThreads;
  }
  if (isThread0_Blk0()) {
    printf("  ERROR: %s( \"%s\", patchID: %i, matl: %i )  unknown variable\n\n", msg, label, patchID, matlIndex);
    assert(0);
  }

#else
  printf("\t ERROR: %s( \"%s\", patchID: %i, matl: %i )  unknown variable\n", msg, label, patchID, matlIndex);
#endif
}

} // end namespace Uintah
