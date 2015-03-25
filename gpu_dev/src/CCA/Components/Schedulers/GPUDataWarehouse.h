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

#ifndef GPU_DW_H
#define GPU_DW_H

#include <sci_defs/cuda_defs.h>
#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Grid/Variables/GPUPerPatch.h>

#include <map> //for host code only.
#include <string>
#include <vector>
#include <Core/Thread/CrowdMonitor.h>

#define MAX_ITEM 20  //If we have 100 patches, 2 materials, and 20 grid vars,
                       //that's 100 * 2 * 20 = 4000.  Make sure we have room!
#define MAX_GHOST_CELLS 100  //MAX_ITEM * 6 one for each face.
#define MAX_MATERIALS 1
#define MAX_LVITEM 1
#define MAX_LABEL   20

namespace Uintah {

enum materialType {
  IDEAL_GAS = 0/*,
  HARD_SPHERE_GAS_EOS = 1,
  TST_EOS = 2,
  JWL_EOS = 3,
  JWLC_EOS = 4,
  MURNAGHAN_EOS = 5,
  BIRCH_MURNAGHAN_EOS = 6,
  GRUNEISEN_EOS = 7,
  TILLOTSON_EOS = 8,
  THOMSEN_HARTKA_WATER_EOS = 9,
  KNAUSS_SEA_WATER_EOS = 10,
  KUMARI_DASS_EOS = 11 */
};

class GPUDataWarehouse {

public:

  GPUDataWarehouse(): allocateLock("allocate lock"), varLock("var lock"){
    d_numItems = 0;
    d_numMaterials = 0;
    d_numGhostCells = 0;
    d_device_copy = NULL;
    d_debug = false;
    d_dirty = true;
    d_device_id = -1;
    resetdVarDB();

  }
  virtual ~GPUDataWarehouse() {};

  enum GhostType {
    None,
    AroundNodes,
    AroundCells,
    AroundFacesX,
    AroundFacesY,
    AroundFacesZ,
    AroundFaces,
    numGhostTypes // 7
  };

  struct dataItem {   // flat array
    char       label[MAX_LABEL];
    int        domainID;
    int        matlIndex;
    int3       var_offset;  
    int3       var_size;
    size_t     num_elems;
    void*      var_ptr;
    int        xstride;
    GhostType  gtype;
    int        numGhostCells;
    bool       validOnGPU; //true if the GPU copy is the "live" copy and not an old version of the data.
    bool       validOnCPU; //true if the CPU copy is the current "live" copy. (It's possible to be both.)
    bool       queueingOnGPU; //true if we've created the variable but we haven't yet set validOnGPU to true.
  };

  struct materialItem {
      materialType material;
      char       simulationType[MAX_LABEL]; //ICE, MPM, etc.  Currently unused
  };

  struct ghostCellItem {

    //The pointer to the task that needs this ghost cell copy
    void* cpuDetailedTaskOwner;
    bool copied;

    //This assumes the ghost cell is already in the GPU's memory
    //We only need to know the source patch, the destination patch
    //the material number, and the shared coordinates.
    int3 sharedLowCoordinates;
    int3 sharedHighCoordinates;

    //Wasatch has virtual patches, which come as a result of periodic boundary cells,
    //which wrap around on each other.  (Like a label wrapping around a soup can, but
    //for all boundary faces of the grid).  While they're technically sharing the same
    //coordinates (just wrapped around once), from our perspective we need their actual indexes
    //this offset helps us get that.
    int3 virtualOffset;

    //So we can look up the size and offset information in the d_varDB
    int dest_varDB_index;

    //Ghost cells can come from two locations.  The first way is that they are already in the
    //d_varDB (such as same GPU->same GPU ghost cell, or CPU ghost cell->GPU).
    //The other way is if the data comes from another GPU on the same node,
    //and if so, then the data will be found in d_ghostCellDB instead of d_varDB.
    //If it's the former, then this source_varDB_index will be the correspondinb d_varDB index.
    //If it's the latter, then this source_varDB_index will be -1.
    int source_varDB_index;

    int3 var_offset;
    int3 var_size;
    void * source_ptr;
    int xstride;

  };

  struct contiguousArrayInfo {
    void * allocatedDeviceMemory;
    void * allocatedHostMemory;
    size_t sizeOfAllocatedMemory;
    size_t assignedOffset;
    size_t copiedOffset;
    //The default constructor
    contiguousArrayInfo() {
      allocatedDeviceMemory = NULL;
      allocatedHostMemory = NULL;
      sizeOfAllocatedMemory = 0;
      assignedOffset = 0; //To keep up to the point where data has been "put".  Computes data will be assigned
      copiedOffset = 0; //To keep up to the point where data will need to be copied.  Required data will be copied
    }
    //Overloaded constructor
    contiguousArrayInfo(double * allocatedDeviceMemory,
                        double * allocatedHostMemory,
                        size_t sizeOfAllocatedMemory) {
      this->allocatedDeviceMemory = allocatedDeviceMemory;
      this->allocatedHostMemory = allocatedHostMemory;
      this->sizeOfAllocatedMemory = sizeOfAllocatedMemory;
      assignedOffset = 0; //To keep up to the point where data has been "put"
      copiedOffset = 0; //To keep up to the point where data has been copied.
    }
  };


  struct allVarPointersInfo {
    void*      device_ptr;   //Where it is on the device
    int3       device_offset;
    int3       device_size;
    void*      host_contiguousArrayPtr;  //Use this address only if partOfContiguousArray is set to true.
    GridVariableBase* gridVar;  //The host variable, which also holds the host pointer to the data, and
                                //retains a reference so the variable isn't deleted

    int        varDB_index;     //Where this also shows up in the varDB.  We can use this to
                                //get the rest of the information we need.

  };


  struct charlabelPatchMatl {
    std::string     label;
    int        patchID;
    int        matlIndex;
    charlabelPatchMatl(const char * label, int patchID, int matlIndex) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndex = matlIndex;
    }
    //This so it can be used in an STL map
    bool operator<(const charlabelPatchMatl& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndex < right.matlIndex)) {
        return true;
      } else {
        return false;
      }

    }

  };


  struct tempGhostCellInfo {  //We only need enough information to copy a linear chunk of data.
    std::string     label;
    int        patchID;
    int        matlIndex;

    void* cpuDetailedTaskOwner;   //Only the task that placed the item in should be the one that copies it.

    void* toDetailedTask;  //When we move this into the d_ghostDB, we need to know what task gets to do the copies.
    int        xstride;

    void*      device_ptr;
    int        memSize;
    int3 ghostCellLow;
    int3 ghostCellHigh;
    int toPatchID;
    int fromDeviceIndex;
    int toDeviceIndex;
    bool usingVarDBData;
    //int fromresource;
    //int toresource;
    //bool copied;
  };

  //______________________________________________________________________
  // GPU GridVariable methods
  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, int patchID, int matlIndex);

  HOST_DEVICE void getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex);

  //HOST_DEVICE void put(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);
  HOST_DEVICE void put(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);

  HOST_DEVICE void allocateAndPut(GPUGridVariableBase& var, char const* label, int patchID, int matlID, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype = None, int numGhostCells = 0);
  HOST_DEVICE void allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndex);
  HOST_DEVICE void allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex, size_t sizeOfDataType);

  //HOST_DEVICE void* getPointer(char const* label, int patchID, int matlIndex);
  HOST_DEVICE void putContiguous(GPUGridVariableBase &var, char const* indexID, char const* label, int patchID, int matlID, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost);
  HOST_DEVICE void allocate(const char* indexID, size_t size);
private:

  HOST_DEVICE void put(GPUGridVariableBase &var, char const* label, int patchID, int matlIndex, size_t xstride, GhostType gtype = None, int numGhostCells = 0, GridVariableBase* gridVar = NULL, void* hostPtr = NULL);
  HOST_DEVICE void put(GPUPerPatchBase& var, char const* label, int patchID, int matlIndex, size_t xstride, GPUPerPatchBase* gridVar = NULL, void* hostPtr = NULL);

public:

  //______________________________________________________________________
  // GPU DataWarehouse support methods
  
  HOST_DEVICE bool exist(char const* label, int patchID, int matlID);
  HOST_DEVICE bool exist(char const* label, int patchID, int matlID, int3 host_size, int3 host_offset, bool skipContiguous = false, bool onlyContiguous = false);
  HOST_DEVICE bool existContiguously(char const* label, int patchID, int matlIndex, int3 host_size, int3 host_offset);
  HOST_DEVICE bool remove(char const* label, int patchID, int matlID);
  HOST_DEVICE void init_device(int id);
  HOST_DEVICE void syncto_device(void *cuda_stream);
  HOST_DEVICE void clear();
  HOST_DEVICE GPUDataWarehouse* getdevice_ptr(){return d_device_copy;};
  HOST_DEVICE void setDebug(bool s){d_debug=s;}
  HOST_DEVICE cudaError_t copyDataHostToDevice(char const* indexID, void *cuda_stream);
  HOST_DEVICE cudaError_t copyDataDeviceToHost(char const* indexID, void *cuda_stream);
  HOST_DEVICE void copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndex);

  //______________________________________________________________________
  //Additional support methods
  HOST_DEVICE void putMaterials(std::vector< std::string > materials);
  HOST_DEVICE materialType getMaterial(int i) const;
  HOST_DEVICE int getNumMaterials() const;
  HOST_DEVICE void putGhostCell(void* dtask, char const* label, int sourcePatchID, int destPatchID, int matlID,
                                int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset,
                                bool sourceIsInTempGhostCells, void * data_ptr, int3 var_offset, int3 var_size, int xstride);
  HOST_DEVICE bool getValidOnGPU(char const* label, int patchID, int matlID);
  HOST_DEVICE void setValidOnGPU(char const* label, int patchID, int matlID);
  HOST_DEVICE bool getValidOnCPU(char const* label, int patchID, int matlID);
  HOST_DEVICE void setValidOnCPU(char const* label, int patchID, int matlID);

  //Regarding ghost cells...
  //Scenario 1: ghost cell on CPU -> var on same CPU:  Managed by getGridVar().
  //Scenario 2: ghost cell on CPU -> var on different CPU:  Managed by sendMPI()
  //Scenario 3: ghost cell on different CPU -> var on CPU:  Managed by recvMPI() and then getGridVar().

  //Scenario 4: ghost cell on GPU -> var on different host's CPU:  Immediately after a task runs,
  // prepareGPUDependencies is invoked, and it discovers all external dependencies.  It then calls
  // the correct GPU DW's prepareGpuGhostCellIntoGpuArray which stages the ghost cell into an array on
  // that GPU.  Then sendMPI() is called, which calls copyTempGhostCellsToHostVar(), which finds the
  // correct array previously made and copies it to a host grid var.  From there the MPI engine
  // can manage it normally.

  //Scenario 5: ghost cell on GPU -> var on different host's GPU:  Immediately after a task runs,
  // prepareGPUDependencies is invoked, and it discovers all external dependencies.  It then calls
  // the correct GPU DW's prepareGpuGhostCellIntoGpuArray which stages the ghost cell into an array on
  // that GPU.  Then sendMPI() is called, which calls copyTempGhostCellsToHostVar(), which finds the
  // correct array previously made and copies it to a host grid var.  From there the receiving host
  // moves onto scenario #6.

  //Scenario 6: ghost cell different GPU -> var on current host's GPU: recvMPI() is called and processes
  // data from other nodes.  These are added to the CPU DW. Then it follows the steps listed in the prior
  // scenario. The UnifiedScheduler sees that the destination is
  // valid on the GPU but the ghost cell is not, it is instead on the CPU.  So it puts the CPU ghost cell
  // in the GPU DW.  It also adds the copying information and task ID information to correct GPU DW's d_ghostCellDB.
  // Once d_ghostCellDB is on the GPU then a GPU kernel is called via copyGpuGhostCellsToGpuVarsInvoker
  // which process all entries owned by the same task ID listed earlier.  All data is then copied into the
  // correct destination GPU var.

  //Scenario 7: ghost cell different CPU -> var on current host's GPU: Same as scenario 5.

  //Scenario 8: ghost cell on CPU -> var on current host's GPU: The UnifiedScheduler sees that the destination is
  // valid on the GPU but the ghost cell is not, it is instead on the CPU.  So it puts the CPU ghost cell
  // in the GPU DW.  It also adds the copying information and task ID information to correct GPU DW's d_ghostCellDB.
  // Once d_ghostCellDB is on the GPU then a GPU kernel is called via copyGpuGhostCellsToGpuVarsInvoker
  // which process all entries owned by the same task ID listed earlier.  All data is then copied into the
  // correct destination GPU var.

  //Scenario 9: ghost cell on GPU -> var on same GPU:  initiateH2D recognizes the destination var is valid
  // and so it adds the copying information and task ID information to correct GPU DW's d_ghostCellDB.
  // Once d_ghostCellDB is on the GPU then a GPU kernel is called via copyGpuGhostCellsToGpuVarsInvoker
  // which process all entries owned by the same task ID listed earlier.  All data is then copied into the
  // correct destination GPU var.

  //Scenario 10: var and ghost cell on CPU -> var on GPU: initiateH2D recognizes the data is on the CPU but not on the GPU.
  // It calls the CPU's getGridVar indicating how many ghost cells, resulting in a CPU grid var with ghost cells in it.
  // It is added to the GPU DW by adding it to the host's varPointers collection and d_varDB collection.  The
  // d_varDB is copied to the GPU.

  //Scenario 11: var and ghost cell on GPU -> var on CPU: The schedule recognizes the task is on the CPU, so initiateD2H
  // checks if any data is valid on the GPU (see TODO).  If the GPU var has larger dimensions, then resize the CPU var to match
  // (any task not needing ghost cells will just ignore it anyway).  Copy the data to the CPU.  Then mark CPU data
  // as valid.

  //Scenario 12: ghost cells and vars are on both CPU and a GPU -> var on different GPU: This happens after an data output
  //task runs.  Instead of doing a GPU->different GPU copy, instead treat it like Scenario 9.

  //Scenario 13: Ghost cells on one GPU -> var on different GPU but same host:  Immediately after a task runs,
  // prepareGPUDependencies is invoked, and it discovers all internal dependencies.  It then calls
  // the correct GPU DW's prepareGpuGhostCellIntoGpuArray which stages the ghost cell into an array on
  // that GPU.  postMPISends() is invoked.  Then copyGPUInternalDependencies() is invoked.  It gets a collection
  // of all tempGhostCells items belonging to that task.  For each ghost cell it then calls
  // prepareGpuToGpuGhostCellDestination(), which creates a same sized array on the destination GPU.  This info
  // is added into that GPU DW tempGhostCells.  Then a cudaMemcpyPeer() is called copying data between devices.
  //
  //

  //Useful for sendMPI.  Processes the tempGhostCells collection for the exact var and size that sendMPI requested.
  //It pulls data out of the GPU into a host array variable.  From there it's managed like any other CPU variable.
  HOST_DEVICE void  copyTempGhostCellsToHostVar(void* hostVarPointer, int3 ghostCellLow, int3 ghostCellHigh, char const* label, int patchID, int matlID);

  //Called by prepareGPUDependencies, handles the GPU->anywhere else but the same GPU
  //It does this by staging the ghost cell data into an array for something else to pick up and copy.
  //These arrays are tracked by the tempGhostCells collection.
  HOST_DEVICE void prepareGpuGhostCellIntoGpuArray(void* cpuDetailedTaskOwner, void* toDetailedTask,
                                                    int3 ghostCellLow, int3 ghostCellHigh,
                                                    int xstride,
                                                    char const* label, int matlID,
                                                    int fromPatchID, int toPatchID,
                                                    int fromDeviceIndex, int toDeviceIndex,
                                                    int fromresource, int toresource);
  //Called by prepareGpuGhostCellIntoGpuArray, this is a method for a kernel which
  //goes through everything listed in the d_ghostCellDB and copies them to a specified array.
  HOST_DEVICE void copyGhostCellsToArray(void* d_ghostCellData, int index, int3 ghostCellLow, int3 ghostCellHigh);


  //This and the function below go through the d_ghostCellData array and copies data into
  //the correct destination GPU var.  This would be the final step of a GPU ghost cell transfer.
  HOST_DEVICE void copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream, void* taskID);
  HOST_DEVICE void copyGpuGhostCellsToGpuVars(void* taskID);

  //Called by copyGPUGhostCellsBetweenDevices().  If the destination is another GPU on the same physical node
  //then this creates room on the destination GPU and stores that information in d_ghostCellDB to
  //later process.
  HOST_DEVICE void prepareGpuToGpuGhostCellDestination(void* cpuDetailedTaskOwner, void* toDetailedTask,
                                                      int3 ghostCellLow, int3 ghostCellHigh,
                                                      int xstride,
                                                      char const* label, int matlID,
                                                      int fromPatchID, int toPatchID,
                                                      int fromDeviceIndex, int toDeviceIndex,
                                                      void * &data_ptr);

  HOST_DEVICE int getNumGhostCells();
  HOST_DEVICE void markGhostCellsCopied(void* taskID);
  HOST_DEVICE void getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells, char const* label, int patchID, int matlIndex);

  HOST_DEVICE dataItem* getItem(char const* label, int patchID, int matlID);
  HOST_DEVICE void getTempGhostCells(void * dtask, std::vector<tempGhostCellInfo>& temp);
private:
  HOST_DEVICE void resetdVarDB();
 // HOST_DEVICE void copyGpuGhostCellsToGpuVars();

private:

  HOST_DEVICE void printGetError(const char* msg, char const* label, int patchID, int matlIndex);


  int d_numItems;


  int d_numMaterials;

  int d_numGhostCells;

   materialItem d_materialDB[MAX_MATERIALS];
  dataItem d_levelDB[MAX_LVITEM];
  GPUDataWarehouse*  d_device_copy;             //The pointer to the copy of this object in the GPU.
  bool d_dirty;
  int d_device_id;
  bool d_debug;


  //These being here do not pose a problem for the CUDA compiler

  std::map<charlabelPatchMatl, allVarPointersInfo> varPointers; //For the host side.  The variable database.
                                                   //Holds the host ptr, the device ptr, and a staging
                                                   //contiguous host ptr var location.  This is not part
                                                   //of d_varDB because the device doesn't need to know about
                                                   //the host pointers.  Being a map makes this much faster.

  dataItem d_varDB[MAX_ITEM];                      //For the device side.  The variable database.

  std::map<std::string, contiguousArrayInfo> contiguousArrays;

  std::vector<tempGhostCellInfo> tempGhostCells; //For the host side.  Holds both ghost cell data and information about prepared ghosts cell.
                                                 //Data is loaded in here through prepareGPUDependencies, which checks ahead and
                                                 //loads all ghost cells into contiguous chunks ready for copying to the destination.
                                                 //See prepareGPUDependencies for more info.

  ghostCellItem d_ghostCellDB[MAX_GHOST_CELLS];  //For the device side. This contains information about what ghost cell copies need to complete.
                                                 //This only covers GPU->same GPU scenarios.  It is meant to copy the data directly into the
                                                 //correct GPU var, thus completing the ghost cell merging process.


  mutable SCIRun::CrowdMonitor allocateLock;
  mutable SCIRun::CrowdMonitor varLock;

};


} //end namespace Uintah

#endif
