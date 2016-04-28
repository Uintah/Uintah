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

#ifndef CCA_COMPONENTS_SCHEDULERS_GPUDATAWAREHOUSE_H
#define CCA_COMPONENTS_SCHEDULERS_GPUDATAWAREHOUSE_H

#include <sci_defs/cuda_defs.h>
#include <Core/Grid/Variables/GPUVariable.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUReductionVariable.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Grid/Variables/GPUPerPatch.h>
#include <Core/Thread/CrowdMonitor.h>


#include <map> //for host code only.
#include <string>
#include <vector>
#include <Core/Thread/CrowdMonitor.h>

#define MAX_VARDB_ITEMS       10000000  //Due to how it's allocated, it will never use up this much space.
                                        //Only a very small fraction of it.
#define MAX_MATERIALSDB_ITEMS 20
#define MAX_LEVELDB_ITEMS     20        //TODO: Is this needed?
#define MAX_NAME_LENGTH       32        //How big a particular label can be.

#define LEVEL_PATCH_ID = -99999999; //A sentinel value used for a patch ID when we're storing a
                                    //region of patches for a level instead of a regular patch.
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
class OnDemandDataWarehouse;

class GPUDataWarehouse;

class GPUDataWarehouse {

public:

  //GPUDataWarehouse(int id, void * placementNewBuffer) : allocateLock("allocate lock"), varLock("var lock"){ };


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

  struct materialItem {
    materialType material;
    char       simulationType[MAX_NAME_LENGTH]; //ICE, MPM, etc.  Currently unused
  };


  //The dataItem can hold two kinds of data.
  //The first is information related to a regular data variable.
  //The second is information indicating how a ghost cell should be copied from one var to another
  //The two types of information are stored in one struct to allow us to make the size of the GPU
  //data warehouse dynamic.  For small problems the GPUDW can be small, and large problems it can
  //be large.
  //The biggest problem is that multiple CPU threads will be adding to the size of the GPUDW
  //IF we had two separate structs, then we would have to let both grow independently and
  //it would require two copies to the GPU DW instead of one.
  //So the solution is to allocate a large buffer of possible GPUDW data in init_device(),
  //one on the host RAM and device RAM
  //Then on the CPU side a thread running a task will collect various dataItems, and just before
  //it sends it to the GPU DW, it will dump its results into the host side buffer (using locks).
  //Then it can either copy in only as much of the GPUDW as needed, instead of the entire buffer.

  struct VarItem {
    GhostType       gtype;
    unsigned int    numGhostCells;
    bool            staging;
  };

  struct GhostItem {
    //This assumes the ghost cell is already in the GPU's memory
    //We only need to know the source patch, the destination patch
    //the material number, and the shared coordinates.
    int3            sharedLowCoordinates;
    int3            sharedHighCoordinates;

    //Wasatch has virtual patches, which come as a result of periodic boundary cells,
    //which wrap around on each other.  (Like a label wrapping around a soup can, but
    //for all boundary faces of the grid).  While they're technically sharing the same
    //coordinates (just wrapped around once), from our perspective we need their actual indexes
    //this offset helps us get that.
    int3 virtualOffset;

    //So we can look up the size and offset information in the d_varDB
    int dest_varDB_index;  //Will be set to -1 if this d_varDB item isn't a ghost cell item.

  };

  struct dataItem {

    char            label[MAX_NAME_LENGTH];  // VarLabel name
    int             domainID;          // a Patch ID (d_VarDB)
    int             matlIndx;          // the material index
    int             levelIndx;         // level the variable resides on (AMR)
    int3            var_offset;        // offset
    int3            var_size;          // dimensions of GPUGridVariable
    void*           var_ptr;           // raw pointer to the memory
    unsigned int    sizeOfDataType;    // the memory size of a single data element.
    VarItem         varItem;           // If the item is holding variable data, remaining info is found in here
    GhostItem       ghostItem;         // If the item contains only ghost cell copying meta data, its info is found in here

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

  enum status { UNALLOCATED               = 0x00000000,
                ALLOCATING                = 0x00000001,
                ALLOCATED                 = 0x00000002,
                COPYING_IN                = 0x00000004,
                VALID                     = 0x00000008,  //For when a variable has its data, this excludes any knowledge of ghost cells.
                AWAITING_GHOST_COPY       = 0x00000010,  //For when when we know a variable is awaiting ghost cell data
                                                         //It is possible for VALID bit set to 0 or 1 with this bit set,
                                                         //meaning we can know a variable is awaiting ghost copies but we
                                                         //don't know from this bit alone if the variable is valid yet.
                VALID_WITH_GHOSTS         = 0x00000020,  //For when a variable has its data and it has its ghost cells
                                                         //Note: Change to just GHOST_VALID?  Meaning ghost cells could be valid but the
                                                         //non ghost part is unknown?
                UNKNOWN                   = 0x80000040}; //TODO: REMOVE THIS WHEN YOU CAN, IT'S NOT OPTIMAL DESIGN.
                //COPYING_OUT = The remaining bits. See below.


  typedef int atomicDataStatus;

  //    0                   1                   2                   3
  //    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //   |                                                   | | | | | | |
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  //Not allocated/Invalid = If the value is 0x00000000

  //Allocating                = bit 31 - 0x00000001
  //Allocated                 = bit 30 - 0x00000002
  //Copying in                = bit 29 - 0x00000004
  //Valid                     = bit 28 - 0x00000008
  //awaiting ghost data       = bit 27 - 0x00000010
  //Valid with ghost cells    = bit 26 - 0x00000020
  //Copying out               = bits 1 through 25, allowing for 25 copy out sources.

  //With this approach we can allow for multiple copy outs, but only one copy in.
  //We should never attempt to copy unless the status is odd (allocated)
  //We should never copy out if the status isn't valid.



  struct stagingVar {

    int3            device_offset;
    int3            device_size;
    //This so it can be used in an STL map
    bool operator<(const stagingVar& rhs) const {
      if (this->device_offset.x < rhs.device_offset.x) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x
             && (this->device_offset.y < rhs.device_offset.y)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x
             && (this->device_offset.y == rhs.device_offset.y)
             && (this->device_offset.z < rhs.device_offset.z)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x
             && (this->device_offset.y == rhs.device_offset.y)
             && (this->device_offset.z == rhs.device_offset.z)
             && (this->device_size.x < rhs.device_size.x)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x
             && (this->device_offset.y == rhs.device_offset.y)
             && (this->device_offset.z == rhs.device_offset.z)
             && (this->device_size.x == rhs.device_size.x)
             && (this->device_size.y < rhs.device_size.y)) {
        return true;
      } else if (this->device_offset.x == rhs.device_offset.x
             && (this->device_offset.y == rhs.device_offset.y)
             && (this->device_offset.z == rhs.device_offset.z)
             && (this->device_size.x == rhs.device_size.x)
             && (this->device_size.y == rhs.device_size.y)
             && (this->device_size.z < rhs.device_size.z)) {
        return true;
      } else {
        return false;
      }
    }
  };

  struct stagingVarInfo {
    void*           device_ptr;   //Where it is on the device
    void*           host_contiguousArrayPtr;  //Use this address only if partOfContiguousArray is set to true.
    int             varDB_index;
    atomicDataStatus      atomicStatusInHostMemory;
    atomicDataStatus      atomicStatusInGpuMemory;

  };

  struct allVarPointersInfo {
    allVarPointersInfo() {
      __sync_fetch_and_and(&atomicStatusInHostMemory, UNALLOCATED);
      __sync_fetch_and_and(&atomicStatusInGpuMemory, UNALLOCATED);
      varDB_index = -1;
    }
    void*           device_ptr;   //Where it is on the device
    void*           host_contiguousArrayPtr;  //Use this address only if partOfContiguousArray is set to true.
    int3            device_offset;
    int3            device_size;
    unsigned int    sizeOfDataType;

    GhostType       gtype;
    unsigned int    numGhostCells;

    int             varDB_index;     //Where this also shows up in the varDB.  We can use this to
                                //get the rest of the information we need.
    atomicDataStatus      atomicStatusInHostMemory;
    atomicDataStatus      atomicStatusInGpuMemory;
    //bool            validOnGPU; //true if the GPU copy is the "live" copy and not an old version of the data.
    //bool            validOnCPU; //true if the CPU copy is the current "live" copy. (It's possible to be both.)

    std::map<stagingVar, stagingVarInfo> stagingVars;  //When ghost cells in the GPU need to go to another memory space
                                                       //we will be creating temporary contiguous arrays to hold that
                                                       //information.  After many iterations of other attempts, it seems
                                                       //creating a map of staging vars is the cleanest way to go for data
                                                       //in GPU memory.
  };


  struct labelPatchMatlLevel {
    std::string label;
    int         patchID;
    int         matlIndx;
    int         levelIndx;
    labelPatchMatlLevel(const char * label, int patchID, int matlIndx, int levelIndx) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
      this->levelIndx = levelIndx;
    }
    //This so it can be used in an STL map
    bool operator<(const labelPatchMatlLevel& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx < right.matlIndx)) {
        return true;
      } else if (    this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx)
                 && (this->levelIndx < right.levelIndx)) {
        return true;
      } else {
        return false;
      }

    }

  };

  struct gpuMemoryData {
    int            status;  //0 for available, 1 for in use.
    unsigned int   timestep;
    void*          ptr;
  };
  struct gpuMemoryPoolItem {

     unsigned int  device_id;
     size_t        deviceSize;

     gpuMemoryPoolItem(unsigned int device_id, size_t deviceSize) {
       this->device_id = device_id;
       this->deviceSize = deviceSize;
     }
     //This so it can be used in an STL map
     bool operator<(const gpuMemoryPoolItem& right) const {
       if (this->device_id < right.device_id) {
         return true;
       } else if ((this->device_id == right.device_id) && (this->deviceSize < right.deviceSize)) {
         return true;
       } else {
         return false;
       }
     }
   };

  //______________________________________________________________________
  // GPU GridVariable methods
  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, const int patchID, const int matlIndx, const int levelIndx);
  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, const int patchID, const int matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }
  HOST_DEVICE void getStagingVar(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  HOST_DEVICE bool stagingVarExists(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);


  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, const int patchID, const int matlIndx, const int levelIndx);
  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, const int patchID, const int matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, const int patchID, const int matlIndx, const int levelIndx);
  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, const int patchID, const int matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getLevel(const GPUGridVariableBase& var, char const* label, int matlIndx, int levelIndx);


  HOST_DEVICE void getModifiable(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void getModifiable( GPUGridVariableBase&  var, char const* label, int patchID, int matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getModifiable(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void getModifiable( GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getModifiable(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void getModifiable( GPUPerPatchBase&  var, char const* label, int patchID, int matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  //HOST_DEVICE void put(GPUGridVariableBase& var, char const* label, int patchID, int matlIndex, bool overWrite=false);
  __host__ void put(GPUGridVariableBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, bool staging = false, GhostType gtype = None, int numGhostCells = 0, void* hostPtr = NULL);
  __host__ void put(GPUReductionVariableBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, void* hostPtr = NULL);
  __host__ void put(GPUPerPatchBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, void* hostPtr = NULL);

  __host__ void allocateAndPut(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype = None, int numGhostCells = 0);
  __host__ void allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType);
  __host__ void allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType);

  __host__ void putUnallocatedIfNotExists(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 offset, int3 size);
  __host__ void copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW, char const* label,
                                   int patchID, int matlIndx, int levelIndx, bool staging,
                                   int3 offset, int3 size);
  //HOST_DEVICE void* getPointer(char const* label, int patchID, int matlIndex);
  HOST_DEVICE void putContiguous(GPUGridVariableBase &var, char const* indexID, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost);
  HOST_DEVICE void allocate(const char* indexID, size_t size);

  //______________________________________________________________________
  // GPU DataWarehouse support methods
  
  //HOST_DEVICE bool exist(char const* label, int patchID, int matlIndx, int levelIndx);
  //HOST_DEVICE bool exist(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset, bool skipContiguous = false, bool onlyContiguous = false);
  HOST_DEVICE bool existContiguously(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset);
  HOST_DEVICE bool existsLevelDB( char const* name, int matlIndx, int levelIndx);       // levelDB
  HOST_DEVICE bool removeLevelDB( char const* name, int matlIndx, int levelIndx);
  HOST_DEVICE bool remove(char const* label, int patchID, int matlIndx, int levelIndx);

  HOST_DEVICE void syncto_device(void *cuda_stream);
  HOST_DEVICE void clear();
  HOST_DEVICE void deleteSelfOnDevice();
  HOST_DEVICE GPUDataWarehouse* getdevice_ptr(){return d_device_copy;};
  HOST_DEVICE void setDebug(bool s){d_debug=s;}
  //HOST_DEVICE cudaError_t copyDataHostToDevice(char const* indexID, void *cuda_stream);
  //HOST_DEVICE cudaError_t copyDataDeviceToHost(char const* indexID, void *cuda_stream);
  HOST_DEVICE void copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndx, int levelIndx);

  //______________________________________________________________________
  //Additional support methods
  HOST_DEVICE void putMaterials(std::vector< std::string > materials);
  HOST_DEVICE materialType getMaterial(int i) const;
  HOST_DEVICE int getNumMaterials() const;
  HOST_DEVICE void putGhostCell(char const* label, int sourcePatchID, int destPatchID, int matlIndx, int levelIndx,
                                bool sourceStaging, bool deststaging,
                                int3 varOffset, int3 varSize,
                                int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset);

  __host__ bool areAllStagingVarsValid(char const* label, int patchID, int matlIndx, int levelIndx);



  __host__ atomicDataStatus getStatus(atomicDataStatus& status);
  __host__ std::string getDisplayableStatusCodes(atomicDataStatus& status);
  __host__ bool testAndSetAllocating(atomicDataStatus& status);
  __host__ bool testAndSetAllocate(atomicDataStatus& status);
  __host__ bool checkAllocated(atomicDataStatus& status);
  __host__ bool checkValid(atomicDataStatus& status);
  //__host__ bool testAndSetCopying(atomicDataStatus& status);

  __host__ bool isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ bool isValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void setValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void setValidOnGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ bool dwEntryExistsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isValidOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx);
  __host__ void setValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);

  __host__ bool testAndSetAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool testAndSetCopyingIntoGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool testAndSetCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool testAndSetCopyingIntoGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ void setAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void setValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);



  //This and the function below go through the d_ghostCellData array and copies data into
  //the correct destination GPU var.  This would be the final step of a GPU ghost cell transfer.
  HOST_DEVICE void copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream);
  HOST_DEVICE void copyGpuGhostCellsToGpuVars();

  HOST_DEVICE bool ghostCellCopiesNeeded();
  HOST_DEVICE void getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells, char const* label, int patchID, int matlIndx, int levelIndx = 0);
  //HOST_DEVICE void getTempGhostCells(void * dtask, std::vector<tempGhostCellInfo>& temp);

  HOST_DEVICE void* getPlacementNewBuffer();
  
  __host__ void init_device(size_t objectSizeInBytes, unsigned int maxdVarDBItems );
  __host__ void init(int id, std::string internalName);
  __host__ void cleanup();


  __device__ void print();

private:

  HOST_DEVICE dataItem* getItem(char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void resetdVarDB();
 // HOST_DEVICE void copyGpuGhostCellsToGpuVars();


  HOST_DEVICE void printError(const char* msg, const char* methodName, char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void printError(const char* msg, const char* methodName) {
    printError(msg, methodName, "", 0, 0, 0);
  }
  HOST_DEVICE void printGetError( const char* msg, char const* label, int patchID, int matlIndx, int levelIndx);
  HOST_DEVICE void printGetLevelError(const char* msg, char const* label, int levelIndx, int matlIndx);

  __host__ void* allocateCudaSpaceFromPool(int device_id, size_t memSize);
  __host__ bool freeCudaSpaceFromPool(int device_id, size_t memSize, void* addr);

  std::map<labelPatchMatlLevel, allVarPointersInfo> *varPointers;

  //mutable Uintah::CrowdMonitor allocateLock;
  //mutable Uintah::CrowdMonitor varLock;
  mutable Uintah::CrowdMonitor *allocateLock;
  mutable Uintah::CrowdMonitor *varLock;
  static Uintah::CrowdMonitor *gpuPoolLock;
  char _internalName[80];

  materialItem       d_materialDB[MAX_MATERIALSDB_ITEMS];
  dataItem           d_levelDB[MAX_LEVELDB_ITEMS];
  int                d_numVarDBItems;
  int                d_numMaterials;
  int                numGhostCellCopiesNeeded;
  GPUDataWarehouse*  d_device_copy;             //The pointer to the copy of this object in the GPU.
  bool               d_dirty;                   //if this changes, we have to recopy the GPUDW.
  int                d_device_id;
  bool               d_debug;
  size_t             objectSizeInBytes;
  unsigned int       maxdVarDBItems;            //How many items we can add to d_varDB before we run out of capacity.
  void *             placementNewBuffer;        //For task DWs, we want to seraliaze and size this object as small as possible.
                                                //So we create a buffer, and keep track of the start of that buffer here.

  //These STL data structures being here do not pose a problem for the CUDA compiler

  static std::multimap<gpuMemoryPoolItem, gpuMemoryData> *gpuMemoryPool;


  std::map<std::string, contiguousArrayInfo> *contiguousArrays;

  //std::vector<tempGhostCellInfo> tempGhostCells; //For the host side.  Holds both ghost cell data and information about prepared ghosts cell.
                                                 //Data is loaded in here through prepareGPUDependencies, which checks ahead and
                                                 //loads all ghost cells into contiguous chunks ready for copying to the destination.
                                                 //See prepareGPUDependencies for more info.

  dataItem d_varDB[MAX_VARDB_ITEMS];              //For the device side.  The variable database. It's a very large buffer.
                                                  //Important note:
                                                  //We should never transfer a full GPUDataWarehouse object as is.  Instead
                                                  //we should use malloc and only use a section of it.  See here for more
                                                  //information:  http://www.open-std.org/Jtc1/sc22/wg14/www/docs/dr_051.html
                                                  //Doing it this way allows to only need one malloc instead of two.  And the
                                                  //object is serialized allowing for a single copy if needed instead of
                                                  //worrying about two copies (one for the GPU DW and one for the array).
                                                  //A thread first accumulates all items that will go into this array,
                                                  //gets a count and data size, and then only that amount of data is copied
                                                  //to the GPU's memory.
                                                  //***This must be the last data member of the class***
                                                  //This follows C++ 98 standards "Nonstatic data members of a (non-union) class
                                                  //declared without an intervening access-specifier are allocated so that later
                                                  //members have higher addresses within a class object. The order of allocation of
                                                  //nonstatic data members separated by an access-specifier is unspecified (11.1).
                                                  //Implementation alignment requirements might cause two adjacent members not to
                                                  //be allocated immediately after each other; so might requirements for space for
                                                  //managing virtual functions (10.3) and virtual base classes (10.1)."

};



} // end namespace Uintah

#endif // end #ifndef CCA_COMPONENTS_SCHEDULERS_GPUDATAWAREHOUSE_H
