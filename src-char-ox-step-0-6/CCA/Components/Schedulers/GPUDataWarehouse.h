/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Parallel/MasterLock.h>

#include <map> //for host code only.
#include <string>
#include <vector>
#include <memory> //for the shared_ptr code

#define MAX_VARDB_ITEMS       10000000  //Due to how it's allocated, it will never use up this much space.
                                        //Only a very small fraction of it.
#define MAX_MATERIALSDB_ITEMS 20
#define MAX_LEVELDB_ITEMS     20        //TODO: Is this needed?
#define MAX_NAME_LENGTH       32        //How big a particular label can be.

#define LEVEL_PATCH_ID = -99999999;     //A sentinel value used for a patch ID when we're storing a
                                        //region of patches for a level instead of a regular patch.
namespace Uintah {

enum materialType {
    IDEAL_GAS = 0
//  , HARD_SPHERE_GAS_EOS = 1
//  , TST_EOS = 2
//  , JWL_EOS = 3
//  , JWLC_EOS = 4
//  , MURNAGHAN_EOS = 5
//  , BIRCH_MURNAGHAN_EOS = 6
//  , GRUNEISEN_EOS = 7
//  , TILLOTSON_EOS = 8
//  , THOMSEN_HARTKA_WATER_EOS = 9
//  , KNAUSS_SEA_WATER_EOS = 10
//  , KUMARI_DASS_EOS = 11
};

class OnDemandDataWarehouse;

class GPUDataWarehouse {

public:


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
      allocatedDeviceMemory = nullptr;
      allocatedHostMemory = nullptr;
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


  //This status is for concurrency.  This enum largely follows a model of "action -> state".  
  //For example, allocating -> allocated.  The idea is that only one thread should be able 
  //to claim moving into an action, and that winner should be responsible for setting it into the state.  
  //When it hits the state, other threads can utilize the variable.  
  enum status { UNALLOCATED               = 0x00000000,
                ALLOCATING                = 0x00000001,
                ALLOCATED                 = 0x00000002,
                COPYING_IN                = 0x00000004,
                VALID                     = 0x00000008,     //For when a variable has its data, this excludes any knowledge of ghost cells.
                AWAITING_GHOST_COPY       = 0x00000010,     //For when when we know a variable is awaiting ghost cell data
                                                            //It is possible for VALID bit set to 0 or 1 with this bit set,
                                                            //meaning we can know a variable is awaiting ghost copies but we
                                                            //don't know from this bit alone if the variable is valid yet.
                VALID_WITH_GHOSTS         = 0x00000020,     //For when a variable has its data and it has its ghost cells
                                                            //Note: Change to just GHOST_VALID?  Meaning ghost cells could be valid but the
                                                            //non ghost part is unknown?
                DEALLOCATING              = 0x00000040,     //TODO: REMOVE THIS WHEN YOU CAN, IT'S NOT OPTIMAL DESIGN.
                FORMING_SUPERPATCH        = 0x00000080,     //As the name suggests, when a number of individual patches are being formed 
                                                            //into a superpatch, there is a period of time which other threads
                                                            //should wait until all patches have been processed.  
                SUPERPATCH                = 0x00000100,     //Indicates this patch is allocated as part of a superpatch.
                                                            //At the moment superpatches is only implemented for entire domain
                                                            //levels.  But it seems to make the most sense to have another set of
                                                            //logic in level.cc which subdivides a level into superpatches.
                                                            //If this bit is set, you should find the lowest numbered patch ID
                                                            //first and start with concurrency reads/writes there.  (Doing this
                                                            //avoids the Dining Philosopher's problem.
                UNKNOWN                   = 0x00000200};    //Remove this when you can, unknown can be dangerous.
                                                            //It's only here to help track some host variables


                //LEFT_SIXTEEN_BITS                         //Use the other 16 bits as a usage counter.
                                                            //If it is zero we could deallocate.




  typedef int atomicDataStatus;

  //    0                   1                   2                   3
  //    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //   |    16-bit reference counter   |  unsued   | | | | | | | | | | |
  //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  //left sixteen bits is a 16-bit integer reference counter.

  //Not allocated/Invalid = If the value is 0x00000000

  //Allocating                = bit 31 - 0x00000001
  //Allocated                 = bit 30 - 0x00000002
  //Copying in                = bit 29 - 0x00000004
  //Valid                     = bit 28 - 0x00000008
  //awaiting ghost data       = bit 27 - 0x00000010
  //Valid with ghost cells    = bit 26 - 0x00000020
  //Deallocating              = bit 25 - 0x00000040
  //Superpatch                = bit 24 - 0x00000080
  //Unknown                = bit 23 - 0x00000080

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
    void*           device_ptr {nullptr};   //Where it is on the device
    void*           host_contiguousArrayPtr {nullptr};  //TODO, remove this.  It's an old idea that didn't pan out.
    int             varDB_index;
    atomicDataStatus      atomicStatusInHostMemory;
    atomicDataStatus      atomicStatusInGpuMemory;

  };

  //Only raw information about the data itself should go here.  Things that should be shared
  //if two patches are in the same superpatch sharing the same data pointer.  (For example,
  //both could have numGhostCells = 1, it wouldn't make sense for one to have numGhostCells = 1 and another = 2
  //if they're sharing the same data pointer.
  class varInfo {
  public:
    varInfo() {
      atomicStatusInHostMemory = UNALLOCATED;
      atomicStatusInGpuMemory = UNALLOCATED;
      gtype = GhostType::None;
    }
    void*           device_ptr {nullptr};   //Where it is on the device
    void*           host_contiguousArrayPtr {nullptr};  //TODO, remove this.  It's an old idea that didn't pan out.
    int3            device_offset {0,0,0};  //TODO, split this into a device_low and a device_offset.  Device_low goes here
                                            //but device_offset should NOT go here (multiple patches may share a dataInfo, but
                                            //they should have distinct offsets
    int3            device_size {0,0,0};
    unsigned int    sizeOfDataType {0};
    GhostType       gtype;
    unsigned int    numGhostCells {0};
    atomicDataStatus   atomicStatusInHostMemory;  //Shared_ptr because patches in a superpatch share the pointer.
    atomicDataStatus   atomicStatusInGpuMemory;   //TODO, merge into the one above it.

    std::map<stagingVar, stagingVarInfo> stagingVars;  //When ghost cells in the GPU need to go to another memory space
                                                       //we will be creating temporary contiguous arrays to hold that
                                                       //information.  After many iterations of other attempts, it seems
                                                       //creating a map of staging vars is the cleanest way to go for data
                                                       //in GPU memory.
  };

  class allVarPointersInfo {
  public:
    allVarPointersInfo() { var = std::make_shared<varInfo>(); }
    std::shared_ptr<varInfo>  var;
    int3                      device_offset {0,0,0};
    int                       varDB_index {-1};        //Where this also shows up in the varDB.  We can use this to
                                                      //get the rest of the information we need.
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


  //______________________________________________________________________
  // GPU GridVariable methods
  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void get(const GPUGridVariableBase& var, char const* label, const int patchID, const int8_t matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }
  HOST_DEVICE void getStagingVar(const GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  HOST_DEVICE bool stagingVarExists(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);


  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void get(const GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void get(const GPUPerPatchBase& var, char const* label, const int patchID, const int8_t matlIndx) {
    get(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getLevel(const GPUGridVariableBase& var, char const* label, const int8_t matlIndx, const int8_t levelIndx);


  HOST_DEVICE void getModifiable(GPUGridVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void getModifiable( GPUGridVariableBase&  var, char const* label, const int patchID, const int8_t matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getModifiable(GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void getModifiable( GPUReductionVariableBase& var, char const* label, const int patchID, const int8_t matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  HOST_DEVICE void getModifiable(GPUPerPatchBase& var, char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void getModifiable( GPUPerPatchBase&  var, char const* label, const int patchID, const int8_t matlIndx) {
    getModifiable(var, label, patchID, matlIndx, 0);
  }

  __host__ void put(GPUGridVariableBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, bool staging = false, GhostType gtype = None, int numGhostCells = 0, void* hostPtr = nullptr);
  __host__ void put(GPUReductionVariableBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, void* hostPtr = nullptr);
  __host__ void put(GPUPerPatchBase& var, size_t sizeOfDataType, char const* label, int patchID, int matlIndx, int levelIndx = 0, void* hostPtr = nullptr);
  __host__ void copySuperPatchInfo(char const* label, int superPatchBaseID, int superPatchDestinationID, int matlIndx, int levelIndx);

  __host__ void allocateAndPut(GPUGridVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GhostType gtype = None, int numGhostCells = 0);
  __host__ void allocateAndPut(GPUReductionVariableBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType);
  __host__ void allocateAndPut(GPUPerPatchBase& var, char const* label, int patchID, int matlIndx, int levelIndx, size_t sizeOfDataType);

  __host__ void putUnallocatedIfNotExists(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 offset, int3 size);
  __host__ void copyItemIntoTaskDW(GPUDataWarehouse *hostSideGPUDW, char const* label,
                                   int patchID, int matlIndx, int levelIndx, bool staging,
                                   int3 offset, int3 size);

  __host__ void putContiguous(GPUGridVariableBase &var, char const* indexID, char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 low, int3 high, size_t sizeOfDataType, GridVariableBase* gridVar, bool stageOnHost);
  __host__ void allocate(const char* indexID, size_t size);

  //______________________________________________________________________
  // GPU DataWarehouse support methods
  //HOST_DEVICE bool existContiguously(char const* label, int patchID, int matlIndx, int levelIndx, bool staging, int3 host_size, int3 host_offset);
  //HOST_DEVICE bool existsLevelDB( char const* name, int matlIndx, int levelIndx);       // levelDB
  //HOST_DEVICE bool removeLevelDB( char const* name, int matlIndx, int levelIndx);
  __host__ bool remove(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void* getPlacementNewBuffer();
  __host__ void syncto_device(void *cuda_stream);
  __host__ void clear();
  __host__ void deleteSelfOnDevice();
  __host__ GPUDataWarehouse* getdevice_ptr(){return d_device_copy;};
  __host__ void setDebug(bool s){d_debug=s;}
  __host__ void copyHostContiguousToHost(GPUGridVariableBase& device_var, GridVariableBase* host_var, char const* label, int patchID, int matlIndx, int levelIndx);


  //______________________________________________________________________
  //Additional support methods
  __host__ void putMaterials(std::vector< std::string > materials);
  HOST_DEVICE materialType getMaterial(int i) const;
  HOST_DEVICE int getNumMaterials() const;
  __host__ void putGhostCell(char const* label, int sourcePatchID, int destPatchID, int matlIndx, int levelIndx,
                                bool sourceStaging, bool deststaging,
                                int3 varOffset, int3 varSize,
                                int3 sharedLowCoordinates, int3 sharedHighCoordinates, int3 virtualOffset);


  __host__ bool transferFrom(cudaStream_t* stream, GPUGridVariableBase &var_source, GPUGridVariableBase &var_dest, GPUDataWarehouse * from, char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool areAllStagingVarsValid(char const* label, int patchID, int matlIndx, int levelIndx);


  //__host__ atomicDataStatus getStatus(std::shared_ptr<atomicDataStatus>& status);
  __host__ std::string getDisplayableStatusCodes(atomicDataStatus& status);
  __host__ void getStatusFlagsForVariableOnGPU(bool& correctSize, bool& allocating, bool& allocated, bool& copyingIn,
                                               bool& validOnGPU, bool& gatheringGhostCells, bool& validWithGhostCellsOnGPU,
                                               bool& deallocating, bool& formingSuperPatch, bool& superPatch,
                                               char const* label, const int patchID, const int matlIndx, const int levelIndx,
                                               const int3& offset, const int3& size);

  __host__ bool compareAndSwapAllocating(atomicDataStatus& status);
  __host__ bool compareAndSwapAllocate(atomicDataStatus& status);
  __host__ bool checkAllocated(atomicDataStatus& status);
  __host__ bool checkValid(atomicDataStatus& status);

  __host__ bool isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isAllocatedOnGPU(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ bool isValidOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool compareAndSwapSetValidOnGPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx);
  __host__ bool compareAndSwapSetValidOnGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ bool dwEntryExistsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isValidOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx);
  __host__ bool compareAndSwapSetValidOnCPU(char const* const label, int patchID, int matlIndx, int levelIndx);

  __host__ bool compareAndSwapAwaitingGhostDataOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool compareAndSwapCopyingIntoGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool compareAndSwapCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool compareAndSwapCopyingIntoGPUStaging(char const* label, int patchID, int matlIndx, int levelIndx, int3 offset, int3 size);
  __host__ bool isValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void setValidWithGhostsOnGPU(char const* label, int patchID, int matlIndx, int levelIndx);

  __host__ bool compareAndSwapFormASuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool compareAndSwapSetSuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ bool isSuperPatchGPU(char const* label, int patchID, int matlIndx, int levelIndx);
  __host__ void setSuperPatchLowAndSize(char const* const label, const int patchID, const int matlIndx, const int levelIndx,
                                                        const int3& low, const int3& size);
  __host__ bool compareAndSwapDeallocating(atomicDataStatus& status);
  __host__ bool compareAndSwapDeallocate(atomicDataStatus& status);


  //This and the function below go through the d_ghostCellData array and copies data into
  //the correct destination GPU var.  This would be the final step of a GPU ghost cell transfer.
  __host__ void copyGpuGhostCellsToGpuVarsInvoker(cudaStream_t* stream);
  __device__ void copyGpuGhostCellsToGpuVars();
  HOST_DEVICE bool ghostCellCopiesNeeded();
  __host__ void getSizes(int3& low, int3& high, int3& siz, GhostType& gtype, int& numGhostCells, char const* label, int patchID, int matlIndx, int levelIndx = 0);
  

  __host__ void init_device(size_t objectSizeInBytes, unsigned int maxdVarDBItems );
  __host__ void init(int id, std::string internalName);
  __host__ void cleanup();

  __device__ void print();

private:

  __device__ dataItem* getItem(char const* label, const int patchID, const int8_t matlIndx, const int8_t levelIndx);
  HOST_DEVICE void resetdVarDB();


  HOST_DEVICE void printError(const char* msg, const char* methodName, char const* label, int patchID, int8_t matlIndx, int8_t levelIndx);
  HOST_DEVICE void printError(const char* msg, const char* methodName) {
    printError(msg, methodName, "", 0, 0, 0);
  }
  HOST_DEVICE void printGetError( const char* msg, char const* label, int8_t matlIndx, const int patchID, int8_t levelIndx);
  HOST_DEVICE void printGetLevelError(const char* msg, char const* label, int8_t levelIndx, int8_t matlIndx);


  std::map<labelPatchMatlLevel, allVarPointersInfo> *varPointers;


  Uintah::MasterLock * allocateLock;
  Uintah::MasterLock * varLock;

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
  unsigned int       d_maxdVarDBItems;          //How many items we can add to d_varDB before we run out of capacity.
  void *             placementNewBuffer;        //For task DWs, we want to seraliaze and size this object as small as possible.
                                                //So we create a buffer, and keep track of the start of that buffer here.

  //These STL data structures being here do not pose a problem for the CUDA compiler

  std::map<std::string, contiguousArrayInfo> *contiguousArrays;

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
