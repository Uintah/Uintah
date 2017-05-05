/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#ifndef CCA_COMPONENTS_SCHEDULERS_DWVARIABLES_H
#define CCA_COMPONENTS_SCHEDULERS_DWVARIABLES_H

#include <cstdio> //For printf
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/Variable.h>
#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Grid/Grid.h>



/**************************************

CLASS
 DWVariables

 This data warehouse should serve to unify the current host-side data warehouse
 (currently described by DWDatabase.h and OnDemandDataWarehouse.h)
 and also the GPU-side data warehouse (currently described by GPUDataWarehouse.h).
 The two datawarehouses have many similarities, a few differences, and
 and both are missing some features we want in a data warehouse moving forward.

GENERAL INFORMATION

 UnifiedDataWarehouse.h

 Brad Peterson
 University of Utah

KEYWORDS
 DWVariables

DESCRIPTION
 The purpose of this data warehouse is to support all of the following:
 * Allow for variables to utilize Kokkos for data layout, avoiding Array3.h.
 * Allow for this data warehouse to track the status of variables in various
         memory locations (host memory, device memory, etc.)
 * To be fully lock free.  This will utilize a lock free map, instead of the old way which was simply
         a vector sized prior to the timestep executing.  The map will allow us, if ever needed, to
         be able to add variables to the database as the simulation is progressing, instead of needing
         to know all variables ahead of time.
 * To allow support for full concurrency of variables, so that we can determine if a variable
         is being allocated, copied, or gathering in ghost cells, and avoid having any such "action"
         performed twice in a race condition.
 * To support all the Uintah variable types (Grid Variables, PerPatch, Sole, Reduction, and Particles)
 * To have scheduler functions which prepare all variables prior to task execution.  The user should
         only have access to getWritable() and getReadonly().  The user should now no longer have access
         to a put() or allocateAndPut() method.  (TODO: Talk with Alan, this seems to assume that
         we would restrict which schedulers could be used.  Or we would have to make sure all schedulers
         use the same "preparation" functionality.)
 * To avoid "messier" things in the older data warehouse, such as scrub counters, reference counters,
         and using Handle to wrap various data types in PerPatches.    Another messy thing to avoid is
         the "inflating" of grid variables when ghost cells are being copied in.  Yet another problem was
         the prior data structure utilizing polymorphism far too often which greatly inhibited code tweaks
         and additional features.
 * d_combinedMemory = false was the standard for the GPU data warehouse.  This needs to support d_combined = true
         as the standard baseline
 * This data warehouse assumes one accelerator device per NUMA region.  It does not support multiple
         accelerator devices per NUMA region, mainly because Kokkos doesn't.
 * Wasatch still needs raw data pointers.  Do they also need d_combinedMemory = false?

 Later goals:
 * Do an MPI_Send immediately, instead of waiting for send_old_data task to run
 * Supporting host pinned memory and a host pinned memory pool to get full overlapping of memory copies and
         kernel execution.
 * Being smarter about ghost cells for a task, such as listing 'ghost cells stencil 6' or 'ghost cell stencil 26',
         so that corner or edge regions don't invoke costly MPI copies.
 * Particles should be smarter on their layout, so we could sort them using a Z space filling curve (Morton)
         a Hilbert space filling curve.
 * Scrub counters may be preserved when a strategy is developed to vacate variables from device memory if
         problems can fit in host memory but not in device memory.
 * Having knowledge of which memory spaces can communicate with which, and what bandwidth speeds
         (a host to device copy over a CUDA device may go through a slower PCIe bus but a faster device from
         host over NVLINK may go MUCH faster).
 * Possibly allow for MPI calls to pull and push data straight from GPUs (if that approach ends up being
         more efficient overall).
 * Do we allow for free form variables, such as the handful of arrays host and GPU arrays that RMCRT needs for the kernels?


WARNING

**************************************/

namespace Uintah {


class UnifiedDataWarehouse : public DataWarehouse {

public:

  virtual bool exists(const VarLabel*, int matlIndex, const Patch*) { printf("UnifiedDataWarehouse::exists() not implemented yet\n"); }
  virtual bool exists(const VarLabel*, int matlIndex, const Level*) { printf("UnifiedDataWarehouse::exists() not implemented yet\n"); }

  virtual ReductionVariableBase* getReductionVariable( const VarLabel*,
                   int matlIndex,
                   const Level* ) { printf("UnifiedDataWarehouse::getReductionVariable() not implemented yet\n"); return nullptr;}

  // Returns a (const) pointer to the grid.  This pointer can then be
  // used to (for example) get the number of levels in the grid.
  virtual const Grid * getGrid() { printf("UnifiedDataWarehouse::getGrid() not implemented yet\n"); return nullptr;}

  virtual void put(Variable*, const VarLabel*, int matlIndex, const Patch*) { printf("UnifiedDataWarehouse::put() will never be implemented for this class.  Use a get() for everything\n"); }

  // Reduction Variables
  virtual void get(ReductionVariableBase&, const VarLabel*,
       const Level* level = 0, int matlIndex = -1) = 0;
  virtual void put(const ReductionVariableBase&, const VarLabel*,
       const Level* level = 0, int matlIndex = -1)  { printf("UnifiedDataWarehouse::put() will never be implemented for this class.  Use a get() instead \n"); }

  virtual void override(const ReductionVariableBase&, const VarLabel*,
      const Level* level = 0, int matlIndex = -1) = 0;
  virtual void print(std::ostream& intout, const VarLabel* label,
         const Level* level, int matlIndex = -1) = 0;

  // Sole Variables
  virtual bool exists(const VarLabel*) const = 0;
  virtual void get(SoleVariableBase&, const VarLabel*,
       const Level* level = 0, int matlIndex = -1) = 0;
  virtual void put(const SoleVariableBase&, const VarLabel*,
       const Level* level = 0, int matlIndex = -1) { printf("UnifiedDataWarehouse::put() will never be implemented for this class.  Use a get() instead\n"); }

  virtual void override(const SoleVariableBase&, const VarLabel*,
      const Level* level = 0, int matlIndex = -1) = 0;

  virtual void doReserve() = 0;

  // Particle Variables
  // changed way PS's were stored from ghost info to low-high range.
  // we can still keep the getPS function API the same though to not annoy
  // everybody -- BJW, May 05
  virtual ParticleSubset* createParticleSubset(  particleIndex numParticles,
                                                 int matlIndex, const Patch*,
                                                 IntVector low = IntVector(0,0,0),
                                                 IntVector high = IntVector(0,0,0) ) = 0;
  virtual void saveParticleSubset(ParticleSubset* psubset,
                                  int matlIndex, const Patch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0)) = 0;
  virtual bool haveParticleSubset(int matlIndex, const Patch*,
                                  IntVector low = IntVector(0,0,0),
                                  IntVector high = IntVector(0,0,0), bool exact = false) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*,
                                            IntVector low, IntVector high) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*) = 0;
  virtual ParticleSubset* getDeleteSubset(int matlIndex, const Patch*) = 0;
  virtual std::map<const VarLabel*, ParticleVariableBase*>* getNewParticleState(int matlIndex, const Patch*) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, const Patch*,
              Ghost::GhostType,
              int numGhostCells,
              const VarLabel* posvar) = 0;
  virtual ParticleSubset* getParticleSubset(int matlIndex, IntVector low, IntVector high,
                                            const Patch* relPatch,
                                            const VarLabel* posvar, const Level* level=0) = 0;
  virtual void allocateTemporary(ParticleVariableBase&,
         ParticleSubset*) = 0;
  virtual void allocateAndPut(ParticleVariableBase&, const VarLabel*,
            ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
       ParticleSubset*) = 0;
  virtual void get(constParticleVariableBase&, const VarLabel*,
       int matlIndex, const Patch* patch) = 0;
  virtual void getModifiable(ParticleVariableBase&, const VarLabel*,
           ParticleSubset*) = 0;
  virtual void put(ParticleVariableBase&, const VarLabel*,
       bool replace = false) = 0;


  virtual void getCopy(ParticleVariableBase&, const VarLabel*, ParticleSubset*) = 0;
  virtual void copyOut(ParticleVariableBase&, const VarLabel*, ParticleSubset*) = 0;

  virtual void print() = 0;
  virtual void clear() = 0;


  virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
                ParticleSubset*) = 0;
  virtual ParticleVariableBase*
  getParticleVariable(const VarLabel*, int matlIndex, const Patch*) = 0;

  // Generic grid based variables

  virtual void get( constGridVariableBase& var,
                    const VarLabel* label, int matlIndex, const Patch* patch,
                    Ghost::GhostType gtype, int numGhostCells ) = 0;

  virtual void getModifiable( GridVariableBase& var,
                              const VarLabel* label, int matlIndex, const Patch* patch, Ghost::GhostType gtype=Ghost::None, int numGhostCells=0 ) = 0;

  virtual void allocateTemporary( GridVariableBase& var, const Patch* patch,
                                  Ghost::GhostType gtype = Ghost::None, int numGhostCells = 0 ) = 0;
//                                  const IntVector& boundaryLayer ) = 0;
//                                const IntVector& boundaryLayer = IntVector(0,0,0)) = 0;

  virtual void allocateAndPut( GridVariableBase& var,
                               const VarLabel* label, int matlIndex,
                               const Patch* patch, Ghost::GhostType gtype = Ghost::None,
                               int numGhostCells = 0 ) = 0;

  virtual void put(GridVariableBase& var, const VarLabel* label, int matlIndex, const Patch* patch,
            bool replace = false) = 0;

  // returns the constGridVariable for all patches on the level
  virtual void getLevel( constGridVariableBase&,
                         const VarLabel*,
                         int matlIndex,
                         const Level* level) = 0;

  virtual void getRegion(constGridVariableBase&, const VarLabel*,
                          int matlIndex, const Level* level,
                          const IntVector& low, const IntVector& high,
                          bool useBoundaryCells = true) = 0;

  // Copy out of the warehouse into an allocated variable.
  virtual void copyOut(GridVariableBase& var, const VarLabel* label, int matlIndex,
         const Patch* patch, Ghost::GhostType gtype = Ghost::None,
         int numGhostCells = 0) = 0;

  // Makes var a copy of the specified warehouse data, allocating it
  // to the appropriate size first.
  virtual void getCopy(GridVariableBase& var, const VarLabel* label, int matlIndex,
         const Patch* patch, Ghost::GhostType gtype = Ghost::None,
         int numGhostCells = 0) = 0;


  // PerPatch Variables
  virtual void get(PerPatchBase&, const VarLabel*,
       int matlIndex, const Patch*) = 0;
  virtual void put(PerPatchBase&, const VarLabel*,
       int matlIndex, const Patch*, bool replace = false) = 0;

  // this is so we can get reduction information for regridding
  virtual void getVarLabelMatlLevelTriples(std::vector<VarLabelMatl<Level> >& vars ) const = 0;

  // Remove particles that are no longer relevant
  virtual void deleteParticles(ParticleSubset* delset) = 0;

  // Add particles
  virtual void addParticles(const Patch* patch, int matlIndex,
          std::map<const VarLabel*, ParticleVariableBase*>* addedstate) = 0;

  // Move stuff to a different data Warehouse
  virtual void transferFrom(DataWarehouse*, const VarLabel*,
          const PatchSubset*, const MaterialSubset*) = 0;

  virtual void transferFrom(DataWarehouse*, const VarLabel*,
          const PatchSubset*, const MaterialSubset*,
                            bool replace) = 0;

  virtual void transferFrom(DataWarehouse*, const VarLabel*,
          const PatchSubset*, const MaterialSubset*,
                            bool replace, const PatchSubset*) = 0;

  //An overloaded version of transferFrom.  GPU transfers need a stream, and a
  //stream is found in a detailedTask object.
  virtual void transferFrom(DataWarehouse*, const VarLabel*,
                            const PatchSubset*, const MaterialSubset*, void * detailedTask,
                            bool replace, const PatchSubset*) = 0;

  virtual size_t emit(OutputContext&, const VarLabel* label,
        int matlIndex, const Patch* patch) = 0;

#if HAVE_PIDX
  virtual void emitPIDX(PIDXOutputContext&,
                        const VarLabel* label,
                        int matlIndex,
                        const Patch* patch,
                        unsigned char* buffer,
                        size_t bufferSize) = 0;
#endif


  // Scrubbing
  enum ScrubMode {
    ScrubNone,
    ScrubComplete,
    ScrubNonPermanent
  };
  virtual ScrubMode setScrubbing(ScrubMode) = 0;

      // For related datawarehouses
  virtual DataWarehouse* getOtherDataWarehouse(Task::WhichDW) = 0;

  // For the schedulers
  virtual bool isFinalized() const = 0;
  virtual void finalize() = 0;
  virtual void unfinalize() = 0;
  virtual void refinalize() = 0;

  // Returns the generation number (id) of this data warehouse.  Id's
  // start at 0. Each subsequent DW's id is one greater.  SetID should
  // only be called by the SimulationController (and only once) if the
  // DW is being used for a restarted simulation.  This allows the DW
  // generation number to be kept in sync with the actual number of
  // timesteps for the restarted simulation.
  int  getID() const { return d_generation; }
  void setID( int id ) { d_generation = id; }

  // For timestep abort/restart
  virtual bool timestepAborted() = 0;
  virtual bool timestepRestarted() = 0;
  virtual void abortTimestep() = 0;
  virtual void restartTimestep() = 0;

  virtual void reduceMPI(const VarLabel* label, const Level* level,
    const MaterialSubset* matls, int nComm) = 0;

  #ifdef HAVE_CUDA
    GPUDataWarehouse* getGPUDW(int i) const { return d_gpuDWs[i]; }
    GPUDataWarehouse* getGPUDW() const {
      int i;
      CUDA_RT_SAFE_CALL(cudaGetDevice(&i));
      return d_gpuDWs[i];
    }
  #endif
  protected:
    DataWarehouse( const ProcessorGroup* myworld,
       Scheduler* scheduler,
       int generation );
    // These two things should be removed from here if possible - Steve
    const ProcessorGroup* d_myworld;
    Scheduler* d_scheduler;

    // Generation should be const, but is not as during a restart, the
    // generation number of the first DW is updated from 0 (the default
    // for the first DW) to the correct generation number based on how
    // many previous time steps had taken place before the restart.
    int d_generation;

  #ifdef HAVE_CUDA
    std::vector<GPUDataWarehouse*> d_gpuDWs;
  #endif
  private:
    DataWarehouse(const DataWarehouse&);
    DataWarehouse& operator=(const DataWarehouse&);
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
                //UNKNOWN                   = 0x80000040}; //TODO: REMOVE THIS WHEN YOU CAN, IT'S NOT OPTIMAL DESIGN.
                //COPYING_OUT = The remaining bits. See below.



  typedef atomic_ullong AtomicDataStatus;  //See documentation below.

    //This 64 bit value is hard coded for up to 5 memory locations.  It is assumed that no Uintah
    //simulation will ever require the schedule to manage a variable in over 5 memory locations
    //simultaneously.  In fact, I would be surprised to ever see it go past 2 (host memory and device memory).

    //    0                   1                   2                   3
    //    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //   | | | | | | | | | | | | | | | | | | | | | | | | | |             |
    //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //   |_________|_________|_________|_________|_________|
    //        |         |         |         |         |
    //        |         |         |         |         |
    //       from      from      from      from      from
    //      memory    memory    memory    memory    memory
    //      loc. 0    loc. 1    loc. 2    loc. 3    loc. 4

    // Copying out bits are bits 0 through 24.
    // Examples:
    // If data is being copied out of memory location 0 into memory location 1, then bit 1 is set.
    // If data is being copied out of memory location 2 into memory location 0, then bit 10 is set.
    // If data is being copied out of memory location 3 into memory location 4, then bit 19 is set.
    // If data is being copied out of memory location 4 into memory location 2, then bit 22 is set.

    //    3               4                   5                   6
    //    2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
    //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //   |   | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //       |___________|___________|___________|___________|___________|
    //             |           |           |           |           |
    //             |           |           |           |           |
    //           memory      memory      memory      memory      memory
    //         location 4  location 3  location 2  location 1  location 0

    // Allocating memory location 0                = bit 63 - 0x0000000000000001
    // Allocated memory location 0                 = bit 62 - 0x0000000000000002
    // Copying in memory location 0                = bit 61 - 0x0000000000000004
    // Valid in memory location 0                  = bit 60 - 0x0000000000000008
    // Awaiting ghost data in memory location 0    = bit 59 - 0x0000000000000010
    // Valid with ghost cells in memory location 0 = bit 58 - 0x0000000000000020
    // Variable not allocated/Invalid for memory location 0 if bits 58 through 63 are all 0
    // Allocating memory location 1                = bit 57 - 0x0000000000000040
    // Allocated memory location 1                 = bit 56 - 0x0000000000000080
    // Copying in memory location 1                = bit 55 - 0x0000000000000100
    // Valid in memory location 1                  = bit 54 - 0x0000000000000200
    // Awaiting ghost data in memory location 1    = bit 53 - 0x0000000000000400
    // Valid with ghost cells in memory location 1 = bit 52 - 0x0000000000000800
    // Variable not allocated/Invalid for memory location 0 if bits 52 through 57 are all 0
    // And so on through memory location 2, 3, and 4.  Ending at bit 34.

    // With this approach we can allow for multiple copy outs, but only one copy in.

    // We should never attempt to copy into unless that memory location's allocated bit is set.
    // We should never copy out unless that memory location's valid bit is set.

  //labelPatchMatl is the 3 tuple key.  Every variable should have a label.  We need to treat the label as an integer.
  //They all could have a material, if no material exists, they can start at -1 or 0.  That means this is an integer.
  //The third is more interesting.  The third item of the 3-tuple could be a patch or a level or
  //something else (such as a bucket in an unstructured mesh).  In order to differentiate which is which
  //the first two bits of that ID can be used to indicate what it is (00 for patch, 01 for level, 10 for bucket).
  //If it's a patch, within that, the next two bits could be used to represent what it
  //is (00 for normal patch, 01 for periodic boundary condition, 10 for something else, etc.)  (In the old
  //system negative patches were designated to be periodic boundary condition patches...this should go away).
  struct labelPatchMatl {
    std::string label;   //TODO: replace with an int index?  Or give it an int index private data member?
    int         patchID;  //TODO: Give an internal int localID.
    int         matlIndx;
    labelPatchMatl(const char * label, int patchID, int matlIndx) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
    }
    //This so it can be used in an STL map
    bool operator<(const labelPatchMatlLevel& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx < right.matlIndx)) {
        return true;
      } else {
        return false;
      }
    }
  };

  struct var{
    AtomicDataStatus atomicDataStatus {0x0000000000000000};
  };

  struct varItem {
      //allVarPointersInfo() {
      //  __sync_fetch_and_and(&atomicStatusInHostMemory, UNALLOCATED);
      //  __sync_fetch_and_and(&atomicStatusInGpuMemory, UNALLOCATED);
      //  varDB_index = -1;
      //}

     void*           device_ptr;               // Where it is on the device
      int3            device_offset;
      int3            device_size;
      unsigned int    sizeOfDataType;

      GhostType       gtype;
      unsigned int    numGhostCells;

      int             varDB_index {-1};     // Where this also shows up in the varDB.  We can use this to get the rest of the information we need.


      std::map<stagingVar, stagingVarInfo> stagingVars;  //TODO, change to a list.  The host foreign vars is better than my approach I think.
                                                         // When ghost cells in the GPU need to go to another memory space
                                                         // we will be creating temporary contiguous arrays to hold that
                                                         // information.  After many iterations of other attempts, it seems
                                                         // creating a map of staging vars is the cleanest way to go for data
                                                         // in GPU memory.
    };


private:
  static const int ARR_SIZE = 100000;                    //This is sized from the worst case scenario of 16 cores per MPI rank, 10 patches per core,
                                                         //100 labels, and 5 materials (estimated by Todd Harman and Jeremy Thornock).
                                                         //Some simulations may have more labels (Arches devs have said they may do 300 labels per patch)
                                                         //Or more patches (RMCRT may have more patches showing up in a middle layer)
                                                         //But overall nobody at the moment seems to need more than 100000 items per MPI rank.
                                                         //If somebody does need more, then just increase this number.

  varItem* dataWarehouse[ARR_SIZE];                      //varItems need to be a pointer to the varItem, so we can easily do atomic swaps to
                                                         //get objects into the array.
                                                         //All gridIDs, materials, and labels need internal/local IDs that count starting at 0.
                                                         //suppose a simulation has 100 labels, 5 materials, and 300 gridIDs available.  If you
                                                         //wanted patch/grid ID #173, material #2, label #73, you could do
                                                         //173*(5*100) + (2*5) + 73 = 86583, which means you would access index 86583 of the array
                                                         //to retrieve all information for that variable.

                                                          //TODO: How will a varItem store a GridVar,
                                                         //a PerPatch, a SoleVar, a ReductionVar, and a ParticleVar?
};

#endif //#define CCA_COMPONENTS_SCHEDULERS_DWVARIABLES_H
