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

#ifndef UINTAH_COMPONENTS_SCHEDULERS_UNSTRUCTURED_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_UNSTRUCTURED_ONDEMANDDATAWAREHOUSE_H

#include <CCA/Components/Schedulers/UnstructuredOnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/UnstructuredDWDatabase.h>
#include <CCA/Components/Schedulers/UnstructuredSendState.h>
#include <CCA/Ports/UnstructuredDataWarehouse.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/UnstructuredGrid.h>
#include <Core/Grid/Variables/UnstructuredPSPatchMatlGhost.h>
#include <Core/Grid/Variables/UnstructuredVarLabelMatl.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/UintahMPI.h>

#include <iosfwd>
#include <map>
#include <vector>

using Uintah::Max;
using Uintah::FastHashTable;

namespace Uintah {

inline const UnstructuredPatch* getRealDomain(const UnstructuredPatch* patch)
{
  return patch->getRealUnstructuredPatch();
}

inline const UnstructuredLevel* getRealDomain(const UnstructuredLevel* level)
{
  return level;
}

class BufferInfo;
class UnstructuredDependencyBatch;
class UnstructuredDetailedDep;
class UnstructuredDetailedTasks;
class UnstructuredLoadBalancer;
class UnstructuredPatch;
class ProcessorGroup;
class UnstructuredSendState;
class UnstructuredTypeDescription;
class UnstructuredReductionVariableBase;
class UnstructuredSoleVariableBase;
class UnstructuredPerPatchBase;

/**************************************

 CLASS
 UnstructuredOnDemandDataWarehouse

 Short description...

 GENERAL INFORMATION

 UnstructuredOnDemandDataWarehouse.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
 UnstructuredOn_Demand_Data_Warehouse

 DESCRIPTION
 Long description...

 WARNING

 ****************************************/

class UnstructuredOnDemandDataWarehouse : public UnstructuredDataWarehouse {

  public:
    UnstructuredOnDemandDataWarehouse( const ProcessorGroup * myworld,
                                 UnstructuredScheduler      * scheduler,
                           const int              generation,
                           const UnstructuredGridP          & grid,
                           const bool             isInitializationDW = false );

    virtual ~UnstructuredOnDemandDataWarehouse();

    virtual bool exists(const UnstructuredVarLabel*,
                        int matIndex,
                        const UnstructuredPatch*) const;

    virtual bool exists(const UnstructuredVarLabel*,
                        int matIndex,
                        const UnstructuredLevel*) const;

    virtual UnstructuredReductionVariableBase* getReductionVariable( const UnstructuredVarLabel* label,
                                                         int             matlIndex,
                                                         const UnstructuredLevel*    level ) const;

    void copyKeyDB(UnstructuredKeyDatabase<UnstructuredPatch>& varkeyDB,
                   UnstructuredKeyDatabase<UnstructuredLevel>& levekeyDB);

    virtual void doReserve();

    // Returns a (const) pointer to the grid.  This pointer can then be
    // used to (for example) get the number of levels in the grid.
    virtual const UnstructuredGrid * getGrid()
    {
      return d_grid.get_rep();
    }

    // Generic put and allocate, passing Variable as a pointer rather than
    // by reference to avoid ambiguity with other put overloaded methods.
    virtual void put(UnstructuredVariable*,
                     const UnstructuredVarLabel*,
                     int matlIndex,
                     const UnstructuredPatch*);

    // Reduction Variables
    virtual void get(UnstructuredReductionVariableBase&,
                     const UnstructuredVarLabel*,
                     const UnstructuredLevel* level = 0,
                     int matIndex = -1);

    virtual void put(const UnstructuredReductionVariableBase&,
                     const UnstructuredVarLabel*,
                     const UnstructuredLevel* level = 0,
                     int matIndex = -1);

    virtual void override(const UnstructuredReductionVariableBase&,
                          const UnstructuredVarLabel*,
                          const UnstructuredLevel* level = 0,
                          int matIndex = -1);

    virtual void print(std::ostream& intout,
                       const UnstructuredVarLabel* label,
                       const UnstructuredLevel* level,
                       int matlIndex = -1);

    // UnstructuredSole Variables
    virtual bool exists(const UnstructuredVarLabel*) const;

    virtual void get(UnstructuredSoleVariableBase&,
                     const UnstructuredVarLabel*,
                     const UnstructuredLevel* level = 0,
                     int matIndex = -1);

    virtual void put(const UnstructuredSoleVariableBase&,
                     const UnstructuredVarLabel*,
                     const UnstructuredLevel* level = 0,
                     int matIndex = -1);

    virtual void override(const UnstructuredSoleVariableBase&,
                          const UnstructuredVarLabel*,
                          const UnstructuredLevel* level = 0,
                          int matIndex = -1);

    //__________________________________
    // Particle Variables

    virtual UnstructuredParticleSubset* createParticleSubset(particleIndex numParticles,
                                                 int matlIndex,
                                                 const UnstructuredPatch*,
                                                 IntVector low = IntVector(0, 0, 0),
                                                 IntVector high = IntVector(0, 0, 0));

    virtual void saveParticleSubset(UnstructuredParticleSubset*,
                                    int matlIndex,
                                    const UnstructuredPatch*,
                                    IntVector low = IntVector(0, 0, 0),
                                    IntVector high = IntVector(0, 0, 0));

    virtual bool haveParticleSubset(int matlIndex,
                                    const UnstructuredPatch*,
                                    IntVector low = IntVector(0, 0, 0),
                                    IntVector high = IntVector(0, 0, 0),
                                    bool exact = false);

    virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex,
                                              const UnstructuredPatch*,
                                              IntVector low,
                                              IntVector high);

    virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex,
                                              const UnstructuredPatch*,
                                              IntVector low,
                                              IntVector high,
                                              const UnstructuredVarLabel*);

    virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex,
                                              const UnstructuredPatch*);

    virtual UnstructuredParticleSubset* getDeleteSubset(int matlIndex,
                                            const UnstructuredPatch*);

    virtual std::map<const UnstructuredVarLabel*, UnstructuredParticleVariableBase*>* getNewParticleState(int matlIndex,
                                                                                  const UnstructuredPatch*);

    virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex,
                                              const UnstructuredPatch*,
                                              Ghost::GhostType,
                                              int numGhostCells,
                                              const UnstructuredVarLabel* posvar);

    //returns the particle subset in the range of low->high
    //relPatch is used as the key and should be the patch you are querying from
    //level is used if you are querying from an old level
    virtual UnstructuredParticleSubset* getParticleSubset(int matlIndex,
                                              IntVector low,
                                              IntVector high,
                                              const UnstructuredPatch* relPatch,
                                              const UnstructuredVarLabel* posvar,
                                              const UnstructuredLevel* level = 0);

    virtual void allocateTemporary(UnstructuredParticleVariableBase&,
                                   UnstructuredParticleSubset*);

    virtual void allocateAndPut(UnstructuredParticleVariableBase&,
                                const UnstructuredVarLabel*,
                                UnstructuredParticleSubset*);

    virtual void get(constUnstructuredParticleVariableBase&,
                     const UnstructuredVarLabel*,
                     UnstructuredParticleSubset*);

    virtual void get(constUnstructuredParticleVariableBase&,
                     const UnstructuredVarLabel*,
                     int matlIndex,
                     const UnstructuredPatch* patch);

    virtual void getModifiable(UnstructuredParticleVariableBase&,
                               const UnstructuredVarLabel*,
                               UnstructuredParticleSubset*);

    virtual void put(UnstructuredParticleVariableBase&,
                     const UnstructuredVarLabel*,
                     bool replace = false);

    virtual UnstructuredParticleVariableBase* getParticleVariable(const UnstructuredVarLabel*,
                                                      UnstructuredParticleSubset*);

    virtual UnstructuredParticleVariableBase* getParticleVariable(const UnstructuredVarLabel*,
                                                      int matlIndex,
                                                      const UnstructuredPatch* patch);
    void printParticleSubsets();

    virtual void getCopy(UnstructuredParticleVariableBase&,
                         const UnstructuredVarLabel*,
                         UnstructuredParticleSubset*);

    virtual void copyOut(UnstructuredParticleVariableBase&,
                         const UnstructuredVarLabel*,
                         UnstructuredParticleSubset*);

    // Remove particles that are no longer relevant
    virtual void deleteParticles(UnstructuredParticleSubset* delset);

    virtual void addParticles(const UnstructuredPatch* patch,
                              int matlIndex,
                              std::map<const UnstructuredVarLabel*, UnstructuredParticleVariableBase*>* addedstate);

    //__________________________________
    // Grid Variables
    virtual void print();

    virtual void clear();

    void get(constUnstructuredGridVariableBase& var,
             const UnstructuredVarLabel* label,
             int matlIndex,
             const UnstructuredPatch* patch,
             Ghost::GhostType gtype,
             int numGhostCells);

    void getModifiable(UnstructuredGridVariableBase& var,
                       const UnstructuredVarLabel* label,
                       int matlIndex,
                       const UnstructuredPatch* patch,
                       Ghost::GhostType gtype = Ghost::None,
                       int numGhostCells = 0);

    void allocateTemporary(UnstructuredGridVariableBase& var,
                           const UnstructuredPatch* patch,
                           Ghost::GhostType gtype,
                           int numGhostCells);
//                          const IntVector& boundaryLayer);

    void allocateAndPut(UnstructuredGridVariableBase& var,
                        const UnstructuredVarLabel* label,
                        int matlIndex,
                        const UnstructuredPatch* patch,
                        Ghost::GhostType gtype,
                        int numGhostCells);

    void put(UnstructuredGridVariableBase& var,
             const UnstructuredVarLabel* label,
             int matlIndex,
             const UnstructuredPatch* patch,
             bool replace = false);

    // returns the constUnstructuredGridVariable for all patches on the level
    virtual void getLevel(constUnstructuredGridVariableBase&,
                          const UnstructuredVarLabel*,
                          int matlIndex,
                          const UnstructuredLevel* level);

    //meant for the UnifiedScheduler only.  Puts a contiguous level in the *level* database
    //so that it doesn't need to constantly make new deep copies each time the full level
    //is requested.
    void putLevelDB( UnstructuredGridVariableBase* gridVar,
                           const UnstructuredVarLabel* label,
                           const UnstructuredLevel* level,
                           int matlIndex = -1 );

    virtual void getRegion(constUnstructuredGridVariableBase&,
                           const UnstructuredVarLabel*,
                           int matlIndex,
                           const UnstructuredLevel* level,
                           const IntVector& low,
                           const IntVector& high,
                           bool useBoundaryCells = true);

    virtual void getRegionModifiable(UnstructuredGridVariableBase&,
                           const UnstructuredVarLabel*,
                           int matlIndex,
                           const UnstructuredLevel* level,
                           const IntVector& low,
                           const IntVector& high,
                           bool useBoundaryCells = true);

    virtual void copyOut(UnstructuredGridVariableBase& var,
                         const UnstructuredVarLabel* label,
                         int matlIndex,
                         const UnstructuredPatch* patch,
                         Ghost::GhostType gtype = Ghost::None,
                         int numGhostCells = 0);

    virtual void getCopy(UnstructuredGridVariableBase& var,
                         const UnstructuredVarLabel* label,
                         int matlIndex,
                         const UnstructuredPatch* patch,
                         Ghost::GhostType gtype = Ghost::None,
                         int numGhostCells = 0);

    // UnstructuredPerPatch Variables
    virtual void get(UnstructuredPerPatchBase&,
                     const UnstructuredVarLabel*,
                     int matIndex,
                     const UnstructuredPatch*);

    virtual void put(UnstructuredPerPatchBase&,
                     const UnstructuredVarLabel*,
                     int matIndex,
                     const UnstructuredPatch*,
                     bool replace = false);

    virtual ScrubMode setScrubbing(ScrubMode);

    // For related datawarehouses
    virtual UnstructuredDataWarehouse* getOtherDataWarehouse(UnstructuredTask::WhichDW);

    virtual void transferFrom(UnstructuredDataWarehouse*, const UnstructuredVarLabel*,
            const UnstructuredPatchSubset*, const MaterialSubset*);

    virtual void transferFrom(UnstructuredDataWarehouse*,
                              const UnstructuredVarLabel*,
                              const UnstructuredPatchSubset* patches,
                              const MaterialSubset*,
                              bool replace);

    virtual void transferFrom(UnstructuredDataWarehouse*,
                              const UnstructuredVarLabel*,
                              const UnstructuredPatchSubset* patches,
                              const MaterialSubset*,
                              bool replace,
                              const UnstructuredPatchSubset* newPatches);


    virtual void transferFrom(UnstructuredDataWarehouse*,
                              const UnstructuredVarLabel*,
                              const UnstructuredPatchSubset* patches,
                              const MaterialSubset*,
                              void * dtask,
                              bool replace,
                              const UnstructuredPatchSubset* newPatches);


    virtual bool isFinalized() const;

    virtual void finalize();

#ifdef HAVE_CUDA
   static int getNumDevices();
   static void uintahSetCudaDevice(int deviceNum);
   static size_t getTypeDescriptionSize(const TypeDescription::Type& type);
   static GPUGridVariableBase* createGPUGridVariable(const TypeDescription::Type& type);
   static GPUPerPatchBase* createGPUPerPatch(const TypeDescription::Type& type);
   static GPUReductionVariableBase* createGPUReductionVariable(const TypeDescription::Type& type);

#endif

    virtual void unfinalize();

    virtual void refinalize();

    virtual size_t emit(OutputContext&,
                        const UnstructuredVarLabel* label,
                        int matlIndex,
                        const UnstructuredPatch* patch);
#if HAVE_PIDX
     void emitPIDX(PIDXOutputContext&, 
                      const UnstructuredVarLabel* label, 
                      int matlIndex,
                      const UnstructuredPatch* patch, 
                      unsigned char *pidx_buffer,
                      size_t pidx_bufferSize);
#endif
    void exchangeParticleQuantities( UnstructuredDetailedTasks    * dts,
                                     UnstructuredLoadBalancer * lb,
                                     const UnstructuredVarLabel   * pos_var,
                                     int              iteration );

    void sendMPI( UnstructuredDependencyBatch       * batch,
                  const UnstructuredVarLabel        * pos_var,
                  BufferInfo            & buffer,
                  UnstructuredOnDemandDataWarehouse * old_dw,
                  const UnstructuredDetailedDep     * dep,
                  UnstructuredLoadBalancer      * lb );

    void recvMPI( UnstructuredDependencyBatch       * batch,
                  BufferInfo            & buffer,
                  UnstructuredOnDemandDataWarehouse * old_dw,
                  const UnstructuredDetailedDep     * dep,
                  UnstructuredLoadBalancer      * lb);

    void reduceMPI(const UnstructuredVarLabel* label,
                   const UnstructuredLevel* level,
                   const MaterialSubset* matls,
                   int nComm);

    // Scrub counter manipulator functions -- when the scrub count goes to
    // zero, the data is deleted
    void setScrubCount(const UnstructuredVarLabel* label,
                       int matlIndex,
                       const UnstructuredPatch* patch,
                       int count);

    int decrementScrubCount( const UnstructuredVarLabel * label,
                                   int        matlIndex,
                             const UnstructuredPatch    * patch );

    void scrub( const UnstructuredVarLabel * label,
                      int        matlIndex,
                const UnstructuredPatch    * patch);

    void initializeScrubs(       int                        dwid,
                           const FastHashTable<UnstructuredScrubItem> * scrubcounts,
                                 bool                       add );

    // For timestep abort/restart
    virtual bool timestepAborted();

    virtual bool timestepRestarted();

    virtual void abortTimestep();

    virtual void restartTimestep();

    virtual void setRestarted() { d_hasRestarted = true; }

   struct ValidNeighbors {
     UnstructuredGridVariableBase* validNeighbor;
     const UnstructuredPatch* neighborPatch;
     IntVector low;
     IntVector high;

   };
   void getNeighborPatches(const UnstructuredVarLabel* label, const UnstructuredPatch* patch, Ghost::GhostType gtype,
                                  int numGhostCells, std::vector<const UnstructuredPatch *>& adjacentNeighbors);

   void getSizesForVar(const UnstructuredVarLabel* label, int matlIndex, const UnstructuredPatch* patch,
                                         IntVector& low, IntVector& high, IntVector& dataLow,
                                         IntVector& siz, IntVector& strides);

   void getValidNeighbors(const UnstructuredVarLabel* label, int matlIndex, const UnstructuredPatch* patch,
           Ghost::GhostType gtype, int numGhostCells, std::vector<ValidNeighbors>& validNeighbors,
           bool ignoreMissingNeighbors = false);

   void logMemoryUse(std::ostream& out,
                      unsigned long& total,
                      const std::string& tag);

    // must be called by the thread that will run the test
    void pushRunningTask(const UnstructuredTask* task,
                         std::vector<UnstructuredOnDemandDataWarehouseP>* dws);

    void popRunningTask();

    // does a final check to see if gets/puts/etc. consistent with
    // requires/computes/modifies for the current task.
    void checkTasksAccesses(const UnstructuredPatchSubset    * patches,
                            const MaterialSubset * matls);

    ScrubMode getScrubMode() const { return d_scrubMode; }

    // The following is for support of regriding
    virtual void getVarLabelMatlLevelTriples(std::vector<UnstructuredVarLabelMatl<UnstructuredLevel> >& vars) const;

    static bool d_combineMemory;

    friend class UnstructuredSchedulerCommon;
    friend class UnifiedScheduler;

  private:

    enum AccessType {
      NoAccess = 0,
      PutAccess,
      GetAccess,
      ModifyAccess
    };

    struct AccessInfo {
        AccessInfo()
            : accessType(NoAccess), lowOffset(0, 0, 0), highOffset(0, 0, 0) { }

        AccessInfo(AccessType type)
            : accessType(type), lowOffset(0, 0, 0), highOffset(0, 0, 0) { }

        void encompassOffsets(IntVector low,
                              IntVector high)
        {
          lowOffset  = Uintah::Max( low,  lowOffset );
          highOffset = Uintah::Max( high, highOffset );
        }

        AccessType accessType;
        IntVector lowOffset;  // ghost cell access
        IntVector highOffset;
    };

    typedef std::map<UnstructuredVarLabelMatl<UnstructuredPatch>, AccessInfo> VarAccessMap;

    struct RunningTaskInfo {
        RunningTaskInfo()
            : d_task(0), dws(0) { }

        RunningTaskInfo(const UnstructuredTask* task,
                        std::vector<UnstructuredOnDemandDataWarehouseP>* dws)
            : d_task(task), dws(dws) { }

        RunningTaskInfo(const RunningTaskInfo& copy)
            : d_task(copy.d_task), dws(copy.dws), d_accesses(copy.d_accesses) { }

        RunningTaskInfo& operator=(const RunningTaskInfo& copy)
        {
          d_task = copy.d_task;
          dws = copy.dws;
          d_accesses = copy.d_accesses;
          return *this;
        }
        const UnstructuredTask* d_task;
        std::vector<UnstructuredOnDemandDataWarehouseP>* dws;
        VarAccessMap d_accesses;
    };

    virtual UnstructuredDataWarehouse* getOtherDataWarehouse(UnstructuredTask::WhichDW,
                                                 RunningTaskInfo *info);

    void getGridVar(UnstructuredGridVariableBase& var,
                    const UnstructuredVarLabel* label,
                    int matlIndex,
                    const UnstructuredPatch* patch,
                    Ghost::GhostType gtype,
                    int numGhostCells);

    inline UnstructuredTask::WhichDW getWhichDW(RunningTaskInfo *info);

    // These will throw an exception if access is not allowed for the current task.
    inline void checkGetAccess(const UnstructuredVarLabel* label,
                               int matlIndex,
                               const UnstructuredPatch* patch,
                               Ghost::GhostType gtype = Ghost::None,
                               int numGhostCells = 0);

    inline void checkPutAccess(const UnstructuredVarLabel* label,
                               int matlIndex,
                               const UnstructuredPatch* patch,
                               bool replace);

    inline void checkModifyAccess(const UnstructuredVarLabel* label,
                                  int matlIndex,
                                  const UnstructuredPatch* patch);

    // These will return false if access is not allowed for the current task.
    inline bool hasGetAccess(const UnstructuredTask* runningTask,
                             const UnstructuredVarLabel* label,
                             int matlIndex,
                             const UnstructuredPatch* patch,
                             IntVector lowOffset,
                             IntVector highOffset,
                             RunningTaskInfo *info);

    inline bool hasPutAccess(const UnstructuredTask* runningTask,
                             const UnstructuredVarLabel* label,
                             int matlIndex,
                             const UnstructuredPatch* patch,
                             bool replace);

    void checkAccesses(RunningTaskInfo* runningTaskInfo,
                       const UnstructuredTask::Dependency* dep,
                       AccessType accessType,
                       const UnstructuredPatchSubset* patches,
                       const MaterialSubset* matls);
    
    void printDebuggingPutInfo( const UnstructuredVarLabel* label,
                                int matlIndex,
                                const UnstructuredPatch* patch,
                                int line);
                       
                       

    struct dataLocation {
        const UnstructuredPatch * patch;
        int mpiNode;
    };

    typedef std::vector<dataLocation*> variableListType;
    typedef std::map<const UnstructuredVarLabel*, variableListType*, UnstructuredVarLabel::Compare> dataLocationDBtype;
    typedef std::multimap<UnstructuredPSPatchMatlGhost, UnstructuredParticleSubset*> psetDBType;
    typedef std::map<std::pair<int, const UnstructuredPatch*>, std::map<const UnstructuredVarLabel*, UnstructuredParticleVariableBase*>*> psetAddDBType;
    typedef std::map<std::pair<int, const UnstructuredPatch*>, int> particleQuantityType;
#ifdef HAVE_CUDA
   std::map<Patch*, bool> assignedPatches;
   // indicates where a given patch should be stored in an accelerator
#endif

    UnstructuredParticleSubset* queryPSetDB(psetDBType &db,
                                const UnstructuredPatch* patch,
                                int matlIndex,
                                IntVector low,
                                IntVector high,
                                const UnstructuredVarLabel* pos_var,
                                bool exact = false);

    void insertPSetRecord(psetDBType &subsetDB,
                          const UnstructuredPatch* patch,
                          IntVector low,
                          IntVector high,
                          int matlIndex,
                          UnstructuredParticleSubset *psubset);

    UnstructuredDWDatabase<UnstructuredPatch>  d_varDB;
    UnstructuredDWDatabase<UnstructuredLevel>  d_levelDB;
    UnstructuredKeyDatabase<UnstructuredPatch> d_varkeyDB;
    UnstructuredKeyDatabase<UnstructuredLevel> d_levelkeyDB;

    psetDBType           d_psetDB;
    psetDBType           d_delsetDB;
    psetAddDBType        d_addsetDB;
    particleQuantityType d_foreignParticleQuantities;

    // Keep track of when this DW sent some (and which) particle information to another processor
    UnstructuredSendState ss_;

    /*
     // On a timestep restart, sometimes (when an entire patch is sent) on the
     // first try of the timestep the receiving DW creates and stores ParticleSubset
     // which throws off the sending on the next iteration.  This will compensate.
     SendState                         d_timestepRestartPsets;
     */

    // Record of which DataWarehouse has the data for each variable...
    //  Allows us to look up the DW to which we will send a data request.
    dataLocationDBtype d_dataLocation;

    bool                 d_finalized;
    UnstructuredGridP                d_grid;

    // Is this the first DW -- created by the initialization timestep?
    bool d_isInitializationDW;

    inline bool hasRunningTask();

    inline std::list<RunningTaskInfo>* getRunningTasksInfo();

    inline RunningTaskInfo* getCurrentTaskInfo();

    std::map<std::thread::id, std::list<RunningTaskInfo> > d_runningTasks; // a list of running tasks for each std::thread::id

    ScrubMode d_scrubMode;

    bool d_aborted;
    bool d_restart;

    // Whether this (Old) DW is being used for a restarted timestep (the new DWs are cleared out)
    bool d_hasRestarted;

}; // end class UnstructuredOnDemandDataWarehouse

}  // end namespace Uintah

#endif // end #ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
