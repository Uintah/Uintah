#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Core/Thread/CrowdMonitor.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DWDatabase.h>
#include <Packages/Uintah/Core/Grid/VarLabelMatlPatch.h>

#include <map>
#include <iosfwd>
#include <vector>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

namespace SCIRun {
  class Thread;
}

namespace Uintah {

using namespace std;
using namespace SCIRun;
  class BufferInfo;
  class DependencyBatch;
  class DetailedDep;
  class TypeDescription;
  class Patch;
  class ProcessorGroup;
  class SendState;

/**************************************

  CLASS
        OnDemandDataWarehouse
   
	Short description...

  GENERAL INFORMATION

        OnDemandDataWarehouse.h

	Steven G. Parker
	Department of Computer Science
	University of Utah

	Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
	Copyright (C) 2000 SCI Group

  KEYWORDS
        On_Demand_Data_Warehouse

  DESCRIPTION
        Long description...
  
  WARNING
  
****************************************/

class OnDemandDataWarehouse : public DataWarehouse {
public:
   OnDemandDataWarehouse( const ProcessorGroup* myworld,
			  const Scheduler* scheduler, int generation,
			  const GridP& grid,
			  bool isInitializationDW = false);
   virtual ~OnDemandDataWarehouse();
   
   virtual bool exists(const VarLabel*, int matIndex, const Patch*) const; 
  
   // Generic put and allocate, passing Variable as a pointer rather than
   // by reference to avoid ambiguity with other put overloaded methods.
   virtual void put(Variable*, const VarLabel*, int matlIndex,
		    const Patch*);   
   virtual void allocateAndPutGridVar(Variable*, const VarLabel*, 
				      int matlIndex, const Patch*);   
   
   // Reduction Variables
   virtual void allocate(ReductionVariableBase&, const VarLabel*,
			 int matIndex = -1);
   virtual void get(ReductionVariableBase&, const VarLabel*,
		    int matIndex = -1);
   virtual void put(const ReductionVariableBase&, const VarLabel*,
		    int matIndex = -1);
   virtual void override(const ReductionVariableBase&, const VarLabel*,
			 int matIndex = -1);

   // Particle Variables
   virtual ParticleSubset* createParticleSubset(particleIndex numParticles,
						int matlIndex, const Patch*);
   virtual void saveParticleSubset(int matlIndex, const Patch*,
	 			   ParticleSubset*);
   virtual bool haveParticleSubset(int matlIndex, const Patch*);
   virtual ParticleSubset* getParticleSubset(int matlIndex,
					     const Patch*);
   virtual ParticleSubset* getDeleteSubset(int matlIndex,
					     const Patch*);
   virtual ParticleSubset* getParticleSubset(int matlIndex,
			 const Patch*, Ghost::GhostType, int numGhostCells,
			 const VarLabel* posvar);
   virtual void allocateTemporary(ParticleVariableBase&, ParticleSubset*);
   virtual void allocateAndPut(ParticleVariableBase&, const VarLabel*,
			       ParticleSubset*);
   virtual void get(constParticleVariableBase&, const VarLabel*,
		    ParticleSubset*);
   virtual void get(constParticleVariableBase&, const VarLabel*,
		    int matlIndex, const Patch* patch);
   virtual void getModifiable(ParticleVariableBase&, const VarLabel*,
		    ParticleSubset*);
   virtual void put(ParticleVariableBase&, const VarLabel*,
		    bool replace = false);
   virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
						     ParticleSubset*);
   virtual ParticleVariableBase*
   getParticleVariable(const VarLabel*, int matlIndex, const Patch* patch);  

   // NCVariables Variables
   virtual void allocateTemporary(NCVariableBase&, const Patch*,
				  Ghost::GhostType = Ghost::None,
				  int numGhostCells = 0);
   virtual void allocateAndPut(NCVariableBase&, const VarLabel*,
			       int matlIndex, const Patch*,
			       Ghost::GhostType = Ghost::None,
			       int numGhostCells = 0);
   virtual void get(constNCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(NCVariableBase&, const VarLabel*, int matlIndex,
			      const Patch*);
   virtual void put(NCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   // CCVariables Variables
   virtual void allocateTemporary(CCVariableBase&, const Patch*, 
				  Ghost::GhostType = Ghost::None,
				  int numGhostCells = 0);
   virtual void allocateAndPut(CCVariableBase&, const VarLabel*,
			       int matlIndex, const Patch*, 
			       Ghost::GhostType = Ghost::None,
			       int numGhostCells = 0);
   virtual void get(constCCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(CCVariableBase&, const VarLabel*, int matlIndex,
			      const Patch*);
   virtual void put(CCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   // SFC[X-Z]Variables Variables
   virtual void allocateTemporary(SFCXVariableBase&, const Patch*,
				  Ghost::GhostType = Ghost::None,
				  int numGhostCells = 0);
   virtual void allocateAndPut(SFCXVariableBase&, const VarLabel*,
			       int matlIndex, const Patch*,
			       Ghost::GhostType = Ghost::None,
			       int numGhostCells = 0);			 
   virtual void get(constSFCXVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(SFCXVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*);
   virtual void put(SFCXVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   virtual void allocateTemporary(SFCYVariableBase&, const Patch*,
				  Ghost::GhostType = Ghost::None,
				  int numGhostCells = 0); 
   virtual void allocateAndPut(SFCYVariableBase&, const VarLabel*,
			       int matlIndex, const Patch*,
			       Ghost::GhostType = Ghost::None,
			       int numGhostCells = 0);			 
   virtual void get(constSFCYVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(SFCYVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*);
   virtual void put(SFCYVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   virtual void allocateTemporary(SFCZVariableBase&, const Patch*,
				  Ghost::GhostType = Ghost::None,
				  int numGhostCells = 0);	 
   virtual void allocateAndPut(SFCZVariableBase&, const VarLabel*,
			       int matlIndex, const Patch*,
			       Ghost::GhostType = Ghost::None,
			       int numGhostCells = 0);			 
   virtual void get(constSFCZVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(SFCZVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*);
   virtual void put(SFCZVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);
   // PerPatch Variables
   virtual void get(PerPatchBase&, const VarLabel*, int matIndex, const Patch*);
   virtual void put(PerPatchBase&, const VarLabel*,
		    int matIndex, const Patch*, bool replace = false);

   // Remove particles that are no longer relevant
   virtual void deleteParticles(ParticleSubset* delset);

   virtual bool isFinalized() const;
   virtual bool exists(const VarLabel*, const Patch*) const;
   
   virtual void finalize();

   virtual void emit(OutputContext&, const VarLabel* label,
		     int matlIndex, const Patch* patch);

   virtual void print(ostream& intout, const VarLabel* label,
		      int matlIndex = -1);

   void sendMPI(SendState& ss, DependencyBatch* batch,
		const ProcessorGroup* world, const VarLabel* pos_var,
		BufferInfo& buffer, OnDemandDataWarehouse* old_dw,
		const DetailedDep* dep);
   void recvMPI(BufferInfo& buffer, DependencyBatch* batch,
		const ProcessorGroup* world, OnDemandDataWarehouse* old_dw,
		const DetailedDep* dep);
   void reduceMPI(const VarLabel* label, const MaterialSubset* matls,
		  const ProcessorGroup* world);

   // Scrub counter manipulator functions -- when the scrub count goes to
   // zero, the data is scrubbed.
   virtual void setScrubCountIfZero(const VarLabel* label, int matlIndex,
				    const Patch* patch, int count)
   { addScrubCount(label, matlIndex, patch, 0, count); }
   virtual void decrementScrubCount(const VarLabel* label, int matlIndex,
				    const Patch* patch, int count = 1,
				    unsigned int addIfZero = 0)
   { addScrubCount(label, matlIndex, patch, -count, addIfZero); }

   // scrub everything with a scrubCount of zero
   virtual void scrubExtraneous();
  
   void logMemoryUse(ostream& out, unsigned long& total, const string& tag);

   // must be called by the thread that will run the test
   virtual void pushRunningTask(const Task* task);
   virtual void popRunningTask();  

   // does a final check to see if gets/puts/etc. consistent with
   // requires/computes/modifies for the current task.
   virtual void checkTasksAccesses(const PatchSubset* patches,
				   const MaterialSubset* matls);
private:
   void addScrubCount(const VarLabel* label, int matlIndex,
		      const Patch* patch, int count = 1,
		      unsigned int addIfZero = 0);
  
   enum AccessType {
     NoAccess = 0, PutAccess, GetAccess, ModifyAccess
   };

   struct AccessInfo {
     AccessInfo()
       : accessType(NoAccess), lowOffset(0, 0, 0), highOffset(0, 0, 0) {}
     AccessInfo(AccessType type)
       : accessType(type), lowOffset(0, 0, 0), highOffset(0, 0, 0) {}
     void encompassOffsets(IntVector low, IntVector high)
     { lowOffset = Max(low, lowOffset); highOffset = Max(high, highOffset); }
     
     AccessType accessType;
     IntVector lowOffset; // ghost cell access
     IntVector highOffset;
   };
  
   typedef map<VarLabelMatlPatch, AccessInfo> VarAccessMap;

   struct RunningTaskInfo {
     RunningTaskInfo()
       : d_task(0) {}
     RunningTaskInfo(const Task* task)
       : d_task(task) {}
     RunningTaskInfo(const RunningTaskInfo& copy)
       : d_task(copy.d_task), d_accesses(copy.d_accesses) {}
     RunningTaskInfo& operator=(const RunningTaskInfo& copy)
     { d_task = copy.d_task; d_accesses = copy.d_accesses; return *this; }
     const Task* d_task;
     VarAccessMap d_accesses;
   };  

   // Generic get function used by the get functions for grid-based
   // (node or cell) variables to avoid code duplication.
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif  
   template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
   void getGridVar(VariableBase& var, DWDatabase& db,
		   const VarLabel* label, int matlIndex, const Patch* patch,
		   Ghost::GhostType gtype, int numGhostCells);

   template <Patch::VariableBasis basis, class VariableBase>
   void allocateTemporaryGridVar(VariableBase& var, const Patch* patch,
				 Ghost::GhostType gtype, int numGhostCells);

   template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
   void allocateAndPutGridVar(VariableBase& var, DWDatabase& db,
			      const VarLabel* label, int matlIndex, 
			      const Patch* patch,
			      Ghost::GhostType gtype, int numGhostCells);

   template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
   void putGridVar(VariableBase& var, DWDatabase& db,
		   const VarLabel* label, int matlIndex, const Patch* patch,
		   bool replace = false);
   template <class VariableBase, class DWDatabase>
   void recvMPIGridVar(DWDatabase& db, BufferInfo& buffer, 
		       const DetailedDep* dep, const VarLabel* label, 
		       int matlIndex, const Patch* patch);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif  

  // These will throw an exception if access is not allowed for the
  // curent task.
  inline void checkGetAccess(const VarLabel* label, int matlIndex,
			     const Patch* patch,
			     Ghost::GhostType gtype = Ghost::None,
			     int numGhostCells = 0);
  inline void checkPutAccess(const VarLabel* label, int matlIndex,
			     const Patch* patch, bool replace);
  inline void checkModifyAccess(const VarLabel* label, int matlIndex,
				const Patch* patch);
  
  // These will return false if access is not allowed for
  // the current task.
  inline bool hasGetAccess(const Task* runningTask, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   IntVector lowOffset, IntVector highOffset);
  inline bool hasPutAccess(const Task* runningTask, const VarLabel* label,
			   int matlIndex, const Patch* patch, bool replace);

  void checkAccesses(RunningTaskInfo* runningTaskInfo,
		     const Task::Dependency* dep,
		     AccessType accessType, const PatchSubset* patches,
		     const MaterialSubset* matls);
  
   struct dataLocation {
      const Patch   * patch;
            int        mpiNode;
   };

   typedef vector<dataLocation*> variableListType;
   typedef map<const VarLabel*, variableListType*, VarLabel::Compare> dataLocationDBtype;
   typedef map<pair<int, const Patch*>, ParticleSubset*> psetDBType;

   DWDatabase<NCVariableBase>        d_ncDB;
   DWDatabase<CCVariableBase>        d_ccDB;
   DWDatabase<SFCXVariableBase>      d_sfcxDB;
   DWDatabase<SFCYVariableBase>      d_sfcyDB;
   DWDatabase<SFCZVariableBase>      d_sfczDB;
   DWDatabase<ParticleVariableBase>  d_particleDB;
   DWDatabase<ReductionVariableBase> d_reductionDB;
   DWDatabase<PerPatchBase>          d_perpatchDB;
   psetDBType                        d_psetDB;
   psetDBType                        d_delsetDB;

   // Record of which DataWarehouse has the data for each variable...
   //  Allows us to look up the DW to which we will send a data request.
   dataLocationDBtype      d_dataLocation;

   //////////
   // Insert Documentation Here:
   mutable CrowdMonitor    d_lock;
   bool                    d_finalized;
   GridP                   d_grid;

   // Is this the first DW -- created by the initialization timestep?
   bool d_isInitializationDW;
  
   inline bool hasRunningTask();
   inline list<RunningTaskInfo>* getRunningTasksInfo();
   inline RunningTaskInfo* getCurrentTaskInfo();
    
   map<Thread*, list<RunningTaskInfo> > d_runningTasks;
  
   // Internal VarLabel for the position of this DataWarehouse
   // ??? with respect to what ???? 
   //const VarLabel * d_positionLabel;
};

} // end namespace Uintah

#endif
