#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Core/Thread/CrowdMonitor.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DWDatabase.h>

#include <map>
#include <iosfwd>
#include <vector>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

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
  
   // Generic put, passing Variable as a pointer rather than by reference
   // to avoid ambiguity with other put overloaded methods.
   virtual void put(Variable*, const VarLabel*, int matlIndex,
		    const Patch*);   
   
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
   virtual ParticleSubset* getParticleSubset(int matlIndex,
			 const Patch*, Ghost::GhostType, int numGhostCells,
			 const VarLabel* posvar);
   virtual void allocate(ParticleVariableBase&, const VarLabel*,
			 ParticleSubset*);
   virtual void get(constParticleVariableBase&, const VarLabel*,
		    ParticleSubset*);
   virtual void getModifiable(ParticleVariableBase&, const VarLabel*,
		    ParticleSubset*);
   virtual void put(ParticleVariableBase&, const VarLabel*,
		    bool replace = false);
   virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
						     ParticleSubset*);

   // NCVariables Variables
   virtual void allocate(NCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*,
			 Ghost::GhostType = Ghost::None,
			 int numGhostCells = 0);
   virtual void get(constNCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(NCVariableBase&, const VarLabel*, int matlIndex,
			      const Patch*);
   virtual void put(NCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   // CCVariables Variables -- fron Tan... need to be fixed...
   virtual void allocate(CCVariableBase&, const VarLabel*,
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
   virtual void allocate(SFCXVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*,
			 Ghost::GhostType = Ghost::None,
			 int numGhostCells = 0);			 
   virtual void get(constSFCXVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(SFCXVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*);
   virtual void put(SFCXVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   virtual void allocate(SFCYVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*,
			 Ghost::GhostType = Ghost::None,
			 int numGhostCells = 0);			 
   virtual void get(constSFCYVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void getModifiable(SFCYVariableBase&, const VarLabel*,
			      int matlIndex, const Patch*);
   virtual void put(SFCYVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*, bool replace = false);

   virtual void allocate(SFCZVariableBase&, const VarLabel*,
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

   void scrub(const VarLabel*);

   void logMemoryUse(ostream& out, unsigned long& total, const string& tag);

   // must be called by the thread that will run the test
   virtual void setCurrentTask(const Task* task);
  
   // does a final check to see if gets/puts/etc. consistent with
   // requires/computes/modifies for the current task.
   virtual void checkTasksAccesses(const PatchSubset* patches,
				   const MaterialSubset* matls);
private:
   enum AccessType {
     NoAccess = 0, PutAccess, GetAccess, ModifyAccess
   };
  
   // Generic get function used by the get functions for grid-based
   // (node or cell) variables to avoid code duplication.
#ifdef __sgi
#pragma set woff 1424
#endif  
   template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
   void getGridVar(VariableBase& var, DWDatabase& db,
		   const VarLabel* label, int matlIndex, const Patch* patch,
		   Ghost::GhostType gtype, int numGhostCells);
   template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
   void
   allocateGridVar(VariableBase& var, DWDatabase& db,
		   const VarLabel* label, int matlIndex, const Patch* patch,
		   Ghost::GhostType gtype, int numGhostCells);
#ifdef __sgi
#pragma reset woff 1424
#endif  

  // These will throw an exception if access is not allowed for the
  // curent task.
  inline void checkGetAccess(const VarLabel* label, int matlIndex,
			     const Patch* patch);
  inline void checkPutAccess(const VarLabel* label, int matlIndex,
			     const Patch* patch, bool replace);
  inline void checkAllocation(const Variable& var, const VarLabel* label);
  inline void checkModifyAccess(const VarLabel* label, int matlIndex,
				const Patch* patch);
  
  // These will return false if access is not allowed for
  // the current task.
  inline bool hasGetAccess(const VarLabel* label, int matlIndex,
			   const Patch* patch);
  inline bool hasPutAccess(const VarLabel* label, int matlIndex,
			   const Patch* patch, bool replace);

  void checkAccesses(const Task* currentTask, const Task::Dependency* dep,
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

   inline const Task* getCurrentTask();	 
   map<Thread*, const Task*> d_runningTasks;
  
  struct SpecificVarLabel {
    SpecificVarLabel(const VarLabel* label, int matlIndex, const Patch* patch)
      : label_(label), matlIndex_(matlIndex), patch_(patch) {}
    SpecificVarLabel(const SpecificVarLabel& copy)
      : label_(copy.label_), matlIndex_(copy.matlIndex_), patch_(copy.patch_)
    {}
    SpecificVarLabel& operator=(const SpecificVarLabel& copy)
    {
      label_=copy.label_; matlIndex_=copy.matlIndex_; patch_=copy.patch_;
      return *this;
    }
    
    bool operator<(const SpecificVarLabel& other) const;
    const VarLabel* label_;
    int matlIndex_;
    const Patch* patch_;    
  };
  
  map<SpecificVarLabel, AccessType> d_currentTaskAccesses;
  
   // Internal VarLabel for the position of this DataWarehouse
   // ??? with respect to what ???? 
   //const VarLabel * d_positionLabel;
};

} // end namespace Uintah

#endif
