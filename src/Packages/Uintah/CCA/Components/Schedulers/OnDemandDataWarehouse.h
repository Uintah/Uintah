#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Core/Thread/CrowdMonitor.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DWDatabase.h>

#include <map>
#include <iosfwd>
#include <vector>
#include <mpi.h>

namespace Uintah {

using namespace std;
using namespace SCIRun;

class DataItem;
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
   OnDemandDataWarehouse( const ProcessorGroup* myworld, int generation,
			  DataWarehouseP& parent);
   virtual ~OnDemandDataWarehouse();
   
   virtual void setGrid(const GridP&);

   virtual bool exists(const VarLabel*, int matIndex, const Patch*) const;

   // Generic put, passing Variable as a pointer rather than by reference
   // to avoid ambiguity with other put overloaded methods.
   virtual void put(const Variable*, const VarLabel*, int matlIndex,
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

   // Scatther/gather.  This will need a VarLabel if anyone but the
   // scheduler ever wants to use it.
   virtual void scatter(ScatterGatherBase*, const Patch*, const Patch*);
   virtual ScatterGatherBase* gather(const Patch*, const Patch*);
      
   // Particle Variables
   virtual ParticleSubset* createParticleSubset(particleIndex numParticles,
						int matlIndex, const Patch*);
   virtual bool haveParticleSubset(int matlIndex, const Patch*);
   virtual ParticleSubset* getParticleSubset(int matlIndex,
					     const Patch*);
   virtual ParticleSubset* getParticleSubset(int matlIndex,
			 const Patch*, Ghost::GhostType, int numGhostCells,
			 const VarLabel* posvar);
   virtual void allocate(ParticleVariableBase&, const VarLabel*,
			 ParticleSubset*);
   virtual void get(ParticleVariableBase&, const VarLabel*,
		    ParticleSubset*);
   virtual void put(const ParticleVariableBase&, const VarLabel*);
   virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
						     ParticleSubset*);

   // NCVariables Variables
   virtual void allocate(NCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*,
			 const IntVector gc = IntVector(0,0,0));
   virtual void get(NCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const NCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // CCVariables Variables -- fron Tan... need to be fixed...
   virtual void allocate(CCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*, 
			 const IntVector gc = IntVector(0,0,0));
   virtual void get(CCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const CCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // SFC[X-Z]Variables Variables
   virtual void allocate(SFCXVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(SFCXVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const SFCXVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   virtual void allocate(SFCYVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(SFCYVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const SFCYVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   virtual void allocate(SFCZVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(SFCZVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const SFCZVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);
   // PerPatch Variables
   virtual void get(PerPatchBase&, const VarLabel*, int matIndex, const Patch*);
   virtual void put(const PerPatchBase&, const VarLabel*,
				 int matIndex, const Patch*);

   // Remove particles that are no longer relevant
   virtual void deleteParticles(ParticleSubset* delset);

   virtual bool isFinalized() const;
   virtual bool exists(const VarLabel*, const Patch*) const;
   
   virtual void finalize();

   virtual void emit(OutputContext&, const VarLabel* label,
		     int matlIndex, const Patch* patch) const;

   virtual void print(ostream& intout, const VarLabel* label,
		      int matlIndex = -1) const;

   void sendParticleSubset(SendState& ss,
			   ParticleSubset* pset, const VarLabel* pos_var,
			   const Task::Dependency* dep, const Patch* toPatch,
			   const ProcessorGroup* world, int* size);
   void sendMPI(SendState& ss, const VarLabel* label, int matlIndex,
		const Patch* patch, const ProcessorGroup* world,
		const Task::Dependency*, int dest,
		int tag, int* size, MPI_Request* requestid);
   void recvMPI(SendState& ss, DataWarehouseP& old_dw, 
		const VarLabel* label, int matlIndex,
		const Patch* patch, const ProcessorGroup* world,
		const Task::Dependency*, int src,
		int tag, int* size, MPI_Request* requestid);
   void reduceMPI(const VarLabel* label, int matlIndex,
		  const ProcessorGroup* world);
private:

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

   map<pair<const Patch*, const Patch*>, ScatterGatherBase* > d_sgDB;


   // Record of which DataWarehouse has the data for each variable...
   //  Allows us to look up the DW to which we will send a data request.
   dataLocationDBtype      d_dataLocation;

   //////////
   // Insert Documentation Here:
   mutable CrowdMonitor    d_lock;
   bool                    d_finalized;
   GridP                   d_grid;

   // Internal VarLabel for the position of this DataWarehouse
   // ??? with respect to what ???? 
   //const VarLabel * d_positionLabel;
};

} // end namespace Uintah

#endif
