#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <SCICore/Thread/CrowdMonitor.h>

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Components/Schedulers/DWDatabase.h>

#include <map>
#include <iosfwd>
#include <vector>
#include <mpi.h>

using std::vector;
using std::map;

using SCICore::Thread::CrowdMonitor;

namespace Uintah {

class DataItem;
class TypeDescription;
class Patch;
class ProcessorGroup;

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

   // Reduction Variables
   virtual void allocate(ReductionVariableBase&, const VarLabel*);
   virtual void get(ReductionVariableBase&, const VarLabel*);
   virtual void put(const ReductionVariableBase&, const VarLabel*);
   virtual void override(const ReductionVariableBase&, const VarLabel*);

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
			 int matlIndex, const Patch*);
   virtual void get(NCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const NCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // CCVariables Variables -- fron Tan... need to be fixed...
   virtual void allocate(CCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(CCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const CCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // FC Variables -- fron jas ... need to be fixed...
   virtual void allocate(XFCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(XFCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const XFCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // FC Variables -- fron jas ... need to be fixed...
   virtual void allocate(YFCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(YFCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const YFCVariableBase&, const VarLabel*,
		    int matlIndex, const Patch*);

   // FC Variables -- fron jas ... need to be fixed...
   virtual void allocate(ZFCVariableBase&, const VarLabel*,
			 int matlIndex, const Patch*);
   virtual void get(ZFCVariableBase&, const VarLabel*, int matlIndex,
		    const Patch*, Ghost::GhostType, int numGhostCells);
   virtual void put(const ZFCVariableBase&, const VarLabel*,
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

   //////////
   // Adds a variable to the save set
   virtual void pleaseSave(const VarLabel* label, int number);
       
   // Adds a variable to the integrated quantities save set
   virtual void pleaseSaveIntegrated(const VarLabel* label);

   //////////
   // Retrieves the saveset
   virtual void getSaveSet(vector<const VarLabel*>&, vector<int>&) const;
       
   virtual void getIntegratedSaveSet(vector<const VarLabel*>&) const;
       
   virtual void emit(OutputContext&, const VarLabel* label,
		     int matlIndex, const Patch* patch) const;

   virtual void emit(ostream& intout, const VarLabel* label) const;

   void sendMPI(const VarLabel* label, int matlIndex,
		const Patch* patch, const ProcessorGroup* world,
		int dest, int tag, int* size, MPI_Request* requestid);
   void recvMPI(DataWarehouseP& old_dw, 
		const VarLabel* label, int matlIndex,
		const Patch* patch, const ProcessorGroup* world,
		int dest, int tag, int* size, MPI_Request* requestid);
   void reduceMPI(const VarLabel* label, const ProcessorGroup* world);
private:

   struct dataLocation {
      const Patch   * patch;
            int        mpiNode;
   };

   struct ReductionRecord {
      ReductionVariableBase* var;
      ReductionRecord(ReductionVariableBase*);
   };

   typedef vector<dataLocation*> variableListType;
   typedef map<const VarLabel*, ReductionRecord*, VarLabel::Compare> reductionDBtype;
   typedef map<const VarLabel*, variableListType*, VarLabel::Compare> dataLocationDBtype;
   typedef map<pair<int, const Patch*>, ParticleSubset*> psetDBType;

   DWDatabase<NCVariableBase>       d_ncDB;
   DWDatabase<CCVariableBase>       d_ccDB;
   DWDatabase<XFCVariableBase>       d_xfcDB;
   DWDatabase<YFCVariableBase>       d_yfcDB;
   DWDatabase<ZFCVariableBase>       d_zfcDB;
   DWDatabase<SFCXVariableBase>     d_sfcxDB;
   DWDatabase<SFCYVariableBase>     d_sfcyDB;
   DWDatabase<SFCZVariableBase>     d_sfczDB;
   DWDatabase<ParticleVariableBase> d_particleDB;
   reductionDBtype                  d_reductionDB;
   DWDatabase<PerPatchBase>         d_perpatchDB;
   psetDBType			    d_psetDB;

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

   vector<const VarLabel*> d_saveset;
   vector<const VarLabel*> d_saveset_integrated;
   vector<int>             d_savenumbers;
};

} // end namespace Uintah

//
// $Log$
// Revision 1.40  2000/11/28 03:55:22  jas
// Added X,Y,Z FCVariables to the data warehouse.
//
// Revision 1.39  2000/09/28 02:15:51  dav
// updates due to not sending 0 particles
//
// Revision 1.38  2000/09/27 02:10:48  dav
// removed registerOwnership and findMpiNode which are nolonger used
//
// Revision 1.37  2000/09/20 16:00:28  sparker
// Added external interface to LoadBalancer (for per-processor tasks)
// Added message logging functionality. Put the tag <MessageLog/> in
//    the ups file to enable
//
// Revision 1.36  2000/08/24 21:04:33  dav
// Removed DWMpiHandler Stuff
//
// Revision 1.35  2000/07/28 22:45:15  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.33  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.32  2000/06/27 23:20:03  rawat
// added functions to deal with staggered cell variables. Also modified get function
// for CCVariables.
//
// Revision 1.31  2000/06/21 20:50:03  guilkey
// Added deleteParticles, a currently empty function that will remove
// particles that are no longer relevant to the simulation.
//
// Revision 1.30  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.29  2000/06/16 19:48:55  sparker
// Eliminated carryForward
//
// Revision 1.28  2000/06/16 05:03:08  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.27  2000/06/15 21:57:12  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.26  2000/06/14 23:39:26  jas
// Added FCVariables.
//
// Revision 1.25  2000/06/05 19:50:22  guilkey
// Added functionality for PerPatch variable.
//
// Revision 1.24  2000/06/03 05:27:24  sparker
// Fixed dependency analysis for reduction variables
// Removed warnings
// Now allow for task patch to be null
// Changed DataWarehouse emit code
//
// Revision 1.23  2000/06/01 23:14:04  guilkey
// Added pleaseSaveIntegrated functionality to save ReductionVariables
// to an archive.
//
// Revision 1.22  2000/05/30 20:19:23  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.21  2000/05/30 17:09:38  dav
// MPI stuff
//
// Revision 1.20  2000/05/15 20:04:36  dav
// Mpi Stuff
//
// Revision 1.19  2000/05/15 19:39:43  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.18  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.17  2000/05/10 20:02:53  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.16  2000/05/07 06:02:08  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.15  2000/05/05 06:42:43  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.14  2000/05/02 17:54:29  sparker
// Implemented more of SerialMPM
//
// Revision 1.13  2000/05/02 06:07:16  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.12  2000/04/28 07:35:34  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.11  2000/04/27 23:18:48  sparker
// Added problem initialization for MPM
//
// Revision 1.10  2000/04/26 06:48:33  sparker
// Streamlined namespaces
//
// Revision 1.9  2000/04/24 15:17:01  sparker
// Fixed unresolved symbols
//
// Revision 1.8  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.7  2000/04/19 21:20:03  dav
// more MPI stuff
//
// Revision 1.6  2000/04/19 05:26:11  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/13 06:50:57  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/03/22 00:36:37  sparker
// Added new version of getPatchData
//
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//

#endif
