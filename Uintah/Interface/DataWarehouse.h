#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H


#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/CCVariableBase.h>
#include <Uintah/Grid/Ghost.h>
#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Grid/ParticleVariableBase.h>
#include <Uintah/Grid/NCVariableBase.h>
#include <Uintah/Grid/SFCXVariableBase.h>
#include <Uintah/Grid/SFCYVariableBase.h>
#include <Uintah/Grid/SFCZVariableBase.h>
#include <Uintah/Grid/ReductionVariableBase.h>
#include <Uintah/Grid/PerPatchBase.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/SchedulerP.h>

#include <iosfwd>

using namespace std;
namespace SCICore {
namespace Geometry {
  class Vector;
}
}

namespace Uintah {
   class OutputContext;
   class ProcessorGroup;
   class VarLabel;
   class ScatterGatherBase;

/**************************************
	
CLASS
   DataWarehouse
	
   Short description...
	
GENERAL INFORMATION
	
   DataWarehouse.h
	
   Steven G. Parker
   Department of Computer Science
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
   Copyright (C) 2000 SCI Group
	
KEYWORDS
   DataWarehouse
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/
      
   class DataWarehouse : public RefCounted {

   public:
      virtual ~DataWarehouse();
      
      DataWarehouseP getTop() const;
      
      virtual void setGrid(const GridP&)=0;

      virtual bool exists(const VarLabel*, int matlIndex, const Patch*) const =0;
      
      // Reduction Variables
      virtual void allocate(ReductionVariableBase&, const VarLabel*,
			    int matlIndex = -1) = 0;
      virtual void get(ReductionVariableBase&, const VarLabel*,
		       int matlIndex = -1) = 0;
      virtual void put(const ReductionVariableBase&, const VarLabel*,
		       int matlIndex = -1) = 0;
      virtual void override(const ReductionVariableBase&, const VarLabel*,
			    int matlIndex = -1) = 0;

      // Scatther/gather.  This will need a VarLabel if anyone but the
      // scheduler ever wants to use it.
      virtual void scatter(ScatterGatherBase*, const Patch*, const Patch*) = 0;
      virtual ScatterGatherBase* gather(const Patch*, const Patch*) = 0;
      
      // Particle Variables
      virtual ParticleSubset* createParticleSubset(particleIndex numParticles,
				        int matlIndex, const Patch*) = 0;
      virtual bool haveParticleSubset(int matlIndex, const Patch*) = 0;
      virtual ParticleSubset* getParticleSubset(int matlIndex,
					const Patch*) = 0;
      virtual ParticleSubset* getParticleSubset(int matlIndex,
			 const Patch*, Ghost::GhostType, int numGhostCells,
			 const VarLabel* posvar) = 0;
      virtual void allocate(ParticleVariableBase&, const VarLabel*,
			    ParticleSubset*) = 0;
      virtual void get(ParticleVariableBase&, const VarLabel*,
		       ParticleSubset*) = 0;
      virtual void put(const ParticleVariableBase&, const VarLabel*) = 0;
      virtual ParticleVariableBase* getParticleVariable(const VarLabel*,
							ParticleSubset*) = 0;
      
      // Node Centered (NC) Variables
      virtual void allocate(NCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(NCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const NCVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*) = 0;
      
      // Cell Centered (CC) Variables
      virtual void allocate(CCVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(CCVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const CCVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*) = 0;

      // Staggered Variables in all three directions (SFCX, SFCY, SFCZ)
      virtual void allocate(SFCXVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCXVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCXVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*) = 0;

      virtual void allocate(SFCYVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCYVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCYVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*) = 0;

      virtual void allocate(SFCZVariableBase&, const VarLabel*,
			    int matlIndex, const Patch*) = 0;
      virtual void get(SFCZVariableBase&, const VarLabel*, int matlIndex,
		       const Patch*, Ghost::GhostType, int numGhostCells) = 0;
      virtual void put(const SFCZVariableBase&, const VarLabel*,
		       int matlIndex, const Patch*) = 0;

      // PerPatch Variables
      virtual void get(PerPatchBase&, const VarLabel*,
				int matlIndex, const Patch*) = 0;
      virtual void put(const PerPatchBase&, const VarLabel*,
				int matlIndex, const Patch*) = 0;
     
      // Remove particles that are no longer relevant
      virtual void deleteParticles(ParticleSubset* delset) = 0;


      virtual void emit(OutputContext&, const VarLabel* label,
			int matlIndex, const Patch* patch) const = 0;

      virtual void emit(ostream& intout, const VarLabel* label,
			int matlIndex = -1) const = 0;

      // For the schedulers
      virtual bool isFinalized() const = 0;
      virtual bool exists(const VarLabel*, const Patch*) const = 0;
      virtual void finalize() = 0;

      int getID() const {
	 return d_generation;
      }
   protected:
      DataWarehouse(const ProcessorGroup* myworld, int generation, DataWarehouseP& parent_dw);
      // These two things should be removed from here if possible - Steve
      const ProcessorGroup* d_myworld;
      const int d_generation;

      
   private:
      
      DataWarehouse(const DataWarehouse&);
      DataWarehouse& operator=(const DataWarehouse&);
      DataWarehouseP& d_parent;
   };

} // end namespace Uintah

//
// $Log$
// Revision 1.43  2000/12/22 00:11:12  jas
// Got rid of the X,Y,Z FCVariable stuff.
//
// Revision 1.42  2000/12/07 01:35:32  witzel
// Nixed the pleaseSave stuff (that is now handle in DataArchiver via
// the problem specification).
//
// Revision 1.41  2000/12/07 00:04:29  witzel
// Change to allow reduction variables for particular materials
//
// Revision 1.40  2000/11/28 04:10:53  jas
// Added X,Y,Z FCVariables and got rid of some compiler warnings.
//
// Revision 1.39  2000/09/26 21:40:54  dav
// made exists() const, removed registerOwnership and findMpiNode as they are not used anymore
//
// Revision 1.38  2000/08/23 22:51:42  dav
// Made d_generation a const
//
// Revision 1.37  2000/07/28 22:45:17  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.36  2000/07/28 03:01:07  rawat
// modified createDatawarehouse and added getTop function
//
// Revision 1.35  2000/07/27 22:39:53  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.34  2000/06/27 23:20:52  rawat
// added staggered variables
//
// Revision 1.33  2000/06/21 20:47:58  guilkey
// Added deleteParticles, a currently empty function that
// will remove irrelevant particles from the simulation.
//
// Revision 1.32  2000/06/17 07:06:46  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.31  2000/06/16 19:46:46  sparker
// Output interface doesn't need old_dw
// DataWarehouse no longer needs carryForward
//
// Revision 1.30  2000/06/16 05:03:11  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.29  2000/06/15 21:57:22  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.28  2000/06/14 23:38:55  jas
// Added FCVariables.
//
// Revision 1.27  2000/06/05 19:45:43  guilkey
// Added some functionality to the DW for PerPatch variables.
//
// Revision 1.26  2000/06/03 05:32:10  sparker
// Removed include of fstream
// Changed include of iostream to iosfwd
//
// Revision 1.25  2000/06/03 05:30:27  sparker
// Changed emit (primary for reduction variables) to use ostream instead
// of ofstream
//
// Revision 1.24  2000/06/01 23:17:53  guilkey
// Added virtual pleaseSaveIntegrated functions.
//
// Revision 1.23  2000/05/30 20:19:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.22  2000/05/15 19:39:52  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.21  2000/05/11 20:10:23  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.20  2000/05/10 20:28:55  tan
// CCVariable is needed.  Virtual functions allocate, get and put are
// set here to make the compilation work.
//
// Revision 1.19  2000/05/10 20:03:05  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.18  2000/05/05 06:42:46  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.17  2000/05/02 17:54:34  sparker
// Implemented more of SerialMPM
//
// Revision 1.16  2000/04/28 07:35:39  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.15  2000/04/27 23:18:51  sparker
// Added problem initialization for MPM
//
// Revision 1.14  2000/04/26 06:49:10  sparker
// Streamlined namespaces
//
// Revision 1.13  2000/04/25 00:41:22  dav
// more changes to fix compilations
//
// Revision 1.12  2000/04/24 15:17:02  sparker
// Fixed unresolved symbols
//
// Revision 1.11  2000/04/21 20:31:25  dav
// added some allocates
//
// Revision 1.10  2000/04/20 18:56:35  sparker
// Updates to MPM
//
// Revision 1.9  2000/04/19 21:20:04  dav
// more MPI stuff
//
// Revision 1.8  2000/04/19 05:26:17  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.7  2000/04/13 06:51:05  sparker
// More implementation to get this to work
//
// Revision 1.6  2000/04/11 07:10:53  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.5  2000/03/22 00:37:17  sparker
// Added accessor for PerPatch data
//
// Revision 1.4  2000/03/17 18:45:43  dav
// fixed a few more namespace problems
//
// Revision 1.3  2000/03/16 22:08:22  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
