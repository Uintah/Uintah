#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

using SCICore::Geometry::Vector;

namespace Uintah {
namespace MPM {
   
class HeatConduction;
class Fracture;
class ThermalContact;

/**************************************

CLASS
   SerialMPM
   
   Short description...

GENERAL INFORMATION

   SerialMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SerialMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SerialMPM : public UintahParallelComponent, public MPMInterface {
public:
  SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP&);
	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&,
				  DataWarehouseP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&,
					     DataWarehouseP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(double t, double dt,
				   const LevelP& level, SchedulerP&,
				   DataWarehouseP&, DataWarehouseP&);

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };
protected:
  //////////
  // Insert Documentation Here:
  void actuallyInitialize(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

  void pleaseSaveParticlesToGrid(const VarLabel* var,
                                 const VarLabel* varweight, int number,
				 DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw);
  //////////
  // Insert Documentation Here:
  void interpolateParticlesToGrid(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);
  //////////
  // Insert Documentation Here:
  void computeStressTensor(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);

  //////////
  // Check to see if any particles are ready to burn
  void checkIfIgnited(const ProcessorGroup*,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);

  //////////
  // Compute the amount of mass of each particle that burns
  // up in a given timestep
  void computeMassRate(const ProcessorGroup*,
		       const Patch* patch,
		       DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw);

  //////////
  // update the Surface Normal Of Boundary Particles according to their
  // velocity gradient during the deformation
  //
  void updateSurfaceNormalOfBoundaryParticle(const ProcessorGroup*,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalForce(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalHeatRate(
			       const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void solveEquationsMotion(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void solveHeatEquations(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void integrateAcceleration(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void integrateTemperatureRate(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void checkLeave(const ProcessorGroup*,
		  const Patch* patch,
		  DataWarehouseP& /*old_dw*/,
		  DataWarehouseP& new_dw);

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  bool             d_burns;

  vector<MPMPhysicalBC*> d_physicalBCs;
};
      
} // end namespace MPM
} // end namespace Uintah
   
//
// $Log$
// Revision 1.48  2000/08/07 00:37:33  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Current implemented force boundary conditions.
//
// Revision 1.47  2000/07/17 23:30:49  tan
// Fixed some problems in MPM heat conduction.
//
// Revision 1.46  2000/07/05 23:43:30  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.45  2000/06/22 21:22:24  tan
// MPMPhysicalModules class is created to handle all the physical modules
// in MPM, currently those physical submodules include HeatConduction,
// Fracture, Contact, and ThermalContact.
//
// Revision 1.44  2000/06/20 18:23:54  tan
// Arranged the physical models implemented in MPM.
//
// Revision 1.43  2000/06/20 04:12:41  tan
// WHen d_thermalContactModel != NULL, heat conduction will be included in MPM
// algorithm.  The d_thermalContactModel is set by ThermalContactFactory according
// to the information in ProblemSpec from input file.
//
// Revision 1.42  2000/06/19 23:52:13  guilkey
// Added boolean d_burns so that certain stuff only gets done
// if a burn model is present.  Not to worry, the if's on this
// are not inside of inner loops.
//
// Revision 1.41  2000/06/17 07:06:33  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.40  2000/06/16 23:23:36  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.39  2000/06/15 21:57:01  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.38  2000/06/08 16:56:51  guilkey
// Added tasks and VarLabels for HE burn model stuff.
//
// Revision 1.37  2000/05/31 18:30:10  tan
// Create linkage to ThermalContact model.
//
// Revision 1.36  2000/05/30 20:18:59  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.35  2000/05/30 18:17:05  dav
// few more fixes
//
// Revision 1.34  2000/05/30 17:07:34  dav
// Removed commented out labels.  Other MPI fixes.  Changed delt to delT so I would stop thinking of it as just delta.
//
// Revision 1.33  2000/05/26 21:37:30  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.32  2000/05/26 17:14:43  tan
// Added solveHeatEquations on grid.
//
// Revision 1.31  2000/05/26 02:27:38  tan
// Added computeHeatRateGeneratedByInternalHeatFlux() for thermal field
// computation.
//
// Revision 1.30  2000/05/25 22:06:21  tan
// A boolean variable d_heatConductionInvolved is set to true when
// heat conduction considered in the simulation.
//
// Revision 1.29  2000/05/23 02:26:53  tan
// Added gSelfContactLabel NCVariable for farcture usage.
//
// Revision 1.28  2000/05/19 23:18:01  guilkey
// Added VarLabel pSurfLabel
//
// Revision 1.27  2000/05/18 18:50:26  jas
// Now using the gravity from the input file.
//
// Revision 1.26  2000/05/16 00:40:52  guilkey
// Added code to do boundary conditions, print out tecplot files, and a
// few other things.  Most of this is now commented out.
//
// Revision 1.25  2000/05/11 20:10:12  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.24  2000/05/10 18:34:00  tan
// Added computations on self-contact cells for cracked surfaces.
//
// Revision 1.23  2000/05/10 05:01:48  tan
// linked to farcture model.
//
// Revision 1.22  2000/05/08 17:04:04  tan
// Added grid VarLabel selfContactLabel
//
// Revision 1.21  2000/05/04 23:40:29  tan
// Added fracture interface to general MPM.
//
// Revision 1.20  2000/05/04 19:10:52  guilkey
// Added code to apply boundary conditions.  This currently calls empty
// functions which will be implemented soon.
//
// Revision 1.19  2000/05/04 17:31:17  tan
//   Add surfaceNormal for boundary particle tracking.
//
// Revision 1.18  2000/05/02 18:41:15  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.17  2000/05/02 17:54:21  sparker
// Implemented more of SerialMPM
//
// Revision 1.16  2000/04/26 06:48:12  sparker
// Streamlined namespaces
//
// Revision 1.15  2000/04/25 22:57:29  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.14  2000/04/24 21:04:24  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.13  2000/04/20 23:20:26  dav
// updates
//
// Revision 1.12  2000/04/20 22:13:41  dav
// making SerialMPM compile
//
// Revision 1.11  2000/04/20 18:56:16  sparker
// Updates to MPM
//
// Revision 1.10  2000/04/19 22:38:16  dav
// Make SerialMPM a UintahParallelComponent
//
// Revision 1.9  2000/04/19 05:26:01  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.8  2000/04/13 06:50:55  sparker
// More implementation to get this to work
//
// Revision 1.7  2000/03/23 20:42:16  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.6  2000/03/17 21:01:50  dav
// namespace mods
//
// Revision 1.5  2000/03/17 09:29:32  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  2000/03/17 02:57:02  dav
// more namespace, cocoon, etc
//
// Revision 1.3  2000/03/15 22:13:04  jas
// Added log and changed header file locations.
//

#endif

