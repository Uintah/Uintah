#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Components/MPM/Contact/Contact.h>

namespace Uintah {
   class VarLabel;
   namespace MPM {
   
     class Fracture;

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
	 SerialMPM( int MpiRank, int MpiProcesses );
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
      private:
	 //////////
	 // Insert Documentation Here:
	 void actuallyInitialize(const ProcessorContext*,
				 const Region* region,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void actuallyComputeStableTimestep(const ProcessorContext*,
					    const Region* region,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void interpolateParticlesToGrid(const ProcessorContext*,
					 const Region* region,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void computeStressTensor(const ProcessorContext*,
				  const Region* region,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);

	 //////////
	 // update the Surface Normal Of Boundary Particles according to their
	 // velocity gradient during the deformation
	 //
	 void updateSurfaceNormalOfBoundaryParticle(
	                            const ProcessorContext*,
				    const Region* region,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);

	 //////////
	 // Insert Documentation Here:
	 void computeInternalForce(const ProcessorContext*,
				   const Region* region,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void solveEquationsMotion(const ProcessorContext*,
				   const Region* region,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void integrateAcceleration(const ProcessorContext*,
				    const Region* region,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);
	 //////////
	 // Insert Documentation Here:
	 void interpolateToParticlesAndUpdate(const ProcessorContext*,
					      const Region* region,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw);

	 //////////
	 // check the stress on each boundary particle to see
	 // if the microcrack will grow.  If fracture occur,
	 // more interior particles become boundary particles
         void crackGrow(const ProcessorContext*,
                        const Region* region,
                        DataWarehouseP& old_dw,
                        DataWarehouseP& new_dw);

	 SerialMPM(const SerialMPM&);
	 SerialMPM& operator=(const SerialMPM&);
	 
	 SimulationStateP d_sharedState;
	 Contact*         d_contactModel;
	 Fracture*        d_fractureModel;
	 
	 const VarLabel* deltLabel;
	 
	 const VarLabel* pDeformationMeasureLabel;
	 const VarLabel* pStressLabel;
	 const VarLabel* pVolumeLabel;
	 const VarLabel* pMassLabel;
	 const VarLabel* pVelocityLabel;
	 const VarLabel* pExternalForceLabel;
	 const VarLabel* pXLabel;
	 const VarLabel* pSurfaceNormalLabel; //for fracture
	 
	 const VarLabel* gMassLabel;
	 const VarLabel* gAccelerationLabel;
	 const VarLabel* gMomExedAccelerationLabel;
	 const VarLabel* gVelocityLabel;
	 const VarLabel* gMomExedVelocityLabel;
	 const VarLabel* gVelocityStarLabel;
	 const VarLabel* gMomExedVelocityStarLabel;
	 const VarLabel* gExternalForceLabel;
	 const VarLabel* gInternalForceLabel;
	 const VarLabel* cSelfContactLabel; //for fracture, CCVariable
	 const VarLabel* cSurfaceNormalLabel; //for fracture, CCVariable
	 
      };
      
   } // end namespace MPM
} // end namespace Uintah
   
//
// $Log$
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

