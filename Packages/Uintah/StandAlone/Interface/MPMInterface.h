#ifndef UINTAH_HOMEBREW_MPMInterface_H
#define UINTAH_HOMEBREW_MPMInterface_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/SchedulerP.h>

namespace Uintah {

/**************************************

CLASS
   MPMInterface
   
   Short description...

GENERAL INFORMATION

   MPMInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class MPMInterface : public UintahParallelPort {
   public:
      MPMInterface();
      virtual ~MPMInterface();
      
      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& state) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, SchedulerP&,
				       const DataWarehouseP&, DataWarehouseP&) = 0;
   private:
      MPMInterface(const MPMInterface&);
      MPMInterface& operator=(const MPMInterface&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.12  2000/04/26 06:49:11  sparker
// Streamlined namespaces
//
// Revision 1.11  2000/04/24 21:04:40  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.10  2000/04/20 18:56:36  sparker
// Updates to MPM
//
// Revision 1.9  2000/04/19 05:26:18  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.8  2000/04/13 06:51:05  sparker
// More implementation to get this to work
//
// Revision 1.7  2000/04/11 07:10:53  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.6  2000/03/23 20:42:24  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.5  2000/03/23 20:00:17  jas
// Changed the include files, namespace, and using statements to reflect the
// move of ProblemSpec from Grid/ to Interface/.
//
// Revision 1.4  2000/03/17 09:30:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
