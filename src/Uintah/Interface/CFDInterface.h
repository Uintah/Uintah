#ifndef UINTAH_HOMEBREW_CFDINTERFACE_H
#define UINTAH_HOMEBREW_CFDINTERFACE_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
    
      /**************************************
	
	CLASS
          CFDInterface
	
	  Short description...
	
	GENERAL INFORMATION
	
          CFDInterface.h
	
	  Steven G. Parker
	  Department of Computer Science
	  University of Utah
	
	  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
	  Copyright (C) 2000 SCI Group
	
	KEYWORDS
          CFD_Interface
	
	DESCRIPTION
          Long description...
	
	WARNING
	
	****************************************/
      
   class CFDInterface : public UintahParallelPort {
   public:
      CFDInterface();
      virtual ~CFDInterface();
      
      //////////
      // Insert Documentation Here:
      virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& state) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&,
						 DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleInitialize(const LevelP& level,
				      SchedulerP&,
				      DataWarehouseP&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleTimeAdvance(double t, double dt,
				       const LevelP& level, SchedulerP&,
				       const DataWarehouseP&, DataWarehouseP&) = 0;
   private:
      CFDInterface(const CFDInterface&);
      CFDInterface& operator=(const CFDInterface&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.12  2000/04/26 06:49:10  sparker
// Streamlined namespaces
//
// Revision 1.11  2000/04/24 21:04:39  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.10  2000/04/20 18:56:35  sparker
// Updates to MPM
//
// Revision 1.9  2000/04/19 05:26:17  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.8  2000/04/13 06:51:04  sparker
// More implementation to get this to work
//
// Revision 1.7  2000/04/11 07:10:52  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.6  2000/03/23 20:42:24  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.5  2000/03/23 20:00:16  jas
// Changed the include files, namespace, and using statements to reflect the
// move of ProblemSpec from Grid/ to Interface/.
//
// Revision 1.4  2000/03/17 09:30:02  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:22  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

