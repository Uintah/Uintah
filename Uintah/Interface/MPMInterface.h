#ifndef UINTAH_HOMEBREW_MPMInterface_H
#define UINTAH_HOMEBREW_MPMInterface_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/MPMInterfaceP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/ProblemSpecP.h>
#include <Uintah/Interface/SchedulerP.h>

namespace Uintah {
namespace Interface {

using Uintah::Parallel::UintahParallelPort;
using Uintah::Grid::ProblemSpecP;
using Uintah::Grid::LevelP;
using Uintah::Grid::GridP;

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
			      DataWarehouseP&)=0;

    //////////
    // Insert Documentation Here:
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&) = 0;

    //////////
    // Insert Documentation Here:
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&) = 0;
private:
    MPMInterface(const MPMInterface&);
    MPMInterface& operator=(const MPMInterface&);
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
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
