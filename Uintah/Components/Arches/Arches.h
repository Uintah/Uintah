#ifndef UINTAH_COMPONENT_ARCHES_ARCHES_H
#define UINTAH_COMPONENT_ARCHES_ARCHES_H

/*
 * Placeholder - nothing here yet
 */

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>

namespace Uintah {
namespace Components {

using Uintah::Parallel::UintahParallelComponent;
using Uintah::Interface::CFDInterface;

/**************************************

CLASS
   Arches
   
   Short description...

GENERAL INFORMATION

   Arches.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Arches

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Arches : public UintahParallelComponent, public CFDInterface {
public:
    Arches();
    virtual ~Arches();

    //////////
    // Insert Documentation Here:
    virtual void problemSetup(const ProblemSpecP& params, 
			      GridP& grid,
			      DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&,
				       DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    virtual void timeStep(double t, 
			  double dt,
			  const LevelP& level,
			  SchedulerP&,
			  const DataWarehouseP&,
			  DataWarehouseP&);

private:
    Arches(const Arches&);
    Arches& operator=(const Arches&);
};

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:26:14  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

