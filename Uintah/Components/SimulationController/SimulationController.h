#ifndef UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H
#define UINTAH_HOMEBREW_SIMULATIONCONTROLLER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/ProblemSpecP.h>

namespace Uintah {
namespace Components {

using Uintah::Parallel::UintahParallelComponent;
using Uintah::Grid::ProblemSpecP;
using Uintah::Grid::LevelP;
using Uintah::Grid::GridP;
using Uintah::Interface::SchedulerP;
using Uintah::Interface::DataWarehouseP;

/**************************************

CLASS
   SimulationController
   
   Short description...

GENERAL INFORMATION

   SimulationController.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Simulation_Controller

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SimulationController : public UintahParallelComponent {
public:
    SimulationController();
    virtual ~SimulationController();

    void run();
private:
    void problemSetup(const ProblemSpecP&, GridP&);
    void computeStableTimestep(LevelP&, SchedulerP&, DataWarehouseP&);
    void timeAdvance(double t, double delt, LevelP&, SchedulerP&,
		     const DataWarehouseP&, DataWarehouseP&);

    SimulationController(const SimulationController&);
    SimulationController& operator=(const SimulationController&);
};

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/17 20:58:31  dav
// namespace updates
//
//

#endif
