

#ifndef UINTAH_HOMEBREW_SimulationController_H
#define UINTAH_HOMEBREW_SimulationController_H

#include "ChemistryInterface.h"
#include "CFDInterface.h"
#include "DataWarehouseP.h"
#include "GridP.h"
#include "Handle.h"
#include "Scheduler.h"
#include "MPMInterface.h"
#include "ProblemSpecP.h"

class SimulationController {
public:
    SimulationController(int argc, char* argv[]);
    ~SimulationController();

    void run();
private:
    void problemSetup(const ProblemSpecP&, GridP&);
    void computeStableTimestep(LevelP&, SchedulerP&, DataWarehouseP&);
    void timeAdvance(double t, double delt, LevelP&, SchedulerP&,
		     const DataWarehouseP&, DataWarehouseP&);

    ChemistryInterfaceP chem;
    MPMInterfaceP mpm;
    CFDInterfaceP cfd;
    SchedulerP scheduler;
    
    SimulationController(const SimulationController&);
    SimulationController& operator=(const SimulationController&);
};

#endif
