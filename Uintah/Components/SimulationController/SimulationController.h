

#ifndef UINTAH_HOMEBREW_SimulationController_H
#define UINTAH_HOMEBREW_SimulationController_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/ProblemSpecP.h>

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

#endif
