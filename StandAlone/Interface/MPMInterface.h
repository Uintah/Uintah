
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

class MPMInterface : public UintahParallelPort {
public:
    MPMInterface();
    virtual ~MPMInterface();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&)=0;
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&) = 0;
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&) = 0;
private:
    MPMInterface(const MPMInterface&);
    MPMInterface& operator=(const MPMInterface&);
};

#endif
