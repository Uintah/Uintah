
#ifndef UINTAH_HOMEBREW_CFDInterface_H
#define UINTAH_HOMEBREW_CFDInterface_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/CFDInterfaceP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>
#include <Uintah/Grid/ProblemSpecP.h>

class CFDInterface : public UintahParallelPort {
public:
    CFDInterface();
    virtual ~CFDInterface();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&)=0;
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&) = 0;
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&) = 0;
private:
    CFDInterface(const CFDInterface&);
    CFDInterface& operator=(const CFDInterface&);
};

#endif
