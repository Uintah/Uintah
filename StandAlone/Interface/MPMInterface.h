
#ifndef UINTAH_HOMEBREW_MPMInterface_H
#define UINTAH_HOMEBREW_MPMInterface_H

#include "MPMInterfaceP.h"
#include "DataWarehouseP.h"
#include "GridP.h"
#include "Handle.h"
#include "LevelP.h"
#include "RefCounted.h"
#include "ProblemSpecP.h"
#include "SchedulerP.h"

class MPMInterface : public RefCounted {
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
