
#ifndef UINTAH_HOMEBREW_CFDInterface_H
#define UINTAH_HOMEBREW_CFDInterface_H

#include "CFDInterfaceP.h"
#include "DataWarehouseP.h"
#include "Handle.h"
#include "RefCounted.h"
#include "GridP.h"
#include "LevelP.h"
#include "SchedulerP.h"
#include "ProblemSpecP.h"

class CFDInterface : public RefCounted {
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
