
#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/*
 * Placeholder - nothing here yet
 */

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
class NonlinearSolver;
const int NDIM = 3;
class Arches : public UintahParallelComponent, public CFDInterface {
public:
    Arches();
    virtual ~Arches();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&);
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&);
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
    void Arches::actuallyComputeStableTimestep(const LevelP& level,
			   DataWarehouseP& dw);
    void Arches::advanceTimeStep(const ProcessorContext*,
			     const Region* region,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);
private:
    Arches(const Arches&);
    Arches& operator=(const Arches&);
    double d_deltaT;
    NonlinearSolver* d_nlSolver;
};

#endif

