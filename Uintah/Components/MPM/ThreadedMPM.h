
#ifndef UINTAH_HOMEBREW_ThreadedMPM_H
#define UINTAH_HOMEBREW_ThreadedMPM_H

#include "MPMInterface.h"
class ProcessorContext;
class Region;

class ThreadedMPM : public MPMInterface {
public:
    ThreadedMPM();
    virtual ~ThreadedMPM();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&);
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&);
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
private:
    void actuallyComputeStableTimestep(const ProcessorContext*,
				       const Region* region,
				       const DataWarehouseP&,
				       DataWarehouseP&);
    void findOwners(const ProcessorContext*,
		    const Region* region,
		    const DataWarehouseP&,
		    DataWarehouseP&);
    void interpolateParticlesToGrid(const ProcessorContext*,
				    const Region* region,
				    const DataWarehouseP&,
				    DataWarehouseP&);
    void computeStressTensor(const ProcessorContext*,
			     const Region* region,
			     const DataWarehouseP&,
			     DataWarehouseP&);
    void computeInternalForce(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    void solveEquationsMotion(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    void integrateAcceleration(const ProcessorContext*,
			       const Region* region,
			       const DataWarehouseP&,
			       DataWarehouseP&);
    void interpolateToParticlesAndUpdate(const ProcessorContext*,
					 const Region* region,
					 const DataWarehouseP&,
					 DataWarehouseP&);

    ThreadedMPM(const ThreadedMPM&);
    ThreadedMPM& operator=(const ThreadedMPM&);
};

#endif
