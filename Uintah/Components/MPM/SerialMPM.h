
#ifndef UINTAH_HOMEBREW_SerialMPM_H
#define UINTAH_HOMEBREW_SerialMPM_H

#include <Uintah/Interface/MPMInterface.h>
class ProcessorContext;
class Region;

class SerialMPM : public MPMInterface {
public:
    SerialMPM();
    virtual ~SerialMPM();

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

    SerialMPM(const SerialMPM&);
    SerialMPM& operator=(const SerialMPM&);
};

#endif

// $Log$
// Revision 1.3  2000/03/15 22:13:04  jas
// Added log and changed header file locations.
//
