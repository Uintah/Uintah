#ifndef UINTAH_HOMEBREW_THREADEDMPM_H
#define UINTAH_HOMEBREW_THREADEDMPM_H

#include <Packages/Uintah/CCA/Ports/MPMInterface.h>

namespace Uintah {

CLASS
   ThreadedMPM
   
   Short description...

GENERAL INFORMATION

   ThreadedMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Threaded

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ThreadedMPM : public MPMInterface {
public:
    ThreadedMPM();
    virtual ~ThreadedMPM();

   virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			     const SimulationStateP&);

   virtual void scheduleInitialize(const LevelP& level,
				   SchedulerP&,
				   DataWarehouseP&);
	 
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&,
					       DataWarehouseP&);
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
private:
    void actuallyComputeStableTimestep(const ProcessorGroup*,
				       const Patch* patch,
				       const DataWarehouseP&,
				       DataWarehouseP&);
    void findOwners(const ProcessorGroup*,
		    const Patch* patch,
		    const DataWarehouseP&,
		    DataWarehouseP&);
    void interpolateParticlesToGrid(const ProcessorGroup*,
				    const Patch* patch,
				    const DataWarehouseP&,
				    DataWarehouseP&);
    void computeStressTensor(const ProcessorGroup*,
			     const Patch* patch,
			     const DataWarehouseP&,
			     DataWarehouseP&);
    void computeInternalForce(const ProcessorGroup*,
			      const Patch* patch,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    void solveEquationsMotion(const ProcessorGroup*,
			      const Patch* patch,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    void integrateAcceleration(const ProcessorGroup*,
			       const Patch* patch,
			       const DataWarehouseP&,
			       DataWarehouseP&);
    void interpolateToParticlesAndUpdate(const ProcessorGroup*,
					 const Patch* patch,
					 const DataWarehouseP&,
					 DataWarehouseP&);

    ThreadedMPM(const ThreadedMPM&);
    ThreadedMPM& operator=(const ThreadedMPM&);
};
} // End namespace Uintah



#endif
