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
				   DataWarehouse*);
	 
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&,
					       DataWarehouse*);
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouse*, DataWarehouse*);
private:
    void actuallyComputeStableTimestep(const ProcessorGroup*,
				       const Patch* patch,
				       const DataWarehouse*,
				       DataWarehouse*);
    void findOwners(const ProcessorGroup*,
		    const Patch* patch,
		    const DataWarehouse*,
		    DataWarehouse*);
    void interpolateParticlesToGrid(const ProcessorGroup*,
				    const Patch* patch,
				    const DataWarehouse*,
				    DataWarehouse*);
    void computeStressTensor(const ProcessorGroup*,
			     const Patch* patch,
			     const DataWarehouse*,
			     DataWarehouse*);
    void computeInternalForce(const ProcessorGroup*,
			      const Patch* patch,
			      const DataWarehouse*,
			      DataWarehouse*);
    void solveEquationsMotion(const ProcessorGroup*,
			      const Patch* patch,
			      const DataWarehouse*,
			      DataWarehouse*);
    void integrateAcceleration(const ProcessorGroup*,
			       const Patch* patch,
			       const DataWarehouse*,
			       DataWarehouse*);
    void interpolateToParticlesAndUpdate(const ProcessorGroup*,
					 const Patch* patch,
					 const DataWarehouse*,
					 DataWarehouse*);

    ThreadedMPM(const ThreadedMPM&);
    ThreadedMPM& operator=(const ThreadedMPM&);
};
} // End namespace Uintah



#endif
