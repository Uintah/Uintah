#ifndef UINTAH_HOMEBREW_THREADEDMPM_H
#define UINTAH_HOMEBREW_THREADEDMPM_H

#include <Uintah/Interface/MPMInterface.h>

namespace Uintah {

namespace Parallel {
  class ProcessorContext;
}

namespace Grid {
  class Region;
}

namespace Components {

/**************************************

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

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/17 21:01:51  dav
// namespace mods
//
//

#endif
