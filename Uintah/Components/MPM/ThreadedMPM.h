#ifndef UINTAH_HOMEBREW_THREADEDMPM_H
#define UINTAH_HOMEBREW_THREADEDMPM_H

#include <Uintah/Interface/MPMInterface.h>

namespace Uintah {
   class ProcessorGroup;
   class Patch;
   namespace MPM {
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

} // end namespace MPM
} // end namespace Uintah

//
// $Log$
// Revision 1.8  2000/06/17 07:06:33  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/05/30 20:18:59  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/04/26 06:48:12  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/20 18:56:16  sparker
// Updates to MPM
//
// Revision 1.4  2000/04/19 05:26:01  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/03/17 21:01:51  dav
// namespace mods
//
//

#endif
