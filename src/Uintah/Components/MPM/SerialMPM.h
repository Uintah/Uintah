#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>

namespace Uintah {

namespace Parallel {
  class ProcessorContext;
}

namespace Grid {
  class Region;
}

namespace Components {

using Uintah::Interface::MPMInterface;
using Uintah::Interface::DataWarehouseP;
using Uintah::Interface::SchedulerP;
using Uintah::Parallel::ProcessorContext;
using Uintah::Grid::Region;
using Uintah::Grid::LevelP;
using Uintah::Interface::ProblemSpecP;
using Uintah::Grid::GridP;

/**************************************

CLASS
   SerialMPM
   
   Short description...

GENERAL INFORMATION

   SerialMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SerialMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SerialMPM : public MPMInterface {
public:
    SerialMPM();
    virtual ~SerialMPM();

    //////////
    // Insert Documentation Here:
    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&);

    //////////
    // Insert Documentation Here:
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&);

    //////////
    // Insert Documentation Here:
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
private:
    //////////
    // Insert Documentation Here:
    void actuallyComputeStableTimestep(const ProcessorContext*,
				       const Region* region,
				       const DataWarehouseP&,
				       DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void interpolateParticlesToGrid(const ProcessorContext*,
				    const Region* region,
				    const DataWarehouseP&,
				    DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void computeStressTensor(const ProcessorContext*,
			     const Region* region,
			     const DataWarehouseP&,
			     DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void computeInternalForce(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void solveEquationsMotion(const ProcessorContext*,
			      const Region* region,
			      const DataWarehouseP&,
			      DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void integrateAcceleration(const ProcessorContext*,
			       const Region* region,
			       const DataWarehouseP&,
			       DataWarehouseP&);
    //////////
    // Insert Documentation Here:
    void interpolateToParticlesAndUpdate(const ProcessorContext*,
					 const Region* region,
					 const DataWarehouseP&,
					 DataWarehouseP&);

    SerialMPM(const SerialMPM&);
    SerialMPM& operator=(const SerialMPM&);
};

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/03/23 20:42:16  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.6  2000/03/17 21:01:50  dav
// namespace mods
//
// Revision 1.5  2000/03/17 09:29:32  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  2000/03/17 02:57:02  dav
// more namespace, cocoon, etc
//
// Revision 1.3  2000/03/15 22:13:04  jas
// Added log and changed header file locations.
//

#endif

