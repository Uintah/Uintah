#ifndef UINTAH_HOMEBREW_SERIALMD_H
#define UINTAH_HOMEBREW_SERIALMD_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/MDInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;

class SerialMD : public UintahParallelComponent, public MDInterface {
public:
  SerialMD( const ProcessorGroup* myworld );
  virtual ~SerialMD();
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP&);
	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&,
				  DataWarehouseP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&,
					     DataWarehouseP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(double t, double dt,
				   const LevelP& level, SchedulerP&,
				   DataWarehouseP&, DataWarehouseP&);

private:

  SerialMD(const SerialMD&);
  SerialMD& operator=(const SerialMD&);
	 
  SimulationStateP d_sharedState;
};
} // End namespace Uintah
      

#endif
   
