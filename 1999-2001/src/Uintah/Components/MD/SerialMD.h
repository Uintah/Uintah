#ifndef UINTAH_HOMEBREW_SERIALMD_H
#define UINTAH_HOMEBREW_SERIALMD_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/MDInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <SCICore/Geometry/Vector.h>

using SCICore::Geometry::Vector;

namespace Uintah {
namespace MD {
   
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
      
} // end namespace MD
} // end namespace Uintah

#endif
   
//
// $Log$
// Revision 1.2  2000/06/20 01:55:55  tan
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.1  2000/06/09 18:01:51  tan
// Create SerialMD to do molecular dynamics simulations.
//
