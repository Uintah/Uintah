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
   
/**************************************

CLASS
   SerialMD
   
   Short description...

GENERAL INFORMATION

   SerialMD.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SerialMD

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SerialMD : public UintahParallelComponent, public MDInterface {
public:
  SerialMD( int MpiRank, int MpiProcesses);
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
// Revision 1.1  2000/06/09 18:01:51  tan
// Create SerialMD to do molecular dynamics simulations.
//
