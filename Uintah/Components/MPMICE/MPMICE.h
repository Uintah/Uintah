#ifndef UINTAH_HOMEBREW_MPMICE_H
#define UINTAH_HOMEBREW_MPMICE_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/MPMCFDInterface.h>
#include <Uintah/Interface/MPMInterface.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

using SCICore::Geometry::Vector;
using namespace Uintah::MPM;
using namespace Uintah::ICESpace;

namespace Uintah {
namespace MPMICESpace {
   
/**************************************

CLASS
   MPMICE
   
   Short description...

GENERAL INFORMATION

   MPMICE.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPMICE

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class MPMICE : public UintahParallelComponent, public MPMCFDInterface {
public:
  MPMICE(const ProcessorGroup* myworld);
  virtual ~MPMICE();
	 
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

  //////////
  // Insert Documentation Here:
  void interpolateNCToCC(const ProcessorGroup*,
                         const Patch* patch,
                         DataWarehouseP& old_dw,
                         DataWarehouseP& new_dw);

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

protected:

  MPMICE(const MPMICE&);
  MPMICE& operator=(const MPMICE&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* Mlb;
  ICELabel* Ilb;
  bool             d_burns;
  double           d_nextOutputTime;
  double           d_outputInterval;
  SerialMPM*       d_mpm;
  ICE*             d_ice;

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
};
      
} // end namespace MPMICE
} // end namespace Uintah
   
#endif

