#ifndef UINTAH_HOMEBREW_MPMICE_H
#define UINTAH_HOMEBREW_MPMICE_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/MPMCFDInterface.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

namespace Uintah {

using namespace SCIRun;

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

  void scheduleInterpolateNCToCC(const Patch* patch,
                                 SchedulerP&,
                                 DataWarehouseP&,
                                 DataWarehouseP&);

  void scheduleCCMomExchange(const Patch* patch,
                             SchedulerP&,
                             DataWarehouseP&,
                             DataWarehouseP&);

  //////////
  // Insert Documentation Here:
  void interpolateNCToCC(const ProcessorGroup*,
                         const Patch* patch,
                         DataWarehouseP& old_dw,
                         DataWarehouseP& new_dw);

  void doCCMomExchange(const ProcessorGroup*,
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
  MPMICELabel* MIlb;
  bool             d_burns;
  double           d_nextOutputTime;
  double           d_outputInterval;
  SerialMPM*       d_mpm;
  ICE*             d_ice;

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
};
      
} // End namespace Uintah
      
#endif
