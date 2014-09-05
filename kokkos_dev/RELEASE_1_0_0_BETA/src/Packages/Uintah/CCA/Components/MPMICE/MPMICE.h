#ifndef UINTAH_HOMEBREW_MPMICE_H
#define UINTAH_HOMEBREW_MPMICE_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/MPMCFDInterface.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>

#include <Core/Geometry/Vector.h>

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

                            
  void scheduleInterpolateNCToCC_0(const Patch* patch,
                                   SchedulerP&,
                                   DataWarehouseP&,
                                   DataWarehouseP&);

  void scheduleInterpolateVelIncFCToNC(const Patch* patch,
                                       SchedulerP&,
                                       DataWarehouseP&,
                                       DataWarehouseP&);

  void scheduleInterpolateNCToCC(const Patch* patch,
                                 SchedulerP&,
                                 DataWarehouseP&,
                                 DataWarehouseP&);

  void scheduleCCMomExchange(const Patch* patch,
                             SchedulerP&,
                             DataWarehouseP&,
                             DataWarehouseP&);

  void scheduleInterpolateCCToNC(const Patch* patch,
				 SchedulerP&,
				 DataWarehouseP&,
				 DataWarehouseP&);

  void scheduleComputeEquilibrationPressure(const Patch* patch,
					    SchedulerP&,
					    DataWarehouseP&,
					    DataWarehouseP&);


  void scheduleInterpolatePressCCToPressNC(const Patch* patch,
					   SchedulerP&,
					   DataWarehouseP&,
					   DataWarehouseP&);

  void scheduleInterpolatePAndGradP(const Patch* patch,
				    SchedulerP&,
				    DataWarehouseP&,
				    DataWarehouseP&);
                                
  void  scheduleMassExchange(const Patch* patch, 
                             SchedulerP&,
                             DataWarehouseP&,
                             DataWarehouseP&);


//______________________________________________________________________
//       A C T U A L   S T E P S : 
                          
                                                    
  void interpolateNCToCC_0(const ProcessorGroup*,
                           const Patch* patch,
                           DataWarehouseP& old_dw,
                           DataWarehouseP& new_dw);

  void interpolateVelIncFCToNC(const ProcessorGroup*,
                               const Patch* patch,
                               DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw);

  void interpolateNCToCC(const ProcessorGroup*,
                         const Patch* patch,
                         DataWarehouseP& old_dw,
                         DataWarehouseP& new_dw);

  void computeEquilibrationPressure(const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP&, 
				    DataWarehouseP&);

  void doCCMomExchange(const ProcessorGroup*,
                       const Patch* patch,
                       DataWarehouseP& old_dw,
                       DataWarehouseP& new_dw);

  void interpolateCCToNC(const ProcessorGroup*,
                         const Patch* patch,
                         DataWarehouseP& old_dw,
                         DataWarehouseP& new_dw);

  void interpolatePressCCToPressNC(const ProcessorGroup*,
				   const Patch* patch,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw);


  void interpolatePAndGradP(const ProcessorGroup*,
                            const Patch* patch,
                            DataWarehouseP& old_dw,
                            DataWarehouseP& new_dw);
                            
  void massExchange(const ProcessorGroup*,
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
  SerialMPM*       d_mpm;
  ICE*             d_ice;
  vector<double>   d_K_mom, d_K_heat;

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
  double d_SMALL_NUM;
};
      
} // End namespace Uintah
      
#endif
