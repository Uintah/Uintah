#ifndef Uintah_Component_MPMArches_h
#define Uintah_Component_MPMArches_h

/**************************************

CLASS
   MPMArches
   
   Short description...

GENERAL INFORMATION

   MPMArches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 University of Utah

KEYWORDS
   MPMArches

DESCRIPTION
   Long description...
  
WARNING

****************************************/

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
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>

#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;


class MPMArches : public UintahParallelComponent, public MPMCFDInterface {
public:
  MPMArches(const ProcessorGroup* myworld);
  virtual ~MPMArches();
	 
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


  void scheduleInterpolateNCToCC(const LevelP& level,
				 SchedulerP&,
				 DataWarehouseP&,
				 DataWarehouseP&);

  void scheduleInterpolateCCToFC(const LevelP& level,
				 SchedulerP&,
				 DataWarehouseP&,
				 DataWarehouseP&);


  void scheduleComputeVoidFrac(const LevelP& level,
			       SchedulerP& sched,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

  void scheduleMomExchange(const LevelP& level,
			   SchedulerP&,
			   DataWarehouseP&,
			   DataWarehouseP&);



protected:

  void interpolateNCToCC(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);

  void interpolateCCToFC(const ProcessorGroup*,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw);

  void computeVoidFrac(const ProcessorGroup*,
		       const Patch* patch,
		       DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw) ;


  void doMomExchange(const ProcessorGroup*,
		     const Patch* patch,
		     DataWarehouseP& old_dw,
		     DataWarehouseP& new_dw);

  double d_SMALL_NUM;
  // GROUP: Constructors (Private):
  ////////////////////////////////////////////////////////////////////////
  // Default MPMArches constructor
  MPMArches();
  ////////////////////////////////////////////////////////////////////////
  // MPMArches copy constructor

  MPMArches(const MPMArches&);
  MPMArches& operator=(const MPMArches&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* Mlb;
  const ArchesLabel* d_Alab;
  const MPMArchesLabel* d_MAlb;
  SerialMPM*       d_mpm;
  Arches*          d_arches;
  bool             d_fracture;
};
      
} // End namespace Uintah
      
#endif
