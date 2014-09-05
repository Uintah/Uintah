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
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>

#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;


class MPMArches : public UintahParallelComponent, public SimulationInterface {
public:
  MPMArches(const ProcessorGroup* myworld);
  virtual ~MPMArches();
	 
  // Read inputs from ups file for MPMArches case
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP&);

  // Set up initial conditions for MPMArches problem	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);

  // Interpolate particle information from particles to grid
  // for the initial condition
  virtual void scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls);

  //////////
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&);
	 
  //////////
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

  // Interpolate Particle variables from Nodes to cell centers
  void scheduleInterpolateNCToCC(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls);

  // Interpolate relevant particle variables from cell center to
  // appropriate face centers
  void scheduleInterpolateCCToFC(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls);

  // Calculate gas void fraction by subtraction of solid volume
  // fraction from one
  void scheduleComputeVoidFrac(SchedulerP& sched,
			       const PatchSet* patches,
			       const MaterialSet* arches_matls,
			       const MaterialSet* mpm_matls,
			       const MaterialSet* all_matls);

  // Calculate momentum exchange terms for gas-solid interface
  void scheduleMomExchange(SchedulerP& sched,
			   const PatchSet* patches,
			   const MaterialSet* arches_matls,
			   const MaterialSet* mpm_matls,
			   const MaterialSet* all_matls);

  // Calculate heat exchange terms for gas-solid interface
  void scheduleEnergyExchange(SchedulerP& sched,
			      const PatchSet* patches,
			      const MaterialSet* arches_matls,
			      const MaterialSet* mpm_matls,
			      const MaterialSet* all_matls);

  // Interpolate all momentum and energy exchange sources from
  // face centers and accumulate with cell-center sources
  void schedulePutAllForcesOnCC(SchedulerP& sched,
			        const PatchSet* patches,
			        const MaterialSet* mpm_matls);

  // Interpolate all momentum and energy sources from cell-centers
  // to node centers for MPM use
  void schedulePutAllForcesOnNC(SchedulerP& sched,
			        const PatchSet* patches,
			        const MaterialSet* mpm_matls);

 protected:

  void interpolateParticlesToGrid(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

  void interpolateNCToCC(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw,
			 DataWarehouse* new_dw);

  void interpolateCCToFC(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw,
			 DataWarehouse* new_dw);

  void computeVoidFrac(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset*,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);


  void doMomExchange(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset*,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);

  void collectToCCGasMomExchSrcs(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);

  void interpolateCCToFCGasMomExchSrcs(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* ,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

#if 0
  void redistributeDragForceFromCCtoFC(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* ,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

#endif

  void doEnergyExchange(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset*,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

  void collectToCCGasEnergyExchSrcs(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset*,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw);

  void putAllForcesOnCC(const ProcessorGroup*,
		        const PatchSubset* patches,
		        const MaterialSubset*,
		        DataWarehouse* old_dw,
		        DataWarehouse* new_dw);

  void putAllForcesOnNC(const ProcessorGroup*,
		        const PatchSubset* patches,
		        const MaterialSubset*,
		        DataWarehouse* old_dw,
		        DataWarehouse* new_dw);

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

  double d_htcoeff;
  bool d_calcEnergyExchange;

};
      
} // End namespace Uintah
      
#endif
