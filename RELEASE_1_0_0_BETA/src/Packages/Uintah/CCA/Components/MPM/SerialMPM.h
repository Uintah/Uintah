#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

#include <Packages/Uintah/CCA/Ports/PatchDataAnalyze.h>

namespace Uintah {

using namespace SCIRun;

class HeatConduction;
class Fracture;
class ThermalContact;

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

class SerialMPM : public UintahParallelComponent, public MPMInterface {
public:
  SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();
	 
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

  void setSharedState(SimulationStateP& ssp)
  {
	d_sharedState = ssp;
  };

  void setMPMLabel(MPMLabel* Mlb)
  {
	lb = Mlb;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  void setAnalyze(PatchDataAnalyze* analyze);

private:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  void actuallyInitialize(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw);

  void interpolateParticlesForSaving(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw);


  //////////
  // Insert Documentation Here:
  void computeFracture(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void stressRelease(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);


  //////////
  // Insert Documentation Here:

  void computeConnectivity(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void computeBoundaryContact(
                   const ProcessorGroup*,
		   const Patch* patch,
		   DataWarehouseP& old_dw,
		   DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void carryForwardVariables( const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateParticlesToGrid(const ProcessorGroup*,
				  const Patch* patch,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw);
  //////////
  // Insert Documentation Here:
  void computeStressTensor(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);

  //////////
  // Check to see if any particles are ready to burn
  void checkIfIgnited(const ProcessorGroup*,
		      const Patch* patch,
		      DataWarehouseP& old_dw,
		      DataWarehouseP& new_dw);

  //////////
  // Compute the amount of mass of each particle that burns
  // up in a given timestep
  void computeMassRate(const ProcessorGroup*,
		       const Patch* patch,
		       DataWarehouseP& old_dw,
		       DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalForce(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalHeatRate(
			       const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void solveEquationsMotion(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void solveHeatEquations(const ProcessorGroup*,
			  const Patch* patch,
			  DataWarehouseP& /*old_dw*/,
			  DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void integrateAcceleration(const ProcessorGroup*,
			     const Patch* patch,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void integrateTemperatureRate(const ProcessorGroup*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void setPositions( const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw);

  //////////
  // Insert Documentation Here:
  void checkLeave(const ProcessorGroup*,
		  const Patch* patch,
		  DataWarehouseP& /*old_dw*/,
		  DataWarehouseP& new_dw);

  void scheduleSetPositions(const Patch* patch,
				     SchedulerP&,
				     DataWarehouseP&,
				     DataWarehouseP&);

  void scheduleComputeFracture(const Patch* patch,
				     SchedulerP&,
				     DataWarehouseP&,
				     DataWarehouseP&);

  void scheduleComputeConnectivity(const Patch* patch,
				     SchedulerP&,
				     DataWarehouseP&,
				     DataWarehouseP&);

  void scheduleComputeBoundaryContact(const Patch* patch,
			              SchedulerP&,
				      DataWarehouseP&,
				      DataWarehouseP&);

  void scheduleCarryForwardVariables(const Patch* patch,
					       SchedulerP&,
					       DataWarehouseP&,
					       DataWarehouseP&);

  void scheduleInterpolateParticlesToGrid(const Patch* patch,
					  SchedulerP&,
					  DataWarehouseP&,
					  DataWarehouseP&);

  void scheduleComputeHeatExchange(const Patch* patch,
				   SchedulerP&,
				   DataWarehouseP&,
				   DataWarehouseP&);

  void scheduleExMomInterpolated(const Patch* patch,
				 SchedulerP&,
				 DataWarehouseP&,
				 DataWarehouseP&);

  void scheduleComputeStressTensor(const Patch* patch,
				   SchedulerP&,
				   DataWarehouseP&,
				   DataWarehouseP&);

  void scheduleComputeInternalForce(const Patch* patch,
				    SchedulerP&,
				    DataWarehouseP&,
				    DataWarehouseP&);

  void scheduleComputeInternalHeatRate(const Patch* patch,
				       SchedulerP&,
				       DataWarehouseP&,
				       DataWarehouseP&);

  void scheduleSolveEquationsMotion(const Patch* patch,
				    SchedulerP&,
				    DataWarehouseP&,
				    DataWarehouseP&);

  void scheduleSolveHeatEquations(const Patch* patch,
				  SchedulerP&,
				  DataWarehouseP&,
				  DataWarehouseP&);

  void scheduleIntegrateAcceleration(const Patch* patch,
				     SchedulerP&,
				     DataWarehouseP&,
				     DataWarehouseP&);

  void scheduleIntegrateTemperatureRate(const Patch* patch,
				        SchedulerP&,
				        DataWarehouseP&,
				        DataWarehouseP&);

  void scheduleExMomIntegrated(const Patch* patch,
			       SchedulerP&,
			       DataWarehouseP&,
			       DataWarehouseP&);

  void scheduleInterpolateToParticlesAndUpdate(const Patch* patch,
					       SchedulerP&,
					       DataWarehouseP&,
					       DataWarehouseP&);

  void scheduleComputeMassRate(const Patch* patch,
			       SchedulerP&,
			       DataWarehouseP&,
			       DataWarehouseP&);

  void scheduleInterpolateParticlesForSaving(const Patch* patch,
					     SchedulerP&,
					     DataWarehouseP&,
					     DataWarehouseP&);

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  bool             d_burns;
  double           d_nextOutputTime;
  double           d_outputInterval;

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;

  PatchDataAnalyze*  d_analyze;
};
      
} // end namespace Uintah

#endif
