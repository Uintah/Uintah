#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

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

class SerialMPM : public UintahParallelComponent, public SimulationInterface {
public:
  SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();

  Contact*         contactModel;
  ThermalContact*  thermalContactModel;
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP&);
	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

  void setSharedState(SimulationStateP& ssp);

  void setMPMLabel(MPMLabel* Mlb)
  {
        delete lb;
	lb = Mlb;
  };

  void setWithICE()
  {
	d_with_ice = true;
  };

  void setWithArches()
  {
	d_with_arches = true;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  bool withFracture() const
  {
    return d_fracture;
  };

private:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  void actuallyInitialize(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

  void printParticleCount(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

  void interpolateParticlesForSaving(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* matls,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);


  //////////
  // Insert Documentation Here:
  void computeCrackExtension(
                   const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* ,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeFracture(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void stressRelease(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);


  //////////
  // Insert Documentation Here:
  void computeConnectivity(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeBoundaryContact(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void carryForwardVariables( const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* matls,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateParticlesToGrid(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here:
  void computeStressTensor(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

  //////////
  // Check to see if any particles are ready to burn
  void checkIfIgnited(const ProcessorGroup*,
		      const PatchSubset* patches,
		      const MaterialSubset* matls,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw);

  //////////
  // Compute the amount of mass of each particle that burns
  // up in a given timestep
  void computeMassRate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalForce(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeInternalHeatRate(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void solveEquationsMotion(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void solveHeatEquations(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* /*old_dw*/,
			  DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void integrateAcceleration(const ProcessorGroup*,
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void integrateTemperatureRate(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here:                            
  void setGridBoundaryConditions(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here:
  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void setPositions( const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void checkLeave(const ProcessorGroup*,
		  const PatchSubset* patches,
		  const MaterialSubset* matls,
		  DataWarehouse* /*old_dw*/,
		  DataWarehouse* new_dw);

  void scheduleSetPositions(SchedulerP&, const PatchSet*, const MaterialSet*);

  void scheduleComputeCrackExtension(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleComputeFracture(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleComputeConnectivity(SchedulerP&, const PatchSet*,
				   const MaterialSet*);

  void scheduleComputeBoundaryContact(SchedulerP&, const PatchSet*,
				      const MaterialSet*);

  void scheduleCarryForwardVariables(SchedulerP&, const PatchSet*,
				     const MaterialSet*);

  void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
					  const MaterialSet*);

  void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
				   const MaterialSet*);

  void scheduleExMomInterpolated(SchedulerP&, const PatchSet*,
				 const MaterialSet*);

  void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
				   const MaterialSet*);

  void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
				    const MaterialSet*);

  void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
				       const MaterialSet*);

  void scheduleSolveEquationsMotion(SchedulerP&, const PatchSet*,
				    const MaterialSet*);

  void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
				  const MaterialSet*);

  void scheduleIntegrateAcceleration(SchedulerP&, const PatchSet*,
				     const MaterialSet*);

  void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
					const MaterialSet*);

  void scheduleExMomIntegrated(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls);
                                                 
  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, const PatchSet*,
					       const MaterialSet*);

  void scheduleComputeMassRate(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleInterpolateParticlesForSaving(SchedulerP&, const PatchSet*,
					     const MaterialSet*);

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  bool             d_burns;
  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
  bool             d_with_ice;
  bool             d_with_arches;
};
      
} // end namespace Uintah

#endif
