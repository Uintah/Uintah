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
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

namespace Uintah {

using namespace SCIRun;

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

class SerialMPM : public SimulationInterface, public UintahParallelComponent {
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
	 
  void schedulePrintParticleCount(const LevelP& level, 
                                  SchedulerP& sched);
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, 
				   SchedulerP&, int step, int nsteps );

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

  int get8or27()
  {
       return d_8or27;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  enum IntegratorType {
    Explicit,
    Implicit
  };

protected:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  virtual void materialProblemSetup(const ProblemSpecP& prob_spec, 
				    SimulationStateP& sharedState,
				    MPMLabel* lb, int n8or27,
				    string integrator, bool haveLoadCurve,
				    bool doErosion);
	 
  virtual void actuallyInitialize(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

  void printParticleCount(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

  //////////
  // Initialize particle data with a default values using 
  // a temporary variable
  void setParticleDefaultWithTemp(constParticleVariable<double>& pvar,
                                  ParticleSubset* pset,
                                  DataWarehouse* new_dw,
                                  double val);

  //////////
  // Initialize particle data with a default values in the
  // new datawarehouse
  void setParticleDefault(ParticleVariable<double>& pvar,
                          const VarLabel* label,
                          ParticleSubset* pset,
                          DataWarehouse* new_dw,
                          double val);
  void setParticleDefault(ParticleVariable<Vector>& pvar,
			  const VarLabel* label, 
			  ParticleSubset* pset,
			  DataWarehouse* new_dw,
			  const Vector& val);
  void setParticleDefault(ParticleVariable<Matrix3>& pvar,
			  const VarLabel* label, 
			  ParticleSubset* pset,
			  DataWarehouse* new_dw,
			  const Matrix3& val);

  void scheduleInitializePressureBCs(const LevelP& level,
				     SchedulerP&);
	 
  void countMaterialPointsPerLoadCurve(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

  void initializePressureBC(const ProcessorGroup*,
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
  virtual void computeStressTensor(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw);

  //////////
  // Compute Accumulated Strain Energy
  void computeAccStrainEnergy(const ProcessorGroup*,
			      const PatchSubset*,
			      const MaterialSubset*,
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
  void computeArtificialViscosity(const ProcessorGroup*,
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
  // This task is to be used for setting particle external force
  // and external heat rate.  I'm creating a separate task so that
  // user defined schemes for setting these can be implemented without
  // editing the core routines
  void applyExternalLoads(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  //////////
  // Calculate the rate of evolution of the damping coefficient
  void calculateDampingRate(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  void addNewParticles(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);


  //////////
  // Insert Documentation Here:
  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);


  virtual void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleComputeHeatExchange(               SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleExMomInterpolated(                 SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleComputeStressTensor(       SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleComputeAccStrainEnergy(            SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleComputeInternalForce(      SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleComputeArtificialViscosity(        SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleComputeInternalHeatRate(           SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleSolveEquationsMotion(      SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleSolveHeatEquations(                SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleIntegrateAcceleration(             SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleIntegrateTemperatureRate(          SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleExMomIntegrated(                   SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  void scheduleSetGridBoundaryConditions(         SchedulerP&, const PatchSet*,
                                                  const MaterialSet* matls);
                                                 
  void scheduleApplyExternalLoads(                SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void scheduleAddNewParticles( SchedulerP&, const PatchSet*,
				const MaterialSet*);

  void scheduleCalculateDampingRate(              SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  bool             d_artificial_viscosity;
  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;
  int              d_8or27;  // Number of nodes a particle can interact with
  double           d_min_part_mass; // Minimum particle mass before it's deleted
  double           d_max_vel; // Maxmimum particle velocity before it's deleted
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.

  double           d_artificialDampCoeff; // Artificial damping coefficient
  bool             d_accStrainEnergy; // Flag for accumulating strain energy
  bool             d_useLoadCurves; // Flag for using load curves
  bool             d_doErosion; // Flag to decide whether to erode or not
  std::string      d_erosionAlgorithm; // Algorithm used to erode material points

  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
  bool             d_with_ice;
  bool             d_with_arches;
  IntegratorType d_integrator;

private:

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
};
      
} // end namespace Uintah

#endif
