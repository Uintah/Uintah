#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
// put here to avoid template problems
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {

using namespace SCIRun;

class ThermalContact;
class HeatConduction;

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
  HeatConduction* heatConductionModel;
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP&,
			    SimulationStateP&);
	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);

  virtual void addMaterial(const ProblemSpecP& params,
                           GridP& grid,
                           SimulationStateP&);

  virtual void scheduleInitializeAddedMaterial(const LevelP& level,
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

  void scheduleRefine(const PatchSet* patches, SchedulerP& scheduler);

  void scheduleRefineInterface(const LevelP& fineLevel, SchedulerP& scheduler,
                               int step, int nsteps);

  void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark flags for AMR regridding
  void scheduleErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched);
  
  /// Schedule to mark initial flags for AMR regridding
  void scheduleInitialErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched);

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

  int get8or27()
  {
       return flags->d_8or27;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  enum IntegratorType {
    Explicit,
    Implicit,
    Fracture
  };

protected:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  virtual void materialProblemSetup(const ProblemSpecP& prob_spec, 
				    SimulationStateP& sharedState,
				    MPMLabel* lb, MPMFlags* flags);
	 
  virtual void actuallyInitialize(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

  virtual void actuallyInitializeAddedMaterial(const ProcessorGroup*,
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

  void printParticleLabels(vector<const VarLabel*> label,DataWarehouse* dw,
			   int dwi, const Patch* patch);

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
  virtual void interpolateParticlesToGrid(const ProcessorGroup*,
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
  // Insert Documentation Here: for thermal stress analysis
  virtual void computeParticleTempFromGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw);  

  /*! Update the erosion parameter is mass is to be removed */
  void updateErosionParameter(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
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
  virtual void computeContactArea(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);
  
  virtual void computeInternalForce(const ProcessorGroup*,
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
  virtual void solveEquationsMotion(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void integrateAcceleration(const ProcessorGroup*,
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


  /*!  Convert the localized particles into particles of a new material
       with a different velocity field */
  void convertLocalizedParticles(const ProcessorGroup*,
		                 const PatchSubset* patches,
		                 const MaterialSubset* matls,
		                 DataWarehouse* old_dw,
		                 DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void interpolateToParticlesAndUpdate(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

  void refine(const ProcessorGroup*,
              const PatchSubset* patches,
              const MaterialSubset* matls,
              DataWarehouse*,
              DataWarehouse* new_dw);

  void errorEstimate(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw);

  void initialErrorEstimate(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse* new_dw);

  virtual void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
					   const MaterialSet*);

  virtual void scheduleExMomInterpolated(SchedulerP&, const PatchSet*,
					 const MaterialSet*);

  virtual void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
					   const MaterialSet*);
  
  // for thermal stress analysis
  virtual void scheduleComputeParticleTempFromGrid(SchedulerP&, const PatchSet*,
                                           const MaterialSet*); 

  void scheduleUpdateErosionParameter(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls);
  void scheduleComputeAccStrainEnergy(SchedulerP&, const PatchSet*,
				      const MaterialSet*);

  virtual void scheduleComputeContactArea(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);
  
  virtual void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
					    const MaterialSet*);

  void scheduleComputeArtificialViscosity(SchedulerP&, const PatchSet*,
					  const MaterialSet*);

  virtual void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
					       const MaterialSet*);

  virtual void scheduleSolveEquationsMotion(SchedulerP&, const PatchSet*,
					    const MaterialSet*);

  virtual void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
					  const MaterialSet*);

  virtual void scheduleIntegrateAcceleration(SchedulerP&, const PatchSet*,
					     const MaterialSet*);

  virtual void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleExMomIntegrated(SchedulerP&, const PatchSet*,
				       const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP&, const PatchSet*,
					 const MaterialSet* matls);
                                                 
  void scheduleApplyExternalLoads(SchedulerP&, const PatchSet*,
				  const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
						       const PatchSet*,
						       const MaterialSet*);

  void scheduleAddNewParticles(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleConvertLocalizedParticles(SchedulerP&, const PatchSet*,
					 const MaterialSet*);

  void scheduleCalculateDampingRate(SchedulerP&, const PatchSet*,
				    const MaterialSet*);

  virtual void scheduleParticleVelocityField(SchedulerP&, const PatchSet*,
					     const MaterialSet*);

  virtual void scheduleAdjustCrackContactInterpolated(SchedulerP&, 
						      const PatchSet*,
						      const MaterialSet*);

  virtual void scheduleAdjustCrackContactIntegrated(SchedulerP&, 
						    const PatchSet*,
						    const MaterialSet*);

  virtual void scheduleCalculateFractureParameters(SchedulerP&,const PatchSet*,
						   const MaterialSet*);

  virtual void scheduleDoCrackPropagation(SchedulerP& sched, 
					  const PatchSet* patches,
					  const MaterialSet* matls);

  virtual void scheduleMoveCracks(SchedulerP& sched,const PatchSet* patches,
				  const MaterialSet* matls);

  virtual void scheduleUpdateCrackFront(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls);

   void scheduleCheckNeedAddMPMMaterial(SchedulerP&,
					const PatchSet* patches,
                                        const MaterialSet*);
                                                                             
  //////////
  // Insert Documentation Here:
  virtual void checkNeedAddMPMMaterial(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

   void scheduleSetNeedAddMaterialFlag(SchedulerP&,
                                       const LevelP& level,
                                       const MaterialSet*);


   void setNeedAddMaterialFlag(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse*);

      virtual bool needRecompile(double time, double dt,
                                 const GridP& grid);

  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;
  Output* dataArchiver;

  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;
  bool             d_doGridReset;  // Default is true, standard MPM
  double           d_min_part_mass; // Minimum particle mass before it's deleted
  double           d_max_vel; // Maxmimum particle velocity before it's deleted
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  
  list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, yminus, ...
  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
  bool             d_with_ice;
  bool             d_recompile;
  IntegratorType d_integrator;

private:

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
};
      
} // end namespace Uintah

#endif
