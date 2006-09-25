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
#include <Packages/Uintah/CCA/Components/MPM/MPMCommon.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

#include <Packages/Uintah/CCA/Components/MPM/share.h>

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

class SCISHARE SerialMPM : public MPMCommon, public SimulationInterface, public UintahParallelComponent {
public:
  SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();

  Contact*         contactModel;
  ThermalContact*  thermalContactModel;
  HeatConduction* heatConductionModel;
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& materials_ps, GridP&,
			    SimulationStateP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);
	 
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
				   SchedulerP&);

  virtual void scheduleRefine(const PatchSet* patches, SchedulerP& scheduler);

  virtual void scheduleRefineInterface(const LevelP& fineLevel, SchedulerP& scheduler,
                                       bool needCoarse, bool needFine);

  virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark flags for AMR regridding
  virtual void scheduleErrorEstimate(const LevelP& coarseLevel, 
                                     SchedulerP& sched);
  
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
	flags->d_with_ice = true;
  };
  
  // special variables for soil and foam
  struct SoilFoam_vars{
     bool usingSoilFoam_CM;
     MaterialSet* matl;
     MaterialSubset* matl_sub;
   };
   SoilFoam_vars* d_SF_vars;
  

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
  virtual void setBCsInterpolated(const ProcessorGroup*,
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

  virtual void scheduleSetBCsInterpolated(SchedulerP&, const PatchSet*,
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
					     const MaterialSet*) {};

  virtual void scheduleAdjustCrackContactInterpolated(SchedulerP&, 
						      const PatchSet*,
						      const MaterialSet*){};

  virtual void scheduleAdjustCrackContactIntegrated(SchedulerP&, 
						    const PatchSet*,
						    const MaterialSet*){};

  virtual void scheduleCalculateFractureParameters(SchedulerP&,const PatchSet*,
						   const MaterialSet*){};

  virtual void scheduleDoCrackPropagation(SchedulerP& sched, 
					  const PatchSet* patches,
					  const MaterialSet* matls){};

  virtual void scheduleMoveCracks(SchedulerP& sched,const PatchSet* patches,
				  const MaterialSet* matls){};

  virtual void scheduleUpdateCrackFront(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls){};


  void scheduleCheckNeedAddMPMMaterial(SchedulerP&,
					const PatchSet* patches,
                                        const MaterialSet*);
                                                                             
  //////////
  // Insert Documentation Here:
  void checkNeedAddMPMMaterial(const ProcessorGroup*,
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
  
  bool needRecompile(double time, double dt,
                             const GridP& grid);
  
  
  virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
  
  virtual void switchTest(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse*);
                   
  void printSchedule(const PatchSet* patches,
                     const string& where);
  
  void printSchedule(const LevelP& level,
                     const string& where);
                     
  void printTask(const PatchSubset* patches,
                 const Patch* patch,
                 const string& where);
  
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;
  Output* dataArchiver;

  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  
  list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, yminus, ...
  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_fracture;
  bool             d_recompile;
  IntegratorType d_integrator;

private:

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
	 
};
      
} // end namespace Uintah

#endif
