#ifndef UINTAH_HOMEBREW_IMP_MPM_H
#define UINTAH_HOMEBREW_IMP_MPM_H

#include <sci_defs/petsc_defs.h>

#include <Core/Geometry/Vector.h>

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>


#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using namespace SCIRun;
 class DataWarehouse;
 class MPMLabel;
 class ProcessorGroup;
 class VarLabel;
 class Task; 
 class MPMPetscSolver;
 class SimpleSolver;
 class ImplicitHeatConduction;
 class ThermalContact;

/**************************************

CLASS
   ImpMPM
   
   Short description...

GENERAL INFORMATION

   ImpMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ImpMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ImpMPM : public UintahParallelComponent, public SimulationInterface {
public:
  ImpMPM(const ProcessorGroup* myworld);
  virtual ~ImpMPM();

  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			    SimulationStateP&);
	 
  virtual void scheduleInitialize(           const LevelP& level, SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(          const LevelP& level, SchedulerP&,
				    int step, int nsteps );

  virtual bool restartableTimesteps();
  virtual double recomputeTimestep(double new_dt);

  void setSharedState(SimulationStateP& ssp);

  void setMPMLabel(MPMLabel* Mlb)
  {

        delete lb;
	lb = Mlb;
  };


  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };
  
  enum IntegratorType {
    Explicit,
    Implicit 
  };

private:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;

  inline bool compare(double num1, double num2)
    {
      double EPSILON=1.e-16;
      
      return (fabs(num1-num2) <= EPSILON);
    };


  void actuallyInitialize(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void printParticleCount(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(  const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void applyExternalLoads(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void interpolateParticlesToGrid(     const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void rigidBody(                      const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void destroyMatrix(                  const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const bool recursion);

  void createMatrix(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void applyBoundaryConditions(        const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void computeContact(                 const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void findFixedDOF(                   const ProcessorGroup*, 
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls, 
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  // This is for the computation with the 24 x 24 matrix
  void computeStressTensor(            const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const bool recursion);

  // No matrix calculations are performed.
  void computeStressTensor(            const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void formStiffnessMatrix(            const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here:
  void computeInternalForce(           const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void iterate(                        const ProcessorGroup* pg,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       LevelP level, Scheduler* sched);

  void formQ(                          const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void solveForDuCG(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void solveForTemp(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void getDisplacementIncrement(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

  void getTemperatureIncrement(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);


  void updateGridKinematics(           const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void checkConvergence(               const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);
			
  //////////
  // Insert Documentation Here:
  void computeAcceleration(            const ProcessorGroup*,
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
  //////////
  // Insert Documentation Here:
  void interpolateStressToGrid(        const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* matls,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

  void scheduleComputeStressTensor( SchedulerP&, const PatchSet*,
                                    const MaterialSet*, const bool recursion);

  void scheduleFormStiffnessMatrix( SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleFormHCStiffnessMatrix( SchedulerP&, const PatchSet*,
                                      const MaterialSet*);

  void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
				    const MaterialSet*);

  void scheduleFormQ(               SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleFormHCQ(             SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleAdjustHCQAndHCKForBCs(SchedulerP&, const PatchSet*,
                                     const MaterialSet*);

  void scheduleUpdateGridKinematics(SchedulerP&, const PatchSet*, 
				    const MaterialSet*);

  void scheduleApplyExternalLoads(             SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleInterpolateParticlesToGrid(     SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
                                   const MaterialSet*);
  
  void scheduleRigidBody(                      SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleDestroyMatrix(      SchedulerP&, const PatchSet*,
                                   const MaterialSet*, const bool recursion);

  void scheduleDestroyHCMatrix(    SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleCreateMatrix(       SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleCreateHCMatrix(     SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleApplyBoundaryConditions(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleApplyHCBoundaryConditions(      SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeContact(                 SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleFindFixedDOF(                   SchedulerP&, const PatchSet*, 
                                               const MaterialSet*);

  void scheduleFindFixedHCDOF(                 SchedulerP&, const PatchSet*, 
                                               const MaterialSet*);

  void scheduleComputeStressTensor(            SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeInternalHeatRate(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveHeatEquations(             SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleIntegrateTemperatureRate(       SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveForDuCG(                   SchedulerP&,const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveForTemp(                   SchedulerP&,const PatchSet*,
                                               const MaterialSet*);

  void scheduleGetDisplacementIncrement(       SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleGetTemperatureIncrement(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeAcceleration(            SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, const PatchSet*,
					       const MaterialSet*);

  void scheduleInterpolateStressToGrid(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleIterate(             SchedulerP&, const LevelP&,const PatchSet*, 
                                    const MaterialSet*);
  
  void scheduleCheckConvergence(    SchedulerP&, const LevelP&, const PatchSet*,
                                    const MaterialSet*);


  ImpMPM(const ImpMPM&);
  ImpMPM& operator=(const ImpMPM&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;

  ImplicitHeatConduction* heatConductionModel;
  ThermalContact* thermalContactModel;

  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;
  double           d_conv_crit_disp;
  double           d_conv_crit_energy;
  double           d_initialDt;
  double           d_forceIncrementFactor; // Increment in ForceBC applied force
  int              d_numIterations;
  bool             d_doGridReset;  // Default is true, standard MPM
  Vector           d_contact_dirs; // For rigid body contact
  int              d_max_num_iterations;  // restart timestep
  int              d_num_iters_to_decrease_delT;
  int              d_num_iters_to_increase_delT;
  double           d_delT_decrease_factor;
  double           d_delT_increase_factor;

  std::list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, ...

  const PatchSet* d_perproc_patches;

#ifdef HAVE_PETSC
  MPMPetscSolver* d_solver;
#else
  SimpleSolver* d_solver;
#endif

  bool d_dynamic;
  bool d_rigid_body;
  bool d_single_velocity;
  bool d_useLoadCurves;
  
  IntegratorType d_integrator;

};
      
} // end namespace Uintah

#endif
