#ifndef UINTAH_HOMEBREW_IMP_MPM_H
#define UINTAH_HOMEBREW_IMP_MPM_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Math/Sparse.h>
#include <valarray>

namespace Uintah {

using namespace SCIRun;
 class DataWarehouse;
 class MPMLabel;
 class ProcessorGroup;
 class Patch;
 class VarLabel;
 class Task; 

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


  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

private:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;

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
  void actuallyComputeStableTimestep(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* matls,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

  void initializeForFirstIterationAfterConverged(const ProcessorGroup*,
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

  void applySymmetryBoundaryConditions(const ProcessorGroup*,
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
			   DataWarehouse* new_dw,
			   const bool recursion);

  void computeStressTensorOnly(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

  //////////
  // Check to see if any particles are ready to burn
  void formStiffnessMatrix(const ProcessorGroup*,
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
			    DataWarehouse* new_dw,
			    const bool recursion);

  void iterate(const ProcessorGroup* pg,
	       const PatchSubset* patches,
	       const MaterialSubset* matls,
	       DataWarehouse* old_dw, DataWarehouse* new_dw,
	       LevelP level, SchedulerP sched);

  void formQ(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls, DataWarehouse* old_dw,
	     DataWarehouse* new_dw);


  void solveForDuCG(const ProcessorGroup*, const PatchSubset* patches,
		    const MaterialSubset* matls, DataWarehouse* old_dw,
		    DataWarehouse* new_dw);


  void updateGridKinematics(const ProcessorGroup*, const PatchSubset* patches,
			    const MaterialSubset* matls, DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void checkConvergence(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw,
			LevelP level, SchedulerP sched);

  //////////
  // Insert Documentation Here:
  void computeAcceleration(const ProcessorGroup*,
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

  void interpolateStressToGrid(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);



  void scheduleInitializeForFirstIterationAfterConverged(SchedulerP&, 
							 const PatchSet*,
							 const MaterialSet*);


  void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
					  const MaterialSet*);


  void scheduleApplySymmetryBoundaryConditions(SchedulerP&, const PatchSet*,
					       const MaterialSet*);

  void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
				   const MaterialSet*,
				   const bool recursion = false);

  void scheduleComputeStressTensorOnly(SchedulerP&, const PatchSet*,
				       const MaterialSet*,
				       const bool recursion = false);

  void scheduleFormStiffnessMatrix(SchedulerP&, const PatchSet*,
				   const MaterialSet*);


  void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
				    const MaterialSet*,
				   const bool recursion);

  void scheduleIterate(SchedulerP&, const LevelP&,const PatchSet*, 
		       const MaterialSet*);

  void scheduleFormQ(SchedulerP&, const PatchSet*, const MaterialSet*);

  void scheduleSolveForDuCG(SchedulerP&, const PatchSet*, const MaterialSet*);

  void scheduleUpdateGridKinematics(SchedulerP&, const PatchSet*, 
				    const MaterialSet*);

  void scheduleCheckConvergence(SchedulerP&, const LevelP&, const PatchSet*,
				const MaterialSet*);

  void scheduleComputeAcceleration(SchedulerP&, const PatchSet*,
				   const MaterialSet*);

                                                 
  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, const PatchSet*,
					       const MaterialSet*);

  void scheduleInterpolateStressToGrid(SchedulerP&, const PatchSet*,
			       const MaterialSet*);

  void scheduleInterpolateParticlesForSaving(SchedulerP&, const PatchSet*,
					     const MaterialSet*);

  ImpMPM(const ImpMPM&);
  ImpMPM& operator=(const ImpMPM&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* lb;

  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_MPM;

  SparseMatrix<double,int> KK,KKK;

  // right hand side
  valarray<double> Q;

  bool dynamic;
  

};
      
} // end namespace Uintah

#endif
