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
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <map>

using namespace std;

#ifdef HAVE_PETSC
extern "C" {
#include "petscsles.h"
}
#endif

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


  //////////
  // Insert Documentation Here:
  void interpolateParticlesToGrid(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

  void applyBoundaryConditions(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

  void createMatrix(const ProcessorGroup*,
		    const PatchSubset* patches,
		    const MaterialSubset* matls,
		    DataWarehouse* old_dw,
		    DataWarehouse* new_dw);

  void destroyMatrix(const ProcessorGroup*,
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
			   bool recursion);

  void computeStressTensorOnly(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

  void formStiffnessMatrix(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw,
			   const bool recursion);

  void formStiffnessMatrixPetsc(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw,
				const bool recursion);



  //////////
  // Insert Documentation Here:
  void computeInternalForce(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw,
			    bool recursion);


  void iterate(const ProcessorGroup* pg,
	       const PatchSubset* patches,
	       const MaterialSubset* matls,
	       DataWarehouse* old_dw, DataWarehouse* new_dw,
	       LevelP level, SchedulerP sched);

  void moveData(const ProcessorGroup*,
		 const PatchSubset* patches,
		 const MaterialSubset* matls,
		 DataWarehouse* old_dw,
		 DataWarehouse* new_dw);

  void formQ(const ProcessorGroup*, const PatchSubset* patches,
	     const MaterialSubset* matls, DataWarehouse* old_dw,
	     DataWarehouse* new_dw,const bool recursion);

  void formQPetsc(const ProcessorGroup*, const PatchSubset* patches,
		  const MaterialSubset* matls, DataWarehouse* old_dw,
		  DataWarehouse* new_dw,const bool recursion);

  void applyRigidBodyCondition(const ProcessorGroup*, 
			       const PatchSubset* patches,
			       const MaterialSubset* matls, 
			       DataWarehouse* old_dw,
			       DataWarehouse* new_dw);

  void removeFixedDOF(const ProcessorGroup*, const PatchSubset* patches,
		      const MaterialSubset* matls, DataWarehouse* old_dw,
		      DataWarehouse* new_dw, const bool recursion);

  void removeFixedDOFPetsc(const ProcessorGroup*, 
			   const PatchSubset* patches,
			   const MaterialSubset* matls, 
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw, const bool recursion);


  void solveForDuCG(const ProcessorGroup*, const PatchSubset* patches,
		    const MaterialSubset* matls, DataWarehouse* old_dw,
		    DataWarehouse* new_dw, const bool recursion);

  void solveForDuCGPetsc(const ProcessorGroup*, const PatchSubset* patches,
			 const MaterialSubset* matls, DataWarehouse* old_dw,
			 DataWarehouse* new_dw, const bool recursion);


  void updateGridKinematics(const ProcessorGroup*, const PatchSubset* patches,
			    const MaterialSubset* matls, DataWarehouse* old_dw,
			    DataWarehouse* new_dw, const bool recursion);

  //////////
  // Insert Documentation Here:
  void checkConvergence(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw,
			LevelP level, SchedulerP sched,
			const bool recursion);


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


  void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
					  const MaterialSet*);


  void scheduleApplyBoundaryConditions(SchedulerP&, const PatchSet*,
				       const MaterialSet*);

  void scheduleCreateMatrix(SchedulerP&, const PatchSet*,const MaterialSet*);

  void scheduleDestroyMatrix(SchedulerP&, const PatchSet*,const MaterialSet*);

  void scheduleComputeStressTensorI(SchedulerP&, const PatchSet*,
				    const MaterialSet*,
				    bool recursion = false);

  void scheduleComputeStressTensorR(SchedulerP&, const PatchSet*,
				    const MaterialSet*,
				    bool recursion = false);

  void scheduleComputeStressTensorOnly(SchedulerP&, const PatchSet*,
				       const MaterialSet*,
				       bool recursion = false);

  void scheduleFormStiffnessMatrixI(SchedulerP&, const PatchSet*,
				   const MaterialSet*,const bool recursion);

  void scheduleFormStiffnessMatrixR(SchedulerP&, const PatchSet*,
				   const MaterialSet*,const bool recursion);


  void scheduleComputeInternalForceI(SchedulerP&, const PatchSet*,
				     const MaterialSet*,
				     bool recursion);

  void scheduleComputeInternalForceII(SchedulerP&, const PatchSet*,
				      const MaterialSet*,
				      bool recursion);

  void scheduleComputeInternalForceR(SchedulerP&, const PatchSet*,
				    const MaterialSet*,
				    const bool recursion);

  void scheduleMoveData(SchedulerP&, const LevelP&, const PatchSet*,
				     const MaterialSet*);

  void scheduleIterate(SchedulerP&, const LevelP&,const PatchSet*, 
		       const MaterialSet*);

  void scheduleFormQI(SchedulerP&, const PatchSet*, const MaterialSet*,
		      const bool recursion);
  
  void scheduleFormQR(SchedulerP&, const PatchSet*, const MaterialSet*,
		      const bool recursion);

  void scheduleApplyRigidBodyConditionI(SchedulerP&, const PatchSet*, 
					 const MaterialSet*);

  void scheduleApplyRigidBodyConditionR(SchedulerP&, const PatchSet*, 
					 const MaterialSet*);

  void scheduleRemoveFixedDOFI(SchedulerP&, const PatchSet*, 
			       const MaterialSet*, const bool recursion);

  void scheduleRemoveFixedDOFR(SchedulerP&, const PatchSet*, 
			       const MaterialSet*,const bool recursion);

  void scheduleSolveForDuCGI(SchedulerP&, const PatchSet*, const MaterialSet*,
			     bool recursion);

  void scheduleSolveForDuCGR(SchedulerP&, const PatchSet*, const MaterialSet*,
			     bool recursion);

  void scheduleUpdateGridKinematicsI(SchedulerP&, const PatchSet*, 
				    const MaterialSet*,const bool recursion);

  void scheduleUpdateGridKinematicsR(SchedulerP&, const PatchSet*, 
				    const MaterialSet*,const bool recursion);

  void scheduleCheckConvergenceI(SchedulerP&, const LevelP&, const PatchSet*,
				const MaterialSet*, bool recursion);


  void scheduleCheckConvergenceR(SchedulerP&, const LevelP&, const PatchSet*,
				const MaterialSet*, bool recursion);

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

  SparseMatrix<double,int> KK;

  // right hand side
  valarray<double> Q;


#ifdef HAVE_PETSC
   Mat A;
   Vec petscQ,petscTemp2,d_x, d_b, d_u;
   SLES sles;
#endif

   const PatchSet* d_perproc_patches;
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;

   bool dynamic;
   
   IntegratorType d_integrator;

};
      
} // end namespace Uintah

#endif
