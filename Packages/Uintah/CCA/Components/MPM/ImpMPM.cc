#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h> // 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Util/DebugStream.h>


#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

static DebugStream cout_doing("IMPM_DOING_COUT", false);
// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

ImpMPM::ImpMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;

}

ImpMPM::~ImpMPM()
{
  delete lb;

}

void ImpMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& /*grid*/,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

  
   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

   ProblemSpecP mpm_ps = prob_spec->findBlock("MPM");
   if (!mpm_ps->get("dynamic",dynamic))
       dynamic = true;

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, 8);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
   }

   cout << "Number of materials: " << d_sharedState->getNumMatls() << endl;


   // Load up all the VarLabels that will be used in each of the
   // physical models
   lb->d_particleState.resize(d_sharedState->getNumMPMMatls());
   lb->d_particleState_preReloc.resize(d_sharedState->getNumMPMMatls());

   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     lb->registerPermanentParticleState(m,lb->pVelocityLabel,
					lb->pVelocityLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pAccelerationLabel,
					lb->pAccelerationLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pExternalForceLabel,
					lb->pExternalForceLabel_preReloc);

     lb->registerPermanentParticleState(m,lb->pParticleIDLabel,
					lb->pParticleIDLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pMassLabel,
					lb->pMassLabel_preReloc);
     lb->registerPermanentParticleState(m,lb->pVolumeLabel,
					lb->pVolumeLabel_preReloc);
     
     mpm_matl->getConstitutiveModel()->addParticleState(lb->d_particleState[m],
					lb->d_particleState_preReloc[m]);
   }
}

void ImpMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched)
{
  Task* t = scinew Task("ImpMPM::actuallyInitialize",
			this, &ImpMPM::actuallyInitialize);
  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel);

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  t = scinew Task("ImpMPM::printParticleCount",
		  this, &ImpMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
}

void ImpMPM::scheduleComputeStableTimestep(const LevelP&, SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void ImpMPM::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleInterpolateParticlesToGrid(sched, patches, matls);

#if 0
  scheduleApplySymmetryBoundaryConditions(sched,patches,matls);
#endif

  scheduleComputeStressTensorI(sched, patches, matls,false);

  scheduleFormStiffnessMatrixI(sched,patches,matls);

  scheduleComputeInternalForceI(sched, patches, matls,false);

  scheduleFormQI(sched, patches, matls);

  scheduleSolveForDuCGI(sched, patches, matls,false);

  scheduleUpdateGridKinematicsI(sched, patches, matls);

  scheduleCheckConvergenceI(sched,level, patches, matls, false);

  scheduleIterate(sched,level,patches,matls);

  scheduleComputeStressTensorOnly(sched,patches,matls,false);

  scheduleComputeInternalForceII(sched,patches,matls,true);

  scheduleComputeAcceleration(sched,patches,matls);

  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleInterpolateStressToGrid(sched,patches,matls);



  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc, 
				    lb->d_particleState_preReloc,
				    lb->pXLabel, lb->d_particleState,
				    lb->pParticleIDLabel, matls);

}



void ImpMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						const PatchSet* patches,
						const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("ImpMPM::interpolateParticlesToGrid",
			this,&ImpMPM::interpolateParticlesToGrid);
  t->requires(Task::OldDW, lb->pMassLabel,           Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,       Ghost::AroundNodes,1);
  
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVelocityLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pXLabel,            Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pExternalForceLabel,Ghost::AroundNodes,1);



  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->TotalMassLabel);
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleApplySymmetryBoundaryConditions(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::applySymmetryBoundaryConditions",
		        this, &ImpMPM::applySymmetryBoundaryConditions);

  sched->addTask(t, patches, matls);
}



void ImpMPM::scheduleComputeStressTensorI(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls,
					 const bool recursion)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ImpMPM::computeStressTensorI",
		    this, &ImpMPM::computeStressTensor,recursion);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequiresImplicit(t, mpm_matl, patches,recursion);
  }

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleComputeStressTensorR(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls,
					 const bool recursion)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ImpMPM::computeStressTensorR",
		    this, &ImpMPM::computeStressTensor,recursion);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequiresImplicit(t, mpm_matl, patches,recursion);
  }

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeStressTensorOnly(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls,
					 const bool recursion)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ImpMPM::computeStressTensorOnly",
		    this, &ImpMPM::computeStressTensorOnly);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequiresImplicitOnly(t, mpm_matl, patches,recursion);
  }

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleFormStiffnessMatrixI(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::formStiffnessMatrixI",
		    this, &ImpMPM::formStiffnessMatrix);

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleFormStiffnessMatrixR(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::formStiffnessMatrixR",
		    this, &ImpMPM::formStiffnessMatrix);

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeInternalForceI(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::computeInternalForceI",
			this, &ImpMPM::computeInternalForce,recursion);
  
  if (recursion)
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  else
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);

  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);

  t->computes(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleComputeInternalForceII(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::computeInternalForceII",
			this, &ImpMPM::computeInternalForce,recursion);
  
  if (recursion)
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,
		1);
  else
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,
		1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);
  t->modifies(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleComputeInternalForceR(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::computeInternalForceR",
			this, &ImpMPM::computeInternalForce,recursion);
  
  if (recursion)
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,
		1);
  else
    t->requires(Task::NewDW,lb->pStressLabel,Ghost::AroundNodes,
		1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);
  t->computes(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}



void ImpMPM::scheduleIterate(SchedulerP& sched,const LevelP& level,
			     const PatchSet* patches, const MaterialSet* matls)
{

  // NOT DONE

  Task* task = scinew Task("scheduleIterate", this, &ImpMPM::iterate,level,
			   sched);
  task->hasSubScheduler();

  task->computes(lb->dispNewLabel);

  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches = lb->createPerProcessorPatchSet(level, 
								   d_myworld);
  sched->addTask(task,perproc_patches,matls);

  
}


void ImpMPM::iterate(const ProcessorGroup* pg,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     LevelP level, SchedulerP sched)
{
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize();
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);


  scheduleComputeStressTensorR(subsched,level->eachPatch(),
			      d_sharedState->allMPMMaterials(),
			      true);

  
  scheduleFormStiffnessMatrixR(subsched,level->eachPatch(),
			      d_sharedState->allMPMMaterials());

  scheduleComputeInternalForceR(subsched,level->eachPatch(),
			      d_sharedState->allMPMMaterials(), true);

  
  scheduleFormQR(subsched,level->eachPatch(),d_sharedState->allMPMMaterials());

  scheduleSolveForDuCGR(subsched,level->eachPatch(),
		       d_sharedState->allMPMMaterials(), true);
  scheduleUpdateGridKinematicsR(subsched,level->eachPatch(),
			       d_sharedState->allMPMMaterials());

  scheduleCheckConvergenceR(subsched,level,level->eachPatch(),
			       d_sharedState->allMPMMaterials(), true);

  subsched->compile(d_myworld,false);


  bool_and_vartype converged_var;
  new_dw->get(converged_var, lb->converged);
  
  while(bool(converged_var) == false) {
    subsched->execute(d_myworld);
    subsched->advanceDataWarehouse(grid);
    subsched->get_new_dw()->get(converged_var, lb->converged);

  }


}

void ImpMPM::scheduleFormQI(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::formQI", this, 
			&ImpMPM::formQ);

  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleFormQR(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::formQR", this, 
			&ImpMPM::formQ);

  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleSolveForDuCGI(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls,
				   const bool recursion)
{
  Task* t = scinew Task("ImpMPM::solveForDuCGI", this, 
			&ImpMPM::solveForDuCG);
  if (recursion)
    t->modifies(lb->dispIncLabel);
  else
    t->computes(lb->dispIncLabel);
  
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleSolveForDuCGR(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls,
				   const bool recursion)
{
  Task* t = scinew Task("ImpMPM::solveForDuCGR", this, 
			&ImpMPM::solveForDuCG);
  if (recursion)
    t->modifies(lb->dispIncLabel);
  else
    t->computes(lb->dispIncLabel);
  
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleUpdateGridKinematicsI(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::updateGridKinematicsI", this, 
			&ImpMPM::updateGridKinematics);
  
  t->computes(lb->dispNewLabel);
  t->modifies(lb->gVelocityLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleUpdateGridKinematicsR(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::updateGridKinematicsR", this, 
			&ImpMPM::updateGridKinematics);
  
  t->computes(lb->dispNewLabel);
  t->modifies(lb->gVelocityLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}



void ImpMPM::scheduleCheckConvergenceI(SchedulerP& sched, const LevelP& level,
				       const PatchSet* patches,
				       const MaterialSet* matls,
				       const bool recursion)
{
  // NOT DONE

  Task* t = scinew Task("ImpMPM::checkConvergenceI", this,
			&ImpMPM::checkConvergence,level,sched);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  if (recursion) {
    t->modifies(lb->dispIncNormMax);
    t->modifies(lb->dispIncQNorm0);
    t->modifies(lb->converged);
  } else {
    t->computes(lb->dispIncNormMax);
    t->computes(lb->dispIncQNorm0);
    t->computes(lb->converged);
  }


  t->hasSubScheduler();

  LoadBalancer* loadbal = sched->getLoadBalancer();
  const PatchSet* perproc_patches = loadbal->createPerProcessorPatchSet(level, 
								   d_myworld);
  sched->addTask(t,perproc_patches,matls);

  

}

void ImpMPM::scheduleCheckConvergenceR(SchedulerP& sched, const LevelP& level,
				       const PatchSet* patches,
				       const MaterialSet* matls,
				       const bool recursion)
{
  // NOT DONE

  Task* t = scinew Task("ImpMPM::checkConvergenceR", this,
			&ImpMPM::checkConvergence,level,sched);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  if (recursion) {
    t->modifies(lb->dispIncNormMax);
    t->modifies(lb->dispIncQNorm0);
    t->modifies(lb->converged);
  } else {
    t->computes(lb->dispIncNormMax);
    t->computes(lb->dispIncQNorm0);
    t->computes(lb->converged);
  }

  t->hasSubScheduler();

  LoadBalancer* loadbal = sched->getLoadBalancer();
  const PatchSet* perproc_patches = loadbal->createPerProcessorPatchSet(level, 
								   d_myworld);
  sched->addTask(t,perproc_patches,matls);

  

}



void ImpMPM::scheduleComputeAcceleration(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{
  /* computeAcceleration
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  Task* t = scinew Task("ImpMPM::computeAcceleration",
			    this, &ImpMPM::computeAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->modifies(lb->gAccelerationLabel);
  t->requires(Task::NewDW, lb->gVelocityLabel,Ghost::None);
  t->requires(Task::NewDW, lb->dispNewLabel,Ghost::None);

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
 /*
  * interpolateToParticlesAndUpdate
  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
  *   operation(interpolate acceleration and v* to particles and
  *   integrate these to get new particle velocity and position)
  * out(P.VELOCITY, P.X, P.NAT_X) */

  Task* t=scinew Task("ImpMPM::interpolateToParticlesAndUpdate",
		    this, &ImpMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,  Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->dispNewLabel,Ghost::None);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);


  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pAccelerationLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pExternalForceLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleInterpolateStressToGrid(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
  /* interpolateStressToGrid
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  Task* t = scinew Task("ImpMPM::interpolateStressToGrid",
			    this, &ImpMPM::interpolateStressToGrid);

  t->requires(Task::NewDW, lb->pXLabel_preReloc,Ghost::AroundNodes,1);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,Ghost::AroundNodes,1);
  t->requires(Task::NewDW, lb->pStressLabel_preReloc,Ghost::AroundNodes,1);

  t->computes(lb->gStressLabel);

  sched->addTask(t, patches, matls);
}

void ImpMPM::printParticleCount(const ProcessorGroup* pg,
				   const PatchSubset*,
				   const MaterialSubset*,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    static bool printed=false;
    if(!printed){
      sumlong_vartype pcount;
      new_dw->get(pcount, lb->partCountLabel);
      cerr << "Created " << pcount << " total particles\n";
      printed=true;
    }
  }
}

void ImpMPM::actuallyInitialize(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing actuallyInitialize on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl << endl;

    CCVariable<short int> cellNAPID;
    new_dw->allocate(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);
    
    for(int m=0;m<matls->size();m++){
      //cerrLock.lock();
      //NOT_FINISHED("not quite right - mapping of matls, use matls->get()");
      //cerrLock.unlock();
       MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
       particleIndex numParticles = mpm_matl->countParticles(patch);
       totalParticles+=numParticles;

       mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

       mpm_matl->getConstitutiveModel()->initializeCMData(patch,
						mpm_matl, new_dw);
       

    }
    new_dw->put(cellNAPID, lb->pCellNAPIDLabel, 0, patch);

  }
  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);

}


void ImpMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					      const PatchSubset*,
					      const MaterialSubset*,
					      DataWarehouse*,
					      DataWarehouse*)
{
}


void ImpMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const PatchSubset* patches,
					   const MaterialSubset* ,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  // DONE
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID()
	       <<"\t\t MPM"<< endl << endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal;
    new_dw->allocate(gmassglobal,lb->gMassLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume;
      constParticleVariable<Vector> pvelocity, pacceleration,pexternalforce;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pacceleration,  lb->pAccelerationLabel,  pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity,gacceleration;
      NCVariable<Vector> gexternalforce;

      new_dw->allocate(gmass,lb->gMassLabel,      matlindex, patch);
      new_dw->allocate(gvolume,lb->gVolumeLabel,    matlindex, patch);
      new_dw->allocate(gvelocity,lb->gVelocityLabel,  matlindex, patch);
      new_dw->allocate(gacceleration,lb->gAccelerationLabel,matlindex, patch);
      new_dw->allocate(gexternalforce,lb->gExternalForceLabel,matlindex,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gacceleration.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	
	// Get the node indices that surround the cell
	IntVector ni[8];
	double S[8];
	
	patch->findCellAndWeights(px[idx], ni, S);
	
	total_mom += pvelocity[idx]*pmass[idx];
	
	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	  if(patch->containsNode(ni[k])) {
	    gmassglobal[ni[k]]    += pmass[idx]          * S[k];
	    gmass[ni[k]]          += pmass[idx]          * S[k];
	    gvolume[ni[k]]        += pvolume[idx]        * S[k];
	    gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	    gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	    gacceleration[ni[k]] += pacceleration[idx] * S[k];
	    totalmass += pmass[idx] * S[k];
	  }
	}
      }
      
      
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	gvelocity[*iter] /= gmass[*iter];
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

      new_dw->put(gmass,         lb->gMassLabel,          matlindex, patch);
      new_dw->put(gvolume,       lb->gVolumeLabel,        matlindex, patch);
      new_dw->put(gvelocity, lb->gVelocityLabel,      matlindex, patch);
      new_dw->put(gacceleration,lb->gAccelerationLabel,matlindex,patch);
      new_dw->put(gexternalforce,lb->gExternalForceLabel, matlindex, patch);
 

    }  // End loop over materials

     new_dw->put(gmassglobal, lb->gMassLabel,
			d_sharedState->getAllInOneMatl()->get(0), patch);
  }  // End loop over patches
}

void ImpMPM::applySymmetryBoundaryConditions(const ProcessorGroup*,
					     const PatchSubset* patches,
					     const MaterialSubset* ,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
{
  // NOT DONE
  cout_doing <<"Doing applysymmetryBoundaryConditions " <<"\t\t\t\t MPM"
	     << endl << endl;

#if 0
   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
   }
#endif
}

void ImpMPM::computeStressTensor(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // DONE

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t MPM"<< endl << endl;


  KK.clear();
  
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->computeStressTensorImplicit(patches, mpm_matl, old_dw, new_dw,KK,
				    recursion);
  }
  
}

void ImpMPM::computeStressTensorOnly(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  // DONE

  cout_doing <<"Doing computeStressTensorOnly " <<"\t\t\t\t MPM"<< endl 
	     << endl;

  KK.clear();
  
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->computeStressTensorImplicitOnly(patches, mpm_matl, old_dw, new_dw);
  }
  
}

void ImpMPM::formStiffnessMatrix(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  // DONE
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
	       <<"\t\t\t\t MPM"<< endl << endl;

    IntVector nodes = patch->getNNodes();

    int matlindex = 0;
    constNCVariable<double> gmass;
    new_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);

    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label() );

    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x()+1)*(n.y()) + (nodes.y()+1)*
	(nodes.x()+1)*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
      KK[dof[0]][dof[0]] = KK[dof[0]][dof[0]] + gmass[*iter]*(4./(dt*dt));
      KK[dof[1]][dof[1]] = KK[dof[1]][dof[1]] + gmass[*iter]*(4./(dt*dt));
      KK[dof[2]][dof[2]] = KK[dof[2]][dof[2]] + gmass[*iter]*(4./(dt*dt));
    }
		
  }
}
	    
void ImpMPM::computeInternalForce(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
				  const bool recursion)
{
  // DONE
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl << endl;

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    
    int numMPMMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel, pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      if (recursion)
	new_dw->get(pstress, lb->pStressLabel_preReloc, pset);
      else
	new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

      new_dw->allocate(internalforce,lb->gInternalForceLabel,matlindex,patch);

      internalforce.initialize(Vector(0,0,0));

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	
	// Get the node indices that surround the cell
	IntVector ni[8];
	Vector d_S[8];
	double S[8];
	
	patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
	
	for (int k = 0; k < 8; k++){
	  if(patch->containsNode(ni[k])){
	    Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
		       d_S[k].z()*oodx[2]);
	    internalforce[ni[k]] -= (div * pstress[idx] * pvol[idx]);
	  }
	}
      }
     
      new_dw->put(internalforce, lb->gInternalForceLabel,   matlindex, patch);
      
    }
  }

}


void ImpMPM::formQ(const ProcessorGroup*, const PatchSubset* patches,
		   const MaterialSubset*, DataWarehouse* old_dw,
		   DataWarehouse* new_dw)
{
  // DONE
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formQ on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl << endl;


    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());
    double fodts = 4./(dt*dt);
    double fodt = 4./dt;

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x()+1)*(nodes.y()+1)*(nodes.z()+1);
    valarray<double> temp2(0.,3.*num_nodes);

    int matlindex = 0;

    constNCVariable<Vector> externalForce, internalForce;
    constNCVariable<Vector> dispNew,velocity,accel;
    constNCVariable<double> mass;
    new_dw->get(internalForce,lb->gInternalForceLabel,matlindex,patch,
		Ghost::None,0);
    new_dw->get(externalForce,lb->gExternalForceLabel,matlindex,patch,
		Ghost::None,0);
    old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::None,0);
    old_dw->get(velocity,lb->gVelocityLabel,matlindex,patch,
		Ghost::None,0);
    new_dw->get(accel,lb->gAccelerationLabel,matlindex,patch,
		Ghost::None,0);

    new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);

    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x()+1)*(n.y()) + (nodes.y()+1)*
	(nodes.x()+1)*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;

      Q[dof[0]] = externalForce[n].x() + internalForce[n].x();
      Q[dof[1]] = externalForce[n].y() + internalForce[n].y();
      Q[dof[2]] = externalForce[n].z() + internalForce[n].z();

      // temp2 = M*a^(k-1)(t+dt)

      temp2[dof[0]] = (dispNew[n].x()*fodts - velocity[n].x()*fodt -
			accel[n].x())*mass[n];
      temp2[dof[1]] = (dispNew[n].y()*fodts - velocity[n].y()*fodt -
			accel[n].y())*mass[n];
      temp2[dof[2]] = (dispNew[n].z()*fodts - velocity[n].z()*fodt -
			accel[n].z())*mass[n];

    }
    if (dynamic)
      Q -= temp2;
  }

}

void ImpMPM::solveForDuCG(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw)

{
  // DONE
  int conflag = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x()+1)*(nodes.y()+1)*(nodes.z()+1);

    valarray<double> x(0.,3*num_nodes);
    int matlindex = 0;
    
    x = cgSolve(KKK,Q,conflag);
    
    NCVariable<Vector> dispInc;
    new_dw->getModifiable(dispInc,lb->dispIncLabel,matlindex,patch);
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x()+1)*(n.y()) + (nodes.y()+1)*
	(nodes.x()+1)*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
      dispInc[n] = Vector(x[dof[0]],x[dof[1]],x[dof[2]]);
    }
  
    new_dw->put(dispInc,lb->dispIncLabel,matlindex,patch);
    
  }

  //return conflag;
    

}

void ImpMPM::updateGridKinematics(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)

{
  // DONE
  for (int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    
    int matlindex = 0;

    NCVariable<Vector> dispNew,velocity;
    constNCVariable<Vector> dispInc;

    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());

    new_dw->allocate(dispNew, lb->dispNewLabel, matlindex,patch);
    new_dw->getModifiable(velocity, lb->gVelocityLabel, matlindex,patch);
    new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);

    if (dynamic) {
      for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++) {
	dispNew[*iter] += dispInc[*iter];
	velocity[*iter] = dispNew[*iter]*(2./dt) - velocity[*iter];
      }
    } else {
      for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++) {
	dispNew[*iter] += dispInc[*iter];
	velocity[*iter] = dispNew[*iter]*(2./dt);
      }
    }

    new_dw->put(velocity,lb->gVelocityLabel,matlindex,patch);
    new_dw->put(dispNew,lb->dispNewLabel,matlindex,patch);
  }

}



void ImpMPM::checkConvergence(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* ,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      LevelP level, SchedulerP sched)
{

#if 0
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize();
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);

#endif

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector nodes = patch->getNNodes();


    cout_doing <<"Doing checkConvergence on patch " << patch->getID()
	       <<"\t\t MPM"<< endl << endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constNCVariable<Vector> dispInc;
      new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);

      double dispIncNorm = 0.;
      double dispIncQNorm = 0.;

      for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++) {
	IntVector n = *iter;
	int dof[3];
	int node_num = n.x() + (nodes.x()+1)*(n.y()) + (nodes.y()+1)*
	(nodes.x()+1)*(n.z());
	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;

	dispIncNorm += Dot(dispInc[n],dispInc[n]);
	dispIncQNorm += dispInc[n].x()*Q[dof[0]] + dispInc[n].y()*Q[dof[1]] +
	  dispInc[n].z()*Q[dof[2]];
      }

      double d_dispIncQNorm0,d_dispIncNormMax = 0.;
      sum_vartype dispIncQNorm0,dispIncNormMax;
      new_dw->override(dispIncQNorm0,lb->dispIncQNorm0);
      new_dw->override(dispIncNormMax,lb->dispIncNormMax);

      bool d_convergence;
      bool_and_vartype convergence;
      new_dw->override(convergence,lb->converged);

      if (dispIncQNorm0 == 0.0)
	d_dispIncQNorm0 = dispIncQNorm;

      if (dispIncNorm > dispIncNormMax)
	d_dispIncNormMax = dispIncNorm;

      if ((dispIncNorm/d_dispIncNormMax <= 1.e-8) && 
	  (dispIncQNorm/d_dispIncQNorm0 <= 1.e-8)) 
	d_convergence = true;
      else
	d_convergence = false;

      new_dw->put(bool_and_vartype(d_convergence),lb->converged);
      new_dw->put(sum_vartype(d_dispIncNormMax),lb->dispIncNormMax);
      new_dw->put(sum_vartype(d_dispIncQNorm0),lb->dispIncQNorm0);

    }  // End of loop over materials
  }  // End of loop over patches

  // Check what the convergence flag is and fire off the appropriate tasks

#if 0
  bool_and_vartype convergence;
  new_dw->get(convergence, lb->converged);

  if (!convergence) {
    scheduleComputeStressTensor(subsched,level->eachPatch(),
				d_sharedState->allMPMMaterials());
    scheduleFormStiffnessMatrix(subsched,level->eachPatch(),
				d_sharedState->allMPMMaterials());
  } else {
    scheduleComputeStressTensorOnly(subsched,level->eachPatch(),
				    d_sharedState->allMPMMaterials());
  }
#endif
  
}

void ImpMPM::computeAcceleration(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  // DONE
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeAcceleration on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl << endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<Vector> acceleration;
      constNCVariable<Vector> velocity,dispNew;
      delt_vartype delT;

      new_dw->getModifiable(acceleration,lb->gAccelerationLabel,dwindex,patch);
      new_dw->get(velocity,lb->gVelocityLabel,dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::None,0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double fodts = 4./(delT*delT);
      double fodt = 4./(delT);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	acceleration[*iter] = dispNew[*iter]*fodts - velocity[*iter]*fodt
	  - acceleration[*iter];
      }

      // Put the result in the datawarehouse
      new_dw->put(acceleration, lb->gAccelerationLabel, dwindex, patch);
    }
  }
}




void ImpMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  // DONE
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
	       << patch->getID() <<"\t MPM"<< endl << endl;

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
    Vector disp(0.0,0.0,0.0);
    Vector acc(0.0,0.0,0.0);
  
    // DON'T MOVE THESE!!!
    Vector CMX(0.0,0.0,0.0);
    Vector CMV(0.0,0.0,0.0);
    double ke=0;
    double massLost=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<Vector> pvelocity, pacceleration,pexternalForce;
      ParticleVariable<Vector> pvelocitynew, pexternalForceNew;
      constParticleVariable<double> pmass, pvolume;
      ParticleVariable<double> pmassNew,pvolumeNew;
  
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> dispNew, gacceleration;
      constNCVariable<double> dTdt;

      delt_vartype delT;

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);

      ParticleSubset* delete_particles = scinew ParticleSubset
	(pset->getParticleSet(),false,dwindex,patch);
    
      old_dw->get(px,                    lb->pXLabel,                    pset);
      old_dw->get(pmass,                 lb->pMassLabel,                 pset);
      new_dw->get(pvolume,               lb->pVolumeDeformedLabel,       pset);
      old_dw->get(pexternalForce,        lb->pExternalForceLabel,        pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,             pset);
      old_dw->get(pacceleration,         lb->pAccelerationLabel,         pset);
      new_dw->allocate(pvelocitynew,     lb->pVelocityLabel_preReloc,    pset);
      new_dw->allocate(pxnew,            lb->pXLabel_preReloc,           pset);
      new_dw->allocate(pmassNew,         lb->pMassLabel_preReloc,        pset);
      new_dw->allocate(pvolumeNew,       lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocate(pexternalForceNew,lb->pExternalForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::None,0);

      new_dw->get(gacceleration,      lb->gAccelerationLabel,
			dwindex, patch, Ghost::AroundCells, 1);

     
      NCVariable<double> dTdt_create, massBurnFraction_create;	
      new_dw->allocate(dTdt_create, lb->dTdt_NCLabel,
		       dwindex,patch,Ghost::AroundCells,1);
      dTdt_create.initialize(0.);
      dTdt = dTdt_create; // reference created data
            

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double rho_init=mpm_matl->getInitialDensity();

      IntVector ni[8];

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	
	double S[8];
	Vector d_S[8];
	
	// Get the node indices that surround the cell
	patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
	
	disp = Vector(0.0,0.0,0.0);
	acc = Vector(0.0,0.0,0.0);
	
	// Accumulate the contribution from each surrounding vertex
	for (int k = 0; k < 8; k++) {
	  disp      += dispNew[ni[k]]  * S[k];
	  acc      += gacceleration[ni[k]]   * S[k];
	}
	
          // Update the particle's position and velocity
          pxnew[idx]           = px[idx] + disp;
          pvelocitynew[idx] = pvelocity[idx] + (pacceleration[idx]+acc)*(.5* delT);
    
          double rho;
	  if(pvolume[idx] > 0.){
	    rho = pmass[idx]/pvolume[idx];
	  }
	  else{
	    rho = rho_init;
	  }
          pmassNew[idx]        = pmass[idx];
          pvolumeNew[idx]      = pmassNew[idx]/rho;
#if 1
	  if(pmassNew[idx] <= 3.e-15){
	    delete_particles->addParticle(idx);
	    pvelocitynew[idx] = Vector(0.,0.,0);
	    pxnew[idx] = px[idx];
	  }
#endif

          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
          massLost += (pmass[idx] - pmassNew[idx]);
        }
      
      
      // Store the new result
      new_dw->put(pxnew,           lb->pXLabel_preReloc);
      new_dw->put(pvelocitynew,    lb->pVelocityLabel_preReloc);
      new_dw->put(pexternalForceNew, lb->pExternalForceLabel_preReloc);
      new_dw->put(pmassNew,        lb->pMassLabel_preReloc);
      new_dw->put(pvolumeNew,      lb->pVolumeLabel_preReloc);
      new_dw->deleteParticles(delete_particles);
      delete delete_particles;

      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->allocate(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);
      new_dw->put(pids_new, lb->pParticleIDLabel_preReloc);
    }
    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

//  cout << "Solid mass lost this timestep = " << massLost << endl;
//  cout << "Solid momentum after advection = " << CMV << endl;

//  cout << "THERMAL ENERGY " << thermal_energy << endl;
  }
}


void ImpMPM::interpolateStressToGrid(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* ,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  // NOT DONE
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateStressToGrid on patch " << patch->getID()
	       <<"\t\t MPM"<< endl << endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pstress;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      new_dw->get(px,lb->pXLabel_preReloc,pset);
      new_dw->get(pmass,lb->pMassLabel_preReloc,pset);
      new_dw->get(pstress,lb->pStressLabel_preReloc,pset);

      NCVariable<Matrix3> gstress;

      new_dw->allocate(gstress,lb->gStressLabel,matlindex,patch);

      gstress.initialize(Matrix3(0.));

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
         double S[8];

         patch->findCellAndWeights(px[idx], ni, S);

         for (int k = 0; k < 8; k++){
	   if (patch->containsNode(ni[k])) {
	     gstress[ni[k]] += pstress[idx]*pmass[idx]*S[k];
	   }
         }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){

      }

      new_dw->put(gstress, lb->gStressLabel, matlindex,patch);


    }  // End of loop over materials
  }  // End of loop over patches
}


void ImpMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}
