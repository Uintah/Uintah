#include <sci_defs.h>
#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h> 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
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
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <Packages/Uintah/CCA/Components/MPM/Solver.h>
#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
#include <set>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream cout_doing("IMPM_DOING_COUT", false);

ImpMPM::ImpMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=0.;
  d_solver = 0;
}

ImpMPM::~ImpMPM()
{
  delete lb;

  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  delete d_solver;

}

void ImpMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& /*grid*/,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

  
   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

   ProblemSpecP mpm_ps = prob_spec->findBlock("MPM");

   string integrator_type;
   if (mpm_ps) {
     mpm_ps->get("time_integrator",integrator_type);
     if (integrator_type == "implicit")
       d_integrator = Implicit;
     else
       if (integrator_type == "explicit")
	 d_integrator = Explicit;
   } else
     d_integrator = Implicit;

    if (!mpm_ps->get("dynamic",dynamic))
       dynamic = true;

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, 8,integrator_type);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
   }
   string solver;
   if (!mpm_ps->get("solver",solver))
     solver = "petsc";

   if (solver == "petsc")
     d_solver = scinew MPMPetscSolver();
   else if (solver == "simple")
     d_solver = scinew SimpleSolver();
   d_solver->initialize();

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
  t->computes(lb->pVolumeOldLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pAccelerationLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel);
  t->computes(lb->bElBarLabel);
  t->computes(lb->dispIncQNorm0);
  t->computes(lb->dispIncNormMax);

  LoadBalancer* loadbal = sched->getLoadBalancer();
  d_perproc_patches = loadbal->createPerProcessorPatchSet(level,d_myworld);
  d_perproc_patches->addReference();

  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());

  t = scinew Task("ImpMPM::printParticleCount",
		  this, &ImpMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());

}

void ImpMPM::scheduleComputeStableTimestep(const LevelP&, SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void
ImpMPM::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched,
			     int, int )
{
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleInterpolateParticlesToGrid(sched,d_perproc_patches, matls);

  scheduleApplyBoundaryConditions(sched,d_perproc_patches,matls);

  scheduleDestroyMatrix(sched, d_perproc_patches, matls,false);

  scheduleCreateMatrix(sched, d_perproc_patches, matls,false);

  scheduleComputeStressTensorI(sched, d_perproc_patches, matls,false);

  scheduleFormStiffnessMatrixI(sched,d_perproc_patches,matls,false);

  scheduleComputeInternalForceI(sched, d_perproc_patches, matls,false);

  scheduleFormQI(sched, d_perproc_patches, matls,false);

#if 0
  scheduleApplyRigidBodyConditionI(sched, d_perproc_patches,matls);
#endif

  scheduleRemoveFixedDOFI(sched, d_perproc_patches, matls,false);

  scheduleSolveForDuCGI(sched, d_perproc_patches, matls,false);

  scheduleUpdateGridKinematicsI(sched, d_perproc_patches, matls,false);

  scheduleCheckConvergenceI(sched,level, d_perproc_patches, matls, false);

  scheduleIterate(sched,level,d_perproc_patches,matls);

  scheduleComputeStressTensorOnly(sched,d_perproc_patches,matls,false);

  scheduleComputeInternalForceII(sched,d_perproc_patches,matls,false);

  scheduleComputeAcceleration(sched,d_perproc_patches,matls);

  scheduleInterpolateToParticlesAndUpdate(sched, d_perproc_patches, matls);
#if 0
  scheduleInterpolateStressToGrid(sched,d_perproc_patches,matls);
#endif
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
  t->requires(Task::OldDW, lb->pVolumeOldLabel,       Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVelocityLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pXLabel,            Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pExternalForceLabel,Ghost::AroundNodes,1);



  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gVelocityOldLabel);
  t->computes(lb->dispNewLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalMassLabel);
  
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleApplyBoundaryConditions(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::applyBoundaryCondition",
		        this, &ImpMPM::applyBoundaryConditions);

  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gAccelerationLabel);

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleCreateMatrix(SchedulerP& sched,
				  const PatchSet* patches,
				  const MaterialSet* matls,
				  const bool recursion)
{

  Task* t = scinew Task("ImpMPM::createMatrix",this,&ImpMPM::createMatrix,
			recursion);
  sched->addTask(t, patches, matls);

}

void ImpMPM::scheduleDestroyMatrix(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls,
				   const bool recursion)
{

  Task* t = scinew Task("ImpMPM::destroyMatrix",this,&ImpMPM::destroyMatrix,
			recursion);
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
					  const MaterialSet* matls,
					  const bool recursion)
{

  Task* t = scinew Task("ImpMPM::formStiffnessMatrixI",
		    this, &ImpMPM::formStiffnessMatrix,recursion);

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleFormStiffnessMatrixR(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::formStiffnessMatrixR",
		    this, &ImpMPM::formStiffnessMatrix,recursion);

  t->requires(Task::ParentNewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::ParentOldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeInternalForceI(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::computeInternalForceI",
			this, &ImpMPM::computeInternalForce,recursion);
  
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);
  t->modifies(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleComputeInternalForceII(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
  Task* t = scinew Task("ImpMPM::computeInternalForceII",
			this, &ImpMPM::computeInternalForce,recursion);
  
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
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

  if (recursion) {
    t->requires(Task::ParentOldDW,lb->pXLabel,Ghost::AroundNodes,1);
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  }
  else {
    t->requires(Task::NewDW,lb->pStressLabel,Ghost::AroundNodes,1);
    t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);
  }
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->computes(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleIterate(SchedulerP& sched,const LevelP& level,
			     const PatchSet* patches, const MaterialSet* matl)
{

  // NOT DONE

  Task* task = scinew Task("scheduleIterate", this, &ImpMPM::iterate,level,
			   sched.get_rep());

  task->hasSubScheduler();

  // Required in computeStressTensor
  //task->requires(Task::NewDW,lb->dispNewLabel,Ghost::None,0);
#if 0
  task->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,0);
#endif

  // Trying out as was done with gVelocityOld
  // We get the parent's old_dw
  task->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeOldLabel,Ghost::None,0);

  task->requires(Task::OldDW,lb->pDeformationMeasureLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->bElBarLabel,Ghost::None,0);

  task->modifies(lb->dispNewLabel);
  task->modifies(lb->gVelocityLabel);

  task->requires(Task::NewDW,lb->gVelocityOldLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);

  task->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);

  task->requires(Task::OldDW,d_sharedState->get_delt_label());
  task->requires(Task::NewDW,lb->dispIncQNorm0);
  task->requires(Task::NewDW,lb->dispIncNormMax);
  task->requires(Task::NewDW,lb->dispIncQNorm);
  task->requires(Task::NewDW,lb->dispIncNorm);

  sched->addTask(task,d_perproc_patches,d_sharedState->allMaterials());

}


void ImpMPM::iterate(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset*,
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     LevelP level, Scheduler* sched)
{
  SchedulerP subsched = sched->createSubScheduler();
  DataWarehouse::ScrubMode old_dw_scrubmode = old_dw->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode new_dw_scrubmode = new_dw->setScrubbing(DataWarehouse::ScrubNone);
  subsched->initialize(3, 1, old_dw, new_dw);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3);
  
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);

  // Create the tasks

  scheduleDestroyMatrix(subsched, level->eachPatch(),
			d_sharedState->allMPMMaterials(),true);

  scheduleCreateMatrix(subsched, level->eachPatch(), 
		       d_sharedState->allMPMMaterials(),true);
  
  scheduleComputeStressTensorR(subsched,level->eachPatch(),
			      d_sharedState->allMPMMaterials(),
			      true);

  scheduleFormStiffnessMatrixR(subsched,level->eachPatch(),
			       d_sharedState->allMPMMaterials(),true);

  scheduleComputeInternalForceR(subsched,level->eachPatch(),
				d_sharedState->allMPMMaterials(), true);

  
  scheduleFormQR(subsched,level->eachPatch(),d_sharedState->allMPMMaterials(),
		 true);

  scheduleRemoveFixedDOFR(subsched,level->eachPatch(),
			  d_sharedState->allMPMMaterials(),true);

  scheduleSolveForDuCGR(subsched,level->eachPatch(),
		       d_sharedState->allMPMMaterials(), true);
  scheduleUpdateGridKinematicsR(subsched,level->eachPatch(),
			       d_sharedState->allMPMMaterials(),true);

  scheduleCheckConvergenceR(subsched,level,level->eachPatch(),
			       d_sharedState->allMPMMaterials(), true);
 
  subsched->compile(d_myworld);

  sum_vartype dispIncNorm,dispIncNormMax,dispIncQNorm,dispIncQNorm0;
  new_dw->get(dispIncNorm,lb->dispIncNorm);
  new_dw->get(dispIncQNorm,lb->dispIncQNorm); 
  new_dw->get(dispIncNormMax,lb->dispIncNormMax);
  new_dw->get(dispIncQNorm0,lb->dispIncQNorm0);
  cerr << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax << "\n";
  cerr << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 << "\n";
  
  int count = 0;
  bool dispInc = false;
  bool dispIncQ = false;
  double error = 1.e-30;
  
  if (dispIncNorm/dispIncNormMax <= error)
    dispInc = true;
  if (dispIncQNorm/dispIncQNorm0 <= 4.*error)
    dispIncQ = true;

  // Get all of the required particle data that is in the old_dw and put it 
  // in the subscheduler's  new_dw.  Then once dw is advanced, subscheduler
  // will be pulling data out of the old_dw.

  for (int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing iterate on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      ParticleSubset* pset = subsched->get_dw(0)->getParticleSubset(matlindex, 
								    patch);
      cerr << "number of particles = " << pset->numParticles() << "\n";
      // Need to pull out the pX, pVolume, and pVolumeOld from old_dw
      // and put it in the subschedulers new_dw.  Once it is advanced,
      // the subscheduler will be pulling data from the old_dw per the
      // task specifications in the scheduleIterate.
      // Get out some grid quantities: gmass, gInternalForce, gExternalForce

      constNCVariable<Vector> internal_force,dispInc;

      NCVariable<Vector> dispNew,velocity;
      new_dw->getModifiable(dispNew,lb->dispNewLabel,matlindex,patch);
      new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
      new_dw->get(internal_force,lb->gInternalForceLabel,matlindex,patch,
      		  Ghost::None,0);

      new_dw->getModifiable(velocity,lb->gVelocityLabel,matlindex,patch);
      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label());
      sum_vartype dispIncQNorm0,dispIncNormMax;
      new_dw->get(dispIncQNorm0,lb->dispIncQNorm);
      new_dw->get(dispIncNormMax,lb->dispIncNormMax);

      // New data to be stored in the subscheduler
      NCVariable<Vector> newdisp,new_int_force,	new_vel,new_disp_inc;

      double new_dt;
      subsched->get_dw(3)->allocateAndPut(newdisp,lb->dispNewLabel,
					     matlindex, patch);
      subsched->get_dw(3)->allocateAndPut(new_disp_inc,lb->dispIncLabel,
					     matlindex, patch);
      subsched->get_dw(3)->allocateAndPut(new_int_force,
					     lb->gInternalForceLabel,
					     matlindex,patch);
      subsched->get_dw(3)->allocateAndPut(new_vel,lb->gVelocityLabel,
					  matlindex,patch);

      subsched->get_dw(3)->saveParticleSubset(matlindex, patch, pset);
      newdisp.copyData(dispNew);
      new_disp_inc.copyData(dispInc);
      new_int_force.copyData(internal_force);
      new_vel.copyData(velocity);

      new_dt = dt;
      // These variables are ultimately retrieved from the subschedulers
      // old datawarehouse after the advancement of the data warehouse.
      subsched->get_dw(3)->put(delt_vartype(new_dt),
				  d_sharedState->get_delt_label());
      subsched->get_dw(3)->put(dispIncQNorm0,lb->dispIncQNorm0);
      subsched->get_dw(3)->put(dispIncNormMax,lb->dispIncNormMax);
      
    }
  }

  subsched->get_dw(3)->finalize();
  subsched->advanceDataWarehouse(grid);
  cerr << "dispInc = " << dispInc << " dispIncQ = " << dispIncQ << "\n";
  while(!dispInc && !dispIncQ) {
    cerr << "Iteration = " << count++ << "\n";
    subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute(d_myworld);
    subsched->get_dw(3)->get(dispIncNorm,lb->dispIncNorm);
    subsched->get_dw(3)->get(dispIncQNorm,lb->dispIncQNorm); 
    subsched->get_dw(3)->get(dispIncNormMax,lb->dispIncNormMax);
    subsched->get_dw(3)->get(dispIncQNorm0,lb->dispIncQNorm0);
    cerr << "Before dispIncNorm/dispIncNormMax . . . ." << endl;
    cerr << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax 
	 << "\n";
    cerr << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 
	 << "\n";
    if (dispIncNorm/dispIncNormMax <= error)
      dispInc = true;
    if (dispIncQNorm/dispIncQNorm0 <= 4.*error)
      dispIncQ = true;
    subsched->advanceDataWarehouse(grid);
  }

  // Move the particle data from subscheduler to scheduler.
  for (int p = 0; p < patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Getting the recursive data on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      // Needed in computeAcceleration 
      constNCVariable<Vector> velocity, dispNew;
      subsched->get_dw(2)->get(velocity,lb->gVelocityLabel,matlindex,patch,
				  Ghost::None,0);
      subsched->get_dw(2)->get(dispNew,lb->dispNewLabel,matlindex,patch,
				  Ghost::None,0);
      NCVariable<Vector> velocity_new, dispNew_new;
      new_dw->getModifiable(velocity_new,lb->gVelocityLabel,matlindex,patch);
      new_dw->getModifiable(dispNew_new,lb->dispNewLabel,matlindex,patch);
      velocity_new.copyData(velocity);
      dispNew_new.copyData(dispNew);
    }
  }
  old_dw->setScrubbing(old_dw_scrubmode);
  new_dw->setScrubbing(new_dw_scrubmode);
}

void ImpMPM::scheduleFormQI(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls, const bool recursion)
{
  Task* t = scinew Task("ImpMPM::formQI", this, 
			&ImpMPM::formQ, recursion);

  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleFormQR(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls,const bool recursion)
{
  Task* t = scinew Task("ImpMPM::formQR", this, 
			&ImpMPM::formQ,recursion);

  t->requires(Task::ParentOldDW,d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);

  // Old version used OldDW (had to copy), new version uses ParentNewDW
  // now no copying is required.
  t->requires(Task::ParentNewDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gVelocityOldLabel,Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gAccelerationLabel,Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gMassLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleApplyRigidBodyConditionI(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::applyRigidBodyConditionI", this, 
			&ImpMPM::applyRigidBodyCondition);
#if 0
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->modifies(Task::NewDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
#endif
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleApplyRigidBodyConditionR(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::applyRigidBodyConditionR", this, 
			&ImpMPM::applyRigidBodyCondition);
#if 0
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->modifies(Task::NewDW,lb->dispNewLabel);
  t->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
#endif
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleRemoveFixedDOFI(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls,
				     const bool recursion)
{

  Task* t = scinew Task("ImpMPM::removeFixedDOFI", this, 
			&ImpMPM::removeFixedDOF,recursion);

  t->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);

  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleRemoveFixedDOFR(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls,
				     const bool recursion)
{
  Task* t = scinew Task("ImpMPM::removeFixedDOFR", this, 
			&ImpMPM::removeFixedDOF,recursion);

  t->requires(Task::ParentNewDW,lb->gMassLabel,Ghost::None,0);

  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleSolveForDuCGI(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls,
				   const bool recursion)
{
  Task* t = scinew Task("ImpMPM::solveForDuCGI", this, 
			&ImpMPM::solveForDuCG,recursion);

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
			&ImpMPM::solveForDuCG,recursion);

  t->computes(lb->dispIncLabel);
    
  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleUpdateGridKinematicsI(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls,
					   const bool recursion)
{
  Task* t = scinew Task("ImpMPM::updateGridKinematicsI", this, 
			&ImpMPM::updateGridKinematics,recursion);
  
  t->modifies(lb->dispNewLabel);
  t->modifies(lb->gVelocityLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::NewDW,lb->gVelocityOldLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleUpdateGridKinematicsR(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls,
					   const bool recursion)
{
  Task* t = scinew Task("ImpMPM::updateGridKinematicsR", this, 
			&ImpMPM::updateGridKinematics,recursion);

  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->computes(lb->dispNewLabel);
  t->computes(lb->gVelocityLabel);
  t->requires(Task::ParentOldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gVelocityOldLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}



void ImpMPM::scheduleCheckConvergenceI(SchedulerP& sched, const LevelP& level,
				       const PatchSet* patches,
				       const MaterialSet* matls,
				       const bool recursion)
{
  // NOT DONE

  Task* t = scinew Task("ImpMPM::checkConvergenceI", this,
			&ImpMPM::checkConvergence, recursion);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispIncQNorm0);
  t->requires(Task::OldDW,lb->dispIncNormMax);

  t->computes(lb->dispIncNormMax);
  t->computes(lb->dispIncQNorm0);
  t->computes(lb->dispIncNorm);
  t->computes(lb->dispIncQNorm);
  
  sched->addTask(t,patches,matls);

  

}

void ImpMPM::scheduleCheckConvergenceR(SchedulerP& sched, const LevelP& level,
				       const PatchSet* patches,
				       const MaterialSet* matls,
				       const bool recursion)
{
  // NOT DONE

  Task* t = scinew Task("ImpMPM::checkConvergenceR", this,
			&ImpMPM::checkConvergence, recursion);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispIncQNorm0);
  t->requires(Task::OldDW,lb->dispIncNormMax);

  t->computes(lb->dispIncNormMax);
  t->computes(lb->dispIncQNorm0);
  t->computes(lb->dispIncNorm);
  t->computes(lb->dispIncQNorm);

  sched->addTask(t,patches,matls);

  

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
  t->requires(Task::NewDW, lb->gVelocityOldLabel,Ghost::None);
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
  t->requires(Task::NewDW, lb->gVelocityLabel,  Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->dispNewLabel,Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);
  t->requires(Task::OldDW, lb->pVolumeOldLabel,        Ghost::None);


  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pAccelerationLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pExtForceLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVolumeOldLabel_preReloc);

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
      cerr << "Created " << (long) pcount << " total particles\n";
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
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);
    new_dw->put(sum_vartype(0.),lb->dispIncQNorm0);
    new_dw->put(sum_vartype(0.),lb->dispIncNormMax);
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( matl );
      particleIndex numParticles = mpm_matl->countParticles(patch);
      totalParticles+=numParticles;
      
      mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

      mpm_matl->getConstitutiveModel()->initializeCMData(patch,
							 mpm_matl, new_dw);
       

    }
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
	       <<"\t\t IMPM"<< "\n" << "\n";

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal;
    new_dw->allocateAndPut(gmassglobal,lb->gMassLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume,pvolumeold;
      constParticleVariable<Vector> pvelocity, pacceleration,pexternalforce;

      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvolumeold,     lb->pVolumeOldLabel,     pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pacceleration,  lb->pAccelerationLabel,  pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity,gvelocity_old,gacceleration,dispNew;
      NCVariable<Vector> gexternalforce,ginternalforce;

      new_dw->allocateAndPut(gmass,lb->gMassLabel,      matlindex, patch);
      new_dw->allocateAndPut(gvolume,lb->gVolumeLabel,    matlindex, patch);
      new_dw->allocateAndPut(gvelocity,lb->gVelocityLabel,  matlindex, patch);
      new_dw->allocateAndPut(gvelocity_old,lb->gVelocityOldLabel,  matlindex,
			     patch);
      new_dw->allocateAndPut(dispNew,lb->dispNewLabel,  matlindex, patch);
      new_dw->allocateAndPut(gacceleration,lb->gAccelerationLabel,matlindex,
			     patch);
      new_dw->allocateAndPut(gexternalforce,lb->gExternalForceLabel,matlindex,
			     patch);
      new_dw->allocateAndPut(ginternalforce,lb->gInternalForceLabel,matlindex,
			     patch);


      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gvelocity_old.initialize(Vector(0,0,0));
      dispNew.initialize(Vector(0,0,0));
      gacceleration.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      ginternalforce.initialize(Vector(0,0,0));

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

	// cerr << "particle accel = " << pacceleration[idx] << "\n";

	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	  if(patch->containsNode(ni[k])) {
	    gmassglobal[ni[k]]    += pmass[idx]          * S[k];
	    gmass[ni[k]]          += pmass[idx]          * S[k];
	    gvolume[ni[k]]        += pvolumeold[idx]        * S[k];
	    gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	    gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	    gacceleration[ni[k]] += pacceleration[idx] * pmass[idx]* S[k];
	    totalmass += pmass[idx] * S[k];
	  }
	}
      }
      
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	if (!compare(gmass[*iter],0.)) {
	  gvelocity[*iter] /= gmass[*iter];
	  gacceleration[*iter] /= gmass[*iter];
	  //	  cerr << "gmass = " << gmass[*iter] << "\n";
	}
      }
      gvelocity_old.copyData(gvelocity);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);
    }  // End loop over materials
  }  // End loop over patches
}

void ImpMPM::applyBoundaryConditions(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  // NOT DONE
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing applyBoundaryConditions " <<"\t\t\t\t IMPM"
	       << "\n" << "\n";
    
  
    // Apply grid boundary conditions to the velocity before storing the data
    IntVector offset =  IntVector(0,0,0);
    for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      NCVariable<Vector> gvelocity,gacceleration;

      new_dw->getModifiable(gvelocity,lb->gVelocityLabel,matlindex,patch);
      new_dw->getModifiable(gacceleration,lb->gAccelerationLabel,matlindex,
			    patch);


      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	const BoundCondBase *vel_bcs, *sym_bcs;
	if (patch->getBCType(face) == Patch::None) {
	  vel_bcs  = patch->getBCValues(matlindex,"Velocity",face);
	  sym_bcs  = patch->getBCValues(matlindex,"Symmetric",face);
	} else
	  continue;
	
	if (vel_bcs != 0) {
	  const VelocityBoundCond* bc =
	    dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	  if (bc->getKind() == "Dirichlet") {
	    //cerr << "Velocity bc value = " << bc->getValue() << "\n";
	    fillFace(gvelocity,patch, face,bc->getValue(),offset);
	    fillFace(gacceleration,patch, face,bc->getValue(),offset);
	  }
	}
	if (sym_bcs != 0) 
	  fillFaceNormal(gvelocity,patch, face,offset);
	  fillFaceNormal(gacceleration,patch, face,offset);
	
      }
    }
  }
}

void ImpMPM::createMatrix(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw,
			  const bool recursion)

{
  if (recursion)
    return;

  d_solver->createLocalToGlobalMapping(d_myworld,d_perproc_patches,patches);
  d_solver->createMatrix(d_myworld);

}

void ImpMPM::destroyMatrix(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* ,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw,
			   const bool recursion)
{
  cout_doing <<"Doing destroyMatrix " <<"\t\t\t\t\t IMPM"
	       << "\n" << "\n";

  d_solver->destroyMatrix(recursion);

}

void ImpMPM::computeStressTensor(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // DONE

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t IMPM"<< "\n" << "\n";

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->computeStressTensorImplicit(patches, mpm_matl, old_dw, new_dw,
				    d_solver, recursion);
  }
  
}

void ImpMPM::computeStressTensorOnly(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  // DONE

  cout_doing <<"Doing computeStressTensorOnly " <<"\t\t\t\t IMPM"<< "\n" 
	     << "\n";

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
				      DataWarehouse* new_dw,
				      const bool recursion)

{
  // DONE

  int nn = 0;
  IntVector nodes(0,0,0);
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    IntVector num_nodes = patch->getNNodes();
    nn += (num_nodes.x())*(num_nodes.y())*(num_nodes.z())*3;
    nodes = IntVector(Max(num_nodes.x(),nodes.x()),
		      Max(num_nodes.y(),nodes.y()),
		      Max(num_nodes.z(),nodes.z()));
  }
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
   
      constNCVariable<double> gmass;
      delt_vartype dt;
      if (recursion) {
	DataWarehouse* parent_new_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentNewDW);
	parent_new_dw->get(gmass, lb->gMassLabel,matlindex,patch,
			   Ghost::None,0);
	DataWarehouse* parent_old_dw =
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW);
	parent_old_dw->get(dt,d_sharedState->get_delt_label());
      } else {
	new_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
      	old_dw->get(dt, d_sharedState->get_delt_label() );
      }
     
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); 
	   iter++) {
	IntVector n = *iter;
	int dof[3];
	int l2g_node_num = l2g[n];
	dof[0] = l2g_node_num;
	dof[1] = l2g_node_num+1;
	dof[2] = l2g_node_num+2;

	double v = gmass[*iter]*(4./(dt*dt));
	d_solver->fillMatrix(dof[0],dof[0],v);
	d_solver->fillMatrix(dof[1],dof[1],v);
	d_solver->fillMatrix(dof[2],dof[2],v);
      }
    } 
  }
  d_solver->finalizeMatrix();
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
	       <<"\t\t\t IMPM"<< "\n" << "\n";
    
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
      constParticleVariable<double>  pvol;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;
      ParticleSubset* pset;
      
      if (recursion) {
	DataWarehouse* parent_old_dw = 
	  new_dw->getOtherDataWarehouse(Task::ParentOldDW);
	pset = parent_old_dw->getParticleSubset(matlindex, patch,
						Ghost::AroundNodes, 1,
						lb->pXLabel);
	parent_old_dw->get(px,lb->pXLabel, pset);
      	new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,matlindex,
			       patch);
      } else {
	pset = old_dw->getParticleSubset(matlindex, patch,
						Ghost::AroundNodes, 1,
						lb->pXLabel);
	old_dw->get(px,lb->pXLabel,pset);

	new_dw->getModifiable(internalforce,lb->gInternalForceLabel,matlindex,
			      patch);
      }
      
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

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
    }
  }

}


void ImpMPM::formQ(const ProcessorGroup*, const PatchSubset* patches,
			const MaterialSubset*, DataWarehouse* old_dw,
			DataWarehouse* new_dw, const bool recursion)
{
  // DONE
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formQ on patch " << patch->getID()
	       <<"\t\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    delt_vartype dt;

    int matlindex = 0;

    constNCVariable<Vector> externalForce, internalForce;
    constNCVariable<Vector> dispNew,velocity,accel;
    constNCVariable<double> mass;
    if (recursion) {
      DataWarehouse* parent_new_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      DataWarehouse* parent_old_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      parent_old_dw->get(dt,d_sharedState->get_delt_label());
      new_dw->get(internalForce,lb->gInternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      parent_new_dw->get(externalForce,lb->gExternalForceLabel,matlindex,patch,
			 Ghost::None,0);
      old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      parent_new_dw->get(velocity,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      parent_new_dw->get(accel,lb->gAccelerationLabel,matlindex,patch,
		Ghost::None,0);
      parent_new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);

      
    } else {
      new_dw->get(internalForce,lb->gInternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      new_dw->get(externalForce,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      new_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      new_dw->get(velocity,lb->gVelocityLabel,matlindex,patch,
		  Ghost::None,0);
      new_dw->get(accel,lb->gAccelerationLabel,matlindex,patch,
		Ghost::None,0);
      new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
      old_dw->get(dt, d_sharedState->get_delt_label());
    }
    double fodts = 4./(dt*dt);
    double fodt = 4./dt;
    
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int l2g_node_num = l2g[n];
      dof[0] = l2g_node_num;
      dof[1] = l2g_node_num+1;
      dof[2] = l2g_node_num+2;

      double v[3];
      v[0] = externalForce[n].x() + internalForce[n].x();
      v[1] = externalForce[n].y() + internalForce[n].y();
      v[2] = externalForce[n].z() + internalForce[n].z();
      
      // temp2 = M*a^(k-1)(t+dt)
      if (dynamic) {
	v[0] -= (dispNew[n].x()*fodts - velocity[n].x()*fodt -
		 accel[n].x())*mass[n];
	v[1] -= (dispNew[n].y()*fodts - velocity[n].y()*fodt -
		 accel[n].y())*mass[n];
	v[2] -= (dispNew[n].z()*fodts - velocity[n].z()*fodt -
		 accel[n].z())*mass[n];
      }
      d_solver->fillVector(dof[0],double(v[0]));
      d_solver->fillVector(dof[1],double(v[1]));
      d_solver->fillVector(dof[2],double(v[2]));
    }
  }
  d_solver->assembleVector();
}


void ImpMPM::applyRigidBodyCondition(const ProcessorGroup*, 
				      const PatchSubset* patches,
				      const MaterialSubset*, 
				      DataWarehouse*,
				      DataWarehouse*)
{
  // NOT DONE
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing (NOT DONE) applyRigidbodyCondition on patch " 
	       << patch->getID()  <<"\t\t IMPM"<< "\n" << "\n";

#if 0
    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;

    int matlindex = 0;

    NCVariable<Vector> dispNew;
    constNCVariable<Vector> velocity;

    new_dw->getModifiable(dispNew,lb->dispNewLabel,matlindex,patch);
    new_dw->get(velocity,lb->gVelocityLabel,matlindex,patch,
		Ghost::None,0);
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
    }
#endif
  }

}



void ImpMPM::removeFixedDOF(const ProcessorGroup*, 
				 const PatchSubset* patches,
				 const MaterialSubset*, 
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // NOT DONE
  
  int num_nodes = 0;
  int matlindex = 0;
  set<int> fixedDOF;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing removeFixedDOF on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    IntVector nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z())*3;
    
    constNCVariable<double> mass;
    if (recursion) {
      DataWarehouse* parent_new_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      parent_new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    }  else
      new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int l2g_node_num = l2g[n];
      dof[0] = l2g_node_num;
      dof[1] = l2g_node_num+1;
      dof[2] = l2g_node_num+2;
      
      // Just look on the grid to see if the gmass is 0 and then remove that  
      if (compare(mass[n],0.)) {
	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }

    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      if (patch->getBCType(face)==Patch::None) { 
	IntVector l,h;
	patch->getFaceNodes(face,0,l,h);
	for(NodeIterator it(l,h); !it.done(); it++) {
	  IntVector n = *it;
	  int dof[3];
	  int l2g_node_num = l2g[n];
	  dof[0] = l2g_node_num;
	  dof[1] = l2g_node_num+1;
	  dof[2] = l2g_node_num+2;
	  
	  fixedDOF.insert(dof[0]);
	  fixedDOF.insert(dof[1]);
	  fixedDOF.insert(dof[2]);
	}
      }
    }
  }
  d_solver->removeFixedDOF(fixedDOF,num_nodes);


}


void ImpMPM::solveForDuCG(const ProcessorGroup* pg,
			       const PatchSubset* patches,
			       const MaterialSubset* ,
			       DataWarehouse*,
			       DataWarehouse* new_dw,
			       const bool /*recursion*/)

{
  // DONE
  int num_nodes = 0;
  IntVector nodes(0,0,0);
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z())*3;
  }

  d_solver->solve();

  int matlindex = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    
    cout_doing <<"Doing solveForDuCG on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";
    nodes = patch->getNNodes();
    NCVariable<Vector> dispInc;
    
    new_dw->allocateAndPut(dispInc,lb->dispIncLabel,matlindex,patch);
    dispInc.initialize(Vector(0.,0.,0.));

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    vector<double> x;
    int begin = d_solver->getSolution(x);

    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int l2g_node_num = l2g[n] - begin;
      dof[0] = l2g_node_num;
      dof[1] = l2g_node_num+1;
      dof[2] = l2g_node_num+2;
      dispInc[n] = Vector(x[dof[0]],x[dof[1]],x[dof[2]]);
    }
  }

}

void ImpMPM::updateGridKinematics(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw,
				  const bool recursion)

{
  // DONE
  for (int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing updateGridKinematics on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    int matlindex = 0;

    NCVariable<Vector> dispNew,velocity;
    constNCVariable<Vector> dispInc,dispNew_old,velocity_old;

    delt_vartype dt;

    if (recursion) {
      DataWarehouse* parent_new_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      DataWarehouse* parent_old_dw = 
	new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      parent_old_dw->get(dt, d_sharedState->get_delt_label());
      old_dw->get(dispNew_old, lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
      new_dw->allocateAndPut(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->allocateAndPut(velocity, lb->gVelocityLabel, matlindex,patch);
      parent_new_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
			 Ghost::None,0);
    }
    else {
      new_dw->getModifiable(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->getModifiable(velocity, lb->gVelocityLabel, matlindex,patch);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
      new_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(dt, d_sharedState->get_delt_label());
    } 

    
    if (recursion) {
      if (dynamic) {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] = dispNew_old[*iter] + dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt) - velocity_old[*iter];
	}
      } else {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] = dispNew_old[*iter] + dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt);
	}
      }
    } else {
      if (dynamic) {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] += dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt) - velocity[*iter];
	}
      } else {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] += dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt);
	}
      }
    }
  }

}



void ImpMPM::checkConvergence(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* ,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw,
			      const bool recursion)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector nodes = patch->getNNodes();

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    cout_doing <<"Doing checkConvergence on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constNCVariable<Vector> dispInc;
      new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
      
      double dispIncNorm = 0.;
      double dispIncQNorm = 0.;
      vector<double> getQ;
      int begin = d_solver->getSolution(getQ);
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++) {
	IntVector n = *iter;
	int dof[3];
	int l2g_node_num = l2g[n] - begin;
	dof[0] = l2g_node_num;
	dof[1] = l2g_node_num+1;
	dof[2] = l2g_node_num+2;
	dispIncNorm += Dot(dispInc[n],dispInc[n]);
	dispIncQNorm += dispInc[n].x()*getQ[dof[0]] + 
	  dispInc[n].y()*getQ[dof[1]] +  dispInc[n].z()*getQ[dof[2]];
      }
      // We are computing both dispIncQNorm0 and dispIncNormMax (max residuals)
      // We are computing both dispIncQNorm and dispIncNorm (current residuals)

      double dispIncQNorm0,dispIncNormMax;
      sum_vartype dispIncQNorm0_var,dispIncNormMax_var;
      old_dw->get(dispIncQNorm0_var,lb->dispIncQNorm0);
      old_dw->get(dispIncNormMax_var,lb->dispIncNormMax);

      cerr << "dispIncQNorm0_var = " << dispIncQNorm0_var << "\n";
      cerr << "dispIncNormMax_var = " << dispIncNormMax_var << "\n";
      cerr << "dispIncNorm = " << dispIncNorm << "\n";
      cerr << "dispIncNormQ = " << dispIncQNorm << "\n";
      dispIncQNorm0 = dispIncQNorm0_var;
      dispIncNormMax = dispIncNormMax_var;

      if (!recursion || dispIncQNorm0 == 0.)
	dispIncQNorm0 = dispIncQNorm;

      if (dispIncNorm > dispIncNormMax)
	dispIncNormMax = dispIncNorm;

      cerr << "dispIncQNorm0 = " << dispIncQNorm0 << "\n";
      cerr << "dispIncQNorm = " << dispIncQNorm << "\n";
      cerr << "dispIncNormMax = " << dispIncNormMax << "\n";
      cerr << "dispIncNorm = " << dispIncNorm << "\n";

      new_dw->put(sum_vartype(dispIncNormMax),lb->dispIncNormMax);
      new_dw->put(sum_vartype(dispIncQNorm0),lb->dispIncQNorm0);
      new_dw->put(sum_vartype(dispIncNorm),lb->dispIncNorm);
      new_dw->put(sum_vartype(dispIncQNorm),lb->dispIncQNorm);

    }  // End of loop over materials
  }  // End of loop over patches

  
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
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get required variables for this patch
      NCVariable<Vector> acceleration;
      constNCVariable<Vector> velocity,dispNew;
      delt_vartype delT;

      new_dw->getModifiable(acceleration,lb->gAccelerationLabel,dwindex,patch);
      new_dw->get(velocity,lb->gVelocityOldLabel,dwindex, patch,
		  Ghost::None, 0);
      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::None,0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double fodts = 4./(delT*delT);
      double fodt = 4./(delT);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	acceleration[*iter] = dispNew[*iter]*fodts - velocity[*iter]*fodt
	  - acceleration[*iter];
      }

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
	       << patch->getID() <<"\t IMPM"<< "\n" << "\n";

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
      ParticleVariable<Vector> pvelocitynew, pexternalForceNew, paccNew;
      constParticleVariable<double> pmass, pvolume,pvolumeold;
      ParticleVariable<double> pmassNew,pvolumeNew,newpvolumeold;
  
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> dispNew, gacceleration,gvelocity;
      constNCVariable<double> dTdt;

      delt_vartype delT;

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);

      ParticleSubset* delete_particles = scinew ParticleSubset
	(pset->getParticleSet(),false,dwindex,patch);
    
      old_dw->get(px,                    lb->pXLabel,                    pset);
      old_dw->get(pmass,                 lb->pMassLabel,                 pset);
      new_dw->get(pvolume,               lb->pVolumeDeformedLabel,       pset);
      old_dw->get(pvolumeold,            lb->pVolumeOldLabel,            pset);
      old_dw->get(pexternalForce,        lb->pExternalForceLabel,        pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,             pset);
      old_dw->get(pacceleration,         lb->pAccelerationLabel,         pset);
      new_dw->allocateAndPut(pvelocitynew,lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(paccNew,    lb->pAccelerationLabel_preReloc,pset);
      new_dw->allocateAndPut(pxnew,      lb->pXLabel_preReloc,           pset);
      new_dw->allocateAndPut(pmassNew,   lb->pMassLabel_preReloc,        pset);
      new_dw->allocateAndPut(pvolumeNew, lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(newpvolumeold,lb->pVolumeOldLabel_preReloc, pset);
      new_dw->allocateAndPut(pexternalForceNew,
			     lb->pExtForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::AroundCells,1);

      new_dw->get(gacceleration,lb->gAccelerationLabel,dwindex, patch, 
		  Ghost::AroundCells, 1);

      new_dw->get(gvelocity,      lb->gVelocityLabel,
		  dwindex, patch, Ghost::AroundCells, 1);
     
      NCVariable<double> dTdt_create, massBurnFraction_create;	
      new_dw->allocateTemporary(dTdt_create, patch,Ghost::None,0);
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
    
	  paccNew[idx] = acc;
	  cerr << "position = " << pxnew[idx] << "\n";
	  cerr << "acceleration = " << paccNew[idx] << "\n";
          double rho;
	  if(pvolume[idx] > 0.){
	    rho = pmass[idx]/pvolume[idx];
	  }
	  else{
	    rho = rho_init;
	  }
          pmassNew[idx]        = pmass[idx];
          pvolumeNew[idx]      = pmassNew[idx]/rho;
	  newpvolumeold[idx] = pvolumeold[idx];

	  if(pmassNew[idx] <= 3.e-15){
	    delete_particles->addParticle(idx);
	    pvelocitynew[idx] = Vector(0.,0.,0);
	    pxnew[idx] = px[idx];
	  }

          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
          massLost += (pmass[idx] - pmassNew[idx]);
        }
      
           
      new_dw->deleteParticles(delete_particles);
      
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);
     
    }
    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

//  cerr << "Solid mass lost this timestep = " << massLost << "\n";
//  cerr << "Solid momentum after advection = " << CMV << "\n";

//  cerr << "THERMAL ENERGY " << thermal_energy << "\n";
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
	       <<"\t\t IMPM"<< "\n" << "\n";

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pstress;

      ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch,
					       Ghost::AroundNodes, 1,
					       lb->pXLabel_preReloc);

      new_dw->get(px,lb->pXLabel_preReloc,pset);
      new_dw->get(pmass,lb->pMassLabel_preReloc,pset);
      new_dw->get(pstress,lb->pStressLabel_preReloc,pset);

      NCVariable<Matrix3> gstress;

      new_dw->allocateAndPut(gstress,lb->gStressLabel,matlindex,patch);

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
    }  // End of loop over materials
  }  // End of loop over patches
}


void ImpMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}
