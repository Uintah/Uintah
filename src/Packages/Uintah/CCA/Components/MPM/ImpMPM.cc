#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h> // 
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
#include <set>
#include <iostream>
#include <fstream>


using namespace Uintah;
using namespace SCIRun;
using namespace std;

#define PETSC_DEBUG

static DebugStream cout_doing("IMPM_DOING_COUT", false);
// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

ImpMPM::ImpMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=0.;

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

   cout << "integrator type = " << integrator_type << " " << d_integrator << "\n";
   
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

   cout << "Number of materials: " << d_sharedState->getNumMatls() << "\n";


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
     lb->registerPermanentParticleState(m,lb->pVolumeOldLabel,
					lb->pVolumeOldLabel_preReloc);
     mpm_matl->getConstitutiveModel()->addParticleState(lb->d_particleState[m],
					lb->d_particleState_preReloc[m]);
   }
#ifdef HAVE_PETSC
   int argc = 1;
   char** argv;
   argv = new char*[argc];
   argv[0] = "ImpMPM::problemSetup";
   PetscInitialize(&argc,&argv, PETSC_NULL, PETSC_NULL);
#endif
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

  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());
#if 0
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
#endif

  t = scinew Task("ImpMPM::printParticleCount",
		  this, &ImpMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());
#if 0
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
#endif
}

void ImpMPM::scheduleComputeStableTimestep(const LevelP&, SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void ImpMPM::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  //  const PatchSet* patches = level->eachPatch();
#if 0
  LoadBalancer* loadbal = sched->getLoadBalancer();
  d_perproc_patches = loadbal->createPerProcessorPatchSet(level,d_myworld);
#endif

  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleInterpolateParticlesToGrid(sched,d_perproc_patches, matls);

  scheduleApplyBoundaryConditions(sched,d_perproc_patches,matls);

  scheduleDestroyMatrix(sched, d_perproc_patches, matls);
#if 1
  scheduleCreateMatrix(sched, d_perproc_patches, matls);
#endif

  scheduleComputeStressTensorI(sched, d_perproc_patches, matls,false);

  scheduleFormStiffnessMatrixI(sched,d_perproc_patches,matls,false);

  scheduleComputeInternalForceI(sched, d_perproc_patches, matls,true);

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
				  const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::createMatrix",this,&ImpMPM::createMatrix);
  sched->addTask(t, patches, matls);

}

void ImpMPM::scheduleDestroyMatrix(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls)
{

  Task* t = scinew Task("ImpMPM::destroyMatrix",this,&ImpMPM::destroyMatrix);
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
  t->assumeDataInNewDW();
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

#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::formStiffnessMatrixI",
		    this, &ImpMPM::formStiffnessMatrixPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::formStiffnessMatrixI",
		    this, &ImpMPM::formStiffnessMatrix,recursion);
#endif

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleFormStiffnessMatrixR(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls,
					  const bool recursion)
{
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::formStiffnessMatrixR",
		    this, &ImpMPM::formStiffnessMatrixPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::formStiffnessMatrixR",
		    this, &ImpMPM::formStiffnessMatrix,recursion);
#endif
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,lb->gMassLabel, Ghost::None);
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
  t->assumeDataInNewDW();
  if (recursion)
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  else
    t->requires(Task::NewDW,lb->pStressLabel,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::AroundNodes,1);
  t->computes(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleMoveData(SchedulerP& sched,const LevelP& level,
			     const PatchSet* patches, const MaterialSet* matls)
{

  Task* task = scinew Task("moveData",this,&ImpMPM::moveData);

  task->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeOldLabel,Ghost::None,0);

  task->requires(Task::OldDW,lb->pDeformationMeasureLabel_preReloc,
		 Ghost::None,0);
  task->requires(Task::OldDW,lb->bElBarLabel_preReloc,Ghost::None,0);
  task->requires(Task::OldDW,lb->gMassLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gVelocityOldLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gExternalForceLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gAccelerationLabel,Ghost::None,0);

  task->requires(Task::OldDW,d_sharedState->get_delt_label());
  task->computes(lb->pXLabel);
  task->computes(lb->pVolumeLabel);
  task->computes(lb->pVolumeOldLabel);

  task->modifies(lb->pDeformationMeasureLabel_preReloc);
  task->modifies(lb->bElBarLabel_preReloc);

  task->computes(lb->gMassLabel);
  task->computes(lb->gExternalForceLabel);

  task->computes(lb->gVelocityOldLabel);
  task->computes(lb->gAccelerationLabel);
  task->computes(d_sharedState->get_delt_label());


  sched->addTask(task,level->eachPatch(),d_sharedState->allMPMMaterials());

}

void ImpMPM::scheduleIterate(SchedulerP& sched,const LevelP& level,
			     const PatchSet*, const MaterialSet* matl)
{

  // NOT DONE

  Task* task = scinew Task("scheduleIterate", this, &ImpMPM::iterate,level,
			   sched);

  task->assumeDataInNewDW();
  task->hasSubScheduler();

  // Required in computeStressTensor
  //task->requires(Task::NewDW,lb->dispNewLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,0);
  task->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeOldLabel,Ghost::None,0);

  task->modifies(lb->pDeformationMeasureLabel_preReloc);
  task->modifies(lb->bElBarLabel_preReloc);



  task->modifies(lb->dispNewLabel);
  task->requires(Task::NewDW,lb->gVelocityOldLabel,Ghost::None,0);
  task->modifies(lb->gVelocityLabel);

  task->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  task->requires(Task::OldDW,d_sharedState->get_delt_label());
  task->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  //  task->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);

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
		     LevelP level, SchedulerP sched)
{
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize();
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);

  // Create the tasks

  scheduleDestroyMatrix(subsched, level->eachPatch(),
			d_sharedState->allMPMMaterials());

  scheduleCreateMatrix(subsched, level->eachPatch(), 
		       d_sharedState->allMPMMaterials());
  
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

  scheduleMoveData(subsched,level,level->eachPatch(),
		   d_sharedState->allMPMMaterials());

 
  subsched->compile(d_myworld,false);

  sum_vartype dispIncNorm,dispIncNormMax,dispIncQNorm,dispIncQNorm0;
  new_dw->get(dispIncNorm,lb->dispIncNorm);
  new_dw->get(dispIncQNorm,lb->dispIncQNorm); 
  new_dw->get(dispIncNormMax,lb->dispIncNormMax);
  new_dw->get(dispIncQNorm0,lb->dispIncQNorm0);
  cout << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax << "\n";
  cout << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 << "\n";
  
  int count = 0;
  bool dispInc = false;
  bool dispIncQ = false;
  double error = 1.e-30;
  
  if (dispIncNorm/dispIncNormMax <= error)
    dispInc = true;
  if (dispIncQNorm/dispIncQNorm0 <= 4.*error)
    dispIncQ = true;

  // subsched->set_old_dw(old_dw);
  //  subsched->set_new_dw(new_dw);
 
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
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
      cout << "number of particles = " << pset->numParticles() << "\n";
      // Need to pull out the pX, pVolume, and pVolumeOld from old_dw
      // and put it in the subschedulers new_dw.  Once it is advanced,
      // the subscheduler will be pulling data from the old_dw per the
      // task specifications in the scheduleIterate.
      // Get out some grid quantities: gmass, gInternalForce, gExternalForce

      constNCVariable<double> gmass;
      constNCVariable<Vector> internal_force,external_force,acc,dispInc,
	velocity_old;
      NCVariable<Vector> dispNew,velocity;
      new_dw->getModifiable(dispNew,lb->dispNewLabel,matlindex,patch);
      new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
      new_dw->get(internal_force,lb->gInternalForceLabel,matlindex,patch,
      		  Ghost::None,0);
      new_dw->get(external_force,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      // We need both the old velocity and new_dw velocity
      new_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      new_dw->getModifiable(velocity,lb->gVelocityLabel,matlindex,patch);
      new_dw->get(acc,lb->gAccelerationLabel,matlindex,patch, Ghost::None,0);
      new_dw->get(gmass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label());
      sum_vartype dispIncQNorm0,dispIncNormMax;
      new_dw->get(dispIncQNorm0,lb->dispIncQNorm);
      new_dw->get(dispIncNormMax,lb->dispIncNormMax);
      constParticleVariable<Matrix3> pstress;
      ParticleVariable<Matrix3> bElBar,deformationGradient;
      new_dw->get(pstress,lb->pStressLabel_preReloc,pset);

      new_dw->getModifiable(deformationGradient, 
			    lb->pDeformationMeasureLabel_preReloc, pset);
      new_dw->getModifiable(bElBar, lb->bElBarLabel_preReloc, pset);
      constParticleVariable<Point> px;
      old_dw->get(px,lb->pXLabel,pset);
      constParticleVariable<double> pvolume,pvolumeold;
      old_dw->get(pvolume,lb->pVolumeLabel,pset);
      old_dw->get(pvolumeold,lb->pVolumeOldLabel,pset);
      // New data to be stored in the subscheduler
      NCVariable<Vector> newdisp,new_int_force,new_ext_force,new_vel_old,
	new_vel,new_acc,new_disp_inc;
      NCVariable<double> newgmass;
      ParticleVariable<Matrix3> newpstress,newdefGrad,newbElBar;
      ParticleVariable<Point> newpx;
      ParticleVariable<double> newpvolume,newpvolumeold;
      double new_dt;
      subsched->get_new_dw()->allocateAndPut(newdisp,lb->dispNewLabel,
					     matlindex, patch);
      subsched->get_new_dw()->allocateAndPut(new_disp_inc,lb->dispIncLabel,
					     matlindex, patch);
      subsched->get_new_dw()->allocateAndPut(new_int_force,
					     lb->gInternalForceLabel,
					     matlindex,patch);
      subsched->get_new_dw()->allocateAndPut(new_ext_force,
					     lb->gExternalForceLabel,
					     matlindex,patch);
      subsched->get_new_dw()->allocateAndPut(new_vel,lb->gVelocityLabel,
					     matlindex,patch);
      subsched->get_new_dw()->allocateAndPut(new_vel_old,lb->gVelocityOldLabel,
					     matlindex,patch);
      subsched->get_new_dw()->allocateAndPut(new_acc,lb->gAccelerationLabel,
					     matlindex,patch);
      subsched->get_new_dw()->allocateAndPut(newgmass,lb->gMassLabel,matlindex,
					     patch);
      subsched->get_new_dw()->allocateAndPut(newpstress,
					     lb->pStressLabel_preReloc,
					     pset);
#if 0
      subsched->get_new_dw()->allocateAndPut(newdefGrad,
					     lb->pDeformationMeasureLabel,
					     pset);
#endif
      subsched->get_new_dw()->allocateAndPut(newdefGrad,
					 lb->pDeformationMeasureLabel_preReloc,
					     pset);
      // subsched->get_new_dw()->allocateAndPut(newbElBar,lb->bElBarLabel,pset);
      subsched->get_new_dw()->allocateAndPut(newbElBar,
					     lb->bElBarLabel_preReloc,
					     pset);
      subsched->get_new_dw()->allocateAndPut(newpx,lb->pXLabel, pset);
      subsched->get_new_dw()->allocateAndPut(newpvolume,lb->pVolumeLabel,pset);
      subsched->get_new_dw()->allocateAndPut(newpvolumeold,lb->pVolumeOldLabel,
					     pset);
      subsched->get_new_dw()->saveParticleSubset(matlindex, patch, pset);
      newdisp.copyData(dispNew);
      new_disp_inc.copyData(dispInc);
      new_int_force.copyData(internal_force);
      new_ext_force.copyData(external_force);
      new_vel.copyData(velocity);
      new_vel_old.copyData(velocity_old);
      new_acc.copyData(acc);
      newgmass.copyData(gmass);
      newpstress.copyData(pstress);
      newdefGrad.copyData(deformationGradient);
      newbElBar.copyData(bElBar);
      newpx.copyData(px);
      newpvolume.copyData(pvolume);
      newpvolumeold.copyData(pvolumeold);
      new_dt = dt;
      // These variables are ultimately retrieved from the subschedulers
      // old datawarehouse after the advancement of the data warehouse.
      subsched->get_new_dw()->put(delt_vartype(new_dt),
				  d_sharedState->get_delt_label());
      subsched->get_new_dw()->put(dispIncQNorm0,lb->dispIncQNorm0);
      subsched->get_new_dw()->put(dispIncNormMax,lb->dispIncNormMax);
      
    }
  }

  subsched->get_new_dw()->finalize();
  subsched->advanceDataWarehouse(grid);
  cout << "dispInc = " << dispInc << " dispIncQ = " << dispIncQ << "\n";
  while(!dispInc && !dispIncQ) {
    cout << "Iteration = " << count++ << "\n";
    //    subsched->advanceDataWarehouse(grid);
    subsched->execute(d_myworld);
    subsched->get_new_dw()->get(dispIncNorm,lb->dispIncNorm);
    subsched->get_new_dw()->get(dispIncQNorm,lb->dispIncQNorm); 
    subsched->get_new_dw()->get(dispIncNormMax,lb->dispIncNormMax);
    subsched->get_new_dw()->get(dispIncQNorm0,lb->dispIncQNorm0);
    cout << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax 
	 << "\n";
    cout << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 
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
      ParticleSubset* pset = 
	subsched->get_old_dw()->getParticleSubset(matlindex, patch);
      cout << "number of particles = " << pset->numParticles() << "\n";
      // Needed in computeStressTensorOnly
      constParticleVariable<Matrix3> bElbar,deformationGradient;
      subsched->get_old_dw()->get(bElbar,lb->bElBarLabel_preReloc, pset);
      subsched->get_old_dw()->get(deformationGradient,
				  lb->pDeformationMeasureLabel_preReloc, pset);
      ParticleVariable<Matrix3> bElbar_new,deformationGradient_new;
      new_dw->getModifiable(bElbar_new,lb->bElBarLabel_preReloc,pset);
      new_dw->getModifiable(deformationGradient_new,
			    lb->pDeformationMeasureLabel_preReloc,pset);
      bElbar_new.copyData(bElbar);
      deformationGradient_new.copyData(deformationGradient);

      // Needed in computeAcceleration 
      constNCVariable<Vector> velocity, dispNew;
      subsched->get_old_dw()->get(velocity,lb->gVelocityLabel,matlindex,patch,
				  Ghost::None,0);
      subsched->get_old_dw()->get(dispNew,lb->dispNewLabel,matlindex,patch,
				  Ghost::None,0);
      NCVariable<Vector> velocity_new, dispNew_new;
      new_dw->getModifiable(velocity_new,lb->gVelocityLabel,matlindex,patch);
      new_dw->getModifiable(dispNew_new,lb->dispNewLabel,matlindex,patch);
      velocity_new.copyData(velocity);
      dispNew_new.copyData(dispNew);
    }
  }
}

void ImpMPM::moveData(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset*,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw)
{
  // It appears like the purpose of moveData is to take any particle
  // variables that are in the old_dw and move them to the new_dw
  for (int p = 0; p < patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing moveData on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      ParticleSubset* pset = 
	old_dw->getParticleSubset(matlindex, patch);
      cout << "number of particles in move = " << pset->numParticles() << "\n";
      
      constParticleVariable<Point> px_old;
      old_dw->get(px_old,lb->pXLabel,pset);
      constParticleVariable<double> pvol_old,pvol_old_old;
      old_dw->get(pvol_old,lb->pVolumeLabel,pset);
      old_dw->get(pvol_old_old,lb->pVolumeOldLabel,pset);
      constParticleVariable<Matrix3> deformationGradient,bElBar;

      old_dw->get(deformationGradient, lb->pDeformationMeasureLabel_preReloc, 
		  pset);

      old_dw->get(bElBar, lb->bElBarLabel_preReloc, pset);
      constNCVariable<double> gmass;
      old_dw->get(gmass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label());
      constNCVariable<Vector> ext_force,acc,dispInc,velocity_old;
      old_dw->get(ext_force,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(acc,lb->gAccelerationLabel,matlindex,patch,Ghost::None,0);
      old_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      
      ParticleVariable<Point> newpx;
      ParticleVariable<double> newpvol,newpvolold;
      ParticleVariable<Matrix3> newdefGrad,newbElBar;
      NCVariable<double> newgmass;
      NCVariable<Vector> newext_force,newacc,newdispInc,newvelocity_old;
      
      double newdt = dt;
      new_dw->allocateAndPut(newpx,lb->pXLabel,pset);
      new_dw->allocateAndPut(newpvol,lb->pVolumeLabel,pset);
      new_dw->allocateAndPut(newpvolold,lb->pVolumeOldLabel,pset);

      new_dw->getModifiable(newdefGrad,lb->pDeformationMeasureLabel_preReloc,
			    pset);

      new_dw->getModifiable(newbElBar,lb->bElBarLabel_preReloc,pset);
      new_dw->allocateAndPut(newgmass,lb->gMassLabel,matlindex,patch);
      new_dw->allocateAndPut(newvelocity_old,lb->gVelocityOldLabel,matlindex,
			     patch);
      new_dw->allocateAndPut(newext_force,lb->gExternalForceLabel,matlindex,
			     patch);
      new_dw->allocateAndPut(newacc,lb->gAccelerationLabel,matlindex,patch);
      
      // We need to copy the old data, since the computation of the 
      // deformation gradient and bElBar in computeStressTensorImplicit
      // updates these values and according to Jim's algorithm, the values
      // aren't updated until computeStressTensorImplicitOnly.
      newpx.copyData(px_old);
      newpvol.copyData(pvol_old);
      newpvolold.copyData(pvol_old_old);
      newdefGrad.copyData(deformationGradient);
      newbElBar.copyData(bElBar);
      newgmass.copyData(gmass);
      newext_force.copyData(ext_force);
      newacc.copyData(acc);
      newvelocity_old.copyData(velocity_old);
      new_dw->saveParticleSubset(matlindex,patch,pset);
      new_dw->put(delt_vartype(newdt), d_sharedState->get_delt_label());
    }
  }
  
  
}


void ImpMPM::scheduleFormQI(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls, const bool recursion)
{
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::formQI", this, 
			&ImpMPM::formQPetsc, recursion);
#else
  Task* t = scinew Task("ImpMPM::formQI", this, 
			&ImpMPM::formQ, recursion);
#endif

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
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::formQR", this, 
			&ImpMPM::formQPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::formQR", this, 
			&ImpMPM::formQ,recursion);
#endif
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gVelocityOldLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gAccelerationLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gMassLabel,Ghost::None,0);
  
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
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::removeFixedDOFI", this, 
			&ImpMPM::removeFixedDOFPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::removeFixedDOFI", this, 
			&ImpMPM::removeFixedDOF,recursion);
#endif
  t->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);

  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleRemoveFixedDOFR(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls,
				     const bool recursion)
{
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::removeFixedDOFR", this, 
			&ImpMPM::removeFixedDOFPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::removeFixedDOFR", this, 
			&ImpMPM::removeFixedDOF,recursion);
#endif
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,lb->gMassLabel,Ghost::None,0);

  sched->addTask(t, patches, matls);
  
}


void ImpMPM::scheduleSolveForDuCGI(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls,
				   const bool recursion)
{
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::solveForDuCGI", this, 
			&ImpMPM::solveForDuCGPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::solveForDuCGI", this, 
			&ImpMPM::solveForDuCG,recursion);
#endif
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
#ifdef HAVE_PETSC
  Task* t = scinew Task("ImpMPM::solveForDuCGR", this, 
			&ImpMPM::solveForDuCGPetsc,recursion);
#else
  Task* t = scinew Task("ImpMPM::solveForDuCGR", this, 
			&ImpMPM::solveForDuCG,recursion);
#endif
  t->assumeDataInNewDW();
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
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->computes(lb->dispNewLabel);
  t->computes(lb->gVelocityLabel);
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gVelocityOldLabel,Ghost::None,0);
  
  sched->addTask(t, patches, matls);
  
}



void ImpMPM::scheduleCheckConvergenceI(SchedulerP& sched, const LevelP& level,
				       const PatchSet* patches,
				       const MaterialSet* matls,
				       const bool recursion)
{
  // NOT DONE

  Task* t = scinew Task("ImpMPM::checkConvergenceI", this,
			&ImpMPM::checkConvergence,level,sched,recursion);

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
			&ImpMPM::checkConvergence,level,sched,recursion);

  t->assumeDataInNewDW();
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
  t->requires(Task::NewDW, lb->dispNewLabel,Ghost::None);
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
  t->computes(lb->pExternalForceLabel_preReloc);
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
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);
    new_dw->put(sum_vartype(0.),lb->dispIncQNorm0);
    new_dw->put(sum_vartype(0.),lb->dispIncNormMax);
    
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
    cout << "number of patches = " << patches->size() << endl;
    cout << "p = " << p << endl;

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
      NCVariable<Vector> gexternalforce;

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

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gvelocity_old.initialize(Vector(0,0,0));
      dispNew.initialize(Vector(0,0,0));
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

	cout << "particle accel = " << pacceleration[idx] << "\n";
	
	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	  if(patch->containsNode(ni[k])) {
	    cout << "Shape = " << S[k] << endl;
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
      
      cout << "Interpolate particles to grid . . " << "\n";
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	if (!compare(gmass[*iter],0.)) {
	  gvelocity[*iter] /= gmass[*iter];
	  gacceleration[*iter] /= gmass[*iter];
	  cout << "gmass = " << gmass[*iter] << "\n";
	}
	cout << "velocity = " << gvelocity[*iter] << "\n";
	cout << "acceleration = " << gacceleration[*iter] << "\n";
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
	    //cout << "Velocity bc value = " << bc->getValue() << "\n";
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
			  DataWarehouse* new_dw)

{

#if 0
  KK.clear();
#ifdef HAVE_PETSC
  PetscTruth exists;
  PetscObjectExists((PetscObject)A,&exists);
  // Probably should zero out things instead of destroying it.
  if (exists == PETSC_TRUE)
    MatDestroy(A);
#endif
#endif

  int numProcessors = d_myworld->size();
  vector<int> numNodes(numProcessors, 0);
  vector<int> startIndex(numProcessors);
  int totalNodes = 0;

  for (int p = 0; p < d_perproc_patches->size(); p++) {
    startIndex[p] = totalNodes;
    int mytotal = 0;
    const PatchSubset* patchsub = d_perproc_patches->getSubset(p);
    for (int ps = 0; ps<patchsub->size(); ps++) {
    cout_doing <<"Doing createMatrix " <<"\t\t\t\t\t IMPM"
	       << "\n" << "\n";
    const Patch* patch = patchsub->get(ps);
    IntVector plowIndex = patch->getNodeLowIndex();
    IntVector phighIndex = patch->getNodeHighIndex();

    long nn = (phighIndex[0]-plowIndex[0])*
	(phighIndex[1]-plowIndex[1])*
	(phighIndex[2]-plowIndex[2])*3;
    cout << "nn = " << nn << "\n";
    d_petscGlobalStart[patch]=totalNodes;
    totalNodes+=nn;
    mytotal+=nn;
    
    }
    numNodes[p] = mytotal;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex();
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();
    Level::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      
      IntVector plow = neighbor->getNodeLowIndex();
      IntVector phigh = neighbor->getNodeHighIndex();
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	  || ( high.z() < low.z() ) )
	throw InternalError("Patch doesn't overlap?");
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dnodes.x()*dnodes.y()*3
	+start.y()*dnodes.x()*2 + start.x();
#ifdef PETSC_DEBUG
      cerr << "Looking at patch: " << neighbor->getID() << '\n';
      cerr << "low=" << low << '\n';
      cerr << "high=" << high << '\n';
      cerr << "dnodes=" << dnodes << '\n';
      cerr << "start at: " << d_petscGlobalStart[neighbor] << '\n';
      cerr << "globalIndex = " << petscglobalIndex << '\n';
#endif
      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
	int idx_slab = petscglobalIndex;
	cerr << "idx_slab = " << idx_slab << "\n";
	petscglobalIndex += dnodes.x()*dnodes.y()*3;
	cerr << "petscglobalIndex = " << petscglobalIndex << "\n";
	
	for (int colY = low.y(); colY < high.y(); colY ++) {
	  int idx = idx_slab;
	  idx_slab += dnodes.x()*3;
	  for (int colX = low.x(); colX < high.x(); colX ++) {
	    l2g[IntVector(colX, colY, colZ)] = idx;
	    idx += 3;
	  }
	}
      }
      IntVector d = high-low;
      totalNodes+=d.x()*d.y()*d.z()*3;
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
#ifdef PETSC_DEBUG
    {	
      IntVector l = l2g.getWindow()->getLowIndex();
      IntVector h = l2g.getWindow()->getHighIndex();
      for(int z=l.z();z<h.z();z++){
	for(int y=l.y();y<h.y();y++){
	  for(int x=l.x();x<h.x();x++){
	    IntVector idx(x,y,z);
	    cerr << "l2g" << idx << "=" << l2g[idx] << '\n';
	  }
	}
      }
    }
#endif
  }

  int me = d_myworld->myrank();
  int numlrows = numNodes[me];
  int numlcolumns = numlrows;
  int globalrows = (int)totalNodes;
  int globalcolumns = (int)totalNodes;

#ifdef PETSC_DEBUG
  cerr << "matrixCreate: local size: " << numlrows << ", " << numlcolumns << ", global size: " << globalrows << ", " << globalcolumns << "\n";
#endif
#ifdef HAVE_PETSC
   MatCreateMPIAIJ(PETSC_COMM_WORLD, numlrows, numlcolumns, globalrows,
		   globalcolumns, PETSC_DEFAULT, PETSC_NULL, PETSC_DEFAULT,
		   PETSC_NULL, &A);

   /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */

   VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&petscQ);
   VecDuplicate(petscQ,&petscTemp2);
   VecDuplicate(petscQ,&diagonal);
#endif

   KK.setSize(globalrows,globalcolumns);

}

void ImpMPM::destroyMatrix(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw)
{
  cout_doing <<"Doing destroyMatrix " <<"\t\t\t\t\t IMPM"
	       << "\n" << "\n";
  KK.clear();
#ifdef HAVE_PETSC
  PetscTruth exists;
  PetscObjectExists((PetscObject)A,&exists);
  if (exists == PETSC_TRUE)
    MatDestroy(A);

  PetscObjectExists((PetscObject)petscQ,&exists);
  if (exists == PETSC_TRUE)
    VecDestroy(petscQ);
  
  PetscObjectExists((PetscObject)petscTemp2,&exists);
  if (exists == PETSC_TRUE)
    VecDestroy(petscTemp2);

  PetscObjectExists((PetscObject)diagonal,&exists);
  if (exists == PETSC_TRUE)
    VecDestroy(diagonal);
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

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t IMPM"<< "\n" << "\n";

#if 0

  KK.clear();
#ifdef HAVE_PETSC
  PetscTruth exists;
  PetscObjectExists((PetscObject)A,&exists);
  if (exists == PETSC_TRUE)
    MatDestroy(A);
#endif

#endif
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
#ifdef HAVE_PETSC
    cm->computeStressTensorImplicit(patches, mpm_matl, old_dw, new_dw,KK,A,
				    d_petscLocalToGlobal, recursion);
#else
    cm->computeStressTensorImplicit(patches, mpm_matl, old_dw, new_dw,KK,
				    recursion);
#endif

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

#if 0
  KK.clear();
#ifdef HAVE_PETSC
  PetscTruth exists;
  PetscObjectExists((PetscObject)A,&exists);
  if (exists == PETSC_TRUE)
    MatDestroy(A);
#endif

#endif
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
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector nodes = patch->getNNodes();

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
   
      constNCVariable<double> gmass;
      if (recursion)
	old_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
      else
	new_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
      
      delt_vartype dt;
      old_dw->get(dt, d_sharedState->get_delt_label() );
            
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); 
	   iter++) {
	IntVector n = *iter;
	int dof[3];
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	  (nodes.x())*(n.z());
	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;
#if 1
	cout << "gmass[" << *iter << "]= " << gmass[*iter] << "\n";
	cout << "KK[" << dof[0] << "][" << dof[0] << "]= " 
	     << KK[dof[0]][dof[0]] << "\n";
	cout << "KK[" << dof[1] << "][" << dof[1] << "]= " 
	     << KK[dof[1]][dof[1]] << "\n";
	cout << "KK[" << dof[2] << "][" << dof[2] << "]= " 
	     << KK[dof[2]][dof[2]] << "\n";
#endif
	KK[dof[0]][dof[0]] = KK[dof[0]][dof[0]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[1]][dof[1]] = KK[dof[1]][dof[1]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[2]][dof[2]] = KK[dof[2]][dof[2]] + gmass[*iter]*(4./(dt*dt));
      }
    } 
  }
}

void ImpMPM::formStiffnessMatrixPetsc(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      const bool recursion)

{
  // DONE

  int nn = 0;
  IntVector nodes(0,0,0);
  cout << "nodes = " << nodes << endl;
  cout << "number of patches = " << patches->size() << endl;
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    IntVector num_nodes = patch->getNNodes();
    nn += (num_nodes.x())*(num_nodes.y())*(num_nodes.z())*3;
    nodes = IntVector(Max(num_nodes.x(),nodes.x()),
		      Max(num_nodes.y(),nodes.y()),
		      Max(num_nodes.z(),nodes.z()));
    cout << "nodes = " << nodes << endl;
  }
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrixPetsc " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    // IntVector nodes = patch->getNNodes();
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex();
    Array3<int> l2g(lowIndex,highIndex);
    l2g.copy(d_petscLocalToGlobal[patch]);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
   
      constNCVariable<double> gmass;
      if (recursion)
	old_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
      else
	new_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
      
      delt_vartype dt;
      old_dw->get(dt, d_sharedState->get_delt_label() );

    
            
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); 
	   iter++) {
	IntVector n = *iter;
	int dof[3];
#if 0
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	  (nodes.x())*(n.z());

	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;
#endif
	int l2g_node_num = l2g[n];
	dof[0] = l2g_node_num;
	dof[1] = l2g_node_num+1;
	dof[2] = l2g_node_num+2;

#if 1
	cout << "gmass[" << *iter << "]= " << gmass[*iter] << "\n";
	cout << "KK[" << dof[0] << "][" << dof[0] << "]= " 
	     << KK[dof[0]][dof[0]] << "\n";
	cout << "KK[" << dof[1] << "][" << dof[1] << "]= " 
	     << KK[dof[1]][dof[1]] << "\n";
	cout << "KK[" << dof[2] << "][" << dof[2] << "]= " 
	     << KK[dof[2]][dof[2]] << "\n";
#endif
	KK[dof[0]][dof[0]] = KK[dof[0]][dof[0]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[1]][dof[1]] = KK[dof[1]][dof[1]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[2]][dof[2]] = KK[dof[2]][dof[2]] + gmass[*iter]*(4./(dt*dt));
#ifdef HAVE_PETSC
	PetscScalar v = gmass[*iter]*(4./(dt*dt));
	MatSetValues(A,1,&dof[0],1,&dof[0],&v,ADD_VALUES);
	MatSetValues(A,1,&dof[1],1,&dof[1],&v,ADD_VALUES);
	MatSetValues(A,1,&dof[2],1,&dof[2],&v,ADD_VALUES);
#endif

      }
    } 
  }
#ifdef HAVE_PETSC
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
#endif
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
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;
      
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
						       Ghost::AroundNodes, 1,
						       lb->pXLabel);
      
      old_dw->get(px,      lb->pXLabel, pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel, pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);
      
      if (recursion) 
	new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,matlindex,
			       patch);
      else
	new_dw->getModifiable(internalforce,lb->gInternalForceLabel,matlindex,
			      patch);
      
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


    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());
    double fodts = 4./(dt*dt);
    double fodt = 4./dt;

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z()) * 3;
    valarray<double> temp2(0.,num_nodes);
    Q.resize(num_nodes);

    int matlindex = 0;

    constNCVariable<Vector> externalForce, internalForce;
    constNCVariable<Vector> dispNew,velocity,accel;
    constNCVariable<double> mass;
    if (recursion) {
      new_dw->get(internalForce,lb->gInternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(externalForce,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      old_dw->get(velocity,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(accel,lb->gAccelerationLabel,matlindex,patch,
		Ghost::None,0);
      old_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
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
    }
    
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;

      cout << "external force = " << externalForce[n] << " internal force = " 
	   << internalForce[n] << "\n";
      Q[dof[0]] = externalForce[n].x() + internalForce[n].x();
      Q[dof[1]] = externalForce[n].y() + internalForce[n].y();
      Q[dof[2]] = externalForce[n].z() + internalForce[n].z();


      // temp2 = M*a^(k-1)(t+dt)
      cout << "dispNew = " << dispNew[n] << "\n";
      cout << "velocity = " << velocity[n] << "\n";
      cout << "accel = " << accel[n] << "\n";

      cout << "dispNew.x*fodts = " << dispNew[n].x() * fodts << "\n";
      cout << "velocity.x*fodt = " << velocity[n].x() * fodt << "\n";
      cout << "dispNew - velocity = " << dispNew[n].x() * fodts - 
	velocity[n].x() * fodt << "\n";

      temp2[dof[0]] = (dispNew[n].x()*fodts - velocity[n].x()*fodt -
			accel[n].x())*mass[n];
      temp2[dof[1]] = (dispNew[n].y()*fodts - velocity[n].y()*fodt -
			accel[n].y())*mass[n];
      temp2[dof[2]] = (dispNew[n].z()*fodts - velocity[n].z()*fodt -
			accel[n].z())*mass[n];

      cout << "temp2 = " << temp2[dof[0]] << " " << temp2[dof[1]] << " " <<
	temp2[dof[2]] << "\n";

    }
    if (dynamic)
      Q = Q - temp2;
  }

}

void ImpMPM::formQPetsc(const ProcessorGroup*, const PatchSubset* patches,
			const MaterialSubset*, DataWarehouse* old_dw,
			DataWarehouse* new_dw, const bool recursion)
{
  // DONE
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formQPetsc on patch " << patch->getID()
	       <<"\t\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex();
    Array3<int> l2g(lowIndex,highIndex);
    l2g.copy(d_petscLocalToGlobal[patch]);

    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());
    double fodts = 4./(dt*dt);
    double fodt = 4./dt;

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z()) * 3;
    valarray<double> temp2(0.,num_nodes);
    Q.resize(num_nodes);
#ifdef HAVE_PETSC
#if 0
    PetscTruth exists;
    PetscObjectExists((PetscObject)petscQ,&exists);
    if (exists == PETSC_TRUE)
      VecDestroy(petscQ);

    PetscObjectExists((PetscObject)petscTemp2,&exists);
    if (exists == PETSC_TRUE)
      VecDestroy(petscTemp2);

    VecCreateMPI(PETSC_COMM_WORLD,num_nodes,PETSC_DETERMINE,&petscQ);
    VecCreateMPI(PETSC_COMM_WORLD,num_nodes,PETSC_DETERMINE,&petscTemp2);
#endif
    PetscScalar v = 0.;
    VecSet(&v,petscQ);
    VecSet(&v,petscTemp2);
    VecAssemblyBegin(petscQ);
    VecAssemblyEnd(petscQ);
    VecAssemblyBegin(petscTemp2);
    VecAssemblyEnd(petscTemp2);
#endif
    
    int matlindex = 0;

    constNCVariable<Vector> externalForce, internalForce;
    constNCVariable<Vector> dispNew,velocity,accel;
    constNCVariable<double> mass;
    if (recursion) {
      new_dw->get(internalForce,lb->gInternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(externalForce,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      old_dw->get(velocity,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(accel,lb->gAccelerationLabel,matlindex,patch,
		Ghost::None,0);
      old_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
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
    }
    
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
#if 0
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
#endif
      int l2g_node_num = l2g[n];
      dof[0] = l2g_node_num;
      dof[1] = l2g_node_num+1;
      dof[2] = l2g_node_num+2;


      cout << "external force = " << externalForce[n] << " internal force = " 
	   << internalForce[n] << "\n";
      Q[dof[0]] = externalForce[n].x() + internalForce[n].x();
      Q[dof[1]] = externalForce[n].y() + internalForce[n].y();
      Q[dof[2]] = externalForce[n].z() + internalForce[n].z();

#ifdef HAVE_PETSC
      PetscScalar v[3];
      v[0] = externalForce[n].x() + internalForce[n].x();
      v[1] = externalForce[n].y() + internalForce[n].y();
      v[2] = externalForce[n].z() + internalForce[n].z();

      VecSetValues(petscQ,3,dof,v,INSERT_VALUES);
#endif

      // temp2 = M*a^(k-1)(t+dt)
      cout << "dispNew = " << dispNew[n] << "\n";
      cout << "velocity = " << velocity[n] << "\n";
      cout << "accel = " << accel[n] << "\n";

      cout << "dispNew.x*fodts = " << dispNew[n].x() * fodts << "\n";
      cout << "velocity.x*fodt = " << velocity[n].x() * fodt << "\n";
      cout << "dispNew - velocity = " << dispNew[n].x() * fodts - 
	velocity[n].x() * fodt << "\n";

      temp2[dof[0]] = (dispNew[n].x()*fodts - velocity[n].x()*fodt -
			accel[n].x())*mass[n];
      temp2[dof[1]] = (dispNew[n].y()*fodts - velocity[n].y()*fodt -
			accel[n].y())*mass[n];
      temp2[dof[2]] = (dispNew[n].z()*fodts - velocity[n].z()*fodt -
			accel[n].z())*mass[n];

#ifdef HAVE_PETSC
      v[0] = (dispNew[n].x()*fodts - velocity[n].x()*fodt -
			accel[n].x())*mass[n];
      v[1] = (dispNew[n].y()*fodts - velocity[n].y()*fodt -
			accel[n].y())*mass[n];
      v[2] = (dispNew[n].z()*fodts - velocity[n].z()*fodt -
			accel[n].z())*mass[n];
      VecSetValues(petscTemp2,3,dof,v,INSERT_VALUES);
#endif

      cout << "temp2 = " << temp2[dof[0]] << " " << temp2[dof[1]] << " " <<
	temp2[dof[2]] << "\n";

    }
#ifdef HAVE_PETSC
    VecAssemblyBegin(petscQ);
    VecAssemblyEnd(petscQ);
    VecAssemblyBegin(petscTemp2);
    VecAssemblyEnd(petscTemp2);
#endif
    if (dynamic) {
      Q = Q - temp2;
#ifdef HAVE_PETSC
      PetscScalar minusone = -1.;
      VecAXPY(&minusone,petscTemp2,petscQ);
      //VecDestroy(petscTemp2);
#endif
    }
#ifdef HAVE_PETSC
    VecView(petscQ,PETSC_VIEWER_STDOUT_WORLD);
#endif
  }

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

    cout_doing <<"Doing (NOT DONE) applyRigidbodyCondition on patch " << patch->getID()
	       <<"\t\t IMPM"<< "\n" << "\n";

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
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing removeFixedDOF on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    // Just look on the grid to see if the gmass is 0 and then remove that

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;
    
    int matlindex = 0;

    constNCVariable<double> mass;
    if (recursion)
      old_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    else
      new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    set<int> fixedDOF;

    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
      
      if (compare(mass[n],0.)) {
	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }

#if 0
    cout << "Patch cell_lo = " << patch->getCellLowIndex() << " cell_hi = " 
	 << patch->getCellHighIndex() << "\n";

    cout << "Patch node_lo = " << patch->getNodeLowIndex() << " node_hi = " 
	 << patch->getNodeHighIndex() << "\n";

    for (CellIterator it = patch->getCellIterator(); !it.done(); it++) {
      cout << "cell iterator = " << *it << "\n";
    }

    for (NodeIterator it = patch->getNodeIterator(); !it.done(); it++) {
      cout << "node = " << *it << "\n";
    }


    
    // IntVector l(0,0,0),h(1,2,2);  // xminus
    // IntVector l(1,0,0),h(2,2,2);  // xplus
    // IntVector l(0,0,0),h(2,1,2);  // yminus
    // IntVector l(0,1,0),h(2,2,2);  // yplus
    // IntVector l(0,0,0),h(2,2,1);  // zminus
     IntVector l(0,0,1),h(2,2,2);  // zplus
    for (NodeIterator it(l,h); !it.done(); it++) {
      cout << "node new = " << *it << "\n";
    }
#endif
    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFaceNodes(face,0,l,h);
#if 0
      cout << "face = " << face << " l = " << l << " h = " << h << "\n";
#endif
      for(NodeIterator it(l,h); !it.done(); it++) {
	IntVector n = *it;
	int dof[3];
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	  (nodes.x())*(n.z());
#if 0
	cout << "node = " << n << " node_num = " << node_num << "\n";
#endif
	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;
#if 0
	cout << "dofs = " << dof[0] << "\t" << dof[1] << "\t" << dof[2] 
	     << "\n";
#endif
	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }


    SparseMatrix<double,int> KKK(KK.Rows(),KK.Columns());
    for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
	 itr != KK.end(); itr++) {
      int i = KK.Index1(itr);
      int j = KK.Index2(itr);
      set<int>::iterator find_itr_j = fixedDOF.find(j);
      set<int>::iterator find_itr_i = fixedDOF.find(i);

      if (find_itr_j != fixedDOF.end() && i == j)
	KKK[i][j] = 1.;

      else if (find_itr_i != fixedDOF.end() && i == j)
	KKK[i][j] = 1.;

      else
	KKK[i][j] = KK[i][j];
    }
    // Zero out the Q elements that have entries in the fixedDOF container

    for (set<int>::iterator iter = fixedDOF.begin(); iter != fixedDOF.end(); 
	 iter++) {
      Q[*iter] = 0.;
    }

    // Make sure the nodes that are outside of the material have values 
    // assigned and solved for.  The solutions will be 0.

    for (int j = 0; j < num_nodes; j++) {
      if (compare(KK[j][j],0.)) {
	KKK[j][j] = 1.;
	Q[j] = 0.;
      }
    }
    KK.clear();
    KK = KKK;
    KKK.clear();

  }

}

void ImpMPM::removeFixedDOFPetsc(const ProcessorGroup*, 
				 const PatchSubset* patches,
				 const MaterialSubset*, 
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // NOT DONE
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing removeFixedDOFPetsc on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex();
    Array3<int> l2g(lowIndex,highIndex);
    l2g.copy(d_petscLocalToGlobal[patch]);
    // Just look on the grid to see if the gmass is 0 and then remove that

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;
    
    int matlindex = 0;

    constNCVariable<double> mass;
    if (recursion)
      old_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    else
      new_dw->get(mass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
    set<int> fixedDOF;

    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      
      int dof[3];
#if 0
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
#endif

      int l2g_node_num = l2g[n];
      dof[0] = l2g_node_num;
      dof[1] = l2g_node_num+1;
      dof[2] = l2g_node_num+2;
      
      
      if (compare(mass[n],0.)) {
	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }


    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFaceNodes(face,0,l,h);
      for(NodeIterator it(l,h); !it.done(); it++) {
	IntVector n = *it;
	int dof[3];
#if 0
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	  (nodes.x())*(n.z());

	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;

#endif
	
	int l2g_node_num = l2g[n];
	dof[0] = l2g_node_num;
	dof[1] = l2g_node_num+1;
	dof[2] = l2g_node_num+2;


	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }

#ifdef HAVE_PETSC
    IS is;
    int* indices;
    int in = 0;
    PetscMalloc(fixedDOF.size() * sizeof(int), &indices);
    for (set<int>::iterator iter = fixedDOF.begin(); iter != fixedDOF.end(); 
	 iter++) {
      indices[in++] = *iter;
    }    
    ISCreateGeneral(PETSC_COMM_SELF,fixedDOF.size(),indices,&is);
    PetscFree(indices);

    PetscScalar one = 1.0;
    MatZeroRows(A,is,&one);
    MatTranspose(A,PETSC_NULL);
    MatZeroRows(A,is,&one);
    MatTranspose(A,PETSC_NULL);
#endif
    SparseMatrix<double,int> KKK(KK.Rows(),KK.Columns());
    for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
	 itr != KK.end(); itr++) {
      int i = KK.Index1(itr);
      int j = KK.Index2(itr);
      set<int>::iterator find_itr_j = fixedDOF.find(j);
      set<int>::iterator find_itr_i = fixedDOF.find(i);

      if (find_itr_j != fixedDOF.end() && i == j)
	KKK[i][j] = 1.;

      else if (find_itr_i != fixedDOF.end() && i == j)
	KKK[i][j] = 1.;

      else
	KKK[i][j] = KK[i][j];
    }
    // Zero out the Q elements that have entries in the fixedDOF container

    for (set<int>::iterator iter = fixedDOF.begin(); iter != fixedDOF.end(); 
	 iter++) {
      Q[*iter] = 0.;
#ifdef HAVE_PETSC
      PetscScalar v = 0.;
      const int index = *iter;
      VecSetValues(petscQ,1,&index,&v,INSERT_VALUES);
#endif
    }
#ifdef HAVE_PETSC
    VecAssemblyBegin(petscQ);
    VecAssemblyEnd(petscQ);
#endif
    // Make sure the nodes that are outside of the material have values 
    // assigned and solved for.  The solutions will be 0.

#ifdef HAVE_PETSC
#if 0
    Vec diagonal;
    VecCreateMPI(PETSC_COMM_WORLD,num_nodes,PETSC_DETERMINE,&diagonal);
#endif
    MatGetDiagonal(A,diagonal);
    PetscScalar* diag;
    VecGetArray(diagonal,&diag);
    for (int j = 0; j < num_nodes; j++) {
      if (compare(diag[j],0.)) {
	VecSetValues(diagonal,1,&j,&one,INSERT_VALUES);
	PetscScalar v = 0.;
	VecSetValues(petscQ,1,&j,&v,INSERT_VALUES);
      }
    }
    VecRestoreArray(diagonal,&diag);
#endif
    for (int j = 0; j < num_nodes; j++) {
      if (compare(KK[j][j],0.)) {
	KKK[j][j] = 1.;
	Q[j] = 0.;
	
      }
    }
#ifdef HAVE_PETSC
    VecAssemblyBegin(petscQ);
    VecAssemblyEnd(petscQ);
    VecAssemblyBegin(diagonal);
    VecAssemblyEnd(diagonal);
    MatDiagonalSet(A,diagonal,INSERT_VALUES);
    //VecDestroy(diagonal);
#endif
    KK.clear();
    KK = KKK;
    KKK.clear();

  }

}



void ImpMPM::solveForDuCG(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse*,
			  DataWarehouse* new_dw,
			  const bool /*recursion*/)

{
  // DONE
  int conflag = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveForDuCG on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;

    valarray<double> x(0.,num_nodes);
    int matlindex = 0;

#if 1    
    for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
	 itr != KK.end(); itr++) {
      int i = KK.Index1(itr);
      int j = KK.Index2(itr);
      cout << "KK[" << i << "][" << j <<"] = " << KK[i][j] << "\n";
    }
#endif
#if 1    
    for (unsigned int i = 0; i < Q.size(); i++) {
      cout << "Q[" << i << "]= " << Q[i] << "\n";
    }
#endif    
    x = cgSolve(KK,Q,conflag);
#if 1    
    for (unsigned int i = 0; i < x.size(); i++) {
      cout << "x[" << i << "]= " << x[i] << "\n";
    }
#endif
    NCVariable<Vector> dispInc;
#if 0
    if (recursion)
      new_dw->getModifiable(dispInc,lb->dispIncLabel,matlindex,patch);
    else {
#endif
      new_dw->allocateAndPut(dispInc,lb->dispIncLabel,matlindex,patch);
      dispInc.initialize(Vector(0.,0.,0.));
#if 0
    }
#endif
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
      dispInc[n] = Vector(x[dof[0]],x[dof[1]],x[dof[2]]);
    }    
  }

}

void ImpMPM::solveForDuCGPetsc(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* ,
			       DataWarehouse*,
			       DataWarehouse* new_dw,
			       const bool /*recursion*/)

{
  // DONE
  int conflag = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveForDuCGPetsc on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;

    valarray<double> x(0.,num_nodes);
    int matlindex = 0;

#if 1    
    for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
	 itr != KK.end(); itr++) {
      int i = KK.Index1(itr);
      int j = KK.Index2(itr);
      cout << "KK[" << i << "][" << j <<"] = " << KK[i][j] << "\n";
    }
#endif
#ifdef HAVE_PETSC
    PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INDEX );
    MatView(A,PETSC_VIEWER_STDOUT_SELF);
#endif
#if 1    
    for (unsigned int i = 0; i < Q.size(); i++) {
      cout << "Q[" << i << "]= " << Q[i] << "\n";
    }
#endif    
#ifdef HAVE_PETSC
    
    VecView(petscQ,PETSC_VIEWER_STDOUT_WORLD);
#endif
#ifdef HAVE_PETSC
    PC          pc;           
    KSP         ksp;
    SLESCreate(PETSC_COMM_WORLD,&sles);
    SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);
    SLESGetKSP(sles,&ksp);
    SLESGetPC(sles,&pc);
    KSPSetType(ksp,KSPCG);
    PCSetType(pc,PCJACOBI);
    KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

    int its;
    VecCreateMPI(PETSC_COMM_WORLD,num_nodes,PETSC_DETERMINE,&d_x);
    SLESSolve(sles,petscQ,d_x,&its);
    SLESView(sles,PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"Iterations %d\n",its);
    PetscScalar* xPetsc;
    VecGetArray(d_x,&xPetsc);
    for (int i = 0; i < num_nodes; i++) {
      PetscPrintf(PETSC_COMM_WORLD,"d_x[%d] = %g\n",i,xPetsc[i]);
    }

    SLESDestroy(sles);
#endif

    x = cgSolve(KK,Q,conflag);
#if 1    
    for (unsigned int i = 0; i < x.size(); i++) {
      cout << "x[" << i << "]= " << x[i] << "\n";
    }
#endif
    NCVariable<Vector> dispInc;
#if 0
    if (recursion)
      new_dw->getModifiable(dispInc,lb->dispIncLabel,matlindex,patch);
    else {
#endif
      new_dw->allocateAndPut(dispInc,lb->dispIncLabel,matlindex,patch);
      dispInc.initialize(Vector(0.,0.,0.));
#if 0
    }
#endif
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
      dof[0] = 3*node_num;
      dof[1] = 3*node_num+1;
      dof[2] = 3*node_num+2;
#ifdef HAVE_PETSC
      dispInc[n] = Vector(xPetsc[dof[0]],xPetsc[dof[1]],xPetsc[dof[2]]);
#else
      dispInc[n] = Vector(x[dof[0]],x[dof[1]],x[dof[2]]);
#endif
    }
#ifdef HAVE_PETSC
    VecRestoreArray(d_x,&xPetsc);
    VecDestroy(d_x);
#endif
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
    old_dw->get(dt, d_sharedState->get_delt_label());

    if (recursion) {
      old_dw->get(dispNew_old, lb->dispNewLabel,matlindex,patch,Ghost::None,0);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
      new_dw->allocateAndPut(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->allocateAndPut(velocity, lb->gVelocityLabel, matlindex,patch);
      old_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
    }
    else {
      new_dw->getModifiable(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->getModifiable(velocity, lb->gVelocityLabel, matlindex,patch);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
      new_dw->get(velocity_old,lb->gVelocityOldLabel,matlindex,patch,
		  Ghost::None,0);
    } 

    
    if (recursion) {
      if (dynamic) {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] = dispNew_old[*iter] + dispInc[*iter];
	  cout << "velocity_old = " << velocity_old[*iter] << "\n";
	  velocity[*iter] = dispNew[*iter]*(2./dt) - velocity_old[*iter];
	  cout << "dispNew = " << dispNew[*iter] << "\n";
	  cout << "velocity_new = " << velocity[*iter] << "\n";
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
	  cout << "velocity_old_old = " << velocity_old[*iter] << "\n";
	  cout << "velocity_old = " << velocity[*iter] << "\n";
	  dispNew[*iter] += dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt) - velocity[*iter];
	  cout << "dispNew = " << dispNew[*iter] << "\n";
	  cout << "velocity_new = " << velocity[*iter] << "\n";
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
			      LevelP, SchedulerP,
			      const bool recursion)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector nodes = patch->getNNodes();

    cout_doing <<"Doing checkConvergence on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

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
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;

	dispIncNorm += Dot(dispInc[n],dispInc[n]);
	dispIncQNorm += dispInc[n].x()*Q[dof[0]] + dispInc[n].y()*Q[dof[1]] +
	  dispInc[n].z()*Q[dof[2]];
      }
      // We are computing both dispIncQNorm0 and dispIncNormMax (max residuals)
      // We are computing both dispIncQNorm and dispIncNorm (current residuals)

      double dispIncQNorm0,dispIncNormMax;
      sum_vartype dispIncQNorm0_var,dispIncNormMax_var;
      if (recursion) {
	old_dw->get(dispIncQNorm0_var,lb->dispIncQNorm0);
	old_dw->get(dispIncNormMax_var,lb->dispIncNormMax);
      } else {
	old_dw->get(dispIncQNorm0_var,lb->dispIncQNorm0);
	old_dw->get(dispIncNormMax_var,lb->dispIncNormMax);
      }
      cout << "dispIncQNorm0_var = " << dispIncQNorm0_var << "\n";
      cout << "dispIncNormMax_var = " << dispIncNormMax_var << "\n";
      cout << "dispIncNorm = " << dispIncNorm << "\n";
      cout << "dispIncNormQ = " << dispIncQNorm << "\n";
      dispIncQNorm0 = dispIncQNorm0_var;
      dispIncNormMax = dispIncNormMax_var;

      if (!recursion || dispIncQNorm0 == 0.)
	dispIncQNorm0 = dispIncQNorm;

      if (dispIncNorm > dispIncNormMax)
	dispIncNormMax = dispIncNorm;

      cout << "dispIncQNorm0 = " << dispIncQNorm0 << "\n";
      cout << "dispIncQNorm = " << dispIncQNorm << "\n";
      cout << "dispIncNormMax = " << dispIncNormMax << "\n";
      cout << "dispIncNorm = " << dispIncNorm << "\n";

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
	cout << "before acceleration = " << acceleration[*iter] << "\n";
	cout << "dispNew*fodts = " << dispNew[*iter]*fodts << "\n";
	cout << "velocity*fodt = " << velocity[*iter]*fodt << "\n";
	acceleration[*iter] = dispNew[*iter]*fodts - velocity[*iter]*fodt
	  - acceleration[*iter];
	cout << "after acceleration = " << acceleration[*iter] << "\n";
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
			     lb->pExternalForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::None,0);

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

      // Print out the grid velocity and acceleration
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	cout << "acceleration = " << gacceleration[*iter] << "\n";
	cout << "velocity = " << gvelocity[*iter] << "\n";
      }

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
	  cout << "position = " << pxnew[idx] << "\n";
	  cout << "acceleration = " << paccNew[idx] << "\n";
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

//  cout << "Solid mass lost this timestep = " << massLost << "\n";
//  cout << "Solid momentum after advection = " << CMV << "\n";

//  cout << "THERMAL ENERGY " << thermal_energy << "\n";
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
