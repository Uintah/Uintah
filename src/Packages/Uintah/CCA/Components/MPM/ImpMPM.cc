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

//#define Y_ONLY
#undef Y_ONLY

static DebugStream cout_doing("IMPM", false);

ImpMPM::ImpMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=0.;
  d_rigid_body = false;
  d_numIterations=0;
}

ImpMPM::~ImpMPM()
{
  delete lb;

  if(d_perproc_patches && d_perproc_patches->removeReference()) { 
    delete d_perproc_patches;
    cout << "Freeing patches!!\n";
  }

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    delete d_solver[m];
  }

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
     if (integrator_type == "implicit"){
       d_integrator = Implicit;
       d_conv_crit_disp   = 1.e-10;
       d_conv_crit_energy = 4.e-10;
       mpm_ps->get("convergence_criteria_disp",  d_conv_crit_disp);
       mpm_ps->get("convergence_criteria_energy",d_conv_crit_energy);
     }
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

   ProblemSpecP child = mpm_mat_ps->findBlock("contact");
   std::string con_type = "null";
   child->get("type",con_type);
   d_single_velocity = false;
   d_rigid_body = false;

   if (con_type == "rigid"){
      d_rigid_body = true;
   }
   else if (con_type == "single_velocity"){
      d_single_velocity = true;
      cout << "single" << endl;
   }

   int numMatls=0;
   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, 8,integrator_type,
                                           false, false);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
     numMatls++;
   }
   string solver;
   if (!mpm_ps->get("solver",solver))
     solver = "simple";

   d_solver = vector<Solver*>(numMatls);
   for(int m=0;m<numMatls;m++){
     if (solver == "petsc")
       d_solver[m] = scinew MPMPetscSolver();
     else if (solver == "simple")
       d_solver[m] = scinew SimpleSolver();
     d_solver[m]->initialize();
   }

  // Pull out from Time section
  d_initialDt = 10000.0;
  ProblemSpecP time_ps = prob_spec->findBlock("Time");
  time_ps->get("delt_init",d_initialDt);
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
  t->computes(lb->pCellNAPIDLabel);
  t->computes(d_sharedState->get_delt_label());

  LoadBalancer* loadbal = sched->getLoadBalancer();
  d_perproc_patches = loadbal->createPerProcessorPatchSet(level,d_myworld);
  d_perproc_patches->addReference();

  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());

  t = scinew Task("ImpMPM::printParticleCount",
		  this, &ImpMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, d_perproc_patches, d_sharedState->allMPMMaterials());

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

void ImpMPM::scheduleComputeStableTimestep(const LevelP& lev, SchedulerP& sched)
{

  Task* t;
  cout_doing << "ImpMPM::scheduleComputeStableTimestep " << endl;
  t = scinew Task("ImpMPM::actuallyComputeStableTimestep",
                     this, &ImpMPM::actuallyComputeStableTimestep);

  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->computes(            d_sharedState->get_delt_label());

  sched->addTask(t,lev->eachPatch(), matls);
}

void
ImpMPM::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, int, int )
{
  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  if (!d_perproc_patches) {
    LoadBalancer* loadbal = sched->getLoadBalancer();
    d_perproc_patches = loadbal->createPerProcessorPatchSet(level,d_myworld);
    d_perproc_patches->addReference();
  }

  scheduleInterpolateParticlesToGrid(     sched, d_perproc_patches,matls);
  scheduleDestroyMatrix(                  sched, d_perproc_patches,matls,false);
  scheduleCreateMatrix(                   sched, d_perproc_patches,matls);
  scheduleApplyBoundaryConditions(        sched, d_perproc_patches,matls);
  scheduleComputeContact(                 sched, d_perproc_patches,matls);
  scheduleFindFixedDOF(                   sched, d_perproc_patches,matls);

  scheduleIterate(                   sched,level,d_perproc_patches,matls);

  scheduleComputeStressTensor(            sched, d_perproc_patches,matls);
  scheduleComputeAcceleration(            sched, d_perproc_patches,matls);
  scheduleInterpolateToParticlesAndUpdate(sched, d_perproc_patches,matls);
  scheduleInterpolateStressToGrid(        sched, d_perproc_patches,matls);

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
  t->requires(Task::OldDW, lb->pMassLabel,         Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,       Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeOldLabel,    Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pAccelerationLabel, Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVelocityLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pXLabel,            Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pExternalForceLabel,Ghost::AroundNodes,1);

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityOldLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->dispNewLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalMassLabel);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleRigidBody(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::rigidBody",
			this,&ImpMPM::rigidBody);

  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityOldLabel,  Ghost::None);
  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->modifies(lb->dispNewLabel);

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

void ImpMPM::scheduleCreateMatrix(SchedulerP& sched,
				  const PatchSet* patches,
				  const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::createMatrix",this,&ImpMPM::createMatrix);

  t->requires(Task::OldDW, lb->pXLabel,Ghost::AroundNodes,1);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleApplyBoundaryConditions(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::applyBoundaryCondition",
		        this, &ImpMPM::applyBoundaryConditions);

  t->modifies(lb->gVelocityOldLabel);
  t->modifies(lb->gAccelerationLabel);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeContact(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::computeContact",
                         this, &ImpMPM::computeContact);

  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());

  t->modifies(lb->gVelocityOldLabel);
  t->modifies(lb->gAccelerationLabel);
  if(d_rigid_body){
    t->modifies(lb->dispNewLabel);
  }

  t->computes(lb->gContactLabel);  

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleFindFixedDOF(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::findFixedDOF", this, 
			&ImpMPM::findFixedDOF);

  t->requires(Task::NewDW, lb->gMassLabel,    Ghost::None, 0);
  t->requires(Task::NewDW, lb->gContactLabel, Ghost::None, 0);

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleComputeStressTensor(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls,
					 const bool recursion)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ImpMPM::computeStressTensor",
		    this, &ImpMPM::computeStressTensor,recursion);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches,recursion);
  }
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleFormStiffnessMatrix(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::formStiffnessMatrix",
		    this, &ImpMPM::formStiffnessMatrix);

  t->requires(Task::ParentNewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::ParentOldDW,d_sharedState->get_delt_label());

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::computeInternalForce",
                         this, &ImpMPM::computeInternalForce);

  t->requires(Task::ParentOldDW,lb->pXLabel,              Ghost::AroundNodes,1);
  t->requires(Task::NewDW,      lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,      lb->pVolumeDeformedLabel, Ghost::AroundNodes,1);

  t->computes(lb->gInternalForceLabel);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleFormQ(SchedulerP& sched,const PatchSet* patches,
			   const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::formQ", this, 
			&ImpMPM::formQ);

  Ghost::GhostType  gnone = Ghost::None;

  t->requires(Task::ParentOldDW,d_sharedState->get_delt_label());
  t->requires(Task::NewDW,      lb->gInternalForceLabel,gnone,0);
  t->requires(Task::ParentNewDW,lb->gExternalForceLabel,gnone,0);
  t->requires(Task::OldDW,      lb->dispNewLabel,       gnone,0);
  t->requires(Task::ParentNewDW,lb->gVelocityOldLabel,  gnone,0);
  t->requires(Task::ParentNewDW,lb->gAccelerationLabel, gnone,0);
  t->requires(Task::ParentNewDW,lb->gMassLabel,         gnone,0);
  
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleSolveForDuCG(SchedulerP& sched,
				  const PatchSet* patches,
				  const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::solveForDuCG", this, 
			&ImpMPM::solveForDuCG);

  sched->addTask(t, patches, matls);
}


void ImpMPM::scheduleGetDisplacementIncrement(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::getDisplacementIncrement", this, 
			&ImpMPM::getDisplacementIncrement);

  t->computes(lb->dispIncLabel);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleUpdateGridKinematics(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::updateGridKinematics", this, 
			&ImpMPM::updateGridKinematics);

  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->computes(lb->dispNewLabel);
  t->computes(lb->gVelocityLabel);
  t->requires(Task::ParentOldDW, d_sharedState->get_delt_label() );
  t->requires(Task::NewDW,      lb->dispIncLabel,         Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gVelocityOldLabel,    Ghost::None,0);
  t->requires(Task::ParentNewDW,lb->gContactLabel,        Ghost::None,0);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleCheckConvergence(SchedulerP& sched, 
				      const LevelP& /* level */,
				      const PatchSet* patches,
				      const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::checkConvergence", this,
			&ImpMPM::checkConvergence);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispIncQNorm0);
  t->requires(Task::OldDW,lb->dispIncNormMax);

  t->computes(lb->dispIncNormMax);
  t->computes(lb->dispIncQNorm0);
  t->computes(lb->dispIncNorm);
  t->computes(lb->dispIncQNorm);

  sched->addTask(t,patches,matls);
}

void ImpMPM::scheduleIterate(SchedulerP& sched,const LevelP& level,
			     const PatchSet*, const MaterialSet*)
{

  Task* task = scinew Task("scheduleIterate", this, &ImpMPM::iterate,level,
			   sched.get_rep());

  task->hasSubScheduler();

  task->requires(Task::OldDW,lb->pXLabel,                 Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,            Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeOldLabel,         Ghost::None,0);
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel,Ghost::None,0);

  task->modifies(lb->dispNewLabel);
  task->modifies(lb->gVelocityLabel);

  task->requires(Task::NewDW,lb->gVelocityOldLabel,    Ghost::None,0);
  task->requires(Task::NewDW,lb->gMassLabel,           Ghost::None,0);
  task->requires(Task::NewDW,lb->gExternalForceLabel,  Ghost::None,0);
  task->requires(Task::NewDW,lb->gAccelerationLabel,   Ghost::None,0);
  task->requires(Task::NewDW,lb->gContactLabel,        Ghost::None,0);

  task->requires(Task::OldDW,d_sharedState->get_delt_label());

  sched->addTask(task,d_perproc_patches,d_sharedState->allMaterials());
}

void ImpMPM::scheduleComputeStressTensor(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ImpMPM::computeStressTensor",
		    this, &ImpMPM::computeStressTensor);

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleComputeAcceleration(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::computeAcceleration",
			    this, &ImpMPM::computeAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->modifies(lb->gAccelerationLabel);
  t->requires(Task::NewDW, lb->gVelocityOldLabel,Ghost::None);
  t->requires(Task::NewDW, lb->dispNewLabel,     Ghost::None);

  sched->addTask(t, patches, matls);
}

void ImpMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
  Task* t=scinew Task("ImpMPM::interpolateToParticlesAndUpdate",
		    this, &ImpMPM::interpolateToParticlesAndUpdate);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,     Ghost::AroundCells,1);
  t->requires(Task::NewDW, lb->dispNewLabel,           Ghost::AroundCells,1);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);
  t->requires(Task::OldDW, lb->pVolumeOldLabel,        Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);

  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pAccelerationLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pExtForceLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVolumeOldLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  sched->addTask(t, patches, matls);
}
void ImpMPM::scheduleInterpolateStressToGrid(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t=scinew Task("ImpMPM::interpolateStressToGrid",
		    this, &ImpMPM::interpolateStressToGrid);

  // This task is done for visualization only

  t->requires(Task::OldDW,lb->pXLabel,              Ghost::AroundNodes,1);
  t->requires(Task::OldDW,lb->pMassLabel,           Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::AroundNodes,1);
  t->requires(Task::NewDW,lb->gMassLabel,           Ghost::None);

  t->computes(lb->gStressForSavingLabel);
  sched->addTask(t, patches, matls);
}

void ImpMPM::iterate(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset*,
		     DataWarehouse* old_dw, DataWarehouse* new_dw,
		     LevelP level, Scheduler* sched)
{
  SchedulerP subsched = sched->createSubScheduler();
  DataWarehouse::ScrubMode old_dw_scrubmode =
                           old_dw->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode new_dw_scrubmode =
                           new_dw->setScrubbing(DataWarehouse::ScrubNone);
  subsched->initialize(3, 1, old_dw, new_dw);
  subsched->clearMappings();
  subsched->mapDataWarehouse(Task::ParentOldDW, 0);
  subsched->mapDataWarehouse(Task::ParentNewDW, 1);
  subsched->mapDataWarehouse(Task::OldDW, 2);
  subsched->mapDataWarehouse(Task::NewDW, 3);
  
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  // Create the tasks

  // This task only zeros out the stiffness matrix it doesn't free any memory.
  scheduleDestroyMatrix(           subsched,level->eachPatch(),matls,true);

  scheduleComputeStressTensor(     subsched,level->eachPatch(),matls,true);
  scheduleFormStiffnessMatrix(     subsched,level->eachPatch(),matls);
  scheduleComputeInternalForce(    subsched,level->eachPatch(),matls);
  scheduleFormQ(                   subsched,level->eachPatch(),matls);
  scheduleSolveForDuCG(            subsched,d_perproc_patches, matls);
  scheduleGetDisplacementIncrement(subsched,level->eachPatch(),matls);
  scheduleUpdateGridKinematics(    subsched,level->eachPatch(),matls);
  scheduleCheckConvergence(subsched,level,  level->eachPatch(),matls);

  subsched->compile(d_myworld);

  int count = 0;
  bool dispInc = false;
  bool dispIncQ = false;
  sum_vartype dispIncQNorm,dispIncNorm,dispIncQNorm0,dispIncNormMax;

  // Get all of the required particle data that is in the old_dw and put it 
  // in the subscheduler's  new_dw.  Then once dw is advanced, subscheduler
  // will be pulling data out of the old_dw.

  for (int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing iterate on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();
      ParticleSubset* pset = 
	subsched->get_dw(0)->getParticleSubset(matl, patch);

      NCVariable<Vector> dispNew,velocity;
      new_dw->getModifiable(dispNew, lb->dispNewLabel,      matl,patch);

      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label());
      subsched->get_dw(3)->put(sum_vartype(0.0),lb->dispIncQNorm0);
      subsched->get_dw(3)->put(sum_vartype(0.0),lb->dispIncNormMax);

      // New data to be stored in the subscheduler
      NCVariable<Vector> newdisp;
      double new_dt;

      subsched->get_dw(3)->allocateAndPut(newdisp,lb->dispNewLabel,matl,patch);
      subsched->get_dw(3)->saveParticleSubset(matl, patch, pset);

      newdisp.copyData(dispNew);

      new_dt = dt;
      // These variables are ultimately retrieved from the subschedulers
      // old datawarehouse after the advancement of the data warehouse.
      subsched->get_dw(3)->put(delt_vartype(new_dt),
				  d_sharedState->get_delt_label());
    }
  }

  subsched->get_dw(3)->finalize();
  subsched->advanceDataWarehouse(grid);

//  double change_dt = .02;
//  old_dw->override(delt_vartype(change_dt),d_sharedState->get_delt_label());

  while(!(dispInc && dispIncQ)) {
    if(d_myworld->myrank() == 0){
     cerr << "Beginning Iteration = " << count++ << "\n";
    }
    subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute(d_myworld);  // THIS ACTUALLY GETS THE WORK DONE
    subsched->get_dw(3)->get(dispIncNorm,   lb->dispIncNorm);
    subsched->get_dw(3)->get(dispIncQNorm,  lb->dispIncQNorm); 
    subsched->get_dw(3)->get(dispIncNormMax,lb->dispIncNormMax);
    subsched->get_dw(3)->get(dispIncQNorm0, lb->dispIncQNorm0);

    if(d_myworld->myrank() == 0){
      cerr << "dispIncNorm/dispIncNormMax = "
           << dispIncNorm/(dispIncNormMax + 1.e-100) << "\n";
      cerr << "dispIncQNorm/dispIncQNorm0 = "
           << dispIncQNorm/(dispIncQNorm0 + 1.e-100) << "\n";
    }
    if ((dispIncNorm/(dispIncNormMax + 1e-100) <= d_conv_crit_disp) &&
                                                          dispIncNormMax != 0.0)
      dispInc = true;
    if (dispIncQNorm/(dispIncQNorm0 + 1e-100) <= d_conv_crit_energy)
      dispIncQ = true;
    subsched->advanceDataWarehouse(grid);
  }
  d_numIterations = count;

  // Move the particle data from subscheduler to scheduler.
  for (int p = 0; p < patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Getting the recursive data on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();

      // Needed in computeAcceleration 
      constNCVariable<Vector> velocity, dispNew;
      subsched->get_dw(2)->get(velocity, lb->gVelocityLabel,matl,patch,gnone,0);
      subsched->get_dw(2)->get(dispNew,  lb->dispNewLabel,  matl,patch,gnone,0);

      NCVariable<Vector> velocity_new, dispNew_new;
      new_dw->getModifiable(velocity_new,lb->gVelocityLabel,      matl,patch);
      new_dw->getModifiable(dispNew_new, lb->dispNewLabel,        matl,patch);
      velocity_new.copyData(velocity);
      dispNew_new.copyData(dispNew);
    }
  }
  old_dw->setScrubbing(old_dw_scrubmode);
  new_dw->setScrubbing(new_dw_scrubmode);
}

void ImpMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
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
      int matl = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume,pvolumeold;
      constParticleVariable<Vector> pvelocity, pacceleration,pexternalforce;

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch,
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
      NCVariable<double> gmass,gvolume;
      NCVariable<Vector> gvelocity_old,gacceleration,dispNew,gvelocity;
      NCVariable<Vector> gexternalforce,ginternalforce;
      NCVariable<Vector> dispInc;

      new_dw->allocateAndPut(gmass,         lb->gMassLabel,         matl,patch);
      new_dw->allocateAndPut(gvolume,       lb->gVolumeLabel,       matl,patch);
      new_dw->allocateAndPut(gvelocity_old, lb->gVelocityOldLabel,  matl,patch);
      new_dw->allocateAndPut(gvelocity,     lb->gVelocityLabel,     matl,patch);
      new_dw->allocateAndPut(dispNew,       lb->dispNewLabel,       matl,patch);
      new_dw->allocateAndPut(gacceleration, lb->gAccelerationLabel, matl,patch);
      new_dw->allocateAndPut(gexternalforce,lb->gExternalForceLabel,matl,patch);
      new_dw->allocateAndPut(ginternalforce,lb->gInternalForceLabel,matl,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity_old.initialize(Vector(0,0,0));
      gvelocity.initialize(Vector(0,0,0));
      dispNew.initialize(Vector(0,0,0));
      gacceleration.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      ginternalforce.initialize(Vector(0,0,0));

      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);
      Vector pmom, pmassacc;

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	IntVector ni[8];
	double S[8];
	
	patch->findCellAndWeights(px[idx], ni, S);
	  
        pmassacc    = pacceleration[idx]*pmass[idx];
        pmom        = pvelocity[idx]*pmass[idx];
	total_mom  += pvelocity[idx]*pmass[idx];
	totalmass  += pmass[idx];

	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	  if(patch->containsNode(ni[k])) {
	    gmass[ni[k]]          += pmass[idx]          * S[k];
	    gvolume[ni[k]]        += pvolumeold[idx]     * S[k];
	    gexternalforce[ni[k]] += pexternalforce[idx] * S[k];
	    gvelocity_old[ni[k]]  += pmom                * S[k];
	    gacceleration[ni[k]]  += pmassacc            * S[k];
	  }
	}
      }
      
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
	IntVector c = *iter;
        gvelocity_old[c] /= (gmass[*iter] + 1.e-200);
        gacceleration[c] /= (gmass[*iter] + 1.e-200);
        gmassglobal[c]   += gmass[c];
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);
    }  // End loop over materials
  }  // End loop over patches
}

void ImpMPM::rigidBody(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* ,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing rigidBody on patch " << patch->getID()
               <<"\t\t IMPM"<< "\n" << "\n";

    // The purpose of this task is to set the known new displacements
    // based on the rigid body velocity so that the stresses are
    // computed correctly in the first pass through computeStressTensor
    // This is not necessary, but not doing it wastes an iteration each
    // timestep.

    // Get rigid body data
    constNCVariable<Vector> vel_rigid;
    constNCVariable<double> mass_rigid;
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      if(mpm_matl->getIsRigid()){
        int RM = mpm_matl->getDWIndex();
        new_dw->get(vel_rigid, lb->gVelocityOldLabel,RM,patch,Ghost::None,0);
        new_dw->get(mass_rigid,lb->gMassLabel,       RM,patch,Ghost::None,0);
      }
    }


    // Get and modify non-rigid data
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();
      if(!mpm_matl->getIsRigid()){
        NCVariable<Vector> dispNew;                     
        new_dw->getModifiable(dispNew,lb->dispNewLabel,matl, patch);

        delt_vartype dt;
        old_dw->get(dt, d_sharedState->get_delt_label() );

        for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          IntVector n = *iter;
          if(!compare(mass_rigid[n],0.0)){
#ifdef Y_ONLY
            dispNew[n] = Vector(0.0,vel_rigid[n].y()*dt,0.0);
#else
            dispNew[n] = vel_rigid[n]*dt;  // ALL Components
#endif
          }
        }
      }
    }
  }
}

void ImpMPM::destroyMatrix(const ProcessorGroup*,
			   const PatchSubset* /*patches*/,
			   const MaterialSubset* ,
			   DataWarehouse* /* old_dw */,
			   DataWarehouse* /* new_dw */,
			   const bool recursion)
{
  cout_doing <<"Doing destroyMatrix " <<"\t\t\t\t\t IMPM" << "\n" << "\n";

  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
    d_solver[m]->destroyMatrix(recursion);
  }
}

void ImpMPM::createMatrix(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  for (int m = 0; m < numMatls; m++) {
    map<int,int> dof_diag;
    for(int pp=0;pp<patches->size();pp++){
      const Patch* patch = patches->get(pp);
      cout_doing <<"Doing createMatrix on patch " << patch->getID() 
		 << "\t\t\t\t IMPM"    << "\n" << "\n";
      d_solver[m]->createLocalToGlobalMapping(d_myworld,d_perproc_patches,
					      patches);
      
      IntVector lowIndex = patch->getNodeLowIndex();
      IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);

      Array3<int> l2g(lowIndex,highIndex);
      d_solver[m]->copyL2G(l2g,patch);

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();    
      constParticleVariable<Point> px;
      ParticleSubset* pset;

      pset = old_dw->getParticleSubset(dwi,patch, Ghost::AroundNodes,1,
                                                          lb->pXLabel);
      old_dw->get(px,lb->pXLabel,pset);
      
      CCVariable<int> visited;
      new_dw->allocateTemporary(visited,patch,Ghost::AroundCells,1);
      visited.initialize(0);
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	IntVector cell,ni[8];
	patch->findCell(px[idx],cell);
	if (visited[cell] == 0 ) {
	  visited[cell] = 1;
	  patch->findNodesFromCell(cell,ni);
	  vector<int> dof(0);
	  int l2g_node_num;
	  for (int k = 0; k < 8; k++) {
	    if (patch->containsNode(ni[k]) ) {
	      l2g_node_num = l2g[ni[k]] - l2g[lowIndex];
	      dof.push_back(l2g_node_num);
	      dof.push_back(l2g_node_num+1);
	      dof.push_back(l2g_node_num+2);
	    }
	  }
	  for (int I = 0; I < (int) dof.size(); I++) {
	    int dofi = dof[I];
	    for (int J = 0; J < (int) dof.size(); J++) {
	      dof_diag[dofi] += 1;
	    }
	  }
	  
	}
      }
      
    }
    d_solver[m]->createMatrix(d_myworld,dof_diag);
  }
  
}

void ImpMPM::applyBoundaryConditions(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* /*old_dw*/,
				     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing applyBoundaryConditions " <<"\t\t\t\t IMPM"
	       << "\n" << "\n";
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    // Apply grid boundary conditions to the velocity before storing the data
    IntVector offset =  IntVector(0,0,0);
    for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
      d_solver[m]->copyL2G(l2g,patch);
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();
      
      NCVariable<Vector> gacceleration,gvelocity_old;

      new_dw->getModifiable(gvelocity_old,lb->gVelocityOldLabel, matl, patch);
      new_dw->getModifiable(gacceleration,lb->gAccelerationLabel,matl, patch);

      IntVector low,high;
      patch->getLevel()->findNodeIndexRange(low,high);
      high -= IntVector(1,1,1);
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	const BoundCondBase *vel_bcs, *sym_bcs;
	if (patch->getBCType(face) == Patch::None) {
	  vel_bcs  = patch->getBCValues(matl,"Velocity",face);
	  sym_bcs  = patch->getBCValues(matl,"Symmetric",face);
	} else
	  continue;
	
	if (vel_bcs != 0) {
	  const VelocityBoundCond* bc =
	    dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	  if (bc->getKind() == "Dirichlet") {
	    fillFace(gvelocity_old,patch, face,bc->getValue(),offset);
	    fillFace(gacceleration,patch, face,bc->getValue(),offset);
	    IntVector l,h;
	    patch->getFaceNodes(face,0,l,h);
            for(NodeIterator it(l,h); !it.done(); it++) {
               IntVector n = *it;
               int dof[3];
               int l2g_node_num = l2g[n];
               dof[0] = l2g_node_num;
               dof[1] = l2g_node_num+1;
               dof[2] = l2g_node_num+2;
               d_solver[m]->d_DOF.insert(dof[0]);
               d_solver[m]->d_DOF.insert(dof[1]);
               d_solver[m]->d_DOF.insert(dof[2]);
            }
	  }
	}

	if (sym_bcs != 0) { 
	  fillFaceNormal(gvelocity_old,patch, face,offset);
	  fillFaceNormal(gacceleration,patch, face,offset);

	  IntVector l,h;
	  patch->getFaceNodes(face,0,l,h);
	  for(NodeIterator it(l,h); !it.done(); it++) {
	    IntVector n = *it;

            // The DOF is an IntVector which is initially (0,0,0).
            // Inserting a 1 into any of the components indicates that 
            // the component should be inserted into the DOF array.
            IntVector DOF(0,0,0);
	    
            if (face == Patch::xminus || face == Patch::xplus){
              DOF = IntVector(max(DOF.x(),1),max(DOF.y(),0),max(DOF.z(),0));
            }
            if (face == Patch::yminus || face == Patch::yplus){
              DOF = IntVector(max(DOF.x(),0),max(DOF.y(),1),max(DOF.z(),0));
            }
            if (face == Patch::zminus || face == Patch::zplus){
              DOF = IntVector(max(DOF.x(),0),max(DOF.y(),0),max(DOF.z(),1));
            }

            int dof[3];
            int l2g_node_num = l2g[n];
            dof[0] = l2g_node_num;
            dof[1] = l2g_node_num+1;
            dof[2] = l2g_node_num+2;
            if (DOF.x())
              d_solver[m]->d_DOF.insert(dof[0]);
            if (DOF.y())
              d_solver[m]->d_DOF.insert(dof[1]);
            if (DOF.z())
              d_solver[m]->d_DOF.insert(dof[2]);
	  }
	}
      }
    }
  }

}

void ImpMPM::computeContact(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeContact on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    delt_vartype dt;

    Ghost::GhostType  gnone = Ghost::None;
    int numMatls = d_sharedState->getNumMPMMatls();
    StaticArray<constNCVariable<double> >  gmass(numMatls);
    StaticArray<NCVariable<Vector> >  gvel(numMatls);
    StaticArray<NCVariable<Vector> >  gacc(numMatls);
    StaticArray<NCVariable<int> >  contact(numMatls);
    for(int n = 0; n < numMatls; n++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
      int dwi = mpm_matl->getDWIndex();

      new_dw->get(gmass[n],              lb->gMassLabel,     dwi,patch,gnone,0);
      new_dw->getModifiable(gvel[n],     lb->gVelocityOldLabel,   dwi,patch);
      new_dw->getModifiable(gacc[n],     lb->gAccelerationLabel,  dwi,patch);
      new_dw->allocateAndPut(contact[n], lb->gContactLabel,       dwi,patch);
      contact[n].initialize(0);
    }

    Vector centerOfMassVelocity;
    Vector centerOfMassAcceleration;

    if(d_single_velocity){
     for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      Vector centerOfMassMom(0.,0.,0.);
      Vector centerOfMassAcc(0.,0.,0.);
      double centerOfMassMass=0.;
      for(int n = 0; n < numMatls; n++){
        centerOfMassMom +=gvel[n][c] * gmass[n][c];
        centerOfMassAcc +=gacc[n][c] * gmass[n][c];
        centerOfMassMass+=gmass[n][c];
      }

      for(int n = 0; n < numMatls; n++){
	if(!compare(gmass[n][c],centerOfMassMass)){
          centerOfMassVelocity    =centerOfMassMom/centerOfMassMass;
          centerOfMassAcceleration=centerOfMassAcc/centerOfMassMass;
          gvel[n][c]  = centerOfMassVelocity;
          gacc[n][c]  = centerOfMassAcceleration;
          contact[n][c] = 1;
	}
      }
     }
    }

    if(d_rigid_body){
     constNCVariable<Vector> vel_rigid;
     constNCVariable<double> mass_rigid;
     int numMatls = d_sharedState->getNumMPMMatls();
     for(int n = 0; n < numMatls; n++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
      if(mpm_matl->getIsRigid()){
        int RM = mpm_matl->getDWIndex();
        new_dw->get(vel_rigid, lb->gVelocityOldLabel,RM,patch,Ghost::None,0);
        new_dw->get(mass_rigid,lb->gMassLabel,       RM,patch,Ghost::None,0);
      }
     }

     // Get and modify non-rigid data
     for(int n = 0; n < numMatls; n++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
      int matl = mpm_matl->getDWIndex();
        NCVariable<Vector> dispNew;                     
        new_dw->getModifiable(dispNew,lb->dispNewLabel,matl, patch);

        delt_vartype dt;
        old_dw->get(dt, d_sharedState->get_delt_label() );

        for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          IntVector c = *iter;
          if(!compare(mass_rigid[c],0.0)){
#ifdef Y_ONLY
            dispNew[c]=Vector(0.0,vel_rigid[c].y()*dt,0.0);
#else
            dispNew[c] = vel_rigid[c]*dt;  // ALL Components
#endif
            contact[n][c] = 2;
          }
        }
     }
    }
  }
}

void ImpMPM::findFixedDOF(const ProcessorGroup*, 
                          const PatchSubset* patches,
                          const MaterialSubset*, 
                          DataWarehouse* ,
                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing findFixedDOF on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      d_solver[m]->copyL2G(l2g,patch);
        constNCVariable<double> mass;
        constNCVariable<int> contact;
        new_dw->get(mass,   lb->gMassLabel,   matlindex,patch,Ghost::None,0);
        new_dw->get(contact,lb->gContactLabel,matlindex,patch,Ghost::None,0);

        for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          IntVector n = *iter;
          int dof[3];
          int l2g_node_num = l2g[n];
          dof[0] = l2g_node_num;
          dof[1] = l2g_node_num+1;
          dof[2] = l2g_node_num+2;

          // Just look on the grid to see if the gmass is 0 and then remove that
          if (compare(mass[n],0.)) {
            d_solver[m]->d_DOF.insert(dof[0]);
            d_solver[m]->d_DOF.insert(dof[1]);
            d_solver[m]->d_DOF.insert(dof[2]);
          }
          if (contact[n] > 0) {  // Contact imposed on these nodes
#ifdef Y_ONLY
            d_solver[m]->d_DOF.insert(dof[1]);
#else
            d_solver[m]->d_DOF.insert(dof[0]);
            d_solver[m]->d_DOF.insert(dof[1]);
            d_solver[m]->d_DOF.insert(dof[2]);
#endif
          }
        }
    }
  }
}

void ImpMPM::computeStressTensor(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t IMPM"<< "\n" << "\n";

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw,
			    d_solver[m], recursion);
  }
  
}

void ImpMPM::formStiffnessMatrix(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      d_solver[m]->copyL2G(l2g,patch);
   
      constNCVariable<double> gmass;
      delt_vartype dt;
      DataWarehouse* parent_new_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      parent_new_dw->get(gmass, lb->gMassLabel,matlindex,patch,Ghost::None,0);
      DataWarehouse* parent_old_dw =
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      parent_old_dw->get(dt,d_sharedState->get_delt_label());

      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
	IntVector n = *iter;
	int dof[3];
	int l2g_node_num = l2g[n];
	dof[0] = l2g_node_num;
	dof[1] = l2g_node_num+1;
	dof[2] = l2g_node_num+2;

	double v = gmass[*iter]*(4./(dt*dt));
	d_solver[m]->fillMatrix(dof[0],dof[0],v);
	d_solver[m]->fillMatrix(dof[1],dof[1],v);
	d_solver[m]->fillMatrix(dof[2],dof[2],v);
      }
    } 
  }
  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++)
    d_solver[m]->finalizeMatrix();
}
	    
void ImpMPM::computeInternalForce(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
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
      int dwi = mpm_matl->getDWIndex();

      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Vector>             int_force;
      
      DataWarehouse* parent_old_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);

      ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch,
                                              Ghost::AroundNodes,1,lb->pXLabel);

      parent_old_dw->get(px,   lb->pXLabel,    pset);
      new_dw->allocateAndPut(int_force,lb->gInternalForceLabel,  dwi, patch);

      new_dw->get(pvol,    lb->pVolumeDeformedLabel,  pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

      int_force.initialize(Vector(0,0,0));
      IntVector ni[8];
      Vector d_S[8];

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	
	// Get the node indices that surround the cell
	patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
	
	for (int k = 0; k < 8; k++){
	  if(patch->containsNode(ni[k])){
	   Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
           int_force[ni[k]] -= (div * pstress[idx])  * pvol[idx];    
	  }
	}
      }
    }
  }
}

void ImpMPM::formQ(const ProcessorGroup*, const PatchSubset* patches,
                   const MaterialSubset*, DataWarehouse* old_dw,
                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formQ on patch " << patch->getID()
	       <<"\t\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      d_solver[m]->copyL2G(l2g,patch);

      delt_vartype dt;
      Ghost::GhostType  gnone = Ghost::None;

      constNCVariable<Vector> extForce, intForce;
      constNCVariable<Vector> dispNew,velocity,accel;
      constNCVariable<double> mass;
      DataWarehouse* parent_new_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      DataWarehouse* parent_old_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);

      parent_old_dw->get(dt,d_sharedState->get_delt_label());
      new_dw->get(       intForce, lb->gInternalForceLabel,dwi,patch,gnone,0);
      old_dw->get(       dispNew,  lb->dispNewLabel,       dwi,patch,gnone,0);
      parent_new_dw->get(extForce, lb->gExternalForceLabel,dwi,patch,gnone,0);
      parent_new_dw->get(velocity, lb->gVelocityOldLabel,  dwi,patch,gnone,0);
      parent_new_dw->get(accel,    lb->gAccelerationLabel, dwi,patch,gnone,0);
      parent_new_dw->get(mass,     lb->gMassLabel,         dwi,patch,gnone,0);

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
        v[0] = extForce[n].x() + intForce[n].x();
        v[1] = extForce[n].y() + intForce[n].y();
        v[2] = extForce[n].z() + intForce[n].z();

        // temp2 = M*a^(k-1)(t+dt)
        if (dynamic) {
          v[0] -= (dispNew[n].x()*fodts - velocity[n].x()*fodt -
                   accel[n].x())*mass[n];
          v[1] -= (dispNew[n].y()*fodts - velocity[n].y()*fodt -
                   accel[n].y())*mass[n];
          v[2] -= (dispNew[n].z()*fodts - velocity[n].z()*fodt -
                   accel[n].z())*mass[n];
        }
        d_solver[m]->fillVector(dof[0],double(v[0]));
        d_solver[m]->fillVector(dof[1],double(v[1]));
        d_solver[m]->fillVector(dof[2],double(v[2]));
      }
      d_solver[m]->assembleVector();
    }
  }
}

void ImpMPM::solveForDuCG(const ProcessorGroup* /*pg*/,
			  const PatchSubset* patches,
			  const MaterialSubset* ,
			  DataWarehouse*,
			  DataWarehouse* new_dw)

{
  int num_nodes = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    
    cout_doing <<"Doing solveForDuCG on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z())*3;
  }
    
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    if(!mpm_matl->getIsRigid()){  // i.e. Leave dispInc zero for Rigd Bodies
      // remove fixed degrees of freedom and solve K*du = Q
      d_solver[m]->removeFixedDOF(num_nodes);
      d_solver[m]->solve();   
    }
  }
}

void ImpMPM::getDisplacementIncrement(const ProcessorGroup* /*pg*/,
				      const PatchSubset* patches,
				      const MaterialSubset* ,
				      DataWarehouse*,
				      DataWarehouse* new_dw)

{
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    
    cout_doing <<"Doing getDisplacementIncrement on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< "\n" << "\n";

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      NCVariable<Vector> dispInc;
      new_dw->allocateAndPut(dispInc,lb->dispIncLabel,matlindex,patch);
      dispInc.initialize(Vector(0.));

      if(!mpm_matl->getIsRigid()){  // i.e. Leave dispInc zero for Rigd Bodies
	// remove fixed degrees of freedom and solve K*du = Q

        IntVector lowIndex = patch->getNodeLowIndex();
        IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
        Array3<int> l2g(lowIndex,highIndex);
        d_solver[m]->copyL2G(l2g,patch);
  
        vector<double> x;
        int begin = d_solver[m]->getSolution(x);
  
        for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
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
  }
}

void ImpMPM::updateGridKinematics(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing updateGridKinematics on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    Ghost::GhostType  gnone = Ghost::None;

    int rig_index=-99;
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
       MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
       if(mpm_matl->getIsRigid()){
         rig_index = mpm_matl->getDWIndex();
       }
    }

    constNCVariable<Vector> velocity_rig;
    if(d_rigid_body){
      DataWarehouse* parent_new_dw = 
                              new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      parent_new_dw->get(velocity_rig,
                                 lb->gVelocityOldLabel,rig_index,patch,gnone,0);
    }

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      NCVariable<Vector> dispNew,velocity;
      constNCVariable<Vector> dispInc,dispNew_old,velocity_old;
      constNCVariable<int> contact;

      delt_vartype dt;

      DataWarehouse* parent_new_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      DataWarehouse* parent_old_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      parent_old_dw->get(dt, d_sharedState->get_delt_label());
      old_dw->get(dispNew_old,         lb->dispNewLabel,   dwi,patch,gnone,0);
      new_dw->get(dispInc,             lb->dispIncLabel,   dwi,patch,gnone,0);
      new_dw->allocateAndPut(dispNew,  lb->dispNewLabel,   dwi,patch);
      new_dw->allocateAndPut(velocity, lb->gVelocityLabel, dwi,patch);
      parent_new_dw->get(velocity_old, lb->gVelocityOldLabel,
                                                           dwi,patch,gnone,0);
      parent_new_dw->get(contact,      lb->gContactLabel,  dwi,patch,gnone,0);

      double oneifdyn = 0.;
      if(dynamic){
        oneifdyn = 1.;
      }

      dispNew.copyData(dispNew_old);

      for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        dispNew[n] += dispInc[n];
        velocity[n] = dispNew[n]*(2./dt) - oneifdyn*velocity_old[n];
      }

      if(d_single_velocity){  // overwrite some of the values computed above
        for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
          IntVector n = *iter;
          if(contact[n]==1){
            dispNew[n]  = velocity_old[n]*dt;
            velocity[n] = dispNew[n]*(2./dt) - oneifdyn*velocity_old[n];
          }
        }
      }

      if(d_rigid_body){  // overwrite some of the values computed above
        for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  IntVector n = *iter;
          if(contact[n]==2){
#ifdef Y_ONLY
              dispNew[n] = Vector(dispNew[n].x(),velocity_rig[n].y()*dt,
                                  dispNew[n].z());
#else
            dispNew[n]  = velocity_rig[n]*dt;
#endif
            velocity[n] = dispNew[n]*(2./dt) - oneifdyn*velocity_old[n];
          } // if contact == 2
        } // for
      } // if d_rigid_body
    }
  }
}

void ImpMPM::checkConvergence(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* ,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    cout_doing <<"Doing checkConvergence on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     int matlindex = mpm_matl->getDWIndex();
     d_solver[m]->copyL2G(l2g,patch);

     if(!mpm_matl->getIsRigid()){
      constNCVariable<Vector> dispInc;
      new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
      
      double dispIncNorm  = 0.;
      double dispIncQNorm = 0.;
      vector<double> getQ;
      int begin = d_solver[m]->getRHS(getQ);
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
      old_dw->get(dispIncQNorm0_var, lb->dispIncQNorm0);
      old_dw->get(dispIncNormMax_var,lb->dispIncNormMax);

      dispIncQNorm0 = dispIncQNorm0_var;
      dispIncNormMax = dispIncNormMax_var;

      bool first_iteration=false;
      if (compare(dispIncQNorm0,0.)){
        first_iteration = true;
	dispIncQNorm0 = dispIncQNorm;
      }

      if (dispIncNorm > dispIncNormMax){
	dispIncNormMax = dispIncNorm;
      }

      // The following is being done because the denominator in the
      // convergence criteria is carried forward each iteration.  Since 
      // every patch puts this into the sum_vartype, the value is multiplied
      // by the number of patches.  Predividing by numPatches fixes this.
      int numPatches = patch->getLevel()->numPatches();
      if(!first_iteration){
        dispIncQNorm0/=((double) numPatches);
        dispIncNormMax/=((double) numPatches);
      }

      new_dw->put(sum_vartype(dispIncNorm),   lb->dispIncNorm);
      new_dw->put(sum_vartype(dispIncQNorm),  lb->dispIncQNorm);
      new_dw->put(sum_vartype(dispIncNormMax),lb->dispIncNormMax);
      new_dw->put(sum_vartype(dispIncQNorm0), lb->dispIncQNorm0);
     }

    }  // End of loop over materials
  }  // End of loop over patches

}

void ImpMPM::computeStressTensor(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  cout_doing <<"Doing computeStressTensor" <<"\t\t\t\t IMPM"<< "\n" 
	     << "\n";

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++) {
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
  }
}

void ImpMPM::computeAcceleration(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeAcceleration on patch " << patch->getID()
	       <<"\t\t\t IMPM"<< "\n" << "\n";

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      NCVariable<Vector> acceleration;
      constNCVariable<Vector> velocity,dispNew;
      delt_vartype delT;

      new_dw->getModifiable(acceleration,lb->gAccelerationLabel,dwindex,patch);
      new_dw->get(velocity,lb->gVelocityOldLabel,dwindex, patch, gnone, 0);
      new_dw->get(dispNew, lb->dispNewLabel,     dwindex, patch, gnone, 0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      double fodts = 4./(delT*delT);
      double fodt = 4./(delT);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	IntVector c = *iter;
	acceleration[c] = dispNew[c]*fodts - velocity[c]*fodt - acceleration[c];
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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
	       << patch->getID() <<"\t IMPM"<< "\n" << "\n";

    Ghost::GhostType  gac = Ghost::AroundCells;

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and displacement to the particles to update their
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
      constParticleVariable<double> pmass, pvolume,pvolumeold,pTempOld;
      ParticleVariable<double> pmassNew,pvolumeNew,newpvolumeold,pTemp;
  
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> dispNew, gacceleration;
      constNCVariable<double> dTdt;

      delt_vartype delT;

      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);

      ParticleSubset* delete_particles = scinew ParticleSubset
	(pset->getParticleSet(),false,dwindex,patch, 0);
    
      old_dw->get(px,                    lb->pXLabel,                    pset);
      old_dw->get(pmass,                 lb->pMassLabel,                 pset);
      new_dw->get(pvolume,               lb->pVolumeDeformedLabel,       pset);
      old_dw->get(pvolumeold,            lb->pVolumeOldLabel,            pset);
      old_dw->get(pexternalForce,        lb->pExternalForceLabel,        pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,             pset);
      old_dw->get(pacceleration,         lb->pAccelerationLabel,         pset);
      old_dw->get(pTempOld,              lb->pTemperatureLabel,          pset);
      new_dw->allocateAndPut(pvelocitynew,lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(paccNew,    lb->pAccelerationLabel_preReloc,pset);
      new_dw->allocateAndPut(pxnew,      lb->pXLabel_preReloc,           pset);
      new_dw->allocateAndPut(pmassNew,   lb->pMassLabel_preReloc,        pset);
      new_dw->allocateAndPut(pvolumeNew, lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(newpvolumeold,lb->pVolumeOldLabel_preReloc, pset);
      new_dw->allocateAndPut(pTemp,      lb->pTemperatureLabel_preReloc, pset);
      new_dw->allocateAndPut(pexternalForceNew,
			     lb->pExtForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(dispNew,      lb->dispNewLabel,      dwindex, patch, gac, 1);
      new_dw->get(gacceleration,lb->gAccelerationLabel,dwindex, patch, gac, 1);
     
      NCVariable<double> dTdt_create, massBurnFraction_create;	
      new_dw->allocateTemporary(dTdt_create, patch,Ghost::None,0);
      dTdt_create.initialize(0.);
      dTdt = dTdt_create; // reference created data
            

      old_dw->get(delT, d_sharedState->get_delt_label() );

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
	  disp      += dispNew[ni[k]]       * S[k];
	  acc       += gacceleration[ni[k]] * S[k];
	}
	
        // Update the particle's position and velocity
        pxnew[idx]        = px[idx] + disp;
        pvelocitynew[idx] = pvelocity[idx] 
                          + (pacceleration[idx]+acc)*(.5* delT);

        paccNew[idx]         = acc;
        pmassNew[idx]        = pmass[idx];
        pvolumeNew[idx]      = pvolume[idx];
        newpvolumeold[idx]   = pvolumeold[idx];
        pTemp[idx]           = pTempOld[idx];

        if(pmassNew[idx] <= 0.0){
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
  }
}

void ImpMPM::interpolateStressToGrid(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing interpolateStressToGrid on patch " << patch->getID()
	       <<"\t\t IMPM"<< "\n" << "\n";

    // This task is done for visualization only

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constParticleVariable<Point>   px;
      constParticleVariable<double>  pmass;
      constParticleVariable<Matrix3> pstress;
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                              Ghost::AroundNodes,1,lb->pXLabel);
      old_dw->get(px,   lb->pXLabel,    pset);
      old_dw->get(pmass,lb->pMassLabel, pset);
      new_dw->get(gmass,lb->gMassLabel, dwi, patch,Ghost::None,0);
      new_dw->allocateAndPut(gstress,  lb->gStressForSavingLabel,dwi, patch);

      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

      gstress.initialize(Matrix3(0.));
      IntVector ni[8];
      double S[8];
      Matrix3 stressmass;

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        patch->findCellAndWeights(px[idx], ni, S);

        stressmass  = pstress[idx]*pmass[idx];

        for (int k = 0; k < 8; k++){
          if(patch->containsNode(ni[k])){
           gstress[ni[k]]       += stressmass * S[k];
          }
        }
      }
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstress[c] /= (gmass[c]+1.e-200);
      }
    }
  }
}

void ImpMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
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

void ImpMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					   const PatchSubset*,
					   const MaterialSubset*,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
    if(d_numIterations==0){
      new_dw->put(delt_vartype(d_initialDt), lb->delTLabel);
    }
    else{
      delt_vartype old_delT;
      old_dw->get(old_delT, d_sharedState->get_delt_label());
      new_dw->put(old_delT, lb->delTLabel);
    }
}
