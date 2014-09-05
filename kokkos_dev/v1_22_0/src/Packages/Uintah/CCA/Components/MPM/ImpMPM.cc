
#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#define _CPP_CMATH
#endif

#include <sci_defs.h>
#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h> 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/NormalForceBC.h>
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
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <sgi_stl_warnings_off.h>
#include <set>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream cout_doing("IMPM", false);

ImpMPM::ImpMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=0.;
  d_rigid_body = false;
  d_numIterations=0;
  d_doGridReset = true;
  d_conv_crit_disp   = 1.e-10;
  d_conv_crit_energy = 4.e-10;
  d_forceIncrementFactor = 1.0;
  d_integrator = Implicit;
  d_dynamic = true;
}

bool ImpMPM::restartableTimesteps()
{
  return true;
}

ImpMPM::~ImpMPM()
{
  delete lb;
  delete flags;

  if(d_perproc_patches && d_perproc_patches->removeReference()) { 
    delete d_perproc_patches;
    cout << "Freeing patches!!\n";
  }

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
     mpm_ps->get("time_integrator",flags->d_integrator_type);
     mpm_ps->get("do_grid_reset",  d_doGridReset);
     mpm_ps->get("ForceBC_force_increment_factor",flags->d_forceIncrementFactor);
     mpm_ps->get("use_load_curves", flags->d_useLoadCurves);
     if (flags->d_integrator_type == "implicit"){
       d_integrator = Implicit;
       mpm_ps->get("convergence_criteria_disp",  d_conv_crit_disp);
       mpm_ps->get("convergence_criteria_energy",d_conv_crit_energy);
       mpm_ps->get("dynamic",d_dynamic);
       mpm_ps->getWithDefault("iters_before_timestep_restart",
                               d_max_num_iterations, 25);
       mpm_ps->getWithDefault("num_iters_to_decrease_delT",
                               d_num_iters_to_decrease_delT, 12);
       mpm_ps->getWithDefault("num_iters_to_increase_delT",
                               d_num_iters_to_increase_delT, 4);
       mpm_ps->getWithDefault("delT_decrease_factor",
                               d_delT_decrease_factor, .6);
       mpm_ps->getWithDefault("delT_increase_factor",
                               d_delT_increase_factor, 2);
     }
     else{
      throw ProblemSetupException("Can't use explicit integration with -impm");
     }
   }

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   ProblemSpecP child = mpm_mat_ps->findBlock("contact");
   std::string con_type = "null";
   child->get("type",con_type);
   d_rigid_body = false;

   if (con_type == "rigid"){
      d_rigid_body = true;
      Vector defaultDir(1,1,1);
      child->getWithDefault("direction",d_contact_dirs, defaultDir);
   }

   if(flags->d_useLoadCurves){
    MPMPhysicalBCFactory::create(prob_spec);
    if( (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size()==0) {
     throw ProblemSetupException("No load curve in ups, d_useLoadCurve==true?");
    }
   }

   int numMatls=0;
   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, flags);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
     numMatls++;
   }

#ifdef HAVE_PETSC
   d_solver = scinew MPMPetscSolver();
#else
   d_solver = scinew SimpleSolver();
#endif
   d_solver->initialize();

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
  d_perproc_patches = loadbal->createPerProcessorPatchSet(level);
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
  t->requires(Task::NewDW, lb->pVelocityLabel,   Ghost::None);
  t->computes(            d_sharedState->get_delt_label());

  sched->addTask(t,lev->eachPatch(), matls);
}

void
ImpMPM::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, int, int )
{
  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  if (!d_perproc_patches) {
    LoadBalancer* loadbal = sched->getLoadBalancer();
    d_perproc_patches = loadbal->createPerProcessorPatchSet(level);
    d_perproc_patches->addReference();
  }

  scheduleApplyExternalLoads(             sched, d_perproc_patches,matls);
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

void ImpMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
                                                                                
{
  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &ImpMPM::applyExternalLoads);
                                                                                
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);

  sched->addTask(t, patches, matls);
                                                                                
}

void ImpMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  Task* t = scinew Task("ImpMPM::interpolateParticlesToGrid",
                        this,&ImpMPM::interpolateParticlesToGrid);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeLabel,           Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVolumeOldLabel,        Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::AroundNodes,1);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::AroundNodes,1);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,Ghost::AroundNodes,1);

  t->computes(lb->gMassLabel);

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

  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  if(d_rigid_body){
    t->modifies(lb->dispNewLabel);
    t->requires(Task::NewDW,lb->gMassLabel,        Ghost::None);
    t->requires(Task::NewDW,lb->gVelocityOldLabel, Ghost::None);
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
  t->requires(Task::OldDW, lb->pDispLabel,             Ghost::None);

  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pAccelerationLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pXXLabel);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVolumeOldLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pDispLabel_preReloc);

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
  t->computes(lb->NTractionZMinusLabel);
  t->computes(lb->integralAreaLabel);
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

  subsched->compile();

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
      subsched->get_dw(3)->saveParticleSubset(pset, matl, patch);

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

  while(!(dispInc && dispIncQ)) {
    if(d_myworld->myrank() == 0){
     cerr << "Beginning Iteration = " << count << "\n";
    }
    count++;
    subsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubComplete);
    subsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
    subsched->execute();  // THIS ACTUALLY GETS THE WORK DONE
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
    if (dispIncQNorm/(dispIncQNorm0 + 1e-100) <= d_conv_crit_energy &&
        dispIncQNorm/(dispIncQNorm0 + 1e-100) > 0.0)
      dispIncQ = true;
    // Check to see if the residual is likely a nan, if so, we'll restart.
    bool restart_nan=false;
    bool restart_neg_residual=false;
    bool restart_num_iters=false;
    if (isnan(dispIncQNorm/dispIncQNorm0)||isnan(dispIncNorm/dispIncNormMax)){
      restart_nan=true;
      cerr << "Restarting due to a nan residual" << endl;
    }
    if (dispIncQNorm/dispIncQNorm0 < 0. ||dispIncNorm/dispIncNormMax < 0.){
      restart_neg_residual=true;
      cerr << "Restarting due to a negative residual" << endl;
    }
    if (count > d_max_num_iterations){
      restart_num_iters=true;
      cerr << "Restarting due to exceeding max number of iterations" << endl;
    }
    if (restart_nan || restart_neg_residual || restart_num_iters){
      new_dw->abortTimestep();
      new_dw->restartTimestep();
      return;
    }

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

void ImpMPM::applyExternalLoads(const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();
  cout_doing << "Current Time (applyExternalLoads) = " << time << endl;
                                                                                
  // Calculate the force vector at each particle for each bc
  std::vector<double> forceMagPerPart;
  std::vector<NormalForceBC*> nfbcP;
  if (flags->d_useLoadCurves) {
    // Currently, only one load curve at a time is supported, but
    // I've left the infrastructure in place to go to multiple
    for (int ii = 0;
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {

      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {
        cerr << "Pressure BCs not supported in ImpMPM" << endl;
      }
      if (bcs_type == "NormalForce") {
        NormalForceBC* nfbc =
         dynamic_cast<NormalForceBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        nfbcP.push_back(nfbc);

        // Calculate the force per particle at current time
        forceMagPerPart.push_back(nfbc->getLoad(time));
      }
    }
  }
                                                                                
  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
                                                                                
    cout_doing <<"Doing applyExternalLoads on patch "
               << patch->getID() << "\t MPM"<< endl;
                                                                                
    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.
                                                                                
    int numMPMMatls=d_sharedState->getNumMPMMatls();
                                                                                
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
                                                                                
      if (flags->d_useLoadCurves) {
        // Get the external force data and allocate new space for
        // external force
        constParticleVariable<Vector> pExternalForce;
        ParticleVariable<Vector> pExternalForce_new;
        old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pExternalForce_new,
                               lb->pExtForceLabel_preReloc,  pset);

        double mag = forceMagPerPart[0];
        // Iterate over the particles
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          // For particles with an existing external force, apply the
          // new magnitude to the same direction.
          if(pExternalForce[idx].length() > 1.e-7){
            pExternalForce_new[idx] = mag*
                       (pExternalForce[idx]/pExternalForce[idx].length());
          } else{
            pExternalForce_new[idx] = Vector(0.,0.,0.);
          }
        }
      } else {
                                                                                
        // Get the external force data and allocate new space for
        // external force and copy the data
        constParticleVariable<Vector> pExternalForce;
        ParticleVariable<Vector> pExternalForce_new;
        old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pExternalForce_new,
                               lb->pExtForceLabel_preReloc,  pset);
                                                                                
        // Iterate over the particles
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pExternalForce_new[idx] = pExternalForce[idx]
                                       *flags->d_forceIncrementFactor;
        }
      }
    } // matl loop
  }  // patch loop
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
    // Create arrays for the grid data
    StaticArray<NCVariable<double> > gmass(numMatls),gvolume(numMatls);
    StaticArray<NCVariable<Vector> > gvel_old(numMatls),gacc(numMatls);
    StaticArray<NCVariable<Vector> > dispNew(numMatls),gvelocity(numMatls);
    StaticArray<NCVariable<Vector> > gextforce(numMatls),gintforce(numMatls);
    StaticArray<NCVariable<Vector> > dispInc(numMatls);

    NCVariable<double> GMASS, GVOLUME;
    NCVariable<Vector> GVEL_OLD, GACC, GEXTFORCE;
    new_dw->allocateTemporary(GMASS,     patch,Ghost::None,0);
    new_dw->allocateTemporary(GVOLUME,   patch,Ghost::None,0);
    new_dw->allocateTemporary(GVEL_OLD,  patch,Ghost::None,0);
    new_dw->allocateTemporary(GACC,      patch,Ghost::None,0);
    new_dw->allocateTemporary(GEXTFORCE, patch,Ghost::None,0);
    GMASS.initialize(0.);
    GVOLUME.initialize(0.);
    GVEL_OLD.initialize(Vector(0.,0.,0.));
    GACC.initialize(Vector(0.,0.,0.));
    GEXTFORCE.initialize(Vector(0.,0.,0.));

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

      old_dw->get(px,             lb->pXLabel,                 pset);
      old_dw->get(pmass,          lb->pMassLabel,              pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,            pset);
      old_dw->get(pvolumeold,     lb->pVolumeOldLabel,         pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,          pset);
      old_dw->get(pacceleration,  lb->pAccelerationLabel,      pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);

      new_dw->allocateAndPut(gmass[m],      lb->gMassLabel,         matl,patch);
      new_dw->allocateAndPut(gvolume[m],    lb->gVolumeLabel,       matl,patch);
      new_dw->allocateAndPut(gvel_old[m],   lb->gVelocityOldLabel,  matl,patch);
      new_dw->allocateAndPut(gvelocity[m],  lb->gVelocityLabel,     matl,patch);
      new_dw->allocateAndPut(dispNew[m],    lb->dispNewLabel,       matl,patch);
      new_dw->allocateAndPut(gacc[m],       lb->gAccelerationLabel, matl,patch);
      new_dw->allocateAndPut(gextforce[m],  lb->gExternalForceLabel,matl,patch);
      new_dw->allocateAndPut(gintforce[m],  lb->gInternalForceLabel,matl,patch);

      gmass[m].initialize(d_SMALL_NUM_MPM);
      gvolume[m].initialize(0);
      gvel_old[m].initialize(Vector(0,0,0));
      gacc[m].initialize(Vector(0,0,0));
      gextforce[m].initialize(Vector(0,0,0));

      dispNew[m].initialize(Vector(0,0,0));
      gvelocity[m].initialize(Vector(0,0,0));
      gintforce[m].initialize(Vector(0,0,0));

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
            gmass[m][ni[k]]          += pmass[idx]          * S[k];
            gvolume[m][ni[k]]        += pvolumeold[idx]     * S[k];
            gextforce[m][ni[k]]      += pexternalforce[idx] * S[k];
            gvel_old[m][ni[k]]       += pmom                * S[k];
            gacc[m][ni[k]]           += pmassacc            * S[k];
          }
        }
      }

      if(mpm_matl->getIsRigid()){
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          IntVector c = *iter;
          gvel_old[m][c] /= (gmass[m][c] + 1.e-200);
          gacc[m][c]     /= (gmass[m][c] + 1.e-200);
        }
      }

      if(!mpm_matl->getIsRigid()){
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
          IntVector c = *iter;
          GMASS[c]+=gmass[m][c];
          GVOLUME[c]+=gvolume[m][c];
          GEXTFORCE[c]+=gextforce[m][c];
          GVEL_OLD[c]+=gvel_old[m][c];
          GACC[c]+=gacc[m][c];
        }
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);
    }  // End loop over materials

    // Give all non-rigid materials the same nodal values
    for(int m = 0; m < numMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid()){
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector c = *iter;
        gmass[m][c]=GMASS[c];
        gvolume[m][c]=GVOLUME[c];
        gextforce[m][c]=GEXTFORCE[c];
        gvel_old[m][c]=GVEL_OLD[c]/(GMASS[c] + 1.e-200);
        gacc[m][c]=GACC[c]/(GMASS[c] + 1.e-200);
      }
     }
    }  // End loop over materials
  }  // End loop over patches
}

void ImpMPM::destroyMatrix(const ProcessorGroup*,
                           const PatchSubset* /*patches*/,
                           const MaterialSubset* ,
                           DataWarehouse* /* old_dw */,
                           DataWarehouse* /* new_dw */,
                           const bool recursion)
{
  cout_doing <<"Doing destroyMatrix " <<"\t\t\t\t\t IMPM" << "\n" << "\n";

  d_solver->destroyMatrix(recursion);
}

void ImpMPM::createMatrix(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  map<int,int> dof_diag;
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    cout_doing <<"Doing createMatrix on patch " << patch->getID() 
               << "\t\t\t\t IMPM"    << "\n" << "\n";
    d_solver->createLocalToGlobalMapping(d_myworld,d_perproc_patches,patches);
    
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);

    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);

    CCVariable<int> visited;
    new_dw->allocateTemporary(visited,patch,Ghost::AroundCells,1);
    visited.initialize(0);

    int numMatls = d_sharedState->getNumMPMMatls();
    for (int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();    
      constParticleVariable<Point> px;
      ParticleSubset* pset;

      pset = old_dw->getParticleSubset(dwi,patch, Ghost::AroundNodes,1,
                                                          lb->pXLabel);
      old_dw->get(px,lb->pXLabel,pset);
      
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
  }
  d_solver->createMatrix(d_myworld,dof_diag);
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
    d_solver->copyL2G(l2g,patch);

    // Apply grid boundary conditions to the velocity before storing the data
    IntVector offset =  IntVector(0,0,0);
    for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();
      
      NCVariable<Vector> gacceleration,gvelocity_old;

      new_dw->getModifiable(gvelocity_old,lb->gVelocityOldLabel, matl, patch);
      new_dw->getModifiable(gacceleration,lb->gAccelerationLabel,matl, patch);

      for(Patch::FaceType face = Patch::startFace;
          face <= Patch::endFace; face=Patch::nextFace(face)){
        const BoundCondBase *vel_bcs,*sym_bcs;
        if (patch->getBCType(face) == Patch::None) {
          int numChildren = 
            patch->getBCDataArray(face)->getNumberChildren(matl);
          for (int child = 0; child < numChildren; child++) {
            vector<IntVector> bound,nbound,sfx,sfy,sfz;
            vector<IntVector>::const_iterator boundary;
            vel_bcs = patch->getArrayBCValues(face,matl,"Velocity",bound,
                                              nbound,sfx,sfy,sfz,child);
            sym_bcs  = patch->getArrayBCValues(face,matl,"Symmetric",bound,
                                               nbound,sfx,sfy,sfz,child);
            if (vel_bcs != 0) {
              const VelocityBoundCond* bc =
                dynamic_cast<const VelocityBoundCond*>(vel_bcs);
              if (bc->getKind() == "Dirichlet") {
                for (boundary=nbound.begin(); boundary != nbound.end();
                     boundary++) {
                  gvelocity_old[*boundary] = bc->getValue();
                  gacceleration[*boundary] = bc->getValue();
                  
                }
                IntVector l,h;
                patch->getFaceNodes(face,0,l,h);
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector n = *it;
                  int dof[3];
                  int l2g_node_num = l2g[n];
                  dof[0] = l2g_node_num;
                  dof[1] = l2g_node_num+1;
                  dof[2] = l2g_node_num+2;
                  d_solver->d_DOF.insert(dof[0]);
                  d_solver->d_DOF.insert(dof[1]);
                  d_solver->d_DOF.insert(dof[2]);
                }
              }
              delete vel_bcs;
            }
            if (sym_bcs != 0) {
              if (face == Patch::xplus || face == Patch::xminus)
                for (boundary=nbound.begin(); boundary != nbound.end(); 
                     boundary++) {
                  gvelocity_old[*boundary] = 
                    Vector(0.,gvelocity_old[*boundary].y(),
                           gvelocity_old[*boundary].z());
                  gacceleration[*boundary] = 
                    Vector(0.,gacceleration[*boundary].y(),
                           gacceleration[*boundary].z());
                }
              if (face == Patch::yplus || face == Patch::yminus)
                for (boundary=nbound.begin(); boundary != nbound.end(); 
                     boundary++) {
                  gvelocity_old[*boundary] = 
                    Vector(gvelocity_old[*boundary].x(),0.,
                           gvelocity_old[*boundary].z());
                  gacceleration[*boundary] = 
                    Vector(gacceleration[*boundary].x(),0.,
                           gacceleration[*boundary].z());
                }
              if (face == Patch::zplus || face == Patch::zminus)
                for (boundary=nbound.begin(); boundary != nbound.end(); 
                     boundary++) {
                  gvelocity_old[*boundary] = 
                    Vector(gvelocity_old[*boundary].x(),
                           gvelocity_old[*boundary].y(),0.);
                  gacceleration[*boundary] = 
                    Vector(gacceleration[*boundary].x(),
                           gacceleration[*boundary].y(),0.);
                }
              IntVector l,h;
              patch->getFaceNodes(face,0,l,h);
              for(NodeIterator it(l,h); !it.done(); it++) {
                IntVector n = *it;
                // The DOF is an IntVector which is initially (0,0,0).
                // Inserting a 1 into any of the components indicates that 
                // the component should be inserted into the DOF array.
                IntVector DOF(0,0,0);
                if (face == Patch::xminus || face == Patch::xplus)
                  DOF=IntVector(max(DOF.x(),1),max(DOF.y(),0),max(DOF.z(),0));
                if (face == Patch::yminus || face == Patch::yplus)
                  DOF=IntVector(max(DOF.x(),0),max(DOF.y(),1),max(DOF.z(),0));
                if (face == Patch::zminus || face == Patch::zplus)
                  DOF=IntVector(max(DOF.x(),0),max(DOF.y(),0),max(DOF.z(),1));
                
                int dof[3];
                int l2g_node_num = l2g[n];
                dof[0] = l2g_node_num;
                dof[1] = l2g_node_num+1;
                dof[2] = l2g_node_num+2;
                if (DOF.x())
                  d_solver->d_DOF.insert(dof[0]);
                if (DOF.y())
                  d_solver->d_DOF.insert(dof[1]);
                if (DOF.z())
                  d_solver->d_DOF.insert(dof[2]);
              }
              delete sym_bcs;
            }
          }
        } else
          continue;
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

    int numMatls = d_sharedState->getNumMPMMatls();
    StaticArray<NCVariable<int> >  contact(numMatls);
    for(int n = 0; n < numMatls; n++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( n );
      int dwi = mpm_matl->getDWIndex();
      new_dw->allocateAndPut(contact[n], lb->gContactLabel,       dwi,patch);
      contact[n].initialize(0);
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
            dispNew[c] = Vector(vel_rigid[c].x()*dt*d_contact_dirs.x(),
                                vel_rigid[c].y()*dt*d_contact_dirs.y(),
                                vel_rigid[c].z()*dt*d_contact_dirs.z());
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
    d_solver->copyL2G(l2g,patch);

    bool firstTimeThrough=true;
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid() && firstTimeThrough){ 
      firstTimeThrough=false;
      int matlindex = mpm_matl->getDWIndex();
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
          d_solver->d_DOF.insert(dof[0]);
          d_solver->d_DOF.insert(dof[1]);
          d_solver->d_DOF.insert(dof[2]);
        }
        if (contact[n] == 2) {  // Rigid Contact imposed on these nodes
          for(int i=0;i<3;i++){
            if(d_contact_dirs[i]==1){
             d_solver->d_DOF.insert(dof[i]);  // specifically, these DOFs
            }
          }
        }// contact ==2
      }  // node iterator
     }   // if not rigid
    }    // loop over matls
  }      // patches
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
                            d_solver, recursion);
  }
  
}

void ImpMPM::formStiffnessMatrix(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)
{
  if (!d_dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
               <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    bool firstTimeThrough=true;
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid() && firstTimeThrough){ 
      firstTimeThrough=false;
      int matlindex = mpm_matl->getDWIndex();
      d_solver->copyL2G(l2g,patch);
   
      constNCVariable<double> gmass;
      delt_vartype dt;
      DataWarehouse* parent_new_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      parent_new_dw->get(gmass, lb->gMassLabel,matlindex,patch,Ghost::None,0);
      DataWarehouse* parent_old_dw =
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);
      parent_old_dw->get(dt,d_sharedState->get_delt_label());

#ifdef HAVE_PETSC
      PetscScalar v[1];
#else
      double v[1];
#endif

      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        int dof[1];
        int l2g_node_num = l2g[n];
        v[0] = gmass[*iter]*(4./(dt*dt));
        for(int ii=0;ii<3;ii++){
          dof[0] = l2g_node_num+ii;
          d_solver->fillMatrix(1,dof,1,dof,v);
        }
      }  // node iterator
     }   // if
    }    // matls
  }
  d_solver->finalizeMatrix();
}
            
void ImpMPM::computeInternalForce(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* ,
                                  DataWarehouse* /*old_dw*/,
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

    StaticArray<NCVariable<Vector> > int_force(numMPMMatls);
    NCVariable<Vector> INT_FORCE;
    new_dw->allocateTemporary(INT_FORCE,     patch,Ghost::None,0);
    INT_FORCE.initialize(Vector(0,0,0));

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol;
      constParticleVariable<Matrix3> pstress;

      new_dw->allocateAndPut(int_force[m],lb->gInternalForceLabel,  dwi, patch);
      int_force[m].initialize(Vector(0,0,0));

      DataWarehouse* parent_old_dw = 
        new_dw->getOtherDataWarehouse(Task::ParentOldDW);

      ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch,
                                              Ghost::AroundNodes,1,lb->pXLabel);

      parent_old_dw->get(px,   lb->pXLabel,    pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel,  pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

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
           int_force[m][ni[k]] -= (div * pstress[idx])  * pvol[idx];    
          }
        }
      }
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        INT_FORCE[n]+=int_force[m][n];
      }
    }  // matls

    for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid()){
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        int_force[m][n]=INT_FORCE[n];
      }
     }
    }  // matls
  }    // patches
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
    d_solver->copyL2G(l2g,patch);

    bool firstTimeThrough=true;
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid() && firstTimeThrough){ 
      firstTimeThrough=false;
      int dwi = mpm_matl->getDWIndex();

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
        if (d_dynamic) {
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
      d_solver->assembleVector();
     } // first time through non-rigid
    }  // matls
  }    // patches
}

void ImpMPM::solveForDuCG(const ProcessorGroup* /*pg*/,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse*,
                          DataWarehouse* /*new_dw*/)

{
  int num_nodes = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    
    cout_doing <<"Doing solveForDuCG on patch " << patch->getID()
               <<"\t\t\t\t IMPM"<< "\n" << "\n";

    IntVector nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z())*3;
  }

  d_solver->removeFixedDOF(num_nodes);
  d_solver->solve();   
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

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    d_solver->copyL2G(l2g,patch);
 
    vector<double> x;
    int begin = d_solver->getSolution(x);
  
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      NCVariable<Vector> dispInc;
      new_dw->allocateAndPut(dispInc,lb->dispIncLabel,matlindex,patch);
      dispInc.initialize(Vector(0.));
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
      if(d_dynamic){
        oneifdyn = 1.;
      }

      dispNew.copyData(dispNew_old);

      if(!mpm_matl->getIsRigid()){
       for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        dispNew[n] += dispInc[n];
        velocity[n] = dispNew[n]*(2./dt) - oneifdyn*velocity_old[n];
       }
      }

      if(d_rigid_body){  // overwrite some of the values computed above
        for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
          IntVector n = *iter;
          if(contact[n]==2){
            dispNew[n] = Vector((1.-d_contact_dirs.x())*dispNew[n].x() +
                                d_contact_dirs.x()*velocity_rig[n].x()*dt,
                                (1.-d_contact_dirs.y())*dispNew[n].y() +
                                d_contact_dirs.y()*velocity_rig[n].y()*dt,
                                (1.-d_contact_dirs.z())*dispNew[n].z() +
                                d_contact_dirs.z()*velocity_rig[n].z()*dt);

            velocity[n] = dispNew[n]*(2./dt) - oneifdyn*velocity_old[n];
          } // if contact == 2
        } // for
      } // if d_rigid_body
    }   // matls
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

   cout_doing <<"Doing checkConvergence on patch " << patch->getID()
              <<"\t\t\t IMPM"<< "\n" << "\n";

   IntVector lowIndex = patch->getNodeLowIndex();
   IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
   Array3<int> l2g(lowIndex,highIndex);
   d_solver->copyL2G(l2g,patch);

   int matlindex = 0;

    constNCVariable<Vector> dispInc;
    new_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
    
    double dispIncNorm  = 0.;
    double dispIncQNorm = 0.;
    vector<double> getQ;
    int begin = d_solver->getRHS(getQ);
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
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
  if (!d_dynamic)
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

    double move_particles=1.;
    if(!d_doGridReset){
      move_particles=0.;
    }

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity, pacceleration,pexternalForce;
      constParticleVariable<Vector> pDispOld;
      ParticleVariable<Vector> pvelocitynew, pexternalForceNew, paccNew,pDisp;
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
      old_dw->get(pDispOld,              lb->pDispLabel,                 pset);
      new_dw->allocateAndPut(pvelocitynew,lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(paccNew,    lb->pAccelerationLabel_preReloc,pset);
      new_dw->allocateAndPut(pxnew,      lb->pXLabel_preReloc,           pset);
      new_dw->allocateAndPut(pxx,        lb->pXXLabel,                   pset);
      new_dw->allocateAndPut(pmassNew,   lb->pMassLabel_preReloc,        pset);
      new_dw->allocateAndPut(pvolumeNew, lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(newpvolumeold,lb->pVolumeOldLabel_preReloc, pset);
      new_dw->allocateAndPut(pTemp,      lb->pTemperatureLabel_preReloc, pset);
      new_dw->allocateAndPut(pDisp,      lb->pDispLabel_preReloc,        pset);

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
        pxnew[idx]        = px[idx] + disp*move_particles;
        pDisp[idx]        = pDispOld[idx] + disp;
        pvelocitynew[idx] = pvelocity[idx] 
                          + (pacceleration[idx]+acc)*(.5* delT);

        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pDisp[idx];

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
    double integralTraction = 0.;
    double integralArea = 0.;

    NCVariable<Matrix3>       GSTRESS;
    new_dw->allocateTemporary(GSTRESS, patch, Ghost::None,0);
    GSTRESS.initialize(Matrix3(0.));
    StaticArray<NCVariable<Matrix3> >         gstress(numMatls);
    StaticArray<constNCVariable<double> >     gmass(numMatls);

    Vector dx = patch->dCell();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constParticleVariable<Point>   px;
      constParticleVariable<double>  pmass;
      constParticleVariable<Matrix3> pstress;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                              Ghost::AroundNodes,1,lb->pXLabel);
      old_dw->get(px,      lb->pXLabel,    pset);
      old_dw->get(pmass,   lb->pMassLabel, pset);
      new_dw->get(gmass[m],lb->gMassLabel, dwi, patch,Ghost::None,0);
      new_dw->allocateAndPut(gstress[m],lb->gStressForSavingLabel,dwi, patch);

      new_dw->get(pstress, lb->pStressLabel_preReloc, pset);

      gstress[m].initialize(Matrix3(0.));
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
           gstress[m][ni[k]]       += stressmass * S[k];
          }
        }
      }
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        GSTRESS[c] += (gstress[m][c]);
      }
      IntVector low = patch-> getInteriorNodeLowIndex();
      IntVector hi  = patch-> getInteriorNodeHighIndex();
      for(Patch::FaceType face = Patch::startFace;
        face <= Patch::endFace; face=Patch::nextFace(face)){

        // I assume we have the patch variable
        // Check if the face is on the boundary
        Patch::BCType bc_type = patch->getBCType(face);
        if (bc_type == Patch::None) {
          // We are on the boundary, i.e. not on an interior patch
          // boundary, so do the traction accumulation . . .
          if(face==Patch::yminus){
            int J=low.y();
            for (int i = low.x(); i<hi.x(); i++) {
              for (int k = low.z(); k<hi.z(); k++) {
                integralTraction +=
                  gstress[m][IntVector(i,J,k)](2,2)*dx.x()*dx.z();
                if(fabs(gstress[m][IntVector(i,J,k)](2,2)) > 1.e-12){
                  integralArea+=dx.x()*dx.z();
                }
              }
            }
          }  // if the yminus face
        } // end of if (bc_type == Patch::None)
      } // Loop over faces
    }  // Loop over matls
    new_dw->put(sum_vartype(integralTraction), lb->NTractionZMinusLabel);
    new_dw->put(sum_vartype(integralArea),     lb->integralAreaLabel);

    for(int m = 0; m < numMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     if(!mpm_matl->getIsRigid()){
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstress[m][c] = GSTRESS[c]/(gmass[m][c]+1.e-200);
      }
     }
    }  // Loop over matls
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
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
   const Patch* patch = patches->get(p);
                                                                                
   cout_doing <<"Doing actuallyComputeStableTimestep on patch "
              << patch->getID() <<"\t IMPM"<< "\n" << "\n";

   if(d_numIterations==0){
    new_dw->put(delt_vartype(d_initialDt), lb->delTLabel);
   }
   else{
    Vector dx = patch->dCell();
    delt_vartype old_delT;
    old_dw->get(old_delT, d_sharedState->get_delt_label());

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      ParticleSubset* pset = new_dw->getParticleSubset(dwindex, patch);

      constParticleVariable<Vector> pvelocity;
      new_dw->get(pvelocity, lb->pVelocityLabel, pset);

      Vector ParticleSpeed(1.e-12,1.e-12,1.e-12);

      for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
        particleIndex idx = *iter;
        ParticleSpeed=Vector(Max(fabs(pvelocity[idx].x()),ParticleSpeed.x()),
                             Max(fabs(pvelocity[idx].y()),ParticleSpeed.y()),
                             Max(fabs(pvelocity[idx].z()),ParticleSpeed.z()));
      }
      ParticleSpeed = dx/ParticleSpeed;
      double delT_new = .8*ParticleSpeed.minComponent();
      double old_dt=old_delT;
      if(d_numIterations <= d_num_iters_to_increase_delT){
        old_dt = d_delT_increase_factor*old_delT;
      }
      if(d_numIterations >= d_num_iters_to_decrease_delT){
        old_dt = d_delT_decrease_factor*old_delT;
      }
      delT_new = min(delT_new, old_dt);
      new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    }
   }
  }
}

double ImpMPM::recomputeTimestep(double current_dt)
{
  return current_dt*d_delT_decrease_factor;
}
