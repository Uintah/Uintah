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

static DebugStream cout_doing("IMPM_DOING_COUT", true);
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

   cout << "integrator type = " << integrator_type << " " << d_integrator << endl;
   
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
  t->computes(lb->pAccelerationLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel);
  t->computes(lb->bElBarLabel);
  t->computes(lb->dispIncQNorm0);
  t->computes(lb->dispIncNormMax);

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

  scheduleApplyBoundaryConditions(sched,patches,matls);

#if 1
  scheduleComputeStressTensorI(sched, patches, matls,false);

  scheduleFormStiffnessMatrixI(sched,patches,matls,false);

  scheduleComputeInternalForceI(sched, patches, matls,true);

  scheduleFormQI(sched, patches, matls,false);

#if 0
  scheduleApplyRigidBodyConditionI(sched, patches,matls);
#endif

  scheduleRemoveFixedDOFI(sched, patches, matls,false);

  scheduleSolveForDuCGI(sched, patches, matls,false);

  scheduleUpdateGridKinematicsI(sched, patches, matls,false);

  scheduleCheckConvergenceI(sched,level, patches, matls, false);
#endif

  scheduleIterate(sched,level,patches,matls);

#if 0
  scheduleMoveData(sched,level,patches,matls);
#endif

  scheduleComputeStressTensorOnly(sched,patches,matls,false);

  scheduleComputeInternalForceII(sched,patches,matls,false);

  scheduleComputeAcceleration(sched,patches,matls);

  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
#if 0
  scheduleInterpolateStressToGrid(sched,patches,matls);
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
  t->requires(Task::OldDW, lb->pMassLabel,           Ghost::None,0);
  t->requires(Task::OldDW, lb->pVolumeLabel,       Ghost::None,0);
  
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::None,0);
  t->requires(Task::OldDW, lb->pVelocityLabel,     Ghost::None,0);
  t->requires(Task::OldDW, lb->pXLabel,            Ghost::None,0);
  t->requires(Task::OldDW, lb->pExternalForceLabel,Ghost::None,0);



  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
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

  t->assumeDataInNewDW();
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
  
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,0);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);

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
  
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,0);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
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
    t->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,
		0);
  else
    t->requires(Task::NewDW,lb->pStressLabel,Ghost::None,
		0);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  t->computes(lb->gInternalForceLabel);  
  
  sched->addTask(t, patches, matls);
  
}

void ImpMPM::scheduleMoveData(SchedulerP& sched,const LevelP& level,
			     const PatchSet* patches, const MaterialSet* matls)
{

  Task* task = scinew Task("moveData",this,&ImpMPM::moveData);
 
  //  task->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->bElBarLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gMassLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gExternalForceLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->gAccelerationLabel,Ghost::None,0);
  //task->requires(Task::OldDW,lb->gInternalForceLabel,Ghost::None,0);
  //task->requires(Task::OldDW,lb->dispIncLabel,Ghost::None,0);
  task->requires(Task::OldDW,d_sharedState->get_delt_label());
  //task->computes(lb->dispNewLabel);
  task->computes(lb->pXLabel);
  task->computes(lb->pVolumeLabel);
  task->computes(lb->pDeformationMeasureLabel);
  task->computes(lb->bElBarLabel);
  task->computes(lb->gMassLabel);
  task->computes(lb->gExternalForceLabel);
  //task->computes(lb->gInternalForceLabel);
  task->computes(lb->gAccelerationLabel);
  task->computes(d_sharedState->get_delt_label());
  //task->computes(lb->dispIncLabel);

  sched->addTask(task,level->eachPatch(),d_sharedState->allMPMMaterials());

}

void ImpMPM::scheduleIterate(SchedulerP& sched,const LevelP& level,
			     const PatchSet*, const MaterialSet*)
{

  // NOT DONE

  Task* task = scinew Task("scheduleIterate", this, &ImpMPM::iterate,level,
			   sched);

  task->assumeDataInNewDW();
  task->hasSubScheduler();

  task->requires(Task::NewDW,lb->dispNewLabel,Ghost::AroundCells,1);
  //task->requires(Task::OldDW,lb->dispNewLabel,Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc,Ghost::None,0);
  task->requires(Task::OldDW,lb->pXLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pVolumeLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->pDeformationMeasureLabel,Ghost::None,0);
  task->requires(Task::OldDW,lb->bElBarLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gMassLabel,Ghost::None,0);
  task->requires(Task::OldDW,d_sharedState->get_delt_label());
  task->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gExternalForceLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gVelocityLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->gAccelerationLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
  task->requires(Task::NewDW,lb->dispIncQNorm0);
  task->requires(Task::NewDW,lb->dispIncNormMax);
  task->requires(Task::NewDW,lb->dispIncQNorm);
  task->requires(Task::NewDW,lb->dispIncNorm);

#if 0
  task->computes(lb->pXLabel);
  task->computes(lb->dispNewLabel);
  task->computes(lb->gVelocityLabel);
#endif

  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches = lb->createPerProcessorPatchSet(level, 
								   d_myworld);
  sched->addTask(task,perproc_patches,d_sharedState->allMaterials());

  
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


  // Create the tasks

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
#if 0
  scheduleMoveData(subsched,level,level->eachPatch(),
		   d_sharedState->allMPMMaterials());
#endif
 

  subsched->compile(d_myworld,false);

  sum_vartype dispIncNorm,dispIncNormMax,dispIncQNorm,dispIncQNorm0;
  new_dw->get(dispIncNorm,lb->dispIncNorm);
  new_dw->get(dispIncQNorm,lb->dispIncQNorm); 
  new_dw->get(dispIncNormMax,lb->dispIncNormMax);
  new_dw->get(dispIncQNorm0,lb->dispIncQNorm0);
  cout << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax << endl;
  cout << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 << endl;
  
  int count = 0;
  bool dispInc = false;
  bool dispIncQ = false;
  if (dispIncNorm/dispIncNormMax <= 1.e-8)
    dispInc = true;
  if (dispIncQNorm/dispIncQNorm0 <= 1.e-8)
    dispIncQ = true;

  cout << "dispInc = " << dispInc << " dispIncQ = " << dispIncQ << endl;

  subsched->set_old_dw(old_dw);
  subsched->set_new_dw(new_dw);
  while(!dispInc && !dispIncQ) {
    cout << "Iteration = " << count++ << endl;
    subsched->execute(d_myworld);
    subsched->get_new_dw()->get(dispIncNorm,lb->dispIncNorm);
    subsched->get_new_dw()->get(dispIncQNorm,lb->dispIncQNorm); 
    subsched->get_new_dw()->get(dispIncNormMax,lb->dispIncNormMax);
    subsched->get_new_dw()->get(dispIncQNorm0,lb->dispIncQNorm0);
    cout << "dispIncNorm/dispIncNormMax = " << dispIncNorm/dispIncNormMax 
	 << endl;
    cout << "dispIncQNorm/dispIncQNorm0 = " << dispIncQNorm/dispIncQNorm0 
	 << endl;
    if (dispIncNorm/dispIncNormMax <= 1.e-8)
      dispInc = true;
    if (dispIncQNorm/dispIncQNorm0 <= 1.e-8)
      dispIncQ = true;
    if (!dispInc && !dispIncQ)
      subsched->advanceDataWarehouse(grid);
  }

#if 0
  // Get the final data
 for (int p = 0; p < patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing moveData on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< endl << endl;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      ParticleSubset* pset = 
	subsched->get_old_dw()->getParticleSubset(matlindex, patch);
      cout << "number of particles = " << pset->numParticles() << endl;
      
      constParticleVariable<Point> px_old;
      subsched->get_old_dw()->get(px_old,lb->pXLabel,pset);
      constParticleVariable<double> pvol_old;
      subsched->get_old_dw()->get(pvol_old,lb->pVolumeLabel,pset);
      constParticleVariable<Matrix3> deformationGradient,bElBar;
      subsched->get_old_dw()->get(deformationGradient, 
				  lb->pDeformationMeasureLabel, pset);
      subsched->get_old_dw()->get(bElBar, lb->bElBarLabel, pset);
#if 0
      constNCVariable<double> gmass;
      subsched->get_old_dw()->get(gmass,lb->gMassLabel,matlindex,patch,
				  Ghost::None,0);
#endif
      delt_vartype dt;
      subsched->get_old_dw()->get(dt,d_sharedState->get_delt_label());
#if 0
      constNCVariable<Vector> ext_force,int_force,acc,dispInc;
      subsched->get_old_dw()->get(ext_force,lb->gExternalForceLabel,
				  matlindex,patch, Ghost::None,0);
      subsched->get_old_dw()->get(int_force,lb->gInternalForceLabel,
				  matlindex,patch, Ghost::None,0);
      subsched->get_old_dw()->get(acc,lb->gAccelerationLabel,matlindex
				  ,patch,Ghost::None,0);
      subsched->get_old_dw()->get(dispInc,lb->dispIncLabel,matlindex,
				  patch,Ghost::None,0);
#endif
      
      ParticleVariable<Point> newpx;
      ParticleVariable<double> newpvol;
      ParticleVariable<Matrix3> newdefGrad,newbElBar;
#if 0
      NCVariable<double> newgmass;
      NCVariable<Vector> newext_force,newacc,newdispInc;
#endif      
      double newdt = dt;
      new_dw->allocate(newpx,lb->pXLabel,pset);
      new_dw->allocate(newpvol,lb->pVolumeLabel,pset);
      new_dw->allocate(newdefGrad,lb->pDeformationMeasureLabel,pset);
      new_dw->allocate(newbElBar,lb->bElBarLabel,pset);
#if 0
      new_dw->allocate(newgmass,lb->gMassLabel,matlindex,patch);
      new_dw->allocate(newext_force,lb->gExternalForceLabel,matlindex,
		       patch);
      new_dw->allocate(newint_force,lb->gInternalForceLabel,matlindex,patch);
      new_dw->allocate(newacc,lb->gAccelerationLabel,matlindex,patch);
      new_dw->allocate(newdispInc,lb->dispIncLabel,matlindex,patch);
#endif
      newpx.copyData(px_old);
      newpvol.copyData(pvol_old);
      newdefGrad.copyData(deformationGradient);
      newbElBar.copyData(bElBar);
#if 0
      newgmass.copyData(gmass);
      newext_force.copyData(ext_force);
      newint_force.copyData(int_force);
      newacc.copyData(acc);
      newdispInc.copyData(dispInc);
#endif
      new_dw->put(newpx,lb->pXLabel);
      new_dw->put(newpvol,lb->pVolumeLabel);
      new_dw->put(newdefGrad,lb->pDeformationMeasureLabel);
      new_dw->put(newbElBar,lb->bElBarLabel);
#if 0
      new_dw->put(newgmass,lb->gMassLabel,matlindex,patch);
      new_dw->put(newext_force,lb->gExternalForceLabel,matlindex,patch);
      new_dw->put(newint_force,lb->gInternalForceLabel,matlindex,patch);
      new_dw->put(newacc,lb->gAccelerationLabel,matlindex,patch);
      new_dw->put(newdispInc,lb->dispIncLabel,matlindex,patch);
#endif
      new_dw->saveParticleSubset(matlindex,patch,pset);
      new_dw->put(delt_vartype(newdt), d_sharedState->get_delt_label());
    }
 }
#endif

}

void ImpMPM::moveData(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset*,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw)
{
  
  for (int p = 0; p < patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing moveData on patch " << patch->getID()
	       <<"\t\t\t\t IMPM"<< endl << endl;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      ParticleSubset* pset = 
	old_dw->getParticleSubset(matlindex, patch);
      cout << "number of particles = " << pset->numParticles() << endl;
      
      constParticleVariable<Point> px_old;
      old_dw->get(px_old,lb->pXLabel,pset);
      constParticleVariable<double> pvol_old;
      old_dw->get(pvol_old,lb->pVolumeLabel,pset);
      constParticleVariable<Matrix3> deformationGradient,bElBar;
      old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
      old_dw->get(bElBar, lb->bElBarLabel, pset);
      constNCVariable<double> gmass;
      old_dw->get(gmass,lb->gMassLabel,matlindex,patch,Ghost::None,0);
      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label());
      constNCVariable<Vector> dispNew,ext_force,int_force,acc,dispInc;
      //old_dw->get(dispNew,lb->dispNewLabel,matlindex,patch, Ghost::None,0);
      old_dw->get(ext_force,lb->gExternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(acc,lb->gAccelerationLabel,matlindex,patch,Ghost::None,0);
#if 0
      old_dw->get(int_force,lb->gInternalForceLabel,matlindex,patch,
		  Ghost::None,0);
      old_dw->get(dispInc,lb->dispIncLabel,matlindex,patch,Ghost::None,0);
#endif
      ParticleVariable<Point> newpx;
      ParticleVariable<double> newpvol;
      ParticleVariable<Matrix3> newdefGrad,newbElBar;

      NCVariable<double> newgmass;

      NCVariable<Vector> newdisp,newext_force,newint_force,newacc,newdispInc;

      double newdt = dt;
      new_dw->allocateAndPut(newpx, lb->pXLabel,pset);
      new_dw->allocateAndPut(newpvol, lb->pVolumeLabel,pset);
      new_dw->allocateAndPut(newdefGrad, lb->pDeformationMeasureLabel,pset);
      new_dw->allocateAndPut(newbElBar, lb->bElBarLabel,pset);
      new_dw->allocateAndPut(newgmass, lb->gMassLabel,matlindex,patch);
      new_dw->allocateAndPut(newext_force, lb->gExternalForceLabel,matlindex,patch);
      //new_dw->allocate(newint_force,lb->gInternalForceLabel,matlindex,patch);
      new_dw->allocateAndPut(newacc, lb->gAccelerationLabel,matlindex,patch);
      //new_dw->allocate(newdispInc,lb->dispIncLabel,matlindex,patch);

      newpx.copyData(px_old);
      newpvol.copyData(pvol_old);
      newdefGrad.copyData(deformationGradient);
      newbElBar.copyData(bElBar);
      newgmass.copyData(gmass);
      //newdisp.copyData(dispNew);
      newext_force.copyData(ext_force);
      //newint_force.copyData(int_force);
      newacc.copyData(acc);
      //newdispInc.copyData(dispInc);

      // allocateAndPut instead:
      /* new_dw->put(newpx,lb->pXLabel); */;
      // allocateAndPut instead:
      /* new_dw->put(newpvol,lb->pVolumeLabel); */;
      // allocateAndPut instead:
      /* new_dw->put(newdefGrad,lb->pDeformationMeasureLabel); */;
      // allocateAndPut instead:
      /* new_dw->put(newbElBar,lb->bElBarLabel); */;
      // allocateAndPut instead:
      /* new_dw->put(newgmass,lb->gMassLabel,matlindex,patch); */;
      // allocateAndPut instead:
      /* new_dw->put(newext_force,lb->gExternalForceLabel,matlindex,patch); */;
      //new_dw->put(newint_force,lb->gInternalForceLabel,matlindex,patch);
      // allocateAndPut instead:
      /* new_dw->put(newacc,lb->gAccelerationLabel,matlindex,patch); */;
      // new_dw->put(newdispInc,lb->dispIncLabel,matlindex,patch);

      new_dw->saveParticleSubset(matlindex,patch,pset);
      new_dw->put(delt_vartype(newdt), d_sharedState->get_delt_label());
    }
  }
  
  
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
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gInternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gExternalForceLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->dispNewLabel,Ghost::None,0);
  t->requires(Task::OldDW,lb->gVelocityLabel,Ghost::None,0);
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
  t->assumeDataInNewDW();
  t->requires(Task::OldDW,lb->gMassLabel,Ghost::None,0);

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
  t->assumeDataInNewDW();
  if (recursion)
    t->computes(lb->dispIncLabel);
  else
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
  //t->requires(Task::OldDW,lb->dispNewLabel,Ghost::AroundCells,1);
  t->modifies(lb->dispNewLabel);
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
			&ImpMPM::checkConvergence,level,sched,recursion);

  t->requires(Task::NewDW,lb->dispIncLabel,Ghost::None,0);
#if 0
  if (recursion) {
    t->modifies(lb->dispIncNormMax);
    t->modifies(lb->dispIncQNorm0);
    t->modifies(lb->converged);
  } 
#endif

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

  t->requires(Task::NewDW, lb->gAccelerationLabel,  Ghost::None,0);
  t->requires(Task::NewDW, lb->dispNewLabel,Ghost::None);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pAccelerationLabel,     Ghost::None);
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

  t->requires(Task::NewDW, lb->pXLabel_preReloc,Ghost::None,0);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,Ghost::None,0);
  t->requires(Task::NewDW, lb->pStressLabel_preReloc,Ghost::None,0);

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
	       <<"\t\t\t IMPM"<< endl << endl;

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
    // allocateAndPut instead:
    /* new_dw->put(cellNAPID, lb->pCellNAPIDLabel, 0, patch); */;

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
	       <<"\t\t IMPM"<< endl << endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
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
					       Ghost::None, 0,
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
      NCVariable<Vector> gvelocity,gacceleration,dispNew;
      NCVariable<Vector> gexternalforce;

      new_dw->allocateAndPut(gmass, lb->gMassLabel,      matlindex, patch);
      new_dw->allocateAndPut(gvolume, lb->gVolumeLabel,    matlindex, patch);
      new_dw->allocateAndPut(gvelocity, lb->gVelocityLabel,  matlindex, patch);
      new_dw->allocateAndPut(dispNew, lb->dispNewLabel,  matlindex, patch);
      new_dw->allocateAndPut(gacceleration, lb->gAccelerationLabel,matlindex, patch);
      new_dw->allocateAndPut(gexternalforce, lb->gExternalForceLabel,matlindex,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
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

      // allocateAndPut instead:
      /* new_dw->put(gmass,         lb->gMassLabel,          matlindex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(gvolume,       lb->gVolumeLabel,        matlindex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(gvelocity, lb->gVelocityLabel,      matlindex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(dispNew, lb->dispNewLabel,      matlindex, patch); */;
      // allocateAndPut instead:
      /* new_dw->put(gacceleration,lb->gAccelerationLabel,matlindex,patch); */;
      // allocateAndPut instead:
      /* new_dw->put(gexternalforce,lb->gExternalForceLabel, matlindex, patch); */;
 

    }  // End loop over materials

     // allocateAndPut instead:
     /* new_dw->put(gmassglobal, lb->gMassLabel,
			d_sharedState->getAllInOneMatl()->get(0), patch); */;
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
	       << endl << endl;
    
  
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
	    //cout << "Velocity bc value = " << bc->getValue() << endl;
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

void ImpMPM::computeStressTensor(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* ,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // DONE

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t IMPM"<< endl << endl;


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

  cout_doing <<"Doing computeStressTensorOnly " <<"\t\t\t\t IMPM"<< endl 
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
				 DataWarehouse* new_dw,
				 const bool recursion)
{
  // DONE
  if (!dynamic)
    return;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing formStiffnessMatrix " << patch->getID()
	       <<"\t\t\t\t IMPM"<< endl << endl;

    IntVector nodes = patch->getNNodes();

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
   
      constNCVariable<double> gmass;
      if (recursion)
	new_dw->get(gmass, lb->gMassLabel,matlindex,patch, Ghost::None,0);
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
#if 0
	cout << "gmass[" << *iter << "]= " << gmass[*iter] << endl;
	cout << "KK[" << dof[0] << "][" << dof[0] << "]= " 
	     << KK[dof[0]][dof[0]] << endl;
	cout << "KK[" << dof[1] << "][" << dof[1] << "]= " 
	     << KK[dof[1]][dof[1]] << endl;
	cout << "KK[" << dof[2] << "][" << dof[2] << "]= " 
	     << KK[dof[2]][dof[2]] << endl;
#endif
	KK[dof[0]][dof[0]] = KK[dof[0]][dof[0]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[1]][dof[1]] = KK[dof[1]][dof[1]] + gmass[*iter]*(4./(dt*dt));
	KK[dof[2]][dof[2]] = KK[dof[2]][dof[2]] + gmass[*iter]*(4./(dt*dt));
      }
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
	       <<"\t\t\t IMPM"<< endl << endl;
    
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
						       Ghost::None, 0,
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
	       <<"\t\t\t\t\t IMPM"<< endl << endl;


    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());
    double fodts = 4./(dt*dt);
    double fodt = 4./dt;

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z());
    valarray<double> temp2(0.,3*num_nodes);
    Q.resize(3*num_nodes);
    
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
      old_dw->get(velocity,lb->gVelocityLabel,matlindex,patch,
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
      cout << "dispNew = " << dispNew[*iter] << endl;
    }
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[3];
      int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	(nodes.x())*(n.z());
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
	       <<"\t\t IMPM"<< endl << endl;

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
	       <<"\t\t\t\t IMPM"<< endl << endl;

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
      
      if (mass[n] <= d_SMALL_NUM_MPM) {
	fixedDOF.insert(dof[0]);
	fixedDOF.insert(dof[1]);
	fixedDOF.insert(dof[2]);
      }
    }

#if 0
    cout << "Patch cell_lo = " << patch->getCellLowIndex() << " cell_hi = " 
	 << patch->getCellHighIndex() << endl;

    cout << "Patch node_lo = " << patch->getNodeLowIndex() << " node_hi = " 
	 << patch->getNodeHighIndex() << endl;

    for (CellIterator it = patch->getCellIterator(); !it.done(); it++) {
      cout << "cell iterator = " << *it << endl;
    }

    for (NodeIterator it = patch->getNodeIterator(); !it.done(); it++) {
      cout << "node = " << *it << endl;
    }


    
    // IntVector l(0,0,0),h(1,2,2);  // xminus
    // IntVector l(1,0,0),h(2,2,2);  // xplus
    // IntVector l(0,0,0),h(2,1,2);  // yminus
    // IntVector l(0,1,0),h(2,2,2);  // yplus
    // IntVector l(0,0,0),h(2,2,1);  // zminus
     IntVector l(0,0,1),h(2,2,2);  // zplus
    for (NodeIterator it(l,h); !it.done(); it++) {
      cout << "node new = " << *it << endl;
    }
#endif
    for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
      IntVector l,h;
      patch->getFaceNodes(face,0,l,h);
#if 0
      cout << "face = " << face << " l = " << l << " h = " << h << endl;
#endif
      for(NodeIterator it(l,h); !it.done(); it++) {
	IntVector n = *it;
	int dof[3];
	int node_num = n.x() + (nodes.x())*(n.y()) + (nodes.y())*
	  (nodes.x())*(n.z());
#if 0
	cout << "node = " << n << " node_num = " << node_num << endl;
#endif
	dof[0] = 3*node_num;
	dof[1] = 3*node_num+1;
	dof[2] = 3*node_num+2;
#if 0
	cout << "dofs = " << dof[0] << "\t" << dof[1] << "\t" << dof[2] 
	     << endl;
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
      if (KK[j][j] == 0.) {
	KKK[j][j] = 1.;
	Q[j] = 0.;
      }
    }
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
	       <<"\t\t\t\t IMPM"<< endl << endl;

    IntVector nodes = patch->getNNodes();
    int num_nodes = (nodes.x())*(nodes.y())*(nodes.z())*3;

    valarray<double> x(0.,num_nodes);
    int matlindex = 0;

#if 0    
    for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
	 itr != KK.end(); itr++) {
      int i = KK.Index1(itr);
      int j = KK.Index2(itr);
      cout << "KK[" << i << "][" << j <<"] = " << KK[i][j] << endl;
    }
#endif
#if 0    
    for (unsigned int i = 0; i < Q.size(); i++) {
      cout << "Q[" << i << "]= " << Q[i] << endl;
    }
#endif    
    x = cgSolve(KK,Q,conflag);
#if 1    
    for (unsigned int i = 0; i < x.size(); i++) {
      cout << "x[" << i << "]= " << x[i] << endl;
    }
#endif
    NCVariable<Vector> dispInc;
#if 0
    if (recursion)
      new_dw->getModifiable(dispInc,lb->dispIncLabel,matlindex,patch);
    else {
#endif
      new_dw->allocateAndPut(dispInc, lb->dispIncLabel,matlindex,patch);
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
  
    // allocateAndPut instead:
    /* new_dw->put(dispInc,lb->dispIncLabel,matlindex,patch); */;
    
  }

  //return conflag;
    

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
	       <<"\t\t\t IMPM"<< endl << endl;

    int matlindex = 0;

    NCVariable<Vector> dispNew,velocity;
    constNCVariable<Vector> dispInc,dispNew_old;

    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label());

    if (recursion) {
      old_dw->get(dispNew_old, lb->dispNewLabel,matlindex,patch,
		  Ghost::AroundCells,1);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
      new_dw->allocateAndPut(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->allocateAndPut(velocity, lb->gVelocityLabel, matlindex,patch);
    }
    else {
      new_dw->getModifiable(dispNew, lb->dispNewLabel, matlindex,patch);
      new_dw->getModifiable(velocity, lb->gVelocityLabel, matlindex,patch);
      new_dw->get(dispInc, lb->dispIncLabel, matlindex,patch,Ghost::None,0);
    } 

    
    if (recursion) {
      if (dynamic) {
	for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
	  dispNew[*iter] = dispNew_old[*iter] + dispInc[*iter];
	  velocity[*iter] = dispNew[*iter]*(2./dt) - velocity[*iter];
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
    if (recursion) {
      // allocateAndPut instead:
      /* new_dw->put(dispNew,lb->dispNewLabel,matlindex,patch); */;
      // allocateAndPut instead:
      /* new_dw->put(velocity,lb->gVelocityLabel,matlindex,patch); */;
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
	       <<"\t\t\t IMPM"<< endl << endl;

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
      cout << "dispIncQNorm0_var = " << dispIncQNorm0_var << endl;
      cout << "dispIncNormMax_var = " << dispIncNormMax_var << endl;
      cout << "dispIncNorm = " << dispIncNorm << endl;
      cout << "dispIncNormQ = " << dispIncQNorm << endl;
      dispIncQNorm0 = dispIncQNorm0_var;
      dispIncNormMax = dispIncNormMax_var;

      if (!recursion || dispIncQNorm0 == 0.)
	dispIncQNorm0 = dispIncQNorm;

      if (dispIncNorm > dispIncNormMax)
	dispIncNormMax = dispIncNorm;

      cout << "dispIncQNorm0 = " << dispIncQNorm0 << endl;
      cout << "dispIncQNorm = " << dispIncQNorm << endl;
      cout << "dispIncNormMax = " << dispIncNormMax << endl;
      cout << "dispIncNorm = " << dispIncNorm << endl;

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
	       <<"\t\t\t IMPM"<< endl << endl;

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
	       << patch->getID() <<"\t IMPM"<< endl << endl;

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
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,    pset);
      new_dw->allocateAndPut(paccNew, lb->pAccelerationLabel_preReloc,pset);
      new_dw->allocateAndPut(pxnew, lb->pXLabel_preReloc,           pset);
      new_dw->allocateAndPut(pmassNew, lb->pMassLabel_preReloc,        pset);
      new_dw->allocateAndPut(pvolumeNew, lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(pexternalForceNew, lb->pExternalForceLabel_preReloc,pset);
      pexternalForceNew.copyData(pexternalForce);

      new_dw->get(dispNew,lb->dispNewLabel,dwindex,patch,Ghost::None,0);

      new_dw->get(gacceleration,      lb->gAccelerationLabel,
			dwindex, patch, Ghost::None, 0);

     
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
	  }
#endif

          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	  CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	  CMV += pvelocitynew[idx]*pmass[idx];
          massLost += (pmass[idx] - pmassNew[idx]);
        }
      
      
      // Store the new result
      // allocateAndPut instead:
      /* new_dw->put(pxnew,           lb->pXLabel_preReloc); */;
      // allocateAndPut instead:
      /* new_dw->put(pvelocitynew,    lb->pVelocityLabel_preReloc); */;
      // allocateAndPut instead:
      /* new_dw->put(paccNew,    lb->pAccelerationLabel_preReloc); */;
      // allocateAndPut instead:
      /* new_dw->put(pexternalForceNew, lb->pExternalForceLabel_preReloc); */;
      // allocateAndPut instead:
      /* new_dw->put(pmassNew,        lb->pMassLabel_preReloc); */;
      // allocateAndPut instead:
      /* new_dw->put(pvolumeNew,      lb->pVolumeLabel_preReloc); */;
      new_dw->deleteParticles(delete_particles);

      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      old_dw->get(pids, lb->pParticleIDLabel, pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);
      // allocateAndPut instead:
      /* new_dw->put(pids_new, lb->pParticleIDLabel_preReloc); */;
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
	       <<"\t\t IMPM"<< endl << endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pstress;

      ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch,
					       Ghost::None, 0,
					       lb->pXLabel_preReloc);

      new_dw->get(px,lb->pXLabel_preReloc,pset);
      new_dw->get(pmass,lb->pMassLabel_preReloc,pset);
      new_dw->get(pstress,lb->pStressLabel_preReloc,pset);

      NCVariable<Matrix3> gstress;

      new_dw->allocateAndPut(gstress, lb->gStressLabel,matlindex,patch);

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

      // allocateAndPut instead:
      /* new_dw->put(gstress, lb->gStressLabel, matlindex,patch); */;


    }  // End of loop over materials
  }  // End of loop over patches
}


void ImpMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}
