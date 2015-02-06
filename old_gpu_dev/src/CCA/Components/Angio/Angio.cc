/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/Angio/Angio.h>
#include <CCA/Components/Angio/AngioMaterial.h>
#include <CCA/Components/Angio/AngioParticleCreator.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <fstream>
#include <sstream>


using namespace Uintah;
using namespace SCIRun;

using namespace std;

static DebugStream cout_doing("Angio_doing", false);
static DebugStream cout_dbg("Angio_dbg", false);
static DebugStream amr_doing("AMRAngio", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

Angio::Angio(const ProcessorGroup* myworld) : UintahParallelComponent(myworld)
{
  lb = scinew AngioLabel();
  flags = scinew AngioFlags();

  d_nextOutputTime=0.;
  d_SMALL_NUM_Angio=1e-200;
  d_PI = 3.14159265359;
  NGP     = 1;
  NGN     = 1;
  d_recompile = false;
  dataArchiver = 0;
}

Angio::~Angio()
{
  delete lb;
  delete flags;
}

void Angio::problemSetup(const ProblemSpecP& prob_spec, 
                             const ProblemSpecP& restart_prob_spec,GridP& grid,
                             SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);

  ProblemSpecP restart_mat_ps = 0;
  ProblemSpecP prob_spec_mat_ps =
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  if (prob_spec_mat_ps)
    restart_mat_ps = prob_spec;
  else if (restart_prob_spec)
    restart_mat_ps = restart_prob_spec;
  else
    restart_mat_ps = prob_spec;

  ProblemSpecP angio_soln_ps = restart_mat_ps->findBlock("Angio");
  if (!angio_soln_ps){
    ostringstream warn;
    warn<<"ERROR:Angio:\n missing Angio section in the input file\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  // Read all Angio flags (look in AngioFlags.cc)
  flags->readAngioFlags(restart_mat_ps);
 
  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flags->d_8or27==27 || flags->d_8or27==64){
    NGP=2;
    NGN=2;
  }

  d_sharedState->setParticleGhostLayer(Ghost::AroundNodes, NGP);

  ProblemSpecP p = prob_spec->findBlock("DataArchiver");
  if(!p->get("outputInterval", d_outputInterval))
    d_outputInterval = 1.0;

  materialProblemSetup(restart_mat_ps, d_sharedState, flags);
}

void Angio::outputProblemSpec(ProblemSpecP& root_ps)
{
  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP flags_ps = root->appendChild("Angio");
  flags->outputProblemSpec(flags_ps);

  ProblemSpecP mat_ps = 0;
  mat_ps = root->findBlockWithOutAttribute("MaterialProperties");

  if (mat_ps == 0)
    mat_ps = root->appendChild("MaterialProperties");
    
  ProblemSpecP angio_ps = mat_ps->appendChild("Angio");
  for (int i = 0; i < d_sharedState->getNumAngioMatls();i++) {
    AngioMaterial* mat = d_sharedState->getAngioMaterial(i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(angio_ps);
  }
}

void Angio::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  Task* t = scinew Task("Angio::actuallyInitialize",
                        this, &Angio::actuallyInitialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->pXLabel);
  t->computes(lb->pGrowthLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pLengthLabel);
  t->computes(lb->pPhiLabel);
  t->computes(lb->pRadiusLabel);
  t->computes(lb->pTimeOfBirthLabel);
  t->computes(lb->pRecentBranchLabel);
  t->computes(lb->pTip0Label);
  t->computes(lb->pTip1Label);
  t->computes(lb->pParentLabel);
  t->computes(lb->pParticleIDLabel);

  t->computes(lb->VesselDensityLabel);
  t->computes(lb->SmoothedVesselDensityLabel);
  t->computes(lb->VesselDensityGradientLabel);
  t->computes(lb->CollagenThetaLabel);
  t->computes(lb->CollagenDevLabel);

  t->computes(lb->partCountLabel);
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);

  sched->addTask(t, level->eachPatch(), d_sharedState->allAngioMaterials());

  schedulePrintParticleCount(level, sched);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

}

void Angio::schedulePrintParticleCount(const LevelP& level, 
                                           SchedulerP& sched)
{
  Task* t = scinew Task("Angio::printParticleCount",
                        this, &Angio::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level), d_sharedState->allAngioMaterials());
}

void Angio::scheduleComputeStableTimestep(const LevelP& level,
                                          SchedulerP& sched)
{
  Task* t = 0;
  cout_doing << d_myworld->myrank() 
             << " Angio::scheduleComputeStableTimestep \t\t\t\tL-"
             << level->getIndex() << endl;
  t = scinew Task("Angio::actuallyComputeStableTimestep",
                   this, &Angio::actuallyComputeStableTimestep);

  const MaterialSet* angio_matls = d_sharedState->allAngioMaterials();

  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  sched->addTask(t,level->eachPatch(), angio_matls);
}

void Angio::scheduleTimeAdvance(const LevelP & level,
                                SchedulerP   & sched)
{
  MALLOC_TRACE_TAG_SCOPE("Angio::scheduleTimeAdvance()");

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allAngioMaterials();

//  scheduleCarryForwardOldSegments(        sched, patches, matls);
  scheduleGrowAtTips(                     sched, patches, matls);
  scheduleSetBCsInterpolated(             sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleAddNewParticles(                sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_sharedState->d_particleState_preReloc,
                                    lb->pXLabel, 
                                    d_sharedState->d_particleState,
                                    lb->pParticleIDLabel, matls);
}

void Angio::scheduleCarryForwardOldSegments(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{       
  printSchedule(patches,cout_doing,"Angio::carryForwardOldSegments\t\t\t");
        
  Task* t = scinew Task("Angio::carryForwardOldSegments",
                        this,&Angio::carryForwardOldSegments);
  Ghost::GhostType  gnone = Ghost::None;

  t->requires(Task::OldDW, lb->pXLabel,                gnone);
  t->requires(Task::OldDW, lb->pGrowthLabel,           gnone);
  t->requires(Task::OldDW, lb->pMassLabel,             gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       gnone);
  t->requires(Task::OldDW, lb->pTip0Label,             gnone);
  t->requires(Task::OldDW, lb->pTip1Label,             gnone);
  t->requires(Task::OldDW, lb->pLengthLabel,           gnone);
  t->requires(Task::OldDW, lb->pRadiusLabel,           gnone);
  t->requires(Task::OldDW, lb->pPhiLabel,              gnone);
  t->requires(Task::OldDW, lb->pParentLabel,           gnone);
  t->requires(Task::OldDW, lb->pTimeOfBirthLabel,      gnone);
  t->requires(Task::OldDW, lb->pRecentBranchLabel,     gnone);

  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pGrowthLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pTip0Label_preReloc);
  t->computes(lb->pTip1Label_preReloc);
  t->computes(lb->pLengthLabel_preReloc);
  t->computes(lb->pRadiusLabel_preReloc);
  t->computes(lb->pPhiLabel_preReloc);
  t->computes(lb->pParentLabel_preReloc);
  t->computes(lb->pTimeOfBirthLabel_preReloc);
  t->computes(lb->pRecentBranchLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void Angio::scheduleGrowAtTips(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"Angio::growAtTips\t\t\t");
  
  Task* t = scinew Task("Angio::growAtTips",
                        this,&Angio::growAtTips);
  Ghost::GhostType  gnone = Ghost::None;

  t->requires(Task::OldDW, lb->pXLabel,                gnone);
  t->requires(Task::OldDW, lb->pGrowthLabel,           gnone);
  t->requires(Task::OldDW, lb->pMassLabel,             gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       gnone);
  t->requires(Task::OldDW, lb->pTip0Label,             gnone);
  t->requires(Task::OldDW, lb->pTip1Label,             gnone);
  t->requires(Task::OldDW, lb->pLengthLabel,           gnone);
  t->requires(Task::OldDW, lb->pRadiusLabel,           gnone);
  t->requires(Task::OldDW, lb->pPhiLabel,              gnone);
  t->requires(Task::OldDW, lb->pParentLabel,           gnone);
  t->requires(Task::OldDW, lb->pTimeOfBirthLabel,      gnone);
  t->requires(Task::OldDW, lb->pRecentBranchLabel,     gnone);
                                                                                
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pGrowthLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pTip0Label_preReloc);
  t->computes(lb->pTip1Label_preReloc);
  t->computes(lb->pLengthLabel_preReloc);
  t->computes(lb->pRadiusLabel_preReloc);
  t->computes(lb->pPhiLabel_preReloc);
  t->computes(lb->pParentLabel_preReloc);
  t->computes(lb->pTimeOfBirthLabel_preReloc);
  t->computes(lb->pRecentBranchLabel_preReloc);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->requires(Task::OldDW, lb->pCellNAPIDLabel,  zeroth_matl,gnone,Ghost::None);
  t->computes(lb->pCellNAPIDLabel,               zeroth_matl);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  sched->addTask(t, patches, matls);
}

void Angio::scheduleSetBCsInterpolated(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"Angio::scheduleSetBCsInterpolated\t\t\t");

  Task* t = scinew Task("Angio::setBCsInterpolated",
                        this,&Angio::setBCsInterpolated);
/*
  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gVelocityInterpLabel);
*/
  sched->addTask(t, patches, matls);
}

void Angio::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  /*
   * computeInternalForce
   *   in(P.CONMOD, P.NAT_X, P.VOLUME)
   *   operation(evaluate the divergence of the stress (stored in
   *   P.CONMOD) using P.NAT_X and the gradients of the
   *   shape functions)
   * out(G.F_INTERNAL) */

  printSchedule(patches,cout_doing,"Angio::scheduleComputeInternalForce\t\t\t\t");
   
  Task* t = scinew Task("Angio::computeInternalForce",
                        this, &Angio::computeInternalForce);

/*
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW,lb->pErosionLabel,              gan,NGP);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::OldDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
*/
  sched->addTask(t, patches, matls);
}

void Angio::scheduleSolveEquationsMotion(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  /* solveEquationsMotion
   *   in(G.MASS, G.F_INTERNAL)
   *   operation(acceleration = f/m)
   *   out(G.ACCELERATION) */

  printSchedule(patches,cout_doing,"Angio::scheduleSolveEquationsMotione\t\t\t\t");
  
  Task* t = scinew Task("Angio::solveEquationsMotion",
                        this, &Angio::solveEquationsMotion);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());
/*

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->computes(lb->gAccelerationLabel);
*/

  sched->addTask(t, patches, matls);
}

void Angio::scheduleIntegrateAcceleration(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  /* integrateAcceleration
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  printSchedule(patches,cout_doing,"Angio::scheduleIntegrateAcceleration\t\t\t\t");
  
  Task* t = scinew Task("Angio::integrateAcceleration",
                        this, &Angio::integrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
/*

  t->requires(Task::NewDW, lb->gAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->computes(lb->gVelocityStarLabel);
*/

  sched->addTask(t, patches, matls);
}

void Angio::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  printSchedule(patches,cout_doing,"Angio::scheduleSetGridBoundaryConditions\t\t\t");
  Task* t=scinew Task("Angio::setGridBoundaryConditions",
                      this, &Angio::setGridBoundaryConditions);
                  
/*  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityInterpLabel,   Ghost::None);

  if(!flags->d_doGridReset){
    t->requires(Task::OldDW, lb->gDisplacementLabel,    Ghost::None);
    t->computes(lb->gDisplacementLabel);
  }
*/
  sched->addTask(t, patches, matls);
}

void Angio::scheduleAddNewParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
//if  manual_new_material==false, DON't do this task OR
//if  create_new_particles==true, DON'T do this task
//  if (!flags->d_addNewMaterial || flags->d_createNewParticles) return;

//if  manual__new_material==true, DO this task OR
//if  create_new_particles==false, DO this task

  printSchedule(patches,cout_doing,"Angio::scheduleAddNewParticles\t\t");
  Task* t=scinew Task("Angio::addNewParticles", this, 
                      &Angio::addNewParticles);
/*
  int numMatls = d_sharedState->getNumAngioMatls();

  for(int m = 0; m < numMatls; m++){
    AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial(m);
    mpm_matl->getParticleCreator()->allocateVariablesAddRequires(t, mpm_matl,
                                                                 patches);
  }
*/
  sched->addTask(t, patches, matls);
}

void Angio::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls)
{
  /*
   * interpolateToParticlesAndUpdate
   *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
   *   operation(interpolate acceleration and v* to particles and
   *   integrate these to get new particle velocity and position)
   * out(P.VELOCITY, P.X, P.NAT_X) */

  printSchedule(patches,cout_doing,"Angio::scheduleInterpolateToParticlesAndUpdate\t\t\t");
  
  Task* t=scinew Task("Angio::interpolateToParticlesAndUpdate",
                      this, &Angio::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

/*
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->modifies(lb->pVolumeLabel_preReloc);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,          gnone);
    
  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress 
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pXXLabel);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->TotalMomentumLabel);
  
*/
  sched->addTask(t, patches, matls);
}

void Angio::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"Angio::scheduleRefine\t\t");
  Task* t = scinew Task("Angio::refine", this, &Angio::refine);
  t->computes(lb->pXLabel);
/*
  t->computes(lb->p_qLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pTempPreviousLabel); // for therma  stresm analysis
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pErosionLabel);
*/

  sched->addTask(t, patches, d_sharedState->allAngioMaterials());
}

void Angio::scheduleRefineInterface(const LevelP& /*fineLevel*/, 
                                        SchedulerP& /*scheduler*/,
                                        bool, bool)
{
  // do nothing for now
}

void Angio::scheduleCoarsen(const LevelP& /*coarseLevel*/, 
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}
//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void Angio::scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the 
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (amr_doing.active())
    amr_doing << "Angio::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // The simulation controller should not schedule it every time step
  Task* task = scinew Task("errorEstimate", this, &Angio::errorEstimate);
  
  // if the finest level, compute flagged cells
  if (coarseLevel->getIndex() == coarseLevel->getGrid()->numLevels()-1) {
    task->requires(Task::NewDW, lb->pXLabel, Ghost::AroundCells, 0);
  }
  else {
    task->requires(Task::NewDW, d_sharedState->get_refineFlag_label(),
                   0, Task::FineLevel, d_sharedState->refineFlagMaterials(), 
                   Task::NormalDomain, Ghost::None, 0);
  }
  task->modifies(d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allAngioMaterials());

}
//______________________________________________________________________
// Schedule to mark initial flags for AMR regridding
void Angio::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void Angio::printParticleCount(const ProcessorGroup* pg,
                                   const PatchSubset*,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  sumlong_vartype pcount;
  new_dw->get(pcount, lb->partCountLabel);
  
  if(pg->myrank() == 0){
    cerr << "Created " << (long) pcount << " total particles\n";
  }
}

void Angio::actuallyInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch,cout_doing,"Doing actuallyInitialize\t\t\t");

    // Helper variable needed to give each particle a unique ID
    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

    // Create and initialize the particle based variables
    //  angio_matl makes calls to the AngioParticleCreator
    for(int m=0;m<matls->size();m++){
      AngioMaterial* angio_matl = d_sharedState->getAngioMaterial( m );

      particleIndex numParticles = angio_matl->countParticles(patch);
      totalParticles+=numParticles;

      angio_matl->createParticles(numParticles,cellNAPID,patch,new_dw);
    }

    // Declare and initialize the grid data
    NCVariable<double> density, smoo_dens, theta, dev;
    NCVariable<Vector> dens_grad;

    new_dw->allocateAndPut(density,   lb->VesselDensityLabel,        0,patch);
    new_dw->allocateAndPut(smoo_dens, lb->SmoothedVesselDensityLabel,0,patch);
    new_dw->allocateAndPut(theta,     lb->CollagenThetaLabel,        0,patch);
    new_dw->allocateAndPut(dev,       lb->CollagenDevLabel,          0,patch);
    new_dw->allocateAndPut(dens_grad, lb->VesselDensityGradientLabel,0,patch);

    density.initialize(0.);
    smoo_dens.initialize(0.);
    dev.initialize(0.5);
    dens_grad.initialize(Vector(0.));

    const double pi = 3.14159265359;
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector n = *iter;
      theta[n] = drand48()*2*pi - pi;
    }
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}

void Angio::actuallyComputeStableTimestep(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse*,
                                          DataWarehouse* new_dw)
{
    //cout << "Need to add some code to computeStableTimestep" << endl;
  const Level* level = getLevel(patches);
  double delt=1;
  for(int pa=0;pa<patches->size();pa++){
    const Patch* patch = patches->get(pa);
    cout_doing << d_myworld->myrank() 
               << " Doing Compute Stable Timestep on patch " << patch->getID()
               << "\t\t Angio \tL-" <<level->getIndex()<< endl;

    //Compute time step which causes vessel to grow half a grid cell
    Vector dx = patch->dCell();
    double half_cell_l = 0.5*dx.minComponent();
    const double E = 2.718281828;
    double a  = flags->d_Grow_a;
    double b  = flags->d_Grow_b;
    double x0 = flags->d_Grow_x0;
    double time = d_sharedState->getElapsedTime();

    double q = pow(E,-(time-x0)/b);
    double p = -(half_cell_l + half_cell_l*q + a*q)/
                (half_cell_l + half_cell_l*q - a);

    static bool first_adj=true;

    if (p > 0){
      if (first_adj) {
         delt = (time-x0+log(p)*b)/6.;
         first_adj = false;
      } else{
        delt = (time-x0+log(p)*b);
      }
    } else {
       cerr <<  "p is negative in actuallyComputeStableTimestep" << endl;
    }
  } // patches

  new_dw->put(delt_vartype(delt), lb->delTLabel, level);
}

void Angio::carryForwardOldSegments(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  
    printTask(patches,patch,cout_doing,"Doing carryForwardOldSegments\t\t");

    int numMatls = d_sharedState->getNumAngioMatls();

    for(int m = 0; m < numMatls; m++){
      AngioMaterial* angio_matl = d_sharedState->getAngioMaterial( m );
      int dwi = angio_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<Vector> pgrowth;
      constParticleVariable<double> pmass, pvolume,plength,pradius;
      constParticleVariable<double> phi,tofb,recent_branch;
      constParticleVariable<long64> pids;
      constParticleVariable<int>    ptip0,ptip1,parent;

      ParticleVariable<Point>  px_new;
      ParticleVariable<Vector> pgrowth_new;
      ParticleVariable<double> pmass_new,pvolume_new,plength_new,pradius_new;
      ParticleVariable<double> phi_new,tofb_new,recent_branch_new;
      ParticleVariable<long64> pids_new;
      ParticleVariable<int>    ptip0_new,ptip1_new,parent_new;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                              Ghost::None, 0, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pgrowth,        lb->pGrowthLabel,        pset);
      old_dw->get(ptip0,          lb->pTip0Label,          pset);
      old_dw->get(ptip1,          lb->pTip1Label,          pset);
      old_dw->get(plength,        lb->pLengthLabel,        pset);
      old_dw->get(pradius,        lb->pRadiusLabel,        pset);
      old_dw->get(phi,            lb->pPhiLabel,           pset);
      old_dw->get(parent,         lb->pParentLabel,        pset);
      old_dw->get(tofb,           lb->pTimeOfBirthLabel,   pset);
      old_dw->get(recent_branch,  lb->pRecentBranchLabel,  pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pids,           lb->pParticleIDLabel,    pset);

      new_dw->allocateAndPut(px_new,       lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pgrowth_new,  lb->pGrowthLabel_preReloc,     pset);
      new_dw->allocateAndPut(ptip0_new,    lb->pTip0Label_preReloc,       pset);
      new_dw->allocateAndPut(ptip1_new,    lb->pTip1Label_preReloc,       pset);
      new_dw->allocateAndPut(plength_new,  lb->pLengthLabel_preReloc,     pset);
      new_dw->allocateAndPut(pradius_new,  lb->pRadiusLabel_preReloc,     pset);
      new_dw->allocateAndPut(phi_new,      lb->pPhiLabel_preReloc,        pset);
      new_dw->allocateAndPut(parent_new,   lb->pParentLabel_preReloc,     pset);
      new_dw->allocateAndPut(tofb_new,     lb->pTimeOfBirthLabel_preReloc,pset);
      new_dw->allocateAndPut(recent_branch_new,
                                          lb->pRecentBranchLabel_preReloc,pset);
      new_dw->allocateAndPut(pmass_new,    lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolume_new,  lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);

      // Carry forward existing particle data to the next patch, but
      // only do this for particles that live on this patch (pset)
      px_new.copyData(px);
      pgrowth_new.copyData(pgrowth);
      ptip0_new.copyData(ptip0);
      ptip1_new.copyData(ptip1);
      plength_new.copyData(plength);
      pradius_new.copyData(pradius);
      phi_new.copyData(phi);
      tofb_new.copyData(tofb);
      parent_new.copyData(parent);
      recent_branch_new.copyData(recent_branch);
      pmass_new.copyData(pmass);
      pvolume_new.copyData(pvolume);
      pids_new.copyData(pids);
    }
  }
}

void Angio::growAtTips(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* ,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing growAtTips\t\t");

    int numMatls = d_sharedState->getNumAngioMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    double time = d_sharedState->getElapsedTime();

    constCCVariable<short int> OldCellNAPID;
    old_dw->get(OldCellNAPID, lb->pCellNAPIDLabel, 0, patch,Ghost::None,0);

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);

    cellNAPID.copyData(OldCellNAPID);

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < numMatls; m++){
      AngioMaterial* angio_matl = d_sharedState->getAngioMaterial( m );
      int dwi = angio_matl->getDWIndex();

      double density = angio_matl->getInitialDensity();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<Vector> pgrowth;
      constParticleVariable<double> pmass, pvolume,plength,pradius;
      constParticleVariable<double> phi,tofb,recent_branch;
      constParticleVariable<long64> pids;
      constParticleVariable<int>    ptip0, ptip1, parent;

      ParticleVariable<Point>  px_new;
      ParticleVariable<Vector> pgrowth_new;
      ParticleVariable<double> pmass_new,pvolume_new,plength_new,pradius_new;
      ParticleVariable<double> phi_new,tofb_new,recent_branch_new;
      ParticleVariable<long64> pids_new;
      ParticleVariable<int>    ptip0_new,ptip1_new,parent_new;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                           gnone,0,lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pgrowth,        lb->pGrowthLabel,        pset);
      old_dw->get(plength,        lb->pLengthLabel,        pset);
      old_dw->get(pradius,        lb->pRadiusLabel,        pset);
      old_dw->get(phi,            lb->pPhiLabel,           pset);
      old_dw->get(parent,         lb->pParentLabel,        pset);
      old_dw->get(tofb,           lb->pTimeOfBirthLabel,   pset);
      old_dw->get(recent_branch,  lb->pRecentBranchLabel,  pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pids,           lb->pParticleIDLabel,    pset);
      old_dw->get(ptip0,          lb->pTip0Label,          pset);
      old_dw->get(ptip1,          lb->pTip1Label,          pset);

      new_dw->allocateAndPut(px_new,       lb->pXLabel_preReloc,          pset);      new_dw->allocateAndPut(pgrowth_new,  lb->pGrowthLabel_preReloc,     pset);      new_dw->allocateAndPut(ptip0_new,    lb->pTip0Label_preReloc,       pset);      new_dw->allocateAndPut(ptip1_new,    lb->pTip1Label_preReloc,       pset);      new_dw->allocateAndPut(plength_new,  lb->pLengthLabel_preReloc,     pset);      new_dw->allocateAndPut(pradius_new,  lb->pRadiusLabel_preReloc,     pset);      new_dw->allocateAndPut(phi_new,      lb->pPhiLabel_preReloc,        pset);      new_dw->allocateAndPut(parent_new,   lb->pParentLabel_preReloc,     pset);      new_dw->allocateAndPut(tofb_new,     lb->pTimeOfBirthLabel_preReloc,pset);      new_dw->allocateAndPut(recent_branch_new,
                                          lb->pRecentBranchLabel_preReloc,pset);      new_dw->allocateAndPut(pmass_new,    lb->pMassLabel_preReloc,       pset);      new_dw->allocateAndPut(pvolume_new,  lb->pVolumeLabel_preReloc,     pset);      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);                                                                                
      // Carry forward existing particle data to the next patch, but
      // only do this for particles that live on this patch (pset)
      px_new.copyData(px);
      pgrowth_new.copyData(pgrowth);
      ptip0_new.copyData(ptip0);
      ptip1_new.copyData(ptip1);
      plength_new.copyData(plength);
      pradius_new.copyData(pradius);
      phi_new.copyData(phi);
      tofb_new.copyData(tofb);
      parent_new.copyData(parent);
      recent_branch_new.copyData(recent_branch);
      pmass_new.copyData(pmass);
      pvolume_new.copyData(pvolume);
      pids_new.copyData(pids);

      Point x_new;
      Vector growth_new(0.,0.,0.);
      double l_new,rad_new,ang_new,t_new,r_b_new,pm_new,pv_new;
      int pt0_new=0,pt1_new=0,par_new;

      vector<Point> vx_new;
      vector<Vector> vgrowth_new;
      vector<double> vl_new,vrad_new,vang_new,vt_new,vr_b_new,vpm_new,vpvol_new;
      vector<int> vpt0_new,vpt1_new,vpar_new;
      vector<IntVector> vcell_idx;

      // count number of expected segments to be added
      int num_new_segs=0;
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(ptip0[idx]!=0){ num_new_segs++; }
        if(ptip1[idx]!=0){ num_new_segs++; }
      }
      cout << "num_potential_new_segs = " << num_new_segs << endl;

      // reserve space for the new segment data
      vx_new.reserve(num_new_segs);
      vgrowth_new.reserve(num_new_segs);
      vl_new.reserve(num_new_segs);
      vrad_new.reserve(num_new_segs);
      vang_new.reserve(num_new_segs);
      vt_new.reserve(num_new_segs);
      vr_b_new.reserve(num_new_segs);
      vpm_new.reserve(num_new_segs);
      vpvol_new.reserve(num_new_segs);
      vpt0_new.reserve(num_new_segs);
      vpt1_new.reserve(num_new_segs);
      vpar_new.reserve(num_new_segs);
      vcell_idx.reserve(num_new_segs);

      IntVector cell_idx;

      // Grow new segments from the tips which are "active"
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Grow from "right" tip
        if(ptip0[idx]!=0){
          l_new = findLength(delT);
          ang_new=findAngle(phi[idx]);
          t_new=time+delT;
          r_b_new=recent_branch[idx];
          rad_new=pradius[idx];
          pv_new=l_new*(rad_new*rad_new)*d_PI;
          pm_new=pv_new*density;
          par_new=parent[idx];
          if(ptip0[idx]==1){
            x_new=px[idx]+pgrowth[idx];
            growth_new=Vector(l_new*cos(ang_new),l_new*sin(ang_new),0.);
            pt1_new=1;
            pt0_new=0;
          }
          if(ptip0[idx]==-1){
            growth_new=Vector(-l_new*cos(ang_new),-l_new*sin(ang_new),0.);
            x_new=px[idx]+growth_new;
            pt0_new=-1;
            pt1_new=0;
          }

//          adjustXNewForPeriodic(x_new,bmin,bmax);

          // Because of frustrating parallelism issues, push the
          // OLD particle's data into the add set, and put the
          // new particle data into the old particle's spot
          patch->findCell(px[idx],cell_idx);

          vx_new.push_back(px[idx]);
          vgrowth_new.push_back(pgrowth[idx]);
          vl_new.push_back(plength[idx]);
          vrad_new.push_back(pradius[idx]);
          vang_new.push_back(phi[idx]);
          vt_new.push_back(tofb[idx]);
          vr_b_new.push_back(recent_branch[idx]);
          vpm_new.push_back(pmass[idx]);
          vpvol_new.push_back(pvolume[idx]);
          vpt0_new.push_back(0);
          vpt1_new.push_back(ptip1[idx]);
          vpar_new.push_back(parent[idx]);
          vcell_idx.push_back(cell_idx);

          px_new[idx]=x_new;
          pgrowth_new[idx]=growth_new;
          plength_new[idx]=l_new;
          pmass_new[idx]=pm_new;
          pvolume_new[idx]=pv_new;
          pradius_new[idx]=rad_new;
          phi_new[idx]=ang_new;
          tofb_new[idx]=t_new;
          recent_branch_new[idx]=r_b_new;
          parent_new[idx]=par_new;
          ptip0_new[idx]=pt0_new;
          ptip1_new[idx]=pt1_new;
        }

        // Grow from "left" tip
        if(ptip1[idx]!=0){
          l_new = findLength(delT);
          ang_new=findAngle(phi[idx]);
          t_new=time+delT;
          r_b_new=recent_branch[idx];
          rad_new=pradius[idx];
          pv_new=l_new*(rad_new*rad_new)*d_PI;
          pm_new=pv_new*density;
          par_new=parent[idx];
          if(ptip1[idx]==1){
            x_new=px[idx]+pgrowth[idx];
            growth_new=Vector(l_new*cos(ang_new),l_new*sin(ang_new),0.);
            pt1_new=1;
            pt0_new=0;
          }
          if(ptip1[idx]==-1){
            growth_new=Vector(-l_new*cos(ang_new),-l_new*sin(ang_new),0.);
            x_new=px[idx]+growth_new;
            pt0_new=-1;
            pt1_new=0;
          }
//          adjustXNewForPeriodic(x_new,bmin,bmax);

          // Because of frustrating parallelism issues, push the
          // OLD particle's data into the add set, and put the
          // new particle data into the old particle's spot
          patch->findCell(px[idx],cell_idx);

          if(vx_new[vx_new.size()-1]!=px[idx]){
            vx_new.push_back(px[idx]);
            vgrowth_new.push_back(pgrowth[idx]);
            vl_new.push_back(plength[idx]);
            vrad_new.push_back(pradius[idx]);
            vang_new.push_back(phi[idx]);
            vt_new.push_back(tofb[idx]);
            vr_b_new.push_back(recent_branch[idx]);
            vpm_new.push_back(pmass[idx]);
            vpvol_new.push_back(pvolume[idx]);
            vpt0_new.push_back(ptip0[idx]);
            vpt1_new.push_back(0);
            vpar_new.push_back(parent[idx]);
            vcell_idx.push_back(cell_idx);

            px_new[idx]=x_new;
            pgrowth_new[idx]=growth_new;
            plength_new[idx]=l_new;
            pmass_new[idx]=pm_new;
            pvolume_new[idx]=pv_new;
            pradius_new[idx]=rad_new;
            phi_new[idx]=ang_new;
            tofb_new[idx]=t_new;
            recent_branch_new[idx]=r_b_new;
            parent_new[idx]=par_new;
            ptip0_new[idx]=pt0_new;
            ptip1_new[idx]=pt1_new;
          }
        }
      } // End of particle loop
      int num_new_particles=vx_new.size();
      AngioParticleCreator* particle_creator = angio_matl->getParticleCreator();
      ParticleSubset* addset=scinew ParticleSubset(num_new_particles,dwi,patch);

      map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

      particle_creator->allocateVariablesAdd(new_dw,addset,newState,
                                             vx_new,vgrowth_new,vl_new,vrad_new,
                                             vang_new,vt_new,vr_b_new,vpm_new,
                                             vpvol_new,vpt0_new,vpt1_new,
                                             vpar_new,vcell_idx,cellNAPID);

      new_dw->addParticles(patch,dwi,newState);

    }  // End loop over materials
  }  // End loop over patches
}

double Angio::findLength(const double& delt)
{

  const double E = 2.718281828;
  double a  = flags->d_Grow_a;
  double b  = flags->d_Grow_b;
  double x0 = flags->d_Grow_x0;
  double time = d_sharedState->getElapsedTime();

  double lc;
  lc  = a/(1.+pow(E,-(time-x0)/b));
  lc -= a/(1.+pow(E,-(time-delt-x0)/b));

  return lc;
}

double Angio::findAngle(const double& old_ang)
{
  double new_ang;

  new_ang = old_ang + drand48()*d_PI - d_PI/2.;

  return new_ang;
}

void Angio::setBCsInterpolated(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing setBCsInterpolated\t\t\t");
/*
    int numMatls = d_sharedState->getNumAngioMatls();
    string inter_type = flags->d_interpolator_type;
    for(int m = 0; m < numMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      NCVariable<Vector> gvelocity,gvelocityInterp;
      new_dw->getModifiable(gvelocity,      lb->gVelocityLabel,      dwi,patch);
      new_dw->getModifiable(gvelocityInterp,lb->gVelocityInterpLabel,dwi,patch);

      gvelocityInterp.copyData(gvelocity);

      // Apply grid boundary conditions to the velocity before storing the data
      AngioBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity,      inter_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity,      inter_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocityInterp,inter_type);
    }
*/
  }  // End loop over patches
}

void Angio::computeInternalForce(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // node based forces
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeInternalForce\t\t\t\t");
/*
    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());


    int numAngioMatls = d_sharedState->getNumAngioMatls();

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gvolumeglobal;
    new_dw->get(gvolumeglobal,  lb->gVolumeLabel,
                d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
                           d_sharedState->getAllInOneMatl()->get(0), patch);

    for(int m = 0; m < numAngioMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol;
      constParticleVariable<double>  p_pressure;
      constParticleVariable<double>  p_q;
      constParticleVariable<Matrix3> pstress;
      constParticleVariable<Vector>  psize;
      constParticleVariable<double>  pErosion;
      NCVariable<Vector>             internalforce;
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gvolume;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      old_dw->get(pvol,    lb->pVolumeLabel,                 pset);
      old_dw->get(pstress, lb->pStressLabel,                 pset);
      old_dw->get(psize,   lb->pSizeLabel,                   pset);
      old_dw->get(pErosion,lb->pErosionLabel,                pset);

      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);

      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);

      if(flags->d_with_ice){
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }
      else {
        ParticleVariable<double>  p_pressure_create;
        new_dw->allocateTemporary(p_pressure_create,  pset);
        for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
          p_pressure_create[*it]=0.0;
        }
        p_pressure = p_pressure_create; // reference created data
      }

      if(flags->d_artificial_viscosity){
        old_dw->get(p_q,lb->p_qLabel, pset);
      }
      else {
        ParticleVariable<double>  p_q_create;
        new_dw->allocateTemporary(p_q_create,  pset);
        for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
          p_q_create[*it]=0.0;
        }
        p_q = p_q_create; // reference created data
      }

      internalforce.initialize(Vector(0,0,0));

      Matrix3 stressvol;
      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        stressvol  = pstress[idx]*pvol[idx];
        //stresspress = pstress[idx] + Id*p_pressure[idx];
        stresspress = pstress[idx] + Id*p_pressure[idx] - Id*p_q[idx];
        partvoldef += pvol[idx];

        for (int k = 0; k < n8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            div *= pErosion[idx];
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
            gstress[ni[k]]       += stressvol * S[k];
          }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstressglobal[c] += gstress[c];
        gstress[c] /= gvolume[c];
      }

      // save boundary forces before apply symmetry boundary condition.
      for(list<Patch::FaceType>::const_iterator fit(d_bndy_traction_faces.begin()); 
          fit!=d_bndy_traction_faces.end();fit++) {       
        Patch::FaceType face = *fit;
        
        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
           continue;
        
        const int iface = (int)face;
      
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 

        IntVector projlow, projhigh;
        patch->getFaceNodes(face, 0, projlow, projhigh);
        Vector norm = face_norm(face);
        double celldepth  = dx[iface/2]; // length in dir. perp. to boundary

        // loop over face nodes to find boundary forces, ave. stress (traction).
        // Note that nodearea incorporates a factor of two as described in the
        // bndyCellArea calculation in order to get node face areas.
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);        
              
              // flip sign so that pushing on boundary gives positive force
              bndyForce[iface] -= internalforce[ijk];

              double nodearea   = 2.0*gvolume[ijk]/celldepth; // node area
              for(int ic=0;ic<3;ic++) for(int jc=0;jc<3;jc++) {
               bndyTraction[iface][ic] += gstress[ijk](ic,jc)*norm[jc]*nodearea;
              }
            }
          }
        }
      } // faces
      
      string interp_type = flags->d_interpolator_type;
      AngioBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,interp_type);
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      gstressglobal[c] /= gvolumeglobal[c];
    }
    delete interpolator;
  }
  new_dw->put(sum_vartype(partvoldef), lb->TotalVolumeDeformedLabel);
*/
  }
}


void Angio::solveEquationsMotion(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing solveEquationsMotion\t\t\t\t");

/*
    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    Ghost::GhostType  gnone = Ghost::None;
    string interp_type = flags->d_interpolator_type;
    for(int m = 0; m < d_sharedState->getNumAngioMatls(); m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<double> mass;
 
      new_dw->get(internalforce,lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,         lb->gMassLabel,          dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));
  
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        Vector acc(0.0,0.0,0.0);
          acc = (internalforce[c] + externalforce[c])/mass[c] ;
          acceleration[c] = acc +  gravity;
      }
    }
*/
  }
}



void Angio::integrateAcceleration(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing integrateAcceleration\t\t\t\t");

/*
    string interp_type = flags->d_interpolator_type;
    for(int m = 0; m < d_sharedState->getNumAngioMatls(); m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<Vector>  acceleration, velocity;
      delt_vartype delT;

      new_dw->get(acceleration,lb->gAccelerationLabel,dwi, patch,Ghost::None,0);
      new_dw->get(velocity,    lb->gVelocityLabel,    dwi, patch,Ghost::None,0);

      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      velocity_star.initialize(Vector(0,0,0));

      for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
        IntVector c = *iter;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }    // matls
*/
  }
}


void Angio::setGridBoundaryConditions(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing setGridBoundaryConditions\t\t\t");
/*
    int numAngioMatls=d_sharedState->getNumAngioMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    string interp_type = flags->d_interpolator_type;
    for(int m = 0; m < numAngioMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocityInterp;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocityInterp,         lb->gVelocityInterpLabel,dwi, patch,
                                                                Ghost::None,0);
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles

      AngioBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();
                                                                iter++){
        IntVector c = *iter;
        gacceleration[c] = (gvelocity_star[c] - gvelocityInterp[c])/delT;
      }

      if(!flags->d_doGridReset){
        NCVariable<Vector> displacement;
        constNCVariable<Vector> displacementOld;
        new_dw->allocateAndPut(displacement,lb->gDisplacementLabel,dwi,patch);
        old_dw->get(displacementOld,        lb->gDisplacementLabel,dwi,patch,
                                                               Ghost::None,0);
        for(NodeIterator iter=patch->getExtraNodeIterator();
                         !iter.done();iter++){
           IntVector c = *iter;
           displacement[c] = displacementOld[c] + gvelocity_star[c] * delT;
        }
      }  // d_doGridReset
      // Set symmetry BCs on acceleration if called for
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gacceleration, interp_type);
    } // matl loop
*/
  }  // patch loop
}

void Angio::addNewParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing addNewParticles\t\t\t");

/*
    int numAngioMatls=d_sharedState->getNumAngioMatls();
    // Find the mpm material that the void particles are going to change
    // into.
    AngioMaterial* null_matl = 0;
    int null_dwi = -1;
    for (int void_matl = 0; void_matl < numAngioMatls; void_matl++) {
      null_dwi = d_sharedState->getAngioMaterial(void_matl)->nullGeomObject();

      if (cout_dbg.active())
        cout_dbg << "Null DWI = " << null_dwi << endl;

      if (null_dwi != -1) {
        null_matl = d_sharedState->getAngioMaterial(void_matl);
        null_dwi = null_matl->getDWIndex();
        break;
      }
    }
    for(int m = 0; m < numAngioMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      if (dwi == null_dwi)
        continue;

      ParticleVariable<int> damage;
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      new_dw->allocateTemporary(damage,pset);
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
           iter++) 
        damage[*iter] = 0;

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
           iter++) {
        if (damage[*iter]) {

          if (cout_dbg.active())
            cout_dbg << "damage[" << *iter << "]=" << damage[*iter] << endl;

          delset->addParticle(*iter);
        }
      }
      
      // Find the mpm material that corresponds to the void particles.
      // Will probably be the same type as the deleted ones, but have
      // different parameters.
      
      
      int numparticles = delset->numParticles();

      if (cout_dbg.active())
        cout_dbg << "Num Failed Particles = " << numparticles << endl;

      if (numparticles != 0) {

        if (cout_dbg.active())
          cout_dbg << "Deleted " << numparticles << " particles" << endl;

        ParticleCreator* particle_creator = null_matl->getParticleCreator();
        ParticleSubset* addset = scinew ParticleSubset(numparticles,null_dwi,patch);

        if (cout_dbg.active()) {
          cout_dbg << "Address of delset = " << delset << endl;
          cout_dbg << "Address of pset = " << pset << endl;
          cout_dbg << "Address of addset = " << addset << endl;
        }

        
        map<const VarLabel*, ParticleVariableBase*>* newState
          = scinew map<const VarLabel*, ParticleVariableBase*>;

        if (cout_dbg.active()) {
          cout_dbg << "Address of newState = " << newState << endl;
          cout_dbg << "Null Material" << endl;
        }

        if (cout_dbg.active())
          cout_dbg << "Angio Material" << endl;

        particle_creator->allocateVariablesAdd(new_dw,addset,newState,
                                               delset,old_dw);
        
        // Need to carry forward the cellNAPID for each time step;
        // Move the particle variable declarations in ParticleCreator.h to one
        // of the functions to save on memory;

        if (cout_dbg.active()){
          cout_dbg << "addset num particles = " << addset->numParticles()
                   << " for material " << addset->getMatlIndex() << endl;
        }

        new_dw->addParticles(patch,null_dwi,newState);

        if (cout_dbg.active())
           cout_dbg << "Calling deleteParticles for material: " << dwi << endl;

        new_dw->deleteParticles(delset);
        
      } else
        delete delset;
    }
*/
  }
}

void Angio::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing interpolateToParticlesAndUpdate\t\t\t");
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    int numAngioMatls=d_sharedState->getNumAngioMatls();

    for(int m = 0; m < numAngioMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

/*
      // Delete particles that have left the domain
      // This is only needed if extra cells are being used.
      // Also delete particles whose mass is too small (due to combustion)
      // For particles whose new velocity exceeds a maximum set in the input
      // file, set their velocity back to the velocity that it came into
      // this step with
      for(ParticleSubset::iterator iter  = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if ((pmassNew[idx] <= flags->d_min_part_mass) || pTempNew[idx] < 0. ){
          delset->addParticle(idx);
//        cout << "Material = " << m << " Deleted Particle = " << idx 
//             << " xold = " << px[idx] << " xnew = " << pxnew[idx]
//             << " vold = " << pvelocity[idx] << " vnew = "<< pvelocitynew[idx]
//             << " massold = " << pmass[idx] << " massnew = " << pmassNew[idx]
//             << " tempold = " << pTemperature[idx] 
//             << " tempnew = " << pTempNew[idx]
//             << " volnew = " << pvolume[idx] << endl;
        }
        if(pvelocitynew[idx].length() > flags->d_max_vel){
          pvelocitynew[idx]=pvelocity[idx];
        }
      }

*/
      new_dw->deleteParticles(delset);      
    }

    delete interpolator;
  }
  
}

void 
Angio::setParticleDefault(ParticleVariable<double>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              double val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void 
Angio::setParticleDefault(ParticleVariable<Vector>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              const Vector& val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void 
Angio::setParticleDefault(ParticleVariable<Matrix3>& pvar,
                              const VarLabel* label, 
                              ParticleSubset* pset,
                              DataWarehouse* new_dw,
                              const Matrix3& val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void Angio::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}

//______________________________________________________________________
void
Angio::initialErrorEstimate(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialErrorEstimate\t\t\t\t");

/*
    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                          0, patch);
    new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    

    for(int m = 0; m < d_sharedState->getNumAngioMatls(); m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Loop over particles
      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      new_dw->get(px, lb->pXLabel, pset);
      
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        refineFlag[patch->getLevel()->getCellIndex(px[*iter])] = true;
        refinePatch->set();
      }
    }
*/
  }
}
//______________________________________________________________________
void
Angio::errorEstimate(const ProcessorGroup* group,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  if (level->getIndex() == level->getGrid()->numLevels()-1) {
    // on finest level, we do the same thing as initialErrorEstimate, so call it
    initialErrorEstimate(group, patches, matls, old_dw, new_dw);
  }
  else {
    // coarsen the errorflag.
//    const Level* fineLevel = level->getFinerLevel().get_rep();
  
    for(int p=0;p<patches->size();p++){  
      const Patch* coarsePatch = patches->get(p);
      printTask(patches, coarsePatch,cout_doing,
                "Doing errorEstimate\t\t\t\t\t");
/*    
      CCVariable<int> refineFlag;
      PerPatch<PatchFlagP> refinePatchFlag;
      
      new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                            0, coarsePatch);
      new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                  0, coarsePatch);

      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      
      // coarsen the fineLevel flag
      for(int i=0;i<finePatches.size();i++){
        const Patch* finePatch = finePatches[i];
 
        IntVector cl, ch, fl, fh;
        getFineLevelRange(coarsePatch, finePatch, cl, ch, fl, fh);
        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
          continue;
        }
        constCCVariable<int> fineErrorFlag;
        new_dw->getRegion(fineErrorFlag, 
                          d_sharedState->get_refineFlag_label(), 0, 
                          fineLevel,fl, fh, false);
        
        //__________________________________
        //if the fine level flag has been set
        // then set the corrsponding coarse level flag
        for(CellIterator iter(cl, ch); !iter.done(); iter++){
          IntVector fineStart(level->mapCellToFiner(*iter));
          
          for(CellIterator inside(IntVector(0,0,0), 
               fineLevel->getRefinementRatio()); !inside.done(); inside++){
               
            if (fineErrorFlag[fineStart+*inside]) {
              refineFlag[*iter] = 1;
              refinePatch->set();
            }
          }
        }  // coarse patch iterator
      }  // fine patch loop
*/
    } // coarse patch loop 
  }
}  

void
Angio::refine(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* /*matls*/,
                  DataWarehouse*,
                  DataWarehouse* new_dw)
{
  // just create a particle subset if one doesn't exist
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing refine\t\t\t");

/*
    int numAngioMatls=d_sharedState->getNumAngioMatls();
    for(int m = 0; m < numAngioMatls; m++){
      AngioMaterial* mpm_matl = d_sharedState->getAngioMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      if (cout_doing.active()) {
        cout_doing <<"Doing refine on patch "
                   << patch->getID() << " material # = " << dwi << endl;
      }

      // this is a new patch, so create empty particle variables.
      if (!new_dw->haveParticleSubset(dwi, patch)) {
        ParticleSubset* pset = new_dw->createParticleSubset(0, dwi, patch);

        // Create arrays for the particle data
        ParticleVariable<Point>  px;
        ParticleVariable<double> pmass, pvolume, pTemperature;
        ParticleVariable<Vector> pvelocity, pexternalforce, psize, pdisp;
        ParticleVariable<double> pErosion, pTempPrev,p_q;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(p_q,            lb->p_qLabel,            pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        new_dw->allocateAndPut(psize,          lb->pSizeLabel,          pset);
        new_dw->allocateAndPut(pErosion,       lb->pErosionLabel,       pset);
      }
    }
*/
  }

} // end refine()

bool Angio::needRecompile(double , double , const GridP& )
{
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}

void Angio::materialProblemSetup(const ProblemSpecP& prob_spec,
                                 SimulationStateP& sharedState,
                                 AngioFlags* flags)
{
  //Search for the MaterialProperties block and then get the Angio section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP angio_mat_ps = mat_ps->findBlock("Angio");
  for (ProblemSpecP ps = angio_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;
                                                                                
    id >> index_val;
                                                                                
    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems (redstorm) it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";
                                                                                
    //Create and register as an Angio material
    AngioMaterial *mat = scinew AngioMaterial(ps, sharedState, flags);
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its
    // index number.
    if (index_val > -1){
      sharedState->registerAngioMaterial(mat,index_val);
    }
    else{
      sharedState->registerAngioMaterial(mat);
    }
  }  // material
}


void Angio::adjustXNewForPeriodic(Point &x, const Point min, const Point max)
{
  double tx=x.x();
  double ty=x.y();
  double tz=x.z();

  //Adjust x
  if(x.x()<min.x()){
    tx=max.x() - (min.x() - x.x());
  }
  if(x.x()>max.x()){
    tx=min.x() + (x.x() - max.x());
  }
  //Adjust y
  if(x.y()<min.y()){
    ty=max.y() - (min.y() - x.y());
  }
  if(x.y()>max.y()){
    ty=min.y() + (x.y() - max.y());
  }
  //Adjust z
  if(x.z()<min.z()){
    tz=max.z() - (min.z() - x.z());
  }
  if(x.z()>max.z()){
    tz=min.z() + (x.z() - max.z());
  }

  x=Point(tx,ty,tz);
}
