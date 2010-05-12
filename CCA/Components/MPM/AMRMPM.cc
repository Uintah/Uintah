/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <process.h>
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream cout_doing("AMRMPM", false);
static DebugStream amr_doing("AMRMPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

AMRMPM::AMRMPM(const ProcessorGroup* myworld) :SerialMPM(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags(myworld);
  flags->d_minGridLevel = 0;
  flags->d_maxGridLevel = 1000;

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  NGP     = 1;
  NGN     = 1;
  d_recompile = false;
  dataArchiver = 0;
}

AMRMPM::~AMRMPM()
{
  delete lb;
  delete flags;
}

void AMRMPM::problemSetup(const ProblemSpecP& prob_spec, 
                          const ProblemSpecP& restart_prob_spec,GridP& grid,
                          SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);

  ProblemSpecP restart_mat_ps = 0;
  if (restart_prob_spec){
    restart_mat_ps = restart_prob_spec;
  }
  else{
    restart_mat_ps = prob_spec;
  }

  ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");

  if(mpm_soln_ps) {

    // Read all MPM flags (look in MPMFlags.cc)
    flags->readMPMFlags(restart_mat_ps);
    if (flags->d_integrator_type == "implicit"){
      throw ProblemSetupException("Can't use implicit integration with -mpm",
                                   __FILE__, __LINE__);
    }

    std::vector<std::string> bndy_face_txt_list;
    mpm_soln_ps->get("boundary_traction_faces", bndy_face_txt_list);
    
    // convert text representation of face into FaceType
    for(std::vector<std::string>::const_iterator ftit(bndy_face_txt_list.begin());
        ftit!=bndy_face_txt_list.end();ftit++) {
        Patch::FaceType face = Patch::invalidFace;
        for(Patch::FaceType ft=Patch::startFace;ft<=Patch::endFace;
            ft=Patch::nextFace(ft)) {
          if(Patch::getFaceName(ft)==*ftit) face =  ft;
        }
        if(face!=Patch::invalidFace) {
          d_bndy_traction_faces.push_back(face);
        } else {
          cerr << "warning: ignoring unknown face '" << *ftit<< "'" << endl;
        }
    }
  }
    
  //__________________________________
  //  bulletproofing
  if(!d_sharedState->isLockstepAMR()){
    ostringstream msg;
    msg << "\n ERROR: You must add \n"
        << " <useLockStep> true </useLockStep> \n"
        << " inside of the <AMR> section. \n"; 
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }  
    
  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flags->d_8or27==27){
    NGP=2;
    NGN=2;
  }

  d_sharedState->setParticleGhostLayer(Ghost::AroundNodes, NGP);

  ProblemSpecP p = prob_spec->findBlock("DataArchiver");
  if(!p->get("outputInterval", d_outputInterval))
    d_outputInterval = 1.0;


  materialProblemSetup(restart_mat_ps, d_sharedState,flags);
}

//______________________________________________________________________
//
void AMRMPM::outputProblemSpec(ProblemSpecP& root_ps)
{
  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP flags_ps = root->appendChild("MPM");
  flags->outputProblemSpec(flags_ps);

  ProblemSpecP mat_ps = 0;
  mat_ps = root->findBlockWithOutAttribute("MaterialProperties");

  if (mat_ps == 0)
    mat_ps = root->appendChild("MaterialProperties");
    
  ProblemSpecP mpm_ps = mat_ps->appendChild("MPM");
  for (int i = 0; i < d_sharedState->getNumMPMMatls();i++) {
    MPMMaterial* mat = d_sharedState->getMPMMaterial(i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{

  if (flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    cout << "doMPMOnLevel = " << level->getIndex() << endl;
  }
  else{
    cout << "DontDoMPMOnLevel = " << level->getIndex() << endl;
  }
  
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;
  Task* t = scinew Task("AMRMPM::actuallyInitialize",
                  this, &AMRMPM::actuallyInitialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pFiberDirLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pTempPreviousLabel); // for therma  stress analysis
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pErosionLabel);
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);

  if(!flags->d_doGridReset){
    t->computes(lb->gDisplacementLabel);
  }
  
  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
  }

  if (flags->d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }

  if (flags->d_accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  schedulePrintParticleCount(level, sched);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

}
//______________________________________________________________________
//
void AMRMPM::schedulePrintParticleCount(const LevelP& level, 
                                        SchedulerP& sched)
{
  Task* t = scinew Task("AMRMPM::printParticleCount",
                  this, &AMRMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level), d_sharedState->allMPMMaterials());
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeStableTimestep(const LevelP&,
                                              SchedulerP&)
{
  // Nothing to do here - delt is computed as a by-product of the
  // consitutive model
}
//______________________________________________________________________
//
void AMRMPM::scheduleTimeAdvance(const LevelP & inlevel,
                                 SchedulerP   & sched)
{
  if(inlevel->getIndex() > 0)  // only schedule once
    return;

  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  int maxLevels = inlevel->getGrid()->numLevels();
  GridP grid = inlevel->getGrid();
  
  
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeZoneOfInfluence(         sched, patches, matls);
    scheduleApplyExternalLoads(             sched, patches, matls);
  }

  
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce(           sched, patches, matls);
  }
  
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
    scheduleSetGridBoundaryConditions(      sched, patches, matls);
  }
  
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeStressTensor(            sched, patches, matls);
  }
  
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{
  if (level->getIndex() == 0) {
    const MaterialSet* matls = d_sharedState->allMPMMaterials();
    sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                      d_sharedState->d_particleState_preReloc,
                                      lb->pXLabel, 
                                      d_sharedState->d_particleState,
                                      lb->pParticleIDLabel, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleComputeZoneOfInfluence(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
                                                                                
  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeZoneOfInfluence\t\t\t");                                                                                
  Task* t = scinew Task("AMRMPM::computeZoneOfInfluence",
                  this, &AMRMPM::computeZoneOfInfluence);
                                                                                
  t->computes(lb->gZOILabel, one_matl);
                                                                                
  sched->addTask(t, patches, matls);

  if (one_matl->removeReference())
    delete one_matl;
}

//______________________________________________________________________
//
void AMRMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AMRMPM::scheduleApplyExternalLoads\t\t\t\t");

  Task* t=scinew Task("AMRMPM::applyExternalLoads",
                this, &AMRMPM::applyExternalLoads);

  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void AMRMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateParticlesToGrid\t\t\t");
  

  Task* t = scinew Task("AMRMPM::interpolateParticlesToGrid",
                   this,&AMRMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,               gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                  gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,        gan,NGP);
  t->requires(Task::OldDW, lb->pErosionLabel,            gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan,NGP);
  //t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  t->computes(lb->gMassLabel);              
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->TotalMassLabel);
  
  sched->addTask(t, patches, matls);
}


#if 0
//______________________________________________________________________
//
void AMRMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateParticlesToGrid\t");
  

  Task* t = scinew Task("AMRMPM::interpolateParticlesToGrid",
                   this,&AMRMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,gan,NGP);
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pErosionLabel,          gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
  t->requires(Task::NewDW, lb->gZOILabel,              gac,NGN);

  MaterialSubset* one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

  //Task::WhichDW C_oldDW = Task::CoarseOldDW;
  
  if(getLevel(patches)->hasCoarserLevel()){
    const MaterialSubset* mss = matls->getUnion();
    Task::DomainSpec DS = Task::NormalDomain;
    t->requires(Task::CoarseOldDW, lb->pXLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pMassLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pSizeLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pVolumeLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pErosionLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pVelocityLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseOldDW, lb->pTemperatureLabel, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseNewDW, lb->pExtForceLabel_preReloc, 0,
                Task::CoarseLevel, mss, DS, gan, NGP);
    t->requires(Task::CoarseNewDW, lb->gZOILabel, 0,
                Task::CoarseLevel, one_matl, DS, gac, NGN);
  }
  if(getLevel(patches)->hasFinerLevel()){
    const MaterialSubset* mss = matls->getUnion();
    Task::DomainSpec DS = Task::NormalDomain;
    bool  fat = false;  // possibly (F)rom (A)nother (T)askgraph
    t->requires(Task::OldDW, lb->pXLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP, fat);
    t->requires(Task::OldDW, lb->pMassLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::OldDW, lb->pSizeLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::OldDW, lb->pVolumeLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::OldDW, lb->pErosionLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::OldDW, lb->pVelocityLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::OldDW, lb->pTemperatureLabel, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::NewDW, lb->pExtForceLabel_preReloc, 0,
                Task::FineLevel, mss, DS, gan, NGP,fat);
    t->requires(Task::NewDW, lb->gZOILabel, 0,
                Task::FineLevel, one_matl, DS, gac, NGN, fat);
  }

  t->computes(lb->gMassLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->TotalMassLabel);

  sched->addTask(t, patches, matls);

  if (one_matl->removeReference())
    delete one_matl;
}
#endif

//______________________________________________________________________
//
/////////////////////////////////////////////////////////////////////////
/*!  **WARNING** In addition to the stresses and deformations, the internal 
 *               heat rate in the particles (pdTdtLabel) 
 *               is computed here */
/////////////////////////////////////////////////////////////////////////
void AMRMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  
  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeStressTensor\t\t\t\t");
  
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("AMRMPM::computeStressTensor",
                  this, &AMRMPM::computeStressTensor);
                  
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();
    
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
       
    t->computes(lb->p_qLabel_preReloc, matlset);
  }

  t->computes(d_sharedState->get_delt_label(),getLevel(patches));
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  // Schedule update of the erosion parameter
  scheduleUpdateErosionParameter(sched, patches, matls);

  if (flags->d_accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"AMRMPM::scheduleUpdateErosionParameter\t\t\t");

  Task* t = scinew Task("AMRMPM::updateErosionParameter",
                  this, &AMRMPM::updateErosionParameter);
                  
  t->requires(Task::OldDW, lb->pErosionLabel,          Ghost::None);
  int numMatls = d_sharedState->getNumMPMMatls();
  
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }
  
  t->computes(lb->pErosionLabel_preReloc);
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeInternalForce\t\t\t\t");
   
  Task* t = scinew Task("AMRMPM::computeInternalForce",
                  this, &AMRMPM::computeInternalForce);

 
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW,lb->pErosionLabel,              gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,   gan,NGP);


  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  t->computes(lb->gStressForSavingLabel);
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeAndIntegrateAcceleration\t\t\t\t");

  Task* t = scinew Task("AMRMPM::computeAndIntegrateAcceleration",
                  this, &AMRMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"AMRMPM::scheduleSetGridBoundaryConditions\t");
  Task* t=scinew Task("AMRMPM::setGridBoundaryConditions",
               this, &AMRMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

  if(!flags->d_doGridReset){
    t->requires(Task::OldDW, lb->gDisplacementLabel,    Ghost::None);
    t->computes(lb->gDisplacementLabel);
  }

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateToParticlesAndUpdate\t\t\t");
  
  Task* t=scinew Task("AMRMPM::interpolateToParticlesAndUpdate",
                this, &AMRMPM::interpolateToParticlesAndUpdate);

  Task::DomainSpec DS = Task::NormalDomain;
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  bool  fat = false;  // possibly (F)rom (A)nother (T)askgraph
  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gZOILabel,                       gac,NGN);
  
  if(getLevel(patches)->hasCoarserLevel()){
    t->requires(Task::CoarseNewDW, lb->gAccelerationLabel, 0, Task::CoarseLevel, mss, DS, gac, NGN);
    t->requires(Task::CoarseNewDW, lb->gVelocityStarLabel, 0, Task::CoarseLevel, mss, DS, gac, NGN);
  }
  if(getLevel(patches)->hasFinerLevel()){
    t->requires(Task::NewDW, lb->gAccelerationLabel, 0,Task::FineLevel,  mss, DS, gac, NGN, fat);
    t->requires(Task::NewDW, lb->gVelocityStarLabel, 0,Task::FineLevel,  mss, DS, gac, NGN, fat);
  }
  t->requires(Task::OldDW, lb->pXLabel,                            gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                         gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                   gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,                  gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                     gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                         gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                         gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                       gnone);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,             gnone);
  t->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,  gnone);
  t->modifies(lb->pVolumeLabel_preReloc);

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
  
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleRefine\t\t");
  Task* t = scinew Task("AMRMPM::refine", this, &AMRMPM::refine);

  t->computes(lb->pXLabel);
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

  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
  }
                                                                                
  if (flags->d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }
                                                                                
  if (flags->d_accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }
                                                                                
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }
                                                                                
  sched->addTask(t, patches, d_sharedState->allMPMMaterials());
}
//______________________________________________________________________
//
void AMRMPM::scheduleRefineInterface(const LevelP& /*fineLevel*/, 
                                     SchedulerP& /*scheduler*/,
                                     bool, bool)
{
  // do nothing for now
}
//______________________________________________________________________
//
void AMRMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/, 
                             SchedulerP& /*sched*/)
{
  // do nothing for now
}
//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void AMRMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the 
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (amr_doing.active())
    amr_doing << "AMRMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // The simulation controller should not schedule it every time step
  Task* task = scinew Task("AMRMPM::errorEstimate", this, &AMRMPM::errorEstimate);
  
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
  sched->addTask(task, coarseLevel->eachPatch(), d_sharedState->allMPMMaterials());

}
//______________________________________________________________________
// Schedule to mark initial flags for AMR regridding
void AMRMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

//______________________________________________________________________
//
void AMRMPM::scheduleSwitchTest(const LevelP& level, 
                                 SchedulerP& sched)
{
  Task* task = scinew Task("AMRMPM::switchTest",this, &AMRMPM::switchTest);

  task->requires(Task::OldDW, d_sharedState->get_delt_label() );
  task->computes(d_sharedState->get_switch_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(),d_sharedState->allMaterials());

}
//______________________________________________________________________
//
void AMRMPM::printParticleCount(const ProcessorGroup* pg,
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
//______________________________________________________________________
//
void AMRMPM::actuallyInitialize(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch,cout_doing,"Doing actuallyInitialize\t\t\t");

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      
      if(!flags->d_doGridReset){
        NCVariable<Vector> gDisplacement;
        new_dw->allocateAndPut(gDisplacement,lb->gDisplacementLabel,indx,patch);
        gDisplacement.initialize(Vector(0.));
      }
      
      particleIndex numParticles = mpm_matl->countParticles(patch);
      totalParticles+=numParticles;

      mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

      mpm_matl->getConstitutiveModel()->initializeCMData(patch,mpm_matl,new_dw);
    }
  }

  if (flags->d_accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}
//______________________________________________________________________
//
void AMRMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                           const PatchSubset*,
                                           const MaterialSubset*,
                                           DataWarehouse*,
                                           DataWarehouse*)
{
}

//______________________________________________________________________
//
void AMRMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing interpolateParticlesToGrid\t\t\t");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    string interp_type = flags->d_interpolator_type;

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      constParticleVariable<double> pErosion;
      constParticleVariable<Matrix3> pDeformationMeasure;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,                   lb->pXLabel,                  pset); 
      old_dw->get(pmass,                lb->pMassLabel,               pset); 
      old_dw->get(pvolume,              lb->pVolumeLabel,             pset); 
      old_dw->get(pvelocity,            lb->pVelocityLabel,           pset); 
      old_dw->get(pTemperature,         lb->pTemperatureLabel,        pset); 
      old_dw->get(psize,                lb->pSizeLabel,               pset); 
      old_dw->get(pErosion,             lb->pErosionLabel,            pset); 
      old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);
      new_dw->get(pexternalforce,       lb->pExtForceLabel_preReloc, pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gTemperature;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,         dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,       dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,     dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,  dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);
      Vector pmom;
      int n8or27=flags->d_8or27;

      //double pSp_vol = 1./mpm_matl->getInitialDensity();
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);

        pmom = pvelocity[idx]*pmass[idx];
        total_mom += pmom;

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices
        IntVector node;
        for(int k = 0; k < n8or27; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
            S[k] *= pErosion[idx];
            gmass[node]          += pmass[idx]                     * S[k];
            gvelocity[node]      += pmom                           * S[k];
            gvolume[node]        += pvolume[idx]                   * S[k];
            gexternalforce[node] += pexternalforce[idx]            * S[k];
            gTemperature[node]   += pTemperature[idx] * pmass[idx] * S[k];
          }
        }
      } // End of particle loop

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter; 
        totalmass         += gmass[c];
        gvelocity[c]      /= gmass[c];
        gTemperature[c]   /= gmass[c];
      }

      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,   interp_type);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);
    }  // End loop over materials

    delete interpolator;
  }  // End loop over patches
}

#if 0
//______________________________________________________________________
//
void AMRMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing interpolateParticlesToGrid\t\t");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    
    string interp_type = flags->d_interpolator_type;
          
    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;

    constNCVariable<Stencil7> zoi_cur,zoi_fine,zoi_coarse;
    constNCVariable<Stencil7> ZOI_CUR,ZOI_FINE;
    new_dw->get(zoi_cur, lb->gZOILabel, 0, patch, gac, NGN);


    IntVector cl, ch, fl, fh, CL, CH, FL, FH;
    // Determine extents for coarser level particle data
    const Level* coarseLevel = 0;
    const Level* fineLevel = 0;
    if(getLevel(patches)->hasCoarserLevel()){
      coarseLevel = getLevel(patches)->getCoarserLevel().get_rep();

      getCoarseLevelRangeNodes(patch, coarseLevel, CL, CH, FL, FH, 1);
      new_dw->getRegion(zoi_coarse, lb->gZOILabel, 0, coarseLevel,CL,CH,false);
    }
    // Determine extents for finer level particle data
    if(getLevel(patches)->hasFinerLevel()){
      fineLevel = getLevel(patches)->getFinerLevel().get_rep();

      patch->computeVariableExtents(Patch::NodeBased, IntVector(0,0,0),
                                    gan, 1, cl, ch);

      fl = patch->getLevel()->mapNodeToFiner(cl);// - ghost;
      fh = patch->getLevel()->mapNodeToFiner(ch);// + ghost;
      new_dw->getRegion(zoi_fine, lb->gZOILabel, 0, fineLevel, fl, fh, false);
    }

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
 
      vector<ParticleSubset* > UberPset(3);

      // Create arrays for the grid data
      NCVariable<double> gmass, gvolume;
      NCVariable<Vector> gvelocity,gexternalforce;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
                                                                     dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      double totalmass = 0;

      // Create arrays for the particle data on this patch
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      constParticleVariable<double> pErosion;

      ParticleSubset* pset=0;
      for(int whichLevel=0;whichLevel<3;whichLevel++){
        bool doit = false;
        //bool get_finer = false;
        bool coarse_part = false;
        if(getLevel(patches)->hasCoarserLevel() && whichLevel==0){
          pset = old_dw->getParticleSubset(dwi, CL, CH, coarseLevel, NULL,
                                                             lb->pXLabel);
          doit = true;
          ZOI_CUR=zoi_coarse;
          ZOI_FINE=zoi_cur;
          coarse_part = true;
        }
        if(whichLevel==1){
          pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, lb->pXLabel);
          doit = true;
          ZOI_CUR=zoi_cur;
          ZOI_FINE=zoi_fine;
        }
        if(getLevel(patches)->hasFinerLevel() && whichLevel==2){
          pset = old_dw->getParticleSubset(dwi, fl, fh, fineLevel, NULL,
                                                           lb->pXLabel);
          doit = true;
          ZOI_CUR=zoi_cur;
          ZOI_FINE=zoi_fine;
        }

        if(doit){
          old_dw->get(px,                 lb->pXLabel,                  pset);
          old_dw->get(pmass,              lb->pMassLabel,               pset);
          old_dw->get(pvolume,            lb->pVolumeLabel,             pset);
          old_dw->get(pvelocity,          lb->pVelocityLabel,           pset);
          old_dw->get(pTemperature,       lb->pTemperatureLabel,        pset);
          old_dw->get(psize,              lb->pSizeLabel,               pset);
	  old_dw->get(pDeformationMeasure,lb->pDeformationMeasureLabel, pset);
          old_dw->get(pErosion,           lb->pErosionLabel,            pset);
          new_dw->get(pexternalforce,     lb->pExtForceLabel_preReloc,  pset);


          int n8or27=flags->d_8or27;
          //int num_cur, num_fine, num_coarse;
          for (ParticleSubset::iterator iter = pset->begin();
               iter != pset->end(); 
               iter++){
            particleIndex idx = *iter;

/*`==========TESTING==========*/    
#if 0
            // Get the node indices that surround the cell
            interpolator->findCellAndWeights(px[idx],ni,S,ZOI_CUR,ZOI_FINE,
                                             get_finer,num_cur,num_fine,
                                             num_coarse,psize[idx],coarse_part,
                                             patch);
#endif
                                             

            // Get the node indices that surround the cell
            interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]); 
/*===========TESTING==========`*/

            Vector pmom = pvelocity[idx]*pmass[idx];
    
            // Add each particles contribution to the local mass & velocity 
            // Must use the node indices
            IntVector node;
/*`==========TESTING==========*/
//            for(int k = 0; k < num_cur; k++) {
            for(int k = 0; k < n8or27; k++) { 
/*===========TESTING==========`*/
              node = ni[k];
              if(patch->containsNode(node)) {
                S[k] *= pErosion[idx];
                gmass[node]          += pmass[idx]                     * S[k];
                gvelocity[node]      += pmom                           * S[k];
                gvolume[node]        += pvolume[idx]                   * S[k];
                gexternalforce[node] += pexternalforce[idx]            * S[k];
              }
            }
          } // End of particle loop
        }
      }


      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter; 
        totalmass       += gmass[c];
        gvelocity[c]    /= gmass[c];
      }
      
      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);
    }  // End loop over materials
    
    delete interpolator;
  }  // End loop over patches
}
#endif

//______________________________________________________________________
//
void AMRMPM::computeStressTensor(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  printTask(patches, patches->get(0),cout_doing,
            "Doing computeStressTensor\t\t\t");

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){

    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

    cm->setWorld(UintahParallelComponent::d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
  }

}
//______________________________________________________________________
//
void AMRMPM::updateErosionParameter(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing updateErosionParameter\t\t\t\t");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the erosion data
      constParticleVariable<double> pErosion;
      ParticleVariable<double> pErosion_new;
      old_dw->get(pErosion, lb->pErosionLabel, pset);
      new_dw->allocateAndPut(pErosion_new, lb->pErosionLabel_preReloc, pset);

      // Get the localization info
      ParticleVariable<int> isLocalized;
      new_dw->allocateTemporary(isLocalized, pset);
      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

      iter = pset->begin(); 
      for (; iter != pset->end(); iter++) {
        pErosion_new[*iter] = pErosion[*iter];
        if (isLocalized[*iter]) {
          if (flags->d_erosionAlgorithm == "RemoveMass") {
            pErosion_new[*iter] = 0.1*pErosion[*iter];
          } 
        } 
      }
    }
  }
}
//______________________________________________________________________
//
void AMRMPM::computeInternalForce(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeInternalForce\t\t\t\t");

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

    string interp_type = flags->d_interpolator_type;


    int numMPMMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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
      constParticleVariable<Matrix3> pDeformationMeasure;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,         pset);             
      old_dw->get(pvol,    lb->pVolumeLabel,    pset);             
      old_dw->get(pstress, lb->pStressLabel,    pset);             
      old_dw->get(psize,   lb->pSizeLabel,      pset);             
      old_dw->get(pErosion,lb->pErosionLabel,   pset);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);

      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);

      internalforce.initialize(Vector(0,0,0));

      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],pDeformationMeasure[idx]);

        stresspress = pstress[idx];

        for (int k = 0; k < n8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],
                       d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
                       
            div *= pErosion[idx];
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
          }
        }
      }

      string interp_type = flags->d_interpolator_type;
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,interp_type);
    }

    delete interpolator;
  }
}
//______________________________________________________________________
//
void AMRMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeAndIntegrateAcceleration\t\t\t\t");

    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get required variables for this patch
      constNCVariable<Vector> internalforce, externalforce, velocity;
      constNCVariable<double> mass;

      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->get(internalforce,lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,         lb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->get(velocity,     lb->gVelocityLabel,      dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> velocity_star,acceleration;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      new_dw->allocateAndPut(acceleration,  lb->gAccelerationLabel, dwi, patch);

      acceleration.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
        IntVector c = *iter;
        Vector acc = (internalforce[c] + externalforce[c])/mass[c];
        acceleration[c]  = acc +  gravity;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }    // matls
  }
}
//______________________________________________________________________
//
void AMRMPM::setGridBoundaryConditions(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing setGridBoundaryConditions\t\t\t");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    string interp_type = flags->d_interpolator_type;

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocity;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi, patch, Ghost::None,0);
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter = patch->getExtraNodeIterator(); !iter.done();
                                                               iter++){
        IntVector c = *iter;
        gacceleration[c] = (gvelocity_star[c] - gvelocity[c])/delT;
      }

      if(!flags->d_doGridReset){
        NCVariable<Vector> displacement;
        constNCVariable<Vector> displacementOld;
        new_dw->allocateAndPut(displacement,lb->gDisplacementLabel,dwi,patch);
        old_dw->get(displacementOld,        lb->gDisplacementLabel,dwi,patch,
                                                               Ghost::None,0);
        for(NodeIterator iter = patch->getExtraNodeIterator();
                         !iter.done();iter++){
           IntVector c = *iter;
           displacement[c] = displacementOld[c] + gvelocity_star[c] * delT;
        }
      }  // d_doGridReset
      // Set symmetry BCs on acceleration if called for
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gacceleration,interp_type);
    } // matl loop
  }  // patch loop
}
//______________________________________________________________________
//
void AMRMPM::computeZoneOfInfluence(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  NCVariable<Stencil7> zoi;
  
  //__________________________________
  //  Initialize the interior nodes
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    
    printTask(patches, patch,cout_doing,"Doing computeZoneOfInfluence\t\t\t\t");
    new_dw->allocateAndPut(zoi, lb->gZOILabel, 0, patch);
    
    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      zoi[c].p=-9876543210e99;
      zoi[c].w=dx.x();
      zoi[c].e=dx.x();
      zoi[c].s=dx.y();
      zoi[c].n=dx.y();
      zoi[c].b=dx.z();
      zoi[c].t=dx.z();
    }
  }

  //__________________________________
  // set the ZOI coarse
  // look up for at the finer level patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
  
    new_dw->getModifiable(zoi, lb->gZOILabel, 0,patch);
  
    if(level->hasFinerLevel()) {
      const Level* fineLevel = level->getFinerLevel().get_rep();
     
      Level::selectType finePatches;
      patch->getFineLevelPatches(finePatches);
      
      for(int p=0;p<finePatches.size();p++){  
        const Patch* finePatch = finePatches[p];
        Vector fine_dx = finePatch->dCell();
 
        //__________________________________
        // Iterate over coarsefine interface faces
        if(finePatch->hasCoarseFaces() ){
          vector<Patch::FaceType> cf;
          finePatch->getCoarseFaces(cf);
          
          vector<Patch::FaceType>::const_iterator iter;  
          for (iter  = cf.begin(); iter != cf.end(); ++iter){
            Patch::FaceType patchFace = *iter;

            //cout << " working on face " << finePatch->getFaceName(patchFace)<<  endl;

            // determine the iterator on the coarse level.
            NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            bool isRight_CP_FP_pair;
            
            coarseLevel_CFI_NodeIterator( patchFace,patch, finePatch, fineLevel,
                                          n_iter ,isRight_CP_FP_pair);
            // The ZOI element is opposite
            // of the patch face
            int element = patchFace;
            if(patchFace == Patch::xminus || 
               patchFace == Patch::yminus || 
               patchFace == Patch::zminus){
              element += 1;  // e, n, t 
            }
            if(patchFace == Patch::xplus || 
               patchFace == Patch::yplus || 
               patchFace == Patch::zplus){
              element -= 1;   // w, s, b
            }
            IntVector dir = patch->getFaceAxes(patchFace);        // face axes
            int p_dir = dir[0];                                    // normal direction 
            
            // eject if this is not the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              for(; !n_iter.done(); n_iter++) {
                IntVector c = *n_iter;
                //cout << " coarseLevels CFI Cells L-" << level->getIndex() << " " << c << endl;
                zoi[c][element]=fine_dx[p_dir];
              }
            }
          }  // patch face loop
        }  // hasCoarseFaces
      }  // finePatches loop
    }  // has finer level
  }  // patches loop
   
   
  //__________________________________
  // set the ZOI in cells in which there are overlaping coarse level nodes
  // look down for coarse level patches 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    new_dw->getModifiable(zoi, lb->gZOILabel, 0,patch);
      
    // underlying coarse level
    if( level->hasCoarserLevel() ) {
      //const Level* coarseLevel = level->getCoarserLevel().get_rep();
      
      Level::selectType coarsePatches;
      patch->getCoarseLevelPatches(coarsePatches);

      for(int p=0;p<coarsePatches.size();p++){  
        const Patch* coarsePatch = coarsePatches[p];
        Vector coarse_dx = coarsePatch->dCell();
        
        //__________________________________
        // Iterate over coarsefine interface faces
        if(patch->hasCoarseFaces() ){
          vector<Patch::FaceType> cf;
          patch->getCoarseFaces(cf);
          
          vector<Patch::FaceType>::const_iterator iter;  
          for (iter  = cf.begin(); iter != cf.end(); ++iter){
            Patch::FaceType patchFace = *iter;

            //cout << " working on face " << patch->getFaceName(patchFace)<<  endl;

            // determine the iterator on the coarse level.
            NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            bool isRight_CP_FP_pair;
            
            fineLevel_CFI_NodeIterator( patchFace,coarsePatch, patch,
                                          n_iter ,isRight_CP_FP_pair);
                                          
            // The ZOI element is opposite
            // of the patch face
            int element = patchFace;
            if(patchFace == Patch::xminus || 
               patchFace == Patch::yminus || 
               patchFace == Patch::zminus){
              element += 1;  // e, n, t 
            }
            if(patchFace == Patch::xplus || 
               patchFace == Patch::yplus || 
               patchFace == Patch::zplus){
              element -= 1;   // w, s, b
            }
            
            IntVector dir = patch->getFaceAxes(patchFace);        // face axes
            int p_dir = dir[0];                                    // normal direction 
            
            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              for(; !n_iter.done(); n_iter++) {
                IntVector c = *n_iter;
                //cout << " fineLevel CFI Cells L-" << level->getIndex() << " " << c << endl;
                zoi[c][element]=coarse_dx[p_dir];
              }
            }

          }  // face interator
        }  // patch has coarse face
      }  // coarsePatches loop
    }  // has finer level                                                                              

  }  // patch loop
}


#if 0

//______________________________________________________________________
//
void AMRMPM::computeZoneOfInfluence(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeZoneOfInfluence\t\t\t\t");

    IntVector cl, ch, fl, fh, CL, CH, FL, FH;
    // Determine extents for coarser level particle data
    const Level* coarseLevel = 0;
    const Level* fineLevel = 0;
    const Level* curLevel = 0;
    bool coarser = false;
    bool finer = false;
    curLevel = getLevel(patches);

    Vector dx = patch->dCell();
    Vector dxfine=dx;
    Vector dxcoarse=dx;
    IntVector ref_rat_fine;

    string interp_type = flags->d_interpolator_type;

    // Find coarser level
    if(getLevel(patches)->hasCoarserLevel()){
      coarseLevel = getLevel(patches)->getCoarserLevel().get_rep();
      coarser = true;
      dxcoarse = coarseLevel->dCell();
    }
    // Find finer level
    if(getLevel(patches)->hasFinerLevel()){
      fineLevel = getLevel(patches)->getFinerLevel().get_rep();
      finer = true;
      dxfine = fineLevel->dCell();
      ref_rat_fine= IntVector((int) (dx.x()/dxfine.x()),
                              (int) (dx.y()/dxfine.y()),
                              (int) (dx.z()/dxfine.z()));
    }
                                                                                
    NCVariable<Stencil7> zoi;
    new_dw->allocateAndPut(zoi, lb->gZOILabel, 0, patch);

    if(!coarser && !finer){
      for(NodeIterator iter = patch->getExtraNodeIterator();
          !iter.done();iter++){
        IntVector c = *iter;
        zoi[c].p=1.;
        zoi[c].w=dx.x();
        zoi[c].e=dx.x();
        zoi[c].s=dx.y();
        zoi[c].n=dx.y();
        zoi[c].b=dx.z();
        zoi[c].t=dx.z();
      }
      return;
    }

    // T-B is z+,z-, E-W is x+,x-, N-S is y+ y-
    Point TBNSEW[8];
    Vector TBNSEWh[8];
    Vector tne = .25*Vector( dx.x(), dx.y(), dx.z());
    Vector tnw = .25*Vector(-dx.x(), dx.y(), dx.z());
    Vector tse = .25*Vector( dx.x(),-dx.y(), dx.z());
    Vector tsw = .25*Vector(-dx.x(),-dx.y(), dx.z());
    Vector bne = .25*Vector( dx.x(), dx.y(),-dx.z());
    Vector bnw = .25*Vector(-dx.x(), dx.y(),-dx.z());
    Vector bse = .25*Vector( dx.x(),-dx.y(),-dx.z());
    Vector bsw = .25*Vector(-dx.x(),-dx.y(),-dx.z());

    cout << "Patch node high index = " << patch->getExtraNodeHighIndex() << endl;
    cout << "Patch cell high index = " << patch->getExtraCellHighIndex() << endl;

    for(NodeIterator iter = patch->getExtraNodeIterator();!iter.done();iter++){
      IntVector c = *iter;

      Point node_pos = curLevel->getNodePosition(c);
      TBNSEW[0] = node_pos+tne;
      TBNSEW[1] = node_pos+tnw;
      TBNSEW[2] = node_pos+tse;
      TBNSEW[3] = node_pos+tsw;
      TBNSEW[4] = node_pos+bne;
      TBNSEW[5] = node_pos+bnw;
      TBNSEW[6] = node_pos+bse;
      TBNSEW[7] = node_pos+bsw;


      for(int i=0;i<8;i++){
        if(curLevel->containsPointIncludingExtraCells(TBNSEW[i])){
          // The resolution at that point is at least the current resolution
          TBNSEWh[i]=dx;
          if(finer){  // If there's a finer level, check for the point there
            if(fineLevel->containsPointIncludingExtraCells(TBNSEW[i])){
              // The resolution is that of the finer level at this point
              TBNSEWh[i]=dxfine;
            }
          }
        }
        else{
          // Point is either off the edge of the fine level, either on
          // coarse level or outside the domain.
          if(coarser){  // If there's a finer level, check for the point there
            if(coarseLevel->containsPointIncludingExtraCells(TBNSEW[i])){
              // The resolution is that of the coarser level at this point
              TBNSEWh[i]=dxcoarse;
            }
            else{
              // There's a coarser level, but this point isn't in it
              TBNSEWh[i]=dxcoarse;  // or 0.;
            }
          }
          else{
            // There isn't a coarser level, and the point isn't in curLevel
            TBNSEWh[i]=dx;  // or 0.;
          }
        }
      }

      if(finer){
        IntVector c_finer = c*ref_rat_fine;
        if(c==IntVector(2,0,0)){
           cout << "The Finer Node = " << c_finer << endl;
        }
        if(fineLevel->selectPatchForNodeIndex(c_finer)!=0){
          zoi[c].p=0.;
        }
        else{
          zoi[c].p=1.;
        }
      }
      else{
        zoi[c].p=1.;
      }

      zoi[c].t=min(min(TBNSEWh[0].z(),TBNSEWh[1].z()),
                   min(TBNSEWh[2].z(),TBNSEWh[3].z()));
      zoi[c].b=min(min(TBNSEWh[4].z(),TBNSEWh[5].z()),
                   min(TBNSEWh[6].z(),TBNSEWh[7].z()));
      zoi[c].e=min(min(TBNSEWh[0].x(),TBNSEWh[2].x()),
                   min(TBNSEWh[4].x(),TBNSEWh[6].x()));
      zoi[c].w=min(min(TBNSEWh[1].x(),TBNSEWh[3].x()),
                   min(TBNSEWh[5].x(),TBNSEWh[7].x()));
      zoi[c].n=min(min(TBNSEWh[0].y(),TBNSEWh[1].y()),
                   min(TBNSEWh[4].y(),TBNSEWh[5].y()));
      zoi[c].s=min(min(TBNSEWh[2].y(),TBNSEWh[3].y()),
                   min(TBNSEWh[6].y(),TBNSEWh[7].y()));

//      if(coarser){
//      cout << "node = " << c << endl
//           << "zoi.p = " << zoi[c].p << endl
//           << "zoi.t = " << zoi[c].t << endl
//           << "zoi.b = " << zoi[c].b << endl
//           << "zoi.n = " << zoi[c].n << endl
//           << "zoi.s = " << zoi[c].s << endl
//           << "zoi.e = " << zoi[c].e << endl
//           << "zoi.w = " << zoi[c].w << endl;
//      }
      if(c==IntVector(2,0,0)){
         cout << "The Node" << endl;
         cout << "zoi.p = " << zoi[c].p << endl;
      }

    }
  }
}

#endif
//______________________________________________________________________
//
void AMRMPM::applyExternalLoads(const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();
  
  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalLoads) = " << time << endl;

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing applyExternalLoads\t\t\t\t");
    
    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.
    
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Carry forward the old pEF, scale by d_forceIncrementFactor
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
        pExternalForce_new[idx] = 
                pExternalForce[idx]*flags->d_forceIncrementFactor;
      }
    } // matl loop
  }  // patch loop
}
//______________________________________________________________________
//
void AMRMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing interpolateToParticlesAndUpdate\t\t\t");

    double thermal_energy = 0.0;
    double ke = 0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    Ghost::GhostType  gac = Ghost::AroundCells;

    constNCVariable<Stencil7> ZOI_CUR,ZOI_FINE;
    new_dw->get(ZOI_CUR, lb->gZOILabel, 0, patch, gac, NGN);

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    double move_particles=1.;
    if(!flags->d_doGridReset){
      move_particles=0.;
    }
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity, psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<Vector> pvelocitynew, psizeNew;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<double> pErosion;
      ParticleVariable<double> pTempPreNew;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gv_star_coarse, gacc_coarse;
      constNCVariable<Vector> gv_star_fine, gacc_fine;
      
      double Cp =mpm_matl->getSpecificHeat();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      old_dw->get(pvolume,      lb->pVolumeLabel,                    pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(pErosion,     lb->pErosionLabel_preReloc,          pset);
      new_dw->get(pDeformationMeasure,   lb->pDeformationMeasureLabel_preReloc, pset);
      new_dw->getModifiable(pvolumeNew,  lb->pVolumeLabel_preReloc,             pset);
      
      
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);

      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

      ParticleSubset* delset = scinew ParticleSubset(0,dwi,patch);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);

      // Get finer level data
      const Patch* finePatch = NULL;
      // FIX: finePatch business
      
      
      if(getLevel(patches)->hasFinerLevel()){
         const Level* fineLevel = getLevel(patches)->getFinerLevel().get_rep();
         Level::selectType finePatches;
         patch->getFineLevelPatches(finePatches);
         IntVector one(1,1,1);
         IntVector FH(-9999,-9999,-9999);
         IntVector FL(9999,9999,9999);
         
         for(int i=0;i<finePatches.size();i++){
            finePatch = finePatches[i];

            IntVector cl, ch, fl, fh;
            getFineLevelRangeNodes(patch, finePatch, cl, ch, fl, fh, one);
            FL=Min(fl,FL);
            FH=Max(fh,FH);
         }
         new_dw->getRegion(gv_star_fine,  lb->gVelocityStarLabel, dwi,
                                          fineLevel, FL, FH, false);
         new_dw->getRegion(gacc_fine,     lb->gAccelerationLabel, dwi,
                                          fineLevel, FL, FH, false);
         new_dw->getRegion(ZOI_FINE,      lb->gZOILabel, 0,
                                          fineLevel, FL, FH, false);
      }

      // Get coarser level data
      if(getLevel(patches)->hasCoarserLevel()){
         const Level* coarseLevel = getLevel(patches)->getCoarserLevel().get_rep();
         IntVector cl, ch, fl, fh;
         getCoarseLevelRangeNodes(patch, coarseLevel, cl, ch, fl, fh, 1);
         
         new_dw->getRegion(gv_star_coarse, lb->gVelocityStarLabel, dwi,
                                           coarseLevel, cl, ch, false);
         new_dw->getRegion(gacc_coarse,    lb->gAccelerationLabel, dwi,
                                           coarseLevel, cl, ch, false);
//         new_dw->getRegion(zoi_coarse,     lb->gZOILabel, 0,
//                                           coarseLevel, cl, ch, false);
      }

      // Loop over particles
      //bool get_finer = true;
      //bool coarse_part = false;
      //int num_cur, num_fine, num_coarse;
      int n8or27=flags->d_8or27;

      
      
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

/*`==========TESTING==========*/
#if 0
        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,ZOI_CUR,ZOI_FINE,
                                         get_finer,num_cur,num_fine,
                                         num_coarse,psize[idx],
                                         coarse_part,finePatch); 
#endif
        // Get the node indices that surround the cell                
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);    
/*===========TESTING==========`*/

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);

        // Accumulate the contribution from vertices on this level
 /*`==========TESTING==========*/
#if 0
       for (int k = 0; k < num_cur; k++) { 
#endif
       for(int k = 0; k < n8or27; k++) {
/*===========TESTING==========`*/
          IntVector node = ni[k];
          S[k] *= pErosion[idx];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];
        }
        
        
/*`==========TESTING==========*/
#if 0
        // Accumulate the contribution from vertices on the finer level
        for (int k = num_cur; k < num_cur+num_fine; k++) {
          IntVector node = ni[k];
          S[k] *= pErosion[idx];
          vel      += gv_star_fine[node]  * S[k];
          acc      += gacc_fine[node]     * S[k];
        } 
#endif
/*===========TESTING==========`*/

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + acc*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx];
        pTempPreNew[idx]     = pTemperature[idx]; //
        pmassNew[idx]        = pmass[idx];
        pvolumeNew[idx]      = pvolume[idx];
        

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom += pvelocitynew[idx]*pmass[idx];
        
      }
      new_dw->deleteParticles(delset);  
     
      new_dw->put(sum_vartype(ke),              lb->KineticEnergyLabel);
      new_dw->put(sum_vartype(thermal_energy),  lb->ThermalEnergyLabel);
      new_dw->put(sumvec_vartype(CMX),          lb->CenterOfMassPositionLabel);
      new_dw->put(sumvec_vartype(totalMom),     lb->TotalMomentumLabel);
      //__________________________________
      //  particle debugging label-- carry forward
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }     
    }
    delete interpolator;
  }
}
//______________________________________________________________________
//
void AMRMPM::setParticleDefault(ParticleVariable<double>& pvar,
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

//______________________________________________________________________
//
void  AMRMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
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
//______________________________________________________________________
//
void 
AMRMPM::setParticleDefault(ParticleVariable<Matrix3>& pvar,
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
//______________________________________________________________________
//
void AMRMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}

void AMRMPM::printParticleLabels(vector<const VarLabel*> labels,
                                 DataWarehouse* dw, int dwi, 
                                 const Patch* patch)
{
  for (vector<const VarLabel*>::const_iterator it = labels.begin(); 
       it != labels.end(); it++) {
    if (dw->exists(*it,dwi,patch))
      cout << (*it)->getName() << " does exists" << endl;
    else
      cout << (*it)->getName() << " does NOT exists" << endl;
  }
}

//______________________________________________________________________
void
AMRMPM::initialErrorEstimate(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialErrorEstimate\t\t\t\t");

    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                          0, patch);
    new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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
  }
}
//______________________________________________________________________
void
AMRMPM::errorEstimate(const ProcessorGroup* group,
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
    const Level* fineLevel = level->getFinerLevel().get_rep();
  
    for(int p=0;p<patches->size();p++){  
      const Patch* coarsePatch = patches->get(p);
      printTask(patches, coarsePatch,cout_doing,
                "Doing errorEstimate\t\t\t\t\t");
     
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
    } // coarse patch loop 
  }
}  
//______________________________________________________________________
//
void AMRMPM::refine(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* /*matls*/,
                    DataWarehouse*,
                    DataWarehouse* new_dw)
{
  // just create a particle subset if one doesn't exist
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing refine\t\t\t");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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
        ParticleVariable<double> pErosion, pTempPrev;
        ParticleVariable<int>    pLoadCurve;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        if (flags->d_useLoadCurves){
          new_dw->allocateAndPut(pLoadCurve,   lb->pLoadCurveIDLabel,   pset);
        }
        new_dw->allocateAndPut(psize,          lb->pSizeLabel,          pset);
        new_dw->allocateAndPut(pErosion,       lb->pErosionLabel,       pset);

        mpm_matl->getConstitutiveModel()->initializeCMData(patch,
                                                           mpm_matl,new_dw);
      }
    }
  }

} // end refine()
//______________________________________________________________________
//
void AMRMPM::scheduleCheckNeedAddMPMMaterial(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleCheckNeedAddMPMMateria\t\t");

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("ARMMPM::checkNeedAddMPMMaterial",
                  this, &AMRMPM::checkNeedAddMPMMaterial);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->scheduleCheckNeedAddMPMMaterial(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::checkNeedAddMPMMaterial(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->checkNeedAddMPMMaterial(patches, mpm_matl, old_dw, new_dw);
  }
}
//______________________________________________________________________
//
void AMRMPM::scheduleSetNeedAddMaterialFlag(SchedulerP& sched,
                                            const LevelP& level,
                                            const MaterialSet* all_matls)
{
  printSchedule(level,cout_doing,"AMRMPM::scheduleSetNeedAddMaterialFlag\t\t");

  Task* t= scinew Task("AMRMPM::setNeedAddMaterialFlag",
                 this, &AMRMPM::setNeedAddMaterialFlag);
  t->requires(Task::NewDW, lb->NeedAddMPMMaterialLabel);
  sched->addTask(t, level->eachPatch(), all_matls);
}
//______________________________________________________________________
//
void AMRMPM::setNeedAddMaterialFlag(const ProcessorGroup*,
                                       const PatchSubset* ,
                                       const MaterialSubset* ,
                                       DataWarehouse* ,
                                       DataWarehouse* new_dw)
{
    sum_vartype need_add_flag;
    new_dw->get(need_add_flag, lb->NeedAddMPMMaterialLabel);

    if(need_add_flag < -0.1){
      d_sharedState->setNeedAddMaterial(-99);
      flags->d_canAddMPMMaterial=false;
      cout << "AMRMPM setting NAM to -99" << endl;
    }
    else{
      d_sharedState->setNeedAddMaterial(0);
    }
}
//______________________________________________________________________
//
bool AMRMPM::needRecompile(double , double , const GridP& )
{
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
//______________________________________________________________________
//
void AMRMPM::switchTest(const ProcessorGroup* group,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  int time_step = d_sharedState->getCurrentTopLevelTimeStep();
  double sw = 0;
  if (time_step == 6 )
    sw = 1;
  else
    sw = 0;

  max_vartype switch_condition(sw);
  new_dw->put(switch_condition,d_sharedState->get_switch_label(),getLevel(patches));
}
