#include <Packages/Uintah/CCA/Components/MPM/AMRMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/AMR.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

#ifdef _WIN32
#include <process.h>
#endif

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SerialMPM", false);
static DebugStream cout_convert("MPMConv", false);
static DebugStream cout_heat("MPMHeat", false);
static DebugStream amr_doing("AMRMPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

static Vector face_norm(Patch::FaceType f)
{
  switch(f) { 
  case Patch::xminus: return Vector(-1,0,0);
  case Patch::xplus:  return Vector( 1,0,0);
  case Patch::yminus: return Vector(0,-1,0);
  case Patch::yplus:  return Vector(0, 1,0);
  case Patch::zminus: return Vector(0,0,-1);
  case Patch::zplus:  return Vector(0,0, 1);
  default:
    return Vector(0,0,0); // oops !
  }
}

AMRMPM::AMRMPM(const ProcessorGroup* myworld) :
  SerialMPM(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags();

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
                          const ProblemSpecP& materials_ps,GridP& grid,
                          SimulationStateP& sharedState)
{
  d_sharedState = sharedState;

  ProblemSpecP restart_mat_ps = 0;
  if (materials_ps){
    restart_mat_ps = materials_ps;
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

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flags->d_8or27==27){
    NGP=2;
    NGN=2;
  }

  ProblemSpecP p = prob_spec->findBlock("DataArchiver");
  if(!p->get("outputInterval", d_outputInterval))
    d_outputInterval = 1.0;


  materialProblemSetup(restart_mat_ps, d_sharedState,flags);
}

void AMRMPM::outputProblemSpec(ProblemSpecP& root_ps)
{
  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP flags_ps = root->appendChild("MPM");
  flags->outputProblemSpec(flags_ps);

  ProblemSpecP mat_ps = 0;
  mat_ps = root->findBlock("MaterialProperties");

  if (mat_ps == 0)
    mat_ps = root->appendChild("MaterialProperties");
    
  ProblemSpecP mpm_ps = mat_ps->appendChild("MPM");
  for (int i = 0; i < d_sharedState->getNumMPMMatls();i++) {
    MPMMaterial* mat = d_sharedState->getMPMMaterial(i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }
}

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
  Task* t = scinew Task("MPM::actuallyInitialize",
                        this, &AMRMPM::actuallyInitialize);

  sched->setPositionVar(lb->pXLabel);
  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->partCountLabel);
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
  t->computes(d_sharedState->get_delt_label());
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

  // artificial damping coeff initialized to 0.0
  if (cout_dbg.active())
    cout_dbg << "Artificial Damping Coeff = " << flags->d_artificialDampCoeff 
             << " 8 or 27 = " << flags->d_8or27 << endl;

  if (flags->d_artificialDampCoeff > 0.0) {
    t->computes(lb->pDampingRateLabel); 
    t->computes(lb->pDampingCoeffLabel); 
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

void AMRMPM::schedulePrintParticleCount(const LevelP& level, 
                                        SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &AMRMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
}

void AMRMPM::scheduleComputeStableTimestep(const LevelP&,
                                              SchedulerP&)
{
  // Nothing to do here - delt is computed as a by-product of the
  // consitutive model
}

void
AMRMPM::scheduleTimeAdvance(const LevelP & inlevel,
                            SchedulerP   & sched)
{
//  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
//    return;
  if(inlevel->getIndex() > 0)
    return;

  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleApplyExternalLoads(             sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleSetBCsInterpolated(             sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeStressTensor(            sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce(           sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleSolveEquationsMotion(           sched, patches, matls);
    scheduleIntegrateAcceleration(          sched, patches, matls);
    scheduleSetGridBoundaryConditions(      sched, patches, matls);
  }
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  }

  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& level = inlevel->getGrid()->getLevel(l);
    sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                      d_sharedState->d_particleState_preReloc,
                                      lb->pXLabel, 
                                      d_sharedState->d_particleState,
                                      lb->pParticleIDLabel, matls);
  }
}

void AMRMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleApplyExternalLoads\t\t\t\t");

  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &AMRMPM::applyExternalLoads);

  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateParticlesToGrid\t");
  

  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&AMRMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pErosionLabel,          gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);

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
  }

  t->computes(lb->gMassLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gVelocityInterpLabel);
  t->computes(lb->gExternalForceLabel);

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleSetBCsInterpolated(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleSetBCsInterpolated\t\t\t");

  Task* t = scinew Task("MPM::setBCsInterpolated",
                        this,&AMRMPM::setBCsInterpolated);

  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gVelocityInterpLabel);

  sched->addTask(t, patches, matls);
}

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
  
  printSchedule(patches,cout_doing,"MPM::scheduleComputeStressTensor\t\t\t\t");
  
  // for thermal stress analysis
  scheduleComputeParticleTempFromGrid(sched, patches, matls); 

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &AMRMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
  }

  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  // Schedule update of the erosion parameter
  scheduleUpdateErosionParameter(sched, patches, matls);

  if (flags->d_accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);

  if(flags->d_artificial_viscosity){
    scheduleComputeArtificialViscosity(   sched, patches, matls);
  }
}

// Compute particle temperature by interpolating grid temperature
// for thermal stress analysis
void AMRMPM::scheduleComputeParticleTempFromGrid(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"MPM::scheduleComputeParticleTempFromGrid");

  Task* t = scinew Task("MPM::computeParticleTempFromGrid",
                        this, &AMRMPM::computeParticleTempFromGrid);
  t->requires(Task::OldDW, lb->pTemperatureLabel, Ghost::None);
  t->computes(lb->pTempCurrentLabel);
  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"MPM::scheduleUpdateErosionParameter\t\t\t");

  Task* t = scinew Task("MPM::updateErosionParameter",
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

void AMRMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalForce\t\t\t\t");
   
  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &AMRMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,      gan,NGP);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
    
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,    gan, NGP);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::NewDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  
  sched->addTask(t, patches, matls);
}


void AMRMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleSolveEquationsMotione\t\t\t");
  
  Task* t = scinew Task("MPM::solveEquationsMotion",
                        this, &AMRMPM::solveEquationsMotion);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleIntegrateAcceleration(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleIntegrateAcceleration\t\t\t");
  
  Task* t = scinew Task("MPM::integrateAcceleration",
                        this, &AMRMPM::integrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->computes(lb->gVelocityStarLabel);

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSetGridBoundaryConditions\t");
  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &AMRMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityInterpLabel,   Ghost::None);

  if(!flags->d_doGridReset){
    t->requires(Task::OldDW, lb->gDisplacementLabel,    Ghost::None);
    t->computes(lb->gDisplacementLabel);
  }

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateToParticlesAndUpdate\t\t\t");
  
  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &AMRMPM::interpolateToParticlesAndUpdate);

  Task::DomainSpec DS = Task::NormalDomain;
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  bool  fat = false;  // possibly (F)rom (A)nother (T)askgraph
  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  if(getLevel(patches)->hasCoarserLevel()){
    t->requires(Task::CoarseNewDW, lb->gAccelerationLabel, 0,
                Task::CoarseLevel, mss, DS, gac, NGN);
    t->requires(Task::CoarseNewDW, lb->gVelocityStarLabel, 0,
                Task::CoarseLevel, mss, DS, gac, NGN);
  }
  if(getLevel(patches)->hasFinerLevel()){
    t->requires(Task::NewDW, lb->gAccelerationLabel,
                 0, Task::FineLevel,  mss, DS, gac, NGN, fat);
    t->requires(Task::NewDW, lb->gVelocityStarLabel,
                 0, Task::FineLevel,  mss, DS, gac, NGN, fat);
  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,            gnone);
  t->requires(Task::NewDW, lb->pErosionLabel_preReloc,          gnone);

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pXXLabel);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }
  
  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"MPM::scheduleRefine\t\t");
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
                                                                                
  if (flags->d_artificialDampCoeff > 0.0) {
    t->computes(lb->pDampingRateLabel);
    t->computes(lb->pDampingCoeffLabel);
  }
                                                                                
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }
                                                                                
  sched->addTask(t, patches, d_sharedState->allMPMMaterials());
}

void AMRMPM::scheduleRefineInterface(const LevelP& /*fineLevel*/, 
                                     SchedulerP& /*scheduler*/,
                                     bool, bool)
{
  // do nothing for now
}

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
  Task* task = scinew Task("errorEstimate", this, &AMRMPM::errorEstimate);
  
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

void AMRMPM::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("switchTest",this, &AMRMPM::switchTest);

  task->requires(Task::OldDW, d_sharedState->get_delt_label() );
  task->computes(d_sharedState->get_switch_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(),d_sharedState->allMaterials());

}

void AMRMPM::printParticleCount(const ProcessorGroup* pg,
                                const PatchSubset*,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  sumlong_vartype pcount;
  new_dw->get(pcount, lb->partCountLabel);
  
  if(pg->myrank() == 0){
    static bool printed=false;
    if(!printed){
      cerr << "Created " << (long) pcount << " total particles\n";
      printed=true;
    }
  }
}

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

      // scalar used for debugging
      if(flags->d_with_color) {
        ParticleVariable<double> pcolor;
        ParticleSubset* pset = new_dw->getParticleSubset(indx, patch);
        setParticleDefault(pcolor, lb->pColorLabel, pset, new_dw, 0.0);
      }

    }
  }

  if (flags->d_accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  // Initialize the artificial damping ceofficient (alpha) to zero
  if (flags->d_artificialDampCoeff > 0.0) {
    double alpha = 0.0;    
    double alphaDot = 0.0;    
    new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}

void AMRMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                           const PatchSubset*,
                                           const MaterialSubset*,
                                           DataWarehouse*,
                                           DataWarehouse*)
{
}

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
    int n8or27=flags->d_8or27;
    Ghost::GhostType  gan = Ghost::AroundNodes;

    IntVector cl, ch, fl, fh, CL, CH, FL, FH;
    // Determine extents for coarser level particle data
    const Level* coarseLevel = 0;
    const Level* fineLevel = 0;
    if(getLevel(patches)->hasCoarserLevel()){
      coarseLevel = getLevel(patches)->getCoarserLevel().get_rep();

      getCoarseLevelRangeNodes(patch, coarseLevel, CL, CH, FL, FH, 1);
    }
    // Determine extents for finer level particle data
    if(getLevel(patches)->hasFinerLevel()){
      fineLevel = getLevel(patches)->getFinerLevel().get_rep();

      patch->computeVariableExtents(Patch::NodeBased, IntVector(0,0,0),
                                    gan, 1, cl, ch);

      fl = patch->getLevel()->mapNodeToFiner(cl);// - ghost;
      fh = patch->getLevel()->mapNodeToFiner(ch);// + ghost;
    }

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      vector<ParticleSubset* > UberPset(3);

      // Create arrays for the grid data
      NCVariable<double> gmass, gvolume;
      NCVariable<Vector> gvelocity,gvelocityInterp,gexternalforce;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gvelocityInterp,  lb->gVelocityInterpLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
                                                                     dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));

      // Create arrays for the particle data on this patch
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      constParticleVariable<double> pErosion;

      ParticleSubset* pset=0;
      for(int whichLevel=0;whichLevel<3;whichLevel++){
        bool doit = false;
        if(getLevel(patches)->hasCoarserLevel() && whichLevel==0){
          pset = old_dw->getParticleSubset(dwi, CL, CH, coarseLevel, NULL,
                                                             lb->pXLabel);
          doit = true;
        }
        if(whichLevel==1){
          pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, lb->pXLabel);
          doit = true;
        }
        if(getLevel(patches)->hasFinerLevel() && whichLevel==2){
          pset = old_dw->getParticleSubset(dwi, fl, fh, fineLevel, NULL,
                                                           lb->pXLabel);
          doit = true;
        }

        if(doit){
          old_dw->get(px,             lb->pXLabel,                 pset);
          old_dw->get(pmass,          lb->pMassLabel,              pset);
          old_dw->get(pvolume,        lb->pVolumeLabel,            pset);
          old_dw->get(pvelocity,      lb->pVelocityLabel,          pset);
          old_dw->get(pTemperature,   lb->pTemperatureLabel,       pset);
          old_dw->get(psize,          lb->pSizeLabel,              pset);
          old_dw->get(pErosion,       lb->pErosionLabel,           pset);
          new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);

          for (ParticleSubset::iterator iter = pset->begin();
               iter != pset->end(); 
               iter++){
            particleIndex idx = *iter;
    
            // Get the node indices that surround the cell
            interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);
    
            Vector pmom = pvelocity[idx]*pmass[idx];
    
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
              }
            }
          } // End of particle loop
        }
      }

      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter; 
        gvelocity[c]    /= gmass[c];
        gvelocityInterp[c]=gvelocity[c];
      }
    }  // End loop over materials
    delete interpolator;
  }  // End loop over patches
}

void AMRMPM::setBCsInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing setBCsInterpolated\t\t\t");

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      NCVariable<Vector> gvelocity,gvelocityInterp;
      new_dw->getModifiable(gvelocity,      lb->gVelocityLabel,      dwi,patch);
      new_dw->getModifiable(gvelocityInterp,lb->gVelocityInterpLabel,dwi,patch);
      int n8or27=flags->d_8or27;

      gvelocityInterp.copyData(gvelocity);

      // Apply grid boundary conditions to the velocity before storing the data
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",   gvelocity,       n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,       n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocityInterp, n8or27);
    }
  }  // End loop over patches
}

void AMRMPM::computeStressTensor(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0),cout_doing,
            "Doing computeSTressTensor\t\t\t");

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){

    if (cout_dbg.active()) {
      cout_dbg << " Patch = " << (patches->get(0))->getID();
      cout_dbg << " Mat = " << m;
    }

    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

    if (cout_dbg.active())
      cout_dbg << " MPM_Mat = " << mpm_matl;

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

    if (cout_dbg.active())
      cout_dbg << " CM = " << cm;

    cm->setWorld(UintahParallelComponent::d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);

    if (cout_dbg.active())
      cout_dbg << " Exit\n" ;

  }
}

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

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel,         pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc,        pset);
      old_dw->get(psize,   lb->pSizeLabel,                   pset);
      new_dw->get(pErosion,lb->pErosionLabel_preReloc,       pset);

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
                                                            psize[idx]);

        stresspress = pstress[idx];

        for (int k = 0; k < n8or27; k++){
          if(patch->containsNode(ni[k])){
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
            div *= pErosion[idx];
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
          }
        }
      }

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,n8or27);
    }

    delete interpolator;
  }
}


void AMRMPM::solveEquationsMotion(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing solveEquationsMotion\t\t\t\t");
    
    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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

      for(NodeIterator iter = patch->getNodeIterator(flags->d_8or27);
                       !iter.done();iter++){
        IntVector c = *iter;
        Vector acc(0.0,0.0,0.0);
        //if (mass[c] > 1.0e-199)
          acc = (internalforce[c] + externalforce[c])/mass[c] ;
          acceleration[c] = acc +  gravity;
      }
    }
  }
}

void AMRMPM::integrateAcceleration(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing integrateAcceleration\t\t\t\t");

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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

      for(NodeIterator iter = patch->getNodeIterator(flags->d_8or27);
                        !iter.done();iter++){
        IntVector c = *iter;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }    // matls
  }
}


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
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocityInterp;
      int n8or27=flags->d_8or27;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocityInterp,         lb->gVelocityInterpLabel,dwi, patch,
                                                                Ghost::None,0);
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",     gvelocity_star, n8or27);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",    gvelocity_star, n8or27);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter = patch->getNodeIterator(n8or27); !iter.done();
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
        for(NodeIterator iter = patch->getNodeIterator(flags->d_8or27);
                         !iter.done();iter++){
           IntVector c = *iter;
           displacement[c] = displacementOld[c] + gvelocity_star[c] * delT;
        }
      }  // d_doGridReset
      // Set symmetry BCs on acceleration if called for
      bc.setBoundaryCondition(patch,dwi,"Symmetric",    gacceleration,  n8or27);

    } // matl loop
  }  // patch loop
}

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

// for thermal stress analysis
void AMRMPM::computeParticleTempFromGrid(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeParticleTempFromGrid\t\t\t");
                                                                                
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
                                                                                
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
                                                                                
      constParticleVariable<double> pTemp;
      old_dw->get(pTemp, lb->pTemperatureLabel, pset);
                                                                                
      ParticleVariable<double> pTempCur;
      new_dw->allocateAndPut(pTempCur,lb->pTempCurrentLabel,pset);
                                                                                
      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++) {
        particleIndex idx = *iter;
        pTempCur[idx]=pTemp[idx];
      } // End of loop over iter
    } // End of loop over m
  } // End of loop over p
}

void AMRMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
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
    vector<Vector> d_S(interpolator->size());

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

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      new_dw->get(pvolume,      lb->pVolumeDeformedLabel,            pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(pErosion,     lb->pErosionLabel_preReloc,          pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);

      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false,dwi,patch, 0);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);

      // Get finer level data
      if(getLevel(patches)->hasFinerLevel()){
         const Level* fineLevel = getLevel(patches)->getFinerLevel().get_rep();
         Level::selectType finePatches;
         patch->getFineLevelPatches(finePatches);
         IntVector one(1,1,1);
         for(int i=0;i<finePatches.size();i++){
            const Patch* finePatch = finePatches[i];

            IntVector cl, ch, fl, fh;
            getFineLevelRangeNodes(patch, finePatch, cl, ch, fl, fh, one);
            constNCVariable<Vector> gv_star_fine, gacc_fine;
            new_dw->getRegion(gv_star_fine,  lb->gVelocityStarLabel, dwi,
                                             fineLevel, fl, fh, false);
            new_dw->getRegion(gacc_fine,     lb->gAccelerationLabel, dwi,
                                             fineLevel, fl, fh, false);
//            for(NodeIterator iter(fl, fh); !iter.done(); iter++){
//              IntVector n = *iter;
//              cout << "F = " << n << " " <<  gv_star_fine[n] << endl;
//            }
         }
      }
      // Get coarser level data
      if(getLevel(patches)->hasCoarserLevel()){
         const Level* coarseLevel = getLevel(patches)->getCoarserLevel().get_rep();
         IntVector cl, ch, fl, fh;
         getCoarseLevelRangeNodes(patch, coarseLevel, cl, ch, fl, fh, 1);
         constNCVariable<Vector> gv_star_coarse, gacc_coarse;
         new_dw->getRegion(gv_star_coarse, lb->gVelocityStarLabel, dwi,
                                           coarseLevel, cl, ch, false);
         new_dw->getRegion(gacc_coarse,    lb->gAccelerationLabel, dwi,
                                           coarseLevel, cl, ch, false);
//         for(NodeIterator iter(cl, ch); !iter.done(); iter++){
//           IntVector n = *iter;
//           cout << "C = " << n << " " <<  gv_star_coarse[n] << endl;
//         }
      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          S[k] *= pErosion[idx];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];
        }

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
      }

      new_dw->deleteParticles(delset);      

      #if 0
      //__________________________________
      //  particle debugging label-- carry forward
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }
      #endif
    }
    delete interpolator;
  }
}

void 
AMRMPM::setParticleDefault(ParticleVariable<double>& pvar,
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
AMRMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
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

void AMRMPM::scheduleCheckNeedAddMPMMaterial(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"MPM::scheduleCheckNeedAddMPMMateria\t\t");

  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("MPM::checkNeedAddMPMMaterial",
                        this, &AMRMPM::checkNeedAddMPMMaterial);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->scheduleCheckNeedAddMPMMaterial(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

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

void AMRMPM::scheduleSetNeedAddMaterialFlag(SchedulerP& sched,
                                            const LevelP& level,
                                            const MaterialSet* all_matls)
{
  printSchedule(level,cout_doing,"MPM::scheduleSetNeedAddMaterialFlag\t\t");

  Task* t= scinew Task("AMRMPM::setNeedAddMaterialFlag",
               this, &AMRMPM::setNeedAddMaterialFlag);
  t->requires(Task::NewDW, lb->NeedAddMPMMaterialLabel);
  sched->addTask(t, level->eachPatch(), all_matls);
}

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
      cout << "MPM setting NAM to -99" << endl;
    }
    else{
      d_sharedState->setNeedAddMaterial(0);
    }
}

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
