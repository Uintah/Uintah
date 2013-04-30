/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/Contact/ContactFactory.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/CubicPolyRoots.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <fstream>
#include <sstream>

//#define GE_Proj
#undef GE_Proj

using namespace Uintah;

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

SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  MPMCommon(myworld), UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags(myworld);

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  contactModel        = 0;
  thermalContactModel = 0;
  heatConductionModel = 0;
  NGP     = 1;
  NGN     = 1;
  d_recompile = false;
  dataArchiver = 0;
  d_loadCurveIndex=0;
  d_switchCriteria = 0;
}

SerialMPM::~SerialMPM()
{
  delete lb;
  delete flags;
  delete contactModel;
  delete thermalContactModel;
  delete heatConductionModel;
  MPMPhysicalBCFactory::clean();
  
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      delete *iter;
    }
  }
  
  if(d_switchCriteria) {
    delete d_switchCriteria;
  }
  
  
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, 
                             const ProblemSpecP& restart_prob_spec,GridP& grid,
                             SimulationStateP& sharedState)
{
  cout_doing<<"Doing problemSetup\t\t\t\t\t MPM"<<endl;
  d_sharedState = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);
  
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!dataArchiver){
    throw InternalError("MPM:couldn't get output port", __FILE__, __LINE__);
  }

  ProblemSpecP restart_mat_ps = 0;
  ProblemSpecP prob_spec_mat_ps = 
    prob_spec->findBlockWithOutAttribute("MaterialProperties");

  if (prob_spec_mat_ps)
    restart_mat_ps = prob_spec;
  else if (restart_prob_spec)
    restart_mat_ps = restart_prob_spec;
  else
    restart_mat_ps = prob_spec;

  ProblemSpecP mpm_soln_ps = restart_mat_ps->findBlock("MPM");
  if (!mpm_soln_ps){
    ostringstream warn;
    warn<<"ERROR:MPM:\n missing MPM section in the input file\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
 
  // Read all MPM flags (look in MPMFlags.cc)
  flags->readMPMFlags(restart_mat_ps, dataArchiver);
  if (flags->d_integrator_type == "implicit"){
    throw ProblemSetupException("Can't use implicit integration with -mpm",
                                 __FILE__, __LINE__);
  }

  // convert text representation of face into FaceType
  for(std::vector<std::string>::const_iterator ftit(flags->d_bndy_face_txt_list.begin());
      ftit!=flags->d_bndy_face_txt_list.end();ftit++) {
      Patch::FaceType face = Patch::invalidFace;
      for(Patch::FaceType ft=Patch::startFace;ft<=Patch::endFace;
          ft=Patch::nextFace(ft)) {
        if(Patch::getFaceName(ft)==*ftit) face =  ft;
      }
      if(face!=Patch::invalidFace) {
        d_bndy_traction_faces.push_back(face);
      } else {
        std::cerr << "warning: ignoring unknown face '" << *ftit<< "'" << std::endl;
      }
  }

  // read in AMR flags from the main ups file
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");
  if (amr_ps) {
    ProblemSpecP mpm_amr_ps = amr_ps->findBlock("MPM");
    if(!mpm_amr_ps){
      ostringstream warn;
      warn<<"ERROR:MPM:\n missing MPM section in the AMR section of the input file\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    mpm_amr_ps->getWithDefault("min_grid_level", flags->d_minGridLevel, 0);
    mpm_amr_ps->getWithDefault("max_grid_level", flags->d_maxGridLevel, 1000);
  }

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  if (flags->d_prescribeDeformation){
    readPrescribedDeformations(flags->d_prescribedDeformationFile);
  }
  if (flags->d_insertParticles){
    readInsertParticlesFile(flags->d_insertParticlesFile);
  }

  d_sharedState->setParticleGhostLayer(Ghost::AroundNodes, NGP);

  MPMPhysicalBCFactory::create(restart_mat_ps, grid, flags);

  contactModel = ContactFactory::create(UintahParallelComponent::d_myworld, restart_mat_ps,sharedState,lb,flags);
  thermalContactModel =
    ThermalContactFactory::create(restart_mat_ps, sharedState, lb,flags);

  heatConductionModel = scinew HeatConduction(sharedState,lb,flags);

  materialProblemSetup(restart_mat_ps, d_sharedState,flags);

  cohesiveZoneProblemSetup(restart_mat_ps, d_sharedState,flags);
  
  //__________________________________
  //  create analysis modules
  // call problemSetup  
  if(!flags->d_with_ice && !flags->d_with_arches){    // mpmice or mpmarches handles this
    d_analysisModules = AnalysisModuleFactory::create(prob_spec, sharedState, dataArchiver);
    
    if(d_analysisModules.size() != 0){
      vector<AnalysisModule*>::iterator iter;
      for( iter  = d_analysisModules.begin();
           iter != d_analysisModules.end(); iter++){
        AnalysisModule* am = *iter;
        am->problemSetup(prob_spec,restart_prob_spec, grid, sharedState);
      }
    }
  }
  
  //__________________________________
  //  create the switching criteria port
  d_switchCriteria = dynamic_cast<SwitchingCriteria*>(getPort("switch_criteria"));
   
  if (d_switchCriteria) {
     d_switchCriteria->problemSetup(restart_mat_ps,restart_prob_spec,d_sharedState);
  }
}

void SerialMPM::outputProblemSpec(ProblemSpecP& root_ps)
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

  contactModel->outputProblemSpec(mpm_ps);
  thermalContactModel->outputProblemSpec(mpm_ps);

  for (int i = 0; i < d_sharedState->getNumCZMatls();i++) {
    CZMaterial* mat = d_sharedState->getCZMaterial(i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }
  
  ProblemSpecP physical_bc_ps = root->appendChild("PhysicalBC");
  ProblemSpecP mpm_ph_bc_ps = physical_bc_ps->appendChild("MPM");
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
    MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->outputProblemSpec(mpm_ph_bc_ps);
  }

}

void SerialMPM::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;
  Task* t = scinew Task("MPM::actuallyInitialize",
                        this, &SerialMPM::actuallyInitialize);

  const PatchSet* patches = level->eachPatch();
  printSchedule(patches,cout_doing,"MPM::scheduleInitialize");
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
  t->computes(lb->pTempPreviousLabel); // for therma  stresm analysis
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pSizeLabel);
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);
  t->computes(lb->NC_CCweightLabel,zeroth_matl);

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

  if (flags->d_reductionVars->accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  if(flags->d_artificial_viscosity){
    t->computes(lb->p_qLabel);
  }

  // artificial damping coeff initialized to 0.0
  if (cout_dbg.active())
    cout_dbg << "Artificial Damping Coeff = " << flags->d_artificialDampCoeff 
             << " 8 or 27 = " << flags->d_8or27 << endl;

  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, d_sharedState->allMPMMaterials());

  schedulePrintParticleCount(level, sched);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  if (flags->d_useLoadCurves) {
    // Schedule the initialization of pressure BCs per particle
    scheduleInitializePressureBCs(level, sched);
  }

  // dataAnalysis 
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }
  
  int numCZM = d_sharedState->getNumCZMatls();
  for(int m = 0; m < numCZM; m++){
    CZMaterial* cz_matl = d_sharedState->getCZMaterial(m);
    CohesiveZone* ch = cz_matl->getCohesiveZone();
    ch->scheduleInitialize(level, sched, cz_matl);
  }

}

/* _____________________________________________________________________
 Purpose:   Set variables that are normally set during the initialization
            phase, but get wiped clean when you restart
_____________________________________________________________________*/
void SerialMPM::restartInitialize()
{
  cout_doing<<"Doing restartInitialize\t\t\t\t\t MPM"<<endl;

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->restartInitialize();
    }
  }  
}

//______________________________________________________________________
void SerialMPM::schedulePrintParticleCount(const LevelP& level, 
                                           SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &SerialMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level),
                 d_sharedState->allMPMMaterials());
}
//__________________________________
//  Diagnostic task: compute the total number of particles
void SerialMPM::scheduleTotalParticleCount(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())){
    return;
  }

  Task* t = scinew Task("SerialMPM::totalParticleCount",
                  this, &SerialMPM::totalParticleCount);
  t->computes(lb->partCountLabel);
  
  sched->addTask(t, patches,matls);
}
//__________________________________
//  Diagnostic task: compute the total number of particles
void SerialMPM::totalParticleCount(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    long int totalParticles = 0;
    
    for(int m=0;m<matls->size();m++){  
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      int numParticles  = pset->end() - pset->begin();
      
      totalParticles+=numParticles;
    }
    new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
  }
}


void SerialMPM::scheduleInitializePressureBCs(const LevelP& level,
                                              SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  
  
  d_loadCurveIndex = scinew MaterialSubset();
  d_loadCurveIndex->add(0);
  d_loadCurveIndex->addReference();

  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure"){
      d_loadCurveIndex->add(nofPressureBCs++);
    }
  }
  if (nofPressureBCs > 0) {
    printSchedule(patches,cout_doing,"MPM::countMaterialPointsPerLoadCurve");
    printSchedule(patches,cout_doing,"MPM::scheduleInitializePressureBCs");
    // Create a task that calculates the total number of particles
    // associated with each load curve.  
    Task* t = scinew Task("MPM::countMaterialPointsPerLoadCurve",
                          this, &SerialMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, d_loadCurveIndex,
                Task::OutOfDomain);
    sched->addTask(t, patches, d_sharedState->allMPMMaterials());

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializePressureBC",
                    this, &SerialMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel,                        Ghost::None);
    t->requires(Task::NewDW, lb->pSizeLabel,                     Ghost::None);
    t->requires(Task::NewDW, lb->pDeformationMeasureLabel,       Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel,              Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel,
                            d_loadCurveIndex, Task::OutOfDomain, Ghost::None);
    t->modifies(lb->pExternalForceLabel);
    if (flags->d_useCBDI) {
       t->computes(             lb->pExternalForceCorner1Label);
       t->computes(             lb->pExternalForceCorner2Label);
       t->computes(             lb->pExternalForceCorner3Label);
       t->computes(             lb->pExternalForceCorner4Label);
    }
    sched->addTask(t, patches, d_sharedState->allMPMMaterials());
  }

  if(d_loadCurveIndex->removeReference())
    delete d_loadCurveIndex;
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP& level,
                                              SchedulerP& sched)
{
  // Nothing to do here - delt is computed as a by-product of the
  // constitutive model
  // However, this task needs to do something in the case that MPM
  // is being run on more than one level.
  Task* t = 0;
  cout_doing << UintahParallelComponent::d_myworld->myrank() << " MPM::scheduleComputeStableTimestep \t\t\t\tL-" <<level->getIndex() << endl;

  t = scinew Task("MPM::actuallyComputeStableTimestep",
                   this, &SerialMPM::actuallyComputeStableTimestep);

  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();

  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  sched->addTask(t,level->eachPatch(), mpm_matls);
}

void
SerialMPM::scheduleTimeAdvance(const LevelP & level,
                               SchedulerP   & sched)
{
  MALLOC_TRACE_TAG_SCOPE("SerialMPM::scheduleTimeAdvance()");
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  const MaterialSet* cz_matls = d_sharedState->allCZMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();

  const MaterialSubset* mpm_matls_sub = matls->getUnion();
  const MaterialSubset* cz_matls_sub  = cz_matls->getUnion();

  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleExMomInterpolated(              sched, patches, matls);
  if(flags->d_useCohesiveZones){
    scheduleUpdateCohesiveZones(          sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
    scheduleAddCohesiveZoneForces(        sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
  }
  scheduleComputeContactArea(             sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);

  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleSetPrescribedMotion(            sched, patches, matls);
  if(flags->d_doExplicitHeatConduction){
    scheduleComputeHeatExchange(          sched, patches, matls);
    scheduleComputeInternalHeatRate(      sched, patches, matls);
    scheduleComputeNodalHeatFlux(         sched, patches, matls);
    scheduleSolveHeatEquations(           sched, patches, matls);
    scheduleIntegrateTemperatureRate(     sched, patches, matls);
  }
  if(!flags->d_use_momentum_form){
    scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
    scheduleComputeStressTensor(            sched, patches, matls);
    scheduleFinalParticleUpdate(            sched, patches, matls);
  }
  if(flags->d_use_momentum_form){
    scheduleInterpolateToParticlesAndUpdateMom1(sched, patches, matls);
    scheduleInterpolateParticleVelToGridMom(    sched, patches, matls);
    scheduleExMomIntegrated(                    sched, patches, matls);
    scheduleSetGridBoundaryConditions(          sched, patches, matls);
    scheduleComputeStressTensor(                sched, patches, matls);
    scheduleInterpolateToParticlesAndUpdateMom2(sched, patches, matls);
  }
  scheduleInsertParticles(                    sched, patches, matls);
  if(flags->d_computeScaleFactor){
    scheduleComputeParticleScaleFactor(       sched, patches, matls);
  }

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis_preReloc( sched, level);
    }
  }

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_sharedState->d_particleState_preReloc,
                                    lb->pXLabel, 
                                    d_sharedState->d_particleState,
                                    lb->pParticleIDLabel, matls, 1);

 if(flags->d_useCohesiveZones){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_sharedState->d_cohesiveZoneState_preReloc,
                                    lb->pXLabel, 
                                    d_sharedState->d_cohesiveZoneState,
                                    lb->czIDLabel, cz_matls,2);
  }

  //__________________________________
  //  on the fly analysis
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis( sched, level);
    }
  }
}

void SerialMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"MPM::scheduleApplyExternalLoads");
 /*
  * applyExternalLoads
  *   in(p.externalForce, p.externalheatrate)
  *   out(p.externalForceNew, p.externalheatrateNew) */
  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &SerialMPM::applyExternalLoads);
                  
  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel,              Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::OldDW, lb->pDispLabel,              Ghost::None);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,     Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);
  if (flags->d_useLoadCurves) {
    t->requires(Task::OldDW, lb->pLoadCurveIDLabel,     Ghost::None);
    t->computes(             lb->pLoadCurveIDLabel_preReloc);
    if (flags->d_useCBDI) {
       t->computes(             lb->pExternalForceCorner1Label);
       t->computes(             lb->pExternalForceCorner2Label);
       t->computes(             lb->pExternalForceCorner3Label);
       t->computes(             lb->pExternalForceCorner4Label);
    }
  }
//  t->computes(Task::OldDW, lb->pExternalHeatRateLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateParticlesToGrid");
  
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&SerialMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
#ifdef GE_Proj
  t->requires(Task::OldDW, lb->pVelGradLabel,          gan,NGP);
#endif
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,gan,NGP);
  if (flags->d_useCBDI) {
    t->requires(Task::NewDW,  lb->pExternalForceCorner1Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner2Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner3Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner4Label,gan,NGP);
    t->requires(Task::OldDW,  lb->pLoadCurveIDLabel,gan,NGP);
  }

  //t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gVolumeLabel,      d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gVelocityLabel,    d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);
  t->computes(lb->gSp_volLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gTemperatureRateLabel);
  t->computes(lb->gExternalHeatRateLabel);

  if(flags->d_with_ice){
    t->computes(lb->gVelocityBCLabel);
  }
  
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleAddCohesiveZoneForces(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSubset* mpm_matls,
                                              const MaterialSubset* cz_matls,
                                              const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleAddCohesiveZoneForces");

  Task* t = scinew Task("MPM::addCohesiveZoneForces",
                        this,&SerialMPM::addCohesiveZoneForces);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pXLabel,                     cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czLengthLabel_preReloc,      cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czForceLabel_preReloc,       cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czTopMatLabel_preReloc,      cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czBotMatLabel_preReloc,      cz_matls, gan,NGP);

  t->modifies(lb->gExternalForceLabel, mpm_matls);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeHeatExchange(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  /* computeHeatExchange
   *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
   *   operation(peform heat exchange which will cause each of
   *   velocity fields to exchange heat according to 
   *   the temperature differences)
   *   out(G.EXTERNAL_HEAT_RATE) */

  printSchedule(patches,cout_doing,"MPM::scheduleComputeHeatExchange");

  Task* t = scinew Task("ThermalContact::computeHeatExchange",
                        thermalContactModel,
                        &ThermalContact::computeHeatExchange);

  thermalContactModel->addComputesAndRequires(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleExMomInterpolated(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleExMomInterpolated");
  
  contactModel->addComputesAndRequiresInterpolated(sched, patches, matls);
}

/////////////////////////////////////////////////////////////////////////
/*!  **WARNING** In addition to the stresses and deformations, the internal 
 *               heat rate in the particles (pdTdtLabel) 
 *               is computed here */
/////////////////////////////////////////////////////////////////////////
void SerialMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  
  printSchedule(patches,cout_doing,"MPM::scheduleComputeStressTensor");
  
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &SerialMPM::computeStressTensor);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();
    t->computes(lb->p_qLabel_preReloc, matlset);
  }

  t->computes(d_sharedState->get_delt_label(),getLevel(patches));
  
  if (flags->d_reductionVars->accStrainEnergy ||
      flags->d_reductionVars->strainEnergy) {
    t->computes(lb->StrainEnergyLabel);
  }
  
  sched->addTask(t, patches, matls);

  // Schedule update of the erosion parameter
  scheduleUpdateErosionParameter(sched, patches, matls);
  scheduleFindRogueParticles(sched, patches, matls);

  if (flags->d_reductionVars->accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);

}


void SerialMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"MPM::scheduleUpdateErosionParameter");

  Task* t = scinew Task("MPM::updateErosionParameter",
                        this, &SerialMPM::updateErosionParameter);
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }
  t->computes(lb->pLocalizedMPMLabel);

  if(flags->d_deleteRogueParticles){
    t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
    t->computes(lb->numLocInCellLabel);
    t->computes(lb->numInCellLabel);
  }

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleFindRogueParticles(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  if(flags->d_deleteRogueParticles) {
    printSchedule(patches,cout_doing,"MPM::scheduleFindRogueParticles");

    Task* t = scinew Task("MPM::findRogueParticles",
                          this, &SerialMPM::findRogueParticles);
    Ghost::GhostType gac   = Ghost::AroundCells;
    t->requires(Task::NewDW, lb->numLocInCellLabel,       gac, 1);
    t->requires(Task::NewDW, lb->numInCellLabel,          gac, 1);
    t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
    t->modifies(lb->pLocalizedMPMLabel);

    sched->addTask(t, patches, matls);
  }
}

// Compute the accumulated strain energy
void SerialMPM::scheduleComputeAccStrainEnergy(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeAccStrainEnergy");

  Task* t = scinew Task("MPM::computeAccStrainEnergy",
                        this, &SerialMPM::computeAccStrainEnergy);
  t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
  t->requires(Task::NewDW, lb->StrainEnergyLabel);
  t->computes(lb->AccStrainEnergyLabel);
  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeContactArea(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  /** computeContactArea */
  if(d_bndy_traction_faces.size()>0) {
  
    printSchedule(patches,cout_doing,"MPM::scheduleComputeContactArea");
    Task* t = scinew Task("MPM::computeContactArea",
                          this, &SerialMPM::computeContactArea);
    
    Ghost::GhostType  gnone = Ghost::None;
    t->requires(Task::NewDW, lb->gVolumeLabel, gnone);
    for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
      int iface = (int)(*ftit);
      t->computes(lb->BndyContactCellAreaLabel[iface]);
    }
    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  /*
   * computeInternalForce
   *   in(P.CONMOD, P.NAT_X, P.VOLUME)
   *   operation(evaluate the divergence of the stress (stored in
   *   P.CONMOD) using P.NAT_X and the gradients of the
   *   shape functions)
   * out(G.F_INTERNAL) */

  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalForce");
   
  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &SerialMPM::computeInternalForce);

 
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,  gan,NGP);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  if(flags->d_artificial_viscosity){
    t->requires(Task::OldDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);
  
  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    t->requires(Task::NewDW, lb->BndyContactCellAreaLabel[iface]);
    t->computes(lb->BndyForceLabel[iface]);
    t->computes(lb->BndyContactAreaLabel[iface]);
    t->computes(lb->BndyTractionLabel[iface]);
  }
  
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}


void SerialMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{  
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalHeatRate");
  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}
void SerialMPM::scheduleComputeNodalHeatFlux(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{  
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeNodalHeatFlux");
  heatConductionModel->scheduleComputeNodalHeatFlux(sched,patches,matls);
}

void SerialMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSolveHeatEquations");
  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void SerialMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeAndIntegrateAcceleration");
  
  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &SerialMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleIntegrateTemperatureRate");
  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

void SerialMPM::scheduleExMomIntegrated(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  /* exMomIntegrated
   *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
   *   operation(peform operations which will cause each of
   *              velocity fields to feel the influence of the
   *              the others according to specific rules)
   *   out(G.VELOCITY_STAR, G.ACCELERATION) */
  printSchedule(patches,cout_doing,"MPM::scheduleExMomIntegrated");
  contactModel->addComputesAndRequiresIntegrated(sched, patches, matls);
}

void SerialMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSetGridBoundaryConditions");
  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &SerialMPM::setGridBoundaryConditions);
                  
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

void SerialMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  /*
   * interpolateToParticlesAndUpdate
   *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
   *   operation(interpolate acceleration and v* to particles and
   *   integrate these to get new particle velocity and position)
   * out(P.VELOCITY, P.X, P.NAT_X) */

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateToParticlesAndUpdate");
  
  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &SerialMPM::interpolateToParticlesAndUpdate);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

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
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress 
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pXXLabel);

  //__________________________________
  //  reduction variables
  if(flags->d_reductionVars->momentum){
    t->computes(lb->TotalMomentumLabel);
  }
  if(flags->d_reductionVars->KE){
    t->computes(lb->KineticEnergyLabel);
  }
  if(flags->d_reductionVars->thermalEnergy){
    t->computes(lb->ThermalEnergyLabel);
  }
  if(flags->d_reductionVars->centerOfMass){
    t->computes(lb->CenterOfMassPositionLabel);
  }
  if(flags->d_reductionVars->mass){
    t->computes(lb->TotalMassLabel);
  }
  if(flags->d_reductionVars->volDeformed){
    t->computes(lb->TotalVolumeDeformedLabel);
  }

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  t->requires(Task::OldDW, lb->NC_CCweightLabel, z_matl, Ghost::None);
  t->computes(             lb->NC_CCweightLabel, z_matl);

  sched->addTask(t, patches, matls);

  // The task will have a reference to z_matl
  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

void SerialMPM::scheduleFinalParticleUpdate(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleFinalParticleUpdate");
  
  Task* t=scinew Task("MPM::finalParticleUpdate",
                      this, &SerialMPM::finalParticleUpdate);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->pdTdtLabel_preReloc,             gnone);
  t->requires(Task::NewDW, lb->pLocalizedMPMLabel,              gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdateMom1(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateToParticlesAndUpdateMom1");

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdateMom1",
                      this, &SerialMPM::interpolateToParticlesAndUpdateMom1);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

 
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);

  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);

  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pXXLabel);
  t->computes(lb->pDispLabel_preReloc);

  //__________________________________
  //  reduction variables
  if(flags->d_reductionVars->momentum){
    t->computes(lb->TotalMomentumLabel);
  }
  if(flags->d_reductionVars->KE){
    t->computes(lb->KineticEnergyLabel);
  }
  if(flags->d_reductionVars->centerOfMass){
    t->computes(lb->CenterOfMassPositionLabel);
  }

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdateMom2(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateToParticlesAndUpdate2");

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdateMom2",
                      this, &SerialMPM::interpolateToParticlesAndUpdateMom2);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,gnone);
  t->requires(Task::NewDW, lb->pdTdtLabel_preReloc,             gnone);
  t->requires(Task::NewDW, lb->pLocalizedMPMLabel,              gnone);

  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->modifies(lb->pVolumeLabel_preReloc);

  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress 
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);

  //__________________________________
  //  reduction variables
  if(flags->d_reductionVars->thermalEnergy){
    t->computes(lb->ThermalEnergyLabel);
  }
  if(flags->d_reductionVars->mass){
    t->computes(lb->TotalMassLabel);
  }
  if(flags->d_reductionVars->volDeformed){
    t->computes(lb->TotalVolumeDeformedLabel);
  }

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  t->requires(Task::OldDW, lb->NC_CCweightLabel, z_matl, Ghost::None);
  t->computes(             lb->NC_CCweightLabel, z_matl);

  sched->addTask(t, patches, matls);

  // The task will have a reference to z_matl
  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...

}

void SerialMPM::scheduleUpdateCohesiveZones(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* cz_matls,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleUpdateCohesiveZones");

  Task* t=scinew Task("MPM::updateCohesiveZones",
                      this, &SerialMPM::updateCohesiveZones);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityLabel,     mpm_matls,   gac,NGN);
  t->requires(Task::NewDW, lb->gMassLabel,         mpm_matls,   gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,            cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czLengthLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czNormLabel,        cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czTangLabel,        cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czDispTopLabel,     cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czDispBottomLabel,  cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czSeparationLabel,  cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czForceLabel,       cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czTopMatLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czBotMatLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czFailedLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czIDLabel,          cz_matls,    gnone);

  t->computes(lb->pXLabel_preReloc,           cz_matls);
  t->computes(lb->czLengthLabel_preReloc,     cz_matls);
  t->computes(lb->czNormLabel_preReloc,       cz_matls);
  t->computes(lb->czTangLabel_preReloc,       cz_matls);
  t->computes(lb->czDispTopLabel_preReloc,    cz_matls);
  t->computes(lb->czDispBottomLabel_preReloc, cz_matls);
  t->computes(lb->czSeparationLabel_preReloc, cz_matls);
  t->computes(lb->czForceLabel_preReloc,      cz_matls);
  t->computes(lb->czTopMatLabel_preReloc,     cz_matls);
  t->computes(lb->czBotMatLabel_preReloc,     cz_matls);
  t->computes(lb->czFailedLabel_preReloc,     cz_matls);
  t->computes(lb->czIDLabel_preReloc,         cz_matls);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInsertParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  if(flags->d_insertParticles){
    printSchedule(patches,cout_doing,"MPM::scheduleInsertParticles");

    Task* t=scinew Task("MPM::insertParticles",this,
                  &SerialMPM::insertParticles);

    t->requires(Task::OldDW, d_sharedState->get_delt_label() );

    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);

    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleComputeParticleScaleFactor(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeParticleScaleFactor");

  Task* t=scinew Task("MPM::computeParticleScaleFactor",this,
                &SerialMPM::computeParticleScaleFactor);

  t->requires(Task::OldDW, lb->pSizeLabel,  Ghost::None);
  t->computes(lb->pScaleFactorLabel_preReloc);

  sched->addTask(t, patches, matls);
}




void SerialMPM::scheduleInterpolateParticleVelToGridMom(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateParticleVelToGridMom");

  Task* t = scinew Task("MPM::interpolateParticleVelToGridMom",
                        this,&SerialMPM::interpolateParticleVelToGridMom);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,              gan,NGP);
  t->requires(Task::NewDW, lb->pVelocityLabel_preReloc, gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                 gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,              gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,gan,NGP);

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->modifies(lb->gVelocityStarLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleSetPrescribedMotion(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  if (flags->d_prescribeDeformation){
    printSchedule(patches,cout_doing,"MPM::scheduleSetPrescribedMotion");
  
    Task* t=scinew Task("MPM::setPrescribedMotion",
                      this, &SerialMPM::setPrescribedMotion);

    const MaterialSubset* mss = matls->getUnion();
    t->modifies(             lb->gAccelerationLabel,     mss);
    t->modifies(             lb->gVelocityStarLabel,     mss);
    t->requires(Task::OldDW, d_sharedState->get_delt_label() );
    if(!flags->d_doGridReset){
      t->requires(Task::OldDW, lb->gDisplacementLabel,    Ghost::None);
      t->modifies(lb->gDisplacementLabel, mss);
    }

    sched->addTask(t, patches, matls);
   }
}

void SerialMPM::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"MPM::scheduleRefine");
  Task* t = scinew Task("SerialMPM::refine", this, &SerialMPM::refine);

  t->computes(lb->pXLabel);
  t->computes(lb->p_qLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pTempPreviousLabel); // for therma  stresm analysis
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->NC_CCweightLabel);
  t->computes(d_sharedState->get_delt_label(),getLevel(patches));

  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
  }
                                                                                
  if (flags->d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }
                                                                                
  if (flags->d_reductionVars->accStrainEnergy) {
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

void SerialMPM::scheduleRefineInterface(const LevelP& /*fineLevel*/, 
                                        SchedulerP& /*scheduler*/,
                                        bool, bool)
{
  //  do nothing for now
}

void SerialMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/, 
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}
//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void SerialMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the 
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (amr_doing.active())
    amr_doing << "SerialMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // The simulation controller should not schedule it every time step
  Task* task = scinew Task("errorEstimate", this, &SerialMPM::errorEstimate);
  
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
void SerialMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void SerialMPM::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  if (d_switchCriteria) {
    d_switchCriteria->scheduleSwitchTest(level,sched);
  }
}

void SerialMPM::printParticleCount(const ProcessorGroup* pg,
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

void SerialMPM::computeAccStrainEnergy(const ProcessorGroup*,
                                       const PatchSubset*,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  // Get the totalStrainEnergy from the old datawarehouse
  max_vartype accStrainEnergy;
  old_dw->get(accStrainEnergy, lb->AccStrainEnergyLabel);

  // Get the incremental strain energy from the new datawarehouse
  sum_vartype incStrainEnergy;
  new_dw->get(incStrainEnergy, lb->StrainEnergyLabel);
  
  // Add the two a put into new dw
  double totalStrainEnergy = 
    (double) accStrainEnergy + (double) incStrainEnergy;
  new_dw->put(max_vartype(totalStrainEnergy), lb->AccStrainEnergyLabel);
}

// Calculate the number of material points per load curve
void SerialMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* ,
                                                DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0) ,cout_doing,"countMaterialPointsPerLoadCurve");
  // Find the number of pressure BCs in the problem
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {
      nofPressureBCs++;

      // Loop through the patches and count
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        int numMPMMatls=d_sharedState->getNumMPMMatls();
        int numPts = 0;
        for(int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<int> pLoadCurveID;
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == (nofPressureBCs)) ++numPts;
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts), 
                    lb->materialPointsPerLoadCurveLabel, 0, nofPressureBCs-1);
      }  // patch loop
    }
  }
}

// Calculate the number of material points per load curve
void SerialMPM::initializePressureBC(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* ,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
  printTask(patches, patches->get(0),cout_doing,"Doing initializePressureBC");
  if (cout_dbg.active())
    cout_dbg << "Current Time (Initialize Pressure BC) = " << time << endl;


  // Calculate the force vector at each particle
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      new_dw->get(px, lb->pXLabel, pset);
      new_dw->get(psize, lb->pSizeLabel, pset);
      new_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);
      constParticleVariable<int> pLoadCurveID;
      new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
      ParticleVariable<Vector> pExternalForce;
      new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);

      ParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                              pExternalForceCorner3, pExternalForceCorner4;
      if (flags->d_useCBDI) {
        new_dw->allocateAndPut(pExternalForceCorner1,
                               lb->pExternalForceCorner1Label, pset);
        new_dw->allocateAndPut(pExternalForceCorner2,
                               lb->pExternalForceCorner2Label, pset);
        new_dw->allocateAndPut(pExternalForceCorner3,
                               lb->pExternalForceCorner3Label, pset);
        new_dw->allocateAndPut(pExternalForceCorner4,
                               lb->pExternalForceCorner4Label, pset);
      }
      int nofPressureBCs = 0;
      for(int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
        string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        if (bcs_type == "Pressure") {

          // Get the material points per load curve
          sumlong_vartype numPart = 0;
          new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel,
                      0, nofPressureBCs++);

          // Save the material points per load curve in the PressureBC object
          PressureBC* pbc =
            dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
          pbc->numMaterialPoints(numPart);

          if (cout_dbg.active())
          cout_dbg << "    Load Curve = "
                   << nofPressureBCs << " Num Particles = " << numPart << endl;

          // Calculate the force per particle at t = 0.0
          double forcePerPart = pbc->forcePerParticle(time);

          // Loop through the patches and calculate the force vector
          // at each particle

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == nofPressureBCs) {
              if (flags->d_useCBDI) {
               Vector dxCell = patch->dCell();
               pExternalForce[idx] = pbc->getForceVectorCBDI(px[idx],psize[idx],
                                    pDeformationMeasure[idx],forcePerPart,time,
                                    pExternalForceCorner1[idx],
                                    pExternalForceCorner2[idx],
                                    pExternalForceCorner3[idx],
                                    pExternalForceCorner4[idx],
                                    dxCell);
              } else {
               pExternalForce[idx] = pbc->getForceVector(px[idx],
                                                        forcePerPart,time);
              }// if CBDI
            } // if pLoadCurveID...
          }  // loop over particles
        }   // if pressure loop
      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

void SerialMPM::actuallyInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch,cout_doing,"Doing actuallyInitialize");

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

    NCVariable<double> NC_CCweight;
    new_dw->allocateAndPut(NC_CCweight, lb->NC_CCweightLabel,    0, patch);

    //__________________________________
    // - Initialize NC_CCweight = 0.125
    // - Find the walls with symmetry BC and double NC_CCweight
    NC_CCweight.initialize(0.125);
    for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
      int mat_id = 0;

      if (patch->haveBC(face,mat_id,"symmetry","Symmetric")) {
        for(CellIterator iter = patch->getFaceIterator(face,Patch::FaceNodes);
                                                  !iter.done(); iter++) {
          NC_CCweight[*iter] = 2.0*NC_CCweight[*iter];
        }
      }
    }

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
    IntVector num_extra_cells=patch->getExtraCells();
    IntVector periodic=patch->getLevel()->getPeriodicBoundaries();
    string interp_type = flags->d_interpolator_type;
    if(interp_type=="linear" && num_extra_cells!=IntVector(0,0,0)){
      if(!flags->d_with_ice && !flags->d_with_arches){
        ostringstream msg;
        msg << "\n ERROR: When using <interpolator>linear</interpolator> \n"
            << " you should also use <extraCells>[0,0,0]</extraCells> \n"
            << " unless you are running an MPMICE or MPMARCHES case.\n";
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }
    else if(((interp_type=="gimp" || interp_type=="3rdorderBS" 
          || interp_type=="cpdi")
                          && ((num_extra_cells+periodic)!=IntVector(1,1,1)
                          && ((num_extra_cells+periodic)!=IntVector(1,1,0) 
                          && flags->d_axisymmetric)))){
        ostringstream msg;
        msg << "\n ERROR: When using <interpolator>gimp</interpolator> \n"
            << " or <interpolator>3rdorderBS</interpolator> \n"
            << " or <interpolator>cpdi</interpolator> \n"
            << " you must also use extraCells and/or periodicBCs such\n"
            << " the sum of the two is [1,1,1].\n"
            << " If using axisymmetry, the sum of the two can be [1,1,0].\n";
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }

    // Only allow axisymmetric runs if the grid is one cell thick in the theta dir.
    if(flags->d_axisymmetric){
      IntVector patchLowNode = patch->getNodeLowIndex();
      IntVector patchHighNode = patch->getNodeHighIndex();
      int num_cells_in_theta = (patchHighNode.z() - patchLowNode.z()) - 1;
      if(num_cells_in_theta > 1){
        ostringstream msg;
        msg << "\n ERROR: When using <axisymmetric>true</axisymmetric> \n"
            << "the grid can only have one cell in the circumferential direction.\n";
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }
    }
  }

  if (flags->d_reductionVars->accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);

}

void SerialMPM::readPrescribedDeformations(string filename)
{
 
 if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening prescribed deformation file '"+filename+"'\n",
                                  __FILE__, __LINE__);
    }
    double t0(-1.e9);
    while(is) {
        double t1,F11,F12,F13,F21,F22,F23,F31,F32,F33,Theta,a1,a2,a3;
        is >> t1 >> F11 >> F12 >> F13 >> F21 >> F22 >> F23 >> F31 >> F32 >> F33 >> Theta >> a1 >> a2 >> a3;
        if(is) {
            if(t1<=t0){
              throw ProblemSetupException("ERROR: Time in prescribed deformation file is not monotomically increasing", __FILE__, __LINE__);
            }
            d_prescribedTimes.push_back(t1);
            d_prescribedF.push_back(Matrix3(F11,F12,F13,F21,F22,F23,F31,F32,F33));
            d_prescribedAngle.push_back(Theta);
            d_prescribedRotationAxis.push_back(Vector(a1,a2,a3));
        }
        t0 = t1;
    }
    if(d_prescribedTimes.size()<2) {
        throw ProblemSetupException("ERROR: Failed to generate valid deformation profile",
                                    __FILE__, __LINE__);
    }
  }
}

void SerialMPM::readInsertParticlesFile(string filename)
{
 
 if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening particle insertion file '"+filename+"'\n",
                                  __FILE__, __LINE__);
    }

    double t0(-1.e9);
    while(is) {
        double t1,color,transx,transy,transz,v_new_x,v_new_y,v_new_z;
        is >> t1 >> color >> transx >> transy >> transz >> v_new_x >> v_new_y >> v_new_z;
        if(is) {
            if(t1<=t0){
              throw ProblemSetupException("ERROR: Time in insertParticleFile is not monotomically increasing", __FILE__, __LINE__);
            }
            d_IPTimes.push_back(t1);
            d_IPColor.push_back(color);
            d_IPTranslate.push_back(Vector(transx,transy,transz));
            d_IPVelNew.push_back(Vector(v_new_x,v_new_y,v_new_z));
        }
        t0 = t1;
    }
  }
}

void SerialMPM::actuallyInitializeAddedMaterial(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* /*matls*/,
                                                DataWarehouse*,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing actuallyInitializeAddedMaterial");

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    cout << "num MPM Matls = " << numMPMMatls << endl;
    CCVariable<short int> cellNAPID;
    int m=numMPMMatls-1;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    particleIndex numParticles = mpm_matl->countParticles(patch);

    new_dw->unfinalize();
    mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

    mpm_matl->getConstitutiveModel()->initializeCMData(patch, mpm_matl, new_dw);
    new_dw->refinalize();
  }
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* ,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  // Put something here to satisfy the need for a reduction operation in
  // the case that there are multiple levels present
  const Level* level = getLevel(patches);
  new_dw->put(delt_vartype(999.0), lb->delTLabel, level);
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing interpolateParticlesToGrid");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch); 
    ParticleInterpolator* linear_interpolator=scinew LinearInterpolator(patch);

    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    NCVariable<double> gmassglobal,gtempglobal,gvolumeglobal;
    NCVariable<Vector> gvelglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal, lb->gTemperatureLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gvolumeglobal, lb->gVolumeLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gvelglobal, lb->gVelocityLabel,
                           d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gvolumeglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
    gvelglobal.initialize(Vector(0.0));
    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce;
      constParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                                   pExternalForceCorner3, pExternalForceCorner4;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pFOld;
      constParticleVariable<Matrix3> pVelGrad;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
#ifdef GE_Proj
      old_dw->get(pVelGrad,       lb->pVelGradLabel,       pset);
#endif
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(psize,          lb->pSizeLabel,          pset);
      old_dw->get(pFOld,          lb->pDeformationMeasureLabel,pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);
      constParticleVariable<int> pLoadCurveID;
      if (flags->d_useCBDI) {
        new_dw->get(pExternalForceCorner1,
                   lb->pExternalForceCorner1Label, pset);
        new_dw->get(pExternalForceCorner2,
                   lb->pExternalForceCorner2Label, pset);
        new_dw->get(pExternalForceCorner3,
                   lb->pExternalForceCorner3Label, pset);
        new_dw->get(pExternalForceCorner4,
                   lb->pExternalForceCorner4Label, pset);
        old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
      }
      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gSp_vol;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gTemperatureRate;
      //NCVariable<double> gnumnearparticles;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gSp_vol,          lb->gSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gTemperatureRate, lb->gTemperatureRateLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
                             dwi,patch);
      new_dw->allocateAndPut(gexternalheatrate,lb->gExternalHeatRateLabel,
                             dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gTemperatureRate.initialize(0);
      gexternalheatrate.initialize(0);
      gSp_vol.initialize(0.);
      //gnumnearparticles.initialize(0.);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      Vector total_mom(0.0,0.0,0.0);
      Vector pmom;
      int n8or27=flags->d_8or27;
      double pSp_vol = 1./mpm_matl->getInitialDensity();
      //loop over all particles in the patch:
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);
        pmom = pvelocity[idx]*pmass[idx];
        total_mom += pmom;

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices
        IntVector node;
        for(int k = 0; k < n8or27; k++) { // Iterates through the nodes which receive information from the current particle
          node = ni[k];
          if(patch->containsNode(node)) {
#ifdef GE_Proj
            Point gpos = patch->getNodePosition(node);
            Vector distance = px[idx] - gpos;
            Vector pvel_ext = pvelocity[idx] - pVelGrad[idx]*distance;
            pmom = pvel_ext*pmass[idx];
#endif
            gmass[node]          += pmass[idx]                     * S[k];
            gvelocity[node]      += pmom                           * S[k];
            gvolume[node]        += pvolume[idx]                   * S[k];
            if (!flags->d_useCBDI) {
              gexternalforce[node] += pexternalforce[idx]          * S[k];
            }
            gTemperature[node]   += pTemperature[idx] * pmass[idx] * S[k];
            gSp_vol[node]        += pSp_vol           * pmass[idx] * S[k];
            //gnumnearparticles[node] += 1.0;
            //gexternalheatrate[node] += pexternalheatrate[idx]      * S[k];
          }
        }
        if (flags->d_useCBDI && pLoadCurveID[idx]>0) {
          vector<IntVector> niCorner1(linear_interpolator->size());
          vector<IntVector> niCorner2(linear_interpolator->size());
          vector<IntVector> niCorner3(linear_interpolator->size());
          vector<IntVector> niCorner4(linear_interpolator->size());
          vector<double> SCorner1(linear_interpolator->size());
          vector<double> SCorner2(linear_interpolator->size()); 
          vector<double> SCorner3(linear_interpolator->size()); 
          vector<double> SCorner4(linear_interpolator->size());
          linear_interpolator->findCellAndWeights(pExternalForceCorner1[idx],
                                 niCorner1,SCorner1,psize[idx],pFOld[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner2[idx],
                                 niCorner2,SCorner2,psize[idx],pFOld[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner3[idx],
                                 niCorner3,SCorner3,psize[idx],pFOld[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner4[idx],
                                 niCorner4,SCorner4,psize[idx],pFOld[idx]);
          for(int k = 0; k < 8; k++) { // Iterates through the nodes which receive information from the current particle
            node = niCorner1[k];
            if(patch->containsNode(node)) {
              gexternalforce[node] += pexternalforce[idx] * SCorner1[k];
            }
            node = niCorner2[k];
            if(patch->containsNode(node)) {
              gexternalforce[node] += pexternalforce[idx] * SCorner2[k];
            }
            node = niCorner3[k];
            if(patch->containsNode(node)) {
              gexternalforce[node] += pexternalforce[idx] * SCorner3[k];
            }
            node = niCorner4[k];
            if(patch->containsNode(node)) {
              gexternalforce[node] += pexternalforce[idx] * SCorner4[k];
            }
          }
        }
      } // End of particle loop
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter; 
        gmassglobal[c]    += gmass[c];
        gvolumeglobal[c]  += gvolume[c];
        gvelglobal[c]     += gvelocity[c];
        gvelocity[c]      /= gmass[c];
        gtempglobal[c]    += gTemperature[c];
        gTemperature[c]   /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gSp_vol[c]        /= gmass[c];
      }

      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,   interp_type);

      // If an MPMICE problem, create a velocity with BCs variable for NCToCC_0
      if(flags->d_with_ice){
        NCVariable<Vector> gvelocityWBC;
        new_dw->allocateAndPut(gvelocityWBC,lb->gVelocityBCLabel,dwi,patch);
        gvelocityWBC.copyData(gvelocity);
        bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocityWBC,interp_type);
        bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocityWBC,interp_type);
      }
    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gtempglobal[c] /= gmassglobal[c];
      gvelglobal[c] /= gmassglobal[c];
    }
    delete interpolator;
    delete linear_interpolator;
  }  // End loop over patches
}

void SerialMPM::addCohesiveZoneForces(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing addCohesiveZoneForces");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    StaticArray<NCVariable<Vector> > gext_force(numMPMMatls);
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* cz_matl = d_sharedState->getMPMMaterial( m );
      int dwi = cz_matl->getDWIndex();

      new_dw->getModifiable(gext_force[m], lb->gExternalForceLabel, dwi, patch);
    }

    Ghost::GhostType  gan = Ghost::AroundNodes;
    int numCZMatls=d_sharedState->getNumCZMatls();
    for(int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = d_sharedState->getCZMaterial( m );
      int dwi = cz_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      constParticleVariable<double> czlength;
      constParticleVariable<Vector> czforce;
      constParticleVariable<int> czTopMat, czBotMat;
      constParticleVariable<Matrix3> pDeformationMeasure;

      old_dw->get(czx,          lb->pXLabel,                          pset);
      new_dw->get(czlength,     lb->czLengthLabel_preReloc,           pset);
      new_dw->get(czforce,      lb->czForceLabel_preReloc,            pset);
      new_dw->get(czTopMat,     lb->czTopMatLabel_preReloc,           pset);
      new_dw->get(czBotMat,     lb->czBotMatLabel_preReloc,           pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

//        double length = sqrt(czlength[idx]);
        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);
        Matrix3 defgrad;
        defgrad.Identity();

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(czx[idx],ni,S,size,defgrad);

        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          if(patch->containsNode(node)) {
            gext_force[BotMat][node] = gext_force[BotMat][node] 
                                     + czforce[idx] * S[k];
            gext_force[TopMat][node] = gext_force[TopMat][node] 
                                     - czforce[idx] * S[k];
          }
        }
      }
    }
    delete interpolator;
  }
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0),cout_doing,
            "Doing computeStressTensor");

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

void SerialMPM::updateErosionParameter(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing updateErosionParameter");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: material # = " << m << endl;

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      if (cout_dbg.active()){
        cout_dbg << "updateErosionParameter:: mpm_matl* = " << mpm_matl 
                 << " dwi = " << dwi << " pset* = " << pset << endl;
      }

      // Get the localization info
      ParticleVariable<int> isLocalized;
      new_dw->allocateAndPut(isLocalized, lb->pLocalizedMPMLabel, pset);
      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++){
        isLocalized[*iter] = 0;
      }
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: Got Damage Parameter" << endl;

      if(flags->d_deleteRogueParticles){
        // The following looks for localized particles that are isolated
        // either individually or in small groups
        //Ghost::GhostType  gac = Ghost::AroundCells;
        CCVariable<int> numLocInCell,numInCell;
        new_dw->allocateAndPut(numLocInCell, lb->numLocInCellLabel, dwi, patch);
        new_dw->allocateAndPut(numInCell,    lb->numInCellLabel,    dwi, patch);
        numLocInCell.initialize(0);
        numInCell.initialize(0);

        constParticleVariable<Point> px;
        old_dw->get(px, lb->pXLabel, pset);

        // Count the number of localized particles in each cell
        for (iter = pset->begin(); iter != pset->end(); iter++) {
          IntVector c;
          patch->findCell(px[*iter],c);
          numInCell[c]++;
          if (isLocalized[*iter]) {
            numLocInCell[c]++;
          }
        }
      } // if d_deleteRogueParticles

      if (cout_dbg.active())
        cout_dbg << "updateErosionParameter:: Updated Erosion " << endl;

    }

    if (cout_dbg.active())
      cout_dbg <<"Done updateErosionParamter on patch "  << patch->getID() << "\t MPM"<< endl;

  }
}

void SerialMPM::findRogueParticles(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing findRogueParticles");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // The following looks for localized particles that are isolated
      // either individually or in small groups
      Ghost::GhostType  gac = Ghost::AroundCells;
      constCCVariable<int> numLocInCell,numInCell;
      constParticleVariable<Point> px;

      ParticleVariable<int> isLocalized;

      new_dw->get(numLocInCell, lb->numLocInCellLabel, dwi, patch, gac, 1);
      new_dw->get(numInCell,    lb->numInCellLabel,    dwi, patch, gac, 1);
      old_dw->get(px, lb->pXLabel, pset);
      new_dw->getModifiable(isLocalized, lb->pLocalizedMPMLabel, pset);

      // Look at the number of localized particles in the current and
      // surrounding cells
      for (ParticleSubset::iterator iter = pset->begin(); 
                                    iter != pset->end(); iter++) {
        if(isLocalized[*iter]==1){
          IntVector c;
          patch->findCell(px[*iter],c);
          int totalInCells = 0;
          for(int i=-1;i<2;i++){
            for(int j=-1;j<2;j++){
              for(int k=-1;k<2;k++){
                IntVector cell = c + IntVector(i,j,k);
                totalInCells += numInCell[cell];
              }
            }
          }
          // If the localized particles are sufficiently isolated, set
          // a flag for deletion in interpolateToParticlesAndUpdate
          if (numLocInCell[c]<=3 && totalInCells<=3) {
              isLocalized[*iter]=-999;
          }
        }  // if localized
      }  // particles
    }  // matls
  }  // patches
}

void SerialMPM::computeContactArea(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* /*old_dw*/,
                                   DataWarehouse* new_dw)
{
  // six indices for each of the faces
  double bndyCArea[6] = {0,0,0,0,0,0};
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeContactArea");
    
    Vector dx = patch->dCell();
    
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<double> gvolume;
      
      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
     
      for(list<Patch::FaceType>::const_iterator
          fit(d_bndy_traction_faces.begin()); 
          fit!=d_bndy_traction_faces.end();fit++) {       
        Patch::FaceType face = *fit;
        int iface = (int)(face);
        
        // Check if the face is on an external boundary
        if(patch->getBCType(face)==Patch::Neighbor)
           continue;
        
        // We are on the boundary, i.e. not on an interior patch
        // boundary, and also on the correct side, 

        // loop over face nodes to find boundary areas
// Because this calculation uses gvolume, particle volumes interpolated to
// the nodes, it will give 1/2 the expected value because the particle values
// are distributed to all nodes, not just those on this face.  It would require
// particles on the other side of the face to "fill" the nodal volumes and give
// the correct area when divided by the face normal cell dimension (celldepth).
// To correct for this, nodearea incorporates a factor of two.

        IntVector projlow, projhigh;
        patch->getFaceNodes(face, 0, projlow, projhigh);
        const double celldepth  = dx[iface/2];
        
        for (int i = projlow.x(); i<projhigh.x(); i++) {
          for (int j = projlow.y(); j<projhigh.y(); j++) {
            for (int k = projlow.z(); k<projhigh.z(); k++) {
              IntVector ijk(i,j,k);
              double nodearea         = 2.0*gvolume[ijk]/celldepth; // node area
              bndyCArea[iface] += nodearea;

            }
          }
        }
      } // faces
    } // materials
  } // patches
  
  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator 
      ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    new_dw->put(sum_vartype(bndyCArea[iface]),
                lb->BndyContactCellAreaLabel[iface]);
  }
}

void SerialMPM::computeInternalForce(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // node based forces
  Vector bndyForce[6];
  Vector bndyTraction[6];
  for(int iface=0;iface<6;iface++) {
    bndyForce   [iface]  = Vector(0.);
    bndyTraction[iface]  = Vector(0.);
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeInternalForce");

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

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gvolumeglobal;
    new_dw->get(gvolumeglobal,  lb->gVolumeLabel,
                d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
                           d_sharedState->getAllInOneMatl()->get(0), patch);

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
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pFOld;
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
      old_dw->get(pFOld,   lb->pDeformationMeasureLabel,     pset);

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

      // for the non axisymmetric case:
      if(!flags->d_axisymmetric){
        for (ParticleSubset::iterator iter = pset->begin();
             iter != pset->end(); 
             iter++){
          particleIndex idx = *iter;
  
          // Get the node indices that surround the cell
          interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                         psize[idx],pFOld[idx]);
          stressvol  = pstress[idx]*pvol[idx];
          stresspress = pstress[idx] + Id*(p_pressure[idx] - p_q[idx]);

          for (int k = 0; k < n8or27; k++){
            if(patch->containsNode(ni[k])){
              Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                         d_S[k].z()*oodx[2]);
              internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
              gstress[ni[k]]       += stressvol * S[k];
            }
          }
        }
      }

      // for the axisymmetric case
      if(flags->d_axisymmetric){
        for (ParticleSubset::iterator iter = pset->begin();
             iter != pset->end();
             iter++){
          particleIndex idx = *iter;

          interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                         psize[idx],pFOld[idx]);

          stressvol   = pstress[idx]*pvol[idx];
          stresspress = pstress[idx] + Id*(p_pressure[idx] - p_q[idx]);
  
          // r is the x direction, z (axial) is the y direction
          double IFr=0.,IFz=0.;
          for (int k = 0; k < n8or27; k++){
            if(patch->containsNode(ni[k])){
              IFr = d_S[k].x()*oodx[0]*stresspress(0,0) +
                    d_S[k].y()*oodx[1]*stresspress(0,1) +
                    d_S[k].z()*stresspress(2,2);
              IFz = d_S[k].x()*oodx[0]*stresspress(0,1)
                  + d_S[k].y()*oodx[1]*stresspress(1,1);
              internalforce[ni[k]] -=  Vector(IFr,IFz,0.0) * pvol[idx];
              gstress[ni[k]]       += stressvol * S[k];
            }
          }
        }
      }

      for(NodeIterator iter =patch->getNodeIterator();!iter.done();iter++){
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
      
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,interp_type);
    }

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      gstressglobal[c] /= gvolumeglobal[c];
    }
    delete interpolator;
  }
  
  // be careful only to put the fields that we have built
  // that way if the user asks to output a field that has not been built
  // it will fail early rather than just giving zeros.
  for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
      ftit!=d_bndy_traction_faces.end();ftit++) {
    int iface = (int)(*ftit);
    new_dw->put(sumvec_vartype(bndyForce[iface]),lb->BndyForceLabel[iface]);
    
    sum_vartype bndyContactCellArea_iface;
    new_dw->get(bndyContactCellArea_iface, lb->BndyContactCellAreaLabel[iface]);
    
    if(bndyContactCellArea_iface>0)
      bndyTraction[iface] /= bndyContactCellArea_iface;
    
    new_dw->put(sumvec_vartype(bndyTraction[iface]),
                               lb->BndyTractionLabel[iface]);
    
    // Use the face force and traction calculations to provide a second estimate
    // of the contact area.
    double bndyContactArea_iface = bndyContactCellArea_iface;
    if(bndyTraction[iface][iface/2]*bndyTraction[iface][iface/2]>1.e-12)
      bndyContactArea_iface = bndyForce[iface][iface/2]
                            / bndyTraction[iface][iface/2];

    new_dw->put(sum_vartype(bndyContactArea_iface),
                            lb->BndyContactAreaLabel[iface]);
  }
}

void SerialMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeAndIntegrateAcceleration");

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
      double damp_coef = flags->d_artificialDampCoeff;

      for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
        IntVector c = *iter;

        Vector acc(0.,0.,0.);
        if (mass[c] > flags->d_min_mass_for_acceleration){
          acc  = (internalforce[c] + externalforce[c])/mass[c];
          acc -= damp_coef*velocity[c];
        }
        acceleration[c] = acc +  gravity;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }    // matls
  }
}

void SerialMPM::setGridBoundaryConditions(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing setGridBoundaryConditions");

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
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi, patch,
                                                                Ghost::None,0);

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();
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
        for(NodeIterator iter=patch->getExtraNodeIterator();
                         !iter.done();iter++){
           IntVector c = *iter;
           displacement[c] = displacementOld[c] + gvelocity_star[c] * delT;
        }
      }  // d_doGridReset
    } // matl loop
  }  // patch loop
}

void SerialMPM::setPrescribedMotion(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{



 for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing setPrescribedMotion");

    // Get the current time
    double time = d_sharedState->getElapsedTime();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );


    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;

      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);

      gacceleration.initialize(Vector(0.0));
      Matrix3 Fdot(0.);

      // Get F and Q from file by interpolating between available times
      int s;  // This time index will be the lower of the two we interpolate from
      int smin = 0;
      int smax = (int) (d_prescribedTimes.size()-1);
      double tmin = d_prescribedTimes[smin];
      double tmax = d_prescribedTimes[smax];

      if(time<=tmin) {
          s=smin;
      } else if(time>=tmax) {
          s=smax;
      } else {
        while (smax>smin+1) {
          int smid = (smin+smax)/2;
          if(d_prescribedTimes[smid]<time){
            smin = smid;
          } else{
            smax = smid;
          }
        }
        s = smin;
      }

      Matrix3 F_high = d_prescribedF[s+1]; //next prescribed deformation gradient
      Matrix3 F_low  = d_prescribedF[s]; //last prescribed deformation gradient
      double t1 = d_prescribedTimes[s];    // time of last prescribed deformation
      double t2 = d_prescribedTimes[s+1];  //time of next prescribed deformation


      //Interpolate to get the deformation gradient at the current time:
      Matrix3 Ft = F_low*(t2-time)/(t2-t1) + F_high*(time-t1)/(t2-t1);

      // Calculate the rate of the deformation gradient without the rotation:
      Fdot = (F_high - F_low)/(t2-t1);

      // Now we need to construct the rotation matrix and its time rate:
      // We are only interested in the rotation information at the next specified time since the rotations specified should be relative to the previously specified time.  For example if I specify Theta=90 at time=1.0, and Theta = 91 and time=2.0 the total rotation at time=2.0 will be 181 degrees.
      //
      const double pi = M_PI; //3.1415926535897932384626433832795028841972;
      const double degtorad= pi/180.0;
      double PrescribedTheta = d_prescribedAngle[s+1]; //The final angle of rotation
      double thetat = PrescribedTheta*degtorad*(time-t1)/(t2-t1); // rotation angle at current time
      Vector a = d_prescribedRotationAxis[s+1];  // The axis of rotation
      Matrix3 Ident;
      Ident.Identity();
      const double costhetat = cos(thetat);
      const double sinthetat = sin(thetat);
      Matrix3 aa(a,a);
      Matrix3 A(0.0,-a.z(),a.y(),a.z(),0.0,-a.x(),-a.y(),a.x(),0.0);
      Matrix3 Qt;
      Qt = (Ident-aa)*costhetat+A*sinthetat + aa;
     
      //calculate thetadot:
      double thetadot = PrescribedTheta*(degtorad)/(t2-t1);


      if (flags->d_exactDeformation)//Exact Deformation Update
      {
         double t3 = d_prescribedTimes[s+2];    
         double t4 = d_prescribedTimes[s+3];  
         if (time == 0 && t4 != 0)
         {
            new_dw->put(delt_vartype(t3 - t2), d_sharedState->get_delt_label(), getLevel(patches));
         } 
         else
         {
            F_high = d_prescribedF[s + 2]; //next prescribed deformation gradient
            F_low  = d_prescribedF[s + 1]; //last prescribed deformation gradient
            t3 = d_prescribedTimes[s+2];
            t4 = d_prescribedTimes[s+3];
            double tst = t4 - t3; 
            Ft = F_low*(t2-time)/(t2-t1) + F_high*(time-t1)/(t2-t1);
            Fdot = (F_high - F_low)/(t3-t2);
            thetadot = PrescribedTheta*(degtorad)/(t3-t2);
            new_dw->put(delt_vartype(tst), d_sharedState->get_delt_label(), getLevel(patches));
          }
       }

      //construct Rdot:
      Matrix3 Qdot(0.0);
      Qdot = (Ident-aa)*(-sinthetat*thetadot) + A*costhetat*thetadot;



      Matrix3 Previous_Rotations;
      Previous_Rotations.Identity();
      int i;
      //now we need to compute the total previous rotation:
      for(i=0;i<s+1;i++){
              Vector ai;
              double thetai = d_prescribedAngle[i]*degtorad;
              ai = d_prescribedRotationAxis[i];
              const double costhetati = cos(thetai);
              const double sinthetati = sin(thetai);

              Matrix3 aai(ai,ai);
              Matrix3 Ai(0.0,-ai.z(),ai.y(),ai.z(),0.0,-ai.x(),-ai.y(),ai.x(),0.0);
              Matrix3 Qi;
              Qi = (Ident-aai)*costhetati+Ai*sinthetati + aai;

              Previous_Rotations = Qi*Previous_Rotations;
      }
     


      // Fstar is the deformation gradient with the superimposed rotations included
      // Fdotstar is the rate of the deformation gradient with superimposed rotations included
      Matrix3 Fstar;
      Matrix3 Fdotstar;
      Fstar = Qt*Previous_Rotations*Ft;
      Fdotstar = Qdot*Previous_Rotations*Ft + Qt*Previous_Rotations*Fdot;
      
      
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done(); iter++){
        IntVector n = *iter;

        Vector NodePosition = patch->getNodePosition(n).asVector();

        if (flags->d_exactDeformation)//Exact Deformation Update
        {
           gvelocity_star[n] = (F_high*F_low.Inverse() - Ident)*Previous_Rotations.Inverse()*Qt.Transpose()*NodePosition/delT;
        }
        else           
        {
           gvelocity_star[n] = Fdotstar*Ft.Inverse()*Previous_Rotations.Inverse()*Qt.Transpose()*NodePosition;
        }

      } // Node Iterator
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
    }   // matl loop
  }     // patch loop
}


void SerialMPM::applyExternalLoads(const ProcessorGroup* ,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();
  
  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalLoads) = " << time << endl;


  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> forcePerPart;
  std::vector<PressureBC*> pbcP;
  if (flags->d_useLoadCurves) {
    for (int ii = 0;ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {

        PressureBC* pbc = 
          dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // Calculate the force per particle at current time
        forcePerPart.push_back(pbc->forcePerParticle(time));
      }
    }
  }

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing applyExternalLoads");
    
    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.
    
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<Vector> pExternalForce_new;

      old_dw->get(px, lb->pXLabel, pset);
      old_dw->get(psize, lb->pSizeLabel, pset);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);
      new_dw->allocateAndPut(pExternalForce_new, 
                             lb->pExtForceLabel_preReloc,  pset);

      if (flags->d_useLoadCurves) {
        bool do_PressureBCs=false;
        for (int ii = 0; 
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
          string bcs_type = 
            MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
          if (bcs_type == "Pressure") {
            do_PressureBCs=true;
          }
        }

        // Get the load curve data
        constParticleVariable<int> pLoadCurveID;
        old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
        // Recycle the loadCurveIDs
        ParticleVariable<int> pLoadCurveID_new;
        new_dw->allocateAndPut(pLoadCurveID_new, 
                               lb->pLoadCurveIDLabel_preReloc, pset);
        pLoadCurveID_new.copyData(pLoadCurveID);

        if(do_PressureBCs){
          // Get the external force data and allocate new space for
          // external force
          constParticleVariable<Vector> pExternalForce;
          old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);

          ParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                                  pExternalForceCorner3, pExternalForceCorner4;
          if (flags->d_useCBDI) {
            new_dw->allocateAndPut(pExternalForceCorner1,
                                  lb->pExternalForceCorner1Label, pset);
            new_dw->allocateAndPut(pExternalForceCorner2,
                                  lb->pExternalForceCorner2Label, pset);
            new_dw->allocateAndPut(pExternalForceCorner3,
                                  lb->pExternalForceCorner3Label, pset);
            new_dw->allocateAndPut(pExternalForceCorner4,
                                  lb->pExternalForceCorner4Label, pset);
           }

          // Iterate over the particles
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            int loadCurveID = pLoadCurveID[idx]-1;
            if (loadCurveID < 0) {
              pExternalForce_new[idx] = pExternalForce[idx];
            } else {
              PressureBC* pbc = pbcP[loadCurveID];
              double force = forcePerPart[loadCurveID];
              
              if (flags->d_useCBDI) {
               Vector dxCell = patch->dCell();
               pExternalForce_new[idx] = pbc->getForceVectorCBDI(px[idx],
                                 psize[idx],pDeformationMeasure[idx],force,time,
                                    pExternalForceCorner1[idx],
                                    pExternalForceCorner2[idx],
                                    pExternalForceCorner3[idx],
                                    pExternalForceCorner4[idx],
                                    dxCell);
              } else {
               pExternalForce_new[idx] = pbc->getForceVector(px[idx],force,time);
              }
            }
          }
        } else {
           for(ParticleSubset::iterator iter = pset->begin();
               iter != pset->end(); iter++){
             pExternalForce_new[*iter] = 0.;
           }
        }
      } else {
// MMS
        string mms_type = flags->d_mms_type;
         if(!mms_type.empty()) {
           MMS MMSObject;                                                                                
           MMSObject.computeExternalForceForMMS(old_dw,new_dw,time,pset,lb,flags,pExternalForce_new);    
         } else { 
          // Get the external force data and allocate new space for
          // external force and copy the data
          constParticleVariable<Vector> pExternalForce;
          old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);

          for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalForce_new[idx] = pExternalForce[idx]*flags->d_forceIncrementFactor;
          }
        }
      }
    } // matl loop
  }  // patch loop
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing interpolateToParticlesAndUpdate");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};


    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
    double partvoldef = 0.;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
//    bool combustion_problem=false;

    Material* reactant;
//    int RMI = -99;
    reactant = d_sharedState->getMaterialByName("reactant");
    if(reactant != 0){
//      RMI = reactant->getDWIndex();
      //combustion_problem=true;
    }
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
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass, pTemperature;
      ParticleVariable<double> pmassNew,pvolume,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Matrix3> pFNew,pVelGrad;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew; 

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);

      new_dw->allocateAndPut(pvelocitynew,lb->pVelocityLabel_preReloc,    pset);
      new_dw->allocateAndPut(pxnew,       lb->pXLabel_preReloc,           pset);
      new_dw->allocateAndPut(pxx,         lb->pXXLabel,                   pset);
      new_dw->allocateAndPut(pdispnew,    lb->pDispLabel_preReloc,        pset);
      new_dw->allocateAndPut(pmassNew,    lb->pMassLabel_preReloc,        pset);
      new_dw->allocateAndPut(pvolume,     lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(pVelGrad,    lb->pVelGradLabel_preReloc,     pset);
      new_dw->allocateAndPut(pFNew,       lb->pDeformationMeasureLabel_preReloc,
                                                                          pset);
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);
      new_dw->allocateAndPut(pTempNew,    lb->pTemperatureLabel_preReloc, pset);

      //Carry forward ParticleID and pSize
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
      old_dw->get(psize,               lb->pSizeLabel,                pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,       pset);
      pids_new.copyData(pids);
      psizeNew.copyData(psize);

      //Carry forward color particle (debugging label)
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }    

      //Carry forward NC_CCweight
      constNCVariable<double> NC_CCweight;
      NCVariable<double> NC_CCweight_new;
      Ghost::GhostType  gnone = Ghost::None;
      old_dw->get(NC_CCweight,       lb->NC_CCweightLabel,  0, patch, gnone, 0);
      new_dw->allocateAndPut(NC_CCweight_new, lb->NC_CCweightLabel,0,patch);
      NC_CCweight_new.copyData(NC_CCweight);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate,lb->frictionalWorkLabel,  dwi,patch,gac,NGP);
      if(flags->d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
        new_dw->get(massBurnFrac,  lb->massBurnFractionLabel,dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        new_dw->allocateTemporary(massBurnFrac_create,           patch,gac,NGP);
        dTdt_create.initialize(0.);
        massBurnFrac_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
        massBurnFrac = massBurnFrac_create;         // reference created data
      }

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();
//      double rho_frac_min = 0.;
//      if(m == RMI){
//        rho_frac_min = .1;
//      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double fricTempRate = 0.0;
        double tempRate = 0.0;
        double burnFraction = 0.0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];

          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       fricTempRate)   * S[k];
          burnFraction += massBurnFrac[node]     * S[k];
        }
        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + acc*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx]     = pTemperature[idx]; // for thermal stress

        if (cout_heat.active()) {
          cout_heat << "MPM::Particle = " << pids[idx]
                    << " T_old = " << pTemperature[idx]
                    << " Tdot = " << tempRate
                    << " dT = " << (tempRate*delT)
                    << " T_new = " << pTempNew[idx] << endl;
        }

        pmassNew[idx]     = Max(pmass[idx]*(1.    - burnFraction),0.);

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom   += pvelocitynew[idx]*pmass[idx];
        totalmass  += pmass[idx];
      }

      // Compute velocity gradient and deformation gradient on every particle
      // This can/should be combined into the loop above, once it is working
      Matrix3 Identity;
      Identity.Identity();
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Matrix3 tensorL(0.0);
        if(!flags->d_axisymmetric){
         // Get the node indices that surround the cell
         interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                   pFOld[idx]);

         computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity_star);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                          psize[idx],pFOld[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity_star,
                                                                   px[idx]);
        }
        pVelGrad[idx]=tensorL;

        if(flags->d_min_subcycles_for_F>0){
          double Lnorm_dt = tensorL.Norm()*delT;
          int num_scs = min(max(flags->d_min_subcycles_for_F,
                                 2*((int) Lnorm_dt)),10000);
          //if(num_scs > 1000){
          //  cout << "NUM_SCS = " << num_scs << endl;
          //}
          double dtsc = delT/(double (num_scs));
          Matrix3 OP_tensorL_DT = Identity + tensorL*dtsc;
          Matrix3 F = pFOld[idx];
          for(int n=0;n<num_scs;n++){
            F=OP_tensorL_DT*F;
          }
          pFNew[idx]=F;
        }
        else{
          Matrix3 Amat = tensorL*delT;
          Matrix3 Finc = Amat.Exponential(abs(flags->d_min_subcycles_for_F));
          pFNew[idx] = Finc*pFOld[idx];
        }
#if 0
        Matrix3 Finc = tensorL*delT + Identity;
        pFNew[idx] = Finc*pFOld[idx];
#endif

        double J=pFNew[idx].Determinant();
        pvolume[idx]=(pmassNew[idx]/rho_init)*J;
        partvoldef += pvolume[idx];
      }

      // The following is used only for pressure stabilization
      CCVariable<double> J_CC;
      new_dw->allocateTemporary(J_CC,       patch);

      if(flags->d_doPressureStabilization) {
        CCVariable<double> vol_0_CC;
        CCVariable<double> vol_CC;
        new_dw->allocateTemporary(vol_0_CC, patch);
        new_dw->allocateTemporary(vol_CC,   patch);

        J_CC.initialize(0.);
        vol_0_CC.initialize(0.);
        vol_CC.initialize(0.);
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // get the volumetric part of the deformation
          double J = pFNew[idx].Determinant();

          IntVector cell_index;
          patch->findCell(px[idx],cell_index);

          vol_CC[cell_index]  +=pvolume[idx];
//  either of the following is correct
          vol_0_CC[cell_index]+=pvolume[idx]/J;
//          vol_0_CC[cell_index]+=pmassNew[idx]/rho_init;
        }

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          J_CC[c]=vol_CC[c]/vol_0_CC[c];
        }

        double ThreedelT  = 3.0*delT;
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;
          IntVector cell_index;
          patch->findCell(px[idx],cell_index);

          // get the original volumetric part of the deformation
          double J = pFNew[idx].Determinant();

          // Change F such that the determinant is equal to the average for
          // the cell
          pFNew[idx]*=cbrt(J_CC[cell_index]/J);
          // Change L such that it is consistent with the F          
          pVelGrad[idx]+= Identity*((log(J_CC[cell_index]/J))/ThreedelT);
        }
      } //end of pressureStabilization loop  at the patch level

      // scale back huge particle velocities.  Default for d_max_vel is 3.e105, hence the conditional
      if(flags->d_max_vel < 1.e105){
       for(ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(pvelocitynew[idx].length() > flags->d_max_vel){
          if(pvelocitynew[idx].length() >= pvelocity[idx].length()){
	     pvelocitynew[idx]=(pvelocitynew[idx]/pvelocitynew[idx].length())*(flags->d_max_vel*.9);	  
	     cout<<endl<<"Warning: particle "<<pids[idx]<<" hit speed ceiling #1. Modifying particle velocity accordingly."<<endl;
            //pvelocitynew[idx]=pvelocity[idx];
          }
        }
      }
     }
    }

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
    if(flags->d_reductionVars->mass){
      new_dw->put(sum_vartype(totalmass),      lb->TotalMassLabel);
    }
    if(flags->d_reductionVars->volDeformed){
      new_dw->put(sum_vartype(partvoldef),     lb->TotalVolumeDeformedLabel);
    }
    if(flags->d_reductionVars->momentum){
      new_dw->put(sumvec_vartype(totalMom),    lb->TotalMomentumLabel);
    }
    if(flags->d_reductionVars->KE){
      new_dw->put(sum_vartype(ke),             lb->KineticEnergyLabel);
    }
    if(flags->d_reductionVars->thermalEnergy){
      new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);
    }
    if(flags->d_reductionVars->centerOfMass){
      new_dw->put(sumvec_vartype(CMX),         lb->CenterOfMassPositionLabel);
    }

    delete interpolator;
  }
  
}

void SerialMPM::finalParticleUpdate(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing finalParticleUpdate");

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<int> pLocalized;
      constParticleVariable<double> pdTdt,pmassNew;
      ParticleVariable<double> pTempNew;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      new_dw->get(pdTdt,        lb->pdTdtLabel_preReloc,             pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);
      new_dw->get(pLocalized,   lb->pLocalizedMPMLabel,              pset);

      new_dw->getModifiable(pTempNew, lb->pTemperatureLabel_preReloc,pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pTempNew[idx] += pdTdt[idx]*delT;

        // Delete particles whose mass is too small (due to combustion),
        // whose pLocalized flag has been set to -999 or who have a negative temperature
        if ((pmassNew[idx] <= flags->d_min_part_mass) || pTempNew[idx] < 0. ||
             (pLocalized[idx]==-999)){
          delset->addParticle(idx);
        }

      } // particles
      new_dw->deleteParticles(delset);    
    } // materials
  } // patches
}

void SerialMPM::interpolateToParticlesAndUpdateMom1(const ProcessorGroup*,
                                                    const PatchSubset* patches,
                                                    const MaterialSubset* ,
                                                    DataWarehouse* old_dw,
                                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing interpolateToParticlesAndUpdateMom1");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // DON'T MOVE THESE!!!
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;

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
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<Vector> pvelocitynew;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pFOld;
      constParticleVariable<long64> pids;
	  
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(psize,        lb->pSizeLabel,                      pset);
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
	  
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);  
	  
      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + acc*delT;
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];

        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom += pvelocitynew[idx]*pmass[idx];
      }

      // Delete particles that have left the domain
      // This is only needed if extra cells are being used.
      // Also delete particles whose mass is too small (due to combustion)
      // For particles whose new velocity exceeds a maximum set in the input
      // file, set their velocity back to the velocity that it came into
      // this step with
      for(ParticleSubset::iterator iter  = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(pvelocitynew[idx].length() > flags->d_max_vel){
	      pvelocitynew[idx]=(pvelocitynew[idx]/pvelocitynew[idx].length())*flags->d_max_vel;
		  cout<<endl<<"Warning: particle "<<pids[idx]<<" hit speed ceiling #2. Modifying particle velocity accordingly."<<endl;
          //pvelocitynew[idx]=pvelocity[idx];
        }
      }
    }

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
    if(flags->d_reductionVars->momentum){
      new_dw->put(sumvec_vartype(totalMom),    lb->TotalMomentumLabel);
    }
    if(flags->d_reductionVars->KE){
      new_dw->put(sum_vartype(ke),             lb->KineticEnergyLabel);
    }
    if(flags->d_reductionVars->centerOfMass){
      new_dw->put(sumvec_vartype(CMX),         lb->CenterOfMassPositionLabel);
    }    

    delete interpolator;
  }

}

void SerialMPM::interpolateToParticlesAndUpdateMom2(const ProcessorGroup*,
                                                    const PatchSubset* patches,
                                                    const MaterialSubset* ,
                                                    DataWarehouse* old_dw,
                                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing interpolateToParticlesAndUpdateMom2");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
    double partvoldef = 0.;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
//    bool combustion_problem=false;

    Material* reactant;
    int RMI = -99;
    reactant = d_sharedState->getMaterialByName("reactant");
    if(reactant != 0){
      RMI = reactant->getDWIndex();
      //combustion_problem=true;
    }

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass, pTemperature, pdTdt;
      ParticleVariable<double> pmassNew,pvolume,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<int> pLocalized;
      constParticleVariable<Matrix3> pFNew,pFOld;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(pdTdt,        lb->pdTdtLabel_preReloc,             pset);
      new_dw->get(pFNew,        lb->pDeformationMeasureLabel_preReloc, pset);
      new_dw->get(pLocalized,         lb->pLocalizedMPMLabel,        pset);

      new_dw->getModifiable(pvolume,  lb->pVolumeLabel_preReloc,     pset);

      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);

      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

      //Carry forward NC_CCweight
      constNCVariable<double> NC_CCweight;
      NCVariable<double> NC_CCweight_new;
      Ghost::GhostType  gnone = Ghost::None;
      old_dw->get(NC_CCweight,       lb->NC_CCweightLabel,  0, patch, gnone, 0);
      new_dw->allocateAndPut(NC_CCweight_new, lb->NC_CCweightLabel,0,patch);
      NC_CCweight_new.copyData(NC_CCweight);

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate,lb->frictionalWorkLabel,  dwi,patch,gac,NGP);
      if(flags->d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
        new_dw->get(massBurnFrac,  lb->massBurnFractionLabel,dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        new_dw->allocateTemporary(massBurnFrac_create,           patch,gac,NGP);
        dTdt_create.initialize(0.);
        massBurnFrac_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
        massBurnFrac = massBurnFrac_create;         // reference created data
      }

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();
      double rho_frac_min = 0.;
      if(m == RMI){
        rho_frac_min = .1;
      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFNew[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double fricTempRate = 0.0;
        double tempRate = 0.0;
        double burnFraction = 0.0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];

          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       fricTempRate)   * S[k];
          burnFraction += massBurnFrac[node]     * S[k];
        }

        // Update the particle's position and velocity
        pTempNew[idx]        = pTemperature[idx] + (tempRate+pdTdt[idx])*delT;
        pTempPreNew[idx]     = pTemperature[idx]; // for thermal stress

        if (cout_heat.active()) {
          cout_heat << "MPM::Particle = " << idx
                    << " T_old = " << pTemperature[idx]
                    << " Tdot = " << tempRate
                    << " dT = " << (tempRate*delT)
                    << " T_new = " << pTempNew[idx] << endl;
        }

        double rho;
        if(pvolume[idx] > 0.){
          rho = max(pmass[idx]/pvolume[idx],rho_frac_min*rho_init);
        }
        else{
          rho = rho_init;
        }
        pmassNew[idx]     = Max(pmass[idx]*(1.    - burnFraction),0.);
        pvolume[idx]      = pmassNew[idx]/rho;

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
      }

      // Delete particles that have left the domain
      // This is only needed if extra cells are being used.
      // Also delete particles whose mass is too small (due to combustion)
      // For particles whose new velocity exceeds a maximum set in the input
      // file, set their velocity back to the velocity that it came into
      // this step with
      for(ParticleSubset::iterator iter  = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if ((pmassNew[idx] <= flags->d_min_part_mass) || pTempNew[idx] < 0. ||
             (pLocalized[idx]==-999)){
          delset->addParticle(idx);
//        cout << "Material = " << m << " Deleted Particle = " << idx 
//             << " xold = " << px[idx] << " xnew = " << pxnew[idx]
//             << " vold = " << pvelocity[idx] << " vnew = "<< pvelocitynew[idx]
//             << " massold = " << pmass[idx] << " massnew = " << pmassNew[idx]
//             << " tempold = " << pTemperature[idx] 
//             << " tempnew = " << pTempNew[idx]
//             << " volnew = " << pvolume[idx] << endl;
        }
      }

      new_dw->deleteParticles(delset);
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

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
    if(flags->d_reductionVars->mass){
      new_dw->put(sum_vartype(totalmass),      lb->TotalMassLabel);
    }
    if(flags->d_reductionVars->volDeformed){
      new_dw->put(sum_vartype(partvoldef),     lb->TotalVolumeDeformedLabel);
    }
    if(flags->d_reductionVars->thermalEnergy){
      new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);
    }
    
    delete interpolator;
  }
}

void SerialMPM::updateCohesiveZones(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing updateCohesiveZones");

    // The following is adapted from "Simulation of dynamic crack growth
    // using the generalized interpolation material point (GIMP) method"
    // Daphalapurkar, N.P., et al., Int. J. Fracture, 143, 79-102, 2007.

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    StaticArray<constNCVariable<Vector> > gvelocity(numMPMMatls);
    StaticArray<constNCVariable<double> > gmass(numMPMMatls);
    //double rho_init[numMPMMatls];
    Vector dx = patch-> dCell();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      //rho_init[m]=mpm_matl->getInitialDensity();
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity[m], lb->gVelocityLabel,dwi, patch, gac, NGN);
      new_dw->get(gmass[m],     lb->gMassLabel,    dwi, patch, gac, NGN);
    }

/*
    double time = d_sharedState->getElapsedTime();
    string outfile_name = "force_sep.dat";
    ofstream dest;
    dest.open(outfile_name.c_str(),ios::app);
    if(!dest){
      cerr << "File " << outfile_name << " can't be opened." << endl;
    }
*/

    int numCZMatls=d_sharedState->getNumCZMatls();
    for(int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = d_sharedState->getCZMaterial( m );
      int dwi = cz_matl->getDWIndex();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);      

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      ParticleVariable<Point> czx_new;
      constParticleVariable<double> czlength;
      ParticleVariable<double> czlength_new;
      constParticleVariable<long64> czids;
      ParticleVariable<long64> czids_new;
      constParticleVariable<Vector> cznorm, cztang, czDispTop;
      ParticleVariable<Vector> cznorm_new, cztang_new, czDispTop_new;
      constParticleVariable<Vector> czDispBot, czsep, czforce;
      ParticleVariable<Vector> czDispBot_new, czsep_new, czforce_new;
      constParticleVariable<int> czTopMat, czBotMat, czFailed;
      ParticleVariable<int> czTopMat_new, czBotMat_new, czFailed_new;

      old_dw->get(czx,          lb->pXLabel,                         pset);
      old_dw->get(czlength,     lb->czLengthLabel,                   pset);
      old_dw->get(cznorm,       lb->czNormLabel,                     pset);
      old_dw->get(cztang,       lb->czTangLabel,                     pset);
      old_dw->get(czDispTop,    lb->czDispTopLabel,                  pset);
      old_dw->get(czDispBot,    lb->czDispBottomLabel,               pset);
      old_dw->get(czsep,        lb->czSeparationLabel,               pset);
      old_dw->get(czforce,      lb->czForceLabel,                    pset);
      old_dw->get(czids,        lb->czIDLabel,                       pset);
      old_dw->get(czTopMat,     lb->czTopMatLabel,                   pset);
      old_dw->get(czBotMat,     lb->czBotMatLabel,                   pset);
      old_dw->get(czFailed,     lb->czFailedLabel,                   pset);

      new_dw->allocateAndPut(czx_new,      lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(czlength_new, lb->czLengthLabel_preReloc,    pset);
      new_dw->allocateAndPut(cznorm_new,   lb->czNormLabel_preReloc,      pset);
      new_dw->allocateAndPut(cztang_new,   lb->czTangLabel_preReloc,      pset);
      new_dw->allocateAndPut(czDispTop_new,lb->czDispTopLabel_preReloc,   pset);
      new_dw->allocateAndPut(czDispBot_new,lb->czDispBottomLabel_preReloc,pset);
      new_dw->allocateAndPut(czsep_new,    lb->czSeparationLabel_preReloc,pset);
      new_dw->allocateAndPut(czforce_new,  lb->czForceLabel_preReloc,     pset);
      new_dw->allocateAndPut(czids_new,    lb->czIDLabel_preReloc,        pset);
      new_dw->allocateAndPut(czTopMat_new, lb->czTopMatLabel_preReloc,    pset);
      new_dw->allocateAndPut(czBotMat_new, lb->czBotMatLabel_preReloc,    pset);
      new_dw->allocateAndPut(czFailed_new, lb->czFailedLabel_preReloc,    pset);


      czlength_new.copyData(czlength);
      czids_new.copyData(czids);
      czTopMat_new.copyData(czTopMat);
      czBotMat_new.copyData(czBotMat);

      double sig_max = cz_matl->getCohesiveNormalStrength();
      double delta_n = cz_matl->getCharLengthNormal();
      double tau_max = cz_matl->getCohesiveTangentialStrength();
      double delta_t = cz_matl->getCharLengthTangential();
      double delta_s = delta_t;
      bool rotate_CZs= cz_matl->getDoRotation();

      double phi_n = M_E*sig_max*delta_n;
      double phi_t = sqrt(M_E/2)*tau_max*delta_t;
      double q = phi_t/phi_n;
      // From the text following Eq. 15 in Nitin's paper it is a little hard
      // to tell what r should be, but zero seems like a reasonable value
      // based on the example problem in that paper
      double r=0.;

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

//        double length = sqrt(czlength[idx]);
//        Vector size(length,length,length);
        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);
        Matrix3 defgrad;
        defgrad.Identity();

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(czx[idx],ni,S,size,defgrad);

        Vector velTop(0.0,0.0,0.0);
        Vector velBot(0.0,0.0,0.0);
        double massTop = 0.0;
        double massBot = 0.0;
        double mass_ratio = 0.0;
        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];
        double cell_volume = dx.x()*dx.y()*dx.z();
        //double denseTop = rho_init[TopMat];
        //double denseBot = rho_init[BotMat];
        double TOPMAX = 0.0;
        double BOTMAX = 0.0;
        
//      if (denseBot != denseTop){
//         throw ProblemSetupException("Different densities not allowed for Bottom and Top Material of Cohesive Zone",
//                                 __FILE__, __LINE__);
//      }

        //double density_ratio = denseTop/denseBot;
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          velTop      += gvelocity[TopMat][node] * S[k];
          velBot      += gvelocity[BotMat][node] * S[k];
          massTop     += gmass[TopMat][node]*S[k];
          TOPMAX      += cell_volume;
          massBot     += gmass[BotMat][node]*S[k];
          BOTMAX      += cell_volume;
        }
        massTop = massTop/TOPMAX;
        massBot = massBot/BOTMAX;
        if (massBot > 0.0) {
            mass_ratio = massTop/massBot;
            mass_ratio = min(mass_ratio,1.0/mass_ratio);
        }
        else {
            mass_ratio = 0.0;
        }
        double mass_correction_factor = mass_ratio;

        // Update the cohesive zone's position and displacements
        czx_new[idx]         = czx[idx]       + .5*(velTop + velBot)*delT;
        czDispTop_new[idx]   = czDispTop[idx] + velTop*delT;
        czDispBot_new[idx]   = czDispBot[idx] + velBot*delT;
        czsep_new[idx]       = czDispTop_new[idx] - czDispBot_new[idx];

        double disp = czsep_new[idx].length();
        if (disp > 0.0 && rotate_CZs){
          Matrix3 Rotation;
          Matrix3 Rotation_tang;
          cz_matl->computeRotationMatrix(Rotation, Rotation_tang,
                                         cznorm[idx],czsep_new[idx]);

          cznorm_new[idx] = Rotation*cznorm[idx];
          cztang_new[idx] = Rotation_tang*cztang[idx];
        }
        else {
          cznorm_new[idx]=cznorm[idx];
          cztang_new[idx]=cztang[idx];
        }

        Vector cztang2 = Cross(cztang_new[idx],cznorm_new[idx]);

        double D_n  = Dot(czsep_new[idx],cznorm_new[idx]);
        double D_t1 = Dot(czsep_new[idx],cztang_new[idx]);
        double D_t2 = Dot(czsep_new[idx],cztang2);

        // Determine if a CZ has failed.  Currently hardwiring failure criteria
        // to fail zone if normal sep is > 4*delta_n or 2*delta_t
        double czf=0.0;
        if(czFailed[idx]>0 ){
          czFailed_new[idx]=czFailed[idx];
          czf=1.0;
        }
        else if(D_n > 4.0*delta_n){
          czFailed_new[idx]=1;
          czf=1.0;
        }
        else if( fabs(D_t1) > 2.0*delta_t){
          czFailed_new[idx]=2;
          czf=1.0;
        } 
        else if( fabs(D_t2) > 2.0*delta_s){
          czFailed_new[idx]=2;
          czf=1.0;
        }
        else {
          czFailed_new[idx]=0;
        }

        double normal_stress  = (phi_n/delta_n)*exp(-D_n/delta_n)*
                              ((D_n/delta_n)*exp((-D_t1*D_t1)/(delta_t*delta_t))
                              + ((1.-q)/(r-1.))
                       *(1.-exp(-D_t1*D_t1/(delta_t*delta_t)))*(r-D_n/delta_n));

        double tang1_stress =(phi_n/delta_n)*(2.*delta_n/delta_t)*(D_t1/delta_t)
                              * (q
                              + ((r-q)/(r-1.))*(D_n/delta_n))
                              * exp(-D_n/delta_n)
                              * exp(-D_t1*D_t1/(delta_t*delta_t));

        double tang2_stress =(phi_n/delta_n)*(2.*delta_n/delta_s)*(D_t2/delta_s)
                              * (q
                              + ((r-q)/(r-1.))*(D_n/delta_n))
                              * exp(-D_n/delta_n)
                              * exp(-D_t2*D_t2/(delta_s*delta_s));

        czforce_new[idx]     = mass_correction_factor*(normal_stress*cznorm_new[idx]*czlength_new[idx]
                             + tang1_stress*cztang_new[idx]*czlength_new[idx]
                             + tang2_stress*cztang2*czlength_new[idx])
                             * (1.0 - czf);

/*
        dest << time << " " << czsep_new[idx].x() << " " << czsep_new[idx].y() << " " << czforce_new[idx].x() << " " << czforce_new[idx].y() << endl;
        if(fabs(normal_force) >= 0.0){
          cout << "czx_new " << czx_new[idx] << endl;
          cout << "czforce_new " << czforce_new[idx] << endl;
          cout << "czsep_new " << czsep_new[idx] << endl;
          cout << "czDispTop_new " << czDispTop_new[idx] << endl;
          cout << "czDispBot_new " << czDispBot_new[idx] << endl;
          cout << "velTop " << velTop << endl;
          cout << "velBot " << velBot << endl;
          cout << "delT " << delT << endl;
        }
*/
      
      }
    }

    delete interpolator;
  }
}

void SerialMPM::insertParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing insertParticles");

    // Get current time and timestep size
    double time = d_sharedState->getElapsedTime();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    int index = -999;

    for(int i = 0; i<(int) d_IPTimes.size(); i++){
       if(time+delT > d_IPTimes[i] && time <= d_IPTimes[i]){
         index = i;
       }
    }

    if(index>=0){
      int numMPMMatls=d_sharedState->getNumMPMMatls();
      for(int m = 0; m < numMPMMatls; m++){
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int dwi = mpm_matl->getDWIndex();
        ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

        // Get the arrays of particle values to be changed
        ParticleVariable<Point> px;
        ParticleVariable<Vector> pvelocity;
        constParticleVariable<double> pcolor;

        old_dw->get(pcolor,               lb->pColorLabel,              pset);
        new_dw->getModifiable(px,         lb->pXLabel_preReloc,         pset);
        new_dw->getModifiable(pvelocity,  lb->pVelocityLabel_preReloc,  pset);

        // Loop over particles here
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
           if(pcolor[idx]==d_IPColor[index]){
             pvelocity[idx]=d_IPVelNew[index];
             px[idx] = px[idx] + d_IPTranslate[index];
           }
        }
      }
    }
  }
}

void SerialMPM::computeParticleScaleFactor(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // This task computes the particles initial physical size, to be used
  // in scaling particles for the deformed particle vis feature

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing computeParticleScaleFactor");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize;
      ParticleVariable<Matrix3> pScaleFactor;
      old_dw->get(psize,                   lb->pSizeLabel,                pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(dataArchiver->isOutputTimestep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = (psize[idx]*Matrix3(dx[0],0,0,
                                                  0,dx[1],0,
                                                  0,0,dx[2]));

        } // for particles
      } // isOutputTimestep
    } // matls
  } // patches

}

void SerialMPM::interpolateParticleVelToGridMom(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing interpolateParticleVelToGridMom");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pFOld;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,                  pset);
      old_dw->get(pmass,          lb->pMassLabel,               pset);
      old_dw->get(psize,          lb->pSizeLabel,               pset);
      new_dw->get(pvelocity,      lb->pVelocityLabel_preReloc,  pset);
      old_dw->get(pFOld,          lb->pDeformationMeasureLabel, pset);

      // Create arrays for the grid data
      constNCVariable<double> gmass;
      NCVariable<Vector> gvelocity_star;
      new_dw->get(gmass,               lb->gMassLabel, dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star, lb->gVelocityStarLabel,  dwi,patch);
      gvelocity_star.initialize(Vector(0,0,0));

      int n8or27=flags->d_8or27;
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end();
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pFOld[idx]);

        Vector pmom = pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices
        for(int k = 0; k < n8or27; k++) {
          IntVector node = ni[k];
          if(patch->containsNode(node)) {
            gvelocity_star[node]      += pmom   * S[k];
          }
        }
      } // End of particle loop

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        gvelocity_star[c]      /= gmass[c];
      }

//    setGridBoundaryConditions handles the BCs for gvelocity_star
    }  // end of materials loop

    delete interpolator;

  }  // End loop over patches
}

void 
SerialMPM::setParticleDefault(ParticleVariable<double>& pvar,
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
SerialMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
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
SerialMPM::setParticleDefault(ParticleVariable<Matrix3>& pvar,
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


void SerialMPM::printParticleLabels(vector<const VarLabel*> labels,
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
SerialMPM::initialErrorEstimate(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* /*matls*/,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialErrorEstimate");

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
SerialMPM::errorEstimate(const ProcessorGroup* group,
                         const PatchSubset* coarsePatches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  if (coarseLevel->getIndex() == coarseLevel->getGrid()->numLevels()-1) {
    // on finest level, we do the same thing as initialErrorEstimate, so call it
    initialErrorEstimate(group, coarsePatches, matls, old_dw, new_dw);
  }
  else {
    // coarsen the errorflag.
    const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
    for(int p=0;p<coarsePatches->size();p++){  
      const Patch* coarsePatch = coarsePatches->get(p);
      printTask(coarsePatches, coarsePatch,cout_doing,
                "Doing errorEstimate");
     
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
        for(CellIterator iter(fl, fh); !iter.done(); iter++){

          IntVector coarseCell(fineLevel->mapCellToCoarser(*iter));

          if (fineErrorFlag[*iter]) {
            refineFlag[coarseCell] = 1;
            refinePatch->set();
          }
        }
      }  // fine patch loop
    } // coarse patch loop 
  }
}  

void
SerialMPM::refine(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* /*matls*/,
                  DataWarehouse*,
                  DataWarehouse* new_dw)
{
  // just create a particle subset if one doesn't exist
  // and initialize NC_CCweights

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing refine");

    int numMPMMatls=d_sharedState->getNumMPMMatls();

    // First do NC_CCweight 
    NCVariable<double> NC_CCweight;
    new_dw->allocateAndPut(NC_CCweight, lb->NC_CCweightLabel,  0, patch);
    //__________________________________
    // - Initialize NC_CCweight = 0.125
    // - Find the walls with symmetry BC and
    //   double NC_CCweight
    NC_CCweight.initialize(0.125);
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for (iter  = bf.begin(); iter != bf.end(); ++iter){
      Patch::FaceType face = *iter;
      int mat_id = 0;
      if (patch->haveBC(face,mat_id,"symmetry","Symmetric")) {

        for(CellIterator iter = patch->getFaceIterator(face,Patch::FaceNodes);
            !iter.done(); iter++) {
          NC_CCweight[*iter] = 2.0*NC_CCweight[*iter];
        }
      }
    }

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
        ParticleVariable<Vector> pvelocity, pexternalforce, pdisp;
        ParticleVariable<Matrix3> psize, pVelGrad;
        ParticleVariable<double> pTempPrev,p_q;
        ParticleVariable<int>    pLoadCurve;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(p_q,            lb->p_qLabel,            pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pVelGrad,       lb->pVelGradLabel,       pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        if (flags->d_useLoadCurves){
          new_dw->allocateAndPut(pLoadCurve,   lb->pLoadCurveIDLabel,   pset);
        }
        new_dw->allocateAndPut(psize,          lb->pSizeLabel,          pset);

        mpm_matl->getConstitutiveModel()->initializeCMData(patch,
                                                           mpm_matl,new_dw);
#if 0
          if(flags->d_with_color) {
            ParticleVariable<double> pcolor;
            int index = mpm_matl->getDWIndex();
            ParticleSubset* pset = new_dw->getParticleSubset(index, patch);
            setParticleDefault(pcolor, lb->pColorLabel, pset, new_dw, 0.0);
          }
#endif
      }
    }
  }

} // end refine()

bool SerialMPM::needRecompile(double , double , const GridP& )
{
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
