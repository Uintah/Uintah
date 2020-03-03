/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/MPM/SingleFieldMPM.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/Materials/Contact/ContactFactory.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <fstream>

#define RIGID
//#undef RIGID

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SingleFieldMPM", false);
static DebugStream cout_convert("MPMConv", false);
static DebugStream cout_heat("MPMHeat", false);
static DebugStream amr_doing("AMRMPM", false);

static Vector face_norm(Patch::FaceType f)
{
  switch(f) {
  case Patch::xminus: return Vector(-1,0,0);
  case Patch::xplus:  return Vector( 1,0,0);
  case Patch::yminus: return Vector(0,-1,0);
  case Patch::yplus:  return Vector(0, 1,0);
  case Patch::zminus: return Vector(0,0,-1);
  case Patch::zplus:  return Vector(0,0, 1);
  default:            return Vector(0,0,0); // oops !
  }
}

SingleFieldMPM::SingleFieldMPM( const ProcessorGroup* myworld,
                                const MaterialManagerP materialManager) :
  MPMCommon( myworld, materialManager )
{
  flags = scinew MPMFlags(myworld);

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  contactModel        = 0;
  thermalContactModel = 0;
  heatConductionModel = 0;
  NGP     = 1;
  NGN     = 1;
  d_loadCurveIndex=0;
  d_switchCriteria = 0;
  flags->d_SingleFieldMPM=true;

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();
}

SingleFieldMPM::~SingleFieldMPM()
{
  delete flags;
  delete contactModel;
  delete thermalContactModel;
  delete heatConductionModel;
  MPMPhysicalBCFactory::clean();

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->releaseComponents();
      delete am;
    }
  }

  if (d_one_matl->removeReference())
    delete d_one_matl;

  if(d_switchCriteria) {
    delete d_switchCriteria;
  }
}

void SingleFieldMPM::problemSetup(const ProblemSpecP& prob_spec,
                             const ProblemSpecP& restart_prob_spec,
                             GridP& grid)
{
  cout_doing<<"Doing MPM::problemSetup\t\t\t\t\t MPM"<<endl;

  m_scheduler->setPositionVar(lb->pXLabel);

  ProblemSpecP restart_mat_ps = 0;
  ProblemSpecP prob_spec_mat_ps =
    prob_spec->findBlockWithOutAttribute("MaterialProperties");

  bool isRestart = false;
  if (prob_spec_mat_ps){
    restart_mat_ps = prob_spec;
  } else if (restart_prob_spec){
    isRestart = true;
    restart_mat_ps = restart_prob_spec;
  } else{
    restart_mat_ps = prob_spec;
  }

  ProblemSpecP mpm_soln_ps = restart_mat_ps->findBlock("MPM");
  if (!mpm_soln_ps){
    ostringstream warn;
    warn<<"ERROR:MPM:\n missing MPM section in the input file\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  // Read all MPM flags (look in MPMFlags.cc)
  flags->readMPMFlags(restart_mat_ps, m_output);
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
    ProblemSpecP refine_ps =
                     mpm_amr_ps->findBlock("Refinement_Criteria_Thresholds");
    //__________________________________
    // Pull out the refinement threshold criteria 
    if( refine_ps != nullptr ){
      for( ProblemSpecP var_ps = refine_ps->findBlock( "Variable" ); var_ps != nullptr; var_ps = var_ps->findNextBlock( "Variable" ) ) {
        thresholdVar data;
        string name, value, matl;

        map<string,string> input;
        var_ps->getAttributes(input);
        name  = input["name"];
        value = input["value"];
        matl  = input["matl"];

        stringstream n_ss(name);
        stringstream v_ss(value);
        stringstream m_ss(matl);

        n_ss >> data.name;
        v_ss >> data.value;
        m_ss >> data.matl;

        if( !n_ss || !v_ss || (!m_ss && matl!="all") ) {
          cerr << "WARNING: AMRMPM.cc: stringstream failed...\n";
        }

        unsigned int numMatls = m_materialManager->getNumMatls();

        //__________________________________
        // if using "all" matls 
        if(matl == "all"){
          for (unsigned int m = 0; m < numMatls; m++){
            data.matl = m;
            d_thresholdVars.push_back(data);
          }
        }else{
          d_thresholdVars.push_back(data);
        }
      }
    } // refine_ps
  } // amr_ps
  
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

  setParticleGhostLayer(Ghost::AroundNodes, NGP);

  MPMPhysicalBCFactory::create(restart_mat_ps, grid, flags);

  bool needNormals = false;
  bool useLR       = false;
  contactModel = ContactFactory::create(d_myworld,
                                        restart_mat_ps,m_materialManager,lb,
                                        flags, needNormals, useLR);

  flags->d_computeNormals=needNormals;
  flags->d_useLogisticRegression=useLR;

  thermalContactModel =
    ThermalContactFactory::create(restart_mat_ps, m_materialManager, lb,flags);

  heatConductionModel = scinew HeatConduction(m_materialManager,lb,flags);

  materialProblemSetup(restart_mat_ps,flags, isRestart);

  cohesiveZoneProblemSetup(restart_mat_ps, flags);

  //__________________________________
  //  create analysis modules
  // call problemSetup
  if(!flags->d_with_ice && !flags->d_with_arches){ // mpmice or mpmarches handles this
    d_analysisModules = AnalysisModuleFactory::create(d_myworld,
						      m_materialManager,
						      prob_spec);

    if(d_analysisModules.size() != 0){
      vector<AnalysisModule*>::iterator iter;
      for( iter  = d_analysisModules.begin();
           iter != d_analysisModules.end(); iter++) {
        AnalysisModule* am = *iter;
        am->setComponents( dynamic_cast<ApplicationInterface*>( this ) );
        am->problemSetup(prob_spec,restart_prob_spec, grid,
			 d_particleState, d_particleState_preReloc);
      }
    }
  }

  //__________________________________
  //  create the switching criteria port
  d_switchCriteria = dynamic_cast<SwitchingCriteria*>(getPort("switch_criteria"));

  if (d_switchCriteria) {
    d_switchCriteria->problemSetup(restart_mat_ps,
                                   restart_prob_spec, m_materialManager);
  }
}
//______________________________________________________________________
//
void SingleFieldMPM::outputProblemSpec(ProblemSpecP& root_ps)
{
  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP flags_ps = root->appendChild("MPM");
  flags->outputProblemSpec(flags_ps);

  ProblemSpecP mat_ps = root->findBlockWithOutAttribute( "MaterialProperties" );

  if( mat_ps == nullptr ) {
    mat_ps = root->appendChild( "MaterialProperties" );
  }

  ProblemSpecP mpm_ps = mat_ps->appendChild("MPM");
  for (unsigned int i = 0; i < m_materialManager->getNumMatls( "MPM" );i++) {
    MPMMaterial* mat = (MPMMaterial*) m_materialManager->getMaterial( "MPM", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  contactModel->outputProblemSpec(mpm_ps);
  thermalContactModel->outputProblemSpec(mpm_ps);

  for (unsigned int i = 0; i < m_materialManager->getNumMatls( "CZ" );i++) {
    CZMaterial* mat = (CZMaterial*) m_materialManager->getMaterial( "CZ", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  ProblemSpecP physical_bc_ps = root->appendChild("PhysicalBC");
  ProblemSpecP mpm_ph_bc_ps = physical_bc_ps->appendChild("MPM");
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
    MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->outputProblemSpec(mpm_ph_bc_ps);
  }

  //__________________________________
  //  output data analysis modules. Mpmice or mpmarches handles this
  if(!flags->d_with_ice && !flags->d_with_arches && d_analysisModules.size() != 0){

    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;

      am->outputProblemSpec( root_ps );
    }
  }

}

void SingleFieldMPM::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())) {
    return;
  }
  Task* t = scinew Task( "MPM::actuallyInitialize", this, &SingleFieldMPM::actuallyInitialize );

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
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->pRefinedLabel);
  t->computes(lb->delTLabel,level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);
  t->computes(lb->NC_CCweightLabel,zeroth_matl);
  t->computes(lb->pSurfLabel);
  t->computes(lb->pSurfGradLabel);

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

  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addInitialComputesAndRequires( t, mpm_matl );
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addInitialComputesAndRequires( t, mpm_matl );
  }

  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));

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

  unsigned int numCZM = m_materialManager->getNumMatls( "CZ" );
  for(unsigned int m = 0; m < numCZM; m++){
    CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial( "CZ", m);
    CohesiveZone* ch = cz_matl->getCohesiveZone();
    ch->scheduleInitialize(level, sched, cz_matl);
  }

  if (flags->d_deleteGeometryObjects) {
    scheduleDeleteGeometryObjects(level, sched);
  }

}
//______________________________________________________________________
//
void SingleFieldMPM::scheduleRestartInitialize(const LevelP& level,
                                          SchedulerP& sched)
{
}
/* _____________________________________________________________________
 Purpose:   Set variables that are normally set during the initialization
            phase, but get wiped clean when you restart
_____________________________________________________________________*/
void SingleFieldMPM::restartInitialize()
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
void SingleFieldMPM::schedulePrintParticleCount(const LevelP& level,
                                           SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &SingleFieldMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "MPM" ));
}
//__________________________________
//  Diagnostic task: compute the total number of particles
void SingleFieldMPM::scheduleTotalParticleCount(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())){
    return;
  }

  Task* t = scinew Task("SingleFieldMPM::totalParticleCount",
                  this, &SingleFieldMPM::totalParticleCount);
  t->computes(lb->partCountLabel);

  sched->addTask(t, patches,matls);
}
//__________________________________
//  Diagnostic task: compute the total number of particles
void SingleFieldMPM::totalParticleCount(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    long int totalParticles = 0;

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      int numParticles  = pset->end() - pset->begin();

      totalParticles+=numParticles;
    }
    new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
  }
}

void SingleFieldMPM::scheduleInitializePressureBCs(const LevelP& level,
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
                          this, &SingleFieldMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, d_loadCurveIndex,
                Task::OutOfDomain);
    sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializePressureBC",
                    this, &SingleFieldMPM::initializePressureBC);
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
    sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
  }

  if(d_loadCurveIndex->removeReference())
    delete d_loadCurveIndex;
}

void SingleFieldMPM::scheduleDeleteGeometryObjects(const LevelP& level,
                                              SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

  Task* t = scinew Task("MPM::deleteGeometryObjects",
                  this, &SingleFieldMPM::deleteGeometryObjects);
  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
}

void SingleFieldMPM::scheduleComputeStableTimeStep(const LevelP& level,
                                                   SchedulerP& sched)
{
  // Nothing to do here - delt is computed as a by-product of the
  // constitutive model
  // However, this task needs to do something in the case that MPM
  // is being run on more than one level.
  Task* t = 0;
  cout_doing << d_myworld->myRank() << " MPM::scheduleComputeStableTimeStep \t\t\t\tL-" <<level->getIndex() << endl;

  t = scinew Task("MPM::actuallyComputeStableTimestep",
                   this, &SingleFieldMPM::actuallyComputeStableTimestep);

  const MaterialSet* mpm_matls = m_materialManager->allMaterials( "MPM" );

  t->computes(lb->delTLabel,level.get_rep());
  sched->addTask(t,level->eachPatch(), mpm_matls);
}

void
SingleFieldMPM::scheduleTimeAdvance(const LevelP & level,
                               SchedulerP   & sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = m_materialManager->allMaterials( "MPM" );
//  const MaterialSet* cz_matls = m_materialManager->allMaterials( "CZ" );
//  const MaterialSet* all_matls = m_materialManager->allMaterials();

// const MaterialSubset* mpm_matls_sub = (   matls ?    matls->getUnion() : nullptr);;
// const MaterialSubset*  cz_matls_sub = (cz_matls ? cz_matls->getUnion() : nullptr);

  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleInterpolateSurfaceToGrid(       sched, patches, matls);
  scheduleComputeParticleSurfaceGradient( sched, patches, matls);
  scheduleComputeGridSurfaceGradient(     sched, patches, matls);

//#ifdef RIGID
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
//#endif
//  if(flags->d_computeNormals){
//    scheduleComputeNormals(               sched, patches, matls);
//  }
//  scheduleExMomInterpolated(              sched, patches, matls);
//  if(flags->d_useCohesiveZones){
//    scheduleUpdateCohesiveZones(          sched, patches, mpm_matls_sub,
//                                                          cz_matls_sub,
//                                                          all_matls);
//    scheduleAddCohesiveZoneForces(        sched, patches, mpm_matls_sub,
//                                                          cz_matls_sub,
//                                                          all_matls);
//  }
//  if(d_bndy_traction_faces.size()>0) {
//    scheduleComputeContactArea(           sched, patches, matls);
//  }
  scheduleComputeInternalForce(           sched, patches, matls);

  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
  scheduleSingleFieldContact(             sched, patches, matls);
//  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
//  if (flags->d_prescribeDeformation){
//    scheduleSetPrescribedMotion(          sched, patches, matls);
//  }
//  if(flags->d_XPIC2){
//    scheduleComputeSSPlusVp(              sched, patches, matls);
//    scheduleComputeSPlusSSPlusVp(         sched, patches, matls);
//  }
//  if(flags->d_doExplicitHeatConduction){
//    scheduleComputeHeatExchange(          sched, patches, matls);
//    scheduleComputeInternalHeatRate(      sched, patches, matls);
//    //scheduleComputeNodalHeatFlux(       sched, patches, matls);
//    scheduleSolveHeatEquations(           sched, patches, matls);
//    scheduleIntegrateTemperatureRate(     sched, patches, matls);
//  }
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleComputeParticleGradients(       sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleFinalParticleUpdate(            sched, patches, matls);
//  scheduleInsertParticles(                sched, patches, matls);
  if(flags->d_computeScaleFactor){
    scheduleComputeParticleScaleFactor(   sched, patches, matls);
  }
//  if(flags->d_refineParticles){
//    scheduleAddParticles(                 sched, patches, matls);
//  }

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis_preReloc( sched, level);
    }
  }

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_particleState_preReloc,
                                    lb->pXLabel,
                                    d_particleState,
				    lb->pParticleIDLabel, matls, 1);

#if 0
 if(flags->d_useCohesiveZones){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_cohesiveZoneState_preReloc,
                                    lb->pXLabel,
                                    d_cohesiveZoneState,
                                    lb->czIDLabel, cz_matls,2);
  }

#endif
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

void SingleFieldMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleApplyExternalLoads");

  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &SingleFieldMPM::applyExternalLoads);

  t->requires(Task::OldDW, lb->simulationTimeLabel);

  if (!flags->d_mms_type.empty()) {
    //MMS problems need displacements
    t->requires(Task::OldDW, lb->pDispLabel,            Ghost::None);
  }

  if (flags->d_useLoadCurves || flags->d_useCBDI) {
    t->requires(Task::OldDW,    lb->pXLabel,                  Ghost::None);
    t->requires(Task::OldDW,    lb->pLoadCurveIDLabel,        Ghost::None);
    t->computes(                lb->pLoadCurveIDLabel_preReloc);
    if (flags->d_useCBDI) {
       t->requires(Task::OldDW, lb->pSizeLabel,               Ghost::None);
       t->requires(Task::OldDW, lb->pDeformationMeasureLabel, Ghost::None);
       t->computes(             lb->pExternalForceCorner1Label);
       t->computes(             lb->pExternalForceCorner2Label);
       t->computes(             lb->pExternalForceCorner3Label);
       t->computes(             lb->pExternalForceCorner4Label);
    }
  }
//  t->computes(Task::OldDW, lb->pExternalHeatRateLabel_preReloc);
  t->computes(             lb->pExtForceLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateParticlesToGrid");

  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&SingleFieldMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
//  t->requires(Task::OldDW, lb->pColorLabel,            gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  if (flags->d_GEVelProj) {
    t->requires(Task::OldDW, lb->pVelGradLabel,             gan,NGP);
    t->requires(Task::OldDW, lb->pTemperatureGradientLabel, gan,NGP);
  }
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

  t->computes(lb->gMassLabel,        m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain);
  t->computes(lb->gVolumeLabel,      m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain);
  t->computes(lb->gVelocityLabel,    m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain);
  t->computes(lb->gMassLabel);
  t->computes(lb->gSp_volLabel);
  t->computes(lb->gVolumeLabel);
//  t->computes(lb->gColorLabel);
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

void SingleFieldMPM::scheduleInterpolateSurfaceToGrid(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateSurfaceToGrid");

  Task* t = scinew Task("MPM::interpolateSurfaceToGrid",
                        this,&SingleFieldMPM::interpolateSurfaceToGrid);

  Ghost::GhostType  gan = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pXLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pMassLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pSurfLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan, NGP);

  t->computes(lb->gSurfLabel,     d_one_matl);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::interpolateSurfaceToGrid(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing interpolateSurfaceToGrid");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    Ghost::GhostType  gan = Ghost::AroundNodes;

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    NCVariable<double>  gsurf, gmassAll;
    new_dw->allocateAndPut(gsurf,     lb->gSurfLabel,      0, patch);
    new_dw->allocateTemporary(gmassAll,     patch,gan,NGP);
    gsurf.initialize(0.0);
    gmassAll.initialize(d_SMALL_NUM_MPM);

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pmass, psurf;
      constParticleVariable<Matrix3> psize, pFOld;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(psurf,          lb->pSurfLabel,          pset);
      old_dw->get(psize,          lb->pSizeLabel,          pset);
      old_dw->get(pFOld,          lb->pDeformationMeasureLabel,pset);

      //loop over all particles in the patch:
      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        IntVector node;
#if 0
        int NN = 8;
        IntVector ni8[8];
        patch->findCellNodes(px[idx],ni8);
        // Iterate through the nodes that receive data from the current particle
        for(int k = 0; k < NN; k++) {
          node = ni8[k];
          if(patch->containsNode(node)) {
             gsurf[ni8[k]] = max(gsurf[ni8[k]], psurf[idx]);
          }
        }
#endif
#if 1
        int NN =
           interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);
        for(int k = 0; k < NN; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
             gsurf[ni[k]]        += psurf[idx]* pmass[idx] * S[k];
             gmassAll[ni[k]]     +=             pmass[idx] * S[k];
          }
        }
#endif
      } // End of particle loop
    }  // End loop over materials

#if 1
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gsurf[c] /= gmassAll[c];
    }
#endif
    delete interpolator;
  }  // End loop over patches
}

void SingleFieldMPM::scheduleComputeSSPlusVp(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeSSPlusVp");

  Task* t=scinew Task("MPM::computeSSPlusVp",
                      this, &SingleFieldMPM::computeSSPlusVp);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);

  t->requires(Task::NewDW, lb->gVelocityLabel,                  gac,NGN);

  t->computes(lb->pVelocitySSPlusLabel);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeSPlusSSPlusVp(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeSPlusSSPlusVp");

  Task* t=scinew Task("MPM::computeSPlusSSPlusVp",
                      this, &SingleFieldMPM::computeSPlusSSPlusVp);

  Ghost::GhostType gan = Ghost::AroundNodes;
  Ghost::GhostType gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                     gan, NGP);
  t->requires(Task::OldDW, lb->pMassLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,    gan, NGP);
  t->requires(Task::NewDW, lb->pVelocitySSPlusLabel,        gan, NGP);
  t->requires(Task::NewDW, lb->gMassLabel,                  gac, NGN);

  t->computes(lb->gVelSPSSPLabel);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleAddCohesiveZoneForces(SchedulerP& sched,
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
                        this,&SingleFieldMPM::addCohesiveZoneForces);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                     cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czForceLabel_preReloc,       cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czTopMatLabel_preReloc,      cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->czBotMatLabel_preReloc,      cz_matls, gan,NGP);
  t->requires(Task::NewDW, lb->gMassLabel,                  mpm_matls,gac,NGN);

  t->modifies(lb->gExternalForceLabel, mpm_matls);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeHeatExchange(SchedulerP& sched,
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

void SingleFieldMPM::scheduleExMomInterpolated(SchedulerP& sched,
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
void SingleFieldMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeStressTensor");

  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &SingleFieldMPM::computeStressTensor);
  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);

    t->computes(lb->p_qLabel_preReloc, matlset);
  }

  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->computes(lb->delTLabel,getLevel(patches));

  if (flags->d_reductionVars->accStrainEnergy ||
      flags->d_reductionVars->strainEnergy) {
    t->computes(lb->StrainEnergyLabel);
  }

  sched->addTask(t, patches, matls);
  
  //__________________________________
  //  Additional tasks
  scheduleUpdateStress_DamageErosionModels( sched, patches, matls );

  if (flags->d_reductionVars->accStrainEnergy)
    scheduleComputeAccStrainEnergy(sched, patches, matls);

}


//______________________________________________________________________
//
// Compute the accumulated strain energy
void SingleFieldMPM::scheduleComputeAccStrainEnergy(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeAccStrainEnergy");

  Task* t = scinew Task("MPM::computeAccStrainEnergy",
                        this, &SingleFieldMPM::computeAccStrainEnergy);
  t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
  t->requires(Task::NewDW, lb->StrainEnergyLabel);
  t->computes(lb->AccStrainEnergyLabel);
  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeContactArea(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  /** computeContactArea */

    printSchedule(patches,cout_doing,"MPM::scheduleComputeContactArea");
    Task* t = scinew Task("MPM::computeContactArea",
                          this, &SingleFieldMPM::computeContactArea);

    Ghost::GhostType  gnone = Ghost::None;
    t->requires(Task::NewDW, lb->gVolumeLabel, gnone);
    for(std::list<Patch::FaceType>::const_iterator ftit(d_bndy_traction_faces.begin());
        ftit!=d_bndy_traction_faces.end();ftit++) {
      int iface = (int)(*ftit);
      t->computes(lb->BndyContactCellAreaLabel[iface]);
    }
    sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalForce");

  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &SingleFieldMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW,lb->pDeformationMeasureLabel,   gan,NGP);

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
//    t->requires(Task::NewDW, lb->BndyContactCellAreaLabel[iface]);
    t->computes(lb->BndyForceLabel[iface]);
    t->computes(lb->BndyContactAreaLabel[iface]);
    t->computes(lb->BndyTractionLabel[iface]);
  }

  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalHeatRate");
  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}

void SingleFieldMPM::scheduleComputeNodalHeatFlux(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeNodalHeatFlux");
  heatConductionModel->scheduleComputeNodalHeatFlux(sched,patches,matls);
}

void SingleFieldMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSolveHeatEquations");
  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void SingleFieldMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &SingleFieldMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, lb->delTLabel );

//  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
//  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
//  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
//  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->requires(Task::NewDW, lb->gMassF0Label,           d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityF0Label,       d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceF0Label,  d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceF0Label,  d_one_matl, Ghost::None);

  t->requires(Task::NewDW, lb->gMassF1Label,           d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityF1Label,       d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceF1Label,  d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceF1Label,  d_one_matl, Ghost::None);

//  t->computes(lb->gVelocityStarLabel);
//  t->computes(lb->gAccelerationLabel);

  t->computes(lb->gVelocityStarF0Label, d_one_matl);
  t->computes(lb->gVelocityStarF1Label, d_one_matl);
  t->computes(lb->gAccelerationF0Label, d_one_matl);
  t->computes(lb->gAccelerationF1Label, d_one_matl);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleIntegrateTemperatureRate");
  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

void SingleFieldMPM::scheduleExMomIntegrated(SchedulerP& sched,
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

void SingleFieldMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSetGridBoundaryConditions");
  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &SingleFieldMPM::setGridBoundaryConditions);

  t->requires(Task::OldDW, lb->delTLabel );

  t->modifies(             lb->gAccelerationF0Label,   d_one_matl);
  t->modifies(             lb->gVelocityStarF0Label,   d_one_matl);
  t->requires(Task::NewDW, lb->gVelocityF0Label,       d_one_matl, Ghost::None);
#ifdef RIGID
  t->requires(Task::NewDW, lb->gMassLabel,             d_one_matl, Ghost::None);
#endif

  t->modifies(             lb->gAccelerationF1Label,   d_one_matl);
  t->modifies(             lb->gVelocityStarF1Label,   d_one_matl);
  t->requires(Task::NewDW, lb->gVelocityF1Label,       d_one_matl, Ghost::None);

//  const MaterialSubset* mss = matls->getUnion();
//  t->modifies(             lb->gAccelerationLabel,     mss);
//  t->modifies(             lb->gVelocityStarLabel,     mss);
//  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);
//
//  if(!flags->d_doGridReset){
//    t->requires(Task::OldDW, lb->gDisplacementLabel,    Ghost::None);
//    t->computes(lb->gDisplacementLabel);
//  }

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateToParticlesAndUpdate");

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &SingleFieldMPM::interpolateToParticlesAndUpdate);

  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::OldDW, lb->simulationTimeLabel);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
//  t->requires(Task::NewDW, lb->gAccelerationLabel,               gac,NGN);
//  t->requires(Task::NewDW, lb->gVelocityStarLabel,               gac,NGN);
//  t->requires(Task::NewDW, lb->gTemperatureRateLabel,            gac,NGN);
//  t->requires(Task::NewDW, lb->frictionalWorkLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gAccelerationF0Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gAccelerationF1Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarF0Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarF1Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gSurfGradLabel,       d_one_matl, gac,NGN);
//  if(flags->d_XPIC2){
//    t->requires(Task::NewDW, lb->gVelSPSSPLabel,                gac,NGN);
//    t->requires(Task::NewDW, lb->pVelocitySSPlusLabel,          gnone);
//  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);
  t->requires(Task::OldDW, lb->pSurfLabel,                      gnone);
  t->requires(Task::NewDW, lb->pSurfGradLabel_preReloc,         gnone);

//  if(flags->d_with_ice){
//    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
//    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
//  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pSurfLabel_preReloc);

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

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  // Carry Forward particle refinement flag
  if(flags->d_refineParticles){
    t->requires(Task::OldDW, lb->pRefinedLabel,                Ghost::None);
    t->computes(             lb->pRefinedLabel_preReloc);
  }

  t->requires(Task::OldDW, lb->NC_CCweightLabel, d_one_matl, Ghost::None);
  t->computes(             lb->NC_CCweightLabel, d_one_matl);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleComputeParticleGradients(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeParticleGradients");

  Task* t=scinew Task("MPM::computeParticleGradients",
                      this, &SingleFieldMPM::computeParticleGradients);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
//  t->requires(Task::NewDW, lb->gVelocityStarLabel,               gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarF0Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarF1Label, d_one_matl, gac,NGN);
  t->requires(Task::NewDW, lb->gSurfGradLabel,       d_one_matl, gac,NGN);
//  if (flags->d_doExplicitHeatConduction){
//    t->requires(Task::NewDW, lb->gTemperatureStarLabel,         gac,NGN);
//  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);
  t->requires(Task::NewDW, lb->pSurfGradLabel_preReloc,         gnone);

  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pTemperatureGradientLabel_preReloc);

  if(flags->d_reductionVars->volDeformed){
    t->computes(lb->TotalVolumeDeformedLabel);
  }

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleFinalParticleUpdate(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleFinalParticleUpdate");

  Task* t=scinew Task("MPM::finalParticleUpdate",
                      this, &SingleFieldMPM::finalParticleUpdate);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->pdTdtLabel,                      gnone);
  t->requires(Task::NewDW, lb->pLocalizedMPMLabel_preReloc,     gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::scheduleUpdateCohesiveZones(SchedulerP& sched,
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
                      this, &SingleFieldMPM::updateCohesiveZones);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityLabel,     mpm_matls,   gac,NGN);
  t->requires(Task::NewDW, lb->gMassLabel,         mpm_matls,   gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,            cz_matls,    gnone);
  t->requires(Task::OldDW, lb->czAreaLabel,        cz_matls,    gnone);
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
  t->computes(lb->czAreaLabel_preReloc,       cz_matls);
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

void SingleFieldMPM::scheduleInsertParticles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  if(flags->d_insertParticles){
    printSchedule(patches,cout_doing,"MPM::scheduleInsertParticles");

    Task* t=scinew Task("MPM::insertParticles",this,
                  &SingleFieldMPM::insertParticles);

    t->requires(Task::OldDW, lb->simulationTimeLabel);
    t->requires(Task::OldDW, lb->delTLabel );

    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);

    sched->addTask(t, patches, matls);
  }
}

void SingleFieldMPM::scheduleAddParticles(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls)

{
  if( !flags->doMPMOnLevel( getLevel(patches)->getIndex(), getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleAddParticles" );

  Task * t = scinew Task("MPM::addParticles", this, &SingleFieldMPM::addParticles );

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->modifies(lb->pParticleIDLabel_preReloc);
  t->modifies(lb->pXLabel_preReloc);
  t->modifies(lb->pVolumeLabel_preReloc);
  t->modifies(lb->pVelocityLabel_preReloc);
  t->modifies(lb->pMassLabel_preReloc);
  t->modifies(lb->pSizeLabel_preReloc);
  t->modifies(lb->pDispLabel_preReloc);
  t->modifies(lb->pStressLabel_preReloc);
  
  if (flags->d_with_color) {
    t->modifies(lb->pColorLabel_preReloc);
  }
  if (flags->d_useLoadCurves) {
    t->modifies(lb->pLoadCurveIDLabel_preReloc);
  }
  t->modifies(lb->pLocalizedMPMLabel_preReloc);
  t->modifies(lb->pExtForceLabel_preReloc);
  t->modifies(lb->pTemperatureLabel_preReloc);
  t->modifies(lb->pTemperatureGradientLabel_preReloc);
  t->modifies(lb->pTempPreviousLabel_preReloc);
  t->modifies(lb->pDeformationMeasureLabel_preReloc);
  t->modifies(lb->pRefinedLabel_preReloc);
  if(flags->d_computeScaleFactor){
    t->modifies(lb->pScaleFactorLabel_preReloc);
  }
  t->modifies(lb->pVelGradLabel_preReloc);

  t->requires(Task::OldDW, lb->pCellNAPIDLabel, zeroth_matl, Ghost::None);
  t->computes(             lb->pCellNAPIDLabel, zeroth_matl);

  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

void
SingleFieldMPM::scheduleComputeParticleScaleFactor(       SchedulerP  & sched,
                                               const PatchSet    * patches,
                                               const MaterialSet * matls )
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleComputeParticleScaleFactor" );

  Task * t = scinew Task( "MPM::computeParticleScaleFactor",this, &SingleFieldMPM::computeParticleScaleFactor );

  t->requires( Task::NewDW, lb->pSizeLabel_preReloc,                Ghost::None );
  t->requires( Task::NewDW, lb->pDeformationMeasureLabel_preReloc,  Ghost::None );
  t->computes( lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void
SingleFieldMPM::scheduleSetPrescribedMotion(       SchedulerP  & sched,
                                        const PatchSet    * patches,
                                        const MaterialSet * matls )
{
  if ( !flags->doMPMOnLevel( getLevel(patches)->getIndex(), 
        getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule(patches,cout_doing,"MPM::scheduleSetPrescribedMotion");

  Task * t = scinew Task( "MPM::setPrescribedMotion", this,
                           &SingleFieldMPM::setPrescribedMotion );

  const MaterialSubset* mss = matls->getUnion();
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->delTLabel );

  sched->addTask(t, patches, matls);
}

void
SingleFieldMPM::scheduleRefine( const PatchSet   * patches,
                                 SchedulerP & sched )
{
  printSchedule(patches,cout_doing,"MPM::scheduleRefine");
  Task* t = scinew Task( "SingleFieldMPM::refine", this, &SingleFieldMPM::refine );

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
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->NC_CCweightLabel);
  t->computes(lb->delTLabel,getLevel(patches));

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

  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addInitialComputesAndRequires(t, mpm_matl);
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addInitialComputesAndRequires(t, mpm_matl);
  }

  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
}

void
SingleFieldMPM::scheduleRefineInterface( const LevelP& /*fineLevel*/,
                                          SchedulerP& /*scheduler*/,
                                          bool /* ??? */,
                                          bool /* ??? */)
{
  //  do nothing for now
}

void SingleFieldMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/,
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}
//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void SingleFieldMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (amr_doing.active())
    amr_doing << "SingleFieldMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // The simulation controller should not schedule it every time step
  Task* task = scinew Task("MPM::errorEstimate", this, &SingleFieldMPM::errorEstimate);

  // if the finest level, compute flagged cells
  if (coarseLevel->getIndex() == coarseLevel->getGrid()->numLevels()-1) {
    task->requires(Task::NewDW, lb->pXLabel, Ghost::AroundCells, 0);
  }
  else {
    task->requires(Task::NewDW, m_regridder->getRefineFlagLabel(),
                   0, Task::FineLevel, m_regridder->refineFlagMaterials(),
                   Task::NormalDomain, Ghost::None, 0);
  }
  task->modifies(m_regridder->getRefineFlagLabel(),      m_regridder->refineFlagMaterials());
  task->modifies(m_regridder->getRefinePatchFlagLabel(), m_regridder->refineFlagMaterials());
  sched->addTask(task, coarseLevel->eachPatch(), m_materialManager->allMaterials( "MPM" ));

}
//______________________________________________________________________
// Schedule to mark initial flags for AMR regridding
void SingleFieldMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void SingleFieldMPM::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  if (d_switchCriteria) {
    d_switchCriteria->scheduleSwitchTest(level,sched);
  }
}
//______________________________________________________________________
//
void SingleFieldMPM::printParticleCount(const ProcessorGroup* pg,
                                   const PatchSubset*,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  sumlong_vartype pcount;
  new_dw->get(pcount, lb->partCountLabel);

  if(pg->myRank() == 0){
    std::cout << "Created " << (long) pcount << " total particles" << std::endl;
  }

  //__________________________________
  //  bulletproofing
  if(pcount == 0 && flags->d_with_arches == false){
    ostringstream msg;
    msg << "\n ERROR: zero particles were created. \n"
        << "  Possible causes: \n"
        << "    1) The geom_objects are outside of the computational domain.\n"
        << "    2) Insufficient grid resolution.  On single/multi-level (MPMICE) problems particles have to created\n"
        << "       on the coarsest level for each geom_object.";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
}
//______________________________________________________________________
//
void SingleFieldMPM::computeAccStrainEnergy(const ProcessorGroup*,
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
void SingleFieldMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
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
        unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
        int numPts = 0;
        for(unsigned int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<IntVector> pLoadCurveID;
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            for(int k = 0;k<3;k++){
              if (pLoadCurveID[idx](k) == (nofPressureBCs)){
                ++numPts;
              }
            }
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts),
                    lb->materialPointsPerLoadCurveLabel, 0, nofPressureBCs-1);
      }  // patch loop
    }
  }
}

// Calculate the number of material points per load curve
void SingleFieldMPM::initializePressureBC(const ProcessorGroup*,
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
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      constParticleVariable<IntVector> pLoadCurveID;
      ParticleVariable<Vector> pExternalForce;
      new_dw->get(px,    lb->pXLabel, pset);
      new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
      new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);

      ParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                              pExternalForceCorner3, pExternalForceCorner4;
      if (flags->d_useCBDI) {
        new_dw->get(psize,               lb->pSizeLabel,               pset);
        new_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);
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
            pExternalForce[idx] = Vector(0.,0.,0.);
            for(int k=0;k<3;k++){
             if (pLoadCurveID[idx](k) == nofPressureBCs) {
              if (flags->d_useCBDI) {
               Vector dxCell = patch->dCell();
               pExternalForce[idx] +=pbc->getForceVectorCBDI(px[idx],psize[idx],
                                    pDeformationMeasure[idx],forcePerPart,time,
                                    pExternalForceCorner1[idx],
                                    pExternalForceCorner2[idx],
                                    pExternalForceCorner3[idx],
                                    pExternalForceCorner4[idx],
                                    dxCell);
              } else {
               pExternalForce[idx] += pbc->getForceVector(px[idx],
                                                        forcePerPart,time);
              }// if CBDI
            } // if pLoadCurveID...
           } // Loop over elements of the loadCurveID IntVector
          }  // loop over particles
        }   // if pressure loop
      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

void SingleFieldMPM::deleteGeometryObjects(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* ,
                                      DataWarehouse* new_dw)
{
  cout << "Deleting Geometry Objects " << endl;
  unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMPMMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
    mpm_matl->deleteGeomObjects();
  }
}

void SingleFieldMPM::actuallyInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing SingleFieldMPM::actuallyInitialize");

    CCVariable<int> cellNAPID;
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
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      particleIndex numParticles = mpm_matl->createParticles(cellNAPID,
                                                             patch, new_dw);

      totalParticles+=numParticles;
      mpm_matl->getConstitutiveModel()->initializeCMData(patch,mpm_matl,new_dw);
      
      //initialize Damage model
      mpm_matl->getDamageModel()->initializeLabels( patch, mpm_matl, new_dw );
      
      mpm_matl->getErosionModel()->initializeLabels( patch, mpm_matl, new_dw );
    }
  } // patches

  const Level* level = getLevel(patches);
  IntVector lowNode, highNode;
  level->findInteriorNodeIndexRange(lowNode, highNode);
  string interp_type = flags->d_interpolator_type;

  // Determine dimensionality for particle splitting
  // To be recognized as 2D, must be in the x-y plane
  d_ndim=3;
  if(highNode.z() - lowNode.z()==2) {
     d_ndim=2;
  }

  // Only allow axisymmetric runs if the grid is one cell
  // thick in the theta dir.
  if(flags->d_axisymmetric){
    int num_cells_in_theta = (highNode.z() - lowNode.z()) - 1;
    if(num_cells_in_theta > 1 ){
     ostringstream msg;
      msg << "\n ERROR: When using <axisymmetric>true</axisymmetric> the \n"
          << "grid can only have one cell in the circumferential direction.\n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }

  // Bulletproofing for extra cells/interpolators/periodic BCs
  IntVector num_extra_cells=level->getExtraCells();
  IntVector periodic=level->getPeriodicBoundaries();
  if(interp_type=="linear" && num_extra_cells!=IntVector(0,0,0)){
    if(!flags->d_with_ice && !flags->d_with_arches){
      ostringstream msg;
      msg << "\n ERROR: When using <interpolator>linear</interpolator> \n"
          << " you should also use <extraCells>[0,0,0]</extraCells> \n"
          << " unless you are running an MPMICE or MPMARCHES case.\n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }
  else if(((interp_type=="gimp"       ||
            interp_type=="3rdorderBS" ||
            interp_type=="fast_cpdi" ||
            interp_type=="cpdi")                          && 
            (  (num_extra_cells+periodic)!=IntVector(1,1,1) && 
            (!((num_extra_cells+periodic)==IntVector(1,1,0) && 
             flags->d_axisymmetric))))){
      ostringstream msg;
      msg << "\n ERROR: When using <interpolator>gimp</interpolator> \n"
          << " or <interpolator>3rdorderBS</interpolator> \n"
          << " or <interpolator>cpdi</interpolator> \n"
          << " or <interpolator>fast_cpdi</interpolator> \n"
          << " you must also use extraCells and/or periodicBCs such\n"
          << " that the sum of the two is [1,1,1].\n"
          << " If using axisymmetry, the sum of the two can be [1,1,0].\n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }

  if (flags->d_reductionVars->accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);

}

void SingleFieldMPM::readPrescribedDeformations(string filename)
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

void SingleFieldMPM::readInsertParticlesFile(string filename)
{

 if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening particle insertion file '"+filename+"'\n",
                                  __FILE__, __LINE__);
    }
    while(is) {
        double t1,color,transx,transy,transz,v_new_x,v_new_y,v_new_z;
        is >> t1 >> color >> transx >> transy >> transz >> v_new_x >> v_new_y >> v_new_z;
        if(is) {
            d_IPTimes.push_back(t1);
            d_IPColor.push_back(color);
            d_IPTranslate.push_back(Vector(transx,transy,transz));
            d_IPVelNew.push_back(Vector(v_new_x,v_new_y,v_new_z));
        }
    }
  }
}

void SingleFieldMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* ,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  // Put something here to satisfy the need for a reduction operation in
  // the case that there are multiple levels present
  const Level* level = getLevel(patches);
  new_dw->put(delt_vartype(1.0e10), lb->delTLabel, level);
}

void SingleFieldMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,
              "Doing SingleFieldMPM::interpolateParticlesToGrid");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    ParticleInterpolator* linear_interpolator=scinew LinearInterpolator(patch);

    string interp_type = flags->d_interpolator_type;

    NCVariable<double> gmassglobal,gtempglobal,gvolumeglobal;
    NCVariable<Vector> gvelglobal;
    int globMatID = m_materialManager->getAllInOneMatls()->get(0);
    new_dw->allocateAndPut(gmassglobal,  lb->gMassLabel,       globMatID,patch);
    new_dw->allocateAndPut(gtempglobal,  lb->gTemperatureLabel,globMatID,patch);
    new_dw->allocateAndPut(gvolumeglobal,lb->gVolumeLabel,     globMatID,patch);
    new_dw->allocateAndPut(gvelglobal,   lb->gVelocityLabel,   globMatID,patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gvolumeglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
    gvelglobal.initialize(Vector(0.0));
    Ghost::GhostType  gan = Ghost::AroundNodes;

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature, pColor;
      constParticleVariable<Vector> pvelocity, pexternalforce;
      constParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                                   pExternalForceCorner3, pExternalForceCorner4;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pFOld;
      constParticleVariable<Matrix3> pVelGrad;
      constParticleVariable<Vector>  pTempGrad;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
//    old_dw->get(pColor,         lb->pColorLabel,         pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      if (flags->d_GEVelProj){
        old_dw->get(pVelGrad,     lb->pVelGradLabel,             pset);
        old_dw->get(pTempGrad,    lb->pTemperatureGradientLabel, pset);
      }
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(psize,          lb->pSizeLabel,          pset);
      old_dw->get(pFOld,          lb->pDeformationMeasureLabel,pset);
      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);
      constParticleVariable<IntVector> pLoadCurveID;
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
//    NCVariable<double> gColor;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gTemperatureRate;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
//    new_dw->allocateAndPut(gColor,           lb->gColorLabel,      dwi,patch);
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
//      gColor.initialize(0.0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gTemperatureRate.initialize(0);
      gexternalheatrate.initialize(0);
      gSp_vol.initialize(0.);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      Vector total_mom(0.0,0.0,0.0);
      double pSp_vol = 1./mpm_matl->getInitialDensity();
      //loop over all particles in the patch:
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end();
           iter++){
        particleIndex idx = *iter;
        int NN =
           interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);
        Vector pmom = pvelocity[idx]*pmass[idx];
        double ptemp_ext = pTemperature[idx];
        total_mom += pmom;

        // Add each particles contribution to the local mass & velocity
        // Must use the node indices
        IntVector node;
        // Iterate through the nodes that receive data from the current particle
        for(int k = 0; k < NN; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
            if (flags->d_GEVelProj){
              Point gpos = patch->getNodePosition(node);
              Vector distance = px[idx] - gpos;
              Vector pvel_ext = pvelocity[idx] - pVelGrad[idx]*distance;
              pmom = pvel_ext*pmass[idx];
              ptemp_ext = pTemperature[idx] - Dot(pTempGrad[idx],distance);
            }
            gmass[node]          += pmass[idx]                     * S[k];
            gvelocity[node]      += pmom                           * S[k];
            gvolume[node]        += pvolume[idx]                   * S[k];
//            gColor[node]         += pColor[idx]*pmass[idx]         * S[k];
            if (!flags->d_useCBDI) {
              gexternalforce[node] += pexternalforce[idx]          * S[k];
            }
            gTemperature[node]   += ptemp_ext * pmass[idx] * S[k];
            gSp_vol[node]        += pSp_vol   * pmass[idx] * S[k];
            //gexternalheatrate[node] += pexternalheatrate[idx]      * S[k];
          }
        }
        if (flags->d_useCBDI && pLoadCurveID[idx].x()>0) {
          vector<IntVector> niCorner1(linear_interpolator->size());
          vector<IntVector> niCorner2(linear_interpolator->size());
          vector<IntVector> niCorner3(linear_interpolator->size());
          vector<IntVector> niCorner4(linear_interpolator->size());
          vector<double> SCorner1(linear_interpolator->size());
          vector<double> SCorner2(linear_interpolator->size());
          vector<double> SCorner3(linear_interpolator->size());
          vector<double> SCorner4(linear_interpolator->size());
          linear_interpolator->findCellAndWeights(pExternalForceCorner1[idx],
                                 niCorner1,SCorner1,psize[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner2[idx],
                                 niCorner2,SCorner2,psize[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner3[idx],
                                 niCorner3,SCorner3,psize[idx]);
          linear_interpolator->findCellAndWeights(pExternalForceCorner4[idx],
                                 niCorner4,SCorner4,psize[idx]);
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
//        gColor[c]         /= gmass[c];
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

void SingleFieldMPM::computeSSPlusVp(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeSSPlusVp");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Vector> pvelSSPlus;
      constNCVariable<Vector> gvelocity;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,       lb->pXLabel,                         pset);
      new_dw->get(psize,    lb->pCurSizeLabel,                   pset);
      old_dw->get(pFOld,    lb->pDeformationMeasureLabel,        pset);

      new_dw->allocateAndPut(pvelSSPlus,lb->pVelocitySSPlusLabel,    pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity,    lb->gVelocityLabel,   dwi,patch,gac,NGP);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(px[idx], ni, S, psize[idx]);
        // Accumulate the contribution from each surrounding vertex
        Vector vel(0.0,0.0,0.0);
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel      += gvelocity[node]  * S[k];
        }
        pvelSSPlus[idx]    = vel;
      }
    }
    delete interpolator;
  }
}

void SingleFieldMPM::computeSPlusSSPlusVp(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeSPlusSSPlusVp");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize, pFOld;
      constParticleVariable<Vector> pvelSSPlus;
      constParticleVariable<double> pmass;

      NCVariable<Vector> gvelSPSSP;
      constNCVariable<double> gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,         lb->pXLabel,                         pset);
      old_dw->get(pmass,      lb->pMassLabel,                      pset);
      new_dw->get(psize,      lb->pCurSizeLabel,                   pset);
      new_dw->get(pvelSSPlus, lb->pVelocitySSPlusLabel,            pset);
      new_dw->get(gmass,      lb->gMassLabel,         dwi,patch,gac,NGP);
      new_dw->allocateAndPut(gvelSPSSP,   lb->gVelSPSSPLabel,   dwi,patch);

      gvelSPSSP.initialize(Vector(0,0,0));

      // Loop over particles
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;
        int NN =
           interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);
        Vector pmom = pvelSSPlus[idx]*pmass[idx];

        IntVector node;
        for(int k = 0; k < NN; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
            gvelSPSSP[node] += pmom * S[k];
          }
        }
      } // End of particle loop
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        gvelSPSSP[c] /= gmass[c];
      }
    }
    delete interpolator;
  }
}

void SingleFieldMPM::addCohesiveZoneForces(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing addCohesiveZoneForces");

//    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    ParticleInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > gext_force(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->getModifiable(gext_force[m], lb->gExternalForceLabel, dwi, patch);
      new_dw->get(gmass[m],                lb->gMassLabel,dwi, patch, gac, NGN);
    }

    unsigned int numCZMatls=m_materialManager->getNumMatls( "CZ" );
    for(unsigned int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial( "CZ",  m );
      int dwi = cz_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      constParticleVariable<Vector> czforce;
      constParticleVariable<int> czTopMat, czBotMat;
      constParticleVariable<Matrix3> pDeformationMeasure;

      old_dw->get(czx,          lb->pXLabel,                          pset);
      new_dw->get(czforce,      lb->czForceLabel_preReloc,            pset);
      new_dw->get(czTopMat,     lb->czTopMatLabel_preReloc,           pset);
      new_dw->get(czBotMat,     lb->czBotMatLabel_preReloc,           pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(czx[idx],ni,S,size);

        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];

        double totMassTop = 0.;
        double totMassBot = 0.;
//        double sumSTop = 0.;
//        double sumSBot = 0.;

        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          totMassTop += S[k]*gmass[TopMat][node];
          totMassBot += S[k]*gmass[BotMat][node];
#if 0
          if(gmass[TopMat][node]>d_SMALL_NUM_MPM){
            sumSTop     += S[k];
          }
          if(gmass[BotMat][node]>d_SMALL_NUM_MPM){
            sumSBot     += S[k];
          }
#endif
        }

        // This currently contains three methods for distributing the CZ force
        // to the nodes.
        // The first of these distributes the force from the CZ
        // to the nodes based on a distance*mass weighting.  
        // The second distributes the force to the nodes that have mass,
        // but only uses distance weighting.  So, a node that is near the CZ
        // but relatively far from particles may get a large acceleration
        // compared to other nodes, thereby inducing a velocity gradient.
        // The third simply does a distance weighting from the CZ to the nodes.
        // For this version, it is possible that nodes with no material mass
        // will still acquire force from the CZ, leading to ~infinite
        // acceleration, and thus, badness.

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          if(patch->containsNode(node)) {
            // Distribute force according to material mass on the nodes
            // to get an approximately equal contribution to the acceleration
            gext_force[BotMat][node] += czforce[idx]*S[k]*gmass[BotMat][node]
                                                                 /totMassBot;
            gext_force[TopMat][node] -= czforce[idx]*S[k]*gmass[TopMat][node]
                                                                 /totMassTop;

//            gext_force[BotMat][node] += czforce[idx]*S[k]/sumSBot;
//            gext_force[TopMat][node] -= czforce[idx]*S[k]/sumSTop;

//            gext_force[BotMat][node] = gext_force[BotMat][node]
//                                     + czforce[idx] * S[k];
//            gext_force[TopMat][node] = gext_force[TopMat][node]
//                                     - czforce[idx] * S[k];
          }
        }
      }
#if 0
      // This is debugging output which is being left in for now (5/10/18)
      // as it may be helpful in generating figures for reports and papers.
      Vector sumForceTop = Vector(0.);
      Vector sumForceBot = Vector(0.);
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        if(gext_force[1][c].length() > 1.e-100){
           cout << "gEF_BM[" << c << "] = " << gext_force[1][c] 
                << ", " << gext_force[1][c]/gmass[1][c] << endl;
           sumForceBot += gext_force[1][c];
        }
        if(gext_force[2][c].length() > 1.e-100){
           cout << "gEF_BM[" << c << "] = " << gext_force[2][c]
                << ", " << gext_force[2][c]/gmass[2][c] << endl;
           sumForceTop += gext_force[2][c];
        }
        if(gext_force[1][c].length() > 1.e-100 &&
           gext_force[2][c].length() > 1.e-100){
           cout << "ratio = " << (gext_force[1][c].x()/gmass[1][c])/
                                 (gext_force[2][c].x()/gmass[2][c]) << endl;
        }
      }
      cout << "SFB = " << sumForceBot << endl;
      cout << "SFT = " << sumForceTop << endl;
#endif
    }
    delete interpolator;
  }
}

void SingleFieldMPM::computeStressTensor(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0),cout_doing,
            "Doing computeStressTensor");

  for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){

    if (cout_dbg.active()) {
      cout_dbg << " Patch = " << (patches->get(0))->getID();
      cout_dbg << " Mat = " << m;
    }

    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);

    if (cout_dbg.active())
      cout_dbg << " MPM_Mat = " << mpm_matl;

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

    if (cout_dbg.active())
      cout_dbg << " CM = " << cm;

    cm->setWorld(d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);

    if (cout_dbg.active())
      cout_dbg << " Exit\n" ;

  }
}


//______________________________________________________________________
//
void SingleFieldMPM::computeContactArea(const ProcessorGroup*,
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

    unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

void SingleFieldMPM::computeInternalForce(const ProcessorGroup*,
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

    unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gvolumeglobal;
    new_dw->get(gvolumeglobal,  lb->gVolumeLabel,
                m_materialManager->getAllInOneMatls()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel,
                           m_materialManager->getAllInOneMatls()->get(0), patch);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
      new_dw->get(psize,   lb->pCurSizeLabel,                pset);

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

      // for the non axisymmetric case:
      if(!flags->d_axisymmetric){
        for (ParticleSubset::iterator iter = pset->begin();
             iter != pset->end();
             iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN =
            interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,
                                                     d_S,psize[idx]);
          stressvol  = pstress[idx]*pvol[idx];
          stresspress = pstress[idx] + Id*(p_pressure[idx] - p_q[idx]);

          for (int k = 0; k < NN; k++){
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

          int NN =
            interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,
                                                                d_S,psize[idx]);

          stressvol   = pstress[idx]*pvol[idx];
          stresspress = pstress[idx] + Id*(p_pressure[idx] - p_q[idx]);

          // r is the x direction, z (axial) is the y direction
          double IFr=0.,IFz=0.;
          for (int k = 0; k < NN; k++){
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

#if 0
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
#endif
  }
}

void SingleFieldMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
                       "Doing SingleFieldMPM::computeAndIntegrateAcceleration");

    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    // Get required variables for this patch
    constNCVariable<Vector> internalforce0, internalforce1,velocity0, velocity1;
    constNCVariable<double> mass0, mass1;
    constNCVariable<Vector> externalforce0, externalforce1;

    new_dw->get(internalforce0,lb->gInternalForceF0Label,   0, patch, gnone, 0);
    new_dw->get(externalforce0,lb->gExternalForceF0Label,   0, patch, gnone, 0);
    new_dw->get(mass0,         lb->gMassF0Label,            0, patch, gnone, 0);
    new_dw->get(velocity0,     lb->gVelocityF0Label,        0, patch, gnone, 0);
    new_dw->get(internalforce1,lb->gInternalForceF1Label,   0, patch, gnone, 0);
    new_dw->get(externalforce1,lb->gExternalForceF1Label,   0, patch, gnone, 0);
    new_dw->get(mass1,         lb->gMassF1Label,            0, patch, gnone, 0);
    new_dw->get(velocity1,     lb->gVelocityF1Label,        0, patch, gnone, 0);

    // Create variables for the results
    NCVariable<Vector> velocity_star0,acceleration0;
    NCVariable<Vector> velocity_star1,acceleration1;
    new_dw->allocateAndPut(velocity_star0,lb->gVelocityStarF0Label, 0, patch);
    new_dw->allocateAndPut(velocity_star1,lb->gVelocityStarF1Label, 0, patch);
    new_dw->allocateAndPut(acceleration0, lb->gAccelerationF0Label, 0, patch);
    new_dw->allocateAndPut(acceleration1, lb->gAccelerationF1Label, 0, patch);

    acceleration0.initialize(Vector(0.,0.,0.));
    acceleration1.initialize(Vector(0.,0.,0.));
    double damp_coef = flags->d_artificialDampCoeff;

    for(NodeIterator iter=patch->getExtraNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;

      Vector acc0(0.,0.,0.);
      Vector acc1(0.,0.,0.);
      if (mass0[c] > flags->d_min_mass_for_acceleration){
        acc0  = (internalforce0[c]+externalforce0[c])/mass0[c];
        acc0 -= damp_coef*velocity0[c];
      }
      acceleration0[c]  = acc0 +  gravity;
      velocity_star0[c] = velocity0[c] + acceleration0[c] * delT;
      if (mass1[c] > flags->d_min_mass_for_acceleration){
        acc1  = (internalforce1[c]+externalforce1[c])/mass1[c];
        acc1 -= damp_coef*velocity1[c];
      }
      acceleration1[c]  = acc1 +  gravity;
      velocity_star1[c] = velocity1[c] + acceleration1[c] * delT;
    }

#if 0
    for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // Get required variables for this patch
      constNCVariable<Vector> internalforce, externalforce, velocity;
      constNCVariable<double> mass;

      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel, getLevel(patches) );

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
        acceleration[c]  = acc +  gravity;
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }    // matls
#endif
  }
}

void SingleFieldMPM::setGridBoundaryConditions(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing setGridBoundaryConditions");

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    string interp_type = flags->d_interpolator_type;

    NCVariable<Vector> gvelocity_star0, gacceleration0;
    constNCVariable<Vector> gvelocity0;
    NCVariable<Vector> gvelocity_star1, gacceleration1;
    constNCVariable<Vector> gvelocity1;

    new_dw->getModifiable(gacceleration0, lb->gAccelerationF0Label,    0,patch);
    new_dw->getModifiable(gvelocity_star0,lb->gVelocityStarF0Label,    0,patch);
    new_dw->get(gvelocity0,               lb->gVelocityF0Label,        0, patch,
                                                                 Ghost::None,0);
#ifdef RIGID
    constNCVariable<double> gMass;
    new_dw->get(gMass,                    lb->gMassLabel,              0, patch,
                                                                 Ghost::None,0);
#endif
    new_dw->getModifiable(gacceleration1, lb->gAccelerationF1Label,    0,patch);
    new_dw->getModifiable(gvelocity_star1,lb->gVelocityStarF1Label,    0,patch);
    new_dw->get(gvelocity1,               lb->gVelocityF1Label,        0, patch,
                                                                 Ghost::None,0);

    MPMBoundCond bc;
    bc.setBoundaryCondition(patch,0,"Velocity", gvelocity_star0,interp_type);
    bc.setBoundaryCondition(patch,0,"Symmetric",gvelocity_star0,interp_type);
    bc.setBoundaryCondition(patch,0,"Velocity", gvelocity_star1,interp_type);
    bc.setBoundaryCondition(patch,0,"Symmetric",gvelocity_star1,interp_type);

    // Now recompute acceleration as the difference between the velocity
    // interpolated to the grid (no bcs applied) and the new velocity_star
    for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();
                                                              iter++){
      IntVector c = *iter;
      gacceleration0[c] = (gvelocity_star0[c] - gvelocity0[c])/delT;
      gacceleration1[c] = (gvelocity_star1[c] - gvelocity1[c])/delT;
#ifdef RIGID
      if(gMass[c]>1.1*d_SMALL_NUM_MPM){
        gacceleration0[c] = Vector(0.0);
        gacceleration1[c] = Vector(0.0);
      }
#endif
    }

#if 0
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocity;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi,patch,
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
    } // matl loop
#endif
  }  // patch loop
}

void SingleFieldMPM::setPrescribedMotion(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
 // Get the current simulation time
 simTime_vartype simTimeVar;
 old_dw->get(simTimeVar, lb->simulationTimeLabel);
 double time = simTimeVar;

 delt_vartype delT;
 old_dw->get(delT, lb->delTLabel, getLevel(patches) );

 for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing setPrescribedMotion");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
          s=smax-1;
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

      Matrix3 F_high = d_prescribedF[s+1]; // next prescribed deformation gradient
      Matrix3 F_low  = d_prescribedF[s];   // last prescribed deformation gradient
      double t1 = d_prescribedTimes[s];    // time of last prescribed deformation
      double t2 = d_prescribedTimes[s+1];  // time of next prescribed deformation

      //Interpolate to get the deformation gradient at the current time:
      Matrix3 Ft = F_low*(t2-time)/(t2-t1) + F_high*(time-t1)/(t2-t1);

      // Calculate the rate of the deformation gradient without the rotation:
      Fdot = (F_high - F_low)/(t2-t1);

      // Now we need to construct the rotation matrix and its time rate:
      // We are only interested in the rotation information at the next
      // specified time since the rotations specified should be relative
      // to the previously specified time.  For example if I specify Theta=90
      // at time=1.0, and Theta = 91 and time=2.0 the total rotation at
      // time=2.0 will be 181 degrees.
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

      if (flags->d_exactDeformation){  //Exact Deformation Update
         double t3 = d_prescribedTimes[s+2];
         double t4 = d_prescribedTimes[s+3];
         if (time == 0 && t4 != 0) {
           new_dw->put(delt_vartype(t3 - t2), lb->delTLabel, getLevel(patches));
         }
         else {
           F_high = d_prescribedF[s + 2]; //next prescribed deformation gradient
           F_low  = d_prescribedF[s + 1]; //last prescribed deformation gradient
           t3 = d_prescribedTimes[s+2];
           t4 = d_prescribedTimes[s+3];
           double tst = t4 - t3;
           Ft = F_low*(t2-time)/(t2-t1) + F_high*(time-t1)/(t2-t1);
           Fdot = (F_high - F_low)/(t3-t2);
           thetadot = PrescribedTheta*(degtorad)/(t3-t2);
           new_dw->put(delt_vartype(tst), lb->delTLabel, getLevel(patches));
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

      // Fstar is the def grad with the superimposed rotations included
      // Fdotstar is the rate of the def grad with superimposed rotations incl.
      Matrix3 Fstar;
      Matrix3 Fdotstar;
      Fstar = Qt*Previous_Rotations*Ft;
      Fdotstar = Qdot*Previous_Rotations*Ft + Qt*Previous_Rotations*Fdot;

      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done(); iter++){
        IntVector n = *iter;

        Vector NodePosition = patch->getNodePosition(n).asVector();

        if (flags->d_exactDeformation){ //Exact Deformation Update
           gvelocity_star[n] = (F_high*F_low.Inverse() - Ident)*Previous_Rotations.Inverse()*Qt.Transpose()*NodePosition/delT;
        } else {
           gvelocity_star[n] = Fdotstar*Ft.Inverse()*Previous_Rotations.Inverse()*Qt.Transpose()*NodePosition;
        }
      } // Node Iterator
    }   // matl loop
  }     // patch loop
}

void SingleFieldMPM::applyExternalLoads(const ProcessorGroup* ,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Get the current simulation time
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, lb->simulationTimeLabel);
  double time = simTimeVar;

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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<Vector>       pExternalForce_new;
      old_dw->get(px,    lb->pXLabel,    pset);
      new_dw->allocateAndPut(pExternalForce_new,
                             lb->pExtForceLabel_preReloc,  pset);

      // pExternalForce is either:
      //  set using load curves
      //  set using an MMS formulation
      //  set to zero

      string mms_type = flags->d_mms_type;
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
        constParticleVariable<IntVector> pLoadCurveID;
        ParticleVariable<IntVector> pLoadCurveID_new;
        // Recycle the loadCurveIDs
        old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
        new_dw->allocateAndPut(pLoadCurveID_new,
                               lb->pLoadCurveIDLabel_preReloc, pset);
        pLoadCurveID_new.copyData(pLoadCurveID);
        if(do_PressureBCs){
          // Get the external force data and allocate new space for
          // external force on particle corners
          //constParticleVariable<Vector> pExternalForce;
          //old_dw->get(pExternalForce, lb->pExternalForceLabel, pset);

          ParticleVariable<Point> pExternalForceCorner1, pExternalForceCorner2,
                                  pExternalForceCorner3, pExternalForceCorner4;
          if (flags->d_useCBDI) {
            old_dw->get(psize,               lb->pSizeLabel,              pset);
            old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel,pset);
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
           pExternalForce_new[idx] = Vector(0.,0.,0.);
           for(int k=0;k<3;k++){
            int loadCurveID = pLoadCurveID[idx](k)-1;
            if (loadCurveID >= 0) {
              PressureBC* pbc = pbcP[loadCurveID];
              double force = forcePerPart[loadCurveID];

              if (flags->d_useCBDI) {
               Vector dxCell = patch->dCell();
               pExternalForce_new[idx] += pbc->getForceVectorCBDI(px[idx],
                                    psize[idx], pDeformationMeasure[idx],
                                    force, time,
                                    pExternalForceCorner1[idx],
                                    pExternalForceCorner2[idx],
                                    pExternalForceCorner3[idx],
                                    pExternalForceCorner4[idx],
                                    dxCell);
              } else {
               pExternalForce_new[idx]+=pbc->getForceVector(px[idx],force,time);
              }
            } // loadCurveID >=0
           }  // loop over elements of the IntVector
          }
        } else {  // using load curves, but not pressure BCs
          // Set to zero
          for(ParticleSubset::iterator iter = pset->begin();
                                       iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalForce_new[idx] = Vector(0.,0.,0.);
          }
        }
      } else if(!mms_type.empty()) {
        // MMS
        MMS MMSObject;
        MMSObject.computeExternalForceForMMS(old_dw,new_dw,time,pset,
                                            lb,flags,pExternalForce_new);
      } else {
        // Set to zero
        for(ParticleSubset::iterator iter = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pExternalForce_new[idx] = Vector(0.,0.,0.);
        }
      }
    } // matl loop
  }  // patch loop
}

void SingleFieldMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
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

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    //Carry forward NC_CCweight (put outside of matl loop, only need for matl 0)
    constNCVariable<double> NC_CCweight;
    NCVariable<double> NC_CCweight_new;
    Ghost::GhostType  gnone = Ghost::None;
    old_dw->get(NC_CCweight,       lb->NC_CCweightLabel,  0, patch, gnone, 0);
    new_dw->allocateAndPut(NC_CCweight_new, lb->NC_CCweightLabel,0,patch);
    NC_CCweight_new.copyData(NC_CCweight);

    // Get single velocity field stuff
    Ghost::GhostType  gac = Ghost::AroundCells;
    constNCVariable<Vector> gSurfGrad, gacc0, gacc1, gvelstar0, gvelstar1;
    new_dw->get(gSurfGrad, lb->gSurfGradLabel,          0,  patch, gac,NGP);
    new_dw->get(gacc0,     lb->gAccelerationF0Label,    0,  patch, gac,NGP);
    new_dw->get(gacc1,     lb->gAccelerationF1Label,    0,  patch, gac,NGP);
    new_dw->get(gvelstar0, lb->gVelocityStarF0Label,    0,  patch, gac,NGP);
    new_dw->get(gvelstar1, lb->gVelocityStarF1Label,    0,  patch, gac,NGP);

#ifdef RIGID
    std::vector< std::pair<double, Vector> > d_vel_profile;
    d_vel_profile.push_back(std::pair<double,Vector>(  0., 
                                                     Vector(0.,0.,0.0)) );
    d_vel_profile.push_back(std::pair<double,Vector>(  20.,
                                                     Vector(0.,0.,-.0001715)));
    d_vel_profile.push_back(std::pair<double,Vector>( 800.,
                                                     Vector(0.,0.,-.0001715)));
    d_vel_profile.push_back(std::pair<double,Vector>( 820.,
                                                     Vector(0.,0.,0.0)));
    d_vel_profile.push_back(std::pair<double,Vector>( 840.,
                                                     Vector(0.,0.,0.0001715)));
    d_vel_profile.push_back(std::pair<double,Vector>(1640.,
                                                     Vector(0.,0.,0.0001715)));
    d_vel_profile.push_back(std::pair<double,Vector>(1680.,
                                                     Vector(0.,0.,0.0)));

    int smin = 0; int smax = (int)(d_vel_profile.size())-1;
    double tmin = d_vel_profile[0].first;
    double tmax = d_vel_profile[smax].first;

    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, lb->simulationTimeLabel);
    double t = simTimeVar;
    Vector RIGID_VEL;
    if(t<=tmin) {
      RIGID_VEL = d_vel_profile[0].second;
    }
    else if(t>=tmax) {
      RIGID_VEL = d_vel_profile[smax].second;
    }
    else {
      // bisection search on table
      // could probably speed this up by keeping copy of last successful
      // search, and looking at that point and a couple to the right
      //
      while (smax>smin+1) {
          int smid = (smin+smax)/2;
          if(d_vel_profile[smid].first<t){
            smin = smid;
          }
          else{
            smax = smid;
          }
      }
      double l  = (d_vel_profile[smin+1].first-d_vel_profile[smin].first);
      double xi = (t-d_vel_profile[smin].first)/l;
      double vx = xi*d_vel_profile[smin+1].second[0]+(1-xi)*d_vel_profile[smin].second[0];
      double vy = xi*d_vel_profile[smin+1].second[1]+(1-xi)*d_vel_profile[smin].second[1];
      double vz = xi*d_vel_profile[smin+1].second[2]+(1-xi)*d_vel_profile[smin].second[2];

      RIGID_VEL = Vector(vx,vy,vz);
    }
#endif

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pvelocity, pvelSSPlus, pdisp;
      constParticleVariable<Vector> pSurfGrad;
      constParticleVariable<Matrix3> psize, pFOld;
      constParticleVariable<double> pmass, pVolumeOld, pTemperature,pSurf;
      constParticleVariable<long64> pids;
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pvelnew, pdispnew;
      ParticleVariable<Matrix3> psizeNew;
      ParticleVariable<double> pmassNew,pTempNew,pSurfNew;
      ParticleVariable<long64> pids_new;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew;

#if 0
      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration, gvelSPSSP;
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;
#endif

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(pVolumeOld,   lb->pVolumeLabel,                    pset);
      new_dw->get(pSurfGrad,    lb->pSurfGradLabel_preReloc,         pset);

      if(flags->d_XPIC2){
        new_dw->get(pvelSSPlus, lb->pVelocitySSPlusLabel,            pset);
      }

      new_dw->allocateAndPut(pxnew,      lb->pXLabel_preReloc,            pset);
      new_dw->allocateAndPut(pvelnew,    lb->pVelocityLabel_preReloc,     pset);
      new_dw->allocateAndPut(pdispnew,   lb->pDispLabel_preReloc,         pset);
      new_dw->allocateAndPut(pmassNew,   lb->pMassLabel_preReloc,         pset);
      new_dw->allocateAndPut(pTempPreNew,lb->pTempPreviousLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,   lb->pTemperatureLabel_preReloc,  pset);

      //Carry forward ParticleID and pSize
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
      new_dw->get(psize,               lb->pCurSizeLabel,             pset);
      old_dw->get(pSurf,               lb->pSurfLabel,                pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,       pset);
      new_dw->allocateAndPut(pSurfNew, lb->pSurfLabel_preReloc,       pset);
      pids_new.copyData(pids);
      pSurfNew.copyData(pSurf);

      //Carry forward color particle (debugging label)
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }
      if(flags->d_refineParticles){
        constParticleVariable<int> pRefinedOld;
        ParticleVariable<int> pRefinedNew;
        old_dw->get(pRefinedOld,            lb->pRefinedLabel,          pset);
        new_dw->allocateAndPut(pRefinedNew, lb->pRefinedLabel_preReloc, pset);
        pRefinedNew.copyData(pRefinedOld);
      }

      double Cp=mpm_matl->getSpecificHeat();

#if 0
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      if(flags->d_XPIC2){
        new_dw->get(gvelSPSSP,     lb->gVelSPSSPLabel,       dwi,patch,gac,NGP);
      }
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

      if(flags->d_XPIC2){
        // Loop over particles
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S,
                                                    psize[idx], pFOld[idx]);
          Vector vel(0.0,0.0,0.0);
          Vector velSSPSSP(0.0,0.0,0.0);
          Vector acc(0.0,0.0,0.0);
          double fricTempRate = 0.0;
          double tempRate = 0.0;
          double burnFraction = 0.0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel      += gvelocity_star[node]  * S[k];
            velSSPSSP+= gvelSPSSP[node]       * S[k];
            acc      += gacceleration[node]   * S[k];

            fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
            tempRate += (gTemperatureRate[node] + dTdt[node] +
                         fricTempRate)   * S[k];
            burnFraction += massBurnFrac[node]     * S[k];
          }

          // Update particle vel and pos using Nairn's XPIC(2) method
          pxnew[idx] = px[idx]    + vel*delT
                     - 0.5*(acc*delT + (pvelocity[idx] - 2.0*pvelSSPlus[idx])
                                                       + velSSPSSP)*delT;
          pvelnew[idx]  = 2.0*pvelSSPlus[idx] - velSSPSSP   + acc*delT;
          pdispnew[idx] = pdisp[idx] + (pxnew[idx]-px[idx]);
#if 0
          // PIC, or XPIC(1)
          pxnew[idx]    = px[idx]    + vel*delT
                     - 0.5*(acc*delT + (pvelocity[idx] - pvelSSPlus[idx]))*delT;
          pvelnew[idx]   = pvelSSPlus[idx]    + acc*delT;
#endif
          pTempNew[idx]    = pTemperature[idx] + tempRate*delT;
          pTempPreNew[idx] = pTemperature[idx]; // for thermal stress
          pmassNew[idx]    = Max(pmass[idx]*(1.    - burnFraction),0.);
          psizeNew[idx]    = (pmassNew[idx]/pmass[idx])*psize[idx];

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelnew[idx].length2();
          CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
          totalMom   += pvelnew[idx]*pmass[idx];
          totalmass  += pmass[idx];
        }
      } else {  // Not XPIC(2)
#endif
        // Loop over particles
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S, psize[idx]);
          Vector vel(0.0,0.0,0.0);
          Vector acc(0.0,0.0,0.0);
          Vector vel01(0.0,0.0,0.0);
          Vector acc01(0.0,0.0,0.0);
//          double fricTempRate = 0.0;
          double tempRate = 0.0;
          double burnFraction = 0.0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            double dotProd = Dot(pSurfGrad[idx],gSurfGrad[ni[k]]); 
            if(dotProd>0.0){
              vel01  += gvelstar0[node]  * S[k];
              acc01  += gacc0[node]      * S[k];
            } else {
              vel01  += gvelstar1[node]  * S[k];
              acc01  += gacc1[node]      * S[k];
            }
//            vel      += gvelocity_star[node]  * S[k];
//            acc      += gacceleration[node]   * S[k];

//            fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
//            tempRate += (gTemperatureRate[node] + dTdt[node] +
//                         fricTempRate)   * S[k];
//            burnFraction += massBurnFrac[node]     * S[k];
          }

          // Update the particle's pos and vel using std "FLIP" method
          pxnew[idx]   = px[idx]        + vel01*delT;
          pdispnew[idx]= pdisp[idx]     + vel01*delT;
          pvelnew[idx] = pvelocity[idx] + acc01*delT;

#ifdef RIGID
          if(m==0){
            pxnew[idx]   = px[idx]        + RIGID_VEL*delT;
            pdispnew[idx]= pdisp[idx]     + RIGID_VEL*delT;
            pvelnew[idx] = RIGID_VEL;
          }
#endif

          pTempNew[idx]    = pTemperature[idx] + tempRate*delT;
          pTempPreNew[idx] = pTemperature[idx]; // for thermal stress
          pmassNew[idx]    = Max(pmass[idx]*(1.    - burnFraction),0.);
          psizeNew[idx]    = (pmassNew[idx]/pmass[idx])*psize[idx];

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelnew[idx].length2();
          CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
          totalMom   += pvelnew[idx]*pmass[idx];
          totalmass  += pmass[idx];
        }
#if 0
      } // use XPIC(2) or not
#endif

      // scale back huge particle velocities.
      // Default for d_max_vel is 3.e105, hence the conditional
      if(flags->d_max_vel < 1.e105){
       for(ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(pvelnew[idx].length() > flags->d_max_vel){
          if(pvelnew[idx].length() >= pvelocity[idx].length()){
            pvelnew[idx]=(pvelnew[idx]/pvelnew[idx].length())
                             *(flags->d_max_vel*.9);
            cout << endl <<"Warning: particle " <<pids[idx]
                 <<" hit speed ceiling #1. Modifying particle vel. accordingly."
		 << "  " << pvelnew[idx].length()
		 << "  " << flags->d_max_vel
		 << "  " << pvelocity[idx].length()
                 << endl;
          } // if
        } // if
       }// for particles
      } // max velocity flag
    }  // loop over materials

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
    if(flags->d_reductionVars->mass){
      new_dw->put(sum_vartype(totalmass),      lb->TotalMassLabel);
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

void SingleFieldMPM::computeParticleGradients(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeParticleGradients");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    double partvoldef = 0.;

    // Get single velocity field stuff
    Ghost::GhostType  gac = Ghost::AroundCells;
    constNCVariable<Vector> gSurfGrad, gvelstar0, gvelstar1;
    new_dw->get(gSurfGrad, lb->gSurfGradLabel,          0,  patch, gac,NGP);
    new_dw->get(gvelstar0, lb->gVelocityStarF0Label,    0,  patch, gac,NGP);
    new_dw->get(gvelstar1, lb->gVelocityStarF1Label,    0,  patch, gac,NGP);

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<double> pVolumeOld,pmass,pmassNew;
      constParticleVariable<int> pLocalized;
      constParticleVariable<Matrix3> pFOld;
      constParticleVariable<Vector> pSurfGrad;
      ParticleVariable<double> pvolume,pTempNew;
      ParticleVariable<Matrix3> pFNew,pVelGrad;
      ParticleVariable<Vector> pTempGrad;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector>  gvelocity_star;
      constNCVariable<double>  gTempStar;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(pVolumeOld,   lb->pVolumeLabel,                    pset);
      old_dw->get(pLocalized,   lb->pLocalizedMPMLabel,              pset);
      new_dw->get(pSurfGrad,    lb->pSurfGradLabel_preReloc,         pset);

      new_dw->allocateAndPut(pvolume,    lb->pVolumeLabel_preReloc,       pset);
      new_dw->allocateAndPut(pVelGrad,   lb->pVelGradLabel_preReloc,      pset);
      new_dw->allocateAndPut(pTempGrad,  lb->pTemperatureGradientLabel_preReloc,
                                                                          pset);

//      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
//      if (flags->d_doExplicitHeatConduction){
//        new_dw->get(gTempStar,     lb->gTemperatureStarLabel,dwi,patch,gac,NGP);
//      }

      // Compute velocity gradient and deformation gradient on every particle
      // This can/should be combined into the loop above, once it is working
      Matrix3 Identity;
      Identity.Identity();
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        int NN=flags->d_8or27;
        Matrix3 tensorL(0.0);
#if 0
        if(!flags->d_axisymmetric){
#endif
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndShapeDerivatives(px[idx],ni,
                                                       d_S,psize[idx]);
         for(int k = 0; k < NN; k++) {
          double dotProd = Dot(pSurfGrad[idx],gSurfGrad[ni[k]]); 
          Vector gvel;
          if(dotProd>0.0){
           gvel = gvelstar0[ni[k]];
          } else {
           gvel = gvelstar1[ni[k]];
          }
           for (int j = 0; j<3; j++){
             double d_SXoodx = d_S[k][j]*oodx[j];
             for (int i = 0; i<3; i++) {
               tensorL(i,j) += gvel[i] * d_SXoodx;
             }
           }
         }
#if 0
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,
                                                   S,d_S,psize[idx],pFOld[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity_star,
                                                                   px[idx],NN);
        }
#endif
        pVelGrad[idx]=tensorL;
        pTempGrad[idx] = Vector(0.0,0.0,0.0);
        if (flags->d_doExplicitHeatConduction){
         if(flags->d_axisymmetric){
           cout << "Fix the pTempGradient calc for axisymmetry" << endl;
         }
         // Get the node indices that surround the cell
          for (int k = 0; k < NN; k++){
            for (int j = 0; j<3; j++) {
              pTempGrad[idx][j] += gTempStar[ni[k]] * d_S[k][j]*oodx[j];
            }
          } // Loop over local node
        }

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

        double J   =pFNew[idx].Determinant();
        double JOld=pFOld[idx].Determinant();
        pvolume[idx]=pVolumeOld[idx]*(J/JOld)*(pmassNew[idx]/pmass[idx]);
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
          vol_0_CC[cell_index]+=pvolume[idx]/J;
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

          double JOld=pFOld[idx].Determinant();
          pvolume[idx]=pVolumeOld[idx]*(J/JOld)*(pmassNew[idx]/pmass[idx]);
        }
      } //end of pressureStabilization loop  at the patch level

      //__________________________________
      //  Apply Erosion
      ErosionModel* em = mpm_matl->getErosionModel();
      em->updateVariables_Erosion( pset, pLocalized, pFOld, pFNew, pVelGrad );

    }  // for materials

    if(flags->d_reductionVars->volDeformed){
      new_dw->put(sum_vartype(partvoldef),     lb->TotalVolumeDeformedLabel);
    }

    delete interpolator;
  }

}

void SingleFieldMPM::finalParticleUpdate(const ProcessorGroup*,
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
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<int> pLocalized;
      constParticleVariable<double> pdTdt,pmassNew;
      ParticleVariable<double> pTempNew;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      new_dw->get(pdTdt,        lb->pdTdtLabel,                      pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);
      new_dw->get(pLocalized,   lb->pLocalizedMPMLabel_preReloc,     pset);

      new_dw->getModifiable(pTempNew, lb->pTemperatureLabel_preReloc,pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pTempNew[idx] += pdTdt[idx]*delT;

        // Delete particles whose mass is too small (due to combustion),
        // whose pLocalized flag has been set to -999 or who have 
        // a negative temperature
        if ((pmassNew[idx] <= flags->d_min_part_mass) || pTempNew[idx] < 0. ||
            (pLocalized[idx]==-999)){
          delset->addParticle(idx);
        }

      } // particles
      new_dw->deleteParticles(delset);
    } // materials
  } // patches
}

void SingleFieldMPM::updateCohesiveZones(const ProcessorGroup*,
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

    ParticleInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity[m], lb->gVelocityLabel,dwi, patch, gac, NGN);
      new_dw->get(gmass[m],     lb->gMassLabel,        dwi, patch, gac, NGN);
    }

    unsigned int numCZMatls=m_materialManager->getNumMatls( "CZ" );
    for(unsigned int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial( "CZ",  m );
      int dwi = cz_matl->getDWIndex();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      ParticleVariable<Point> czx_new;
      constParticleVariable<double> czarea;
      ParticleVariable<double> czarea_new;
      constParticleVariable<long64> czids;
      ParticleVariable<long64> czids_new;
      constParticleVariable<Vector> cznorm, cztang, czDispTop;
      ParticleVariable<Vector> cznorm_new, cztang_new, czDispTop_new;
      constParticleVariable<Vector> czDispBot, czsep, czforce;
      ParticleVariable<Vector> czDispBot_new, czsep_new, czforce_new;
      constParticleVariable<int> czTopMat, czBotMat, czFailed;
      ParticleVariable<int> czTopMat_new, czBotMat_new, czFailed_new;

      old_dw->get(czx,          lb->pXLabel,                         pset);
      old_dw->get(czarea,       lb->czAreaLabel,                     pset);
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
      new_dw->allocateAndPut(czarea_new,   lb->czAreaLabel_preReloc,      pset);
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

      czarea_new.copyData(czarea);
      czids_new.copyData(czids);
      czTopMat_new.copyData(czTopMat);
      czBotMat_new.copyData(czBotMat);

      double sig_max = cz_matl->getCohesiveNormalStrength();
      double delta_n = cz_matl->getCharLengthNormal();
      double delta_t = cz_matl->getCharLengthTangential();
      double tau_max = cz_matl->getCohesiveTangentialStrength();
      double delta_s = delta_t;
      double delta_n_fail = cz_matl->getNormalFailureDisplacement();
      double delta_t_fail = cz_matl->getTangentialFailureDisplacement();
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

        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(czx[idx],ni,S,size);

        Vector velTop(0.0,0.0,0.0);
        Vector velBot(0.0,0.0,0.0);
        double massTop = 0.0;
        double massBot = 0.0;
        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];
        double sumSTop = 0.;
        double sumSBot = 0.;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          if(gmass[TopMat][node]>2.e-200){
            velTop      += gvelocity[TopMat][node]* S[k];
            sumSTop     += S[k];
          }
          if(gmass[BotMat][node]>2.e-200){
            velBot      += gvelocity[BotMat][node]* S[k];
            sumSBot     += S[k];
          }
          massTop     += gmass[TopMat][node]*S[k];
          massBot     += gmass[BotMat][node]*S[k];
        }
        velTop/=sumSTop;
        velBot/=sumSBot;

#if 0
        // I'm not sure what this was here for in the first place,
        // but it is disabled for now
        double mass_ratio = 0.0;
        if (massBot > 0.0) {
          mass_ratio = massTop/massBot;
          mass_ratio = min(mass_ratio,1.0/mass_ratio);
        }
        else {
          mass_ratio = 0.0;
        }

        double mass_correction_factor = mass_ratio;
#endif

        // Update the cohesive zone's position and displacements
        czx_new[idx]         = czx[idx]       + .5*(velTop + velBot)*delT;
        czDispTop_new[idx]   = czDispTop[idx] + velTop*delT;
        czDispBot_new[idx]   = czDispBot[idx] + velBot*delT;
        czsep_new[idx]       = czDispTop_new[idx] - czDispBot_new[idx];

        double disp_old = czsep[idx].length();
        double disp     = czsep_new[idx].length();
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

        // Determine if a CZ has failed.
        double czf=0.0;
        if(czFailed[idx]>0 ){
          if(disp>=disp_old){
           czFailed_new[idx]=min(czFailed[idx]+1,1000);
          } else {
           czFailed_new[idx]=czFailed[idx];
          }
          czf =.001*((double) czFailed_new[idx]);
        }
        else if(fabs(D_n) > delta_n_fail){
          cout << "czFailed, D_n =  " << endl;
          czFailed_new[idx]=1;
        }
        else if( fabs(D_t1) > delta_t_fail){
          czFailed_new[idx]=1;
        }
        else if( fabs(D_t2) > delta_t_fail){
          czFailed_new[idx]=1;
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

        czforce_new[idx]     = ((normal_stress*cznorm_new[idx]
                             +   tang1_stress*cztang_new[idx]
                             +   tang2_stress*cztang2)*czarea_new[idx])
                             *   (1.0 - czf);
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

void SingleFieldMPM::insertParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing insertParticles");

    // Get the current simulation time
    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, lb->simulationTimeLabel);
    double time = simTimeVar;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    int index = -999;
    for(int i = 0; i<(int) d_IPTimes.size(); i++){
      if(time+delT > d_IPTimes[i] && time <= d_IPTimes[i]){
        index = i;
        if(index>=0){
          unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
          for(unsigned int m = 0; m < numMPMMatls; m++){
            MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
            int dwi = mpm_matl->getDWIndex();
            ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

            // Get the arrays of particle values to be changed
            ParticleVariable<Point> px;
            ParticleVariable<Vector> pvelocity;
            constParticleVariable<double> pcolor;

            old_dw->get(pcolor,             lb->pColorLabel,              pset);
            new_dw->getModifiable(px,       lb->pXLabel_preReloc,         pset);
            new_dw->getModifiable(pvelocity,lb->pVelocityLabel_preReloc,  pset);

            // Loop over particles here
            for(ParticleSubset::iterator iter  = pset->begin();
                                         iter != pset->end();   iter++){
              particleIndex idx = *iter;
              if(pcolor[idx]==d_IPColor[index]){
               pvelocity[idx]=d_IPVelNew[index];
               px[idx] = px[idx] + d_IPTranslate[index];
              } // end if
            }   // end for
          }     // end for
        }       // end if
      }         // end if
    }           // end for
  }             // end for
}

void SingleFieldMPM::addParticles(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* ,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    printTask(patches, patch,cout_doing, "Doing addParticles");
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    //Carry forward CellNAPID
    constCCVariable<int> NAPID;
    CCVariable<int> NAPID_new;
    Ghost::GhostType  gnone = Ghost::None;
    old_dw->get(NAPID,               lb->pCellNAPIDLabel,    0,patch,gnone,0);
    new_dw->allocateAndPut(NAPID_new,lb->pCellNAPIDLabel,    0,patch);
    NAPID_new.copyData(NAPID);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      ParticleVariable<Point> px;
      ParticleVariable<Matrix3> pF,pSize,pstress,pvelgrad,pscalefac;
      ParticleVariable<long64> pids;
      ParticleVariable<double> pvolume,pmass,ptemp,ptempP,pcolor;
      ParticleVariable<double> pESF; 
      ParticleVariable<Vector> pvelocity,pextforce,pdisp,ptempgrad;
      ParticleVariable<int> pref,ploc,prefOld,pSplitR1R2R3;
      ParticleVariable<IntVector> pLoadCID;
      new_dw->getModifiable(px,       lb->pXLabel_preReloc,            pset);
      new_dw->getModifiable(pids,     lb->pParticleIDLabel_preReloc,   pset);
      new_dw->getModifiable(pmass,    lb->pMassLabel_preReloc,         pset);
      new_dw->getModifiable(pSize,    lb->pSizeLabel_preReloc,         pset);
      new_dw->getModifiable(pdisp,    lb->pDispLabel_preReloc,         pset);
      new_dw->getModifiable(pstress,  lb->pStressLabel_preReloc,       pset);
      new_dw->getModifiable(pvolume,  lb->pVolumeLabel_preReloc,       pset);
      new_dw->getModifiable(pvelocity,lb->pVelocityLabel_preReloc,     pset);
      if(flags->d_computeScaleFactor){
        new_dw->getModifiable(pscalefac,lb->pScaleFactorLabel_preReloc,pset);
      }
      new_dw->getModifiable(pextforce,lb->pExtForceLabel_preReloc,     pset);
      new_dw->getModifiable(ptemp,    lb->pTemperatureLabel_preReloc,  pset);
      new_dw->getModifiable(ptempgrad,lb->pTemperatureGradientLabel_preReloc,
                                                                       pset);
      new_dw->getModifiable(ptempP,   lb->pTempPreviousLabel_preReloc, pset);
      new_dw->getModifiable(pref,     lb->pRefinedLabel_preReloc,      pset);
      new_dw->getModifiable(ploc,     lb->pLocalizedMPMLabel_preReloc, pset);
      new_dw->getModifiable(pvelgrad, lb->pVelGradLabel_preReloc,      pset);
      new_dw->getModifiable(pF,  lb->pDeformationMeasureLabel_preReloc,pset);
      if (flags->d_with_color) {
        new_dw->getModifiable(pcolor, lb->pColorLabel_preReloc,        pset);
      }
      if (flags->d_useLoadCurves) {
        new_dw->getModifiable(pLoadCID,lb->pLoadCurveIDLabel_preReloc, pset);
      }

      new_dw->allocateTemporary(prefOld,       pset);
      new_dw->allocateTemporary(pSplitR1R2R3,  pset);

      unsigned int numNewPartNeeded=0;
      bool splitForStretch=false;
      bool splitForAny=false;
      // Put refinement criteria here
      const unsigned int origNParticles = pset->addParticles(0);
      for( unsigned int pp=0; pp<origNParticles; ++pp ){
       prefOld[pp] = pref[pp];
       // Conditions to refine particle based on physical state
       // TODO:  Check below, should be < or <= in first conditional
       bool splitCriteria=false;
       //__________________________________
       // Only set the refinement flags for certain materials
       for(int i = 0; i< (int)d_thresholdVars.size(); i++ ){
          thresholdVar data = d_thresholdVars[i];
          string name  = data.name;
          double thresholdValue = data.value;

          if((int)m==data.matl){
            pSplitR1R2R3[pp]=0;
            if(name=="stressNorm"){
               double stressNorm = pstress[pp].Norm();
               if(stressNorm > thresholdValue){
                 splitCriteria = true;
                 splitForAny = true;
               }
            }
            if(name=="stretchRatio"){
              // This is the same R-vector equation used in CPDI interpolator
              // The "size" is relative to the grid cell size at this point
//              Matrix3 dsize = pF[pp]*pSize[pp];
              Matrix3 dsize = pF[pp]*pSize[pp]*Matrix3(dx[0],0,0,
                                                       0,dx[1],0,
                                                       0,0,dx[2]);
              Vector R1(dsize(0,0), dsize(1,0), dsize(2,0));
              Vector R2(dsize(0,1), dsize(1,1), dsize(2,1));
              Vector R3(dsize(0,2), dsize(1,2), dsize(2,2));
              double R1L=R1.length2();
              double R2L=R2.length2();
              double R3L=R3.length2();
              double R1_R2_ratSq = R1L/R2L;
              double R1_R3_ratSq = R1L/R3L;
              double R2_R3_ratSq = R2L/R3L;
              double tVSq = thresholdValue*thresholdValue;
              double tV_invSq = 1.0/tVSq;
//              cout << "R1L = " << R1L << endl;
//              cout << "R2L = " << R2L << endl;
//              cout << "R3L = " << R3L << endl;
              if (R1_R2_ratSq > tVSq){
                pSplitR1R2R3[pp]=1;
              } else if (R1_R2_ratSq < tV_invSq) {
                pSplitR1R2R3[pp]=-1;
              } else if (R1_R3_ratSq > tVSq && d_ndim==3){
                pSplitR1R2R3[pp]=2;
              } else if (R1_R3_ratSq < tV_invSq && d_ndim==3){
                pSplitR1R2R3[pp]=-2;
              } else if (R2_R3_ratSq > tVSq && d_ndim==3){
                 pSplitR1R2R3[pp]=3;
              } else if (R2_R3_ratSq < tV_invSq && d_ndim==3){
                 pSplitR1R2R3[pp]=-3;
              } else {
                 pSplitR1R2R3[pp]=0;
              }

              if(pSplitR1R2R3[pp]){
//                cout << "pSplit = " << pSplitR1R2R3[pp] << endl;
                splitCriteria  = true;
                splitForStretch = true;
                splitForAny = true;
              }
           }
         } // if this matl is in the list
       } // loop over criteria

       if(splitCriteria && prefOld[pp]==0){
         pref[pp]++;
         numNewPartNeeded++;
       }
      }  // Loop over original particles

      int fourOrEight=pow(2,d_ndim);
      if(splitForStretch){
        fourOrEight=4;
      }
      double fourthOrEighth = 1./((double) fourOrEight);
      numNewPartNeeded*=(fourOrEight+0);

      const unsigned int oldNumPar = pset->addParticles(numNewPartNeeded);

//      cout << "oldNumPar = " << oldNumPar << endl;
//      cout << "numNewPartNeeded = " << numNewPartNeeded << endl;
//      const unsigned int newNumPar = pset->addParticles(0);
//      cout << "newNumPar = " << newNumPar << endl;

      ParticleVariable<Point> pxtmp;
      ParticleVariable<Matrix3> pFtmp,psizetmp,pstrstmp,pvgradtmp,pSFtmp;
      ParticleVariable<long64> pidstmp;
      ParticleVariable<double> pvoltmp, pmasstmp,ptemptmp,ptempPtmp,pcolortmp;
      ParticleVariable<Vector> pveltmp,pextFtmp,pdisptmp,ptempgtmp;
      ParticleVariable<int> preftmp,ploctmp;
      ParticleVariable<IntVector> pLoadCIDtmp;
      new_dw->allocateTemporary(pidstmp,  pset);
      new_dw->allocateTemporary(pxtmp,    pset);
      new_dw->allocateTemporary(pvoltmp,  pset);
      new_dw->allocateTemporary(pveltmp,  pset);
      if(flags->d_computeScaleFactor){
        new_dw->allocateTemporary(pSFtmp, pset);
      }
      new_dw->allocateTemporary(pextFtmp, pset);
      new_dw->allocateTemporary(ptemptmp, pset);
      new_dw->allocateTemporary(ptempgtmp,pset);
      new_dw->allocateTemporary(ptempPtmp,pset);
      new_dw->allocateTemporary(pFtmp,    pset);
      new_dw->allocateTemporary(psizetmp, pset);
      new_dw->allocateTemporary(pdisptmp, pset);
      new_dw->allocateTemporary(pstrstmp, pset);
      new_dw->allocateTemporary(pmasstmp, pset);
      new_dw->allocateTemporary(preftmp,  pset);
      new_dw->allocateTemporary(ploctmp,  pset);
      new_dw->allocateTemporary(pvgradtmp,pset);
      if (flags->d_with_color) {
        new_dw->allocateTemporary(pcolortmp,pset);
      }
      if (flags->d_useLoadCurves) {
        new_dw->allocateTemporary(pLoadCIDtmp,  pset);
      }

      // copy data from old variables for particle IDs and the position vector
      for( unsigned int pp=0; pp<oldNumPar; ++pp ){
        pidstmp[pp]  = pids[pp];
        pxtmp[pp]    = px[pp];
        pvoltmp[pp]  = pvolume[pp];
        pveltmp[pp]  = pvelocity[pp];
        pextFtmp[pp] = pextforce[pp];
        ptemptmp[pp] = ptemp[pp];
        ptempgtmp[pp]= ptempgrad[pp];
        ptempPtmp[pp]= ptempP[pp];
        pFtmp[pp]    = pF[pp];
        psizetmp[pp] = pSize[pp];
        pdisptmp[pp] = pdisp[pp];
        pstrstmp[pp] = pstress[pp];
        if(flags->d_computeScaleFactor){
          pSFtmp[pp]   = pscalefac[pp];
        }
        if (flags->d_with_color) {
          pcolortmp[pp]= pcolor[pp];
        }
        if (flags->d_useLoadCurves) {
          pLoadCIDtmp[pp]= pLoadCID[pp];
        }
        pmasstmp[pp] = pmass[pp];
        preftmp[pp]  = pref[pp];
        ploctmp[pp]  = ploc[pp];
        pvgradtmp[pp]= pvelgrad[pp];
      }

      int numRefPar=0;
      if(splitForAny){
       // Don't loop over particles unless at least one needs to be refined
      for( unsigned int idx=0; idx<oldNumPar; ++idx ){
       if(pref[idx]!=prefOld[idx]){  // do refinement!
        IntVector c_orig;
        patch->findCell(px[idx],c_orig);
        vector<Point> new_part_pos;

        Matrix3 dsize = (pF[idx]*pSize[idx]*Matrix3(dx[0],0,0,
                                                    0,dx[1],0,
                                                    0,0,dx[2]));

        // Find vectors to new particle locations, based on particle size and
        // deformation (patterned after CPDI interpolator code)
        Vector r[4];
        if(fourOrEight==8){
          r[0]=Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                      -dsize(1,0)-dsize(1,1)+dsize(1,2),
                      -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.25;
          r[1]=Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                       dsize(1,0)-dsize(1,1)+dsize(1,2),
                       dsize(2,0)-dsize(2,1)+dsize(2,2))*0.25;
          r[2]=Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                       dsize(1,0)+dsize(1,1)+dsize(1,2),
                       dsize(2,0)+dsize(2,1)+dsize(2,2))*0.25;
          r[3]=Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                      -dsize(1,0)+dsize(1,1)+dsize(1,2),
                      -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.25;

          new_part_pos.push_back(px[idx]+r[0]);
          new_part_pos.push_back(px[idx]+r[1]);
          new_part_pos.push_back(px[idx]+r[2]);
          new_part_pos.push_back(px[idx]+r[3]);
          new_part_pos.push_back(px[idx]-r[0]);
          new_part_pos.push_back(px[idx]-r[1]);
          new_part_pos.push_back(px[idx]-r[2]);
          new_part_pos.push_back(px[idx]-r[3]);
        } else if(fourOrEight==4){
          if(pSplitR1R2R3[idx]){
            // divide the particle in the direction of longest relative R-vector
            Vector R(0.,0.,0.);
            if(pSplitR1R2R3[idx]==1 || pSplitR1R2R3[idx]==2){
              //cout << "split in R1-direction!" << endl;
              R = Vector(dsize(0,0), dsize(1,0), dsize(2,0));
            } else if(pSplitR1R2R3[idx]==3 || pSplitR1R2R3[idx]==-1){
              //cout << "split in R2-direction!" << endl;
              R = Vector(dsize(0,1), dsize(1,1), dsize(2,1));
            } else if(pSplitR1R2R3[idx]==-2 || pSplitR1R2R3[idx]==-3){
              // Grab the third R-vector
              R = Vector(dsize(0,2), dsize(1,2), dsize(2,2));
              //cout << "split in R3-direction!" << endl;
            }
            new_part_pos.push_back(px[idx]-.375*R);
            new_part_pos.push_back(px[idx]-.125*R);
            new_part_pos.push_back(px[idx]+.125*R);
            new_part_pos.push_back(px[idx]+.375*R);
          } else {
            // divide the particle along x and y direction
            r[0]=Vector(-dsize(0,0)-dsize(0,1),
                        -dsize(1,0)-dsize(1,1),
                         0.0)*0.25;
            r[1]=Vector( dsize(0,0)-dsize(0,1),
                         dsize(1,0)-dsize(1,1),
                         0.0)*0.25;

            new_part_pos.push_back(px[idx]+r[0]);
            new_part_pos.push_back(px[idx]+r[1]);
            new_part_pos.push_back(px[idx]-r[0]);
            new_part_pos.push_back(px[idx]-r[1]);
          }
        }

        for(int i = 0;i<fourOrEight;i++){
          long64 cellID = ((long64)c_orig.x() << 16) |
                          ((long64)c_orig.y() << 32) |
                          ((long64)c_orig.z() << 48);

          int& myCellNAPID = NAPID_new[c_orig];
          int new_index;
          if(i==0){
             new_index=idx;
          } else {
             new_index=(oldNumPar-1)+(fourOrEight-1)*numRefPar+i;
          }
          pidstmp[new_index]    = (cellID | (long64) myCellNAPID);
          pxtmp[new_index]      = new_part_pos[i];
          pvoltmp[new_index]    = fourthOrEighth*pvolume[idx];
          pmasstmp[new_index]   = fourthOrEighth*pmass[idx];
          pveltmp[new_index]    = pvelocity[idx];
          if (flags->d_useLoadCurves) {
            pLoadCIDtmp[new_index]  = pLoadCID[idx];
          }
          if (flags->d_with_color) {
            pcolortmp[new_index]  = pcolor[idx];
          }
          if(fourOrEight==8){
            if(flags->d_computeScaleFactor){
              pSFtmp[new_index]   = 0.5*pscalefac[idx];
            }
            psizetmp[new_index]   = 0.5*pSize[idx];
          } else if(fourOrEight==4){
           if(pSplitR1R2R3[idx]){
            // Divide psize in the direction of the biggest R-vector
            Matrix3 dSNew;
            if(pSplitR1R2R3[idx]==1 || pSplitR1R2R3[idx]==2){
              // Split across the first R-vector
              dSNew = Matrix3(0.25*dsize(0,0), dsize(0,1), dsize(0,2),
                              0.25*dsize(1,0), dsize(1,1), dsize(1,2),
                              0.25*dsize(2,0), dsize(2,1), dsize(2,2));
            } else if(pSplitR1R2R3[idx]==3 || pSplitR1R2R3[idx]==-1){
              // Split across the second R-vector
              dSNew = Matrix3(dsize(0,0), 0.25*dsize(0,1), dsize(0,2),
                              dsize(1,0), 0.25*dsize(1,1), dsize(1,2),
                              dsize(2,0), 0.25*dsize(2,1), dsize(2,2));
            } else if(pSplitR1R2R3[idx]==-2 || pSplitR1R2R3[idx]==-3){
              // Split across the third R-vector
              dSNew = Matrix3(dsize(0,0), dsize(0,1), 0.25*dsize(0,2),
                              dsize(1,0), dsize(1,1), 0.25*dsize(1,2),
                              dsize(2,0), dsize(2,1), 0.25*dsize(2,2));
            }
            if(flags->d_computeScaleFactor){
              pSFtmp[new_index]  = dSNew;
            }
            psizetmp[new_index]= pF[idx].Inverse()*dSNew*Matrix3(1./dx[0],0.,0.,
                                                              0.,1./dx[1],0.,
                                                              0.,0.,1./dx[2]);
           } else {
              // Divide psize by two in both x and y directions
            if(flags->d_computeScaleFactor){
              Matrix3 ps=pscalefac[idx];
              Matrix3 tmp(0.5*ps(0,0), 0.5*ps(0,1), 0.0,
                          0.5*ps(1,0), 0.5*ps(1,1), 0.0,
                          0.0,         0.0,         ps(2,2));
              pSFtmp[new_index]     = tmp;
             }
             Matrix3 ps = pSize[idx];
             Matrix3 tmp(0.5*ps(0,0), 0.5*ps(0,1), 0.0,
                         0.5*ps(1,0), 0.5*ps(1,1), 0.0,
                         0.0,         0.0,         ps(2,2));
             psizetmp[new_index]   = tmp;
           }
          } // if fourOrEight==4
          pextFtmp[new_index]   = pextforce[idx];
          pFtmp[new_index]      = pF[idx];
          pdisptmp[new_index]   = pdisp[idx];
          pstrstmp[new_index]   = pstress[idx];
          ptemptmp[new_index]   = ptemp[idx];
          ptempgtmp[new_index]  = ptempgrad[idx];
          ptempPtmp[new_index]  = ptempP[idx];
          preftmp[new_index]    = 1;
          ploctmp[new_index]    = ploc[idx];
          pvgradtmp[new_index]  = pvelgrad[idx];
          NAPID_new[c_orig]++;
        }
        numRefPar++;
       }  // if particle flagged for refinement
      } // for particles
      } // if any particles flagged for refinement

      cm->splitCMSpecificParticleData(patch, dwi, fourOrEight, prefOld, pref,
                                      oldNumPar, numNewPartNeeded,
                                      old_dw, new_dw);

      // put back temporary data
      new_dw->put(pidstmp,  lb->pParticleIDLabel_preReloc,           true);
      new_dw->put(pxtmp,    lb->pXLabel_preReloc,                    true);
      new_dw->put(pvoltmp,  lb->pVolumeLabel_preReloc,               true);
      new_dw->put(pveltmp,  lb->pVelocityLabel_preReloc,             true);
      if(flags->d_computeScaleFactor){
        new_dw->put(pSFtmp, lb->pScaleFactorLabel_preReloc,          true);
      }
      new_dw->put(pextFtmp, lb->pExtForceLabel_preReloc,             true);
      new_dw->put(pmasstmp, lb->pMassLabel_preReloc,                 true);
      new_dw->put(ptemptmp, lb->pTemperatureLabel_preReloc,          true);
      new_dw->put(ptempgtmp,lb->pTemperatureGradientLabel_preReloc,  true);
      new_dw->put(ptempPtmp,lb->pTempPreviousLabel_preReloc,         true);
      new_dw->put(psizetmp, lb->pSizeLabel_preReloc,                 true);
      new_dw->put(pdisptmp, lb->pDispLabel_preReloc,                 true);
      new_dw->put(pstrstmp, lb->pStressLabel_preReloc,               true);
      if (flags->d_with_color) {
        new_dw->put(pcolortmp,lb->pColorLabel_preReloc,              true);
      }
      if (flags->d_useLoadCurves) {
        new_dw->put(pLoadCIDtmp,lb->pLoadCurveIDLabel_preReloc,      true);
      }
      new_dw->put(pFtmp,    lb->pDeformationMeasureLabel_preReloc,   true);
      new_dw->put(preftmp,  lb->pRefinedLabel_preReloc,              true);
      new_dw->put(ploctmp,  lb->pLocalizedMPMLabel_preReloc,         true);
      new_dw->put(pvgradtmp,lb->pVelGradLabel_preReloc,              true);
    }  // for matls
  }    // for patches
}

void SingleFieldMPM::computeParticleScaleFactor(const ProcessorGroup*,
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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->get(pF,           lb->pDeformationMeasureLabel_preReloc,    pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(m_output->isOutputTimeStep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = (pF[idx]*(Matrix3(dx[0],0,0,
                                               0,dx[1],0,
                                               0,0,dx[2])*psize[idx]));

        } // for particles
      } // isOutputTimestep
    } // matls
  } // patches

}

void
SingleFieldMPM::setParticleDefault(ParticleVariable<double>& pvar,
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
SingleFieldMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
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
SingleFieldMPM::setParticleDefault(ParticleVariable<Matrix3>& pvar,
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


void SingleFieldMPM::printParticleLabels(vector<const VarLabel*> labels,
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
SingleFieldMPM::initialErrorEstimate(const ProcessorGroup*,
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
    new_dw->getModifiable(refineFlag, m_regridder->getRefineFlagLabel(),
                          0, patch);
    new_dw->get(refinePatchFlag, m_regridder->getRefinePatchFlagLabel(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();


    for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
SingleFieldMPM::errorEstimate(const ProcessorGroup* group,
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

      new_dw->getModifiable(refineFlag, m_regridder->getRefineFlagLabel(),
                            0, coarsePatch);
      new_dw->get(refinePatchFlag, m_regridder->getRefinePatchFlagLabel(),
                  0, coarsePatch);

      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);

      // coarsen the fineLevel flag
      for(unsigned int i=0;i<finePatches.size();i++){
        const Patch* finePatch = finePatches[i];

        IntVector cl, ch, fl, fh;
        getFineLevelRange(coarsePatch, finePatch, cl, ch, fl, fh);

        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
          continue;
        }
        constCCVariable<int> fineErrorFlag;
        new_dw->getRegion(fineErrorFlag,
                          m_regridder->getRefineFlagLabel(), 0,
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
SingleFieldMPM::refine(const ProcessorGroup*,
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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

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

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
        ParticleVariable<Vector> pvelocity, pexternalforce, pdisp,pTempGrad;
        ParticleVariable<Matrix3> psize, pVelGrad;
        ParticleVariable<double> pTempPrev,p_q;
        ParticleVariable<IntVector> pLoadCurve,pLoc;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress;

        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(p_q,            lb->p_qLabel,            pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pVelGrad,       lb->pVelGradLabel,       pset);
        new_dw->allocateAndPut(pTempGrad,      lb->pTemperatureGradientLabel,
                                                                        pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        new_dw->allocateAndPut(pLoc,           lb->pLocalizedMPMLabel,  pset);
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

//
void SingleFieldMPM::scheduleComputeNormals(SchedulerP   & sched,
                                       const PatchSet * patches,
                                       const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"SingleFieldMPM::scheduleComputeNormals");
  
  Task* t = scinew Task("MPM::computeNormals", this, 
                        &SingleFieldMPM::computeNormals);

  t->requires(Task::OldDW, lb->pXLabel,                  particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pMassLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pDispLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pVolumeLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pSizeLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pStressLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,d_one_matl,Ghost::None);

  t->computes(lb->gSurfNormLabel);
  t->computes(lb->gStressLabel);
  t->computes(lb->gNormTractionLabel);
  t->computes(lb->gPositionLabel);
  t->computes(lb->gDisplacementLabel);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void SingleFieldMPM::computeNormals(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * ,
                                     DataWarehouse  * old_dw,
                                     DataWarehouse  * new_dw)
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
  std::vector<constNCVariable<double> >  gmass(numMPMMatls);
  std::vector<NCVariable<Point> >        gposition(numMPMMatls);
  std::vector<NCVariable<Vector> >       gdisp(numMPMMatls);
  std::vector<NCVariable<Vector> >       gvelocity(numMPMMatls);
  std::vector<NCVariable<Vector> >       gsurfnorm(numMPMMatls);
  std::vector<NCVariable<double> >       gnormtraction(numMPMMatls);
  std::vector<NCVariable<Matrix3> >      gstress(numMPMMatls);

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    constNCVariable<double>    NC_CCweight;
    old_dw->get(NC_CCweight,   lb->NC_CCweightLabel,  0, patch, gnone, 0);

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    printTask(patches, patch, cout_doing, "Doing computeNormals");

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[m],                lb->gMassLabel,     dwi,patch,gan,1);

      new_dw->allocateAndPut(gsurfnorm[m],    lb->gSurfNormLabel,    dwi,patch);
      new_dw->allocateAndPut(gposition[m],    lb->gPositionLabel,    dwi,patch);
      new_dw->allocateAndPut(gdisp[m],        lb->gDisplacementLabel,dwi,patch);
      new_dw->allocateAndPut(gstress[m],      lb->gStressLabel,      dwi,patch);
      new_dw->allocateAndPut(gnormtraction[m],lb->gNormTractionLabel,dwi,patch);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Point> px;
      constParticleVariable<Vector> pdisp;
      constParticleVariable<double> pmass, pvolume;
      constParticleVariable<Matrix3> psize, pstress;
      constParticleVariable<Matrix3> deformationGradient;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(pdisp,               lb->pDispLabel,               pset);
      old_dw->get(pmass,               lb->pMassLabel,               pset);
      old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
      new_dw->get(psize,               lb->pCurSizeLabel,            pset);
      old_dw->get(pstress,             lb->pStressLabel,             pset);

      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));
      gposition[m].initialize(Point(0.0,0.0,0.0));
      gdisp[m].initialize(Vector(0.0,0.0,0.0));
      gnormtraction[m].initialize(0.0);
      gstress[m].initialize(Matrix3(0.0));

      int NN = flags->d_8or27;
      if(flags->d_axisymmetric){
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          NN = interpolator->findCellAndWeightsAndShapeDerivatives(
                          px[idx],ni,S,d_S,psize[idx]);
          double rho = pmass[idx]/pvolume[idx];
          for(int k = 0; k < NN; k++) {
            if (patch->containsNode(ni[k])){
              Vector G(d_S[k].x(),d_S[k].y(),0.0);
              gsurfnorm[m][ni[k]] += rho * G;
              gposition[m][ni[k]] += px[idx].asVector()*pmass[idx] * S[k];
              gdisp[m][ni[k]]     += pdisp[idx]*pmass[idx] * S[k];
              gstress[m][ni[k]]   += pstress[idx] * S[k];
            }
          }
        }
      } else {
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          NN = interpolator->findCellAndWeightsAndShapeDerivatives(
                          px[idx],ni,S,d_S,psize[idx]);
          for(int k = 0; k < NN; k++) {
            if (patch->containsNode(ni[k])){
              Vector grad(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                          d_S[k].z()*oodx[2]);
              gsurfnorm[m][ni[k]] += pmass[idx] * grad;
              gposition[m][ni[k]] += px[idx].asVector()*pmass[idx] * S[k];
              gdisp[m][ni[k]]     += pdisp[idx]*pmass[idx] * S[k];
              gstress[m][ni[k]]   += pstress[idx] * S[k];
            }
          }
        }
      } // axisymmetric conditional
    }   // matl loop

    // Make normal vectors colinear by setting all norms to be
    // in the opposite direction of the norm with the largest magnitude
    if(flags->d_computeColinearNormals){
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        double max_mag = gsurfnorm[0][c].length();
        unsigned int max_mag_matl = 0;
        for(unsigned int m=1; m<numMPMMatls; m++){
          double mag = gsurfnorm[m][c].length();
          if(mag > max_mag){
             max_mag = mag;
             max_mag_matl = m;
          }
        }  // loop over matls

        for(unsigned int m=0; m<numMPMMatls; m++){
	  if(m!=max_mag_matl){
           gsurfnorm[m][c] = -gsurfnorm[max_mag_matl][c];
         }
        }  // loop over matls
      }
    }

    // Make norms unit length
    for(unsigned int m=0;m<numMPMMatls;m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gsurfnorm[m],interp_type);

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
         IntVector c = *iter;
         double length = gsurfnorm[m][c].length();
         if(length>1.0e-15){
            gsurfnorm[m][c] = gsurfnorm[m][c]/length;
         }
         Vector norm = gsurfnorm[m][c];
         gnormtraction[m][c] = Dot((norm*gstress[m][c]),norm);
         gposition[m][c]    /= gmass[m][c];
         gdisp[m][c]        /= gmass[m][c];
      }
    }  // loop over matls

    delete interpolator;
  }    // patches
}

void SingleFieldMPM::scheduleComputeParticleSurfaceGradient(SchedulerP   & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,
                "SingleFieldMPM::scheduleComputeParticleSurfaceGradient");

  Task* t = scinew Task("MPM::computeParticleSurfaceGradient", this, 
                        &SingleFieldMPM::computeParticleSurfaceGradient);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType gnone = Ghost::None;

  t->requires(Task::OldDW, lb->timeStepLabel);
  t->requires(Task::OldDW, lb->pXLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pMassLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pSurfLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan, NGP);
  t->requires(Task::OldDW, lb->pSurfGradLabel,           gnone);
//t->requires(Task::NewDW, lb->gSurfLabel, d_one_matl, Ghost::AroundCells, NGN);

  t->computes(lb->pSurfGradLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::computeParticleSurfaceGradient(const ProcessorGroup *,
                                               const PatchSubset    * patches,
                                               const MaterialSubset * ,
                                                     DataWarehouse  * old_dw,
                                                     DataWarehouse  * new_dw)
{
  unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
  Ghost::GhostType  gan = Ghost::AroundNodes;

  timeStep_vartype timeStep;
  old_dw->get(timeStep, lb->timeStepLabel);
  int timestep = timeStep;

  // Should we make this an input file parameter?
  int interval = 10;
  int doit=timestep%interval;

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double rho=dx.length();
    double rho2=rho*rho;
    double rho3=rho2*rho;
    double invrho2 = 1./rho2;
    double invrho3 = 1./rho3;

    printTask(patches,patch,cout_doing,"Doing computeParticleSurfaceGradient");

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // This particle set includes ghost particles
      ParticleSubset* pset   = old_dw->getParticleSubset(dwi, patch,
                                                         gan, NGP, lb->pXLabel);

      // This is the on patch (OP) particle set
      ParticleSubset* psetOP = old_dw->getParticleSubset(dwi, patch);

      // Create arrays for the particle data
      constParticleVariable<Point>   px,pxOP;
      constParticleVariable<double>  psurf;
      ParticleVariable<Vector>       pSurfGrad;
      constParticleVariable<Vector>  pSurfGradOld;

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(psurf,          lb->pSurfLabel,          pset);

      old_dw->get(pxOP,           lb->pXLabel,             psetOP);
      old_dw->get(pSurfGradOld,   lb->pSurfGradLabel,      psetOP);

      new_dw->allocateAndPut(pSurfGrad,lb->pSurfGradLabel_preReloc, psetOP);

      vector<double> psurfv;
      vector<Point>  pxv;
      bool carryForward=true;
      if(timestep==1 || doit==0){
        carryForward=false;
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          psurfv.push_back(psurf[*it]);
          pxv.push_back(px[*it]);
        }
      }
      int vecl = psurfv.size();

      // Either carry forward the particle surface data, or recompute it every
      // N timesteps.
      if(carryForward){
       // Carry forward particle surface information
       for (ParticleSubset::iterator iter = psetOP->begin();
                                     iter != psetOP->end();
                                     iter++){
         particleIndex idx = *iter;
         pSurfGrad[idx]=pSurfGradOld[idx];
       }
      } else {
       for(ParticleSubset::iterator itop=psetOP->begin();
                                    itop!=psetOP->end();
                                    itop++){
        particleIndex idxOP = *itop;
        Vector gradD(0.,0.,0.);
        Vector gradS(0.,0.,0.);
        double S = 0;
        double D = 0;
        for(int ip=0;ip<vecl;ip++){
          Vector xminusxp = pxv[ip]-pxOP[idxOP];
          double r2 = xminusxp.length2();
          if(r2<rho2){
            double r = sqrt(r2);
            Vector gradw  = (r*invrho3 - invrho2)*xminusxp;
            gradD += psurfv[ip]*gradw;
            gradS +=            gradw;
            double w = 1. - 3.*r2*invrho2 + 2.*r2*r*invrho3;
            S     +=            w;
            D     += psurfv[ip]*w;
          }
        }
        pSurfGrad[idxOP] = -6.0*(gradD/S - D*gradS/(S*S));
      }
     }   // matl loop
    }   // endif
  }    // patches
}

void SingleFieldMPM::scheduleComputeGridSurfaceGradient(SchedulerP   & sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,
                "SingleFieldMPM::scheduleComputeGridSurfaceGradient");

  Task* t = scinew Task("MPM::computeGridSurfaceGradient", this, 
                        &SingleFieldMPM::computeGridSurfaceGradient);

  t->requires(Task::OldDW, lb->pXLabel,                  particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pSizeLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pMassLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pSurfLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pStressLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pVolumeLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pVelocityLabel,           particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pExternalForceLabel,      particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->pSurfGradLabel_preReloc,  particle_ghost_type, particle_ghost_layer);

  t->computes(lb->gSurfGradLabel,          d_one_matl);
  t->computes(lb->gSurfNormF0Label,        d_one_matl);
  t->computes(lb->gSurfNormF1Label,        d_one_matl);
  t->computes(lb->gMassF0Label,            d_one_matl);
  t->computes(lb->gMassF1Label,            d_one_matl);
  t->computes(lb->gStressF0Label,          d_one_matl);
  t->computes(lb->gStressF1Label,          d_one_matl);
  t->computes(lb->gVolumeF0Label,          d_one_matl);
  t->computes(lb->gVolumeF1Label,          d_one_matl);
  t->computes(lb->gPositionF0Label,        d_one_matl);
  t->computes(lb->gPositionF1Label,        d_one_matl);
  t->computes(lb->gVelocityF0Label,        d_one_matl);
  t->computes(lb->gVelocityF1Label,        d_one_matl);
  t->computes(lb->gNormTractionF0Label,    d_one_matl);
  t->computes(lb->gNormTractionF1Label,    d_one_matl);
  t->computes(lb->gInternalForceF0Label,   d_one_matl);
  t->computes(lb->gInternalForceF1Label,   d_one_matl);
  t->computes(lb->gExternalForceF0Label,   d_one_matl);
  t->computes(lb->gExternalForceF1Label,   d_one_matl);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::computeGridSurfaceGradient(const ProcessorGroup *,
                                           const PatchSubset    * patches,
                                           const MaterialSubset * ,
                                                 DataWarehouse  * old_dw,
                                                 DataWarehouse  * new_dw)
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;

  unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    NCVariable<double>  gMassF0, gMassF1, gVolumeF0, gVolumeF1;
    NCVariable<double>  gNormTracF0, gNormTracF1;
    NCVariable<Point>   gPositionF0, gPositionF1;
    NCVariable<Vector>  gSurfGrad, gVelF0, gVelF1, gIntForceF0, gIntForceF1;
    NCVariable<Vector>  gSurfNormF0, gSurfNormF1,  gExtForceF0, gExtForceF1;
    NCVariable<Matrix3> gStressF0, gStressF1;
    new_dw->allocateAndPut(gSurfGrad,   lb->gSurfGradLabel,           0, patch);
    new_dw->allocateAndPut(gSurfNormF0, lb->gSurfNormF0Label,         0, patch);
    new_dw->allocateAndPut(gSurfNormF1, lb->gSurfNormF1Label,         0, patch);
    new_dw->allocateAndPut(gMassF0,     lb->gMassF0Label,             0, patch);
    new_dw->allocateAndPut(gMassF1,     lb->gMassF1Label,             0, patch);
    new_dw->allocateAndPut(gStressF0,   lb->gStressF0Label,           0, patch);
    new_dw->allocateAndPut(gStressF1,   lb->gStressF1Label,           0, patch);
    new_dw->allocateAndPut(gVolumeF0,   lb->gVolumeF0Label,           0, patch);
    new_dw->allocateAndPut(gVolumeF1,   lb->gVolumeF1Label,           0, patch);
    new_dw->allocateAndPut(gPositionF0, lb->gPositionF0Label,         0, patch);
    new_dw->allocateAndPut(gPositionF1, lb->gPositionF1Label,         0, patch);
    new_dw->allocateAndPut(gVelF0,      lb->gVelocityF0Label,         0, patch);
    new_dw->allocateAndPut(gVelF1,      lb->gVelocityF1Label,         0, patch);
    new_dw->allocateAndPut(gIntForceF0, lb->gInternalForceF0Label,    0, patch);
    new_dw->allocateAndPut(gIntForceF1, lb->gInternalForceF1Label,    0, patch);
    new_dw->allocateAndPut(gExtForceF0, lb->gExternalForceF0Label,    0, patch);
    new_dw->allocateAndPut(gExtForceF1, lb->gExternalForceF1Label,    0, patch);
    new_dw->allocateAndPut(gNormTracF0, lb->gNormTractionF0Label,     0, patch);
    new_dw->allocateAndPut(gNormTracF1, lb->gNormTractionF1Label,     0, patch);
    gSurfGrad.initialize(Vector(0.0,0.0,0.0));
    gSurfNormF0.initialize(Vector(0.0,0.0,0.0));
    gSurfNormF1.initialize(Vector(0.0,0.0,0.0));
    gMassF0.initialize(d_SMALL_NUM_MPM);
    gMassF1.initialize(d_SMALL_NUM_MPM);
    gVolumeF0.initialize(d_SMALL_NUM_MPM);
    gVolumeF1.initialize(d_SMALL_NUM_MPM);
    gVelF0.initialize(Vector(0.0,0.0,0.0));
    gVelF1.initialize(Vector(0.0,0.0,0.0));
    gPositionF0.initialize(Point(0.0,0.0,0.0));
    gPositionF1.initialize(Point(0.0,0.0,0.0));
    gIntForceF0.initialize(Vector(0.0,0.0,0.0));
    gIntForceF1.initialize(Vector(0.0,0.0,0.0));
    gExtForceF0.initialize(Vector(0.0,0.0,0.0));
    gExtForceF1.initialize(Vector(0.0,0.0,0.0));
    gNormTracF0.initialize(0.0);
    gNormTracF1.initialize(0.0);
    gStressF0.initialize(Matrix3(0.0));
    gStressF1.initialize(Matrix3(0.0));

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    printTask(patches, patch, cout_doing, "Doing computeGridSurfaceGradient");

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Point>   px;
      constParticleVariable<double>  psurf;
      constParticleVariable<Vector>  pSurfGrad, pvelocity,pextforce;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(psurf,               lb->pSurfLabel,               pset);
      new_dw->get(pSurfGrad,           lb->pSurfGradLabel_preReloc,  pset);

      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;

//       if(psurf[idx]>0.0){
//        int NN = interpolator->findCellAndWeights(
//                        px[idx],ni,S,psize[idx],deformationGradient[idx]);
        int NN = 8;
        IntVector ni8[8];
        patch->findCellNodes(px[idx],ni8);
        double pSurfGradMag2 = pSurfGrad[idx].length2();
        for (int k = 0; k < NN; k++){
          if(patch->containsNode(ni8[k])) {
            double gSurfGradMag2 = gSurfGrad[ni8[k]].length2();
            if(pSurfGradMag2 > gSurfGradMag2){
              gSurfGrad[ni8[k]]=pSurfGrad[idx];
            } // if Mag
          }   // if patch
        }     // Loop over local node
//       }
      }       // for particles
    }   // matl loop

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Point>   px;
      constParticleVariable<double>  pmass,pvol;
      constParticleVariable<Matrix3> psize, deformationGradient,pstress;
      constParticleVariable<Vector>  pSurfGrad, pvelocity, pextforce;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(pmass,               lb->pMassLabel,               pset);
      old_dw->get(pvol,                lb->pVolumeLabel,             pset);
      new_dw->get(psize,               lb->pCurSizeLabel,            pset);
      old_dw->get(pstress,             lb->pStressLabel,             pset);
      old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
      old_dw->get(pextforce,           lb->pExternalForceLabel,      pset);
      new_dw->get(pSurfGrad,           lb->pSurfGradLabel_preReloc,  pset);

      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;
        int NN = interpolator->findCellAndWeightsAndShapeDerivatives(
                        px[idx],ni,S,d_S,psize[idx]);
        for (int k = 0; k < NN; k++){
          if(patch->containsNode(ni[k])) {
            Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                                          d_S[k].z()*oodx[2]);
            double dotProd = Dot(pSurfGrad[idx],gSurfGrad[ni[k]]); 
            Matrix3 stresspress = pstress[idx] /*- Id*p_q[idx]*/;
            if(dotProd>0.0){
              // Push Marker mass to field zero
              gMassF0[ni[k]]     += S[k] * pmass[idx];
              gVelF0[ni[k]]      += S[k] * pvelocity[idx]    * pmass[idx];
              gPositionF0[ni[k]] += S[k] * px[idx].asVector()* pmass[idx];
              gVolumeF0[ni[k]]   += S[k] * pvol[idx];
              gStressF0[ni[k]]   += S[k] * stresspress;
              gExtForceF0[ni[k]] += S[k] * pextforce[idx];
              gSurfNormF0[ni[k]] += pmass[idx] * div;
              gIntForceF0[ni[k]] -= (div * stresspress)  * pvol[idx];
            } else {
              // Push Marker mass to field one
              gMassF1[ni[k]]     += S[k] * pmass[idx];
              gVelF1[ni[k]]      += S[k] * pvelocity[idx]    * pmass[idx];
              gPositionF1[ni[k]] += S[k] * px[idx].asVector()* pmass[idx];
              gVolumeF1[ni[k]]   += S[k] * pvol[idx];
              gStressF1[ni[k]]   += S[k] * stresspress;
              gExtForceF1[ni[k]] += S[k] * pextforce[idx];
              gSurfNormF1[ni[k]] += pmass[idx] * div;
              gIntForceF1[ni[k]] -= (div * stresspress)  * pvol[idx];
            }
          }
        }
      }       // for particles
    }   // matl loop

    MPMBoundCond bc;
    bc.setBoundaryCondition(patch, 0, "Symmetric", gSurfNormF0, interp_type);
    bc.setBoundaryCondition(patch, 0, "Symmetric", gSurfNormF1, interp_type);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gVelF0[c]      /= gMassF0[c];
      gVelF1[c]      /= gMassF1[c];
      gPositionF0[c] /= gMassF0[c];
      gPositionF1[c] /= gMassF1[c];

      Vector normDiff = 0.5*(gSurfNormF0[c] - gSurfNormF1[c]);
      gSurfNormF0[c] = normDiff;
      gSurfNormF1[c] = -1.0*normDiff;

      double length0 = gSurfNormF0[c].length();
      double length1 = gSurfNormF1[c].length();
      if(length0>1.0e-15){
         gSurfNormF0[c]/=length0;
      }
      if(length1>1.0e-15){
         gSurfNormF1[c]/=length1;
      }
//      if(length0>length1){
//        gSurfNormF1[c] = -1.0*gSurfNormF0[c];
//      }else{
//        gSurfNormF0[c] = -1.0*gSurfNormF1[c];
//      }

      gNormTracF0[c]= Dot((gSurfNormF0[c]*gStressF0[c]),gSurfNormF0[c]);
      gNormTracF1[c]= Dot((gSurfNormF1[c]*gStressF1[c]),gSurfNormF1[c]);
    }
    bc.setBoundaryCondition(patch, 0, "Symmetric",  gVelF0,      interp_type);
    bc.setBoundaryCondition(patch, 0, "Symmetric",  gVelF1,      interp_type);
    bc.setBoundaryCondition(patch, 0, "Symmetric",  gIntForceF0, interp_type);
    bc.setBoundaryCondition(patch, 0, "Symmetric",  gIntForceF1, interp_type);
    delete interpolator;
  }    // patches
}

void SingleFieldMPM::scheduleSingleFieldContact(SchedulerP   & sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,
                "SingleFieldMPM::scheduleSingleFieldContact");

  Task* t = scinew Task("MPM::computeSingleFieldContact", this, 
                        &SingleFieldMPM::computeSingleFieldContact);

  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,     d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gSurfLabel,           d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gSurfNormF0Label,     d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gSurfNormF1Label,     d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gMassF0Label,         d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gMassF1Label,         d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeF0Label,       d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeF1Label,       d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gPositionF0Label,     d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gPositionF1Label,     d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gNormTractionF0Label, d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gNormTractionF1Label, d_one_matl, Ghost::None);
#ifdef RIGID
  t->requires(Task::NewDW, lb->gMassLabel,           d_one_matl, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,       d_one_matl, Ghost::None);
#endif
  t->modifies(             lb->gVelocityStarF0Label, d_one_matl);
  t->modifies(             lb->gVelocityStarF1Label, d_one_matl);

  sched->addTask(t, patches, matls);
}

void SingleFieldMPM::computeSingleFieldContact(const ProcessorGroup *,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * ,
                                                DataWarehouse  * old_dw,
                                                DataWarehouse  * new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeSingleFieldContact");

    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();

    constNCVariable<double> NC_CCweight, gNormTrac0, gNormTrac1;
    constNCVariable<double> gmass0, gmass1, gvol0, gvol1, gSurf;
    constNCVariable<Vector> gSurfNorm0, gSurfNorm1;
    NCVariable<Vector> gvelstar0, gvelstar1;
    constNCVariable<Point>  gpos0, gpos1;
    constNCVariable<Vector> gVelocity;
    constNCVariable<double> gMass;

    old_dw->get(NC_CCweight,lb->NC_CCweightLabel,     0, patch, gnone, 0);
    new_dw->get(gSurf,      lb->gSurfLabel,           0, patch, gnone, 0);
    new_dw->get(gSurfNorm0, lb->gSurfNormF0Label,     0, patch, gnone, 0);
    new_dw->get(gSurfNorm1, lb->gSurfNormF1Label,     0, patch, gnone, 0);
    new_dw->get(gNormTrac0, lb->gNormTractionF0Label, 0, patch, gnone, 0);
    new_dw->get(gNormTrac1, lb->gNormTractionF1Label, 0, patch, gnone, 0);

#ifdef RIGID
    new_dw->get(gMass,     lb->gMassLabel,            0, patch, gnone, 0);
    new_dw->get(gVelocity, lb->gVelocityLabel,        0, patch, gnone, 0);
#endif

    new_dw->get(gmass0,     lb->gMassF0Label,         0, patch, gnone, 0);
    new_dw->get(gmass1,     lb->gMassF1Label,         0, patch, gnone, 0);
    new_dw->get(gvol0,      lb->gVolumeF0Label,       0, patch, gnone, 0);
    new_dw->get(gvol1,      lb->gVolumeF1Label,       0, patch, gnone, 0);
    new_dw->get(gpos0,      lb->gPositionF0Label,     0, patch, gnone, 0);
    new_dw->get(gpos1,      lb->gPositionF1Label,     0, patch, gnone, 0);
    new_dw->getModifiable(gvelstar0, lb->gVelocityStarF0Label, 0, patch);
    new_dw->getModifiable(gvelstar1, lb->gVelocityStarF1Label, 0, patch);

    double d_mu     = 0.5; double d_sepFac = 0.9; double d_vol_const = 0.0;
    double sepDis = d_sepFac*cbrt(cell_vol);

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;

      double sumMass  =  gmass0[c] + gmass1[c];
      Vector mean_vel = (gmass0[c]*gvelstar0[c] + gmass1[c]*gvelstar1[c])/
                         sumMass;

      if(gSurf[c] > 0.49){  // if node is a surface node
       Vector Dv0(0.,0.,0.); Vector Dv1(0.,0.,0.);
       if(fabs(gmass0[c] - sumMass)>1.e-15 &&
          fabs(gmass1[c] - sumMass)>1.e-15){  // if node has mass from 2 fields

        double totalNodalVol = (gvol0[c] + gvol1[c])*8.0*NC_CCweight[c];

        if((totalNodalVol/cell_vol) > d_vol_const){
          Vector centerOfMassPos = (gmass0[c]*gpos0[c].asVector()
                                 +  gmass1[c]*gpos1[c].asVector())/
                                    sumMass;

          Vector sepvec0  = (sumMass/(sumMass - gmass0[c]))
                          * (centerOfMassPos - gpos0[c]);
          Vector sepvec1  = (sumMass/(sumMass - gmass1[c]))
                          * (centerOfMassPos - gpos1[c]);
          double sepscal0 = sepvec0.length();
          double sepscal1 = sepvec1.length();

          Vector deltaVelocity0=gvelstar0[c] - mean_vel;
          Vector deltaVelocity1=gvelstar1[c] - mean_vel;
          Vector normal0 = gSurfNorm0[c];
          Vector normal1 = gSurfNorm1[c];
          double normalDeltaVel0=Dot(deltaVelocity0,normal0);
          double normalDeltaVel1=Dot(deltaVelocity1,normal1);
          double Tn0 = gNormTrac0[c];
          double Tn1 = gNormTrac1[c];

          if(sepscal0 < sepDis){
           if((normalDeltaVel0 > 0.0 || Tn0 < -1.e-12)){
            Vector normal_normaldV0 = normal0*normalDeltaVel0;
            Vector dV_normaldV0     = deltaVelocity0 - normal_normaldV0;

            if(dV_normaldV0.length2() < 1.e-15){
              Dv0 = -normal_normaldV0;
            } else if(fabs(normalDeltaVel0) > 1.e-15) {
              Vector surfaceTangent = dV_normaldV0/dV_normaldV0.length();
              double tangentDeltaVelocity = Dot(deltaVelocity0,surfaceTangent);
              double frictionCoefficient=
                     Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel0));

              // Calculate velocity change needed to enforce contact
              Dv0 = -normal_normaldV0
                    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel0);
            }
           }
          }

          if(sepscal1 < sepDis){
           if((normalDeltaVel1 > 0.0 || Tn1 < -1.e-12)){
            Vector normal_normaldV1 = normal1*normalDeltaVel1;
            Vector dV_normaldV1     = deltaVelocity1 - normal_normaldV1;
            if(dV_normaldV1.length2() < 1.e-15){
              Dv1 = -normal_normaldV1;
            } else if(fabs(normalDeltaVel1) > 1.e-15) {
              Vector surfaceTangent = dV_normaldV1/dV_normaldV1.length();
              double tangentDeltaVelocity = Dot(deltaVelocity1,surfaceTangent);
              double frictionCoefficient=
                     Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel1));

              // Calculate velocity change needed to enforce contact
              Dv1 = -normal_normaldV1
                    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel1);
            }
           }
          }
         }
       }// Node has mass from both fields
       gvelstar0[c] += Dv0;
       gvelstar1[c] += Dv1;
      } else {  // Node is (code above) or is not (code below) a surface node
        gvelstar0[c] = mean_vel;
        gvelstar1[c] = mean_vel;
      }
#ifdef RIGID
      if(gMass[c]>1.e-100){
        gvelstar0[c] = gVelocity[c];
        gvelstar1[c] = gVelocity[c];
      }
#endif
    } // NodeIterator
  }    // patches
}
