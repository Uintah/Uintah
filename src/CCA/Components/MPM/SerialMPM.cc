/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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
#include <CCA/Components/MPM/SerialMPM.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/CZLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/Materials/Contact/ContactFactory.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/CohesiveZone/CohesiveZoneTasks.h>
#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/TorqueBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
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
#include <Core/Grid/Variables/PerPatchVars.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

// Diffusion includes
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/SDInterfaceModel.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/Materials/Diffusion/SDInterfaceModelFactory.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModelFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>


#include <iostream>
#include <fstream>
#include <cmath>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SerialMPM", false);
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

//______________________________________________________________________
//

SerialMPM::SerialMPM( const ProcessorGroup* myworld,
                      const MaterialManagerP materialManager)
  : ApplicationCommon( myworld, materialManager), MPMCommon( m_materialManager )
{
  flags = scinew MPMFlags(myworld);

  Cl = scinew CZLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  contactModel        = nullptr;
  thermalContactModel = nullptr;
  heatConductionModel = nullptr;
  cohesiveZoneTasks   = nullptr;
  NGP     = 1;
  NGN     = 1;
  d_loadCurveIndex=0;
  d_switchCriteria = nullptr;

  // Diffusion related
  d_fluxBC           = nullptr;
  d_sdInterfaceModel = nullptr;
}

SerialMPM::~SerialMPM()
{
  delete contactModel;
  delete thermalContactModel;
  delete heatConductionModel;
  delete d_fluxBC;
  delete d_sdInterfaceModel;
  delete flags;
  delete d_switchCriteria;
  delete Cl;
  delete cohesiveZoneTasks;

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
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec,
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

  flags->d_computeNormals        = needNormals;
  flags->d_useLogisticRegression = useLR;

  thermalContactModel =
    ThermalContactFactory::create(restart_mat_ps, m_materialManager, lb,flags);

  heatConductionModel = scinew HeatConduction(m_materialManager,lb,flags);

  materialProblemSetup(restart_mat_ps,flags, isRestart);

  contactModel->setContactMaterialAttributes();

  cohesiveZoneTasks = scinew CohesiveZoneTasks(m_materialManager, flags);

  cohesiveZoneTasks->cohesiveZoneProblemSetup(restart_mat_ps, flags);

  if (flags->d_doScalarDiffusion) {
    d_sdInterfaceModel = SDInterfaceModelFactory::create(restart_mat_ps, m_materialManager, flags, lb);
  }

  d_fluxBC = FluxBCModelFactory::create(m_materialManager, flags);

  //__________________________________
  //  create analysis modules
  // call problemSetup
  if(!flags->d_with_ice ){ // mpmice
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
void SerialMPM::outputProblemSpec(ProblemSpecP& root_ps)
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
  if (flags->d_doScalarDiffusion) {
    d_sdInterfaceModel->outputProblemSpec(mpm_ps);
  }

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
  //  output data analysis modules. Mpmice handles this
  if(!flags->d_with_ice && d_analysisModules.size() != 0){

    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;

      am->outputProblemSpec( root_ps );
    }
  }

}

void SerialMPM::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())) {
    return;
  }
  Task* t = scinew Task( "MPM::actuallyInitialize", this,
                   &SerialMPM::actuallyInitialize );

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
  t->computes(lb->pExternalHeatRateLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->pSurfLabel);
  t->computes(lb->pRefinedLabel);
  t->computes(lb->delTLabel,level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);
  t->computes(lb->NC_CCweightLabel,zeroth_matl);

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

  if (flags->d_doScalarDiffusion) {
    t->computes(lb->diffusion->pArea);
    t->computes(lb->diffusion->pConcentration);
    t->computes(lb->diffusion->pConcPrevious);
    t->computes(lb->diffusion->pGradConcentration);
    t->computes(lb->diffusion->pExternalScalarFlux);
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

    if (flags->d_doScalarDiffusion) {
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->addInitialComputesAndRequires(t, mpm_matl, patches);
    }
  }

  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));

  schedulePrintParticleCount(level, sched);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  if (flags->d_useLoadCurves && !flags->d_doScalarDiffusion) {
    // Schedule the initialization of pressure BCs per particle
    scheduleInitializePressureBCs(level, sched);
  }

  if (flags->d_useLoadCurves && flags->d_doScalarDiffusion) {
    // Schedule the initialization of scalar fluxBCs per particle
    d_fluxBC->scheduleInitializeScalarFluxBCs(level, sched);
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
    CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial("CZ",m);
    CohesiveZone* ch = cz_matl->getCohesiveZone();
    ch->scheduleInitialize(level, sched, cz_matl);

    schedulePrintCZCount(level, sched);
  }

  if (flags->d_deleteGeometryObjects) {
    scheduleDeleteGeometryObjects(level, sched);
  }
}

//______________________________________________________________________
//  Task:  On a restart if the values are going to be modified there must first
//         be computes( label ) before it can be modified.  This is a hack and
//         a problem with the infrastructure.
void SerialMPM::scheduleRestartInitializeHACK( SchedulerP   & sched,
                                               const LevelP & level)
{
  const string schedName = "SerialMPM::sched_restartInitializeHACK";
  printSchedule(level, cout_doing, schedName);

  const string taskName = "SerialMPM::restartInitializeHACK";
  Task* t = scinew Task(taskName, this, &SerialMPM::restartInitializeHACK);

  t->computes( lb->pTemperatureLabel  );
  sched->addTask(t, level->eachPatch(), m_materialManager->allMaterials("MPM"));
}


//______________________________________________________________________
//
void SerialMPM::scheduleRestartInitialize(const LevelP& level,
                                          SchedulerP& sched)
{


/*`==========TESTING==========*/
Task* t = scinew Task("SerialMPM::restartInitializeTask", this,
                      &SerialMPM::restartInitializeTask);

  const PatchSet* patches = level->eachPatch();
  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );

  for(unsigned int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addReinitializeComputesAndRequires(t, mpm_matl, patches);
  }


/*===========TESTING==========`*/


#if 0
  // to modify the particle temperature
  scheduleRestartInitializeHACK( sched, level );
  t->modifies( lb->pTemperatureLabel );
#endif

#if 0
  unsigned int numCZM = m_materialManager->getNumMatls( "CZ" );
  for(unsigned int m = 0; m < numCZM; m++){
    CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial("CZ",m);
    CohesiveZone* ch = cz_matl->getCohesiveZone();
    ch->scheduleInitialize(level, sched, cz_matl);
  }
#endif

  sched->addTask(t, level->eachPatch(),  m_materialManager->allMaterials( "MPM" ));
}

//______________________________________________________________________
void SerialMPM::schedulePrintParticleCount(const LevelP& level,
                                           SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &SerialMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "MPM" ));
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
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      int numParticles  = pset->end() - pset->begin();

      totalParticles+=numParticles;
    }
    new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
  }
}

//______________________________________________________________________
void SerialMPM::schedulePrintCZCount(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printCZCount",
                        this, &SerialMPM::printCZCount);
  t->requires(Task::NewDW, Cl->czCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "CZ" ));
}
//______________________________________________________________________
//
void SerialMPM::printCZCount(const ProcessorGroup* pg,
                                 const PatchSubset*,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{
  sumlong_vartype trcount;
  new_dw->get(trcount, Cl->czCountLabel);

  if(pg->myRank() == 0){
   std::cout << "Created " << (long) trcount << " total cohesive zones" << std::endl;
  }

  //__________________________________
  //  bulletproofing
  if(trcount == 0){
    ostringstream msg;
    msg << "\n ERROR: zero cohesive zones were created.";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
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
    if (bcs_type == "Pressure" || bcs_type == "Torque" || bcs_type=="HeatFlux"){
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
    sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializePressureBC",
                    this, &SerialMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel,                        Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel,              Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel,
                            d_loadCurveIndex, Task::OutOfDomain, Ghost::None);
    t->modifies(lb->pExternalForceLabel);
    t->modifies(lb->pExternalHeatRateLabel);
    if (flags->d_useCBDI) {
       t->requires(Task::NewDW, lb->pSizeLabel,                  Ghost::None);
       t->requires(Task::NewDW, lb->pDeformationMeasureLabel,    Ghost::None);
       t->computes(             lb->pExternalForceCorner1Label);
       t->computes(             lb->pExternalForceCorner2Label);
       t->computes(             lb->pExternalForceCorner3Label);
       t->computes(             lb->pExternalForceCorner4Label);
    }
    sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
  }

  if(d_loadCurveIndex->removeReference()) delete d_loadCurveIndex;
}

void SerialMPM::scheduleDeleteGeometryObjects(const LevelP& level,
                                              SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

  Task* t = scinew Task("MPM::deleteGeometryObjects",
                  this, &SerialMPM::deleteGeometryObjects);
  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
}

void SerialMPM::scheduleComputeStableTimeStep(const LevelP& level,
                                              SchedulerP& sched)
{
  // Nothing to do here - delt is computed as a by-product of the
  // constitutive model
  // However, this task needs to do something in the case that MPM
  // is being run on more than one level.
  Task* t = 0;
  cout_doing << d_myworld->myRank() << " MPM::scheduleComputeStableTimeStep \t\t\t\tL-" <<level->getIndex() << endl;

  t = scinew Task("MPM::actuallyComputeStableTimestep",
                   this, &SerialMPM::actuallyComputeStableTimestep);

  const MaterialSet* mpm_matls = m_materialManager->allMaterials( "MPM" );

  t->computes(lb->delTLabel,level.get_rep());
  sched->addTask(t,level->eachPatch(), mpm_matls);
}

void
SerialMPM::scheduleTimeAdvance(const LevelP & level,
                               SchedulerP   & sched)
{
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels()))
    return;

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = m_materialManager->allMaterials( "MPM" );
  const MaterialSet* cz_matls = m_materialManager->allMaterials( "CZ" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();

  const MaterialSubset* mpm_matls_sub = (   matls ?    matls->getUnion() : nullptr);;
  const MaterialSubset*  cz_matls_sub = (cz_matls ? cz_matls->getUnion() : nullptr);

  scheduleComputeCurrentParticleSize(     sched, patches, matls);
  scheduleApplyExternalLoads(             sched, patches, matls);
  if(flags->d_doScalarDiffusion) {
    d_fluxBC->scheduleApplyExternalScalarFlux(sched, patches, matls);
  }
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  if(flags->d_computeNormals){
    scheduleComputeNormals(               sched, patches, matls);
  }
  if(flags->d_useLogisticRegression){
    scheduleFindSurfaceParticles(         sched, patches, matls);
    scheduleComputeLogisticRegression(    sched, patches, matls);
  }
  scheduleExMomInterpolated(              sched, patches, matls);
  if(flags->d_doScalarDiffusion) {
    scheduleConcInterpolated(             sched, patches, matls);
  }
  if(flags->d_useCohesiveZones){
    cohesiveZoneTasks->scheduleUpdateCohesiveZones(
                                          sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
    cohesiveZoneTasks->scheduleAddCohesiveZoneForces(
                                          sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
  }
  if(d_bndy_traction_faces.size()>0) {
    scheduleComputeContactArea(           sched, patches, matls);
  }
  scheduleComputeInternalForce(           sched, patches, matls);
  if (flags->d_doScalarDiffusion) {
    scheduleComputeFlux(                  sched, patches, matls);
    scheduleComputeDivergence(            sched, patches, matls);
    scheduleDiffusionInterfaceDiv(        sched, patches, matls);
  }

  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
  if (flags->d_doScalarDiffusion) {
    scheduleComputeAndIntegrateDiffusion( sched, patches, matls);
  }
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  if (flags->d_prescribeDeformation){
    scheduleSetPrescribedMotion(          sched, patches, matls);
  }
  if(flags->d_XPIC2){
    scheduleComputeSSPlusVp(              sched, patches, matls);
    scheduleComputeSPlusSSPlusVp(         sched, patches, matls);
  }
  if(flags->d_doExplicitHeatConduction){
    scheduleComputeHeatExchange(          sched, patches, matls);
    scheduleComputeInternalHeatRate(      sched, patches, matls);
    scheduleComputeNodalHeatFlux(         sched, patches, matls);
    scheduleSolveHeatEquations(           sched, patches, matls);
    scheduleIntegrateTemperatureRate(     sched, patches, matls);
  }

  // For use with Force/Torque Transmitting materials
  scheduleReduceVars(                     sched, patches, matls);
  scheduleComputeGridVelocityForFTM(      sched, patches, matls);

  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleComputeParticleGradients(       sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);

  if(flags->d_computeScaleFactor){
    scheduleComputeParticleScaleFactor(   sched, patches, matls);
  }
  if(flags->d_doGranularMPM){ //MJ
    scheduleGranularMPM(                    sched, patches, matls);
  }

  scheduleFinalParticleUpdate(            sched, patches, matls);
  scheduleInsertParticles(                sched, patches, matls);
  if(flags->d_refineParticles){
    scheduleAddParticles(                 sched, patches, matls);
  }

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis_preReloc( sched, level);
    }
  }

  SerialMPM::scheduleParticleRelocation(   sched, level,  matls, cz_matls);

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
//______________________________________________________________________
//  Purpose:  If needed update the labels and materialSubsets that are passed
//             to Relocate and schedule particle relocate
void SerialMPM::scheduleParticleRelocation( SchedulerP        & sched,
                                            const LevelP      & level,
                                            const MaterialSet * matls,
                                            const MaterialSet * cz_matls)
{


  //__________________________________
  //  Unmodified labels and matls subset
  std::vector<std::vector<const VarLabel* > > old_labels;
  std::vector<std::vector<const VarLabel* > > new_labels;

  old_labels = d_particleState_preReloc;
  new_labels = d_particleState;

  const MaterialSubset*  old_mss = matls->getSubset(0);
  MaterialSubset*        new_mss = scinew MaterialSubset();
  new_mss->addReference();

  new_mss->addSubset( old_mss );

  //__________________________________
  // If needed concatenate the labels and matls that are passed into
  // the ParticleRelocate
  if(flags->d_useCohesiveZones){

    //update the mss
    const MaterialSubset*  mss  = cz_matls->getSubset(0);
    new_mss->addSubset( mss );

    // update the labels
    int numLabels = cohesiveZoneTasks->d_cohesiveZoneState_preReloc.size();
    for( int i=0; i<numLabels; i++){
      old_labels.push_back(cohesiveZoneTasks->d_cohesiveZoneState_preReloc[i]);
      new_labels.push_back(cohesiveZoneTasks->d_cohesiveZoneState[i] );
    }
  }

  //__________________________________
  //  create a new material set containing the
  //  the updated matlSubset.
  MaterialSet* newMatlSet = scinew MaterialSet();
  newMatlSet->addSubset( new_mss );
  newMatlSet->addReference();

  sched->scheduleParticleRelocation(level,
                                    lb->pXLabel_preReloc,
                                    old_labels,
                                    lb->pXLabel,
                                    new_labels,
                                    lb->pParticleIDLabel,
                                    newMatlSet );

  if(newMatlSet && newMatlSet->removeReference()) {
    delete newMatlSet;
  }
}



//______________________________________________________________________
//
void SerialMPM::scheduleComputeCurrentParticleSize(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeCurrentParticleSize");

  Task* t=scinew Task("MPM::computeCurrentParticleSize",
                    this, &SerialMPM::computeCurrentParticleSize);

  t->requires(Task::OldDW, lb->pSizeLabel,               Ghost::None);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, Ghost::None);

  t->computes(             lb->pCurSizeLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleApplyExternalLoads");

  Task* t=scinew Task("MPM::applyExternalLoads",
                    this, &SerialMPM::applyExternalLoads);

  t->requires(Task::OldDW, lb->simulationTimeLabel);

  if (!flags->d_mms_type.empty()) {
    //MMS problems need displacements
    t->requires(Task::OldDW, lb->pDispLabel,            Ghost::None);
  }

  if (flags->d_useLoadCurves || flags->d_useCBDI) {
    t->requires(Task::OldDW,    lb->pXLabel,                  Ghost::None);
    t->requires(Task::OldDW,    lb->pLoadCurveIDLabel,        Ghost::None);
    if(flags->d_keepPressBCNormalToSurface){
      t->requires(Task::OldDW,  lb->pDeformationMeasureLabel, Ghost::None);
    }
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
  t->computes(lb->pExtForceLabel_preReloc);
  t->computes(lb->pExternalHeatRateLabel_preReloc);

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

  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&SerialMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  if (flags->d_with_color) {
   t->requires(Task::OldDW, lb->pColorLabel,            gan,NGP);
  }
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  if (flags->d_GEVelProj) {
    t->requires(Task::OldDW, lb->pVelGradLabel,             gan,NGP);
    t->requires(Task::OldDW, lb->pTemperatureGradientLabel, gan,NGP);
  }
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
  t->requires(Task::NewDW, lb->pExternalHeatRateLabel_preReloc,
                                                       gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::NewDW, lb->pCurSizeLabel,          gan,NGP);
  if (flags->d_useCBDI) {
    t->requires(Task::NewDW,  lb->pExternalForceCorner1Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner2Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner3Label,gan,NGP);
    t->requires(Task::NewDW,  lb->pExternalForceCorner4Label,gan,NGP);
    t->requires(Task::OldDW,  lb->pLoadCurveIDLabel,gan,NGP);
  }
  if (flags->d_doScalarDiffusion) {
    t->requires(Task::OldDW, lb->pStressLabel,              gan, NGP);
    t->requires(Task::OldDW, lb->diffusion->pConcentration, gan, NGP);
    if (flags->d_GEVelProj) {
      t->requires(Task::OldDW, lb->diffusion->pGradConcentration, gan, NGP);
    }
    t->requires(Task::NewDW, lb->diffusion->pExternalScalarFlux_preReloc, gan, NGP);
    t->computes(lb->diffusion->gConcentration);
    t->computes(lb->diffusion->gConcentrationNoBC);
    t->computes(lb->diffusion->gHydrostaticStress);
    t->computes(lb->diffusion->gExternalScalarFlux);
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
  if (flags->d_with_color) {
    t->computes(lb->gColorLabel);
  }
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

void SerialMPM::scheduleComputeSSPlusVp(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeSSPlusVp");

  Task* t=scinew Task("MPM::computeSSPlusVp",
                      this, &SerialMPM::computeSSPlusVp);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   gnone);

  t->requires(Task::NewDW, lb->gVelocityLabel,                  gac,NGN);

  t->computes(lb->pVelocitySSPlusLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeSPlusSSPlusVp(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeSPlusSSPlusVp");

  Task* t=scinew Task("MPM::computeSPlusSSPlusVp",
                      this, &SerialMPM::computeSPlusSSPlusVp);

  Ghost::GhostType gan = Ghost::AroundNodes;
  Ghost::GhostType gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                     gan, NGP);
  t->requires(Task::OldDW, lb->pMassLabel,                  gan, NGP);
  t->requires(Task::NewDW, lb->pCurSizeLabel,               gan, NGP);
  t->requires(Task::NewDW, lb->pVelocitySSPlusLabel,        gan, NGP);
  t->requires(Task::NewDW, lb->gMassLabel,                  gac, NGN);

  t->computes(lb->gVelSPSSPLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeHeatExchange(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

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

  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &SerialMPM::computeStressTensor);
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

void SerialMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalForce");

  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &SerialMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::NewDW,lb->pCurSizeLabel,              gan,NGP);

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
  t->computes(lb->gStressForSavingLabel, m_materialManager->getAllInOneMatls(),
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
  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}

void SerialMPM::scheduleComputeNodalHeatFlux(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  heatConductionModel->scheduleComputeNodalHeatFlux(sched,patches,matls);
}

void SerialMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void SerialMPM::scheduleComputeAndIntegrateDiffusion(       SchedulerP  & sched
                                                    ,const  PatchSet    * patches
                                                    ,const  MaterialSet * matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) return;
  printSchedule(patches, cout_doing,
                "MPM::scheduleComputeAndIntegrateDiffusion");

  Task* t = scinew Task("SerialMPM::computeAndIntegrateDiffusion",
                        this, &SerialMPM::computeAndIntegrateDiffusion);

  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::NewDW, lb->gMassLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->diffusion->gConcentrationNoBC, Ghost::None);
  t->requires(Task::NewDW, lb->diffusion->gConcentration,     Ghost::None);
  t->requires(Task::NewDW, lb->diffusion->gExternalScalarFlux, Ghost::None);
  t->requires(Task::NewDW, d_sdInterfaceModel->getInterfaceFluxLabel(), Ghost::None);
  t->modifies(lb->diffusion->gConcentrationRate);
  t->computes(lb->diffusion->gConcentrationStar);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                                "MPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &SerialMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes( VarLabel::find(abortTimeStep_name) );
  t->computes( VarLabel::find(recomputeTimeStep_name) );
  //__________________________________
  //  reduction variables
  //  Create reductionMatlSubSet that includes all mpm matls
  // and the global matl.  Needed to output per matl reduction variables
  const MaterialSubset* global_mss = t->getGlobalMatlSubset();
  const MaterialSubset* mpm_mss    = (matls ?  matls->getUnion() : nullptr);

  MaterialSubset* reduction_mss = scinew MaterialSubset();
  reduction_mss->add( global_mss->get(0) );

  unsigned int nMatls = m_materialManager->getNumMatls( "MPM" );

  if( nMatls > 1 ){  // ignore for single matl problems
    for (unsigned int m = 0; m < nMatls; m++ ) {
      reduction_mss->add( mpm_mss->get(m) );
    }
  }

  reduction_mss->addReference();

  if(flags->d_reductionVars->mass ||
     flags->d_reductionVars->sumTransmittedForce){
    t->computes(lb->TotalMassLabel,           reduction_mss, Task::OutOfDomain);
  }

  if( flags->d_reductionVars->sumTransmittedForce ){
    // Tell scheduler to not automatically reduce variable.
    lb->SumTransmittedForceLabel->schedReductionTask(false);
    t->computes(lb->SumTransmittedForceLabel, reduction_mss, Task::OutOfDomain);
  }

  sched->addTask(t, patches, matls);

  if (reduction_mss && reduction_mss->removeReference()){
    delete reduction_mss;
  }
}

void SerialMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

void SerialMPM::scheduleExMomIntegrated(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

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
  t->requires(Task::OldDW, lb->delTLabel );

  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                               "MPM::scheduleInterpolateToParticlesAndUpdate");

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &SerialMPM::interpolateToParticlesAndUpdate);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             gac,NGN);
  if(flags->d_XPIC2){
    t->requires(Task::NewDW, lb->gVelSPSSPLabel,                gac,NGN);
    t->requires(Task::NewDW, lb->pVelocitySSPlusLabel,          gnone);
  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);

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
  t->computes(lb->pSizeLabel_preReloc);

  if(flags->d_doScalarDiffusion) {
    t->requires(Task::OldDW, lb->diffusion->pConcentration,     gnone     );
    t->requires(Task::NewDW, lb->diffusion->gConcentrationRate, gac,  NGN );

    t->computes(lb->diffusion->pConcentration_preReloc);
    t->computes(lb->diffusion->pConcPrevious_preReloc);

    if (flags->d_doAutoCycleBC) {
      if (flags->d_autoCycleUseMinMax) {
        t->computes(lb->diffusion->rMinConcentration);
        t->computes(lb->diffusion->rMaxConcentration);
      } else {
        t->computes(lb->diffusion->rTotalConcentration);
      }
    }
  }

  //__________________________________
  //  reduction variables
  //  Create reductionMatlSubSet that could include all mpm matls
  // and the global matl.  Needed for per matl reduction variables

  const MaterialSubset* global_mss = t->getGlobalMatlSubset();
  const MaterialSubset* mpm_mss    = matls->getUnion();

  MaterialSubset* reduction_mss = scinew MaterialSubset();
  reduction_mss->add( global_mss->get(0) );

  unsigned int nMatls = m_materialManager->getNumMatls( "MPM" );

  if( nMatls > 1 ){  // ignore for single matl problems
    for (unsigned int m = 0; m < nMatls; m++ ) {
      reduction_mss->add( mpm_mss->get(m) );
    }
  }

  reduction_mss->addReference();

  if(flags->d_reductionVars->momentum){
    t->computes(lb->TotalMomentumLabel, reduction_mss, Task::OutOfDomain);
  }
  if(flags->d_reductionVars->KE){
    t->computes(lb->KineticEnergyLabel, reduction_mss, Task::OutOfDomain);
  }
  if(flags->d_reductionVars->thermalEnergy){
    t->computes(lb->ThermalEnergyLabel, reduction_mss, Task::OutOfDomain);
  }
  if(flags->d_reductionVars->centerOfMass){
    t->computes(lb->CenterOfMassPositionLabel, reduction_mss,Task::OutOfDomain);
  }

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  // Carry Forward particle refinement flag
  if(flags->d_refineParticles){
    t->requires(Task::OldDW, lb->pRefinedLabel,  Ghost::None);
    t->computes(             lb->pRefinedLabel_preReloc);
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

  if (reduction_mss && reduction_mss->removeReference()){
    delete reduction_mss;
  }
}

void SerialMPM::scheduleComputeParticleGradients(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeParticleGradients");

  Task* t=scinew Task("MPM::computeParticleGradients",
                      this, &SerialMPM::computeParticleGradients);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  if (flags->d_doExplicitHeatConduction){
    t->requires(Task::NewDW, lb->gTemperatureStarLabel,         gac,NGN);
  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);

  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pTemperatureGradientLabel_preReloc);

  // JBH -- Need code to use these variables -- FIXME TODO
  if(flags->d_doScalarDiffusion) {
    t->requires(Task::NewDW, lb->diffusion->gConcentrationStar, gac, NGN);
    t->requires(Task::OldDW, lb->diffusion->pArea,              gnone);
    t->computes(lb->diffusion->pGradConcentration_preReloc);
    t->computes(lb->diffusion->pArea_preReloc);
  }

  if(flags->d_reductionVars->volDeformed){
    t->computes(lb->TotalVolumeDeformedLabel);
  }

  sched->addTask(t, patches, matls);
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

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->pdTdtLabel,                      gnone);
  t->requires(Task::NewDW, lb->pLocalizedMPMLabel_preReloc,     gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

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

    t->requires(Task::OldDW, lb->simulationTimeLabel);
    t->requires(Task::OldDW, lb->delTLabel );

    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);

    sched->addTask(t, patches, matls);
  }
}

void SerialMPM::scheduleAddParticles(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls)

{
  if( !flags->doMPMOnLevel( getLevel(patches)->getIndex(), getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleAddParticles" );

  Task * t = scinew Task("MPM::addParticles", this, &SerialMPM::addParticles );

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

  // JBH -- Add code for these variables -- FIXME TODO
  if (flags->d_doScalarDiffusion) {
    t->modifies(lb->diffusion->pConcentration_preReloc);
    t->modifies(lb->diffusion->pConcPrevious_preReloc);
    t->modifies(lb->diffusion->pGradConcentration_preReloc);
    t->modifies(lb->diffusion->pExternalScalarFlux_preReloc);
    t->modifies(lb->diffusion->pArea_preReloc);
    t->modifies(lb->diffusion->pDiffusivity_preReloc);
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
    if (flags->d_doScalarDiffusion) {
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
    }
  }

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//  Schedule the reduction of variables that are computed multiple times in a timestep
//  Use Label->schedReductionTask( true );
//  to the tell scheduler to perform the reduction.
//  The actual task is inside MPIScheduler.
void
SerialMPM::scheduleReduceVars(       SchedulerP  & sched,
                               const PatchSet    * patches,
                               const MaterialSet * matls )
{

  if( !flags->d_reductionVars->sumTransmittedForce ){
    return;
  }

  printSchedule( patches,cout_doing,"MPM::scheduleReduceVars");

  Task* t = scinew Task( "MPM::reductionTask", Task::Reduction);

  //__________________________________
  // Create reductionMatlSubSet that includes all mpm matls
  // and the global matl.
  const MaterialSubset* global_mss = t->getGlobalMatlSubset();
  const MaterialSubset* mpm_mss    = (matls ?  matls->getUnion() : nullptr);

  MaterialSubset* reduction_mss = scinew MaterialSubset();
  reduction_mss->add( global_mss->get(0) );

  unsigned int nMatls = m_materialManager->getNumMatls( "MPM" );

  if( nMatls > 1 ){  // ignore for single matl problems
    for (unsigned int m = 0; m < nMatls; m++ ) {
      reduction_mss->add( mpm_mss->get(m) );
    }
  }

  reduction_mss->addReference();

  // Tell the scheduler to reduce this variable
  lb->SumTransmittedForceLabel->schedReductionTask(true);
  t->computes( lb->SumTransmittedForceLabel, reduction_mss, Task::OutOfDomain );

  sched->addTask(t, patches, matls);

  if (reduction_mss && reduction_mss->removeReference()){
    delete reduction_mss;
  }
}

void
SerialMPM::scheduleComputeGridVelocityForFTM(SchedulerP  & sched,
                                             const PatchSet    * patches,
                                             const MaterialSet * matls )
{

  if( !flags->d_reductionVars->sumTransmittedForce ){
    return;
  }

  printSchedule( patches,cout_doing,"MPM::scheduleComputeGridVelocityForFTM");

  Task* t = scinew Task( "MPM::computeGridVelocityForFTM", this,
                          &SerialMPM::computeGridVelocityForFTM);

  //__________________________________
  // Create reductionMatlSubSet that includes all mpm matls
  // and the global matl.
  const MaterialSubset* global_mss = t->getGlobalMatlSubset();
  const MaterialSubset* mpm_mss    = (matls ?  matls->getUnion() : nullptr);

  MaterialSubset* reduction_mss = scinew MaterialSubset();
  reduction_mss->add( global_mss->get(0) );

  unsigned int nMatls = m_materialManager->getNumMatls( "MPM" );

  if( nMatls > 1 ){  // ignore for single matl problems
    for (unsigned int m = 0; m < nMatls; m++ ) {
      reduction_mss->add( mpm_mss->get(m) );
    }
  }

  reduction_mss->addReference();

  if( flags->d_reductionVars->mass ||
      flags->d_reductionVars->sumTransmittedForce){
    t->requires(Task::NewDW, lb->TotalMassLabel, nullptr,
                reduction_mss, Task::OutOfDomain, Task::SearchTG::NewTG);
  }
  if( flags->d_reductionVars->sumTransmittedForce ){
    t->requires(Task::NewDW, lb->SumTransmittedForceLabel, nullptr,
                reduction_mss, Task::OutOfDomain, Task::SearchTG::NewTG);
  }

  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, lb->delTLabel );

  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

  sched->addTask(t, patches, matls);

  if (reduction_mss && reduction_mss->removeReference()){
    delete reduction_mss;
  }
}

//______________________________________________________________________
//
void
SerialMPM::scheduleComputeParticleScaleFactor(       SchedulerP  & sched,
                                               const PatchSet    * patches,
                                               const MaterialSet * matls )
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing,"MPM::scheduleComputeParticleScaleFactor");

  Task * t = scinew Task( "MPM::computeParticleScaleFactor",this,
                          &SerialMPM::computeParticleScaleFactor );

  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,               Ghost::None);
  t->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc, Ghost::None);
  t->computes(lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void
SerialMPM::scheduleSetPrescribedMotion(       SchedulerP  & sched,
                                        const PatchSet    * patches,
                                        const MaterialSet * matls )
{
  if ( !flags->doMPMOnLevel( getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule(patches,cout_doing,"MPM::scheduleSetPrescribedMotion");

  Task * t = scinew Task( "MPM::setPrescribedMotion", this,
                           &SerialMPM::setPrescribedMotion );

  const MaterialSubset* mss = matls->getUnion();
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->delTLabel );

  sched->addTask(t, patches, matls);
}

void
SerialMPM::scheduleRefine( const PatchSet   * patches,
                                 SchedulerP & sched )
{
  printSchedule(patches,cout_doing,"MPM::scheduleRefine");
  Task* t = scinew Task( "SerialMPM::refine", this, &SerialMPM::refine );

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
  t->computes(lb->pExternalHeatRateLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pCurSizeLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->NC_CCweightLabel);
  t->computes(lb->delTLabel,getLevel(patches));

  // JBH -- Add code to support these variables FIXME TODO
  if (flags->d_doScalarDiffusion) {
    t->computes(lb->diffusion->pConcentration);
    t->computes(lb->diffusion->pConcPrevious);
    t->computes(lb->diffusion->pGradConcentration);
    t->computes(lb->diffusion->pArea);
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
SerialMPM::scheduleRefineInterface( const LevelP& /*fineLevel*/,
                                          SchedulerP& /*scheduler*/,
                                          bool /* ??? */,
                                          bool /* ??? */)
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
  Task* task = scinew Task("MPM::errorEstimate", this, &SerialMPM::errorEstimate);

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

//MJ: This function schedul GranularMPM and should take place before FinalParticleUpdate
void SerialMPM::scheduleGranularMPM(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    printSchedule(patches, cout_doing, "MPM::scheduleGranularMPM");

    Task* t = scinew Task("MPM::GranularMPM", this, &SerialMPM::GranularMPM);
  

//    MaterialSubset* zeroth_matl = scinew MaterialSubset();
//    zeroth_matl->add(0);
//    zeroth_matl->addReference();

    t->modifies(lb->pParticleIDLabel_preReloc);
    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pVolumeLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);
    t->modifies(lb->pMassLabel_preReloc);
    t->modifies(lb->pSizeLabel_preReloc);
    t->modifies(lb->pDispLabel_preReloc);
    t->modifies(lb->pStressLabel_preReloc);
    t->modifies(lb->pdTdtLabel);

    if (flags->d_with_color) {
        t->modifies(lb->pColorLabel_preReloc);
    }
    if (flags->d_useLoadCurves) {
        t->modifies(lb->pLoadCurveIDLabel_preReloc);
    }

    // JBH -- Add code for these variables -- FIXME TODO
    if (flags->d_doScalarDiffusion) {
        t->modifies(lb->diffusion->pConcentration_preReloc);
        t->modifies(lb->diffusion->pConcPrevious_preReloc);
        t->modifies(lb->diffusion->pGradConcentration_preReloc);
        t->modifies(lb->diffusion->pExternalScalarFlux_preReloc);
        t->modifies(lb->diffusion->pArea_preReloc);
        t->modifies(lb->diffusion->pDiffusivity_preReloc);
    }
    t->modifies(lb->pLocalizedMPMLabel_preReloc);
    t->modifies(lb->pExtForceLabel_preReloc);
    t->modifies(lb->pTemperatureLabel_preReloc);
    t->modifies(lb->pTemperatureGradientLabel_preReloc);
    t->modifies(lb->pTempPreviousLabel_preReloc);
    t->modifies(lb->pDeformationMeasureLabel_preReloc);
    //t->modifies(lb->pRefinedLabel_preReloc);
    if (flags->d_computeScaleFactor) {
        t->modifies(lb->pScaleFactorLabel_preReloc);
    }
    t->modifies(lb->pVelGradLabel_preReloc);

    //t->requires(Task::OldDW, lb->pCellNAPIDLabel, zeroth_matl, Ghost::None);
    //t->computes(lb->pCellNAPIDLabel, zeroth_matl);

    // Need to figure out if this is needed, and if not, why not?
#if 0
    unsigned int numMatls = m_materialManager->getNumMatls("MPM");
    for (unsigned int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
        ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
        cm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
        if (flags->d_doScalarDiffusion) {
            ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
            sdm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
        }
    }
#endif

    sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void SerialMPM::printParticleCount(const ProcessorGroup* pg,
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
  if(pcount == 0 ){
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

  printTask(patches, patches->get(0),cout_doing,
                                     "MPM::countMaterialPointsPerLoadCurve");
  // Find the number of pressure BCs in the problem
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure" || bcs_type == "Torque" || bcs_type=="HeatFlux"){
      nofPressureBCs++;

      // Loop through the patches and count
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        unsigned int numMPMMatls=m_materialManager->getNumMatls("MPM");
        int numPts = 0;
        for(unsigned int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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
void SerialMPM::initializePressureBC(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* ,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
  printTask(patches, patches->get(0),cout_doing,
                                            "Doing MPM::initializePressureBC");
  if (cout_dbg.active())
    cout_dbg << "Current Time (Initialize Pressure BC) = " << time << endl;

  // Calculate the force vector at each particle
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      constParticleVariable<IntVector> pLoadCurveID;
      ParticleVariable<Vector> pExternalForce;
      ParticleVariable<double> pExternalHeatRate;

      new_dw->get(px, lb->pXLabel, pset);
      new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
      new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);
      new_dw->getModifiable(pExternalHeatRate, 
                                            lb->pExternalHeatRateLabel, pset);

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
//      int nofTorqueBCs = 0;
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
        } else if (bcs_type == "Torque") {
          // Get the material points per load curve
          sumlong_vartype numPart = 0;
          new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel,
                      0, nofPressureBCs++);

          // Save the material points per load curve in the PressureBC object
          TorqueBC* tbc =
            dynamic_cast<TorqueBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
          tbc->numMaterialPoints(numPart);

          if (cout_dbg.active())
          cout_dbg << "    Load Curve = "
                   << nofPressureBCs << " Num Particles = " << numPart << endl;

          // Calculate the force per particle at t = 0.0
          double torquePerPart = tbc->torquePerParticle(time);

          // Loop through the patches and calculate the force vector
          // at each particle

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalForce[idx] = Vector(0.,0.,0.);
            for(int k=0;k<3;k++){
             if (pLoadCurveID[idx](k) == nofPressureBCs) {
               pExternalForce[idx] += tbc->getForceVector(px[idx],
                                                          torquePerPart,time);
            } // if pLoadCurveID...
           } // Loop over elements of the loadCurveID IntVector
          }  // loop over particles
        } else if (bcs_type == "HeatFlux") {
          // Get the material points per load curve
          sumlong_vartype numPart = 0;
          new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel,
                      0, nofPressureBCs++);

          // Save the material points per load curve in the PressureBC object
          HeatFluxBC* hfbc =
            dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
          hfbc->numMaterialPoints(numPart);

          if (cout_dbg.active())
          cout_dbg << "    Load Curve = "
                   << nofPressureBCs << " Num Particles = " << numPart << endl;

          // Calculate the flux per particle at t = 0.0
          double heatFluxPerPart = hfbc->fluxPerParticle(time);

          // Loop through the patches and calculate the flux at each particle

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalHeatRate[idx] = 0.;
            for(int k=0;k<3;k++){
             if (pLoadCurveID[idx](k) == nofPressureBCs) {
               pExternalHeatRate[idx] += hfbc->getFlux(px[idx],heatFluxPerPart);
            } // if pLoadCurveID...
           } // Loop over elements of the loadCurveID IntVector
          }  // loop over particles
        }  // if pressure loop
      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

//______________________________________________________________________
//
void SerialMPM::deleteGeometryObjects(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* ,
                                      DataWarehouse* new_dw)
{
   printTask( cout_doing,"Doing MPM::deleteGeometryObjects");

   unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
   for(unsigned int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
     cout << "MPM::Deleting Geometry Objects  matl: " << mpm_matl->getDWIndex() << "\n";
     mpm_matl->deleteGeomObjects();
   }

    // The call below is necessary because the GeometryPieceFactory holds on to a pointer
    // to all geom_pieces (so that it can look them up by name during initialization)
    // The pieces are never actually deleted until the factory is destroyed at the end
    // of the program. resetFactory() will rid of the pointer (lookup table) and
    // allow the deletion of the unneeded pieces.

    GeometryPieceFactory::resetFactory();
}
//______________________________________________________________________
//
void SerialMPM::actuallyInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;

  const Level* level = getLevel(patches);
  IntVector lowNode, highNode;
  level->findInteriorNodeIndexRange(lowNode, highNode);

  // Determine dimensionality for particle splitting
  // To be recognized as 2D, must be in the x-y plane
  // A 1D problem must be in the x-direction.
  flags->d_ndim=3;
  if(highNode.z() - lowNode.z()==2) {
     flags->d_ndim=2;
    if(highNode.y() - lowNode.y()==2) {
       flags->d_ndim=1;
    }
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing MPM::actuallyInitialize");

    CCVariable<int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

    NCVariable<double> NC_CCweight;
    new_dw->allocateAndPut(NC_CCweight, lb->NC_CCweightLabel,    0, patch);

    //__________________________________
    // - Initialize NC_CCweight = 0.125
    // - Find the walls with symmetry BC and double NC_CCweight
    NC_CCweight.initialize(0.125);

    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( auto itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;

      int mat_id = 0;

      if (patch->haveBC(face,mat_id,"symmetry","Symmetric")) {
        for(CellIterator iter = patch->getFaceIterator(face,Patch::FaceNodes);
                                                  !iter.done(); iter++) {
          NC_CCweight[*iter] = 2.0*NC_CCweight[*iter];
        }
      }
    }

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      particleIndex numParticles = mpm_matl->createParticles(cellNAPID,
                                                             patch, new_dw);

      totalParticles+=numParticles;
      mpm_matl->getConstitutiveModel()->initializeCMData(patch,mpm_matl,new_dw);

      //initialize Damage model
      mpm_matl->getDamageModel()->initializeLabels( patch, mpm_matl, new_dw );

      mpm_matl->getErosionModel()->initializeLabels( patch, mpm_matl, new_dw );

      if (flags->d_doScalarDiffusion) {
        ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        sdm->initializeTimeStep(patch, mpm_matl, new_dw);
        sdm->initializeSDMData( patch, mpm_matl, new_dw);
      }
    }
  } // patches

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
  string interp_type = flags->d_interpolator_type;
  if(interp_type=="linear" && num_extra_cells!=IntVector(0,0,0)){
    if( !flags->d_with_ice ){
      ostringstream msg;
      msg << "\n ERROR: When using <interpolator>linear</interpolator> \n"
          << " you should also use <extraCells>[0,0,0]</extraCells> \n"
          << " unless you are running an MPMICE case.\n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }
  else if(((interp_type=="gimp"       ||
            interp_type=="3rdorderBS" ||
            interp_type=="fast_cpdi"  ||
            interp_type=="cpti"       ||
            interp_type=="cpdi")                          &&
            (  (num_extra_cells+periodic)!=IntVector(1,1,1) &&
            (!((num_extra_cells+periodic)==IntVector(1,1,0) &&
             flags->d_axisymmetric))))){
      ostringstream msg;
      msg << "\n ERROR: When using <interpolator>gimp</interpolator> \n"
          << " or <interpolator>3rdorderBS</interpolator> \n"
          << " or <interpolator>cpdi</interpolator> \n"
          << " or <interpolator>fast_cpdi</interpolator> \n"
          << " or <interpolator>cpti</interpolator> \n"
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

  // JBH -- Fix these concentrations to be dilineated in non-absolute form. FIXME TODO
  if (flags->d_doAutoCycleBC && flags->d_doScalarDiffusion) {
    if (flags->d_autoCycleUseMinMax) {
      new_dw->put(min_vartype(5e11), lb->diffusion->rMinConcentration);
      new_dw->put(max_vartype(-5e11), lb->diffusion->rMaxConcentration);
    } else {
      new_dw->put(sum_vartype(0.0), lb->diffusion->rTotalConcentration);
    }
  }
}

//______________________________________________________________________
//  Task:  SerialMPM::restartInitializeTask
//  Purpose:  Modify variables on a restart.  You MUST schedule a
//            computes<label> in the restartInitializeHACK.
//______________________________________________________________________
void SerialMPM::restartInitializeTask(const ProcessorGroup *,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * ,
                                      DataWarehouse        * old_dw,
                                      DataWarehouse        * new_dw)
{
  //__________________________________
  // Patches loop
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    const string msg = "Doing SerialMPM::restartInitializeTask";
    printTask(patches, patch, cout_doing, msg);

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

/*`==========TESTING==========*/
      if( cm->d_reinitializeCMData ){
        cm->reinitializeCMData(patch, mpm_matl, new_dw);
      }
/*===========TESTING==========`*/

#if 0  // used to modify the particle temperature.
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset  = new_dw->getParticleSubset( dwi, patch );

      ParticleVariable<double> pTemperature;
      new_dw->getModifiable(pTemperature,   lb->pTemperatureLabel,   pset);

      ParticleSubset::iterator iter;
      for (iter = pset->begin(); iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pTemperature[idx] = 310;
      }
#endif
    }
  }
}

//______________________________________________________________________
//
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

void SerialMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* ,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  // Put something here to satisfy the need for a reduction operation in
  // the case that there are multiple levels present
  const Level* level = getLevel(patches);
  // JBH -- Convert this to a global variable that goes into all of the default
  //        timesteps -- FIXME TODO
  new_dw->put(delt_vartype(1.0e10), lb->delTLabel, level);
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,
              "Doing MPM::interpolateParticlesToGrid");

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
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature, pColor;
      constParticleVariable<double> pexternalheatrate;
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
      if (flags->d_with_color) {
        old_dw->get(pColor,       lb->pColorLabel,         pset);
      }
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      if (flags->d_GEVelProj){
        old_dw->get(pVelGrad,     lb->pVelGradLabel,             pset);
        old_dw->get(pTempGrad,    lb->pTemperatureGradientLabel, pset);
      }
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      new_dw->get(psize,          lb->pCurSizeLabel,       pset);

      // JBH -- Scalar diffusion related
      constParticleVariable<double> pConcentration, pExternalScalarFlux;
      constParticleVariable<Vector> pConcGrad;
      constParticleVariable<Matrix3> pStress;
      if (flags->d_doScalarDiffusion) {
        new_dw->get(pExternalScalarFlux, lb->diffusion->pExternalScalarFlux_preReloc, pset);
        old_dw->get(pConcentration,      lb->diffusion->pConcentration,   pset);
        old_dw->get(pStress,             lb->pStressLabel,                pset);
        if (flags->d_GEVelProj) {
          old_dw->get(pConcGrad, lb->diffusion->pGradConcentration, pset);
        }
      }

      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc,         pset);
      new_dw->get(pexternalheatrate, 
                                  lb->pExternalHeatRateLabel_preReloc, pset);
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
      NCVariable<double> gColor;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gTemperatureRate;

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
      if (flags->d_with_color) {
        new_dw->allocateAndPut(gColor,         lb->gColorLabel,      dwi,patch);
        gColor.initialize(0.);
      }

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gTemperatureRate.initialize(0);
      gexternalheatrate.initialize(0);
      gSp_vol.initialize(0.);

      // JBH -- Scalar diffusion related
      NCVariable<double>  gConcentration, gConcentrationNoBC;
      NCVariable<double>  gHydrostaticStress, gExtScalarFlux;
      if (flags->d_doScalarDiffusion) {
        new_dw->allocateAndPut(gConcentration,lb->diffusion->gConcentration,
                                               dwi, patch);
        new_dw->allocateAndPut(gConcentrationNoBC, 
                                              lb->diffusion->gConcentrationNoBC,
                                                   dwi, patch);
        new_dw->allocateAndPut(gHydrostaticStress, 
                                              lb->diffusion->gHydrostaticStress,
                                                   dwi, patch);
        new_dw->allocateAndPut(gExtScalarFlux,
                                              lb->diffusion->gExternalScalarFlux,
                                               dwi, patch);
        gConcentration.initialize(0);
        gConcentrationNoBC.initialize(0);
        gHydrostaticStress.initialize(0);
        gExtScalarFlux.initialize(0);
      }

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
     if(mpm_matl->getIsActive()){
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
            if (flags->d_with_color) {
              gColor[node]       += pColor[idx]*pmass[idx]         * S[k];
            }
            if (!flags->d_useCBDI) {
              gexternalforce[node] += pexternalforce[idx]          * S[k];
            }
            gTemperature[node]   += ptemp_ext * pmass[idx] * S[k];
            gSp_vol[node]        += pSp_vol   * pmass[idx] * S[k];
            gexternalheatrate[node] += pexternalheatrate[idx]      * S[k];
          }
        }
        if (flags->d_doScalarDiffusion) {
          double one_third = 1./3.;
          double pHydroStress = one_third*pStress[idx].Trace();
          double pConc_Ext = pConcentration[idx];
          for (int k = 0; k < NN; ++k) {
            node = ni[k];
            if (patch->containsNode(node)) {
              if (flags->d_GEVelProj) {
                Point gpos = patch->getNodePosition(node);
                Vector pointOffset = px[idx]-gpos;
                pConc_Ext -= Dot(pConcGrad[idx],pointOffset);
              }
              double massWeight = pmass[idx]*S[k];
              gHydrostaticStress[node]  += pHydroStress             *massWeight;
              gConcentration[node]      += pConc_Ext                *massWeight;
              gExtScalarFlux[node]      += pExternalScalarFlux[idx] *massWeight;
            }
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
        if (flags->d_with_color) {
          gColor[c]       /= gmass[c];
        }
        gtempglobal[c]    += gTemperature[c];
        gTemperature[c]   /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gSp_vol[c]        /= gmass[c];
      }

      if (flags->d_doScalarDiffusion) {
        for (NodeIterator iter=patch->getExtraNodeIterator();
             !iter.done(); ++iter) {
          IntVector c = *iter;
          gConcentration[c]     /= gmass[c];
          gHydrostaticStress[c] /= gmass[c];
          gConcentrationNoBC[c]  = gConcentration[c];
        }
      }

      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gvelocity,   interp_type);
      if (flags->d_doScalarDiffusion) {
        bc.setBoundaryCondition(patch, dwi, "SD-Type", gConcentration,
                                                                   interp_type);
      }

      // If an MPMICE problem, create a velocity with BCs variable for NCToCC_0
      if(flags->d_with_ice){
        NCVariable<Vector> gvelocityWBC;
        new_dw->allocateAndPut(gvelocityWBC,lb->gVelocityBCLabel,dwi,patch);
        gvelocityWBC.copyData(gvelocity);
        bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocityWBC,interp_type);
        bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocityWBC,interp_type);
      }
     } // is material active
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

void SerialMPM::computeSSPlusVp(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::computeSSPlusVp");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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

      new_dw->allocateAndPut(pvelSSPlus,lb->pVelocitySSPlusLabel,    pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity,    lb->gVelocityLabel,   dwi,patch,gac,NGP);

      if(!mpm_matl->getIsRigid()){
        // Loop over particles
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S,
                                                    psize[idx]);
          // Accumulate the contribution from each surrounding vertex
          Vector vel(0.0,0.0,0.0);
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel      += gvelocity[node]  * S[k];
          }
          pvelSSPlus[idx]    = vel;
        }
      }  // only do this if the material is not rigid
    }  // loop over materials
    delete interpolator;
  } // loop over patches
}

void SerialMPM::computeSPlusSSPlusVp(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::computeSPlusSSPlusVp");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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

      if(!mpm_matl->getIsRigid()){
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
        }  // loop over nodes
      }  // only do if not a rigid material
    } // loop over materials
    delete interpolator;
  }
}

void SerialMPM::computeStressTensor(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0),cout_doing,
            "Doing MPM::computeStressTensor");

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
    printTask(patches, patch,cout_doing,"Doing MPM::computeContactArea");

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
    printTask(patches, patch,cout_doing,"Doing MPM::computeInternalForce");

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
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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

      if(!mpm_matl->getIsRigid() && mpm_matl->getIsActive()){
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
        } else {
          // for the axisymmetric case
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

          // loop over face nodes to find boundary forces, ave. stress(traction)
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
                 bndyTraction[iface][ic]+=gstress[ijk](ic,jc)*norm[jc]*nodearea;
                }
              }
            }
          }
        } // faces

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch,dwi, "Symmetric",
                                internalforce, interp_type);
      } // is not rigid and is active
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

void SerialMPM::computeAndIntegrateDiffusion(const  ProcessorGroup  *
                                            ,const  PatchSubset     * patches
                                            ,const  MaterialSubset  *
                                            ,       DataWarehouse   * old_dw
                                            ,       DataWarehouse   * new_dw  )
{
  for (int p=0; p < patches->size(); ++p) {
    const Patch*  patch = patches->get(p);
    printTask(patches, patch, cout_doing, "Doing MPM::computeAndIntegrateDiffusion");

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    Ghost::GhostType  gnone = Ghost::None;
    for (unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); ++m) {
      MPMMaterial*  mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();

      // Get required variables for this patch
      constNCVariable<double> mass;
      new_dw->get(mass,         lb->gMassLabel,          dwi, patch, gnone, 0);

      // Scalar Diffusion Related Variables -- JBH
      constNCVariable<double> gSD_IF_FluxRate;
      constNCVariable<double> gConcentration, gConcNoBC, gExternalScalarFlux;
      NCVariable<double> gConcRate, gConcStar;
      const VarLabel* SD_IF_FluxLabel =
                        d_sdInterfaceModel->getInterfaceFluxLabel();
      new_dw->get(gSD_IF_FluxRate,      SD_IF_FluxLabel,
                  dwi,  patch, gnone, 0);
      new_dw->get(gConcentration,       lb->diffusion->gConcentration,
                  dwi,  patch, gnone, 0);
      new_dw->get(gConcNoBC,            lb->diffusion->gConcentrationNoBC,
                  dwi,  patch, gnone, 0);
      new_dw->get(gExternalScalarFlux,  lb->diffusion->gExternalScalarFlux,
                  dwi,  patch, gnone, 0);

      new_dw->getModifiable( gConcRate, lb->diffusion->gConcentrationRate,
                             dwi, patch);
      new_dw->allocateAndPut(gConcStar, lb->diffusion->gConcentrationStar,
                             dwi, patch);

      // JBH -- Variables associated with scalar diffusion

      for (NodeIterator iter=patch->getExtraNodeIterator();
           !iter.done(); ++iter) {
        IntVector node = *iter;
//        gConcRate[node] /= mass[node];
        gConcStar[node]  = gConcentration[node] +
                            (gConcRate[node]/mass[node] + gSD_IF_FluxRate[node]);
      }
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch, dwi, "SD-Type", gConcStar,
                              flags->d_interpolator_type);
      for (NodeIterator iter=patch->getExtraNodeIterator();
           !iter.done(); ++iter) {
        IntVector node = *iter;
        gConcRate[node] = (gConcStar[node] - gConcNoBC[node]) / delT
                           + gExternalScalarFlux[node]/mass[node];
      }
    }
  }
}

void SerialMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  IntVector lowNode, highNode;
  level->findInteriorNodeIndexRange(lowNode, highNode);
  string interp_type = flags->d_interpolator_type;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
                       "Doing MPM::computeAndIntegrateAcceleration");

    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    map<int,double> totalMass;
    double allMatls_totalMass  = 0.0;
    map<int,Vector> STF;
    Vector allMatls_STF  = Vector(0.,0.,0.);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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

      totalMass[dwi] = 0.0;
      STF[dwi] = Vector(0.,0.,0.);

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

        totalMass[dwi]     += mass[c];
        allMatls_totalMass += mass[c];
        STF[dwi]           +=externalforce[c];
        allMatls_STF       +=externalforce[c];
      }

      if(mpm_matl->getIsFTM()){
        acceleration.initialize(Vector(0.,0.,0.));
        velocity_star.copyData(velocity);
      }

      // Check the integrated nodal velocity and if the product of velocity
      // and timestep size is larger than half the cell size, restart the
      // timestep with 10% as large of a timestep (see recomputeDelT in this
      // file).
      if(flags->d_restartOnLargeNodalVelocity){
        Vector dxCell = patch->dCell();
        double cell_size_sq = dxCell.length2();
        for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
          IntVector c = *iter;
          if(c.x()>lowNode.x() && c.x()<highNode.x() &&
             c.y()>lowNode.y() && c.y()<highNode.y() &&
             c.z()>lowNode.z() && c.z()<highNode.z()){
            if((velocity_star[c]*delT).length2() > 0.25*cell_size_sq){
             cerr << "Aborting timestep, velocity star too large" << endl;
             cerr << "velocity_star[" << c << "] = " << velocity_star[c] << endl;
             new_dw->put( bool_or_vartype(true),
                          VarLabel::find(abortTimeStep_name));
             new_dw->put( bool_or_vartype(true),
                          VarLabel::find(recomputeTimeStep_name));
            }
          }
        }
      }
    }    // matls

    //__________________________________
    //  put the reduction variables
    const MaterialSubset* matls = 
                           m_materialManager->allMaterials( "MPM" )->getUnion();

    if( flags->d_reductionVars->mass ||
        flags->d_reductionVars->sumTransmittedForce){
      new_dw->put( sum_vartype(allMatls_totalMass),
                                          lb->TotalMassLabel, nullptr, -1);
      new_dw->put_sum_vartype( totalMass, lb->TotalMassLabel, matls );
    }
    if( flags->d_reductionVars->sumTransmittedForce ){
      new_dw->put(sumvec_vartype(allMatls_STF),
                                     lb->SumTransmittedForceLabel, nullptr, -1);
      new_dw->put_sum_vartype( STF,  lb->SumTransmittedForceLabel, matls);
    }
  }
}

void SerialMPM::computeGridVelocityForFTM(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::computeGridVelocityForFTM");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    // DON'T MOVE THESE!!!
    map<int,double> zeroD = initializeMap(0.0);
    map<int,Vector> zeroV = initializeMap(Vector(0.));
    map<int,double> totalMass  = zeroD;
    map<int,double> totalMOI  = zeroD;
    map<int,Vector> totalSTF   = zeroV;
    map<int,Vector> totalSTT   = zeroV;
    map<int,Vector> totalSRI   = zeroV;
    map<int,Vector> totalMom   = zeroV;
    map<int,Vector> CMX        = zeroV;

    Vector allMatls_totalMom(0.0,0.0,0.0);
    Vector allMatls_CMX(0.0,0.0,0.0);

    if(flags->d_reductionVars->mass ||
       flags->d_reductionVars->sumTransmittedForce){
      totalMass = new_dw->get_sum_vartypeD(lb->TotalMassLabel, matls);
      totalSTF = new_dw->get_sum_vartypeV(lb->SumTransmittedForceLabel,  matls);
    }

    string interp_type = flags->d_interpolator_type;
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      if(mpm_matl->getIsFTM()){
        int dwi = mpm_matl->getDWIndex();
        Vector FTM_acc   = totalSTF[dwi]/totalMass[dwi];

        NCVariable<Vector> gvelocity_star, gacceleration;
        constNCVariable<Vector> gvelocity;

        new_dw->getModifiable(gacceleration, lb->gAccelerationLabel, dwi,patch);
        new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel, dwi,patch);
        new_dw->get(gvelocity,               lb->gVelocityLabel,     dwi,patch,
                                                                 Ghost::None,0);

        // Now recompute acceleration as the difference between the velocity
        // interpolated to the grid (no bcs applied) and the new velocity_star
        for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();
                                                                  iter++){
          IntVector c = *iter;
          gacceleration[c] = FTM_acc;
          gvelocity_star[c] = gvelocity[c] + FTM_acc*delT;
        }
      } // is FTM
    } // matl loop
  }  // patch loop

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
              "Doing MPM::setGridBoundaryConditions");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    string interp_type = flags->d_interpolator_type;
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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
  }  // patch loop
}

void SerialMPM::setPrescribedMotion(const ProcessorGroup*,
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
    printTask(patches, patch,cout_doing, "Doing MPM::setPrescribedMotion");

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

void SerialMPM::computeCurrentParticleSize(const ProcessorGroup* ,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,
              "Doing MPM::computeCurrentParticleSize");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    string interp_type = flags->d_interpolator_type;

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Matrix3> pSize;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Matrix3> pCurSize;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(pSize,                lb->pSizeLabel,               pset);
      old_dw->get(pFOld,                lb->pDeformationMeasureLabel, pset);
      new_dw->allocateAndPut(pCurSize,  lb->pCurSizeLabel,            pset);

      if(interp_type == "cpdi" || interp_type == "fast_cpdi"
                               || interp_type == "cpti"){
        if(flags->d_axisymmetric){
          for (ParticleSubset::iterator iter = pset->begin();
               iter != pset->end(); iter++){
            particleIndex idx = *iter;
            Matrix3 defgrad1=Matrix3(pFOld[idx](0,0),pFOld[idx](0,1),0.0,
                                     pFOld[idx](1,0),pFOld[idx](1,1),0.0,
                                     0.0,            0.0,            1.0);

            pCurSize[idx] = defgrad1*pSize[idx];
          }
        } else {
          for (ParticleSubset::iterator iter = pset->begin();
               iter != pset->end(); iter++){
            particleIndex idx = *iter;

            pCurSize[idx] = pFOld[idx]*pSize[idx];
          }
        }
      } else {
        pCurSize.copyData(pSize);
#if 0
        for (ParticleSubset::iterator iter = pset->begin();
             iter != pset->end(); iter++){
          particleIndex idx = *iter;

          pCurSize[idx] = pSize[idx];
        }
#endif
      }
    }
  }
}

void SerialMPM::applyExternalLoads(const ProcessorGroup* ,
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
  std::vector<double> torquePerPart;
  std::vector<double> fluxPerPart;
  std::vector<PressureBC*> pbcP;
  std::vector<TorqueBC*> TBC;
  std::vector<HeatFluxBC*> HFBC;

  if (flags->d_useLoadCurves) {
    for (int ii = 0;ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {

        PressureBC* pbc =
          dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // Calculate the force per particle at current time
        forcePerPart.push_back(pbc->forcePerParticle(time));
      } else if (bcs_type == "Torque") {

        TorqueBC* tbc =
          dynamic_cast<TorqueBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        TBC.push_back(tbc);

        // Calculate the force per particle at current time
        torquePerPart.push_back(tbc->torquePerParticle(time));
      } else if (bcs_type == "HeatFlux") {

        HeatFluxBC* hfbc =
          dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        HFBC.push_back(hfbc);

        // Calculate the force per particle at current time
        fluxPerPart.push_back(hfbc->fluxPerParticle(time));
      }
    }
  }


  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing MPM::applyExternalLoads");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<Vector>       pExternalForce_new;
      ParticleVariable<double>       pExternalHeatRate_new;
      new_dw->allocateAndPut(pExternalForce_new,
                             lb->pExtForceLabel_preReloc,          pset);
      new_dw->allocateAndPut(pExternalHeatRate_new,
                             lb->pExternalHeatRateLabel_preReloc,  pset);

      // pExternalForce is either:
      //  set to zero (default)
      //  set using load curves
      //  set using an MMS formulation

      ParticleSubset::iterator iter = pset->begin();
      for(;iter != pset->end(); iter++){
         particleIndex idx = *iter;
         pExternalForce_new[idx] = Vector(0.,0.,0.);
         pExternalHeatRate_new[idx] = 0.;
      }

      string mms_type = flags->d_mms_type;
      if (flags->d_useLoadCurves) {
        bool do_PressureBCs=false;
        bool do_TorqueBCs  =false;
        bool do_HeatFluxBCs=false;
        int numPressureLCs = 0;
        int numTorqueLCs   = 0;
        int numHeatFluxLCs = 0;
        old_dw->get(px,    lb->pXLabel,    pset);
        for (int ii = 0;
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
          string bcs_type =
            MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
          if (bcs_type == "Pressure") {
            do_PressureBCs=true;
            numPressureLCs++;
          } else if (bcs_type == "Torque") {
            do_TorqueBCs=true;
            numTorqueLCs++;
          } else if (bcs_type == "HeatFlux") {
            do_HeatFluxBCs=true;
            numHeatFluxLCs++;
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

        // Get the load curve data
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
           for(int k=0;k<3;k++){
            int loadCurveID = pLoadCurveID[idx](k)-1;
            if (loadCurveID >= 0 && loadCurveID < numPressureLCs) {
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
        }

        if(flags->d_keepPressBCNormalToSurface){
         old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel,pset);
         for(ParticleSubset::iterator iter = pset->begin();
                                      iter != pset->end(); iter++){
            particleIndex idx = *iter;
            Matrix3 F = pDeformationMeasure[idx];
            Matrix3 R, V;
            F.polarDecompositionRMB(V, R);
            Vector pF = pExternalForce_new[idx];

            pExternalForce_new[idx] = R*pF;
          }
        }

        // Get the load curve data
        if(do_TorqueBCs){
          // Iterate over the particles
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
           particleIndex idx = *iter;
           for(int k=0;k<3;k++){
            //int loadCurveIndex = pLoadCurveID[idx](k)-numPressureLCs;
            int loadCurveIndex = pLoadCurveID[idx](k)-1;
            if (loadCurveIndex >= 0 && pLoadCurveID[idx](k) > numPressureLCs) {
              TorqueBC* tbc = TBC[loadCurveIndex - numPressureLCs];
              double torque = torquePerPart[loadCurveIndex-numPressureLCs];

              pExternalForce_new[idx]+=tbc->getForceVector(px[idx],torque,time);
            } // loadCurveID >=0
           }  // loop over elements of the IntVector
          }
        }

        // Get the load curve data
        if(do_HeatFluxBCs){
          // Iterate over the particles
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
           particleIndex idx = *iter;
           for(int k=0;k<3;k++){
            int loadCurveIndex = pLoadCurveID[idx](k)-1;
            if (loadCurveIndex >= 0) {
              HeatFluxBC* hfbc = HFBC[loadCurveIndex];
              double flux = fluxPerPart[loadCurveIndex];

              pExternalHeatRate_new[idx]+=hfbc->getFlux(px[idx], flux);
            } // loadCurveID >=0
           }  // loop over elements of the IntVector
          }
        }
      } else if(!mms_type.empty()) {
        // MMS
        MMS MMSObject;
        MMSObject.computeExternalForceForMMS(old_dw,new_dw,time,pset,
                                            lb,flags,pExternalForce_new);
      }
    } // matl loop
  }  // patch loop
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* matls ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::interpolateToParticlesAndUpdate");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    // DON'T MOVE THESE!!!
    map<int,double> zeroD = initializeMap(0.0);
    map<int,Vector> zeroV = initializeMap(Vector(0.));
    map<int,double> kineticEng = zeroD;
    map<int,double> thermalEng = zeroD;
    map<int,Vector> totalMom   = zeroV;
    map<int,Vector> CMX        = zeroV;

//    double allMatls_angMom     = 0.0;
    double allMatls_kineticEng = 0.0;
    double allMatls_thermalEng = 0.0;
    Vector allMatls_totalMom(0.0,0.0,0.0);
    Vector allMatls_CMX(0.0,0.0,0.0);

    double totalConc    =   0.0;
    double minPatchConc =  5e11;
    double maxPatchConc = -5e11;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    //Carry forward NC_CCweight (put outside of matl loop, only need for matl 0)
    constNCVariable<double> NC_CCweight;
    NCVariable<double> NC_CCweight_new;
    Ghost::GhostType  gnone = Ghost::None;
    old_dw->get(NC_CCweight,       lb->NC_CCweightLabel,  0, patch, gnone, 0);
    new_dw->allocateAndPut(NC_CCweight_new, lb->NC_CCweightLabel,0,patch);
    NC_CCweight_new.copyData(NC_CCweight);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pvelocity, pvelSSPlus, pdisp;
      constParticleVariable<Matrix3> psize, pFOld, pcursize;
      constParticleVariable<double> pmass, pVolumeOld, pTemperature;
      constParticleVariable<long64> pids;
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pvelnew, pdispnew;
      ParticleVariable<Matrix3> psizeNew;
      ParticleVariable<double> pmassNew,pTempNew;
      ParticleVariable<long64> pids_new;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration, gvelSPSSP;
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      old_dw->get(pVolumeOld,   lb->pVolumeLabel,                    pset);
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
      old_dw->get(psize,               lb->pSizeLabel,                pset);
      new_dw->get(pcursize,            lb->pCurSizeLabel,             pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,       pset);
      pids_new.copyData(pids);

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

      Ghost::GhostType  gac = Ghost::AroundCells;
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

      double Cp=mpm_matl->getSpecificHeat();

      // Diffusion related - JBH
      double sdmMaxEffectiveConc = -999;
      double sdmMinEffectiveConc =  999;
      constParticleVariable<double> pConcentration;
      constNCVariable<double>       gConcentrationRate;

      ParticleVariable<double>      pConcentrationNew, pConcPreviousNew;
      if (flags->d_doScalarDiffusion) {
        // Grab min/max concentration and conc. tolerance for particle loop.
        ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        sdmMaxEffectiveConc = sdm->getMaxConcentration() 
                            - sdm->getConcentrationTolerance();
        sdmMinEffectiveConc = sdm->getMinConcentration() 
                            + sdm->getConcentrationTolerance();

        old_dw->get(pConcentration,     lb->diffusion->pConcentration,    pset);
        new_dw->get(gConcentrationRate, lb->diffusion->gConcentrationRate,
                                                         dwi,  patch, gac, NGP);

        new_dw->allocateAndPut(pConcentrationNew,
                                  lb->diffusion->pConcentration_preReloc, pset);
        new_dw->allocateAndPut(pConcPreviousNew,
                                  lb->diffusion->pConcPrevious_preReloc,  pset);
      }

      if(flags->d_XPIC2 && !mpm_matl->getIsRigid()){
        // Loop over particles
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S,
                                                    pcursize[idx]);
          Vector vel(0.0,0.0,0.0);
          Vector velSSPSSP(0.0,0.0,0.0);
          Vector acc(0.0,0.0,0.0);
          double fricTempRate = 0.0;
          double tempRate = 0.0;
          double concRate = 0.0;
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

          if (flags->d_doScalarDiffusion) {
            for (int k = 0; k < NN; ++k) {
              IntVector node = ni[k];
              concRate += gConcentrationRate[node] * S[k];
            }

            pConcentrationNew[idx] = pConcentration[idx] + concRate * delT;
            if (pConcentrationNew[idx] < sdmMinEffectiveConc) {
              pConcentrationNew[idx] = sdmMinEffectiveConc;
            }
            if (pConcentrationNew[idx] > sdmMaxEffectiveConc) {
              pConcentrationNew[idx] = sdmMaxEffectiveConc;
            }

            pConcPreviousNew[idx] = pConcentration[idx];
            if (mpm_matl->doConcReduction()) {
              if (flags->d_autoCycleUseMinMax) {
                if (pConcentrationNew[idx] > maxPatchConc)
                  maxPatchConc = pConcentrationNew[idx];
                if (pConcentrationNew[idx] < minPatchConc)
                  minPatchConc = pConcentrationNew[idx];
              } else {
                totalConc += pConcentration[idx];
              }
            }
          }

          //__________________________________
          // reduction variables

          Vector centerOfMass = (pxnew[idx]*pmass[idx]).asVector();
          CMX[dwi]      += centerOfMass;
          allMatls_CMX  += centerOfMass;

          double thermalEngy   = pTemperature[idx] * pmass[idx] * Cp;
          thermalEng[dwi]     += thermalEngy;
          allMatls_thermalEng += thermalEngy;

          double ke            = .5*pmass[idx]*pvelnew[idx].length2();
          kineticEng[dwi]     += ke;
          allMatls_kineticEng += ke;

          Vector mom         = pvelnew[idx]*pmass[idx];
          allMatls_totalMom += mom;
          totalMom[dwi]     += mom;

//          Vector angular_mom = pmass[idx]*Cross(pvelnew[idx],pxnew[idx].asVector());
//          allMatls_angMom += angular_mom.length();
        }
      } else {  // Not XPIC(2)
        // Loop over particles
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S,
                                                    pcursize[idx]);
          Vector vel(0.0,0.0,0.0);
          Vector acc(0.0,0.0,0.0);
          double fricTempRate = 0.0;
          double tempRate = 0.0;
          double concRate = 0.0;
          double burnFraction = 0.0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel      += gvelocity_star[node]  * S[k];
            acc      += gacceleration[node]   * S[k];

            fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
            tempRate += (gTemperatureRate[node] + dTdt[node] +
                         fricTempRate)   * S[k];
            burnFraction += massBurnFrac[node]     * S[k];
          }

          // Update the particle's pos and vel using std "FLIP" method
          pxnew[idx]   = px[idx]        + vel*delT;
          pdispnew[idx]= pdisp[idx]     + vel*delT;
          pvelnew[idx] = pvelocity[idx] + acc*delT;

          pTempNew[idx]    = pTemperature[idx] + tempRate*delT;
          pTempPreNew[idx] = pTemperature[idx]; // for thermal stress
          pmassNew[idx]    = Max(pmass[idx]*(1.    - burnFraction),0.);
          psizeNew[idx]    = (pmassNew[idx]/pmass[idx])*psize[idx];

          if (flags->d_doScalarDiffusion) {
            for (int k = 0; k < NN; ++k) {
              IntVector node = ni[k];
              concRate += gConcentrationRate[node] * S[k];
            }

            pConcentrationNew[idx] = pConcentration[idx] + concRate * delT;
            if (pConcentrationNew[idx] < sdmMinEffectiveConc) {
              pConcentrationNew[idx] = sdmMinEffectiveConc;
            }
            if (pConcentrationNew[idx] > sdmMaxEffectiveConc) {
              pConcentrationNew[idx] = sdmMaxEffectiveConc;
            }

            pConcPreviousNew[idx] = pConcentration[idx];
            if (mpm_matl->doConcReduction()) {
              if (flags->d_autoCycleUseMinMax) {
                if (pConcentrationNew[idx] > maxPatchConc)
                  maxPatchConc = pConcentrationNew[idx];
                if (pConcentrationNew[idx] < minPatchConc)
                  minPatchConc = pConcentrationNew[idx];
              } else {
                totalConc += pConcentration[idx];
              }
            }
          }

          //__________________________________
          // reduction variables

          Vector centerOfMass = (pxnew[idx]*pmass[idx]).asVector();
          CMX[dwi]      += centerOfMass;
          allMatls_CMX  += centerOfMass;

          double thermalEngy   = pTemperature[idx] * pmass[idx] * Cp;
          thermalEng[dwi]     += thermalEngy;
          allMatls_thermalEng += thermalEngy;

          double ke            = .5*pmass[idx]*pvelnew[idx].length2();
          kineticEng[dwi]     += ke;
          allMatls_kineticEng += ke;

          Vector mom = pvelnew[idx]*pmass[idx];
          allMatls_totalMom += mom;
          totalMom[dwi]     += mom;

//          Vector angular_mom = pmass[idx]*Cross(pvelnew[idx],pxnew[idx].asVector());
//          allMatls_angMom += angular_mom.length();
        }
      } // use XPIC(2) or not
    }  // loop over materials

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
    if( flags->d_reductionVars->momentum ){
      new_dw->put( sumvec_vartype(allMatls_totalMom), 
                   lb->TotalMomentumLabel, nullptr, -1);

      new_dw->put_sum_vartype( totalMom,   lb->TotalMomentumLabel, matls );
    }

    if( flags->d_reductionVars->KE ){
      new_dw->put( sum_vartype(allMatls_kineticEng),  
                   lb->KineticEnergyLabel, nullptr, -1);

      new_dw->put_sum_vartype( kineticEng, lb->KineticEnergyLabel, matls );
    }

    if( flags->d_reductionVars->thermalEnergy ){
      new_dw->put( sum_vartype(allMatls_thermalEng),  
                   lb->ThermalEnergyLabel, nullptr, -1);

      new_dw->put_sum_vartype( thermalEng, lb->ThermalEnergyLabel, matls );

    }

    if(flags->d_reductionVars->centerOfMass){
      new_dw->put(sumvec_vartype(allMatls_CMX), 
                  lb->CenterOfMassPositionLabel, nullptr, -1);

      new_dw->put_sum_vartype( CMX,  lb->CenterOfMassPositionLabel, matls );
    }

    delete interpolator;
  }
}

void SerialMPM::computeParticleGradients(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::computeParticleGradients");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    double partvoldef = 0.;

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize, pSizeOrig;
      constParticleVariable<double> pVolumeOld,pmass,pmassNew;
      constParticleVariable<int> pLocalized;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<double> pvolume,pTempNew;
      ParticleVariable<Matrix3> pFNew,pVelGrad;
      ParticleVariable<Vector> pTempGrad;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector>  gvelocity_star;
      constNCVariable<double>  gTempStar;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);
      old_dw->get(pSizeOrig,    lb->pSizeLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(pVolumeOld,   lb->pVolumeLabel,                    pset);
      old_dw->get(pLocalized,   lb->pLocalizedMPMLabel,              pset);

      new_dw->allocateAndPut(pvolume,    lb->pVolumeLabel_preReloc,       pset);
      new_dw->allocateAndPut(pVelGrad,   lb->pVelGradLabel_preReloc,      pset);
      new_dw->allocateAndPut(pTempGrad,  lb->pTemperatureGradientLabel_preReloc,
                                                                          pset);
      new_dw->allocateAndPut(pFNew,      lb->pDeformationMeasureLabel_preReloc,
                                                                          pset);

      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      if (flags->d_doExplicitHeatConduction){
        new_dw->get(gTempStar,     lb->gTemperatureStarLabel,dwi,patch,gac,NGP);
      }

      // Compute velocity gradient and deformation gradient on every particle
      // This can/should be combined into the loop above, once it is working
      Matrix3 Identity;
      Identity.Identity();

      // JBH -- Scalar diffusion related variables
      constParticleVariable<Vector> pArea;
      constNCVariable<double>       gConcStar;
      ParticleVariable<Vector>      pConcGradNew, pAreaNew;
      if (flags->d_doScalarDiffusion) {
        old_dw->get(pArea, lb->diffusion->pArea, pset);
        new_dw->get(gConcStar, lb->diffusion->gConcentrationStar,
                    dwi, patch, gac, NGP);

        new_dw->allocateAndPut(pAreaNew, lb->diffusion->pArea_preReloc,
                               pset);
        new_dw->allocateAndPut(pConcGradNew, lb->diffusion->pGradConcentration_preReloc,
                               pset);
      }

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        int NN=flags->d_8or27;
        Matrix3 tensorL(0.0);
        if(!flags->d_axisymmetric){
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndShapeDerivatives(px[idx],ni,
                                                     d_S,psize[idx]);
         computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity_star,NN);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,
                                                   S,d_S,psize[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity_star,
                                                                   px[idx],NN);
        }
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

        if (flags->d_doScalarDiffusion) {
          pAreaNew[idx] = pArea[idx];
          pConcGradNew[idx] = Vector(0.0, 0.0, 0.0);
          for (int k = 0; k < NN; ++k) {
            for (int j = 0; j < 3; ++j) {
              pConcGradNew[idx][j] += gConcStar[ni[k]] * d_S[k][j] * oodx[j];
            }
          }
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
        if(flags->d_doGranularMPM){ //MJ //JG
          double Vcrix = flags->d_GranularMPM_Vcrix;
          double Vcriy = flags->d_GranularMPM_Vcriy;
          double Vcriz = flags->d_GranularMPM_Vcriz;
          if(flags->d_ndim<=2){
            Vcriz=1.0;
          }
          double r1=dx.x()*pSizeOrig[idx](0,0);
          double r2=dx.y()*pSizeOrig[idx](1,1);
          double r3=dx.z()*pSizeOrig[idx](2,2);
          const double Vcri = (Vcrix*r1 * Vcriy*r2 * Vcriz*r3);
          if (pvolume[idx]> (Vcri + 1.0e-12)){
            pFNew[idx] = pFOld[idx];
            pvolume[idx] = pVolumeOld[idx];
          }
        } // end if Granular MPM
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

          if(flags->d_doGranularMPM){ //MJ //JG
            double Vcrix = flags->d_GranularMPM_Vcrix;
            double Vcriy = flags->d_GranularMPM_Vcriy;
            double Vcriz = flags->d_GranularMPM_Vcriz;
            if(flags->d_ndim<=2){
              Vcriz=1.0;
            }
            double r1=dx.x()*pSizeOrig[idx](0,0);
            double r2=dx.y()*pSizeOrig[idx](1,1);
            double r3=dx.z()*pSizeOrig[idx](2,2);
            const double Vcri = (Vcrix*r1 * Vcriy*r2 * Vcriz*r3);
            if (pvolume[idx]> (Vcri + 1.0e-12)){
               pFNew[idx] = pFOld[idx];
               pvolume[idx] = pVolumeOld[idx];
            }
          } // end if Granular MPM
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

void SerialMPM::finalParticleUpdate(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::finalParticleUpdate");

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

void SerialMPM::insertParticles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing MPM::insertParticles");

    // Get the current simulation time
    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, lb->simulationTimeLabel);
    double time = simTimeVar;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    // activate materials if it is their time
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                       (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      if(time >= mpm_matl->getActivationTime()){
         mpm_matl->setIsActive(true);
      }
    }

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

void SerialMPM::addParticles(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* ,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    printTask(patches, patch,cout_doing, "Doing MPM::addParticles");
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    //Carry forward CellNAPID
    constCCVariable<int> NAPID;
    CCVariable<int> NAPID_new;
    Ghost::GhostType  gnone = Ghost::None;
    old_dw->get(NAPID,               lb->pCellNAPIDLabel,    0,patch,gnone,0);
    new_dw->allocateAndPut(NAPID_new,lb->pCellNAPIDLabel,    0,patch);
    NAPID_new.copyData(NAPID);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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
      // JBH -- Scalard diffusion variables
      ParticleVariable<double> pConc, pConcPrev, pD, pESFlux;
      ParticleVariable<Vector> pGradConc, pArea;
      if (flags->d_doScalarDiffusion) {
        new_dw->getModifiable(pConc,      lb->diffusion->pConcentration_preReloc,     pset);
        new_dw->getModifiable(pConcPrev,  lb->diffusion->pConcPrevious_preReloc,      pset);
        new_dw->getModifiable(pGradConc,  lb->diffusion->pGradConcentration_preReloc, pset);
        new_dw->getModifiable(pESFlux,    lb->diffusion->pExternalScalarFlux_preReloc,pset);
        new_dw->getModifiable(pArea,      lb->diffusion->pArea_preReloc,              pset);
        new_dw->getModifiable(pD,         lb->diffusion->pDiffusivity_preReloc,       pset);
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
              } else if (R1_R3_ratSq > tVSq && flags->d_ndim==3){
                pSplitR1R2R3[pp]=2;
              } else if (R1_R3_ratSq < tV_invSq && flags->d_ndim==3){
                pSplitR1R2R3[pp]=-2;
              } else if (R2_R3_ratSq > tVSq && flags->d_ndim==3){
                 pSplitR1R2R3[pp]=3;
              } else if (R2_R3_ratSq < tV_invSq && flags->d_ndim==3){
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

      int fourOrEight=pow(2,flags->d_ndim);
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

      // JBH - Scalar Diffusion Variables
      ParticleVariable<double> pConcTmp, pConcPrevTmp, pESFluxTmp, pDTmp;
      ParticleVariable<Vector> pGradConcTmp, pAreaTmp;
      if (flags->d_doScalarDiffusion) {
        new_dw->allocateTemporary(pConcTmp,     pset);
        new_dw->allocateTemporary(pConcPrevTmp, pset);
        new_dw->allocateTemporary(pGradConcTmp, pset);
        new_dw->allocateTemporary(pESFluxTmp,   pset);
        new_dw->allocateTemporary(pDTmp,        pset);
        new_dw->allocateTemporary(pAreaTmp,     pset);
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

      if (flags->d_doScalarDiffusion) {
        for (unsigned int pp=0; pp < oldNumPar; ++pp) {
          pConcTmp[pp]      = pConc[pp];
          pConcPrevTmp[pp]  = pConcPrev[pp];
          pGradConcTmp[pp]  = pGradConc[pp];
          pESFluxTmp[pp]    = pESFlux[pp];
          pAreaTmp[pp]      = pArea[pp];
          pDTmp[pp]         = pD[pp];
        }
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

        int comp = 0;
        int last_index = -999;
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
              comp=0;
              dSNew = Matrix3(0.25*dsize(0,0), dsize(0,1), dsize(0,2),
                              0.25*dsize(1,0), dsize(1,1), dsize(1,2),
                              0.25*dsize(2,0), dsize(2,1), dsize(2,2));
            } else if(pSplitR1R2R3[idx]==3 || pSplitR1R2R3[idx]==-1){
              // Split across the second R-vector
              comp=1;
              dSNew = Matrix3(dsize(0,0), 0.25*dsize(0,1), dsize(0,2),
                              dsize(1,0), 0.25*dsize(1,1), dsize(1,2),
                              dsize(2,0), 0.25*dsize(2,1), dsize(2,2));
            } else if(pSplitR1R2R3[idx]==-2 || pSplitR1R2R3[idx]==-3){
              // Split across the third R-vector
              comp=2;
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
          if (flags->d_doScalarDiffusion) {
            pConcTmp[new_index]     = pConc[idx];
            pConcPrevTmp[new_index] = pConcPrev[idx];
            pGradConcTmp[new_index] = pGradConc[idx];
            pESFluxTmp[new_index]   = pESFlux[idx];
            pDTmp[new_index]        = pD[idx];
            if(( fabs(pArea[idx].x()) > 0.0 && fabs(pArea[idx].y()) > 0.0) ||
               ( fabs(pArea[idx].x()) > 0.0 && fabs(pArea[idx].z()) > 0.0) ||
               ( fabs(pArea[idx].y()) > 0.0 && fabs(pArea[idx].z()) > 0.0) ||
               ( fabs(pArea[idx][comp]) < 1.e-12)) {
                 pAreaTmp[new_index] = fourthOrEighth*pArea[idx];
            } else {
              if (i==0) {
                pAreaTmp[new_index] = pArea[idx];
              } else {
                if (  pxtmp[new_index].asVector().length2() >
                      pxtmp[last_index].asVector().length2()  ) {
                  pAreaTmp[last_index]    = 0.0;
                  pAreaTmp[new_index]     = pArea[idx];
                  pLoadCIDtmp[last_index] = IntVector(0.0, 0.0, 0.0);
                  pLoadCIDtmp[new_index]  = pLoadCID[idx];
                } else {
                  pAreaTmp[new_index]     = 0.0;
                  pLoadCIDtmp[new_index]  = IntVector(0.0, 0.0, 0.0);
                } // if pxtmp
              } // if i==0
            } // if pArea
          } // if diffusion
          ptemptmp[new_index]   = ptemp[idx];
          ptempgtmp[new_index]  = ptempgrad[idx];
          ptempPtmp[new_index]  = ptempP[idx];
          preftmp[new_index]    = 1;
          ploctmp[new_index]    = ploc[idx];
          pvgradtmp[new_index]  = pvelgrad[idx];
          NAPID_new[c_orig]++;
          last_index = new_index;
        }
        numRefPar++;
       }  // if particle flagged for refinement
      } // for particles
      } // if any particles flagged for refinement

      cm->splitCMSpecificParticleData(patch, dwi, fourOrEight, prefOld, pref,
                                      oldNumPar, numNewPartNeeded,
                                      old_dw, new_dw);

      if (flags->d_doScalarDiffusion) {
        ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        sdm->splitSDMSpecificParticleData(patch, dwi, fourOrEight, prefOld, pref,
                                          oldNumPar, numNewPartNeeded, old_dw, new_dw);
      }
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
      if (flags->d_doScalarDiffusion) {
        new_dw->put(pConcTmp,     lb->diffusion->pConcentration_preReloc, true);
        new_dw->put(pConcPrevTmp, lb->diffusion->pConcPrevious_preReloc,  true);
        new_dw->put(pGradConcTmp, lb->diffusion->pGradConcentration_preReloc,   true);
        new_dw->put(pESFluxTmp,   lb->diffusion->pExternalScalarFlux_preReloc,  true);
        new_dw->put(pAreaTmp,     lb->diffusion->pArea_preReloc,          true);
        new_dw->put(pDTmp,        lb->diffusion->pDiffusivity_preReloc,   true);
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
    printTask(patches,patch,cout_doing,"Doing MPM::computeParticleScaleFactor");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                       (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
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
    printTask(patches, patch,cout_doing,"Doing MPM::initialErrorEstimate");

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
                "Doing MPM::errorEstimate");

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
    printTask(patches, patch,cout_doing,"Doing MPM::refine");

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
        ParticleVariable<Matrix3> psize, pVelGrad, pcursize;
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
        new_dw->allocateAndPut(pcursize,       lb->pCurSizeLabel,       pset);

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
void SerialMPM::scheduleComputeNormals(SchedulerP   & sched,
                                       const PatchSet * patches,
                                       const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"MPM::scheduleComputeNormals");

  Task* t = scinew Task("MPM::computeNormals", this,
                        &SerialMPM::computeNormals);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  t->requires(Task::OldDW, lb->pXLabel,                  particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pMassLabel,               particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pVolumeLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->pCurSizeLabel,            particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pStressLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::AroundNodes, 1);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);

  t->computes(lb->gSurfNormLabel);
  t->computes(lb->gStressLabel);
  t->computes(lb->gNormTractionLabel);
  t->computes(lb->gPositionLabel);

  sched->addTask(t, patches, matls);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

//______________________________________________________________________
//
void SerialMPM::computeNormals(const ProcessorGroup *,
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
  std::vector<NCVariable<Vector> >       gsurfnorm(numMPMMatls);
  std::vector<NCVariable<double> >       gnormtraction(numMPMMatls);
  std::vector<NCVariable<Matrix3> >      gstress(numMPMMatls);

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask(patches, patch, cout_doing, "Doing MPM::computeNormals");

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

    // Find surface normal at each material based on a gradient of nodal mass
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                    (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[m],                lb->gMassLabel,   dwi,patch,gan,  1);

      new_dw->allocateAndPut(gsurfnorm[m],    lb->gSurfNormLabel,    dwi,patch);
      new_dw->allocateAndPut(gposition[m],    lb->gPositionLabel,    dwi,patch);
      new_dw->allocateAndPut(gstress[m],      lb->gStressLabel,      dwi,patch);
      new_dw->allocateAndPut(gnormtraction[m],lb->gNormTractionLabel,dwi,patch);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Point> px;
      constParticleVariable<double> pmass, pvolume;
      constParticleVariable<Matrix3> psize, pstress;
      constParticleVariable<Matrix3> deformationGradient;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(pmass,               lb->pMassLabel,               pset);
      old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
      new_dw->get(psize,               lb->pCurSizeLabel,            pset);
      old_dw->get(pstress,             lb->pStressLabel,             pset);

      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));
      gposition[m].initialize(Point(0.0,0.0,0.0));
      gnormtraction[m].initialize(0.0);
      gstress[m].initialize(Matrix3(0.0));

      if(flags->d_axisymmetric){
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          int NN = interpolator->findCellAndWeightsAndShapeDerivatives(
                                                   px[idx],ni,S,d_S,psize[idx]);
          double rho = pmass[idx]/pvolume[idx];
          for(int k = 0; k < NN; k++) {
            if (patch->containsNode(ni[k])){
              Vector G(d_S[k].x(),d_S[k].y(),0.0);
              gsurfnorm[m][ni[k]] += rho * G;
              gposition[m][ni[k]] += px[idx].asVector()*pmass[idx] * S[k];
              gstress[m][ni[k]]   += pstress[idx] * S[k];
            }
          }
        }
      } else {
        for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
          particleIndex idx = *it;

          int NN = interpolator->findCellAndWeightsAndShapeDerivatives(
                                                   px[idx],ni,S,d_S,psize[idx]);
          for(int k = 0; k < NN; k++) {
            if (patch->containsNode(ni[k])){
              Vector grad(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                          d_S[k].z()*oodx[2]);
              gsurfnorm[m][ni[k]] += pmass[idx] * grad;
              gposition[m][ni[k]] += px[idx].asVector()*pmass[idx] * S[k];
              gstress[m][ni[k]]   += pstress[idx] * S[k];
            }
          }
        }
      } // axisymmetric conditional
    }   // matl loop

#if 0
    // Make normal vectors colinear by setting all norms to be
    // in the opposite direction of the norm with the largest magnitude
    if(flags->d_computeColinearNormals){
      //cout << "Fix colinearNormals" << endl;
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
#endif

    // Make normal vectors colinear by taking an average with the
    // other materials at a node
    if(flags->d_computeColinearNormals){
     for(NodeIterator iter=patch->getExtraNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      vector<Vector> norm_temp(numMPMMatls);
      for(unsigned int m=0; m<numMPMMatls; m++){
       norm_temp[m]=Vector(0.,0.,0);
       if(gmass[m][c]>1.e-200){
        Vector mWON(0.,0.,0.);
        double mON=0.0;
        for(unsigned int n=0; n<numMPMMatls; n++){
          if(n!=m){
            mWON += gmass[n][c]*gsurfnorm[n][c];
            mON  += gmass[n][c];
          }
        }  // loop over other matls
        mWON/=(mON+1.e-100);
        norm_temp[m]=0.5*(gsurfnorm[m][c] - mWON);

       } // If node has mass
      }  // Outer loop over materials

      // Now put temporary norm into main array
      for(unsigned int m=0; m<numMPMMatls; m++){
        gsurfnorm[m][c] = norm_temp[m];
      }  // Outer loop over materials

     }   // Loop over nodes
    }    // if(flags..)

    // Make traditional norms unit length, compute gnormtraction
    for(unsigned int m=0;m<numMPMMatls;m++){
      MPMMaterial* mpm_matl =
                   (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
      }
    }

    delete interpolator;
  }    // patches
}

//
void SerialMPM::scheduleComputeLogisticRegression(SchedulerP   & sched,
                                                  const PatchSet * patches,
                                                  const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"MPM::scheduleComputeLogisticRegression");

  Task* t = scinew Task("MPM::computeLogisticRegression", this,
                        &SerialMPM::computeLogisticRegression);

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  t->requires(Task::OldDW, lb->pXLabel,                  particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->pCurSizeLabel,            particle_ghost_type, particle_ghost_layer);
  t->requires(Task::OldDW, lb->pVolumeLabel,             particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->pSurfLabel_preReloc,      particle_ghost_type, particle_ghost_layer);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,
           m_materialManager->getAllInOneMatls(),Task::OutOfDomain,Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,Ghost::None);

  t->computes(lb->gMatlProminenceLabel);
  t->computes(lb->gAlphaMaterialLabel);
  t->computes(lb->gNormAlphaToBetaLabel,z_matl);

  sched->addTask(t, patches, matls);

  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
}

//______________________________________________________________________
//
void SerialMPM::computeLogisticRegression(const ProcessorGroup *,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * ,
                                                DataWarehouse  * old_dw,
                                                DataWarehouse  * new_dw)
{

  // As of 5/22/19, this uses John Nairn's and Chad Hammerquist's
  // Logistic Regression method for finding normals used in contact.
  // These "NormAlphaToBeta" are then used to find each material's
  // greatest prominence at a node.  That is, the portion of a
  // particle that projects farthest along the direction of that normal.
  // One material at each multi-material node is identified as the
  // alpha material, this is the material with the most mass at the node.
  // All other materials are beta materials, and the NormAlphaToBeta
  // is perpendicular to the plane separating those materials
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
  std::vector<constNCVariable<double> >  gmass(numMPMMatls);
  std::vector<constParticleVariable<Point> > px(numMPMMatls);
  std::vector<constParticleVariable<double> > pvolume(numMPMMatls);

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing MPM::computeLogisticRegression");

    Vector dx = patch->dCell();

    constNCVariable<double>    NC_CCweight;
    old_dw->get(NC_CCweight,   lb->NC_CCweightLabel,  0, patch, gnone, 0);

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    // Declare and allocate storage for use in the Logistic Regression
    NCVariable<int> alphaMaterial;
    NCVariable<int> NumMatlsOnNode;
    NCVariable<int> NumParticlesOnNode;
    NCVariable<Vector> normAlphaToBeta;
    std::vector<NCVariable<Int130> > ParticleList(numMPMMatls);
    std::vector<bool> IsRigidMaterial(numMPMMatls);
    std::vector<bool> PossibleAlpha(numMPMMatls);

    new_dw->allocateAndPut(alphaMaterial,  lb->gAlphaMaterialLabel,   0, patch);
    new_dw->allocateAndPut(normAlphaToBeta,lb->gNormAlphaToBetaLabel, 0, patch);
    new_dw->allocateTemporary(NumMatlsOnNode,     patch);
    new_dw->allocateTemporary(NumParticlesOnNode, patch);
    alphaMaterial.initialize(-99);
    NumMatlsOnNode.initialize(0);
    NumParticlesOnNode.initialize(0);
    normAlphaToBeta.initialize(Vector(-99.-99.-99.));

    // Get out the mass first, need access to the mass of all materials
    // at each node.
    // Also, at each multi-material node, we need the position of the
    // particles around that node.  Rather than store the positions, for now,
    // store a list of particle indices for each material at each node and
    // use those to point into the particle set to get the particle positions
    constNCVariable<double>  gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
         m_materialManager->getAllInOneMatls()->get(0), patch, gnone, 0);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                    (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[m],                lb->gMassLabel,   dwi,patch,gnone,0);
      new_dw->allocateTemporary(ParticleList[m], patch);
      IsRigidMaterial[m] = mpm_matl->getIsRigid();
      PossibleAlpha[m] = mpm_matl->getPossibleAlphaMaterial();
    }

    // Here, find out two things:
    // 1.  How many materials have mass on a node
    // 2.  Which material has the most mass on a node.  That is the alpha matl.
    for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      double maxMass=-9.e99;
      for(unsigned int m = 0; m < numMPMMatls; m++){
        if(!PossibleAlpha[m]) continue;
        if(gmass[m][c] > 1.e-8*gmassglobal[c] && gmass[m][c] > 1.e-16){
          NumMatlsOnNode[c]++;
          if(gmass[m][c]>maxMass){
            // This is the alpha material, all other matls are beta
            alphaMaterial[c]=m;
            maxMass=gmass[m][c];
          }
        }
      } // Loop over materials

      double maxRigidMass=1.e-90;
      for(unsigned int m = 0; m < numMPMMatls; m++){
        if(IsRigidMaterial[m] && PossibleAlpha[m]){
          if(gmass[m][c] > maxRigidMass){
            maxRigidMass=gmass[m][c];
            alphaMaterial[c]=m;
          }
        }
      } // Loop over materials

      if(NumMatlsOnNode[c]<2){
        alphaMaterial[c]=-99;
      }
    }   // Node Iterator

    // In this section of code, we find the particles that are in the
    // vicinity of a multi-material node and put their indices in a list
    // so we can retrieve their positions later.

    // I hope to improve on this ParticleList later, but for now,
    // the last element in the array holds the number of entries in the
    // array.  I don't yet know how to allocate an STL container on the nodes.
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                    (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);
      constParticleVariable<Matrix3> cursize;

      old_dw->get(px[m],                lb->pXLabel,               pset);
      new_dw->get(cursize,              lb->pCurSizeLabel,         pset);
      old_dw->get(pvolume[m],           lb->pVolumeLabel,          pset);

      // Initialize the ParticleList
      for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        for(unsigned int p = 0; p < 400; p++){
          ParticleList[m][c][p]=0;
        }
      }

      // Loop over particles and find which multi-mat nodes they contribute to
      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;

        int NN = interpolator->findCellAndWeights(px[m][idx],ni,S,cursize[idx]);

        set<IntVector> nodeList;

        for(int k = 0; k < NN; k++) {
          if (patch->containsNode(ni[k]) &&
              NumMatlsOnNode[ni[k]]>1    &&
              S[k]>1.e-8){
            nodeList.insert(ni[k]);
          } // conditional
        }   // loop over nodes returned by interpolator
        for (set<IntVector>::iterator it1 = nodeList.begin();
                                      it1!= nodeList.end();  it1++){
          if(ParticleList[m][*it1][399] < 399){
            ParticleList[m][*it1][ParticleList[m][*it1][399]]=idx;
            ParticleList[m][*it1][399]++;
            NumParticlesOnNode[*it1]++;
          }
        }
        nodeList.clear();
      }    // Loop over Particles
    }  // Loop over materials

    // Do an additional check to make sure that nodes identified as
    // multi-material have multiple materials with particles contributing
    // If not, set alphaMaterial back to -99
    for(NodeIterator iter =patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      int numMatlsWParticles=0;
      for(unsigned int m = 0; m < numMPMMatls; m++){
        if(ParticleList[m][c][399] > 0){
          numMatlsWParticles++;
        }
      }  // Loop over materials
      if(numMatlsWParticles<2){
        alphaMaterial[c]=-99;
      }
    }  // Loop over nodes

    // This is the Logistic Regression code that finds the normal to the
    // plane that separates two materials.  This is as directly as possible
    // from Nairn & Hammerquist, 2019.
    double lam = 1.e-7*dx.x()*dx.x();
    double lambda[4]={lam,lam,lam,0};
    double wp = 1.0;
    double RHS[4];
    for(NodeIterator iter =patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      // Only work on multi-material nodes
      if(alphaMaterial[c]>=0){
       bool converged = false;
       int num_iters=0;
       double tol = 1.e-7;
       double phi[4]={0.,0.,0.,0.};
       Vector nhat_k(phi[0],phi[1],phi[2]);
       Vector nhat_backup(0.);
       double error_min=1.0;
       while(!converged){
        num_iters++;
        // Initialize the coefficient matrix
        FastMatrix FMJtWJ(4,4);
        FMJtWJ.zero();
        for(int i = 0; i<4; i++){
          FMJtWJ(i,i) = lambda[i];
        }
        for(int i = 0; i<4; i++){
          RHS[i] = -1.0*lambda[i]*phi[i];
        }
        for(unsigned int m = 0; m < numMPMMatls; m++){
          double cp=0.;
          if(alphaMaterial[c]==(int) m){
            cp=-1.;
          } else {
            cp=1.;
          }
          for(int p=0;p<ParticleList[m][c][399];p++){
            Point xp = px[m][ParticleList[m][c][p]];
            double xp4[4]={xp.x(),xp.y(),xp.z(),1.0};
            double XpDotPhi = xp4[0]*phi[0]
                            + xp4[1]*phi[1]
                            + xp4[2]*phi[2]
                            + xp4[3]*phi[3];
            double expterm = exp(-XpDotPhi);
            double num     = 2.0*expterm;
            double denom   = (1.0 + expterm)*(1.0 + expterm);
            double fEq20   = 2./(1+expterm) - 1.0;
            double psi = num/denom;
            double psi2wp = psi*psi*wp;
            // Construct coefficient matrix, Eqs. 54 and 55
            // Inner terms
            for(int i = 0; i<3; i++){
              for(int j = 0; j<3; j++){
                FMJtWJ(i,j)+=psi2wp*xp(i)*xp(j);
              }
            }
            // Other terms
            FMJtWJ(0,3)+=psi2wp*xp(0);
            FMJtWJ(1,3)+=psi2wp*xp(1);
            FMJtWJ(2,3)+=psi2wp*xp(2);
            FMJtWJ(3,0)=FMJtWJ(0,3);
            FMJtWJ(3,1)=FMJtWJ(1,3);
            FMJtWJ(3,2)=FMJtWJ(2,3);
            FMJtWJ(3,3)+=psi2wp;
            // Construct RHS
            for(int i = 0; i<4; i++){
              RHS[i]+=psi*wp*(cp - fEq20)*xp4[i];
            }
          } // Loop over each material's particle list
        }     // Loop over materials

        // Solve (FMJtWJ)^(-1)*RHS.  The solution comes back in the RHS array
        FMJtWJ.destructiveSolve(RHS);

        for(int i = 0; i<4; i++){
          phi[i]+=RHS[i];
        }
        Vector nhat_kp1(phi[0],phi[1],phi[2]);
        nhat_kp1/=(nhat_kp1.length()+1.e-100);
        double error = 1.0 - Dot(nhat_kp1,nhat_k);
        if(error < error_min){
          error_min = error;
          nhat_backup = nhat_kp1;
        }
        if(error < tol || num_iters > 50){
          converged=true;
          if(num_iters > 50){
           normAlphaToBeta[c] = nhat_backup;
          } else {
           normAlphaToBeta[c] = nhat_kp1;
          }
        } else{
          nhat_k=nhat_kp1;
        }
       } // while(!converged) loop
      }  // If this node has more than one particle on it
    }    // Loop over nodes

    MPMBoundCond bc;
    bc.setBoundaryCondition(patch,0,"Symmetric", normAlphaToBeta, interp_type);

    // Renormalize normal vectors after setting BCs
    for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      normAlphaToBeta[c]/=(normAlphaToBeta[c].length()+1.e-100);
      if(alphaMaterial[c]==-99){
        normAlphaToBeta[c]=Vector(0.);
      }
      if(!(normAlphaToBeta[c].length() >= 0.0)){
        cout << "Node  = " << c << endl;
        cout << "normAlphaToBeta[c] = " << normAlphaToBeta[c] << endl;
        cout << "alphaMaterial[c] = " << alphaMaterial[c] << endl;
        cout << "NumMatlsOnNode[c] = " << NumMatlsOnNode[c] << endl;
        cout << "NumParticlesOnNode[c] = " << NumParticlesOnNode[c] << endl;
      }
    }    // Loop over nodes

    // Loop over all the particles, find the nodes they interact with
    // For the alpha material (alphaMaterial) find g.position as the
    // maximum of the dot product between each particle corner and the
    // gNormalAlphaToBeta vector.
    // For the beta materials (every other material) find g.position as the
    // minimum of the dot product between each particle corner and the
    // gNormalAlphaToBeta vector.
    // Compute "MatlProminence" as the min/max of the dot product between
    // the normal and the particle corners

    std::vector<NCVariable<double> >      d_x_p_dot_n(numMPMMatls);

    for(unsigned int m=0;m<numMPMMatls;m++){
      MPMMaterial* mpm_matl =
                   (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      constParticleVariable<Matrix3> pcursize;
      constParticleVariable<double> psurf;

      new_dw->get(pcursize,                 lb->pCurSizeLabel,         pset);
      new_dw->get(psurf,                    lb->pSurfLabel_preReloc,   pset);
      new_dw->allocateAndPut(d_x_p_dot_n[m],lb->gMatlProminenceLabel,dwi,patch);

      d_x_p_dot_n[m].initialize(-99.);

      NCVariable<double> projMax, projMin;
      new_dw->allocateTemporary(projMax,                  patch,    gnone);
      new_dw->allocateTemporary(projMin,                  patch,    gnone);
      projMax.initialize(-9.e99);
      projMin.initialize( 9.e99);

      for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
        particleIndex idx = *it;

      if(psurf[idx]>0.9){
       int NN = interpolator->findCellAndWeights(px[m][idx],ni,S,pcursize[idx]);

        Matrix3 dsize = pcursize[idx]*Matrix3(dx[0],0,0,
                                              0,dx[1],0,
                                              0,0,dx[2]);

#if 0
        // This version uses particle corners to compute prominence
        // Compute vectors from particle center to the corners
        Vector RNL[8];
        RNL[0] = Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                        -dsize(1,0)-dsize(1,1)+dsize(1,2),
                        -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
        RNL[1] = Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                         dsize(1,0)-dsize(1,1)+dsize(1,2),
                         dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
        RNL[2] = Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                         dsize(1,0)+dsize(1,1)+dsize(1,2),
                         dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
        RNL[3] = Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                        -dsize(1,0)+dsize(1,1)+dsize(1,2),
                        -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
        RNL[4] = Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                        -dsize(1,0)-dsize(1,1)-dsize(1,2),
                        -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
        RNL[5] = Vector( dsize(0,0)-dsize(0,1)-dsize(0,2),
                         dsize(1,0)-dsize(1,1)-dsize(1,2),
                         dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
        RNL[6] = Vector( dsize(0,0)+dsize(0,1)-dsize(0,2),
                         dsize(1,0)+dsize(1,1)-dsize(1,2),
                         dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
        RNL[7] = Vector(-dsize(0,0)+dsize(0,1)-dsize(0,2),
                        -dsize(1,0)+dsize(1,1)-dsize(1,2),
                        -dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;

        for(int k = 0; k < NN; k++) {
          if (patch->containsNode(ni[k])){
            if(S[k] > 0. && NumParticlesOnNode[ni[k]] > 1){
              for(int ic=0;ic<8;ic++){
                Vector xp_xi = (px[m][idx].asVector()+RNL[ic]);
                double proj = Dot(xp_xi, normAlphaToBeta[ni[k]]);
                if((int) m==alphaMaterial[ni[k]]){
                  if(proj>projMax[ni[k]]){
                     projMax[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                } else {
                  if(proj<projMin[ni[k]]){
                     projMin[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                }
              } // Loop over all 8 particle corners
            }  // Only deal with nodes that this particle affects
          }  // If node is on the patch
        } // Loop over nodes near this particle
#endif
#if 0
        // This version uses constant particle radius, here assuming 2 PPC
        // in each direction
        double Rp = cbrt((.75/M_PI)*pvolume[m][idx]);
        for(int k = 0; k < NN; k++) {
          if (patch->containsNode(ni[k])){
            if(S[k] > 0. && NumParticlesOnNode[ni[k]] > 1){
//              Point xi = patch->getNodePosition(ni[k]);
              for(int ic=0;ic<8;ic++){
                Vector xp_xi = (px[m][idx].asVector());// - xi;
                double proj = Dot(xp_xi, normAlphaToBeta[ni[k]]);
                if((int) m==alphaMaterial[ni[k]]){
                  proj+=Rp;
                  if(proj>projMax[ni[k]]){
                     projMax[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                } else {
                  proj-=Rp;
                  if(proj<projMin[ni[k]]){
                     projMin[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                }
              } // Loop over all 8 particle corners
            }  // Only deal with nodes that this particle affects
          }  // If node is on the patch
        } // Loop over nodes near this particle
#endif

#if 1
        // This version uses particle faces to compute prominence.
        // Compute vectors from particle center to the faces
        Vector RFL[6];
        RFL[0] = Vector(-dsize(0,0),-dsize(1,0),-dsize(2,0))*0.5;
        RFL[1] = Vector( dsize(0,0), dsize(1,0), dsize(2,0))*0.5;
        RFL[2] = Vector(-dsize(0,1),-dsize(1,1),-dsize(2,1))*0.5;
        RFL[3] = Vector( dsize(0,1), dsize(1,1), dsize(2,1))*0.5;
        RFL[4] = Vector(-dsize(0,2),-dsize(1,2),-dsize(2,2))*0.5;
        RFL[5] = Vector( dsize(0,2), dsize(1,2), dsize(2,2))*0.5;

        for(int k = 0; k < NN; k++) {
          if (patch->containsNode(ni[k])){
            if(S[k] > 0. && NumParticlesOnNode[ni[k]] > 1){
              for(int ic=0;ic<6;ic++){
                Vector xp_xi = (px[m][idx].asVector()+RFL[ic]);
                double proj = Dot(xp_xi, normAlphaToBeta[ni[k]]);
                if((int) m==alphaMaterial[ni[k]]){
                  if(proj>projMax[ni[k]]){
                     projMax[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                } else {
                  if(proj<projMin[ni[k]]){
                     projMin[ni[k]]=proj;
                     d_x_p_dot_n[m][ni[k]] = proj;
                  }
                }
              } // Loop over all 8 particle corners
            }  // Only deal with nodes that this particle affects
          }  // If node is on the patch
        } // Loop over nodes near this particle
#endif
       } // Is a surface particle
      } // end Particle loop
    }  // loop over matls

    delete interpolator;
  }    // patches
}

//
void SerialMPM::scheduleFindSurfaceParticles(SchedulerP   & sched,
                                             const PatchSet * patches,
                                             const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"SerialMPM::scheduleFindSurfaceParticles");

  Task* t = scinew Task("MPM::findSurfaceParticles", this,
                        &SerialMPM::findSurfaceParticles);

  Ghost::GhostType  gp;
  int ngc_p;
  getParticleGhostLayer(gp, ngc_p);

  t->requires(Task::OldDW, lb->pSurfLabel,               gp, ngc_p);
  t->computes(lb->pSurfLabel_preReloc);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void SerialMPM::findSurfaceParticles(const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * ,
                                           DataWarehouse  * old_dw,
                                           DataWarehouse  * new_dw)
{
  unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask(patches, patch, cout_doing, "Doing findSurfaceParticles");

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial*  mpm_matl  =
                       (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<double> pSurfOld;
      ParticleVariable<double> pSurf;

      old_dw->get(pSurfOld,            lb->pSurfLabel,               pset);
      new_dw->allocateAndPut(pSurf,    lb->pSurfLabel_preReloc,      pset);

      // For now carry forward the particle surface data
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end();
           iter++){
         particleIndex idx = *iter;
         pSurf[idx]=pSurfOld[idx];
      }
    }   // matl loop
  }    // patches
}


//
void SerialMPM::scheduleConcInterpolated(       SchedulerP  & sched
                                        , const PatchSet    * patches
                                        , const MaterialSet * matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) return;

  printSchedule(patches,cout_doing,"MPM::scheduleConcInterpolated");

  d_sdInterfaceModel->addComputesAndRequiresInterpolated(sched, patches, matls);

}

void SerialMPM::scheduleComputeFlux(       SchedulerP  & sched
                                   , const PatchSet    * patches
                                   , const MaterialSet * matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeFlux");

  Task* t = scinew Task("SerialMPM::computeFlux",
                        this, &SerialMPM::computeFlux);

  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  for (unsigned int m = 0; m < numMPM; ++m) {
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->scheduleComputeFlux(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);

}

void SerialMPM::computeFlux(  const ProcessorGroup  *
                           ,  const PatchSubset     * patches
                           ,  const MaterialSubset  * matls
                           ,        DataWarehouse   * old_dw
                           ,        DataWarehouse   * new_dw  )
{
  for (int p = 0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch, cout_doing, "Doing MPM::computeFlux");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMatls; ++m) {
      MPMMaterial*          mpm_matl  = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm       = mpm_matl->getScalarDiffusionModel();
      sdm->computeFlux(patch, mpm_matl, old_dw, new_dw);
    }
  }
}

void SerialMPM::scheduleComputeDivergence(       SchedulerP  & sched
                                         , const PatchSet    * patches
                                         , const MaterialSet * matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeDivergence");

  Task* t = scinew Task("SerialMPM::computeDivergence",
                        this, &SerialMPM::computeDivergence);

  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  for (unsigned int m = 0; m < numMPM; ++m) {
    MPMMaterial*          mpm_matl  = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ScalarDiffusionModel* sdm       = mpm_matl->getScalarDiffusionModel();
    sdm->scheduleComputeDivergence(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

// JBH -- Check these simple compute functions; they seem to ignore the input
//        material list and always run on the global list. FIXME TODO
void SerialMPM::computeDivergence(  const ProcessorGroup  *
                                 ,  const PatchSubset     * patches
                                 ,  const MaterialSubset  * matls
                                 ,        DataWarehouse   * old_dw
                                 ,        DataWarehouse   * new_dw  )
{
  for (int p = 0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch, cout_doing, "Doing MPM::computeDivergence");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for (unsigned int m = 0; m < numMatls; m++) {
      MPMMaterial*  mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->computeDivergence(patch, mpm_matl, old_dw, new_dw);
    }
  }
}

void SerialMPM::scheduleDiffusionInterfaceDiv(       SchedulerP  & sched
                                             , const PatchSet    * patches
                                             , const MaterialSet * matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) return;

  printSchedule(patches,cout_doing,"MPM::scheduleDiffusionInterfaceDiv");

  d_sdInterfaceModel->addComputesAndRequiresDivergence(sched, patches, matls);
}

//______________________________________________________________________
//
double SerialMPM::recomputeDelT( const double delT )
{
  return delT * 0.1;
}

//MJ
//---------------------Granular MPM--------------------------------------------------------------------

// This function will check the material points position and if necessary
// move them to other velocity fields. This function should take place
// before FinalParticleUpdate.
void SerialMPM::GranularMPM(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
    //---------------------Preparing data for task 1--------------------------//

  //1- Finding the number of material points in all materials
  //Counter for finding number of particles in a particles set
  int PsetNumbParticles = 0;
  //Counter for finding number of particles in all materials
  int NumbParticles = 0;

  // For now, different code for 2D and 3D
  int dimensionality=flags->d_ndim;

  // The whole Granular MPM is written to repeat itself for each patch once. 
  for (int p = 0; p < patches->size(); p++) {   
    const Patch* patch = patches->get(p);
    printTask(patches, patch, cout_doing, "Doing MPM::GranularMPM");
    unsigned int numMatls = m_materialManager->getNumMatls("MPM");
    Vector dx = patch->dCell();

    //  cout << "Patch size = " << patches->size() << endl;
    for (unsigned int m = 0; m < numMatls; m++) {   
        MPMMaterial* mpm_matl = 
                         (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
        int dwi = mpm_matl->getDWIndex();
        ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
        PsetNumbParticles = pset->addParticles(0);//Find the number of particles
       // cerr << " PsetNumbParticles=  "  << PsetNumbParticles << endl;
        NumbParticles = NumbParticles + PsetNumbParticles;           
    }       

    //2- creating vector to carry particle data from all materials
    //   and filling them.
    vector<int> MaterialIndex(NumbParticles);
     // a vector for recording original material. The number kept in this vector
     // show the index of original materil. This vector helps in finding 
     // the original data of particle.
    vector<int> ParticleIndex(NumbParticles);
    // a vector for recording original particle index. The number kept in this 
    // vector show the index of materal point in its original material. 
    // This vector helps in finding the original data of particle.

    //a vector for the material point cordinate in x-direction
    vector<double> px1(NumbParticles);
    //a vector for the material point cordinate in y-direction
    vector<double> px2(NumbParticles);
    //a vector for the material point cordinate in z-direction
    vector<double> px3(NumbParticles);

    vector<double> DI1(NumbParticles); // vector for x of domain of interaction
    vector<double> DI2(NumbParticles); // vector for y of domain of interaction 
    vector<double> DI3(NumbParticles); // vector for z of domain of interaction 
    int pcounter = 0;

    //moving on the materials in each pataches
    for (unsigned int m = 0; m < numMatls; m++) {
      MPMMaterial* mpm_matl = 
                         (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      //creating particle arrays
      ParticleVariable<Point> px;
      ParticleVariable<Matrix3>  pSize;
      ParticleVariable<double> pvolume;
      //putting the data in the arrays
      new_dw->getModifiable(px,      lb->pXLabel_preReloc, pset);
      new_dw->getModifiable(pSize,   lb->pSizeLabel_preReloc, pset);
      new_dw->getModifiable(pvolume, lb->pVolumeLabel_preReloc, pset);

      double Vcrix = flags->d_GranularMPM_Vcrix;
      double Vcriy = flags->d_GranularMPM_Vcriy;
      double Vcriz = flags->d_GranularMPM_Vcriz;

      //moving on the particles in each materaial and each pataches
      for (ParticleSubset::iterator iter1 = pset->begin();     
         iter1 != pset->end(); iter1++) {
         particleIndex idx0 = *iter1;

         MaterialIndex[pcounter] = dwi;                  
         ParticleIndex[pcounter] = idx0;
         px1[pcounter] = px[idx0](0);                      
         px2[pcounter] = px[idx0](1);                    
         px3[pcounter] = px[idx0](2);
         double r1=dx.x()*pSize[idx0](0,0);
         double r2=dx.y()*pSize[idx0](1,1);
         double r3=dx.z()*pSize[idx0](2,2);
         if(dimensionality==3){
           //Critical volume of material point.
           const  double Vcri = (Vcrix*r1 * Vcriy*r2 * Vcriz*r3);

           //domain of interaction in x, y and z directions, respectively.
           DI1[pcounter] = cbrt(( 0.125*(r1*r1/r2/r3) * Vcri ));
           DI2[pcounter] = cbrt(( 0.125*(r2*r2/r1/r3) * Vcri ));
           DI3[pcounter] = cbrt(( 0.125*(r3*r3/r1/r2) * Vcri ));
           pcounter = pcounter + 1;
         } else {
           //Critical volume of material point.
           const  double Vcri = (Vcrix*r1 * Vcriy*r2);  // JG - Sort this out

           //domain of interaction in x and y directions.
           DI1[pcounter] = sqrt(( 0.25*(r1/r2) * Vcri ));
           DI2[pcounter] = sqrt(( 0.25*(r2/r1) * Vcri ));
           pcounter = pcounter + 1;
        }  // dimensionality
      } // loop over particles
    }   // loop over materials

    // Creating matrixes for Granular MPM. 
    // We need 3 matrixes in the whole Granular MPM. As the matrix creation 
    // requires some steps, I am making all here so prevent repeating. 
    // I write detailed comment here for each matrix and the task the 
    // matrix will be used in. 

    // First matrix:  Interaction     ---> This matrix will be used in Task 1.
    // It is a matrix for recording indexes of interacting material point. 
    // each row of Interaction has indexes of material points connected 
    // with the row number material point through grid.
    vector<vector<int> > Interaction(NumbParticles);  

    // Second matrix: Grids    ----> This matrix will be use in Task 2. 
    // It is a matrix for recording indexes of interacting material points 
    // and shows the parallel grid in the next step.
    // each row of Grids has indexes of material points belonging to one 
    // grid of the next step.
    vector<vector<int> > Grids(NumbParticles);  
    
    // Third matrix: CurrentGrids    ------> This matrix will be use in Task 3. 
    // It is a matrix for indicating that a material point should be 
    // removed from its current material.
    // each row of Grids has -1 and 1 only. -1 signals that a material point
    // exist here and should be removed. 1 signals that the material point 
    // is Ok to stay or that there is no material point in this position.
    vector<vector<int> > CurrentGrids(NumbParticles);

    for (int i = 0; i < NumbParticles; i++) {
      Interaction[i].resize(NumbParticles);
      Grids[i].resize(NumbParticles);
      CurrentGrids[i].resize(NumbParticles);
     
      for (int j = 0; j < NumbParticles; j++) {
        // -1 is a mere place holder for Interaction matrix. 
        // This allows controling the results
        Interaction[i][j] = {-1};
        //-1 is a mere place holder for Grids matrix.
        Grids[i][j] = {-1};
        // Initially, we assume that all the material points are Ok to stay
        // or that there is no material point in this position. The material
        // that should be moved will be marked with -1 in task 3
        CurrentGrids[i][j] = {1};
      }
    }

    //check for creating vecros with the material points data & cordinates
    /*
        for (int i = 0; i < NumbParticles; i++) {
           cerr << " MaterialIndex[" << i << "]= " << MaterialIndex[i] << endl;
           cerr << " ParticleIndex[" << i << "]= " << ParticleIndex[i] << endl;
           cerr << " px1[" << i << "]= " << px1[i] << endl;
           cerr << " px2[" << i << "]= " << px2[i] << endl;
           cerr << " px3[" << i << "]= " << px3[i] << endl;
           cerr << " DI1[" << i << "]= " << DI1[i] << endl;
           cerr << " DI2[" << i << "]= " << DI2[i] << endl;
           cerr << " DI3[" << i << "]= " << DI3[i] << endl;
        } 
    */
               
    //----------------------------Task 1----------------------------//
    // Creating a vector for recording interaction size for each particle. 
    int InteractionSize[NumbParticles];
    for (int i = 0; i < NumbParticles; i++) { 
      InteractionSize[i] = { 0 };
    }

    // JG - I created a 3D and 2D version for efficiency  These could be made
    // more efficient by restructuring the conditionals.
    if(dimensionality==3){
      // loop over all particles in the patch to find the Interaction 
      // for each material point:           
      for (int idx1 = 0; idx1 < NumbParticles; idx1++) {
        //finding min and max of DI for material point one.
        double ximax = px1[idx1] + DI1[idx1];
        double ximin = px1[idx1] - DI1[idx1];
        double yimax = px2[idx1] + DI2[idx1];
        double yimin = px2[idx1] - DI2[idx1];
        double zimax = px3[idx1] + DI3[idx1];
        double zimin = px3[idx1] - DI3[idx1];
  
        for (int idx2 = 0; idx2 < NumbParticles; idx2++) {
          //finding min and max of DI for material point two.
          double xjmax = px1[idx2] + DI1[idx2];
          double xjmin = px1[idx2] - DI1[idx2];
          double yjmax = px2[idx2] + DI2[idx2];
          double yjmin = px2[idx2] - DI2[idx2];
          double zjmax = px3[idx2] + DI3[idx2];
          double zjmin = px3[idx2] - DI3[idx2];
  
          // if domains of interaction intersect, then the material 
          //point index should be added to interaction.
          if ((ximax > xjmin) && (ximin < xjmax) && 
              (yimax > yjmin) && (yimin < yjmax) && 
              (zimax > zjmin) && (zimin < zjmax)){
              Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
              InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((ximax > xjmin) && (ximin < xjmax) && 
                   (yimax > yjmin) && (yimin < yjmax) && 
                   (zjmax > zimin) && (zjmin < zimax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((ximax > xjmin) && (ximin < xjmax) && 
                   (yjmax > yimin) && (yjmin < yimax) && 
                   (zimax > zjmin) && (zimin < zjmax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((ximax > xjmin) && (ximin < xjmax) && 
                   (yjmax > yimin) && (yjmin < yimax) && 
                   (zjmax > zimin) && (zjmin < zimax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((xjmax > ximin) && (xjmin < ximax) && 
                   (yimax > yjmin) && (yimin < yjmax) && 
                   (zimax > zjmin) && (zimin < zjmax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((xjmax > ximin) && (xjmin < ximax) && 
                   (yimax > yjmin) && (yimin < yjmax) && 
                   (zjmax > zimin) && (zjmin < zimax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((xjmax > ximin) && (xjmin < ximax) && 
                   (yjmax > yimin) && (yjmin < yimax) && 
                   (zimax > zjmin) && (zimin < zjmax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
          else if ((xjmax > ximin) && (xjmin < ximax) && 
                   (yjmax > yimin) && (yjmin < yimax) && 
                   (zjmax > zimin) && (zjmin < zimax)) {
            Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
            InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
        } // end for idx2
      } // end for idx1
    } else {  // 2D (or 1D, for now) version
      for (int idx1 = 0; idx1 < NumbParticles; idx1++) {
        //finding min and max of DI for material point one.
        double ximax = px1[idx1] + DI1[idx1];
        double ximin = px1[idx1] - DI1[idx1];
        double yimax = px2[idx1] + DI2[idx1];
        double yimin = px2[idx1] - DI2[idx1];
//      cout << "ximax = " << ximax << endl;

        for (int idx2 = 0; idx2 < NumbParticles; idx2++) {      
           //finding min and max of DI for material point two.
           double xjmax = px1[idx2] + DI1[idx2];
           double xjmin = px1[idx2] - DI1[idx2];
           double yjmax = px2[idx2] + DI2[idx2];
           double yjmin = px2[idx2] - DI2[idx2];
//         cout << "xjmin = " << xjmin << endl;

           // if domains of interaction intersect, then the 
           // material point index should be added to interaction.
           if ((ximax > xjmin) && (ximin < xjmax) && 
               (yimax > yjmin) && (yimin < yjmax)) {
             Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
             InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
             if(MaterialIndex[idx1]!=MaterialIndex[idx2]){
//                cout << "Here1" << endl;
//                cout << "px1 = " << px1[idx1] << endl;
//                cout << "px2 = " << px1[idx2] << endl;
//                cout << "idx1 = " << idx1 << endl;
//                cout << "idx2 = " << idx2 << endl;
//                cout << "ximax = " << ximax << endl;
//                cout << "xjmin = " << xjmin << endl;
//                cout << "ximin = " << ximin << endl;
//                cout << "xjmax = " << xjmax << endl;
             }
           } else if ((ximax > xjmin) && (ximin < xjmax) && 
                      (yjmax > yimin) && (yjmin < yimax)) {
                        Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
                        InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          } else if ((xjmax > ximin) && (xjmin < ximax) && 
                     (yimax > yjmin) && (yimin < yjmax)) {
                        Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
                        InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          } else if ((xjmax > ximin) && (xjmin < ximax) && 
                     (yjmax > yimin) && (yjmin < yimax)) {
                        Interaction[idx1][(InteractionSize[idx1])] = { idx2 };
                        InteractionSize[idx1] = { InteractionSize[idx1] + 1 };
          }
        } // loop over idx2
      } // loop over idx1
    }

    //Task 1 check 
    /*
    cout << "NumbParticles = " << NumbParticles << endl;
    for (int i = 0; i < NumbParticles; i++)            {
      cerr << " InteractionSize[" << i << "]= " << InteractionSize[i]<< endl;
        for (int j = 0; j < NumbParticles; j++) {                            
        cerr << " Interaction[" << i << "]" << "[" << j << "]= " << Interaction[i][j] << endl;
        } 
    }
    */
    //---------------------------End of Task 1-------------------//

    //------------------------------Task 2----------------------//
    // Initiating the necessary vectors and counters            
    int GridsSize[NumbParticles];  //vector for saving size of each Grid
    int Assigned[NumbParticles];   //vector for marking Assigned material points
    //initiating a vector for keeping index of UnAssigned material points. 
    int UnAssigned[NumbParticles]; 
    // Counter for the number of Grids. 
    // Initially none of material points are Assigned.
    int GridCount = 0;

    // Counter for the number of UnAssigned material ponts. 
    // Initially all the material points are UnAssigned.
    int UnAssignCount = NumbParticles; 
    // Counter for repeated material points in a grid.
    // This counter helps to prevent repetation in a grid.
    int RepeatCounter = 0;   

    // Putting initial value in the matrix and vectors
    for (int i = 0; i < NumbParticles; i++) {
        // Initially there is no grid and thus the size of all grid is 0
        GridsSize[i] = { 0 };
       // -1 signals that it is not.     1 signals that it is.
       //  Initially none of material points are Assigned.
        Assigned[i] = { -1 };
        // Initially all material points are UnAssigned. Therefore, 
        // UnAssigned carries all material points indexs just here 
        // before the assiging process begins. Later, it won't carry 
        // all the material points indexes as they get assigned to a grid.
        UnAssigned[i] = { i };     
        // There is a differece between what Assigned and UnAssigned 
        // vectors carry. The Assigned carries a signal while UnAssigned 
        // carries index of UnAssigned material points.               
    }

    // This loop ensures that every material point is assigned to a grid
    while (UnAssignCount > 0) {
      GridCount = GridCount + 1;
      // Assigning the first unassigned material point to a new grid.
      for (int i = 0; i < InteractionSize[UnAssigned[0]]; i++) {
          Grids[GridCount - 1][i] = Interaction[UnAssigned[0]][i];
          GridsSize[GridCount - 1] = GridsSize[GridCount - 1] + 1;
      }
      // Marking this material point as Assigned now
      Assigned[UnAssigned[0]] = 1;

      // Variable for stopping the second while loop
      int stop = 0;
      // Counter for the second while loop
      int stopCounter = 0;

      // Second while loop that assigns material points in the 
      // current grid if not already assigned                
      while (stop < 1) {
        //moving on the existing material points in the grid
        for (int j = 0; j < GridsSize[GridCount - 1]; j++) {      
          // checking if the material point is assigned or not. 
          // If it is not Assigned, the next lines will assign it. 
          if (Assigned[Grids[GridCount - 1][j]] == -1) {
            // moving on the Interaction line of the unassigned material 
            // point. Every material points in this line should be added 
            // to the current grid unless it is already there. 
            for (int k = 0; k < InteractionSize[Grids[GridCount-1][j]];k++){
              // making sure that the material point on interaction is 
              // not already there and then adding it to the current grid
              //reseting the counter which prevent repetation in a grid.
              RepeatCounter = 0;
              //moving on the current material points of grid.
              for (int m = 0; m < GridsSize[GridCount - 1]; m++) {
                // If the material points on Interaction is 
                // already on the current grid
                if (Interaction[Grids[GridCount - 1][j]][k] == 
                                               Grids[GridCount - 1][m]){
                   //then increase the RepeatCounter
                    RepeatCounter = RepeatCounter + 1;
                }
              }
              // If the RepeatCounter is still 0, the material point on 
              // interaction is not on the current grid and should be added 
              if (RepeatCounter < 1) {
                //Increasing the size of grid
                GridsSize[GridCount - 1] = GridsSize[GridCount - 1] + 1;
                //Adding the material point to the current grid
                Grids[GridCount - 1][GridsSize[GridCount - 1] - 1] = 
                                    Interaction[Grids[GridCount - 1][j]][k];
              }
            }
            //now this material point is Assigned and should be so marked
            Assigned[Grids[GridCount - 1][j]] = 1;                   
          }
        }

        // controlling second while loop. 
        // This loop stops when the size remain unchanged
        if (stopCounter == GridsSize[GridCount - 1]) {
            stop = 1;
        } else {
            stopCounter = GridsSize[GridCount - 1];
        }
      } // end while(stop < 1)

      //Finding UnAssigned material points. 
      //resetting the counter for for the number of UnAssigned material ponts.
      UnAssignCount = 0;
      for (int i = 0; i < NumbParticles; i++) {
        // -1 signals that the material point is not UnAssigned 
        // In this for loop, we assume that all 
        // material points are alrready assigned unless the next if 
        // condition find otherwise.
        UnAssigned[i] = -1;
        if (Assigned[i] == -1) {
          UnAssignCount = UnAssignCount + 1;
          UnAssigned[UnAssignCount - 1] = i;
        }
      }
    } // end while (UnAssignCount > 0)

    //Task 2 check 
    cerr << " GridCount " << GridCount << endl;
            
    /*                        
    for (int i = 0; i < NumbParticles; i++) {
      cerr << " GridsSize[" << i << "]= " << GridsSize[i]<< endl;
      for (int j = 0; j < NumbParticles; j++) {                           
        cerr << "Grids[" << i << "]" << "[" << j << "]=" << Grids[i][j] << endl;
      }
    }
    */
    //---------------------------End of Task 2-------------------//


    //------------------------------Task 3----------------------//
    // This task check the particles to make sure they are on the correct 
    // materials. If a particle is not on the correct material, the task 
    // will move that particle to the correct material, copies its 
    // variables and remove the particle from its current material.             
    // moving on the rows of Grids. This for loop make sure that all 
    // the different grids are controlled one by one.

    for (int i = 0; i < GridCount; i++) {
      // Initiating the necessary vectors and counters                      
      // an intger for checking if contact is necessary for a grid or not. 
      int ContactCheck = int((numMatls - 1) / 2);
      // This value is determined once per each grid and helps to move 
      //  particles into matrials with or without contact.
      // int ( numMatls /2) for this value signals that there is no need 
      // for contact. 0 for this value signals the need for contact.
      // initially we assume that there is no need for contact 
      // (thus we set the value to int ( numMatls /2). In 1, we check that
      // and if there is a need the value will be changed to 0.

      //1- Checking if the contact is necessary for this grid or not.
      for (int j = 0; j < GridsSize[i]; j++) {
        // if MaterialIndex of even one material point is 0, then contact 
        // is necessary because there is an elastic/rigid material there.
        if (MaterialIndex[Grids[i][j]]==0) {
           ContactCheck = 0;  // so we change the value of ContactCheck to 0.
        }
      }
                                     
      // 2-moving on the particles of a grid. This for check make sure 
      // that all the different grids are controlled one by one.  
      for (int j = 0; j < GridsSize[i]; j++) {
        // the check to see if a particle is not on the correct material. 
        // If it goes in, this means that the particle on "Grids[i][j]" 
        // needs to move from "MaterialIndex[Grids[i][j]]" to "i" material.
        // This will take care of it.
        if (MaterialIndex[Grids[i][j]]!=0 && 
            MaterialIndex[Grids[i][j]]!=(i+1+ContactCheck)) {
           //----Creating necessary vector and putting old date in them---//
          //cerr << " We are here1 " << endl;

          //1-getting the origin material and Creating necessary vectors 
          MPMMaterial* mpm_matl1 = (MPMMaterial*)m_materialManager->getMaterial("MPM", MaterialIndex[Grids[i][j]]);
          int dwi1 = mpm_matl1->getDWIndex();
          ParticleSubset* pset1 = old_dw->getParticleSubset(dwi1, patch);
          //ConstitutiveModel* cm1 = mpm_matl1->getConstitutiveModel();

          ParticleVariable<Point> px1;
          ParticleVariable<Matrix3> pF1, pSize1, pstress1, pvelgrad1,pscalefac1;
          ParticleVariable<long64> pids1;
          ParticleVariable<double> pvolume1, pmass1, ptemp1, ptempP1, pcolor1;
          ParticleVariable<double> pESF1;
          ParticleVariable<Vector> pvelocity1, pextforce1, pdisp1, ptempgrad1;
          ParticleVariable<int> pref1, ploc1, prefOld1, pSplitR1R2R31;
          ParticleVariable<IntVector> pLoadCID1;

          //2- putting origin data in the created vectors
          new_dw->getModifiable(px1,    lb->pXLabel_preReloc, pset1);
          new_dw->getModifiable(pids1,  lb->pParticleIDLabel_preReloc, pset1);
          new_dw->getModifiable(pmass1, lb->pMassLabel_preReloc, pset1);
          new_dw->getModifiable(pSize1, lb->pSizeLabel_preReloc, pset1);
          new_dw->getModifiable(pdisp1, lb->pDispLabel_preReloc, pset1);
          new_dw->getModifiable(pstress1, lb->pStressLabel_preReloc, pset1);
          new_dw->getModifiable(pvolume1, lb->pVolumeLabel_preReloc, pset1);
          new_dw->getModifiable(pvelocity1, lb->pVelocityLabel_preReloc, pset1);
          if (flags->d_computeScaleFactor) {
            new_dw->getModifiable(pscalefac1, lb->pScaleFactorLabel_preReloc, pset1);
          }
          new_dw->getModifiable(pextforce1, lb->pExtForceLabel_preReloc, pset1);
          new_dw->getModifiable(ptemp1, lb->pTemperatureLabel_preReloc, pset1);
          new_dw->getModifiable(ptempgrad1, lb->pTemperatureGradientLabel_preReloc,
              pset1);
          new_dw->getModifiable(ptempP1, lb->pTempPreviousLabel_preReloc, pset1);
          //new_dw->getModifiable(pref1, lb->pRefinedLabel_preReloc, pset1);
          new_dw->getModifiable(ploc1, lb->pLocalizedMPMLabel_preReloc, pset1);
          new_dw->getModifiable(pvelgrad1, lb->pVelGradLabel_preReloc, pset1);
          new_dw->getModifiable(pF1, lb->pDeformationMeasureLabel_preReloc, pset1);
          if (flags->d_with_color) {
            new_dw->getModifiable(pcolor1, lb->pColorLabel_preReloc, pset1);
          }
          // JBH -- Scalard diffusion variables
          ParticleVariable<double> pConc1, pConcPrev1, pD1, pESFlux1;
          ParticleVariable<Vector> pGradConc1, pArea1;
          if (flags->d_doScalarDiffusion) {
            new_dw->getModifiable(pConc1, lb->diffusion->pConcentration_preReloc, pset1);
            new_dw->getModifiable(pConcPrev1, lb->diffusion->pConcPrevious_preReloc, pset1);
            new_dw->getModifiable(pGradConc1, lb->diffusion->pGradConcentration_preReloc, pset1);
            new_dw->getModifiable(pESFlux1, lb->diffusion->pExternalScalarFlux_preReloc, pset1);
            new_dw->getModifiable(pArea1, lb->diffusion->pArea_preReloc, pset1);
            new_dw->getModifiable(pD1, lb->diffusion->pDiffusivity_preReloc, pset1);
          }
          if (flags->d_useLoadCurves) {
            new_dw->getModifiable(pLoadCID1, lb->pLoadCurveIDLabel_preReloc, pset1);
          }

          new_dw->allocateTemporary(prefOld1, pset1);
          new_dw->allocateTemporary(pSplitR1R2R31, pset1);

          //2- check                            
            /*
          int jj = pset1->addParticles(0);
          cout << "NumbParticles = " << jj << endl;                                                         
          for (int ii = 0; ii < jj; ii++){                              
              cerr << " pmass1[" << ii << "]= " << pmass1[ii]<< endl;
              cerr << " pvolume1[" << ii << "]= " << pvolume1[ii] << endl;
              cerr << " px1[" << ii << ", 0 "<< "]= " << px1[ii](0)<< endl;
              cerr << " px1[" << ii << ", 1 "<< "]= " << px1[ii](1)<< endl;
              cerr << " px1[" << ii << ", 2 "<< "]= " << px1[ii](2)<< endl;                           
          } 
           */

          // 3-getting the destination material,  Creating necessary vectors
          // and putting its data in the created vectors
          cout << "ContactCheck = " << ContactCheck << endl;
          MPMMaterial* mpm_matl2 = (MPMMaterial*)m_materialManager->getMaterial("MPM", (i+1+ContactCheck));
          int dwi2 = mpm_matl2->getDWIndex();
          ParticleSubset* pset2 = old_dw->getParticleSubset(dwi2, patch);
          //ConstitutiveModel* cm2 = mpm_matl2->getConstitutiveModel();

          ParticleVariable<Point>  px2;
          ParticleVariable<Matrix3>  pF2,pSize2,pstress2, pvelgrad2, pscalefac2;
          ParticleVariable<long64>  pids2;
          ParticleVariable<double>  pvolume2, pmass2, ptemp2, ptempP2, pcolor2;
          ParticleVariable<double>  pESF2;
          ParticleVariable<Vector>  pvelocity2, pextforce2, pdisp2, ptempgrad2;
          ParticleVariable<int>  pref2, ploc2, prefOld2, pSplitR1R2R32;
          ParticleVariable<IntVector> pLoadCID2;

          new_dw->getModifiable(px2,        lb->pXLabel_preReloc, pset2);
          new_dw->getModifiable(pids2,     lb->pParticleIDLabel_preReloc,pset2);
          new_dw->getModifiable(pmass2,     lb->pMassLabel_preReloc, pset2);
          new_dw->getModifiable(pSize2,     lb->pSizeLabel_preReloc, pset2);
          new_dw->getModifiable(pdisp2,     lb->pDispLabel_preReloc, pset2);
          new_dw->getModifiable(pstress2,   lb->pStressLabel_preReloc, pset2);
          new_dw->getModifiable(pvolume2,   lb->pVolumeLabel_preReloc, pset2);
          new_dw->getModifiable(pvelocity2, lb->pVelocityLabel_preReloc, pset2);
          if (flags->d_computeScaleFactor) {
            new_dw->getModifiable(pscalefac2, lb->pScaleFactorLabel_preReloc, pset2);
          }
          new_dw->getModifiable(pextforce2, lb->pExtForceLabel_preReloc, pset2);
          new_dw->getModifiable(ptemp2,     lb->pTemperatureLabel_preReloc, pset2);
          new_dw->getModifiable(ptempgrad2, lb->pTemperatureGradientLabel_preReloc, pset2);
          new_dw->getModifiable(ptempP2, lb->pTempPreviousLabel_preReloc, pset2);
          //new_dw->getModifiable(pref2, lb->pRefinedLabel_preReloc, pset2);
          new_dw->getModifiable(ploc2, lb->pLocalizedMPMLabel_preReloc, pset2);
          new_dw->getModifiable(pvelgrad2, lb->pVelGradLabel_preReloc, pset2);
          new_dw->getModifiable(pF2, lb->pDeformationMeasureLabel_preReloc, pset2);
          if (flags->d_with_color) {
            new_dw->getModifiable(pcolor2, lb->pColorLabel_preReloc, pset2);
          }
          // JBH -- Scalard diffusion variables
          ParticleVariable<double> pConc2, pConcPrev2, pD2, pESFlux2;
          ParticleVariable<Vector> pGradConc2, pArea2;
          if (flags->d_doScalarDiffusion) {
            new_dw->getModifiable(pConc2, lb->diffusion->pConcentration_preReloc, pset2);
            new_dw->getModifiable(pConcPrev2, lb->diffusion->pConcPrevious_preReloc, pset2);
            new_dw->getModifiable(pGradConc2, lb->diffusion->pGradConcentration_preReloc, pset2);
            new_dw->getModifiable(pESFlux2, lb->diffusion->pExternalScalarFlux_preReloc, pset2);
            new_dw->getModifiable(pArea2, lb->diffusion->pArea_preReloc, pset2);
            new_dw->getModifiable(pD2, lb->diffusion->pDiffusivity_preReloc, pset2);
          }
          if (flags->d_useLoadCurves) {
            new_dw->getModifiable(pLoadCID2, lb->pLoadCurveIDLabel_preReloc, pset2);
          }

          new_dw->allocateTemporary(prefOld2, pset2);
          new_dw->allocateTemporary(pSplitR1R2R32, pset2);

         //3- check 
         /*
           int jj = pset2->addParticles(0);
           cout << "NumbParticles = " << jj << endl;
          for (int ii = 0; ii < jj; ii++){                              
              cerr << " pmass2[" << ii << "]= " << pmass2[ii] << endl;
              cerr << " pvolume2[" << ii << "]= " << pvolume2[ii] << endl;
              cerr << " px2[" << ii << ", 0 " << "]= " << px2[ii](0) << endl;
              cerr << " px2[" << ii << ", 1 " << "]= " << px2[ii](1) << endl;
              cerr << " px2[" << ii << ", 2 " << "]= " << px2[ii](2) << endl;
          } 
         */

          // 4-Increasing the size of destination material so the new 
          // particle can be added
          const unsigned int oldNumPar = pset2->addParticles(1);
          // cout << "oldNumPar = " << oldNumPar << endl;
          const unsigned int newNumPar = pset2->addParticles(0);
          // cout << "newNumPar = " << newNumPar << endl;

          //4- check 
          /*
          int jj = pset2->addParticles(0);
          cout << "NumbParticles = " << jj << endl;
           
          for (int ii = 0; ii < jj; ii++){
              cerr << " pids2[" << ii << "]= " << pids2[ii] << endl;
              cerr << " px2[" << ii << ", 0 "<< "]= " << px2[ii](0)<< endl;
              cerr << " px2[" << ii << ", 1 "<< "]= " << px2[ii](1)<< endl;
              cerr << " px2[" << ii << ", 2 "<< "]= " << px2[ii](2)<< endl;
          }
          */

          //5-Creating temporary vectors for the new destnation material
          ParticleVariable<Point> pxtmp;
          ParticleVariable<Matrix3> pFtmp, psizetmp, pstrstmp, pvgradtmp,pSFtmp;
          ParticleVariable<long64> pidstmp;
          ParticleVariable<double> pvoltmp, pmasstmp, ptemptmp;
          ParticleVariable<double>  ptempPtmp, pcolortmp, pTempNewtmp;
          ParticleVariable<Vector> pveltmp, pextFtmp, pdisptmp, ptempgtmp;
          ParticleVariable<int> preftmp, ploctmp;
          ParticleVariable<IntVector> pLoadCIDtmp;
          ParticleVariable<double> pConcTmp, pConcPrevTmp, pESFluxTmp, pDTmp;
          ParticleVariable<Vector> pGradConcTmp, pAreaTmp;
         
          // 6-temporary allocation of destination material 
          // to created temporary vectors
          new_dw->allocateTemporary(pidstmp, pset2);
          new_dw->allocateTemporary(pxtmp, pset2);
          new_dw->allocateTemporary(pvoltmp, pset2);
          new_dw->allocateTemporary(pveltmp, pset2);
          if (flags->d_computeScaleFactor) {
              new_dw->allocateTemporary(pSFtmp, pset2);
          }
          new_dw->allocateTemporary(pextFtmp,  pset2);
          new_dw->allocateTemporary(ptemptmp,  pset2);
          new_dw->allocateTemporary(ptempgtmp, pset2);
          new_dw->allocateTemporary(ptempPtmp, pset2);
          new_dw->allocateTemporary(pFtmp,     pset2);
          new_dw->allocateTemporary(psizetmp,  pset2);
          new_dw->allocateTemporary(pdisptmp,  pset2);
          new_dw->allocateTemporary(pstrstmp,  pset2);
          new_dw->allocateTemporary(pmasstmp,  pset2);
          new_dw->allocateTemporary(preftmp,   pset2);
          new_dw->allocateTemporary(ploctmp,   pset2);
          new_dw->allocateTemporary(pvgradtmp, pset2);
          if (flags->d_with_color) {
              new_dw->allocateTemporary(pcolortmp, pset2);
          }

          // JBH - Scalar Diffusion Variables

          if (flags->d_doScalarDiffusion) {
            new_dw->allocateTemporary(pConcTmp, pset2);
            new_dw->allocateTemporary(pConcPrevTmp, pset2);
            new_dw->allocateTemporary(pGradConcTmp, pset2);
            new_dw->allocateTemporary(pESFluxTmp, pset2);
            new_dw->allocateTemporary(pDTmp, pset2);
            new_dw->allocateTemporary(pAreaTmp, pset2);
          }

          if (flags->d_useLoadCurves) {
            new_dw->allocateTemporary(pLoadCIDtmp, pset2);
          }

          //6- check 
          /*
          int jj = pset2->addParticles(0);
           cout << "NumbParticles = " << jj << endl;
          for (int ii = 0; ii < jj; ii++){
              cerr << " pidstmp[" << ii << "]= " << pidstmp[ii] << endl;
              cerr << " pxtmp[" << ii << ", 0 "<< "]= " << pxtmp[ii](0)<< endl;
              cerr << " pxtmp[" << ii << ", 1 "<< "]= " << pxtmp[ii](1)<< endl;
              cerr << " pxtmp[" << ii << ", 2 "<< "]= " << pxtmp[ii](2)<< endl;
          }
         */

          //7-copying data from variables of destnation to temporary particles
          for (unsigned int pp = 0; pp < oldNumPar; ++pp) {
            pidstmp[pp] = pids2[pp];
            pxtmp[pp] = px2[pp];
            pvoltmp[pp] = pvolume2[pp];
            pveltmp[pp] = pvelocity2[pp];
            pextFtmp[pp] = pextforce2[pp];
            ptemptmp[pp] = ptemp2[pp];
            ptempgtmp[pp] = ptempgrad2[pp];
            ptempPtmp[pp] = ptempP2[pp];
            pFtmp[pp] = pF2[pp];
            psizetmp[pp] = pSize2[pp];
            pdisptmp[pp] = pdisp2[pp];
            pstrstmp[pp] = pstress2[pp];
            if (flags->d_computeScaleFactor) {
                pSFtmp[pp] = pscalefac2[pp];
            }
            if (flags->d_with_color) {
                pcolortmp[pp] = pcolor2[pp];
            }
            if (flags->d_useLoadCurves) {
                pLoadCIDtmp[pp] = pLoadCID2[pp];
            }
            pmasstmp[pp] = pmass2[pp];
            //preftmp[pp] = pref2[pp];
            ploctmp[pp] = ploc2[pp];
            pvgradtmp[pp] = pvelgrad2[pp];
          }

          if (flags->d_doScalarDiffusion) {
            for (unsigned int pp = 0; pp < oldNumPar; ++pp) {
              pConcTmp[pp] = pConc2[pp];
              pConcPrevTmp[pp] = pConcPrev2[pp];
              pGradConcTmp[pp] = pGradConc2[pp];
              pESFluxTmp[pp] = pESFlux2[pp];
              pAreaTmp[pp] = pArea2[pp];
              pDTmp[pp] = pD2[pp];
            }
          }

          //7- check 
          /*
           int jj = pset2->addParticles(0);
           cout << "NumbParticles = " << jj << endl;
          for (int ii = 0; ii < jj; ii++){
              cerr << " pidstmp[" << ii << "]= " << pidstmp[ii] << endl;
              cerr << " pxtmp[" << ii << ", 0 "<< "]= " << pxtmp[ii](0)<< endl;
              cerr << " pxtmp[" << ii << ", 1 "<< "]= " << pxtmp[ii](1)<< endl;
              cerr << " pxtmp[" << ii << ", 2 "<< "]= " << pxtmp[ii](2)<< endl;
          }
         */

          // 8-copying data from variables of origin and removing it from origin
          // Currently, I am adding just one particle. so no loop is nedded. 
          // Later, i will do it more efficently.                               

          pidstmp[newNumPar - 1] = pids1[ParticleIndex[Grids[i][j]]];
          pxtmp[newNumPar-1] = px1[ParticleIndex[Grids[i][j]]];
          pvoltmp[newNumPar-1] = pvolume1[ParticleIndex[Grids[i][j]]];
          pveltmp[newNumPar-1] = pvelocity1[ParticleIndex[Grids[i][j]]];
          pextFtmp[newNumPar-1] = pextforce1[ParticleIndex[Grids[i][j]]];
          ptemptmp[newNumPar-1] = ptemp1[ParticleIndex[Grids[i][j]]];
          ptempgtmp[newNumPar-1] = ptempgrad1[ParticleIndex[Grids[i][j]]];
          ptempPtmp[newNumPar-1] = ptempP1[ParticleIndex[Grids[i][j]]];
          pFtmp[newNumPar-1] = pF1[ParticleIndex[Grids[i][j]]];
          psizetmp[newNumPar-1] = pSize1[ParticleIndex[Grids[i][j]]];
          pdisptmp[newNumPar-1] = pdisp1[ParticleIndex[Grids[i][j]]];
          pstrstmp[newNumPar-1] = pstress1[ParticleIndex[Grids[i][j]]];
          if (flags->d_computeScaleFactor) {
            pSFtmp[newNumPar-1] = pscalefac1[ParticleIndex[Grids[i][j]]];
          }
          if (flags->d_with_color) {
            pcolortmp[newNumPar-1] = pcolor1[ParticleIndex[Grids[i][j]]];
          }
          if (flags->d_useLoadCurves) {
            pLoadCIDtmp[newNumPar-1] = pLoadCID1[ParticleIndex[Grids[i][j]]];
          }
          pmasstmp[newNumPar-1] = pmass1[ParticleIndex[Grids[i][j]]];
          //preftmp[newNumPar-1] = pref1[ParticleIndex[Grids[i][j]]];
          ploctmp[newNumPar-1] = ploc1[ParticleIndex[Grids[i][j]]];
          pvgradtmp[newNumPar-1] = pvelgrad1[ParticleIndex[Grids[i][j]]];
          
          // Now that the data are copied we should sign the material point
          // to be removed from the origin. 

          // The -1 signals that this material point should be removed
          CurrentGrids[MaterialIndex[Grids[i][j]]][ParticleIndex[Grids[i][j]]]
                                                                      = { -1 };

          //ParticleSubset* delset = scinew ParticleSubset(0, dwi1, patch);
          
          //delset->addParticle(ParticleIndex[Grids[i][j]]);  
          //new_dw->deleteParticles(delset);

          //8- check 
          /*
           int jj = pset2->addParticles(0);
          cout << "NumbParticles = " << jj << endl;
          for (int ii = 0; ii < jj; ii++){
              cerr << " pidstmp[" << ii << "]= " << pidstmp[ii] << endl;
              cerr << " pxtmp[" << ii << ", 0 "<< "]= " << pxtmp[ii](0)<< endl;
              cerr << " pxtmp[" << ii << ", 1 "<< "]= " << pxtmp[ii](1)<< endl;
              cerr << " pxtmp[" << ii << ", 2 "<< "]= " << pxtmp[ii](2)<< endl;
          }
         */

          //9-putting back temporary data
          new_dw->put(pidstmp, lb->pParticleIDLabel_preReloc, true);
          new_dw->put(pxtmp,   lb->pXLabel_preReloc, true);
          new_dw->put(pvoltmp, lb->pVolumeLabel_preReloc, true);
          new_dw->put(pveltmp, lb->pVelocityLabel_preReloc, true);
          if (flags->d_computeScaleFactor) {
            new_dw->put(pSFtmp, lb->pScaleFactorLabel_preReloc, true);
          }
          new_dw->put(pextFtmp,  lb->pExtForceLabel_preReloc, true);
          new_dw->put(pmasstmp,  lb->pMassLabel_preReloc, true);
          new_dw->put(ptemptmp,  lb->pTemperatureLabel_preReloc, true);
          new_dw->put(ptempgtmp, lb->pTemperatureGradientLabel_preReloc, true);
          new_dw->put(ptempPtmp, lb->pTempPreviousLabel_preReloc, true);
          new_dw->put(psizetmp,  lb->pSizeLabel_preReloc, true);
          new_dw->put(pdisptmp,  lb->pDispLabel_preReloc, true);
          new_dw->put(pstrstmp,  lb->pStressLabel_preReloc, true);
          if (flags->d_with_color) {
            new_dw->put(pcolortmp, lb->pColorLabel_preReloc, true);
          }
          if (flags->d_doScalarDiffusion) {
            new_dw->put(pConcTmp, lb->diffusion->pConcentration_preReloc, true);
            new_dw->put(pConcPrevTmp,lb->diffusion->pConcPrevious_preReloc, true);
            new_dw->put(pGradConcTmp,lb->diffusion->pGradConcentration_preReloc, true);
            new_dw->put(pESFluxTmp,  lb->diffusion->pExternalScalarFlux_preReloc, true);
            new_dw->put(pAreaTmp,    lb->diffusion->pArea_preReloc, true);
            new_dw->put(pDTmp,       lb->diffusion->pDiffusivity_preReloc,true);
          }

          if (flags->d_useLoadCurves) {
            new_dw->put(pLoadCIDtmp, lb->pLoadCurveIDLabel_preReloc, true);
          }
          new_dw->put(pFtmp, lb->pDeformationMeasureLabel_preReloc, true);
          //new_dw->put(preftmp, lb->pRefinedLabel_preReloc, true);
          new_dw->put(ploctmp, lb->pLocalizedMPMLabel_preReloc, true);
          new_dw->put(pvgradtmp, lb->pVelGradLabel_preReloc, true);

          //9- check 

          /*
          int jj = pset2->addParticles(0);
          // cout << "NumbParticles = " << jj << endl;
          for (int ii = 0; ii < jj; ii++){
           //cerr << " pidstmp[" << ii << "]= " << pidstmp[ii] << endl;
           //cerr << " pveltmp[" << ii << "]= " << pveltmp[ii] << endl;
           cerr << " pstrstmp[" << ii << ", 0 "<< "]= " << pstrstmp[ii](<< endl;
           //cerr << " pxtmp[" << ii << ", 1 "<< "]= " << pxtmp[ii](1)<< endl;
           //cerr << " pxtmp[" << ii << ", 2 "<< "]= " << pxtmp[ii](2)<< endl;
          }
         */
        } //end of the check to see if a particle is not on the correct material
      } //end of moving on the particles of a grid.
    } //end of moving on the rows of Grids.

    //10-removing the material points from origin material
    unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");
    for (unsigned int m = 1; m < numMPMMatls; m++) {
      MPMMaterial* mpm_matl = 
                         (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
     
      for (ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          particleIndex idx = *iter;
          if (CurrentGrids[dwi][idx] ==  -1 ) {
              delset->addParticle(idx);
          }
      } // particles
      new_dw->deleteParticles(delset);
    } // materials  

  //---------------------------------End of Task 3-------------------------// 
  }  // loop over patches
}
