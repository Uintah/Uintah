/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/MPM/SingleHydroMPM.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
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

// Hydro-mechanical coupling includes
#include <CCA/Components/MPM/PhysicalBC/HydrostaticBC.h>
#include <CCA/Components/MPM/Materials/Contact/FluidContact.h>

#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/HydroMPMLabel.h>
#include <CCA/Components/MPM/Core/CZLabel.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);
static DebugStream cout_dbg("SingleHydroMPM", false);
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

SingleHydroMPM::SingleHydroMPM( const ProcessorGroup* myworld,
                      const MaterialManagerP materialManager) :
  MPMCommon( myworld, materialManager )
{
  flags = scinew MPMFlags(myworld);
  Hlb = scinew HydroMPMLabel();
  Cl = scinew CZLabel();

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  contactModel        = nullptr;
  thermalContactModel = nullptr;
  heatConductionModel = nullptr;
  fluidContactModel = nullptr;
  NGP     = 1;
  NGN     = 1;
  d_loadCurveIndex=0;
  d_switchCriteria = nullptr;

  d_fracture = false;

  // Diffusion related
  d_fluxBC           = nullptr;
  d_sdInterfaceModel = nullptr;


  d_mpm = scinew SerialMPM(myworld, m_materialManager);
}

SingleHydroMPM::~SingleHydroMPM()
{
  delete contactModel;
  delete thermalContactModel;
  delete heatConductionModel;
  delete fluidContactModel;
  delete d_fluxBC;
  delete d_sdInterfaceModel;
  delete flags;
  delete Hlb;
  delete d_switchCriteria;

  d_mpm->releaseComponents();
  delete d_mpm;

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

void SingleHydroMPM::problemSetup(const ProblemSpecP& prob_spec,
                             const ProblemSpecP& restart_prob_spec,
                             GridP& grid)
{
  cout_doing<<"Doing MPM::problemSetup\t\t\t\t\t MPM"<<endl;

  //__________________________________
  //  M P M
  //d_mpm->setComponents(this);
  //dynamic_cast<ApplicationCommon*>(d_mpm)->problemSetup(prob_spec);
  //d_mpm->problemSetup(prob_spec, restart_prob_spec, grid);

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

  fluidContactModel = scinew FluidContact(d_myworld, m_materialManager, lb,Hlb, flags);

  thermalContactModel =
    ThermalContactFactory::create(restart_mat_ps, m_materialManager, lb,flags);

  heatConductionModel = scinew HeatConduction(m_materialManager,lb,flags);

  materialProblemSetup(restart_mat_ps,flags, isRestart);

  cohesiveZoneTasks = scinew CohesiveZoneTasks(m_materialManager, flags);

  cohesiveZoneTasks->cohesiveZoneProblemSetup(restart_mat_ps, flags);

  if (flags->d_doScalarDiffusion) {
    d_sdInterfaceModel = SDInterfaceModelFactory::create(restart_mat_ps, m_materialManager, flags, lb);
  }
  d_fluxBC = FluxBCModelFactory::create(m_materialManager, flags);

  //__________________________________
  //  create analysis modules
  // call problemSetup
  if(!flags->d_with_ice){ // mpmice or mpmarches handles this
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
void SingleHydroMPM::outputProblemSpec(ProblemSpecP& root_ps)
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
  fluidContactModel->outputProblemSpec(mpm_ps);
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
  //  output data analysis modules. Mpmice or mpmarches handles this
  if(!flags->d_with_ice && d_analysisModules.size() != 0){

    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;

      am->outputProblemSpec( root_ps );
    }
  }

}

void SingleHydroMPM::scheduleInitialize(const LevelP& level,
                                   SchedulerP& sched)
{

    //printSchedule(level, cout_doing, "SingleHydroMPM::scheduleInitialize");

   // d_mpm->scheduleInitialize(level, sched);
    
    
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())) {
    return;
  }
  Task* t = scinew Task( "MPM::actuallyInitialize", this, &SingleHydroMPM::actuallyInitialize );

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

  // Hydro-mechanical coupling
  if (flags->d_coupledflow) {
      t->computes(Hlb->pFluidVelocityLabel);
      t->computes(Hlb->pFluidMassLabel);
      t->computes(Hlb->pSolidMassLabel);
      t->computes(Hlb->pPorePressureLabel);
      t->computes(Hlb->pPrescribedPorePressureLabel);
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
void SingleHydroMPM::scheduleRestartInitialize(const LevelP& level,
                                          SchedulerP& sched)
{
    printSchedule(level, cout_doing, "SingleHydroMPM::scheduleInitialize");

    d_mpm->scheduleRestartInitialize(level, sched);

    //__________________________________
    // dataAnalysis 
    if (d_analysisModules.size() != 0) {
        vector<AnalysisModule*>::iterator iter;
        for (iter = d_analysisModules.begin();
            iter != d_analysisModules.end(); iter++) {
            AnalysisModule* am = *iter;
            am->scheduleRestartInitialize(sched, level);
        }
    }
}

/* _____________________________________________________________________
 Purpose:   Set variables that are normally set during the initialization
            phase, but get wiped clean when you restart
_____________________________________________________________________*/
void SingleHydroMPM::restartInitialize()
{
    if (cout_doing.active())
        cout_doing << "Doing restartInitialize \t\t\t SingleHydroMPM" << endl;

    d_mpm->restartInitialize();

    if (d_analysisModules.size() != 0) {
        vector<AnalysisModule*>::iterator iter;
        for (iter = d_analysisModules.begin();
            iter != d_analysisModules.end(); iter++) {
            AnalysisModule* am = *iter;
            am->restartInitialize();
        }
    }
}

//______________________________________________________________________
void SingleHydroMPM::schedulePrintParticleCount(const LevelP& level,
                                           SchedulerP& sched)
{

    
  Task* t = scinew Task("MPM::printParticleCount",
                        this, &SingleHydroMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "MPM" ));
  
}

//__________________________________
//  Diagnostic task: compute the total number of particles
void SingleHydroMPM::scheduleTotalParticleCount(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())){
    return;
  }

  Task* t = scinew Task("SingleHydroMPM::totalParticleCount",
                  this, &SingleHydroMPM::totalParticleCount);
  t->computes(lb->partCountLabel);

  sched->addTask(t, patches,matls);
}

//__________________________________
//  Diagnostic task: compute the total number of particles
void SingleHydroMPM::totalParticleCount(const ProcessorGroup*,
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

void SingleHydroMPM::scheduleInitializePressureBCs(const LevelP& level,
                                              SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

  d_loadCurveIndex = scinew MaterialSubset();
  d_loadCurveIndex->add(0);
  d_loadCurveIndex->addReference();

  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure" || bcs_type == "Hydrostatic"){
      d_loadCurveIndex->add(nofPressureBCs++);
    }
  }
  if (nofPressureBCs > 0) {
    printSchedule(patches,cout_doing,"MPM::countMaterialPointsPerLoadCurve");
    printSchedule(patches,cout_doing,"MPM::scheduleInitializePressureBCs");
    // Create a task that calculates the total number of particles
    // associated with each load curve.
    Task* t = scinew Task("MPM::countMaterialPointsPerLoadCurve",
                          this, &SingleHydroMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, d_loadCurveIndex,
                Task::OutOfDomain);

    t->computes(Hlb->boundaryPointsPerCellLabel);

    sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializePressureBC",
                    this, &SingleHydroMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel,                        Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel,              Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel,
                            d_loadCurveIndex, Task::OutOfDomain, Ghost::None);
    t->modifies(lb->pExternalForceLabel);
   
    if (flags->d_coupledflow) {
        t->requires(Task::NewDW, Hlb->boundaryPointsPerCellLabel, Ghost::None);
        t->modifies(Hlb->pPrescribedPorePressureLabel);
    }

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

void SingleHydroMPM::scheduleDeleteGeometryObjects(const LevelP& level,
                                              SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

  Task* t = scinew Task("MPM::deleteGeometryObjects",
                  this, &SingleHydroMPM::deleteGeometryObjects);
  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
}

void SingleHydroMPM::scheduleComputeStableTimeStep(const LevelP& level,
                                              SchedulerP& sched)
{
    // Schedule computing the MPM stable timestep
    d_mpm->scheduleComputeStableTimeStep(level, sched);
    // MPM stable timestep is a by product of the CM
}

void
SingleHydroMPM::scheduleTimeAdvance(const LevelP & level,
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

  d_mpm->scheduleComputeCurrentParticleSize(     sched, patches, matls);

  // Cannot recall ApplyExternalLoads in mpm?
  //d_mpm->scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleApplyExternalFluidLoads(sched, patches, matls);

  if(flags->d_doScalarDiffusion) {
      d_mpm->d_fluxBC->scheduleApplyExternalScalarFlux(sched, patches, matls);
  }
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  if(flags->d_computeNormals){
      d_mpm->scheduleComputeNormals(               sched, patches, matls);
  }
  if(flags->d_useLogisticRegression){
      d_mpm->scheduleFindSurfaceParticles(         sched, patches, matls);
      d_mpm->scheduleComputeLogisticRegression(    sched, patches, matls);
  }
  scheduleExMomInterpolated(                       sched, patches, matls);
  if(flags->d_doScalarDiffusion) {
      d_mpm->scheduleConcInterpolated(             sched, patches, matls);
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
      d_mpm->scheduleComputeContactArea(           sched, patches, matls);
  }
  scheduleComputeInternalForce(           sched, patches, matls);
  if (flags->d_doScalarDiffusion) {
      d_mpm->scheduleComputeFlux(                  sched, patches, matls);
      d_mpm->scheduleComputeDivergence(            sched, patches, matls);
      d_mpm->scheduleDiffusionInterfaceDiv(        sched, patches, matls);
  }

  if (flags->d_coupledflow) {
      if (flags->d_coupledflow_contact) {  // Coupled flow contact only with
                                           // friction contact
          scheduleFluidExMomInterpolated(sched, patches, matls);
      }
      scheduleComputeFluidDragForce(sched, patches, matls);
      scheduleComputeAndIntegrateFluidAcceleration(sched, patches, matls);

      if (flags->d_coupledflow_contact) {  // Coupled flow contact only with
                                           // friction contact
          scheduleFluidExMomIntegrated(sched, patches, matls);
      }
  }

  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
  if (flags->d_doScalarDiffusion) {
      d_mpm->scheduleComputeAndIntegrateDiffusion( sched, patches, matls);
  }
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  if (flags->d_prescribeDeformation){
      d_mpm->scheduleSetPrescribedMotion(          sched, patches, matls);
  }
  if(flags->d_XPIC2){
      d_mpm->scheduleComputeSSPlusVp(              sched, patches, matls);
      d_mpm->scheduleComputeSPlusSSPlusVp(         sched, patches, matls);
  }
  if(flags->d_doExplicitHeatConduction){ // somehow cannot use d_mpm, it leads to errors
    scheduleComputeHeatExchange(          sched, patches, matls);
    scheduleComputeInternalHeatRate(      sched, patches, matls);
    scheduleComputeNodalHeatFlux(         sched, patches, matls);
    scheduleSolveHeatEquations(           sched, patches, matls);
    scheduleIntegrateTemperatureRate(     sched, patches, matls);
  }
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleComputeParticleGradients(       sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleFinalParticleUpdate(            sched, patches, matls);

  if (flags->d_PorePressureFilter) {
      scheduleInterpolateParticleToGridFilter(sched, patches, matls);
      scheduleInterpolateGridToParticleFilter(sched, patches, matls);
  }

  scheduleInsertParticles(                    sched, patches, matls);
  if(flags->d_computeScaleFactor){
      d_mpm->scheduleComputeParticleScaleFactor(       sched, patches, matls);
  }
  if(flags->d_refineParticles){
      d_mpm->scheduleAddParticles(                     sched, patches, matls);
  }

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis_preReloc( sched, level);
    }
  }

 d_mpm->scheduleParticleRelocation(           sched, level,  matls,  
                                                             cz_matls);

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

void SingleHydroMPM::scheduleApplyExternalFluidLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"SingleHydroMPM::scheduleApplyExternalLoads");

  Task* t=scinew Task("SingleHydroMPM::applyExternalFluidLoads",
                    this, &SingleHydroMPM::applyExternalFluidLoads);

  t->requires(Task::OldDW, lb->simulationTimeLabel);

  if (!flags->d_mms_type.empty()) {
    //MMS problems need displacements
    t->requires(Task::OldDW, lb->pDispLabel,            Ghost::None);
  }

  if (flags->d_useLoadCurves || flags->d_useCBDI) {
    t->requires(Task::OldDW,    lb->pXLabel,                  Ghost::None);
    t->requires(Task::OldDW,    lb->pLoadCurveIDLabel,        Ghost::None);
    t->computes(                lb->pLoadCurveIDLabel_preReloc);

    //if (flags->d_coupledflow) {
       // t->requires(Task::OldDW, lb->pPrescribedPorePressureLabel, Ghost::None);
       // t->computes(lb->pPrescribedPorePressureLabel);
   // }

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

void SingleHydroMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleInterpolateParticlesToGrid");

  Task* t = scinew Task("MPM::interpolateParticlesToGrid",
                        this,&SingleHydroMPM::interpolateParticlesToGrid);
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

  if (flags->d_coupledflow) {
      t->requires(Task::OldDW, Hlb->pFluidMassLabel, gan, NGP);
      t->requires(Task::OldDW, Hlb->pSolidMassLabel, gan, NGP);
      t->requires(Task::OldDW, Hlb->pFluidVelocityLabel, gan, NGP);
      t->requires(Task::OldDW, Hlb->pPorePressureLabel, gan, NGP);
      //t->requires(Task::NewDW, lb->pPrescribedPorePressureLabel, gan, NGP);

      t->computes(Hlb->gExternalFluidForceLabel);

      t->computes(Hlb->gFluidMassLabel);
      t->computes(Hlb->gFluidMassBarLabel);
      t->computes(Hlb->gFluidVelocityLabel);
      t->computes(Hlb->gFluidMassLabel, m_materialManager->getAllInOneMatls(),
          Task::OutOfDomain);
      t->computes(Hlb->gFluidMassBarLabel, m_materialManager->getAllInOneMatls(),
          Task::OutOfDomain);
      t->computes(Hlb->gFluidVelocityLabel, m_materialManager->getAllInOneMatls(),
          Task::OutOfDomain);
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

void SingleHydroMPM::scheduleComputeHeatExchange(SchedulerP& sched,
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

void SingleHydroMPM::scheduleExMomInterpolated(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleExMomInterpolated");

  contactModel->addComputesAndRequiresInterpolated(sched, patches, matls);
}

void SingleHydroMPM::scheduleFluidExMomInterpolated(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls) {
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;
    printSchedule(patches, cout_doing, "MPM::scheduleFluidExMomInterpolated");

    fluidContactModel->addComputesAndRequiresInterpolated(sched, patches, matls);
}


/////////////////////////////////////////////////////////////////////////
/*!  **WARNING** In addition to the stresses and deformations, the internal
 *               heat rate in the particles (pdTdtLabel)
 *               is computed here */
/////////////////////////////////////////////////////////////////////////
void SingleHydroMPM::scheduleComputeStressTensor(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeStressTensor");

  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  Task* t = scinew Task("MPM::computeStressTensor",
                        this, &SingleHydroMPM::computeStressTensor);
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
void SingleHydroMPM::scheduleComputeAccStrainEnergy(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleComputeAccStrainEnergy");

  Task* t = scinew Task("MPM::computeAccStrainEnergy",
                        this, &SingleHydroMPM::computeAccStrainEnergy);
  t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
  t->requires(Task::NewDW, lb->StrainEnergyLabel);
  t->computes(lb->AccStrainEnergyLabel);
  sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeInternalForce");

  Task* t = scinew Task("MPM::computeInternalForce",
                        this, &SingleHydroMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, m_materialManager->getAllInOneMatls(),
              Task::OutOfDomain, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::NewDW,lb->pCurSizeLabel,              gan,NGP);

  //if(flags->d_with_ice){
  //  t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  //}

  if(flags->d_artificial_viscosity){
    t->requires(Task::OldDW, lb->p_qLabel,                gan,NGP);
  }

  t->computes(lb->gInternalForceLabel);

  if (flags->d_coupledflow) {
      t->requires(Task::OldDW, Hlb->pPorePressureLabel, gan, NGP);
      // t->requires(Task::NewDW, Hlb->pPorePressureLabel_preReloc, gan, NGP);
      t->computes(Hlb->gInternalFluidForceLabel);

  }
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

void SingleHydroMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  heatConductionModel->scheduleComputeInternalHeatRate(sched,patches,matls);
}

void SingleHydroMPM::scheduleComputeNodalHeatFlux(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  heatConductionModel->scheduleComputeNodalHeatFlux(sched,patches,matls);
}

void SingleHydroMPM::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  heatConductionModel->scheduleSolveHeatEquations(sched,patches,matls);
}

void SingleHydroMPM::scheduleComputeFluidDragForce(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls) {
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    /*
     * computeFluidDragForce
     *   in(P.VELOCITIES, P.NAT_X, P.MASS)
     *   operation(evaluate the divergence of the stress (stored in
     *   P.CONMOD) using P.NAT_X and the gradients of the
     *   shape functions)
     * out(G.DRAG_INTERNAL) */

    printSchedule(patches, cout_doing, "SingleHydroMPM::scheduleComputeFluidDragForce");

    Task* t = scinew Task("SingleHydroMPM::computeFluidDragForce", this,
        &SingleHydroMPM::computeFluidDragForce);

    Ghost::GhostType gan = Ghost::AroundNodes;
    //Ghost::GhostType gnone = Ghost::None;
    t->requires(Task::OldDW, lb->pVelocityLabel, gan, NGP);
    t->requires(Task::OldDW, Hlb->pFluidVelocityLabel, gan, NGP);
    t->requires(Task::OldDW, Hlb->pFluidMassLabel, gan, NGP);

    t->requires(Task::NewDW, lb->gVelocityLabel, gan, NGP);
    t->requires(Task::NewDW, Hlb->gFluidVelocityLabel, gan, NGP);

    t->requires(Task::OldDW, lb->pXLabel, gan, NGP);
    //t->requires(Task::OldDW, lb->pSizeLabel, gan, NGP);
    t->requires(Task::NewDW, lb->pCurSizeLabel, gan, NGP);
    t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan, NGP);

    t->computes(Hlb->gInternalDragForceLabel);

    sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleComputeAndIntegrateFluidAcceleration(
    SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls) {
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    printSchedule(patches, cout_doing,
        "SingleHydroMPM::scheduleComputeAndIntegrateFluidAcceleration");

    Task* t = scinew Task("SingleHydroMPM::computeAndIntegrateFluidAcceleration", this,
        &SingleHydroMPM::computeAndIntegrateFluidAcceleration);

    t->requires(Task::OldDW, lb->delTLabel);
    t->requires(Task::NewDW, Hlb->gFluidMassLabel, Ghost::None);
    t->requires(Task::NewDW, Hlb->gFluidVelocityLabel, Ghost::None);
    t->requires(Task::NewDW, Hlb->gExternalFluidForceLabel, Ghost::None);
    t->requires(Task::NewDW, Hlb->gInternalFluidForceLabel, Ghost::None);
    t->requires(Task::NewDW, Hlb->gInternalDragForceLabel, Ghost::None);

    t->computes(Hlb->gFluidVelocityStarLabel);
    t->computes(Hlb->gFluidAccelerationLabel);

    sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &SingleHydroMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  if (flags->d_coupledflow) {
      t->requires(Task::NewDW, Hlb->gFluidMassLabel, Ghost::None);
      t->requires(Task::NewDW, Hlb->gFluidMassBarLabel, Ghost::None);
      t->requires(Task::NewDW, Hlb->gInternalFluidForceLabel, Ghost::None);
      t->requires(Task::NewDW, Hlb->gFluidVelocityLabel, Ghost::None);
      t->requires(Task::NewDW, Hlb->gFluidAccelerationLabel, Ghost::None);
  }

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes( VarLabel::find(abortTimeStep_name) );
  t->computes( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  heatConductionModel->scheduleIntegrateTemperatureRate(sched,patches,matls);
}

void SingleHydroMPM::scheduleExMomIntegrated(SchedulerP& sched,
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

void SingleHydroMPM::scheduleFluidExMomIntegrated(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls) {
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    /* FluidExMomIntegrated
     *   in(G.MASS, G.VELOCITY_STAR, G.FLUIDVELOCITY_STAR, G.FLUIDACCELERATION)
     *   operation(peform operations which will cause the fluid
     *              velocity field to feel the influence of the
     *              the impermeable fields according to specific rules)
     *   out(G.FLUIDVELOCITY_STAR, G.FLUIDACCELERATION) */
    printSchedule(patches, cout_doing, "MPM::scheduleFluidExMomIntegrated");
    fluidContactModel->addComputesAndRequiresIntegrated(sched, patches, matls);
}

void SingleHydroMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleSetGridBoundaryConditions");
  Task* t=scinew Task("MPM::setGridBoundaryConditions",
                      this, &SingleHydroMPM::setGridBoundaryConditions);

  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, lb->delTLabel );

  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

  if (flags->d_coupledflow) {
      t->modifies(Hlb->gFluidAccelerationLabel, mss);
      t->modifies(Hlb->gFluidVelocityStarLabel, mss);
      t->requires(Task::NewDW, Hlb->gFluidVelocityLabel, Ghost::None);
  }

  sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                               "MPM::scheduleInterpolateToParticlesAndUpdate");

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &SingleHydroMPM::interpolateToParticlesAndUpdate);

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

  if (flags->d_coupledflow) {
      t->requires(Task::OldDW, Hlb->pSolidMassLabel, gnone);
      t->requires(Task::OldDW, Hlb->pFluidMassLabel, gnone);
      t->requires(Task::OldDW, Hlb->pFluidVelocityLabel, gnone);
      t->requires(Task::OldDW, Hlb->pPorosityLabel, gnone);
      t->requires(Task::NewDW, Hlb->gFluidAccelerationLabel, gac, NGN);

      t->computes(Hlb->pFluidMassLabel_preReloc);
      t->computes(Hlb->pSolidMassLabel_preReloc);
      t->computes(Hlb->pFluidVelocityLabel_preReloc);
      t->computes(Hlb->pPorosityLabel_preReloc);
  }

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

void SingleHydroMPM::scheduleComputeParticleGradients(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeParticleGradients");

  Task* t=scinew Task("MPM::computeParticleGradients",
                      this, &SingleHydroMPM::computeParticleGradients);

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
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);

  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pTemperatureGradientLabel_preReloc);

  // Hydro mechanical coupling
  if (flags->d_coupledflow) {
      t->requires(Task::OldDW, Hlb->pPorePressureLabel, gnone);
      t->requires(Task::NewDW, Hlb->gFluidVelocityStarLabel, gac, NGN);
      t->computes(Hlb->pPorePressureLabel_preReloc);
  }

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

void SingleHydroMPM::scheduleFinalParticleUpdate(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleFinalParticleUpdate");

  Task* t=scinew Task("MPM::finalParticleUpdate",
                      this, &SingleHydroMPM::finalParticleUpdate);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->pdTdtLabel,                      gnone);
  t->requires(Task::NewDW, lb->pLocalizedMPMLabel_preReloc,     gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

  /*
  if (flags->d_coupledflow) {
      t->requires(Task::NewDW, Hlb->pFluidMassLabel_preReloc, gnone);
      t->requires(Task::NewDW, Hlb->pSolidMassLabel_preReloc, gnone);
      t->requires(Task::NewDW, Hlb->pVelGradLabel_preReloc, gnone);
      t->modifies(Hlb->pVolumeLabel_preReloc);
  }
  */
  sched->addTask(t, patches, matls);
}

// Null space filter using local method
void SingleHydroMPM::scheduleInterpolateParticleToGridFilter(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    printSchedule(patches, cout_doing, "SingleHydroMPM::scheduleInterpolateParticleToGridFilter");

    Task* t = scinew Task("SingleHydroMPM::InterpolateParticleToGridFilter",
        this, &SingleHydroMPM::InterpolateParticleToGridFilter);

    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gnone = Ghost::None;

    t->requires(Task::NewDW, lb->gVolumeLabel, gnone);
    t->requires(Task::NewDW, lb->gVolumeLabel, m_materialManager->getAllInOneMatls(),
        Task::OutOfDomain, gnone);
    t->requires(Task::OldDW, lb->pVolumeLabel, gan, NGP);
    t->requires(Task::OldDW, lb->pXLabel, gan, NGP);
    t->requires(Task::NewDW, lb->pCurSizeLabel, gan, NGP);

    t->requires(Task::NewDW, Hlb->pPorePressureLabel_preReloc, gan, NGP);
    t->computes(Hlb->gPorePressureFilterLabel);
    t->computes(Hlb->gPorePressureFilterLabel, m_materialManager->getAllInOneMatls(),
        Task::OutOfDomain);

    sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleInterpolateGridToParticleFilter(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)

{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    printSchedule(patches, cout_doing,
        "SingleHydroMPM::scheduleInterpolateGridToParticleFilter");

    Task* t = scinew Task("SingleHydroMPM::InterpolateGridToParticleFilter",
        this, &SingleHydroMPM::InterpolateGridToParticleFilter);

    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gnone = Ghost::None;
    t->requires(Task::OldDW, lb->pXLabel, gnone);
    t->requires(Task::NewDW, lb->pCurSizeLabel, gnone);

    t->requires(Task::NewDW, Hlb->gPorePressureFilterLabel, gac, NGN);
    t->computes(Hlb->pPorePressureFilterLabel_preReloc);

    sched->addTask(t, patches, matls);
}

void SingleHydroMPM::scheduleInsertParticles(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)

{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
        getLevel(patches)->getGrid()->numLevels()))
        return;

    if (flags->d_insertParticles) {
        printSchedule(patches, cout_doing, "MPM::scheduleInsertParticles");

        Task* t = scinew Task("MPM::insertParticles", this,
            &SingleHydroMPM::insertParticles);

        t->requires(Task::OldDW, lb->simulationTimeLabel);
        t->requires(Task::OldDW, lb->delTLabel);

        t->modifies(lb->pXLabel_preReloc);
        t->modifies(lb->pVelocityLabel_preReloc);
        t->requires(Task::OldDW, lb->pColorLabel, Ghost::None);

        sched->addTask(t, patches, matls);
    }
}

void
SingleHydroMPM::scheduleRefine( const PatchSet   * patches,
                                 SchedulerP & sched )
{
  printSchedule(patches,cout_doing,"SingleHydroMPM::scheduleRefine");
  Task* t = scinew Task( "SingleHydroMPM::refine", this, &SingleHydroMPM::refine );

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
SingleHydroMPM::scheduleRefineInterface( const LevelP& /*fineLevel*/,
                                          SchedulerP& /*scheduler*/,
                                          bool /* ??? */,
                                          bool /* ??? */)
{
  //  do nothing for now
}

void SingleHydroMPM::scheduleCoarsen(const LevelP& /*coarseLevel*/,
                                SchedulerP& /*sched*/)
{
  // do nothing for now
}

//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void SingleHydroMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched)
{
  // main way is to count particles, but for now we only want particles on
  // the finest level.  Thus to schedule cells for regridding during the
  // execution, we'll coarsen the flagged cells (see coarsen).

  if (amr_doing.active())
    amr_doing << "SingleHydroMPM::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';

  // The simulation controller should not schedule it every time step
  Task* task = scinew Task("MPM::errorEstimate", this, &SingleHydroMPM::errorEstimate);

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
void SingleHydroMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                             SchedulerP& sched)
{
  scheduleErrorEstimate(coarseLevel, sched);
}

void SingleHydroMPM::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  if (d_switchCriteria) {
    d_switchCriteria->scheduleSwitchTest(level,sched);
  }
}

//______________________________________________________________________
//
void SingleHydroMPM::printParticleCount(const ProcessorGroup* pg,
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
  if(pcount == 0){
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
void SingleHydroMPM::computeAccStrainEnergy(const ProcessorGroup*,
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
void SingleHydroMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* ,
                                                DataWarehouse* new_dw)
{

  printTask(patches, patches->get(0) ,cout_doing,"MPM::countMaterialPointsPerLoadCurve");
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

          CCVariable<int> numPtsPerCell;
          new_dw->allocateAndPut(numPtsPerCell, Hlb->boundaryPointsPerCellLabel,
              dwi, patch);
          numPtsPerCell.initialize(0);

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<IntVector> pLoadCurveID;
          constParticleVariable<Point> px;
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
          new_dw->get(px, lb->pXLabel, pset);
          new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            for(int k = 0;k<3;k++){
              if (pLoadCurveID[idx](k) == (nofPressureBCs)){
                ++numPts;
                ++numPtsPerCell[patch->getCellIndex(px[idx])]; // 999 is code for hydrostatic
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
void SingleHydroMPM::initializePressureBC(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* ,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
  printTask(patches, patches->get(0),cout_doing,"Doing MPM::initializePressureBC");
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

      new_dw->get(px, lb->pXLabel, pset);
      new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
      new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);

      // Hydro-echanical coupling
      constCCVariable<int> numPtsPerCell;
      ParticleVariable<Vector> prescribedporepressure;
      if (flags->d_coupledflow) {
          new_dw->get(numPtsPerCell, Hlb->boundaryPointsPerCellLabel, dwi, patch, Ghost::None, 0);
          new_dw->getModifiable(prescribedporepressure, Hlb->pPrescribedPorePressureLabel, pset);
      }

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


        if (bcs_type == "Hydrostatic") {
            cout << "Boundary condition is hydrostatic." << endl;
            // Save the material points per load curve in the PressureBC object
            HydrostaticBC* pbc = dynamic_cast<HydrostaticBC*>(
                MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);

            double cellsurfacearea = pbc->getCellSurfaceArea(patch);
            Vector normal = pbc->getSurfaceNormal();
            ParticleSubset::iterator iter = pset->begin();
            for (; iter != pset->end(); iter++) {
                particleIndex idx = *iter;
                if (pLoadCurveID[idx] == 999) {
                    IntVector cellindex = patch->getCellIndex(px[idx]);
                    prescribedporepressure[idx] =
                        normal *
                        (cellsurfacearea *
                            pbc->getCellAveragePorePressure(cellindex, patch)) /
                        numPtsPerCell[cellindex];
                }  // if pLoadCurveID...
            }    // loop over particles
        } // if Hydrostatic loop

      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

void SingleHydroMPM::deleteGeometryObjects(const ProcessorGroup*,
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
}


void SingleHydroMPM::actuallyInitialize(const ProcessorGroup*,
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
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
    if(!flags->d_with_ice){
      ostringstream msg;
      msg << "\n ERROR: When using <interpolator>linear</interpolator> \n"
          << " you should also use <extraCells>[0,0,0]</extraCells> \n"
          << " unless you are running an MPMICE or MPMARCHES case.\n";
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

  // The call below is necessary because the GeometryPieceFactory holds on to a pointer
  // to all geom_pieces (so that it can look them up by name during initialization)
  // The pieces are never actually deleted until the factory is destroyed at the end
  // of the program. resetFactory() will rid of the pointer (lookup table) and
  // allow the deletion of the unneeded pieces.  
  
  GeometryPieceFactory::resetFactory();
 
}


void SingleHydroMPM::readPrescribedDeformations(string filename)
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

void SingleHydroMPM::readInsertParticlesFile(string filename)
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

void SingleHydroMPM::interpolateParticlesToGrid(const ProcessorGroup*,
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

    NCVariable<Vector> gfluidvelglobal;
    NCVariable<double> gfluidmassglobal;
    // TODO: Global fluidmass and velocity not needed, I think. Should be
   // confirmed.
    if (flags->d_coupledflow) {
        new_dw->allocateAndPut(gfluidmassglobal, Hlb->gFluidMassLabel,
            globMatID, patch);
        new_dw->allocateAndPut(gfluidvelglobal, Hlb->gFluidVelocityLabel,
            globMatID, patch);
        gfluidvelglobal.initialize(Vector(0.0));
        gfluidmassglobal.initialize(0.0);
    }

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
      //old_dw->get(pmass,          lb->pMassLabel,          pset);
//    old_dw->get(pColor,         lb->pColorLabel,         pset);
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

      constParticleVariable<double> pfluidmass, ptotalmass, pporepressure;
      //constParticleVariable<Vector> prescribedporepressure;
      constParticleVariable<Vector> pfluidvelocity;
     //if (hasFluid(m)) {
          old_dw->get(pfluidmass, Hlb->pFluidMassLabel, pset);
          //cout << "Can I haz fluidvelocity plz?" << endl;
          old_dw->get(pfluidvelocity, Hlb->pFluidVelocityLabel, pset);
          //cout << "Kthxbye" << endl;
          old_dw->get(pmass, Hlb->pSolidMassLabel, pset);
          old_dw->get(ptotalmass, lb->pMassLabel, pset);
          old_dw->get(pporepressure, Hlb->pPorePressureLabel, pset);
         // new_dw->get(prescribedporepressure, Hlb->pPrescribedPorePressureLabel,
          //    pset);
     // }
     // else {
      //    old_dw->get(pmass, lb->pMassLabel, pset);
      //}

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

      // JBH -- Scalar diffusion related
      NCVariable<double>  gConcentration, gConcentrationNoBC;
      NCVariable<double>  gHydrostaticStress, gExtScalarFlux;
      if (flags->d_doScalarDiffusion) {
        new_dw->allocateAndPut(gConcentration, lb->diffusion->gConcentration,
                                               dwi, patch);
        new_dw->allocateAndPut(gConcentrationNoBC, lb->diffusion->gConcentrationNoBC,
                                                   dwi, patch);
        new_dw->allocateAndPut(gHydrostaticStress, lb->diffusion->gHydrostaticStress,
                                                   dwi, patch);
        new_dw->allocateAndPut(gExtScalarFlux, lb->diffusion->gExternalScalarFlux,
                                               dwi, patch);
        gConcentration.initialize(0);
        gConcentrationNoBC.initialize(0);
        gHydrostaticStress.initialize(0);
        gExtScalarFlux.initialize(0);
      }

      NCVariable<double> gfluidmass, gfluidmassbar;
      NCVariable<Vector> gfluidvelocity;
      NCVariable<Vector> gexternalfluidforce;
      if (flags->d_coupledflow) {
          new_dw->allocateAndPut(gfluidmass, Hlb->gFluidMassLabel, dwi, patch);
          new_dw->allocateAndPut(gfluidmassbar, Hlb->gFluidMassBarLabel, dwi,
              patch);
          new_dw->allocateAndPut(gfluidvelocity, Hlb->gFluidVelocityLabel, dwi,
              patch);
          new_dw->allocateAndPut(gexternalfluidforce,
              Hlb->gExternalFluidForceLabel, dwi, patch);
          gfluidvelocity.initialize(Vector(0.0, 0.0, 0.0));
          gfluidmass.initialize(d_SMALL_NUM_MPM);
          gfluidmassbar.initialize(d_SMALL_NUM_MPM);
          gexternalfluidforce.initialize(Vector(0., 0., 0.));
      }

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

        Vector pfluidmom(0.0);
        if (flags->d_coupledflow && !mpm_matl->getIsRigid()) {
            double n = mpm_matl->getPorosity();
            pmom = (1.0 - n) * pmom;
            pfluidmom = n * pfluidmass[idx] * pfluidvelocity[idx];
        }

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

            if (flags->d_coupledflow &&
                !mpm_matl
                ->getIsRigid()) {  // Need some additional mass matrices
                double n = mpm_matl->getPorosity();
                gmass[node] -=
                    n * pmass[idx] * S[k];  // M_s = sum_p (1-n)m_s,p*S,p
                gfluidmass[node] += pfluidmass[idx] * S[k];
                gfluidmassbar[node] += n * pfluidmass[idx] * S[k];
                gfluidvelocity[node] += pfluidmom * S[k];
            }
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
              gHydrostaticStress[node]  += pHydroStress             * massWeight;
              gConcentration[node]      += pConc_Ext                * massWeight;
              gExtScalarFlux[node]      += pExternalScalarFlux[idx] * massWeight;
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
        gtempglobal[c]    += gTemperature[c];
        gTemperature[c]   /= gmass[c];
//        gColor[c]         /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gSp_vol[c]        /= gmass[c];

        if (flags->d_coupledflow) {
            gfluidmassglobal[c] += gfluidmass[c];
            gfluidvelglobal[c] += gfluidvelocity[c];
            gfluidvelocity[c] /= gfluidmass[c];
        }
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

      if (flags->d_coupledflow)
      bc.setBoundaryCondition(patch, dwi, "Symmetric", gfluidvelocity, interp_type);

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

      if (flags->d_coupledflow) gfluidvelglobal[c] /= gfluidmassglobal[c];
    }
    delete interpolator;
    delete linear_interpolator;
  }  // End loop over patches
}

void SingleHydroMPM::computeStressTensor(const ProcessorGroup*,
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


void SingleHydroMPM::computeInternalForce(const ProcessorGroup*,
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

      NCVariable<Vector> internalfluidforce, internaldragforce;

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

      //if(flags->d_with_ice){
        //new_dw->get(p_pressure,lb->pPressureLabel, pset);
      //}
      //else {

     // if (hasFluid(m)) {
          old_dw->get(p_pressure, Hlb->pPorePressureLabel, pset);
          new_dw->allocateAndPut(internalfluidforce, Hlb->gInternalFluidForceLabel,
              dwi, patch);
          internalfluidforce.initialize(Vector(0., 0., 0.));
      //}
     // else {
      //    new_dw->allocateTemporary(p_pressure_create, pset);
      //    for (ParticleSubset::iterator it = pset->begin(); it != pset->end(); it++) {
      //        p_pressure_create[*it] = 0.0;
      //    }
      //    p_pressure = p_pressure_create; // reference created data
       // }
      //}



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
      Matrix3 fluidpress;

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
          
           //stresspress = pstress[idx];

          if (flags->d_coupledflow) fluidpress = Id * p_pressure[idx];

          for (int k = 0; k < NN; k++){
            if(patch->containsNode(ni[k])){
              Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                         d_S[k].z()*oodx[2]);
              internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
              gstress[ni[k]]       += stressvol * S[k];

              //if (hasFluid(m))
                  internalfluidforce[ni[k]] -= (div * fluidpress) * pvol[idx];
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

          if (flags->d_coupledflow) fluidpress = Id * p_pressure[idx];

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

             // if (hasFluid(m)) {
                  IFr = d_S[k].x() * oodx[0] * fluidpress(0, 0) +
                      d_S[k].y() * oodx[1] * fluidpress(0, 1) +
                      d_S[k].z() * fluidpress(2, 2);
                  IFz = d_S[k].x() * oodx[0] * fluidpress(0, 1) +
                      d_S[k].y() * oodx[1] * fluidpress(1, 1);
                  internalfluidforce[ni[k]] -= Vector(IFr, IFz, 0.0) * pvol[idx];
              //}
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
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce, interp_type);

     // if (hasFluid(m))
          bc.setBoundaryCondition(patch, dwi, "Symmetric", internalfluidforce, interp_type);
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

void SingleHydroMPM::computeFluidDragForce(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw) {
    for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        printTask(patches, patch, cout_doing, "Doing computeFluidDragForce");

        Matrix3 Id;
        Id.Identity();

        ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
        vector<IntVector> ni(interpolator->size());
        vector<double> S(interpolator->size());
        string interp_type = flags->d_interpolator_type;

        unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

        for (unsigned int m = 0; m < numMPMMatls; m++) {
           // if (!hasFluid(m)) continue;

            MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
            int dwi = mpm_matl->getDWIndex();

            // need the gravitational acceleration to be consistent with input
            // before adding body forces with acceleration, a quick peak
            // on the magnitude of the water density will give a clue of
            // the unit of gravitation. This assumes that the time is given in
            // seconds.
            double water_rho = mpm_matl->getWaterDensity();
            int factor = -log10(water_rho) / 3.0 + 1.0;
            double g = 10 * pow(10, factor);  // [m/s^2]

            // Create arrays for the particle position, volume
            // and the constitutive model
            constParticleVariable<Point> px;
            //constParticleVariable<Matrix3> psize;
            constParticleVariable<Matrix3> pcursize;
            constParticleVariable<Matrix3> pFOld;

            constParticleVariable<Vector> pvelocity, pfluidvelocity;
            constParticleVariable<double> pfluidmass;
            NCVariable<Vector> internaldragforce;
            NCVariable<double> littleq;
            constNCVariable<Vector> gvelocity, gfluidvelocity;

            ParticleSubset* pset = old_dw->getParticleSubset(
                dwi, patch, Ghost::AroundNodes, NGP, lb->pXLabel);

            // For finding shape functions
            old_dw->get(px, lb->pXLabel, pset);
            //old_dw->get(psize, lb->pSizeLabel, pset);
            new_dw->get(pcursize, lb->pCurSizeLabel, pset);
            old_dw->get(pFOld, lb->pDeformationMeasureLabel, pset);

            old_dw->get(pvelocity, lb->pVelocityLabel, pset);
            old_dw->get(pfluidvelocity, Hlb->pFluidVelocityLabel, pset);
            old_dw->get(pfluidmass, Hlb->pFluidMassLabel, pset);

            Ghost::GhostType gac = Ghost::AroundCells;
            new_dw->get(gvelocity, lb->gVelocityLabel, dwi, patch, gac, NGP);
            new_dw->get(gfluidvelocity, Hlb->gFluidVelocityLabel, dwi, patch, gac,
                NGP);

            new_dw->allocateAndPut(internaldragforce, Hlb->gInternalDragForceLabel,
                dwi, patch);
            new_dw->allocateTemporary(littleq, patch, gac, NGP);
            internaldragforce.initialize(Vector(0, 0, 0));
            littleq.initialize(0.0);

            double q = 0.0;
            double n = mpm_matl->getPorosity();
            double perm = mpm_matl->getPermeability();

            // for the non axisymmetric case:
            // EDIT: Should be valid for all cases
            // if(!flags->d_axisymmetric){
            for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();
                iter++) {
                particleIndex idx = *iter;

                // Get the node indices that surround the cell
                int NN = interpolator->findCellAndWeights(px[idx], ni, S, pcursize[idx]);

                q = n / perm * pfluidmass[idx] * g;
               
                for (int k = 0; k < NN; k++) {
                    if (patch->containsNode(ni[k])) {
                        // Make sure it's diagonal!
                        littleq[ni[k]] += q * S[k];
                    }
                }
            }  // end of particle loop
         
            for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
                IntVector c = *iter;
                internaldragforce[c] = littleq[c] * (gfluidvelocity[c] - gvelocity[c]);
            }  // end of nodal loop
            // }

            // for the axisymmetric case
            // if(flags->d_axisymmetric){
            // throw ProblemSetupException("Axisymmetry for coupled flow not
            // implemented yet", __FILE__, __LINE__);
            // }
            // save boundary forces before apply symmetry boundary condition
            // Boundary conditions of drag force would be 0 because v=w
            // Not sure how to implement this
            MPMBoundCond bc;
            bc.setBoundaryCondition(patch, dwi, "Symmetric", internaldragforce,
                interp_type);
        }
        delete interpolator;
    }
}

void SingleHydroMPM::computeAndIntegrateFluidAcceleration(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw) {
    for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        printTask(patches, patch, cout_doing,
            "Doing computeAndIntegrateFluidAcceleration");

        Ghost::GhostType gnone = Ghost::None;

        Vector gravity = flags->d_gravity;
        double damp_coef = flags->d_artificialDampCoeff;

        double water_damping_coef = flags->d_waterdampingCoeff;

        vector<string> v;

        unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

        for (unsigned int m = 0; m < numMPMMatls; m++) {
            MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
            int dwi = mpm_matl->getDWIndex();

            // If material is rigid, there is no fluid acceleration to integrate,
            // and we can get on with our lives.
            if (mpm_matl->getIsRigid()) continue;

            // Get required variables for this patchcomputeAnd
            delt_vartype delT;
            constNCVariable<Vector> externalfluidforce, internalfluidforce,
                internaldragforce, fluidvelocity;
            constNCVariable<double> fluidmass;
            NCVariable<Vector> fluidvelocity_star, fluidacceleration;

            old_dw->get(delT, lb->delTLabel, getLevel(patches));
            new_dw->get(externalfluidforce, Hlb->gExternalFluidForceLabel, dwi, patch,
                gnone, 0);
            new_dw->get(internalfluidforce, Hlb->gInternalFluidForceLabel, dwi, patch,
                gnone, 0);
            new_dw->get(internaldragforce, Hlb->gInternalDragForceLabel, dwi, patch,
                gnone, 0);
            new_dw->get(fluidmass, Hlb->gFluidMassLabel, dwi, patch, gnone, 0);
            new_dw->get(fluidvelocity, Hlb->gFluidVelocityLabel, dwi, patch, gnone, 0);

            new_dw->allocateAndPut(fluidvelocity_star, Hlb->gFluidVelocityStarLabel,
                dwi, patch);
            new_dw->allocateAndPut(fluidacceleration, Hlb->gFluidAccelerationLabel,
                dwi, patch);

            fluidacceleration.initialize(Vector(0., 0., 0.));

            for (NodeIterator iter = patch->getExtraNodeIterator(); !iter.done();
                iter++) {
                IntVector c = *iter;

                Vector fluiddamp(0., 0., 0.);
                Vector fluidacc(0., 0., 0.);
                Vector fluidvel = fluidvelocity[c];
                fluidvel.safe_normalize();
                if (fluidmass[c] > flags->d_min_mass_for_acceleration) {
                    // TODO: externalfluidforce
                    fluiddamp =
                        -water_damping_coef *
                        (internalfluidforce[c] + fluidmass[c] * gravity).length() *
                        fluidvel;

                    fluidacc = (-externalfluidforce[c] + internalfluidforce[c] -
                        internaldragforce[c]) /
                        fluidmass[c];
                    fluidacc -= damp_coef * fluidvelocity[c];
                }
                fluidacceleration[c] = fluidacc + gravity;
                fluidvelocity_star[c] = fluidvelocity[c] + fluidacceleration[c] * delT;

                //cerr << "internaldragforce[c] " << internaldragforce[c] << endl;

                // Do magic to contact velocity
                // Computing the contact velocity for fluid, the true fluid
                // velocity and acceleration is needed for the mixture
                // 1. Need the surface normals
                // 2. Need to compare velocities
                // 3. Update water velocity and acceleration
            }  // loop over nodes
        }    // matls
    }
}

void SingleHydroMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
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
    double damp_coef = flags->d_artificialDampCoeff;

    double water_damping_coef = flags->d_waterdampingCoeff;
    double solid_damping_coef = flags->d_soliddampingCoeff;

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

      // Hydro mechanical coupling
      constNCVariable<double> fluidmassbar, fluidmass;
      constNCVariable<Vector> fluidacceleration, fluidvelocity,
          internalfluidforce;
     //if (hasFluid(m)) {
          new_dw->get(internalfluidforce, Hlb->gInternalFluidForceLabel, dwi,
              patch, gnone, 0);
          new_dw->get(fluidmass, Hlb->gFluidMassLabel, dwi, patch, gnone, 0);
          new_dw->get(fluidmassbar, Hlb->gFluidMassBarLabel, dwi, patch, gnone, 0);
          new_dw->get(fluidvelocity, Hlb->gFluidVelocityLabel, dwi, patch, gnone,
              0);
          new_dw->get(fluidacceleration, Hlb->gFluidAccelerationLabel, dwi, patch,
              gnone, 0);

     // }
      // Create variables for the results
      NCVariable<Vector> velocity_star,acceleration;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      new_dw->allocateAndPut(acceleration,  lb->gAccelerationLabel, dwi, patch);

      acceleration.initialize(Vector(0.,0.,0.));
      
      for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
        IntVector c = *iter;

        //double totalmass = (hasFluid(m) ? (fluidmassbar[c] + mass[c]) : mass[c]);

        double totalmass =(fluidmassbar[c] + mass[c]);

        Vector acc(0.,0.,0.);

        Vector fluiddamp(0., 0., 0.);
        Vector soliddamp(0., 0., 0.);
        Vector fluidvel(0., 0., 0.);
        Vector solidvel(0., 0., 0.);
       // if (hasFluid(m)) {
            fluidvel = fluidvelocity[c];
            fluidvel.safe_normalize();
            solidvel = velocity[c];
            solidvel.safe_normalize();
       // }

        if (mass[c] > flags->d_min_mass_for_acceleration){
          //acc  = (internalforce[c] + externalforce[c])/mass[c];
          //acc -= damp_coef*velocity[c];

            acc = internalforce[c] + externalforce[c];

          //  if (hasFluid(m)) {
                // Experimental: local damping
                fluiddamp =
                    -water_damping_coef *
                    (internalfluidforce[c] + fluidmass[c] * gravity).length() *
                    fluidvel;
                soliddamp = -solid_damping_coef *
                    (acc + gravity * totalmass -
                        (internalfluidforce[c] + fluidmass[c] * gravity))
                    .length() *
                    solidvel;
                // In case of fluid flow and porous material, add contribution from
                // fluid
                acc -=
                    fluidmassbar[c] * fluidacceleration[c] + fluiddamp + soliddamp;
            //}
            acc /= mass[c];
            //if (hasFluid(m)) {
                acc -= damp_coef * velocity[c];
           // }

        }
        //acceleration[c]  = acc +  gravity;

        acceleration[c] = acc + gravity * totalmass / mass[c];
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
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
  }
}

void SingleHydroMPM::setGridBoundaryConditions(const ProcessorGroup*,
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
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<Vector> gvelocity;
      NCVariable<Vector> gwelocity_star, gfluidacceleration;
      constNCVariable<Vector> gwelocity;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi,patch,
                                                                 Ghost::None,0);

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

     // if (hasFluid(m)) {
          new_dw->getModifiable(gfluidacceleration, Hlb->gFluidAccelerationLabel,
              dwi, patch);
          new_dw->getModifiable(gwelocity_star, Hlb->gFluidVelocityStarLabel, dwi,
              patch);
          new_dw->get(gwelocity, Hlb->gFluidVelocityLabel, dwi, patch, Ghost::None,
              0);

          bc.setBoundaryCondition(patch, dwi, "Velocity", gwelocity_star, interp_type);
          bc.setBoundaryCondition(patch, dwi, "Symmetric", gwelocity_star, interp_type);
     // }
      // Now recompute acceleration as the difference between the velocity
      // interpolated to the grid (no bcs applied) and the new velocity_star
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();
                                                                iter++){
        IntVector c = *iter;
        gacceleration[c] = (gvelocity_star[c] - gvelocity[c])/delT;
      //  if (hasFluid(m))
            gfluidacceleration[c] = (gwelocity_star[c] - gwelocity[c]) / delT;
      }
    } // matl loop
  }  // patch loop
}

void SingleHydroMPM::applyExternalFluidLoads(const ProcessorGroup* ,
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
    cout_doing << "Current Time (applyExternalFluidLoads) = " << time << endl;

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
    printTask(patches, patch,cout_doing,"Doing SingleHydroMPM::applyExternalFluidLoads");

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

     // ParticleVariable<Vector> prescribedporepressure_new;
     // if (flags->d_coupledflow) {
      //    new_dw->allocateAndPut(prescribedporepressure_new,
      //        lb->pPrescribedPorePressureLabel, pset);
     // }

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

            //constParticleVariable<Vector> prescribedporepressure;
            //if (flags->d_coupledflow) {
            //    old_dw->get(prescribedporepressure,
            //        lb->pPrescribedPorePressureLabel, pset);
            //}

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

          // if (flags->d_coupledflow) {

           //    cerr << "flags->d_coupledflow" << flags->d_coupledflow << endl;
           //    prescribedporepressure_new[idx] = 0;
           //}

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

              // Should apply pore pressure here, temporary empty!



            } // loadCurveID >=0
           }  // loop over elements of the IntVector
          }
        } else {  // using load curves, but not pressure BCs
          // Set to zero
          for(ParticleSubset::iterator iter = pset->begin();
                                       iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalForce_new[idx] = Vector(0.,0.,0.);

           // if (flags->d_coupledflow) {
            //    prescribedporepressure_new[idx] = 0;
           // }
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

         // if (flags->d_coupledflow) {
         //     prescribedporepressure_new[idx] = 0;
         // }
        }
      }
    } // matl loop
  }  // patch loop
}

void SingleHydroMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
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

    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;

    double totalConc    =   0.0;
    double minPatchConc =  5e11;
    double maxPatchConc = -5e11;

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

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

      // Hydro mechanical coupling
      constParticleVariable<double> pfluidmass, psolidmass, porosity;
      ParticleVariable<double> pfluidmassNew, psolidmassNew, porosityNew;
      constParticleVariable<Vector> pfluidvelocity;
      ParticleVariable<Vector> pfluidvelNew;
      constNCVariable<Vector> gfluidacceleration;

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

      // Hydro mechanical coupling
      //if (hasFluid(m)) {
          old_dw->get(psolidmass, Hlb->pSolidMassLabel, pset);
          old_dw->get(pfluidmass, Hlb->pFluidMassLabel, pset);
          old_dw->get(pfluidvelocity, Hlb->pFluidVelocityLabel, pset);
          old_dw->get(porosity, Hlb->pPorosityLabel, pset);
          new_dw->get(gfluidacceleration, Hlb->gFluidAccelerationLabel, dwi, patch,
              Ghost::AroundCells, NGP);
          new_dw->allocateAndPut(pfluidmassNew, Hlb->pFluidMassLabel_preReloc, pset);
          new_dw->allocateAndPut(psolidmassNew, Hlb->pSolidMassLabel_preReloc, pset);
          new_dw->allocateAndPut(pfluidvelNew, Hlb->pFluidVelocityLabel_preReloc, pset);
          new_dw->allocateAndPut(porosityNew, Hlb->pPorosityLabel_preReloc, pset);
      //}

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
        sdmMaxEffectiveConc = sdm->getMaxConcentration() - sdm->getConcentrationTolerance();
        sdmMinEffectiveConc = sdm->getMinConcentration() + sdm->getConcentrationTolerance();

        old_dw->get(pConcentration,     lb->diffusion->pConcentration,    pset);
        new_dw->get(gConcentrationRate, lb->diffusion->gConcentrationRate,
                                          dwi,  patch, gac, NGP);

        new_dw->allocateAndPut(pConcentrationNew,
                                        lb->diffusion->pConcentration_preReloc, pset);
        new_dw->allocateAndPut(pConcPreviousNew,
                                        lb->diffusion->pConcPrevious_preReloc,  pset);
      }


      if(flags->d_XPIC2){
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

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelnew[idx].length2();
          CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
          totalMom   += pvelnew[idx]*pmass[idx];
          totalmass  += pmass[idx];
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
          Vector fluidacc(0.0, 0.0, 0.0);
          double fricTempRate = 0.0;
          double tempRate = 0.0;
          double concRate = 0.0;
          double burnFraction = 0.0;

          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel      += gvelocity_star[node]  * S[k];
            acc      += gacceleration[node]   * S[k];
            //if (hasFluid(m)) 
                fluidacc += gfluidacceleration[node] * S[k];

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

          //if (hasFluid(m)) {
              pfluidmassNew[idx] = pfluidmass[idx];
              psolidmassNew[idx] = psolidmass[idx];
              pfluidvelNew[idx] = pfluidvelocity[idx] + fluidacc * delT;
              porosityNew[idx] = porosity[idx];
          //}

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

          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelnew[idx].length2();
          CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
          totalMom   += pvelnew[idx]*pmass[idx];
          totalmass  += pmass[idx];
        }
      } // use XPIC(2) or not

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

void SingleHydroMPM::computeParticleGradients(const ProcessorGroup*,
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
      constParticleVariable<Matrix3> psize;
      constParticleVariable<double> pVolumeOld,pmass,pmassNew;
      constParticleVariable<int> pLocalized;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<double> pvolume,pTempNew;
      ParticleVariable<Matrix3> pFNew,pVelGrad;
      ParticleVariable<Vector> pTempGrad;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector>  gvelocity_star;
      constNCVariable<double>  gTempStar;

      // Hydro-mechanical coupling
      constParticleVariable<double> pporepressure;
      ParticleVariable<double> pporepressureNew;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);
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

      constNCVariable<Vector> gfluidvelocity_star;
      //if (hasFluid(m)) {
          old_dw->get(pporepressure, Hlb->pPorePressureLabel, pset);
          new_dw->allocateAndPut(pporepressureNew, Hlb->pPorePressureLabel_preReloc, pset);
          new_dw->get(gfluidvelocity_star, Hlb->gFluidVelocityStarLabel, dwi, patch, gac, NGP);
     // }          

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

      // critical density and volume
      double rho_0 = mpm_matl->getInitialDensity();
      double rho_critical_lowerbound = 0.9 * rho_0;
      double rho_critical_upperbound = 1.1 * rho_0;

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        int NN=flags->d_8or27;
        Matrix3 tensorL(0.0);
        Matrix3 tensorLw(0.0);

        if(!flags->d_axisymmetric){
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndShapeDerivatives(px[idx],ni,
                                                     d_S,psize[idx]);
         computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity_star,NN);

         //if (hasFluid(m))
         computeVelocityGradient(tensorLw, ni, d_S, oodx, gfluidvelocity_star, NN);

        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         NN =interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,
                                                   S,d_S,psize[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity_star,
                                                                   px[idx],NN);

         //if (hasFluid(m))
             computeAxiSymVelocityGradient(tensorLw, ni, d_S, S, oodx,
                 gfluidvelocity_star, px[idx], NN);
        }

        pVelGrad[idx]=tensorL;
        pTempGrad[idx] = Vector(0.0,0.0,0.0);

        //if (hasFluid(m)) {
            double n = mpm_matl->getPorosity();  // TODO: This is supposed to be a
                                                 // state variable
            double bulkWater =
                300.0e6;  // TODO: Can be an input parameter in stead
            pporepressureNew[idx] =
                pporepressure[idx] +
                (delT * bulkWater / n *
                    ((1.0 - n) * tensorL.Trace() + n * tensorLw.Trace()));
        //}

            /*
            // Drain boundary condition
            if (px[idx].y() > 0.98) {
                pporepressureNew[idx] = 0;
            }
            */

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

              // Analytical equation to update volume (Dunatunga & Kamrin 2018)
              //Matrix3 Amat = tensorL * delT;
              //double traceAmat = Amat.Trace();
              //double dJ = exp(traceAmat);
              //double pvolume_trial = pVolumeOld[idx] * dJ;
              double Jtest = pFNew[idx].Determinant();
              double JOldtest = pFOld[idx].Determinant();
              double pvolume_trial = pVolumeOld[idx] * (Jtest / JOldtest) * (pmassNew[idx] / pmass[idx]);
              double pvolume_critical_upperbound = pmass[idx] / rho_critical_lowerbound;
              double pvolume_critical_lowerbound = pmass[idx] / rho_critical_upperbound;
              //double rho_cur = rho_0 / J; //current density
              //partvoldef += pvolume[idx];

              if (flags->d_doCapDensity) {
                  if (pvolume_trial < pvolume_critical_upperbound && pvolume_trial > pvolume_critical_lowerbound) {
                      // Deformation gradient
                      //Matrix3 Finc = Amat.Exponential(abs(flags->d_min_subcycles_for_F));
                      //pFNew[idx] = Finc * pFOld[idx];
                      pvolume[idx] = pvolume_trial;
                      partvoldef += pvolume[idx];
                      //double J = pFNew[idx].Determinant();
                      //double JOld = pFOld[idx].Determinant();
                      //pvolume[idx] = pVolumeOld[idx] * (J / JOld);
                  }
                  else {
                      pvolume[idx] = pVolumeOld[idx];
                      pFNew[idx] = pFOld[idx];
                      partvoldef += pvolume[idx];
                  }
              }
              else {
                  pvolume[idx] = pvolume_trial;
                  partvoldef += pvolume[idx];
              }
        }
        else{
          Matrix3 Amat = tensorL*delT;
          Matrix3 Finc = Amat.Exponential(abs(flags->d_min_subcycles_for_F));
          pFNew[idx] = Finc*pFOld[idx];

          double J   =pFNew[idx].Determinant();
          double JOld=pFOld[idx].Determinant();
          pvolume[idx]=pVolumeOld[idx]*(J/JOld)*(pmassNew[idx]/pmass[idx]);
          partvoldef += pvolume[idx];
        }

        //double J   =pFNew[idx].Determinant();
        //double JOld=pFOld[idx].Determinant();
        //pvolume[idx]=pVolumeOld[idx]*(J/JOld)*(pmassNew[idx]/pmass[idx]);
        //partvoldef += pvolume[idx];
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

void SingleHydroMPM::finalParticleUpdate(const ProcessorGroup*,
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

void SingleHydroMPM::InterpolateParticleToGridFilter(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
    for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        printTask(patches, patch, cout_doing, "Doing InterpolateParticleToGridFilter");

        ParticleInterpolator* linear_interpolator = scinew LinearInterpolator(patch);
        vector<IntVector> ni(linear_interpolator->size());
        vector<double> S(linear_interpolator->size());

        unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

        constNCVariable<double>   gvolumeglobal;
        new_dw->get(gvolumeglobal, lb->gVolumeLabel,
            m_materialManager->getAllInOneMatls()->get(0), patch, Ghost::None, 0);

        NCVariable<double>       gPorePressureglobal;
        new_dw->allocateAndPut(gPorePressureglobal, Hlb->gPorePressureFilterLabel,
            m_materialManager->getAllInOneMatls()->get(0), patch);
        gPorePressureglobal.initialize(0.0);

        for (unsigned int m = 0; m < numMPMMatls; m++) {

            MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
            int dwi = mpm_matl->getDWIndex();
            ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                Ghost::AroundNodes, NGP, lb->pXLabel);

            constParticleVariable<Point>   px;
            constParticleVariable<double>  pvol;
            constParticleVariable<Matrix3> psize;
            constNCVariable<double>        gvolume;

            old_dw->get(px, lb->pXLabel, pset);
            old_dw->get(pvol, lb->pVolumeLabel, pset);
            new_dw->get(psize, lb->pCurSizeLabel, pset);
            new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);

            constParticleVariable<double> pPorePressure;
            NCVariable<double>		      gPorePressure;
            new_dw->get(pPorePressure, Hlb->pPorePressureLabel_preReloc, pset);
            new_dw->allocateAndPut(gPorePressure, Hlb->gPorePressureFilterLabel, dwi, patch);
            gPorePressure.initialize(0.0);
            double PoreVol = 0.0;

            for (ParticleSubset::iterator iter = pset->begin();
                iter != pset->end();
                iter++) {
                particleIndex idx = *iter;

                // Get the node indices that surround the cell
                int  NN = linear_interpolator->findCellAndWeights(px[idx], ni, S,
                    psize[idx]);

                //PoreVol = pPoreTensor[idx] * pvol[idx];
                PoreVol = pPorePressure[idx] * pvol[idx];

                for (int k = 0; k < NN; k++) {
                    if (patch->containsNode(ni[k])) {
                        gPorePressure[ni[k]] += PoreVol * S[k];
                    }
                }
            } // End particle loop

            for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
                IntVector c = *iter;

                // Pore Tensor
                gPorePressureglobal[c] += gPorePressure[c];
                gPorePressure[c] /= gvolume[c];
            }
            //}
        }

        for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
            IntVector c = *iter;
            gPorePressureglobal[c] /= gvolumeglobal[c];
        }
        delete linear_interpolator;
    }
}

void SingleHydroMPM::InterpolateGridToParticleFilter(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
    for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        printTask(patches, patch, cout_doing,
            "Doing InterpolateGridToParticleFilter");

        ParticleInterpolator* linear_interpolator = scinew LinearInterpolator(patch);
        vector<IntVector> ni(linear_interpolator->size());
        vector<double> S(linear_interpolator->size());

        unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

        for (unsigned int m = 0; m < numMPMMatls; m++) {
            MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
            int dwi = mpm_matl->getDWIndex();
            Ghost::GhostType  gac = Ghost::AroundCells;
            ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

            constParticleVariable<Point> px;
            constParticleVariable<Matrix3> psize;
            old_dw->get(px, lb->pXLabel, pset);
            new_dw->get(psize, lb->pCurSizeLabel, pset);

            constNCVariable<double> gPorePressure;
            ParticleVariable<double> pPorePressurenew;
            new_dw->get(gPorePressure, Hlb->gPorePressureFilterLabel, dwi, patch, gac, NGP);
            new_dw->allocateAndPut(pPorePressurenew, Hlb->pPorePressureFilterLabel_preReloc, pset);

            // Loop over particles
            for (ParticleSubset::iterator iter = pset->begin();
                iter != pset->end(); iter++) {
                particleIndex idx = *iter;

                // Get the node indices that surround the cell
                int NN = linear_interpolator->findCellAndWeights(px[idx], ni, S,
                    psize[idx]);

                double PorePressurenew = 0.0;

                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                    IntVector node = ni[k];
                    PorePressurenew += gPorePressure[node] * S[k];
                }

                pPorePressurenew[idx] = PorePressurenew;
            }

            //}
        }  // loop over materials
        delete linear_interpolator;
    }
}

void SingleHydroMPM::insertParticles(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
    for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        printTask(patches, patch, cout_doing, "Doing MPM::insertParticles");

        // Get the current simulation time
        simTime_vartype simTimeVar;
        old_dw->get(simTimeVar, lb->simulationTimeLabel);
        double time = simTimeVar;

        delt_vartype delT;
        old_dw->get(delT, lb->delTLabel, getLevel(patches));

        int index = -999;
        for (int i = 0; i < (int)d_IPTimes.size(); i++) {
            if (time + delT > d_IPTimes[i] && time <= d_IPTimes[i]) {
                index = i;
                if (index >= 0) {
                    unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");
                    for (unsigned int m = 0; m < numMPMMatls; m++) {
                        MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);
                        int dwi = mpm_matl->getDWIndex();
                        ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

                        // Get the arrays of particle values to be changed
                        ParticleVariable<Point> px;
                        ParticleVariable<Vector> pvelocity;
                        constParticleVariable<double> pcolor;

                        old_dw->get(pcolor, lb->pColorLabel, pset);
                        new_dw->getModifiable(px, lb->pXLabel_preReloc, pset);
                        new_dw->getModifiable(pvelocity, lb->pVelocityLabel_preReloc, pset);

                        // Loop over particles here
                        for (ParticleSubset::iterator iter = pset->begin();
                            iter != pset->end();   iter++) {
                            particleIndex idx = *iter;
                            if (pcolor[idx] == d_IPColor[index]) {
                                pvelocity[idx] = d_IPVelNew[index];
                                px[idx] = px[idx] + d_IPTranslate[index];
                            } // end if
                        }   // end for
                    }     // end for
                }       // end if
            }         // end if
        }           // end for
    }             // end for
}

void
SingleHydroMPM::setParticleDefault(ParticleVariable<double>& pvar,
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
SingleHydroMPM::setParticleDefault(ParticleVariable<Vector>& pvar,
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
SingleHydroMPM::setParticleDefault(ParticleVariable<Matrix3>& pvar,
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


void SingleHydroMPM::printParticleLabels(vector<const VarLabel*> labels,
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
SingleHydroMPM::initialErrorEstimate(const ProcessorGroup*,
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
SingleHydroMPM::errorEstimate(const ProcessorGroup* group,
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
SingleHydroMPM::refine(const ProcessorGroup*,
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

//______________________________________________________________________
//
double SingleHydroMPM::recomputeDelT( const double delT )
{
  return delT * 0.1;
}

// Check for rigid material and coupled flow
bool SingleHydroMPM::hasFluid(int m) {

    MPMMaterial* mpm_matl = (MPMMaterial*)m_materialManager->getMaterial("MPM", m);

    return (flags->d_coupledflow &&
        !mpm_matl->getIsRigid());
}
