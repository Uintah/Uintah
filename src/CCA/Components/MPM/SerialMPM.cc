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
#include <CCA/Components/MPM/SerialMPM.h>

#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Components/MPM/Core/TriangleLabel.h>
#include <CCA/Components/MPM/Core/TracerLabel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/Materials/Contact/ContactFactory.h>
#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>
#include <CCA/Components/MPM/Materials/Dissolution/DissolutionFactory.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/Tracer/TracerMaterial.h>
#include <CCA/Components/MPM/Tracer/Tracer.h>
#include <CCA/Components/MPM/LineSegment/LineSegmentMaterial.h>
#include <CCA/Components/MPM/LineSegment/LineSegment.h>
#include <CCA/Components/MPM/Triangle/Triangle.h>
#include <CCA/Components/MPM/Triangle/TriangleMaterial.h>
#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/BurialHistory.h>
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
#include <Core/Grid/cpdiInterpolator.h>
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
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Int130.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/DOUT.hpp>
#include <Core/OS/Dir.h>

#include <dirent.h>
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

SerialMPM::SerialMPM( const ProcessorGroup* myworld,
                      const MaterialManagerP materialManager) :
  MPMCommon( myworld, materialManager )
{
  flags               = scinew MPMFlags(myworld);
  burialHistory       = scinew BurialHistory(/*myworld*/);

  TriL = scinew TriangleLabel();
  TraL = scinew TracerLabel();

  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  contactModel        = 0;
  dissolutionModel    = 0;
  thermalContactModel = 0;
  heatConductionModel = 0;
  d_loadCurveIndex    = 0;
  d_switchCriteria    = 0;
  NGP     = 1;
  NGN     = 1;

  d_ndim = 0;
  activateReductionVariable( endSimulation_name, true);
  activateReductionVariable( recomputeTimeStep_name, true);
  activateReductionVariable(     abortTimeStep_name, true);
}

SerialMPM::~SerialMPM()
{
  delete flags;
  delete contactModel;
  delete dissolutionModel;
  delete thermalContactModel;
  delete heatConductionModel;
  delete burialHistory;

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

  if(d_switchCriteria) {
    delete d_switchCriteria;
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

  if(flags->d_useLoadCurves){
    int exists = burialHistory->populate(restart_mat_ps);
    if(exists == 0){
      burialHistory=nullptr;
    }
  } else {
      burialHistory=nullptr;
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

  dissolutionProblemSetup(restart_mat_ps, flags);

  materialProblemSetup(restart_mat_ps,flags, isRestart);

  cohesiveZoneProblemSetup(restart_mat_ps, flags);

  tracerProblemSetup(restart_mat_ps, flags);

  lineSegmentProblemSetup(restart_mat_ps, flags);

  triangleProblemSetup(restart_mat_ps, flags);

  dissolutionModel = 
                  DissolutionFactory::create(UintahParallelComponent::d_myworld,
                                     restart_mat_ps,m_materialManager,lb,flags);
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
  dissolutionModel->outputProblemSpec(mpm_ps);
  thermalContactModel->outputProblemSpec(mpm_ps);

  for (unsigned int i = 0; i < m_materialManager->getNumMatls( "CZ" );i++) {
    CZMaterial* mat = (CZMaterial*) m_materialManager->getMaterial( "CZ", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  for (unsigned int i = 0; i < m_materialManager->getNumMatls("Tracer");i++) {
    TracerMaterial* mat = (TracerMaterial *) 
                                   m_materialManager->getMaterial("Tracer", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  for(unsigned int i = 0;i< m_materialManager->getNumMatls("LineSegment");i++){
    LineSegmentMaterial* mat = (LineSegmentMaterial *) 
                              m_materialManager->getMaterial("LineSegment", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  for(unsigned int i = 0;i< m_materialManager->getNumMatls("Triangle");i++){
    TriangleMaterial* mat = (TriangleMaterial *) 
                              m_materialManager->getMaterial("Triangle", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }

  if(burialHistory != nullptr){
    burialHistory->outputProblemSpec(root_ps);
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
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pModalIDLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->pSurfLabel);
  t->computes(lb->delTLabel,level.get_rep());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);
  t->computes(lb->NC_CCweightLabel,zeroth_matl);
  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->DissolvedMassLabel);
  t->computes(lb->TotalMassLabel);
  t->computes(lb->InitialMassSVLabel);
  t->computes(lb->TotalSurfaceAreaLabel);

  // Debugging Scalar
  if (flags->d_with_color) {
    t->computes(lb->pColorLabel);
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
    t->computes(lb->pLoadCurveIDLabel);
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

  int numCZM = m_materialManager->getNumMatls( "CZ" );
  if(numCZM>0){
    CZMaterial* cz_matl = (CZMaterial*) m_materialManager->getMaterial("CZ", 0);
    CohesiveZone* cz = cz_matl->getCohesiveZone();
    cz->scheduleInitialize(level, sched, m_materialManager);

    schedulePrintCZCount(level, sched);
  }

  int numTracerM = m_materialManager->getNumMatls("Tracer");
  if(numTracerM>0){
   TracerMaterial* tracer_matl = (TracerMaterial *)
                                   m_materialManager->getMaterial("Tracer", 0);
   Tracer* tr = tracer_matl->getTracer();
   tr->scheduleInitialize(level, sched, m_materialManager);

   schedulePrintTracerCount(level, sched);
  }


  int numLineSegmentM = m_materialManager->getNumMatls("LineSegment");
  if(numLineSegmentM>0){
   LineSegmentMaterial* ls_matl = (LineSegmentMaterial *) 
                              m_materialManager->getMaterial("LineSegment", 0);
   LineSegment* ls = ls_matl->getLineSegment();
   ls->scheduleInitialize(level, sched, m_materialManager);
  }

  int numTriangleM = m_materialManager->getNumMatls("Triangle");
  if(numTriangleM>0){
   TriangleMaterial* ls_matl = (TriangleMaterial *) 
                              m_materialManager->getMaterial("Triangle", 0);
   Triangle* ls = ls_matl->getTriangle();
   ls->scheduleInitialize(level, sched, m_materialManager);

   schedulePrintTriangleCount(level, sched);
  }

  if (flags->d_deleteGeometryObjects) {
    scheduleDeleteGeometryObjects(level, sched);
  }
}
//______________________________________________________________________
//
void SerialMPM::scheduleRestartInitialize(const LevelP& level,
                                          SchedulerP& sched)
{
}
/* _____________________________________________________________________
 Purpose:   Set variables that are normally set during the initialization
            phase, but get wiped clean when you restart
_____________________________________________________________________*/
void SerialMPM::restartInitialize()
{
  proc0cout<<"Doing restartInitialize\t\t\t\t\t MPM"<<endl;

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
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "MPM" ));
}

//______________________________________________________________________
void SerialMPM::schedulePrintTriangleCount(const LevelP& level,
                                           SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printTriangleCount",
                        this, &SerialMPM::printTriangleCount);
  t->requires(Task::NewDW, TriL->triangleCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "Triangle" ));
}

//______________________________________________________________________
void SerialMPM::printTriangleCount(const ProcessorGroup* pg,
                                   const PatchSubset*,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  sumlong_vartype trcount;
  new_dw->get(trcount, TriL->triangleCountLabel);

  if(pg->myRank() == 0){
   std::cout << "Created " << (long) trcount << " total triangles" << std::endl;
  }

  //__________________________________
  //  bulletproofing
  if(trcount == 0){
    ostringstream msg;
    msg << "\n ERROR: zero triangles were created.";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
}

//______________________________________________________________________
void SerialMPM::schedulePrintTracerCount(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printTracerCount",
                        this, &SerialMPM::printTracerCount);
  t->requires(Task::NewDW, TraL->tracerCountLabel);
  t->setType(Task::OncePerProc);
  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                 m_materialManager->allMaterials( "Tracer" ));
}
//______________________________________________________________________
//
void SerialMPM::printTracerCount(const ProcessorGroup* pg,
                                 const PatchSubset*, 
                                 const MaterialSubset*, 
                                 DataWarehouse*, 
                                 DataWarehouse* new_dw)
{
  sumlong_vartype trcount;
  new_dw->get(trcount, TraL->tracerCountLabel);

  if(pg->myRank() == 0){
   std::cout << "Created " << (long) trcount << " total tracers" << std::endl;
  }

  //__________________________________
  //  bulletproofing
  if(trcount == 0){
    ostringstream msg;
    msg << "\n ERROR: zero tracers were created.";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
}

//______________________________________________________________________
void SerialMPM::schedulePrintCZCount(const LevelP& level,
                                         SchedulerP& sched)
{
  Task* t = scinew Task("MPM::printCZCount",
                        this, &SerialMPM::printCZCount);
  t->requires(Task::NewDW, lb->czCountLabel);
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
  new_dw->get(trcount, lb->czCountLabel);

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
  const MaterialSet* matls        = m_materialManager->allMaterials( "MPM" );
  const MaterialSet* cz_matls     = m_materialManager->allMaterials( "CZ" );
  const MaterialSet* tracer_matls = m_materialManager->allMaterials("Tracer");
  const MaterialSet* lineseg_matls= m_materialManager->allMaterials("LineSegment");
  const MaterialSet* triangle_matls=m_materialManager->allMaterials("Triangle");
  const MaterialSet* all_matls    = m_materialManager->allMaterials();

  const MaterialSubset* mpm_matls_sub    = 
                            (   matls ?            matls->getUnion() : nullptr);
  const MaterialSubset* cz_matls_sub     = 
                            (cz_matls ?         cz_matls->getUnion() : nullptr);
  const MaterialSubset* tracer_matls_sub = 
                            (tracer_matls ? tracer_matls->getUnion() : nullptr);
  const MaterialSubset* lineseg_matls_sub = 
                          (lineseg_matls ? lineseg_matls->getUnion() : nullptr);
  const MaterialSubset* triangle_matls_sub = 
                        (triangle_matls ? triangle_matls->getUnion() : nullptr);

  if (flags->d_useLoadCurves){
    scheduleModifyLoadCurves(             level, sched,   matls);
  }
  scheduleComputeCurrentParticleSize(     sched, patches, matls);
  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleFindSurfaceParticles(           sched, patches, matls);
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  if(flags->d_useLineSegments){
    scheduleComputeLineSegmentForces(     sched, patches, mpm_matls_sub,
                                                          lineseg_matls_sub,
                                                          all_matls);
  }
  if(flags->d_useTriangles){
    scheduleComputeTriangleForces(        sched, patches, mpm_matls_sub,
                                                          triangle_matls_sub,
                                                          all_matls);
  }
//  scheduleFindGrainCollisions(          sched, patches, matls);
  if(flags->d_computeNormals){
    if(flags->d_useTriangles){
      scheduleComputeNormalsTri(          sched, patches, mpm_matls_sub,
                                                          triangle_matls_sub,
                                                          all_matls);
    } else {
      scheduleComputeNormals(             sched, patches, matls);
    }
  }
  if(flags->d_useTracers && flags->d_doingDissolution) {
    scheduleComputeGridCemVec(                sched, patches, mpm_matls_sub,
                                                          tracer_matls_sub,
                                                          all_matls);
  }
  if(flags->d_useLogisticRegression){
    scheduleComputeLogisticRegression(    sched, patches, matls);
  }
  scheduleExMomInterpolated(              sched, patches, matls);
  if(flags->d_useCohesiveZones){
    scheduleUpdateCohesiveZones(          sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
    scheduleAddCohesiveZoneForces(        sched, patches, mpm_matls_sub,
                                                          cz_matls_sub,
                                                          all_matls);
  }
  if(d_bndy_traction_faces.size()>0) {
    scheduleComputeContactArea(           sched, patches, matls);
  }
  scheduleComputeInternalForce(           sched, patches, matls);

  scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
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
  scheduleComputeMassBurnFrac(            sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleComputeParticleGradients(       sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  if(flags->d_useTracers){
    scheduleUpdateTracers(                sched, patches, mpm_matls_sub,
                                                          tracer_matls_sub,
                                                          all_matls);
  }
  if(flags->d_useLineSegments){
    scheduleUpdateLineSegments(           sched, patches, mpm_matls_sub,
                                                          lineseg_matls_sub,
                                                          all_matls);
  }
  if(flags->d_useTriangles){
    scheduleUpdateTriangles(              sched, patches, mpm_matls_sub,
                                                          triangle_matls_sub,
                                                          all_matls);
  }
  if(flags->d_computeScaleFactor){
    scheduleComputeParticleScaleFactor(   sched, patches, matls);
    if(flags->d_useLineSegments){
      scheduleComputeLineSegScaleFactor(  sched, patches, lineseg_matls);
    }
    if(flags->d_useTriangles){
      scheduleComputeTriangleScaleFactor( sched, patches, triangle_matls);
    }
  }
  scheduleChangeGrainMaterials(           sched, patches, matls);
  scheduleFinalParticleUpdate(            sched, patches, matls);
  scheduleInsertParticles(                sched, patches, matls);
  if(flags->d_canAddParticles){
    if(flags->d_useTracers){
      scheduleAddTracers(                 sched, patches, tracer_matls);
    }
    scheduleAddParticles(                 sched, patches, matls);
  }

//scheduleManageChangeGrainMaterials(     level, sched);
  scheduleManageDoAuthigenesis(           level, sched);

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

 if(flags->d_useCohesiveZones){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_cohesiveZoneState_preReloc,
                                    lb->pXLabel,
                                    d_cohesiveZoneState,
                                    lb->czIDLabel, cz_matls, 2);
 }

 if(flags->d_useTracers){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_tracerState_preReloc,
                                    lb->pXLabel,
                                    d_tracerState,
                                    TraL->tracerIDLabel, tracer_matls, 3);
 }

 if(flags->d_useLineSegments){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_linesegState_preReloc,
                                    lb->pXLabel,
                                    d_linesegState,
                                    lb->linesegIDLabel, lineseg_matls, 4);
 }

 if(flags->d_useTriangles){
  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                    d_triangleState_preReloc,
                                    lb->pXLabel,
                                    d_triangleState,
                                    TriL->triangleIDLabel, triangle_matls, 5);
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

void SerialMPM::scheduleModifyLoadCurves(const LevelP & level,
                                         SchedulerP& sched,
                                         const MaterialSet* matls)
{
  Task* t=scinew Task("MPM::modifyLoadCurves",
                    this, &SerialMPM::modifyLoadCurves);

  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->delTLabel );
  if(flags->d_useTimeAveragedKE){
    t->requires(Task::OldDW, lb->TimeAveSpecificKELabel );
  }
  t->requires(Task::OldDW, lb->KineticEnergyLabel );
  t->computes( VarLabel::find( endSimulation_name));
  t->setType(Task::OncePerProc);

  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level),
                                                                      matls);
}

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
  t->requires(Task::OldDW, lb->delTLabel);

  if (!flags->d_mms_type.empty()) {
    //MMS problems need displacements
    t->requires(Task::OldDW, lb->pDispLabel,            Ghost::None);
  }

  if (flags->d_useLoadCurves || flags->d_useCBDI) {
    t->requires(Task::OldDW,    lb->pXLabel,                  Ghost::None);
    t->requires(Task::OldDW,    lb->pLoadCurveIDLabel,        Ghost::None);
    t->computes(                lb->pLoadCurveIDLabel_preReloc);
    t->computes( VarLabel::find( endSimulation_name));
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
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pColorLabel,            gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  if (flags->d_GEVelProj) {
    t->requires(Task::OldDW, lb->pVelGradLabel,             gan,NGP);
    t->requires(Task::OldDW, lb->pTemperatureGradientLabel, gan,NGP);
  }
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::NewDW, lb->pCurSizeLabel,          gan,NGP);
  t->requires(Task::NewDW, lb->pSurfLabel_preReloc,    gan,NGP);
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
  t->computes(lb->gColorLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gTemperatureRateLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->massBurnFractionLabel);
  t->computes(lb->dLdtDissolutionLabel);
  t->computes(lb->NodalWeightSumLabel);

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
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::OldDW, lb->pXLabel,                     cz_matls,  gan, 1);
  t->requires(Task::NewDW, lb->czForceLabel_preReloc,       cz_matls,  gan, 1);
  t->requires(Task::NewDW, lb->czTopMatLabel_preReloc,      cz_matls,  gan, 1);
  t->requires(Task::NewDW, lb->czBotMatLabel_preReloc,      cz_matls,  gan, 1);
  t->requires(Task::NewDW, lb->gMassLabel,                  mpm_matls, gac, 2);

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
  t->requires(Task::NewDW,lb->gVolumeLabel,               gnone);
  t->requires(Task::NewDW,lb->gVolumeLabel, 
              m_materialManager->getAllInOneMatls(), Task::OutOfDomain, gnone);
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

  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::OldDW, lb->simulationTimeLabel);

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);
  t->computes( VarLabel::find(abortTimeStep_name) );
  t->computes( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(t, patches, matls);
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
  t->requires(Task::OldDW, lb->timeStepLabel);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  if(flags->d_computeNormals){
    t->requires(Task::NewDW, lb->gSurfNormLabel,                gac,NGN);
  }
  if(flags->d_XPIC2){
    t->requires(Task::NewDW, lb->gVelSPSSPLabel,                gac,NGN);
    t->requires(Task::NewDW, lb->pVelocitySSPlusLabel,          gnone);
  }
  t->requires(Task::OldDW, lb->pXLabel,                         gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                      gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);
  t->requires(Task::OldDW, lb->pModalIDLabel,                   gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,               gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                  gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                      gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                      gnone);
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);
  t->requires(Task::NewDW, lb->pSurfLabel_preReloc,             gnone);

  t->requires(Task::NewDW, lb->massBurnFractionLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->NodalWeightSumLabel,             gac,NGN);
  if(flags->d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,                  gac,NGN);
  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pModalIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);

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
    t->requires(Task::OldDW, lb->TotalMassLabel,    Ghost::None);
    t->requires(Task::OldDW, lb->InitialMassSVLabel,Ghost::None);
    t->computes(lb->TotalMassLabel);
    t->computes(lb->InitialMassSVLabel);
    t->computes(lb->DissolvedMassLabel);
    t->computes(lb->PistonMassLabel);
  }
  if(flags->d_reductionVars->volDeformed){
    t->computes(lb->TotalVolumeDeformedLabel);
  }

  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  // Carry Forward particle refinement flag
//  if(flags->d_refineParticles){
//    t->requires(Task::OldDW, lb->pRefinedLabel,                Ghost::None);
//    t->computes(             lb->pRefinedLabel_preReloc);
//  }

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
  t->requires(Task::OldDW, lb->pVolumeLabel,                    gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                gnone);

  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pTemperatureGradientLabel_preReloc);

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
  t->requires(Task::NewDW, lb->pVolumeLabel_preReloc,           gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleUpdateTracers(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSubset* mpm_matls,
                                      const MaterialSubset* tracer_matls,
                                      const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleUpdateTracers");

  Task* t=scinew Task("MPM::updateTracers",
                      this, &SerialMPM::updateTracers);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityStarLabel,   mpm_matls,      gac,NGN+1);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,      gac,NGN+1);
  t->requires(Task::NewDW, lb->dLdtDissolutionLabel, mpm_matls,      gac,NGN+1);
  t->requires(Task::NewDW, lb->gMassLabel,
             m_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  t->requires(Task::NewDW, lb->gVelocityLabel,
             m_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  if (flags->d_doingDissolution) {
    t->requires(Task::NewDW, lb->gSurfNormLabel,     mpm_matls,      gac,NGN+1);
  }

  t->requires(Task::OldDW, lb->pXLabel,            tracer_matls, gnone);
  t->requires(Task::OldDW, TraL->tracerIDLabel,    tracer_matls, gnone);
  t->requires(Task::OldDW, TraL->tracerCemVecLabel,tracer_matls, gnone);

  t->computes(lb->pXLabel_preReloc,            tracer_matls);
  t->computes(TraL->tracerIDLabel_preReloc,    tracer_matls);
  t->computes(TraL->tracerCemVecLabel_preReloc,tracer_matls);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleUpdateLineSegments(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* mpm_matls,
                                           const MaterialSubset* lineseg_matls,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleUpdateLineSegments");

  Task* t=scinew Task("MPM::updateLineSegments",
                      this, &SerialMPM::updateLineSegments);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityStarLabel,   mpm_matls,     gac,NGN+1);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,     gac,NGN+1);
  t->requires(Task::OldDW, lb->pXLabel,              lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,           lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->linesegIDLabel,       lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->lsMidToEndVectorLabel,lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,
                                                     lineseg_matls, gnone);
  t->requires(Task::NewDW, lb->dLdtDissolutionLabel, mpm_matls,     gac,NGN+1);
  if (flags->d_doingDissolution) {
    t->requires(Task::NewDW, lb->gSurfNormLabel,     mpm_matls,     gac,NGN+1);
  }

  t->computes(lb->pXLabel_preReloc,                      lineseg_matls);
  t->computes(lb->pSizeLabel_preReloc,                   lineseg_matls);
  t->computes(lb->linesegIDLabel_preReloc,               lineseg_matls);
  t->computes(lb->lsMidToEndVectorLabel_preReloc,        lineseg_matls);
  t->computes(lb->pDeformationMeasureLabel_preReloc,     lineseg_matls);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleUpdateTriangles(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSubset* mpm_matls,
                                        const MaterialSubset* triangle_matls,
                                        const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleUpdateTriangles");

  Task* t=scinew Task("MPM::updateTriangles",
                      this, &SerialMPM::updateTriangles);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityStarLabel,   mpm_matls,     gac,NGN+1);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,     gac,NGN+1);
  t->requires(Task::NewDW, lb->dLdtDissolutionLabel, mpm_matls,     gac,NGN+1);
  if (flags->d_doingDissolution) {
    t->requires(Task::NewDW, lb->gSurfNormLabel,     mpm_matls,     gac,NGN+1);
  }
  t->requires(Task::NewDW, lb->gMassLabel,
             m_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  t->requires(Task::NewDW, lb->gVelocityLabel,
             m_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  t->requires(Task::OldDW, lb->pXLabel,                  triangle_matls, gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,               triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triangleIDLabel,        triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN0VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN1VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN2VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,
                                                         triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triUseInPenaltyLabel,   triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triAreaLabel,           triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triAreaAtNodesLabel,    triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triClayLabel,           triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMassDispLabel,       triangle_matls, gnone);

  t->computes(lb->pXLabel_preReloc,                      triangle_matls);
  t->computes(lb->pSizeLabel_preReloc,                   triangle_matls);
  t->computes(TriL->triangleIDLabel_preReloc,            triangle_matls);
  t->computes(TriL->triMidToN0VectorLabel_preReloc,      triangle_matls);
  t->computes(TriL->triMidToN1VectorLabel_preReloc,      triangle_matls);
  t->computes(TriL->triMidToN2VectorLabel_preReloc,      triangle_matls);
  t->computes(lb->pDeformationMeasureLabel_preReloc,     triangle_matls);
  t->computes(TriL->triUseInPenaltyLabel_preReloc,       triangle_matls);
  t->computes(TriL->triAreaLabel_preReloc,               triangle_matls);
  t->computes(TriL->triAreaAtNodesLabel_preReloc,        triangle_matls);
  t->computes(TriL->triClayLabel_preReloc,               triangle_matls);
  t->computes(TriL->triMassDispLabel_preReloc,           triangle_matls);
  t->computes(TriL->triNormalLabel_preReloc,             triangle_matls);
  t->computes(TriL->triMultiMatLabel_preReloc,           triangle_matls);
  t->computes(TriL->triNearbyMatsLabel_preReloc,         triangle_matls);
//  t->computes(TriL->triNearbyMatsN0Label_preReloc,       triangle_matls);
//  t->computes(TriL->triNearbyMatsN1Label_preReloc,       triangle_matls);
//  t->computes(TriL->triNearbyMatsN2Label_preReloc,       triangle_matls);

  // Reduction Variable
  t->computes(lb->TotalSurfaceAreaLabel);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeLineSegmentForces(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* lineseg_matls,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeLineSegmentForces");

  Task* t=scinew Task("MPM::computeLineSegmentForces",
                      this, &SerialMPM::computeLineSegmentForces);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->pXLabel,              lineseg_matls, gac, 2);
  t->requires(Task::OldDW, lb->pSizeLabel,           lineseg_matls, gac, 2);
  t->requires(Task::OldDW, lb->lsMidToEndVectorLabel,lineseg_matls, gac, 2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,     gac, NGN+2);

  t->computes(lb->gLSContactForceLabel,              mpm_matls);
  t->computes(lb->gSurfaceAreaLabel,                 mpm_matls);

  sched->addTask(t, patches, matls);
}

void SerialMPM::scheduleComputeTriangleForces(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* triangle_matls,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeTriangleForces");

  Task* t=scinew Task("MPM::computeTriangleForces",
                      this, &SerialMPM::computeTriangleForces);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->pXLabel,                triangle_matls, gac, 2);
  t->requires(Task::OldDW, lb->pSizeLabel,             triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN0VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN1VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN2VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triUseInPenaltyLabel, triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triangleIDLabel,      triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triAreaAtNodesLabel,  triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triClayLabel,         triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMultiMatLabel,     triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triNearbyMatsLabel,   triangle_matls, gac, 2);
  if (flags->d_doingDissolution) {
    t->requires(Task::OldDW, TriL->triMassDispLabel,   triangle_matls, gac, 2);
  }

  t->requires(Task::NewDW, lb->gMassLabel,             mpm_matls,   gac,NGN+3);

  t->computes(lb->gLSContactForceLabel,                mpm_matls);
  t->computes(lb->gInContactMatlLabel,                 mpm_matls);
  if (flags->d_doingDissolution) {
    t->computes(lb->gSurfaceAreaLabel,                 mpm_matls);
    t->computes(lb->gSurfaceClayLabel,                 mpm_matls);
  }
//  t->computes(TriL->triInContactLabel,                triangle_matls);

  sched->addTask(t, patches, matls);
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
  Ghost::GhostType  gan   = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pXLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pColorLabel,              gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan, NGP);
  t->modifies(lb->pParticleIDLabel_preReloc);
  t->modifies(lb->pModalIDLabel_preReloc);
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

void SerialMPM::scheduleAddTracers(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* tracer_matls)

{
  if( !flags->doMPMOnLevel( getLevel(patches)->getIndex(), getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleAddTracers" );

  Task * t = scinew Task("MPM::addTracers", this, &SerialMPM::addTracers );

  t->modifies(TraL->tracerIDLabel_preReloc, tracer_matls);
  t->modifies(lb->pXLabel_preReloc,         tracer_matls);

  sched->addTask(t, patches, tracer_matls);
}

void
SerialMPM::scheduleComputeParticleScaleFactor(       SchedulerP  & sched,
                                               const PatchSet    * patches,
                                               const MaterialSet * matls )
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleComputeParticleScaleFactor" );

  Task * t = scinew Task( "MPM::computeParticleScaleFactor",this, 
                          &SerialMPM::computeParticleScaleFactor);

  t->requires( Task::NewDW, lb->pSizeLabel_preReloc,              Ghost::None );
  t->requires( Task::NewDW, lb->pDeformationMeasureLabel_preReloc,Ghost::None );
  t->computes( lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void
SerialMPM::scheduleComputeLineSegScaleFactor(        SchedulerP  & sched,
                                               const PatchSet    * patches,
                                               const MaterialSet * matls )
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleComputeLineSegScaleFactor" );

  Task * t = scinew Task( "MPM::computeLineSegScaleFactor",this, 
                          &SerialMPM::computeLineSegScaleFactor);

  t->requires( Task::NewDW, lb->pSizeLabel_preReloc,              Ghost::None );
  t->computes( lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void
SerialMPM::scheduleComputeTriangleScaleFactor(       SchedulerP  & sched,
                                               const PatchSet    * patches,
                                               const MaterialSet * matls )
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing,"MPM::scheduleComputeTriangleScaleFactor");

  Task * t = scinew Task( "MPM::computeTriangleScaleFactor",this, 
                          &SerialMPM::computeTriangleScaleFactor);

  t->requires( Task::NewDW, lb->pSizeLabel_preReloc,              Ghost::None );
  t->requires( Task::NewDW, lb->pDeformationMeasureLabel_preReloc,Ghost::None );
  t->computes( lb->pScaleFactorLabel_preReloc );

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

void SerialMPM::scheduleComputeMassBurnFrac(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeMassBurnFrac");
  dissolutionModel->addComputesAndRequiresMassBurnFrac(sched, patches, matls);
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
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pCurSizeLabel);
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
          MPMMaterial* mpm_matl = 
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
  printTask(patches, patches->get(0),cout_doing,"Doing MPM::initializePressureBC");
  if (cout_dbg.active())
    cout_dbg << "Current Time (Initialize Pressure BC) = " << time << endl;


  // Calculate the force vector at each particle
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
          if(isProc0_macro){
            string udaDir = m_output->getOutputLocation();
            string filename=udaDir+"/TimePressure.dat";
            std::ofstream TP(filename.c_str(),ios::app);
            TP << "#BHIndex UintahTime Pressure GeologicTime GeologicTemp " << endl;
          }

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

void SerialMPM::deleteGeometryObjects(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* ,
                                      DataWarehouse* new_dw)
{
   printTask( cout_doing,"Doing MPM::deleteGeometryObjects");

   unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
   for(unsigned int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
     proc0cout << "MPM::Deleting Geometry Objects  matl: " 
               << mpm_matl->getDWIndex() << "\n";
     mpm_matl->deleteGeomObjects();
   }
}

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
  new_dw->put(sum_vartype(0.0), lb->KineticEnergyLabel);
  new_dw->put(sum_vartype(0.0), lb->DissolvedMassLabel);
  new_dw->put(sum_vartype(0.0), lb->TotalMassLabel);
  new_dw->put(sum_vartype(0.0), lb->TotalSurfaceAreaLabel);
  SoleVariable<double> IMSV = 0.0;
  new_dw->put(IMSV, lb->InitialMassSVLabel);

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);

  // The call below is necessary because the GeometryPieceFactory holds on to a pointer
  // to all geom_pieces (so that it can look them up by name during initialization)
  // The pieces are never actually deleted until the factory is destroyed at the end
  // of the program. resetFactory() will rid of the pointer (lookup table) and
  // allow the deletion of the unneeded pieces.  
  
  GeometryPieceFactory::resetFactory();
 
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
      constParticleVariable<double>  pSurf;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pColor,         lb->pColorLabel,         pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      if (flags->d_GEVelProj){
        old_dw->get(pVelGrad,     lb->pVelGradLabel,             pset);
        old_dw->get(pTempGrad,    lb->pTemperatureGradientLabel, pset);
      }
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      new_dw->get(psize,          lb->pCurSizeLabel,       pset);

      new_dw->get(pexternalforce, lb->pExtForceLabel_preReloc, pset);
      new_dw->get(pSurf,          lb->pSurfLabel_preReloc,     pset);

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
      NCVariable<double> gTemperatureRate,massBurnFrac,dLdt;
      NCVariable<double> nodalWeightSum;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
      new_dw->allocateAndPut(gColor,           lb->gColorLabel,      dwi,patch);
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
      new_dw->allocateAndPut(massBurnFrac,     lb->massBurnFractionLabel,
                             dwi,patch);
      new_dw->allocateAndPut(dLdt,             lb->dLdtDissolutionLabel,
                             dwi,patch);
      new_dw->allocateAndPut(nodalWeightSum,   lb->NodalWeightSumLabel,
                             dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gColor.initialize(0.0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gTemperatureRate.initialize(0);
      gexternalheatrate.initialize(0);
      gSp_vol.initialize(0.);
      massBurnFrac.initialize(0.);
      dLdt.initialize(0.);
      nodalWeightSum.initialize(0.0);

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
            gColor[node]         += pColor[idx]*pmass[idx]         * S[k];
            if (!flags->d_useCBDI) {
              gexternalforce[node] += pexternalforce[idx]          * S[k];
            }
            gTemperature[node]   += ptemp_ext * pmass[idx] * S[k];
            gSp_vol[node]        += pSp_vol   * pmass[idx] * S[k];
            //gexternalheatrate[node] += pexternalheatrate[idx]      * S[k];
            if(pSurf[idx] >= 0.99 && S[k]>1.e-20){
              nodalWeightSum[node]+=S[k];
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
        gColor[c]         /= gmass[c];
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

      new_dw->allocateAndPut(pvelSSPlus,lb->pVelocitySSPlusLabel,    pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity,    lb->gVelocityLabel,   dwi,patch,gac,NGP);

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
    }
    delete interpolator;
  }
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

void SerialMPM::addCohesiveZoneForces(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing MPM::addCohesiveZoneForces");

//    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    ParticleInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > gext_force(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    Vector dx = patch->dCell();
    double dxlength=dx.x();

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) m_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();

      new_dw->getModifiable(gext_force[m],lb->gExternalForceLabel,   dwi,patch);
      new_dw->get(gmass[m],               lb->gMassLabel,dwi, patch, gac,2);
    }

    unsigned int numCZMatls=m_materialManager->getNumMatls( "CZ" );
    for(unsigned int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl=(CZMaterial*) m_materialManager->getMaterial( "CZ",m);
      int dwi = cz_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, 1, lb->pXLabel);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      constParticleVariable<Vector> czforce;
      constParticleVariable<int> czTopMat, czBotMat;

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

        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          totMassTop += S[k]*gmass[TopMat][node];
          totMassBot += S[k]*gmass[BotMat][node];
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
            Vector CZFBot = czforce[idx]*S[k]*gmass[BotMat][node]/totMassBot;
            Vector CZFTop = czforce[idx]*S[k]*gmass[TopMat][node]/totMassTop;

	    if(CZFBot.length()/gmass[BotMat][node]*delT*delT < 0.5*dxlength){
              gext_force[BotMat][node] += CZFBot;
	    }
	    if(CZFTop.length()/gmass[TopMat][node]*delT*delT < 0.5*dxlength){
              gext_force[TopMat][node] -= CZFTop;
	    }

//            gext_force[BotMat][node] += czforce[idx]*S[k]*gmass[BotMat][node]
//                                                                 /totMassBot;
//            gext_force[TopMat][node] -= czforce[idx]*S[k]*gmass[TopMat][node]
//                                                                 /totMassTop;

//            gext_force[BotMat][node] += czforce[idx]*S[k]/sumSBot;
//            gext_force[TopMat][node] -= czforce[idx]*S[k]/sumSTop;

//            gext_force[BotMat][node] = gext_force[BotMat][node]
//                                     + czforce[idx] * S[k];
//            gext_force[TopMat][node] = gext_force[TopMat][node]
//                                     - czforce[idx] * S[k];
          }
        }
      }
    }
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
      MPMMaterial* mpm_matl = 
                  (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

  Ghost::GhostType  gnone = Ghost::None;
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
           m_materialManager->getAllInOneMatls()->get(0), patch, gnone,0);
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

      new_dw->get(gvolume,    lb->gVolumeLabel,         dwi, patch, gnone, 0);

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

     if(m!=((unsigned int) flags->d_KEMaterial)){

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
     }  // endif not piston material

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
  const Level* level = getLevel(patches);
  IntVector lowNode, highNode;
  level->findInteriorNodeIndexRange(lowNode, highNode);
  string interp_type = flags->d_interpolator_type;

  //double maxVelMag = 0.;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
                       "Doing MPM::computeAndIntegrateAcceleration");

    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;
#if 0
    simTime_vartype simTimeVar;
    old_dw->get(simTimeVar, lb->simulationTimeLabel);
    double time = simTimeVar;
    if(time<20.0){
      gravity = 0.5*flags->d_gravity*(time/20.);
    } else if(time<50.){
      gravity = 0.5*flags->d_gravity;
    } else if(time<70.){
      gravity = 0.5*flags->d_gravity + 0.5*gravity*((time-50.)/20.);
    } else if(time<100.){
      gravity = flags->d_gravity;
    } else if(time<120.){
      gravity = flags->d_gravity + 0.5*gravity*((time-100.)/20.);
    } else if(time<150.){
      gravity = 1.5*flags->d_gravity;
    } else if(time<170.){
      gravity = 1.5*flags->d_gravity + 0.5*gravity*((time-150.)/20.);
    } else if(time<200.){
      gravity = 2.0*flags->d_gravity;
    }
    proc0cout << "Gravity = " << gravity << endl;
#endif

    for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){
#if 0
      if(m==1){
        gravity*=-1;
      }
#endif
      MPMMaterial* mpm_matl = 
                 (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

      // Check the integrated nodal velocity and if the product of velocity
      // and timestep size is larger than half the cell size, restart the
      // timestep with 50% as large of a timestep (see recomputeDelT in this
      // file).
      if(flags->d_restartOnLargeNodalVelocity){
       Vector dx = patch->dCell();
       double cell_size_sq = dx.length2();
//       double cell_size = dx.length();
//       double cellVol = dx.x()*dx.y()*dx.z();
//       double rho = mpm_matl->getInitialDensity();
       for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        if(c.x()>=lowNode.x() && c.x()<highNode.x() &&
           c.y()>=lowNode.y() && c.y()<highNode.y() &&
           c.z()>=lowNode.z() && c.z()<highNode.z()){
           if((velocity_star[c]*delT).length2() > 0.25*cell_size_sq){
            cerr << "velocity_star[" << c << "] = " << velocity_star[c] << endl;
            cerr << "mass[" << c << "] = " << mass[c] << endl;
            cerr << "internalforce[" << c << "] = " << internalforce[c] << endl;
            cerr << "externalforce[" << c << "] = " << externalforce[c] << endl;
            cerr << "matl = " << m << endl;
              cerr << "Restarting timestep, velocity star too large" << endl;
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
//     if(m==1){  /* Keeping this for testing */
      MPMMaterial* mpm_matl = 
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
//     }
    }   // matl loop
  }     // patch loop
}

void SerialMPM::modifyLoadCurves(const ProcessorGroup* ,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  // Get the current time
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, lb->simulationTimeLabel);
  double time = simTimeVar;
  sum_vartype KE;
  max_vartype timeAveKE;
  delt_vartype delT;
  double KELoadCurve;
  old_dw->get(delT, lb->delTLabel, getLevel(patches) );
  old_dw->get(KE,   lb->KineticEnergyLabel);
  if(flags->d_useTimeAveragedKE){
   old_dw->get(timeAveKE,   lb->TimeAveSpecificKELabel);
   KELoadCurve=timeAveKE;
  } else {
   KELoadCurve=KE;
  }

  for(int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {

      // Save the material points per load curve in the PressureBC object
      PressureBC* pbc =
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      int numPOLC = pbc->getLoadCurve()->numberOfPointsOnLoadCurve();
      int nextIndex = pbc->getLoadCurve()->getNextIndex(time);
      double nextTime = pbc->getLoadCurve()->getTime(nextIndex);
      double timeToNextLoad = nextTime-time;
      if(timeToNextLoad < delT){
        proc0cout << "timeToNextLoad = " << timeToNextLoad << endl;
        proc0cout << "KE = " << KE << endl;
        if(flags->d_useTimeAveragedKE){
         proc0cout << "timeAverageKE = " << timeAveKE << endl;
        }
        proc0cout << "KELoadCurve = " << KELoadCurve << endl;
        proc0cout << "pbc->getLoadCurve()->getMaxKE(nextIndex) = "
                  <<  pbc->getLoadCurve()->getMaxKE(nextIndex) << endl;
        if(KELoadCurve > pbc->getLoadCurve()->getMaxKE(nextIndex)){
          for(int i=nextIndex;i<numPOLC;i++){
            double loadTime = pbc->getLoadCurve()->getTime(i);
            pbc->getLoadCurve()->setTime(i,loadTime+1.0);
          }
        } // if KE
        // This is just here so the user can confirm that this is working right
        for(int i=0; i<numPOLC; i++){
          double time = pbc->getLoadCurve()->getTime(i);
          double load = pbc->getLoadCurve()->getLoad(i);
          proc0cout << "time, load = " << time << " " << load << endl;
        } // Loop over points on load curve
      } //if(timeToNextLoad...
      if(nextIndex>=numPOLC){ // shut down the simulation
        new_dw->put(bool_or_vartype(true), VarLabel::find(endSimulation_name));
      } // endif
    } // if bcs_type == "Pressure"
  }   // Loop over physical BCs
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

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches) );

  const Level* level = getLevel(patches);
  const GridP grid = level->getGrid();

  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalLoads) = " << time << endl;

  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> forcePerPart;
  std::vector<PressureBC*> pbcP;
  int curLCIndex=0;
  int curBHIndex=0;
  double geoTime_MYa=0.;
  double geoTemp_K=0.;
  flags->d_currentPhase = "null";

  if (flags->d_useLoadCurves) {
    for (int ii = 0;ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {
        PressureBC* pbc =
          dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // get the load curve time (not the ID), use that to get the BH index
        curLCIndex = pbc->getLoadCurve()->getNextIndex(time)-1;
        int lastLCIndex = pbc->getLoadCurve()->getNextIndex(time-delT)-1;
        flags->d_currentPhase= pbc->getLoadCurve()->getPhase(curLCIndex);
        bool outputStep = false;
        if(lastLCIndex != curLCIndex){
          m_output->setOutputTimeStep(    true, grid );
          m_output->setCheckpointTimeStep(true, grid );
          outputStep = true;
        }
        if(burialHistory != nullptr){
          curBHIndex = pbc->getLoadCurve()->getBHIndex(curLCIndex);
//          cout << "curBHIndex = " << curBHIndex << endl;
          double uintahDisTime = 
                         burialHistory->getUintahDissolutionTime(curBHIndex);
          double geoInterval = burialHistory->getTime_Ma(curBHIndex) - 
                         burialHistory->getTime_Ma(curBHIndex - 1);
          double qtzGrowthVecF = 
                         burialHistory->getQuartzGrowthVec_fr(curBHIndex);
          // The following is to get an interpolated temperature out
          // of the burial history for use in the dissolution model
 
          if(flags->d_currentPhase=="hold" ||
             flags->d_currentPhase=="dissolution"){
            double holdStartTime = pbc->getLoadCurve()->getTime(curLCIndex);
            double startTemp = burialHistory->getTemperature_K(curBHIndex);
            double endTemp   = burialHistory->getTemperature_K(curBHIndex-1);
            geoTemp_K = startTemp + ((endTemp-startTemp)/uintahDisTime)
                                    *(time-holdStartTime);
          } else if(flags->d_currentPhase=="ramp") {
            geoTemp_K   = burialHistory->getTemperature_K(curBHIndex-1);
          } else if(flags->d_currentPhase=="settle"){
            geoTemp_K   = burialHistory->getTemperature_K(curBHIndex);
          }
          geoTime_MYa = burialHistory->getTime_Ma(curBHIndex);
          bool EOC    = burialHistory->getEndOnCompletion(curBHIndex);
          if(EOC && flags->d_currentPhase=="ramp" && outputStep){
            proc0cout << "Stopping per burial history specification" << endl;
            new_dw->put(bool_or_vartype(true), 
                        VarLabel::find(endSimulation_name));
          }
          if(curBHIndex==0){ // shut down the simulation
            proc0cout << "Reached the end of the burial history" << endl;
            new_dw->put(bool_or_vartype(true), 
                        VarLabel::find(endSimulation_name));
          } // endif
          burialHistory->setCurrentIndex(curBHIndex);
          burialHistory->setCurrentPhaseType(flags->d_currentPhase);

          // DISSOLUTION
          if (flags->d_doingDissolution) {
           dissolutionModel->setTemperature(geoTemp_K);
           dissolutionModel->setPhase(flags->d_currentPhase);
           dissolutionModel->setTimeConversionFactor(geoInterval/uintahDisTime);
           dissolutionModel->setGrowthFractionRate(qtzGrowthVecF/uintahDisTime);
          }
        }
        if(isProc0_macro){
          double curLoad = pbc->getLoadCurve()->getLoad(time);
          string udaDir = m_output->getOutputLocation();
          string filename=udaDir+"/TimePressure.dat";
          std::ofstream TP(filename.c_str(),ios::app);
          TP << curBHIndex << " "  << time        << " " << curLoad   << " " 
             << geoTime_MYa << " " << geoTemp_K   << endl;
        }

        // Calculate the force per particle at current time
        forcePerPart.push_back(pbc->forcePerParticle(time));
      }
    } // loop over BCs
  } // if use load curves

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing MPM::applyExternalLoads");

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

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, lb->timeStepLabel);
  int timestep = timeStep;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing MPM::interpolateToParticlesAndUpdate");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass  = 0;
    sum_vartype OTM;
    old_dw->get(OTM, lb->TotalMassLabel);
    double oldTotalMass   = OTM;
    SoleVariable< double > OIMSV;

    old_dw->get( OIMSV, lb->InitialMassSVLabel);

    if(timestep<=2){
     proc0cout << "timestep = " << timestep 
               << ", oldTotalMass = " << oldTotalMass << endl;
     OIMSV = oldTotalMass;
    }

    double dissolvedmass = 0;
    double pistonmass = 0;
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

    vector<double> useInKECalc(numMPMMatls);
    if(flags->d_KEMaterial >= (int) numMPMMatls){
      ostringstream warn;
      warn << "KEMaterial index (" << flags->d_KEMaterial 
           << ") is greater than number of MPM matls\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    if(flags->d_KEMaterial==-999){
      for(unsigned int m = 0; m < numMPMMatls; m++){
        useInKECalc[m]=1.0;
      }
    } else {
      for(unsigned int m = 0; m < numMPMMatls; m++){
        useInKECalc[m]=0.0;
      }
      useInKECalc[flags->d_KEMaterial]=1.0;
    }

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pvelocity, pvelSSPlus, pdisp;
      constParticleVariable<Matrix3> psize, pFOld, pcursize;
      constParticleVariable<double> pmass, pVolumeOld, pTemperature;
      constParticleVariable<long64> pids;
      constParticleVariable<int> pModID;
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> pvelnew, pdispnew;
      ParticleVariable<Matrix3> psizeNew;
      ParticleVariable<double> pmassNew,pTempNew;
      ParticleVariable<long64> pids_new;
      ParticleVariable<int> pModIDNew;
      constParticleVariable<double> pSurf;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration, gvelSPSSP;
      constNCVariable<double> gTemperatureRate, gTempStar;
      constNCVariable<double> dTdt, massBurnFrac, frictionTempRate;
      constNCVariable<Vector> gGrowthDir;
      constNCVariable<double> nodalWeightSum;

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
      new_dw->get(pSurf,        lb->pSurfLabel_preReloc,             pset);

      new_dw->allocateAndPut(pxnew,      lb->pXLabel_preReloc,            pset);
      new_dw->allocateAndPut(pvelnew,    lb->pVelocityLabel_preReloc,     pset);
      new_dw->allocateAndPut(pdispnew,   lb->pDispLabel_preReloc,         pset);
      new_dw->allocateAndPut(pmassNew,   lb->pMassLabel_preReloc,         pset);
      new_dw->allocateAndPut(pTempPreNew,lb->pTempPreviousLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,   lb->pTemperatureLabel_preReloc,  pset);

      //Carry forward ParticleID and pSize
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
      old_dw->get(pModID,              lb->pModalIDLabel,             pset);
      old_dw->get(psize,               lb->pSizeLabel,                pset);
      new_dw->get(pcursize,            lb->pCurSizeLabel,             pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pModIDNew,lb->pModalIDLabel_preReloc,    pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,       pset);
      pids_new.copyData(pids);
      pModIDNew.copyData(pModID);

      //Carry forward color particle (debugging label)
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      if(flags->d_XPIC2){
        new_dw->get(gvelSPSSP,     lb->gVelSPSSPLabel,       dwi,patch,gac,NGP);
      } else{
        NCVariable<Vector> gvelSPSSP_create;
        new_dw->allocateTemporary(gvelSPSSP_create,              patch,gac,NGP);
        gvelSPSSP_create.initialize(Vector(0.,0.,0.));
        gvelSPSSP = gvelSPSSP_create;               // reference created data
      }

      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(nodalWeightSum,  lb->NodalWeightSumLabel,  dwi,patch,gac,NGP);

      if(flags->d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
      } else{
        NCVariable<double> dTdt_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        dTdt_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
      }

      // gSurfNormLabel may contain a growth direction that is not normal
      if(flags->d_computeNormals){
        new_dw->get(gGrowthDir,    lb->gSurfNormLabel,       dwi,patch,gac,NGP);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                    patch,gac,NGP);
        gSN_create.initialize(Vector(0.));
        gGrowthDir = gSN_create;                     // reference created data
      }

      new_dw->get(massBurnFrac,    lb->massBurnFractionLabel,dwi,patch,gac,NGP);

      double Cp=mpm_matl->getSpecificHeat();
      Vector dx = patch->dCell();

      bool useXPIC=false;

      // The following logic is intended to turn on every 10th dissolution step
      // For problems not involving dissolution, or during phases of the
      // dissolution simulation where dissolution isn't happening, XPIC is
      // carried out as normal.
      if((flags->d_XPIC2 && !(flags->d_currentPhase=="dissolution" && 
                              flags->d_doingDissolution)) ||
         (flags->d_XPIC2 && 
          flags->d_currentPhase=="dissolution" && flags->d_doingDissolution 
                                        && timestep%10==1)){
        useXPIC=true;
      }

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
        Vector pSN(0.0,0.0,0.0);
        double tempRate = 0.0;
        double burnFraction = 0.0;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel      += gvelocity_star[node]  * S[k];
          velSSPSSP+= gvelSPSSP[node]       * S[k];
          acc      += gacceleration[node]   * S[k];
          pSN      += gGrowthDir[node]      * S[k];

          tempRate += (gTemperatureRate[node] + dTdt[node]) * S[k];
        }

        if(pSurf[idx]>=0.99){
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            burnFraction += massBurnFrac[node]*S[k]
                           /(nodalWeightSum[node]+1.e-100);
          }
        }

        // Update particle vel and pos using Nairn's XPIC(2) method
	// if useXPIC, otherwise just standard FLIP
        pxnew[idx]   = px[idx]    + vel*delT;
        if(useXPIC){
          pxnew[idx] -= 0.5*(acc*delT + 
                      (pvelocity[idx] - 2.0*pvelSSPlus[idx]) + velSSPSSP)*delT;
          pvelnew[idx]  = 2.0*pvelSSPlus[idx] - velSSPSSP   + acc*delT;
        } else {
          pvelnew[idx] = pvelocity[idx] + acc*delT;
        }
        pdispnew[idx] = pdisp[idx] + (pxnew[idx]-px[idx]);
        pTempNew[idx]    = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx] = pTemperature[idx]; // for thermal stress
        if (flags->d_doingDissolution){
          if(pSurf[idx]>=0.99 && burnFraction != 0.0){
            // Normalize particle surface normal
            pSN /= (pSN.length() + 1.e-100);
            int maxDir = 0; double maxComp=fabs(pSN.x());
            for(int i = 1; i<3; i++){
              if(fabs(pSN[i])>maxComp){
                maxComp=fabs(pSN[i]);
                maxDir=i;
              }
            }
            int maxDirP1 = (maxDir+1)%3;
            int maxDirP2 = (maxDir+2)%3;
            pmassNew[idx]    = Max(pmass[idx] - burnFraction*delT, 0.);
            double deltaMassFrac = (pmass[idx]-pmassNew[idx])/pmass[idx];
            Vector L[3];
            double Ll[3];
            double dL[3];
            double pSNdotL[3];
            for(int i=0;i<3;i++){
              L[i]=Vector(psize[idx](0,i),psize[idx](1,i),psize[idx](2,i));
              Ll[i] = L[i].length();

              L[i]/=Ll[i];
              pSNdotL[i] = fabs(Dot(pSN,L[i]));
            }

            double dL1overdL0 = pSNdotL[maxDirP1]/pSNdotL[maxDir];
            double dL2overdL0 = pSNdotL[maxDirP2]/pSNdotL[maxDir];

            dL[maxDir] = deltaMassFrac*(Ll[0]*Ll[1]*Ll[2])/
                                       (Ll[maxDirP1]*Ll[maxDirP2] 
                                 + dL1overdL0*Ll[maxDir]*Ll[maxDirP2] 
                                 + dL2overdL0*Ll[maxDir]*Ll[maxDirP1]);

            dL[maxDirP1] = dL1overdL0*dL[maxDir];
            dL[maxDirP2] = dL2overdL0*dL[maxDir];
            L[maxDir]   *= (Ll[maxDir]   - dL[maxDir]);
            L[maxDirP1] *= (Ll[maxDirP1] - dL[maxDirP1]);
            L[maxDirP2] *= (Ll[maxDirP2] - dL[maxDirP2]);

            psizeNew[idx] = Matrix3(L[0].x(), L[1].x(), L[2].x(),
                                    L[0].y(), L[1].y(), L[2].y(),
                                    L[0].z(), L[1].z(), L[2].z());

            Vector deltaPos = 0.5*Vector(dL[0]*dx.x()*pSN.x(),
                                         dL[1]*dx.y()*pSN.y(),
                                         dL[2]*dx.z()*pSN.z());
            pxnew[idx] = pxnew[idx] - deltaPos;
          } else {
            pmassNew[idx] = pmass[idx];
            psizeNew[idx] = psize[idx];
          }
        } else {
          pmassNew[idx]    = Max(pmass[idx]*(1.    - burnFraction),0.);
          psizeNew[idx]    = (pmassNew[idx]/pmass[idx])*psize[idx];
        }

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelnew[idx].length2()*useInKECalc[m];
        CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom   += pvelnew[idx]*pmass[idx];
        if(!mpm_matl->getIsPistonMaterial()){
          totalmass  += pmass[idx];
        }
        dissolvedmass  += Max(0., (pmass[idx] - pmassNew[idx]));
        pistonmass += pmass[idx]*useInKECalc[m];
      }

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
      new_dw->put(sum_vartype(dissolvedmass),  lb->DissolvedMassLabel);
      new_dw->put(sum_vartype(pistonmass),     lb->PistonMassLabel);
      new_dw->put(OIMSV,                       lb->InitialMassSVLabel);
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
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<double> pVolumeOld,pmass,pmassNew;
      constParticleVariable<int> pLocalized;
      constParticleVariable<Matrix3> pFOld;
      constParticleVariable<long64> pids;
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
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);

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

        int imax=0, jmax=0;
        if(pFNew[idx].MaxAbsElemComp(imax, jmax)>10 ||
           !(pFNew[idx].Determinant()>0.1)){
          cerr << "Resetting F for particle " << pids[idx] 
               << " with F = " << pFNew[idx] << endl;
          cerr << "imax, jmax = " << imax << ", " << jmax << endl;
          cerr << "matl  = " << m << endl;
          cerr << "tensorL  = " << tensorL << endl;
          cerr << "pmass = " << pmass[idx] << endl;
          cerr << "px = " << px[idx] << endl;
          cerr << "pvolume = " << pVolumeOld[idx] << endl;
          cerr << "J = " << pFNew[idx].Determinant() << endl;
          cerr << "F is now reset to " << Identity << endl;
          pFNew[idx]=Identity;
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
          if(J_CC[cell_index] > 0.0 && J > 0.0){
           pFNew[idx]*=cbrt(J_CC[cell_index]/J);
          }

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

    // DON'T MOVE THESE!!!
    //__________________________________
    //  reduction variables
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
    Vector dxCell = patch->dCell();
    double cell_vol = dxCell.x()*dxCell.y()*dxCell.z();

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<int> pLocalized;
      constParticleVariable<double> pdTdt,pmassNew,pVolNew;
      ParticleVariable<double> pTempNew;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      new_dw->get(pdTdt,        lb->pdTdtLabel,                      pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);
      new_dw->get(pVolNew,      lb->pVolumeLabel_preReloc,           pset);
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
            (pLocalized[idx]==-999) || 
             pVolNew[idx]<flags->d_min_partVolToCellVolRatio*cell_vol){
          cout << "Adding to delset, m = " << m << endl;
          cout << "pmassNew[idx] = " << pmassNew[idx] << endl;
          cout << "pTempNew[idx] = " << pTempNew[idx] << endl;
          cout << "pdTdt[idx] = " << pdTdt[idx] << endl;
          cout << "pLocalized[idx] = " << pLocalized[idx] << endl;
          delset->addParticle(idx);
        }

      } // particles
      new_dw->deleteParticles(delset);
    } // materials
    // These is only used for moving particles to other materials, but I need
    // a place to reset them out when I'm done with them.
  } // patches
}

void SerialMPM::updateTracers(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing updateTracers");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    BBox domain;
    const Level* level = getLevel(patches);
    level->getInteriorSpatialRange(domain);
    Point dom_min = domain.min();
    Point dom_max = domain.max();
    IntVector periodic = level->getPeriodicBoundaries();

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=m_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    Matrix3 size(0.5,0.,0.,0.,0.5,0.,0.,0.,0.5); // Placeholder, not used

    Ghost::GhostType  gac = Ghost::AroundCells;
    constNCVariable<Vector>  gvelocityglobal;
    constNCVariable<double>  gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
           m_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
    new_dw->get(gvelocityglobal,  lb->gVelocityLabel,
           m_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     m_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity[m], lb->gVelocityStarLabel,  dwi, patch, gac,NGN+1);
      new_dw->get(gmass[m],     lb->gMassLabel,          dwi, patch, gac,NGN+1);
      new_dw->get(dLdt[m],      lb->dLdtDissolutionLabel,dwi, patch, gac,NGN+1);
      if (flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],lb->gSurfNormLabel,     dwi, patch, gac,NGN+1);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+1);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numTracerMatls=m_materialManager->getNumMatls("Tracer");
    for(int tm = 0; tm < numTracerMatls; tm++){
      TracerMaterial* t_matl = (TracerMaterial *)
                                 m_materialManager->getMaterial("Tracer", tm );
      int dwi = t_matl->getDWIndex();

      int adv_matl = t_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<long64> tracer_ids;
      ParticleVariable<long64> tracer_ids_new;
      constParticleVariable<Vector> tracerCemVec;
      ParticleVariable<Vector> tracerCemVec_new;

      old_dw->get(tx,            lb->pXLabel,                         pset);
      old_dw->get(tracer_ids,  TraL->tracerIDLabel,                   pset);
      old_dw->get(tracerCemVec,TraL->tracerCemVecLabel,               pset);

      new_dw->allocateAndPut(tx_new,          lb->pXLabel_preReloc,       pset);
      new_dw->allocateAndPut(tracer_ids_new,TraL->tracerIDLabel_preReloc, pset);
      new_dw->allocateAndPut(tracerCemVec_new,
                                        TraL->tracerCemVecLabel_preReloc, pset);

      tracer_ids_new.copyData(tracer_ids);
      tracerCemVec_new.copyData(tracerCemVec);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(tx[idx],ni,S,size);
        Vector vel(0.0,0.0,0.0);
        Vector surf(0.0,0.0,0.0);
  
        double sumSk=0.0;
        Vector gSN(0.,0.,0.);
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++){
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf  -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
          gSN   += gSurfNorm[adv_matl][node]*S[k];
        }
        if(sumSk > 1.e-90){
          // This is the normal condition, when at least one of the nodes
          // influencing a tracer has mass on it.
          vel/=sumSk;
          tx_new[idx] = tx[idx] + vel*delT;
          tx_new[idx] += (surf/(gSN.length()+1.e-100))*delT;
        } else {
            // This is the "just in case" instance that none of the nodes
            // influencing a vertex has mass on it.  In this case, use an
            // interpolator with a larger footprint
            ParticleInterpolator* cpdiInterp=scinew cpdiInterpolator(patch);
            vector<IntVector> ni_cpdi(cpdiInterp->size());
            vector<double> S_cpdi(cpdiInterp->size());
            Matrix3 size; size.Identity();
            int N = cpdiInterp->findCellAndWeights(tx[idx],ni_cpdi,S_cpdi,size);
            vel  = Vector(0.0,0.0,0.0);
            surf = Vector(0.0,0.0,0.0);
            sumSk= 0.0;
            for (int k = 0; k < N; k++) {
             IntVector node = ni_cpdi[k];
              vel  += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S_cpdi[k];
              sumSk+= gmass[adv_matl][node]*S_cpdi[k];
              surf -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S_cpdi[k];
            }
            vel/=sumSk;
            tx_new[idx] = tx[idx] + vel*delT;
            tx_new[idx] += (surf/(gSN.length()+1.e-100))*delT;

            delete cpdiInterp;
          }
//        } else {
//          // This is the rare "just in case" instance that none of the nodes
//          // influencing a tracer has mass on it.  In this case, use the
//          // "center of mass" velocity to move the vertex
//          double sumSkCoM=0.0;
//          Vector velCoM(0.0,0.0,0.0);
//          for (int k = 0; k < NN; k++) {
//            IntVector node = ni[k];
//            sumSkCoM += gmassglobal[node]*S[k];
//            velCoM   += gvelocityglobal[node]*gmassglobal[node]*S[k];
//          }
//          velCoM/=sumSkCoM;
//          tx_new[idx] = tx[idx] + velCoM*delT;
//        }


#if 1
        // Check to see if a tracer has left the domain
        if(!domain.inside(tx_new[idx])){
          //cout << "tx[idx] = " << tx[idx] << endl;
          //cout << "tx_new[idx] = " << tx_new[idx] << endl;
          double epsilon = 1.e-15;
          static ProgressiveWarning warn("A tracer has moved outside the domain through an x boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
          Point txn = tx_new[idx];
          if(periodic.x()==0){
           if(tx_new[idx].x()<dom_min.x()){
            tx_new[idx] = Point(dom_min.x()+epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
           if(tx_new[idx].x()>dom_max.x()){
            tx_new[idx] = Point(dom_max.x()-epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
          }
          if(periodic.y()==0){
           if(tx_new[idx].y()<dom_min.y()){
            tx_new[idx] = Point(txn.x(),dom_min.y()+epsilon, txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
           if(tx_new[idx].y()>dom_max.y()){
            tx_new[idx] = Point(txn.x(),dom_max.y()-epsilon, txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
          }
          if(periodic.z()==0){
           if(tx_new[idx].z()<dom_min.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_min.z()+epsilon);
            warn.invoke();
           }
           if(tx_new[idx].z()>dom_max.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_max.z()-epsilon);
            warn.invoke();
           }
          }
        } // if tracer has left domain
#endif
      }
    }
    delete interpolator;
  }
}

void SerialMPM::updateLineSegments(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing updateLineSegments");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    Vector dx = patch->dCell();
    Matrix3 size; size.Identity();

    BBox domain;
    const Level* level = getLevel(patches);
    level->getInteriorSpatialRange(domain);
    Point dom_min = domain.min();
    Point dom_max = domain.max();
    IntVector periodic = level->getPeriodicBoundaries();

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=m_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     m_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity[m], lb->gVelocityStarLabel,  dwi, patch, gac,NGN+1);
      new_dw->get(gmass[m],     lb->gMassLabel,          dwi, patch, gac,NGN+1);
      new_dw->get(dLdt[m],      lb->dLdtDissolutionLabel,dwi, patch, gac,NGN+1);
      if (flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],lb->gSurfNormLabel,     dwi, patch, gac,NGN+1);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+1);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numLSMatls=m_materialManager->getNumMatls("LineSegment");
    for(int ls = 0; ls < numLSMatls; ls++){
      LineSegmentMaterial* ls_matl = (LineSegmentMaterial *) 
                              m_materialManager->getMaterial("LineSegment", ls);
      int dwi = ls_matl->getDWIndex();

      int adv_matl = ls_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<Matrix3> tsize, tF;
      ParticleVariable<Matrix3> tsize_new, tF_new;
      constParticleVariable<long64> lineseg_ids;
      ParticleVariable<long64> lineseg_ids_new;
      constParticleVariable<Vector> lsMidToEndVec;
      ParticleVariable<Vector> lsMidToEndVec_new;

      old_dw->get(tx,            lb->pXLabel,                         pset);
      old_dw->get(tsize,         lb->pSizeLabel,                      pset);
      old_dw->get(lineseg_ids,   lb->linesegIDLabel,                  pset);
      old_dw->get(tF,            lb->pDeformationMeasureLabel,        pset);
      old_dw->get(lsMidToEndVec, lb->lsMidToEndVectorLabel,           pset);

      new_dw->allocateAndPut(tx_new,         lb->pXLabel_preReloc,        pset);
      new_dw->allocateAndPut(tsize_new,      lb->pSizeLabel_preReloc,     pset);
      new_dw->allocateAndPut(lineseg_ids_new,lb->linesegIDLabel_preReloc, pset);
      new_dw->allocateAndPut(tF_new,lb->pDeformationMeasureLabel_preReloc,pset);
      new_dw->allocateAndPut(lsMidToEndVec_new,
                                       lb->lsMidToEndVectorLabel_preReloc,pset);

      lineseg_ids_new.copyData(lineseg_ids);
      tF_new.copyData(tF);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // First update the position of the "right" end of the segment
        Point right = tx[idx]+lsMidToEndVec[idx];
        Point left  = tx[idx]-lsMidToEndVec[idx];
        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(right, ni, S, size);
        Vector vel(0.0,0.0,0.0);
        Vector surf(0.0,0.0,0.0);
//        Vector v = left - right;

//        normal = v X (0,0,1)
//        Vector normal = Vector(v.y(), -v.x(), 0.)
//                         / (1.e-100+sqrt(v.y()*v.y()+v.x()*v.x()));
  
        double sumSk =0.0;
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf   -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
        }
        vel/=sumSk;
  
        right += vel*delT;
        right += surf*delT;
  
        // Next update the position of the "left" end of the segment
        // Get the node indices that surround the cell
        NN = interpolator->findCellAndWeights(left, ni, S, size);
        vel = Vector(0.0,0.0,0.0);
        surf = Vector(0.0,0.0,0.0);
  
        sumSk=0.0;
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf   -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
        }
        vel/=sumSk;
  
        left += vel*delT;
        left += surf*delT;

        tx_new[idx] = 0.5*(left+right);

        Vector lsETE = (right-left);
        lsMidToEndVec_new[idx] = 0.5*lsETE;
        Matrix3 size_new =Matrix3(lsETE.x()/dx.x(), 0.1*lsETE.y()/dx.y(), 0.0,
                                  lsETE.y()/dx.x(), -.1*lsETE.x()/dx.y(), 0.0,
                                              0.0,                  0.0, 1.0);
        tsize_new[idx] = size_new;
  
        // Check to see if a line segment has left the domain
        if(!domain.inside(tx_new[idx])){
          double epsilon = 1.e-15;
          Point txn = tx_new[idx];
          if(periodic.x()==0){
           if(tx_new[idx].x()<dom_min.x()){
            tx_new[idx] = Point(dom_min.x()+epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
           }
           if(tx_new[idx].x()>dom_max.x()){
            tx_new[idx] = Point(dom_max.x()-epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through an x boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
          if(periodic.y()==0){
           if(tx_new[idx].y()<dom_min.y()){
            tx_new[idx] = Point(txn.x(),dom_min.y()+epsilon, txn.z());
            txn = tx_new[idx];
           }
           if(tx_new[idx].y()>dom_max.y()){
            tx_new[idx] = Point(txn.x(),dom_max.y()-epsilon, txn.z());
            txn = tx_new[idx];
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through a y boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
          if(periodic.z()==0){
           if(tx_new[idx].z()<dom_min.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_min.z()+epsilon);
           }
           if(tx_new[idx].z()>dom_max.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_max.z()-epsilon);
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through a z boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
        }
      }
    }
    delete interpolator;
  }
}

void SerialMPM::computeLineSegmentForces(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeLineSegmentForces");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
//    ParticleInterpolator* interpolator=scinew cpdiInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType gac = Ghost::AroundCells;
    Vector dxCell = patch->dCell();
    double cell_length2 = dxCell.length2();

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > LSContForce(numMPMMatls);
    std::vector<NCVariable<double> > SurfArea(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<double> stiffness(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      double inv_stiff = mpm_matl->getConstitutiveModel()->getCompressibility();
      stiffness[m] = 1./inv_stiff;

      new_dw->allocateAndPut(LSContForce[m],lb->gLSContactForceLabel,dwi,patch);
      new_dw->allocateAndPut(SurfArea[m],   lb->gSurfaceAreaLabel,   dwi,patch);
      new_dw->get(gmass[m],                 lb->gMassLabel,          dwi,patch,
                                                                     gac,NGN+2);
      LSContForce[m].initialize(Vector(0.0));
      SurfArea[m].initialize(1.0e-100);
    }

    int numLSMatls=m_materialManager->getNumMatls("LineSegment");

    // Get the arrays of particle values to be changed
    std::vector<constParticleVariable<Point>  >  tx0(numLSMatls);
    std::vector<constParticleVariable<Matrix3> > tsize0(numLSMatls);
    std::vector<constParticleVariable<Vector>  > lsMidToEndVec0(numLSMatls);
    std::vector<ParticleSubset*> psetvec;

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      LineSegmentMaterial* t_matl0 = (LineSegmentMaterial *) 
                             m_materialManager->getMaterial("LineSegment", tmo);
      int dwi0 = t_matl0->getDWIndex();

      ParticleSubset* pset0 = old_dw->getParticleSubset(dwi0, patch,
                                                       gac, 2, lb->pXLabel);
      psetvec.push_back(pset0);

      old_dw->get(tx0[tmo],            lb->pXLabel,                   pset0);
      old_dw->get(tsize0[tmo],         lb->pSizeLabel,                pset0);
      old_dw->get(lsMidToEndVec0[tmo], lb->lsMidToEndVectorLabel,     pset0);
    }

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      LineSegmentMaterial* t_matl0 = (LineSegmentMaterial *) 
                             m_materialManager->getMaterial("LineSegment", tmo);
      int adv_matl0 = t_matl0->getAssociatedMaterial();

      ParticleSubset* pset0 = psetvec[tmo];

      // Extrapolate area of line segments to the grid for use in dissolution
      if (flags->d_doingDissolution){
       for(ParticleSubset::iterator iter0 = pset0->begin();
           iter0 != pset0->end(); iter0++){
         particleIndex idx0 = *iter0;

         Point px0=tx0[tmo][idx0] - lsMidToEndVec0[tmo][idx0];
         Point a = tx0[tmo][idx0] + lsMidToEndVec0[tmo][idx0];
         Vector v = px0 - a;
         double vLength = v.length();
         double LSArea = vLength*dxCell.z();
         Matrix3 size0 = tsize0[tmo][idx0];
         int nn = interpolator->findCellAndWeights(px0,ni,S,size0);
         double totMass = 0.;
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           totMass += S[k]*gmass[adv_matl0][node];
        }

         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)) {
             SurfArea[adv_matl0][node] += 0.5*LSArea*S[k]*gmass[adv_matl0][node]
                                          /totMass;
           }
         }

         nn = interpolator->findCellAndWeights(a,ni,S,size0);
         totMass = 0.;
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           totMass += S[k]*gmass[adv_matl0][node];
         }

         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)) {
             SurfArea[adv_matl0][node] += 0.5*LSArea*S[k]*gmass[adv_matl0][node]
                                          /totMass;
           }
         }
        } // loop over particles
      }  // if doingDissolution

      for(int tmi = tmo+1; tmi < numLSMatls; tmi++) {
        LineSegmentMaterial* t_matl1 = (LineSegmentMaterial *) 
                              m_materialManager->getMaterial("LineSegment",tmi);
        int adv_matl1 = t_matl1->getAssociatedMaterial();

        if(adv_matl0==adv_matl1){
          continue;
        }

        ParticleSubset* pset1 = psetvec[tmi];

        int numPar_pset1 = pset1->numParticles();

        double K_l = 10.*(stiffness[adv_matl0] * stiffness[adv_matl1])/
                         (stiffness[adv_matl0] + stiffness[adv_matl1]);

       if(numPar_pset1 > 0){

        // Loop over zeroth line segment subset
        // Only test the "left" end of the line segment for
        // penetration into other segments, because the right
        // end will get checked as the left end of the next segment
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;

          Point px0=tx0[tmo][idx0] - lsMidToEndVec0[tmo][idx0];

          double min_sep  = 9.e99;
          double min_sep2  = 9.e99;
          int closest = 99999;
          int secondClosest = 99999;
          // Loop over other particle subset
          for(ParticleSubset::iterator iter1 = pset1->begin();
              iter1 != pset1->end(); iter1++){
            particleIndex idx1 = *iter1;
            Point px1 = tx0[tmi][idx1];
            double sep = (px1-px0).length2();
            if(sep < min_sep2 && sep < 0.25*cell_length2){
              if(sep < min_sep){
                secondClosest=closest;
                min_sep2=min_sep;
                closest  = idx1;
                min_sep  = sep;
              } 
              else{
                secondClosest=idx1;
                min_sep2  = sep;
              }
            }
          }

          double forceMag=0.0;
          bool done = false;
          double tC1 = 99.9;
          double tC2 = 99.9;
          double overlap1 = 99.9;
          double overlap2 = 99.9;
          Vector normal1(0.,0.,0);
          Vector normal2(0.,0.,0);
          if(closest < 99999){
            // Following the description in stackexchange:
            // https://math.stackexchange.com/questions/2193720/find-a-point-on-a-line-segment-which-is-the-closest-to-other-point-not-on-the-li
           Point A = tx0[tmi][closest] + lsMidToEndVec0[tmi][closest];
           Point B = tx0[tmi][closest] - lsMidToEndVec0[tmi][closest];
           Vector v = B - A;
           double vLength2 = v.length2();

           Vector u = A - px0;
           tC1 = -Dot(v,u)/vLength2;
           Vector fromLineSegToPoint1= px0.asVector() - ((1.-tC1)*A + tC1*B);
           //normal = v X (0,0,1)
           normal1 = Vector(v.y(), -v.x(), 0.)
                         / (1.e-100+sqrt(v.y()*v.y()+v.x()*v.x()));
           overlap1 = Dot(normal1,fromLineSegToPoint1);
           if(tC1 >= 0.0 && tC1 <= 1.0){
              if(overlap1 < 0.0){
               done = true;
               double vLength = sqrt(vLength2);
               double K = K_l*vLength;
               forceMag = overlap1*K;

               Vector tForce1A = (1.-tC1)*forceMag*normal1;
               Vector tForce1B = tC1*forceMag*normal1;

               // See comments in addCohesiveZoneForces for a description of how
               // the force is put on the nodes

               // Get the node indices that surround the cell
               Matrix3 size1 = tsize0[tmi][closest];
               int NN = interpolator->findCellAndWeights(A, ni, S, size1);
  
               double totMass0 = 0.;
               double totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1A*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
//                 }
                 }
               }

               // Get the node indices that surround the cell
               NN = interpolator->findCellAndWeights(B, ni, S, size1);
  
               totMass0 = 0.;
               totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1B*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1B*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
//                 }
                 }
               }

              } // if overlap1
           } // if(tC1 >= 0.0 && tC1 <= 1.0)

           if(!done && secondClosest < 99999){
            Point A2=tx0[tmi][secondClosest]+lsMidToEndVec0[tmi][secondClosest];
            Point B2=tx0[tmi][secondClosest]-lsMidToEndVec0[tmi][secondClosest];
            Vector v2 = B2 - A2;
            double v2Length2 = v2.length2();

            Vector u2 = A2 - px0;
            tC2 = -Dot(v2,u2)/v2Length2;
            Vector fromLineSegToPoint2= px0.asVector() - ((1.-tC2)*A2 + tC2*B2);
            normal2 = Vector(v2.y(), -v2.x(), 0.)
                        /(1.e-100 + sqrt(v2.y()*v2.y() + v2.x()*v2.x()));
            overlap2 = Dot(normal2,fromLineSegToPoint2);
            if(((tC1 < 0.0 && tC2 > 1.0) || (tC1 > 1.0 && tC2 < 0.0)) && 
               overlap1 < 0.0 && overlap2 < 0.0 && !done){
              done = true;

              double vLength = sqrt(vLength2);
              double K = K_l*vLength;

              Point vertex;
              Vector n;
              double sizeWeight1, sizeWeight2;
              Matrix3 size1,size2;
              if((px0-A).length2() < (px0-B).length2()){ // closest
               vertex = A;
               n=normal1;
               size1 = tsize0[tmi][closest];
               size2 = tsize0[tmi][secondClosest];
               sizeWeight1=fabs(tC2-1.)/(fabs(tC2-1.) + fabs(tC1));
               sizeWeight2=fabs(tC1)/(fabs(tC2-1.) + fabs(tC1));
              } else {  // secondClosest;
               vertex = B;
               n=normal2;
               size1 = tsize0[tmi][secondClosest];
               size2 = tsize0[tmi][closest];
               sizeWeight1=fabs(tC1-1.)/(fabs(tC1-1.) + fabs(tC2));
               sizeWeight2=fabs(tC2)/(fabs(tC1-1.) + fabs(tC2));
              }
//              forceMag = overlap1*K;
//              Vector tForce1A = forceMag*normal1;
//              Vector tForce1A = forceMag*n;
              Vector tForce1A = K*(px0-vertex);
              // Get the node indices that surround the cell
//              Matrix3 size_mean = 0.5*(size1+size2);
              Matrix3 size_mean = sizeWeight1*size1 + sizeWeight2*size2;
              int NN = interpolator->findCellAndWeights(px0, ni, S, size_mean);
 
              double totMass0 = 0.;
              double totMass1 = 0.;
              for (int k = 0; k < NN; k++) {
                IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                totMass0 += S[k]*gmass[adv_matl0][node];
                totMass1 += S[k]*gmass[adv_matl1][node];
//              }
              }

              // Accumulate the contribution from each surrounding vertex
              for (int k = 0; k < NN; k++) {
                IntVector node = ni[k];
                if(patch->containsNode(node)) {
                  // Distribute force according to material mass on the nodes
                  // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                  LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                            * gmass[adv_matl0][node]/totMass0;
                  LSContForce[adv_matl1][node] += tForce1A*S[k]
                                            * gmass[adv_matl1][node]/totMass1;
//                }
                }
              }
            } // if(tC1...)
           }  // if secondClosest

          } // closest < 99999

          if(!done && secondClosest < 99999){
            Point A2=tx0[tmi][secondClosest]+lsMidToEndVec0[tmi][secondClosest];
            Point B2=tx0[tmi][secondClosest]-lsMidToEndVec0[tmi][secondClosest];
            Vector v2 = B2 - A2;
            double v2Length2 = v2.length2();

            Vector u2 = A2 - px0;
            tC2 = -Dot(v2,u2)/v2Length2;
            Vector fromLineSegToPoint2= px0.asVector() - ((1.-tC2)*A2 + tC2*B2);
            //normal = v X (0,0,1)
            if(tC2 >= 0.0 && tC2 <= 1.0){
              Vector normal = Vector(v2.y(), -v2.x(), 0.)
                          / (1.e-100+sqrt(v2.y()*v2.y()+v2.x()*v2.x()));
              overlap2 = Dot(normal,fromLineSegToPoint2);
              if(overlap2 < 0.0){
               done = true;
               double v2Length = sqrt(v2Length2);
               double K = K_l*v2Length;
               forceMag = overlap2*K;

               Vector tForce1A = (1.-tC2)*forceMag*normal;
               Vector tForce1B = tC2*forceMag*normal;

               // Get the node indices that surround the cell
               Matrix3 size1 = tsize0[tmi][closest];
               int NN = interpolator->findCellAndWeights(A2, ni, S, size1);
  
               double totMass0 = 0.;
               double totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1A*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
//                 }
                 }
               }

               // Get the node indices that surround the cell
               NN = interpolator->findCellAndWeights(B2, ni, S, size1);
  
               totMass0 = 0.;
               totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1B*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1B*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
//                 }
                 }
               }
              } // if overlap2
            }
          }
        } //  Outer loop over linesegments
       }
      } // inner loop over line segment materials
    } // outer loop over line segment materials
//    cout << "numOverlap = " << numOverlap << endl;
    delete interpolator;
  }
}

void SerialMPM::updateTriangles(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing updateTriangles");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    Ghost::GhostType  gac = Ghost::AroundCells;

    unsigned int numMPMMatls=m_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    std::vector<bool> PistonMaterial(numMPMMatls);

//    constNCVariable<Vector>  gvelocityglobal;
    constNCVariable<double>  gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
           m_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
//    new_dw->get(gvelocityglobal,  lb->gVelocityLabel,
//           m_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     m_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity[m], lb->gVelocityStarLabel,  dwi, patch, gac,NGN+1);
      new_dw->get(gmass[m],     lb->gMassLabel,          dwi, patch, gac,NGN+1);
      new_dw->get(dLdt[m],      lb->dLdtDissolutionLabel,dwi, patch, gac,NGN+1);
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();

      if (flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],lb->gSurfNormLabel,     dwi, patch, gac,NGN+1);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+1);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numLSMatls=m_materialManager->getNumMatls("Triangle");
    for(int ls = 0; ls < numLSMatls; ls++){
      TriangleMaterial* ls_matl = (TriangleMaterial *) 
                              m_materialManager->getMaterial("Triangle", ls);
      int dwi = ls_matl->getDWIndex();

      int adv_matl = ls_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<Matrix3> tsize, tF;
      ParticleVariable<Matrix3> tsize_new, tF_new;
      constParticleVariable<long64> triangle_ids;
      ParticleVariable<long64> tri_ids_new;
      constParticleVariable<Vector> triMidToN0Vec, triMidToN1Vec, triMidToN2Vec;
      ParticleVariable<Vector> triMidToN0Vec_new, 
                               triMidToN1Vec_new, triMidToN2Vec_new;
      constParticleVariable<IntVector> triUseInPenalty;
      ParticleVariable<IntVector>      triUseInPenalty_new;
      constParticleVariable<double> triArea, triClay, triMassDisp;
      ParticleVariable<double>      triArea_new, triClay_new, triMassDisp_new;
      constParticleVariable<Vector> triAreaAtNodes;
      ParticleVariable<Vector>      triAreaAtNodes_new, triNormal_new;
      ParticleVariable<IntVector>   triMultiMat;
//      ParticleVariable<IntVector>   triNearbyMatsN0,triNearbyMatsN1,triNearbyMatsN2;
      ParticleVariable<Matrix3>     triNearbyMats;

      old_dw->get(tx,              lb->pXLabel,                         pset);
      old_dw->get(tsize,           lb->pSizeLabel,                      pset);
      old_dw->get(triangle_ids,    TriL->triangleIDLabel,               pset);
      old_dw->get(tF,              lb->pDeformationMeasureLabel,        pset);
      old_dw->get(triMidToN0Vec,   TriL->triMidToN0VectorLabel,         pset);
      old_dw->get(triMidToN1Vec,   TriL->triMidToN1VectorLabel,         pset);
      old_dw->get(triMidToN2Vec,   TriL->triMidToN2VectorLabel,         pset);
      old_dw->get(triUseInPenalty, TriL->triUseInPenaltyLabel,          pset);
      old_dw->get(triArea,         TriL->triAreaLabel,                  pset);
      old_dw->get(triAreaAtNodes,  TriL->triAreaAtNodesLabel,           pset);
      old_dw->get(triClay,         TriL->triClayLabel,                  pset);
      old_dw->get(triMassDisp,     TriL->triMassDispLabel,              pset);

      new_dw->allocateAndPut(tx_new,         lb->pXLabel_preReloc,        pset);
      new_dw->allocateAndPut(tsize_new,      lb->pSizeLabel_preReloc,     pset);
      new_dw->allocateAndPut(tri_ids_new,  TriL->triangleIDLabel_preReloc,pset);
      new_dw->allocateAndPut(tF_new,lb->pDeformationMeasureLabel_preReloc,pset);
      new_dw->allocateAndPut(triMidToN0Vec_new,
                                    TriL->triMidToN0VectorLabel_preReloc, pset);
      new_dw->allocateAndPut(triMidToN1Vec_new,
                                    TriL->triMidToN1VectorLabel_preReloc, pset);
      new_dw->allocateAndPut(triMidToN2Vec_new,
                                    TriL->triMidToN2VectorLabel_preReloc, pset);
      new_dw->allocateAndPut(triUseInPenalty_new,
                                     TriL->triUseInPenaltyLabel_preReloc, pset);
      new_dw->allocateAndPut(triArea_new,
                                     TriL->triAreaLabel_preReloc,         pset);
      new_dw->allocateAndPut(triAreaAtNodes_new,
                                     TriL->triAreaAtNodesLabel_preReloc,  pset);
      new_dw->allocateAndPut(triClay_new,
                                     TriL->triClayLabel_preReloc,         pset);
      new_dw->allocateAndPut(triMassDisp_new,
                                     TriL->triMassDispLabel_preReloc,     pset);
      new_dw->allocateAndPut(triNormal_new,
                                     TriL->triNormalLabel_preReloc,       pset);
      new_dw->allocateAndPut(triMultiMat,
                                     TriL->triMultiMatLabel_preReloc,     pset);
      new_dw->allocateAndPut(triNearbyMats,
                                   TriL->triNearbyMatsLabel_preReloc,   pset);
//      new_dw->allocateAndPut(triNearbyMatsN0,
//                                   TriL->triNearbyMatsN0Label_preReloc,   pset);
//      new_dw->allocateAndPut(triNearbyMatsN1,
//                                   TriL->triNearbyMatsN1Label_preReloc,   pset);
//      new_dw->allocateAndPut(triNearbyMatsN2,
//                                   TriL->triNearbyMatsN2Label_preReloc,   pset);


      tri_ids_new.copyData(triangle_ids);
      tF_new.copyData(tF);
      triAreaAtNodes_new.copyData(triAreaAtNodes);
      triUseInPenalty_new.copyData(triUseInPenalty);
      triClay_new.copyData(triClay);
      triMassDisp_new.copyData(triMassDisp);

      double totalsurfarea = 0.;

      // Loop over triangles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        triMultiMat[idx]=IntVector(0,0,0);

        Point P[3];
        // Update the positions of the triangle vertices
        P[0] = tx[idx] + triMidToN0Vec[idx];
        P[1] = tx[idx] + triMidToN1Vec[idx];
        P[2] = tx[idx] + triMidToN2Vec[idx];
        // Keep track of how much of the triangle's motion is due to mass change
        Vector surf[3] = {Vector(0.0),Vector(0.0),Vector(0.0)};
 
        // Loop over the vertices
        int deleteThisTriangle = 0;
        Vector vertexVel[3];
        double populatedVertex[3]={0.,0.,0.};
        double DisPrecip = 0.;  // Dissolving if > 0, precipitating if < 0.
        IntVector negnn(-99,-99,-99);
        IntVector matls[3]={negnn,negnn,negnn};

        for(int itv = 0; itv < 3; itv++){
          // Get the node indices that surround the point
          int NN = interpolator->findCellAndWeights(P[itv], ni, S, tsize[idx]);
          Vector vel(0.0,0.0,0.0);
          double sumSk=0.0;
          Vector gSN(0.,0.,0.);
          vector< std::pair <double,int> > matlMass(numMPMMatls);
          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
            sumSk += gmass[adv_matl][node]*S[k];
            surf[itv] -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
            gSN   += gSurfNorm[adv_matl][node]*S[k];
            DisPrecip += dLdt[adv_matl][node]*S[k];
            if(gmass[adv_matl][node] <= 0.70*gmassglobal[node]){
              triMultiMat[idx](itv)=1;
            }
          }
          if(triUseInPenalty[idx](itv)==1){
            for (int k = 0; k < NN; k++) {
              IntVector node = ni[k];
              for(int m=0;m<numMPMMatls;m++){
                 matlMass[m].first = gmass[m][node]*S[k];
                 matlMass[m].second = m;
              }
              sort(matlMass.begin(), matlMass.end());
              matls[itv]=IntVector(matlMass[1].second,  
                                   matlMass[2].second,  
                                   matlMass[3].second);
            }
          }

          if(sumSk > 1.e-90){
            // This is the normal condition, when at least one of the nodes
            // influencing a vertex has mass on it.
            vel/=sumSk;
            P[itv] += vel*delT;
            surf[itv]/=(gSN.length()+1.e-100);
            P[itv] += surf[itv]*delT;
            vertexVel[itv] = vel + surf[itv];
            populatedVertex[itv] = 1.;
          } else {
            deleteThisTriangle++;
          }
        } // loop over vertices

        triNearbyMats[idx](0,0)=matls[0].x();
        triNearbyMats[idx](0,1)=matls[0].y();
        triNearbyMats[idx](0,2)=matls[0].z();
        triNearbyMats[idx](1,0)=matls[1].x();
        triNearbyMats[idx](1,1)=matls[1].y();
        triNearbyMats[idx](1,2)=matls[1].z();
        triNearbyMats[idx](2,0)=matls[2].x();
        triNearbyMats[idx](2,1)=matls[2].y();
        triNearbyMats[idx](2,2)=matls[2].z();

//        cout << "triNearbyMats = " << triNearbyMats[idx] << endl;
//        triNearbyMatsN0[idx]=matls[0];
//        triNearbyMatsN1[idx]=matls[1];
//        triNearbyMatsN2[idx]=matls[2];

        if(DisPrecip <=0 && !PistonMaterial[adv_matl]){
          totalsurfarea+=triArea[idx];
        }

        // Handle the triangles that have vertices that are not near nodes with mass
        if(deleteThisTriangle==3){
          cout << "NOTICE: Deleting " << triangle_ids[idx] << " of group " << adv_matl
               << " because none of its vertices are getting any nodal input." << endl; 
          delset->addParticle(idx);
        } else if(deleteThisTriangle>0){
          Vector velMean(0.);
          double populatedVertices=0.;
          for(int itv = 0; itv < 3; itv++){
            velMean += vertexVel[itv]*populatedVertex[itv];
            populatedVertices+=populatedVertex[itv]; 
          } // loop over vertices
          velMean/=populatedVertices;
          for(int itv = 0; itv < 3; itv++){
            P[itv] += velMean*(1. - populatedVertex[itv])*delT;
          } // loop over vertices
        }

        tx_new[idx] = (P[0]+P[1]+P[2])/3.;
        Vector triNorm = Cross(P[1]-P[0],P[2]-P[0]);
        double triNormLength = triNorm.length()+1.e-100;
        triArea_new[idx]=0.5*triNormLength;
        triNormal_new[idx]=triNorm/triNormLength;
        triMassDisp_new[idx] += Dot(triNormal_new[idx],
                                    (surf[0] + surf[1] + surf[2])*delT/3.);

        triMidToN0Vec_new[idx] = P[0] - tx_new[idx];
        triMidToN1Vec_new[idx] = P[1] - tx_new[idx];
        triMidToN2Vec_new[idx] = P[2] - tx_new[idx];
#if 0
        // No point in updating size unless it is used.  Just carry forward.
        Vector r0 = P[1] - P[0];
        Vector r1 = P[2] - P[0];
        Vector r2 = 0.1*Cross(r1,r0);
        Matrix3 size =Matrix3(r0.x()/dx.x(), r1.x()/dx.x(), r2.x()/dx.x(),
                              r0.y()/dx.y(), r1.y()/dx.y(), r2.y()/dx.y(),
                              r0.z()/dx.z(), r1.z()/dx.z(), r2.z()/dx.z());
        tsize_new[idx] = size;
#endif
        tsize_new[idx] = tsize[idx];

      } // Loop over triangles
      new_dw->deleteParticles(delset);

      new_dw->put(sum_vartype(totalsurfarea),      lb->TotalSurfaceAreaLabel);

#if 0
      // This is for computing updated triAreaAtNodes. Need to create a
      // container to replace the modified Stencil7 that I used in a hack here
      // Loop over triangles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Hit each vertex of the triangle
        double area0=0., area1=0., area2=0.;

        // Vertex 0
        for(int itri=0; itri < triNode0TriIDs[idx][29]; itri++) {
          int triID = triNode0TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area0+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area0/=3.;

        // Vertex 1
        for(int itri=0; itri < triNode1TriIDs[idx][29]; itri++) {
          int triID = triNode1TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area1+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area1/=3.;

        // Vertex 2
        for(int itri=0; itri < triNode2TriIDs[idx][29]; itri++) {
          int triID = triNode2TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area2+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area2/=3.;

        triAreaAtNodes_new[idx]=Vector(area0, area1, area2);
      } // Outer loop over triangles for vertex area calculation
#endif
    }  // matls

    delete interpolator;
  }    // patches
}

void SerialMPM::computeTriangleForces(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeTriangleForces");

    // Get the current simulation time
    //simTime_vartype simTimeVar;
    //old_dw->get(simTimeVar, lb->simulationTimeLabel);
    //double time = simTimeVar;

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType gac = Ghost::AroundCells;
    Vector dxCell = patch->dCell();
    double cell_length2 = dxCell.length2();

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > LSContForce(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<NCVariable<double> > SurfArea(numMPMMatls);
    std::vector<NCVariable<double> > SurfClay(numMPMMatls);
    std::vector<NCVariable<int> > InContactMatl(numMPMMatls);
    std::vector<double> stiffness(numMPMMatls);
    std::vector<bool> PistonMaterial(numMPMMatls);
//    std::vector<Vector> sumTriForce(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();

      double inv_stiff = mpm_matl->getConstitutiveModel()->getCompressibility();
      stiffness[m] = 1./inv_stiff;

      new_dw->allocateAndPut(LSContForce[m],lb->gLSContactForceLabel,dwi,patch);
      LSContForce[m].initialize(Vector(0.0));
      new_dw->allocateAndPut(InContactMatl[m],lb->gInContactMatlLabel,dwi,patch);
      InContactMatl[m].initialize(-99);

      if (flags->d_doingDissolution) {
        new_dw->allocateAndPut(SurfArea[m], lb->gSurfaceAreaLabel,   dwi,patch);
        new_dw->allocateAndPut(SurfClay[m], lb->gSurfaceClayLabel,   dwi,patch);
        SurfArea[m].initialize(0.0);
        SurfClay[m].initialize(0.0);
      }
      new_dw->get(gmass[m],                 lb->gMassLabel,          dwi,patch,
                                                                     gac,NGN+3);
//      sumTriForce[m]=Vector(0.0);
    }

    int numLSMatls=m_materialManager->getNumMatls("Triangle");

    std::vector<constParticleVariable<Point>  >  tx0(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN0Vec(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN1Vec(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN2Vec(numLSMatls);
//    std::vector<ParticleVariable<int>  >         triInContact(numLSMatls);

    std::vector<std::vector<constParticleVariable<Vector>  > >
                                                    triMidToNodeVec(numLSMatls);

    std::vector<constParticleVariable<long64> >     triangle_ids(numLSMatls);
    std::vector<constParticleVariable<double> >     triClay(numLSMatls);
    std::vector<constParticleVariable<IntVector> >  triUseInPenalty(numLSMatls);
    std::vector<constParticleVariable<Vector> >     triAreaAtNodes(numLSMatls);
    std::vector<constParticleVariable<double> >     triMassDisp(numLSMatls);
    std::vector<constParticleVariable<IntVector> >  triMultiMat(numLSMatls);
    std::vector<constParticleVariable<Matrix3> >    triNearbyMats(numLSMatls);
    std::vector<ParticleSubset*> psetvec;
    std::vector<int> psetSize(numLSMatls);
//    std::vector<std::vector<int> > triInContact(numLSMatls);
    Matrix3 size; size.Identity();

    FILE* fp;
    if(m_output->isOutputTimeStep()){
      timeStep_vartype timeStep;
      old_dw->get(timeStep, lb->timeStepLabel);
      int timestep = timeStep;

      string udaDir = m_output->getOutputLocation();
      ostringstream tname;
      tname << setw(5) << setfill('0') << timestep;
      string tnames = tname.str();
      string pPath = udaDir + "/results_contacts";
      DIR *check = opendir(pPath.c_str());
      if ( check == nullptr ) {
        MKDIR( pPath.c_str(), 0777 );
      } else {
        closedir(check);
      }

      stringstream pnum;
      pnum << patch->getID();
      string pnums = pnum.str();
      string fname = pPath + "/TriContact." + pnums + "." + tnames;
      fp = fopen(fname.c_str(), "w");
    }

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      TriangleMaterial* t_matl0 = (TriangleMaterial *) 
                             m_materialManager->getMaterial("Triangle", tmo);
      int dwi0 = t_matl0->getDWIndex();

      ParticleSubset* pset0 = old_dw->getParticleSubset(dwi0, patch,
                                                        gac, 2, lb->pXLabel);
      psetvec.push_back(pset0);
      psetSize[tmo]=(pset0->end() - pset0->begin());
//      triInContact[tmo].resize(psetSize[tmo]);

      old_dw->get(tx0[tmo],            lb->pXLabel,                   pset0);
      old_dw->get(triMidToN0Vec[tmo],  TriL->triMidToN0VectorLabel,   pset0);
      old_dw->get(triMidToN1Vec[tmo],  TriL->triMidToN1VectorLabel,   pset0);
      old_dw->get(triMidToN2Vec[tmo],  TriL->triMidToN2VectorLabel,   pset0);
      old_dw->get(triUseInPenalty[tmo],TriL->triUseInPenaltyLabel,    pset0);
      old_dw->get(triAreaAtNodes[tmo], TriL->triAreaAtNodesLabel,     pset0);
      old_dw->get(triangle_ids[tmo],   TriL->triangleIDLabel,         pset0);
      old_dw->get(triClay[tmo],        TriL->triClayLabel,            pset0);
      old_dw->get(triMultiMat[tmo],    TriL->triMultiMatLabel,        pset0);
      old_dw->get(triNearbyMats[tmo],  TriL->triNearbyMatsLabel,      pset0);

      if (flags->d_doingDissolution) {
        old_dw->get(triMassDisp[tmo],  TriL->triMassDispLabel,        pset0);
      } else {
        ParticleVariable<double>   triMassDisp_tmp;
        new_dw->allocateTemporary(triMassDisp_tmp,  pset0);
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;
          triMassDisp_tmp[idx0] = 0.;
        }
        triMassDisp[tmo]=triMassDisp_tmp;
      }
      triMidToNodeVec[tmo].push_back(triMidToN0Vec[tmo]);
      triMidToNodeVec[tmo].push_back(triMidToN1Vec[tmo]);
      triMidToNodeVec[tmo].push_back(triMidToN2Vec[tmo]);
//      new_dw->allocateAndPut(triInContact[tmo],TriL->triInContactLabel,pset0);
//      for(ParticleSubset::iterator iter0 = pset0->begin();
//          iter0 != pset0->end(); iter0++){
//        particleIndex idx0 = *iter0;
//        triInContact[tmo][idx0] = -1;
//      }
    } // end loop over triangle materials to get data from DW

    int numOverlap=0;
    int numInside=0;
    double totalContactArea    = 0.0;
    double totalContactAreaTri = 0.0;
    Vector totalForce(0.);
    double timefactor=1.0;
//    double timefactor=min(1.0, time/1.0);
//    proc0cout << "timefactor = " << timefactor << endl;

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      TriangleMaterial* t_matl0 = (TriangleMaterial *) 
                             m_materialManager->getMaterial("Triangle", tmo);
      int adv_matl0 = t_matl0->getAssociatedMaterial();

      ParticleSubset* pset0 = psetvec[tmo];

      // Extrapolate area of line segments to the grid for use in dissolution
      if (flags->d_doingDissolution){
       for(ParticleSubset::iterator iter0 = pset0->begin();
           iter0 != pset0->end(); iter0++){
         particleIndex idx0 = *iter0;

         Point vert[3];

         vert[0] = tx0[tmo][idx0] + triMidToN0Vec[tmo][idx0];
         vert[1] = tx0[tmo][idx0] + triMidToN1Vec[tmo][idx0];
         vert[2] = tx0[tmo][idx0] + triMidToN2Vec[tmo][idx0];
         Vector BA = vert[1]-vert[0];
         Vector CA = vert[2]-vert[0];
         double thirdTriArea = 0.5*Cross(BA,CA).length()/3.;

         for(int itv = 0; itv < 3; itv++){
           int nn = interpolator->findCellAndWeights(vert[itv], ni, S, size);
           double totMass = 0.;
           for (int k = 0; k < nn; k++) {
             IntVector node = ni[k];
             totMass += S[k]*gmass[adv_matl0][node];
           }

           for (int k = 0; k < nn; k++) {
             IntVector node = ni[k];
             if(patch->containsNode(node)) {
               SurfArea[adv_matl0][node] += thirdTriArea*S[k]
                                           *gmass[adv_matl0][node]/totMass;
               SurfClay[adv_matl0][node] += thirdTriArea*triClay[tmo][idx0]*S[k]
                                           *gmass[adv_matl0][node]/totMass;
             }
           }
         }
       } // loop over all triangles

       // Now loop over the nodes and normalize SurfClay by the area
       for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
         IntVector c = *iter;
         SurfClay[adv_matl0][c]/=(SurfArea[adv_matl0][c]+1.e-100);
       } // loop over all nodes
      }   // only do this if a dissolution problem

      for(int tmi = tmo+1; tmi < numLSMatls; tmi++) {
       TriangleMaterial* t_matl1 = (TriangleMaterial *) 
                             m_materialManager->getMaterial("Triangle",tmi);
       int adv_matl1 = t_matl1->getAssociatedMaterial();

       if(adv_matl0==adv_matl1 || 
          (PistonMaterial[adv_matl0] && PistonMaterial[adv_matl1])){
         continue;
       }

       ParticleSubset* pset1 = psetvec[tmi];

       int numPar_pset1 = pset1->numParticles();

       double K_l = 10.*(stiffness[adv_matl0] * stiffness[adv_matl1])/
                        (stiffness[adv_matl0] + stiffness[adv_matl1]);
       K_l*=timefactor;

       if(numPar_pset1 > 0){

        // Loop over zeroth triangle subset
        // Then loop over the vertices of the triangle
        // Check to see if they are to be "used" in force
        // calculation.  Every vertex should only be used once.
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;

         for(int iu = 0; iu < 3; iu++){

          if(triUseInPenalty[tmo][idx0](iu)==0 || 
             triMultiMat[tmo][idx0](iu) == 0 ||
            ((int) triNearbyMats[tmo][idx0](iu,0) != adv_matl1 &&
             (int) triNearbyMats[tmo][idx0](iu,1) != adv_matl1 &&
             (int) triNearbyMats[tmo][idx0](iu,2) != adv_matl1)){
            continue;
          }

          Point px0=tx0[tmo][idx0] + triMidToNodeVec[tmo][iu][idx0];

          Vector ptNormal =Cross(triMidToN0Vec[tmo][idx0],
                                 triMidToN1Vec[tmo][idx0]);
          double pNL = ptNormal.length();
          if(pNL>0.0){
            ptNormal /= pNL;
          }

          bool foundOne = false;
          vector<double> triSep;
          vector<int> triIndex;
          vector<double> triOverlap;
          vector<Point> triInPlane;
          vector<Vector> triTriNormal;
          // Loop over other particle subset
          for(ParticleSubset::iterator iter1 = pset1->begin();
              iter1 != pset1->end(); iter1++){
            particleIndex idx1 = *iter1;
            // AP is a vector from the test point px0 
            // to the centroid of the test triangle
            Vector AP = px0 - tx0[tmi][idx1];
            double sep = AP.length2();
            if(sep < 0.25*cell_length2){
              Vector triNormal =Cross(triMidToN0Vec[tmi][idx1],
                                      triMidToN1Vec[tmi][idx1]);
              double tNL = triNormal.length();
              if(tNL>0.0){
                triNormal /= tNL;
              }
              double overlap = Dot(AP,triNormal);
              if(overlap < 0.0 && Dot(ptNormal,triNormal) < -.2){
                // Point is past the plane of the triangle
                numOverlap++;
                triSep.push_back(sep);
                triIndex.push_back(idx1);
                triOverlap.push_back(overlap);
                triTriNormal.push_back(triNormal);
                Point inPlane = px0 - overlap*triNormal;
                triInPlane.push_back(inPlane);
              }    // Point px0 overlaps plane of current triangle
            }  // point is in the neighborhood
          } // inner loop over triangles

          // Sort the triangles according to triSep.
          int aLength = triSep.size(); // initialise to a's length
          int numSorted = min(aLength, 6);

          /* advance the position through the entire array */
          for (int i = 0; i < numSorted-1; i++) {
            /* find the min element in the unsorted a[i .. aLength-1] */

            /* assume the min is the first element */
            int jMin = i;
            /* test against elements after i to find the smallest */
            for (int j = i+1; j < aLength; j++) {
              /* if this element is less, then it is the new minimum */
              if (triSep[j] < triSep[jMin]) {
                  /* found new minimum; remember its index */
                  jMin = j;
              }
            }

            if (jMin != i) {
              swap(triSep[i],        triSep[jMin]);
              swap(triIndex[i],      triIndex[jMin]);
              swap(triOverlap[i],    triOverlap[jMin]);
              swap(triInPlane[i],    triInPlane[jMin]);
              swap(triTriNormal[i],  triTriNormal[jMin]);
            }
          } // for loop over unsorted vector

          // Loop over all triangles that the point px0 overlaps
          for (int i = 0; i < numSorted; i++) {
            //Now, see if that point is inside the triangle or not
            int vecIdx = triIndex[i];
            Point A = tx0[tmi][vecIdx] + triMidToN0Vec[tmi][vecIdx];
            Point B = tx0[tmi][vecIdx] + triMidToN1Vec[tmi][vecIdx];
            Point C = tx0[tmi][vecIdx] + triMidToN2Vec[tmi][vecIdx];
            Vector a = A - triInPlane[i];
            Vector b = B - triInPlane[i];
            Vector c = C - triInPlane[i];
            Vector u = Cross(b,c);
            Vector v = Cross(c,a);
            Vector w = Cross(a,b);
            if(Dot(u,v) >= 0. && Dot(u,w) >= 0.){
              numInside++;
//              triInContact[tmi][closest] = tmo;
              foundOne=true;
//              double Length=((C-B).length()+(B-A).length()+(A-C).length())/3.;
              double Length = sqrt(triAreaAtNodes[tmo][idx0][iu]);
              double K = K_l*Length;
              double forceMag = triOverlap[i]*K;
              // Find the weights for each of the vertices
              // These are actually twice the areas, doesn't matter, dividing
              double areaA = u.length();
              double areaB = v.length();
              double areaC = w.length();
              double totalArea = areaA+areaB+areaC;
              areaA/=totalArea;
              areaB/=totalArea;
              areaC/=totalArea;
              Vector tForceA  = -forceMag*triTriNormal[i]*areaA;
              Vector tForceB  = -forceMag*triTriNormal[i]*areaB;
              Vector tForceC  = -forceMag*triTriNormal[i]*areaC;
              totalContactArea += triAreaAtNodes[tmo][idx0][iu];
              totalContactAreaTri += 0.5*totalArea;

              if(m_output->isOutputTimeStep()){
                // triangle_ids[tmo][idx0] is the triangle that is penetrating
                // iu is the vertex of the penetrating triangle
                // triangle_ids[tmi][vecIdx] is the penetrated triangle
                 fprintf(fp,"%i %i %i %i %ld %ld %i %8.6e %8.6e %8.6e\n",
                 tmo, tmi, adv_matl0, adv_matl1,
                 triangle_ids[tmo][idx0], triangle_ids[tmi][vecIdx],
                 iu, 0.5*totalArea,
                 triAreaAtNodes[tmo][idx0][iu], triMassDisp[tmi][vecIdx]);
                 fflush(fp);
              }

//                cout << "triAreaAtNodes[" << tmo << "][" << idx0 << "][" << iu << "] = " << triAreaAtNodes[tmo][idx0][iu] << endl;
//                cout << "totalAreaA, closest  = " << 0.5*totalArea 
//                     << " "                      << closest << endl;
              totalForce += tForceA;
              totalForce += tForceB;
              totalForce += tForceC;

              // Distribute the force to the grid from the triangles
              // from the triangle vertices.  Use same spatial location
              // for both adv_matls

              // First for Point A
              // Get the node indices that surround the cell
              int NN = interpolator->findCellAndWeights(A, ni, S, size);

              double totMass0 = 0.; double totMass1 = 0.;
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                totMass0 += S[k]*gmass[adv_matl0][node];
                totMass1 += S[k]*gmass[adv_matl1][node];
//               }
              }

              // Accumulate the contribution from each surrounding vertex
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
               if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 // Distribute force according to material mass on the nodes
                 // to get nearly equal contribution to the acceleration
                 LSContForce[adv_matl0][node] += tForceA*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceA*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }

              // Next for Point B
              // Get the node indices that surround the cell
              NN = interpolator->findCellAndWeights(B, ni, S, size);

              totMass0 = 0.; totMass1 = 0.;
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                totMass0 += S[k]*gmass[adv_matl0][node];
                totMass1 += S[k]*gmass[adv_matl1][node];
//               }
              }

              // Accumulate the contribution from each surrounding vertex
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
               if(patch->containsNode(node)) {
                 // Distribute force according to material mass on the nodes
                 // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 LSContForce[adv_matl0][node] += tForceB*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceB*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }

              // Finally for Point C
              // Get the node indices that surround the cell
              NN = interpolator->findCellAndWeights(C, ni, S, size);

              totMass0 = 0.; totMass1 = 0.;
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                totMass0 += S[k]*gmass[adv_matl0][node];
                totMass1 += S[k]*gmass[adv_matl1][node];
//               }
              }

              // Accumulate the contribution from each surrounding vertex
              for (int k = 0; k < NN; k++) {
               IntVector node = ni[k];
               if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 // Distribute force according to material mass on the nodes
                 // to get nearly equal contribution to the acceleration
                 LSContForce[adv_matl0][node] += tForceC*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceC*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }
            }  // inPlane is inside triangle
            if(foundOne){
              break;
            }
          } // loop over overlapped triangles

          if(!foundOne && triIndex.size()>1){
            // check to see if "triInPlane[0]" is inside another triangle
            // that is overlapped
            for (int i = 1; i < numSorted; i++) {
              //Now, see if that point is inside the triangle or not
              int vecIdx = triIndex[i];
              Point A = tx0[tmi][vecIdx] + triMidToN0Vec[tmi][vecIdx];
              Point B = tx0[tmi][vecIdx] + triMidToN1Vec[tmi][vecIdx];
              Point C = tx0[tmi][vecIdx] + triMidToN2Vec[tmi][vecIdx];
              Vector a = A - triInPlane[0];
              Vector b = B - triInPlane[0];
              Vector c = C - triInPlane[0];
              Vector u = Cross(b,c);
              Vector v = Cross(c,a);
              Vector w = Cross(a,b);
              if(Dot(u,v) >= 0. && Dot(u,w) >= 0.){
                numInside++;
                foundOne=true;
                double Length=((C-B).length()+(B-A).length()+(A-C).length())/3.;
                double K = K_l*Length;
                double forceMag = triOverlap[i]*K;
                // Find the weights for each of the vertices
                // These are actually twice the areas, doesn't matter, dividing
                double areaA = u.length();
                double areaB = v.length();
                double areaC = w.length();
                double totalArea = areaA+areaB+areaC;
                areaA/=totalArea;
                areaB/=totalArea;
                areaC/=totalArea;
                Vector tForceA  = -forceMag*triTriNormal[i]*areaA;
                Vector tForceB  = -forceMag*triTriNormal[i]*areaB;
                Vector tForceC  = -forceMag*triTriNormal[i]*areaC;
                totalContactArea += triAreaAtNodes[tmo][idx0][iu];
                totalContactAreaTri += 0.5*totalArea;

                if(m_output->isOutputTimeStep()){
                 fprintf(fp,"%i %i %i %i %ld %ld %i %8.6e %8.6e %8.6e\n",
                 tmo, tmi, adv_matl0, adv_matl1,
                 triangle_ids[tmo][idx0], triangle_ids[tmi][vecIdx],
                 iu, 0.5*totalArea,
                 triAreaAtNodes[tmo][idx0][iu], triMassDisp[tmi][vecIdx]);
                 fflush(fp);
                }

//                cout << "triAreaAtNodes[" << tmo << "][" << idx0 << "][" << iu << "] = " << triAreaAtNodes[tmo][idx0][iu] << endl;
//                cout << "totalAreaA, closest  = " << 0.5*totalArea 
//                     << " "                      << closest << endl;
                totalForce += tForceA;
                totalForce += tForceB;
                totalForce += tForceC;

                // Distribute the force to the grid from the triangles
                // from the triangle vertices.  Use same spatial location
                // for both adv_matls
  
                // First for Point A
                // Get the node indices that surround the cell
                int NN = interpolator->findCellAndWeights(A, ni, S, size);
  
                double totMass0 = 0.; double totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceA*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceA*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
  
                // Next for Point B
                // Get the node indices that surround the cell
                NN = interpolator->findCellAndWeights(B, ni, S, size);
  
                totMass0 = 0.; totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceB*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceB*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
  
                // Finally for Point C
                // Get the node indices that surround the cell
                NN = interpolator->findCellAndWeights(C, ni, S, size);
  
                totMass0 = 0.; totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceC*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceC*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
              } // check dot products
              if(foundOne){
                break;
              }
            } // loop over other nearby triangles
          }  // If multiple overlaps, but penetration point not in triangles
         } // loop over the three vertices of the triangle
        } //  Outer loop over triangles
       }  // if num particles in the inner pset is > 0
      } // inner loop over triangle materials
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch, adv_matl0, "Symmetric", 
                              LSContForce[adv_matl0], "linear");
    } // outer loop over triangle materials
#if 0
    if(totalContactArea > 1.e-22){
      cout << "patchID = " << patch->getID() << endl;
      cout << "numOverlap = " << numOverlap << endl;
      cout << "numInside = " << numInside << endl;
      cout << "totalContactArea = " << time << " " << totalContactArea << endl;
      cout << "totalContactAreaTri = " << time << " " << totalContactAreaTri << endl;
      cout << "totalForce = " << time << " " << totalForce << endl;
    }
#endif
    delete interpolator;
    if(m_output->isOutputTimeStep()){
      fclose(fp);
    }
  } // patches
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
              "Doing MPM::updateCohesiveZones");

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

    Ghost::GhostType  gac = Ghost::AroundCells;
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity[m], lb->gVelocityLabel,dwi, patch, gac, NGN);
      new_dw->get(gmass[m],     lb->gMassLabel,    dwi, patch, gac, NGN);
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
          if(gmass[TopMat][node]>2.e-100){
            velTop      += gvelocity[TopMat][node]* S[k];
            sumSTop     += S[k];
          }
          if(gmass[BotMat][node]>2.e-100){
            velBot      += gvelocity[BotMat][node]* S[k];
            sumSBot     += S[k];
          }
          massTop     += gmass[TopMat][node]*S[k];
          massBot     += gmass[BotMat][node]*S[k];
        }
        velTop/=(sumSTop+1.e-100);
        velBot/=(sumSBot+1.e-100);

        // Update the cohesive zone's position and displacements
        czx_new[idx]         = czx[idx]       + .5*(velTop + velBot)*delT;
        czDispTop_new[idx]   = czDispTop[idx] + velTop*delT;
        czDispBot_new[idx]   = czDispBot[idx] + velBot*delT;
        // This mass check is done in case CZs are placed where one or both
        // of the materials' masses is zero
        if(massTop>1.e-99 && massBot>1.e-99){
          czsep_new[idx]       = czDispTop_new[idx] - czDispBot_new[idx];
        } else {
          czsep_new[idx]       = czsep[idx];
        }

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
        if(czFailed[idx]!=0 ){
          if(disp>=disp_old){
           if(czFailed[idx]>0){
             czFailed_new[idx]=min(czFailed[idx]+1, 1000);
           } else{
             czFailed_new[idx]=max(czFailed[idx]-1,-1000);
           }
          } else {
           czFailed_new[idx]=czFailed[idx];
          }
          czf =.001*fabs((double) czFailed_new[idx]);
        }
        else if(fabs(D_n) > delta_n_fail){
          cout << "czFailed, D_n =  " << D_n << endl;
          czFailed_new[idx]=1;
        }
        else if( fabs(D_t1) > delta_t_fail){
          cout << "czFailed, D_t1 =  " << D_t1 << endl;
          czFailed_new[idx]=-1;
        }
        else if( fabs(D_t2) > delta_t_fail){
          cout << "czFailed, D_t2 =  " << D_t2 << endl;
          czFailed_new[idx]=-1;
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
    printTask(patches, patch,cout_doing, "Doing addParticles");

   //Carry forward CellNAPID
   constCCVariable<int> NAPID;
   CCVariable<int> NAPID_new;
   Ghost::GhostType  gnone = Ghost::None;
   old_dw->get(NAPID,               lb->pCellNAPIDLabel,    0,patch,gnone,0);
   new_dw->allocateAndPut(NAPID_new,lb->pCellNAPIDLabel,    0,patch);
   NAPID_new.copyData(NAPID);

   if(flags->d_doAuthigenesis){
    cout << "Doing addParticles" << endl;

    Vector dx = patch->dCell();
    printTask(patches, patch,cout_doing, "Doing MPM::addParticles");
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    vector<Point> P;
    vector<std::string> filenames;
    vector<double> color,Fcolor;
    vector<double> surface;
    for(unsigned int m = 0; m < numMPMMatls; m++){
      int numNewPartNeeded=0;
      P.clear();
      color.clear();
      Fcolor.clear();

      stringstream mnum;
      mnum << m;
      string mnums = mnum.str();
      string fname=flags->d_authigenesisBaseFilename + mnums;
      cout << "fname  = " << fname  << endl;
      std::ifstream is(fname.c_str());
      filenames.clear();

      if(is) {
       double col;
       std::string filename;
       while(is >> filename >> col){
         filenames.push_back(filename);
         Fcolor.push_back(col);
       }
       is.close();
      } else {
       cout << "No file named  = " << fname  << " can be found." << endl;
       cout << "That's OK if there isn't new geometry for matl. " << m << endl;
      }

      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gan = Ghost::AroundNodes;
      ParticleSubset* pset    = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* pset_wg = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      ParticleVariable<Point> px;
      ParticleVariable<Matrix3> pF,pSize,pstress,pvelgrad,pscalefac;
      ParticleVariable<long64> pids;
      ParticleVariable<double> pvolume,pmass,ptemp,ptempP,pcolor;
      ParticleVariable<double> pSurf;
      ParticleVariable<Vector> pvelocity,pextforce,pdisp,ptempgrad;
      ParticleVariable<int> pref,ploc,prefOld,pModID;
      ParticleVariable<IntVector> pLoadCID;
      new_dw->getModifiable(px,       lb->pXLabel_preReloc,            pset);
      new_dw->getModifiable(pids,     lb->pParticleIDLabel_preReloc,   pset);
      new_dw->getModifiable(pModID,   lb->pModalIDLabel_preReloc,      pset);
      new_dw->getModifiable(pmass,    lb->pMassLabel_preReloc,         pset);
      new_dw->getModifiable(pSize,    lb->pSizeLabel_preReloc,         pset);
      new_dw->getModifiable(pSurf,    lb->pSurfLabel_preReloc,         pset);
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
      new_dw->getModifiable(ploc,     lb->pLocalizedMPMLabel_preReloc, pset);
      new_dw->getModifiable(pvelgrad, lb->pVelGradLabel_preReloc,      pset);
      new_dw->getModifiable(pF,  lb->pDeformationMeasureLabel_preReloc,pset);
      if (flags->d_with_color) {
        new_dw->getModifiable(pcolor, lb->pColorLabel_preReloc,        pset);
      }
      if (flags->d_useLoadCurves) {
        new_dw->getModifiable(pLoadCID,lb->pLoadCurveIDLabel_preReloc, pset);
      }

      // Need these to determine if new particles are inside of old ones
      constParticleVariable<Point>   px_wg;
      constParticleVariable<Matrix3> pF_wg,pSize_wg;
      constParticleVariable<double>  pColor_wg;

      old_dw->get(px_wg,              lb->pXLabel,                  pset_wg);
      old_dw->get(pF_wg,              lb->pDeformationMeasureLabel, pset_wg);
      old_dw->get(pSize_wg,           lb->pSizeLabel,               pset_wg);
      old_dw->get(pColor_wg,          lb->pColorLabel,              pset_wg);

      int PaPeCe = 2;  // This value should perhaps be a run time option
      vector<GeometryPieceP> newGeomPiece;
      for(unsigned int ifile = 0; ifile<filenames.size(); ifile++){
       GeometryPieceP nGP = nullptr;
       nGP = scinew TriGeometryPiece(filenames[ifile]);
       newGeomPiece.push_back(nGP);
 
       ParticleCreator *pc;
       pc = scinew ParticleCreator();
 
       IntVector ppc(PaPeCe,PaPeCe,PaPeCe);
       Vector dxpp = patch->dCell()/ppc;
       Vector dcorner = dxpp*0.5;
 
       for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        Point lower = patch->nodePosition(*iter) + dcorner;
        IntVector c = *iter;

        for(int ix=0; ix < ppc.x(); ix++){
         for(int iy=0; iy < ppc.y(); iy++){
          for(int iz=0; iz < ppc.z(); iz++){
            IntVector idx(ix, iy, iz);
            Point p = lower + dxpp*idx;
            if(newGeomPiece[ifile]->inside(p)){
              // See if new point lies within an existing particle
              bool inExisting = false;
//              cout << "Point inside of new geometry" << endl;
//              cout << "Fcolor =  " << Fcolor[ifile] << endl;
              for(ParticleSubset::iterator piter  = pset_wg->begin();
                                           piter != pset_wg->end(); piter++){
                particleIndex pidx = *piter;
                Matrix3 dsize = (pF_wg[pidx]*(Matrix3(dx[0],0,0,
                                                      0,dx[1],0,
                                                      0,0,dx[2])
                                             *pSize_wg[pidx]));
                if(Fcolor[ifile]==pColor_wg[pidx]){
                  inExisting = isPointInExistingParticle(dsize, p, px_wg[pidx]);
                }

                // Once we determine that the current test point is inside
                // an existing particle, no need to check other particles.
                if(inExisting){
                  break;
                }
              }  // loop over existing particles in the patch
              if(!inExisting){
                 P.push_back(p);
                 color.push_back(Fcolor[ifile]);
                 double isurf=((double) pc->checkForSurface(newGeomPiece[ifile],
                                                            p, dxpp,
                                                            flags->d_ndim));
                 surface.push_back(isurf);
                 numNewPartNeeded++;
              }
            } // if inside of new geometry
          }  // z
         }  // y
        }  // x
       }  // CellIterator

       if(numNewPartNeeded > 0 ){
         cout << "numNewPartNeeded = " << numNewPartNeeded << endl;
       }

      }

      int fourOrEight=pow(PaPeCe,flags->d_ndim);
      double fourthOrEighth = 1./((double) fourOrEight);

      const unsigned int oldNumPar = pset->addParticles(numNewPartNeeded);

      ParticleVariable<Point> pxtmp;
      ParticleVariable<Matrix3> pFtmp,psizetmp,pstrstmp,pvgradtmp,pSFtmp;
      ParticleVariable<long64> pidstmp;
      ParticleVariable<double> psurftmp;
      ParticleVariable<double> pvoltmp, pmasstmp,ptemptmp,ptempPtmp,pcolortmp;
      ParticleVariable<Vector> pveltmp,pextFtmp,pdisptmp,ptempgtmp;
      ParticleVariable<int> preftmp,ploctmp,pMIDtmp;
      ParticleVariable<IntVector> pLoadCIDtmp;
      new_dw->allocateTemporary(pidstmp,  pset);
      new_dw->allocateTemporary(pMIDtmp,pset);
      new_dw->allocateTemporary(pxtmp,    pset);
      new_dw->allocateTemporary(pvoltmp,  pset);
      new_dw->allocateTemporary(pveltmp,  pset);
      new_dw->allocateTemporary(psurftmp, pset);
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
      new_dw->allocateTemporary(ploctmp,  pset);
      new_dw->allocateTemporary(pvgradtmp,pset);
      if (flags->d_with_color) {
        new_dw->allocateTemporary(pcolortmp,pset);
      }

      if (flags->d_useLoadCurves) {
        new_dw->allocateTemporary(pLoadCIDtmp,  pset);
      }

      // copy data from old variables for particle IDs and the position vector
      int counter = 0;
      for( unsigned int pp=0; pp<oldNumPar; ++pp ){
        pidstmp[pp]  = pids[pp];
        pMIDtmp[pp]  = pModID[pp];
        pxtmp[pp]    = px[pp];
        psurftmp[pp] = pSurf[pp];

        for(unsigned int ifile = 0; ifile<filenames.size(); ifile++){
          if(psurftmp[pp]>0){
            Vector RNL[8];
            // Compute R-vectors from particle center to the corners
            Matrix3 dsize = (pF[pp]*(Matrix3(dx[0],0,0,
                                             0,dx[1],0,
                                             0,0,dx[2])
                                         *pSize[pp]));
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
            // If any of the corners of the original particles are outside of
            // the overgrowth geometry, it is a surface, otherwise it is not.
            for(unsigned int ir = 0; ir<8; ir++){
              if(!newGeomPiece[ifile]->inside(pxtmp[pp]+RNL[ir])){
                psurftmp[pp]=2.0;
                counter++;
              }
            }
            if(psurftmp[pp]==2.0){
              psurftmp[pp]=1.0;
            } else {
              psurftmp[pp]=0.0;
            }
          }
        }

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
        ploctmp[pp]  = ploc[pp];
        pvgradtmp[pp]= pvelgrad[pp];
      }

      double cellVol = dx.x()*dx.y()*dx.z();
      double rho = mpm_matl->getInitialDensity();
      int matModalID = mpm_matl->getModalID();
      Matrix3 Id;
      Id.Identity();
      if(numNewPartNeeded>0){
        for(int i = 0;i<numNewPartNeeded;i++){
          IntVector c_orig;
          patch->findCell(P[i],c_orig);
          long64 cellID = ((long64)c_orig.x() << 16) |
                          ((long64)c_orig.y() << 32) |
                          ((long64)c_orig.z() << 48);

          int& myCellNAPID = NAPID_new[c_orig];
          int new_index=oldNumPar+i;
          pidstmp[new_index]    = (cellID | (long64) myCellNAPID);
          pMIDtmp[new_index]    = matModalID;
          pxtmp[new_index]      = P[i];
          pvoltmp[new_index]    = fourthOrEighth*cellVol;
          pmasstmp[new_index]   = rho*pvoltmp[new_index];
          pveltmp[new_index]    = Vector(0.0,0.0,0.0);
          if (flags->d_useLoadCurves) {
            pLoadCIDtmp[new_index]  = 0;
          }
          if (flags->d_with_color) {
            pcolortmp[new_index]  = color[i];
          }
          //double zz_size = min(1.0,1./(((double) flags->d_ndim)-1.));
          //psizetmp[new_index] = Matrix3(0.5,0.0,0.0,
          //                              0.0,0.5,0.0,
          //                              0.0,0.0,zz_size);
          double sz_new = (1.0)/((double) PaPeCe);
          psizetmp[new_index] = Matrix3(sz_new,0.0,0.0,
                                        0.0,sz_new,0.0,
                                        0.0,0.0,sz_new);

          if(flags->d_computeScaleFactor){
            pSFtmp[new_index]   = Matrix3(dx.x(),0.0,0.0,
                                          0.0,dx.y(),0.0,
                                          0.0,0.0,dx.z())*psizetmp[new_index];
          }

          pextFtmp[new_index]   = Vector(0.0);
          pFtmp[new_index]      = Id;
          pdisptmp[new_index]   = Vector(0.0);
          pstrstmp[new_index]   = Matrix3(0.0);
          ptemptmp[new_index]   = 300.;
          ptempgtmp[new_index]  = Vector(0.0);
          ptempPtmp[new_index]  = 300.;
          psurftmp[new_index]   = surface[i];
          ploctmp[new_index]    = 0.;
          pvgradtmp[new_index]  = Matrix3(0.0);
          NAPID_new[c_orig]++;
        }
      } // if any particles flagged for refinement

      cm->splitCMSpecificParticleData(patch, dwi, fourOrEight, prefOld, pref,
                                      oldNumPar, numNewPartNeeded,
                                      old_dw, new_dw);

      // put back temporary data
      new_dw->put(pidstmp,  lb->pParticleIDLabel_preReloc,           true);
      new_dw->put(pMIDtmp,  lb->pModalIDLabel_preReloc,              true);
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
      new_dw->put(psurftmp, lb->pSurfLabel_preReloc,                 true);
      new_dw->put(pdisptmp, lb->pDispLabel_preReloc,                 true);
      new_dw->put(pstrstmp, lb->pStressLabel_preReloc,               true);
      if (flags->d_with_color) {
        new_dw->put(pcolortmp,lb->pColorLabel_preReloc,              true);
      }

      if (flags->d_useLoadCurves) {
        new_dw->put(pLoadCIDtmp,lb->pLoadCurveIDLabel_preReloc,      true);
      }
      new_dw->put(pFtmp,    lb->pDeformationMeasureLabel_preReloc,   true);
      new_dw->put(ploctmp,  lb->pLocalizedMPMLabel_preReloc,         true);
      new_dw->put(pvgradtmp,lb->pVelGradLabel_preReloc,              true);
    }  // for matls
   }    // if doAuth && AddedNewParticles<1.0....
  }   // for patches
//   flags->d_doAuthigenesis = false;
}

void SerialMPM::addTracers(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* ,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing addTracers");

    if(flags->d_doAuthigenesis){
      cout << "Doing addTracers" << endl;

      int tm = 0;  // Only one tracer material now
      int numNewTracers=0;

      TracerMaterial* tr_matl = (TracerMaterial*) 
                                   m_materialManager->getMaterial("Tracer", tm);
      int dwi = tr_matl->getDWIndex();
      Tracer* tr = tr_matl->getTracer();

      string filename = tr_matl->getTracerFilename();
      numNewTracers = tr->countTracers(patch,filename);

//      cout << "numNewTracers = " << numNewTracers << endl;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      ParticleVariable<Point> px;
      ParticleVariable<long64> pids;
      new_dw->getModifiable(px,         lb->pXLabel_preReloc,           pset);
      new_dw->getModifiable(pids,     TraL->tracerIDLabel_preReloc,     pset);

      ParticleSubset* psetnew = 
                new_dw->createParticleSubset(numNewTracers,dwi,patch);

      ParticleVariable<Point> pxtmp;
      ParticleVariable<long64> pidstmp;
      new_dw->allocateTemporary(pidstmp,  psetnew);
      new_dw->allocateTemporary(pxtmp,    psetnew);

      std::ifstream is(filename.c_str());

      double p1,p2,p3;
      string line;
      particleIndex start = 0;
      while (getline(is, line)) {
       istringstream ss(line);
       string token;
       long64 tid;
       ss >> token;
       tid = stoull(token);
       ss >> token;
       p1 = stof(token);
       ss >> token;
       p2 = stof(token);
       ss >> token;
       p3 = stof(token);
//     cout << tid << " " << p1 << " " << p2 << " " << p3 << endl;
       Point pos = Point(p1,p2,p3);

       if(patch->containsPoint(pos)){
         particleIndex pidx = start;
         pxtmp[pidx]   = pos;
         pidstmp[pidx] = tid;
         start++;
       }
      }

      is.close();

      // put back temporary data
      new_dw->put(pxtmp,      lb->pXLabel_preReloc,             true);
      new_dw->put(pidstmp,  TraL->tracerIDLabel_preReloc,       true);
   }    // if doAuth && AddedNewParticles<1.0....
  }   // for patches
//   flags->d_doAuthigenesis = false;
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
    printTask(patches, patch,cout_doing, "Doing MPM::computeParticleScaleFactor");

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
    } // loop over MPM matls
  } // patches
}

void SerialMPM::computeLineSegScaleFactor(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  // This task computes the particles physical size, to be used
  // in scaling particles for the deformed particle vis feature

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing MPM::computeLineSegScaleFactor");

    unsigned int numLSMatls=m_materialManager->getNumMatls( "LineSegment" );
    for(unsigned int m = 0; m < numLSMatls; m++){
      LineSegmentMaterial* ls_matl = 
        (LineSegmentMaterial*) m_materialManager->getMaterial("LineSegment", m);
      int dwi = ls_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(m_output->isOutputTimeStep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = ((Matrix3(dx[0],0,0,
                                        0,dx[1],0,
                                        0,0,dx[2])*psize[idx]));
        } // for particles
      } // isOutputTimestep
    } // loop over LineSegment matls
  } // patches
}

void SerialMPM::computeTriangleScaleFactor(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // This task computes the particles initial physical size, to be used
  // in scaling particles for the deformed particle vis feature

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing MPM::computeTriangleScaleFactor");

    unsigned int numLSMatls=m_materialManager->getNumMatls( "Triangle" );
    for(unsigned int m = 0; m < numLSMatls; m++){
      LineSegmentMaterial* ls_matl = 
        (LineSegmentMaterial*) m_materialManager->getMaterial("Triangle", m);
      int dwi = ls_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(m_output->isOutputTimeStep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = ((Matrix3(dx[0],0,0,
                                        0,dx[1],0,
                                        0,0,dx[2])*psize[idx]));
        } // for particles
      } // isOutputTimestep
    } // loop over Triangle matls
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
  std::vector<NCVariable<Vector> >       gvelocity(numMPMMatls);
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

    // Keep the standard surface normal code around for now
    // Needed to compute the normal traction
    // At this point, everything from here on is only for diagnostics
    printTask(patches, patch, cout_doing, "Doing MPM::computeNormals");

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

void SerialMPM::scheduleComputeNormalsTri(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSubset* triangle_matls,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeNormalsTri");

  Task* t=scinew Task("MPM::computeNormalsTri",
                      this, &SerialMPM::computeNormalsTri);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->pXLabel,              triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triNormalLabel,     triangle_matls, gac, 2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,      gac,NGN+3);

  t->computes(lb->gSurfNormLabel,                    mpm_matls);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void SerialMPM::computeNormalsTri(const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset * ,
                                        DataWarehouse  * old_dw,
                                        DataWarehouse  * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeNormalsTri");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    Ghost::GhostType gan   = Ghost::AroundNodes;

    unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> >       gsurfnorm(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->allocateAndPut(gsurfnorm[m],    lb->gSurfNormLabel,    dwi,patch);
      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));
    }

    int numTriMatls=m_materialManager->getNumMatls("Triangle");
    Matrix3 size; size.Identity(); size*=0.5;

    for(int tmo = 0; tmo < numTriMatls; tmo++) {
      TriangleMaterial* t_matl = (TriangleMaterial *) 
                             m_materialManager->getMaterial("Triangle", tmo);
      int dwi_tri = t_matl->getDWIndex();
      int adv_matl = t_matl->getAssociatedMaterial();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi_tri, patch,
                                                        gan, 2, lb->pXLabel);
      constParticleVariable<Point>  tx;
      constParticleVariable<Vector> triNormal;
      old_dw->get(tx,         lb->pXLabel,            pset);
      old_dw->get(triNormal,  TriL->triNormalLabel,   pset);

      for(ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
         particleIndex idx = *iter;
         int nn = interpolator->findCellAndWeights(tx[idx], ni, S, size);
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)){
             gsurfnorm[adv_matl][node] += triNormal[idx]*S[k];
           }
         }
      } // triangles
    }   // triangle materials

    for(unsigned int m = 0; m < numMPMMatls; m++){
      for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        gsurfnorm[m][c] /= (gsurfnorm[m][c].length()+1.e-100);
      } // Nodes
    }   // MPM materials
  }     // patches
}

void SerialMPM::scheduleComputeGridCemVec(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSubset* tracer_matls,
                                          const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeGridCemVec");

  Task* t=scinew Task("MPM::computeGridCemVec",
                      this, &SerialMPM::computeGridCemVec);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->pXLabel,              tracer_matls, gac, 2);
  t->requires(Task::OldDW, TraL->tracerCemVecLabel,  tracer_matls, gac, 2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,    gac,NGN+3);

  t->computes(lb->gCemVecLabel,                      mpm_matls);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void SerialMPM::computeGridCemVec(const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset * ,
                                        DataWarehouse  * old_dw,
                                        DataWarehouse  * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeGridCemVec");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    Ghost::GhostType gan   = Ghost::AroundNodes;

    unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> >       gcemvec(numMPMMatls);
    std::vector<NCVariable<double> >       SumS(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->allocateAndPut(gcemvec[m],    lb->gCemVecLabel,  dwi, patch);
      new_dw->allocateTemporary(SumS[m],                            patch);
      gcemvec[m].initialize(Vector(0.0,0.0,0.0));
      SumS[m].initialize(0.0);
    }

    int numTraMatls=m_materialManager->getNumMatls("Tracer");
    Matrix3 size; size.Identity();

    for(int tmo = 0; tmo < numTraMatls; tmo++) {
      TracerMaterial* t_matl = (TracerMaterial *) 
                             m_materialManager->getMaterial("Tracer", tmo);
      int dwi_tra = t_matl->getDWIndex();
      int adv_matl = t_matl->getAssociatedMaterial();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi_tra, patch,
                                                        gan, 2, lb->pXLabel);
      constParticleVariable<Point>  tx;
      constParticleVariable<Vector> tracerCemVec;
      old_dw->get(tx,               lb->pXLabel,             pset);
      old_dw->get(tracerCemVec,   TraL->tracerCemVecLabel,   pset);

      for(ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
         particleIndex idx = *iter;
         int nn = interpolator->findCellAndWeights(tx[idx], ni, S, size);
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)){
             gcemvec[adv_matl][node] += tracerCemVec[idx]*S[k];
             SumS[adv_matl][node]    += S[k];
           }
         }
      } // triangles
    }   // triangle materials

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                   (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        gcemvec[m][c] /= (SumS[m][c]+1.e-100);
      } // Nodes

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gcemvec[m],interp_type);
    }   // MPM materials
  }     // patches
}
//______________________________________________________________________
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
    }

    // Here, find out two things:
    // 1.  How many materials have mass on a node
    // 2.  Which material has the most mass on a node.  That is the alpha matl.
    for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      double maxMass=-9.e99;
      for(unsigned int m = 0; m < numMPMMatls; m++){
        if(gmass[m][c] > 1.e-8*gmassglobal[c] && gmass[m][c] > 1.e-16){
          NumMatlsOnNode[c]++;
          if(gmass[m][c]>maxMass){
            // This is the alpha material, all other matls are beta
            alphaMaterial[c]=m;
            maxMass=gmass[m][c];
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
          ParticleList[m][*it1][ParticleList[m][*it1][399]]=idx;
          ParticleList[m][*it1][399]++;
          NumParticlesOnNode[*it1]++;
        }
        nodeList.clear();
      }    // Loop over Particles
    }

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
        double Rp = 0.25*dx.x();
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

  t->requires(Task::OldDW, lb->timeStepLabel);
  t->requires(Task::OldDW, lb->pXLabel,                  gp, ngc_p);
  t->requires(Task::OldDW, lb->pSurfLabel,               gp, ngc_p);
  t->requires(Task::OldDW, lb->pColorLabel,              gp, ngc_p);
  t->requires(Task::OldDW, lb->pParticleIDLabel,         gp, ngc_p);

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
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  unsigned int numMPMMatls = m_materialManager->getNumMatls("MPM");

  timeStep_vartype timeStep;
  old_dw->get(timeStep, lb->timeStepLabel);
  int timestep = timeStep;

  // Should we make this an input file parameter?
  int interval=INT_MAX;
  if (flags->d_doingDissolution){
     interval = 20;
  }

  int doit=timestep%interval;

//  const Level* level = getLevel(patches);
//  IntVector low, hi;
//  IntVector periodic=level->getPeriodicBoundaries();
//  level->findNodeIndexRange(low, hi);

  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    printTask(patches, patch, cout_doing, "Doing findSurfaceParticles");

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial*  mpm_matl  =
                       (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();
      bool needSurfaceParticles = mpm_matl->getNeedSurfaceParticles();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);
      ParticleSubset* psetOP = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Point> px, pxOP;
      constParticleVariable<double> pSurfOld, pcolor, pcolorOP;
      ParticleVariable<double> pSurf;
      constParticleVariable<long64> pids, pidsOP;

      old_dw->get(px,                  lb->pXLabel,                  pset);
      old_dw->get(pids,                lb->pParticleIDLabel,         pset);
      old_dw->get(pcolor,              lb->pColorLabel,              pset);

      old_dw->get(pxOP,                lb->pXLabel,                  psetOP);
      old_dw->get(pidsOP,              lb->pParticleIDLabel,         psetOP);
      old_dw->get(pcolorOP,            lb->pColorLabel,              psetOP);
      old_dw->get(pSurfOld,            lb->pSurfLabel,               psetOP);

      new_dw->allocateAndPut(pSurf,    lb->pSurfLabel_preReloc,      psetOP);

      double cellVol=dx.x()*dx.y()*dx.z();
      double tol = 0.100*cbrt(cellVol);
      int nclose = 2*flags->d_ndim;

      // Either carry forward the particle surface data, or recompute it every
      // N timesteps.
      if(timestep==1 || doit!=1 || !needSurfaceParticles){
        // Carry forward particle surface information
        for (ParticleSubset::iterator iter = psetOP->begin();
             iter != psetOP->end();
             iter++){
           particleIndex idx = *iter;
           pSurf[idx]=pSurfOld[idx];
         }
      } else {
        // Find new particle surface information
        for (ParticleSubset::iterator iter = psetOP->begin();
             iter != psetOP->end();
             iter++){
          particleIndex idx = *iter;
#if 0
          // Determine if particle is in a domain boundary cell
          bool onBoundary = false;
          IntVector cI = patch->findCell(px[idx],cI);
          int lowx=low.x(); int hix=hi.x();
          int lowy=low.y(); int hiy=hi.y();
          int lowz=low.z(); int hiz=hi.z();
          if(periodic.x()==0){
             lowx++; hix-=3;
          }
          if(periodic.y()==0){
             lowy++; hiy-=3;
          }
          if(periodic.z()==0){
             lowz++; hiz-=3;
          }
          if(cI.x()==lowx || cI.x()==hix){
            onBoundary=true;
          }
          if(flags->d_ndim>1){
            if(cI.y()==lowy || cI.y()==hiy){
              onBoundary=true;
            }
            if(flags->d_ndim>2){
              if(cI.z()==lowz || cI.z()==hiz){
                onBoundary=true;
              }
            }
          }

          if(pSurfOld[idx]>0.99 || onBoundary){
#endif
          if(pSurfOld[idx]>0.99){
           pSurf[idx]=pSurfOld[idx];
          } else {
          vector<particleIndex> close(nclose);
          vector<double> closestSep(nclose);
          for(int i=0;i<nclose;i++){
            close[i]=-999;
            closestSep[i]=9.e99;
          }
          for (ParticleSubset::iterator iter2 = pset->begin();
             iter2 != pset->end();
             iter2++){
            particleIndex idx2 = *iter2;
            double sep;
            int howclose;
            // Check particles that are NOT the current particle but are
            // in the same grain
            if(pidsOP[idx]!=pids[idx2] && pcolorOP[idx]==pcolor[idx2]){
              sep = (pxOP[idx]-px[idx2]).length2();
              if(sep<closestSep[nclose-1]){
                howclose=nclose-1;
                for(int j=nclose-2;j>=0;j--){
                  if(sep<closestSep[j]){
                    howclose=j;
                  }  // endif
                }
                for(int k=nclose-2;k>=howclose;k--){
                  closestSep[k+1]=closestSep[k];
                  close[k+1]=close[k];
                }
                closestSep[howclose]=sep;
                close[howclose]=idx2;
              }  // if the inner particle is closer than the current least close
            } // if the outer particle isn't the same as the inner particle
          } // Inner particle loop

          // Get the centroid of the nearest neighbors
          Vector neighborCent(0.,0.,0.);
          for(int i=0;i<nclose;i++){
            neighborCent+= px[close[i]].asVector();
          }
          neighborCent/=((double) nclose);

          // Compare centroid of neighbors to the location of current particle
          Vector posDiff = pxOP[idx].asVector() - neighborCent;
          if(posDiff.length() < tol){
            pSurf[idx] = pSurfOld[idx];
          }else{
            pSurf[idx] = 1.0;
          }
         } // if particle is/is not already a surface particle
        } // outer loop over particles
      }
    }   // matl loop
  }    // patches
}

bool SerialMPM::isPointInExistingParticle(Matrix3 dsize,Point p, Point px)
{

  bool inExisting = false;

  // Compute R-vectors from particle center to the corners
  Vector RNLA = Vector(-dsize(0,0)-dsize(0,1)+dsize(0,2),
                       -dsize(1,0)-dsize(1,1)+dsize(1,2),
                       -dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  Vector RNLB = Vector( dsize(0,0)-dsize(0,1)+dsize(0,2),
                        dsize(1,0)-dsize(1,1)+dsize(1,2),
                        dsize(2,0)-dsize(2,1)+dsize(2,2))*0.5;
  Vector RNLC = Vector( dsize(0,0)+dsize(0,1)+dsize(0,2),
                        dsize(1,0)+dsize(1,1)+dsize(1,2),
                        dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  Vector RNLD = Vector(-dsize(0,0)+dsize(0,1)+dsize(0,2),
                       -dsize(1,0)+dsize(1,1)+dsize(1,2),
                       -dsize(2,0)+dsize(2,1)+dsize(2,2))*0.5;
  Vector RNLE = Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                       -dsize(1,0)-dsize(1,1)-dsize(1,2),
                       -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
  Vector RNLF = Vector( dsize(0,0)-dsize(0,1)-dsize(0,2),
                        dsize(1,0)-dsize(1,1)-dsize(1,2),
                        dsize(2,0)-dsize(2,1)-dsize(2,2))*0.5;
//  Vector RNLG = Vector( dsize(0,0)+dsize(0,1)-dsize(0,2),
//                        dsize(1,0)+dsize(1,1)-dsize(1,2),
//                        dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;
  Vector RNLH = Vector(-dsize(0,0)+dsize(0,1)-dsize(0,2),
                       -dsize(1,0)+dsize(1,1)-dsize(1,2),
                       -dsize(2,0)+dsize(2,1)-dsize(2,2))*0.5;

  // Find the outward normals for all 6 faces of a particle
  Vector normal[6];
  normal[0]=Cross(RNLB-RNLA,RNLD-RNLA);
  normal[1]=Cross(RNLH-RNLE,RNLF-RNLE);
  normal[2]=Cross(RNLE-RNLA,RNLB-RNLA);
  normal[3]=Cross(RNLC-RNLD,RNLH-RNLD);
  normal[4]=Cross(RNLD-RNLA,RNLE-RNLA);
  normal[5]=Cross(RNLF-RNLB,RNLC-RNLB);
  // Unitize the normal vectors
  for(int itest = 0; itest<6; itest++){
    normal[itest]=normal[itest]/normal[itest].length();
  }
  // Dot the normal to each particle face with a vector
  // from the candidate point to a point on that face.
  // If the dot product is < 0, then it is on the "inside"
  // of that face.

  // Currently using a "zero" that is slightly positive
  // so that points that end up on an edge are considered inside.
  // I would like to find a better way...
  if( (Dot(p - (px + RNLA), normal[0]) <= 0.001  &&
       Dot(p - (px + RNLE), normal[1]) <= 0.001  &&
       Dot(p - (px + RNLA), normal[2]) <= 0.001  &&
       Dot(p - (px + RNLD), normal[3]) <= 0.001  &&
       Dot(p - (px + RNLA), normal[4]) <= 0.001  &&
       Dot(p - (px + RNLB), normal[5]) <= 0.001)){
    inExisting = true;
  }

  return inExisting;
}

//
void SerialMPM::scheduleFindGrainCollisions(SchedulerP   & sched,
                                           const PatchSet * patches,
                                           const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"SerialMPM::scheduleFindGrainCollisions");
  
  Task* t = scinew Task("MPM::findGrainCollisions", this, 
                        &SerialMPM::findGrainCollisions);

  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gan   = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pXLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pSurfLabel,               gan, NGP);
  t->requires(Task::OldDW, lb->pColorLabel,              gan, NGP);
  t->requires(Task::NewDW, lb->pCurSizeLabel,            gan, NGP);

  t->requires(Task::NewDW, lb->gColorLabel,              gnone);
  t->computes(lb->TotalLocalizedParticleLabel);

  sched->addTask(t, patches, matls);

  Task* t2 = scinew Task("MPM::communicateGrainCollisions", this, 
                         &SerialMPM::communicateGrainCollisions);

//  t2->setType( Task::OncePerProc );

  t2->requires(Task::NewDW, lb->TotalLocalizedParticleLabel);
  sched->addTask(t2, patches, matls);

}


void SerialMPM::findGrainCollisions(const ProcessorGroup *,
                                    const PatchSubset    * patches,
                                    const MaterialSubset * ,
                                          DataWarehouse  * old_dw,
                                          DataWarehouse  * new_dw)
{
//  timeStep_vartype timeStep;
//  old_dw->get(timeStep, lb->timeStepLabel);
//  int timestep = timeStep;

  // Should we make this an input file parameter?
//  int interval=INT_MAX;

//  int doit=timestep%interval;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int patchID =  patch->getID();

    printTask(patches,patch,cout_doing,
              "Doing SerialMPM::findGrainCollisions");

   if(flags->d_changeGrainMaterials){
    proc0cout << "looking for collisions" << endl;
    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = flags->d_interpolator_type;

    Ghost::GhostType gnone = Ghost::None;
    Ghost::GhostType  gan   = Ghost::AroundNodes;

    set<double> collideColors;

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) m_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pColor;
      constParticleVariable<Matrix3> psize, pFOld;
      constNCVariable<double>        gColor;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pColor,       lb->pColorLabel,                     pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);
      new_dw->get(gColor,       lb->gColorLabel,      dwi, patch, gnone, 0);

      cout.precision(15);
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end();
           iter++){
        particleIndex idx = *iter;
        int NN =
           interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);

        // Add each particles contribution to the local mass & velocity
        // Must use the node indices
        IntVector node;
        // Iterate through the nodes that receive data from the current particle
        bool collides = false;
        for(int k = 0; k < NN; k++) {
          node = ni[k];
          if(patch->containsNode(node) && S[k]>0.0) {
            if(fabs(pColor[idx]-gColor[node])>1.e-8){
              collides = true;
            }
          }
          if(collides){
             collideColors.insert(pColor[idx]);
          }
        }
      } // End of particle loop
    }
#if 0
    for (set<double>::iterator it1 = collideColors.begin(); 
                               it1!= collideColors.end();  it1++){
      cout << "Color " << *it1 << " collides " << endl;
    }
#endif

    ostringstream pnum; pnum << patchID;
    string filename = "collideColors." + pnum.str();
    ofstream colout(filename.c_str());
    if(!colout){
      cerr << "file not opened:  " << filename << endl;
      cerr << "exiting" << endl;
      exit(1);
    }
    colout.precision(12);
    for (set<double>::iterator it1 = collideColors.begin(); 
                               it1!= collideColors.end();  it1++){
      colout << *it1 << endl;
    }
    colout.close();

   } // If this stuff hasn't already been done.
   new_dw->put(sum_vartype(1),     lb->TotalLocalizedParticleLabel);
  }
}

void SerialMPM::communicateGrainCollisions(const ProcessorGroup * pg,
                                           const PatchSubset    * patches,
                                           const MaterialSubset * ,
                                                 DataWarehouse  * old_dw,
                                                 DataWarehouse  * new_dw)
{
   int numPatches = 0;
   sum_vartype TLP;
   new_dw->get(TLP,   lb->TotalLocalizedParticleLabel);
   for(int p=0;p<patches->size();p++){
     const Patch* patch = patches->get(p);
     numPatches = patch->getLevel()->numPatches();

     printTask(patches,patch,cout_doing,
               "Doing SerialMPM::communicateGrainCollisions");
   }
   if(flags->d_changeGrainMaterials){
    for(int p=0;p<numPatches;p++){
     ostringstream pnum; pnum << p;
     string filename = "collideColors." + pnum.str();
     ifstream colin(filename.c_str());
      if(!colin){
       cerr << "file not opened:  " << filename << endl;
       cerr << "exiting" << endl;
       exit(1);
     }

     double cc;
     while(colin >> cc){
       d_collideColors.insert(cc); 
     }
     colin.close();
    }

    for (set<double>::iterator it1 = d_collideColors.begin(); 
                               it1!= d_collideColors.end();  it1++){
     proc0cout << "Color " << *it1 << " collides " << endl;
    }
   } // If I'm doing this

//   cout << "In Communicate Grain Collisions" << endl;
//   int rank = pg->myRank();
//   ReduceSet(rank);
}

void SerialMPM::scheduleChangeGrainMaterials(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)

{
  if( !flags->doMPMOnLevel( getLevel(patches)->getIndex(), 
      getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule( patches, cout_doing, "MPM::scheduleChangeGrainMaterials" );

  Task * t = scinew Task("MPM::changeGrainMaterials", this, 
                   &SerialMPM::changeGrainMaterials);

  t->modifies(lb->pParticleIDLabel_preReloc);
  t->modifies(lb->pModalIDLabel_preReloc);
  t->modifies(lb->pXLabel_preReloc);
  t->modifies(lb->pVolumeLabel_preReloc);
  t->modifies(lb->pVelocityLabel_preReloc);
  t->modifies(lb->pMassLabel_preReloc);
  t->modifies(lb->pSurfLabel_preReloc);
  t->modifies(lb->pSurfLabel_preReloc);
  t->modifies(lb->pdTdtLabel);
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
  if(flags->d_computeScaleFactor){
    t->modifies(lb->pScaleFactorLabel_preReloc);
  }
  t->modifies(lb->pVelGradLabel_preReloc);

#if 0
  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
  }
#endif

  sched->addTask(t, patches, matls);
}

void SerialMPM::changeGrainMaterials(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing changeGrainMaterials");

   if(flags->d_changeGrainMaterials){
    proc0cout << "Doing changeGrainMaterials" << endl;

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    unsigned int numAcceptorMaterials = flags->d_acceptorMaterialIndex.size();
    for(unsigned int aMI_index = 0; 
                     aMI_index < numAcceptorMaterials; 
                     aMI_index++){
      unsigned int aMI = flags->d_acceptorMaterialIndex[aMI_index];

      vector<double> donorColors;
      donorColors.clear();
      int d_CCIndex=0;
      for (set<double>::iterator it1 = d_collideColors.begin(); 
                                 it1!= d_collideColors.end();  it1++){
        if(d_CCIndex%(numAcceptorMaterials+1)==aMI_index){
         donorColors.push_back(*it1);
        }
        d_CCIndex++;
      }

      // Loop over all materials, except for acceptor, look for particles with
      // color = donorColor. If found, copy particle data into acceptor matl and
      // then set pLocalized to -999 so that finalizeParticles will remove it.

      MPMMaterial* ami_matl =
                     (MPMMaterial*) m_materialManager->getMaterial( "MPM", aMI);
      int dw_ami = ami_matl->getDWIndex();
      ParticleSubset* pset_ami = old_dw->getParticleSubset(dw_ami, patch);
      //ConstitutiveModel* cm_ami = ami_matl->getConstitutiveModel();

      unsigned int numNewPartNeeded = 0;
      for(unsigned int m = 0; m < numMPMMatls; m++){
       bool notAcceptorMatl = true;
       for(unsigned int index = 0; 
                        index < numAcceptorMaterials; 
                        index++){
         if(m==((unsigned int) flags->d_acceptorMaterialIndex[index])){
           notAcceptorMatl = false;
         }
       }

       if(notAcceptorMatl){
        MPMMaterial* mpm_matl =
                       (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
        int dwi = mpm_matl->getDWIndex();
        ParticleSubset* pset  = old_dw->getParticleSubset(dwi, patch);
        ParticleVariable<double> pcolor;
        new_dw->getModifiable(pcolor, lb->pColorLabel_preReloc,        pset);
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;
          for(unsigned int i=0;i<donorColors.size();i++){
            if(pcolor[idx]==donorColors[i]){
              numNewPartNeeded++;
            }
          }
        }
       }
      }

      const unsigned int oldNumPar = pset_ami->addParticles(numNewPartNeeded);

      cout << "numNewPartNeeded = " << numNewPartNeeded << endl;
      cout << "oldNumPar = " << oldNumPar << endl;

      ParticleVariable<Point>  pxtmp;
      ParticleVariable<Matrix3> pFtmp,psizetmp,pstrstmp,pvgradtmp,pSFtmp;
      ParticleVariable<Matrix3> belbartmp;
      ParticleVariable<double> pYStmp, pPStmp;
      ParticleVariable<long64> pidstmp;
      ParticleVariable<double> psurftmp, pdTdttmp;
      ParticleVariable<double> pvoltmp, pmasstmp,ptemptmp,ptempPtmp,pcolortmp;
      ParticleVariable<Vector> pveltmp,pextFtmp,pdisptmp,ptempgtmp;
      ParticleVariable<int> preftmp,ploctmp,pMIDtmp;
      ParticleVariable<IntVector> pLoadCIDtmp;
      new_dw->allocateTemporary(pidstmp,  pset_ami);
      new_dw->allocateTemporary(pMIDtmp,  pset_ami);
      new_dw->allocateTemporary(pxtmp,    pset_ami);
      new_dw->allocateTemporary(pvoltmp,  pset_ami);
      new_dw->allocateTemporary(pveltmp,  pset_ami);
      new_dw->allocateTemporary(psurftmp, pset_ami);
      new_dw->allocateTemporary(pdTdttmp, pset_ami);
      if(flags->d_computeScaleFactor){
        new_dw->allocateTemporary(pSFtmp, pset_ami);
      }
      new_dw->allocateTemporary(pextFtmp, pset_ami);
      new_dw->allocateTemporary(ptemptmp, pset_ami);
      new_dw->allocateTemporary(ptempgtmp,pset_ami);
      new_dw->allocateTemporary(ptempPtmp,pset_ami);
      new_dw->allocateTemporary(pFtmp,    pset_ami);
      new_dw->allocateTemporary(psizetmp, pset_ami);
      new_dw->allocateTemporary(pdisptmp, pset_ami);
      new_dw->allocateTemporary(pstrstmp, pset_ami);
      new_dw->allocateTemporary(pmasstmp, pset_ami);
      new_dw->allocateTemporary(ploctmp,  pset_ami);
      new_dw->allocateTemporary(pvgradtmp,pset_ami);
      if (flags->d_with_color) {
        new_dw->allocateTemporary(pcolortmp,pset_ami);
      }
  
      if (flags->d_useLoadCurves) {
        new_dw->allocateTemporary(pLoadCIDtmp,  pset_ami);
      }
      new_dw->allocateTemporary(belbartmp,   pset_ami);
      new_dw->allocateTemporary(pYStmp,      pset_ami);
      new_dw->allocateTemporary(pPStmp,      pset_ami);
  
      unsigned int pp = 0;

      for(unsigned int m = 0; m < numMPMMatls; m++){
       bool notAcceptorMatl = true;
       for(unsigned int index = 0; 
                        index < numAcceptorMaterials; 
                        index++){
         if(m==((unsigned int) flags->d_acceptorMaterialIndex[index])){
           notAcceptorMatl = false;
         }
       }
       if(notAcceptorMatl){
        MPMMaterial* mpm_matl =
                       (MPMMaterial*) m_materialManager->getMaterial("MPM", m);
        int dwi = mpm_matl->getDWIndex();
        ParticleSubset* pset  = old_dw->getParticleSubset(dwi, patch);
        //ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
  
        ParticleVariable<Point> px;
        ParticleVariable<Matrix3> pF,pSize,pstress,pvelgrad,pscalefac;
        ParticleVariable<Matrix3> belbar;
        ParticleVariable<double> pYS, pPS;
        ParticleVariable<long64> pids;
        ParticleVariable<double> pvolume,pmass,ptemp,ptempP,pcolor;
        ParticleVariable<double> pSurf, pdTdt;
        ParticleVariable<Vector> pvelocity,pextforce,pdisp,ptempgrad;
        ParticleVariable<int> pref,ploc,prefOld,pModID;
        ParticleVariable<IntVector> pLoadCID;
        new_dw->getModifiable(px,       lb->pXLabel_preReloc,            pset);
        new_dw->getModifiable(pids,     lb->pParticleIDLabel_preReloc,   pset);
        new_dw->getModifiable(pModID,   lb->pModalIDLabel_preReloc,      pset);
        new_dw->getModifiable(pmass,    lb->pMassLabel_preReloc,         pset);
        new_dw->getModifiable(pSize,    lb->pSizeLabel_preReloc,         pset);
        new_dw->getModifiable(pSurf,    lb->pSurfLabel_preReloc,         pset);
        new_dw->getModifiable(pdTdt,    lb->pdTdtLabel,                  pset);
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
        new_dw->getModifiable(ploc,     lb->pLocalizedMPMLabel_preReloc, pset);
        new_dw->getModifiable(pvelgrad, lb->pVelGradLabel_preReloc,      pset);
        new_dw->getModifiable(belbar,   lb->bElBarLabel_preReloc,        pset);
        new_dw->getModifiable(pYS,      lb->pYieldStressLabel_preReloc,  pset);
        new_dw->getModifiable(pPS,      lb->pPlasticStrainLabel_preReloc,pset);
        new_dw->getModifiable(pF,  lb->pDeformationMeasureLabel_preReloc,pset);
        if (flags->d_with_color) {
          new_dw->getModifiable(pcolor, lb->pColorLabel_preReloc,        pset);
        }
        if (flags->d_useLoadCurves) {
          new_dw->getModifiable(pLoadCID,lb->pLoadCurveIDLabel_preReloc, pset);
        }

        // copy data from old variables for particle IDs and the position vector
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          particleIndex idx = *iter;
           for(unsigned int i=0;i<donorColors.size();i++){
            if(pcolor[idx]==donorColors[i]){
             pidstmp[pp]  = pids[idx];
             pMIDtmp[pp]  = pModID[idx];
             pxtmp[pp]    = px[idx];
             psurftmp[pp] = pSurf[idx];
             pdTdttmp[pp] = pdTdt[idx];
             pvoltmp[pp]  = pvolume[idx];
             pveltmp[pp]  = pvelocity[idx];
             pextFtmp[pp] = pextforce[idx];
             ptemptmp[pp] = ptemp[idx];
             ptempgtmp[pp]= ptempgrad[idx];
             ptempPtmp[pp]= ptempP[idx];
             pFtmp[pp]    = pF[idx];
             psizetmp[pp] = pSize[idx];
             pdisptmp[pp] = pdisp[idx];
             pstrstmp[pp] = pstress[idx];
             if(flags->d_computeScaleFactor){
               pSFtmp[pp]   = pscalefac[idx];
             }
             if (flags->d_with_color) {
               pcolortmp[pp]= pcolor[idx];
             }
             if (flags->d_useLoadCurves) {
               pLoadCIDtmp[pp]= pLoadCID[idx];
             }
             pmasstmp[pp] = pmass[idx];
             ploctmp[pp]  = ploc[idx];
             ploc[idx]  = -999;
             pvgradtmp[pp]= pvelgrad[idx];
             belbartmp[pp]= belbar[idx];
             pYStmp[pp]= pYS[idx];
             pPStmp[pp]= pPS[idx];
             pp++;
           } // Color == donorColor
          } // Loop over donorColors
        } // Loop over particles
//      TODO
//      cm->changeCMSpecificParticleData(patch, dwi, 8,
//                                      oldNumPar, numNewPartNeeded,
//                                      old_dw, new_dw);

       }  //  if material is not acceptor material
      }  // for matls
      // put back temporary data
      new_dw->put(pidstmp,  lb->pParticleIDLabel_preReloc,           true);
      new_dw->put(pMIDtmp,  lb->pModalIDLabel_preReloc,              true);
      new_dw->put(pxtmp,    lb->pXLabel_preReloc,                    true);
      new_dw->put(pvoltmp,  lb->pVolumeLabel_preReloc,               true);
      new_dw->put(pveltmp,  lb->pVelocityLabel_preReloc,             true);
      new_dw->put(pdTdttmp, lb->pdTdtLabel,                          true);
      if(flags->d_computeScaleFactor){
        new_dw->put(pSFtmp, lb->pScaleFactorLabel_preReloc,          true);
      }
      new_dw->put(pextFtmp, lb->pExtForceLabel_preReloc,             true);
      new_dw->put(pmasstmp, lb->pMassLabel_preReloc,                 true);
      new_dw->put(ptemptmp, lb->pTemperatureLabel_preReloc,          true);
      new_dw->put(ptempgtmp,lb->pTemperatureGradientLabel_preReloc,  true);
      new_dw->put(ptempPtmp,lb->pTempPreviousLabel_preReloc,         true);
      new_dw->put(psizetmp, lb->pSizeLabel_preReloc,                 true);
      new_dw->put(psurftmp, lb->pSurfLabel_preReloc,                 true);
      new_dw->put(pdisptmp, lb->pDispLabel_preReloc,                 true);
      new_dw->put(pstrstmp, lb->pStressLabel_preReloc,               true);
      if (flags->d_with_color) {
        new_dw->put(pcolortmp,lb->pColorLabel_preReloc,              true);
      }
  
      if (flags->d_useLoadCurves) {
        new_dw->put(pLoadCIDtmp,lb->pLoadCurveIDLabel_preReloc,      true);
      }
      new_dw->put(pFtmp,    lb->pDeformationMeasureLabel_preReloc,   true);
      new_dw->put(ploctmp,  lb->pLocalizedMPMLabel_preReloc,         true);
      new_dw->put(pvgradtmp,lb->pVelGradLabel_preReloc,              true);
      new_dw->put(belbartmp,lb->bElBarLabel_preReloc,                true);
      new_dw->put(pYStmp,   lb->pYieldStressLabel_preReloc,          true);
      new_dw->put(pPStmp,   lb->pPlasticStrainLabel_preReloc,        true);
    }  // Loop over acceptor materials
   }    // if d_changeGrainMaterial
  }   // for patches
}

//
void SerialMPM::scheduleManageChangeGrainMaterials(const LevelP& level,
                                                   SchedulerP& sched)
{

  Task* t = scinew Task("MPM::manageChangeGrainMaterials", this, 
                        &SerialMPM::manageChangeGrainMaterials);

  t->setType( Task::OncePerProc );

  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                    m_materialManager->allMaterials( "MPM" ));
}

//
void SerialMPM::scheduleManageDoAuthigenesis(const LevelP& level,
                                             SchedulerP& sched)
{

  Task* t = scinew Task("MPM::manageDoAuthigenesis", this, 
                        &SerialMPM::manageDoAuthigenesis);

  t->setType( Task::OncePerProc );

  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level),
                    m_materialManager->allMaterials( "MPM" ));
}

//______________________________________________________________________
//
void SerialMPM::manageChangeGrainMaterials(const ProcessorGroup* pg,
                                           const PatchSubset*,
                                           const MaterialSubset*,
                                           DataWarehouse*,
                                           DataWarehouse* new_dw)
{
  if(flags->d_changeGrainMaterials){
#if 0
   flags->d_acceptorMaterialIndex.erase(flags->d_acceptorMaterialIndex.begin());
   unsigned int numAcceptorMaterials = flags->d_acceptorMaterialIndex.size();
   for(unsigned int aMI_index = 0; 
                    aMI_index < numAcceptorMaterials; 
                    aMI_index++){
     unsigned int aMI = flags->d_acceptorMaterialIndex[aMI_index];

     cout << "aMI = " << aMI << endl;
   }
#endif
   d_collideColors.clear();
   flags->d_changeGrainMaterials = false;
  }
}

//______________________________________________________________________
//
void SerialMPM::manageDoAuthigenesis(const ProcessorGroup* pg,
                                     const PatchSubset*,
                                     const MaterialSubset*,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  if(flags->d_canAddParticles){
//   flags->d_canAddParticles = false;
   flags->d_doAuthigenesis  = false;
  }
}

//______________________________________________________________________
//
double SerialMPM::recomputeDelT( const double delT )
{
  return delT * 0.1;
}
