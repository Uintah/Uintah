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
 */#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
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
#include <Core/Math/MinMax.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>


#include <iostream>
#include <fstream>
#include <sstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("AMRMPM", false);
static DebugStream amr_doing("AMRMPM", false);

//#define USE_DEBUG_TASK
//#define DEBUG_VEL
//#define DEBUG_ACC

//__________________________________
//   TODO:
// - We only need to compute ZOI when the grid changes not every timestep
//
// - InterpolateParticlesToGrid_CFI()  Need to account for gimp when getting particles on coarse level.
//
// - scheduleTimeAdvance:  Do we need to schedule each task in a separate level loop?  I suspect that we only need
//                         to in the CFI tasks
//
//  What is going on in refine & coarsen
//  To Test:
//    Symetric BC
//    compilicated grids  
//    multi processors 
//
//  Need to Add gimp interpolation

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

AMRMPM::AMRMPM(const ProcessorGroup* myworld) :SerialMPM(myworld)
{
  lb = scinew MPMLabel();
  flags = scinew MPMFlags(myworld);
  flags->d_minGridLevel = 0;
  flags->d_maxGridLevel = 1000;

  d_SMALL_NUM_MPM=1e-200;
  NGP     = -9;
  NGN     = -9;
  d_nPaddingCells_Coarse = -9;
  dataArchiver = 0;
  d_acc_ans = Vector(0,0,0);
  d_acc_tol = 1e-7;
  d_vel_ans = Vector(-100,0,0);
  d_vel_tol = 1e-7;
  
  pDbgLabel = VarLabel::create("p.dbg",ParticleVariable<double>::getTypeDescription());
  gSumSLabel= VarLabel::create("g.sum_S",NCVariable<double>::getTypeDescription());
}

AMRMPM::~AMRMPM()
{
  delete lb;
  VarLabel::destroy(pDbgLabel);
  VarLabel::destroy(gSumSLabel);
  
  delete flags;
  for (int i = 0; i< (int)d_refine_geom_objs.size(); i++) {
    delete d_refine_geom_objs[i];
  }
}

//______________________________________________________________________
//
void AMRMPM::problemSetup(const ProblemSpecP& prob_spec, 
                          const ProblemSpecP& restart_prob_spec,
                          GridP& grid,
                          SimulationStateP& sharedState)
{
  cout_doing<<"Doing problemSetup\t\t\t\t\t AMRMPM"<<endl;
  
  d_sharedState = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);

  Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));

  if(!dataArchiver){
    throw InternalError("AMRMPM:couldn't get output port", __FILE__, __LINE__);
  }
   
  ProblemSpecP mat_ps = 0;
  if (restart_prob_spec){
    mat_ps = restart_prob_spec;
  } else{
    mat_ps = prob_spec;
  }

  ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");


  // Read all MPM flags (look in MPMFlags.cc)
  flags->readMPMFlags(mat_ps, dataArchiver);
  if (flags->d_integrator_type == "implicit"){
    throw ProblemSetupException("Can't use implicit integration with AMRMPM",
                                 __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in the AMR section
  ProblemSpecP mpm_ps;
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");
  if (amr_ps){
    mpm_ps = amr_ps->findBlock("MPM");
  }
  
  
  if(!mpm_ps){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <MPM>  block not found inside of <AMR> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
  //__________________________________
  // read in the regions that user would like 
  // refined if the grid has not been setup manually
  bool manualGrid;
  mpm_ps->getWithDefault("manualGrid", manualGrid, false);
  
  if(!manualGrid){
    ProblemSpecP refine_ps = mpm_ps->findBlock("Refine_Regions");
    if(!refine_ps ){
      string warn;
      warn ="\n INPUT FILE ERROR:\n <Refine_Regions> "
           " block not found inside of <MPM> block \n";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    // Read in the refined regions geometry objects
    int piece_num = 0;
    list<GeometryObject::DataItem> geom_obj_data;
    geom_obj_data.push_back(GeometryObject::DataItem("level", GeometryObject::Integer));

    for (ProblemSpecP geom_obj_ps = refine_ps->findBlock("geom_object");
          geom_obj_ps != 0;
          geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

        vector<GeometryPieceP> pieces;
        GeometryPieceFactory::create(geom_obj_ps, pieces);

        GeometryPieceP mainpiece;
        if(pieces.size() == 0){
           throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
        } else if(pieces.size() > 1){
           mainpiece = scinew UnionGeometryPiece(pieces);
        } else {
           mainpiece = pieces[0];
        }
        piece_num++;
        d_refine_geom_objs.push_back(scinew GeometryObject(mainpiece,geom_obj_ps,geom_obj_data));
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
  } else if(flags->d_8or27==27 || flags->d_8or27==64){
    NGP=2;
    NGN=2;
  }
  // Determine extents for coarser level particle data
  // Linear Interpolation:  1 layer of coarse level cells
  // Gimp Interpolation:    2 layers
  
/*`==========TESTING==========*/
  d_nPaddingCells_Coarse = 1;
  NGP = 1; 
/*===========TESTING==========`*/

  d_sharedState->setParticleGhostLayer(Ghost::AroundNodes, NGP);

  materialProblemSetup(mat_ps, d_sharedState,flags);
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
    proc0cout << "doMPMOnLevel = " << level->getIndex() << endl;
  }
  else{
    proc0cout << "DontDoMPMOnLevel = " << level->getIndex() << endl;
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

  if (flags->d_reductionVars->accStrainEnergy) {
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
void AMRMPM::scheduleTimeAdvance(const LevelP & level,
                                 SchedulerP   & sched)
{
  if(level->getIndex() > 0)  // only schedule once
    return;

  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  int maxLevels = level->getGrid()->numLevels();
  GridP grid = level->getGrid();


  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    schedulePartitionOfUnity(               sched, patches, matls);
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
    scheduleInterpolateParticlesToGrid_CFI( sched, patches, matls);
  }
  
#ifdef USE_DEBUG_TASK
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleDebug_CFI( sched, patches, matls);
  }
#endif

  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleCoarsenNodalData_CFI( sched, patches, matls, coarsenData);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleNodal_velocity_temperature( sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce(           sched, patches, matls);
  }
#if 1
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce_CFI(       sched, patches, matls);
  }
#endif
#if 1
  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleCoarsenNodalData_CFI2( sched, patches, matls);
  }
#endif
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
 
 // zero the nodal data at the CFI on the coarse level 
 for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleCoarsenNodalData_CFI( sched, patches, matls, zeroData);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate_CFI(sched, patches, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{

  const PatchSet* patches = level->eachPatch();
  scheduleCountParticles(patches,sched);

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
void AMRMPM::schedulePartitionOfUnity(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"AMRMPM::partitionOfUnity");
  Task* t = scinew Task("AMRMPM::partitionOfUnity",
                  this, &AMRMPM::partitionOfUnity);
                  
  t->requires(Task::OldDW, lb->pXLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);
  t->computes(lb->pPartitionUnityLabel);
  sched->addTask(t, patches, matls);
}


//______________________________________________________________________
//
void AMRMPM::scheduleComputeZoneOfInfluence(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  if(L_indx > 0 ){
    MaterialSubset* one_matl = scinew MaterialSubset();
    one_matl->add(0);
    one_matl->addReference();

    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeZoneOfInfluence");
    Task* t = scinew Task("AMRMPM::computeZoneOfInfluence",
                    this, &AMRMPM::computeZoneOfInfluence);

    t->computes(lb->gZOILabel, one_matl);

    sched->addTask(t, patches, matls);

    if (one_matl->removeReference())
      delete one_matl;
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleApplyExternalLoads");

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

  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateParticlesToGrid");


  Task* t = scinew Task("AMRMPM::interpolateParticlesToGrid",
                   this,&AMRMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,               gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,           gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                  gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,        gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan,NGP);
  //t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gExternalForceLabel);
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//  You need particle data from the coarse levels at the CFI on the fine level
void AMRMPM::scheduleInterpolateParticlesToGrid_CFI(SchedulerP& sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();

  if(L_indx > 0 ){
    printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateParticlesToGrid_CFI");

    Task* t = scinew Task("AMRMPM::interpolateParticlesToGrid_CFI",
                     this,&AMRMPM::interpolateParticlesToGrid_CFI);

    Ghost::GhostType  gac  = Ghost::AroundCells;
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    
/*`==========TESTING==========*/
    // Linear 1 coarse Level cells:
    // Gimp:  2 coarse level cells:
    int npc = d_nPaddingCells_Coarse;  
/*===========TESTING==========`*/
    
    #define allPatches 0
    #define allMatls 0
    //__________________________________
    //  Note: were using nPaddingCells to extract the region of coarse level
    // particles around every fine patch.   Technically, these are ghost
    // cells but somehow it works.
    t->requires(Task::NewDW, lb->gZOILabel,   Ghost::None, 0);
    t->requires(Task::OldDW, lb->pMassLabel,               allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pVolumeLabel,             allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pVelocityLabel,           allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pXLabel,                  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pTemperatureLabel,        allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pDeformationMeasureLabel, allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);

    t->modifies(lb->gMassLabel);
    t->modifies(lb->gVolumeLabel);
    t->modifies(lb->gVelocityLabel);
    t->modifies(lb->gTemperatureLabel);
    t->modifies(lb->gExternalForceLabel);

    sched->addTask(t, patches, matls);
  }
}

//______________________________________________________________________
//  This task does one of two operations on the coarse nodes along
//  the coarse fine interface.  The input parameter "flag" determines
//  which.
//  Coarsen:  copy fine patch node data to the coarse level at CFI
//  Zero:     zero the coarse level nodal data directly under the fine level
void AMRMPM::scheduleCoarsenNodalData_CFI(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const coarsenFlag flag)
{
  string txt = "(zero)";
  if (flag == coarsenData){
    txt = "(coarsen)";
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleCoarsenNodalData_CFI" + txt);

  Task* t = scinew Task("AMRMPM::coarsenNodalData_CFI",
                   this,&AMRMPM::coarsenNodalData_CFI, flag);

  Ghost::GhostType  gn  = Ghost::None;
  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0

  t->requires(Task::NewDW, lb->gMassLabel,          allPatches, Task::FineLevel,allMatls, ND, gn,0);
  t->requires(Task::NewDW, lb->gVolumeLabel,        allPatches, Task::FineLevel,allMatls, ND, gn,0);
  t->requires(Task::NewDW, lb->gVelocityLabel,      allPatches, Task::FineLevel,allMatls, ND, gn,0);
  t->requires(Task::NewDW, lb->gTemperatureLabel,   allPatches, Task::FineLevel,allMatls, ND, gn,0);

  t->requires(Task::NewDW, lb->gExternalForceLabel, allPatches, Task::FineLevel,allMatls, ND, gn,0);
  
  t->modifies(lb->gMassLabel);
  t->modifies(lb->gVolumeLabel);
  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gTemperatureLabel);
  t->modifies(lb->gExternalForceLabel);

  if (flag == zeroData){
    t->requires(Task::NewDW, lb->gAccelerationLabel,  allPatches, Task::FineLevel,allMatls, ND, gn,0);
    t->requires(Task::NewDW, lb->gVelocityStarLabel,  allPatches, Task::FineLevel,allMatls, ND, gn,0);
    t->modifies(lb->gAccelerationLabel);
    t->modifies(lb->gVelocityStarLabel);
  }


  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//  This task copies fine patch node data to the coarse level at CFI
void AMRMPM::scheduleCoarsenNodalData_CFI2(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  printSchedule( patches, cout_doing,"AMRMPM::scheduleCoarsenNodalData_CFI2" );

  Task* t = scinew Task( "AMRMPM::coarsenNodalData_CFI2",
                    this,&AMRMPM::coarsenNodalData_CFI2 );

  Ghost::GhostType  gn  = Ghost::None;
  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0

  t->requires(Task::NewDW, lb->gMassLabel,          allPatches, Task::FineLevel,allMatls, ND, gn,0);
  t->requires(Task::NewDW, lb->gInternalForceLabel, allPatches, Task::FineLevel,allMatls, ND, gn,0);
  
  t->modifies(lb->gInternalForceLabel);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//  compute the nodal velocity and temperature after coarsening the fine
//  nodal data
void AMRMPM::scheduleNodal_velocity_temperature(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleNodal_velocity_temperature");

  Task* t = scinew Task("AMRMPM::Nodal_velocity_temperature",
                   this,&AMRMPM::Nodal_velocity_temperature);
                   
  t->requires(Task::NewDW, lb->gMassLabel,  Ghost::None);

  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gTemperatureLabel);
  sched->addTask(t, patches, matls);
}


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
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }
  
  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeStressTensor");
  
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

  if (flags->d_reductionVars->accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleUpdateErosionParameter(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleUpdateErosionParameter");

  Task* t = scinew Task("AMRMPM::updateErosionParameter",
                  this, &AMRMPM::updateErosionParameter);

  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addRequiresDamageParameter(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeInternalForce");
   
  Task* t = scinew Task("AMRMPM::computeInternalForce",
                  this, &AMRMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gVolumeLabel, gnone);
  t->requires(Task::OldDW,lb->pStressLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pSizeLabel,                 gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,  gan,NGP);

  t->computes( gSumSLabel );
  t->computes(lb->gInternalForceLabel);
  t->computes(lb->TotalVolumeDeformedLabel);
  t->computes(lb->gStressForSavingLabel);

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeInternalForce_CFI(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();

  if(L_indx > 0 ){
    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeInternalForce_CFI");

    Task* t = scinew Task("AMRMPM::computeInternalForce_CFI",
                    this, &AMRMPM::computeInternalForce_CFI);

    Ghost::GhostType  gac  = Ghost::AroundCells;
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;

    /*`==========TESTING==========*/
      // Linear 1 coarse Level cells:
      // Gimp:  2 coarse level cells:
     int npc = d_nPaddingCells_Coarse;  
    /*===========TESTING==========`*/
 
    #define allPatches 0
    #define allMatls 0
    //__________________________________
    //  Note: were using nPaddingCells to extract the region of coarse level
    // particles around every fine patch.   Technically, these are ghost
    // cells but somehow it works.
    t->requires(Task::NewDW, lb->gZOILabel,   Ghost::None,0);
    t->requires(Task::OldDW, lb->pXLabel,       allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pStressLabel,  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pVolumeLabel,  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    
    t->modifies( gSumSLabel );
    t->modifies(lb->gInternalForceLabel);

    sched->addTask(t, patches, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeAndIntegrateAcceleration");

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
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleSetGridBoundaryConditions");

  Task* t=scinew Task("AMRMPM::setGridBoundaryConditions",
               this,  &AMRMPM::setGridBoundaryConditions);

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
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateToParticlesAndUpdate");
  
  Task* t=scinew Task("AMRMPM::interpolateToParticlesAndUpdate",
                this, &AMRMPM::interpolateToParticlesAndUpdate);
                
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  
  t->requires(Task::OldDW, lb->pXLabel,                            gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                         gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                   gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,                  gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                     gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                         gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,                         gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                       gnone);
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

  t->computes(lb->TotalMassLabel);
  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->TotalMomentumLabel);

#ifndef USE_DEBUG_TASK
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  gnone);
    t->computes(lb->pColorLabel_preReloc);
  }
#endif  
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void AMRMPM::scheduleInterpolateToParticlesAndUpdate_CFI(SchedulerP& sched,
                                                         const PatchSet* patches,
                                                         const MaterialSet* matls)

{
  const Level* level = getLevel(patches);
  
  if(level->hasFinerLevel()){

    printSchedule(patches,cout_doing,"AMRMPM::scheduleInterpolateToParticlesAndUpdate_CFI");

    Task* t=scinew Task("AMRMPM::interpolateToParticlesAndUpdate_CFI",
                  this, &AMRMPM::interpolateToParticlesAndUpdate_CFI);

    Ghost::GhostType  gn  = Ghost::None;
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    t->requires(Task::OldDW, d_sharedState->get_delt_label() );
    
    t->requires(Task::OldDW, lb->pXLabel, gn);
    t->requires(Task::NewDW, lb->gVelocityStarLabel, allPatches, Task::FineLevel,allMatls, ND, gn,0);
    t->requires(Task::NewDW, lb->gAccelerationLabel, allPatches, Task::FineLevel,allMatls, ND, gn,0);
    t->requires(Task::NewDW, lb->gZOILabel,          allPatches, Task::FineLevel,allMatls, ND, gn,0);
    
    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pDispLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);

    sched->addTask(t, patches, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleRefine(const PatchSet* patches, 
                               SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleRefine");
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
    
  printSchedule(coarseLevel,cout_doing,"AMRMPM::scheduleErrorEstimate");
  
  Task* task = scinew Task("AMRMPM::errorEstimate", this, 
                           &AMRMPM::errorEstimate);

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
  const Level* level = getLevel(patches);
  int levelIndex = level->getIndex();
  particleIndex totalParticles=0;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    
    printTask(patches, patch,cout_doing,"Doing AMRMPM::actuallyInitialize");

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
      
      
      //__________________________________
      // color particles according to what level they're on
      if (flags->d_with_color) {
        ParticleSubset* pset = new_dw->getParticleSubset(indx, patch);
        ParticleVariable<double> pColor;
        new_dw->getModifiable(pColor, lb->pColorLabel, pset);

        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pColor[idx] =levelIndex;
        }
      }
    }  // matl loop
  }

  if (flags->d_reductionVars->accStrainEnergy) {
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
//  This task computes the partition of unity for each particle
//
void AMRMPM::partitionOfUnity(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing AMRMPM::partitionOfUnity");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    const Matrix3 notUsed;

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<double>p_partitionUnity;
    
      old_dw->get(px,     lb->pXLabel,     pset);
      old_dw->get(psize,  lb->pSizeLabel,  pset);
      new_dw->allocateAndPut(p_partitionUnity,  lb->pPartitionUnityLabel, pset);
      
      int n8or27=flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],notUsed);

        p_partitionUnity[idx] = 0;
         
        for(int k = 0; k < n8or27; k++) {
          p_partitionUnity[idx] += S[k];
        }
      }
    }  // loop over materials
    delete interpolator;
  }  // loop over patches
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

    printTask(patches,patch,cout_doing,"Doing AMRMPM::interpolateParticlesToGrid");

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
      constParticleVariable<Vector> pvelocity, pexternalforce;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,                   lb->pXLabel,                  pset);
      old_dw->get(pmass,                lb->pMassLabel,               pset);
      old_dw->get(pvolume,              lb->pVolumeLabel,             pset);
      old_dw->get(pvelocity,            lb->pVelocityLabel,           pset);
      old_dw->get(pTemperature,         lb->pTemperatureLabel,        pset);
      old_dw->get(psize,                lb->pSizeLabel,               pset);
      old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);
      new_dw->get(pexternalforce,       lb->pExtForceLabel_preReloc,  pset);
      
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
      
      Vector pmom;
      int n8or27=flags->d_8or27;

      //double pSp_vol = 1./mpm_matl->getInitialDensity();
      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);

        pmom = pvelocity[idx]*pmass[idx];
        
        // Add each particles contribution to the local mass & velocity 
        IntVector node;
        for(int k = 0; k < n8or27; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
            gmass[node]          += pmass[idx]                     * S[k];
            gvelocity[node]      += pmom                           * S[k];
            gvolume[node]        += pvolume[idx]                   * S[k];
            gexternalforce[node] += pexternalforce[idx]            * S[k];
            gTemperature[node]   += pTemperature[idx] * pmass[idx] * S[k];
          }
        }
      }  // End of particle loop
    }  // End loop over materials
    delete interpolator;
  }  // End loop over patches
}

//______________________________________________________________________
//  At the CFI fine patch nodes add contributions from the coarse level particles.
void AMRMPM::interpolateParticlesToGrid_CFI(const ProcessorGroup*,
                                            const PatchSubset* finePatches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(finePatches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  IntVector refineRatio(fineLevel->getRefinementRatio());
  
  for(int fp=0; fp<finePatches->size(); fp++){
    const Patch* finePatch = finePatches->get(fp);
    printTask(finePatches,finePatch,cout_doing,"Doing AMRMPM::interpolateParticlesToGrid_CFI");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(finePatch);
    
    constNCVariable<Stencil7> zoi_fine;
    new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

    // Determine extents for coarser level particle data
    // Linear Interpolation:  1 layer of coarse level cells
    // Gimp Interpolation:    2 layers
/*`==========TESTING==========*/
    IntVector nLayers(d_nPaddingCells_Coarse, d_nPaddingCells_Coarse, d_nPaddingCells_Coarse );
    IntVector nPaddingCells = nLayers * (fineLevel->getRefinementRatio());
    //cout << " nPaddingCells " << nPaddingCells << "nLayers " << nLayers << endl;
/*===========TESTING==========`*/

    int nGhostCells = 0;
    bool returnExclusiveRange=false;
    IntVector cl_tmp, ch_tmp, fl, fh;

    getCoarseLevelRange(finePatch, coarseLevel, cl_tmp, ch_tmp, fl, fh, 
                        nPaddingCells, nGhostCells,returnExclusiveRange);
                        
    cl_tmp -= finePatch->neighborsLow() * nLayers;  //  expand cl_tmp when a neighor patch exists.
                                                    //  This patch owns the low nodes.  You need particles
                                                    //  from the neighbor patch.

    // find the coarse patches under the fine patch.  You must add a single layer of padding cells.
    int padding = 1;
    Level::selectType coarsePatches;
    finePatch->getOtherLevelPatches(-1, coarsePatches, padding);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // get fine level nodal data
      NCVariable<double> gMass_fine;
      NCVariable<double> gVolume_fine;
      NCVariable<Vector> gVelocity_fine;
      NCVariable<Vector> gExternalforce_fine;
      NCVariable<double> gTemperature_fine;
            
      new_dw->getModifiable(gMass_fine,            lb->gMassLabel,         dwi,finePatch);
      new_dw->getModifiable(gVolume_fine,          lb->gVolumeLabel,       dwi,finePatch);
      new_dw->getModifiable(gVelocity_fine,        lb->gVelocityLabel,     dwi,finePatch);
      new_dw->getModifiable(gTemperature_fine,     lb->gTemperatureLabel,  dwi,finePatch);
      new_dw->getModifiable(gExternalforce_fine,   lb->gExternalForceLabel,dwi,finePatch);
      
      // loop over the coarse patches under the fine patches.
      for(int cp=0; cp<coarsePatches.size(); cp++){
        const Patch* coarsePatch = coarsePatches[cp];
        
        // get coarse level particle data
        constParticleVariable<Point>  pX_coarse;
        constParticleVariable<double> pMass_coarse;
        constParticleVariable<double> pVolume_coarse;
        constParticleVariable<double> pTemperature_coarse;
        constParticleVariable<Vector> pVelocity_coarse;
        constParticleVariable<Vector> pExternalforce_coarse;
        constParticleVariable<Matrix3> pDefMeasure_coarse;

        // coarseLow and coarseHigh cannot lie outside of the coarse patch
        IntVector cl = Max(cl_tmp, coarsePatch->getCellLowIndex());
        IntVector ch = Min(ch_tmp, coarsePatch->getCellHighIndex());
        
        ParticleSubset* pset=0;
        
        pset = old_dw->getParticleSubset(dwi, cl, ch, coarsePatch ,lb->pXLabel);
#if 0
        cout << "  coarseLevel: " << coarseLevel->getIndex() << endl;
        cout << " cl_tmp: "<< cl_tmp << " ch_tmp: " << ch_tmp << endl;
        cout << " cl:     " << cl    << " ch:     " << ch<< " fl: " << fl << " fh " << fh << endl;
        cout << "  " << *pset << endl;
#endif        
        old_dw->get(pX_coarse,             lb->pXLabel,                  pset);
        old_dw->get(pMass_coarse,          lb->pMassLabel,               pset);
        old_dw->get(pVolume_coarse,        lb->pVolumeLabel,             pset);
        old_dw->get(pVelocity_coarse,      lb->pVelocityLabel,           pset);
        old_dw->get(pTemperature_coarse,   lb->pTemperatureLabel,        pset);
        old_dw->get(pDefMeasure_coarse,    lb->pDeformationMeasureLabel, pset);
        new_dw->get(pExternalforce_coarse, lb->pExtForceLabel_preReloc,  pset);

        for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the fine patch cell
          vector<IntVector> ni;
          vector<double> S;

          interpolator->findCellAndWeights_CFI(pX_coarse[idx],ni,S,zoi_fine);

          Vector pmom = pVelocity_coarse[idx]*pMass_coarse[idx];

          // Add each particle's contribution to the local mass & velocity 
          IntVector fineNode;
          for(int k = 0; k < (int) ni.size(); k++) {
            fineNode = ni[k];
            
            //S[k] *= pErosion[idx];
            
            gMass_fine[fineNode]          += pMass_coarse[idx]          * S[k];
            gVelocity_fine[fineNode]      += pmom                       * S[k];
            gVolume_fine[fineNode]        += pVolume_coarse[idx]        * S[k];
            gExternalforce_fine[fineNode] += pExternalforce_coarse[idx] * S[k];
            gTemperature_fine[fineNode]   += pTemperature_coarse[idx] 
                                           * pMass_coarse[idx] * S[k];

  /*`==========TESTING==========*/
#if 0
//            if( fineNode.y() == 30 && fineNode.z() == 1 && (fineNode.x() > 37 && fineNode.x() < 43) ){
              cout << "    fineNode " << fineNode  << " S[k] " << S[k] << " \t gMass_fine " << gMass_fine[fineNode]
                   << " gVelocity " << gVelocity_fine[fineNode]/gMass_fine[fineNode] << " px: " << pX_coarse[idx] << " \t zoi " << (zoi_fine[fineNode]) << endl; 
            }
#endif
  /*===========TESTING==========`*/
          }
        }  // End of particle loop
      }  // loop over coarse patches
    }  // End loop over materials  
    delete interpolator;
  }  // End loop over fine patches
}

//______________________________________________________________________
//  copy the fine level nodal data to the underlying coarse nodes at the CFI.
void AMRMPM::coarsenNodalData_CFI(const ProcessorGroup*,
                                  const PatchSubset* coarsePatches,
                                  const MaterialSubset* ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const coarsenFlag flag)
{
  Level::selectType coarseCFI_Patches;
  
  coarseLevelCFI_Patches(coarsePatches,coarseCFI_Patches );
  
  //__________________________________
  // From the coarse patch look up to the fine patches that have
  // coarse fine interfaces.
  const Level* coarseLevel = getLevel(coarsePatches);
  
  for(int p=0;p<coarseCFI_Patches.size();p++){
    const Patch* coarsePatch = coarseCFI_Patches[p];

    string txt = "(zero)";
    if (flag == coarsenData){
      txt = "(coarsen)";
    }
    printTask(coarsePatch,cout_doing,"Doing AMRMPM::coarsenNodalData_CFI"+txt);
    
    int numMatls = d_sharedState->getNumMPMMatls();                  
    for(int m = 0; m < numMatls; m++){                               
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );    
      int dwi = mpm_matl->getDWIndex();
      
      // get coarse level data                                                                                    
      NCVariable<double> gMass_coarse;                                                                            
      NCVariable<double> gVolume_coarse;                                                                          
      NCVariable<Vector> gVelocity_coarse;
      NCVariable<Vector> gVelocityStar_coarse;
      NCVariable<Vector> gAcceleration_coarse;                                                                 
      NCVariable<Vector> gExternalforce_coarse;                                                                   
      NCVariable<double> gTemperature_coarse;    
    
      new_dw->getModifiable(gMass_coarse,            lb->gMassLabel,           dwi,coarsePatch);                  
      new_dw->getModifiable(gVolume_coarse,          lb->gVolumeLabel,         dwi,coarsePatch);                  
      new_dw->getModifiable(gVelocity_coarse,        lb->gVelocityLabel,       dwi,coarsePatch);                  
      new_dw->getModifiable(gTemperature_coarse,     lb->gTemperatureLabel,    dwi,coarsePatch);                  
      new_dw->getModifiable(gExternalforce_coarse,   lb->gExternalForceLabel,  dwi,coarsePatch);
      
      if(flag == zeroData){
        new_dw->getModifiable(gVelocityStar_coarse,  lb->gVelocityStarLabel,     dwi,coarsePatch);
        new_dw->getModifiable(gAcceleration_coarse,  lb->gAccelerationLabel,     dwi,coarsePatch);
      }                                                                               
        
      //__________________________________
      // Iterate over coarse/fine interface faces

      ASSERT(coarseLevel->hasFinerLevel());
      const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);

      // loop over all the fine level patches
      for(int fp=0;fp<finePatches.size();fp++){  
        const Patch* finePatch = finePatches[fp];
        if(finePatch->hasCoarseFaces() ){

          // get fine level data                                                                                  
          constNCVariable<double> gMass_fine;                                                                     
          constNCVariable<double> gVolume_fine;                                                                   
          constNCVariable<Vector> gVelocity_fine;                                                                 
          constNCVariable<Vector> gExternalforce_fine;                                                            
          constNCVariable<double> gTemperature_fine;                                                                   
          Ghost::GhostType  gn = Ghost::None;                                                                     

          if(flag == coarsenData){
            new_dw->get(gMass_fine,             lb->gMassLabel,          dwi, finePatch, gn, 0);
            new_dw->get(gVolume_fine,           lb->gVolumeLabel,        dwi, finePatch, gn, 0);
            new_dw->get(gVelocity_fine,         lb->gVelocityLabel,      dwi, finePatch, gn, 0);
            new_dw->get(gTemperature_fine,      lb->gTemperatureLabel,   dwi, finePatch, gn, 0);
            new_dw->get(gExternalforce_fine,    lb->gExternalForceLabel, dwi, finePatch, gn, 0);
          }                                                                                     

          vector<Patch::FaceType> cf;
          finePatch->getCoarseFaces(cf);

          // Iterate over coarse/fine interface faces
          vector<Patch::FaceType>::const_iterator iter;  
          for (iter  = cf.begin(); iter != cf.end(); ++iter){
            Patch::FaceType patchFace = *iter;
            
            
            // determine the iterator on the coarse level.
            NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            bool isRight_CP_FP_pair;

            coarseLevel_CFI_NodeIterator( patchFace,coarsePatch, finePatch,fineLevel,
                                          n_iter ,isRight_CP_FP_pair);

            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              //cout << UintahParallelComponent::d_myworld->myrank() << "    fine patch face " << finePatch->getFaceName(patchFace)
              //     << "    CoarseLevel CFI iterator: " << n_iter << endl;

              for(; !n_iter.done(); n_iter++) {
                IntVector c_node = *n_iter;

                IntVector f_node = coarseLevel->mapNodeToFiner(c_node);

                switch(flag)
                {
                  case coarsenData:
                    // only overwrite coarse data if there is non-zero fine data
                    if( gMass_fine[f_node] > 2 * d_SMALL_NUM_MPM ){

                      //cout << "    coarsen:  c_node: " << c_node << " f_node: "<< f_node << " gmass_coarse: " 
                      //     << gMass_coarse[c_node] << " gmass_fine: " << gMass_fine[f_node] << endl;

                      gMass_coarse[c_node]           = gMass_fine[f_node];
                      gVolume_coarse[c_node]         = gVolume_fine[f_node];
                      gVelocity_coarse[c_node]       = gVelocity_fine[f_node];
                      gTemperature_coarse[c_node]    = gTemperature_fine[f_node];
                      gExternalforce_coarse[c_node]  = gExternalforce_fine[f_node];
                    }
                   break;
                  case zeroData:
                  
                    // cout << "    zero:  c_node: " << c_node << " f_node: "<< f_node << " gmass_coarse: " 
                    //       << gMass_coarse[c_node] << endl;
                  
                    gMass_coarse[c_node]          = 0;
                    gVolume_coarse[c_node]        = 0;
                    gVelocity_coarse[c_node]      = Vector(0,0,0);
                    gVelocityStar_coarse[c_node]  = Vector(0,0,0);
                    gAcceleration_coarse[c_node]  = Vector(0,0,0);
                    gTemperature_coarse[c_node]   = 0;
                    gExternalforce_coarse[c_node] = Vector(0,0,0);
                    break;
                }
              }
            }  //  isRight_CP_FP_pair

          }  //  end CFI face loop
        }  //  if finepatch has coarse face
      }  //  end fine Patch loop
    }  //  end matl loop
  }  // end coarse patch loop
}


//______________________________________________________________________
//  copy the fine level nodal data to the underlying coarse nodes at the CFI.
void AMRMPM::coarsenNodalData_CFI2(const ProcessorGroup*,
                                  const PatchSubset* coarsePatches,
                                  const MaterialSubset* ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  Level::selectType coarseCFI_Patches;
  coarseLevelCFI_Patches(coarsePatches,coarseCFI_Patches );

  //__________________________________
  // From the coarse patch look up to the fine patches that have
  // coarse fine interfaces.
  const Level* coarseLevel = getLevel(coarsePatches);
  
  for(int p=0;p<coarseCFI_Patches.size();p++){
    const Patch* coarsePatch = coarseCFI_Patches[p];

    printTask(coarsePatch,cout_doing,"Doing AMRMPM::coarsenNodalData_CFI2");
    
    int numMatls = d_sharedState->getNumMPMMatls();                  
    for(int m = 0; m < numMatls; m++){                               
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );    
      int dwi = mpm_matl->getDWIndex();
      
      // get coarse level data                                                             
      NCVariable<Vector> internalForce_coarse;                    
      new_dw->getModifiable(internalForce_coarse, lb->gInternalForceLabel, dwi,coarsePatch);                                                                                
        
      //__________________________________
      // Iterate over coarse/fine interface faces

      ASSERT(coarseLevel->hasFinerLevel());
      const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);

      // loop over all the coarse level patches
      for(int fp=0;fp<finePatches.size();fp++){  
        const Patch* finePatch = finePatches[fp];
        if(finePatch->hasCoarseFaces() ){

          // get fine level data                                                                                  
          constNCVariable<double> gMass_fine;                                                                 
          constNCVariable<Vector> internalForce_fine;                                                                  
          Ghost::GhostType  gn = Ghost::None;                                                                     
          new_dw->get(gMass_fine,          lb->gMassLabel,          dwi, finePatch, gn, 0);
          new_dw->get(internalForce_fine,  lb->gInternalForceLabel, dwi, finePatch, gn, 0);

          vector<Patch::FaceType> cf;
          finePatch->getCoarseFaces(cf);

          // Iterate over coarse/fine interface faces
          vector<Patch::FaceType>::const_iterator iter;  
          for (iter  = cf.begin(); iter != cf.end(); ++iter){
            Patch::FaceType patchFace = *iter;
            
            
            // determine the iterator on the coarse level.
            NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            bool isRight_CP_FP_pair;

            coarseLevel_CFI_NodeIterator( patchFace,coarsePatch, finePatch,fineLevel,
                                          n_iter ,isRight_CP_FP_pair);

            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
               
              for(; !n_iter.done(); n_iter++) {
                IntVector c_node = *n_iter;

                IntVector f_node = coarseLevel->mapNodeToFiner(c_node);
 
                // only overwrite coarse data if there is non-zero fine data
                if( gMass_fine[f_node] > 2 * d_SMALL_NUM_MPM ){
               
                 internalForce_coarse[c_node] = internalForce_fine[f_node];
                 
/*`==========TESTING==========*/
#if 0
                  if( internalForce_coarse[c_node].length()  >1e-8){
                    ostringstream warn;
                    warn << "Too Big: " << c_node << " f_node " << f_node 
                         << "    L-"<< fineLevel->getIndex()
                         <<" InternalForce_fine   " << internalForce_fine[f_node] 
                         <<" InternalForce_coarse " << internalForce_coarse[c_node] << endl;
                     
                    throw InternalError(warn.str(), __FILE__, __LINE__);
                  } 
#endif
/*===========TESTING==========`*/
                  
                }
              }  //  node loop
            }  //  isRight_CP_FP_pair
          }  //  end CFI face loop
        }  //  if finepatch has coarse face
      }  //  end fine Patch loop
    }  //  end matl loop
  }  //  end coarse patch loop
}

//______________________________________________________________________
// Divide gVelocity and gTemperature by gMass
void AMRMPM::Nodal_velocity_temperature(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* ,
                                        DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing AMRMPM::Nodal_velocity_temperature");

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // get  level nodal data
      constNCVariable<double> gMass;
      NCVariable<Vector> gVelocity;
      NCVariable<double> gTemperature;
      Ghost::GhostType  gn = Ghost::None;
      
      new_dw->get(gMass,                   lb->gMassLabel,         dwi,patch, gn, 0);
      new_dw->getModifiable(gVelocity,     lb->gVelocityLabel,     dwi,patch, gn, 0);
      new_dw->getModifiable(gTemperature,  lb->gTemperatureLabel,  dwi,patch, gn, 0);
      
      //__________________________________
      //  back out the nodal quantities
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        gVelocity[n]     /= gMass[n];
        gTemperature[n]  /= gMass[n];
      }
      
      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      string interp_type = flags->d_interpolator_type;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gVelocity,   interp_type);
    }  // End loop over materials
  }  // End loop over fine patches
}

//______________________________________________________________________
//
void AMRMPM::computeStressTensor(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  printTask(patches, patches->get(0),cout_doing,"Doing AMRMPM::computeStressTensor");

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
    printTask(patches, patch,cout_doing, "Doing AMRMPM::updateErosionParameter");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the localization info
      ParticleVariable<int> isLocalized;
      new_dw->allocateTemporary(isLocalized, pset);
      ParticleSubset::iterator iter = pset->begin(); 
      for (; iter != pset->end(); iter++) isLocalized[*iter] = 0;
      mpm_matl->getConstitutiveModel()->getDamageParameter(patch, isLocalized,
                                                           dwi, old_dw,new_dw);

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
    printTask(patches, patch,cout_doing,"Doing AMRMPM::computeInternalForce");

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);

    
    string interp_type = flags->d_interpolator_type;

    int numMPMMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol;
      constParticleVariable<double>  p_pressure;
      constParticleVariable<double>  p_q;
      constParticleVariable<Matrix3> pstress;
      constParticleVariable<Matrix3> psize;
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
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);

      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);
      gstress.initialize(Matrix3(0));
      internalforce.initialize(Vector(0,0,0));
      // getParticleSubset_CFI
      
/*`==========TESTING==========*/
      NCVariable<double> gSumS;
      new_dw->allocateAndPut(gSumS, gSumSLabel,  dwi, patch); 
      gSumS.initialize(0); 
/*===========TESTING==========`*/ 
      
      //__________________________________
      //  fine Patch     
      gstress.initialize(Matrix3(0));

      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;
      vector<IntVector> ni(interpolator->size());
      vector<double> S(interpolator->size());
      vector<Vector> d_S(interpolator->size());
    

      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end();  iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S,
                                                            psize[idx],pDeformationMeasure[idx]);

        stresspress = pstress[idx];
        
/*`==========TESTING==========*/
//        stresspress = Matrix3(0);                 //HARDWIRED
/*===========TESTING==========`*/

        for (int k = 0; k < n8or27; k++){
          
          if(patch->containsNode(ni[k])){ 
            Vector div(d_S[k].x()*oodx[0],
                       d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
                       
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
            
            // cout << " CIF: ni: " << ni[k] << " div " << div << "\t internalForce " << internalforce[ni[k]] << endl;
            // cout << " div " << div[k] << " stressPress: " << stresspress  << endl;
            
            if( isinf( internalforce[ni[k]].length() ) || isnan( internalforce[ni[k]].length() ) ){
                cout << "INF: " << ni[k] << " " << internalforce[ni[k]] << " div: " << div << " stressPress: " << stresspress 
                      << " pvol " << pvol[idx] << endl;
              }
/*`==========TESTING==========*/
            gSumS[ni[k]] +=S[k]; 
/*===========TESTING==========`*/
          }
        }
      }

      string interp_type = flags->d_interpolator_type;
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce,interp_type);
    }  // End matl loop
    delete interpolator;
  }  // End patch loop
}

//______________________________________________________________________
//
void AMRMPM::computeInternalForce_CFI(const ProcessorGroup*,
                                      const PatchSubset* finePatches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(finePatches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  IntVector refineRatio(fineLevel->getRefinementRatio());
  
  for(int p=0;p<finePatches->size();p++){
    const Patch* finePatch = finePatches->get(p);
    printTask(finePatches, finePatch,cout_doing,"Doing AMRMPM::computeInternalForce_CFI");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(finePatch);

    //__________________________________
    //          AT CFI
    if( fineLevel->hasCoarserLevel() &&  finePatch->hasCoarseFaces() ){


      // Determine extents for coarser level particle data
      // Linear Interpolation:  1 layer of coarse level cells
      // Gimp Interpolation:    2 layers
  /*`==========TESTING==========*/
      IntVector nLayers(d_nPaddingCells_Coarse, d_nPaddingCells_Coarse, d_nPaddingCells_Coarse );
      IntVector nPaddingCells = nLayers * (fineLevel->getRefinementRatio());
      //cout << " nPaddingCells " << nPaddingCells << "nLayers " << nLayers << endl;
  /*===========TESTING==========`*/

      int nGhostCells = 0;
      bool returnExclusiveRange=false;
      IntVector cl_tmp, ch_tmp, fl, fh;

      getCoarseLevelRange(finePatch, coarseLevel, cl_tmp, ch_tmp, fl, fh, 
                          nPaddingCells, nGhostCells,returnExclusiveRange);
                          
      cl_tmp -= finePatch->neighborsLow() * nLayers;  //  expand cl_tmp when a neighor patch exists.
                                                      //  This patch owns the low nodes.  You need particles
                                                      //  from the neighbor patch.

      // find the coarse patches under the fine patch.  You must add a single layer of padding cells.
      int padding = 1;
      Level::selectType coarsePatches;
      finePatch->getOtherLevelPatches(-1, coarsePatches, padding);
        

        
      constNCVariable<Stencil7> zoi_fine;
      new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );
  
      int numMPMMatls = d_sharedState->getNumMPMMatls();
      
      for(int m = 0; m < numMPMMatls; m++){
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int dwi = mpm_matl->getDWIndex();
        
        NCVariable<Vector> internalforce;
        new_dw->getModifiable(internalforce,lb->gInternalForceLabel,  dwi, finePatch);

  /*`==========TESTING==========*/
        NCVariable<double> gSumS;
        new_dw->getModifiable(gSumS, gSumSLabel,  dwi, finePatch);
  /*===========TESTING==========`*/ 
        
        // loop over the coarse patches under the fine patches.
        for(int cp=0; cp<coarsePatches.size(); cp++){
          const Patch* coarsePatch = coarsePatches[cp];

          // get coarse level particle data                                                       
          ParticleSubset* pset_coarse;    
          constParticleVariable<Point> px_coarse;
          constParticleVariable<Matrix3> pstress_coarse;
          constParticleVariable<double>  pvol_coarse;
          
          // coarseLow and coarseHigh cannot lie outside of the coarse patch
          IntVector cl = Max(cl_tmp, coarsePatch->getCellLowIndex());
          IntVector ch = Min(ch_tmp, coarsePatch->getCellHighIndex());

          pset_coarse = old_dw->getParticleSubset(dwi, cl, ch, coarsePatch ,lb->pXLabel);

   #if 0
          cout << " fine patch : " << finePatch->getGridIndex() << endl;
          cout << " cl_tmp: "<< cl_tmp << " ch_tmp: " << ch_tmp << endl;
          cout << " cl:     " << cl    << " ch:     " << ch<< " fl: " << fl << " fh " << fh << endl;                                                     
          cout << "  " << *pset_coarse << endl;
   #endif

          // coarse level data
          old_dw->get(px_coarse,       lb->pXLabel,       pset_coarse);
          old_dw->get(pvol_coarse,     lb->pVolumeLabel,  pset_coarse);
          old_dw->get(pstress_coarse,  lb->pStressLabel,  pset_coarse);

          //__________________________________
          //  Iterate over the coarse level particles and 
          // add their contribution to the internal stress on the fine patch
          for (ParticleSubset::iterator iter = pset_coarse->begin(); iter != pset_coarse->end();  iter++){
            particleIndex idx = *iter;

            vector<IntVector> ni;
            vector<double> S;
            vector<Vector> div;
            interpolator->findCellAndWeightsAndShapeDerivatives_CFI( px_coarse[idx], ni, S, div, zoi_fine );

            Matrix3 stresspress = pstress_coarse[idx];
/*`==========TESTING==========*/
//            stresspress = Matrix3(0);              // hardwire
/*===========TESTING==========`*/

            IntVector fineNode;
            for(int k = 0; k < (int)ni.size(); k++) {   
              fineNode = ni[k];

              if( finePatch->containsNode( fineNode ) ){
                gSumS[fineNode] +=S[k];

                Vector Increment ( (div[k] * stresspress)  * pvol_coarse[idx] );
                Vector Before = internalforce[fineNode];
                Vector After  = Before - Increment;

                internalforce[fineNode] -=  Increment;


                //  cout << " CIF_CFI: ni: " << ni[k] << " div " << div[k] << "\t internalForce " << internalforce[fineNode] << endl;
                //  cout << "    before " << Before << " After " << After << " Increment " << Increment << endl;
                //  cout << "    div " << div[k] << " stressPress: " << stresspress << " pvol_coarse " << pvol_coarse[idx] << endl;


  /*`==========TESTING==========*/
                if(isinf( internalforce[fineNode].length() ) ||  isnan( internalforce[fineNode].length() )){
                  cout << "INF: " << fineNode << " " << internalforce[fineNode] << " div[k]: " << div[k] << " stressPress: " << stresspress 
                        << " pvol " << pvol_coarse[idx] << endl;
                }

   #if 0             
                if( internalforce[fineNode].length()  >1e-10){
                  cout << "CIF_CFI: " << fineNode
                       << "    L-"<< getLevel(finePatches)->getIndex()
                       <<" InternalForce " << internalforce[fineNode] << " div[k]: " << div[k] << " stressPress: " << stresspress 
                       << " pvol " << pvol_coarse[idx] << endl;
                  cout << "          Before: " << Before << " Increment " << Increment << endl;
                }
  #endif
  /*===========TESTING==========`*/


              }  // contains node
            }  // node loop          
          }  // pset loop
        }  // coarse Patch loop
        
        //__________________________________
        //  Set boundary conditions 
        string interp_type = flags->d_interpolator_type;
        MPMBoundCond bc;
        bc.setBoundaryCondition( finePatch,dwi,"Symmetric",internalforce,interp_type); 
              
      }  // End matl loop 
    }  // patch has CFI faces
    delete interpolator;
  }  // End fine patch loop
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
    printTask(patches, patch,cout_doing,"Doing AMRMPM::computeAndIntegrateAcceleration");

    Ghost::GhostType  gnone = Ghost::None;
    Vector gravity = flags->d_gravity;
    
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<Vector> gvelocity;
      constNCVariable<double> gmass;

      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->get(internalforce,lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass,         lb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->get(gvelocity,     lb->gVelocityLabel,      dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      new_dw->allocateAndPut(gvelocity_star, lb->gVelocityStarLabel, dwi, patch);
      new_dw->allocateAndPut(gacceleration,  lb->gAccelerationLabel, dwi, patch);

      gacceleration.initialize(Vector(0.,0.,0.));
      gvelocity_star.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter=patch->getExtraNodeIterator(); !iter.done();iter++){
        IntVector n = *iter;
        if (gmass[n] > flags->d_min_mass_for_acceleration){
          Vector acc = (internalforce[n] + externalforce[n])/gmass[n];
          gacceleration[n]  = acc +  gravity;
          gvelocity_star[n] = gvelocity[n] + gacceleration[n] * delT;
/*`==========TESTING==========*/
#ifdef DEBUG_ACC
          if( abs(gacceleration[n].length() - d_acc_ans.length()) > d_acc_tol ) {
            Vector diff = gacceleration[n] - d_acc_ans;
            cout << "    L-"<< getLevel(patches)->getIndex() << " node: "<< n << " gacceleration: " << gacceleration[n] 
                 <<  " externalForce: " <<externalforce[n] << " internalforce: " << internalforce[n] 
                 << " diff: " << diff
                 << " gmass: " << gmass[n] 
                 << " gravity: " << gravity << endl;
          }
#endif 
/*===========TESTING==========`*/
        }
      }
    }  // matls
  }  // patches
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
    printTask(patches, patch,cout_doing,"Doing AMRMPM::setGridBoundaryConditions");

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
    
    string interp_type = flags->d_interpolator_type;

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      constNCVariable<Vector> gvelocity;

      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,  dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,  dwi,patch);
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi, patch, Ghost::None,0);
          
          
      //__________________________________
      // Apply grid boundary conditions to velocity_star and acceleration
      if( patch->hasBoundaryFaces() && !patch->hasCoarseFaces() ){
        IntVector node(10,10,0);
        //cout << "    L-"<< getLevel(patches)->getIndex() << " before setBC  gvelocity_star: " << gvelocity_star[node] << endl;
        

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
        
        //cout << "    L-"<< getLevel(patches)->getIndex() << " After setBC  gvelocity_star: " << gvelocity_star[node] << endl;
        
        bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

       //cout << "    L-"<< getLevel(patches)->getIndex() <<  " After setBC2  gvelocity_star: " << gvelocity_star[node] << endl;
       
        // Now recompute acceleration as the difference between the velocity
        // interpolated to the grid (no bcs applied) and the new velocity_star
        for(NodeIterator iter = patch->getExtraNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          gacceleration[c] = (gvelocity_star[c] - gvelocity[c])/delT;
        }
        // Set symmetry BCs on acceleration
        bc.setBoundaryCondition(patch,dwi,"Symmetric",gacceleration,interp_type);
      } 
      
      //__________________________________
      //
      if(!flags->d_doGridReset){
        NCVariable<Vector> displacement;
        constNCVariable<Vector> displacementOld;
        new_dw->allocateAndPut(displacement,lb->gDisplacementLabel,dwi,patch);
        old_dw->get(displacementOld,        lb->gDisplacementLabel,dwi,patch,
                                                               Ghost::None,0);
        for(NodeIterator iter = patch->getExtraNodeIterator();!iter.done();iter++){
           IntVector c = *iter;
           displacement[c] = displacementOld[c] + gvelocity_star[c] * delT;
        }
      }  // d_doGridReset

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
  
  ASSERT(level->hasCoarserLevel() );
  
  //__________________________________
  //  Initialize the interior nodes
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    
    printTask(patches, patch,cout_doing,"Doing AMRMPM::computeZoneOfInfluence");
    NCVariable<Stencil7> zoi;
    new_dw->allocateAndPut(zoi, lb->gZOILabel, 0, patch);
    
    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      zoi[c].p=-9876543210e99;
      zoi[c].w= dx.x();
      zoi[c].e= dx.x();
      zoi[c].s= dx.y();
      zoi[c].n= dx.y();
      zoi[c].b= dx.z();
      zoi[c].t= dx.z();
    }
  }
  
  //__________________________________
  // Set the ZOI on the current level.
  // Look up at the finer level patches
  // for coarse-fine interfaces
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    NCVariable<Stencil7> zoi;
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
            IntVector dir = patch->getFaceAxes(patchFace);         // face axes
            int p_dir = dir[0];                                    // normal direction 
            
            // eject if this is not the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              
//              cout << "  A) Setting ZOI  " 
//                   << " \t On L-" << level->getIndex() << " patch  " << patch->getID()
//                   << ", beneath patch " << finePatch->getID() << ", face: "  << finePatch->getFaceName(patchFace) 
//                   << ", isRight_CP_FP_pair: " << isRight_CP_FP_pair  << " n_iter: " << n_iter << endl;
              
              for(; !n_iter.done(); n_iter++) {
                IntVector c = *n_iter;
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
    const Patch* finePatch = patches->get(p);
    NCVariable<Stencil7> zoi_fine;
    new_dw->getModifiable(zoi_fine, lb->gZOILabel, 0,finePatch);
      
    // underlying coarse level
    if( level->hasCoarserLevel() ) {
      
      Level::selectType coarsePatches;
      finePatch->getCoarseLevelPatches(coarsePatches);
      //__________________________________
      // Iterate over coarsefine interface faces
      if(finePatch->hasCoarseFaces() ){
        vector<Patch::FaceType> cf;
        finePatch->getCoarseFaces(cf);

        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = cf.begin(); iter != cf.end(); ++iter){
          Patch::FaceType patchFace = *iter;
          bool setFace = false;
            
          for(int p=0;p<coarsePatches.size();p++){  
            const Patch* coarsePatch = coarsePatches[p];
            Vector coarse_dx = coarsePatch->dCell();

            // determine the iterator on the coarse level.
            NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
            bool isRight_CP_FP_pair;
            
            fineLevel_CFI_NodeIterator( patchFace,coarsePatch, finePatch,
                                          n_iter ,isRight_CP_FP_pair);
                                          
            int element = patchFace;
            
            IntVector dir = finePatch->getFaceAxes(patchFace);        // face axes
            int p_dir = dir[0];                                       // normal direction 
            

            
            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              setFace = true; 
                   
//              cout << "  B) Setting ZOI  " 
//                   << " \t On L-" << level->getIndex() << " patch  " << finePatch->getID()
//                   << "   CFI face: "  << finePatch->getFaceName(patchFace) 
//                   << " isRight_CP_FP_pair: " << isRight_CP_FP_pair  << " n_iter: " << n_iter << endl;             
              
              for(; !n_iter.done(); n_iter++) {
                IntVector c = *n_iter;
                zoi_fine[c][element]=coarse_dx[p_dir];
              }
            }

          }  // coarsePatches loop
          
          // bulletproofing
          if( !setFace ){                                                               
              ostringstream warn;                                                      
              warn << "\n ERROR: computeZoneOfInfluence:Fine Level: Did not find node   iterator! "
                   << "\n coarse: L-" << level->getIndex()                       
                   << "\n coarePatches size: " << coarsePatches.size()                              
                   << "\n fine patch:   " << *finePatch                                
                   << "\n fine patch face: " << finePatch->getFaceName(patchFace);     

              throw ProblemSetupException(warn.str(), __FILE__, __LINE__);             
          }                                                                            
        }  // face interator
      }  // patch has coarse face 
    }  // has finer level
  }  // patch loop

}

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
    
    printTask(patches, patch,cout_doing,"Doing AMRMPM::applyExternalLoads");
    
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
        pExternalForce_new[idx] = pExternalForce[idx]*flags->d_forceIncrementFactor;
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
    printTask(patches, patch,cout_doing, "Doing AMRMPM::interpolateToParticlesAndUpdate");
    
    double totalmass = 0;
    double thermal_energy = 0.0;
    double ke = 0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

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
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      ParticleVariable<double> pTempPreNew;

      // Get the arrays of grid data on which the new particle values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      double Cp =mpm_matl->getSpecificHeat();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pids,         lb->pParticleIDLabel,                pset);
      old_dw->get(pvolume,      lb->pVolumeLabel,                    pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
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
      new_dw->get(gvelocity_star,    lb->gVelocityStarLabel,   dwi,patch,gac,NGN);
      new_dw->get(gacceleration,     lb->gAccelerationLabel,   dwi,patch,gac,NGN);

      // Loop over particles
      int n8or27=flags->d_8or27;
         
      for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell                
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);

        Vector acc(0.0, 0.0, 0.0); 
        Vector vel(0.0, 0.0, 0.0);

        // Accumulate the contribution from vertices on this level
       for(int k = 0; k < n8or27; k++) {
          IntVector node = ni[k];
          //S[k] *= pErosion[idx];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];
        }
        
        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]         + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx]      + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]  + acc*delT;
        
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx];
        pTempPreNew[idx]     = pTemperature[idx]; //
        pmassNew[idx]        = pmass[idx];
        pvolumeNew[idx]      = pvolume[idx];
        
/*`==========TESTING==========*/
#ifdef DEBUG_VEL
        Vector diff = ( pvelocitynew[idx] - d_vel_ans );
       if( abs(diff.length() ) > d_vel_tol ) {
         cout << "    L-"<< getLevel(patches)->getIndex() << " px: "<< pxnew[idx] << " pvelocitynew: " << pvelocitynew[idx] <<  " pvelocity " << pvelocity[idx]
                         << " diff " << diff << endl;
       }
#endif 
#ifdef DEBUG_ACC 
  #if 1
       if( abs(acc.length() - d_acc_ans.length() ) > d_acc_tol ) {
         cout << "    L-"  << getLevel(patches)->getIndex() << " particle: "<< idx 
              <<  " cell: " << getLevel(patches)->getCellIndex(px[idx])
              <<  " acc: " << acc 
              <<  " diff: " << acc.length() - d_acc_ans.length() << endl;
       }   
  #endif                                                                                    
#endif
/*===========TESTING==========`*/ 
        
        totalmass      += pmass[idx];
        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke             += .5*pmass[idx] * pvelocitynew[idx].length2();
        CMX            = CMX + (pxnew[idx] * pmass[idx]).asVector();
        totalMom       += pvelocitynew[idx] * pmass[idx];
      }
      new_dw->deleteParticles(delset);  
      
      new_dw->put(sum_vartype(totalmass),       lb->TotalMassLabel);
      new_dw->put(sum_vartype(ke),              lb->KineticEnergyLabel);
      new_dw->put(sum_vartype(thermal_energy),  lb->ThermalEnergyLabel);
      new_dw->put(sumvec_vartype(CMX),          lb->CenterOfMassPositionLabel);
      new_dw->put(sumvec_vartype(totalMom),     lb->TotalMomentumLabel);
#ifndef USE_DEBUG_TASK  
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

//______________________________________________________________________
//
void AMRMPM::interpolateToParticlesAndUpdate_CFI(const ProcessorGroup*,
                                                 const PatchSubset* coarsePatches,
                                                 const MaterialSubset* ,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), coarseLevel );
  
  double move_particles=1.;
  if(!flags->d_doGridReset){
    move_particles=0.;
  }
  
  //__________________________________
  //Loop over the coarse level patches
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    printTask(coarsePatches,coarsePatch,cout_doing,"AMRMPM::interpolateToParticlesAndUpdate_CFI");

    int numMatls = d_sharedState->getNumMPMMatls();
        
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);
    
    //__________________________________
    //  Fine patch loop
    for(int i=0;i<finePatches.size();i++){
      const Patch* finePatch = finePatches[i]; 
      
      if(finePatch->hasCoarseFaces()){

        ParticleInterpolator* interpolator = flags->d_interpolator->clone(finePatch);
        
        constNCVariable<Stencil7> zoi_fine;
        new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

        for(int m = 0; m < numMatls; m++){
          MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
          int dwi = mpm_matl->getDWIndex();

          // get fine level grid data
          constNCVariable<double> gmass_fine;
          constNCVariable<Vector> gvelocity_star_fine;
          constNCVariable<Vector> gacceleration_fine;
          
          Ghost::GhostType  gn  = Ghost::None;
          new_dw->get(gvelocity_star_fine,  lb->gVelocityStarLabel, dwi, finePatch, gn, 0);
          new_dw->get(gacceleration_fine,   lb->gAccelerationLabel, dwi, finePatch, gn, 0);
          
          // get coarse level particle data
          ParticleVariable<Point>  pxnew_coarse;
          ParticleVariable<Vector> pdispnew_coarse;
          ParticleVariable<Vector> pvelocitynew_coarse;
          constParticleVariable<Point>  pxold_coarse;
          
          ParticleSubset* pset=0;
          
/*`==========TESTING==========*/
          // get the particles for the entire coarse patch
          // Ideally you only need the particle subset in the cells
          // that surround the fine patch.  Currently, getModifiable doesn't
          // allow you to get a pset with a high/low index that does not match
          // the patch low high index  
          pset = old_dw->getParticleSubset(dwi, coarsePatch);
          //cout << *pset << endl; 
/*===========TESTING==========`*/
          old_dw->get(pxold_coarse,                  lb->pXLabel,                 pset);
          new_dw->getModifiable(pxnew_coarse,        lb->pXLabel_preReloc,        pset);
          new_dw->getModifiable(pdispnew_coarse,     lb->pDispLabel_preReloc,     pset);
          new_dw->getModifiable(pvelocitynew_coarse, lb->pVelocityLabel_preReloc, pset);
          

          
          
          for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
            particleIndex idx = *iter;

            // Get the node indices that surround the fine patch cell
            vector<IntVector> ni;
            vector<double> S;
            
            interpolator->findCellAndWeights_CFI(pxold_coarse[idx],ni,S,zoi_fine);

            Vector acc(0.0, 0.0, 0.0); 
            Vector vel(0.0, 0.0, 0.0);

            // Add each nodes contribution to the particle's velocity & acceleration 
            IntVector fineNode;
            for(int k = 0; k < (int)ni.size(); k++) {
              
              fineNode = ni[k];
            
              vel  += gvelocity_star_fine[fineNode] * S[k];
              acc  += gacceleration_fine[fineNode]  * S[k];
/*`==========TESTING==========*/
#ifdef DEBUG_ACC 
              Vector diff = acc - d_acc_ans;
              if( abs(acc.length() - d_acc_ans.length() > d_acc_tol ) ) {
                const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
                cout << "    L-"<< fineLevel->getIndex() << " node: "<< fineNode << " gacceleration: " << gacceleration_fine[fineNode] 
                     << "  diff " << diff << endl;
              }
#endif 
/*===========TESTING==========`*/
            }
            
//            cout << " pvelocitynew_coarse  "<< idx << " "  << pvelocitynew_coarse[idx] << " p.x " << pxnew_coarse[idx] ;
            
            // Update the particle's position and velocity
            pxnew_coarse[idx]         += vel*delT*move_particles;  
            pdispnew_coarse[idx]      += vel*delT;                 
            pvelocitynew_coarse[idx]  += acc*delT; 
            
          } // End of particle loop
        } // End loop over materials 
      
      delete interpolator;
      }  // if has coarse face
    }  // End loop over fine patches 
  }  // End loop over patches
}

//______________________________________________________________________
//
void
AMRMPM::errorEstimate(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse*,
                      DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
    
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing AMRMPM::initialErrorEstimate");

    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch);
    new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(), 0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    
    // loop over all the geometry objects
    for(int obj=0; obj<(int)d_refine_geom_objs.size(); obj++){
      GeometryPieceP piece = d_refine_geom_objs[obj]->getPiece();
      Vector dx = patch->dCell();
      
      int geom_level =  d_refine_geom_objs[obj]->getInitialData_int("level");
     
      //don't add refinement flags if the current level is greater than the geometry level specification
      if(geom_level!=-1 && level->getIndex()>=geom_level)
        continue;

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        Point  lower  = patch->nodePosition(c);
        Vector upperV = lower.asVector() + dx; 
        Point  upper  = upperV.asPoint();
        
        if(piece->inside(upper) && piece->inside(lower))
          refineFlag[c] = true;
          refinePatch->set();
      }
    }

#if 0    
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      // Loop over particles
      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      
      constParticleVariable<Point> px;
      new_dw->get(px, lb->pXLabel, pset);
      
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
        IntVector c = level->getCellIndex(px[*iter]);
        refineFlag[c] = true;
        refinePatch->set();
      }
    }
#endif
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
    printTask(patches, patch,cout_doing,"Doing AMRMPM::refine");

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
        ParticleVariable<Vector> pvelocity, pexternalforce, pdisp;
        ParticleVariable<Matrix3> psize;
        ParticleVariable<double> pTempPrev;
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

        mpm_matl->getConstitutiveModel()->initializeCMData(patch,
                                                           mpm_matl,new_dw);
      }
    }
  }
} // end refine()
//______________________________________________________________________
// Debugging Task that counts the number of particles in the domain.
void AMRMPM::scheduleCountParticles(const PatchSet* patches,
                                    SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleCountParticles");
  Task* t = scinew Task("AMRMPM::countParticles",this, 
                        &AMRMPM::countParticles);
  t->computes(lb->partCountLabel);
  sched->addTask(t, patches, d_sharedState->allMPMMaterials());
}
//______________________________________________________________________
//
void AMRMPM::countParticles(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,                           
                            DataWarehouse* new_dw)
{
  long int totalParticles=0;
  int numMPMMatls = d_sharedState->getNumMPMMatls();
  
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    
    printTask(patches,patch,cout_doing,"Doing AMRMPM::countParticles");
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      totalParticles += pset->end() - pset->begin();
    }
  }
  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}

//______________________________________________________________________
//  

//______________________________________________________________________
// This task colors the particles that are retrieved from the coarse level and
// used on the CFI.  This task mimics interpolateParticlesToGrid_CFI
void AMRMPM::scheduleDebug_CFI(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSet* matls)
{
  const Level* level = getLevel(patches);

  printSchedule(patches,cout_doing,"AMRMPM::scheduleDebug_CFI");

  Task* t = scinew Task("AMRMPM::debug_CFI",
                   this,&AMRMPM::debug_CFI);
                   
   Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, lb->pXLabel,                  gn,0);
  t->requires(Task::OldDW, lb->pSizeLabel,               gn,0);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gn,0);
  
  if(level->hasFinerLevel()){ 
    #define allPatches 0
    #define allMatls 0
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    t->requires(Task::NewDW, lb->gZOILabel, allPatches, Task::FineLevel,allMatls, ND, gn, 0);
  }
  
  t->computes(lb->pColorLabel_preReloc);

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
void AMRMPM::debug_CFI(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* ,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  
  for(int cp=0; cp<patches->size(); cp++){
    const Patch* patch = patches->get(cp);

    printTask(patches,patch,cout_doing,"Doing AMRMPM::debug_CFI");
    
    
    //__________________________________
    //  Write p.color all levels all patches  
    ParticleSubset* pset=0;
    int dwi = 0;
    pset = old_dw->getParticleSubset(dwi, patch);
    
    constParticleVariable<Point>  px;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<Matrix3> pDeformationMeasure;
    ParticleVariable<double>  pColor;
    
    old_dw->get(px,                   lb->pXLabel,                  pset);
    old_dw->get(psize,                lb->pSizeLabel,               pset);
    old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);
    new_dw->allocateAndPut(pColor,    lb->pColorLabel_preReloc,     pset);
    
    ParticleInterpolator* interpolatorCoarse = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolatorCoarse->size());
    vector<double> S(interpolatorCoarse->size());

    for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pColor[idx] = 0;
      
      interpolatorCoarse->findCellAndWeights(px[idx],ni,S,psize[idx],pDeformationMeasure[idx]);
      
      for(int k = 0; k < (int)ni.size(); k++) {
        pColor[idx] += S[k];
      }
    }  

    //__________________________________
    //  Mark the particles that are accessed at the CFI.
    if(level->hasFinerLevel()){  

      // find the fine patches over the coarse patch.  Add a single layer of cells
      // so you will get the required patches when coarse patch and fine patch boundaries coincide.
      Level::selectType finePatches;
      patch->getOtherLevelPatches(1, finePatches, 1);

      const Level* fineLevel = level->getFinerLevel().get_rep();
      IntVector refineRatio(fineLevel->getRefinementRatio());

      for(int fp=0; fp<finePatches.size(); fp++){
        const Patch* finePatch = finePatches[fp];

        // Determine extents for coarser level particle data
        // Linear Interpolation:  1 layer of coarse level cells
        // Gimp Interpolation:    2 layers
    /*`==========TESTING==========*/
        IntVector nLayers(d_nPaddingCells_Coarse, d_nPaddingCells_Coarse, d_nPaddingCells_Coarse);
        IntVector nPaddingCells = nLayers * (refineRatio);
        //cout << " nPaddingCells " << nPaddingCells << "nLayers " << nLayers << endl;
    /*===========TESTING==========`*/

        int nGhostCells = 0;
        bool returnExclusiveRange=false;
        IntVector cl_tmp, ch_tmp, fl, fh;

        getCoarseLevelRange(finePatch, level, cl_tmp, ch_tmp, fl, fh, 
                            nPaddingCells, nGhostCells,returnExclusiveRange);
                            
        cl_tmp -= finePatch->neighborsLow() * nLayers;  //  expand cl_tmp when a neighor patch exists.
                                                        //  This patch owns the low nodes.  You need particles
                                                        //  from the neighbor patch.

        // coarseLow and coarseHigh cannot lie outside of the coarse patch
        IntVector cl = Max(cl_tmp, patch->getCellLowIndex());
        IntVector ch = Min(ch_tmp, patch->getCellHighIndex());

        ParticleSubset* pset2=0;
        pset2 = old_dw->getParticleSubset(dwi, cl, ch, patch,lb->pXLabel);
        
        constParticleVariable<Point>  px_CFI;
        constNCVariable<Stencil7> zoi;
        old_dw->get(px_CFI, lb->pXLabel,  pset2);
        new_dw->get(zoi,    lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

        ParticleInterpolator* interpolatorFine = flags->d_interpolator->clone(finePatch);

        for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
          particleIndex idx = *iter;
          
          for (ParticleSubset::iterator iter2 = pset2->begin();iter2 != pset2->end(); iter2++){
            particleIndex idx2 = *iter2;
            
            if( px[idx] == px_CFI[idx2] ){       
              pColor[idx] = 0;
              vector<IntVector> ni;
              vector<double> S;
              interpolatorFine->findCellAndWeights_CFI(px[idx],ni,S,zoi); 
              for(int k = 0; k < (int)ni.size(); k++) {
                pColor[idx] += S[k];
              }
            }
          }  // pset2 loop
        }  // pset loop
        delete interpolatorFine;
      }  // loop over fine patches
    }  //// hasFinerLevel
    delete interpolatorCoarse;
  }  // End loop over coarse patches
}

//______________________________________________________________________
//  Returns the patches that have coarse fine interfaces
void AMRMPM::coarseLevelCFI_Patches(const PatchSubset* coarsePatches,
                                    Level::selectType& CFI_patches )
{
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);

    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);
    // loop over all the coarse level patches

    for(int fp=0;fp<finePatches.size();fp++){  
      const Patch* finePatch = finePatches[fp];
      
      if(finePatch->hasCoarseFaces() ){
        CFI_patches.push_back( coarsePatch );
      }
    }
  }
}
