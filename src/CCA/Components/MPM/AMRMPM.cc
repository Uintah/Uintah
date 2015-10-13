/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/Contact/ContactFactory.h>
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/SDInterfaceModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/SDInterfaceModelFactory.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
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
#include <Core/Grid/AMR_CoarsenRefine.h>
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
#undef CBDI_FLUXBCS

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
  //lb = scinew MPMLabel();
  //flags = scinew MPMFlags(myworld);
  flags->d_minGridLevel = 0;
  flags->d_maxGridLevel = 1000;

  d_SMALL_NUM_MPM=1e-200;
  contactModel   = 0;
  sdInterfaceModel = 0;
  NGP     = -9;
  NGN     = -9;
  d_nPaddingCells_Coarse = -9;
  dataArchiver = 0;
  d_acc_ans = Vector(0,0,0);
  d_acc_tol = 1e-7;
  d_vel_ans = Vector(-100,0,0);
  d_vel_tol = 1e-7;
  
  pDbgLabel = VarLabel::create("p.dbg",
                               ParticleVariable<double>::getTypeDescription());
  gSumSLabel= VarLabel::create("g.sum_S",
                               NCVariable<double>::getTypeDescription());
  RefineFlagXMaxLabel = VarLabel::create("RefFlagXMax",
                                         max_vartype::getTypeDescription() );
  RefineFlagXMinLabel = VarLabel::create("RefFlagXMin",
                                         min_vartype::getTypeDescription() );
  RefineFlagYMaxLabel = VarLabel::create("RefFlagYMax",
                                         max_vartype::getTypeDescription() );
  RefineFlagYMinLabel = VarLabel::create("RefFlagYMin",
                                         min_vartype::getTypeDescription() );
  RefineFlagZMaxLabel = VarLabel::create("RefFlagZMax",
                                         max_vartype::getTypeDescription() );
  RefineFlagZMinLabel = VarLabel::create("RefFlagZMin",
                                         min_vartype::getTypeDescription() );
  
  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();
}

AMRMPM::~AMRMPM()
{
//  delete lb;
//  delete flags;
  if(flags->d_doScalarDiffusion){
    delete sdInterfaceModel;
  }
  
  VarLabel::destroy(pDbgLabel);
  VarLabel::destroy(gSumSLabel);
  VarLabel::destroy(RefineFlagXMaxLabel);
  VarLabel::destroy(RefineFlagYMaxLabel);
  VarLabel::destroy(RefineFlagZMaxLabel);
  VarLabel::destroy(RefineFlagXMinLabel);
  VarLabel::destroy(RefineFlagYMinLabel);
  VarLabel::destroy(RefineFlagZMinLabel);
  
  if (d_one_matl->removeReference())
    delete d_one_matl;
  
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

  dataArchiver = dynamic_cast<Output*>(getPort("output"));

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
    flags->d_AMR=true;
  } else {
    string warn;
    warn ="\n INPUT FILE ERROR:\n <AMR>  block not found in input file \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
  
  
  if(!mpm_ps){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <MPM>  block not found inside of <AMR> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }

#if 0
  ProblemSpecP refine_ps = mpm_ps->findBlock("Refine_Regions");
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
 }  // if(refine_ps)
#endif

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

  MPMPhysicalBCFactory::create(mat_ps, grid, flags);
  
  contactModel = ContactFactory::create(UintahParallelComponent::d_myworld,
                                        mat_ps,sharedState,lb,flags);

  // Determine extents for coarser level particle data
  // Linear Interpolation:  1 layer of coarse level cells
  // Gimp Interpolation:    2 layers
  
/*`==========TESTING==========*/
  d_nPaddingCells_Coarse = 1;
//  NGP = 1; 
/*===========TESTING==========`*/

  d_sharedState->setParticleGhostLayer(Ghost::AroundNodes, NGP);

  materialProblemSetup(mat_ps, d_sharedState,flags);

  if(flags->d_doScalarDiffusion){
    sdInterfaceModel = SDInterfaceModelFactory::create(mat_ps, sharedState, flags);
  }
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
  contactModel->outputProblemSpec(mpm_ps);
  if (flags->d_doScalarDiffusion){
     sdInterfaceModel->outputProblemSpec(mpm_ps);
  }

  ProblemSpecP physical_bc_ps = root->appendChild("PhysicalBC");
  ProblemSpecP mpm_ph_bc_ps = physical_bc_ps->appendChild("MPM");
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
    MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->outputProblemSpec(mpm_ph_bc_ps);
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
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pTemperatureGradientLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pRefinedLabel);
  t->computes(lb->pLastLevelLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
  t->computes(lb->pCellNAPIDLabel,d_one_matl);

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

  if (flags->d_doScalarDiffusion){
    t->computes(lb->pConcentrationLabel);
    t->computes(lb->pConcPreviousLabel);
    t->computes(lb->pConcGradientLabel);
  }

  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  // ********** Initial computes is now done above ***************
  // Adding initial computes and requires for scalar diffusion models.
  // if (flags->d_doScalarDiffusion){
  //  sdInterfaceModel->addInitialComputesAndRequires(t, patches);
  // }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  if (level->getIndex() == 0 ) schedulePrintParticleCount(level, sched); 

  if (flags->d_useLoadCurves && !flags->d_doScalarDiffusion) {
    // Schedule the initialization of pressure BCs per particle
    cout << "Pressure load curves are untested for multiple levels" << endl;
    scheduleInitializePressureBCs(level, sched);
  }

  if (flags->d_useLoadCurves && flags->d_doScalarDiffusion) {
    // Schedule the initialization of scalar fluxBCs per particle
    cout << "Scalar load curves are untested for multiple levels" << endl;
    scheduleInitializeScalarFluxBCs(level, sched);
  }

}
//______________________________________________________________________
//
void AMRMPM::schedulePrintParticleCount(const LevelP& level, 
                                        SchedulerP& sched)
{
  Task* t = scinew Task("AMRMPM::printParticleCount",
                  this, &AMRMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  t->setType(Task::OncePerProc);

  sched->addTask(t, sched->getLoadBalancer()->getPerProcessorPatchSet(level), d_sharedState->allMPMMaterials());
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeStableTimestep(const LevelP&,
                                              SchedulerP&)
{
  // Nothing to do here - delt is computed as a by-product of the
  // constitutive model
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
    scheduleApplyExternalScalarFlux(        sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateParticlesToGrid(     sched, patches, matls);
    // Need to add a task to do the reductions on the max hydro stress - JG ???
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
    scheduleCoarsenNodalData_CFI(      sched, patches, matls, coarsenData);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleNormalizeNodalVelTempConc(sched, patches, matls);
    scheduleExMomInterpolated(        sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce(           sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeInternalForce_CFI(       sched, patches, matls);
  }

  if(flags->d_doScalarDiffusion){
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      scheduleComputeFlux(              sched, patches, matls);
      scheduleComputeDivergence(        sched, patches, matls);
    }

#if 1
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      scheduleComputeDivergence_CFI(    sched, patches, matls);
    }
#endif
  }

  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleCoarsenNodalData_CFI2( sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeAndIntegrateAcceleration(sched, patches, matls);
    scheduleExMomIntegrated(                sched, patches, matls);
    scheduleSetGridBoundaryConditions(      sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeLAndF(            sched, patches, matls);
  }

#if 0
  // zero the nodal data at the CFI on the coarse level 
  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleCoarsenNodalData_CFI( sched, patches, matls, zeroData);
  }
#endif

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  }

#if 0
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpolateToParticlesAndUpdate_CFI(sched, patches, matls);
  }
#endif

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleComputeStressTensor(            sched, patches, matls);
  }

  if(flags->d_computeScaleFactor){
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      scheduleComputeParticleScaleFactor(       sched, patches, matls);
    }
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleFinalParticleUpdate(            sched, patches, matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    if(flags->d_refineParticles){
      scheduleAddParticles(                 sched, patches, matls);
    }
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleReduceFlagsExtents(             sched, patches, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{

  const PatchSet* patches = level->eachPatch();

  if (level->getIndex() == 0) {
    const MaterialSet* matls = d_sharedState->allMPMMaterials();
    sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                      d_sharedState->d_particleState_preReloc,
                                      lb->pXLabel, 
                                      d_sharedState->d_particleState,
                                      lb->pParticleIDLabel, matls);
  }
  scheduleCountParticles(patches,sched);
}

//______________________________________________________________________
//
void AMRMPM::schedulePartitionOfUnity(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"AMRMPM::schedulePartitionOfUnity");
  Task* t = scinew Task("AMRMPM::partitionOfUnity",
                  this, &AMRMPM::partitionOfUnity);
                  
  t->requires(Task::OldDW, lb->pXLabel,    Ghost::None);

  // Carry forward and update pSize if particles change levels
  t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);
  t->requires(Task::OldDW, lb->pLastLevelLabel, Ghost::None);

  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pLastLevelLabel_preReloc);
  t->computes(lb->pPartitionUnityLabel);
  t->computes(lb->MPMRefineCellLabel, d_one_matl);

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

    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeZoneOfInfluence");
    Task* t = scinew Task("AMRMPM::computeZoneOfInfluence",
                    this, &AMRMPM::computeZoneOfInfluence);

    t->computes(lb->gZOILabel, d_one_matl);

    sched->addTask(t, patches, matls);

  }
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
  if (flags->d_GEVelProj) {
    t->requires(Task::OldDW, lb->pVelGradLabel,          gan,NGP);
  }
  t->requires(Task::OldDW, lb->pXLabel,                  gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,        gan,NGP);
  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,      gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureRateLabel);
  t->computes(lb->gExternalForceLabel);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, lb->pStressLabel,             gan, NGP);
    t->requires(Task::OldDW, lb->pConcentrationLabel,      gan, NGP);
    t->requires(Task::NewDW, lb->pExternalScalarFluxLabel, gan, NGP);
    t->computes(lb->gConcentrationLabel);
    t->computes(lb->gHydrostaticStressLabel);
    t->computes(lb->gExternalScalarFluxLabel);
#ifdef CBDI_FLUXBCS
    if (flags->d_useLoadCurves) {
      t->requires(Task::OldDW, lb->pLoadCurveIDLabel,      gan, NGP);
    }
#endif
  }
  
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
    printSchedule(patches,cout_doing,
                  "AMRMPM::scheduleInterpolateParticlesToGrid_CFI");

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
    t->requires(Task::NewDW, lb->gZOILabel,                d_one_matl,  Ghost::None, 0);
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

    if(flags->d_doScalarDiffusion){
      t->requires(Task::OldDW, lb->pConcentrationLabel,      allPatches,
                                      Task::CoarseLevel,allMatls, ND, gac, npc);
      t->requires(Task::OldDW, lb->pStressLabel,             allPatches,
                                      Task::CoarseLevel,allMatls, ND, gac, npc);
      t->requires(Task::NewDW, lb->pExternalScalarFluxLabel, allPatches,
                                      Task::CoarseLevel,allMatls, ND, gac, npc);
 
      t->modifies(lb->gConcentrationLabel);
      t->modifies(lb->gHydrostaticStressLabel);
      t->modifies(lb->gExternalScalarFluxLabel);
    }

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

  printSchedule(patches,cout_doing,"AMRMPM::scheduleCoarsenNodalData_CFI"+txt);

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

  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->gConcentrationLabel,      allPatches,
                              Task::FineLevel,allMatls, ND, gn, 0);
    t->modifies(lb->gConcentrationLabel);
    t->requires(Task::NewDW, lb->gExternalScalarFluxLabel, allPatches,
                              Task::FineLevel,allMatls, ND, gn, 0);
    t->modifies(lb->gExternalScalarFluxLabel);
  }

  if (flag == zeroData){
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

  t->requires(Task::NewDW, lb->gMassLabel,          allPatches,
                                           Task::FineLevel,allMatls, ND, gn,0);
  t->requires(Task::NewDW, lb->gInternalForceLabel, allPatches,
                                           Task::FineLevel,allMatls, ND, gn,0);
  
  t->modifies(lb->gInternalForceLabel);
  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->gConcentrationRateLabel, allPatches,
                                           Task::FineLevel,allMatls, ND, gn,0);
    t->modifies(lb->gConcentrationRateLabel);
  }

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//  compute the nodal velocity and temperature after coarsening the fine
//  nodal data
void AMRMPM::scheduleNormalizeNodalVelTempConc(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleNormalizeNodalVelTempConc");

  Task* t = scinew Task("AMRMPM::normalizeNodalVelTempConc",
                   this,&AMRMPM::normalizeNodalVelTempConc);
                   
  t->requires(Task::NewDW, lb->gMassLabel,  Ghost::None);

  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gTemperatureLabel);
  if(flags->d_doScalarDiffusion){
    t->modifies(lb->gConcentrationLabel);
    t->computes(lb->gConcentrationNoBCLabel);
    t->modifies(lb->gHydrostaticStressLabel);
  }

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
  t->requires(Task::NewDW,lb->pSizeLabel_preReloc,        gan,NGP);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,  gan,NGP);
  if(flags->d_artificial_viscosity){
    t->requires(Task::OldDW, lb->p_qLabel,                gan,NGP);
  }
  
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
    t->requires(Task::NewDW, lb->gZOILabel,     d_one_matl, Ghost::None,0);
    t->requires(Task::OldDW, lb->pXLabel,       allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pStressLabel,  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    t->requires(Task::OldDW, lb->pVolumeLabel,  allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    
    if(flags->d_artificial_viscosity){
      t->requires(Task::OldDW, lb->p_qLabel,    allPatches, Task::CoarseLevel,allMatls, ND, gac, npc);
    }
    
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

  printSchedule(patches,cout_doing,
                             "AMRMPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("AMRMPM::computeAndIntegrateAcceleration",
                  this, &AMRMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  // This stuff should probably go in its own task, but for expediency...JG
  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->gConcentrationNoBCLabel,  Ghost::None);
    t->requires(Task::NewDW, lb->gConcentrationLabel,      Ghost::None);
    t->requires(Task::NewDW, lb->gExternalScalarFluxLabel, Ghost::None);
    t->modifies(lb->gConcentrationRateLabel);
    t->computes(lb->gConcentrationStarLabel);
  }

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
void AMRMPM::scheduleComputeLAndF(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls)

{
  const Level* level = getLevel(patches);
  if (!flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeLAndF");
  
  Task* t=scinew Task("AMRMPM::computeLAndF",
                this, &AMRMPM::computeLAndF);
                
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  
  t->requires(Task::OldDW, lb->pXLabel,                            gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                         gnone);
  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,                gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,           gnone);

  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->gConcentrationStarLabel,       gac,NGN);
    t->computes(lb->pConcGradientLabel_preReloc);
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

  printSchedule(patches,cout_doing,
                        "AMRMPM::scheduleInterpolateToParticlesAndUpdate");
  
  Task* t=scinew Task("AMRMPM::interpolateToParticlesAndUpdate",
                this, &AMRMPM::interpolateToParticlesAndUpdate);
                
  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             gac,NGN);
  
  t->requires(Task::OldDW, lb->pXLabel,                            gnone);
  t->requires(Task::OldDW, lb->pMassLabel,                         gnone);
  t->requires(Task::OldDW, lb->pParticleIDLabel,                   gnone);
  t->requires(Task::OldDW, lb->pTemperatureLabel,                  gnone);
  t->requires(Task::OldDW, lb->pVelocityLabel,                     gnone);
  t->requires(Task::OldDW, lb->pDispLabel,                         gnone);
  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,                gnone);
  t->requires(Task::OldDW, lb->pVolumeLabel,                       gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,           gnone);
  t->requires(Task::OldDW, lb->pLocalizedMPMLabel,              gnone);

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pLocalizedMPMLabel_preReloc);
  t->computes(lb->pXXLabel);

  // Carry Forward particle refinement flag
  if(flags->d_refineParticles){
    t->requires(Task::OldDW, lb->pRefinedLabel,                Ghost::None);
    t->computes(             lb->pRefinedLabel_preReloc);
  }

  if(flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, lb->pConcentrationLabel,           gnone);
    t->requires(Task::NewDW, lb->gConcentrationRateLabel,       gac, NGN);

    t->computes(lb->pConcentrationLabel_preReloc);
    t->computes(lb->pConcPreviousLabel_preReloc);
  }

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

void AMRMPM::scheduleComputeParticleScaleFactor(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                        "AMRMPM::scheduleComputeParticleScaleFactor");

  Task* t=scinew Task("AMRMPM::computeParticleScaleFactor",this,
                      &AMRMPM::computeParticleScaleFactor);

  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,                Ghost::None);
  t->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,  Ghost::None);
  t->computes(lb->pScaleFactorLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleFinalParticleUpdate(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AMRMPM::scheduleFinalParticleUpdate");

  Task* t=scinew Task("AMRMPM::finalParticleUpdate",
                      this, &AMRMPM::finalParticleUpdate);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->pdTdtLabel,                      gnone);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,             gnone);

  t->modifies(lb->pTemperatureLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void AMRMPM::scheduleAddParticles(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

    printSchedule(patches,cout_doing,"AMRMPM::scheduleAddParticles");

    Task* t=scinew Task("AMRMPM::addParticles",this,
                        &AMRMPM::addParticles);

    t->modifies(lb->pParticleIDLabel_preReloc);
    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pVolumeLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);
    t->modifies(lb->pMassLabel_preReloc);
    t->modifies(lb->pSizeLabel_preReloc);
    t->modifies(lb->pDispLabel_preReloc);
    t->modifies(lb->pStressLabel_preReloc);
    t->modifies(lb->pColorLabel_preReloc);
    t->modifies(lb->pLocalizedMPMLabel_preReloc);
    t->modifies(lb->pExtForceLabel_preReloc);
    t->modifies(lb->pTemperatureLabel_preReloc);
    t->modifies(lb->pTempPreviousLabel_preReloc);
    t->modifies(lb->pDeformationMeasureLabel_preReloc);
    t->modifies(lb->pRefinedLabel_preReloc);
    t->modifies(lb->pScaleFactorLabel_preReloc);
    t->modifies(lb->pLastLevelLabel_preReloc);
    t->modifies(lb->pVelGradLabel_preReloc);
    t->modifies(lb->MPMRefineCellLabel, d_one_matl);

    t->requires(Task::OldDW, lb->pCellNAPIDLabel, d_one_matl, Ghost::None);
    t->computes(             lb->pCellNAPIDLabel, d_one_matl);

    sched->addTask(t, patches, matls);
}


void AMRMPM::scheduleReduceFlagsExtents(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  const Level* level = getLevel(patches);

//  if( !level->hasFinerLevel() ){
  if(level->getIndex() > 0 ){
    printSchedule(patches,cout_doing,"AMRMPM::scheduleReduceFlagsExtents");

    Task* t=scinew Task("AMRMPM::reduceFlagsExtents",
                        this, &AMRMPM::reduceFlagsExtents);

    t->requires(Task::NewDW, lb->MPMRefineCellLabel, d_one_matl, Ghost::None);

    t->computes(RefineFlagXMaxLabel);
    t->computes(RefineFlagXMinLabel);
    t->computes(RefineFlagYMaxLabel);
    t->computes(RefineFlagYMinLabel);
    t->computes(RefineFlagZMaxLabel);
    t->computes(RefineFlagZMinLabel);

    sched->addTask(t, patches, matls);
  }
}

//______________________________________________________________________
////
void AMRMPM::scheduleRefine(const PatchSet* patches, SchedulerP& sched)
{
  printSchedule(patches,cout_doing,"AMRMPM::scheduleRefine");
  Task* t = scinew Task("AMRMPM::refineGrid", this, &AMRMPM::refineGrid);

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
  t->computes(lb->pLastLevelLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->pRefinedLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pCellNAPIDLabel, d_one_matl);

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

  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }
  t->computes(d_sharedState->get_delt_label(),getLevel(patches));


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
void AMRMPM::scheduleCoarsen(const LevelP& coarseLevel,
                             SchedulerP& sched)
{
  // Coarsening the refineCell data so that errorEstimate will have it
  // on all levels
  Ghost::GhostType  gn = Ghost::None;

  Task* task = scinew Task("AMRMPM::coarsen",this,
                           &AMRMPM::coarsen);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.  
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const PatchSet* patch_set = coarseLevel->eachPatch();

  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph

  task->requires(Task::NewDW, lb->MPMRefineCellLabel,
               0, Task::FineLevel,  d_one_matl,oims, gn, 0, fat);

  task->requires(Task::NewDW, RefineFlagXMaxLabel);
  task->requires(Task::NewDW, RefineFlagXMinLabel);
  task->requires(Task::NewDW, RefineFlagYMaxLabel);
  task->requires(Task::NewDW, RefineFlagYMinLabel);
  task->requires(Task::NewDW, RefineFlagZMaxLabel);
  task->requires(Task::NewDW, RefineFlagZMinLabel);

  task->modifies(lb->MPMRefineCellLabel, d_one_matl, oims, fat);

  sched->addTask(task, patch_set, all_matls);
}

void AMRMPM::coarsen(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  GridP grid = coarseLevel->getGrid();
  int numLevels = grid->numLevels();
  IntVector RR = fineLevel->getRefinementRatio();

  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);
    cout_doing << "  patch " << coarsePatch->getID()<< endl;

    CCVariable<double> refineCell;
    new_dw->getModifiable(refineCell, lb->MPMRefineCellLabel, 0, coarsePatch);
    bool computesAve = true;

    fineToCoarseOperator<double>(refineCell, computesAve,
                       lb->MPMRefineCellLabel, 0,   new_dw,
                       coarsePatch, coarseLevel, fineLevel);

    if( coarseLevel->getIndex() == numLevels - 2 ){
//    cout << "coarseLevelIndex = " << coarseLevel->getIndex() << endl;
      max_vartype xmax,ymax,zmax;
      min_vartype xmin,ymin,zmin;
      new_dw->get(xmax, RefineFlagXMaxLabel);
      new_dw->get(ymax, RefineFlagYMaxLabel);
      new_dw->get(zmax, RefineFlagZMaxLabel);
      new_dw->get(xmin, RefineFlagXMinLabel);
      new_dw->get(ymin, RefineFlagYMinLabel);
      new_dw->get(zmin, RefineFlagZMinLabel);

//    cout << "xmax = " << xmax << endl;
//    cout << "ymax = " << ymax << endl;
//    cout << "zmax = " << zmax << endl;
//    cout << "xmin = " << xmin << endl;
//    cout << "ymin = " << ymin << endl;
//    cout << "zmin = " << zmin << endl;

      IntVector fineXYZMaxMin(xmax,ymax,zmax);
      IntVector fineXYZMinMax(xmin,ymin,zmin);
      IntVector fineXZMaxYMin(xmax,ymin,zmax);
      IntVector fineXZMinYMax(xmin,ymax,zmin);
      IntVector fineXYMaxZMin(xmax,ymax,zmin);
      IntVector fineXYMinZMax(xmin,ymin,zmax);
      IntVector fineXMinYZMax(xmin,ymax,zmax);
      IntVector fineXMaxYZMin(xmax,ymin,zmin);

      IntVector coarseMinMax[8];

      coarseMinMax[0] = fineXYZMaxMin/RR;
      coarseMinMax[1] = fineXYZMinMax/RR;

    // Set the refine flags to 1 in all cells in the interior of the minimum
    // and maximum to ensure a rectangular region is refined.
    int imax,jmax,kmax,imin,jmin,kmin;
    imax = coarseMinMax[0].x();
    jmax = coarseMinMax[0].y();
    kmax = coarseMinMax[0].z();
    imin = coarseMinMax[1].x();
    jmin = coarseMinMax[1].y();
    kmin = coarseMinMax[1].z();
    for(CellIterator iter=coarsePatch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      if(c.x() >= imin && c.x() <= imax &&
         c.y() >= jmin && c.y() <= jmax &&
         c.z() >= kmin && c.z() <= kmax){
         refineCell[c]=1;
      }
    }

  }  // end if level
  } // end patch
}

//______________________________________________________________________
// Schedule to mark flags for AMR regridding
void AMRMPM::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
//  cout << "scheduleErrorEstimate" << endl;
  printSchedule(coarseLevel,cout_doing,"AMRMPM::scheduleErrorEstimate");
  
  Task* task = scinew Task("AMRMPM::errorEstimate", this, 
                           &AMRMPM::errorEstimate);

  task->modifies(d_sharedState->get_refineFlag_label(),
                                        d_sharedState->refineFlagMaterials());
  task->modifies(d_sharedState->get_refinePatchFlag_label(),
                                        d_sharedState->refineFlagMaterials());
  task->requires(Task::NewDW, lb->MPMRefineCellLabel, Ghost::None);

  sched->addTask(task, coarseLevel->eachPatch(),
                                        d_sharedState->allMPMMaterials());
}
//______________________________________________________________________
// Schedule to mark initial flags for AMR regridding
void AMRMPM::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
//  cout << "scheduleInitialErrorEstimate" << endl;
//  cout << "Doing nothing for now" << endl;
  
//  scheduleErrorEstimate(coarseLevel, sched);
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
    std::cout << "Created " << (long) pcount << " total particles" << std::endl;
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

    CCVariable<int> cellNAPID;
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
      
      particleIndex numParticles = mpm_matl->createParticles(cellNAPID, patch, new_dw);

      totalParticles+=numParticles;
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

    // ******* Initialization moved to ParticleCreator *********
    // if(flags->d_doScalarDiffusion){
    //   sdInterfaceModel->initializeSDMData(patch, new_dw);
    // }
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
  const Level* curLevel = getLevel(patches);
  int curLevelIndex = curLevel->getIndex();
  Vector dX = curLevel->dCell();
  Vector dX_fine   = 0.5*dX;
  Vector dX_coarse = 2.0*dX;
  Vector RRC = dX/dX_coarse;
  Vector RRF = dX/dX_fine;
  if(curLevel->hasFinerLevel()){
    dX_fine = curLevel->getFinerLevel()->dCell();
    RRF=dX/dX_fine;
    RRC=Vector(1./RRF.x(),1./RRF.y(),1./RRF.z());
  }
  if(curLevel->hasCoarserLevel()){
    dX_coarse = curLevel->getCoarserLevel()->dCell();
    RRC=dX/dX_coarse;
    RRF=Vector(1./RRC.x(),1./RRC.y(),1./RRC.z());
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches,patch,cout_doing,"Doing AMRMPM::partitionOfUnity");

    // Create and Initialize refine flags to be modified later
    CCVariable<double> refineCell;
    new_dw->allocateAndPut(refineCell, lb->MPMRefineCellLabel, 0, patch);
    refineCell.initialize(0.0);

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
      constParticleVariable<int>plastlevel;
      ParticleVariable<Matrix3> psizenew;
      ParticleVariable<int>plastlevelnew;
      ParticleVariable<double>partitionUnity;
    
      old_dw->get(px,                lb->pXLabel,          pset);
      old_dw->get(psize,             lb->pSizeLabel,       pset);
      old_dw->get(plastlevel,        lb->pLastLevelLabel,  pset);
      new_dw->allocateAndPut(psizenew,       lb->pSizeLabel_preReloc,     pset);
      new_dw->allocateAndPut(plastlevelnew,  lb->pLastLevelLabel_preReloc,pset);
      new_dw->allocateAndPut(partitionUnity, lb->pPartitionUnityLabel,    pset);
      
      int n8or27=flags->d_8or27;

      for (ParticleSubset::iterator iter = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Matrix3 ps = psize[idx];

        if(curLevelIndex<plastlevel[idx]){
         psizenew[idx]=Matrix3(ps(0,0)*RRC.x(),ps(0,1)*RRC.x(),ps(0,2)*RRC.x(),
                               ps(1,0)*RRC.y(),ps(1,1)*RRC.y(),ps(1,2)*RRC.y(),
                               ps(2,0)*RRC.z(),ps(2,1)*RRC.z(),ps(2,2)*RRC.z());
        } else if(curLevelIndex>plastlevel[idx]){
         psizenew[idx]=Matrix3(ps(0,0)*RRF.x(),ps(0,1)*RRF.x(),ps(0,2)*RRF.x(),
                               ps(1,0)*RRF.y(),ps(1,1)*RRF.y(),ps(1,2)*RRF.y(),
                               ps(2,0)*RRF.z(),ps(2,1)*RRF.z(),ps(2,2)*RRF.z());
        } else {
          psizenew[idx]  = psize[idx];
        }

        plastlevelnew[idx]= curLevelIndex;

        partitionUnity[idx] = 0;

        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],notUsed);

        for(int k = 0; k < n8or27; k++) {
          partitionUnity[idx] += S[k];
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

    printTask(patches,patch,cout_doing,
                           "Doing AMRMPM::interpolateParticlesToGrid");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

#ifdef CBDI_FLUXBCS
    LinearInterpolator* LPI;
    LPI = scinew LinearInterpolator(patch);
    vector<IntVector> ni_LPI(LPI->size());
    vector<double> S_LPI(LPI->size());
#endif

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<double> pConcentration;
      constParticleVariable<double> pExternalScalarFlux;
      constParticleVariable<Vector> pvelocity, pexternalforce;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      constParticleVariable<Matrix3> pStress;
      constParticleVariable<Matrix3> pVelGrad;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);

      old_dw->get(px,                   lb->pXLabel,                  pset);
      old_dw->get(pmass,                lb->pMassLabel,               pset);
      old_dw->get(pvolume,              lb->pVolumeLabel,             pset);
      old_dw->get(pvelocity,            lb->pVelocityLabel,           pset);
      old_dw->get(pTemperature,         lb->pTemperatureLabel,        pset);

#ifdef CBDI_FLUXBCS
      constParticleVariable<int> pLoadCurveID;
      if (flags->d_useLoadCurves) {
        old_dw->get(pLoadCurveID,       lb->pLoadCurveIDLabel,        pset);
      }
#endif

      new_dw->get(psize,                lb->pSizeLabel_preReloc,      pset);
      old_dw->get(pDeformationMeasure,  lb->pDeformationMeasureLabel, pset);
      new_dw->get(pexternalforce,       lb->pExtForceLabel_preReloc,  pset);
      if (flags->d_GEVelProj){
        old_dw->get(pVelGrad,           lb->pVelGradLabel,            pset);
      }
      if(flags->d_doScalarDiffusion){
        new_dw->get(pExternalScalarFlux,lb->pExternalScalarFluxLabel, pset);
        old_dw->get(pConcentration,     lb->pConcentrationLabel,      pset);
        old_dw->get(pStress,            lb->pStressLabel,             pset);
      }

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gTemperature;
      NCVariable<double> gTemperatureRate;
      NCVariable<double> gconcentration;
      NCVariable<double> gextscalarflux;
      NCVariable<double> ghydrostaticstress;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,           dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,         dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,       dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,    dwi,patch);
      new_dw->allocateAndPut(gTemperatureRate, lb->gTemperatureRateLabel,dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,  dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(d_SMALL_NUM_MPM);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureRate.initialize(0);

      if(flags->d_doScalarDiffusion){
        new_dw->allocateAndPut(gconcentration,     lb->gConcentrationLabel,
                                                               dwi,  patch);
        new_dw->allocateAndPut(ghydrostaticstress, lb->gHydrostaticStressLabel,
                                                               dwi,  patch);
        new_dw->allocateAndPut(gextscalarflux,     lb->gExternalScalarFluxLabel,
                                                                dwi,  patch);
        gconcentration.initialize(0);
        ghydrostaticstress.initialize(0);
        gextscalarflux.initialize(0);
      }
      
      Vector pmom;
      int n8or27=flags->d_8or27;

      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],
                                         pDeformationMeasure[idx]);

        pmom = pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        IntVector node;
        for(int k = 0; k < n8or27; k++) {
          node = ni[k];
          if(patch->containsNode(node)) {
            if (flags->d_GEVelProj){
              Point gpos = patch->getNodePosition(node);
              Vector distance = px[idx] - gpos;
              Vector pvel_ext = pvelocity[idx] - pVelGrad[idx]*distance;
              pmom = pvel_ext*pmass[idx];
            }
            gmass[node]          += pmass[idx]                     * S[k];
            gvelocity[node]      += pmom                           * S[k];
            gvolume[node]        += pvolume[idx]                   * S[k];
            gexternalforce[node] += pexternalforce[idx]            * S[k];
            gTemperature[node]   += pTemperature[idx] * pmass[idx] * S[k];
          }
        }
        if(flags->d_doScalarDiffusion){
          double one_third = 1./3.;
          double phydrostress = one_third*pStress[idx].Trace();
          for(int k = 0; k < n8or27; k++) {
            node = ni[k];
            if(patch->containsNode(node)) {
              ghydrostaticstress[node] += phydrostress        * pmass[idx]*S[k];
              gconcentration[node]     += pConcentration[idx] * pmass[idx]*S[k];
#ifndef CBDI_FLUXBCS
              gextscalarflux[node]     += pExternalScalarFlux[idx]        *S[k];
#endif
            }
          }
        }
      }  // End of particle loop


#ifdef CBDI_FLUXBCS
      Vector dx = patch->dCell();
      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Point flux_pos;
        if(pLoadCurveID[idx]==1){
          flux_pos=Point(px[idx].x()-0.5*psize[idx](0,0)*dx.x(),
                         px[idx].y(),
                         px[idx].z());
        }
        if(pLoadCurveID[idx]==2){
          flux_pos=Point(px[idx].x()+0.5*psize[idx](0,0)*dx.x(),
                         px[idx].y(),
                         px[idx].z());
        }
        if(pLoadCurveID[idx]==3){
          flux_pos=Point(px[idx].x(),
                         px[idx].y()+0.5*psize[idx](1,1)*dx.y(),
                         px[idx].z());
        }
//          cout << "px[idx] = "  << px[idx]  << endl;
//          cout << "flux_pos = " << flux_pos << endl;
        LPI->findCellAndWeights(flux_pos,ni_LPI,S_LPI,psize[idx],
                                       pDeformationMeasure[idx]);
        for(int k = 0; k < (int) ni_LPI.size(); k++) {
          if(patch->containsNode(ni_LPI[k])) {
            gextscalarflux[ni_LPI[k]]  += pExternalScalarFlux[idx] * S_LPI[k];
          }
        }
      }
#endif

      // gvelocity and gtemperature are divided by gmass in 
      // AMRMPM::NormalizeNodalVelTempConc() task
      
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
    printTask(finePatches,finePatch,cout_doing,
                          "Doing AMRMPM::interpolateParticlesToGrid_CFI");

    int numMatls = d_sharedState->getNumMPMMatls();
    ParticleInterpolator* interpolator =flags->d_interpolator->clone(finePatch);

    constNCVariable<Stencil7> zoi_fine;
    new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );

    // Determine extents for coarser level particle data
    // Linear Interpolation:  1 layer of coarse level cells
    // Gimp Interpolation:    2 layers
/*`==========TESTING==========*/
    IntVector nLayers(d_nPaddingCells_Coarse,
                      d_nPaddingCells_Coarse, 
                      d_nPaddingCells_Coarse);
    IntVector nPaddingCells = nLayers * (fineLevel->getRefinementRatio());
    //cout << " nPaddingCells " << nPaddingCells << "nLayers " << nLayers << endl;
/*===========TESTING==========`*/

    int nGhostCells = 0;
    bool returnExclusiveRange=false;
    IntVector cl_tmp, ch_tmp, fl, fh;

    getCoarseLevelRange(finePatch, coarseLevel, cl_tmp, ch_tmp, fl, fh, 
                        nPaddingCells, nGhostCells,returnExclusiveRange);
                        
    //  expand cl_tmp when a neighor patch exists.
    //  This patch owns the low nodes.  You need particles
    //  from the neighbor patch.
    cl_tmp -= finePatch->neighborsLow() * nLayers;

    // find the coarse patches under the fine patch.
    // You must add a single layer of padding cells.
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
      NCVariable<double> gConc_fine;
      NCVariable<double> gExtScalarFlux_fine;
      NCVariable<double> gHStress_fine;

      new_dw->getModifiable(gMass_fine,          lb->gMassLabel,         dwi,finePatch);
      new_dw->getModifiable(gVolume_fine,        lb->gVolumeLabel,       dwi,finePatch);
      new_dw->getModifiable(gVelocity_fine,      lb->gVelocityLabel,     dwi,finePatch);
      new_dw->getModifiable(gTemperature_fine,   lb->gTemperatureLabel,  dwi,finePatch);
      new_dw->getModifiable(gExternalforce_fine, lb->gExternalForceLabel,dwi,finePatch);
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConc_fine,          lb->gConcentrationLabel,    dwi,finePatch);
        new_dw->getModifiable(gExtScalarFlux_fine, lb->gConcentrationLabel,    dwi,finePatch);
        new_dw->getModifiable(gHStress_fine,       lb->gHydrostaticStressLabel,dwi,finePatch);
      }

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
        constParticleVariable<double> pConc_coarse;
        constParticleVariable<double> pExtScalarFlux_c;
        constParticleVariable<Matrix3> pStress_coarse;

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
        new_dw->get(pExternalforce_coarse, lb->pExtForceLabel_preReloc,  pset);
        if(flags->d_doScalarDiffusion){
          old_dw->get(pConc_coarse,        lb->pConcentrationLabel,      pset);
          new_dw->get(pExtScalarFlux_c,    lb->pExternalScalarFluxLabel, pset);
          old_dw->get(pStress_coarse,      lb->pStressLabel,             pset);
        }

        for (ParticleSubset::iterator iter  = pset->begin();
                                      iter != pset->end(); iter++){
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

            gMass_fine[fineNode]          += pMass_coarse[idx]          * S[k];
            gVelocity_fine[fineNode]      += pmom                       * S[k];
            gVolume_fine[fineNode]        += pVolume_coarse[idx]        * S[k];
            gExternalforce_fine[fineNode] += pExternalforce_coarse[idx] * S[k];
            gTemperature_fine[fineNode]   += pTemperature_coarse[idx] 
                                           * pMass_coarse[idx] * S[k];
          }
          if(flags->d_doScalarDiffusion){
            double one_third = 1./3.;
            double ConcMass         = pConc_coarse[idx]*pMass_coarse[idx];
            double phydrostressmass = one_third*pStress_coarse[idx].Trace()
                                               *pMass_coarse[idx];
            double pESFlux_c = pExtScalarFlux_c[idx];

            for(int k = 0; k < (int) ni.size(); k++){
              fineNode = ni[k];
              gConc_fine[fineNode]          += ConcMass         * S[k];
              gExtScalarFlux_fine[fineNode] += pESFlux_c        * S[k];
              gHStress_fine[fineNode]       += phydrostressmass * S[k];
            }
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
      NCVariable<double> gConcentration_coarse;
      NCVariable<double> gExtScalarFlux_coarse;

      new_dw->getModifiable(gMass_coarse,            lb->gMassLabel,           dwi,coarsePatch);                  
      new_dw->getModifiable(gVolume_coarse,          lb->gVolumeLabel,         dwi,coarsePatch);                  
      new_dw->getModifiable(gVelocity_coarse,        lb->gVelocityLabel,       dwi,coarsePatch);                  
      new_dw->getModifiable(gTemperature_coarse,     lb->gTemperatureLabel,    dwi,coarsePatch);                  
      new_dw->getModifiable(gExternalforce_coarse,   lb->gExternalForceLabel,  dwi,coarsePatch);
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConcentration_coarse, lb->gConcentrationLabel,  dwi,coarsePatch);
        new_dw->getModifiable(gExtScalarFlux_coarse, lb->gConcentrationLabel,  dwi,coarsePatch);
      }
      
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
          constNCVariable<double> gTemperature_fine;
          constNCVariable<Vector> gExternalforce_fine;
          constNCVariable<double> gConcentration_fine;
          constNCVariable<double> gExtScalarFlux_fine;

          if(flag == coarsenData){
            // use getRegion() instead of get().  They should be equivalent but 
            // get() throws assert on parallel runs.
            IntVector fl = finePatch->getNodeLowIndex();  
            IntVector fh = finePatch->getNodeHighIndex();
            new_dw->getRegion(gMass_fine,          lb->gMassLabel,          dwi, fineLevel,fl, fh);
            new_dw->getRegion(gVolume_fine,        lb->gVolumeLabel,        dwi, fineLevel,fl, fh);
            new_dw->getRegion(gVelocity_fine,      lb->gVelocityLabel,      dwi, fineLevel,fl, fh);
            new_dw->getRegion(gTemperature_fine,   lb->gTemperatureLabel,   dwi, fineLevel,fl, fh);
            new_dw->getRegion(gExternalforce_fine, lb->gExternalForceLabel, dwi, fineLevel,fl, fh);
            if(flags->d_doScalarDiffusion){
             new_dw->getRegion(gConcentration_fine,lb->gConcentrationLabel, dwi, fineLevel,fl, fh);
             new_dw->getRegion(gExtScalarFlux_fine,lb->gExternalScalarFluxLabel,
                                                                            dwi, fineLevel,fl, fh);
            }
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

            coarseLevel_CFI_NodeIterator(patchFace, coarsePatch,
                                         finePatch, fineLevel,
                                         n_iter, isRight_CP_FP_pair);

            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
              switch(flag){
               case coarsenData:
                for(; !n_iter.done(); n_iter++) {
                  IntVector c_node = *n_iter;
                  IntVector f_node = coarseLevel->mapNodeToFiner(c_node);
                  // only overwrite coarse data if there is non-zero fine data
                  if( gMass_fine[f_node] > 2 * d_SMALL_NUM_MPM ){
                    gMass_coarse[c_node]          = gMass_fine[f_node];
                    gVolume_coarse[c_node]        = gVolume_fine[f_node];
                    gVelocity_coarse[c_node]      = gVelocity_fine[f_node];
                    gTemperature_coarse[c_node]   = gTemperature_fine[f_node];
                    gExternalforce_coarse[c_node] = gExternalforce_fine[f_node];
                    if(flags->d_doScalarDiffusion){
                     gConcentration_coarse[c_node]= gConcentration_fine[f_node];
                     gExtScalarFlux_coarse[c_node]= gExtScalarFlux_fine[f_node];
                    }
                   } // if mass
                } // end node iterator loop
                 break;
               case zeroData:
                for(; !n_iter.done(); n_iter++) {
                  IntVector c_node = *n_iter;
                  IntVector f_node = coarseLevel->mapNodeToFiner(c_node);
                  gMass_coarse[c_node]          = 0;
                  gVolume_coarse[c_node]        = 0;
                  gVelocity_coarse[c_node]      = Vector(0,0,0);
                  gVelocityStar_coarse[c_node]  = Vector(0,0,0);
                  gAcceleration_coarse[c_node]  = Vector(0,0,0);
                  gTemperature_coarse[c_node]   = 0;
                  gExternalforce_coarse[c_node] = Vector(0,0,0);
                  if(flags->d_doScalarDiffusion){
                   gConcentration_coarse[c_node]= 0;
                   gExtScalarFlux_coarse[c_node]= 0;
                  }
                } // end node iterator loop
                break;
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
      new_dw->getModifiable(internalForce_coarse, lb->gInternalForceLabel, 
                                                               dwi,coarsePatch);
      NCVariable<double> gConcRate_coarse;                    
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConcRate_coarse,   lb->gConcentrationRateLabel, 
                                                               dwi,coarsePatch);
      }

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
          constNCVariable<double> gMass_fine,gConcRate_fine;
          constNCVariable<Vector> internalForce_fine;

          // use getRegion() instead of get().  They should be equivalent but 
          // get() throws assert on parallel runs.
          IntVector fl = finePatch->getNodeLowIndex();
          IntVector fh = finePatch->getNodeHighIndex();
          new_dw->getRegion(gMass_fine,          lb->gMassLabel,         
                                                 dwi, fineLevel, fl, fh);
          new_dw->getRegion(internalForce_fine,  lb->gInternalForceLabel,
                                                 dwi, fineLevel, fl, fh);

          if(flags->d_doScalarDiffusion){
            new_dw->getRegion(gConcRate_fine,  lb->gConcentrationRateLabel,
                                                   dwi, fineLevel, fl, fh);
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

            coarseLevel_CFI_NodeIterator(patchFace, coarsePatch,
                                         finePatch, fineLevel,
                                         n_iter, isRight_CP_FP_pair);

            // Is this the right coarse/fine patch pair
            if (isRight_CP_FP_pair){
               
              for(; !n_iter.done(); n_iter++) {
                IntVector c_node = *n_iter;

                IntVector f_node = coarseLevel->mapNodeToFiner(c_node);
 
                // only overwrite coarse data if there is non-zero fine data
                if( gMass_fine[f_node] > 2 * d_SMALL_NUM_MPM ){
               
                 internalForce_coarse[c_node] = internalForce_fine[f_node];

                 if(flags->d_doScalarDiffusion){
                   gConcRate_coarse[c_node] = gConcRate_fine[f_node];
                 }
                 
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
void AMRMPM::normalizeNodalVelTempConc(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* ,
                                       DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing AMRMPM::normalizeNodalVelTempConc");

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // get  level nodal data
      constNCVariable<double> gMass;
      NCVariable<Vector> gVelocity;
      NCVariable<double> gTemperature;
      NCVariable<double> gConcentration;
      NCVariable<double> gConcentrationNoBC;
      NCVariable<double> gHydroStress;
      Ghost::GhostType  gn = Ghost::None;
      
      new_dw->get(gMass,                  lb->gMassLabel,       dwi,patch,gn,0);
      new_dw->getModifiable(gVelocity,    lb->gVelocityLabel,   dwi,patch,gn,0);
      new_dw->getModifiable(gTemperature, lb->gTemperatureLabel,dwi,patch,gn,0);
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConcentration,      
                                    lb->gConcentrationLabel,    dwi,patch,gn,0);
        new_dw->getModifiable(gHydroStress,
                                    lb->gHydrostaticStressLabel,dwi,patch,gn,0);
        new_dw->allocateAndPut(gConcentrationNoBC,
                                    lb->gConcentrationNoBCLabel,dwi,patch);
      }
      
      //__________________________________
      //  back out the nodal quantities
      for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        gVelocity[n]     /= gMass[n];
        gTemperature[n]  /= gMass[n];
      }
      if(flags->d_doScalarDiffusion){
        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done(); iter++){
          IntVector n = *iter;
          gConcentration[n] /= gMass[n];
          gHydroStress[n]   /= gMass[n];
          gConcentrationNoBC[n] = gConcentration[n];
        }
      }
      
      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      string interp_type = flags->d_interpolator_type;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gVelocity,   interp_type);
      bc.setBoundaryCondition(patch,dwi,"SD-Type",    gConcentration,
                                                                   interp_type);
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
  printTask(patches, patches->get(0),cout_doing,
                                           "Doing AMRMPM::computeStressTensor");

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
    printTask(patches, patch,cout_doing,"Doing AMRMPM::updateErosionParameter");

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

      old_dw->get(px,      lb->pXLabel,                              pset);
      old_dw->get(pvol,    lb->pVolumeLabel,                         pset);
      old_dw->get(pstress, lb->pStressLabel,                         pset);
      new_dw->get(psize,   lb->pSizeLabel_preReloc,                  pset);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);

      new_dw->get(gvolume, lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);
      gstress.initialize(Matrix3(0));
      internalforce.initialize(Vector(0,0,0));
      
      // load p_q
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
      
/*`==========TESTING==========*/
      NCVariable<double> gSumS;
      new_dw->allocateAndPut(gSumS, gSumSLabel,  dwi, patch); 
      gSumS.initialize(0); 
/*===========TESTING==========`*/ 
      
      //__________________________________
      //  fine Patch     
      gstress.initialize(Matrix3(0));

      Matrix3 stressvol;
      Matrix3 stresspress;
      int n8or27 = flags->d_8or27;
      vector<IntVector> ni(interpolator->size());
      vector<double> S(interpolator->size());
      vector<Vector> d_S(interpolator->size());
    

      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end();  iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S,
                                                            psize[idx],pDeformationMeasure[idx]);

        stresspress = pstress[idx] + Id*(/*p_pressure*/-p_q[idx]);

        for (int k = 0; k < n8or27; k++){
          
          if(patch->containsNode(ni[k])){ 
            Vector div(d_S[k].x()*oodx[0],
                       d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
                       
            internalforce[ni[k]] -= (div * stresspress)  * pvol[idx];
            
            // cout << " CIF: ni: " << ni[k] << " div " << div << "\t internalForce " << internalforce[ni[k]] << endl;
            // cout << " div " << div[k] << " stressPress: " << stresspress  << endl;
            
            if( std::isinf( internalforce[ni[k]].length() ) || 
                std::isnan( internalforce[ni[k]].length() ) ){
                cout << "INF: " << ni[k] << " " << internalforce[ni[k]] 
                     << " div: " << div << " stressPress: " << stresspress 
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
    printTask(finePatches, finePatch,cout_doing,
                                     "Doing AMRMPM::computeInternalForce_CFI");

    ParticleInterpolator* interpolator =flags->d_interpolator->clone(finePatch);

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
                          
      //  expand cl_tmp when a neighor patch exists.
      //  This patch owns the low nodes.  You need particles
      //  from the neighbor patch.
      cl_tmp -= finePatch->neighborsLow() * nLayers;

      // find the coarse patches under the fine patch.  
      // You must add a single layer of padding cells.
      int padding = 1;
      Level::selectType coarsePatches;
      finePatch->getOtherLevelPatches(-1, coarsePatches, padding);
        
      Matrix3 Id;
      Id.Identity();
        
      constNCVariable<Stencil7> zoi_fine;
      new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );
  
      int numMPMMatls = d_sharedState->getNumMPMMatls();
      
      for(int m = 0; m < numMPMMatls; m++){
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int dwi = mpm_matl->getDWIndex();
        
        NCVariable<Vector> internalforce;
        new_dw->getModifiable(internalforce,lb->gInternalForceLabel,  
                                                     dwi, finePatch);

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
          constParticleVariable<double>  p_q_coarse;
          
          // coarseLow and coarseHigh cannot lie outside of the coarse patch
          IntVector cl = Max(cl_tmp, coarsePatch->getCellLowIndex());
          IntVector ch = Min(ch_tmp, coarsePatch->getCellHighIndex());

          pset_coarse = old_dw->getParticleSubset(dwi, cl, ch, coarsePatch,
                                                               lb->pXLabel);

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
          
          // Artificial Viscosity
          if(flags->d_artificial_viscosity){
            old_dw->get(p_q_coarse,    lb->p_qLabel,      pset_coarse);
          }
          else {
            ParticleVariable<double>  p_q_create;
            new_dw->allocateTemporary(p_q_create,  pset_coarse);
            for(ParticleSubset::iterator it = pset_coarse->begin();
                                         it != pset_coarse->end();it++){
              p_q_create[*it]=0.0;
            }
            p_q_coarse = p_q_create; // reference created data
          }
          
          //__________________________________
          //  Iterate over the coarse level particles and 
          // add their contribution to the internal stress on the fine patch
          for (ParticleSubset::iterator iter = pset_coarse->begin(); 
                                        iter != pset_coarse->end();  iter++){
            particleIndex idx = *iter;

            vector<IntVector> ni;
            vector<double> S;
            vector<Vector> div;
            interpolator->findCellAndWeightsAndShapeDerivatives_CFI(
                                         px_coarse[idx], ni, S, div, zoi_fine );

            Matrix3 stresspress =  pstress_coarse[idx] + Id*(-p_q_coarse[idx]);

            IntVector fineNode;
            for(int k = 0; k < (int)ni.size(); k++) {   
              fineNode = ni[k];

              if( finePatch->containsNode( fineNode ) ){
                gSumS[fineNode] +=S[k];

                Vector Increment ( (div[k] * stresspress)  * pvol_coarse[idx] );
                //Vector Before = internalforce[fineNode];
                //Vector After  = Before - Increment;

                internalforce[fineNode] -=  Increment;


                //  cout << " CIF_CFI: ni: " << ni[k] << " div " << div[k] << "\t internalForce " << internalforce[fineNode] << endl;
                //  cout << "    before " << Before << " After " << After << " Increment " << Increment << endl;
                //  cout << "    div " << div[k] << " stressPress: " << stresspress << " pvol_coarse " << pvol_coarse[idx] << endl;


  /*`==========TESTING==========*/
                if(std::isinf( internalforce[fineNode].length() ) ||  std::isnan( internalforce[fineNode].length() )){
                  cout << "INF: " << fineNode << " " << internalforce[fineNode] 
                       << " div[k]:"<< div[k] << " stressPress: " << stresspress
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
        bc.setBoundaryCondition( finePatch,dwi,"Symmetric",internalforce,
                                                           interp_type); 
              
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
    printTask(patches, patch,cout_doing,
                              "Doing AMRMPM::computeAndIntegrateAcceleration");

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
      constNCVariable<double> gConcentration,gConcNoBC,gExtScalarFlux;

      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->get(internalforce, lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass,         lb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->get(gvelocity,     lb->gVelocityLabel,      dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      NCVariable<double> gConcStar,gConcRate;
      new_dw->allocateAndPut(gvelocity_star, lb->gVelocityStarLabel, dwi,patch);
      new_dw->allocateAndPut(gacceleration,  lb->gAccelerationLabel, dwi,patch);

      if(flags->d_doScalarDiffusion){
        new_dw->get(gConcentration, lb->gConcentrationLabel,      dwi,patch,gnone,0);
        new_dw->get(gConcNoBC,      lb->gConcentrationNoBCLabel,  dwi,patch,gnone,0);
        new_dw->get(gExtScalarFlux, lb->gExternalScalarFluxLabel, dwi,patch,gnone,0);

        new_dw->getModifiable( gConcRate,lb->gConcentrationRateLabel,  dwi,patch);
        new_dw->allocateAndPut(gConcStar,lb->gConcentrationStarLabel,  dwi,patch);
      }

      gacceleration.initialize(Vector(0.,0.,0.));
      double damp_coef = flags->d_artificialDampCoeff;
      gvelocity_star.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter=patch->getExtraNodeIterator(); !iter.done();iter++){
        IntVector n = *iter;
        
        Vector acc(0,0,0);
        if (gmass[n] > flags->d_min_mass_for_acceleration){
          acc = (internalforce[n] + externalforce[n])/gmass[n];
          acc -= damp_coef * gvelocity[n];
        }
        gacceleration[n]  = acc +  gravity;
        gvelocity_star[n] = gvelocity[n] + gacceleration[n] * delT;
          
/*`==========TESTING==========*/
#ifdef DEBUG_ACC
        if( abs(gacceleration[n].length() - d_acc_ans.length()) > d_acc_tol ) {
          Vector diff = gacceleration[n] - d_acc_ans;
          cout << "    L-"<< getLevel(patches)->getIndex() << " node: "<< n << " gacceleration: " << gacceleration[n] 
               <<  " externalForce: " << externalforce[n]
               << " internalforce: "  << internalforce[n] 
               << " diff: " << diff
               << " gmass: " << gmass[n] 
               << " gravity: " << gravity << endl;
        }
#endif 
/*===========TESTING==========`*/
      }
      if(flags->d_doScalarDiffusion){
        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
          IntVector c = *iter;
//          gConcRate[c] /= gmass[c];
          gConcStar[c]  =  gConcentration[c] + gConcRate[c] * delT;
        }

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch, dwi,"SD-Type", gConcStar,
                                                flags->d_interpolator_type);

        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
          IntVector c = *iter;
          gConcRate[c] = (gConcStar[c] - gConcNoBC[c]) / delT 
                       + gExtScalarFlux[c];
        }
      } // if doScalarDiffusion
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
      new_dw->get(gvelocity,               lb->gVelocityLabel,      dwi, patch,
                                                                 Ghost::None,0);
          
          
      //__________________________________
      // Apply grid boundary conditions to velocity_star and acceleration
      if( patch->hasBoundaryFaces() ){
        IntVector node(0,4,4);

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch,dwi,"Velocity", gvelocity_star,interp_type);
        bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star,interp_type);

        // Now recompute acceleration as the difference between the velocity
        // interpolated to the grid (no bcs applied) and the new velocity_star
        for(NodeIterator iter = patch->getExtraNodeIterator();
                        !iter.done(); iter++){
          IntVector c = *iter;
          gacceleration[c] = (gvelocity_star[c] - gvelocity[c])/delT;
        }
      } 
      
      //__________________________________
      //
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

    } // matl loop
  }  // patch loop
}
//______________________________________________________________________

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
                                          n_iter, isRight_CP_FP_pair);

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
            IntVector dir = patch->getFaceAxes(patchFace); // face axes
            int p_dir = dir[0];                            // normal direction
            
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
      finePatch->getOtherLevelPatchesNB(-1,coarsePatches,0);
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
            IntVector dir = finePatch->getFaceAxes(patchFace); // face axes
            int p_dir = dir[0];                                // normal dir 

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
              warn << "\n ERROR: computeZoneOfInfluence:Fine Level: Did not find node iterator! "
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
void AMRMPM::computeLAndF(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing AMRMPM::computeLAndF");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<double> pvolume;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Matrix3> pFNew,pVelGrad;
      ParticleVariable<Vector> pConcGradNew;

      // Get the arrays of grid data on which the new particle values depend
      constNCVariable<Vector> gvelocity_star;
      constNCVariable<double> gConcStar;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(psize,        lb->pSizeLabel_preReloc,             pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);

      new_dw->allocateAndPut(pvolume,     lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(pVelGrad,    lb->pVelGradLabel_preReloc,     pset);
      new_dw->allocateAndPut(pFNew,       lb->pDeformationMeasureLabel_preReloc,
                                                                          pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);

      if(flags->d_doScalarDiffusion){
        new_dw->get(gConcStar,              lb->gConcentrationStarLabel, dwi,
                                                               patch, gac, NGP);
        new_dw->allocateAndPut(pConcGradNew,lb->pConcGradientLabel_preReloc,
                                                                          pset);
      }

      // Compute velocity gradient and deformation gradient on every particle
      // This can/should be combined into the loop above, once it is working
      double rho_init=mpm_matl->getInitialDensity();
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
        if(flags->d_doScalarDiffusion){
          pConcGradNew[idx] = Vector(0.0, 0.0, 0.0);
          for(int k = 0; k < flags->d_8or27; k++) {
            IntVector node = ni[k];
            for(int j = 0; j < 3; j++){
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

        double J=pFNew[idx].Determinant();
        pvolume[idx]=(pmass[idx]/rho_init)*J;
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
        }
      } //end of pressureStabilization loop  at the patch level

    }
    delete interpolator;
  }
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
    printTask(patches, patch,cout_doing,
                            "Doing AMRMPM::interpolateToParticlesAndUpdate");

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    //Vector dx = patch->dCell();

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
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
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass,pTemperature;
      ParticleVariable<double> pmassNew,pvolume,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      ParticleVariable<double> pTempPreNew;
      constParticleVariable<Matrix3> pFOld;

      constParticleVariable<double>  pConcentration;
      ParticleVariable<double> pConcentrationNew;
      ParticleVariable<double> pConcPreviousNew;
      constNCVariable<double>  gConcentrationRate;

      // Get the arrays of grid data on which the new particle values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, frictionTempRate;
      double Cp = mpm_matl->getSpecificHeat();
      //double max_conc = 0.0;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      new_dw->get(psize,        lb->pSizeLabel_preReloc,             pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

      Ghost::GhostType  gac = Ghost::AroundCells;
      if(flags->d_doScalarDiffusion){
        old_dw->get(pConcentration,     lb->pConcentrationLabel,     pset);
        new_dw->get(gConcentrationRate, lb->gConcentrationRateLabel, 
                                                   dwi, patch, gac, NGP);
        new_dw->allocateAndPut(pConcentrationNew,
                                        lb->pConcentrationLabel_preReloc, pset);
        new_dw->allocateAndPut(pConcPreviousNew,
                                        lb->pConcPreviousLabel_preReloc,  pset);
        //ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        //max_conc = sdm->getMaxConcentration();
      }

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      //Carry forward ParticleID and pSize
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);

      ParticleVariable<int> isLocalized;
      new_dw->allocateAndPut(isLocalized, lb->pLocalizedMPMLabel_preReloc,pset);
      ParticleSubset::iterator iter = pset->begin();
      for (; iter != pset->end(); iter++){
        isLocalized[*iter] = 0;
      }

      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate,lb->frictionalWorkLabel,  dwi,patch,gac,NGP);

      if(flags->d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        dTdt_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
      }

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
        double concRate = 0.0;

        // Accumulate the contribution from vertices on this level
        for(int k = 0; k < flags->d_8or27; k++) {
          IntVector node = ni[k];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];

          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       fricTempRate)   * S[k];
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT*move_particles;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + acc*delT;

        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx]    + pdispnew[idx];
        pTempNew[idx]        = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx]     = pTemperature[idx]; // for thermal stress
        pmassNew[idx]        = pmass[idx];

        if(flags->d_doScalarDiffusion){
          for(int k = 0; k < flags->d_8or27; k++) {
            IntVector node = ni[k];
            concRate += gConcentrationRate[node]   * S[k];
          }

          pConcentrationNew[idx]= pConcentration[idx] + concRate*delT;
          //pConcentrationNew[idx]= min(pConcentrationNew[idx],max_conc);
          pConcPreviousNew[idx] = pConcentration[idx];
        }
/*`==========TESTING==========*/
#ifdef DEBUG_VEL
        Vector diff = ( pvelocitynew[idx] - d_vel_ans );
       if( abs(diff.length() ) > d_vel_tol ) {
         cout << "    L-"<< getLevel(patches)->getIndex() << " px: "<< pxnew[idx] << " pvelocitynew: " << pvelocitynew[idx] <<  " pvelocity " << pvelocity[idx]
                         << " diff " << diff << endl;
       }
#endif
#ifdef DEBUG_ACC
#endif
/*===========TESTING==========`*/

        totalmass  += pmass[idx];
        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX         = CMX + (pxnew[idx]*pmass[idx]).asVector();
        totalMom   += pvelocitynew[idx]*pmass[idx];
      }

#if 0 // Until Todd is ready for this, leave inactive
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
//        cout << "Material = " << m << " Deleted Particle = " << pids_new[idx] 
//             << " xold = " << px[idx] << " xnew = " << pxnew[idx]
//             << " vold = " << pvelocity[idx] << " vnew = "<< pvelocitynew[idx]
//             << " massold = " << pmass[idx] << " massnew = " << pmassNew[idx]
//             << " tempold = " << pTemperature[idx] 
//             << " tempnew = " << pTempNew[idx]
//             << " pLocalized = " << pLocalized[idx]
//             << " volnew = " << pvolume[idx] << endl;
        }
        if(pvelocitynew[idx].length() > flags->d_max_vel){
          if(pvelocitynew[idx].length() >= pvelocity[idx].length()){
	     pvelocitynew[idx]=(pvelocitynew[idx]/pvelocitynew[idx].length())*(flags->d_max_vel*.9);	  
	     cout<<endl<<"Warning: particle "<<pids[idx]<<" hit speed ceiling #1. Modifying particle velocity accordingly."<<endl;
            //pvelocitynew[idx]=pvelocity[idx];
          }
        }
      }
#endif

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
      if(flags->d_refineParticles){
        constParticleVariable<int> pRefinedOld;
        ParticleVariable<int> pRefinedNew;
        old_dw->get(pRefinedOld,            lb->pRefinedLabel,          pset);
        new_dw->allocateAndPut(pRefinedNew, lb->pRefinedLabel_preReloc, pset);
        pRefinedNew.copyData(pRefinedOld);
      }
    }
    delete interpolator;
  }
}

void AMRMPM::finalParticleUpdate(const ProcessorGroup*,
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
      constParticleVariable<double> pdTdt,pmassNew;
      ParticleVariable<double> pTempNew;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      new_dw->get(pdTdt,        lb->pdTdtLabel,                      pset);
      new_dw->get(pmassNew,     lb->pMassLabel_preReloc,             pset);

      new_dw->getModifiable(pTempNew, lb->pTemperatureLabel_preReloc,pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pTempNew[idx] += pdTdt[idx]*delT;

        // Delete particles whose mass is too small (due to combustion),
        // whose pLocalized flag has been set to -999 or who have a negative temperature
        if ((pmassNew[idx] <= flags->d_min_part_mass) || pTempNew[idx] < 0.){
          delset->addParticle(idx);
        }

      } // particles
      new_dw->deleteParticles(delset);
    } // materials
  } // patches
}

void AMRMPM::addParticles(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing addParticles");
    int numMPMMatls=d_sharedState->getNumMPMMatls();

    const Level* level = getLevel(patches);
    int levelIndex = level->getIndex();
    bool hasCoarser=false;
    if(level->hasCoarserLevel()){
      hasCoarser=true;
    }

    //Carry forward CellNAPID
    constCCVariable<int> NAPID;
    CCVariable<int> NAPID_new;
    Ghost::GhostType  gnone = Ghost::None;
    old_dw->get(NAPID,               lb->pCellNAPIDLabel,    0,patch,gnone,0);
    new_dw->allocateAndPut(NAPID_new,lb->pCellNAPIDLabel,    0,patch);
    NAPID_new.copyData(NAPID);

    // Mark cells where particles are refined for grid refinement
    CCVariable<double> refineCell;
    new_dw->getModifiable(refineCell, lb->MPMRefineCellLabel, 0, patch);

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      ParticleVariable<Point> px;
      ParticleVariable<Matrix3> pF,pSize,pstress,pvelgrad,pscalefac;
      ParticleVariable<long64> pids;
      ParticleVariable<double> pvolume,pmass,ptemp,ptempP,pcolor;
      ParticleVariable<Vector> pvelocity,pextforce,pdisp;
      ParticleVariable<int> pref,ploc,plal,prefOld;
      new_dw->getModifiable(px,       lb->pXLabel_preReloc,            pset);
      new_dw->getModifiable(pids,     lb->pParticleIDLabel_preReloc,   pset);
      new_dw->getModifiable(pmass,    lb->pMassLabel_preReloc,         pset);
      new_dw->getModifiable(pSize,    lb->pSizeLabel_preReloc,         pset);
      new_dw->getModifiable(pdisp,    lb->pDispLabel_preReloc,         pset);
      new_dw->getModifiable(pstress,  lb->pStressLabel_preReloc,       pset);
      new_dw->getModifiable(pcolor,   lb->pColorLabel_preReloc,        pset);
      new_dw->getModifiable(pvolume,  lb->pVolumeLabel_preReloc,       pset);
      new_dw->getModifiable(pvelocity,lb->pVelocityLabel_preReloc,     pset);
      new_dw->getModifiable(pscalefac,lb->pScaleFactorLabel_preReloc,  pset);
      new_dw->getModifiable(pextforce,lb->pExtForceLabel_preReloc,     pset);
      new_dw->getModifiable(ptemp,    lb->pTemperatureLabel_preReloc,  pset);
      new_dw->getModifiable(ptempP,   lb->pTempPreviousLabel_preReloc, pset);
      new_dw->getModifiable(pref,     lb->pRefinedLabel_preReloc,      pset);
      new_dw->getModifiable(plal,     lb->pLastLevelLabel_preReloc,    pset);
      new_dw->getModifiable(ploc,     lb->pLocalizedMPMLabel_preReloc, pset);
      new_dw->getModifiable(pvelgrad, lb->pVelGradLabel_preReloc,      pset);
      new_dw->getModifiable(pF,  lb->pDeformationMeasureLabel_preReloc,pset);

      new_dw->allocateTemporary(prefOld,  pset);

      int numNewPartNeeded=0;
      // Put refinement criteria here
      const unsigned int origNParticles = pset->addParticles(0);
      for( unsigned int pp=0; pp<origNParticles; ++pp ){
        prefOld[pp] = pref[pp];
        // Conditions to refine particle based on physical state
        // TODO:  Check below, should be < or <= in first conditional
        if(pref[pp]<levelIndex && pstress[pp].Norm() > 1){
          pref[pp]++;
          numNewPartNeeded++;
        }
        if(pref[pp]>prefOld[pp] || pstress[pp].Norm() > 1){
          IntVector c = level->getCellIndex(px[pp]);
          if(patch->containsCell(c)){
            refineCell[c] = 3.0;  // Why did I use 3 here?  JG
          }
        }else{
          if(hasCoarser){  /* see comment below */
            IntVector c = level->getCellIndex(px[pp]);
            if(patch->containsCell(c)){
              refineCell[c] = -100.;
            }
          }
        }
        // Refine particle if it is too big relative to the cell size
        // of the level it is on.  Don't refine the grid.
        if(pref[pp]< levelIndex){
          pref[pp]++;
          numNewPartNeeded++;
        }
      }
      numNewPartNeeded*=8;

      /*  This tomfoolery is in place to keep refined regions that contain
          particles refined.  If a patch with particles coarsens, the particles
          on that patch disappear when the fine patch is deleted.  This
          prevents the deletion of those patches.  Ideally, we'd allow
          coarsening and relocate the orphan particles, but I don't know how to
          do that.  JG */
      bool keep_patch_refined=false;
      IntVector low = patch->getCellLowIndex();
      IntVector high= patch->getCellHighIndex();
      IntVector middle = (low+high)/IntVector(2,2,2);

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        if(refineCell[c]<0.0){
           keep_patch_refined=true;
           refineCell[c]=0.0;
        }
      }
      if(keep_patch_refined==true){
          refineCell[middle]=-100.0;
      }
      /*  End tomfoolery */

      const unsigned int oldNumPar = pset->addParticles(numNewPartNeeded);

      ParticleVariable<Point> pxtmp;
      ParticleVariable<Matrix3> pFtmp,psizetmp,pstrstmp,pvgradtmp,pSFtmp;
      ParticleVariable<long64> pidstmp;
      ParticleVariable<double> pvoltmp, pmasstmp,ptemptmp,ptempPtmp,pcolortmp;
      ParticleVariable<Vector> pveltmp,pextFtmp,pdisptmp;
      ParticleVariable<int> preftmp,ploctmp,plaltmp;
      new_dw->allocateTemporary(pidstmp,  pset);
      new_dw->allocateTemporary(pxtmp,    pset);
      new_dw->allocateTemporary(pvoltmp,  pset);
      new_dw->allocateTemporary(pveltmp,  pset);
      new_dw->allocateTemporary(pSFtmp,   pset);
      new_dw->allocateTemporary(pextFtmp, pset);
      new_dw->allocateTemporary(ptemptmp, pset);
      new_dw->allocateTemporary(ptempPtmp,pset);
      new_dw->allocateTemporary(pFtmp,    pset);
      new_dw->allocateTemporary(psizetmp, pset);
      new_dw->allocateTemporary(pdisptmp, pset);
      new_dw->allocateTemporary(pstrstmp, pset);
      new_dw->allocateTemporary(pcolortmp,pset);
      new_dw->allocateTemporary(pmasstmp, pset);
      new_dw->allocateTemporary(preftmp,  pset);
      new_dw->allocateTemporary(plaltmp,  pset);
      new_dw->allocateTemporary(ploctmp,  pset);
      new_dw->allocateTemporary(pvgradtmp,pset);
      // copy data from old variables for particle IDs and the position vector
      for( unsigned int pp=0; pp<oldNumPar; ++pp ){
        pidstmp[pp]  = pids[pp];
        pxtmp[pp]    = px[pp];
        pvoltmp[pp]  = pvolume[pp];
        pveltmp[pp]  = pvelocity[pp];
        pSFtmp[pp]   = pscalefac[pp];
        pextFtmp[pp] = pextforce[pp];
        ptemptmp[pp] = ptemp[pp];
        ptempPtmp[pp]= ptempP[pp];
        pFtmp[pp]    = pF[pp];
        psizetmp[pp] = pSize[pp];
        pdisptmp[pp] = pdisp[pp];
        pstrstmp[pp] = pstress[pp];
        pcolortmp[pp]= pcolor[pp];
        pmasstmp[pp] = pmass[pp];
        preftmp[pp]  = pref[pp];
        plaltmp[pp]  = plal[pp];
        ploctmp[pp]  = ploc[pp];
        pvgradtmp[pp]= pvelgrad[pp];

//        cout << "pids_orig = " << pidstmp[pp] << endl;
      }

      Vector dx = patch->dCell();
      int numRefPar=0;
      for( unsigned int idx=0; idx<oldNumPar; ++idx ){
       if(pref[idx]!=prefOld[idx]){
        IntVector c_orig;
        patch->findCell(px[idx],c_orig);
        vector<Point> new_part_pos;

        Matrix3 dsize = (pF[idx]*pSize[idx]*Matrix3(dx[0],0,0,
                                                    0,dx[1],0,
                                                    0,0,dx[2]));

        // Find vectors to new particle locations, based on particle size and
        // deformation (patterned after CPDI interpolator code)
        Vector r[4];
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

        cout << "OPP = " << px[idx] << endl;
        for(int i = 0;i<8;i++){
//        cout << "NPP = " << new_part_pos[i] << endl;
          if(!level->containsPoint(new_part_pos[i])){
            Point anchor = level->getAnchor();
            Point orig = new_part_pos[i];
            new_part_pos[i]=Point(max(orig.x(),anchor.x()),
                                  max(orig.y(),anchor.y()),
                                  max(orig.z(),anchor.z()));
          }

          long64 cellID = ((long64)c_orig.x() << 16) |
                          ((long64)c_orig.y() << 32) |
                          ((long64)c_orig.z() << 48);

          int& myCellNAPID = NAPID_new[c_orig];
          int new_index;
          if(i==0){
             new_index=idx;
          } else {
             new_index=oldNumPar+7*numRefPar+i;
          }
//          cout << "new_index = " << new_index << endl;
          pidstmp[new_index]    = (cellID | (long64) myCellNAPID);
          pxtmp[new_index]      = new_part_pos[i];
          pvoltmp[new_index]    = .125*pvolume[idx];
          pmasstmp[new_index]   = .125*pmass[idx];
          pveltmp[new_index]    = pvelocity[idx];
          pSFtmp[new_index]     = 0.5*pscalefac[idx];
          pextFtmp[new_index]   = pextforce[idx];
          pFtmp[new_index]      = pF[idx];
          psizetmp[new_index]   = 0.5*pSize[idx];
          pdisptmp[new_index]   = pdisp[idx];
          pstrstmp[new_index]   = pstress[idx];
          pcolortmp[new_index]  = pcolor[idx];
          ptemptmp[new_index]   = ptemp[idx];
          ptempPtmp[new_index]  = ptempP[idx];
          preftmp[new_index]    = pref[idx];
          plaltmp[new_index]    = plal[idx];
          ploctmp[new_index]    = ploc[idx];
          pvgradtmp[new_index]  = pvelgrad[idx];
          NAPID_new[c_orig]++;
        }
        numRefPar++;
       }  // if particle flagged for refinement
      } // for particles

      // put back temporary data
      new_dw->put(pidstmp,  lb->pParticleIDLabel_preReloc,           true);
      new_dw->put(pxtmp,    lb->pXLabel_preReloc,                    true);
      new_dw->put(pvoltmp,  lb->pVolumeLabel_preReloc,               true);
      new_dw->put(pveltmp,  lb->pVelocityLabel_preReloc,             true);
      new_dw->put(pSFtmp,   lb->pScaleFactorLabel_preReloc,          true);
      new_dw->put(pextFtmp, lb->pExtForceLabel_preReloc,             true);
      new_dw->put(pmasstmp, lb->pMassLabel_preReloc,                 true);
      new_dw->put(ptemptmp, lb->pTemperatureLabel_preReloc,          true);
      new_dw->put(ptempPtmp,lb->pTempPreviousLabel_preReloc,         true);
      new_dw->put(psizetmp, lb->pSizeLabel_preReloc,                 true);
      new_dw->put(pdisptmp, lb->pDispLabel_preReloc,                 true);
      new_dw->put(pstrstmp, lb->pStressLabel_preReloc,               true);
      new_dw->put(pcolortmp,lb->pColorLabel_preReloc,                true);
      new_dw->put(pFtmp,    lb->pDeformationMeasureLabel_preReloc,   true);
      new_dw->put(preftmp,  lb->pRefinedLabel_preReloc,              true);
      new_dw->put(plaltmp,  lb->pLastLevelLabel_preReloc,            true);
      new_dw->put(ploctmp,  lb->pLocalizedMPMLabel_preReloc,         true);
      new_dw->put(pvgradtmp,lb->pVelGradLabel_preReloc,              true);
      // put back temporary data
    }  // for matls
  }    // for patches
}

void AMRMPM::reduceFlagsExtents(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{

  // Currently doing for levels > 0
  const Level* level = getLevel(patches);
  int levelIndex = level->getIndex();
  IntVector RR_thisLevel = level->getRefinementRatio();
  int numLevels = level->getGrid()->numLevels();

  IntVector RR_RelToFinest = IntVector(1,1,1);
  if(level->hasFinerLevel()){
    RR_RelToFinest = RR_thisLevel*(numLevels-levelIndex-1);
  }

//  cout << "rFE levelIndex = " << levelIndex << endl;
//  cout << "RR_RelToFinest = " << RR_RelToFinest << endl;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing reduceFlagsExtents");

    // Mark cells where particles are refined for grid refinement
    Ghost::GhostType  gnone = Ghost::None;
    constCCVariable<double> refineCell;
    new_dw->get(refineCell, lb->MPMRefineCellLabel, 0, patch, gnone, 0);

    int xmax,xmin,ymax,ymin,zmax,zmin;
    xmax = -999;   ymax = -999;   zmax = -999;
    xmin = 999999; ymin = 999999; zmin = 999999;
//    int print = 0;
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      if(refineCell[c]>0){
        xmax=max(xmax,c.x()); ymax=max(ymax,c.y()); zmax=max(zmax,c.z());
        xmin=min(xmin,c.x()); ymin=min(ymin,c.y()); zmin=min(zmin,c.z());
//        print = 1;
      }
    }

    xmax = xmax*RR_RelToFinest.x();
    ymax = ymax*RR_RelToFinest.y();
    zmax = zmax*RR_RelToFinest.z();
    xmin = xmin*RR_RelToFinest.x();
    ymin = ymin*RR_RelToFinest.y();
    zmin = zmin*RR_RelToFinest.z();

/*
    if (print==1){
      cout << "Xmax = " << xmax << endl;
      cout << "Ymax = " << ymax << endl;
      cout << "Zmax = " << zmax << endl;
      cout << "Xmin = " << xmin << endl;
      cout << "Ymin = " << ymin << endl;
      cout << "Zmin = " << zmin << endl;
    }
*/

    new_dw->put(max_vartype(xmax), RefineFlagXMaxLabel);
    new_dw->put(max_vartype(ymax), RefineFlagYMaxLabel);
    new_dw->put(max_vartype(zmax), RefineFlagZMaxLabel);
    new_dw->put(min_vartype(xmin), RefineFlagXMinLabel);
    new_dw->put(min_vartype(ymin), RefineFlagYMinLabel);
    new_dw->put(min_vartype(zmin), RefineFlagZMinLabel);
  }    // for patches
}

void AMRMPM::computeParticleScaleFactor(const ProcessorGroup*,
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

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->get(pF,           lb->pDeformationMeasureLabel_preReloc,    pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(dataArchiver->isOutputTimestep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = (pF[idx]*psize[idx]*Matrix3(dx[0],0,0,
                                                          0,dx[1],0,
                                                          0,0,dx[2]));
        } // for particles
      } // isOutputTimestep
    } // matls
  } // patches

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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing AMRMPM::errorEstimate");

    Ghost::GhostType  gnone = Ghost::None;
    constCCVariable<double> refineCell;
    CCVariable<int>      refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(),
                                                                  0, patch);
    new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(),
                                                                  0, patch);
    new_dw->get(refineCell,     lb->MPMRefineCellLabel, 0, patch, gnone, 0);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

    IntVector low = patch->getCellLowIndex();
    IntVector high= patch->getCellHighIndex();
    IntVector middle = (low+high)/IntVector(2,2,2);

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      if(refineCell[c]>0.0 || refineFlag[c]==true){
        refineFlag[c] = 1;
        refinePatch->set();
      }else if(refineCell[c]<0.0 && ((int) refineCell[c])%100!=0){
        refineFlag[c] = 1;
        refinePatch->set();
      }else{
        refineFlag[c] = 0;
      }
    }

#if 0
    // loop over all the geometry objects
    for(int obj=0; obj<(int)d_refine_geom_objs.size(); obj++){
      GeometryPieceP piece = d_refine_geom_objs[obj]->getPiece();
      Vector dx = patch->dCell();
      
      int geom_level =  d_refine_geom_objs[obj]->getInitialData_int("level");
     
      // don't add refinement flags if the current level is greater than
      // the geometry level specification
      if(geom_level!=-1 && level->getIndex()>=geom_level)
        continue;

      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        Point  lower  = patch->nodePosition(c);
        Vector upperV = lower.asVector() + dx; 
        Point  upper  = upperV.asPoint();
        
        if(piece->inside(upper) && piece->inside(lower))
          refineFlag[c] = true;
          refinePatch->set();
      }
    }
#endif

#if 0
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      // Loop over particles
      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      
      constParticleVariable<Point> px;
      constParticleVariable<int> prefined;
      new_dw->get(px,       lb->pXLabel,       pset);
      new_dw->get(prefined, lb->pRefinedLabel, pset);
#if 0
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        
        if(level->getIndex()==0 &&(c==IntVector(26,1,0) && step > 48)){
          refineFlag[c] = true;
          refinePatch->set();
        } else{
          refineFlag[c] = false;
        }
      }
#endif

#if 1

      for(ParticleSubset::iterator iter = pset->begin();
                                   iter!= pset->end();  iter++){
        if(prefined[*iter]==1){
          IntVector c = level->getCellIndex(px[*iter]);
          refineFlag[c] = true;
          cout << "refineFlag Cell = " << c << endl;
          refinePatch->set();
        }
      }
#endif
    }
#endif
  }

}
//______________________________________________________________________
//
void AMRMPM::refineGrid(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* /*matls*/,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  // just create a particle subset if one doesn't exist
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing AMRMPM::refineGrid");

    CCVariable<int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

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
        ParticleVariable<double> pTempPrev,pColor;
        ParticleVariable<int>    pLoadCurve,pLastLevel,pLocalized,pRefined;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress, pVelGrad;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        new_dw->allocateAndPut(pLastLevel,     lb->pLastLevelLabel,     pset);
        new_dw->allocateAndPut(pLocalized,     lb->pLocalizedMPMLabel,  pset);
        new_dw->allocateAndPut(pRefined,       lb->pRefinedLabel,       pset);
        new_dw->allocateAndPut(pVelGrad,       lb->pVelGradLabel,       pset);
        if (flags->d_useLoadCurves){
          new_dw->allocateAndPut(pLoadCurve,   lb->pLoadCurveIDLabel,   pset);
        }
        if (flags->d_with_color) {
          new_dw->allocateAndPut(pColor,       lb->pColorLabel,         pset);
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
  
//  const Level* level = getLevel(patches);
//  cout << "Level " << level->getIndex() << " has " << level->numPatches() << " patches" << endl;
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    
    printTask(patches,patch,cout_doing,"Doing AMRMPM::countParticles");
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      totalParticles += pset->end() - pset->begin();
    }
//    cout << "patch = " << patch->getID()
//         << ", numParticles = " << totalParticles << endl;
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
  t->requires(Task::NewDW, lb->pSizeLabel,               gn,0);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, gn,0);
  
  if(level->hasFinerLevel()){ 
    #define allPatches 0
    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    t->requires(Task::NewDW, lb->gZOILabel, allPatches, Task::FineLevel,d_one_matl, ND, gn, 0);
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
    new_dw->get(psize,                lb->pSizeLabel,               pset);
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

#if 0
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
    t->requires(Task::NewDW, lb->gVelocityStarLabel, allPatches, Task::FineLevel,allMatls,   ND, gn,0);
    t->requires(Task::NewDW, lb->gAccelerationLabel, allPatches, Task::FineLevel,allMatls,   ND, gn,0);
    t->requires(Task::NewDW, lb->gZOILabel,          allPatches, Task::FineLevel,d_one_matl, ND, gn,0);
    
    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pDispLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);

    sched->addTask(t, patches, matls);
  }
}
#endif

#if 0
//______________________________________________________________________
//
void AMRMPM::interpolateToParticlesAndUpdate_CFI(const ProcessorGroup*,
                                                 const PatchSubset* coarsePatches,
                                                 const MaterialSubset* ,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
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

          // use getRegion() instead of get().  They should be equivalent but 
          // get() throws assert on parallel runs.
          IntVector fl = finePatch->getNodeLowIndex();
          IntVector fh = finePatch->getNodeHighIndex();
          new_dw->getRegion(gvelocity_star_fine,  lb->gVelocityStarLabel, dwi, fineLevel,fl, fh);   
          new_dw->getRegion(gacceleration_fine,   lb->gAccelerationLabel, dwi, fineLevel,fl, fh); 
            
          
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


          for (ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
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
#endif

void AMRMPM::scheduleComputeFlux(SchedulerP& sched, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;

    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeFlux");

    Task* t = scinew Task("AMRMPM::computeFlux",
                     this,&AMRMPM::computeFlux);

    int numMPM = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->scheduleComputeFlux(t, mpm_matl, patches);
    }

    //t->computes(d_sharedState->get_delt_label(),getLevel(patches));

    sched->addTask(t, patches, matls);
}

void AMRMPM::computeFlux(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing AMRMPM::computeFlux");

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->computeFlux(patch, mpm_matl, old_dw, new_dw);
    }
  }
}

void AMRMPM::scheduleComputeDivergence(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeDivergence");

  Task* t = scinew Task("AMRMPM::computeDivergence",
                    this,&AMRMPM::computeDivergence);

#if 1
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->scheduleComputeDivergence(t, mpm_matl, patches);
  }
#endif

  sched->addTask(t, patches, matls);
}

void AMRMPM::computeDivergence(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,
             "Doing AMRMPM::computeDivergence");

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->computeDivergence(patch, mpm_matl, old_dw, new_dw);
    }
  }
}

void AMRMPM::scheduleComputeDivergence_CFI(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();

  if(L_indx > 0 ){
    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeDivergence_CFI");

    Task* t = scinew Task("AMRMPM::computeDivergence_CFI",
                     this,&AMRMPM::computeDivergence_CFI);

#if 1
    int numMPM = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->scheduleComputeDivergence_CFI(t, mpm_matl, patches);
    }
#endif

    sched->addTask(t, patches, matls);
  }
}

void AMRMPM::computeDivergence_CFI(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->computeDivergence_CFI(patches, mpm_matl, old_dw, new_dw);
  }
}

void AMRMPM::scheduleInitializeScalarFluxBCs(const LevelP& level,
                                             SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  
  d_loadCurveIndex = scinew MaterialSubset();
  d_loadCurveIndex->add(0);
  d_loadCurveIndex->addReference();
  
  int nofSFBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "ScalarFlux"){
      d_loadCurveIndex->add(nofSFBCs++);
    }
  }
  if (nofSFBCs > 0) {
   printSchedule(patches,cout_doing,"MPM::countMaterialPointsPerFluxLoadCurve");
   printSchedule(patches,cout_doing,"MPM::scheduleInitializeScalarFluxBCs");
    // Create a task that calculates the total number of particles
    // associated with each load curve.  
    Task* t = scinew Task("MPM::countMaterialPointsPerFluxLoadCurve",
                          this, &AMRMPM::countMaterialPointsPerFluxLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, d_loadCurveIndex,
                Task::OutOfDomain);
    sched->addTask(t, patches, d_sharedState->allMPMMaterials());

#if 1
    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("MPM::initializeScalarFluxBC",
                    this, &AMRMPM::initializeScalarFluxBC);
//    t->requires(Task::NewDW, lb->pXLabel,                        Ghost::None);
//    t->requires(Task::NewDW, lb->pSizeLabel,                     Ghost::None);
//    t->requires(Task::NewDW, lb->pDeformationMeasureLabel,       Ghost::None);
//    t->requires(Task::NewDW, lb->pLoadCurveIDLabel,              Ghost::None);
//    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel,
//                            d_loadCurveIndex, Task::OutOfDomain, Ghost::None);
//    t->modifies(lb->pExternalScalarFluxLabel);
//    if (flags->d_useCBDI) {
//       t->computes(             lb->pExternalForceCorner1Label);
//       t->computes(             lb->pExternalForceCorner2Label);
//       t->computes(             lb->pExternalForceCorner3Label);
//       t->computes(             lb->pExternalForceCorner4Label);
//    }
    sched->addTask(t, patches, d_sharedState->allMPMMaterials());
#endif
  }
  
  if(d_loadCurveIndex->removeReference())
    delete d_loadCurveIndex;
}

// Calculate the number of material points per load curve
void AMRMPM::countMaterialPointsPerFluxLoadCurve(const ProcessorGroup*,
                                                 const PatchSubset* patches,
                                                 const MaterialSubset*,
                                                 DataWarehouse* ,
                                                 DataWarehouse* new_dw)
{
  printTask(patches, patches->get(0), cout_doing,
                     "countMaterialPointsPerLoadCurve");
  // Find the number of pressure BCs in the problem
  int nofSFBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "ScalarFlux") {
      nofSFBCs++;

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
            if (pLoadCurveID[idx] == (nofSFBCs)) ++numPts;
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts),
                    lb->materialPointsPerLoadCurveLabel, 0, nofSFBCs-1);
      }  // patch loop
    }
  }
}

void AMRMPM::initializeScalarFluxBC(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* ,
                                    DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
  printTask(patches,patches->get(0),cout_doing,"Doing initialize ScalarFluxBC");
  if (cout_doing.active())
    cout_doing << "Current Time (Initialize ScalarFlux BC) = " << time << endl;

  // Calculate the scalar flux at each particle
  for(int p=0;p<patches->size();p++){
//    const Patch* patch = patches->get(p);
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
//      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
//      int dwi = mpm_matl->getDWIndex();
//      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
//      ParticleVariable<double> pExtSF;
//      new_dw->getModifiable(pExtSF, lb->pExternalScalarFluxLabel, pset);

#if 0
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      new_dw->get(px, lb->pXLabel, pset);
      new_dw->get(psize, lb->pSizeLabel, pset);
      new_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);
      constParticleVariable<int> pLoadCurveID;
      new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

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
#endif
      int nofSFBCs = 0;
      for(int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
        string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        if (bcs_type == "ScalarFlux") {

          // Get the material points per load curve
          sumlong_vartype numPart = 0;
          new_dw->get(numPart,lb->materialPointsPerLoadCurveLabel,0,nofSFBCs++);

          // Save the material points per load curve in the ScalarFluxBC object
          ScalarFluxBC* pbc =
          dynamic_cast<ScalarFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
          pbc->numMaterialPoints(numPart);

          if (cout_doing.active()){
            cout_doing << "    Load Curve = "
                     << nofSFBCs << " Num Particles = " << numPart << endl; 
          }
          // Calculate the force per particle at t = 0.0
          //double fluxPerPart = pbc->fluxPerParticle(time);

          // Loop through the patches and calculate the force vector
          // at each particle

#if 0
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == nofSFBCs) {
//               pExternalForce[idx] = pbc->getForceVector(px[idx],
//                                                        forcePerPart,time);
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
#endif
        }   // if pressure loop
      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

void AMRMPM::scheduleApplyExternalScalarFlux(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleApplyExternalScalarFlux");

  Task* t=scinew Task("MPM::applyExternalScalarFlux",
                    this, &AMRMPM::applyExternalScalarFlux);

  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel,              Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,Ghost::None);
  t->computes(             lb->pExternalScalarFluxLabel);
  if (flags->d_useLoadCurves) {
    t->requires(Task::OldDW, lb->pLoadCurveIDLabel,     Ghost::None);
  }

  sched->addTask(t, patches, matls);
}

void AMRMPM::applyExternalScalarFlux(const ProcessorGroup* ,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();

  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalScalarFlux) = " << time << endl;

  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> fluxPerPart;
  std::vector<ScalarFluxBC*> pbcP;
  if (flags->d_useLoadCurves) {
    for (int ii = 0;ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "ScalarFlux") {

        ScalarFluxBC* pbc =
          dynamic_cast<ScalarFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // Calculate the force per particle at current time
        fluxPerPart.push_back(pbc->fluxPerParticle(time));
      }
    }
  }

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing applyExternalScalarFlux");

    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.

    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<double> pExternalScalarFlux;

      old_dw->get(px,    lb->pXLabel,    pset);
      old_dw->get(psize, lb->pSizeLabel, pset);
      old_dw->get(pDeformationMeasure, lb->pDeformationMeasureLabel, pset);
      new_dw->allocateAndPut(pExternalScalarFlux,
                                      lb->pExternalScalarFluxLabel,  pset);

      if (flags->d_useLoadCurves) {
        constParticleVariable<int> pLoadCurveID;
        old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
        bool do_FluxBCs=false;
        for (int ii = 0;
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
          string bcs_type =
            MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
          if (bcs_type == "ScalarFlux") {
            do_FluxBCs=true;
          }
        }

        // Get the load curve data
        if(do_FluxBCs){
#if 0
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
#endif
          // Iterate over the particles
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            int loadCurveID = pLoadCurveID[idx]-1;
            if (loadCurveID < 0) {
              pExternalScalarFlux[idx] = 0.0;
            } else {
              //ScalarFluxBC* pbc = pbcP[loadCurveID];

              pExternalScalarFlux[idx] = fluxPerPart[loadCurveID];
#if 0
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
               pExternalForce_new[idx]=pbc->getForceVector(px[idx],force,time);
              }
#endif
            }
          }
        } else {
          for(ParticleSubset::iterator iter = pset->begin();
                                       iter != pset->end(); iter++){
           pExternalScalarFlux[*iter] = 0.;
          }
        }
      } else { // if use load curves
        for(ParticleSubset::iterator iter = pset->begin();
                                     iter != pset->end(); iter++){
         pExternalScalarFlux[*iter] = 0.;
        }
      }
    } // matl loop
  }  // patch loop
}
