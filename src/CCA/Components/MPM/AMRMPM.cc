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
 
// make uintah CXX=/usr/bin/iwyu


#include <CCA/Components/MPM/AMRMPM.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h> // for MPMDiffusionLabel
#include <CCA/Components/MPM/Core/MPMLabel.h>          // for MPMLabel
#include <CCA/Components/MPM/Core/MPMBoundCond.h>      // for MPMBoundCond
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h> // for Contact
#include <CCA/Components/MPM/Materials/Contact/ContactFactory.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/SDInterfaceModel.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/Materials/Diffusion/SDInterfaceModelFactory.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModelFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>
#include <CCA/Components/MPM/SerialMPM.h>                // for SerialMPM
#include <Core/Grid/Variables/PerPatchVars.h>       // for PatchFlagP, etc
#include <CCA/Ports/DataWarehouse.h>                     // for DataWarehouse
#include <CCA/Ports/Output.h>                            // for Output
#include <CCA/Ports/Scheduler.h>                         // for Scheduler
#include <CCA/Ports/Regridder.h>
#include <Core/Disclosure/TypeUtils.h>                   // for long64
#include <Core/Exceptions/InternalError.h>               // for InternalError
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>                     // for IntVector, operator<<, Max, etc
#include <Core/Geometry/Point.h>                         // for Point, operator<<
#include <Core/GeometryPiece/GeometryObject.h>           // for GeometryObject
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>                 // for fineToCoarseOperator
#include <Core/Grid/DbgOutput.h>                         // for printTask, printSchedule
#include <Core/Grid/Ghost.h>                             // for Ghost, etc
#include <Core/Grid/Grid.h>                              // for Grid
#include <Core/Grid/ParticleInterpolator.h>              // for ParticleInterpolator
#include <Core/Grid/MaterialManager.h>                   // for MaterialManager
#include <Core/Grid/Task.h>                              // for Task, Task::WhichDW::OldDW, etc
#include <Core/Grid/Variables/Array3.h>                  // for Array3
#include <Core/Grid/Variables/CCVariable.h>              // for CCVariable, etc
#include <Core/Grid/Variables/CellIterator.h>            // for CellIterator
#include <Core/Grid/Variables/GridVariableBase.h>        // for GridVariableBase
#include <Core/Grid/Variables/NodeIterator.h>            // for NodeIterator
#include <Core/Grid/Variables/ParticleSubset.h>          // for ParticleSubset, etc
#include <Core/Grid/Variables/ParticleVariable.h>        // for ParticleVariable, etc
#include <Core/Grid/Variables/ParticleVariableBase.h>
#include <Core/Grid/Variables/PerPatch.h>                // for PerPatch
#include <Core/Grid/Variables/Stencil7.h>                // for Stencil7
#include <Core/Grid/Variables/VarLabel.h>                // for VarLabel
#include <Core/Grid/Variables/VarTypes.h>                // for delt_vartype, etc
#include <Core/Malloc/Allocator.h>                       // for scinew
#include <Core/Math/Matrix3.h>                           // for Matrix3, swapbytes, etc
#include <Core/Parallel/Parallel.h>                      // for proc0cout
#include <Core/Parallel/ProcessorGroup.h>                // for ProcessorGroup
#include <Core/ProblemSpec/ProblemSpec.h>                // for Vector, IntVector, etc
#include <Core/ProblemSpec/ProblemSpecP.h>               // for ProblemSpecP
#include <Core/Util/DebugStream.h>                       // for DebugStream
#include <Core/Util/Handle.h>                            // for Handle

#include <algorithm>                                     // for max, min
#include <cmath>                                         // for cbrt, isinf, isnan
#include <iostream>                                      // for operator<<, basic_ostream, etc
#include <mpi.h>                                         // for Uintah::MPI::Pack_size
#include <stdlib.h>                                      // for abs
#include <string>                                        // for string, operator==, etc

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("AMRMPM", false);
static DebugStream amr_doing("AMRMPM", false);

//#define USE_DEBUG_TASK
//#define DEBUG_VEL
//#define DEBUG_ACC
#undef CBDI_FLUXBCS
#define USE_FLUX_RESTRICTION

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


AMRMPM::AMRMPM(const ProcessorGroup* myworld,
	       const MaterialManagerP materialManager) :
  SerialMPM(myworld, materialManager)
{
  //lb = scinew MPMLabel();
  //flags = scinew MPMFlags(myworld);
  flags->d_minGridLevel = 0;
  flags->d_maxGridLevel = 1000;

  d_SMALL_NUM_MPM=1e-200;
  contactModel   = 0;
  d_sdInterfaceModel = 0;
  NGP     = -9;
  NGN     = -9;
  d_nPaddingCells_Coarse = -9;
  d_acc_ans = Vector(0,0,0);
  d_acc_tol = 1e-7;
  d_vel_ans = Vector(-100,0,0);
  d_vel_tol = 1e-7;
  
  d_gac = Ghost::AroundCells;  // for readability
  d_gan = Ghost::AroundNodes;
  d_gn  = Ghost::None;
  
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
  
  d_fluxbc = 0;

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();

  d_switchCriteria = nullptr;
  gZOISWBLabel = nullptr;
  gZOINETLabel = nullptr;

}
//______________________________________________________________________
//
AMRMPM::~AMRMPM()
{
//  delete lb;
//  delete flags;
  if(flags->d_doScalarDiffusion){
    delete d_sdInterfaceModel;
  }
  
  delete d_fluxbc;

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
                          GridP& grid)
{
  cout_doing<<"Doing problemSetup\t\t\t\t\t AMRMPM"<<endl;
  
  m_scheduler->setPositionVar(lb->pXLabel);

  bool isRestart = false;
  ProblemSpecP mat_ps = 0;
  if (restart_prob_spec){
    mat_ps = restart_prob_spec;
    isRestart = true;
  } else{
    mat_ps = prob_spec;
  }

  ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");

  // Read all MPM flags (look in MPMFlags.cc)
  flags->readMPMFlags(mat_ps, m_output);
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

  ProblemSpecP refine_ps = mpm_ps->findBlock("Refinement_Criteria_Thresholds");
#if 0
  if(!refine_ps ){
    string warn;
    warn ="\n INPUT FILE ERROR:\n <Refinement_Criteria_Thresholds> "
         " block not found inside of <MPM> block \n";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
  }
#endif

  //__________________________________
  // Pull out the refinement threshold criteria 
  if( refine_ps != nullptr ) {
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
  }

  //__________________________________
  // override CFI_interpolator
  d_CFI_interpolator = flags->d_interpolator_type;
  mpm_ps->get("CFI_interpolator", d_CFI_interpolator );
  
  if(d_CFI_interpolator != flags->d_interpolator_type ){
    proc0cout << "______________________________________________________\n" 
              << "          AMRMPM:  WARNING\n"
              << "  The particle to grid interpolator at the CFI is (" << d_CFI_interpolator
              << "), however the over rest of the domain it is: " << flags->d_interpolator_type
              << "\n______________________________________________________________________" << endl;
  }
  
  // bulletproofing
  int maxLevel = grid->numLevels();
  for(int L = 0; L<maxLevel; L++){
    LevelP level = grid->getLevel(L);
    IntVector ec = level->getExtraCells();
    
    if( ec != IntVector(1,1,1) && flags->d_interpolator_type == "gimp" ){       // This should be generalized
      ostringstream msg;                                                        // Each interpolator should know how many EC needed.
      msg << "\n AMRMPM ERROR:\n The number of extraCells on level ("
          << level->getIndex() << ") is not equal to [1,1,1] required for the GIMP particle interpolator";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);    
    }
  }
  
  
  
  

#if 0  // This allows defining regions to be refined using geometry objects
       // Jim was having a bit of trouble keeping this consistent with other
       // methods of defining finer levels.  Keep for now.
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
  if(!isLockstepAMR()){
    ostringstream msg;
    msg << "\n ERROR: You must add \n"
        << " <useLockStep> true </useLockStep> \n"
        << " inside of the <AMR> section. \n"; 
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }  
    
  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  MPMPhysicalBCFactory::create(mat_ps, grid, flags);
  
  bool needNormals = false;
  bool useLR       = false;
  contactModel = ContactFactory::create(d_myworld,
                                        mat_ps,m_materialManager,lb,
                                        flags, needNormals, useLR);

  flags->d_computeNormals=needNormals;
  flags->d_useLogisticRegression=useLR;

  // Determine extents for coarser level particle data
  // Linear Interpolation:  1 layer of coarse level cells
  // Gimp Interpolation:    2 layers
  
/*`==========TESTING==========*/
  d_nPaddingCells_Coarse = 1;
//  NGP = 1;
/*===========TESTING==========`*/

  setParticleGhostLayer(Ghost::AroundNodes, NGP);

  materialProblemSetup(mat_ps,flags, isRestart);

  if(flags->d_doScalarDiffusion){
    d_sdInterfaceModel = SDInterfaceModelFactory::create(mat_ps, m_materialManager, flags, lb);
  }
  d_fluxbc = FluxBCModelFactory::create(m_materialManager, flags);
}

//______________________________________________________________________
//
void AMRMPM::outputProblemSpec(ProblemSpecP& root_ps)
{
  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP flags_ps = root->appendChild("MPM");
  flags->outputProblemSpec( flags_ps );

  ProblemSpecP mat_ps = nullptr;
  mat_ps = root->findBlockWithOutAttribute("MaterialProperties");

  if( mat_ps == nullptr ) {
    mat_ps = root->appendChild("MaterialProperties");
  }

  ProblemSpecP mpm_ps = mat_ps->appendChild("MPM");
  for (unsigned int i = 0; i < m_materialManager->getNumMatls( "MPM" );i++) {
    MPMMaterial* mat = (MPMMaterial*) m_materialManager->getMaterial( "MPM", i);
    ProblemSpecP cm_ps = mat->outputProblemSpec(mpm_ps);
  }
  contactModel->outputProblemSpec(mpm_ps);
  if (flags->d_doScalarDiffusion){
    d_sdInterfaceModel->outputProblemSpec(mpm_ps);
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
  t->computes(lb->delTLabel,level.get_rep());
  t->computes(lb->pCellNAPIDLabel,d_one_matl);
  t->computes(lb->NC_CCweightLabel,d_one_matl);

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
    t->computes(lb->diffusion->pArea);
    t->computes(lb->diffusion->pConcentration);
    t->computes(lb->diffusion->pConcPrevious);
    t->computes(lb->diffusion->pGradConcentration);
    t->computes(lb->diffusion->pExternalScalarFlux);
  }

  if(flags->d_withGaussSolver){
    t->computes(lb->pPosChargeLabel);
    t->computes(lb->pNegChargeLabel);
    t->computes(lb->pPermittivityLabel);
  }

  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  const PatchSet* patches = level->eachPatch();
  for(unsigned int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addInitialComputesAndRequires(t, mpm_matl);

    ErosionModel* em = mpm_matl->getErosionModel();
    em->addInitialComputesAndRequires(t, mpm_matl);
    
    if(flags->d_doScalarDiffusion){
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->addInitialComputesAndRequires(t, mpm_matl, patches);
    }
  }

  sched->addTask(t, level->eachPatch(), m_materialManager->allMaterials( "MPM" ));

  if (level->getIndex() == 0 ) schedulePrintParticleCount(level, sched);

  if (flags->d_useLoadCurves && !flags->d_doScalarDiffusion) {
    // Schedule the initialization of pressure BCs per particle
    cout << "Pressure load curves are untested for multiple levels" << endl;
    scheduleInitializePressureBCs(level, sched);
  }

  if (flags->d_useLoadCurves && flags->d_doScalarDiffusion) {
    // Schedule the initialization of scalar fluxBCs per particle
    cout << "Scalar load curves are untested for multiple levels" << endl;
    d_fluxbc->scheduleInitializeScalarFluxBCs(level, sched);
    //scheduleInitializeScalarFluxBCs(level, sched);
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

  sched->addTask(t, m_loadBalancer->getPerProcessorPatchSet(level), m_materialManager->allMaterials( "MPM" ));
}
//______________________________________________________________________
//
void AMRMPM::scheduleComputeStableTimeStep(const LevelP&,
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

  const MaterialSet* matls = m_materialManager->allMaterials( "MPM" );
  int maxLevels = level->getGrid()->numLevels();
  GridP grid = level->getGrid();


  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    schedulePartitionOfUnity(               sched, patches, matls);
    scheduleComputeZoneOfInfluence(         sched, patches, matls);
    scheduleComputeCurrentParticleSize(     sched, patches, matls);
    scheduleApplyExternalLoads(             sched, patches, matls);
    d_fluxbc->scheduleApplyExternalScalarFlux( sched, patches, matls);
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
    if(flags->d_computeNormals){
      scheduleComputeNormals(         sched, patches, matls);
    }
    scheduleExMomInterpolated(        sched, patches, matls);
    if(flags->d_doScalarDiffusion){
      scheduleConcInterpolated(sched, patches, matls);
    }
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
      scheduleDiffusionInterfaceDiv(    sched, patches, matls);
    }

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      scheduleComputeDivergence_CFI(    sched, patches, matls);
    }
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

#if 0  // Jim sees no need to do this task, at least not for linear interp.
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

#if 0  // This may need to be reactivated when we enable GIMP
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
  if (level->getIndex() == 0) {
    const MaterialSet* matls = m_materialManager->allMaterials( "MPM" );
    sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
                                      d_particleState_preReloc,
                                      lb->pXLabel, 
                                      d_particleState,
                                      lb->pParticleIDLabel, matls);
  }
}

//______________________________________________________________________
//
void AMRMPM::scheduleAnalysis( const LevelP& level, SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

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

  t->requires(Task::OldDW, lb->pMassLabel,               d_gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,             d_gan,NGP);
  t->requires(Task::OldDW, lb->pVelocityLabel,           d_gan,NGP);
  if (flags->d_GEVelProj) {
    t->requires(Task::OldDW, lb->pVelGradLabel,          d_gan,NGP);
  }
  t->requires(Task::OldDW, lb->pXLabel,                  d_gan,NGP);
  t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  d_gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,        d_gan,NGP);
  t->requires(Task::NewDW, lb->pCurSizeLabel,            d_gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureRateLabel);
  t->computes(lb->gExternalForceLabel);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, lb->pStressLabel,             d_gan, NGP);
    t->requires(Task::OldDW, lb->diffusion->pConcentration,      d_gan, NGP);
    if (flags->d_GEVelProj) {
      t->requires(Task::OldDW, lb->diffusion->pGradConcentration, d_gan, NGP);
    }  
    t->requires(Task::NewDW, lb->diffusion->pExternalScalarFlux_preReloc, d_gan, NGP);
    t->computes(lb->diffusion->gConcentration);
    t->computes(lb->diffusion->gHydrostaticStress);
    t->computes(lb->diffusion->gExternalScalarFlux);
#ifdef CBDI_FLUXBCS
    if (flags->d_useLoadCurves) {
      t->requires(Task::OldDW, lb->pLoadCurveIDLabel,      d_gan, NGP);
    }
#endif
  }
  if(flags->d_withGaussSolver){
    t->requires(Task::OldDW, lb->pPosChargeLabel, d_gan, NGP);
    t->requires(Task::OldDW, lb->pNegChargeLabel, d_gan, NGP);
    t->computes(lb->gPosChargeLabel);
    t->computes(lb->gNegChargeLabel);
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

    Task* t = nullptr;
    if( d_CFI_interpolator == "gimp" ){

      t = scinew Task("AMRMPM::interpolateParticlesToGrid_CFI_GIMP",
                 this,&AMRMPM::interpolateParticlesToGrid_CFI_GIMP);
    }else{
      t = scinew Task("AMRMPM::interpolateParticlesToGrid_CFI",
                 this,&AMRMPM::interpolateParticlesToGrid_CFI);
    }

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
    t->requires(Task::OldDW, lb->pMassLabel,               allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pVolumeLabel,             allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pVelocityLabel,           allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pXLabel,                  allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::NewDW, lb->pExtForceLabel_preReloc,  allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pTemperatureLabel,        allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);

    t->modifies(lb->gMassLabel);
    t->modifies(lb->gVolumeLabel);
    t->modifies(lb->gVelocityLabel);
    t->modifies(lb->gTemperatureLabel);
    t->modifies(lb->gExternalForceLabel);

    if(flags->d_doScalarDiffusion) {
      t->requires(Task::OldDW, lb->diffusion->pConcentration,                 allPatches,
                                      Task::CoarseLevel,  allMatls, ND, d_gac, npc);
      t->requires(Task::OldDW, lb->pStressLabel,                              allPatches,
                                      Task::CoarseLevel,  allMatls, ND, d_gac, npc);
      t->requires(Task::NewDW, lb->diffusion->pExternalScalarFlux_preReloc,   allPatches,
                                      Task::CoarseLevel,  allMatls, ND, d_gac, npc);
 
      t->modifies(lb->diffusion->gConcentration);
      t->modifies(lb->diffusion->gHydrostaticStress);
      t->modifies(lb->diffusion->gExternalScalarFlux);
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
  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0

  t->requires(Task::NewDW, lb->gMassLabel,          allPatches, Task::FineLevel,allMatls, ND, d_gn,0);
  t->requires(Task::NewDW, lb->gVolumeLabel,        allPatches, Task::FineLevel,allMatls, ND, d_gn,0);
  t->requires(Task::NewDW, lb->gVelocityLabel,      allPatches, Task::FineLevel,allMatls, ND, d_gn,0);
  t->requires(Task::NewDW, lb->gTemperatureLabel,   allPatches, Task::FineLevel,allMatls, ND, d_gn,0);
  t->requires(Task::NewDW, lb->gExternalForceLabel, allPatches, Task::FineLevel,allMatls, ND, d_gn,0);
  
  t->modifies(lb->gMassLabel);
  t->modifies(lb->gVolumeLabel);
  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gTemperatureLabel);
  t->modifies(lb->gExternalForceLabel);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->diffusion->gConcentration,      allPatches,
                              Task::FineLevel,allMatls, ND, d_gn, 0);
    t->modifies(lb->diffusion->gConcentration);
    t->requires(Task::NewDW, lb->diffusion->gExternalScalarFlux, allPatches,
                              Task::FineLevel,allMatls, ND, d_gn, 0);
    t->modifies(lb->diffusion->gExternalScalarFlux);
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

  Task::MaterialDomainSpec  ND  = Task::NormalDomain;
  #define allPatches 0
  #define allMatls 0

  t->requires(Task::NewDW, lb->gMassLabel,          allPatches,
                                           Task::FineLevel,allMatls, ND, d_gn,0);
  t->requires(Task::NewDW, lb->gInternalForceLabel, allPatches,
                                           Task::FineLevel,allMatls, ND, d_gn,0);
  
  t->modifies(lb->gInternalForceLabel);
  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->diffusion->gConcentrationRate, allPatches,
                                           Task::FineLevel,allMatls, ND, d_gn,0);
    t->modifies(lb->diffusion->gConcentrationRate);
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
                   
  t->requires(Task::NewDW, lb->gMassLabel,  d_gn);
  t->modifies(lb->gVelocityLabel);
  t->modifies(lb->gTemperatureLabel);
  
  if(flags->d_doScalarDiffusion){
    t->modifies(lb->diffusion->gConcentration);
    t->computes(lb->diffusion->gConcentrationNoBC);
    t->modifies(lb->diffusion->gHydrostaticStress);
  }
  if(flags->d_withGaussSolver){
    t->modifies(lb->gPosChargeLabel);
    t->modifies(lb->gNegChargeLabel);
    t->computes(lb->gPosChargeNoBCLabel);
    t->computes(lb->gNegChargeNoBCLabel);
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
  
  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
  Task* t = scinew Task("AMRMPM::computeStressTensor",
                  this, &AMRMPM::computeStressTensor);
                  
  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();
    
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
       
    t->computes(lb->p_qLabel_preReloc, matlset);
  }

  t->requires(Task::OldDW,lb->simulationTimeLabel);
  t->computes(lb->delTLabel,getLevel(patches));
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);
  
  //__________________________________
  //  Additional tasks
  scheduleUpdateStress_DamageErosionModels( sched, patches, matls );

  if (flags->d_reductionVars->accStrainEnergy) 
    scheduleComputeAccStrainEnergy(sched, patches, matls);
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

  t->requires(Task::NewDW,lb->gVolumeLabel, d_gn);
  t->requires(Task::OldDW,lb->pStressLabel,               d_gan,NGP);
  t->requires(Task::OldDW,lb->pVolumeLabel,               d_gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    d_gan,NGP);
  t->requires(Task::NewDW,lb->pCurSizeLabel,              d_gan,NGP);
  if(flags->d_artificial_viscosity){
    t->requires(Task::OldDW, lb->p_qLabel,                d_gan,NGP);
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
    t->requires(Task::OldDW, lb->pXLabel,       allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pStressLabel,  allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    t->requires(Task::OldDW, lb->pVolumeLabel,  allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
    
    if(flags->d_artificial_viscosity){
      t->requires(Task::OldDW, lb->p_qLabel,    allPatches, Task::CoarseLevel,allMatls, ND, d_gac, npc);
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

  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  // This stuff should probably go in its own task, but for expediency...JG
  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->diffusion->gConcentrationNoBC,  Ghost::None);
    t->requires(Task::NewDW, lb->diffusion->gConcentration,      Ghost::None);
    t->requires(Task::NewDW, lb->diffusion->gExternalScalarFlux, Ghost::None);
    t->requires(Task::NewDW, d_sdInterfaceModel->getInterfaceFluxLabel(), Ghost::None);
    t->modifies(lb->diffusion->gConcentrationRate);
    t->computes(lb->diffusion->gConcentrationStar);
  }

  if(flags->d_withGaussSolver){
    t->requires(Task::NewDW, lb->gPosChargeNoBCLabel, Ghost::None);
    t->requires(Task::NewDW, lb->gNegChargeNoBCLabel, Ghost::None);
    t->requires(Task::NewDW, lb->gPosChargeLabel,     Ghost::None);
    t->requires(Task::NewDW, lb->gNegChargeLabel,     Ghost::None);
    t->computes(lb->gPosChargeStarLabel);
    t->computes(lb->gNegChargeStarLabel);
    t->modifies(lb->gPosChargeRateLabel);
    t->modifies(lb->gNegChargeRateLabel);
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
  t->requires(Task::OldDW, lb->delTLabel );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);
  t->requires(Task::NewDW, lb->gVelocityLabel,   Ghost::None);

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

  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->gVelocityStarLabel,              d_gac,NGN);
  
  t->requires(Task::OldDW, lb->pXLabel,                         d_gn);
  t->requires(Task::OldDW, lb->pMassLabel,                      d_gn);
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   d_gn);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,        d_gn);

  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::NewDW, lb->diffusion->gConcentrationStar,       d_gac,NGN);
    t->requires(Task::OldDW, lb->diffusion->pArea,                    d_gn);
    t->computes(lb->diffusion->pGradConcentration_preReloc);
    t->computes(lb->diffusion->pArea_preReloc);
  }

  if(flags->d_withGaussSolver){
    t->requires(Task::NewDW, lb->gPosChargeStarLabel, d_gac, NGN);
    t->requires(Task::NewDW, lb->gNegChargeStarLabel, d_gac, NGN);
    t->computes(lb->pPosChargeGradLabel_preReloc);
    t->computes(lb->pNegChargeGradLabel_preReloc);
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
                
  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->gAccelerationLabel,              d_gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,              d_gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,           d_gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,             d_gac,NGN);
  
  t->requires(Task::OldDW, lb->pXLabel,                         d_gn);   
  t->requires(Task::OldDW, lb->pMassLabel,                      d_gn);   
  t->requires(Task::OldDW, lb->pParticleIDLabel,                d_gn);   
  t->requires(Task::OldDW, lb->pTemperatureLabel,               d_gn);   
  t->requires(Task::OldDW, lb->pVelocityLabel,                  d_gn);   
  t->requires(Task::OldDW, lb->pDispLabel,                      d_gn);   
  t->requires(Task::NewDW, lb->pCurSizeLabel,                   d_gn);   
  t->requires(Task::OldDW, lb->pVolumeLabel,                    d_gn);   

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress
  t->computes(lb->pMassLabel_preReloc);

  // Carry Forward particle refinement flag
  if(flags->d_refineParticles){
    t->requires(Task::OldDW, lb->pRefinedLabel,                d_gn);
    t->computes(             lb->pRefinedLabel_preReloc);
  }

  t->requires(Task::OldDW, lb->NC_CCweightLabel, d_one_matl, Ghost::None);
  t->computes(             lb->NC_CCweightLabel, d_one_matl);

  if(flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, lb->diffusion->pConcentration,           d_gn);
    t->requires(Task::NewDW, lb->diffusion->gConcentrationRate,       d_gac, NGN);

    t->computes(lb->diffusion->pConcentration_preReloc);
    t->computes(lb->diffusion->pConcPrevious_preReloc);
    if(flags->d_doAutoCycleBC){
      if(flags->d_autoCycleUseMinMax){
        t->computes(lb->diffusion->rMinConcentration);
        t->computes(lb->diffusion->rMaxConcentration);
      }else{
        t->computes(lb->diffusion->rTotalConcentration);
      }
    }
  }

  if(flags->d_withGaussSolver){
    t->requires(Task::OldDW, lb->pPosChargeLabel, d_gn);
    t->requires(Task::OldDW, lb->pNegChargeLabel, d_gn);
    t->requires(Task::OldDW, lb->pPermittivityLabel, d_gn);
    t->requires(Task::NewDW, lb->gPosChargeRateLabel, d_gac, NGN);
    t->requires(Task::NewDW, lb->gNegChargeRateLabel, d_gac, NGN);

    t->computes(lb->pPosChargeLabel_preReloc);
    t->computes(lb->pNegChargeLabel_preReloc);
    t->computes(lb->pPermittivityLabel_preReloc);
  }

  t->computes(lb->TotalMassLabel);
  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->TotalMomentumLabel);

#ifndef USE_DEBUG_TASK
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  d_gn);
    t->computes(lb->pColorLabel_preReloc);
  }
#endif  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
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
//______________________________________________________________________
//
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

  t->requires(Task::OldDW, lb->delTLabel );

  t->requires(Task::NewDW, lb->pdTdtLabel,           d_gn);
  t->requires(Task::NewDW, lb->pMassLabel_preReloc,  d_gn);

  t->modifies(lb->pTemperatureLabel_preReloc);

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::scheduleAddParticles(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls)
{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())){
      return;
    }

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
    if (flags->d_with_color) {
      t->modifies(lb->pColorLabel_preReloc);
    }
    if(flags->d_doScalarDiffusion){
      t->modifies(lb->diffusion->pConcentration_preReloc);
      t->modifies(lb->diffusion->pConcPrevious_preReloc);
      t->modifies(lb->diffusion->pGradConcentration_preReloc);
      t->modifies(lb->diffusion->pExternalScalarFlux_preReloc);
      t->modifies(lb->diffusion->pArea_preReloc);
      t->modifies(lb->diffusion->pDiffusivity_preReloc);
    }
    if (flags->d_useLoadCurves) {
      t->modifies(lb->pLoadCurveIDLabel_preReloc);
    }
    t->modifies(lb->pLocalizedMPMLabel_preReloc);
    t->modifies(lb->pExtForceLabel_preReloc);
    t->modifies(lb->pTemperatureLabel_preReloc);
    t->modifies(lb->pTempPreviousLabel_preReloc);
    t->modifies(lb->pDeformationMeasureLabel_preReloc);
    t->modifies(lb->pRefinedLabel_preReloc);
    if(flags->d_computeScaleFactor){
     t->modifies(lb->pScaleFactorLabel_preReloc);
    }
    t->modifies(lb->pLastLevelLabel_preReloc);
    t->modifies(lb->pVelGradLabel_preReloc);
    t->modifies(lb->MPMRefineCellLabel, d_one_matl);

    t->requires(Task::OldDW, lb->pCellNAPIDLabel, d_one_matl, Ghost::None);
    t->computes(             lb->pCellNAPIDLabel, d_one_matl);

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
      if(flags->d_doScalarDiffusion){
        ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        sdm->addSplitParticlesComputesAndRequires(t, mpm_matl, patches);
      }
    }

    sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
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
  t->computes(lb->pTempPreviousLabel); // for thermal  stress analysis
  t->computes(lb->pdTdtLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->diffusion->pExternalScalarFlux_preReloc);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  if(flags->d_doScalarDiffusion){
    t->computes(lb->diffusion->pConcentration);
    t->computes(lb->diffusion->pConcPrevious);
    t->computes(lb->diffusion->pGradConcentration);
    t->computes(lb->diffusion->pArea);
  }
  t->computes(lb->pLastLevelLabel);
  t->computes(lb->pLocalizedMPMLabel);
  t->computes(lb->pRefinedLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->pVelGradLabel);
  t->computes(lb->pCellNAPIDLabel, d_one_matl);
  t->computes(lb->NC_CCweightLabel);

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
    dm->addInitialComputesAndRequires(t, mpm_matl);
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addInitialComputesAndRequires(t, mpm_matl);
  }
  t->computes(lb->delTLabel,getLevel(patches));


  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
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

  Task* task = scinew Task("AMRMPM::coarsen",this,
                           &AMRMPM::coarsen);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.  
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const PatchSet* patch_set = coarseLevel->eachPatch();

  bool  fat = true;  // possibly (F)rom (A)nother (T)askgraph

  task->requires(Task::NewDW, lb->MPMRefineCellLabel,
               0, Task::FineLevel,  d_one_matl,oims, d_gn, 0, fat);

  task->requires(Task::NewDW, RefineFlagXMaxLabel);
  task->requires(Task::NewDW, RefineFlagXMinLabel);
  task->requires(Task::NewDW, RefineFlagYMaxLabel);
  task->requires(Task::NewDW, RefineFlagYMinLabel);
  task->requires(Task::NewDW, RefineFlagZMaxLabel);
  task->requires(Task::NewDW, RefineFlagZMinLabel);

  task->modifies(lb->MPMRefineCellLabel, d_one_matl, oims, fat);

  sched->addTask(task, patch_set, all_matls);
}
//______________________________________________________________________
//
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

  task->modifies(m_regridder->getRefineFlagLabel(),
                                        m_regridder->refineFlagMaterials());
  task->modifies(m_regridder->getRefinePatchFlagLabel(),
                                        m_regridder->refineFlagMaterials());
  task->requires(Task::NewDW, lb->MPMRefineCellLabel, Ghost::None);

  sched->addTask(task, coarseLevel->eachPatch(),
                                        m_materialManager->allMaterials( "MPM" ));
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
  
  if(pg->myRank() == 0){
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
    
    
    printTask(patches, patch,cout_doing,"Doing AMRMPM::actuallyInitialize");

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
      int indx = mpm_matl->getDWIndex();
      
      particleIndex numParticles = mpm_matl->createParticles(cellNAPID, patch, new_dw);

      totalParticles+=numParticles;
      mpm_matl->getConstitutiveModel()->initializeCMData(patch,mpm_matl,new_dw);
      
      //initialize damage/erosion model
      mpm_matl->getDamageModel()->initializeLabels( patch, mpm_matl, new_dw );
      
      mpm_matl->getErosionModel()->initializeLabels( patch, mpm_matl, new_dw );
      
      if(flags->d_doScalarDiffusion){
    	  mpm_matl->getScalarDiffusionModel()->initializeTimeStep(patch,mpm_matl,new_dw);
    	  mpm_matl->getScalarDiffusionModel()->initializeSDMData(patch,mpm_matl,new_dw);
      }
      
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

  if(flags->d_doAutoCycleBC && flags->d_doScalarDiffusion){
    if(flags->d_autoCycleUseMinMax){
      new_dw->put(min_vartype(5e11), lb->diffusion->rMinConcentration);
      new_dw->put(max_vartype(-5e11), lb->diffusion->rMaxConcentration);
    }else{
      new_dw->put(sum_vartype(0.0), lb->diffusion->rTotalConcentration);
    }
  }
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

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

        int NN = interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);

        for(int k = 0; k < NN; k++) {
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

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

#ifdef CBDI_FLUXBCS
    LinearInterpolator* LPI;
    LPI = scinew LinearInterpolator(patch);
    vector<IntVector> ni_LPI(8);
    vector<double> S_LPI(8);
#endif

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<double> pConcentration;
      constParticleVariable<double> pExternalScalarFlux;
      constParticleVariable<Vector> pvelocity, pexternalforce;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pStress;
      constParticleVariable<Matrix3> pVelGrad;
      constParticleVariable<double> pPosCharge;
      constParticleVariable<double> pNegCharge;
      
      constParticleVariable<Vector> pConcGrad;
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       d_gan, NGP, lb->pXLabel);

      old_dw->get(px,                   lb->pXLabel,                  pset);
      old_dw->get(pmass,                lb->pMassLabel,               pset);
      old_dw->get(pvolume,              lb->pVolumeLabel,             pset);
      old_dw->get(pvelocity,            lb->pVelocityLabel,           pset);
      old_dw->get(pTemperature,         lb->pTemperatureLabel,        pset);

#ifdef CBDI_FLUXBCS
      constParticleVariable<IntVector> pLoadCurveID;
      if (flags->d_useLoadCurves) {
        old_dw->get(pLoadCurveID,       lb->pLoadCurveIDLabel,        pset);
      }
#endif
      new_dw->get(psize,                lb->pCurSizeLabel,            pset);
      new_dw->get(pexternalforce,       lb->pExtForceLabel_preReloc,  pset);
      if (flags->d_GEVelProj){
        old_dw->get(pVelGrad,           lb->pVelGradLabel,            pset);
      }
      if(flags->d_doScalarDiffusion){
        new_dw->get(pExternalScalarFlux,lb->diffusion->pExternalScalarFlux_preReloc, pset);
        old_dw->get(pConcentration,     lb->diffusion->pConcentration,      pset);
        old_dw->get(pStress,            lb->pStressLabel,             pset);
        if (flags->d_GEVelProj) {
          old_dw->get(pConcGrad, lb->diffusion->pGradConcentration, pset);
        }
      }
      if(flags->d_withGaussSolver){
        old_dw->get(pPosCharge, lb->pPosChargeLabel, pset);
        old_dw->get(pNegCharge, lb->pNegChargeLabel, pset);
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
      NCVariable<double> gposcharge;
      NCVariable<double> gnegcharge;

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
        new_dw->allocateAndPut(gconcentration,     lb->diffusion->gConcentration,
                                                               dwi,  patch);
        new_dw->allocateAndPut(ghydrostaticstress, lb->diffusion->gHydrostaticStress,
                                                               dwi,  patch);
        new_dw->allocateAndPut(gextscalarflux,     lb->diffusion->gExternalScalarFlux,
                                                                dwi,  patch);
        gconcentration.initialize(0);
        ghydrostaticstress.initialize(0);
        gextscalarflux.initialize(0);
      }
      if(flags->d_withGaussSolver){
        new_dw->allocateAndPut(gposcharge, lb->gPosChargeLabel, dwi, patch);
        new_dw->allocateAndPut(gnegcharge, lb->gNegChargeLabel, dwi, patch);
        gposcharge.initialize(0.0);
        gnegcharge.initialize(0.0);
      }
      
      Vector pmom;
      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);

        pmom = pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        IntVector node;
        for(int k = 0; k < NN; k++) {
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
          double pConc_Ext = pConcentration[idx];
          for(int k = 0; k < NN; k++) {
            node = ni[k];
            if(patch->containsNode(node)) {
              if (flags->d_GEVelProj) {
                Point gpos = patch->getNodePosition(node);
                Vector pointOffset = px[idx]-gpos;
                pConc_Ext -= Dot(pConcGrad[idx],pointOffset);
              }
              ghydrostaticstress[node] += phydrostress        * pmass[idx]*S[k];
              gconcentration[node]     += pConc_Ext           * pmass[idx]*S[k];
#ifndef CBDI_FLUXBCS
              gextscalarflux[node]+= (pExternalScalarFlux[idx]*pmass[idx])*S[k];
#endif
            }
          }
        }
        if(flags->d_withGaussSolver){
          for(int k = 0; k < NN; k++) {
            node = ni[k];
            if(patch->containsNode(node)) {
              gposcharge[node] += pPosCharge[idx] * pmass[idx]*S[k];
              gnegcharge[node] += pNegCharge[idx] * pmass[idx]*S[k];
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
        if(pLoadCurveID[idx].x()==1){
          flux_pos=Point(px[idx].x()-0.5*psize[idx](0,0)*dx.x(),
                         px[idx].y(),
                         px[idx].z());
        }
        if(pLoadCurveID[idx].x()==2){
          flux_pos=Point(px[idx].x()+0.5*psize[idx](0,0)*dx.x(),
                         px[idx].y(),
                         px[idx].z());
        }
        if(pLoadCurveID[idx].x()==3){
          flux_pos=Point(px[idx].x(),
                         px[idx].y()+0.5*psize[idx](1,1)*dx.y(),
                         px[idx].z());
        }
        LPI->findCellAndWeights(flux_pos,ni_LPI,S_LPI,psize[idx]);
        for(int k = 0; k < 8; k++) {
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

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator =flags->d_interpolator->clone(finePatch);

    constNCVariable<Stencil7> zoi_fine;
    new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, d_gn, 0 );

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
    finePatch->getOtherLevelPatches55902(-1, coarsePatches, padding);

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
        new_dw->getModifiable(gConc_fine,          lb->diffusion->gConcentration,    dwi,finePatch);
        new_dw->getModifiable(gExtScalarFlux_fine, lb->diffusion->gExternalScalarFlux,    dwi,finePatch);
        new_dw->getModifiable(gHStress_fine,       lb->diffusion->gHydrostaticStress,dwi,finePatch);
      }

      // loop over the coarse patches under the fine patches.
      for(size_t cp=0; cp<coarsePatches.size(); cp++){
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
          old_dw->get(pConc_coarse,        lb->diffusion->pConcentration,      pset);
          new_dw->get(pExtScalarFlux_c,    lb->diffusion->pExternalScalarFlux_preReloc, pset);
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
//         G I M P
//  At the CFI fine patch nodes add contributions from the coarse level particles.
void AMRMPM::interpolateParticlesToGrid_CFI_GIMP(const ProcessorGroup*,
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
                          "Doing AMRMPM::interpolateParticlesToGrid_CFI_GIMP");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    ParticleInterpolator* interpolator =flags->d_interpolator->clone(finePatch);

    constNCVariable<Stencil7> zoi_fine;
    new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, d_gn, 0 );

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
    finePatch->getOtherLevelPatches55902(-1, coarsePatches, padding);

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
        new_dw->getModifiable(gConc_fine,          lb->diffusion->gConcentration,    dwi,finePatch);
        new_dw->getModifiable(gExtScalarFlux_fine, lb->diffusion->gExternalScalarFlux,    dwi,finePatch);
        new_dw->getModifiable(gHStress_fine,       lb->diffusion->gHydrostaticStress,dwi,finePatch);
      }

      // loop over the coarse patches under the fine patches.
      for(size_t cp=0; cp<coarsePatches.size(); cp++){
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
          old_dw->get(pConc_coarse,        lb->diffusion->pConcentration,      pset);
          new_dw->get(pExtScalarFlux_c,    lb->diffusion->pExternalScalarFlux_preReloc, pset);
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
            double pESFlux_c = pExtScalarFlux_c[idx]*pMass_coarse[idx];

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
  Level::selectType CFI_coarsePatches;
  Level::selectType CFI_finePatches;
  
  coarseLevelCFI_Patches(coarsePatches,CFI_coarsePatches, CFI_finePatches  );
  
  //__________________________________
  // From the coarse patch look up to the fine patches that have
  // coarse fine interfaces.
  const Level* coarseLevel = getLevel(coarsePatches);
  
  for(size_t p=0;p<CFI_coarsePatches.size();p++){
    const Patch* coarsePatch = CFI_coarsePatches[p];

    string txt = "(zero)";
    if (flag == coarsenData){
      txt = "(coarsen)";
    }
    printTask(coarsePatch,cout_doing,"Doing AMRMPM::coarsenNodalData_CFI"+txt);
    
    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMatls; m++){                               
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );    
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
        new_dw->getModifiable(gConcentration_coarse, lb->diffusion->gConcentration,  dwi,coarsePatch);
        new_dw->getModifiable(gExtScalarFlux_coarse, lb->diffusion->gExternalScalarFlux,  dwi,coarsePatch);
      }

      if(flag == zeroData){
        new_dw->getModifiable(gVelocityStar_coarse,  lb->gVelocityStarLabel,     dwi,coarsePatch);
        new_dw->getModifiable(gAcceleration_coarse,  lb->gAccelerationLabel,     dwi,coarsePatch);
      }

      //__________________________________
      // Iterate over coarse/fine interface faces
      ASSERT(coarseLevel->hasFinerLevel());
      const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

      // loop over all the fine level patches
      for(size_t fp=0;fp<CFI_finePatches.size();fp++){
        const Patch* finePatch = CFI_finePatches[fp];

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
             new_dw->getRegion(gConcentration_fine,lb->diffusion->gConcentration, dwi, fineLevel,fl, fh);
             new_dw->getRegion(gExtScalarFlux_fine,lb->diffusion->gExternalScalarFlux,
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
  Level::selectType CFI_coarsePatches;
  Level::selectType CFI_finePatches;
  coarseLevelCFI_Patches(coarsePatches, CFI_coarsePatches, CFI_finePatches );

  //__________________________________
  // From the coarse patch look up to the fine patches that have
  // coarse fine interfaces.
  const Level* coarseLevel = getLevel(coarsePatches);
  
  for(size_t p=0;p<CFI_coarsePatches.size();p++){
    const Patch* coarsePatch = CFI_coarsePatches[p];

    printTask(coarsePatch,cout_doing,"Doing AMRMPM::coarsenNodalData_CFI2");
    
    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMatls; m++){                               
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );    
      int dwi = mpm_matl->getDWIndex();
      
      // get coarse level data
      NCVariable<Vector> internalForce_coarse;                    
      new_dw->getModifiable(internalForce_coarse, lb->gInternalForceLabel, 
                                                               dwi,coarsePatch);
      NCVariable<double> gConcRate_coarse;                    
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConcRate_coarse,   lb->diffusion->gConcentrationRate,
                                                               dwi,coarsePatch);
      }

      //__________________________________
      // Iterate over coarse/fine interface faces
      ASSERT(coarseLevel->hasFinerLevel());
      const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();

            // loop over all the fine level patches
      for(size_t fp=0;fp<CFI_finePatches.size();fp++){
        const Patch* finePatch = CFI_finePatches[fp];

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
            new_dw->getRegion(gConcRate_fine,  lb->diffusion->gConcentrationRate,
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
                }
              }  //  node loop
            }  //  isRight_CP_FP_pair
          }  //  end CFI face loop
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

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      // get  level nodal data
      constNCVariable<double> gMass;
      NCVariable<Vector> gVelocity;
      NCVariable<double> gTemperature;
      NCVariable<double> gConcentration;
      NCVariable<double> gConcentrationNoBC;
      NCVariable<double> gHydroStress;
      NCVariable<double> gPosCharge;
      NCVariable<double> gNegCharge;
      NCVariable<double> gPosChargeNoBC;
      NCVariable<double> gNegChargeNoBC;
      
      new_dw->get(gMass,                  lb->gMassLabel,       dwi,patch,d_gn,0);
      new_dw->getModifiable(gVelocity,    lb->gVelocityLabel,   dwi,patch,d_gn,0);
      new_dw->getModifiable(gTemperature, lb->gTemperatureLabel,dwi,patch,d_gn,0);
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(gConcentration,      
                                    lb->diffusion->gConcentration,    dwi,patch,d_gn,0);
        new_dw->getModifiable(gHydroStress,
                                    lb->diffusion->gHydrostaticStress,dwi,patch,d_gn,0);
        new_dw->allocateAndPut(gConcentrationNoBC,
                                    lb->diffusion->gConcentrationNoBC,dwi,patch);
      }
      if(flags->d_withGaussSolver){
        new_dw->getModifiable(gPosCharge, lb->gPosChargeLabel, dwi, patch, d_gn,0);
        new_dw->getModifiable(gNegCharge, lb->gNegChargeLabel, dwi, patch, d_gn,0);
        new_dw->allocateAndPut(gPosChargeNoBC, lb->gPosChargeNoBCLabel, dwi, patch);
        new_dw->allocateAndPut(gNegChargeNoBC, lb->gNegChargeNoBCLabel, dwi, patch);
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
      if(flags->d_withGaussSolver){
        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done(); iter++){
          IntVector n = *iter;
          gPosCharge[n] /= gMass[n];
          gNegCharge[n]   /= gMass[n];
          gPosChargeNoBC[n] = gPosCharge[n];
          gNegChargeNoBC[n] = gNegCharge[n];
        }
      }
      
      // Apply boundary conditions to the temperature and velocity (if symmetry)
      MPMBoundCond bc;
      string interp_type = flags->d_interpolator_type;
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature,interp_type);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gVelocity,   interp_type);
      if(flags->d_doScalarDiffusion){
        bc.setBoundaryCondition(patch,dwi,"SD-Type",  gConcentration,
                                                                   interp_type);
      }
      if(flags->d_withGaussSolver){
        bc.setBoundaryCondition(patch,dwi, "PosCharge", gPosCharge, interp_type);
        bc.setBoundaryCondition(patch,dwi, "NegCharge", gNegCharge, interp_type);
      }
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

  for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){

    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);

    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

    cm->setWorld(d_myworld);
    cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
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

    unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                              pset);
      old_dw->get(pvol,    lb->pVolumeLabel,                         pset);
      old_dw->get(pstress, lb->pStressLabel,                         pset);
      new_dw->get(psize,   lb->pCurSizeLabel,                        pset);

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
      vector<IntVector> ni(interpolator->size());
      vector<double> S(interpolator->size());
      vector<Vector> d_S(interpolator->size());
    

      for (ParticleSubset::iterator iter  = pset->begin();
                                    iter != pset->end();  iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,
                                                            S, d_S, psize[idx]);

        stresspress = pstress[idx] + Id*(/*p_pressure*/-p_q[idx]);

        for (int k = 0; k < NN; k++){
          
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
      finePatch->getOtherLevelPatches55902(-1, coarsePatches, padding);
        
      Matrix3 Id;
      Id.Identity();
        
      constNCVariable<Stencil7> zoi_fine;
      new_dw->get(zoi_fine, lb->gZOILabel, 0, finePatch, Ghost::None, 0 );
  
      unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
      
      for(unsigned int m = 0; m < numMPMMatls; m++){
        MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
        int dwi = mpm_matl->getDWIndex();
        
        NCVariable<Vector> internalforce;
        new_dw->getModifiable(internalforce,lb->gInternalForceLabel,  
                                                     dwi, finePatch);

  /*`==========TESTING==========*/
        NCVariable<double> gSumS;
        new_dw->getModifiable(gSumS, gSumSLabel,  dwi, finePatch);
  /*===========TESTING==========`*/ 
        
        // loop over the coarse patches under the fine patches.
        for(size_t cp=0; cp<coarsePatches.size(); cp++){
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


#if 0
  /*`==========TESTING==========*/
                if(std::isinf( internalforce[fineNode].length() ) ||  std::isnan( internalforce[fineNode].length() )){
                  cout << "INF: " << fineNode << " " << internalforce[fineNode] 
                       << " div[k]:"<< div[k] << " stressPress: " << stresspress
                       << " pvol " << pvol_coarse[idx] << endl;
                }
  /*===========TESTING==========`*/
#endif
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

    Vector gravity = flags->d_gravity;
    
    for(unsigned int m = 0; m < m_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      mpm_matl->getScalarDiffusionModel();

      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<Vector> gvelocity;
      constNCVariable<double> gmass;
      constNCVariable<double> gConcentration,gConcNoBC,gExtScalarFlux;
      constNCVariable<double> gSDIFFluxRate;
      constNCVariable<double> gPosCharge, gPosChargeNoBC;
      constNCVariable<double> gNegCharge, gNegChargeNoBC;

      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel, getLevel(patches) );

      new_dw->get(internalforce, lb->gInternalForceLabel, dwi, patch, d_gn, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, dwi, patch, d_gn, 0);
      new_dw->get(gmass,         lb->gMassLabel,          dwi, patch, d_gn, 0);
      new_dw->get(gvelocity,     lb->gVelocityLabel,      dwi, patch, d_gn, 0);

      // Create variables for the results
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      NCVariable<double> gConcStar,gConcRate;
      NCVariable<double> gPosChargeStar, gPosChargeRate;
      NCVariable<double> gNegChargeStar, gNegChargeRate;

      new_dw->allocateAndPut(gvelocity_star, lb->gVelocityStarLabel, dwi,patch);
      new_dw->allocateAndPut(gacceleration,  lb->gAccelerationLabel, dwi,patch);

      if(flags->d_doScalarDiffusion){
        const VarLabel* SDIFFluxVarLabel =
                          d_sdInterfaceModel->getInterfaceFluxLabel();
        new_dw->get(gSDIFFluxRate,  SDIFFluxVarLabel,                   dwi,patch,d_gn,0);                  
        new_dw->get(gConcentration, lb->diffusion->gConcentration,      dwi,patch,d_gn,0);
        new_dw->get(gConcNoBC,      lb->diffusion->gConcentrationNoBC,  dwi,patch,d_gn,0);
        new_dw->get(gExtScalarFlux, lb->diffusion->gExternalScalarFlux, dwi,patch,d_gn,0);

        new_dw->getModifiable( gConcRate,lb->diffusion->gConcentrationRate,dwi,patch);
        new_dw->allocateAndPut(gConcStar,lb->diffusion->gConcentrationStar,dwi,patch);
      }

      if(flags->d_withGaussSolver){
        new_dw->get(gPosCharge,     lb->gPosChargeLabel,     dwi, patch, d_gn, 0);
        new_dw->get(gPosChargeNoBC, lb->gPosChargeNoBCLabel, dwi, patch, d_gn, 0);
        new_dw->get(gNegCharge,     lb->gNegChargeLabel,     dwi, patch, d_gn, 0);
        new_dw->get(gNegChargeNoBC, lb->gNegChargeNoBCLabel, dwi, patch, d_gn, 0);

        new_dw->allocateAndPut(gPosChargeStar, lb->gPosChargeStarLabel, dwi, patch);
        new_dw->allocateAndPut(gNegChargeStar, lb->gNegChargeStarLabel, dwi, patch);

        new_dw->getModifiable(gPosChargeRate, lb->gPosChargeRateLabel,dwi, patch);
        new_dw->getModifiable(gNegChargeRate, lb->gNegChargeRateLabel,dwi, patch);
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
          gConcRate[c] /= gmass[c];
          gConcStar[c]  =  gConcentration[c] + (gConcRate[c] + gSDIFFluxRate[c]) * delT;
        }

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch, dwi,"SD-Type", gConcStar,
                                                flags->d_interpolator_type);

        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
          IntVector c = *iter;
          gConcRate[c] = (gConcStar[c] - gConcNoBC[c]) / delT
                       + gExtScalarFlux[c]/gmass[c];
        }
      } // if doScalarDiffusion

      if(flags->d_withGaussSolver){
        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
          IntVector c = *iter;
          gPosChargeRate[c] /= gmass[c];
          gPosChargeStar[c]  = gPosCharge[c] + gPosChargeRate[c] * delT;
          gNegChargeRate[c] /= gmass[c];
          gNegChargeStar[c]  = gNegCharge[c] + gNegChargeRate[c] * delT;
        }

        MPMBoundCond bc;
        bc.setBoundaryCondition(patch, dwi,"PosCharge", gConcStar,
                                      flags->d_interpolator_type);
        bc.setBoundaryCondition(patch, dwi,"NegCharge", gConcStar,
                                      flags->d_interpolator_type);

        for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
          IntVector c = *iter;
          gPosChargeRate[c] = (gPosChargeStar[c] - gPosChargeNoBC[c]) / delT;
          gNegChargeRate[c] = (gNegChargeStar[c] - gNegChargeNoBC[c]) / delT;
        }
      } // if d_withGaussSolver

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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    
    delt_vartype delT;            
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    
    string interp_type = flags->d_interpolator_type;

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
  
  // find the fine & coarse CFI patches
  Level::selectType CFI_coarsePatches;
  Level::selectType CFI_finePatches;
  
  coarseLevelCFI_Patches(patches,CFI_coarsePatches, CFI_finePatches  );

  //__________________________________
  // Set the ZOI on the current level.
  // Look up at the finer level patches
  // for coarse-fine interfaces
  for(size_t p=0;p<CFI_coarsePatches.size();p++){
    const Patch* coarsePatch = CFI_coarsePatches[p];
    
    NCVariable<Stencil7> zoi;
    new_dw->getModifiable(zoi, lb->gZOILabel, 0,coarsePatch);

    const Level* fineLevel = level->getFinerLevel().get_rep();


    for(size_t p=0;p<CFI_finePatches.size();p++){  
      const Patch* finePatch = CFI_finePatches[p];

      Vector fine_dx = finePatch->dCell();

      //__________________________________
      // Iterate over coarsefine interface faces
      vector<Patch::FaceType> cf;
      finePatch->getCoarseFaces(cf);

      vector<Patch::FaceType>::const_iterator iter;  
      for (iter  = cf.begin(); iter != cf.end(); ++iter){
        Patch::FaceType patchFace = *iter;

        // determine the iterator on the coarse level.
        NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
        bool isRight_CP_FP_pair;

        coarseLevel_CFI_NodeIterator( patchFace,coarsePatch, finePatch, fineLevel,
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
        IntVector dir = coarsePatch->getFaceAxes(patchFace); // face axes
        int p_dir = dir[0];                                  // normal direction

        // eject if this is not the right coarse/fine patch pair
        if (isRight_CP_FP_pair){

//         cout << "  A) Setting ZOI  " 
//              << " \t On L-" << level->getIndex() << " patch  " << coarsePatch->getID()
//              << ", beneath patch " << finePatch->getID() << ", face: "  << finePatch->getFaceName(patchFace) 
//              << ", isRight_CP_FP_pair: " << isRight_CP_FP_pair  << " n_iter: " << n_iter << endl;

          for(; !n_iter.done(); n_iter++) {
            IntVector c = *n_iter;
            zoi[c][element]=fine_dx[p_dir];
          }
        }

      }  // patch face loop
    }  // finePatches loop
  }  // coarse patches loop


  //__________________________________
  // set the ZOI in cells in which there are overlaping coarse level nodes
  // look down for coarse level patches 

  Level::selectType coarsePatches;
  Level::selectType CFI_finePatches2;
  fineLevelCFI_Patches(patches,coarsePatches, CFI_finePatches2  );
  
  for(size_t p=0;p<CFI_finePatches2.size();p++){
    const Patch* finePatch = CFI_finePatches2[p];

    NCVariable<Stencil7> zoi_fine;
    new_dw->getModifiable(zoi_fine, lb->gZOILabel, 0,finePatch);

    //__________________________________
    // Iterate over coarse/fine interface faces
    vector<Patch::FaceType> cf;
    finePatch->getCoarseFaces(cf);

    vector<Patch::FaceType>::const_iterator iter;  
    for (iter  = cf.begin(); iter != cf.end(); ++iter){
      Patch::FaceType patchFace = *iter;
      bool setFace = false;

      for(size_t p=0;p<coarsePatches.size();p++){
        const Patch* coarsePatch = coarsePatches[p];
        Vector coarse_dx = coarsePatch->dCell();

        // determine the iterator on the coarse level.
        NodeIterator n_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
        bool isRight_CP_FP_pair;

        fineLevel_CFI_NodeIterator( patchFace,coarsePatch, finePatch,
                                      n_iter ,isRight_CP_FP_pair);
                   
        // Is this the right coarse/fine patch pair
        if (isRight_CP_FP_pair){
          int element   = patchFace;
          IntVector dir = finePatch->getFaceAxes(patchFace); // face axes
          int p_dir     = dir[0];                           // normal dir
          setFace = true; 

//          cout << "  C) Setting ZOI  "                                                                 
//               << " \t On L-" << level->getIndex() << " patch  " << finePatch->getID()                 
//               << "  coarsePatch " << coarsePatch->getID()                                             
//               << "   CFI face: "  << finePatch->getFaceName(patchFace)                                
//               << " isRight_CP_FP_pair: " << isRight_CP_FP_pair  << " n_iter: " << n_iter << endl;                 

          for(; !n_iter.done(); n_iter++) {
            IntVector c = *n_iter;
            zoi_fine[c][element]=coarse_dx[p_dir];
          }
        }
      }  // coarsePatches loop

      //__________________________________
      // bulletproofing
      if( !setFace ){ 
          ostringstream warn;
          warn << "\n ERROR: computeZoneOfInfluence:Fine Level: Did not find node iterator! "
               << "\n coarse: L-" << level->getIndex()
               << "\n coarePatches size: " << CFI_coarsePatches.size()
               << "\n fine patch:   " << *finePatch
               << "\n fine patch face: " << finePatch->getFaceName(patchFace);
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }  // face interator
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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Vector> parea;
      ParticleVariable<double> pvolume;
      constParticleVariable<double> pmass;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Matrix3> pFNew,pVelGrad;
      ParticleVariable<Vector> pConcGradNew,pareanew;
      ParticleVariable<Vector> pPosChargeGrad, pNegChargeGrad;

      // Get the arrays of grid data on which the new particle values depend
      constNCVariable<Vector> gvelocity_star;
      constNCVariable<double> gConcStar;
      constNCVariable<double> gPosChargeStar;
      constNCVariable<double> gNegChargeStar;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);
      old_dw->get(pFOld,        lb->pDeformationMeasureLabel,        pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);

      new_dw->allocateAndPut(pvolume,     lb->pVolumeLabel_preReloc,      pset);
      new_dw->allocateAndPut(pVelGrad,    lb->pVelGradLabel_preReloc,     pset);
      new_dw->allocateAndPut(pFNew,       lb->pDeformationMeasureLabel_preReloc,
                                                                          pset);
      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel, dwi,patch,d_gac,NGP);

      if(flags->d_doScalarDiffusion){
        old_dw->get(parea,        lb->diffusion->pArea,                      pset);
        new_dw->get(gConcStar,    lb->diffusion->gConcentrationStar, dwi,
                                                             patch, d_gac, NGP);

        new_dw->allocateAndPut(pareanew,    lb->diffusion->pArea_preReloc,      pset);
        new_dw->allocateAndPut(pConcGradNew,lb->diffusion->pGradConcentration_preReloc,
                                                                          pset);
      }

      if(flags->d_withGaussSolver){
        new_dw->get(gPosChargeStar, lb->gPosChargeStarLabel, dwi, patch, d_gac, NGP);
        new_dw->get(gNegChargeStar, lb->gNegChargeStarLabel, dwi, patch, d_gac, NGP);

        new_dw->allocateAndPut(pPosChargeGrad, lb->pPosChargeGradLabel_preReloc, pset);
        new_dw->allocateAndPut(pNegChargeGrad, lb->pNegChargeGradLabel_preReloc, pset);
      }

      double rho_init=mpm_matl->getInitialDensity();
      Matrix3 Identity;
      Identity.Identity();
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        int NN=flags->d_8or27;

        Matrix3 tensorL(0.0);
        if(!flags->d_axisymmetric){
         // Get the node indices that surround the cell
         NN = interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,
                                                        psize[idx]);

         computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity_star,NN);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         NN = interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,
                                                     d_S,psize[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity_star,
                                                                   px[idx],NN);
        }
        pVelGrad[idx]=tensorL;
        if(flags->d_doScalarDiffusion){
          pConcGradNew[idx] = Vector(0.0, 0.0, 0.0);
          for(int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            for(int j = 0; j < 3; j++){
              pConcGradNew[idx][j] += gConcStar[ni[k]] * d_S[k][j] * oodx[j];
            }
          }
        }
        if(flags->d_withGaussSolver){
          pPosChargeGrad[idx] = Vector(0.0, 0.0, 0.0);
          pNegChargeGrad[idx] = Vector(0.0, 0.0, 0.0);
          for(int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            for(int j = 0; j < 3; j++){
              pPosChargeGrad[idx][j] += gPosChargeStar[ni[k]] * d_S[k][j] * oodx[j];
              pNegChargeGrad[idx][j] += gNegChargeStar[ni[k]] * d_S[k][j] * oodx[j];
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
        if(flags->d_doScalarDiffusion){
          pareanew[idx]         = parea[idx];
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

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    double totalmass = 0;
    Vector CMX(0.0,0.0,0.0);
    Vector totalMom(0.0,0.0,0.0);
    double ke=0;

    double totalconc = 0;
    double minPatchConc = 5e11;
    double maxPatchConc =-5e11;

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

      // Used for AutoCycleFluxBC, scalar flux boundary conditions
      bool do_conc_reduction = mpm_matl->doConcReduction();

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
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

      constParticleVariable<double> pPosCharge, pNegCharge, pPermittivity;
      constNCVariable<double> gPosChargeRate, gNegChargeRate;
      ParticleVariable<double> pPosChargeNew, pNegChargeNew, pPermittivityNew;

      // Get the arrays of grid data on which the new particle values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate;
      constNCVariable<double> dTdt, frictionTempRate;
      double Cp = mpm_matl->getSpecificHeat();

      double sdmMaxEffectiveConc = -999;
      double sdmMinEffectiveConc =  999;
      if (flags->d_doScalarDiffusion) {
        // Grab min/max concentration and conc. tolerance for particle loop.
        ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
        sdmMaxEffectiveConc = sdm->getMaxConcentration() - sdm->getConcentrationTolerance();
        sdmMinEffectiveConc = sdm->getMinConcentration() + sdm->getConcentrationTolerance();
      }
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,           lb->pXLabel,                         pset);
      old_dw->get(pdisp,        lb->pDispLabel,                      pset);
      old_dw->get(pmass,        lb->pMassLabel,                      pset);
      old_dw->get(pvelocity,    lb->pVelocityLabel,                  pset);
      old_dw->get(pTemperature, lb->pTemperatureLabel,               pset);
      new_dw->get(psize,        lb->pCurSizeLabel,                   pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);

      if(flags->d_doScalarDiffusion){
        old_dw->get(pConcentration,     lb->diffusion->pConcentration,     pset);
        new_dw->get(gConcentrationRate, lb->diffusion->gConcentrationRate,
                                                   dwi, patch, d_gac, NGP);
        new_dw->allocateAndPut(pConcentrationNew,
                                        lb->diffusion->pConcentration_preReloc, pset);
        new_dw->allocateAndPut(pConcPreviousNew,
                                        lb->diffusion->pConcPrevious_preReloc,  pset);

      }

      if(flags->d_withGaussSolver){
        old_dw->get(pPosCharge,    lb->pPosChargeLabel,    pset);
        old_dw->get(pNegCharge,    lb->pNegChargeLabel,    pset);
        old_dw->get(pPermittivity, lb->pPermittivityLabel, pset);

        new_dw->get(gPosChargeRate, lb->gPosChargeRateLabel, dwi, patch, d_gac, NGP);
        new_dw->get(gNegChargeRate, lb->gNegChargeRateLabel, dwi, patch, d_gac, NGP);

        new_dw->allocateAndPut(pPosChargeNew,    lb->pPosChargeLabel_preReloc,    pset);
        new_dw->allocateAndPut(pNegChargeNew,    lb->pNegChargeLabel_preReloc,    pset);
        new_dw->allocateAndPut(pPermittivityNew, lb->pPermittivityLabel_preReloc, pset);
      }

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      //Carry forward ParticleID and pSize
      old_dw->get(pids,                lb->pParticleIDLabel,          pset);
      new_dw->allocateAndPut(pids_new, lb->pParticleIDLabel_preReloc, pset);
      pids_new.copyData(pids);

      new_dw->get(gvelocity_star,  lb->gVelocityStarLabel,   dwi,patch,d_gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,d_gac,NGP);
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,d_gac,NGP);
      new_dw->get(frictionTempRate,lb->frictionalWorkLabel,  dwi,patch,d_gac,NGP);

      if(flags->d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,d_gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create,massBurnFrac_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,d_gac,NGP);
        dTdt_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);

        Vector vel(0.0,0.0,0.0);
        Vector acc(0.0,0.0,0.0);
        double fricTempRate = 0.0;
        double tempRate = 0.0;
        double concRate = 0.0;

        // Accumulate the contribution from vertices on this level
        for(int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel      += gvelocity_star[node]  * S[k];
          acc      += gacceleration[node]   * S[k];

          fricTempRate = frictionTempRate[node]*flags->d_addFrictionWork;
          tempRate += (gTemperatureRate[node] + dTdt[node] +
                       fricTempRate)   * S[k];
        }

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx]    + vel*delT;
        pdispnew[idx]        = pdisp[idx] + vel*delT;
        pvelocitynew[idx]    = pvelocity[idx]    + acc*delT;

        pTempNew[idx]        = pTemperature[idx] + tempRate*delT;
        pTempPreNew[idx]     = pTemperature[idx]; // for thermal stress
        pmassNew[idx]        = pmass[idx];

        if(flags->d_doScalarDiffusion){
          for(int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            concRate += gConcentrationRate[node]   * S[k];
          }

          pConcentrationNew[idx]= pConcentration[idx] + concRate*delT;
          if(pConcentrationNew[idx] < sdmMinEffectiveConc ){
            pConcentrationNew[idx] = sdmMinEffectiveConc;
          }
          if (pConcentrationNew[idx] > sdmMaxEffectiveConc ) {
            pConcentrationNew[idx] = sdmMaxEffectiveConc;
          }
          pConcPreviousNew[idx] = pConcentration[idx];
          if(do_conc_reduction){
            if(flags->d_autoCycleUseMinMax){
              if(pConcentrationNew[idx] > maxPatchConc)
                maxPatchConc = pConcentrationNew[idx];
              if(pConcentrationNew[idx] < minPatchConc)
                minPatchConc = pConcentrationNew[idx];
            }else{
              totalconc += pConcentration[idx];
            }
          }
        }

        if(flags->d_withGaussSolver){
          double posChargeRate = 0.0;
          double negChargeRate = 0.0;
          for(int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            posChargeRate += gPosChargeRate[node] * S[k];
            negChargeRate += gNegChargeRate[node] * S[k];
          }

          pPosChargeNew[idx] = pPosCharge[idx] + posChargeRate * delT;
          pNegChargeNew[idx] = pNegCharge[idx] + negChargeRate * delT;
          pPermittivityNew[idx] = pPermittivity[idx];
          if(pPosChargeNew[idx] < 0.0)
            pPosChargeNew[idx] = 0.0;
          if(pNegChargeNew[idx] < 0.0)
            pNegChargeNew[idx] = 0.0;
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

      new_dw->deleteParticles(delset);    

      new_dw->put(sum_vartype(totalmass),       lb->TotalMassLabel);
      new_dw->put(sum_vartype(ke),              lb->KineticEnergyLabel);
      new_dw->put(sum_vartype(thermal_energy),  lb->ThermalEnergyLabel);
      new_dw->put(sumvec_vartype(CMX),          lb->CenterOfMassPositionLabel);
      new_dw->put(sumvec_vartype(totalMom),     lb->TotalMomentumLabel);

      if(flags->d_doAutoCycleBC && flags->d_doScalarDiffusion){
        if(flags->d_autoCycleUseMinMax){
          new_dw->put(max_vartype(maxPatchConc),  lb->diffusion->rMaxConcentration);
          new_dw->put(min_vartype(minPatchConc),  lb->diffusion->rMinConcentration);
        }else{
          new_dw->put(sum_vartype(totalconc),     lb->diffusion->rTotalConcentration);
        }
      }

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
//______________________________________________________________________
//
void AMRMPM::finalParticleUpdate(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing finalParticleUpdate");

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
//______________________________________________________________________
//
void AMRMPM::addParticles(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int numLevels = level->getGrid()->numLevels();
  int levelIndex = level->getIndex();
  bool hasCoarser=false;
  if(level->hasCoarserLevel()){
    hasCoarser=true;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    printTask(patches, patch,cout_doing, "Doing addParticles");
    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );

    //Carry forward CellNAPID
    constCCVariable<int> NAPID;
    CCVariable<int> NAPID_new;

    old_dw->get(NAPID,               lb->pCellNAPIDLabel,    0,patch,d_gn,0);
    new_dw->allocateAndPut(NAPID_new,lb->pCellNAPIDLabel,    0,patch);
    NAPID_new.copyData(NAPID);

    // Mark cells where particles are refined for grid refinement
    CCVariable<double> refineCell;
    new_dw->getModifiable(refineCell, lb->MPMRefineCellLabel, 0, patch);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      ParticleVariable<Point> px;
      ParticleVariable<Matrix3> pF,pSize,pstress,pvelgrad,pscalefac;
      ParticleVariable<long64> pids;
      ParticleVariable<double> pvolume,pmass,ptemp,ptempP,pcolor,pconc,pconcpre;
      ParticleVariable<double> pESF,pD;
      ParticleVariable<Vector> pvelocity,pextforce,pdisp,pconcgrad,pArea;
      ParticleVariable<int> pref,ploc,plal,prefOld,pSplitR1R2R3;
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
      new_dw->getModifiable(ptempP,   lb->pTempPreviousLabel_preReloc, pset);
      new_dw->getModifiable(pref,     lb->pRefinedLabel_preReloc,      pset);
      new_dw->getModifiable(plal,     lb->pLastLevelLabel_preReloc,    pset);
      new_dw->getModifiable(ploc,     lb->pLocalizedMPMLabel_preReloc, pset);
      new_dw->getModifiable(pvelgrad, lb->pVelGradLabel_preReloc,      pset);
      new_dw->getModifiable(pF,  lb->pDeformationMeasureLabel_preReloc,pset);
      if (flags->d_with_color) {
        new_dw->getModifiable(pcolor,   lb->pColorLabel_preReloc,        pset);
      }
      if(flags->d_doScalarDiffusion){
        new_dw->getModifiable(pconc,    lb->diffusion->pConcentration_preReloc,pset);
        new_dw->getModifiable(pconcpre, lb->diffusion->pConcPrevious_preReloc, pset);
        new_dw->getModifiable(pconcgrad,lb->diffusion->pGradConcentration_preReloc, pset);
        new_dw->getModifiable(pESF,     lb->diffusion->pExternalScalarFlux_preReloc, pset);
        new_dw->getModifiable(pArea,    lb->diffusion->pArea_preReloc,         pset);
        new_dw->getModifiable(pD,       lb->diffusion->pDiffusivity_preReloc,  pset);
      }
      if (flags->d_useLoadCurves) {
        new_dw->getModifiable(pLoadCID, lb->pLoadCurveIDLabel_preReloc,  pset);
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
                //cout << "pSplit = " << pSplitR1R2R3[pp] << endl;
                splitCriteria  = true;
                splitForStretch = true;
                splitForAny = true;
              }
           }
         } // if this matl is in the list
       } // loop over criteria

        if((pref[pp]< levelIndex && splitCriteria && numLevels > 1 ) ||
           (pref[pp]<=levelIndex && splitCriteria && numLevels == 1)){
          pref[pp]++;
          numNewPartNeeded++;
        }
        if(pref[pp]>prefOld[pp] || splitCriteria) {
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
          splitForAny = true;
          pref[pp]++;
          numNewPartNeeded++;
        }
      }  // Loop over original particles

      int fourOrEight=pow(2,flags->d_ndim);
      if(splitForStretch){
        fourOrEight=4;
      }
      double fourthOrEighth = 1./((double) fourOrEight);
      numNewPartNeeded*=fourOrEight;

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
      ParticleVariable<double> pvoltmp, pmasstmp,ptemptmp,ptempPtmp,pESFtmp;
      ParticleVariable<double> pcolortmp, pconctmp, pconcpretmp,pDtmp;
      ParticleVariable<Vector> pveltmp,pextFtmp,pdisptmp,pconcgradtmp,pareatmp;
      ParticleVariable<int> preftmp,ploctmp,plaltmp;
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
      new_dw->allocateTemporary(ptempPtmp,pset);
      new_dw->allocateTemporary(pFtmp,    pset);
      new_dw->allocateTemporary(psizetmp, pset);
      new_dw->allocateTemporary(pareatmp, pset);
      new_dw->allocateTemporary(pdisptmp, pset);
      new_dw->allocateTemporary(pstrstmp, pset);
      if (flags->d_with_color) {
        new_dw->allocateTemporary(pcolortmp,pset);
      }
      if(flags->d_doScalarDiffusion){
        new_dw->allocateTemporary(pconctmp,     pset);
        new_dw->allocateTemporary(pconcpretmp,  pset);
        new_dw->allocateTemporary(pconcgradtmp, pset);
        new_dw->allocateTemporary(pESFtmp,      pset);
        new_dw->allocateTemporary(pDtmp,        pset);
      }
      if (flags->d_useLoadCurves) {
        new_dw->allocateTemporary(pLoadCIDtmp,  pset);
      }
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
        pextFtmp[pp] = pextforce[pp];
        ptemptmp[pp] = ptemp[pp];
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
        plaltmp[pp]  = plal[pp];
        ploctmp[pp]  = ploc[pp];
        pvgradtmp[pp]= pvelgrad[pp];
      }

      if(flags->d_doScalarDiffusion){
       for( unsigned int pp=0; pp<oldNumPar; ++pp ){
         pconctmp[pp]    = pconc[pp];
         pconcpretmp[pp] = pconcpre[pp];
         pconcgradtmp[pp]= pconcgrad[pp];
         pESFtmp[pp]     = pESF[pp];
         pareatmp[pp]    = pArea[pp];
         pDtmp[pp]       = pD[pp];
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

        // This dsize is now in terms of physical dimensions
        // The additional scaling by the grid cell size is needed
        // for determining new particle positions (below)
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

//        cout << "OPP = " << px[idx] << endl;
        int comp=0;
        int last_index=-999;
        for(int i = 0;i<fourOrEight;i++){
//          cout << "NPP = " << new_part_pos[i] << endl;
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
          int new_idx;
          if(i==0){
             new_idx=idx;
          } else {
             new_idx=oldNumPar+(fourOrEight-1)*numRefPar+i;
          }
//          cout << "new_idx = " << new_idx << endl;
          pidstmp[new_idx]    = (cellID | (long64) myCellNAPID);
          pxtmp[new_idx]      = new_part_pos[i];
          pvoltmp[new_idx]    = fourthOrEighth*pvolume[idx];
          pmasstmp[new_idx]   = fourthOrEighth*pmass[idx];
          pveltmp[new_idx]    = pvelocity[idx];
          if (flags->d_useLoadCurves) {
            pLoadCIDtmp[new_idx]  = pLoadCID[idx];
          }
          if(fourOrEight==8){
            if(flags->d_computeScaleFactor){
              pSFtmp[new_idx]     = 0.5*pscalefac[idx];
            }
            psizetmp[new_idx]   = 0.5*pSize[idx];
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
              pSFtmp[new_idx]  = dSNew;
            }
            psizetmp[new_idx]= pF[idx].Inverse()*dSNew*Matrix3(1./dx[0],0.,0.,
                                                              0.,1./dx[1],0.,
                                                              0.,0.,1./dx[2]);
           } else {
              // Divide psize by two in both x and y directions
              Matrix3 ps=pscalefac[idx];
              Matrix3 tmp(0.5*ps(0,0), 0.5*ps(0,1), 0.0,
                          0.5*ps(1,0), 0.5*ps(1,1), 0.0,
                          0.0,         0.0,         ps(2,2));
              if(flags->d_computeScaleFactor){
               pSFtmp[new_idx]     = tmp;
              }
              ps = pSize[idx];
              tmp = Matrix3(0.5*ps(0,0), 0.5*ps(0,1), 0.0,
                            0.5*ps(1,0), 0.5*ps(1,1), 0.0,
                            0.0,         0.0,         ps(2,2));
              psizetmp[new_idx]   = tmp;
           }
          } // if fourOrEight==4
          pextFtmp[new_idx]   = pextforce[idx];
          pFtmp[new_idx]      = pF[idx];
          pdisptmp[new_idx]   = pdisp[idx];
          pstrstmp[new_idx]   = pstress[idx];
          if (flags->d_with_color) {
            pcolortmp[new_idx]  = pcolor[idx];
          }
          if(flags->d_doScalarDiffusion){
            pconctmp[new_idx]     = pconc[idx];
            pconcpretmp[new_idx]  = pconcpre[idx];
            pconcgradtmp[new_idx] = pconcgrad[idx];
            pESFtmp[new_idx]      = pESF[idx];
            pDtmp[new_idx]        = pD[idx];
            if((fabs(pArea[idx].x()) > 0.0 && fabs(pArea[idx].y()) > 0.0) || 
               (fabs(pArea[idx].x()) > 0.0 && fabs(pArea[idx].z()) > 0.0) ||
               (fabs(pArea[idx].y()) > 0.0 && fabs(pArea[idx].z()) > 0.0) ||
               (fabs(pArea[idx][comp])<1.e-12)) {
              pareatmp[new_idx]     = fourthOrEighth*pArea[idx];
            } else {
              if(i==0){
                pareatmp[new_idx]     = pArea[idx];
              } else {
                if(pxtmp[new_idx].asVector().length2() >
                   pxtmp[last_index].asVector().length2()){
                  pareatmp[last_index]     = 0.0;
                  pareatmp[new_idx]        = pArea[idx];
                  pLoadCIDtmp[last_index]  = IntVector(0,0,0);
                  pLoadCIDtmp[new_idx]     = pLoadCID[idx];
                } else{
                  pareatmp[new_idx]        = 0.0;
                  pLoadCIDtmp[new_idx]     = IntVector(0,0,0);
                } // if pxtmp
              } // if i==0
            } // if pArea
          } // if diffusion
          ptemptmp[new_idx]   = ptemp[idx];
          ptempPtmp[new_idx]  = ptempP[idx];
          preftmp[new_idx]    = pref[idx];
          plaltmp[new_idx]    = plal[idx];
          ploctmp[new_idx]    = ploc[idx];
          pvgradtmp[new_idx]  = pvelgrad[idx];
          NAPID_new[c_orig]++;
          last_index=new_idx;
        }
        numRefPar++;
       }  // if this particle flagged for refinement
      } // for old particles
      } // if any particles flagged for refinement

      cm->splitCMSpecificParticleData(patch, dwi, fourOrEight, prefOld, pref,
                                      oldNumPar, numNewPartNeeded,
                                      old_dw, new_dw);
      if(flags->d_doScalarDiffusion){
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
        new_dw->put(pSFtmp,   lb->pScaleFactorLabel_preReloc,          true);
      }
      new_dw->put(pextFtmp, lb->pExtForceLabel_preReloc,             true);
      new_dw->put(pmasstmp, lb->pMassLabel_preReloc,                 true);
      new_dw->put(ptemptmp, lb->pTemperatureLabel_preReloc,          true);
      new_dw->put(ptempPtmp,lb->pTempPreviousLabel_preReloc,         true);
      new_dw->put(psizetmp, lb->pSizeLabel_preReloc,                 true);
      new_dw->put(pdisptmp, lb->pDispLabel_preReloc,                 true);
      new_dw->put(pstrstmp, lb->pStressLabel_preReloc,               true);
      if (flags->d_with_color) {
        new_dw->put(pcolortmp,lb->pColorLabel_preReloc,              true);
      }
      if(flags->d_doScalarDiffusion){
        new_dw->put(pconctmp,     lb->diffusion->pConcentration_preReloc,  true);
        new_dw->put(pconcpretmp,  lb->diffusion->pConcPrevious_preReloc,   true);
        new_dw->put(pconcgradtmp, lb->diffusion->pGradConcentration_preReloc,   true);
        new_dw->put(pESFtmp,      lb->diffusion->pExternalScalarFlux_preReloc, true);
        new_dw->put(pareatmp,     lb->diffusion->pArea_preReloc,           true);
        new_dw->put(pDtmp,        lb->diffusion->pDiffusivity_preReloc,    true);
      }
      if (flags->d_useLoadCurves) {
        new_dw->put(pLoadCIDtmp,lb->pLoadCurveIDLabel_preReloc,      true);
      }
      new_dw->put(pFtmp,    lb->pDeformationMeasureLabel_preReloc,   true);
      new_dw->put(preftmp,  lb->pRefinedLabel_preReloc,              true);
      new_dw->put(plaltmp,  lb->pLastLevelLabel_preReloc,            true);
      new_dw->put(ploctmp,  lb->pLocalizedMPMLabel_preReloc,         true);
      new_dw->put(pvgradtmp,lb->pVelGradLabel_preReloc,              true);
    }  // for matls
  }    // for patches
}
//______________________________________________________________________
//
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
    constCCVariable<double> refineCell;
    new_dw->get(refineCell, lb->MPMRefineCellLabel, 0, patch, d_gn, 0);

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

    constCCVariable<double> refineCell;
    CCVariable<int>      refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->getModifiable(refineFlag, m_regridder->getRefineFlagLabel(),
                                                                  0, patch);
    new_dw->get(refinePatchFlag, m_regridder->getRefinePatchFlagLabel(),
                                                                  0, patch);
    new_dw->get(refineCell,     lb->MPMRefineCellLabel, 0, patch, d_gn, 0);

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

#if 0  // Alternate method of initializing refined regions.  Inactive, not
       // necessarily compatible with, say, defining levels in Grid section of
       // input.
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

    unsigned int numMPMMatls=m_materialManager->getNumMatls( "MPM" );
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
        ParticleVariable<Vector> pvelocity, pexternalforce, pdisp,pConcGrad;
        ParticleVariable<Matrix3> psize;
        ParticleVariable<Vector>  parea;
        ParticleVariable<double> pTempPrev,pColor,pConc,pConcPrev,pExtScalFlux;
        ParticleVariable<int>    pLastLevel,pRefined;
        ParticleVariable<IntVector> pLoadCurve;
        ParticleVariable<long64> pID;
        ParticleVariable<Matrix3> pdeform, pstress, pVelGrad;
        
        new_dw->allocateAndPut(px,             lb->pXLabel,             pset);
        new_dw->allocateAndPut(pmass,          lb->pMassLabel,          pset);
        new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        pset);
        new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      pset);
        new_dw->allocateAndPut(pTemperature,   lb->pTemperatureLabel,   pset);
        new_dw->allocateAndPut(pTempPrev,      lb->pTempPreviousLabel,  pset);
        new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, pset);
        new_dw->allocateAndPut(pExtScalFlux,   lb->diffusion->pExternalScalarFlux_preReloc,
                                                                        pset);
        new_dw->allocateAndPut(pID,            lb->pParticleIDLabel,    pset);
        new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          pset);
        new_dw->allocateAndPut(pLastLevel,     lb->pLastLevelLabel,     pset);
        new_dw->allocateAndPut(pRefined,       lb->pRefinedLabel,       pset);
        new_dw->allocateAndPut(pVelGrad,       lb->pVelGradLabel,       pset);
        if (flags->d_useLoadCurves){
          new_dw->allocateAndPut(pLoadCurve,   lb->pLoadCurveIDLabel,   pset);
        }
        if (flags->d_with_color) {
          new_dw->allocateAndPut(pColor,       lb->pColorLabel,         pset);
        }
        if(flags->d_doScalarDiffusion){
          new_dw->allocateAndPut(pConc,        lb->diffusion->pConcentration, pset);
          new_dw->allocateAndPut(pConcPrev,    lb->diffusion->pConcPrevious,  pset);
          new_dw->allocateAndPut(pConcGrad,    lb->diffusion->pGradConcentration,  pset);
          new_dw->allocateAndPut(parea,        lb->diffusion->pArea,          pset);
        }
        new_dw->allocateAndPut(psize,          lb->pSizeLabel,          pset);

        mpm_matl->getConstitutiveModel()->initializeCMData(patch,
                                                           mpm_matl,new_dw);
          //initialize Damage/erosion model labels
        mpm_matl->getDamageModel()->initializeLabels( patch, mpm_matl, new_dw );
        
        mpm_matl->getErosionModel()->initializeLabels( patch, mpm_matl, new_dw );
                                                           
        
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
  sched->addTask(t, patches, m_materialManager->allMaterials( "MPM" ));
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
  unsigned int numMPMMatls = m_materialManager->getNumMatls( "MPM" );
  
//  const Level* level = getLevel(patches);
//  cout << "Level " << level->getIndex() << " has " << level->numPatches() << " patches" << endl;
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    
    printTask(patches,patch,cout_doing,"Doing AMRMPM::countParticles");
    
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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

  #define allPatches 0
  Task::MaterialDomainSpec  ND  = Task::NormalDomain;

  Task* t = scinew Task("AMRMPM::debug_CFI",
                   this,&AMRMPM::debug_CFI);
  printSchedule(patches,cout_doing,"AMRMPM::scheduleDebug_CFI");
  if(level->hasFinerLevel()){ 
    t->requires(Task::NewDW, lb->gZOILabel, allPatches, Task::FineLevel,d_one_matl, ND, d_gn, 0);
  }

  t->requires(Task::OldDW, lb->pXLabel,                  d_gn,0);
  t->requires(Task::NewDW, lb->pCurSizeLabel,            d_gn,0);
  
  t->computes(lb->pColorLabel_preReloc);

  sched->addTask(t, patches, matls);
}
//
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
    ParticleVariable<double>  pColor;
    
    old_dw->get(px,                   lb->pXLabel,                  pset);
    new_dw->get(psize,                lb->pCurSizeLabel,            pset);
    new_dw->allocateAndPut(pColor,    lb->pColorLabel_preReloc,     pset);
    
    ParticleInterpolator* interpolatorCoarse = flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolatorCoarse->size());
    vector<double> S(interpolatorCoarse->size());

    for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pColor[idx] = 0;
      
      int NN = interpolatorCoarse->findCellAndWeights(px[idx],ni,S,psize[idx]);
      
      for(int k = 0; k < NN; k++) {
        pColor[idx] += S[k];
      }
    }  

    //__________________________________
    //  Mark the particles that are accessed at the CFI.
    if(level->hasFinerLevel()){  

      // find the fine patches over the coarse patch.  Add a single layer of cells
      // so you will get the required patches when coarse patch and fine patch boundaries coincide.
      Level::selectType finePatches;
      patch->getOtherLevelPatches55902(1, finePatches, 1);

      const Level* fineLevel = level->getFinerLevel().get_rep();
      IntVector refineRatio(fineLevel->getRefinementRatio());

      for(size_t fp=0; fp<finePatches.size(); fp++){
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
//    This removes duplicate entries in the array
void AMRMPM::removeDuplicates( Level::selectType& array)
{
  int length = array.size();
  if ( length <= 1 ){
    return;
  }
  
  int newLength = 1;        // new length of modified array
  int i, j;
  
  for(i=1; i< length; i++){
    for(j=0; j< newLength; j++){
      if( array[i] == array[j] ){
        break;
      }
    }
    // if none of the values in array[0..j] == array[i],
    // then copy the current value to a new position in array

    if (j == newLength ){
      array[newLength++] = array[i];
    }
  }
  array.resize(newLength);
}


//______________________________________________________________________
//  Returns the fine and coarse level patches that have coarse fine interfaces
void AMRMPM::coarseLevelCFI_Patches(const PatchSubset* coarsePatches,
                                    Level::selectType& CFI_coarsePatches,
                                    Level::selectType& CFI_finePatches )
{
  const Level* coarseLevel = getLevel(coarsePatches);
  if( !coarseLevel->hasFinerLevel()) {
    return;
  }

  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    bool addMe = false;

    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(size_t fp=0;fp<finePatches.size();fp++){  
      const Patch* finePatch = finePatches[fp];
      
      if(finePatch->hasCoarseFaces() ){
        addMe = true;
        CFI_finePatches.push_back( finePatch );
      }
    }
    
    if( addMe ){  // only add once
      CFI_coarsePatches.push_back( coarsePatch );
    }
  }

  // remove duplicate patches
  removeDuplicates( CFI_coarsePatches );
  removeDuplicates( CFI_finePatches );

}

//______________________________________________________________________
//  Returns the fine patches that have a CFI and all of the underlying
//  coarse patches beneath those patches.  We don't know which of the coarse
//  patches are beneath the fine patch with the CFI.
// This takes in fine level patches
void AMRMPM::fineLevelCFI_Patches(const PatchSubset* finePatches,
                                  Level::selectType& coarsePatches,
                                  Level::selectType& CFI_finePatches )
{
  const Level* fineLevel = getLevel(finePatches);
      
  if( !fineLevel->hasCoarserLevel()) {
    return;
  }
  
  for(int p=0;p<finePatches->size();p++){
    const Patch* finePatch = finePatches->get(p);
    
    if(finePatch->hasCoarseFaces() ){
      CFI_finePatches.push_back( finePatch );
      
      // need to use the Node Based version of getOtherLevel
      finePatch->getOtherLevelPatchesNB(-1,coarsePatches,0);  
    } 
  }
  removeDuplicates( coarsePatches );
  removeDuplicates( CFI_finePatches );
}


#if 0  // May need to reactivate for GIMP
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

    Task::MaterialDomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    t->requires(Task::OldDW, lb->delTLabel );
    
    t->requires(Task::OldDW, lb->pXLabel, gn);
    t->requires(Task::NewDW, lb->gVelocityStarLabel, allPatches, Task::FineLevel,allMatls,   ND, d_gn,0);
    t->requires(Task::NewDW, lb->gAccelerationLabel, allPatches, Task::FineLevel,allMatls,   ND, d_gn,0);
    t->requires(Task::NewDW, lb->gZOILabel,          allPatches, Task::FineLevel,d_one_matl, ND, d_gn,0);
    
    t->modifies(lb->pXLabel_preReloc);
    t->modifies(lb->pDispLabel_preReloc);
    t->modifies(lb->pVelocityLabel_preReloc);

    sched->addTask(t, patches, matls);
  }
}
#endif

#if 0  // May need to reactivate for GIMP
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
  old_dw->get(delT, lb->delTLabel, coarseLevel );
  
  //__________________________________
  //Loop over the coarse level patches
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    printTask(coarsePatches,coarsePatch,cout_doing,"AMRMPM::interpolateToParticlesAndUpdate_CFI");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );
        
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

        for(unsigned int m = 0; m < numMatls; m++){
          MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM",  m );
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
            pxnew_coarse[idx]         += vel*delT;  
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

void AMRMPM::scheduleConcInterpolated(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;
  printSchedule(patches,cout_doing,"MPM::scheduleConcInterpolated");

  d_sdInterfaceModel->addComputesAndRequiresInterpolated(sched, patches, matls);
}

//______________________________________________________________________
//
void AMRMPM::scheduleComputeFlux(SchedulerP& sched, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls)
{
    if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                           getLevel(patches)->getGrid()->numLevels())){
      return;
    }

    printSchedule(patches,cout_doing,"AMRMPM::scheduleComputeFlux");

    Task* t = scinew Task("AMRMPM::computeFlux",
                     this,&AMRMPM::computeFlux);

    unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->scheduleComputeFlux(t, mpm_matl, patches);
    }

    sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::computeFlux(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing AMRMPM::computeFlux");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->computeFlux(patch, mpm_matl, old_dw, new_dw);
    }
  }
}
//______________________________________________________________________
//
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
  unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
  for(unsigned int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->scheduleComputeDivergence(t, mpm_matl, patches);
  }
#endif

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void AMRMPM::computeDivergence(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing, "Doing AMRMPM::computeDivergence");

    unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

    for(unsigned int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->computeDivergence(patch, mpm_matl, old_dw, new_dw);
    }
  }
}
//______________________________________________________________________
//
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

    unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
    for(unsigned int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      sdm->scheduleComputeDivergence_CFI(t, mpm_matl, patches);
    }

    sched->addTask(t, patches, matls);
  }
}
//______________________________________________________________________
//
void AMRMPM::computeDivergence_CFI(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  unsigned int numMatls = m_materialManager->getNumMatls( "MPM" );

  for(unsigned int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->computeDivergence_CFI(patches, mpm_matl, old_dw, new_dw);
  }
}

void AMRMPM::scheduleDiffusionInterfaceDiv(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels()))
      return;
    printSchedule(patches,cout_doing,"AMRMPM::scheduleDiffusionInterfaceDiv");

    d_sdInterfaceModel->addComputesAndRequiresDivergence(sched, patches, matls);
}

/* This set of functions used for the scalar flux boundary conditions have been
 *  movedinto the FluxBCModel */
