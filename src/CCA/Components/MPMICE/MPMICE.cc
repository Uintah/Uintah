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

// MPMICE.cc
#include <CCA/Components/ICE/AMRICE.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPMICE/MPMICE.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/ShellMPM.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/ModelMaker.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/Utils.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/MiscMath.h>


#include <Core/Containers/StaticArray.h>
#include <cfloat>
#include <cstdio>
#include <Core/Util/DebugStream.h>

#include <iomanip>
#include <errno.h>
#include <fenv.h>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT".....
//  MPMICE_NORMAL_COUT:  dumps out during problemSetup 
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF


static DebugStream cout_norm("MPMICE_NORMAL_COUT", false);  
static DebugStream cout_doing("MPMICE_DOING_COUT", false);
static DebugStream ds_EqPress("DBG_EqPress",false);

MPMICE::MPMICE(const ProcessorGroup* myworld, 
               MPMType mpmtype, const bool doAMR)
  : UintahParallelComponent(myworld)
{
  MIlb = scinew MPMICELabel();
 
  d_rigidMPM = false;
  d_doAMR = doAMR;
  d_testForNegTemps_mpm = true;
  d_recompile = false;
  
  switch(mpmtype) {
  case RIGID_MPMICE:
    d_mpm = scinew RigidMPM(myworld);
    d_rigidMPM = true;
    break;
  case SHELL_MPMICE:
    d_mpm = scinew ShellMPM(myworld);
    break;
  default:
    d_mpm = scinew SerialMPM(myworld);
  }

  // Don't do AMRICE with MPMICE for now...
  if (d_doAMR) {
    d_ice  = scinew AMRICE(myworld);
  }
  else {
    d_ice  = scinew ICE(myworld, false);
  }

  Ilb=d_ice->lb;
  Mlb=d_mpm->lb;

  d_SMALL_NUM = d_ice->d_SMALL_NUM;
  d_TINY_RHO  = 1.e-12;  // Note, within MPMICE, d_TINY_RHO is only applied
                         // to MPM materials, for which its value is hardcoded,
                         // unlike the situation for ice materials

  d_switchCriteria = 0;

  // Turn off all the debuging switches
  switchDebug_InterpolateNCToCC_0 = false;
  switchDebug_InterpolateCCToNC   = false;
  switchDebug_InterpolatePAndGradP= false;
}

MPMICE::~MPMICE()
{
  delete MIlb;
  delete d_mpm;
  delete d_ice;
  
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      delete *iter;
    }
  }
  
}

//__________________________________
//    For restarting timesteps
bool MPMICE::restartableTimesteps()
{
  return true;
}

double MPMICE::recomputeTimestep(double current_dt)
{
  return current_dt/2;
} 
//______________________________________________________________________
//
void MPMICE::problemSetup(const ProblemSpecP& prob_spec, 
                          const ProblemSpecP& restart_prob_spec, 
                          GridP& grid, SimulationStateP& sharedState)
{
  cout_doing << "Doing MPMICE::problemSetup " << endl;
  d_sharedState = sharedState;
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));

  //__________________________________
  //  M P M
  d_mpm->setWithICE();
  d_ice->setMPMICELabel(MIlb);
  d_ice->setWithMPM();
  if(d_rigidMPM){
   d_ice->setWithRigidMPM();
  }
  d_mpm->attachPort("output",dataArchiver);
  d_mpm->attachPort("scheduler",sched);
  d_mpm->problemSetup(prob_spec, restart_prob_spec,grid, d_sharedState);
  d_8or27 = d_mpm->flags->d_8or27; 
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }

  d_switchCriteria = dynamic_cast<SwitchingCriteria*>
    (getPort("switch_criteria"));
  
  if (d_switchCriteria) {
    d_switchCriteria->problemSetup(prob_spec,restart_prob_spec,d_sharedState);
  }

  //__________________________________
  //  I C E
  if(!dataArchiver){
    throw InternalError("MPMICE needs a dataArchiver component to work", __FILE__, __LINE__);
  }
  d_ice->attachPort("output", dataArchiver);
  d_ice->attachPort("scheduler", sched);
  
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver){
    throw InternalError("ICE needs a solver component to work", __FILE__, __LINE__);
  }
  d_ice->attachPort("solver", solver);
  
  ModelMaker* models = dynamic_cast<ModelMaker*>(getPort("modelmaker"));

  if(models){  // of there are models then push the port down to ICE
    d_ice->attachPort("modelmaker",models);
  }
  
  d_ice->problemSetup(prob_spec, restart_prob_spec,grid, d_sharedState);

  if(models){  // some models may need to have access to MPMLabels
    for(vector<ModelInterface*>::iterator iter = d_ice->d_models.begin();
       iter != d_ice->d_models.end(); iter++){
      (*iter)->setMPMLabel(Mlb);
    }
  }
  
  
  //__________________________________
  //  M P M I C E
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {   
    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
        child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"]      == "switchDebug_InterpolateNCToCC_0")
        switchDebug_InterpolateNCToCC_0 = true;
      else if (debug_attr["label"] == "switchDebug_InterpolateCCToNC")
        switchDebug_InterpolateCCToNC   = true;
      else if (debug_attr["label"] == "switchDebug_InterpolatePAndGradP")
        switchDebug_InterpolatePAndGradP   = true;       
    }
  }
  
  ProblemSpecP mpm_ps = 0;
  mpm_ps = prob_spec->findBlock("MPM");
  
  if(!mpm_ps){
    mpm_ps = restart_prob_spec->findBlock("MPM");
  }
  mpm_ps->get("testForNegTemps_mpm",d_testForNegTemps_mpm);
  
  //__________________________________
  //  bulletproofing
  if(d_doAMR && !d_sharedState->isLockstepAMR()){
    ostringstream msg;
    msg << "\n ERROR: You must add \n"
        << " <useLockStep> true </useLockStep> \n"
        << " inside of the <AMR> section for MPMICE and AMR. \n"; 
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
    
  if (cout_norm.active()) {
    cout_norm << "Done with problemSetup \t\t\t MPMICE" <<endl;
    cout_norm << "--------------------------------\n"<<endl;
  }
  
  //__________________________________
  //  Set up data analysis modules
  d_analysisModules = AnalysisModuleFactory::create(prob_spec, sharedState, dataArchiver);

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->problemSetup(prob_spec, grid, sharedState);
    }
  }  
}

//______________________________________________________________________
//
void MPMICE::outputProblemSpec(ProblemSpecP& root_ps)
{
  d_mpm->outputProblemSpec(root_ps);
  d_ice->outputProblemSpec(root_ps);
  
  // Global flags required by mpmice
  ProblemSpecP mpm_ps = root_ps->findBlock("MPM");
  mpm_ps->appendElement("testForNegTemps_mpm", d_testForNegTemps_mpm);

}

//______________________________________________________________________
//
void MPMICE::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  printSchedule(level,cout_doing,"MPMICE::scheduleInitialize");

  d_mpm->scheduleInitialize(level, sched);
  d_ice->scheduleInitialize(level, sched);

  //__________________________________
  //  What isn't initialized in either ice or mpm
  Task* t = scinew Task("MPMICE::actuallyInitialize",
                  this, &MPMICE::actuallyInitialize);
                  
  // Get the material subsets
  const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();
  const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();

  // These values are calculated for ICE materials in d_ice->actuallyInitialize(...)
  //  so they are only needed for MPM
  t->computes(MIlb->vel_CCLabel,       mpm_matls);
  t->computes(Ilb->rho_CCLabel,        mpm_matls); 
  t->computes(Ilb->temp_CCLabel,       mpm_matls);
  t->computes(Ilb->sp_vol_CCLabel,     mpm_matls);
  t->computes(Ilb->speedSound_CCLabel, mpm_matls); 
  t->computes(Mlb->heatRate_CCLabel,   mpm_matls);

  // This is compute in d_ice->actuallyInitalize(...), and it is needed in 
  //  MPMICE's actuallyInitialize()
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, ice_matls, Ghost::None, 0);

  if (d_switchCriteria) {
    d_switchCriteria->scheduleInitialize(level,sched);
  }
  
  //__________________________________
  // dataAnalysis 
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }
    
  sched->addTask(t, level->eachPatch(), d_sharedState->allMaterials());
}

//______________________________________________________________________
//
void MPMICE::restartInitialize()
{
  if (cout_doing.active())
    cout_doing <<"Doing restartInitialize \t\t\t MPMICE" << endl;

  d_mpm->restartInitialize();
  d_ice->restartInitialize();
  
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
//
void MPMICE::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  // Schedule computing the ICE stable timestep
  d_ice->scheduleComputeStableTimestep(level, sched);
  // MPM stable timestep is a by product of the CM
}

//______________________________________________________________________
//
void
MPMICE::scheduleTimeAdvance(const LevelP& inlevel, SchedulerP& sched)
{
  MALLOC_TRACE_TAG_SCOPE("MPMICE::scheduleTimeAdvance()");
  // Only do scheduling on level 0 for lockstep AMR
  if(inlevel->getIndex() > 0 && d_sharedState->isLockstepAMR())
    return;

  // If we have a finer level, then assume that we are doing multilevel MPMICE
  // Otherwise, it is plain-ole MPMICE
  do_mlmpmice = false;
  if(inlevel->hasFinerLevel()){
    do_mlmpmice = true;
  }
  const LevelP& mpm_level = do_mlmpmice? inlevel->getGrid()->getLevel(inlevel->getGrid()->numLevels()-1) : inlevel;

  const PatchSet* mpm_patches = mpm_level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* press_matl   = d_ice->d_press_matl;
  MaterialSubset* one_matl     = d_ice->d_press_matl;

  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();
  cout_doing << "---------------------------------------------------------Level ";
  if(do_mlmpmice){
    cout_doing << inlevel->getIndex() << " (ICE) " << mpm_level->getIndex() << " (MPM)"<< endl;;
  } else {
    cout_doing << inlevel->getIndex()<< endl;
  }

 //__________________________________
 // Scheduling
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
    d_ice->scheduleComputeThermoTransportProperties(sched, ice_level,ice_matls);
    
    d_ice->scheduleMaxMach_on_Lodi_BC_Faces(        sched, ice_level,ice_matls);
  }
   
  d_mpm->scheduleApplyExternalLoads(          sched, mpm_patches, mpm_matls);
  d_mpm->scheduleInterpolateParticlesToGrid(  sched, mpm_patches, mpm_matls);
  d_mpm->scheduleComputeHeatExchange(         sched, mpm_patches, mpm_matls);

  d_mpm->scheduleExMomInterpolated(           sched, mpm_patches, mpm_matls);

  // schedule the interpolation of mass and volume to the cell centers
  scheduleInterpolateNCToCC_0(                sched, mpm_patches, one_matl, 
                                                                  mpm_matls);
  // do coarsens in reverse order, and before the other tasks
  if(do_mlmpmice){
    for (int l = inlevel->getGrid()->numLevels() - 2; l >= 0; l--) {
      const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
      const PatchSet* ice_patches = ice_level->eachPatch();

      scheduleCoarsenCC_0(                      sched, ice_patches, mpm_matls);
      scheduleCoarsenNCMass(                    sched, ice_patches, mpm_matls);
    }
  }

  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
    const PatchSet* ice_patches = ice_level->eachPatch();
    
    scheduleComputePressure(                  sched, ice_patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
    
  
    d_ice->scheduleComputeTempFC(             sched, ice_patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
    d_ice->scheduleComputeModelSources(       sched, ice_level,   all_matls);

    d_ice->scheduleUpdateVolumeFraction(      sched, ice_level,   press_matl,
                                                                  all_matls);
  
    d_ice->scheduleComputeVel_FC(             sched, ice_patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl, 
                                                                  all_matls);
                                                               
    d_ice->scheduleAddExchangeContributionToFCVel(
                                              sched, ice_patches, ice_matls_sub,
                                                                  all_matls,
                                                                  false);  
  }
  if(d_ice->d_impICE) {        //  I M P L I C I T, won't work with AMR yet
    // we should use the AMR multi-level pressure solve
    for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
      const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
      const PatchSet* ice_patches = ice_level->eachPatch();

      sched->overrideVariableBehavior("hypre_solver_label",false,
                                      false,false,true,true);

      d_ice->scheduleSetupRHS(                sched, ice_patches, one_matl, 
                                                                  all_matls,
                                                                  false,
                                                                  "computes");
      d_ice->scheduleCompute_maxRHS(          sched, ice_level,    one_matl,
                                                                   all_matls);
                                                                  
      d_ice->scheduleImplicitPressureSolve(   sched, ice_level,   ice_patches,
                                                                  one_matl, 
                                                                  press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
                                                           
      d_ice->scheduleComputeDel_P(            sched, ice_level,   ice_patches, 
                                                                  one_matl, 
                                                                  press_matl,
                                                                  all_matls);
    }
  }                           //  IMPLICIT AND EXPLICIT

                                                                  
  if(!(d_ice->d_impICE)){       //  E X P L I C I T 
    for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
      const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
      const PatchSet* ice_patches = ice_level->eachPatch();

      d_ice->scheduleComputeDelPressAndUpdatePressCC(
                                              sched, ice_patches, press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
    }
  } 
  
  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
    const PatchSet* ice_patches = ice_level->eachPatch();

    d_ice->scheduleComputePressFC(            sched, ice_patches, press_matl,
                                                                    all_matls);
    d_ice->scheduleAccumulateMomentumSourceSinks(
                                              sched, ice_patches, press_matl,
                                                                  ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
                                                                  
    d_ice->scheduleAccumulateEnergySourceSinks(sched, ice_patches,ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
  }

  if(!d_rigidMPM){
//    if(do_mlmpmice){
//      scheduleRefinePressCC(                  sched, mpm_patches, press_matl,
//                                                                  mpm_matls);
//    }
      
    scheduleInterpolatePressCCToPressNC(      sched, mpm_patches, press_matl,
                                                                  mpm_matls);

    scheduleInterpolatePAndGradP(             sched, mpm_patches, press_matl,
                                                                  one_matl,
                                                                  mpm_matls_sub,
                                                                  mpm_matls);
  }
   
  d_mpm->scheduleComputeInternalForce(        sched, mpm_patches, mpm_matls);
  d_mpm->scheduleComputeInternalHeatRate(     sched, mpm_patches, mpm_matls);
  d_mpm->scheduleComputeNodalHeatFlux(        sched, mpm_patches, mpm_matls);
  d_mpm->scheduleSolveHeatEquations(          sched, mpm_patches, mpm_matls);
  d_mpm->scheduleComputeAndIntegrateAcceleration(sched, mpm_patches, mpm_matls);
  d_mpm->scheduleIntegrateTemperatureRate(    sched, mpm_patches, mpm_matls);
  
  scheduleComputeLagrangianValuesMPM(         sched, mpm_patches, one_matl,
                                                                  mpm_matls); 

  // do coarsens in reverse order, and before the other tasks
  if(do_mlmpmice){
    for (int l = inlevel->getGrid()->numLevels() - 2; l >= 0; l--) {
      const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
      const PatchSet* ice_patches = ice_level->eachPatch();
      scheduleCoarsenLagrangianValuesMPM(     sched, ice_patches, mpm_matls);
    }
  }

  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
    const PatchSet* ice_patches = ice_level->eachPatch();

    d_ice->scheduleComputeLagrangianValues(   sched, ice_patches, ice_matls);

    d_ice->scheduleAddExchangeToMomentumAndEnergy(
                                              sched, ice_patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls); 

    d_ice->scheduleComputeLagrangianSpecificVolume(
                                              sched, ice_patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
                                                                  
    d_ice->scheduleComputeLagrangian_Transported_Vars(
                                              sched, ice_patches, ice_matls);

  }

  scheduleComputeCCVelAndTempRates(           sched, mpm_patches, mpm_matls);

//  if(do_mlmpmice){
//    scheduleRefineCC(                         sched, mpm_patches, mpm_matls);
//  }

  scheduleInterpolateCCToNC(                  sched, mpm_patches, mpm_matls);

  d_mpm->scheduleExMomIntegrated(             sched, mpm_patches, mpm_matls);
  d_mpm->scheduleSetGridBoundaryConditions(   sched, mpm_patches, mpm_matls);
  d_mpm->scheduleComputeStressTensor(         sched, mpm_patches, mpm_matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate(sched, mpm_patches, mpm_matls);
  //d_mpm->scheduleApplyExternalLoads(          sched, mpm_patches, mpm_matls);

  for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
    const LevelP& ice_level = inlevel->getGrid()->getLevel(l);
    const PatchSet* ice_patches = ice_level->eachPatch();
                                   
    d_ice->scheduleAdvectAndAdvanceInTime(   sched, ice_patches,ice_matls_sub,
                                                                ice_matls);
                                                                
    d_ice->scheduleConservedtoPrimitive_Vars(sched, ice_patches,ice_matls_sub,
                                                    ice_matls,"afterAdvection");
  }
  if(d_ice->d_canAddICEMaterial){
     for (int l = 0; l < inlevel->getGrid()->numLevels(); l++) {
       const LevelP& ice_level = inlevel->getGrid()->getLevel(l);

       //  This checks to see if the model on THIS patch says that it's
       //  time to add a new material
       d_ice->scheduleCheckNeedAddMaterial(  sched, ice_level,   all_matls);

       //  This one checks to see if the model on ANY patch says that it's
       //  time to add a new material
       d_ice->scheduleSetNeedAddMaterialFlag(sched, ice_level,   all_matls);
     }
   }

} // end scheduleTimeAdvance()


/* _____________________________________________________________________
MPMICE::scheduleFinalizeTimestep--
This task called at the very bottom of the timestep,
after scheduleTimeAdvance and the scheduleCoarsen.  

This is scheduled on every level.
_____________________________________________________________________*/
void
MPMICE::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{
  cout_doing << "----------------------------"<<endl;
  cout_doing << d_myworld->myrank() << " MPMICE::scheduleFinalizeTimestep\t\t\t\tL-" <<level->getIndex()<< endl;
  
  const PatchSet* ice_patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
                                                          
  d_ice->scheduleConservedtoPrimitive_Vars(sched, ice_patches,ice_matls_sub,
                                                              ice_matls,
                                                              "finalizeTimestep");

  d_ice->scheduleTestConservation(        sched, ice_patches, ice_matls_sub,
                                                             all_matls);
  
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis_preReloc( sched, level);
    }
  }
                                                              
  // only do on finest level until we get AMR MPM
  if (level->getIndex() == level->getGrid()->numLevels()-1)
    sched->scheduleParticleRelocation(level,
                                  Mlb->pXLabel_preReloc, 
                                  d_sharedState->d_particleState_preReloc,
                                  Mlb->pXLabel, d_sharedState->d_particleState,
                                  Mlb->pParticleIDLabel, mpm_matls);
  
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
  cout_doing << "---------------------------------------------------------"<<endl;
}

//______________________________________________________________________
//
void MPMICE::scheduleRefinePressCC(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSubset* press_matl,
                                   const MaterialSet* matls)
{
  printSchedule(patches,cout_doing,"MPMICE::scheduleRefinePressCC");
    
  MaterialSet* press_matls = scinew MaterialSet();
  press_matls->add(0);
  press_matls->addReference();

  scheduleRefineVariableCC(sched,patches, press_matls,Ilb->press_CCLabel);
  if(press_matls->removeReference())
    delete press_matls;
}

//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePressCCToPressNC(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* press_matl,
                                           const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();
  if(!d_mpm->flags->doMPMOnLevel(L_indx,level->getGrid()->numLevels()))
    return;
    
  printSchedule(patches,cout_doing, "MPMICE::scheduleInterpolatePressCCToPressNC"); 
    
  Task* t=scinew Task("MPMICE::interpolatePressCCToPressNC",
                    this, &MPMICE::interpolatePressCCToPressNC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW,Ilb->press_CCLabel, press_matl, gac, 1);
  t->computes(MIlb->press_NCLabel, press_matl);
  
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePAndGradP(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSubset* press_matl,
                                     const MaterialSubset* /*one_matl*/,
                                     const MaterialSubset* mpm_matl,
                                     const MaterialSet* all_matls)
{
  if(!d_mpm->flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                                 getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPMICE::scheduleInterpolatePAndGradP");
 
  Task* t=scinew Task("MPMICE::interpolatePAndGradP",
                      this, &MPMICE::interpolatePAndGradP);
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  t->requires(Task::NewDW, MIlb->press_NCLabel,       press_matl,gac, NGN);
  t->requires(Task::NewDW, MIlb->cMassLabel,          mpm_matl,  gac, 1);
  t->requires(Task::OldDW, Mlb->pXLabel,              mpm_matl,  Ghost::None);
  t->requires(Task::OldDW, Mlb->pSizeLabel,           mpm_matl,  Ghost::None);
  t->requires(Task::OldDW, Mlb->pDeformationMeasureLabel, mpm_matl, Ghost::None);
   
  t->computes(Mlb->pPressureLabel,   mpm_matl);
  sched->addTask(t, patches, all_matls);
}



//______________________________________________________________________
//
void MPMICE::scheduleInterpolateNCToCC_0(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* one_matl,
                                    const MaterialSet* mpm_matls)
{
  if(d_mpm->flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                                getLevel(patches)->getGrid()->numLevels())){
                                
    printSchedule(patches,cout_doing, "MPMICE::scheduleInterpolateNCToCC_0");
 
    /* interpolateNCToCC */
    Task* t=scinew Task("MPMICE::interpolateNCToCC_0",
                        this, &MPMICE::interpolateNCToCC_0);
    const MaterialSubset* mss = mpm_matls->getUnion();
    t->requires(Task::NewDW, Mlb->gMassLabel,       Ghost::AroundCells, 1);
    t->requires(Task::NewDW, Mlb->gVolumeLabel,     Ghost::AroundCells, 1);
    t->requires(Task::NewDW, Mlb->gVelocityBCLabel, Ghost::AroundCells, 1); 
    t->requires(Task::NewDW, Mlb->gTemperatureLabel,Ghost::AroundCells, 1);
    t->requires(Task::NewDW, Mlb->gSp_volLabel,     Ghost::AroundCells, 1);
    t->requires(Task::OldDW, Mlb->NC_CCweightLabel,one_matl,
                                                    Ghost::AroundCells, 1);
    t->requires(Task::OldDW, Ilb->sp_vol_CCLabel,   Ghost::None, 0); 
    t->requires(Task::OldDW, MIlb->temp_CCLabel,    Ghost::None, 0);

    t->computes(MIlb->cMassLabel);
    t->computes(MIlb->vel_CCLabel);
    t->computes(MIlb->temp_CCLabel);
    t->computes(Ilb->sp_vol_CCLabel, mss);
    t->computes(Ilb->rho_CCLabel, mss); 
   
    sched->addTask(t, patches, mpm_matls);
  }
}

//______________________________________________________________________
//
void MPMICE::scheduleCoarsenCC_0(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* mpm_matls)
{
  printSchedule(patches,cout_doing, "MPMICE::scheduleCoarsenCC_0");
 
  bool modifies = false;

  scheduleCoarsenVariableCC(sched, patches, mpm_matls, MIlb->cMassLabel,
                            1.9531e-15,   modifies, "sum");
                            
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, MIlb->temp_CCLabel,
                            0.,           modifies,"massWeighted");
                            
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, MIlb->vel_CCLabel,
                          Vector(0, 0, 0),modifies,"massWeighted");
                          
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->sp_vol_CCLabel,
                            0.8479864471, modifies, "massWeighted");
                            
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->rho_CCLabel,
                            1.e-12,       modifies, "std");
}

//______________________________________________________________________
//
void MPMICE::scheduleCoarsenNCMass(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* mpm_matls)
{
  printSchedule(patches,cout_doing, "MPMICE::scheduleCoarsenNCMass");
                                                                                
  bool modifies = false;
                                                                                
  scheduleCoarsenVariableNC(sched, patches, mpm_matls, Mlb->gMassLabel,
                            1.e-200,  modifies, "sum");
}

//______________________________________________________________________
//
void MPMICE::scheduleComputeLagrangianValuesMPM(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSubset* one_matl,
                                   const MaterialSet* mpm_matls)
{
  if(d_mpm->flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                                getLevel(patches)->getGrid()->numLevels())){

    printSchedule(patches,cout_doing, "MPMICE::scheduleComputeLagrangianValuesMPM");

    Task* t=scinew Task("MPMICE::computeLagrangianValuesMPM",
                        this, &MPMICE::computeLagrangianValuesMPM);

    const MaterialSubset* mss = mpm_matls->getUnion();
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    t->requires(Task::NewDW, Mlb->gVelocityStarLabel, mss, gac,1);
    t->requires(Task::NewDW, Mlb->gMassLabel,              gac,1);
    t->requires(Task::NewDW, Mlb->gTemperatureStarLabel,   gac,1);
    t->requires(Task::OldDW, Mlb->NC_CCweightLabel,       one_matl, gac,1);
    t->requires(Task::NewDW, MIlb->cMassLabel,             gn);
    t->requires(Task::NewDW, Ilb->int_eng_source_CCLabel,  gn);
    t->requires(Task::NewDW, Ilb->mom_source_CCLabel,      gn);

    t->requires(Task::NewDW, MIlb->temp_CCLabel,           gn);
    t->requires(Task::NewDW, MIlb->vel_CCLabel,            gn);

    if(d_ice->d_models.size() > 0 && !do_mlmpmice){
      t->requires(Task::NewDW, Ilb->modelMass_srcLabel,   gn);
      t->requires(Task::NewDW, Ilb->modelMom_srcLabel,    gn);
      t->requires(Task::NewDW, Ilb->modelEng_srcLabel,    gn);
    }

    t->computes( Ilb->mass_L_CCLabel);
    t->computes( Ilb->mom_L_CCLabel);
    t->computes( Ilb->int_eng_L_CCLabel);

    sched->addTask(t, patches, mpm_matls);
  }
}


//______________________________________________________________________
//
void MPMICE::scheduleCoarsenLagrangianValuesMPM(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* mpm_matls)
{
  printSchedule(patches,cout_doing,"MPMICE:scheduleCoarsenLagrangianValues mpm_matls");

  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->rho_CCLabel,
                            1e-12,          true, "std"); // modifies
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->mass_L_CCLabel,
                            1.9531e-15,     false, "sum");
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->mom_L_CCLabel,
                            Vector(0, 0, 0),false, "sum");
  scheduleCoarsenVariableCC(sched, patches, mpm_matls, Ilb->int_eng_L_CCLabel,
                             0.0,           false, "sum");
}

//______________________________________________________________________
//
void MPMICE::scheduleComputeCCVelAndTempRates(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* mpm_matls)
{
  printSchedule(patches, cout_doing, "MPMICE::scheduleComputeCCVelAndTempRates");

  Task* t=scinew Task("MPMICE::computeCCVelAndTempRates",
                      this, &MPMICE::computeCCVelAndTempRates);               

  Ghost::GhostType  gn = Ghost::None;

  t->requires(Task::OldDW, d_sharedState->get_delt_label(),getLevel(patches));
  t->requires(Task::NewDW, Ilb->mass_L_CCLabel,         gn);
  t->requires(Task::NewDW, Ilb->mom_L_CCLabel,          gn);  
  t->requires(Task::NewDW, Ilb->int_eng_L_CCLabel,      gn);  
  t->requires(Task::NewDW, Ilb->mom_L_ME_CCLabel,       gn);
  t->requires(Task::NewDW, Ilb->eng_L_ME_CCLabel,       gn);
  t->requires(Task::NewDW, Ilb->int_eng_source_CCLabel, gn);
  t->requires(Task::NewDW, Ilb->mom_source_CCLabel,     gn);
  t->requires(Task::OldDW, Mlb->heatRate_CCLabel,       gn);

  t->computes(Ilb->dTdt_CCLabel);
  t->computes(Ilb->dVdt_CCLabel);
  t->computes(Mlb->heatRate_CCLabel);

  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//
void MPMICE::scheduleRefineCC(SchedulerP& sched,
                              const PatchSet* patches,
                              const MaterialSet* mpm_matls)
{
  if(!d_mpm->flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                                 getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches, cout_doing, "MPMICE::scheduleRefineCC");
  scheduleRefineVariableCC(sched, patches, mpm_matls, Ilb->dTdt_CCLabel);
  scheduleRefineVariableCC(sched, patches, mpm_matls, Ilb->dVdt_CCLabel);
}

//______________________________________________________________________
//
void MPMICE::scheduleInterpolateCCToNC(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* mpm_matls)
{
  if(!d_mpm->flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                                 getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches, cout_doing, "MPMICE::scheduleInterpolateCCToNC");  
                                                                                
  Task* t=scinew Task("MPMICE::interpolateCCToNC",
                      this, &MPMICE::interpolateCCToNC);
  const MaterialSubset* mss = mpm_matls->getUnion();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
                                                                                
  t->requires(Task::OldDW, d_sharedState->get_delt_label(),getLevel(patches));
  t->requires(Task::NewDW, Ilb->dVdt_CCLabel,       gan,1);
  t->requires(Task::NewDW, Ilb->dTdt_CCLabel,       gan,1);
  
  if(d_ice->d_models.size() > 0){
    t->requires(Task::NewDW, MIlb->cMassLabel,       gac,1);
    t->requires(Task::NewDW,Ilb->modelMass_srcLabel, gac,1);
  }     
                                                                                
  t->modifies(Mlb->gVelocityStarLabel, mss);
  t->modifies(Mlb->gAccelerationLabel, mss);
  t->computes(Mlb->massBurnFractionLabel,mss);
  t->computes(Mlb->dTdt_NCLabel);

  sched->addTask(t, patches, mpm_matls);
}

/*_____________________________________________________________________
 Function~  MPMICE::scheduleComputePressure--
 Note:  Handles both Rate and Equilibration form of solution technique
_____________________________________________________________________*/
void MPMICE::scheduleComputePressure(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSubset* ice_matls,
                                     const MaterialSubset* mpm_matls,
                                     const MaterialSubset* press_matl,
                                     const MaterialSet* all_matls)
{
  Task* t = NULL;

  printSchedule(patches, cout_doing,"MPMICE::scheduleComputeEquilibrationPressure");

  t = scinew Task("MPMICE::computeEquilibrationPressure",
            this, &MPMICE::computeEquilibrationPressure, press_matl);
  
  t->requires(Task::OldDW, d_sharedState->get_delt_label(),getLevel(patches));

                              // I C E
  Ghost::GhostType  gn  = Ghost::None;

  t->requires(Task::OldDW,Ilb->temp_CCLabel,       ice_matls, gn);  
  t->requires(Task::OldDW,Ilb->rho_CCLabel,        ice_matls, gn);  
  t->requires(Task::OldDW,Ilb->sp_vol_CCLabel,     ice_matls, gn);  
  t->requires(Task::NewDW,Ilb->specific_heatLabel, ice_matls, gn);  
  t->requires(Task::NewDW,Ilb->gammaLabel,         ice_matls, gn);  

                              // M P M
  t->requires(Task::NewDW,MIlb->temp_CCLabel,      mpm_matls, gn);  
  t->requires(Task::NewDW,Ilb->rho_CCLabel,        mpm_matls, gn);  
  t->requires(Task::NewDW,Ilb->sp_vol_CCLabel,     mpm_matls, gn);  

  t->requires(Task::OldDW,Ilb->press_CCLabel,      press_matl, gn);
  t->requires(Task::OldDW,Ilb->vel_CCLabel,        ice_matls,  gn);
  t->requires(Task::NewDW,MIlb->vel_CCLabel,       mpm_matls,  gn);


  computesRequires_CustomBCs(t, "EqPress", Ilb, ice_matls, 
                            d_ice->d_customBC_var_basket);
                              
                              //  A L L _ M A T L S
  t->computes(Ilb->f_theta_CCLabel);
  t->computes(Ilb->compressibilityLabel, ice_matls);
  t->computes(Ilb->compressibilityLabel, mpm_matls);

  t->computes(Ilb->speedSound_CCLabel); 
  t->computes(Ilb->vol_frac_CCLabel);
  t->computes(Ilb->sumKappaLabel,       press_matl);
  t->computes(Ilb->TMV_CCLabel,         press_matl);
  t->computes(Ilb->press_equil_CCLabel, press_matl);  
  t->computes(Ilb->sum_imp_delPLabel,   press_matl);  // needed by implicit ICE
  t->modifies(Ilb->sp_vol_CCLabel,      mpm_matls);
  t->computes(Ilb->sp_vol_CCLabel,      ice_matls);
  t->computes(Ilb->rho_CCLabel,         ice_matls);
  
  sched->addTask(t, patches, all_matls);
}

void MPMICE::scheduleSwitchTest(const LevelP& level, SchedulerP& sched)
{
  if (d_switchCriteria) {
    d_switchCriteria->scheduleSwitchTest(level,sched);
  }
}

//______________________________________________________________________
//       A C T U A L   S T E P S :
//______________________________________________________________________
void MPMICE::actuallyInitialize(const ProcessorGroup*, 
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
    printTask(patches, patch, cout_doing,"Doing actuallyInitialize ");
    //__________________________________
    //output material indices
    if(patch->getID() == 0){
      cout << "Materials Indicies:   MPM ["<< *(d_sharedState->allMPMMaterials())  << "] " 
           << "ICE["<< *(d_sharedState->allICEMaterials()) << "]" << endl;

      cout << "Material Names:";
      int numAllMatls = d_sharedState->getNumMatls();
      for (int m = 0; m < numAllMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        cout <<" " << matl->getDWIndex() << ") " << matl->getName();
      }
      cout << "\n";
    }

    // Sum variable for testing that the volume fractions sum to 1
    CCVariable<double> vol_frac_sum;
    new_dw->allocateTemporary(vol_frac_sum, patch);
    vol_frac_sum.initialize(0.0);

    //__________________________________
    //  Initialize CCVaribles for MPM Materials
    //  Even if mass = 0 in a cell you still need
    //  CC Variables defined.
    double junk=-9, tmp;
    int numMPM_matls = d_sharedState->getNumMPMMatls();
    double p_ref = d_ice->getRefPress();
    for (int m = 0; m < numMPM_matls; m++ ) {
      CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC, speedSound, vol_frac_CC;
      CCVariable<Vector> vel_CC;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int indx= mpm_matl->getDWIndex();
      new_dw->allocateTemporary(rho_micro, patch);
      // Allocate volume fraction for use in intializeCCVariables
      new_dw->allocateTemporary(vol_frac_CC,patch);
      new_dw->allocateAndPut(sp_vol_CC,   Ilb->sp_vol_CCLabel,    indx,patch);
      new_dw->allocateAndPut(rho_CC,      Ilb->rho_CCLabel,       indx,patch);
      new_dw->allocateAndPut(speedSound,  Ilb->speedSound_CCLabel,indx,patch);
      new_dw->allocateAndPut(Temp_CC,    MIlb->temp_CCLabel,      indx,patch);
      new_dw->allocateAndPut(vel_CC,     MIlb->vel_CCLabel,       indx,patch);


      CCVariable<double> heatFlux;
      new_dw->allocateAndPut(heatFlux, Mlb->heatRate_CCLabel,    indx, patch);
      heatFlux.initialize(0.0);

/*`==========TESTING==========*/
#if 0
  09/09/11  Jim is going to check with BB to see if we can delete the particle addition
      // Ignore the dummy materials that are used when particles are
      // localized
      if (d_mpm->flags->d_createNewParticles) {
        if (m%2 == 0) // The actual materials
          mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                          Temp_CC,     vel_CC,
                                          vol_frac_CC, patch);  
        else // The dummy materials
          mpm_matl->initializeDummyCCVariables(rho_micro,   rho_CC,
                                               Temp_CC,     vel_CC,  
                                               vol_frac_CC, patch);  
      } else {
        mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                        Temp_CC,     vel_CC,  
                                        vol_frac_CC, patch);  
      }
#endif 
/*===========TESTING==========`*/      
      
      mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                      Temp_CC,     vel_CC,  
                                      vol_frac_CC, patch);
      
      
      

      setBC(rho_CC,    "Density",      patch, d_sharedState, indx, new_dw);    
      setBC(rho_micro, "Density",      patch, d_sharedState, indx, new_dw);    
      setBC(Temp_CC,   "Temperature",  patch, d_sharedState, indx, new_dw);    
      setBC(vel_CC,    "Velocity",     patch, d_sharedState, indx, new_dw);
      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[c] = 1.0/rho_micro[c];

        mpm_matl->getConstitutiveModel()->
            computePressEOSCM(rho_micro[c],junk, p_ref, junk, tmp,mpm_matl,Temp_CC[c]); 
        speedSound[c] = sqrt(tmp);

        // sum volume fraction
        vol_frac_sum[c] += vol_frac_CC[c];
      }
      
      //__________________________________
      //    B U L L E T   P R O O F I N G
      IntVector neg_cell;
      ostringstream warn;
      if( !areAllValuesPositive(rho_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " rho_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(Temp_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " Temp_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(sp_vol_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " sp_vol_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesNumbers(speedSound, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " speedSound is nan\n";
        warn << "speedSound = " << speedSound[neg_cell] << " sp_vol_CC = " << sp_vol_CC[neg_cell]
             << " rho_micro = " << rho_micro[neg_cell] << " Temp_CC = " << Temp_CC[neg_cell] << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }

      //---- P R I N T   D A T A ------        
      if (d_ice->switchDebug_Initialize){      
        ostringstream desc;
        desc << "MPMICE_Initialization_Mat_" << indx << "_patch_"
             << patch->getID();
        d_ice->printData(indx, patch,  1, desc.str(), "rho_CC",      rho_CC);
        d_ice->printData(indx, patch,  1, desc.str(), "rho_micro_CC",rho_micro);
        d_ice->printData(indx, patch,  1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        d_ice->printData(indx, patch,  1, desc.str(), "Temp_CC",     Temp_CC);
        d_ice->printVector(indx, patch,1, desc.str(), "vel_CC", 0,   vel_CC);
      }             
    }  // num_MPM_matls loop 

    //___________________________________
    //   B U L L E T  P R O O F I N G
    // Verify volume fractions sum to 1.0
    // Loop through ICE materials to get their contribution to volume fraction
    int numICE_matls = d_sharedState->getNumICEMatls();
    for (int m = 0; m < numICE_matls; m++ ) {
      constCCVariable<double> vol_frac;
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();

      // Get the Volume Fraction computed in ICE's actuallyInitialize(...)
      new_dw->get(vol_frac, Ilb->vol_frac_CCLabel, indx, patch, Ghost::None, 0);
      
      for (CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        vol_frac_sum[c] += vol_frac[c];
      }
    }  // num_ICE_matls loop

    double errorThresholdTop    = 1.0e0 + 1.0e-10;
    double errorThresholdBottom = 1.0e0 - 1.0e-10;

    for (CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      
      if(!(vol_frac_sum[c] <= errorThresholdTop && vol_frac_sum[c] >= errorThresholdBottom)){\
        ostringstream warn;
        warn << "ERROR MPMICE::actuallyInitialize cell " << *iter << "\n\n"
             << "volume fraction ("<< std::setprecision(13)<< vol_frac_sum[*iter] << ") does not sum to 1.0 +- 1e-10.\n"
             << "Verify that this region of the domain contains at least 1 geometry object.  If you're using the optional\n"
             << "'volumeFraction' tags verify that they're correctly specified.\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
    } // cell iterator for volume fraction
  } // Patch loop
}

//______________________________________________________________________
//
void MPMICE::interpolatePressCCToPressNC(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{                            
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing interpolatePressCCToPressNC");

    constCCVariable<double> pressCC;
    NCVariable<double> pressNC;

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(pressCC, Ilb->press_CCLabel, 0, patch, gac, 1);
    new_dw->allocateAndPut(pressNC, MIlb->press_NCLabel, 0, patch);
    pressNC.initialize(0.0);
    
    IntVector cIdx[8];
    // Interpolate CC pressure to nodes
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
       patch->findCellsFromNode(*iter,cIdx);
      for (int in=0;in<8;in++){
        pressNC[*iter]  += .125*pressCC[cIdx[in]];
      }
    }

    // Apply grid boundary conditions to the pressure before storing the data
    string inter_type = d_mpm->flags->d_interpolator_type;
    MPMBoundCond bc;
    bc.setBoundaryCondition(patch,0,"Pressure",   pressNC,   inter_type);
  }
}

//______________________________________________________________________
//
void MPMICE::interpolatePAndGradP(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing interpolatePressureToParticles");

    ParticleInterpolator* interpolator = d_mpm->flags->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    double p_ref = d_ice->getRefPress();
    constNCVariable<double>   pressNC;    
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(pressNC, MIlb->press_NCLabel,  0, patch, gac, NGN);

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
      ParticleVariable<double> pPressure;
      constParticleVariable<Point> px;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      old_dw->get(psize,                Mlb->pSizeLabel,     pset);     
      old_dw->get(px,                   Mlb->pXLabel,        pset);     
      old_dw->get(deformationGradient,  Mlb->pDeformationMeasureLabel, pset);
      new_dw->allocateAndPut(pPressure, Mlb->pPressureLabel, pset);     

     //__________________________________
     // Interpolate NC pressure to particles
      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
        particleIndex idx = *iter;
        double press = 0.;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx], ni, S,psize[idx],deformationGradient[idx]);

        for (int k = 0; k < d_8or27; k++) {
          press += pressNC[ni[k]] * S[k];
        }
        pPressure[idx] = press-p_ref;
      }
    }  // numMPMMatls
    delete interpolator;
  } //patches
}

//______________________________________________________________________
//
void MPMICE::interpolateNCToCC_0(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing interpolateNCToCC_0");

    int numMatls = d_sharedState->getNumMPMMatls();
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z(); 
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
    
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight, Mlb->NC_CCweightLabel,  0, patch, gac, 1);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume, gtemperature, gSp_vol;
      constNCVariable<Vector> gvelocity;
      CCVariable<double> cmass,Temp_CC, sp_vol_CC, rho_CC;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> Temp_CC_ice, sp_vol_CC_ice;

      new_dw->allocateAndPut(cmass,    MIlb->cMassLabel,     indx, patch);  
      new_dw->allocateAndPut(vel_CC,   MIlb->vel_CCLabel,    indx, patch);  
      new_dw->allocateAndPut(Temp_CC,  MIlb->temp_CCLabel,   indx, patch);  
      new_dw->allocateAndPut(sp_vol_CC, Ilb->sp_vol_CCLabel, indx, patch); 
      new_dw->allocateAndPut(rho_CC,    Ilb->rho_CCLabel,    indx, patch);
      
      double very_small_mass = d_TINY_RHO * cell_vol;
      cmass.initialize(very_small_mass);

      new_dw->get(gmass,        Mlb->gMassLabel,        indx, patch,gac, 1);
      new_dw->get(gvolume,      Mlb->gVolumeLabel,      indx, patch,gac, 1);
      new_dw->get(gvelocity,    Mlb->gVelocityBCLabel,  indx, patch,gac, 1);
      new_dw->get(gtemperature, Mlb->gTemperatureLabel, indx, patch,gac, 1);
      new_dw->get(gSp_vol,      Mlb->gSp_volLabel,      indx, patch,gac, 1);
      old_dw->get(sp_vol_CC_ice,Ilb->sp_vol_CCLabel,    indx, patch,gn, 0); 
      old_dw->get(Temp_CC_ice,  MIlb->temp_CCLabel,     indx, patch,gn, 0);
      IntVector nodeIdx[8];

//      double sp_vol_orig = 1.0/(mpm_matl->getInitialDensity());
      
      //---- P R I N T   D A T A ------ 
#if 0
      if(switchDebug_InterpolateNCToCC_0) {
        ostringstream desc;
        desc<< "TOP_MPMICE::interpolateNCToCC_0_mat_"<<indx<<"_patch_"
            <<  patch->getID();
        printData(     indx, patch, 1,desc.str(), "gmass",       gmass);
        printData(     indx, patch, 1,desc.str(), "gvolume",     gvolume);
        printData(     indx, patch, 1,desc.str(), "gtemperatue", gtemperature);
        printNCVector( indx, patch, 1,desc.str(), "gvelocity", 0, gvelocity);
      }
#endif 
      //__________________________________
      //  compute CC Variables
      for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        patch->findNodesFromCell(*iter,nodeIdx);
 
        double Temp_CC_mpm = 0.0;  
        double sp_vol_mpm = 0.0;   
        Vector vel_CC_mpm  = Vector(0.0, 0.0, 0.0);
        
        for (int in=0;in<8;in++){
          double NC_CCw_mass = NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          cmass[c]    += NC_CCw_mass;
          sp_vol_mpm  += gSp_vol[nodeIdx[in]]      * NC_CCw_mass;
          vel_CC_mpm  += gvelocity[nodeIdx[in]]    * NC_CCw_mass;
          Temp_CC_mpm += gtemperature[nodeIdx[in]] * NC_CCw_mass;
        }
        double inv_cmass = 1.0/cmass[c];
        vel_CC_mpm  *= inv_cmass;    
        Temp_CC_mpm *= inv_cmass;
        sp_vol_mpm  *= inv_cmass;

        //__________________________________
        // set *_CC = to either vel/Temp_CC_mpm or some safe values
        // depending upon if there is cmass.  You need
        // a well defined vel/temp_CC even if there isn't any mass
        // If you change this you must also change 
        // MPMICE::computeLagrangianValuesMPM
        double one_or_zero = (cmass[c] - very_small_mass)/cmass[c];

//        Temp_CC[c]  =(1.0-one_or_zero)*999.        + one_or_zero*Temp_CC_mpm;
//        sp_vol_CC[c]=(1.0-one_or_zero)*sp_vol_orig + one_or_zero*sp_vol_mpm;

        Temp_CC[c]  =(1.0-one_or_zero)*Temp_CC_ice[c]  +one_or_zero*Temp_CC_mpm;
        sp_vol_CC[c]=(1.0-one_or_zero)*sp_vol_CC_ice[c]+one_or_zero*sp_vol_mpm;

        vel_CC[c]   =vel_CC_mpm;
        rho_CC[c]   = cmass[c]/cell_vol;
      }

      //  Set BC's
      setBC(Temp_CC, "Temperature",patch, d_sharedState, indx, new_dw);
      setBC(rho_CC,  "Density",    patch, d_sharedState, indx, new_dw);
      setBC(vel_CC,  "Velocity",   patch, d_sharedState, indx, new_dw);
      //  Set if symmetric Boundary conditions
      setBC(cmass,    "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
      setBC(sp_vol_CC,"set_if_sym_BC",patch, d_sharedState, indx, new_dw); 

      //---- P R I N T   D A T A ------
      if(switchDebug_InterpolateNCToCC_0) {
        ostringstream desc;
        desc<< "BOT_MPMICE::interpolateNCToCC_0_Mat_"<< indx <<"_patch_"
            <<  patch->getID();
        d_ice->printData(   indx, patch, 1,desc.str(), "sp_vol",    sp_vol_CC); 
        d_ice->printData(   indx, patch, 1,desc.str(), "cmass",     cmass);
        d_ice->printData(   indx, patch, 1,desc.str(), "Temp_CC",   Temp_CC);
        d_ice->printData(   indx, patch, 1,desc.str(), "rho_CC",    rho_CC);
        d_ice->printVector( indx, patch, 1,desc.str(), "vel_CC", 0, vel_CC);
      }
      //---- B U L L E T   P R O O F I N G------
      // ignore BP if timestep restart has already been requested
      IntVector neg_cell;
      ostringstream warn;
      bool tsr = new_dw->timestepRestarted();
      
      int L = getLevel(patches)->getIndex();
      if(d_testForNegTemps_mpm){
        if (!areAllValuesPositive(Temp_CC, neg_cell) && !tsr) {
          warn <<"ERROR MPMICE:("<< L<<"):interpolateNCToCC_0, mat "<< indx 
               <<" cell "
               << neg_cell << " Temp_CC " << Temp_CC[neg_cell] << "\n ";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }
      }
      if (!areAllValuesPositive(rho_CC, neg_cell) && !tsr) {
        warn <<"ERROR MPMICE:("<< L<<"):interpolateNCToCC_0, mat "<< indx 
             <<" cell " << neg_cell << " rho_CC " << rho_CC[neg_cell]<< "\n ";
        throw InvalidValue(warn.str(), __FILE__, __LINE__);
      }
      if (!areAllValuesPositive(sp_vol_CC, neg_cell) && !tsr) {
        warn <<"ERROR MPMICE:("<< L<<"):interpolateNCToCC_0, mat "<< indx 
             <<" cell "
             << neg_cell << " sp_vol_CC " << sp_vol_CC[neg_cell]<<"\n ";
        throw InvalidValue(warn.str(), __FILE__, __LINE__);
      } 
    }
  }  //patches
}

//______________________________________________________________________
//
void MPMICE::computeLagrangianValuesMPM(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* ,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing computeLagrangianValuesMPM");

    int numMatls = d_sharedState->getNumMPMMatls();
    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();
    double very_small_mass = d_TINY_RHO * cellVol; 
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;         
         
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight,       Mlb->NC_CCweightLabel, 0,patch, gac, 1);
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();

      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume,gtempstar;
      constNCVariable<Vector> gvelocity;
      CCVariable<Vector> cmomentum;
      CCVariable<double> int_eng_L, mass_L;
      constCCVariable<double> cmass, Temp_CC_sur, int_eng_src;
      constCCVariable<Vector> vel_CC_sur, mom_source;
      new_dw->get(gmass,       Mlb->gMassLabel,             indx,patch,gac,1);
      new_dw->get(gvelocity,   Mlb->gVelocityStarLabel,     indx,patch,gac,1);
      new_dw->get(gtempstar,   Mlb->gTemperatureStarLabel,  indx,patch,gac,1);
      new_dw->get(cmass,       MIlb->cMassLabel,            indx,patch,gn, 0);
      new_dw->get(Temp_CC_sur, MIlb->temp_CCLabel,          indx,patch,gn, 0);
      new_dw->get(vel_CC_sur,  MIlb->vel_CCLabel,           indx,patch,gn, 0);
      new_dw->get(mom_source,   Ilb->mom_source_CCLabel,    indx,patch,gn, 0);
      new_dw->get(int_eng_src,  Ilb->int_eng_source_CCLabel,indx,patch,gn, 0);

      new_dw->allocateAndPut(mass_L,    Ilb->mass_L_CCLabel,   indx,patch); 
      new_dw->allocateAndPut(cmomentum, Ilb->mom_L_CCLabel,    indx,patch);
      new_dw->allocateAndPut(int_eng_L, Ilb->int_eng_L_CCLabel,indx,patch);

      cmomentum.initialize(Vector(0.0, 0.0, 0.0));
      int_eng_L.initialize(0.);
      mass_L.initialize(0.); 
      double cv = mpm_matl->getSpecificHeat();

      IntVector nodeIdx[8];

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebug_LagrangianValues) {
        ostringstream desc;
        desc <<"TOP_MPMICE::computeLagrangianValuesMPM_mat_"<<indx<<"_patch_"
             <<  indx<<patch->getID();
        d_ice->printData(indx, patch,1,desc.str(), "cmass",    cmass);
        printData(     indx, patch,  1,desc.str(), "gmass",    gmass);
        printData(     indx, patch,  1,desc.str(), "gtemStar", gtempstar);
        //printNCVector( indx, patch,  1,desc.str(), "gvelocityStar", 0,
        //                                                       gvelocity);
      }
 
      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                          iter++){ 
        IntVector c = *iter;
        patch->findNodesFromCell(c,nodeIdx);
        double int_eng_L_mpm = 0.0;
        double int_eng_L_sur = cmass[c] * Temp_CC_sur[c] * cv;
        Vector cmomentum_mpm = Vector(0.0, 0.0, 0.0);
        Vector cmomentum_sur = vel_CC_sur[c] * cmass[c];
        
        for (int in=0;in<8;in++){
          double NC_CCw_mass = NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          cmomentum_mpm +=gvelocity[nodeIdx[in]]      * NC_CCw_mass;
          int_eng_L_mpm +=gtempstar[nodeIdx[in]] * cv * NC_CCw_mass;
        }
        int_eng_L_mpm += int_eng_src[c];
        if(!d_rigidMPM){
          cmomentum_mpm += mom_source[c];
        }
        //__________________________________
        // set cmomentum/int_eng_L to either 
        // what's calculated from mpm or 
        // the surrounding value.
        // If you change this you must also change MPMICE::interpolateNCToCC_0  
        double one_or_zero = (cmass[c] - very_small_mass)/cmass[c];
        cmomentum[c] = (1.0 - one_or_zero)* cmomentum_sur + 
                              one_or_zero * cmomentum_mpm;
        int_eng_L[c] = (1.0 - one_or_zero)* int_eng_L_sur + 
                              one_or_zero * int_eng_L_mpm;
      }
      //__________________________________
      //  NO REACTION
      if(d_ice->d_models.size() == 0 || do_mlmpmice)  { 
        for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                    iter++){ 
         IntVector c = *iter;
         mass_L[c]    = cmass[c];
        }
      }
      //__________________________________
      //   M O D E L   B A S E D   E X C H A N G E
      // The reaction can't completely eliminate 
      //  all the mass, momentum and internal E.
      // If it does then we'll get erroneous vel,
      // and temps in CCMomExchange.  If the mass
      // goes to min_mass then cmomentum and int_eng_L
      // need to be scaled by min_mass to avoid inf temp and vel_CC
      // in 
      if(d_ice->d_models.size() > 0 && !do_mlmpmice){
        constCCVariable<double> modelMass_src;
        constCCVariable<double> modelEng_src;
        constCCVariable<Vector> modelMom_src;
        new_dw->get(modelMass_src,Ilb->modelMass_srcLabel,indx, patch, gn, 0);
        new_dw->get(modelMom_src, Ilb->modelMom_srcLabel, indx, patch, gn, 0);
        new_dw->get(modelEng_src, Ilb->modelEng_srcLabel, indx, patch, gn, 0);
                
        for(CellIterator iter = patch->getExtraCellIterator();!iter.done();
                                                    iter++){
          IntVector c = *iter;
          //  must have a minimum mass
          double min_mass = very_small_mass;
          double inv_cmass = 1.0/cmass[c];
          mass_L[c] = std::max( (cmass[c] + modelMass_src[c] ), min_mass);

          //  must have a minimum momentum 
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons
            double min_mom_L = min_mass * cmomentum[c][dir] * inv_cmass;
            double mom_L_tmp = cmomentum[c][dir] + modelMom_src[c][dir];

            // Preserve the original sign on momemtum     
            // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp+d_SMALL_NUM));

            mom_L_tmp = (mom_L_tmp/mass_L[c] ) * (cmass[c] + modelMass_src[c] );
                                
            cmomentum[c][dir] = plus_minus_one *
                                std::max( fabs(mom_L_tmp), fabs(min_mom_L) );
          }
          // must have a minimum int_eng   
          double min_int_eng = min_mass * int_eng_L[c] * inv_cmass;
          int_eng_L[c] = (int_eng_L[c]/mass_L[c]) * (cmass[c]+modelMass_src[c]);
          int_eng_L[c] = std::max((int_eng_L[c] + modelEng_src[c]),min_int_eng);
        }
      }  // if(model.size() >0)      
       
       //__________________________________
       //  Set Boundary conditions
       setBC(cmomentum, "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
       setBC(int_eng_L, "set_if_sym_BC",patch, d_sharedState, indx, new_dw);

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebug_LagrangianValues) {
        ostringstream desc;
        desc<<"BOT_MPMICE::computeLagrangianValuesMPM_mat_"<<indx<<"_patch_"
            <<  patch->getID();
        d_ice->printData(  indx,patch, 1,desc.str(), "mass_L",       mass_L);
        d_ice->printData(  indx,patch, 1,desc.str(), "int_eng_L_CC", int_eng_L);
        d_ice->printVector(indx,patch, 1,desc.str(), "mom_L_CC", 0,  cmomentum);
      }
      
      //---- B U L L E T   P R O O F I N G------
      // ignore BP if timestep restart has already been requested
      IntVector neg_cell;
      ostringstream warn;
      bool tsr = new_dw->timestepRestarted();
      
      if(d_testForNegTemps_mpm){
        if (!areAllValuesPositive(int_eng_L, neg_cell) && !tsr) {
          int L = getLevel(patches)->getIndex();
          warn <<"ERROR MPMICE:("<< L<<"):computeLagrangianValuesMPM, mat "
               << indx<<" cell "
               << neg_cell << " int_eng_L_CC " << int_eng_L[neg_cell] << "\n ";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }
      }
    }  //numMatls
  }  //patches
}

//______________________________________________________________________
//
void MPMICE::computeCCVelAndTempRates(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing computeCCVelAndTempRates");

    //__________________________________
    // This is where I interpolate the CC 
    // changes to NCs for the MPMMatls
    int numMPMMatls = d_sharedState->getNumMPMMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for (int m = 0; m < numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      CCVariable<double> dTdt_CC,heatRate;
      CCVariable<Vector> dVdt_CC;

      constCCVariable<double> mass_L_CC, old_heatRate;
      constCCVariable<Vector> mom_L_ME_CC, old_mom_L_CC, mom_source;
      constCCVariable<double> eng_L_ME_CC, old_int_eng_L_CC, int_eng_src;
      
      double cv = mpm_matl->getSpecificHeat();     

      Ghost::GhostType  gn = Ghost::None;
      new_dw->get(old_mom_L_CC,    Ilb->mom_L_CCLabel,       indx,patch,gn, 0);
      new_dw->get(old_int_eng_L_CC,Ilb->int_eng_L_CCLabel,   indx,patch,gn, 0);
      new_dw->get(mass_L_CC,       Ilb->mass_L_CCLabel,      indx,patch,gn, 0);
      new_dw->get(mom_L_ME_CC,     Ilb->mom_L_ME_CCLabel,    indx,patch,gn, 0);
      new_dw->get(eng_L_ME_CC,     Ilb->eng_L_ME_CCLabel,    indx,patch,gn, 0);
      old_dw->get(old_heatRate,    Mlb->heatRate_CCLabel,    indx,patch,gn, 0);
      new_dw->get(mom_source,      Ilb->mom_source_CCLabel,  indx,patch,gn, 0);
      new_dw->get(int_eng_src,   Ilb->int_eng_source_CCLabel,indx,patch,gn, 0);

      new_dw->allocateAndPut(dTdt_CC,     Ilb->dTdt_CCLabel,    indx, patch);
      new_dw->allocateAndPut(dVdt_CC,     Ilb->dVdt_CCLabel,    indx, patch);
      new_dw->allocateAndPut(heatRate,    Mlb->heatRate_CCLabel,indx, patch);

      dTdt_CC.initialize(0.0);
      dVdt_CC.initialize(Vector(0.0));
      //__________________________________
      for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
         IntVector c = *iter;
         if(!d_rigidMPM){
           dVdt_CC[c] = (mom_L_ME_CC[c] - (old_mom_L_CC[c]-mom_source[c]))
                                                      /(mass_L_CC[c]*delT);
         }
         dTdt_CC[c]   = (eng_L_ME_CC[c] - (old_int_eng_L_CC[c]-int_eng_src[c]))
                           /(mass_L_CC[c] * cv * delT);
         double heatRte  = (eng_L_ME_CC[c] - old_int_eng_L_CC[c])/delT;
         heatRate[c] = .05*heatRte + .95*old_heatRate[c];
      }
      setBC(dTdt_CC,    "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
      setBC(dVdt_CC,    "set_if_sym_BC",patch, d_sharedState, indx, new_dw);
    }
  }  //patches
}

//______________________________________________________________________
//
void MPMICE::interpolateCCToNC(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* ,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing interpolateCCToNC");

    //__________________________________
    // This is where I interpolate the CC 
    // changes to NCs for the MPMMatls
    int numMPMMatls = d_sharedState->getNumMPMMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for (int m = 0; m < numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();
      NCVariable<Vector> gacceleration, gvelocity;
      NCVariable<double> dTdt_NC,massBurnFraction;

      constCCVariable<double> dTdt_CC;
      constCCVariable<Vector> dVdt_CC;
      
      new_dw->getModifiable(gvelocity,    Mlb->gVelocityStarLabel,indx,patch);
      new_dw->getModifiable(gacceleration,Mlb->gAccelerationLabel,indx,patch);
                  
      Ghost::GhostType  gan = Ghost::AroundNodes;
      new_dw->get(dTdt_CC,    Ilb->dTdt_CCLabel,   indx,patch,gan,1);
      new_dw->get(dVdt_CC,    Ilb->dVdt_CCLabel,   indx,patch,gan,1);
      
      new_dw->allocateAndPut(massBurnFraction, 
                                      Mlb->massBurnFractionLabel,indx,patch);
      new_dw->allocateAndPut(dTdt_NC, Mlb->dTdt_NCLabel,    indx, patch);
      
      dTdt_NC.initialize(0.0);
      massBurnFraction.initialize(0.);
      IntVector cIdx[8];
      //__________________________________
      //  
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        patch->findCellsFromNode(*iter,cIdx);
        for(int in=0;in<8;in++){
          gvelocity[*iter]     +=  dVdt_CC[cIdx[in]]*delT*.125;
          gacceleration[*iter] +=  dVdt_CC[cIdx[in]]*.125;
          dTdt_NC[*iter]       +=  dTdt_CC[cIdx[in]]*.125;
        }
      }
      //__________________________________
      //  inter-material phase transformation
      if(d_ice->d_models.size() > 0)  { 
        constCCVariable<double> modelMass_src, mass_CC;
        Ghost::GhostType  gac = Ghost::AroundCells;
        new_dw->get(modelMass_src,Ilb->modelMass_srcLabel,indx,patch, gac,1);
        new_dw->get(mass_CC,      MIlb->cMassLabel,       indx,patch, gac,1);

        for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
          patch->findCellsFromNode(*iter,cIdx);
          for (int in=0;in<8;in++){
            massBurnFraction[*iter] +=
                     (fabs(modelMass_src[cIdx[in]])/mass_CC[cIdx[in]])*.125;

          }
        }
      }  // if(models >0 )
    }  // mpmMatls
  }  //patches
}


/* --------------------------------------------------------------------- 
 Function~  MPMICE::computeEquilibrationPressure-- 
 Reference: Flow of Interpenetrating Material Phases, J. Comp, Phys
               18, 440-464, 1975, see the equilibration section
                   
 Steps
 ----------------
    - Compute rho_micro_CC, SpeedSound, vol_frac for ALL matls

    For each cell
    _ WHILE LOOP(convergence, max_iterations)
        - compute the pressure and dp_drho from the EOS of each material.
        - Compute delta Pressure
        - Compute delta volume fraction and update the 
          volume fraction and the celldensity.
        - Test for convergence of delta pressure and delta volume fraction
    - END WHILE LOOP
    - bulletproofing
    end
 
Note:  The nomenclature follows the reference.
       This is identical to  ICE::computeEquilibrationPressure except
       we now include EOS for MPM matls.                               
_____________________________________________________________________*/
void MPMICE::computeEquilibrationPressure(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     const MaterialSubset* press_matl)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing computeEquilibrationPressure");

    double    converg_coeff = 100.;
    double    convergence_crit = converg_coeff * DBL_EPSILON;
    double    c_2;
    double press_ref= d_ice->getRefPress();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = numICEMatls + numMPMMatls;

    Vector dx       = patch->dCell(); 
    double cell_vol = dx.x()*dx.y()*dx.z();

    StaticArray<double> press_eos(numALLMatls);
    StaticArray<double> dp_drho(numALLMatls),dp_de(numALLMatls);
    StaticArray<double> mat_volume(numALLMatls);

    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > rho_micro(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numALLMatls);
    StaticArray<CCVariable<double> > speedSound(numALLMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numALLMatls);
    StaticArray<CCVariable<double> > f_theta(numALLMatls);
    StaticArray<CCVariable<double> > kappa(numALLMatls);
    StaticArray<constCCVariable<double> > placeHolder(0);
    StaticArray<constCCVariable<double> > cv(numALLMatls);
    StaticArray<constCCVariable<double> > gamma(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls); 
    StaticArray<constCCVariable<double> > Temp(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC_old(numALLMatls);
    StaticArray<constCCVariable<double> > mass_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numALLMatls);
    constCCVariable<double> press;    
    CCVariable<double> press_new, delPress_tmp,sumKappa, TMV_CC;
    CCVariable<double> sum_imp_delP;    
    Ghost::GhostType  gn = Ghost::None;
    //__________________________________
    old_dw->get(press,                  Ilb->press_CCLabel, 0,patch,gn, 0); 
    new_dw->allocateAndPut(press_new,   Ilb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(TMV_CC,      Ilb->TMV_CCLabel,         0,patch);
    new_dw->allocateAndPut(sumKappa,    Ilb->sumKappaLabel,       0,patch);
    new_dw->allocateAndPut(sum_imp_delP,Ilb->sum_imp_delPLabel,   0,patch);
    new_dw->allocateTemporary(delPress_tmp, patch); 
    
    sum_imp_delP.initialize(0.0);

    StaticArray<MPMMaterial*> mpm_matl(numALLMatls);
    StaticArray<ICEMaterial*> ice_matl(numALLMatls);
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ice_matl[m] = dynamic_cast<ICEMaterial*>(matl);
      mpm_matl[m] = dynamic_cast<MPMMaterial*>(matl);
    }

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      if(ice_matl[m]){                    // I C E
        old_dw->get(Temp[m],      Ilb->temp_CCLabel,      indx,patch,gn,0);
        old_dw->get(rho_CC_old[m],Ilb->rho_CCLabel,       indx,patch,gn,0);
        old_dw->get(sp_vol_CC[m], Ilb->sp_vol_CCLabel,    indx,patch,gn,0);
        old_dw->get(vel_CC[m],    Ilb->vel_CCLabel,       indx,patch,gn,0);
        new_dw->get(cv[m],        Ilb->specific_heatLabel,indx,patch,gn,0);
        new_dw->get(gamma[m],     Ilb->gammaLabel,        indx,patch,gn,0);
        new_dw->allocateAndPut(rho_CC_new[m], Ilb->rho_CCLabel,  indx,patch);
      }
      if(mpm_matl[m]){                    // M P M
        new_dw->get(Temp[m],     MIlb->temp_CCLabel, indx,patch,gn,0);
        new_dw->get(vel_CC[m],   MIlb->vel_CCLabel,  indx,patch,gn,0);
        new_dw->get(sp_vol_CC[m],Ilb->sp_vol_CCLabel,indx,patch,gn,0); 
        new_dw->get(rho_CC_old[m],Ilb->rho_CCLabel,  indx,patch,gn,0);
        new_dw->allocateTemporary(rho_CC_new[m],  patch);
      }
      new_dw->allocateTemporary(rho_micro[m],  patch);
      new_dw->allocateAndPut(vol_frac[m],   Ilb->vol_frac_CCLabel,  indx,patch);
      new_dw->allocateAndPut(f_theta[m],    Ilb->f_theta_CCLabel,   indx,patch);
      new_dw->allocateAndPut(speedSound[m], Ilb->speedSound_CCLabel,indx,patch);
      new_dw->allocateAndPut(sp_vol_new[m], Ilb->sp_vol_CCLabel,    indx,patch);
      new_dw->allocateAndPut(kappa[m],    Ilb->compressibilityLabel,indx,patch);
    }

    press_new.copyData(press);

    //__________________________________
    // Compute rho_micro, speedSound, volfrac, rho_CC
    // see Docs/MPMICE.txt for explaination of why we ONlY
    // use eos evaulations for rho_micro_mpm

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){

      const IntVector& c = *iter;
      double total_mat_vol = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        if(ice_matl[m]){                // I C E
         rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
        } else if(mpm_matl[m]){                //  M P M
          rho_micro[m][c] =  mpm_matl[m]->getConstitutiveModel()->
            computeRhoMicroCM(press_new[c],press_ref, mpm_matl[m],Temp[m][c],1.0/sp_vol_CC[m][c]);
        }
        mat_volume[m] = (rho_CC_old[m][c]*cell_vol)/rho_micro[m][c];
        total_mat_vol += mat_volume[m];
      }  // numAllMatls loop

      TMV_CC[c] = total_mat_vol;

      for (int m = 0; m < numALLMatls; m++) {
        vol_frac[m][c] = mat_volume[m]/total_mat_vol;
        rho_CC_new[m][c] = vol_frac[m][c]*rho_micro[m][c];
      }
    }  // cell iterator
    //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_equil_press)  {
      ostringstream desc1;
      desc1<< "TOP_equilibration_patch_"<< patch->getID();
      d_ice->printData( 0, patch, 1, desc1.str(), "Press_CC_top", press); 

      for (int m = 0; m < numALLMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"TOP_equilibration_Mat_"<< indx<<"_patch_"<<patch->getID();
        d_ice->printData( indx,patch,1,desc.str(),"rho_CC_new",rho_CC_new[m]);
        d_ice->printData( indx,patch,1,desc.str(),"rho_micro", rho_micro[m]);
        d_ice->printData( indx,patch,1,desc.str(),"Temp_CC",   Temp[m]);     
        d_ice->printData( indx,patch,1,desc.str(),"vol_frac_CC",vol_frac[m]);
      }
    }
    //______________________________________________________________________
    // Done with preliminary calcs, now loop over every cell
    int count, test_max_iter = 0;

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;  
      double delPress = 0.;
      bool converged  = false;
      double sum;
      count           = 0;
      vector<EqPress_dbg> dbgEqPress;

      while ( count < d_ice->d_max_iter_equilibration && converged == false) {
        count++;
        //__________________________________
        // evaluate press_eos at cell i,j,k
        for (int m = 0; m < numALLMatls; m++)  {
          if(ice_matl[m]){    // ICE
            ice_matl[m]->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                                   cv[m][c],Temp[m][c],
                                                   press_eos[m],
                                                   dp_drho[m],dp_de[m]);
          } else if(mpm_matl[m]){    // MPM
            mpm_matl[m]->getConstitutiveModel()->
              computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                dp_drho[m], c_2,mpm_matl[m],Temp[m][c]);
          }
        }

        //__________________________________
        // - compute delPress
        // - update press_CC     
        double A = 0., B = 0., C = 0.;

        for (int m = 0; m < numALLMatls; m++)   {
          double Q =  press_new[c] - press_eos[m];
          double inv_y =  (vol_frac[m][c] * vol_frac[m][c])
            / (dp_drho[m] * rho_CC_new[m][c] + d_SMALL_NUM);
                                 
          A   +=  vol_frac[m][c];
          B   +=  Q * inv_y;
          C   +=  inv_y;
        } 
        double vol_frac_not_close_packed = 1.;
        delPress = (A - vol_frac_not_close_packed - B)/C;

        press_new[c] += delPress;

        if(press_new[c] < convergence_crit ){
          press_new[c] = fabs(delPress);
        }
        
       //__________________________________
       // backout rho_micro_CC at this new pressure
       // - compute the updated volume fractions
       sum = 0;
       for (int m = 0; m < numALLMatls; m++) {
         if(ice_matl[m]){
           rho_micro[m][c] = 
             ice_matl[m]->getEOS()->computeRhoMicro(press_new[c],gamma[m][c],
                                           cv[m][c],Temp[m][c],rho_micro[m][c]);
         } if(mpm_matl[m]){
           rho_micro[m][c] =  
             mpm_matl[m]->getConstitutiveModel()->computeRhoMicroCM(
                                          press_new[c],press_ref,mpm_matl[m],Temp[m][c],rho_micro[m][c]);
         }
         vol_frac[m][c]   = rho_CC_new[m][c]/rho_micro[m][c];
         sum += vol_frac[m][c];
       }

       //__________________________________
       // - Test for convergence 
       //  If sum of vol_frac_CC ~= 1.0 then converged 
       if (fabs(sum-vol_frac_not_close_packed) < convergence_crit){
         converged = true;
         //__________________________________
         // Find the speed of sound based on the converged solution
         //feclearexcept(FE_ALL_EXCEPT);
         for (int m = 0; m < numALLMatls; m++)  {
           if(ice_matl[m]){
             ice_matl[m]->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                              cv[m][c],Temp[m][c],press_eos[m],
                                              dp_drho[m],dp_de[m]);

             c_2 = dp_drho[m] + dp_de[m] * 
                        (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
           } else if(mpm_matl[m]){
              mpm_matl[m]->getConstitutiveModel()->
                   computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                     dp_drho[m],c_2,mpm_matl[m],Temp[m][c]);
           }
           speedSound[m][c] = sqrt(c_2);         // Isentropic speed of sound

           //____ BB : B U L L E T   P R O O F I N G----
           // catch inf and nan speed sound values
           //if (fetestexcept(FE_INVALID) != 0 || c_2 == 0.0) {
           //  ostringstream warn;
           //  warn<<"ERROR MPMICE::computeEquilPressure, mat= "<< m << " cell= "
           //      << c << " sound speed is imaginary.\n";
           //  warn << "speedSound = " << speedSound[m][c] << " c_2 = " << c_2 
           //       << " press_eos = " << press_eos[m]
           //       << " dp_drho = " << dp_drho[m]
           //       << " dp_de = " << dp_de[m]
           //       << " rho_micro = " << rho_micro[m][c] << " Temp = " << Temp[m][c] << endl;
           //  throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
           //}
         }
       }

       // Save iteration data for output in case of crash
       if(ds_EqPress.active()){
         EqPress_dbg dbg;
         dbg.delPress     = delPress;
         dbg.press_new    = press_new[c];
         dbg.sumVolFrac   = sum;
         dbg.count        = count;

         for (int m = 0; m < numALLMatls; m++) {
           EqPress_dbgMatl dmatl;
           dmatl.press_eos   = press_eos[m];
           dmatl.volFrac     = vol_frac[m][c];
           dmatl.rhoMicro    = rho_micro[m][c];
           dmatl.rho_CC      = rho_CC_new[m][c];
           dmatl.temp_CC     = Temp[m][c];
           dmatl.mat         = m;
           dbg.matl.push_back(dmatl);
         }
         dbgEqPress.push_back(dbg);
       }
       
     }   // end of converged

      delPress_tmp[c] = delPress;

     //__________________________________
     // If the pressure solution has stalled out 
     //  then try a binary search
     if(count >= d_ice->d_max_iter_equilibration) {
        //int lev = patch->getLevel()->getIndex();
        //cout << "WARNING:MPMICE:ComputeEquilibrationPressure "
        //     << " Cell : " << c << " on level " << lev << " having a difficult time converging. \n"
        //    << " Now performing a binary pressure search " << endl;

        binaryPressureSearch( Temp, rho_micro, vol_frac, rho_CC_new,
                              speedSound,  dp_drho,  dp_de, 
                              press_eos, press, press_new, press_ref,
                              cv, gamma, convergence_crit, 
                              numALLMatls, count, sum, c);
     }
     test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      // ignore BP if timestep restart has already been requested
      bool tsr = new_dw->timestepRestarted();
      
      string message;
      bool allTestsPassed = true;
      if(test_max_iter == d_ice->d_max_iter_equilibration && !tsr){
        allTestsPassed = false;
        message += "Max. iterations reached ";
      }
      
      for (int m = 0; m < numALLMatls; m++) {
        ASSERT(( vol_frac[m][c] > 0.0 ) ||( vol_frac[m][c] < 1.0));
      }
      
      if ( fabs(sum - 1.0) > convergence_crit && !tsr) {  
        allTestsPassed = false;
        message += " sum (volumeFractions) != 1 ";
      }
      
      if ( press_new[c] < 0.0 && !tsr) {
        allTestsPassed = false;
        message += " Computed pressure is < 0 ";
      }
      
      for (int m = 0; m < numALLMatls; m++){
        if ((rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) && !tsr) {
          allTestsPassed = false;
          message += " rho_micro < 0 || vol_frac < 0";
        }
      }
      if(allTestsPassed != true){  // throw an exception of there's a problem
        ostringstream warn;
        warn << "\nMPMICE::ComputeEquilibrationPressure: Cell "<< c << ", L-"<<L_indx <<"\n"
             << message
             <<"\nThis usually means that something much deeper has gone wrong with the simulation. "
             <<"\nCompute equilibration pressure task is rarely the problem. "
             << "For more debugging information set the environmental variable:  \n"
             << "   SCI_DEBUG DBG_EqPress:+\n\n";
             
        warn << "INPUTS: \n"; 
        for (int m = 0; m < numALLMatls; m++){
          warn<< "\n matl: " << m << "\n"
               << "   rho_CC:     " << rho_CC_new[m][c] << "\n"
               << "   Temperature:   "<< Temp[m][c] << "\n";
        }
        if(ds_EqPress.active()){
          warn << "\nDetails on iterations " << endl;
          vector<EqPress_dbg>::iterator dbg_iter;
          for( dbg_iter  = dbgEqPress.begin(); dbg_iter != dbgEqPress.end(); dbg_iter++){
            EqPress_dbg & d = *dbg_iter;
            warn << "Iteration:   " << d.count
                 << "  press_new:   " << d.press_new
                 << "  sumVolFrac:  " << d.sumVolFrac
                 << "  delPress:    " << d.delPress << "\n";
            for (int m = 0; m < numALLMatls; m++){
              warn << "  matl: " << d.matl[m].mat
                   << "  press_eos:  " << d.matl[m].press_eos
                   << "  volFrac:    " << d.matl[m].volFrac
                   << "  rhoMicro:   " << d.matl[m].rhoMicro
                   << "  rho_CC:     " << d.matl[m].rho_CC
                   << "  Temp:       " << d.matl[m].temp_CC << "\n";
            }
          }
        } 
      }  // all testsPassed
    }  // end of cell interator
    if (cout_norm.active())
      cout_norm<<"max number of iterations in any cell \t"<<test_max_iter<<endl;

    //__________________________________
    // Now change how rho_CC is defined to 
    // rho_CC = mass/cell_volume  NOT mass/mat_volume 
    for (int m = 0; m < numALLMatls; m++) {
      if(ice_matl[m]){
        rho_CC_new[m].copyData(rho_CC_old[m]);
      }
    }

    //__________________________________
    // - update Boundary conditions
    //   Don't set Lodi bcs, we already compute Press
    //   in all the extra cells.
    // - make copy of press for implicit calc.
    preprocess_CustomBCs("EqPress",old_dw, new_dw, Ilb, patch, 999,
                          d_ice->d_customBC_var_basket);
    
    setBC(press_new,   rho_micro, placeHolder,d_ice->d_surroundingMatl_indx,
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw, 
          d_ice->d_customBC_var_basket);
    
    delete_CustomBCs(d_ice->d_customBC_var_basket);

     
    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numALLMatls; m++)   {  
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        const IntVector& c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c];
      }
      
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setSpecificVolBC(sp_vol_new[m], "SpecificVol", false,rho_CC_new[m],vol_frac[m],
                       patch,d_sharedState, indx);
    } 
    //__________________________________
    //  compute f_theta  
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      sumKappa[c] = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        kappa[m][c] = sp_vol_new[m][c]/(speedSound[m][c]*speedSound[m][c]);
        sumKappa[c] += vol_frac[m][c]*kappa[m][c];
      }
      for (int m = 0; m < numALLMatls; m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m][c]/sumKappa[c];
      }
    }

    //____ BB : B U L L E T   P R O O F I N G----
    //for (int m = 0; m < numALLMatls; m++) {
    //  Material* matl = d_sharedState->getMaterial( m );
    //  int indx = matl->getDWIndex();
    //  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    //    if (isnan(kappa[m][*iter]) || isinf(kappa[m][*iter])) {
    //      ostringstream warn;
    //      warn<<"MPMICE:(L-"<<m<<"):computeEquilibrationPressure, mat "<<indx<<" cell "
    //             << *iter << " kappa = " << kappa[m][*iter] 
    //             << " vol_frac = " << vol_frac[m][*iter] 
    //             << " sp_vol_new = " << sp_vol_new[m][*iter] 
    //             << " speedSound = " << speedSound[m][*iter] << endl;
    //      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    //    }
    //  }
    //}

#if 0
    //__________________________________
    //  compute f_theta  (alternate)
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      sumKappa[c] = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        if(ice_matl[m]){
          kappa[m][c] = sp_vol_new[m][c]/(speedSound[m][c]*speedSound[m][c]);
        }
        if(mpm_matl[m]){
          kappa[m][c] = mpm_matl[m]->getConstitutiveModel()->getCompressibility();
        }
        sumKappa[c] += vol_frac[m][c]*kappa[m][c];
      }
      for (int m = 0; m < numALLMatls; m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m][c]/sumKappa[c];
      }
    }
#endif
    
  //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_equil_press)  { 
      ostringstream desc1;
      desc1<< "BOT_equilibration_patch_"<<patch->getID();

      d_ice->printData( 0, patch, 1, desc1.str(),"Press_CC_equil",press_new);
      d_ice->printData( 0, patch, 1, desc1.str(),"delPress",      delPress_tmp);
      for (int m = 0; m < numALLMatls; m++)  {
         Material* matl = d_sharedState->getMaterial( m );
         int indx = matl->getDWIndex();
         ostringstream desc; 
         desc<< "BOT_equilibration_Mat_"<<indx<<"_patch_"<< patch->getID();
         d_ice->printData(indx,patch,1,desc.str(),"rho_CC",      rho_CC_new[m]);
         d_ice->printData(indx,patch,1,desc.str(),"rho_micro_CC",rho_micro[m]);
         d_ice->printData(indx,patch,1,desc.str(),"vol_frac_CC", vol_frac[m]);
      }
    }
  }  //patches
}

/* --------------------------------------------------------------------- 
 Function~  MPMICE::binaryPressureSearch-- 
 Purpose:   When the technique for find the equilibration pressure
            craps out then try this method.
 Reference:  Se Numerical Methods by Robert W. Hornbeck.
_____________________________________________________________________*/ 
void MPMICE::binaryPressureSearch(  StaticArray<constCCVariable<double> >& Temp,
                            StaticArray<CCVariable<double> >& rho_micro, 
                            StaticArray<CCVariable<double> >& vol_frac, 
                            StaticArray<CCVariable<double> >& rho_CC_new,
                            StaticArray<CCVariable<double> >& speedSound,
                            StaticArray<double> & dp_drho, 
                            StaticArray<double> & dp_de, 
                            StaticArray<double> & press_eos,
                            constCCVariable<double> & press,
                            CCVariable<double> & press_new, 
                            double press_ref,
                            StaticArray<constCCVariable<double> > & cv,
                            StaticArray<constCCVariable<double> > & gamma,
                            double convergence_crit,
                            int numALLMatls,
                            int & count,
                            double & sum,
                            IntVector c )
{
  // Start over for this cell using a binary search
//  cout << " cell " << c << " Starting binary pressure search "<< endl;
  count = 0;
  bool converged = false;
  double c_2;
  double Pleft=0., Pright=0., Ptemp=0., Pm=0.;
  double rhoMicroR=0., rhoMicroL=0.;
  StaticArray<double> vfR(numALLMatls);
  StaticArray<double> vfL(numALLMatls);
  Pm = press[c];
  double residual = 1.0;

  while ( count < d_ice->d_max_iter_equilibration && converged == false) {
    count++;
    sum = 0.;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){        // ICE
        rho_micro[m][c] =
          ice_matl->getEOS()->computeRhoMicro(Pm,gamma[m][c],
                                            cv[m][c],Temp[m][c],rho_micro[m][c]);
      }
      if(mpm_matl){        // MPM
        rho_micro[m][c] =
          mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                       Pm,press_ref,mpm_matl,Temp[m][c],rho_micro[m][c]);
      }
      vol_frac[m][c] = rho_CC_new[m][c]/rho_micro[m][c];
      sum += vol_frac[m][c];
    }  // loop over matls

    residual = 1. - sum;

    if(fabs(residual) <= convergence_crit){
      converged = true;
      press_new[c] = Pm;
      //__________________________________
      // Find the speed of sound at ijk
      // needed by eos and the the explicit
      // del pressure function
      //feclearexcept(FE_ALL_EXCEPT);
      for (int m = 0; m < numALLMatls; m++)  {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        if(ice_matl){       // ICE
          ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                              cv[m][c],Temp[m][c],press_eos[m],
                                              dp_drho[m],dp_de[m]);
          c_2 = dp_drho[m] + dp_de[m] *
                     (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
        }
        if(mpm_matl){       // MPM
           mpm_matl->getConstitutiveModel()->
                computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                  dp_drho[m],c_2,mpm_matl,Temp[m][c]);
        }
        speedSound[m][c] = sqrt(c_2);     // Isentropic speed of sound

        //____ BB : B U L L E T   P R O O F I N G----
        // catch inf and nan speed sound values
        //if (fetestexcept(FE_INVALID) != 0 || c_2 == 0.0) {
        //  ostringstream warn;
        //  warn<<"ERROR MPMICE::binaryPressSearch, mat= "<< m << " cell= "
        //      << c << " sound speed is imaginary.\n";
        //  warn << "speedSound = " << speedSound[m][c] << " c_2 = " << c_2 
        //       << " press_eos = " << press_eos[m]
        //       << " dp_drho = " << dp_drho[m]
        //       << " dp_de = " << dp_de[m]
        //       << " rho_micro = " << rho_micro[m][c] << " Temp = " << Temp[m][c] << endl;
        //  throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
        //}
      }
    }
    // Initial guess
    if(count == 1){
      Pleft = DBL_EPSILON;
      Pright=1.0/DBL_EPSILON;
      Ptemp = .5*(Pleft + Pright);
    }

    double sumR=0., sumL=0.;
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      if(ice_matl){        //  ICE
        rhoMicroR = ice_matl->getEOS()->
         computeRhoMicro(Pright,gamma[m][c],cv[m][c],Temp[m][c],rho_micro[m][c]);
        rhoMicroL = ice_matl->getEOS()->
         computeRhoMicro(Pleft, gamma[m][c],cv[m][c],Temp[m][c],rho_micro[m][c]);
      }
      if(mpm_matl){        //  MPM
        rhoMicroR =
          mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                       Pright,press_ref,mpm_matl,Temp[m][c],rho_micro[m][c]);
        rhoMicroL =
          mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                       Pleft, press_ref,mpm_matl,Temp[m][c],rho_micro[m][c]);
      }
      vfR[m] = rho_CC_new[m][c]/rhoMicroR;
      vfL[m] = rho_CC_new[m][c]/rhoMicroL;
      sumR += vfR[m];
      sumL += vfL[m];
      
//      cout << "matl: " << m << " vol_frac_L: " << vfL[m] << " vol_frac_R: " << vfR[m] 
//           << " rho_CC: " << rho_CC_new[m][c] << " rho_micro_L: " << rhoMicroL << " rhoMicroR: " << rhoMicroR << endl;
    }  // all matls


//    cout << "Pm = " << Pm << "\t P_L: " << Pleft << "\t P_R: " << Pright << "\t 1.-sum " << residual << " \t sumR: " << sumR << " \t sumL " << sumL << endl;

    //__________________________________
    //  come up with a new guess
    double prod = (1.- sumR)*(1. - sumL);
    if(prod < 0.){
      Ptemp = Pleft;
      Pleft = .5*(Pleft + Pright);
    }else{
      Pright = Pleft;
      Pleft  = Ptemp;
      Pleft  = 0.5 * (Pleft + Pright);
    }
    Pm = .5*(Pleft + Pright);
 //   cout << setprecision(15);

  }   // end of converged

#ifdef D_STRICT
  if (count >= d_ice->d_max_iter_equilibration) {
    ostringstream desc;
    desc << "**ERROR** Binary pressure search failed to converge in cell" << c << endl;
    throw ConvergenceFailure(desc.str(), d_ice->d_max_iter_equilibration,
                             fabs(residual), convergence_crit, __FILE__, __LINE__);
  }
#else
  if (count >= d_ice->d_max_iter_equilibration) {
    cout << "**WARNING** Binary pressure search failed to converge in cell " << c << " after "
         << d_ice->d_max_iter_equilibration << " iterations.  Final residual is "
         << fabs(residual) << " and convergence tolerance is " << convergence_crit << endl;
    cout << "  Continuing with unconverged value of pressure = " << Pm << endl;
    press_new[c] = Pm;
    //__________________________________
    // Find the speed of sound at ijk
    // needed by eos and the the explicit
    // del pressure function
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){       // ICE
        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                            cv[m][c],Temp[m][c],press_eos[m],
                                            dp_drho[m],dp_de[m]);
        c_2 = dp_drho[m] + dp_de[m] *
                     (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
      }
      if(mpm_matl){       // MPM
        mpm_matl->getConstitutiveModel()->
                computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                  dp_drho[m],c_2,mpm_matl,Temp[m][c]);
      }
      speedSound[m][c] = sqrt(c_2);     // Isentropic speed of sound
      cout << "    Material " << m << " vol. frac = " << vol_frac[m][c]
           << " rho = " << rho_micro[m][c] << " press = " << press_eos[m] 
           << " dp_drho = " << dp_drho[m] << " c^2 = " << c_2 << endl;
    }
  }
#endif
}

//______________________________________________________________________
bool MPMICE::needRecompile(double time, double dt, const GridP& grid) {
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
                                                                                
void MPMICE::addMaterial(const ProblemSpecP& prob_spec, GridP& grid,
                         SimulationStateP&   sharedState)
{
  d_recompile = true;
  if(d_sharedState->needAddMaterial() > 0){
    d_ice->addMaterial(prob_spec, grid, d_sharedState);
    cout << "Adding an ICE material" << endl;
  }
  if(d_sharedState->needAddMaterial() < 0){
    d_mpm->addMaterial(prob_spec, grid, d_sharedState);
    d_ice->updateExchangeCoefficients(prob_spec, grid, d_sharedState);
    cout << "Adding an MPM material" << endl;
  }
}
//______________________________________________________________________
void MPMICE::scheduleInitializeAddedMaterial(const LevelP& level,
                                             SchedulerP& sched)
{
  printSchedule(level,cout_doing,"MPMICE::scheduleInitializeAddedMaterial");

  if(d_sharedState->needAddMaterial() > 0){
    d_ice->scheduleInitializeAddedMaterial(level,sched);
  }
  if(d_sharedState->needAddMaterial() < 0){
    d_mpm->scheduleInitializeAddedMaterial(level,sched);

    Task* t = scinew Task("MPMICE::actuallyInitializeAddedMPMMaterial",
                           this, &MPMICE::actuallyInitializeAddedMPMMaterial);

    int addedMaterialIndex = d_sharedState->getNumMatls() - 1;

    MaterialSubset* add_matl = scinew MaterialSubset();
    add_matl->add(addedMaterialIndex);
    add_matl->addReference();
    t->computes(MIlb->vel_CCLabel,       add_matl);
    t->computes(Mlb->heatRate_CCLabel,   add_matl);
    t->computes(Ilb->rho_CCLabel,        add_matl);
    t->computes(Ilb->temp_CCLabel,       add_matl);
    t->computes(Ilb->sp_vol_CCLabel,     add_matl);
    t->computes(Ilb->speedSound_CCLabel, add_matl);

    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

    if (add_matl->removeReference())
      delete add_matl; // shouln't happen, but...
  }
}

//______________________________________________________________________
void MPMICE::actuallyInitializeAddedMPMMaterial(const ProcessorGroup*, 
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse*,
                                                DataWarehouse* new_dw)
{
  new_dw->unfinalize();
  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing actuallyInitializeAddedMPMMaterial");
   
    double junk=-9, tmp;
    int m    = d_sharedState->getNumMPMMatls() - 1;
    int indx = d_sharedState->getNumMatls() - 1;
    double p_ref = d_ice->getRefPress();
    CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC, speedSound, vol_frac_CC;
    CCVariable<double>  heatRate_CC;
    CCVariable<Vector> vel_CC;
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    new_dw->allocateTemporary(rho_micro,   patch);
    new_dw->allocateTemporary(vol_frac_CC, patch);
    new_dw->allocateAndPut(sp_vol_CC, Ilb->sp_vol_CCLabel,    indx,patch);
    new_dw->allocateAndPut(rho_CC,    Ilb->rho_CCLabel,       indx,patch);
    new_dw->allocateAndPut(speedSound,Ilb->speedSound_CCLabel,indx,patch);
    new_dw->allocateAndPut(Temp_CC,  MIlb->temp_CCLabel,      indx,patch);
    new_dw->allocateAndPut(vel_CC,   MIlb->vel_CCLabel,       indx,patch);
    new_dw->allocateAndPut(heatRate_CC,Mlb->heatRate_CCLabel, indx,patch);

    heatRate_CC.initialize(0.0);

    mpm_matl->initializeDummyCCVariables(rho_micro,   rho_CC,
                                         Temp_CC,     vel_CC, 
                                         vol_frac_CC, patch);

    setBC(rho_CC,    "Density",      patch, d_sharedState, indx, new_dw);
    setBC(rho_micro, "Density",      patch, d_sharedState, indx, new_dw);
    setBC(Temp_CC,   "Temperature",  patch, d_sharedState, indx, new_dw);
    setBC(vel_CC,    "Velocity",     patch, d_sharedState, indx, new_dw);
    for (CellIterator iter = patch->getExtraCellIterator();
                                                      !iter.done();iter++){
      IntVector c = *iter;
      sp_vol_CC[c] = 1.0/rho_micro[c];
                                                                              
      mpm_matl->getConstitutiveModel()->
          computePressEOSCM(rho_micro[c],junk, p_ref, junk, tmp,mpm_matl,Temp_CC[c]);
      speedSound[c] = sqrt(tmp);
    }
  }
  new_dw->refinalize();
}

//______________________________________________________________________
void MPMICE::scheduleRefineInterface(const LevelP& fineLevel,
                                     SchedulerP& scheduler,
                                     bool needOld, bool needNew)
{
  d_ice->scheduleRefineInterface(fineLevel, scheduler, needOld, needNew);
  d_mpm->scheduleRefineInterface(fineLevel, scheduler, needOld, needNew);

  if(fineLevel->getIndex() > 0 && d_sharedState->isCopyDataTimestep() &&
     d_mpm->flags->doMPMOnLevel(fineLevel->getIndex(),
                                fineLevel->getGrid()->numLevels())) {
    cout_doing << d_myworld->myrank() 
               << " MPMICE::scheduleRefineInterface \t\t\tL-"
               << fineLevel->getIndex() << endl;

    Task* task = scinew Task("MPMICE::refineCoarseFineInterface",
                             this, &MPMICE::refineCoarseFineInterface);

    const MaterialSet* all_matls   = d_sharedState->allMaterials();
    const MaterialSubset* one_matl = d_ice->d_press_matl;

    task->modifies(Mlb->NC_CCweightLabel, one_matl);

    scheduler->addTask(task, fineLevel->eachPatch(), all_matls);
  }
}


void MPMICE::refineCoarseFineInterface(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* fine_old_dw,
                                       DataWarehouse* fine_new_dw)
{
  // This isn't actually refining anything, it is simply reinitializing
  // NC_CCweight after regridding on all levels finer than 0 because
  // copyData doesn't copy extra cell data.
  const Level* level = getLevel(patches);
  if(level->getIndex() > 0){
    cout_doing << d_myworld->myrank()
               << " Doing refineCoarseFineInterface"<< "\t\t\t MPMICE L-"
               << level->getIndex() << " Patches: " << *patches << endl;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      //__________________________________
      //NC_CCweight
      NCVariable<double> NC_CCweight;
      fine_new_dw->getModifiable(NC_CCweight, Mlb->NC_CCweightLabel, 0, patch);
      //__________________________________
      // - Initialize NC_CCweight = 0.125
      // - Find the walls with symmetry BC and double NC_CCweight
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
          } // cell iterator
        } // if symmetry
      } // for patch faces
    } // for patches
  } // if level
}
//______________________________________________________________________
void MPMICE::scheduleRefine(const PatchSet* patches, 
                            SchedulerP& sched)
{
  d_ice->scheduleRefine(patches, sched);
  d_mpm->scheduleRefine(patches, sched);

  printSchedule(patches,cout_doing,"MPMICE::scheduleRefine");

  Task* task = scinew Task("MPMICE::refine", this, &MPMICE::refine);
  
  task->computes(Mlb->heatRate_CCLabel);
  task->computes(Ilb->sp_vol_CCLabel);
  task->computes(MIlb->vel_CCLabel);
  task->computes(Ilb->temp_CCLabel);

  sched->addTask(task, patches, d_sharedState->allMPMMaterials());
}

//______________________________________________________________________    
void MPMICE::scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched)
{
  d_ice->scheduleCoarsen(coarseLevel, sched);
  d_mpm->scheduleCoarsen(coarseLevel, sched);
}

//______________________________________________________________________
void MPMICE::scheduleInitialErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched)
{
  d_ice->scheduleInitialErrorEstimate(coarseLevel, sched);
  d_mpm->scheduleInitialErrorEstimate(coarseLevel, sched);
}

//______________________________________________________________________                                               
void MPMICE::scheduleErrorEstimate(const LevelP& coarseLevel,
                                   SchedulerP& sched)
{
  d_ice->scheduleErrorEstimate(coarseLevel, sched);
  d_mpm->scheduleErrorEstimate(coarseLevel, sched);
}

//______________________________________________________________________
 void MPMICE::scheduleRefineVariableCC(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls,
                                       const VarLabel* variable)
 {
   ostringstream taskName;
   taskName << "MPMICE::refineVariable(" << variable->getName() << ")";
   Task* t;

   // the sgis don't like accepting a templated function over a function call for some reason...
   void (MPMICE::*func)(const ProcessorGroup*, const PatchSubset*, const MaterialSubset*,
                        DataWarehouse*, DataWarehouse*, const VarLabel*);
                        
   switch(variable->typeDescription()->getSubType()->getType()){
   case TypeDescription::double_type:
     func = &MPMICE::refineVariableCC<double>;
     t=scinew Task(taskName.str().c_str(),this, func, variable);
     break;
   case TypeDescription::Vector:
     func = &MPMICE::refineVariableCC<Vector>;
     t=scinew Task(taskName.str().c_str(),this, func, variable);
     break;
   default:
     throw InternalError("Unknown variable type for refine", __FILE__, __LINE__);
   }

   Ghost::GhostType  gac = Ghost::AroundCells;
   t->requires(Task::NewDW, variable, 0, Task::CoarseLevel, 0, Task::NormalDomain, gac, 1);
   t->computes(variable);
   sched->addTask(t, patches, matls);
 }

 //______________________________________________________________________
 template<typename T>
   void MPMICE::scheduleCoarsenVariableCC(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const VarLabel* variable,
                                          T defaultValue, 
                                          bool modifies,
                                          const string& coarsenMethod)
{
  // The SGI compiler does't like accepting a templated function over
  // a function call for some reason...  We use this hack to force it
  // to figure out the correct type of the function.
  void (MPMICE::*func)(const ProcessorGroup*, const PatchSubset*,
                       const MaterialSubset*, DataWarehouse*, DataWarehouse*, 
                       const VarLabel*, T, bool, string);
  func = &MPMICE::coarsenVariableCC<T>;
  ostringstream taskName;

  taskName << "MPMICE::coarsenVariableCC(" << variable->getName() 
           << (modifies?" modified":"") << ")";
  
  Task* t=scinew Task(taskName.str().c_str(),this, func, 
                       variable, defaultValue, modifies, coarsenMethod);
  
  Ghost::GhostType  gn = Ghost::None;
  Task::MaterialDomainSpec ND   = Task::NormalDomain;
  
  t->requires(Task::NewDW, variable, 0, Task::FineLevel, 0, ND,gn,0);
  
  if(coarsenMethod == "massWeighted"){
    t->requires(Task::NewDW, MIlb->cMassLabel, 0, Task::FineLevel, 0, ND,gn,0);
  }
  
  if(modifies){
    t->modifies(variable);
  }else{
    t->computes(variable);
  }
  sched->addTask(t, patches, matls);
}
 

 //______________________________________________________________________
 template<typename T>
   void MPMICE::scheduleCoarsenVariableNC(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const VarLabel* variable,
                                          T defaultValue,
                                          bool modifies,
                                          string coarsenMethod)
{
  // The SGI compiler does't like accepting a templated function over
  // a function call for some reason...  We use this hack to force it
  // to figure out the correct type of the function.
  void (MPMICE::*func)(const ProcessorGroup*, const PatchSubset*,
                       const MaterialSubset*, DataWarehouse*, DataWarehouse*,
                       const VarLabel*, T, bool, string);
  func = &MPMICE::coarsenVariableNC<T>;
  ostringstream taskName;

  taskName << "MPMICE::coarsenVariableNC(" << variable->getName() 
           << (modifies?" modified":"") << ")";

  Task* t=scinew Task(taskName.str().c_str(),this, func,
                       variable, defaultValue, modifies, coarsenMethod);

  //Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Task::MaterialDomainSpec ND   = Task::NormalDomain;

  const LevelP fineLevel = getLevel(patches)->getFinerLevel();
  IntVector refineRatio(fineLevel->getRefinementRatio());
  int ghost = max(refineRatio.x(),refineRatio.y());
  ghost = max(ghost,refineRatio.z());

  t->requires(Task::NewDW, variable, 0, Task::FineLevel, 0, ND, gan, ghost);

  if(modifies){
    t->modifies(variable);
  }else{
    t->computes(variable);
  }
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
void
MPMICE::refine(const ProcessorGroup*,
               const PatchSubset* patches,
               const MaterialSubset* /*matls*/,
               DataWarehouse*,
               DataWarehouse* new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing refine");
     
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      cout_doing << d_myworld->myrank() << " Doing refine on patch "
           << patch->getID() << " material # = " << dwi << endl;
      
      // for now, create 0 heat flux
      CCVariable<double> heatFlux;
      new_dw->allocateAndPut(heatFlux, Mlb->heatRate_CCLabel, dwi, patch);
      heatFlux.initialize(0.0);

      CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC, vol_frac_CC;
      CCVariable<Vector> vel_CC;

      new_dw->allocateTemporary(rho_micro,   patch);
      new_dw->allocateTemporary(rho_CC,      patch);
      new_dw->allocateTemporary(vol_frac_CC, patch);

      new_dw->allocateAndPut(sp_vol_CC,   Ilb->sp_vol_CCLabel,    dwi,patch);
      new_dw->allocateAndPut(Temp_CC,     MIlb->temp_CCLabel,     dwi,patch);
      new_dw->allocateAndPut(vel_CC,      MIlb->vel_CCLabel,      dwi,patch);

      mpm_matl->initializeDummyCCVariables(rho_micro,   rho_CC,
                                           Temp_CC,     vel_CC,
                                           vol_frac_CC, patch);  
      //__________________________________
      //  Set boundary conditions                                     
      setBC(rho_micro, "Density",      patch, d_sharedState, dwi, new_dw);    
      setBC(Temp_CC,   "Temperature",  patch, d_sharedState, dwi, new_dw);    
      setBC(vel_CC,    "Velocity",     patch, d_sharedState, dwi, new_dw);
      for (CellIterator iter = patch->getExtraCellIterator();
           !iter.done();iter++){
        sp_vol_CC[*iter] = 1.0/rho_micro[*iter];
      }

      //__________________________________
      //    B U L L E T   P R O O F I N G
      IntVector neg_cell;
      ostringstream warn;
      if( !areAllValuesPositive(rho_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<dwi<< " cell "
            <<neg_cell << " rho_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(Temp_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<dwi<< " cell "
            <<neg_cell << " Temp_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(sp_vol_CC, neg_cell) ) {
        warn<<"ERROR MPMICE::actuallyInitialize, mat "<<dwi<< " cell "
            <<neg_cell << " sp_vol_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
    }  //mpmMatls
  }  //patches
}

//______________________________________________________________________
//
template<typename T>
void MPMICE::refineVariableCC(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse*,
                              DataWarehouse* new_dw,
                              const VarLabel* variable)
{
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  IntVector refineRatio(fineLevel->getRefinementRatio());

  for(int p=0;p<patches->size();p++){
    const Patch* finePatch = patches->get(p);
    ostringstream message;
    message<<"Doing refineVariableCC (" << variable->getName() << ")\t\t\t";
    printTask(patches,finePatch,cout_doing,message.str());    
    
    // region of fine space that will correspond to the coarse we need to get
    IntVector cl, ch, fl, fh;
    IntVector bl(0,0,0);  // boundary layer cells
    int nGhostCells = 1;
    bool returnExclusiveRange=true;
    
    getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl, 
                        nGhostCells, returnExclusiveRange);

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      CCVariable<T> fine_q_CC;
      new_dw->allocateAndPut(fine_q_CC, variable, indx, finePatch);
      
      constCCVariable<T> coarse_q_CC;

      new_dw->getRegion(coarse_q_CC, variable, indx, coarseLevel, cl, ch, false);
    
      // Only interpolate over the intersection of the fine and coarse patches
      // coarse cell 
//      linearInterpolation<T>(coarse_q_CC, coarseLevel, fineLevel,
//                             refineRatio, lo, hi, fine_q_CC);

      piecewiseConstantInterpolation<T>(coarse_q_CC, fineLevel,
                                        fl, fh, fine_q_CC);
    }
  }
}



//__________________________________
//
template<typename T>
void MPMICE::coarsenDriver_stdNC(IntVector cl,
                                 IntVector ch,
                                 IntVector refinementRatio,
                                 double ratio,
                                 const Level* coarseLevel,
                                 constNCVariable<T>& fine_q_NC,
                                 NCVariable<T>& coarse_q_NC )
{
  T zero(0.0);
  // iterate over coarse level cells
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  Vector DX = coarseLevel->dCell();
  IntVector range(refinementRatio.x()/2, refinementRatio.y()/2, refinementRatio.z()/2);

  IntVector varLow = fine_q_NC.getLowIndex();
  IntVector varHigh = fine_q_NC.getHighIndex();

  for(NodeIterator iter(cl, ch); !iter.done(); iter++){
    IntVector c = *iter;
    IntVector fineNode = coarseLevel->mapNodeToFiner(c);
    Point coarseLoc=coarseLevel->getNodePosition(c);

    IntVector start = Max(fineNode-range, varLow);
    IntVector end = Min(fineNode+range, varHigh);

    // for each coarse level cell iterate over the fine level cells
    T q_NC_tmp(zero);

    for (NodeIterator inner(start, end); !inner.done(); inner++) {
      IntVector fc(*inner);
      Point fineLoc=fineLevel->getNodePosition(fc);
      Vector C2F = fineLoc - coarseLoc;
      Vector Vweight = C2F/DX;
      double weight = (1.-fabs(Vweight.x()))*
        (1.-fabs(Vweight.y()))*
        (1.-fabs(Vweight.z()));
      q_NC_tmp += fine_q_NC[fc]*weight;
    }
    coarse_q_NC[c] =q_NC_tmp;
  }
}

//______________________________________________________________________
//
template<typename T>
void MPMICE::coarsenVariableCC(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse* new_dw,
                               const VarLabel* variable,
                               T defaultValue, 
                               bool modifies,
                               string coarsenMethod)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  IntVector refineRatio(fineLevel->getRefinementRatio());
  double ratio = 1./(refineRatio.x()*refineRatio.y()*refineRatio.z());

  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);
    ostringstream message;
    message<<"Doing CoarsenVariableCC (" << variable->getName() << ")\t\t\t";
    printTask(patches,coarsePatch,cout_doing,message.str());

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      CCVariable<T> coarse_q_CC;
      if(modifies){
        new_dw->getModifiable(coarse_q_CC, variable, indx, coarsePatch);
      }else{
        new_dw->allocateAndPut(coarse_q_CC, variable, indx, coarsePatch);
      }
      coarse_q_CC.initialize(defaultValue);

      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      for(int i=0;i<finePatches.size();i++){
        const Patch* finePatch = finePatches[i];
  
        IntVector cl, ch, fl, fh;
        getFineLevelRange(coarsePatch, finePatch, cl, ch, fl, fh);
        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
          continue;
        }
        
        constCCVariable<T> fine_q_CC;
        new_dw->getRegion(fine_q_CC,  variable, indx, fineLevel, fl, fh, false);
        
        //__________________________________
        //  call the coarsening function
        ASSERT((coarsenMethod=="std" || coarsenMethod=="sum" 
                                     || coarsenMethod=="massWeighted"));
        if(coarsenMethod == "std"){
          coarsenDriver_std(cl, ch, fl, fh, refineRatio,ratio, coarseLevel, 
                            fine_q_CC, coarse_q_CC);
        }
        if(coarsenMethod =="sum"){
          ratio = 1.0;
          coarsenDriver_std(cl, ch, fl, fh, refineRatio,ratio, coarseLevel, 
                            fine_q_CC, coarse_q_CC);
        }
        if(coarsenMethod == "massWeighted"){
          constCCVariable<double> cMass;
          new_dw->getRegion(cMass,  MIlb->cMassLabel, indx, fineLevel, fl, fh, false);
          
          coarsenDriver_massWeighted(cl,ch, fl, fh, refineRatio,coarseLevel,
                                     cMass, fine_q_CC, coarse_q_CC );
        }
      }  // fine patches
      // Set BCs on coarsened data.  This sucks--Steve
      if(variable->getName()=="temp_CC"){
       setBC(coarse_q_CC, "Temperature",coarsePatch,d_sharedState,indx,new_dw);
      }
      else if(variable->getName()=="rho_CC"){
       setBC(coarse_q_CC, "Density",    coarsePatch,d_sharedState,indx,new_dw);
      }
      else if(variable->getName()=="vel_CC"){
       setBC(coarse_q_CC, "Velocity",   coarsePatch,d_sharedState,indx,new_dw);
      }
      else if(variable->getName()=="c.mass"       ||
              variable->getName()=="sp_vol_CC"    ||
              variable->getName()=="mom_L_CC"     ||
              variable->getName()=="int_eng_L_CC" ){
       setBC(coarse_q_CC,"set_if_sym_BC",coarsePatch,d_sharedState,indx,new_dw);
      }
    }  // matls
  }  // coarse level
}

//______________________________________________________________________
//
template<typename T>
void MPMICE::coarsenVariableNC(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse* new_dw,
                               const VarLabel* variable,
                               T defaultValue, 
                               bool modifies,
                               string coarsenMethod)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  IntVector refineRatio(fineLevel->getRefinementRatio());
  double ratio = 1./(refineRatio.x()*refineRatio.y()*refineRatio.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    ostringstream message;
    message<<"Doing CoarsenVariableNC (" << variable->getName() << ")\t\t\t";
    printTask(patches,coarsePatch,cout_doing,message.str());

    for(int m = 0;m<matls->size();m++){
      int indx = matls->get(m);

      NCVariable<T> coarse_q_NC;
      if(modifies){
        new_dw->getModifiable(coarse_q_NC, variable, indx, coarsePatch);
      }else{
        new_dw->allocateAndPut(coarse_q_NC, variable, indx, coarsePatch);
      }
      coarse_q_NC.initialize(defaultValue);

      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      for(int i=0;i<finePatches.size();i++){
        const Patch* finePatch = finePatches[i];
        
        IntVector cl, ch, fl, fh;

        IntVector padding(refineRatio.x()/2,refineRatio.y()/2,refineRatio.z()/2);
        getFineLevelRangeNodes(coarsePatch, finePatch, cl, ch, fl, fh,padding);
        

        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
          continue;
        }

        constNCVariable<T> fine_q_NC;
        new_dw->getRegion(fine_q_NC,  variable, indx, fineLevel, fl, fh, false);

        //__________________________________
        //  call the coarsening function
        ASSERT(coarsenMethod=="sum"); 
        if(coarsenMethod == "sum"){
          coarsenDriver_stdNC(cl, ch, refineRatio,ratio, coarseLevel, 
                              fine_q_NC, coarse_q_NC);
        }
      }  // fine patches
    }  // matls
  }  // coarse level
}
