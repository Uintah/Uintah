// MPMICE.cc

#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/MathToolbox.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/StaticArray.h>
#include <float.h>
#include <stdio.h>
#include <Core/Util/DebugStream.h>


using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT".....
//  MPMICE_NORMAL_COUT:  dumps out during problemSetup 
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF
static DebugStream cout_norm("MPMICE_NORMAL_COUT", false);  
static DebugStream cout_doing("MPMICE_DOING_COUT", false);

#define EOSCM
//#undef EOSCM
//#define IDEAL_GAS
#undef IDEAL_GAS
//#define FAKE_BURN_MODEL
#undef FAKE_BURN_MODEL

MPMICE::MPMICE(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  d_mpm      = scinew SerialMPM(myworld);
  d_ice      = scinew ICE(myworld);
  d_SMALL_NUM = 1.e-100;
  // Turn off all the debuging switches
  switchDebug_InterpolateNCToCC_0 = false;
  switchDebug_InterpolateCCToNC   = false;
}

MPMICE::~MPMICE()
{
  delete MIlb;
  delete d_mpm;
  delete d_ice;
}

//______________________________________________________________________
//
void MPMICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                       SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  //__________________________________
  //  M P M
  d_mpm->setMPMLabel(Mlb);
  d_mpm->setWithICE();
  d_mpm->problemSetup(prob_spec, grid, d_sharedState);
  
  //__________________________________
  //  I C E
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(dataArchiver == 0){
    cout<<"dataArhiver in MPMICE is null now exiting; "<<endl;
    exit(1);
  }
  d_ice->attachPort("output", dataArchiver);
  d_ice->setICELabel(Ilb);
  d_ice->problemSetup(prob_spec, grid, d_sharedState);
  //__________________________________
  //  M P M I C E
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {
    d_dbgStartTime = 0.;
    d_dbgStopTime = 1.;
    d_dbgOutputInterval = 0.0;
    d_dbgBeginIndx = IntVector(0,0,0);
    d_dbgEndIndx   = IntVector(0,0,0); 
    d_dbgSigFigs   = 5;
    
    debug_ps->get("dbg_timeStart",     d_dbgStartTime);
    debug_ps->get("dbg_timeStop",      d_dbgStopTime);
    debug_ps->get("dbg_outputInterval",d_dbgOutputInterval);
    debug_ps->get("d_dbgBeginIndx",    d_dbgBeginIndx);
    debug_ps->get("d_dbgEndIndx",      d_dbgEndIndx );
    debug_ps->get("dbg_SigFigs",       d_dbgSigFigs );

    d_dbgOldTime = -d_dbgOutputInterval;
    d_dbgNextDumpTime = 0.0;
    
    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
        child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"] == "switchDebug_InterpolateNCToCC_0")
        switchDebug_InterpolateNCToCC_0 = true;
      else if (debug_attr["label"] == "switchDebug_InterpolateCCToNC")
        switchDebug_InterpolateCCToNC = true;       
    }
  }
  cout_norm << "Done with problemSetup \t\t\t MPMICE" <<endl;
  cout_norm << "--------------------------------\n"<<endl;
}
//______________________________________________________________________
//
void MPMICE::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  cout_doing << "\nDoing scheduleInitialize \t\t\t MPMICE" << endl;
  d_mpm->scheduleInitialize(level, sched);
  d_ice->scheduleInitialize(level, sched);

  //__________________________________
  //  What isn't initialized in either ice or mpm
  Task* t = scinew Task("MPMICE::actuallyInitialize",
                  this, &MPMICE::actuallyInitialize);
  t->computes(Mlb->doMechLabel);
  t->computes(MIlb->NC_CCweightLabel);
  t->computes(MIlb->vel_CCLabel);
  t->computes(Ilb->rho_CCLabel); 
  t->computes(Ilb->temp_CCLabel);
  t->computes(Ilb->sp_vol_CCLabel);
  
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  cout_doing << "Done with Initialization \t\t\t MPMICE" <<endl;
  cout_norm << "--------------------------------\n"<<endl; 
}

void MPMICE::restartInitialize()
{
  d_mpm->restartInitialize();
  d_ice->restartInitialize();
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
void MPMICE::scheduleTimeAdvance(const LevelP&   level,
                             SchedulerP&     sched)
{
  int numALLMatls=d_sharedState->getNumMatls();
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  MaterialSubset* press_matl   = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();

  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();
  //__________________________________
  //  Find the product and reactant matl subset
  MaterialSubset* prod_sub  = scinew MaterialSubset();
  prod_sub->addReference();
  MaterialSubset* react_sub = scinew MaterialSubset();
  react_sub->addReference();
 
  for (int m = 0; m < numALLMatls; m++) {
    Material* matl = d_sharedState->getMaterial(m);
    if (matl->getRxProduct() == Material::product) {
     //cerr << "Product Material: " << m << endl;
     prod_sub->add(m);
    }
    if (matl->getRxProduct() == Material::reactant) {
     //cerr << "reactant Material: " << m << endl;
     react_sub->add(m);
    }
  }
 //__________________________________
 // Scheduling
  if( d_mpm->withFracture() ) {
    d_mpm->scheduleSetPositions(                  sched, patches, mpm_matls);
    d_mpm->scheduleComputeBoundaryContact(        sched, patches, mpm_matls);
    d_mpm->scheduleComputeConnectivity(           sched, patches, mpm_matls);
  }
  d_mpm->scheduleInterpolateParticlesToGrid(      sched, patches, mpm_matls);

  d_mpm->scheduleComputeHeatExchange(             sched, patches, mpm_matls);

  // schedule the interpolation of mass and volume to the cell centers
  scheduleInterpolateNCToCC_0(                    sched, patches, one_matl,
                                                                  mpm_matls);

  scheduleComputePressure(                        sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
  if (d_ice->d_RateForm) {
    d_ice->scheduleComputeFCPressDiffRF(          sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
  }
  d_ice->scheduleComputeFaceCenteredVelocities(   sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  press_matl,
                                                                  all_matls);
                                                               
  d_ice->scheduleAddExchangeContributionToFCVel(  sched, patches, all_matls);

  scheduleHEChemistry(                            sched, patches, react_sub,
                                                                  prod_sub,
                                                                  press_matl,
                                                                  all_matls);
                                                                   
  d_ice->scheduleComputeDelPressAndUpdatePressCC( sched, patches, press_matl,
                                                                  ice_matls_sub, 
                                                                  mpm_matls_sub,
                                                                  all_matls);
  
  d_mpm->scheduleExMomInterpolated(               sched, patches, mpm_matls);
  d_mpm->scheduleComputeStressTensor(             sched, patches, mpm_matls);

  scheduleInterpolateMassBurnFractionToNC(        sched, patches, mpm_matls);

  d_ice->scheduleComputePressFC(                  sched, patches, press_matl,
                                                                  all_matls);
  d_ice->scheduleAccumulateMomentumSourceSinks(   sched, patches, press_matl,
                                                                  ice_matls_sub,
                                                                  all_matls);
  d_ice->scheduleAccumulateEnergySourceSinks(     sched, patches, press_matl,
                                                                  all_matls);

  scheduleInterpolatePressCCToPressNC(            sched, patches, press_matl,
                                                                  mpm_matls);
  scheduleInterpolatePAndGradP(                   sched, patches, press_matl,
                                                                  one_matl,
                                                                  mpm_matls_sub,
                                                                  mpm_matls);
   
  d_mpm->scheduleComputeInternalForce(            sched, patches, mpm_matls);
  d_mpm->scheduleComputeInternalHeatRate(         sched, patches, mpm_matls);
  d_mpm->scheduleSolveEquationsMotion(            sched, patches, mpm_matls);
  d_mpm->scheduleSolveHeatEquations(              sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateAcceleration(           sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateTemperatureRate(        sched, patches, mpm_matls);

  scheduleComputeLagrangianValuesMPM(             sched, patches, one_matl,
                                                                  mpm_matls); 

  d_ice->scheduleComputeLagrangianValues(         sched, patches, ice_matls);

  scheduleCCMomExchange(                          sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);

  d_ice->scheduleComputeLagrangianSpecificVolume( sched, patches, press_matl,
                                                                  ice_matls_sub,
                                                                  all_matls);
  scheduleInterpolateCCToNC(                      sched, patches, mpm_matls);
  d_mpm->scheduleExMomIntegrated(                 sched, patches, mpm_matls);
  d_mpm->scheduleSetGridBoundaryConditions(       sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate( sched, patches, mpm_matls);


  if( d_mpm->withFracture() ) {
    d_mpm->scheduleComputeFracture(               sched, patches, mpm_matls);
    d_mpm->scheduleComputeCrackExtension(         sched, patches, mpm_matls);
  }

  d_mpm->scheduleCarryForwardVariables(           sched, patches, mpm_matls);
  d_ice->scheduleAdvectAndAdvanceInTime(          sched, patches, ice_matls_sub,
                                                                  mpm_matls_sub,
                                                                  all_matls);
  sched->scheduleParticleRelocation(level,
                                Mlb->pXLabel_preReloc, 
                                Mlb->d_particleState_preReloc,
                                Mlb->pXLabel, Mlb->d_particleState,
                                Mlb->pParticleIDLabel, mpm_matls);

   // whatever tasks use press_matl will have their own reference to it.
  if (press_matl->removeReference())
    delete press_matl; 
  // whatever tasks use one_matl will have their own reference to it.
  if (one_matl->removeReference())
    delete one_matl;
  // whatever tasks use prod_sub will have their own reference to it.
  if (prod_sub->removeReference())
    delete prod_sub;
  // whatever tasks use react_sub will have their own reference to it.
  if (react_sub->removeReference())
    delete react_sub;
}

//______________________________________________________________________
//
void MPMICE::scheduleInterpolatePressCCToPressNC(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* press_matl,
                                           const MaterialSet* matls)
{
  cout_doing << "MPMICE::scheduleInterpolatePressCCToPressNC" << endl;
  
  Task* t=scinew Task("MPMICE::interpolatePressCCToPressNC",
                    this, &MPMICE::interpolatePressCCToPressNC);
  
  t->requires(Task::NewDW,Ilb->press_CCLabel, press_matl,Ghost::AroundCells, 1);
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
  cout_doing << "MPMICE::scheduleInterpolatePAndGradP" << endl;
 
   Task* t=scinew Task("MPMICE::interpolatePAndGradP",
                 this, &MPMICE::interpolatePAndGradP);

   t->requires(Task::OldDW, d_sharedState->get_delt_label());
   
   t->requires(Task::NewDW, MIlb->press_NCLabel,    press_matl, 
                                                    Ghost::AroundCells, 1);
   t->requires(Task::OldDW, Mlb->pXLabel,           mpm_matl,    
                                                    Ghost::None);
   t->requires(Task::NewDW, Ilb->mom_source_CCLabel,mpm_matl, 
                                                    Ghost::AroundCells, 1);
   t->requires(Task::NewDW, MIlb->cMassLabel,       mpm_matl,
                                                    Ghost::AroundCells, 1);

   t->computes(Mlb->pPressureLabel,   mpm_matl);
   t->computes(Mlb->gradPAccNCLabel,  mpm_matl);
   sched->addTask(t, patches, all_matls);
}



//______________________________________________________________________
//
void MPMICE::scheduleInterpolateNCToCC_0(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* one_matl,
                                    const MaterialSet* mpm_matls)
{
  cout_doing << "MPMICE::scheduleInterpolateNCToCC_0" << endl;
 
   /* interpolateNCToCC */
   Task* t=scinew Task("MPMICE::interpolateNCToCC_0",
                 this, &MPMICE::interpolateNCToCC_0);

   t->requires(Task::OldDW, Mlb->doMechLabel, Ghost::None);   
   t->requires(Task::NewDW, Mlb->gMassLabel,       Ghost::AroundCells, 1);
   t->requires(Task::NewDW, Mlb->gVolumeLabel,     Ghost::AroundCells, 1);
   t->requires(Task::NewDW, Mlb->gVelocityLabel,   Ghost::AroundCells, 1); 
   t->requires(Task::NewDW, Mlb->gTemperatureLabel,Ghost::AroundCells, 1);
   t->requires(Task::OldDW, MIlb->NC_CCweightLabel,one_matl,
                                                   Ghost::AroundCells, 1);
   t->requires(Task::OldDW, MIlb->temp_CCLabel,    Ghost::None, 0);
   t->requires(Task::OldDW, MIlb->vel_CCLabel,     Ghost::None, 0);
    
   t->computes(MIlb->cMassLabel);
   t->computes(MIlb->cVolumeLabel);
   t->computes(MIlb->vel_CCLabel);
   t->computes(MIlb->temp_CCLabel);
   t->computes( Mlb->doMechLabel);

   sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//
void MPMICE::scheduleComputeLagrangianValuesMPM(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSubset* one_matl,
                                   const MaterialSet* mpm_matls)
{

  cout_doing << "MPMICE::scheduleComputeLagrangianValuesMPM" << endl;

   /* interpolateNCToCC */

   Task* t=scinew Task("MPMICE::computeLagrangianValuesMPM",
                 this, &MPMICE::computeLagrangianValuesMPM);

   const MaterialSubset* mss = mpm_matls->getUnion();

   t->requires(Task::NewDW, Mlb->gVelocityStarLabel, mss, Ghost::AroundCells,1);
   t->requires(Task::NewDW, Mlb->gMassLabel,              Ghost::AroundCells,1);
   t->requires(Task::NewDW, Mlb->gTemperatureStarLabel,   Ghost::AroundCells,1);
   t->requires(Task::OldDW, MIlb->NC_CCweightLabel,       one_matl,
                                                          Ghost::AroundCells,1);
   t->requires(Task::NewDW, MIlb->cMassLabel,             Ghost::None);
   t->requires(Task::NewDW, MIlb->burnedMassCCLabel,      Ghost::None);
   t->requires(Task::NewDW, Ilb->int_eng_comb_CCLabel,    Ghost::None);
   t->requires(Task::NewDW, Ilb->mom_comb_CCLabel,        Ghost::None);
/*`==========TESTING==========*/
   t->requires(Task::NewDW, MIlb->temp_CCLabel,           Ghost::None);
   t->requires(Task::NewDW, MIlb->vel_CCLabel,            Ghost::None); 
/*==========TESTING==========`*/
   t->modifies(Ilb -> rho_CCLabel); 
   t->computes(Ilb -> mass_L_CCLabel);
   t->computes(Ilb -> mom_L_CCLabel);
   t->computes(Ilb -> int_eng_L_CCLabel);
   t->computes(MIlb-> NC_CCweightLabel, one_matl);

   sched->addTask(t, patches, mpm_matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleCCMomExchange(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSubset* ice_matls,
                               const MaterialSubset* mpm_matls,
                               const MaterialSet* all_matls)
{
  Task* t;
  cout_doing << "MPMICE::scheduleCCMomExchange" << endl;
  t=scinew Task("MPMICE::doCCMomExchange",
                this, &MPMICE::doCCMomExchange);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());
                                 // I C E
  t->computes(Ilb->mom_L_ME_CCLabel,      ice_matls);      
  t->computes(Ilb->int_eng_L_ME_CCLabel,  ice_matls);      

                                 // M P M
  t->computes(MIlb->dTdt_CCLabel,         mpm_matls); 
  t->computes(MIlb->dvdt_CCLabel,         mpm_matls); 
  t->computes(Ilb->mom_L_ME_CCLabel,      mpm_matls);      
  t->computes(Ilb->int_eng_L_ME_CCLabel,  mpm_matls);      

                                // A L L  M A T L S
  t->requires(Task::NewDW,  Ilb->mass_L_CCLabel,    Ghost::None); 
  t->requires(Task::NewDW,  Ilb->mom_L_CCLabel,     Ghost::None);
  t->requires(Task::NewDW,  Ilb->int_eng_L_CCLabel, Ghost::None);
  t->requires(Task::NewDW,  Ilb->sp_vol_CCLabel,    Ghost::None); 
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,  Ghost::None);
  
  if (d_ice->d_RateForm) {      //  R A T E   F O R M
    t->requires(Task::OldDW,  Ilb->temp_CCLabel,  ice_matls, Ghost::None);
    t->requires(Task::NewDW,  Ilb->temp_CCLabel,  mpm_matls, Ghost::None);
    t->computes(Ilb->Tdot_CCLabel);
  }
  sched->addTask(t, patches, all_matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateCCToNC(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* mpm_matls)
{

  cout_doing << "MPMICE::scheduleInterpolateCCToNC" << endl;

  Task* t=scinew Task("MPMICE::interpolateCCToNC",
                this, &MPMICE::interpolateCCToNC);
                
  const MaterialSubset* mss = mpm_matls->getUnion();
  t->modifies(             Mlb->gVelocityStarLabel, mss);
  t->modifies(             Mlb->gAccelerationLabel, mss);
  t->requires(Task::NewDW, MIlb->dTdt_CCLabel,           Ghost::AroundCells,1);
  t->requires(Task::NewDW, MIlb->dvdt_CCLabel,           Ghost::AroundCells,1);
  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->computes(Mlb->dTdt_NCLabel);

  sched->addTask(t, patches, mpm_matls);
}
/* ---------------------------------------------------------------------
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
  Task* t;
  if (d_ice->d_RateForm) {     // R A T E   F O R M
    cout_doing << "MPMICE::scheduleComputeRateFormPressure" << endl;
    t = scinew Task("MPMICE::computeRateFormPressure",
                     this, &MPMICE::computeRateFormPressure);
  }
  if (d_ice->d_EqForm) {       // E Q   F O R M
    cout_doing << "MPMICE::scheduleComputeEquilibrationPressure" << endl;
    t = scinew Task("MPMICE::computeEquilibrationPressure",
                    this, &MPMICE::computeEquilibrationPressure);
  }
                              // I C E
  t->requires(Task::OldDW,Ilb->temp_CCLabel,         ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->rho_CCLabel,          ice_matls, Ghost::None);
  t->requires(Task::OldDW,Ilb->sp_vol_CCLabel,       ice_matls, Ghost::None);

                              // M P M
  t->requires(Task::NewDW,MIlb->temp_CCLabel,        mpm_matls, Ghost::None);
  t->requires(Task::NewDW,MIlb->cVolumeLabel,        mpm_matls, Ghost::None);
  t->requires(Task::NewDW,MIlb->cMassLabel,          mpm_matls, Ghost::None);
 
  if (d_ice->d_EqForm) {      // E Q   F O R M
    t->requires(Task::OldDW,Ilb->press_CCLabel,      press_matl, Ghost::None);
    t->requires(Task::OldDW,Ilb->vel_CCLabel,        ice_matls, Ghost::None);
    t->requires(Task::NewDW,MIlb->vel_CCLabel,       mpm_matls, Ghost::None);
  }

                              //  A L L _ M A T L S
  if (d_ice->d_RateForm) {   
    t->computes(Ilb->matl_press_CCLabel);
    t->computes(Ilb->f_theta_CCLabel);
  }
  t->computes(Ilb->speedSound_CCLabel); 
  t->computes(Ilb->sp_vol_CCLabel);
  t->computes(Ilb->vol_frac_CCLabel);
  t->computes(Ilb->rho_CCLabel);
  t->computes(Ilb->press_equil_CCLabel, press_matl);

  sched->addTask(t, patches, all_matls);
}
/* ---------------------------------------------------------------------
 Function~  MPMICE::scheduleHEChemistry--
_____________________________________________________________________*/
void MPMICE::scheduleHEChemistry(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* react_matls,
                                    const MaterialSubset* prod_matls,
                                    const MaterialSubset* press_matl,
                                    const MaterialSet* all_matls)
{
  cout_doing << "MPMICE::scheduleHEChemistry" << endl;

  Task* t = scinew Task("MPMICE::HEChemistry",
                  this, &MPMICE::HEChemistry);
  
  t->requires(Task::OldDW, d_sharedState->get_delt_label());
  
  //__________________________________
  // Products
  t->requires(Task::OldDW, Ilb->temp_CCLabel,     prod_matls, Ghost::None);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, prod_matls, Ghost::None);
  if (prod_matls->size() > 0){
    t->requires(Task::NewDW, Ilb->press_equil_CCLabel, 
                                                  press_matl, Ghost::None);
  }
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matls, Ghost::None);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matls, Ghost::None);
  t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matls, Ghost::None);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matls, Ghost::None);
  t->requires(Task::NewDW, Mlb->gMassLabel,       react_matls,
                                                       Ghost::AroundCells,1);
  t->requires(Task::OldDW, Mlb->doMechLabel);

  t->computes(MIlb->burnedMassCCLabel);
  t->computes( Ilb->int_eng_comb_CCLabel); 
  t->computes( Ilb->created_vol_CCLabel);
  t->computes( Ilb->mom_comb_CCLabel);
  

  sched->addTask(t, patches, all_matls);
}
//______________________________________________________________________
//
void MPMICE::scheduleInterpolateMassBurnFractionToNC(SchedulerP& sched,
                                           const PatchSet* patches,
                                            const MaterialSet* mpm_matls)
{

  cout_doing << "MPMICE::scheduleInterpolateMassBurnFractionToNC" << endl;

  Task* t = scinew Task("MPMICE::interpolateMassBurnFractionToNC",
                  this, &MPMICE::interpolateMassBurnFractionToNC);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());  
  t->requires(Task::NewDW, MIlb->burnedMassCCLabel, Ghost::AroundCells,1);
  t->requires(Task::NewDW, MIlb->cMassLabel,        Ghost::AroundCells,1);

  t->computes(Mlb->massBurnFractionLabel);

  sched->addTask(t, patches, mpm_matls);
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
    cout_doing << "Doing Initialize on patch " << patch->getID() 
     << "\t\t\t MPMICE" << endl;

    
    NCVariable<double> NC_CCweight;
    new_dw->allocate(NC_CCweight,  MIlb->NC_CCweightLabel,    0, patch);
   //__________________________________
   // - Initialize NC_CCweight = 0.125
   // - Find the walls with symmetry BC and
   //   double NC_CCweight
   NC_CCweight.initialize(0.125);
   for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
      int mat_id = 0; 
      BoundCondBase *sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      if (sym_bcs != 0) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"NC_vars"); 
                                                  !iter.done(); iter++) {
          NC_CCweight[*iter] = 2.0*NC_CCweight[*iter];
        }
      }
    }
    //__________________________________
    //  Initialize CCVaribles for MPM Materials
    //  Even if mass = 0 in a cell you still need
    //  CC Variables defined.
   int numALL_matls = d_sharedState->getNumMatls();
   int numMPM_matls = d_sharedState->getNumMPMMatls();
   for (int m = 0; m < numMPM_matls; m++ ) {
      CCVariable<double> rho_micro, sp_vol_CC, rho_CC, Temp_CC;
      CCVariable<Vector> vel_CC;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int indx= mpm_matl->getDWIndex();
      new_dw->allocate(rho_micro,   Ilb->rho_micro_CCLabel,   indx,patch);
      new_dw->allocate(sp_vol_CC,   Ilb->sp_vol_CCLabel,      indx,patch);
      new_dw->allocate(rho_CC,      Ilb->rho_CCLabel,         indx,patch); 
      new_dw->allocate(Temp_CC,   MIlb->temp_CCLabel,         indx,patch);
      new_dw->allocate(vel_CC,    MIlb->vel_CCLabel,          indx,patch);
      
      mpm_matl->initializeCCVariables(rho_micro,   rho_CC,
                                      Temp_CC,     vel_CC,  
                                      numALL_matls,patch);  

      setBC(rho_CC,    "Density",      patch, d_sharedState, indx);    
      setBC(rho_micro, "Density",      patch, d_sharedState, indx);    
      setBC(Temp_CC,   "Temperature",  patch, d_sharedState, indx);    
      setBC(vel_CC,    "Velocity",     patch, indx);                   
      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[c] = 1.0/rho_micro[c];
      }
      //__________________________________
      //You may need to adjust Temp_CC if g!=0
      // for thermo consistency, for now ignore it 2/11/02
      
      new_dw->put(sp_vol_CC,    Ilb->sp_vol_CCLabel, indx,patch);        
      new_dw->put(rho_CC,       Ilb->rho_CCLabel,    indx,patch);        
      new_dw->put(Temp_CC,     MIlb->temp_CCLabel,   indx,patch);        
      new_dw->put(vel_CC,      MIlb->vel_CCLabel,    indx,patch); 
      
      //---- P R I N T   D A T A ------        
      if (d_ice->switchDebugInitialize){      
        ostringstream desc;
        desc << "MPMICE_Initialization_Mat_" << indx << "_patch_"
             << patch->getID();
        d_ice->printData(patch,   1, desc.str(), "rho_CC",      rho_CC);
        d_ice->printData(patch,   1, desc.str(), "rho_micro_CC",rho_micro);
        d_ice->printData(patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        d_ice->printData(patch,   1, desc.str(), "Temp_CC",     Temp_CC);
        d_ice->printVector(patch, 1, desc.str(), "uvel_CC", 0,  vel_CC);
        d_ice->printVector(patch, 1, desc.str(), "vvel_CC", 1,  vel_CC);
        d_ice->printVector(patch, 1, desc.str(), "wvel_CC", 2,  vel_CC);
      }          
    }  // num_MPM_matls loop  
    
    double doMech = 999.9;
    new_dw->put(delt_vartype(doMech), Mlb->doMechLabel);
    new_dw->put(NC_CCweight,  MIlb->NC_CCweightLabel,    0, patch);
  }
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

    cout_doing<<"Doing interpolatePressCCToPressNC on patch "<<patch->getID()
       <<"\t\t MPMICE" << endl;

    constCCVariable<double> pressCC;
    NCVariable<double> pressNC;

    new_dw->get(pressCC, Ilb->press_CCLabel, 0, patch,Ghost::AroundCells, 1);
    new_dw->allocate(pressNC, MIlb->press_NCLabel, 0, patch);
    pressNC.initialize(0.0);
    
    IntVector cIdx[8];
    // Interpolate CC pressure to nodes
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
       patch->findCellsFromNode(*iter,cIdx);
      for (int in=0;in<8;in++){
        pressNC[*iter]  += .125*pressCC[cIdx[in]];
      }
    }
    new_dw->put(pressNC,MIlb->press_NCLabel,0,patch);
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

    cout_doing<<"Doing interpolatePressureToParticles on patch "<<
      patch->getID()<<"\t\t MPMICE" << endl;

    constNCVariable<double> pressNC;
    IntVector ni[8];
    double S[8];
    IntVector cIdx[8];
    double p_ref = d_sharedState->getRefPress();
    new_dw->get(pressNC,MIlb->press_NCLabel,0,patch,Ghost::AroundCells,1);

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwindex, patch);
      ParticleVariable<double> pPressure;
      constParticleVariable<Point> px;

      new_dw->allocate(pPressure, Mlb->pPressureLabel,      pset);
      old_dw->get(px,             Mlb->pXLabel,             pset);

      // Interpolate NC pressure to particles
      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
        particleIndex idx = *iter;
        double press = 0.;

        // Get the node indices that surround the cell
        patch->findCellAndWeights(px[idx], ni, S);
        for (int k = 0; k < 8; k++) {
          press += pressNC[ni[k]] * S[k];
        }
        pPressure[idx] = press-p_ref;
      }

      constCCVariable<Vector> mom_source;
      constCCVariable<double> mass;
      NCVariable<Vector> gradPAccNC;
      new_dw->get(mom_source,       Ilb->mom_source_CCLabel, dwindex, patch,
                                                  Ghost::AroundCells, 1);
      new_dw->get(mass,             MIlb->cMassLabel,        dwindex, patch,
                                                  Ghost::AroundCells, 1);
      new_dw->allocate(gradPAccNC,  Mlb->gradPAccNCLabel,   dwindex, patch);
      gradPAccNC.initialize(Vector(0.,0.,0.));
      // Interpolate CC pressure gradient (mom_source) to nodes (gradP*dA*dt)
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        patch->findCellsFromNode(*iter,cIdx);
        for (int in=0;in<8;in++){
         gradPAccNC[*iter]+=(mom_source[cIdx[in]]/(mass[cIdx[in]]*delT))*.125;
         }
      }
      new_dw->put(pPressure,   Mlb->pPressureLabel);
      new_dw->put(gradPAccNC,  Mlb->gradPAccNCLabel, dwindex, patch);
    }
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

    cout_doing << "Doing interpolateNCToCC_0 on patch "<< patch->getID()
               <<"\t\t\t MPMICE" << endl;

    int numMatls = d_sharedState->getNumMPMMatls();
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z(); 
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel,  0, patch,
              Ghost::AroundCells, 1);

    int reactant_indx = -1;
    double thresholdTemperature = 1e6;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      if(mpm_matl->getRxProduct() == Material::reactant){
        reactant_indx = m;
        thresholdTemperature =
                     mpm_matl->getBurnModel()->getThresholdTemperature();
      }
    }
    double doMechNew = 999.9;
    delt_vartype doMechOld;
    old_dw->get(doMechOld, Mlb->doMechLabel);
    static int first_small_dt = 0;

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();

      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume, gtemperature;
      constNCVariable<Vector> gvelocity;
      CCVariable<double> cmass, cvolume,Temp_CC;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> Temp_CC_ice;
      constCCVariable<Vector> vel_CC_ice; 

      new_dw->allocate(cmass,     MIlb->cMassLabel,   indx, patch);    
      new_dw->allocate(cvolume,   MIlb->cVolumeLabel, indx, patch);
      new_dw->allocate(vel_CC,    MIlb->vel_CCLabel,  indx, patch);    
      new_dw->allocate(Temp_CC,   MIlb->temp_CCLabel, indx, patch);   

      double rho_orig = mpm_matl->getInitialDensity();
      double very_small_mass = d_SMALL_NUM * cell_vol;
      cmass.initialize(very_small_mass);
      cvolume.initialize( very_small_mass/rho_orig);
         
      new_dw->get(gmass,Mlb->gMassLabel,  indx, patch,Ghost::AroundCells, 1);
      new_dw->get(gvolume,      
                  Mlb->gVolumeLabel,      indx, patch,Ghost::AroundCells, 1);
      new_dw->get(gvelocity,    
                  Mlb->gVelocityLabel,    indx, patch,Ghost::AroundCells, 1);
      new_dw->get(gtemperature, 
                  Mlb->gTemperatureLabel, indx, patch,Ghost::AroundCells, 1);
      old_dw->get(Temp_CC_ice,  MIlb->temp_CCLabel, indx,patch,Ghost::None,0);    
      old_dw->get(vel_CC_ice,   MIlb->vel_CCLabel,  indx,patch,Ghost::None,0);
      IntVector nodeIdx[8];
      
      //---- P R I N T   D A T A ------ 
      if(switchDebug_InterpolateNCToCC_0) {
        char description[50];
        sprintf(description, "TOP_MPMICE::interpolateNCToCC_0_mat_%d_patch_%d ", 
                    indx, patch->getID());
        printData(     patch, 1,description, "gmass",       gmass);
        printData(     patch, 1,description, "gvolume",     gvolume);
        printData(     patch, 1,description, "gtemperatue", gtemperature);
        printNCVector( patch, 1,description, "gvelocity.X", 0, gvelocity);
        printNCVector( patch, 1,description, "gvelocity.Y", 1, gvelocity);
        printNCVector( patch, 1,description, "gvelocity.Z", 2, gvelocity);
      }
      //__________________________________
      //  compute CC Variables
      for(CellIterator iter =patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        patch->findNodesFromCell(*iter,nodeIdx);
 
        double Temp_CC_mpm = 0.0;       
        Vector vel_CC_mpm  = Vector(0.0, 0.0, 0.0);
        
        for (int in=0;in<8;in++){
          cmass[c]    += NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          cvolume[c]  += NC_CCweight[nodeIdx[in]] * gvolume[nodeIdx[in]];
          vel_CC_mpm  += gvelocity[nodeIdx[in]] *
                            NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          Temp_CC_mpm += gtemperature[nodeIdx[in]] *
                            NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
        } 
        vel_CC_mpm  /= cmass[c];    
        Temp_CC_mpm /= cmass[c]; 
        
        //__________________________________
        // set *_CC = to either vel/Temp_CC_ice or vel/Temp_CC_mpm
        // depending upon if there is cmass.  You need
        // a well defined vel/temp_CC even if there isn't any mass
        // If you change this you must also change MPMICE::computeLagrangianValuesMPM
        double one_or_zero = (cmass[c] - very_small_mass)/cmass[c];
        Temp_CC[c] = (1.0 - one_or_zero)* Temp_CC_ice[c] + one_or_zero * Temp_CC_mpm;
        vel_CC[c]  = (1.0 - one_or_zero)* vel_CC_ice[c]  + one_or_zero * vel_CC_mpm;   
      }

      //__________________________________
      // Compute doMechNew
      // Note when sumIntegral > 1 then Temp_CC > thresholdTemp
      double integral = 0.0;
      double sumIntegral = 0.0;  
      if(m == reactant_indx ) {
        for(CellIterator iter =patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          modf(Temp_CC[c]/thresholdTemperature, &integral);
          sumIntegral += integral; 
        }
        
        if(sumIntegral > 1.0 || doMechOld < 0 ) {
          doMechNew = -1.0;
          if(first_small_dt > 0){
            doMechNew = -2.0;
          }
        }
      } 
      //__________________________________
      //  Set BC's
      setBC(vel_CC,  "Velocity",   patch, indx);
      setBC(Temp_CC, "Temperature",patch, d_sharedState, indx);
      setBC(cmass,   "Density",    patch, d_sharedState, indx);
      setBC(cvolume, "Density",    patch, d_sharedState, indx); 
      
/*`==========TESTING==========*/
      //  Set if symmetric Boundary conditions
    #if 0
      setBC(cmass,   "set_if_sym_BC",patch, d_sharedState, indx);
      setBC(cvolume, "set_if_sym_BC",patch, d_sharedState, indx);
    #endif 
/*==========TESTING==========`*/
      
     //---- P R I N T   D A T A ------
     if(switchDebug_InterpolateNCToCC_0) {
        char description[50];
        sprintf(description, "BOT_MPMICE::interpolateNCToCC_0_Mat_%d_patch_%d ", 
                    indx, patch->getID());
        d_ice->printData(   patch, 1,description, "cmass",     cmass);
        d_ice->printData(   patch, 1,description, "cvolume",   cvolume);
        d_ice->printData(   patch, 1,description, "Temp_CC",   Temp_CC);
        d_ice->printVector( patch, 1,description, "uvel_CC", 0,vel_CC);
        d_ice->printVector( patch, 1,description, "uvel_CC", 1,vel_CC);
        d_ice->printVector( patch, 1,description, "uvel_CC", 2,vel_CC);
      } 
     
      new_dw->put(cmass,    MIlb->cMassLabel,   indx, patch);    
      new_dw->put(cvolume,  MIlb->cVolumeLabel, indx, patch);    
      new_dw->put(vel_CC,   MIlb->vel_CCLabel,  indx, patch);    
      new_dw->put(Temp_CC,  MIlb->temp_CCLabel, indx, patch);   
    }
    if(doMechNew < 0.){
       first_small_dt++;
    }
    new_dw->put(delt_vartype(doMechNew), Mlb->doMechLabel);
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
    cout_doing << "Doing computeLagrangianValuesMPM on patch "<< patch->getID()
               <<"\t\t\t MPMICE" << endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();
    double inv_cellVol = 1.0/cellVol;
    double very_small_mass = d_SMALL_NUM * cellVol; 
    
    constNCVariable<double> NC_CCweight;
    NCVariable<double>NC_CCweight_copy;
    new_dw->allocate(NC_CCweight_copy,MIlb->NC_CCweightLabel, 0,patch);
    old_dw->get(NC_CCweight,          MIlb->NC_CCweightLabel, 0,patch,
              Ghost::AroundCells, 1);
              
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int indx = mpm_matl->getDWIndex();

      // Create arrays for the grid data
      constNCVariable<double> gmass, gvolume,gtempstar;
      constNCVariable<Vector> gvelocity;
      CCVariable<Vector> cmomentum;
      CCVariable<double> int_eng_L, rho_CC, mass_L;              
      constCCVariable<double> burnedMassCC, int_eng_comb, cmass; 
      constCCVariable<Vector> mom_comb;
      constCCVariable<double> Temp_CC_sur;
      constCCVariable<Vector> vel_CC_sur; 
      new_dw->get(gmass,        
                  Mlb->gMassLabel,           indx,patch,Ghost::AroundCells,1);
      new_dw->get(gvelocity,    
                  Mlb->gVelocityStarLabel,   indx,patch,Ghost::AroundCells,1);
      new_dw->get(gtempstar,
                  Mlb->gTemperatureStarLabel,indx,patch,Ghost::AroundCells,1);
      new_dw->get(cmass,       
                  MIlb->cMassLabel,          indx,patch,Ghost::None,0);    
      new_dw->get(burnedMassCC,
                  MIlb->burnedMassCCLabel,   indx,patch,Ghost::None,0);    
      new_dw->get(int_eng_comb, 
                  Ilb->int_eng_comb_CCLabel, indx,patch,Ghost::None,0);    
      new_dw->get(mom_comb,     
                  Ilb->mom_comb_CCLabel,     indx,patch,Ghost::None,0);
      new_dw->get(Temp_CC_sur,  
                  MIlb->temp_CCLabel,        indx,patch,Ghost::None,0);    
      new_dw->get(vel_CC_sur,   
                  MIlb->vel_CCLabel,         indx,patch,Ghost::None,0);
                                                           
      new_dw->getModifiable(rho_CC,Ilb->rho_CCLabel,      indx,patch);   
      new_dw->allocate(mass_L,     Ilb->mass_L_CCLabel,   indx,patch);      
      new_dw->allocate(cmomentum,  Ilb->mom_L_CCLabel,    indx,patch);   
      new_dw->allocate(int_eng_L,  Ilb->int_eng_L_CCLabel,indx,patch);   

      cmomentum.initialize(Vector(0.0, 0.0, 0.0));
      int_eng_L.initialize(0.);
      mass_L.initialize(0.); 
      double cv = mpm_matl->getSpecificHeat();

      IntVector nodeIdx[8];

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebugLagrangianValues) {
         ostringstream desc;
         desc <<"TOP_MPMICE::computeLagrangianValuesMPM_mat_%d_patch_%d "<< 
                   indx<<patch->getID();
         d_ice->printData(patch,1,desc.str(), "cmass",    cmass);
         printData(     patch,  1,desc.str(), "gmass",    gmass);
         printData(     patch,  1,desc.str(), "gtemStar", gtempstar);
         printNCVector( patch,  1,desc.str(), "gvelocityStar.X", 0, gvelocity);
         printNCVector( patch,  1,desc.str(), "gvelocityStar.Y", 1, gvelocity);
         printNCVector( patch,  1,desc.str(), "gvelocityStar.Z", 2, gvelocity);
         
      }

      for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        patch->findNodesFromCell(*iter,nodeIdx);
        IntVector c = *iter;
        double int_eng_L_mpm = 0.0;
        double int_eng_L_sur = cmass[c] * Temp_CC_sur[c] * cv;
        Vector cmomentum_mpm = Vector(0.0, 0.0, 0.0);
        Vector cmomentum_sur = vel_CC_sur[c] * cmass[c];
        
        for (int in=0;in<8;in++){
          cmomentum_mpm +=gvelocity[nodeIdx[in]] *
                            NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
          int_eng_L_mpm +=gtempstar[nodeIdx[in]] * cv *
                            NC_CCweight[nodeIdx[in]] * gmass[nodeIdx[in]];
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
      if(d_ice->d_massExchange == false) {
        for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
         IntVector c = *iter;
         mass_L[c]    = cmass[c];
         rho_CC[c]    = mass_L[c] * inv_cellVol;
        }
      }

      //__________________________________
      //  REACTION
      // The reaction can't completely eliminate 
      //  all the mass, momentum and internal E.
      // If it does then we'll get erroneous vel,
      // and temps in CCMomExchange.  If the mass
      // goes to min_mass then cmomentum and int_eng_L
      // need to be scaled by min_mass to avoid inf temp and vel_CC
      // in 
      if(d_ice->d_massExchange)  { 
        for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          //  must have a minimum mass
          double min_mass = d_SMALL_NUM * cellVol;
          double inv_cmass = 1.0/cmass[c];
          mass_L[c] = std::max( (cmass[c] + burnedMassCC[c] ), min_mass);
          rho_CC[c] = mass_L[c] * inv_cellVol;
              
          //  must have a minimum momentum 
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons                        
            double min_mom_L = min_mass * cmomentum[c](dir) * inv_cmass;
            double mom_L_tmp = cmomentum[c](dir) + mom_comb[c](dir);

            // Preserve the original sign on momemtum     
            // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp+d_SMALL_NUM));

            mom_L_tmp = (mom_L_tmp/mass_L[c] ) *
                        (cmass[c] + burnedMassCC[c] );
                                
            cmomentum[c](dir) = plus_minus_one *
                                std::max( fabs(mom_L_tmp), fabs(min_mom_L) );
          }
          // must have a minimum int_eng   
          double min_int_eng = min_mass * int_eng_L[c] * inv_cmass;
          int_eng_L[c] = (int_eng_L[c]/mass_L[c]) * (cmass[c] + burnedMassCC[c]);     
          int_eng_L[c] = std::max((int_eng_L[c] - int_eng_comb[c]),min_int_eng);
        }
      }        
       
       //__________________________________
       //  Set if symmetric Boundary conditions
       setBC(cmomentum, "set_if_sym_BC",patch, indx);
       setBC(int_eng_L, "set_if_sym_BC",patch, d_sharedState, indx);

      //---- P R I N T   D A T A ------ 
      if(d_ice->switchDebugLagrangianValues) {
        char desc[50];
        sprintf(desc, "BOT_MPMICE::computeLagrangianValuesMPM_mat_%d_patch_%d ", 
                 indx, patch->getID());
        d_ice->printData(   patch, 1,desc, "rho_CC",    rho_CC);
        d_ice->printData(   patch, 1,desc, "int_eng_L", int_eng_L);
        d_ice->printVector( patch, 1,desc, "xmom_L_CC", 0, cmomentum);
        d_ice->printVector( patch, 1,desc, "ymom_L_CC", 1, cmomentum);
        d_ice->printVector( patch, 1,desc, "zmom_L_CC", 2, cmomentum);
      }
       new_dw->put(cmomentum,   Ilb->mom_L_CCLabel,    indx, patch);     
       new_dw->put(int_eng_L,   Ilb->int_eng_L_CCLabel,indx, patch);     
       new_dw->put(mass_L,      Ilb->mass_L_CCLabel,   indx, patch);     
    }
    //__________________________________
    // carry forward interpolation weight 
    IntVector low = patch->getNodeLowIndex();
    IntVector hi  = patch->getNodeHighIndex();
    NC_CCweight_copy.copyPatch(NC_CCweight, low,hi);
    new_dw->put(NC_CCweight_copy, MIlb->NC_CCweightLabel, 0,patch);
  }  //patches
}

//______________________________________________________________________
//
void MPMICE::doCCMomExchange(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t MPMICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<Vector> > dvdt_CC(numALLMatls);
    StaticArray<CCVariable<double> > dTdt_CC(numALLMatls);
    StaticArray<NCVariable<double> > dTdt_NC(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_temp(numALLMatls);

    vector<double> b(numALLMatls);
    vector<double> sp_vol(numALLMatls);
    vector<double> cv(numALLMatls);
    double tmp;
    DenseMatrix beta(numALLMatls,numALLMatls),acopy(numALLMatls,numALLMatls);
    DenseMatrix K(numALLMatls,numALLMatls),H(numALLMatls,numALLMatls);
    DenseMatrix a(numALLMatls,numALLMatls), a_inverse(numALLMatls,numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();

    d_ice->getExchangeCoefficients( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      if(mpm_matl){                 // M P M
        new_dw->allocate(vel_CC[m],  MIlb->vel_CC_scratchLabel,  indx, patch);
        new_dw->allocate(Temp_CC[m], MIlb->temp_CC_scratchLabel,indx, patch);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      if(ice_matl){                 // I C E
        new_dw->allocate(vel_CC[m], Ilb->vel_CCLabel,     indx, patch);
        new_dw->allocate(Temp_CC[m],Ilb->temp_CCLabel,    indx, patch); 
        cv[m] = ice_matl->getSpecificHeat();
      }                             // A L L  M A T L S
      new_dw->get(mass_L[m],        Ilb->mass_L_CCLabel,  indx, patch,
                                                   Ghost::None, 0); 
      new_dw->get(sp_vol_CC[m],     Ilb->sp_vol_CCLabel, indx, patch,
                                                   Ghost::None, 0);
      new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,     indx, patch,
                                                   Ghost::None, 0);
      new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel, indx, patch,
                                                   Ghost::None, 0);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,  indx, patch,
                                                   Ghost::None, 0);
      new_dw->allocate(dvdt_CC[m], MIlb->dvdt_CCLabel,      indx, patch);
      new_dw->allocate(dTdt_CC[m], MIlb->dTdt_CCLabel,      indx, patch);
      new_dw->allocate(mom_L_ME[m], Ilb->mom_L_ME_CCLabel,  indx, patch);
      new_dw->allocate(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,indx,patch);
      dvdt_CC[m].initialize(zero);
      dTdt_CC[m].initialize(0.);
    }
    //__________________________________
    //   R A T E   F O R M
    if(d_ice->d_RateForm) { 
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        int indx = matl->getDWIndex();
        if(mpm_matl){               
          new_dw->get(old_temp[m],Ilb->temp_CCLabel,indx,patch,Ghost::None,0);
        }
        if(ice_matl){               
          old_dw->get(old_temp[m],Ilb->temp_CCLabel,indx,patch,Ghost::None,0);
        }                            
        new_dw->allocate(Tdot[m], Ilb->Tdot_CCLabel,indx ,patch);
      }
    }

    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }
    //---- P R I N T   D A T A ------ 
    if (d_ice->switchDebugMomentumExchange_CC ) 
    {
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        char desc[50];
        sprintf(desc, "TOP_CCMomExchange_%d_patch_%d ",indx, patch->getID());
        d_ice->printData(   patch,1, desc,"Temp_CC",    Temp_CC[m]);     
        d_ice->printData(   patch,1, desc,"int_eng_L",  int_eng_L[m]);   
        d_ice->printData(   patch,1, desc,"mass_L",     mass_L[m]);      
        d_ice->printVector( patch,1, desc,"uvel_CC", 0, vel_CC[m]);      
        d_ice->printVector( patch,1, desc,"vvel_CC", 1, vel_CC[m]);      
        d_ice->printVector( patch,1, desc,"wvel_CC", 2, vel_CC[m]);      
      }
    }
 
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      //   Form BETA matrix (a), off diagonal terms
      //  The beta and (a) matrix is common to all momentum exchanges
      for(int m = 0; m < numALLMatls; m++)  {
        sp_vol[m]  = sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta[m][n] = delT * vol_frac_CC[n][c] * K[n][m]*sp_vol[m];
          a[m][n] = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a[m][m] = 1.;
        for(int n = 0; n < numALLMatls; n++) {
          a[m][m] +=  beta[m][n];
        }
      }
      matrixInverse(numALLMatls, a, a_inverse);
      
      //     X - M O M E N T U M  --  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++) {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
          b[m] += beta[m][n] * (vel_CC[n][c].x() - vel_CC[m][c].x());
        }
      }
      //     S O L V E
      //  - Add exchange contribution to orig value     
      vector<double> X(numALLMatls);
      multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++) {
        vel_CC[m][c].x( vel_CC[m][c].x() + X[m] );
        dvdt_CC[m][c].x( X[m]/delT );
      } 

      //     Y - M O M E N T U M  --   F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++) {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
          b[m] += beta[m][n] * (vel_CC[n][c].y() - vel_CC[m][c].y());
        }
      }

      //     S O L V E
      //  - Add exchange contribution to orig value
      multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++)  {
        vel_CC[m][c].y( vel_CC[m][c].y() + X[m] );
        dvdt_CC[m][c].y( X[m]/delT );
      }

      //     Z - M O M E N T U M  --  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;
        for(int n = 0; n < numALLMatls; n++) {
          b[m] += beta[m][n] * (vel_CC[n][c].z() - vel_CC[m][c].z());
        }
      }    

      //     S O L V E
      //  - Add exchange contribution to orig value
      multiplyMatrixAndVector(numALLMatls,a_inverse,b,X);
      for(int m = 0; m < numALLMatls; m++)  {
        vel_CC[m][c].z( vel_CC[m][c].z() + X[m] );
        dvdt_CC[m][c].z( X[m]/delT );
      }

      //---------- E N E R G Y   E X C H A N G E
      //         
      for(int m = 0; m < numALLMatls; m++) {
        tmp = sp_vol_CC[m][c] / cv[m];
        for(int n = 0; n < numALLMatls; n++)  {
          beta[m][n] = delT * vol_frac_CC[n][c] * H[n][m]*tmp;
          a[m][n] = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a[m][m] = 1.;
        for(int n = 0; n < numALLMatls; n++)   {
          a[m][m] +=  beta[m][n];
        }
      }
      // -  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;

       for(int n = 0; n < numALLMatls; n++) {
         b[m] += beta[m][n] * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
      }
      //     S O L V E, Add exchange contribution to orig value
      matrixSolver(numALLMatls,a,b,X);
      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = Temp_CC[m][c] + X[m];
        dTdt_CC[m][c] = X[m]/delT;
      }

    }  //end CellIterator loop

    //__________________________________
    //  Set the Boundary conditions 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      setBC(vel_CC[m], "Velocity",   patch,dwindex);
      setBC(Temp_CC[m],"Temperature",patch, d_sharedState, dwindex);
      
      //__________________________________
      //  Symetry BC dTdt: Neumann = 0
      //             dvdt: tangent components Neumann = 0
      //                   normal component negInterior
      if(mpm_matl){
        setBC(dTdt_CC[m], "set_if_sym_BC", patch, d_sharedState, dwindex);
        setBC(dvdt_CC[m], "set_if_sym_BC", patch, dwindex);
      }
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]          * mass_L[m][c];
      }
    }
    if(d_ice->d_RateForm) {         // RateForm
      for(CellIterator iter = patch->getExtraCellIterator(); 
                                                          !iter.done();iter++){
        IntVector c = *iter;
        for (int m = 0; m < numALLMatls; m++) {
          Tdot[m][c] = (Temp_CC[m][c] - old_temp[m][c])/delT;        
        }
      }
    }
    //---- P R I N T   D A T A ------ 
    if (d_ice->switchDebugMomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int dwindex = matl->getDWIndex();
        char description[50];;
        sprintf(description, "MPMICE_momExchange_CC_%d_patch_%d ", 
                dwindex, patch->getID());
        d_ice->printVector(patch,1, description, "xmom_L_ME", 0, mom_L_ME[m]);
        d_ice->printVector(patch,1, description, "ymom_L_ME", 1, mom_L_ME[m]);
        d_ice->printVector(patch,1, description, "zmom_L_ME", 2, mom_L_ME[m]);
        d_ice->printData(  patch,1, description,"int_eng_L_ME",int_eng_L_ME[m]);
        d_ice->printData(  patch,1, description, "dTdt_CC",       dTdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.X",  0, dvdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.Y",  1, dvdt_CC[m]);
        d_ice->printVector(patch,1, description, "dVdt_CC.Z",  2, dvdt_CC[m]);
      }
    }
    //__________________________________
    //    Put into new_dw
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){           // I C E
        new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    indx,patch);
        new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,indx,patch);
      }
      if(mpm_matl){           // M P M 
        new_dw->put(dvdt_CC[m],     MIlb->dvdt_CCLabel,       indx,patch);
        new_dw->put(dTdt_CC[m],     MIlb->dTdt_CCLabel,       indx,patch);
        new_dw->put(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,    indx,patch);
        new_dw->put(int_eng_L_ME[m],Ilb->int_eng_L_ME_CCLabel,indx,patch);        
      }
      if(d_ice->d_RateForm){ 
        new_dw->put(Tdot[m],        Ilb->Tdot_CCLabel,        indx,patch);
      }
    }  
  } //patches
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

    cout_doing << "Doing interpolateCCToNC on patch "<< patch->getID()
               <<"\t\t\t MPMICE" << endl;

    //__________________________________
    // This is where I interpolate the CC 
    // changes to NCs for the MPMMatls

    int numMPMMatls = d_sharedState->getNumMPMMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for (int m = 0; m < numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();
      constCCVariable<Vector> dvdt_CC;
      constCCVariable<double> dTdt_CC;
      NCVariable<Vector> gacceleration, gvelocity;

      NCVariable<double> dTdt_NC;

      new_dw->getModifiable(gvelocity, Mlb->gVelocityStarLabel,dwindex, patch);
      new_dw->getModifiable(gacceleration,Mlb->gAccelerationLabel,dwindex,
                         patch);
      new_dw->get(dvdt_CC,      MIlb->dvdt_CCLabel,     dwindex, patch,
                  Ghost::AroundCells,1);
      new_dw->get(dTdt_CC,      MIlb->dTdt_CCLabel,     dwindex, patch,
                  Ghost::AroundCells,1);      

      new_dw->allocate(dTdt_NC, Mlb->dTdt_NCLabel,        dwindex, patch);
      dTdt_NC.initialize(0.0);
      IntVector cIdx[8];

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        patch->findCellsFromNode(*iter,cIdx);
       for(int in=0;in<8;in++){
          gvelocity[*iter]     +=  dvdt_CC[cIdx[in]]*.125*delT;
          gacceleration[*iter] += (dvdt_CC[cIdx[in]])*.125;
          dTdt_NC[*iter]       += (dTdt_CC[cIdx[in]])*.125;
        }
      }
  
      //---- P R I N T   D A T A ------ 
      if(switchDebug_InterpolateCCToNC) {
        char description[50];
        sprintf(description, "BOT_MPMICE::interpolateCCToNC_mat_%d_patch_%d ", 
                    dwindex, patch->getID());
        printData(     patch, 1,description, "dTdt_NC",     dTdt_NC);
        printNCVector( patch, 1,description,"gvelocity.X",    0,gvelocity);
        printNCVector( patch, 1,description,"gvelocity.Y",    1,gvelocity);
        printNCVector( patch, 1,description,"gvelocity.Z",    2,gvelocity);
        printNCVector( patch, 1,description,"gacceleration.X",0,gacceleration);
        printNCVector( patch, 1,description,"gacceleration.Y",1,gacceleration);
        printNCVector( patch, 1,description,"gacceleration.Z",2,gacceleration);
      }
      new_dw->put(dTdt_NC,           Mlb->dTdt_NCLabel,       dwindex,patch);
    }  
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
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing<<"Doing computeEquilibrationPressure on patch "
              << patch->getID() <<"\t\t MPMICE" << endl;

    double    converg_coeff = 100.;
    double    convergence_crit = converg_coeff * DBL_EPSILON;
    double    sum, tmp;
    double press_ref= d_sharedState->getRefPress();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numALLMatls = numICEMatls + numMPMMatls;

    Vector dx       = patch->dCell(); 
    double cell_vol = dx.x()*dx.y()*dx.z();
    char warning[100];
    static int n_passes;                  
    n_passes ++; 

    StaticArray<double> press_eos(numALLMatls);
    StaticArray<double> dp_drho(numALLMatls),dp_de(numALLMatls);
    StaticArray<double> mat_volume(numALLMatls);
    StaticArray<double> cv(numALLMatls);
    
    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > rho_micro(numALLMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numALLMatls);
    StaticArray<CCVariable<double> > speedSound_new(numALLMatls);
    StaticArray<CCVariable<double> > speedSound(numALLMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls); 
    StaticArray<constCCVariable<double> > Temp(numALLMatls);
    StaticArray<constCCVariable<double> > mat_vol(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC_old(numALLMatls);
    StaticArray<constCCVariable<double> > mass_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numALLMatls);
    constCCVariable<double> press;    
    CCVariable<double> press_new, delPress_tmp;
    //  P R E S S 
    new_dw->allocate(press_new,    Ilb->press_equil_CCLabel, 0,patch);
    new_dw->allocate(delPress_tmp, Ilb->press_CCLabel,       0,patch); 
    old_dw->get(press, Ilb->press_CCLabel, 0,patch,Ghost::None, 0); 

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){                    // I C E
        old_dw->get(Temp[m],     Ilb->temp_CCLabel,  indx,patch,Ghost::None,0);
        old_dw->get(rho_CC_old[m],Ilb->rho_CCLabel,  indx,patch,Ghost::None,0);
        old_dw->get(sp_vol_CC[m],Ilb->sp_vol_CCLabel,indx,patch,Ghost::None,0);
        old_dw->get(vel_CC[m],   Ilb->vel_CCLabel,   indx,patch,Ghost::None,0);
        cv[m] = ice_matl->getSpecificHeat();
      }
      if(mpm_matl){                    // M P M
        new_dw->get(Temp[m],     MIlb->temp_CCLabel, indx,patch,Ghost::None,0);
        new_dw->get(mat_vol[m],  MIlb->cVolumeLabel, indx,patch,Ghost::None,0);
        new_dw->get(mass_CC[m],  MIlb->cMassLabel,   indx,patch,Ghost::None,0);
        new_dw->get(vel_CC[m],   MIlb->vel_CCLabel,  indx,patch,Ghost::None,0);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      new_dw->allocate(rho_CC_new[m],Ilb->rho_CCLabel,       indx, patch);
      new_dw->allocate(vol_frac[m],  Ilb->vol_frac_CCLabel,  indx, patch);
      new_dw->allocate(rho_micro[m], Ilb->rho_micro_CCLabel, indx, patch);
      new_dw->allocate(speedSound_new[m],Ilb->speedSound_CCLabel,indx,patch);
      new_dw->allocate(sp_vol_new[m],Ilb->sp_vol_CCLabel,    indx, patch);
      speedSound_new[m].initialize(0.0);
    }
    
    press_new.copyData(press);

    //__________________________________
    // Compute rho_micro, speedSound, volfrac, rho_CC
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      double total_mat_vol = 0.0;
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

        if(ice_matl){                // I C E
         rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
         double gamma   = ice_matl->getGamma(); 
         ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma,
                                         cv[m],Temp[m][c],
                                         press_eos[m],dp_drho[m],dp_de[m]);
         
         mat_volume[m] = (rho_CC_old[m][c]*cell_vol) * sp_vol_CC[m][c];

         tmp = dp_drho[m] + dp_de[m] * 
           (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
        } 

        if(mpm_matl){                //  M P M
  #ifdef EOSCM
          rho_micro[m][c] =  mpm_matl->getConstitutiveModel()->
            computeRhoMicroCM(press_new[c],press_ref, mpm_matl);

          mpm_matl->getConstitutiveModel()->
            computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                              dp_drho[m], tmp,mpm_matl);
  #endif
          mat_volume[m] = mat_vol[m][c];

  //    This is the IDEAL GAS stuff
  #ifdef IDEAL_GAS
          double gamma   = mpm_matl->getGamma();
          rho_micro[m][c] = mpm_matl->
            getConstitutiveModel()->computeRhoMicro(press_new[c],gamma,
                                               cv[m],Temp[m][c]);
          mpm_matl->getConstitutiveModel()->
            computePressEOS(rho_micro[m][c],gamma,cv[m],Temp[m][c],
                          press_eos[m],dp_drho[m], dp_de[m]);

          tmp = dp_drho[m] + dp_de[m] *
            (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
  #endif   
        }
/*`==========TESTING==========*/
    //  speedSound_new[m][c] = sqrt(tmp)/gamma[m];  // Isothermal speed of sound
        speedSound_new[m][c] = sqrt(tmp);           // Isentropic speed of sound
/*==========TESTING==========`*/        
        total_mat_vol += mat_volume[m];
      }  // numAllMatls loop

      for (int m = 0; m < numALLMatls; m++) {
        vol_frac[m][c] = mat_volume[m]/total_mat_vol;
        rho_CC_new[m][c] = vol_frac[m][c]*rho_micro[m][c];
      }
    }  // cell iterator
  //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_EQ_RF_press)  {
        char description[50];
        sprintf(description, "TOP_equilibration_patch_%d ", patch->getID());
        d_ice->printData( patch, 1, description, "Press_CC_top", press); 

        for (int m = 0; m < numALLMatls; m++)  {
          Material* matl = d_sharedState->getMaterial( m );
          int dwindex = matl->getDWIndex();
          sprintf(description, "TOP_equilibration_Mat_%d_patch_%d ", 
                   dwindex, patch->getID());
          d_ice->printData( patch,1,description, "rho_CC_new", rho_CC_new[m]);
          d_ice->printData( patch,1,description, "rho_micro",  rho_micro[m]);
          d_ice->printData( patch,0,description, "speedSound",speedSound_new[m]);
          d_ice->printData( patch,1,description, "Temp_CC",    Temp[m]);
          d_ice->printData( patch,1,description, "vol_frac_CC",vol_frac[m]);
        }
      }
  //______________________________________________________________________
  // Done with preliminary calcs, now loop over every cell
    int count, test_max_iter = 0;
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;  
      int i = c.x(), j = c.y(), k = c.z();
      double delPress = 0.;
      bool converged  = false;
      count           = 0;
      while ( count < d_ice->d_max_iter_equilibration && converged == false) {
        count++;
        double A = 0.;
        double B = 0.;
        double C = 0.;
        //__________________________________
       // evaluate press_eos at cell i,j,k
       for (int m = 0; m < numALLMatls; m++)  {
         Material* matl = d_sharedState->getMaterial( m );
         ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
         MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
         if(ice_matl){
            double gamma = ice_matl->getGamma();

            ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma,
                                             cv[m],Temp[m][c],
                                             press_eos[m], dp_drho[m],dp_de[m]);
         }
         if(mpm_matl){
          //__________________________________
          //  Hardwire for an ideal gas
  #ifdef EOSCM
            mpm_matl->getConstitutiveModel()->
                 computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                   dp_drho[m], tmp,mpm_matl);
  #endif
  //    This is the IDEAL GAS stuff
  #ifdef IDEAL_GAS
            double gamma = mpm_matl->getGamma();
            mpm_matl->getConstitutiveModel()->
              computePressEOS(rho_micro[m][c],gamma, cv[m],Temp[m][c],
                           press_eos[m], dp_drho[m], dp_de[m]);
  #endif

         }
       }

       //__________________________________
       // - compute delPress
       // - update press_CC     
       StaticArray<double> Q(numALLMatls),y(numALLMatls);     
       for (int m = 0; m < numALLMatls; m++)   {
         Q[m] =  press_new[c] - press_eos[m];
         y[m] =  dp_drho[m] * ( rho_CC_new[m][c]/
                 (vol_frac[m][c] * vol_frac[m][c]) ); 
         A   +=  vol_frac[m][c];
         B   +=  Q[m]/(y[m] + d_SMALL_NUM);
         C   +=  1.0/(y[m]  + d_SMALL_NUM);
       } 
       double vol_frac_not_close_packed = 1.;
       delPress = (A - vol_frac_not_close_packed - B)/C;

       press_new[c] += delPress;

       //__________________________________
       // backout rho_micro_CC at this new pressure
       for (int m = 0; m < numALLMatls; m++) {
         Material* matl = d_sharedState->getMaterial( m );
         ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
         MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
         if(ice_matl){
           double gamma = ice_matl->getGamma();

           rho_micro[m][c] = 
             ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma,
                                               cv[m],Temp[m][c]);
         }
         if(mpm_matl){
  #ifdef EOSCM
           rho_micro[m][c] =  
             mpm_matl->getConstitutiveModel()->computeRhoMicroCM(
                                          press_new[c],press_ref,mpm_matl);
  #endif
  //    This is the IDEAL GAS stuff
  #ifdef IDEAL_GAS
           double gamma = mpm_matl->getGamma();
           rho_micro[m][c] = 
           mpm_matl->getConstitutiveModel()->computeRhoMicro(press_new[c],
                                     gamma, cv[m],Temp[m][c]);
  #endif
         }
       }
       //__________________________________
       // - compute the updated volume fractions
       for (int m = 0; m < numALLMatls; m++)  {
         vol_frac[m][c]   = rho_CC_new[m][c]/rho_micro[m][c];
       }
       //__________________________________
       // Find the speed of sound at ijk
       // needed by eos and the the explicit
       // del pressure function
       for (int m = 0; m < numALLMatls; m++)  {
         Material* matl = d_sharedState->getMaterial( m );
         ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
         MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
         if(ice_matl){
           double gamma = ice_matl->getGamma();
           ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma,
                                              cv[m],Temp[m][c],
                                              press_eos[m],dp_drho[m],dp_de[m]);

           tmp = dp_drho[m] + dp_de[m] * 
                      (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
         }
         if(mpm_matl){
  #ifdef EOSCM
            mpm_matl->getConstitutiveModel()->
                 computePressEOSCM(rho_micro[m][c],press_eos[m],press_ref,
                                   dp_drho[m],tmp,mpm_matl);
  #endif
  //    This is the IDEAL GAS stuff
  #ifdef IDEAL_GAS
           double gamma = mpm_matl->getGamma();
           mpm_matl->getConstitutiveModel()->
               computePressEOS(rho_micro[m][c],gamma,
                                            cv[m],Temp[m][c],
                                            press_eos[m],dp_drho[m], dp_de[m]);
           tmp = dp_drho[m] + dp_de[m] * 
                      (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
  #endif
         }
/*`==========TESTING==========*/
     //  speedSound_new[m][c] = sqrt(tmp)/gamma[m];// Isothermal speed of sound  
         speedSound_new[m][c] = sqrt(tmp);         // Isentropic speed of sound  
/*==========TESTING==========`*/
       }
       //__________________________________
       // - Test for convergence 
       //  If sum of vol_frac_CC ~= 1.0 then converged 
       sum = 0.0;
       for (int m = 0; m < numALLMatls; m++)  {
         sum += vol_frac[m][c];
       }
       if (fabs(sum-1.0) < convergence_crit)
         converged = true;
     }   // end of converged

      delPress_tmp[c] = delPress;

      test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      if(test_max_iter == d_ice->d_max_iter_equilibration) {
          sprintf(warning, 
          " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
          i,j,k,count,n_passes);
          d_ice->Message(1,"calc_equilibration_press:",
              " Maximum number of iterations was reached ", warning);
      }

      for (int m = 0; m < numALLMatls; m++) {
           ASSERT(( vol_frac[m][c] > 0.0 ) ||
                  ( vol_frac[m][c] < 1.0));
      }
      if ( fabs(sum - 1.0) > convergence_crit)   {
          sprintf(warning, 
          " cell[%d][%d][%d], iter %d, n_passes %d,Now exiting ",
          i,j,k,count,n_passes);
          d_ice ->Message(1,"calc_equilibration_press:",
              " sum(vol_frac_CC) != 1.0", warning);
      }

      if ( press_new[c] < 0.0 )   {
          sprintf(warning, 
          " cell[%d][%d][%d], iter %d, n_passes %d, Now exiting",
           i,j,k, count, n_passes);
         d_ice->Message(1,"calc_equilibration_press:", 
              " press_new[iter*] < 0", warning);
      }

      for (int m = 0; m < numALLMatls; m++)
      if ( rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) {
          sprintf(warning, 
          " cell[%d][%d][%d], mat %d, iter %d, n_passes %d,Now exiting ",
          i,j,k,m,count,n_passes);
          cout << "r_m " << rho_micro[m][c] << endl;
          cout << "v_f " << vol_frac[m][c]  << endl;
          cout << "tmp " << Temp[m][c]      << endl;
          cout << "p_n " << press_new[c]    << endl;
          cout << "m_v " << mat_volume[m]       << endl;
          d_ice->Message(1," calc_equilibration_press:", 
              " rho_micro < 0 || vol_frac < 0", warning);
      }
    }     // end of cell interator

    cout_norm<<"max number of iterations in any cell \t"<<test_max_iter<<endl;

    //__________________________________
    // Now change how rho_CC is defined to 
    // rho_CC = mass/cell_volume  NOT mass/mat_volume 
    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){
        rho_CC_new[m].copyData(rho_CC_old[m]);
      }
      if(mpm_matl){
        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          rho_CC_new[m][c] = mass_CC[m][c]/cell_vol;
        }
      }
    }
    
    setBC(press_new,rho_micro[SURROUND_MAT],
          "rho_micro", "Pressure", patch, d_sharedState, 0, new_dw);
    
    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numALLMatls; m++)   {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c];
      }
    }  
    //__________________________________
    //    Put all matls into new dw
    for (int m = 0; m < numALLMatls; m++)   {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put( vol_frac[m],      Ilb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( speedSound_new[m],Ilb->speedSound_CCLabel, indx, patch);
      new_dw->put( sp_vol_new[m],    Ilb->sp_vol_CCLabel,     indx, patch); 
      new_dw->put( rho_CC_new[m],    Ilb->rho_CCLabel,        indx, patch);
    }
    new_dw->put(press_new,Ilb->press_equil_CCLabel,0,patch);


  //---- P R I N T   D A T A ------
    if(d_ice -> switchDebug_EQ_RF_press)  { 
      char description[50];
      sprintf(description, "BOT_equilibration_patch_%d ",patch->getID());
      d_ice->printData( patch, 1, description, "Press_CC_equil", press_new);
      d_ice->printData( patch, 1, description, "delPress",       delPress_tmp);
   #if 0                 
      for (int m = 0; m < numALLMatls; m++)  {
         Material* matl = d_sharedState->getMaterial( m );
         int dwindex = matl->getDWIndex(); 
         sprintf(description, "BOT_equilibration_Mat_%d_patch_%d ", 
                  dwindex,patch->getID());
         d_ice->printData( patch,1,description, "rho_CC",      rho_CC_new[m]);
         d_ice->printData( patch,1,description, "rho_micro_CC",rho_micro[m]);
         d_ice->printData( patch,1,description, "vol_frac_CC", vol_frac[m]);
      }
    #endif
    }
  }  //patches
}
/* --------------------------------------------------------------------- 
 Function~  MPMICE::HEChemistry--
 Steps:   
    - Pull out temp_CC(matl 0) and press data from ICE
    - Loop over all the mpm matls and compute heat and mass released
    - Put the heat and mass into ICE matl (0).
_____________________________________________________________________*/ 
void MPMICE::HEChemistry(const ProcessorGroup*,

                             const PatchSubset* patches,
                             const MaterialSubset* ,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing HEChemistry on patch "<< patch->getID()
               <<"\t\t\t\t MPMICE" << endl;
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    int numALLMatls=d_sharedState->getNumMatls();
    StaticArray<CCVariable<double> > burnedMass(numALLMatls);
    StaticArray<CCVariable<double> > createdVol(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_react(numALLMatls); 
    StaticArray<CCVariable<Vector> > mom_comb(numALLMatls);
    
    constCCVariable<double> gasPressure;
    constCCVariable<double> gasTemperature;
    constCCVariable<double> gasVolumeFraction;
    
    CCVariable<double>      sumBurnedMass;
    CCVariable<double>      sumCreatedVol;
    CCVariable<double>      sumReleasedHeat;
    CCVariable<Vector>      sumMom_comb;
    
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> solidTemperature;
    constCCVariable<double> solidMass;
    constNCVariable<double> NCsolidMass;
    constCCVariable<double> sp_vol_CC; 
    
    double surfArea, delX, delY, delZ;
    Vector dx;
    dx = patch->dCell();
    delX = dx.x();
    delY = dx.y();
    delZ = dx.z();
    int prod_indx = -1;
    
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      //__________________________________
      //  if no reaction 
      //  burnedMass, createdVol, int_eng_comb
      //  must still be allocated and initialized = 0,
      //  other tasks depend on them.
      new_dw->allocate(burnedMass[m],   MIlb->burnedMassCCLabel,   indx,patch);
      new_dw->allocate(createdVol[m],    Ilb->created_vol_CCLabel, indx,patch);
      new_dw->allocate(int_eng_react[m], Ilb->int_eng_comb_CCLabel,indx,patch);  
      new_dw->allocate(mom_comb[m],      Ilb->mom_comb_CCLabel,    indx,patch);  
            
      burnedMass[m].initialize(0.0);
      createdVol[m].initialize(0.0);
      int_eng_react[m].initialize(0.0); 
      mom_comb[m].initialize(0.0);
      //__________________________________
      // Pull out products data, should be only
      // 1 product matl
      if (ice_matl && (ice_matl->getRxProduct() == Material::product)){
        prod_indx = ice_matl->getDWIndex();
        new_dw->get(gasPressure,          Ilb->press_equil_CCLabel,
                                      0, patch, Ghost::None,0);
        old_dw->get(gasTemperature,       Ilb->temp_CCLabel,
                                      prod_indx, patch, Ghost::None,0);
        new_dw->get(gasVolumeFraction,    Ilb->vol_frac_CCLabel,
                                      prod_indx, patch, Ghost::None,0);
        new_dw->allocate(sumBurnedMass,  MIlb->burnedMassCCLabel,   
                                      prod_indx, patch);
        new_dw->allocate(sumCreatedVol,   Ilb->created_vol_CCLabel,  
                                      prod_indx, patch);
        new_dw->allocate(sumReleasedHeat, Ilb->int_eng_comb_CCLabel, 
                                      prod_indx, patch);
        new_dw->allocate(sumMom_comb,     Ilb->mom_comb_CCLabel, 
                                      prod_indx, patch);
        sumBurnedMass.initialize(0.0);
        sumCreatedVol.initialize(0.0);
        sumReleasedHeat.initialize(0.0);
        sumMom_comb.initialize(0.0);
      }
      //__________________________________
      // Pull out reactant data
      if(mpm_matl && (mpm_matl->getRxProduct() == Material::reactant))  {
        int react_indx = mpm_matl->getDWIndex();  
        new_dw->get(solidTemperature,    MIlb->temp_CCLabel,  
                                      react_indx, patch, Ghost::None,0);
        new_dw->get(solidMass,           MIlb->cMassLabel,    
                                      react_indx, patch, Ghost::None,0);
        new_dw->get(sp_vol_CC,            Ilb->sp_vol_CCLabel,
                                      react_indx, patch, Ghost::None,0);
        new_dw->get(vel_CC,               MIlb->vel_CCLabel,  
                                      react_indx, patch, Ghost::None,0); 
        new_dw->get(NCsolidMass,          Mlb->gMassLabel,    
                                      react_indx, patch, Ghost::AroundCells, 1);
      }
    }
    
    IntVector nodeIdx[8];
    //__________________________________
    // M P M  matls
    // compute the burned mass and released Heat
    // if burnModel != null  && material == reactant
    double total_created_vol = 0;
    for(int m = 0; m < numALLMatls; m++) {  
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl); 

      if(mpm_matl && (mpm_matl->getRxProduct() == Material::reactant))  {
        double delt = delT;
        double cv_solid = mpm_matl->getSpecificHeat();
        
        delt_vartype doMech;
        old_dw->get(doMech, Mlb->doMechLabel);

       if(doMech < -1.5){
        for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
         // Find if the cell contains surface:
         
         patch->findNodesFromCell(*iter,nodeIdx);
         
         double MaxMass = NCsolidMass[nodeIdx[0]];
         double MinMass = NCsolidMass[nodeIdx[0]];
         for (int nodeNumber=0; nodeNumber<8; nodeNumber++) {
           if (NCsolidMass[nodeIdx[nodeNumber]]>MaxMass) {
             MaxMass = NCsolidMass[nodeIdx[nodeNumber]];
           }
           if (NCsolidMass[nodeIdx[nodeNumber]]<MinMass) {
             MinMass = NCsolidMass[nodeIdx[nodeNumber]];
           }
         }         

         double Temperature = 0;
         if (gasVolumeFraction[c] < 1.e-5) 
            Temperature = solidTemperature[c];
          else Temperature = gasTemperature[c];

         /* Here is the new criterion for the surface:
            if (MnodeMax - MnodeMin) / Mcell > 0.5 - consider it a surface
         */
/*`==========TESTING==========*/
         #ifdef FAKE_BURN_MODEL
            MinMass = 0.0;   
         #endif
/*==========TESTING==========`*/
         if ((MaxMass-MinMass)/MaxMass > 0.9){
// && (MaxMass-MinMass)/MaxMass < 0.99999) 

             double gradRhoX, gradRhoY, gradRhoZ;
             double normalX,  normalY,  normalZ;

             gradRhoX = 0.25 * (( NCsolidMass[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[3]] ) -
                              ( NCsolidMass[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[5]]+
                                NCsolidMass[nodeIdx[6]]+
                                NCsolidMass[nodeIdx[7]] )) / delX;
             gradRhoY = 0.25 * (( NCsolidMass[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[5]] ) -
                              ( NCsolidMass[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[3]]+
                                NCsolidMass[nodeIdx[6]]+
                                NCsolidMass[nodeIdx[7]] )) / delY;
             gradRhoZ = 0.25 * (( NCsolidMass[nodeIdx[1]]+
                                NCsolidMass[nodeIdx[3]]+
                                NCsolidMass[nodeIdx[5]]+
                                NCsolidMass[nodeIdx[7]] ) -
                              ( NCsolidMass[nodeIdx[0]]+
                                NCsolidMass[nodeIdx[2]]+
                                NCsolidMass[nodeIdx[4]]+
                                NCsolidMass[nodeIdx[6]] )) / delZ;

             double absGradRho = sqrt(gradRhoX*gradRhoX +
                                   gradRhoY*gradRhoY +
                                   gradRhoZ*gradRhoZ );

             normalX = gradRhoX/absGradRho;
             normalY = gradRhoY/absGradRho;
             normalZ = gradRhoZ/absGradRho;

             double TmpX, TmpY, TmpZ;

             TmpX = normalX*delX;
             TmpY = normalY*delY;
             TmpZ = normalZ*delZ;

             TmpX = fabs(TmpX);
             TmpY = fabs(TmpY);
             TmpZ = fabs(TmpZ);

             surfArea = delX*delY*delZ / (TmpX+TmpY+TmpZ); 
             
             // compute the enthalpy of reaction and burnedMass         
             matl->getBurnModel()->computeBurn(Temperature,
                                          gasPressure[c],
                                          solidMass[c],
                                          solidTemperature[c],
                                          burnedMass[m][c],
                                          sumReleasedHeat[c],
                                          delt, surfArea);
                             
/*`==========FAKE BURN MODEL==========*/
      #ifdef FAKE_BURN_MODEL
               burnedMass[m][c] = solidMass[c]/10.0;
               sumReleasedHeat[c] = 0.0;
      #endif
/*===================================`*/
            
             int_eng_react[m][c] =
                      cv_solid*solidTemperature[c]*burnedMass[m][c];
             createdVol[m][c]    =  burnedMass[m][c] * sp_vol_CC[c];
             mom_comb[m][c]      = -vel_CC[c] * burnedMass[m][c];

             sumBurnedMass[c]   += burnedMass[m][c];
             sumReleasedHeat[c] += int_eng_react[m][c];
             sumCreatedVol[c]   += createdVol[m][c];
             sumMom_comb[c]     += -mom_comb[m][c];
             total_created_vol  += createdVol[m][c];
             burnedMass[m][c]    = -burnedMass[m][c];
             // reactantants: (-)burnedMass
             // int_eng_react  = change in internal energy of the reactants
             //                  
             // products:        (+)burnedMass
             // sumReleasedHeat=  enthalpy of reaction (Q) + change in internal 
             //                   energy of reactants
             // Need the proper sign on burnedMass in ICE::DelPress calc

             // We've gotten all the use we need out of createdVol by
             // accumulating it in sumCreatedVol
             createdVol[m][c]    = 0.0;
         }
         else {
            burnedMass[m][c]      = 0.0;
            int_eng_react[m][c]   = 0.0;
            createdVol[m][c]      = 0.0;
         }  // if (maxMass-MinMass....)
        }  // cell iterator  
       }  // if(doMech)
      }  // if(mpm_matl == reactant)
     }  // numALLMatls loop
    
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if (ice_matl && (ice_matl->getRxProduct() == Material::product)) {
        new_dw->put(sumBurnedMass,  MIlb->burnedMassCCLabel,   indx, patch);
        new_dw->put(sumReleasedHeat, Ilb->int_eng_comb_CCLabel,indx, patch); 
        new_dw->put(sumCreatedVol,   Ilb->created_vol_CCLabel, indx, patch);
        new_dw->put(sumMom_comb,     Ilb->mom_comb_CCLabel,    indx, patch);
      }
      else{
        new_dw->put(burnedMass[m],   MIlb->burnedMassCCLabel,  indx, patch);
        new_dw->put(int_eng_react[m],Ilb->int_eng_comb_CCLabel,indx, patch); 
        new_dw->put(createdVol[m],    Ilb->created_vol_CCLabel,indx, patch);
        new_dw->put(mom_comb[m],      Ilb->mom_comb_CCLabel,   indx, patch);
      }
    }
    //---- P R I N T   D A T A ------ 
    #if 0  //turn off for quality control testing 
    for(int m = 0; m < numALLMatls; m++) {
 //     if (d_ice->switchDebugSource_Sink) 
      {                                         
        Material* matl = d_sharedState->getMaterial( m );                          
        int indx = matl->getDWIndex();                                             
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);                  
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);                  
        char desc[50];                                                             
        if(ice_matl) {                                                             
          sprintf(desc,"ICEsources/sinks_Mat_%d_patch_%d",indx,patch->getID());    
          d_ice->printData( patch,0,desc,"SumburnedMass",  sumBurnedMass);      
          d_ice->printData( patch,0,desc,"sumReleasedHeat",sumReleasedHeat);    
          d_ice->printData( patch,0,desc,"sumCreatedVol",  sumCreatedVol);  
          d_ice->printVector( patch,0,desc,"sum_XMom_comb", 0, sumMom_comb);
          d_ice->printVector( patch,0,desc,"sum_YMom_comb", 1, sumMom_comb);
          d_ice->printVector( patch,0,desc,"sum_ZMom_comb", 2, sumMom_comb);    
        }                                                                          
        if(mpm_matl) {                                                             
          sprintf(desc,"MPMsources/sinks_Mat_%d_patch_%d",indx,patch->getID());    
          d_ice->printData( patch, 0, desc,"burnedMass",   burnedMass[m]);         
          d_ice->printData( patch, 0, desc,"int_eng_react",int_eng_react[m]);      
          d_ice->printData( patch, 0, desc,"createdVol",   createdVol[m]); 
          d_ice->printVector( patch, 0, desc,"mom_Xcomb", 0, mom_comb[m]);  
          d_ice->printVector( patch, 0, desc,"mom_Ycomb", 1, mom_comb[m]);
          d_ice->printVector( patch, 0, desc,"mom_Zcomb", 2, mom_comb[m]);      
        }                                                                          
      }                                                                            
    }
    #endif
  }  // patches
}
//______________________________________________________________________
//
void MPMICE::interpolateMassBurnFractionToNC(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing interpolateMassBurnFractionToNC on patch "
               << patch->getID() <<"\t MPMICE" << endl;

    // Interpolate the CC burn fraction to the nodes

    int numALLMatls = d_sharedState->getNumMPMMatls() + 
      d_sharedState->getNumICEMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int dwindex = matl->getDWIndex();
      if(mpm_matl){
        constCCVariable<double> burnedMassCC;
        constCCVariable<double> massCC;
        NCVariable<double> massBurnFraction;
        new_dw->get(burnedMassCC,     MIlb->burnedMassCCLabel,dwindex,patch,
                                                  Ghost::AroundCells,1);
        new_dw->get(massCC,           MIlb->cMassLabel,       dwindex,patch,
                                                  Ghost::AroundCells,1);
        new_dw->allocate(massBurnFraction,
                                  Mlb->massBurnFractionLabel,dwindex,patch);

        IntVector cIdx[8];
        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
           patch->findCellsFromNode(*iter,cIdx);
          massBurnFraction[*iter]         = 0.0;
          for (int in=0;in<8;in++){
            massBurnFraction[*iter] +=
                     (fabs(burnedMassCC[cIdx[in]])/massCC[cIdx[in]])*.125;

          }
        }
        new_dw->put(massBurnFraction,Mlb->massBurnFractionLabel, dwindex,patch);
      }  //if(mpm_matl)
    }  //ALLmatls  
  }  //patches
}
