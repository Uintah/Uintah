#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/MathToolbox.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <vector>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/StaticArray.h>
#include <sstream>
#include <float.h>
#include <iostream>
#include <Core/Util/DebugStream.h>


using std::vector;
using std::max;
using std::min;
using std::istringstream;
 
using namespace SCIRun;
using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "ICE_NORMAL_COUT:+,ICE_DOING_COUT:+"
//  ICE_NORMAL_COUT:  dumps out during problemSetup 
//  ICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF
static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);

//#define ANNULUSICE
#undef ANNULUSICE

ICE::ICE(const ProcessorGroup* myworld) 
  : UintahParallelComponent(myworld)
{
  lb   = scinew ICELabel();

  // Turn off all the debuging switches
  switchDebugInitialize           = false;
  switchDebug_EQ_RF_press         = false;
  switchDebug_vel_FC              = false;
  switchDebug_PressDiffRF         = false;
  switchDebug_Exchange_FC         = false;
  switchDebug_explicit_press      = false;
  switchDebug_PressFC             = false;
  switchDebugLagrangianValues     = false;
  switchDebugMomentumExchange_CC  = false;
  switchDebugSource_Sink          = false;
  switchDebug_advance_advect      = false;
  switchDebug_advectQFirst        = false;
  switchTestConservation          = false;

  d_massExchange = false;
  d_RateForm     = false;
  d_EqForm       = false;
  
}

ICE::~ICE()
{
  delete lb;
  delete d_advector;

}

/* ---------------------------------------------------------------------
 Function~  ICE::problemSetup--
_____________________________________________________________________*/
void  ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                        SimulationStateP&   sharedState)
{
  d_sharedState = sharedState;
  d_SMALL_NUM   = 1.e-100;
  cout_norm << "In the preprocessor . . ." << endl;
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(dataArchiver == 0){
    cout<<"dataArhiver in ICE is null now exiting; "<<endl;
    exit(1);
  }
  //__________________________________
  // Find the switches
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {
    d_dbgStartTime = 0.;
    d_dbgStopTime  = 1.;
    d_dbgOutputInterval = 0.0;
    d_dbgBeginIndx = IntVector(0,0,0);
    d_dbgEndIndx   = IntVector(0,0,0);
     
    debug_ps->get("dbg_timeStart",     d_dbgStartTime);
    debug_ps->get("dbg_timeStop",      d_dbgStopTime);
    debug_ps->get("dbg_outputInterval",d_dbgOutputInterval);
    debug_ps->get("d_dbgBeginIndx",    d_dbgBeginIndx);
    debug_ps->get("d_dbgEndIndx",      d_dbgEndIndx );
    d_dbgOldTime      = -d_dbgOutputInterval;
    d_dbgNextDumpTime = 0.0;

    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
        child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"]      == "switchDebugInitialize")
       switchDebugInitialize            = true;
      else if (debug_attr["label"] == "switchDebug_EQ_RF_press")
       switchDebug_EQ_RF_press          = true;
      else if (debug_attr["label"] == "switchDebug_PressDiffRF")
       switchDebug_PressDiffRF          = true;
      else if (debug_attr["label"] == "switchDebug_vel_FC")
       switchDebug_vel_FC               = true;
      else if (debug_attr["label"] == "switchDebug_Exchange_FC")
       switchDebug_Exchange_FC          = true;
      else if (debug_attr["label"] == "switchDebug_explicit_press")
       switchDebug_explicit_press       = true;
      else if (debug_attr["label"] == "switchDebug_PressFC")
       switchDebug_PressFC              = true;
      else if (debug_attr["label"] == "switchDebugLagrangianValues")
       switchDebugLagrangianValues      = true;
      else if (debug_attr["label"] == "switchDebugMomentumExchange_CC")
       switchDebugMomentumExchange_CC   = true;
      else if (debug_attr["label"] == "switchDebugSource_Sink")
       switchDebugSource_Sink           = true;
      else if (debug_attr["label"] == "switchDebug_advance_advect")
       switchDebug_advance_advect       = true;
      else if (debug_attr["label"] == "switchDebug_advectQFirst")
       switchDebug_advectQFirst         = true;
      else if (debug_attr["label"] == "switchTestConservation")
        switchTestConservation           = true;
    }
  }
  cout_norm << "Pulled out the debugging switches from input file" << endl;
  cout_norm<< "  debugging starting time "  <<d_dbgStartTime<<endl;
  cout_norm<< "  debugging stopping time "  <<d_dbgStopTime<<endl;
  cout_norm<< "  debugging output interval "<<d_dbgOutputInterval<<endl;
 
  //__________________________________
  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  cfd_ps->require("cfl",d_CFL);
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE");
  cfd_ice_ps->require("max_iteration_equilibration",d_max_iter_equilibration);
  d_advector = AdvectionFactory::create(cfd_ice_ps);
  // Grab the solution technique
  ProblemSpecP child = cfd_ice_ps->findBlock("solution");
  if(!child)
    throw ProblemSetupException("Cannot find Solution Technique tag for ICE");
  std::string solution_technique;
  if(!child->getAttribute("technique",solution_technique))
    throw ProblemSetupException("Nothing specified for solution technique"); 
  if (solution_technique == "RateForm") {
    d_RateForm = true;
    cout_norm << "Solution Technique = Rate Form " << endl;
  } 
  if (solution_technique == "EqForm") {
    d_EqForm = true;
    cout_norm << "Solution Technique = Equilibration Form " << endl;
  }
  if (d_RateForm == false && d_EqForm == false ) {
    throw ProblemSetupException("\n\nERROR:\nMust specify either EqForm or RateForm in ICE solution technique\n\n");
  }
  
  cout_norm << "cfl = " << d_CFL << endl;
  cout_norm << "max_iteration_equilibration " << d_max_iter_equilibration << endl;
  cout_norm << "max_iteration_equilibration " << d_max_iter_equilibration << endl;

  cout_norm << "Pulled out CFD-ICE block of the input file" << endl;
    
  //__________________________________
  // Pull out from Time section
  d_initialDt = 10000.0;
  ProblemSpecP time_ps = prob_spec->findBlock("Time");
  time_ps->get("delt_init",d_initialDt);
  cout_norm << "Initial dt = " << d_initialDt << endl;
  cout_norm << "Pulled out Time block of the input file" << endl;

  //__________________________________
  // Pull out Initial Conditions
  ProblemSpecP mat_ps       =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");  

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    // Extract out the type of EOS and the associated parameters
     ICEMaterial *mat = scinew ICEMaterial(ps);
     sharedState->registerICEMaterial(mat);
  }     
  cout_norm << "Pulled out InitialConditions block of the input file" << endl;

  //__________________________________
  // Pull out the exchange coefficients
  ProblemSpecP exch_ps = mat_ps->findBlock("exchange_properties");
  if (!exch_ps)
    throw ProblemSetupException("Cannot find exchange_properties tag");
  
  ProblemSpecP exch_co_ps = exch_ps->findBlock("exchange_coefficients");
  exch_co_ps->require("momentum",d_K_mom);
  exch_co_ps->require("heat",d_K_heat);

  for (int i = 0; i<(int)d_K_mom.size(); i++)
    cout_norm << "K_mom = " << d_K_mom[i] << endl;
  for (int i = 0; i<(int)d_K_heat.size(); i++)
    cout_norm << "K_heat = " << d_K_heat[i] << endl;
  cout_norm << "Pulled out exchange coefficients of the input file" << endl;

  string mass_exch_in;
  ProblemSpecP mass_exch_ps = exch_ps->get("mass_exchange",mass_exch_in);

  if (mass_exch_ps) {
    istringstream in(mass_exch_in);
    string mass_exch_out;
    in >> mass_exch_out;
    if (mass_exch_out == "true" || mass_exch_out == "TRUE" || 
       mass_exch_out == "1") {
      d_massExchange = true;
    } else 
      d_massExchange = false;
  } else
    d_massExchange = false;

  cout_norm << "Mass exchange = " << d_massExchange << endl;

  //__________________________________
  //  Print out what I've found
  cout_norm << "Number of ICE materials: " 
       << d_sharedState->getNumICEMatls()<< endl;

  if (switchDebugInitialize == true) 
    cout_norm << "switchDebugInitialize is ON" << endl;
  if (switchDebug_EQ_RF_press == true) 
    cout_norm << "switchDebug_EQ_RF_press is ON" << endl;
  if (switchDebug_vel_FC == true) 
    cout_norm << "switchDebug_vel_FC is ON" << endl;
  if (switchDebug_Exchange_FC == true) 
    cout_norm << "switchDebug_Exchange_FC is ON" << endl;
  if (switchDebug_explicit_press == true) 
    cout_norm << "switchDebug_explicit_press is ON" << endl;
  if (switchDebug_PressFC == true) 
    cout_norm << "switchDebug_PressFC is ON" << endl;
  if (switchDebugLagrangianValues == true) 
    cout_norm << "switchDebugLagrangianValues is ON" << endl;
  if (switchDebugSource_Sink == true) 
    cout_norm << "switchDebugSource_Sink is ON" << endl;
  if (switchDebug_advance_advect == true) 
    cout_norm << "switchDebug_advance_advect is ON" << endl;
  if (switchDebug_advectQFirst == true) 
    cout_norm << "switchDebug_advectQFirst is ON" << endl;
  if (switchTestConservation == true)
    cout_norm << "switchTestConservation is ON" << endl;

}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleInitialize--
_____________________________________________________________________*/
void ICE::scheduleInitialize(const LevelP& level,SchedulerP& sched)
{

  cout_doing << "Doing ICE::scheduleInitialize " << endl;
  Task* t = scinew Task("ICE::actuallyInitialize",
                  this, &ICE::actuallyInitialize);
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  t->computes(lb->doMechLabel);
  t->computes(lb->vel_CCLabel);
  t->computes(lb->rho_CCLabel); 
  t->computes(lb->temp_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->rho_micro_CCLabel);
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->press_CCLabel, press_matl);
  t->computes(d_sharedState->get_delt_label());

  sched->addTask(t, level->eachPatch(), d_sharedState->allICEMaterials());

  // The task will have a reference to press_matl
  if (press_matl->removeReference())
    delete press_matl; // shouln't happen, but...
}

void ICE::restartInitialize()
{
  // disregard initial dt when restarting
  d_initialDt = 10000.0;
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeStableTimestep--
_____________________________________________________________________*/
void ICE::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  cout_doing << "ICE::scheduleComputeStableTimestep " << endl;
  Task* task = scinew Task("ICE::actuallyComputeStableTimestep",
                     this, &ICE::actuallyComputeStableTimestep);

  task->requires(Task::NewDW, lb->doMechLabel);
  task->requires(Task::NewDW, lb->vel_CCLabel,        Ghost::None);
  task->requires(Task::NewDW, lb->speedSound_CCLabel, Ghost::None);
  task->computes(d_sharedState->get_delt_label());
  sched->addTask(task,level->eachPatch(), d_sharedState->allICEMaterials());
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleTimeAdvance--
_____________________________________________________________________*/
void ICE::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  cout_doing << "ICE::scheduleTimeAdvance" << endl;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  
  MaterialSubset* press_matl    = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();

  scheduleComputePressure(                sched, patches, press_matl,
                                                          all_matls);
  if(d_RateForm) {                                                         
    scheduleComputeFCPressDiffRF(         sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          press_matl,
                                                          all_matls);
  }
                                                            
  scheduleComputeFaceCenteredVelocities(  sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          press_matl, 
                                                          all_matls);

  scheduleAddExchangeContributionToFCVel( sched, patches, all_matls);    

  scheduleMassExchange(                   sched, patches, all_matls);

  scheduleComputeDelPressAndUpdatePressCC(sched, patches, press_matl,
                                                          ice_matls_sub, 
                                                          mpm_matls_sub,
                                                          all_matls);

  scheduleComputePressFC(                 sched, patches, press_matl,
                                                          all_matls);

  scheduleAccumulateMomentumSourceSinks(  sched, patches, press_matl,
                                                          ice_matls_sub,
                                                          all_matls);

  scheduleAccumulateEnergySourceSinks(    sched, patches, press_matl,
                                                          all_matls);

  scheduleComputeLagrangianValues(        sched, patches, all_matls);

  scheduleAddExchangeToMomentumAndEnergy( sched, patches, all_matls);

  scheduleComputeLagrangianSpecificVolume(sched, patches, press_matl,
                                                          ice_matls_sub, 
                                                          all_matls);

  scheduleAdvectAndAdvanceInTime(         sched, patches, all_matls);
  
  if(switchTestConservation) {
    schedulePrintConservedQuantities(     sched, patches, press_matl,
                                                          all_matls); 
  }

  // whatever tasks use press_matl will have their own reference to it.
  if (press_matl->removeReference())
    delete press_matl;
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputePressure--
_____________________________________________________________________*/
void ICE::scheduleComputePressure(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* press_matl,
                                          const MaterialSet* ice_matls)
{
  Task* t;
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleComputeRateFormPressure" << endl;
    t = scinew Task("ICE::computeRateFormPressure",
                     this, &ICE::computeRateFormPressure);
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::scheduleComputeEquilibrationPressure" << endl;
    t = scinew Task("ICE::computeEquilibrationPressure",
                     this, &ICE::computeEquilibrationPressure);
  }         

  if (d_EqForm) {       // EQ FORM
    t->requires(Task::OldDW,lb->press_CCLabel, press_matl, Ghost::None);
    t->requires(Task::OldDW,lb->rho_CCLabel,               Ghost::None);
    t->requires(Task::OldDW,lb->temp_CCLabel,              Ghost::None); 
    t->requires(Task::OldDW,lb->vel_CCLabel,               Ghost::None);
  }         
  else if (d_RateForm) {     // RATE FORM
    t->requires(Task::OldDW,lb->press_CCLabel, press_matl, Ghost::None);
    t->requires(Task::OldDW,lb->rho_CCLabel,               Ghost::None);
    t->requires(Task::OldDW,lb->temp_CCLabel,              Ghost::None); 
    t->requires(Task::OldDW,lb->sp_vol_CCLabel,            Ghost::None);
    t->computes(lb->matl_press_CCLabel);
    t->computes(lb->f_theta_CCLabel);
  }
                        // EQ & RATE FORM
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->rho_CCLabel);
  t->computes(lb->press_equil_CCLabel, press_matl);
  
  sched->addTask(t, patches, ice_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeFaceCenteredVelocities--
_____________________________________________________________________*/
void ICE::scheduleComputeFaceCenteredVelocities(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSubset* press_matl,
                                          const MaterialSet* all_matls)
{ 
  Task* t;
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleComputeFaceCenteredVelocitiesRF" << endl;
    t = scinew Task("ICE::computeFaceCenteredVelocitiesRF",
                       this, &ICE::computeFaceCenteredVelocitiesRF);
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::scheduleComputeFaceCenteredVelocities" << endl;
    t = scinew Task("ICE::computeFaceCenteredVelocities",
                       this, &ICE::computeFaceCenteredVelocities);
  }
                      // EQ  & RATE FORM 
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW,lb->press_equil_CCLabel, press_matl,
                                                      Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,    /*all_matls*/
                                                      Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->rho_CCLabel,       /*all_matls*/
                                                      Ghost::AroundCells,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,         ice_matls, 
                                                      Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->vel_CCLabel,         mpm_matls, 
                                                      Ghost::AroundCells,1);
  t->requires(Task::OldDW,lb->doMechLabel);
  
  if (d_RateForm) {     //RATE FORM
    t->requires(Task::NewDW,lb->press_diffX_FCLabel, /*all_matls*/
                                                        Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->press_diffY_FCLabel, /*all_matls*/
                                                        Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->press_diffZ_FCLabel, /*all_matls*/
                                                        Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->matl_press_CCLabel,  /*all_matls*/
                                                        Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->vol_frac_CCLabel,    /*all_matls*/
                                                        Ghost::AroundCells,1);
  }

  t->computes(lb->uvel_FCLabel);
  t->computes(lb->vvel_FCLabel);
  t->computes(lb->wvel_FCLabel);
  sched->addTask(t, patches, all_matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAddExchangeContributionToFCVel--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeContributionToFCVel(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleAddExchangeContributionToFCVel" << endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel);

  task->requires(Task::OldDW, lb->delTLabel);  
  task->requires(Task::NewDW,lb->sp_vol_CCLabel,    Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->vol_frac_CCLabel,  Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->uvel_FCLabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW,lb->vvel_FCLabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW,lb->wvel_FCLabel,      Ghost::AroundCells,2);
 
  task->computes(lb->uvel_FCMELabel);
  task->computes(lb->vvel_FCMELabel);
  task->computes(lb->wvel_FCMELabel);
  
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleMassExchange--
_____________________________________________________________________*/
void  ICE::scheduleMassExchange(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleMassExchange" << endl;
  Task* task = scinew Task("ICE::massExchange",
                     this, &ICE::massExchange);
  task->requires(Task::NewDW, lb->rho_CCLabel,  Ghost::None);
  task->requires(Task::OldDW, lb->vel_CCLabel,  Ghost::None); 
  task->requires(Task::OldDW, lb->temp_CCLabel, Ghost::None);
  task->computes(lb->burnedMass_CCLabel);
  task->computes(lb->int_eng_comb_CCLabel);
  task->computes(lb->mom_comb_CCLabel);
  task->computes(lb->created_vol_CCLabel);
  
  sched->addTask(task, patches, matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeDelPressAndUpdatePressCC--
_____________________________________________________________________*/
void ICE::scheduleComputeDelPressAndUpdatePressCC(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* press_matl,
                                            const MaterialSubset* ice_matls,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleComputeDelPressAndUpdatePressCC" << endl;
  Task *task = scinew Task("ICE::computeDelPressAndUpdatePressCC",
                            this, &ICE::computeDelPressAndUpdatePressCC);
  
  task->requires( Task::OldDW, lb->delTLabel);
  task->requires( Task::NewDW, lb->press_equil_CCLabel,
                                          press_matl, Ghost::None);
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,  Ghost::AroundCells,1);
  task->requires( Task::NewDW, lb->uvel_FCMELabel,    Ghost::AroundCells,2);
  task->requires( Task::NewDW, lb->vvel_FCMELabel,    Ghost::AroundCells,2);
  task->requires( Task::NewDW, lb->wvel_FCMELabel,    Ghost::AroundCells,2);
  task->requires( Task::NewDW, lb->sp_vol_CCLabel,    Ghost::None);
  task->requires( Task::NewDW, lb->rho_CCLabel,       Ghost::None);
  task->requires( Task::OldDW, lb->vel_CCLabel,       ice_matls, 
                                                      Ghost::None); 
  task->requires( Task::NewDW, lb->vel_CCLabel,       mpm_matls, 
                                                      Ghost::None);     
  task->requires(Task::NewDW,lb->burnedMass_CCLabel,  Ghost::None);
  if (d_EqForm){          //  E Q   F O R M
    task->requires(Task::NewDW,lb->speedSound_CCLabel,Ghost::None);
  }

  task->computes(lb->press_CCLabel,        press_matl);
  task->computes(lb->delP_DilatateLabel,   press_matl);
  task->computes(lb->delP_MassXLabel,      press_matl);
  
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputePressFC--
_____________________________________________________________________*/
void ICE::scheduleComputePressFC(SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSubset* press_matl,
                             const MaterialSet* matls)
{ 
  cout_doing << "ICE::scheduleComputePressFC" << endl;                   
  Task* task = scinew Task("ICE::computePressFC",
                     this, &ICE::computePressFC);

  task->requires(Task::NewDW,lb->press_CCLabel,
                                     press_matl,Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->rho_CCLabel,   Ghost::AroundCells,1);
  
  task->computes(lb->pressX_FCLabel, press_matl);
  task->computes(lb->pressY_FCLabel, press_matl);
  task->computes(lb->pressZ_FCLabel, press_matl);

  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAccumulateMomentumSourceSinks--
_____________________________________________________________________*/
void ICE::scheduleAccumulateMomentumSourceSinks(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSubset* press_matl,
                                          const MaterialSubset* ice_matls_sub,
                                          const MaterialSet* matls)
{
  Task* t;
  cout_doing << "ICE::scheduleAccumulateMomentumSourceSinks" << endl; 
  t = scinew Task("ICE::accumulateMomentumSourceSinks", 
                   this, &ICE::accumulateMomentumSourceSinks);

                       // EQ  & RATE FORM     
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW,lb->pressX_FCLabel,   press_matl,    
                                                Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->pressY_FCLabel,   press_matl,
                                                Ghost::AroundCells,1);
  t->requires(Task::NewDW,lb->pressZ_FCLabel,   press_matl,
                                                Ghost::AroundCells,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,      ice_matls_sub,
                                                Ghost::AroundCells,2); 
  t->requires(Task::NewDW,lb->rho_CCLabel,      Ghost::None);
  t->requires(Task::NewDW,lb->vol_frac_CCLabel, Ghost::None);
  t->requires(Task::OldDW,lb->doMechLabel);
  if (d_RateForm) {   // RATE FORM
    t->requires(Task::NewDW,lb->press_diffX_FCLabel, Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->press_diffY_FCLabel, Ghost::AroundCells,1);
    t->requires(Task::NewDW,lb->press_diffZ_FCLabel, Ghost::AroundCells,1);
  }
 
  t->computes(lb->doMechLabel);
  t->computes(lb->mom_source_CCLabel);
  sched->addTask(t, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAccumulateEnergySourceSinks--
_____________________________________________________________________*/
void ICE::scheduleAccumulateEnergySourceSinks(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSubset* press_matl,
                                         const MaterialSet* matls)

{
  Task* t;
    cout_doing << "ICE::scheduleAccumulateEnergySourceSinks" << endl;
    t = scinew Task("ICE::accumulateEnergySourceSinks",
                     this, &ICE::accumulateEnergySourceSinks);
  
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,Ghost::None);
  t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,Ghost::None);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,               Ghost::None);
  t->requires(Task::NewDW, lb->speedSound_CCLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,             Ghost::None);

#ifdef ANNULUSICE
  t->requires(Task::NewDW, lb->rho_CCLabel,                  Ghost::None);
#endif
  
  t->computes(lb->int_eng_source_CCLabel);
  
  sched->addTask(t, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE:: scheduleComputeLagrangianValues--
 Note:      Only loop over ICE materials  
_____________________________________________________________________*/
void ICE::scheduleComputeLagrangianValues(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* ice_matls)
{
  cout_doing << "ICE::scheduleComputeLagrangianValues" << endl;
  Task* task = scinew Task("ICE::computeLagrangianValues",
                      this,&ICE::computeLagrangianValues);

  task->requires(Task::NewDW,lb->rho_CCLabel,             Ghost::None);
  task->requires(Task::OldDW,lb->vel_CCLabel,             Ghost::None);
  task->requires(Task::OldDW,lb->temp_CCLabel,            Ghost::None);
  task->requires(Task::NewDW,lb->mom_source_CCLabel,      Ghost::None);
  task->requires(Task::NewDW,lb->mom_comb_CCLabel,        Ghost::None);
  task->requires(Task::NewDW,lb->burnedMass_CCLabel,      Ghost::None);
  task->requires(Task::NewDW,lb->int_eng_comb_CCLabel,    Ghost::None);
  task->requires(Task::NewDW,lb->int_eng_source_CCLabel,  Ghost::None);

  task->computes(lb->mom_L_CCLabel);
  task->computes(lb->int_eng_L_CCLabel);
  task->computes(lb->mass_L_CCLabel);
 
  sched->addTask(task, patches, ice_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE:: scheduleComputeLagrangianSpecificVolume--
_____________________________________________________________________*/
void ICE::scheduleComputeLagrangianSpecificVolume(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSubset* press_matl,
                                               const MaterialSubset* ice_matls,
                                               const MaterialSet* matls)
{
  Task* t;
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleComputeLagrangianSpecificVolumeRF" << endl;
    t = scinew Task("ICE::computeLagrangianSpecificVolumeRF",
                        this,&ICE::computeLagrangianSpecificVolumeRF);
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::scheduleComputeLagrangianSpecificVolume" << endl;
    t = scinew Task("ICE::computeLagrangianSpecificVolume",
                        this,&ICE::computeLagrangianSpecificVolume);
  }

  if (d_EqForm) {     // EQ FORM
    t->requires(Task::NewDW, lb->rho_CCLabel,         ice_matls,
                                                      Ghost::AroundCells,1);
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,      ice_matls,
                                                      Ghost::AroundCells,1);
    t->requires(Task::NewDW, lb->created_vol_CCLabel, ice_matls,
                                                      Ghost::AroundCells,1);
    t->computes(lb->spec_vol_L_CCLabel);
  }
  else if (d_RateForm) {     // RATE FORM
    t->requires(Task::OldDW, lb->delTLabel);
    t->requires(Task::NewDW, lb->rho_CCLabel,       ice_matls, 
                                                    Ghost::AroundCells,1);       
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,    ice_matls,              
                                                    Ghost::AroundCells,1);  
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,  ice_matls, Ghost::None);
    t->requires(Task::OldDW, lb->temp_CCLabel,      ice_matls, Ghost::None);
    t->requires(Task::NewDW, lb->Tdot_CCLabel,      ice_matls, Ghost::None);
    t->requires(Task::NewDW, lb->f_theta_CCLabel,   ice_matls, Ghost::None);
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,Ghost::None);
    t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,Ghost::None);
    t->computes(lb->spec_vol_L_CCLabel);
    t->computes(lb->spec_vol_source_CCLabel);
  }
  sched->addTask(t, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAddExchangeToMomentumAndEnergy--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeToMomentumAndEnergy(SchedulerP& sched,
                                           const PatchSet* patches, 
                                           const MaterialSet* matls)
{
  Task* t;
  cout_doing << "ICE::scheduleAddExchangeToMomentumAndEnergy" << endl;
  t = scinew Task("ICE::addExchangeToMomentumAndEnergy",
                     this, &ICE::addExchangeToMomentumAndEnergy);

                        // EQ & RATE FORM
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->mass_L_CCLabel,   Ghost::None);
  t->requires(Task::NewDW, lb->mom_L_CCLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->int_eng_L_CCLabel,Ghost::None);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel, Ghost::None);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,   Ghost::None);
 
  t->computes(lb->mom_L_ME_CCLabel);
  t->computes(lb->int_eng_L_ME_CCLabel);
  
  if (d_RateForm) {   // RATE FORM
   t->requires(Task::OldDW, lb->temp_CCLabel,     Ghost::None);
   t->computes(lb->Tdot_CCLabel);
  }
  
  sched->addTask(t, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAdvectAndAdvanceInTime--
_____________________________________________________________________*/
void ICE::scheduleAdvectAndAdvanceInTime(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleAdvectAndAdvanceInTime" << endl;
  Task* task = scinew Task("ICE::advectAndAdvanceInTime",
                     this, &ICE::advectAndAdvanceInTime);
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW, lb->uvel_FCMELabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW, lb->vvel_FCMELabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW, lb->wvel_FCMELabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW, lb->mom_L_ME_CCLabel,    Ghost::AroundCells,1);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,      Ghost::AroundCells,1);
  task->requires(Task::NewDW, lb->int_eng_L_ME_CCLabel,Ghost::AroundCells,1);
  task->requires(Task::NewDW, lb->spec_vol_L_CCLabel,  Ghost::AroundCells,1);
 
  task->modifies(lb->rho_CCLabel);
  task->modifies(lb->sp_vol_CCLabel);
  task->computes(lb->temp_CCLabel);
  task->computes(lb->vel_CCLabel);
  sched->addTask(task, patches, matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::schedulePrintConservedQuantities--
_____________________________________________________________________*/
void ICE::schedulePrintConservedQuantities(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSubset* press_matl,
                                      const MaterialSet* matls)
{
  cout_doing << "ICE::schedulePrintConservedQuantities" << endl;
  Task* task = scinew Task("ICE::printConservedQuantities",
                     this, &ICE::printConservedQuantities);

  task->requires(Task::NewDW,lb->delP_DilatateLabel, press_matl,Ghost::None);
  task->requires(Task::NewDW,lb->rho_CCLabel, Ghost::None);
  task->requires(Task::NewDW,lb->vel_CCLabel, Ghost::None);
  task->requires(Task::NewDW,lb->temp_CCLabel,Ghost::None);
  task->computes(lb->TotalMassLabel);
  task->computes(lb->KineticEnergyLabel);
  task->computes(lb->TotalIntEngLabel);
  task->computes(lb->CenterOfMassVelocityLabel); //momentum
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::actuallyComputeStableTimestep--
 Purpose~   Compute next time step based on speed of sound and 
            maximum velocity in the domain
_____________________________________________________________________*/
void ICE::actuallyComputeStableTimestep(const ProcessorGroup*,  
                                    const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                    DataWarehouse* /*old_dw*/,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Compute Stable Timestep on patch " << patch->getID() 
         << "\t\t ICE" << endl;
      
      Vector dx = patch->dCell();
      double delt_CFL = 100000, fudge_factor = 1.;
      constCCVariable<double> speedSound;
      constCCVariable<Vector> vel;

      for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx= ice_matl->getDWIndex();

        new_dw->get(speedSound, lb->speedSound_CCLabel,
                                           indx,patch,Ghost::None, 0);
        new_dw->get(vel, lb->vel_CCLabel, indx,patch,Ghost::None, 0);
       
       for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
         IntVector c = *iter;
         double A = fudge_factor*d_CFL*dx.x()/(speedSound[c] + 
                                        fabs(vel[c].x())+d_SMALL_NUM);
         double B = fudge_factor*d_CFL*dx.y()/(speedSound[c] + 
                                        fabs(vel[c].y())+d_SMALL_NUM);
         double C = fudge_factor*d_CFL*dx.z()/(speedSound[c] + 
                                        fabs(vel[c].z())+d_SMALL_NUM);

         delt_CFL = std::min(A, delt_CFL);
         delt_CFL = std::min(B, delt_CFL);
         delt_CFL = std::min(C, delt_CFL);

        }
      }
      delt_CFL = std::min(delt_CFL, d_initialDt);
      d_initialDt = 10000.0;

      delt_vartype doMech;
      new_dw->get(doMech, lb->doMechLabel);
      if(doMech >= 0.){
        delt_CFL = .0625;
      }
    //__________________________________
    //  Bullet proofing
    if(delt_CFL < 1e-20) {  
      string warn = " E R R O R -------> ICE::ComputeStableTimestep: delT < 1e-20";
      throw InvalidValue(warn);
    }
    
    new_dw->put(delt_vartype(delt_CFL), lb->delTLabel);
  }  // patch loop
  //  update when you should dump debugging data. 
  d_dbgNextDumpTime = d_dbgOldTime + d_dbgOutputInterval;
}

/* --------------------------------------------------------------------- 
 Function~  ICE::actuallyInitialize--
 Purpose~  Initialize the CC and FC variables and the pressure  
 Note that rho_micro, sp_vol, temp and velocity must be defined 
 everywhere in the domain
_____________________________________________________________________*/ 
void ICE::actuallyInitialize(const ProcessorGroup*, 
                          const PatchSubset* patches,
                          const MaterialSubset* /*matls*/,
                          DataWarehouse*, 
                          DataWarehouse* new_dw)
{
 //__________________________________
 //  dump patch limits to screen
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<< "patch: "<<patch->getID()<<
          patch->getCellLowIndex()  << 
          patch->getCellHighIndex() << endl;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID() 
         << "\t\t\t ICE" << endl;
    int numMatls    = d_sharedState->getNumICEMatls();
    int numALLMatls = d_sharedState->getNumMatls();
    Vector grav     = d_sharedState->getGravity();
    StaticArray<CCVariable<double>   > rho_micro(numMatls);
    StaticArray<CCVariable<double>   > sp_vol_CC(numMatls);
    StaticArray<CCVariable<double>   > rho_CC(numMatls); 
    StaticArray<CCVariable<double>   > Temp_CC(numMatls);
    StaticArray<CCVariable<double>   > speedSound(numMatls);
    StaticArray<CCVariable<double>   > vol_frac_CC(numMatls);
    StaticArray<CCVariable<Vector>   > vel_CC(numMatls);
    CCVariable<double>    press_CC;  
    StaticArray<double>   cv(numMatls);
    new_dw->allocate(press_CC,lb->press_CCLabel, 0,patch);
    press_CC.initialize(0.0);

  //__________________________________
  // Note:
  // The press_CC isn't material dependent even though
  // we loop over numMatls below. This is done so we don't need additional
  // machinery to grab the pressure inside a geom_object
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();
      new_dw->allocate(rho_micro[m],   lb->rho_micro_CCLabel,   indx,patch);
      new_dw->allocate(sp_vol_CC[m],   lb->sp_vol_CCLabel,      indx,patch);
      new_dw->allocate(rho_CC[m],      lb->rho_CCLabel,         indx,patch); 
      new_dw->allocate(Temp_CC[m],     lb->temp_CCLabel,        indx,patch);
      new_dw->allocate(speedSound[m],  lb->speedSound_CCLabel,  indx,patch);
      new_dw->allocate(vol_frac_CC[m], lb->vol_frac_CCLabel,    indx,patch);
      new_dw->allocate(vel_CC[m],      lb->vel_CCLabel,         indx,patch);
    }
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      ice_matl->initializeCells(rho_micro[m],  rho_CC[m],
                                Temp_CC[m],   speedSound[m], 
                                vol_frac_CC[m], vel_CC[m], 
                                press_CC,  numALLMatls,    patch, new_dw);

      cv[m] = ice_matl->getSpecificHeat();
      setBC(rho_CC[m],        "Density",      patch, d_sharedState, indx);
      setBC(rho_micro[m],     "Density",      patch, d_sharedState, indx);
      setBC(Temp_CC[m],       "Temperature",  patch, d_sharedState, indx);
      setBC(vel_CC[m],        "Velocity",     patch, indx); 
      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[m][c] = 1.0/rho_micro[m][c];
      }
      //__________________________________
      //  Adjust pressure and Temp field if g != 0
      //  so fields are thermodynamically consistent.
      if ((grav.x() !=0 || grav.y() != 0.0 || grav.z() != 0.0))  {
        hydrostaticPressureAdjustment(patch,
                                      rho_micro[SURROUND_MAT], press_CC);

        setBC(press_CC,  rho_micro[SURROUND_MAT],
            "rho_micro", "Pressure", patch, d_sharedState, 0, new_dw);

        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        double gamma = ice_matl->getGamma();
        ice_matl->getEOS()->computeTempCC(patch, "WholeDomain",
                                     press_CC,   gamma,   cv[m],
                                     rho_micro[m],    Temp_CC[m]);
      }
    }  
    
    for (int m = 0; m < numMatls; m++ ) { 
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex(); 
      new_dw->put(rho_micro[m],     lb->rho_micro_CCLabel,      indx,patch);
      new_dw->put(sp_vol_CC[m],     lb->sp_vol_CCLabel,         indx,patch);
      new_dw->put(rho_CC[m],        lb->rho_CCLabel,            indx,patch);
      new_dw->put(vol_frac_CC[m],   lb->vol_frac_CCLabel,       indx,patch);
      new_dw->put(Temp_CC[m],       lb->temp_CCLabel,           indx,patch);
      new_dw->put(speedSound[m],    lb->speedSound_CCLabel,     indx,patch);
      new_dw->put(vel_CC[m],        lb->vel_CCLabel,            indx,patch);

      if (switchDebugInitialize){
        cout_norm << " Initial Conditions" << endl;       
        ostringstream desc;
        desc << "Initialization_Mat_" << indx << "_patch_"<< patch->getID();
        printData(patch,   1, desc.str(), "rho_CC",      rho_CC[m]);
        printData(patch,   1, desc.str(), "rho_micro_CC",rho_micro[m]);
     // printData(patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC[m]);
        printData(patch,   1, desc.str(), "Temp_CC",     Temp_CC[m]);
        printData(patch,   1, desc.str(), "vol_frac_CC", vol_frac_CC[m]);
        printVector(patch, 1, desc.str(), "uvel_CC", 0,  vel_CC[m]);
        printVector(patch, 1, desc.str(), "vvel_CC", 1,  vel_CC[m]);
        printVector(patch, 1, desc.str(), "wvel_CC", 2,  vel_CC[m]);
      }   
    }
    setBC(press_CC,   rho_micro[SURROUND_MAT], 
          "rho_micro","Pressure", patch, d_sharedState, 0, new_dw);
    if (switchDebugInitialize){
       printData(   patch, 1, "Initialization", "press_CC", press_CC);
    }
    new_dw->put(press_CC,    lb->press_CCLabel,  0,patch);

    double doMech = -999.9;
    new_dw->put(delt_vartype(doMech), lb->doMechLabel);

  }  // patch loop 
}

/* --------------------------------------------------------------------- 
 Function~  ICE::computeEquilibrationPressure--
 Purpose~   Find the equilibration pressure  
 Reference: Flow of Interpenetrating Material Phases, J. Comp, Phys
               18, 440-464, 1975, see the equilibration section
                   
 Steps
 ----------------
    - Compute rho_micro_CC, SpeedSound, vol_frac

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
_____________________________________________________________________*/
void ICE::computeEquilibrationPressure(const ProcessorGroup*,  
                                       const PatchSubset* patches,
                                       const MaterialSubset* /*matls*/,
                                       DataWarehouse* old_dw, 
                                       DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing calc_equilibration_pressure on patch "<<patch->getID() 
         << "\t\t ICE" << endl;
    double    converg_coeff = 15;              
    double    convergence_crit = converg_coeff * DBL_EPSILON;
    double    sum, tmp;

    int       numMatls = d_sharedState->getNumICEMatls();
    static int n_passes;                  
    n_passes ++; 

    StaticArray<double> press_eos(numMatls);
    StaticArray<double> dp_drho(numMatls),dp_de(numMatls);
    StaticArray<CCVariable<double> > vol_frac(numMatls);
    StaticArray<CCVariable<double> > rho_micro(numMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numMatls);
    StaticArray<CCVariable<double> > sp_vol_CC(numMatls);
    StaticArray<CCVariable<double> > speedSound(numMatls);
    StaticArray<CCVariable<double> > speedSound_new(numMatls);
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numMatls);
    CCVariable<int> n_iters_equil_press;
    constCCVariable<double> press;
    CCVariable<double> press_new;
    StaticArray<double> cv(numMatls), gamma(numMatls);

    old_dw->get(press,         lb->press_CCLabel, 0,patch,Ghost::None, 0); 
    new_dw->allocate(press_new,lb->press_equil_CCLabel, 0,patch);

    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],  lb->temp_CCLabel,indx,patch, Ghost::None,0);
      old_dw->get(rho_CC[m],lb->rho_CCLabel, indx,patch, Ghost::None,0);
      old_dw->get(vel_CC[m],lb->vel_CCLabel, indx,patch, Ghost::None,0);

      new_dw->allocate(speedSound_new[m],lb->speedSound_CCLabel,indx, patch);
      new_dw->allocate(rho_micro[m],     lb->rho_micro_CCLabel, indx, patch);
      new_dw->allocate(vol_frac[m],      lb->vol_frac_CCLabel,  indx, patch);
      new_dw->allocate(rho_CC_new[m],    lb->rho_CCLabel,       indx, patch);
      new_dw->allocate(sp_vol_CC[m],     lb->sp_vol_CCLabel,    indx, patch);
      cv[m] = matl->getSpecificHeat();
      gamma[m] = matl->getGamma();
    }

    press_new.copyData(press);
    //__________________________________
    // Compute rho_micro, speedSound, and volfrac
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();
          iter++) {
        rho_micro[m][*iter] = 
         ice_matl->getEOS()->computeRhoMicro(press_new[*iter],gamma[m],cv[m],
                                         Temp[m][*iter]); 

          ice_matl->getEOS()->computePressEOS(rho_micro[m][*iter],gamma[m],
                                          cv[m], Temp[m][*iter],
                                          press_eos[m], dp_drho[m], dp_de[m]);

         double div = 1./rho_micro[m][*iter];
         tmp = dp_drho[m] + dp_de[m] * press_eos[m] * div * div;
         speedSound_new[m][*iter] = sqrt(tmp);
         vol_frac[m][*iter] = rho_CC[m][*iter] * div;
      }
    }

   //---- P R I N T   D A T A ------  
    if (switchDebug_EQ_RF_press) {
    
      new_dw->allocate(n_iters_equil_press, lb->scratchLabel, 0, patch);
#if 0
      ostringstream desc;
      desc << "TOP_equilibration_patch_" << patch->getID();
      printData( patch, 1, desc.str(), "Press_CC_top", press);
     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       desc << "TOP_equilibration_Mat_" << indx << "_patch_"<<patch->getID();
       printData(patch, 1, desc.str(), "rho_CC",       rho_CC[m]);    
       printData(patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);  
       printData(patch, 0, desc.str(), "speedSound",   speedSound_new[m]);
       printData(patch, 1, desc.str(), "Temp_CC",      Temp[m]);       
       printData(patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);   
      }
#endif
    }

  //______________________________________________________________________
  // Done with preliminary calcs, now loop over every cell
    int count, test_max_iter = 0;
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
      IntVector c = *iter;   
      double delPress = 0.;
      bool converged  = false;
      count           = 0;
      while ( count < d_max_iter_equilibration && converged == false) {
        count++;
        double A = 0.;
        double B = 0.;
        double C = 0.;

        //__________________________________
       // evaluate press_eos at cell i,j,k
       for (int m = 0; m < numMatls; m++)  {
         ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
         ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                           cv[m], Temp[m][c],
                                           press_eos[m], dp_drho[m], dp_de[m]);
       }
       //__________________________________
       // - compute delPress
       // - update press_CC     
       for (int m = 0; m < numMatls; m++)   {
         double Q =  press_new[c] - press_eos[m];
         double y =  dp_drho[m] * ( rho_CC[m][c]/
                 (vol_frac[m][c] * vol_frac[m][c]) ); 
        double div_y = 1./y;
         A   +=  vol_frac[m][c];
         B   +=  Q*div_y;
         C   +=  div_y;
       }
       double vol_frac_not_close_packed = 1.;
       delPress = (A - vol_frac_not_close_packed - B)/C;

       press_new[c] += delPress;

       //__________________________________
       // backout rho_micro_CC at this new pressure
       for (int m = 0; m < numMatls; m++) {
         ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
         rho_micro[m][c] = 
           ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[m],
                                               cv[m],Temp[m][c]);

        double div = 1./rho_micro[m][c];
       //__________________________________
       // - compute the updated volume fractions
        vol_frac[m][c]   = rho_CC[m][c]*div;

       //__________________________________
       // Find the speed of sound 
       // needed by eos and the explicit
       // del pressure function
          ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                            cv[m],Temp[m][c],
                                            press_eos[m],dp_drho[m], dp_de[m]);

          tmp = dp_drho[m] + dp_de[m] * press_eos[m]*div*div;
          speedSound_new[m][c] = sqrt(tmp);
       }
       //__________________________________
       // - Test for convergence 
       //  If sum of vol_frac_CC ~= 1.0 then converged 
       sum = 0.0;
       for (int m = 0; m < numMatls; m++)  {
         sum += vol_frac[m][c];
       }
       if (fabs(sum-1.0) < convergence_crit)
         converged = true;

      }   // end of converged

      test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      if(test_max_iter == d_max_iter_equilibration) 
       throw MaxIteration(c,count,n_passes,"MaxIterations reached");

       for (int m = 0; m < numMatls; m++) {
           ASSERT(( vol_frac[m][c] > 0.0 ) ||
                  ( vol_frac[m][c] < 1.0));
       }
       if ( fabs(sum - 1.0) > convergence_crit)  
        throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached vol_frac != 1");
       
       if ( press_new[c] < 0.0 )  
        throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached press_new < 0");

       for (int m = 0; m < numMatls; m++)
        if ( rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) 
          throw
            MaxIteration(c,count,n_passes,
                      "MaxIteration reached rho_micro < 0 || vol_frac < 0");

      if (switchDebug_EQ_RF_press) {
        n_iters_equil_press[c] = count;
      }
    }     // end of cell interator

    cout_norm << "max. iterations in any cell " << test_max_iter << 
                 " on patch "<<patch->getID()<<endl; 

    //__________________________________
    // update Boundary conditions
    setBC(press_new,   rho_micro[SURROUND_MAT],
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw);    
    
    //__________________________________
    // compute sp_vol_CC
    for (int m = 0; m < numMatls; m++)   {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        sp_vol_CC[m][c] = 1.0/rho_micro[m][c];
      }
    }
    //__________________________________
    // carry rho_cc forward 
    // MPMICE computes rho_CC_new
    // therefore need the machinery here
    for (int m = 0; m < numMatls; m++)   {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      rho_CC_new[m].copyData(rho_CC[m]);
      new_dw->put( vol_frac[m],      lb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( speedSound_new[m],lb->speedSound_CCLabel, indx, patch);
      new_dw->put( rho_CC_new[m],    lb->rho_CCLabel,        indx, patch);
      new_dw->put( sp_vol_CC[m],     lb->sp_vol_CCLabel,     indx, patch);
    }
    new_dw->put(press_new,lb->press_equil_CCLabel,0,patch);

   //---- P R I N T   D A T A ------   
    if (switchDebug_EQ_RF_press) {
      ostringstream desc;
      desc << "BOT_equilibration_patch_" << patch->getID();
      printData( patch, 1, desc.str(), "Press_CC_equil", press_new);

     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream desc;
       desc << "BOT_equilibration_Mat_"<< indx << "_patch_"<< patch->getID();
       printData( patch, 1, desc.str(), "rho_CC",       rho_CC[m]);
       printData( patch, 1, desc.str(), "sp_vol_CC",    sp_vol_CC[m]);
       printData( patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);
       printData( patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);
     //printData( patch, 1, desc.str(), "iterations",   n_iters_equil_press);
       
     }
    }
  }  // patch loop
}
 
/* ---------------------------------------------------------------------
 Function~  ICE::computeFaceCenteredVelocities--
 Purpose~   compute the face centered velocities minus the exchange
            contribution.
_____________________________________________________________________*/
template<class T> void ICE::computeVelFace(int dir, CellIterator it,
                                       IntVector adj_offset,double dx,
                                       double delT, double gravity,
                                       constCCVariable<double>& rho_CC,
                                       constCCVariable<double>& sp_vol_CC,
                                       constCCVariable<Vector>& vel_CC,
                                       constCCVariable<double>& press_CC,
                                       T& vel_FC)
{

  for(;!it.done(); it++){
    IntVector c = *it;
    IntVector adj = c + adj_offset; 
    double rho_FC = rho_CC[adj] + rho_CC[c];
    ASSERT(rho_FC > 0.0);
    //__________________________________
    // interpolation to the face
    double term1 = (rho_CC[adj] * vel_CC[adj](dir) +
                  rho_CC[c] * vel_CC[c](dir))/(rho_FC);            
    //__________________________________
    // pressure term           
    double sp_vol_brack = 2.*(sp_vol_CC[adj] * sp_vol_CC[c])/
      (sp_vol_CC[adj] + sp_vol_CC[c]); 
    
    double term2 = delT * sp_vol_brack * (press_CC[c] - press_CC[adj])/dx;
    
    //__________________________________
    // gravity term
    double term3 =  delT * gravity;
    
    vel_FC[c] = term1- term2 + term3;
  } 
}

                       
//______________________________________________________________________
//                       
void ICE::computeFaceCenteredVelocities(const ProcessorGroup*,  
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw, 
                                        DataWarehouse* new_dw)
{
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing compute_face_centered_velocities on patch " 
              << patch->getID() << "\t ICE" << endl;
    int numMatls = d_sharedState->getNumMatls();
    
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);
    Vector dx      = patch->dCell();
    Vector gravity = d_sharedState->getGravity();
    
    constCCVariable<double> press_CC;
    new_dw->get(press_CC,lb->press_equil_CCLabel, 0, patch, 
               Ghost::AroundCells, 1);
    
    
    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      constCCVariable<double> rho_CC, sp_vol_CC;
      constCCVariable<Vector> vel_CC;
      if(ice_matl){
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, 
                  Ghost::AroundCells, 1);
        old_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, 
                  Ghost::AroundCells, 1);
      } else {
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, 
                  Ghost::AroundCells, 1);
        new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, 
                  Ghost::AroundCells, 1);
      }              
      new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch,
                  Ghost::AroundCells, 1);
              
      //---- P R I N T   D A T A ------
  #if 0
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "TOP_vel_FC_Mat_" << indx << "_patch_"<< patch->getID(); 
        printData(  patch, 1, desc.str(), "rho_CC",      rho_CC);
        printData(  patch, 1, desc.str(), "sp_vol_CC",  sp_vol_CC);
        printVector( patch,1, desc.str(), "uvel_CC", 0, vel_CC);
        printVector( patch,1, desc.str(), "vvel_CC", 1, vel_CC);
        printVector( patch,1, desc.str(), "wvel_CC", 2, vel_CC);
      }
  #endif
      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;

      new_dw->allocate(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->allocate(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->allocate(wvel_FC, lb->wvel_FCLabel, indx, patch);
      IntVector lowIndex(patch->getSFCXLowIndex());
      uvel_FC.initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      vvel_FC.initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      wvel_FC.initialize(0.0, lowIndex,patch->getSFCZHighIndex());

      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

      if(doMechOld < -1.5) {
      int offset=0;      // 0=Compute all faces in computational domain
                         // 1=Skip the faces at the border between interior and gc    
      //__________________________________
      //  Compute vel_FC for each face
      computeVelFace<SFCXVariable<double> >(0,patch->getSFCXIterator(offset),
                                      adj_offset[0],dx(0),delT,gravity(0),
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       uvel_FC);

      computeVelFace<SFCYVariable<double> >(1,patch->getSFCYIterator(offset),
                                      adj_offset[1],dx(1),delT,gravity(1),
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       vvel_FC);

      computeVelFace<SFCZVariable<double> >(2,patch->getSFCZIterator(offset),
                                      adj_offset[2],dx(2),delT,gravity(2),
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       wvel_FC);

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()
      new_dw->put(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, indx, patch);

      //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc <<"bottom_of_vel_FC_Mat_" << indx << "_patch_"<< patch->getID();
        printData_FC( patch,1, desc.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, desc.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, desc.str(), "wvel_FC", wvel_FC);
      }
      }  // if doMech
    } // matls loop
  }  // patch loop
}
/*---------------------------------------------------------------------
 Function~  addExchangeContributionToFCVel--
 Purpose~
   This function adds the momentum exchange contribution to the 
   existing face-centered velocities

 Prerequisites:
            The face centered velocity for each material without
            the exchange must be solved prior to this routine.
            
                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_FC[1]  |    
|                                               |   |           |    
| -b21              (1+b21 + b23) -b32          |   |del_FC[2]  |    
|                                               |   |           | 
| -b31              -b32          (1+b31 + b32) |   |del_FC[2]  |

                        =
                        
                        (B)
| b12( uvel_FC[2] - uvel_FC[1] ) + b13 ( uvel_FC[3] -uvel_FC[1])    | 
|                                                                   |
| b21( uvel_FC[1] - uvel_FC[2] ) + b23 ( uvel_FC[3] -uvel_FC[2])    | 
|                                                                   |
| b31( uvel_FC[1] - uvel_FC[3] ) + b32 ( uvel_FC[2] -uvel_FC[3])    | 
 
 Steps for each face:
    1) Comute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for del_FC[*]
    4) Add del_FC[*] to the appropriate velocity
 
 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 ---------------------------------------------------------------------  */
void ICE::addExchangeContributionToFCVel(const ProcessorGroup*,  
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Add_exchange_contribution_to_FC_vel on patch " <<
      patch->getID() << "\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    double tmp, sp_vol_brack;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);

    StaticArray<constCCVariable<double> > vol_frac_CC(numMatls);
    StaticArray<constSFCXVariable<double> > uvel_FC(numMatls);
    StaticArray<constSFCYVariable<double> > vvel_FC(numMatls);
    StaticArray<constSFCZVariable<double> > wvel_FC(numMatls);

    StaticArray<SFCXVariable<double> > uvel_FCME(numMatls);
    StaticArray<SFCYVariable<double> > vvel_FCME(numMatls);
    StaticArray<SFCZVariable<double> > wvel_FCME(numMatls);
    // lowIndex is the same for all vel_FC
    IntVector lowIndex(patch->getSFCXLowIndex()); 
    
    // Extract the momentum exchange coefficients
    vector<double> b(numMatls);
    vector<double> X(numMatls);
    DenseMatrix beta(numMatls,numMatls),a(numMatls,numMatls);
    DenseMatrix K(numMatls,numMatls), junk(numMatls,numMatls);

    beta.zero();
    a.zero();
    K.zero();
    getExchangeCoefficients( K, junk);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(sp_vol_CC[m],    lb->sp_vol_CCLabel,  indx, patch, 
                Ghost::AroundCells, 1);
      new_dw->get(vol_frac_CC[m],  lb->vol_frac_CCLabel,indx, patch, 
                Ghost::AroundCells, 1);
      new_dw->get(uvel_FC[m], lb->uvel_FCLabel, indx, patch, 
                Ghost::AroundCells, 2);
      new_dw->get(vvel_FC[m], lb->vvel_FCLabel, indx, patch, 
                Ghost::AroundCells, 2);
      new_dw->get(wvel_FC[m], lb->wvel_FCLabel, indx, patch, 
                Ghost::AroundCells, 2);

      new_dw->allocate(uvel_FCME[m], lb->uvel_FCMELabel, indx, patch);
      new_dw->allocate(vvel_FCME[m], lb->vvel_FCMELabel, indx, patch);
      new_dw->allocate(wvel_FCME[m], lb->wvel_FCMELabel, indx, patch);

      uvel_FCME[m].initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      vvel_FCME[m].initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      wvel_FCME[m].initialize(0.0, lowIndex,patch->getSFCZHighIndex());
    }   
    //__________________________________
    //    B O T T O M  F A C E -- B  E  T  A      
    //  Note this includes b[m][m]
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    //   - form off diagonal terms of (a) 
    int offset=0;   // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc 
    for(CellIterator iter=patch->getSFCYIterator(offset);!iter.done();iter++){
      IntVector cur = *iter;
      IntVector adj(cur.x(),cur.y()-1,cur.z()); 
      for(int m = 0; m < numMatls; m++) {
        for(int n = 0; n < numMatls; n++) {
          tmp = (vol_frac_CC[n][adj] + vol_frac_CC[n][cur]) * K[n][m];
   
          sp_vol_brack = (sp_vol_CC[m][adj] * sp_vol_CC[m][cur])/
                         (sp_vol_CC[m][adj] + sp_vol_CC[m][cur]);                         
          beta[m][n] = sp_vol_brack * delT * tmp;
          a[m][n] = -beta[m][n];
        }
      }
      //__________________________________
      //  F  O  R  M     M  A  T  R  I  X   (a)
      //  - Diagonal terms      
      for(int m = 0; m < numMatls; m++) {
       a[m][m] = 1.;
       for(int n = 0; n < numMatls; n++) {
         a[m][m] +=  beta[m][n];
       }
      }
      //__________________________________
      //    F  O  R  M     R  H  S  (b)     
      for(int m = 0; m < numMatls; m++) {
       b[m] = 0.0;
       for(int n = 0; n < numMatls; n++)  {
         b[m] += beta[m][n] * (vvel_FC[n][cur] - vvel_FC[m][cur]);
       }
      }
      //__________________________________
      //      S  O  L  V  E  
      //   - backout velocities           
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
       vvel_FCME[m][cur] = vvel_FC[m][cur] + X[m];
      }
    }

    //__________________________________
    //   L E F T  F A C E-- B  E  T  A      
    //  Note this includes b[m][m]
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    //   - form off diagonal terms of (a)
    for(CellIterator iter=patch->getSFCXIterator(offset);!iter.done();iter++){
      IntVector cur = *iter;
      IntVector adj(cur.x()-1,cur.y(),cur.z()); 

      for(int m = 0; m < numMatls; m++)  {
       for(int n = 0; n < numMatls; n++)  {
         tmp = (vol_frac_CC[n][adj] + vol_frac_CC[n][cur]) * K[n][m];

          sp_vol_brack = (sp_vol_CC[m][adj] * sp_vol_CC[m][cur])/
                         (sp_vol_CC[m][adj] + sp_vol_CC[m][cur]);
                           
          beta[m][n] = sp_vol_brack * delT * tmp;
          a[m][n] = -beta[m][n];
       }
      }
      //__________________________________
      //  F  O  R  M     M  A  T  R  I  X   (a)
      //  - Diagonal terms  
      for(int m = 0; m < numMatls; m++) {
       a[m][m] = 1.;
       for(int n = 0; n < numMatls; n++) {
         a[m][m] +=  beta[m][n];
       }
      }

      //__________________________________
      //    F  O  R  M     R  H  S  (b) 
      for(int m = 0; m < numMatls; m++)  {
       b[m] = 0.0;
       for(int n = 0; n < numMatls; n++)  {
         b[m] += beta[m][n] * (uvel_FC[n][cur] - uvel_FC[m][cur]);
       }
      }
      //__________________________________
      //      S  O  L  V  E
      //   - backout velocities
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
       uvel_FCME[m][cur] = uvel_FC[m][cur] + X[m];
      }   
    }
    //__________________________________
    //  B A C K  F A C E -- B  E  T  A      
    //  Note this includes b[m][m]
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    //   - form off diagonal terms of (a)
    for(CellIterator iter=patch->getSFCZIterator(offset);!iter.done();iter++){
      IntVector cur = *iter;
      IntVector adj(cur.x(),cur.y(),cur.z()-1); 
      for(int m = 0; m < numMatls; m++)  {
        for(int n = 0; n < numMatls; n++) {
          tmp = (vol_frac_CC[n][adj] + vol_frac_CC[n][cur]) * K[n][m];
  
          sp_vol_brack = (sp_vol_CC[m][adj] * sp_vol_CC[m][cur])/
                         (sp_vol_CC[m][adj] + sp_vol_CC[m][cur]);
                           
          beta[m][n] = sp_vol_brack * delT * tmp;
          a[m][n] = -beta[m][n];
        }
      }
      //__________________________________
      //  F  O  R  M     M  A  T  R  I  X   (a)
      // - Diagonal terms
      for(int m = 0; m < numMatls; m++) {
       a[m][m] = 1.;
       for(int n = 0; n < numMatls; n++) {
         a[m][m] +=  beta[m][n];
       }
      }
      //__________________________________
      //    F  O  R  M     R  H  S  (b)
      for(int m = 0; m < numMatls; m++) {
       b[m] = 0.0;
       for(int n = 0; n < numMatls; n++) {
         b[m] += beta[m][n] * (wvel_FC[n][cur] - wvel_FC[m][cur]);
       }
      }
      //__________________________________
      //      S  O  L  V  E
      //   - backout velocities 
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
       wvel_FCME[m][cur] = wvel_FC[m][cur] + X[m];
      }
    }
    
    for (int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(uvel_FCME[m],"Velocity","x",patch,indx);
      setBC(vvel_FCME[m],"Velocity","y",patch,indx);
      setBC(wvel_FCME[m],"Velocity","z",patch,indx);
    }

   //---- P R I N T   D A T A ------ 
    if (switchDebug_Exchange_FC ) {
      for (int m = 0; m < numMatls; m++)  {
       Material* matl = d_sharedState->getMaterial( m );
       int indx = matl->getDWIndex();
       ostringstream desc;
       desc <<"Exchange_FC_after_BC_Mat_" << indx <<"_patch_"<<patch->getID();
       printData_FC( patch,1, desc.str(), "uvel_FCME", uvel_FCME[m]);
       printData_FC( patch,1, desc.str(), "vvel_FCME", vvel_FCME[m]);
       printData_FC( patch,1, desc.str(), "wvel_FCME", wvel_FCME[m]);
      }
    }

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(uvel_FCME[m], lb->uvel_FCMELabel, indx, patch);
      new_dw->put(vvel_FCME[m], lb->vvel_FCMELabel, indx, patch);
      new_dw->put(wvel_FCME[m], lb->wvel_FCMELabel, indx, patch);
    }
  }  // patch loop  
}

/*---------------------------------------------------------------------
 Function~  ICE::computeDelPressAndUpdatePressCC--
 Purpose~
   This function calculates the change in pressure explicitly. 
 Note:  Units of delp_Dilatate and delP_MassX are [Pa]
 Reference:  Multimaterial Formalism eq. 1.5
 ---------------------------------------------------------------------  */
void ICE::computeDelPressAndUpdatePressCC(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* old_dw, 
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    cout_doing << "Doing explicit delPress on patch " << patch->getID() 
         <<  "\t\t\t ICE" << endl;

    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();

    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;
    double compressibility;

    
    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_CC, q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> press_CC;
    CCVariable<double> term1, term2, term3;
    constCCVariable<double> burnedMass;
    constCCVariable<double> pressure;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
   
    const IntVector gc(1,1,1);
    new_dw->allocate(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocate(delP_MassX,   lb->delP_MassXLabel, 0, patch);
    new_dw->allocate(press_CC,     lb->press_CCLabel,   0, patch);
    new_dw->allocate(q_advected,   lb->q_advectedLabel, 0, patch);
    new_dw->allocate(term1,        lb->term1Label,      0, patch);
    new_dw->allocate(term2,        lb->term2Label,      0, patch);
    new_dw->allocate(term3,        lb->term3Label,      0, patch);
    new_dw->allocate(q_CC, lb->q_CCLabel, 0, patch, Ghost::AroundCells,1);
    new_dw->get(pressure,  lb->press_equil_CCLabel,0,patch,Ghost::None,0);

    term1.initialize(0.);
    term2.initialize(0.);
    term3.initialize(0.);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      constCCVariable<double> speedSound;
      constCCVariable<double> vol_frac;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> rho_CC;
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      new_dw->get(uvel_FC, lb->uvel_FCMELabel,   indx,  patch,
                Ghost::AroundCells, 2);
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,   indx,  patch,
                Ghost::AroundCells, 2);
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,   indx,  patch,
                Ghost::AroundCells, 2);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,  patch,
                Ghost::AroundCells,1);
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,Ghost::None,0);
      new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel,   indx,patch,Ghost::None,0);
      new_dw->get(burnedMass,  lb->burnedMass_CCLabel,indx,patch,Ghost::None,0);
      if(d_EqForm) {
        new_dw->get(speedSound,lb->speedSound_CCLabel,indx,patch,Ghost::None,0);
      }
      if(ice_matl) {
        old_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }
      if(mpm_matl) {
        new_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }

      //__________________________________
      // Advection preprocessing
      // - divide vol_frac_cc/vol
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch);

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();
         iter++) {
       IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      //__________________________________
      //   First order advection of q_CC
      advector->advectQ(q_CC,patch,q_advected);

      //---- P R I N T   D A T A ------  
      if (switchDebug_explicit_press ) {
        ostringstream desc;
        desc<<"middle_explicit_Pressure_Mat_"<<indx<<"_patch_"<<patch->getID();
        printData_FC( patch,1, desc.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, desc.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, desc.str(), "wvel_FC", wvel_FC);
      }    
      //__________________________________
      //   E Q   F O R M
      if(d_EqForm) {
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
          //   Contributions from reactions
          double inv_mass = sp_vol_CC[m][c] /vol;            
          term1[c] += (burnedMass[c]/delT) * (inv_mass);

          //   Divergence of velocity * face area 
          term2[c] -= q_advected[c];

          term3[c] += vol_frac[c] * sp_vol_CC[m][c]/
                              (speedSound[c]*speedSound[c]);                             
        }  //iter loop
      } 
      //__________________________________
      //    R A T E   F O R M
      if(d_RateForm) {
        if(mpm_matl) {
           compressibility =
                mpm_matl->getConstitutiveModel()->getCompressibility();
        }
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
          //   Contributions from reactions
          double inv_mass = sp_vol_CC[m][c] /vol;            
          term1[c] += (burnedMass[c]/delT) * (inv_mass);

          
          //   Divergence of velocity * face area
          term2[c] -= q_advected[c];

          if(ice_matl) {
            compressibility =
                     ice_matl->getEOS()->getCompressibility(pressure[*iter]);
          }
          // New term3 comes from Kashiwa 4.11
          term3[*iter] += vol_frac[*iter]*compressibility;
        }  //iter loop
      }  // if (RateForm)
    }  //matl loop
    delete advector;
    press_CC.initialize(0.);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    = (delT * term1[c])/term3[c];
      delP_Dilatate[c] = -term2[c]/term3[c];
      press_CC[c]      = pressure[c] + 
                             delP_MassX[c] + delP_Dilatate[c];    
    }
    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);

    new_dw->put(delP_Dilatate, lb->delP_DilatateLabel, 0, patch);
    new_dw->put(delP_MassX,    lb->delP_MassXLabel,    0, patch);
    new_dw->put(press_CC,      lb->press_CCLabel,      0, patch);

   //---- P R I N T   D A T A ------  
    if (switchDebug_explicit_press) {
      ostringstream desc;
      desc << "Bottom_of_explicit_Pressure_patch_" << patch->getID();
      printData( patch, 1,desc.str(), "delP_Dilatate", delP_Dilatate);
    //printData( patch, 1,desc.str(), "delP_MassX",    delP_MassX);
      printData( patch, 1,desc.str(), "Press_CC",      press_CC);
    }
  }  // patch loop
}

/* ---------------------------------------------------------------------  
 Function~  ICE::computePressFC--
 Purpose~
    This function calculates the face centered pressure on each of the 
    cell faces for every cell in the computational domain and a single 
    layer of ghost cells. 
  ---------------------------------------------------------------------  */

template <class T> void ICE::computePressFace(int& numMatls,CellIterator iter, 
                                         IntVector adj_offset,
                        StaticArray<constCCVariable<double> >& rho_CC,
                                        constCCVariable<double>& press_CC,
                                         T& press_FC)
{

  for(;!iter.done(); iter++){
    IntVector c = *iter;
    IntVector adj = c + adj_offset; 
    double sum_rho     = 0.0;
    double sum_rho_adj = 0.0;
    for(int m = 0; m < numMatls; m++) {
      sum_rho      += rho_CC[m][c];
      sum_rho_adj  += rho_CC[m][adj];
    }
    
    press_FC[c] = (press_CC[c] * sum_rho_adj + press_CC[adj] * sum_rho)/
      (sum_rho + sum_rho_adj);
  }

}

void ICE::computePressFC(const ProcessorGroup*,   
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse*,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing press_face_MM on patch " << patch->getID() 
         << "\t\t\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    constCCVariable<double> press_CC;
    new_dw->get(press_CC,lb->press_CCLabel, 0, patch, 
               Ghost::AroundCells, 1);

    SFCXVariable<double> pressX_FC;
    SFCYVariable<double> pressY_FC;
    SFCZVariable<double> pressZ_FC;

    new_dw->allocate(pressX_FC,lb->pressX_FCLabel, 0, patch);
    new_dw->allocate(pressY_FC,lb->pressY_FCLabel, 0, patch);
    new_dw->allocate(pressZ_FC,lb->pressZ_FCLabel, 0, patch);

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(rho_CC[m],lb->rho_CCLabel,indx,patch,Ghost::AroundCells,1);
    }
    //__________________________________
    //  For each face compute the pressure

    computePressFace<SFCXVariable<double> >(numMatls,patch->getSFCXIterator(),
                                       adj_offset[0],rho_CC,press_CC,
                                       pressX_FC);

    computePressFace<SFCYVariable<double> >(numMatls,patch->getSFCYIterator(),
                                       adj_offset[1],rho_CC,press_CC,
                                       pressY_FC);

    computePressFace<SFCZVariable<double> >(numMatls,patch->getSFCZIterator(),
                                       adj_offset[2],rho_CC,press_CC,
                                       pressZ_FC);

    new_dw->put(pressX_FC,lb->pressX_FCLabel, 0, patch);
    new_dw->put(pressY_FC,lb->pressY_FCLabel, 0, patch);
    new_dw->put(pressZ_FC,lb->pressZ_FCLabel, 0, patch);

   //---- P R I N T   D A T A ------ 
    if (switchDebug_PressFC) {
      ostringstream desc;
      desc << "press_FC_patch_" <<patch->getID();
      printData_FC( patch,0,desc.str(), "press_FC_RIGHT", pressX_FC);
      printData_FC( patch,0,desc.str(), "press_FC_TOP",   pressY_FC);
      printData_FC( patch,0,desc.str(), "press_FC_FRONT", pressZ_FC);
    }
  }  // patch loop
}


/* ---------------------------------------------------------------------
 Function~  ICE::massExchange--
 ---------------------------------------------------------------------  */
void ICE::massExchange(const ProcessorGroup*,  
                     const PatchSubset* patches,
                     const MaterialSubset* /*matls*/,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing massExchange on patch " <<
      patch->getID() << "\t\t\t\t ICE" << endl;


   Vector dx        = patch->dCell();
   double vol       = dx.x()*dx.y()*dx.z();

   int numMatls   =d_sharedState->getNumMatls();
   int numICEMatls=d_sharedState->getNumICEMatls();
   StaticArray<constCCVariable<Vector> > vel_CC(numMatls);
   StaticArray<constCCVariable<double> > rho_CC(numMatls);
   StaticArray<constCCVariable<double> > Temp_CC(numMatls);
   StaticArray<CCVariable<double> > created_vol(numMatls);
   StaticArray<CCVariable<double> > burnedMass(numMatls);
   StaticArray<CCVariable<double> > int_eng_comb(numMatls);
   StaticArray<CCVariable<Vector> > mom_comb(numMatls);
   StaticArray<double> cv(numMatls);
   
   int react = -1;

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

      // Look for the reactant material
      if (matl->getRxProduct() == Material::reactant)
       react = matl->getDWIndex();

      int indx = matl->getDWIndex();
      old_dw->get(vel_CC[m], lb->vel_CCLabel, indx,patch,Ghost::None, 0);
      new_dw->get(rho_CC[m], lb->rho_CCLabel, indx,patch,Ghost::None, 0);
      old_dw->get(Temp_CC[m],lb->temp_CCLabel,indx,patch,Ghost::None, 0);
      new_dw->allocate(burnedMass[m],  lb->burnedMass_CCLabel,  indx,patch);
      new_dw->allocate(int_eng_comb[m],lb->int_eng_comb_CCLabel,indx,patch);
      new_dw->allocate(created_vol[m], lb->created_vol_CCLabel, indx,patch);
      new_dw->allocate(mom_comb[m],    lb->mom_comb_CCLabel,  indx,patch);
      burnedMass[m].initialize(0.0);
      int_eng_comb[m].initialize(0.0); 
      created_vol[m].initialize(0.0);
      mom_comb[m].initialize(0.0);
      cv[m] = ice_matl->getSpecificHeat();
    }
    //__________________________________
    // Do the exchange if there is a reactant (react >= 0)
    // and the switch is on.
    if(d_massExchange && (react >= 0)){       
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        double mass_hmx = rho_CC[react][c] * vol;
        if (mass_hmx > d_SMALL_NUM)  {
          double burnedMass_tmp  = (rho_CC[react][c] * vol/2.0); 
          burnedMass[react][c]   = -burnedMass_tmp;
          int_eng_comb[react][c] = -burnedMass_tmp
                                   * cv[react]               
                                   * Temp_CC[react][c];    
          mom_comb[react][c]     = -vel_CC[react][c] * burnedMass_tmp;
          // Commented out for now as I'm not sure that this is appropriate
          // for regular ICE - Jim 7/30/01
//          created_vol[react][c]  =
//                          -burnedMass_tmp/rho_micro_CC[react][c];
        }
      }
      //__________________________________
      // Find the ICE matl which is the products of reaction
      // dump all the mass into that matl.
      // Make this faster
      for(int prods = 0; prods < numICEMatls; prods++) {
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(prods);
        if (ice_matl->getRxProduct() == Material::product) {
          for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
            IntVector c = *iter;
            burnedMass[prods][c]  = -burnedMass[react][c];
            int_eng_comb[prods][c]= -int_eng_comb[react][c];
            mom_comb[prods][c]    = -mom_comb[react][c];
           // Commented out for now as I'm not sure that this is appropriate
          // for regular ICE - Jim 7/30/01
//             created_vol[prods][c]  -= created_vol[m][c];
         }
       }
      }    
    }
    //__________________________________
    // if there is no mass exchange carry forward
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(burnedMass[m],   lb->burnedMass_CCLabel,   indx,patch);
      new_dw->put(int_eng_comb[m], lb->int_eng_comb_CCLabel, indx,patch);
      new_dw->put(mom_comb[m],     lb->mom_comb_CCLabel,     indx,patch);
      new_dw->put(created_vol[m],  lb->created_vol_CCLabel,  indx,patch);
    }

#if 0    // turn off for quality control tests
    //---- P R I N T   D A T A ------ 
    for(int m = 0; m < numMatls; m++) {
      if (switchDebugSource_Sink) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc << "sources/sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printData(patch, 0, desc.str(),"burnedMass",   burnedMass[m]);
        printData(patch, 0, desc.str(),"int_eng_comb", int_eng_comb[m]);
        printData(patch, 0, desc.str(),"mom_comb",   mom_comb[m]);
      }

    }
#endif
  }   // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::accumulateMomentumSourceSinks--
 Purpose~   This function accumulates all of the sources/sinks of momentum
 ---------------------------------------------------------------------  */
void ICE::accumulateMomentumSourceSinks(const ProcessorGroup*,  
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw, 
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing accumulate_momentum_source_sinks_MM on patch " <<
      patch->getID() << "\t ICE" << endl;

    int indx;
    int numMatls  = d_sharedState->getNumMatls();

    IntVector right, left, top, bottom, front, back;
    Vector dx, gravity;
    double pressure_source, mass, vol;
    double viscous_source;
    double viscosity;
    double include_term;

    delt_vartype delT; 
    old_dw->get(delT, d_sharedState->get_delt_label());
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);
 
    dx      = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    gravity = d_sharedState->getGravity();
    vol     = delX * delY * delZ;
    constCCVariable<double>   rho_CC;
    constCCVariable<Vector>   vel_CC;
    constCCVariable<double>   vol_frac;
    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;

    new_dw->get(pressX_FC,lb->pressX_FCLabel, 0, patch,Ghost::AroundCells, 1);
    new_dw->get(pressY_FC,lb->pressY_FCLabel, 0, patch,Ghost::AroundCells, 1);
    new_dw->get(pressZ_FC,lb->pressZ_FCLabel, 0, patch,Ghost::AroundCells, 1);

  //__________________________________
  //  Matl loop 
    for(int m = 0; m < numMatls; m++) {
      Material* matl        = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      indx = matl->getDWIndex();

      new_dw->get(rho_CC,  lb->rho_CCLabel,      indx,patch,Ghost::None, 0);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,patch,Ghost::None, 0);
      CCVariable<Vector>   mom_source;
      new_dw->allocate(mom_source,  lb->mom_source_CCLabel,  indx, patch);
      mom_source.initialize(Vector(0.,0.,0.));

      if(doMechOld < -1.5){
      //__________________________________
      // Compute Viscous Terms 
      SFCXVariable<Vector> tau_X_FC;
      SFCYVariable<Vector> tau_Y_FC;
      SFCZVariable<Vector> tau_Z_FC;  
      // note tau_*_FC is the same size as press(*)_FC
      tau_X_FC.allocate(pressX_FC.getLowIndex(), pressX_FC.getHighIndex());
      tau_Y_FC.allocate(pressY_FC.getLowIndex(), pressY_FC.getHighIndex());
      tau_Z_FC.allocate(pressZ_FC.getLowIndex(), pressZ_FC.getHighIndex());
      
      tau_X_FC.initialize(Vector(0.,0.,0.));
      tau_Y_FC.initialize(Vector(0.,0.,0.));
      tau_Z_FC.initialize(Vector(0.,0.,0.));
      viscosity = 0.0;
      if(ice_matl){
        old_dw->get(vel_CC, lb->vel_CCLabel,  indx,patch,Ghost::AroundCells, 2); 
        viscosity = ice_matl->getViscosity();
        if(viscosity != 0.0){  
          computeTauX_Components( patch, vel_CC, viscosity, dx, tau_X_FC);
          computeTauY_Components( patch, vel_CC, viscosity, dx, tau_Y_FC);
          computeTauZ_Components( patch, vel_CC, viscosity, dx, tau_Z_FC);
        }
        include_term = 1.0;
        // This multiplies terms that are only included in the ice_matls
      }
      if(mpm_matl){
        include_term = 0.0;
      }
      //__________________________________
      //  accumulate sources
      if(d_EqForm){
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          mass = rho_CC[c] * vol;
          right    = c + IntVector(1,0,0);
          left     = c + IntVector(0,0,0);
          top      = c + IntVector(0,1,0);
          bottom   = c + IntVector(0,0,0);
          front    = c + IntVector(0,0,1);
          back     = c + IntVector(0,0,0);
          //__________________________________
          //  WARNING:  Note that vol_frac * div Tau
          //   is not right.  It should be div (vol_frac Tau)
          //   Just trying it out.
          //__________________________________
          //    X - M O M E N T U M 
          pressure_source = (pressX_FC[right]-pressX_FC[left]) * vol_frac[c];

          viscous_source=(tau_X_FC[right].x() - tau_X_FC[left].x())  *delY*delZ+
                         (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())*delX*delZ+
                         (tau_Z_FC[front].x() - tau_Z_FC[back].x())  *delX*delY;

          mom_source[c].x( (-pressure_source * delY * delZ +
                             vol_frac[c] * viscous_source +
                             mass * gravity.x() * include_term) * delT );
          //__________________________________
          //    Y - M O M E N T U M
           pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c];

          viscous_source=(tau_X_FC[right].y() - tau_X_FC[left].y())  *delY*delZ+
                         (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())*delX*delZ+
                         (tau_Z_FC[front].y() - tau_Z_FC[back].y())  *delX*delY;

          mom_source[c].y( (-pressure_source * delX * delZ +
                             vol_frac[c] * viscous_source +
                             mass * gravity.y() * include_term) * delT );       
        //__________________________________
        //    Z - M O M E N T U M
          pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c];
        
          viscous_source=(tau_X_FC[right].z() - tau_X_FC[left].z())  *delY*delZ+
                         (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())*delX*delZ+
                         (tau_Z_FC[front].z() - tau_Z_FC[back].z())  *delX*delY;

          mom_source[c].z( (-pressure_source * delX * delY +
                           vol_frac[c] * viscous_source + 
                           mass * gravity.z() * include_term) * delT );
                                 
        }
      }
      if(d_RateForm){
        constSFCXVariable<double> press_diffX_FC;
        constSFCYVariable<double> press_diffY_FC;
        constSFCZVariable<double> press_diffZ_FC;
        new_dw->get(press_diffX_FC,
                   lb->press_diffX_FCLabel,indx,patch,Ghost::AroundCells, 1);
        new_dw->get(press_diffY_FC,
                   lb->press_diffY_FCLabel,indx,patch,Ghost::AroundCells, 1);
        new_dw->get(press_diffZ_FC,
                   lb->press_diffZ_FCLabel,indx,patch,Ghost::AroundCells, 1);
        double press_diff_source;
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          mass = rho_CC[c] * vol;
          right    = c + IntVector(1,0,0);
          left     = c + IntVector(0,0,0);
          top      = c + IntVector(0,1,0);
          bottom   = c + IntVector(0,0,0);
          front    = c + IntVector(0,0,1);
          back     = c + IntVector(0,0,0);

          // TODD:  Note that here, as in computeFCVelocities, I'm putting
          // a negative sign in front of press_diff_source, since sigma=-p*I

          //__________________________________
          //    X - M O M E N T U M    R A T E   F O R M
          pressure_source = (pressX_FC[right]-pressX_FC[left]) * vol_frac[c];

          press_diff_source = (press_diffX_FC[right]-press_diffX_FC[left]);

          viscous_source=(tau_X_FC[right].x() - tau_X_FC[left].x())  *delY*delZ+
                         (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())*delX*delZ+
                         (tau_Z_FC[front].x() - tau_Z_FC[back].x())  *delX*delY;

          mom_source[c].x( (-pressure_source * delY * delZ +
                             viscous_source -
                             press_diff_source * delY * delZ * include_term +
                             mass * gravity.x() * include_term) * delT);
          //__________________________________
          //    Y - M O M E N T U M   R A T E   F O R M
          pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c];

          press_diff_source = (press_diffY_FC[top]-press_diffY_FC[bottom]);

          viscous_source=(tau_X_FC[right].y() - tau_X_FC[left].y())  *delY*delZ+
                         (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())*delX*delZ+
                         (tau_Z_FC[front].y() - tau_Z_FC[back].y())  *delX*delY;

          mom_source[c].y( (-pressure_source * delX * delZ +
                             viscous_source -
                             press_diff_source * delX * delZ * include_term +
                             mass * gravity.y() * include_term) * delT );
          //__________________________________
          //    Z - M O M E N T U M   R A T E   F O R M
          pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c];

          press_diff_source = (press_diffZ_FC[front]-press_diffZ_FC[back]);

          viscous_source=(tau_X_FC[right].z() - tau_X_FC[left].z())  *delY*delZ+
                         (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())*delX*delZ+
                         (tau_Z_FC[front].z() - tau_Z_FC[back].z())  *delX*delY;

          mom_source[c].z( (-pressure_source * delX * delY +
                             viscous_source -
                             press_diff_source * delX * delY * include_term +
                             mass * gravity.z() * include_term) * delT );

        }
      }
      } // if doMechOld
      new_dw->put(mom_source, lb->mom_source_CCLabel, indx, patch);
      new_dw->put(doMechOld,  lb->doMechLabel);

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc << "sources/sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printVector(patch, 1, desc.str(), "xmom_source", 0, mom_source);
        printVector(patch, 1, desc.str(), "ymom_source", 1, mom_source);
        printVector(patch, 1, desc.str(), "zmom_source", 2, mom_source);
      }
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  ICE::accumulateEnergySourceSinks--
 Purpose~   This function accumulates all of the sources/sinks of energy 
 Currently the kinetic energy isn't included.
 ---------------------------------------------------------------------  */
void ICE::accumulateEnergySourceSinks(const ProcessorGroup*,  
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw)
{
#ifdef ANNULUSICE
  static int n_iter;
  n_iter ++;
#endif

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing accumulate_energy_source_sinks on patch " 
         << patch->getID() << "\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx = patch->dCell();
    double A, B, vol=dx.x()*dx.y()*dx.z();
 
    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> speedSound;
    constCCVariable<double> vol_frac;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;

    new_dw->get(press_CC,     lb->press_CCLabel,      0, patch,Ghost::None, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,Ghost::None, 0);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      new_dw->get(sp_vol_CC,   lb->sp_vol_CCLabel,    indx,patch,Ghost::None,0);
      new_dw->get(speedSound,  lb->speedSound_CCLabel,indx,patch,Ghost::None,0);
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,  indx,patch,Ghost::None,0);

#ifdef ANNULUSICE
      CCVariable<double> rho_CC;
      new_dw->get(rho_CC,      lb->rho_CCLabel,       indx,patch,Ghost::None,0);
#endif
      new_dw->allocate(int_eng_source,  lb->int_eng_source_CCLabel, indx,patch);

      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      int_eng_source.initialize(0.);
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        A = vol * vol_frac[c] * press_CC[c] * sp_vol_CC[c];
        B = speedSound[c] * speedSound[c];
        int_eng_source[c] = (A/B) * delP_Dilatate[c];
      }

#ifdef ANNULUSICE
      if(n_iter <= 1.e10){
        if(m==2){
          for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
            IntVector c = *iter;
            int_eng_source[c] += 1.e10 * delT * rho_CC[c] * vol;
          }
        }
      }
#endif

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc <<  "sources/sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printData(patch,1,desc.str(),"int_eng_source", int_eng_source);
      }
      new_dw->put(int_eng_source, lb->int_eng_source_CCLabel, indx,patch);
    }  // matl loop
  }  // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::computeLagrangianValues--
 Computes lagrangian mass momentum and energy
 Note:    Only loop over ICE materials, mom_L, massL and int_eng_L
           for MPM is computed in computeLagrangianValuesMPM()
 ---------------------------------------------------------------------  */
void ICE::computeLagrangianValues(const ProcessorGroup*,  
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing Lagrangian mass, momentum and energy on patch " <<
      patch->getID() << "\t ICE" << endl;

    int numALLMatls = d_sharedState->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();    
    
    //__________________________________ 
    //  Compute the Lagrangian quantities
    for(int m = 0; m < numALLMatls; m++) {
     Material* matl = d_sharedState->getMaterial( m );
     int indx = matl->getDWIndex();
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     CCVariable<Vector> mom_L; 
     CCVariable<double> int_eng_L; 
     CCVariable<double> mass_L;
     if(ice_matl)  {               //  I C E
      constCCVariable<double> rho_CC, temp_CC;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_source, burnedMass;
      constCCVariable<double> int_eng_comb;
      constCCVariable<Vector> mom_source;
      constCCVariable<Vector> mom_comb;

      new_dw->get(rho_CC,  lb->rho_CCLabel,     indx,patch,Ghost::None, 0);
      old_dw->get(vel_CC,  lb->vel_CCLabel,     indx,patch,Ghost::None, 0);
      old_dw->get(temp_CC, lb->temp_CCLabel,    indx,patch,Ghost::None, 0);
      new_dw->get(burnedMass,     lb->burnedMass_CCLabel,indx,patch,
                                                           Ghost::None, 0);
      new_dw->get(int_eng_comb,   lb->int_eng_comb_CCLabel,indx,patch,
                                                           Ghost::None, 0);
      new_dw->get(mom_source,     lb->mom_source_CCLabel,indx,patch,
                                                           Ghost::None, 0);
      new_dw->get(mom_comb,       lb->mom_comb_CCLabel,  indx,patch,
                                                           Ghost::None, 0);
      new_dw->get(int_eng_source, lb->int_eng_source_CCLabel,indx,patch,
                                                           Ghost::None, 0);
      new_dw->allocate(mom_L,     lb->mom_L_CCLabel,     indx,patch);
      new_dw->allocate(int_eng_L, lb->int_eng_L_CCLabel, indx,patch);
      new_dw->allocate(mass_L,    lb->mass_L_CCLabel,    indx,patch);
      double cv = ice_matl->getSpecificHeat();
      //__________________________________
      //  NO mass exchange
      if(d_massExchange == false) {
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
           iter++) {
         IntVector c = *iter;
          double mass = rho_CC[c] * vol;
          mass_L[c] = mass;

          mom_L[c] = vel_CC[c] * mass + mom_source[c];

          int_eng_L[c] = mass*cv * temp_CC[c] + int_eng_source[c];
        }
      }
      //__________________________________
      //  WITH mass exchange
      // Note that the mass exchange can't completely
      // eliminate all the mass, momentum and internal E
      // If it does then we'll get erroneous vel, and temps
      // after advection.  Thus there is always a mininum amount
      if(d_massExchange)  {
        double massGain = 0.;
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
           iter++) {
         IntVector c = *iter;
           //  must have a minimum mass
          double mass = rho_CC[c] * vol;
          double min_mass = d_SMALL_NUM * vol;

          mass_L[c] = std::max( (mass + burnedMass[c] ), min_mass);

          massGain += burnedMass[c];
          
          //  must have a minimum momentum   
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons                         
            double min_mom_L = vel_CC[c](dir) * min_mass;

         // Todd:  I believe the second term here to be flawed as was
         // int_eng_tmp.  We should create a "burnedMomentum" or something
         // to do this right.  Someday.
            double mom_L_tmp = vel_CC[c](dir) * mass;
                             + mom_comb[c](dir);
  
             // Preserve the original sign on momemtum     
             // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp + d_SMALL_NUM)/
                                    (fabs(mom_L_tmp + d_SMALL_NUM));
            
            mom_L[c](dir) = mom_source[c](dir) +
                  plus_minus_one * std::max( fabs(mom_L_tmp), min_mom_L );
          }

          // must have a minimum int_eng   
          double min_int_eng = min_mass * cv * temp_CC[c];
          double int_eng_tmp = mass * cv * temp_CC[c];

          //  Glossary:
          //  int_eng_tmp    = the amount of internal energy for this
          //                   matl in this cell coming into this task
          //  int_eng_source = thermodynamic work = f(delP_Dilatation)
          //  int_eng_comb   = enthalpy of reaction gained by the
          //                   product gas, PLUS (OR, MINUS) the
          //                   internal energy of the reactant
          //                   material that was liberated in the
          //                   reaction
          // min_int_eng     = a small amount of internal energy to keep
          //                   the equilibration pressure from going nuts


          int_eng_L[c] = int_eng_tmp +
                             int_eng_source[c] +
                             int_eng_comb[c];

          int_eng_L[c] = std::max(int_eng_L[c], min_int_eng);
         }
       cout << "Mass gained by the gas this timestep = " << massGain << endl;
       }  //  if (mass exchange)
  
        //__________________________________
        //  B U L L E T   P R O O F I N G   
        // catch negative internal energies
        double plusMinusOne = 1.0;
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
             iter++) {
         IntVector c = *iter;
          plusMinusOne *= int_eng_L[c]/fabs(int_eng_L[c]);     
        }
        if (plusMinusOne < 0.0) {
         string warn = "ICE::computeLagrangianValues: Negative Internal energy or Temperature detected";
         throw InvalidValue(warn);
        }
        //---- P R I N T   D A T A ------ 
        // Dump out all the matls data
        if (switchDebugLagrangianValues ) {
          ostringstream desc;
          desc <<"Bot_Lagrangian_Values_Mat_"<<indx<< "_patch_"<<patch->getID();
          printVector(patch,1, desc.str(), "xmom_L_CC", 0, mom_L);
          printVector(patch,1, desc.str(), "ymom_L_CC", 1, mom_L);
          printVector(patch,1, desc.str(), "zmom_L_CC", 2, mom_L);
          printData(  patch,1, desc.str(), "int_eng_L_CC", int_eng_L); 

        }
        new_dw->put(mom_L,     lb->mom_L_CCLabel,     indx,patch);
        new_dw->put(int_eng_L, lb->int_eng_L_CCLabel, indx,patch);
        new_dw->put(mass_L,    lb->mass_L_CCLabel,    indx,patch);
      }  // if (ice_matl)
    }  // end numALLMatl loop
  }  // patch loop
}
/* ---------------------------------------------------------------------
 Function~  ICE::computeLagrangianSpecificVolumes--
 ---------------------------------------------------------------------  */
void ICE::computeLagrangianSpecificVolume(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* /*old_dw*/,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing computeLagrangianSpecificVolume " <<
      patch->getID() << "\t\t\t ICE" << endl;

    int numALLMatls = d_sharedState->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    constCCVariable<double> rho_CC, createdVol, sp_vol;
    //__________________________________
    //  Compute the Lagrangian specific volume
    for(int m = 0; m < numALLMatls; m++) {
     Material* matl = d_sharedState->getMaterial( m );
     int indx = matl->getDWIndex();
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     CCVariable<double> spec_vol_L;
     new_dw->allocate(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
     spec_vol_L.initialize(0.);
     if(ice_matl){
       new_dw->get(rho_CC,    lb->rho_CCLabel,        indx,patch,Ghost::None,0);
       new_dw->get(sp_vol,    lb->sp_vol_CCLabel,     indx,patch,Ghost::None,0);
       new_dw->get(createdVol,lb->created_vol_CCLabel,indx,patch,Ghost::None,0);

       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){ 
         IntVector c = *iter;
         spec_vol_L[c] = (rho_CC[c] * vol * sp_vol[c]) + createdVol[c];
       }
     }
     new_dw->put(spec_vol_L,     lb->spec_vol_L_CCLabel,     indx,patch);
    }  // end numALLMatl loop
  }  // patch loop
}

/*---------------------------------------------------------------------
 Function~  ICE::addExchangeToMomentumAndEnergy--
 Purpose~
   This function adds the energy exchange contribution to the 
   existing cell-centered lagrangian temperature

 Prerequisites:
            The face centered velocity for each material without
            the exchange must be solved prior to this routine.
            
                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_data_CC[1]  |    
|                                               |   |                |    
| -b21              (1+b21 + b23) -b32          |   |del_data_CC[2]  |    
|                                               |   |                | 
| -b31              -b32          (1+b31 + b32) |   |del_data_CC[2]  |

                        =
                        
                        (B)
| b12( data_CC[2] - data_CC[1] ) + b13 ( data_CC[3] -data_CC[1])    | 
|                                                                   |
| b21( data_CC[1] - data_CC[2] ) + b23 ( data_CC[3] -data_CC[2])    | 
|                                                                   |
| b31( data_CC[1] - data_CC[3] ) + b32 ( data_CC[2] -data_CC[3])    | 
 
 - set *_L_ME arrays = *_L arrays
 - convert flux variables
 Steps for each cell;
    1) Comute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for del_data_CC[*]
    4) Add del_data_CC[*] to the appropriate Lagrangian data
 - apply Boundary conditions to vel_CC and Temp_CC
 - Stuff fluxes mom_L_ME and int_eng_L_ME back into dw
 
 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 ---------------------------------------------------------------------  */
void ICE::addExchangeToMomentumAndEnergy(const ProcessorGroup*,  
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Heat and momentum exchange on patch " << 
      patch->getID() << "\t\t ICE" << endl;

    int     numMatls  = d_sharedState->getNumICEMatls();
    double  tmp;
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    StaticArray<CCVariable<double> > Temp_CC(numMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numMatls);
    StaticArray<CCVariable<Vector> > mom_L_ME(numMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numMatls);
    StaticArray<CCVariable<double> > Tdot(numMatls);
    StaticArray<constCCVariable<double> > mass_L(numMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numMatls);
    StaticArray<constCCVariable<double> > vol_frac_CC(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numMatls);
    StaticArray<constCCVariable<double> > old_temp(numMatls);

    vector<double> b(numMatls);
    vector<double> cv(numMatls);
    vector<double> X(numMatls);
    DenseMatrix beta(numMatls,numMatls),acopy(numMatls,numMatls);
    DenseMatrix a_inverse(numMatls, numMatls);
    DenseMatrix K(numMatls,numMatls),H(numMatls,numMatls),a(numMatls,numMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();
    getExchangeCoefficients(K, H);

    for(int m = 0; m < numMatls; m++)  {
      ICEMaterial* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(mass_L[m],   lb->mass_L_CCLabel, indx,patch,Ghost::None,0);
      new_dw->get(mom_L[m],    lb->mom_L_CCLabel,  indx,patch,Ghost::None,0);
      new_dw->get(int_eng_L[m],lb->int_eng_L_CCLabel, indx,patch,
                Ghost::None,0);
      new_dw->get(vol_frac_CC[m], lb->vol_frac_CCLabel,indx,patch, 
                Ghost::None,0);   
      new_dw->get(sp_vol_CC[m],   lb->sp_vol_CCLabel,   indx,patch,
                Ghost::None,0);    

      new_dw->allocate( mom_L_ME[m],   lb->mom_L_ME_CCLabel,    indx, patch);
      new_dw->allocate(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,indx, patch);
      new_dw->allocate( vel_CC[m],     lb->vel_CCLabel,         indx, patch);
      new_dw->allocate( Temp_CC[m],    lb->temp_CCLabel,        indx, patch);

      Temp_CC[m].initialize(0.0);
      int_eng_L_ME[m].initialize(0.0);
      vel_CC[m].initialize(Vector(0.0, 0.0, 0.0));
      mom_L_ME[m].initialize(Vector(0.0, 0.0, 0.0));
      
      cv[m] = matl->getSpecificHeat();
    } 
        
    if(d_RateForm) {    //  R A T E   F O R M
      for(int m = 0; m < numMatls; m++)  {
        ICEMaterial* matl = d_sharedState->getICEMaterial( m );
        int indx = matl->getDWIndex(); 
        old_dw->get(old_temp[m],   lb->temp_CCLabel,indx,patch, Ghost::None,0);
        new_dw->allocate( Tdot[m], lb->Tdot_CCLabel,indx,patch);
      }
    }

    //__________________________________
    // Convert vars. flux -> primitive 
    for (int m = 0; m < numMatls; m++) {
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector c = *iter;
	Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m]);
	vel_CC[m][c]  =  mom_L[m][c]/mass_L[m][c];
      }  
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      //   Form BETA matrix (a), off diagonal terms
      //  The beta and (a) matrix is common to all momentum exchanges
      for(int m = 0; m < numMatls; m++)  {
        tmp = sp_vol_CC[m][c];
        for(int n = 0; n < numMatls; n++) {
          beta[m][n] = delT * vol_frac_CC[n][c]  * K[n][m] * tmp;
          a[m][n] = -beta[m][n];
        }
      }

      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numMatls; m++) {
        a[m][m] = 1.0;
        for(int n = 0; n < numMatls; n++) {
          a[m][m] +=  beta[m][n];
        }
      }
      matrixInverse(numMatls, a, a_inverse);
      //---------- X - M O M E N T U M
      // -  F O R M   R H S   (b)
      // -  push a copy of (a) into the solver
      // -  Add exchange contribution to orig value
      for(int m = 0; m < numMatls; m++) {
        b[m] = 0.0;

        for(int n = 0; n < numMatls; n++) {
         b[m] += beta[m][n] *
           (vel_CC[n][c].x() - vel_CC[m][c].x());
        }
      }
      multiplyMatrixAndVector(numMatls,a_inverse,b,X);
      for(int m = 0; m < numMatls; m++) {
        vel_CC[m][c].x( vel_CC[m][c].x() + X[m] );
      }

      //---------- Y - M O M E N T U M
      for(int m = 0; m < numMatls; m++) {
        b[m] = 0.0;

        for(int n = 0; n < numMatls; n++) {
         b[m] += beta[m][n] *
           (vel_CC[n][c].y() - vel_CC[m][c].y());
        }
      }   
      multiplyMatrixAndVector(numMatls,a_inverse,b,X);
      for(int m = 0; m < numMatls; m++) {
         vel_CC[m][c].y( vel_CC[m][c].y() + X[m] );
      }

      //---------- Z - M O M E N T U M
      for(int m = 0; m < numMatls; m++)  {
        b[m] = 0.0;

        for(int n = 0; n < numMatls; n++) {
         b[m] += beta[m][n] *
           (vel_CC[n][c].z() - vel_CC[m][c].z());
        }
      } 
      multiplyMatrixAndVector(numMatls,a_inverse,b,X);
      for(int m = 0; m < numMatls; m++) {
         vel_CC[m][c].z( vel_CC[m][c].z() + X[m] );
      }

      //---------- E N E R G Y   E X C H A N G E
      //   Form BETA matrix (a) off diagonal terms
      for(int m = 0; m < numMatls; m++) {
        tmp = cv[m];
        for(int n = 0; n < numMatls; n++)  {
         beta[m][n]=delT * sp_vol_CC[m][c] * vol_frac_CC[n][c] * H[n][m]/tmp;
         a[m][n] = -beta[m][n];
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numMatls; m++) {
        a[m][m] = 1.;
        for(int n = 0; n < numMatls; n++)   {
         a[m][m] +=  beta[m][n];
        }
      }
      // -  F O R M   R H S   (b)
      for(int m = 0; m < numMatls; m++)  {
        b[m] = 0.0;

       for(int n = 0; n < numMatls; n++) {
         b[m] += beta[m][n] *
           (Temp_CC[n][c] - Temp_CC[m][c]);
        }
      }
      //     S O L V E, Add exchange contribution to orig value
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
        Temp_CC[m][c] = Temp_CC[m][c] + X[m];
      }
    }
    //__________________________________
    //  Set the Boundary condiitions
    for (int m = 0; m < numMatls; m++)  {
      ICEMaterial* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch,indx);
      setBC(Temp_CC[m],"Temperature",patch,d_sharedState,indx);
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for (int m = 0; m < numMatls; m++) {
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
	IntVector c = *iter;
        int_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c] * mass_L[m][c];
      }  
    }
    
    if (d_RateForm) {
      for (int m = 0; m < numMatls; m++) {
	for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
	  IntVector c = *iter;
          Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
        }  
      }
    } 

    for(int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(mom_L_ME[m],    lb->mom_L_ME_CCLabel,    indx, patch);
      new_dw->put(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,indx, patch);
      if(d_RateForm) {
        new_dw->put(Tdot[m],      lb->Tdot_CCLabel,        indx, patch);
      }
    }
  }  // patch loop
}

/* --------------------------------------------------------------------- 
 Function~  ICE::advectAndAdvanceInTime--
 Purpose~
   This function calculates the The cell-centered, time n+1, mass, momentum
   and internal energy

   Need to include kinetic energy 
 ---------------------------------------------------------------------  */
void ICE::advectAndAdvanceInTime(const ProcessorGroup*,  
                                 const PatchSubset* patches,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
 
    cout_doing << "Doing Advect and Advance in Time on patch " << 
      patch->getID() << "\t\t ICE" << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    Vector dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z(),mass;
    double invvol = 1.0/vol;

    // These arrays get re-used for each material, and for each
    // advected quantity
    const IntVector gc(1,1,1);
    CCVariable<double> q_CC, q_advected;
    CCVariable<Vector> qV_CC, qV_advected;
    Advector* advector = d_advector->clone(new_dw,patch);
    
    new_dw->allocate(q_advected, lb->q_advectedLabel, 0, patch);
    new_dw->allocate(qV_advected,lb->qV_advectedLabel,0, patch);
    new_dw->allocate(qV_CC,      lb->qV_CCLabel, 0,patch,Ghost::AroundCells,1);
    new_dw->allocate(q_CC,       lb->q_CCLabel,  0,patch,Ghost::AroundCells,1);

    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();

      CCVariable<double> rho_CC, temp, sp_vol_CC;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_L_ME, mass_L,spec_vol_L;
      constCCVariable<Vector> mom_L_ME;
     
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      new_dw->get(uvel_FC,lb->uvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(vvel_FC,lb->vvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(wvel_FC,lb->wvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(mass_L, lb->mass_L_CCLabel,indx,patch,Ghost::AroundCells,1);

      new_dw->get(mom_L_ME,    lb->mom_L_ME_CCLabel,
                                   indx,patch,Ghost::AroundCells,1);
      new_dw->get(spec_vol_L,  lb->spec_vol_L_CCLabel,
                                   indx,patch,Ghost::AroundCells,1);
      new_dw->get(int_eng_L_ME,lb->int_eng_L_ME_CCLabel,
                                   indx,patch,Ghost::AroundCells,1);
      new_dw->getModifiable(sp_vol_CC,  
                               lb->sp_vol_CCLabel,indx,patch);
      new_dw->getModifiable(rho_CC,  
                               lb->rho_CCLabel,indx,patch);

      new_dw->allocate(temp,      lb->temp_CCLabel,           indx,patch);
      new_dw->allocate(vel_CC,    lb->vel_CCLabel,            indx,patch);
      rho_CC.initialize(0.0);
      temp.initialize(0.0);
      vel_CC.initialize(Vector(0.0,0.0,0.0));
      qV_advected.initialize(Vector(0.0,0.0,0.0));
      double cv = ice_matl->getSpecificHeat();
      //__________________________________
      //   Advection preprocessing
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch);

      //__________________________________
      // Advect mass and backout rho_CC
      for(CellIterator iter=patch->getCellIterator(gc); !iter.done();iter++){
        IntVector c = *iter;
        q_CC[c] = mass_L[c] * invvol;
      }

      advector->advectQ(q_CC,patch,q_advected);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        rho_CC[c]  = (mass_L[c] + q_advected[c]) * invvol;
      }
      setBC(rho_CC, "Density", patch, d_sharedState, indx);

      //__________________________________
      // Advect  momentum and backout vel_CC
      for(CellIterator iter=patch->getCellIterator(gc); !iter.done(); iter++){
        IntVector c = *iter;
        qV_CC[c] = mom_L_ME[c] * invvol;
      }

      advector->advectQ(qV_CC,patch,qV_advected);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        mass = rho_CC[c] * vol;
        vel_CC[c] = (mom_L_ME[c] + qV_advected[c])/mass ;
      }

      setBC(vel_CC, "Velocity", patch,indx);

      //__________________________________
      // Advect internal energy and backout Temp_CC
      for(CellIterator iter=patch->getCellIterator(gc); !iter.done(); iter++){
        IntVector c = *iter;
        q_CC[c] = int_eng_L_ME[c] * invvol;
      }

      advector->advectQ(q_CC,patch,q_advected);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        mass = rho_CC[c] * vol;
        temp[c] = (int_eng_L_ME[c] + q_advected[c])/(mass*cv);
      }
      setBC(temp, "Temperature", patch, d_sharedState, indx);

      //__________________________________
      // Advection of specific volume.  Advected quantity is a volume fraction
      // I am doing this so that we get a reasonable answer for sp_vol
      // in the extra cells.  This calculation will get overwritten in
      // the interior cells.
/*`==========TESTING==========*/
#if 0 
     // Jim I don't think we need this anymore since sp_vol_CC is compute elsewhere
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done(); iter++){
        IntVector c = *iter;
        sp_vol_CC[c] = 1.0/rho_micro[c];
      }
#endif
 /*==========TESTING==========`*/
      for(CellIterator iter=patch->getCellIterator(gc); !iter.done();iter++){
        IntVector c = *iter;
        q_CC[c] = spec_vol_L[c]*invvol;
      }

      advector->advectQ(q_CC,patch,q_advected);

      // After the following expression, sp_vol_CC is the matl volume
      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
       IntVector c = *iter;
       sp_vol_CC[c] = (q_CC[c]*vol + q_advected[c])/(rho_CC[c]*vol);
      }

      //---- P R I N T   D A T A ------   
      if (switchDebug_advance_advect ) {
       ostringstream desc;
       desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"<<patch->getID();
       printVector( patch,1, desc.str(), "xmom_L_CC", 0,mom_L_ME); 
       printVector( patch,1, desc.str(), "ymom_L_CC", 1,mom_L_ME); 
       printVector( patch,1, desc.str(), "zmom_L_CC", 2,mom_L_ME); 
       printData(   patch,1, desc.str(), "int_eng_L_CC",int_eng_L_ME);
       printData(   patch,1, desc.str(), "rho_CC",      rho_CC);
       printData(   patch,1, desc.str(), "Temp_CC",     temp);
       printVector( patch,1, desc.str(), "uvel_CC", 0,  vel_CC);
       printVector( patch,1, desc.str(), "vvel_CC", 1,  vel_CC);
       printVector( patch,1, desc.str(), "wvel_CC", 2,  vel_CC);
      }
      new_dw->put(vel_CC,   lb->vel_CCLabel,           indx,patch);
      new_dw->put(temp,     lb->temp_CCLabel,          indx,patch);

    }
    delete advector;
  }  // patch loop 
}

/* 
 ======================================================================*
 Function:  hydrostaticPressureAdjustment--
 Notes:
            This doesn't take the ghostcells into account, therefore you 
            must adjust boundary conditions after this function.  Material 0 is
            assumed to be the surrounding fluid and therfore we compute the 
            hydrostatic pressure using
            
            press_hydro_ = rho_micro_CC[SURROUNDING_MAT] * grav * some_distance
            
            Currently it is assumed that the top, right and front ghostcells is
            where the reference pressure is defined.
            
CAVEAT:     Only works -gravity.x(), -gravity.y() and -gravity.z()           
_______________________________________________________________________ */
void   ICE::hydrostaticPressureAdjustment(const Patch* patch,
                          const CCVariable<double>& sp_vol_CC,
                                CCVariable<double>& press_CC)
{
  Vector dx             = patch->dCell();
  Vector gravity        = d_sharedState->getGravity();
//  IntVector highIndex   = patch->getInteriorCellHighIndex();
  
                  //ONLY WORKS ON ONE PATCH  Press_ref_* will have to change
  double press_hydro;
  double dist_from_p_ref;
  IntVector HighIndex;
  IntVector L;
  const Level* level= patch->getLevel();
  level->findIndexRange(L, HighIndex);
  int press_ref_x = HighIndex.x() -2;   // we want the interiorCellHighIndex 
  int press_ref_y = HighIndex.y() -2;   // therefore we subtract off 2
  int press_ref_z = HighIndex.z() -2;

  //__________________________________
  //  X direction
  if (gravity.x() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector curcell = *iter;
      dist_from_p_ref   =  fabs((double) (curcell.x() - press_ref_x)) * dx.x();
      press_hydro       = 1.0/sp_vol_CC[curcell] * 
                          fabs(gravity.x() ) * dist_from_p_ref;
      
      press_CC[curcell] += press_hydro;
    }
  }
  //__________________________________
  //  Y direction
  if (gravity.y() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector curcell = *iter;
      dist_from_p_ref   = fabs((double) (curcell.y() - press_ref_y)) * dx.y();
      press_hydro       = 1.0/sp_vol_CC[curcell] * 
                          fabs(gravity.y() ) * dist_from_p_ref;
      
      press_CC[curcell] += press_hydro;
    }
  }
  //__________________________________
  //  Z direction
  if (gravity.z() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector curcell = *iter;
      dist_from_p_ref   = fabs((double) (curcell.z() - press_ref_z)) * dx.z();
      press_hydro       = 1.0/sp_vol_CC[curcell] * 
                          fabs(gravity.z() ) * dist_from_p_ref;
      
      press_CC[curcell] += press_hydro;
    }
  }   
}




/*---------------------------------------------------------------------
 Function~  ICE::computeTauX_Components
 Purpose:   This function computes shear stress tau_xx, ta_xy, tau_xz 
 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use 2 cells
            to compute it.  When you take the difference of the edge velocities 
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially.   
 ---------------------------------------------------------------------  */
void ICE::computeTauX_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCXVariable<Vector>& tau_X_FC)
{
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
  //__________________________________
  // loop over the left cell faces
  // For multipatch problems adjust the iter limits
  // on the left patches to include the right face
  // of the cell at the patch boundary. 
  // We compute tau_ZZ[right]-tau_XX[left] on each patch
  CellIterator hi_lo = patch->getSFCXIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?1:0,
		   patch->getBCType(patch->yplus) ==patch->Neighbor?0:0,
		   patch->getBCType(patch->zplus) ==patch->Neighbor?0:0); 
  CellIterator iterLimits(low,hi); 
  
  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector cell = *iter;
    int i = cell.x();
    int j = cell.y();
    int k = cell.z();

    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector left(i-1, j, k);
    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].x() )/4.0;    
    double uvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].x() )/4.0;
    double uvel_EC_front  = (vel_CC[IntVector(i-1,j  ,k+1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].x() )/4.0;
    double uvel_EC_back   = (vel_CC[IntVector(i-1,j  ,k-1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].x() )/4.0;
    double vvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].y() )/4.0;
    double vvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].y() )/4.0;
    double wvel_EC_front  = (vel_CC[IntVector(i-1,j-1,k+1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].z() )/4.0;
    double wvel_EC_back   = (vel_CC[IntVector(i-1,j-1,k-1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].z() )/4.0;
    //__________________________________
    //  tau_XX
    grad_uvel = (vel_CC[cell].x() - vel_CC[left].x())/delX;
    grad_vvel = (vvel_EC_top      - vvel_EC_bottom)  /delY;
    grad_wvel = (wvel_EC_front    - wvel_EC_back )   /delZ;

    term1 = 2.0 * viscosity * grad_uvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_X_FC[cell].x(term1 - term2);
    //__________________________________
    //  tau_XY
    grad_1 = (uvel_EC_top      - uvel_EC_bottom)  /delY;
    grad_2 = (vel_CC[cell].y() - vel_CC[left].y())/delX;
    tau_X_FC[cell].y(viscosity * (grad_1 + grad_2));

    //__________________________________
    //  tau_XZ
    grad_1 = (uvel_EC_front    - uvel_EC_back)    /delZ;
    grad_2 = (vel_CC[cell].z() - vel_CC[left].z())/delX;
    tau_X_FC[cell].z(viscosity * (grad_1 + grad_2));
    
//     if (i == 0 && k == 0){
//       cout<<cell<<" tau_XX: "<<tau_X_FC[cell].x()<<
//       " tau_XY: "<<tau_X_FC[cell].y()<<
//       " tau_XZ: "<<tau_X_FC[cell].z()<<
//       " patch: " <<patch->getID()<<endl;     
//     } 
  }
}


/*---------------------------------------------------------------------
 Function~  ICE::computeTauY_Components
 Purpose:   This function computes shear stress tau_YY, ta_yx, tau_yz 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use2 cells
            to compute it.  When you take the difference of the edge velocities 
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially. 
 ---------------------------------------------------------------------  */
void ICE::computeTauY_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCYVariable<Vector>& tau_Y_FC)
{
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
  //__________________________________
  // loop over the bottom cell faces
  // For multipatch problems adjust the iter limits
  // on the bottom patches to include the top face
  // of the cell at the patch boundary. 
  // We compute tau_YY[top]-tau_YY[bot] on each patch
  CellIterator hi_lo = patch->getSFCYIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?0:0,
		   patch->getBCType(patch->yplus) ==patch->Neighbor?1:0,
		   patch->getBCType(patch->zplus) ==patch->Neighbor?0:0); 
  CellIterator iterLimits(low,hi); 
  
  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector cell = *iter;
    int i = cell.x();
    int j = cell.y();
    int k = cell.z();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector bottom(i,j-1,k);
    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].x() )/4.0;
    double uvel_EC_left  = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                            vel_CC[IntVector(i-1,j-1,k  )].x() )/4.0;
    double vvel_EC_right = (vel_CC[IntVector(i+1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].y() )/4.0;
    double vvel_EC_left  = (vel_CC[IntVector(i-1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i-1,j-1,k  )].y() )/4.0; 
    double vvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].y() )/4.0;
    double vvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].y() )/4.0;
    double wvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].z() )/4.0;
    double wvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].z() )/4.0;
    //__________________________________
    //  tau_YY
    grad_uvel = (uvel_EC_right    - uvel_EC_left)      /delX;
    grad_vvel = (vel_CC[cell].y() - vel_CC[bottom].y())/delY;
    grad_wvel = (wvel_EC_front    - wvel_EC_back )     /delZ;

    term1 = 2.0 * viscosity * grad_vvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_Y_FC[cell].y(term1 - term2);
    
    //__________________________________
    //  tau_YX
    grad_1 = (vel_CC[cell].x() - vel_CC[bottom].x())/delY;
    grad_2 = (vvel_EC_right    - vvel_EC_left)    /delX;

    tau_Y_FC[cell].x(viscosity * (grad_1 + grad_2) );

    //__________________________________
    //  tau_YZ
    grad_1 = (vvel_EC_front    - vvel_EC_back)    /delZ;
    grad_2 = (vel_CC[cell].z() - vel_CC[bottom].z())/delY;
    tau_Y_FC[cell].z(viscosity * (grad_1 + grad_2));
    
//     if (i == 0 && k == 0){    
//       cout<< cell<< " tau_YX: "<<tau_Y_FC[cell].x()<<
//       " tau_YY: "<<tau_Y_FC[cell].y()<<
//       " tau_YZ: "<<tau_Y_FC[cell].z()<<
//        " patch: "<<patch->getID()<<endl;
//     }
  }
}

/*---------------------------------------------------------------------
 Function~  ICE::computeTauZ_Components
 Purpose:   This function computes shear stress tau_zx, ta_zy, tau_zz 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use 2 cells
            to compute it.  When you take the difference of the edge velocities 
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially.
 ---------------------------------------------------------------------  */
void ICE::computeTauZ_Components( const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const double viscosity,
                          const Vector dx,
                          SFCZVariable<Vector>& tau_Z_FC)
{
  double term1, term2, grad_1, grad_2;
  double grad_uvel, grad_vvel, grad_wvel;
 
  //__________________________________
  // loop over the back cell faces
  // For multipatch problems adjust the iter limits
  // on the back patches to include the front face
  // of the cell at the patch boundary. 
  // We compute tau_ZZ[front]-tau_ZZ[back] on each patch
  CellIterator hi_lo = patch->getSFCZIterator();
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?0:0,
		   patch->getBCType(patch->yplus) ==patch->Neighbor?0:0,
		   patch->getBCType(patch->zplus) ==patch->Neighbor?1:0); 
  CellIterator iterLimits(low,hi); 

  for(CellIterator iter = iterLimits;!iter.done();iter++){ 
    IntVector cell = *iter; 
    int i = cell.x();
    int j = cell.y();
    int k = cell.z();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    IntVector back(i, j, k-1);
    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                             vel_CC[IntVector(i+1,j,  k-1)].x() )/4.0;
    double uvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].x() )/4.0;
    double vvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j+1,k-1)].y() )/4.0;
    double vvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j-1,k-1)].y() )/4.0;
    double wvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].z()  + 
                             vel_CC[IntVector(i+1,j  ,k-1)].z() )/4.0;
    double wvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].z()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].z() )/4.0;
    double wvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].z()   + 
                             vel_CC[IntVector(i  ,j+1,k-1)].z() )/4.0;
    double wvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].z()   + 
                             vel_CC[IntVector(i  ,j-1,k-1)].z() )/4.0;
    //__________________________________
    //  tau_ZX
    grad_1 = (vel_CC[cell].x() - vel_CC[back].x()) /delZ;
    grad_2 = (wvel_EC_right    - wvel_EC_left)     /delX;
    tau_Z_FC[cell].x(viscosity * (grad_1 + grad_2));

    //__________________________________
    //  tau_ZY
    grad_1 = (vel_CC[cell].y() - vel_CC[back].y()) /delZ;
    grad_2 = (wvel_EC_top      - wvel_EC_bottom)   /delX;
    tau_Z_FC[cell].y( viscosity * (grad_1 + grad_2) );

    //__________________________________
    //  tau_ZZ
    grad_uvel = (uvel_EC_right    - uvel_EC_left)    /delX;
    grad_vvel = (vvel_EC_top      - vvel_EC_bottom)  /delY;
    grad_wvel = (vel_CC[cell].z() - vel_CC[back].z())/delZ;

    term1 = 2.0 * viscosity * grad_wvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_Z_FC[cell].z( term1 - term2);
//  cout<<"tau_ZX: "<<tau_Z_FC[cell].x()<<
//        " tau_ZY: "<<tau_Z_FC[cell].y()<<
//        " tau_ZZ: "<<tau_Z_FC[cell].z()<<endl;
  }
}
/*---------------------------------------------------------------------
 Function~  ICE::getExchangeCoefficients--
 ---------------------------------------------------------------------  */
void ICE::getExchangeCoefficients( DenseMatrix& K, DenseMatrix& H  )
{
  int numMatls  = d_sharedState->getNumMatls();
    // Fill in the exchange matrix with the vector of exchange coefficients.
   // The vector of exchange coefficients only contains the upper triagonal
   // matrix

   // Check if the # of coefficients = # of upper triangular terms needed
   int num_coeff = ((numMatls)*(numMatls) - numMatls)/2;

   vector<double>::iterator it=d_K_mom.begin(),it1=d_K_heat.begin();

   if (num_coeff == (int)d_K_mom.size() && num_coeff==(int)d_K_heat.size()) {
     // Fill in the upper triangular matrix
     for (int i = 0; i < numMatls; i++ )  {
      for (int j = i + 1; j < numMatls; j++) {
        K[i][j] = K[j][i] = *it++;
        H[i][j] = H[j][i] = *it1++;
      }
     }
   } else if (2*num_coeff==(int)d_K_mom.size() && 
             2*num_coeff == (int)d_K_heat.size()){
     // Fill in the whole matrix but skip the diagonal terms
     for (int i = 0; i < numMatls; i++ )  {
      for (int j = 0; j < numMatls; j++) {
        if (i == j) continue;
        K[i][j] = *it++;
        H[i][j] = *it1++;
      }
     }
   } else 
     throw InvalidValue("Number of exchange components don't match.");
  
}


#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif


/*______________________________________________________________________
          S H E M A T I C   D I A G R A M S

                                    q_outflux(TOP)

                                        |    (I/O)flux_EF(TOP_BK)
                                        |
  (I/O)flux_CF(TOP_L_BK)       _________|___________
                              /___/_____|_______/__/|   (I/O)flux_CF(TOP_R_BK)
                             /   /      |      /  | |
                            /   /       |     /  /| |
  (I/O)flux_EF(TOP_L)      /   /             /  / |/|
                          /___/_____________/__/ ------ (I/O)flux_EF(TOP_R)
                        _/__ /_____________/__/| /| | 
                        |   |             |  | |/ | |   (I/O)flux_EF(BCK_R)
                        | + |      +      | +| /  | |      
                        |---|----------------|/|  |/| 
                        |   |             |  | | /| /  (I/O)flux_CF(BOT_R_BK)
  (I/O)flux(LEFT_FR)    | + |     i,j,k   | +| |/ /          
                        |   |             |  |/| /   (I/O)flux_EF(BOT_R)
                        |---|----------------| |/
  (I/O)flux_CF(BOT_L_FR)| + |      +      | +|/    (I/O)flux_CF(BOT_R_FR)
                        ---------------------- 
                         (I/O)flux_EF(BOT_FR)       
                         
                                         
                         
                            (TOP)      
   ______________________              ______________________  _
   |   |             |  |              |   |             |  |  |  delY_top
   | + |      +      | +|              | + |      +      | +|  |
   |---|----------------|  --ytop      |---|----------------|  -
   |   |             |  |              |   |             |  |
   | + |     i,j,k   | +| (RIGHT)      | + |     i,j,k   | +|
   |   |             |  |              |   |             |  |
   |---|----------------|  --y0        |---|----------------|  -
   | + |      +      | +|              | + |      +      | +|  | delY_bottom
   ----------------------              ----------------------  -
       |             |                 |---|             |--|
       x0            xright              delX_left         delX_right
       
                            (BACK)
   ______________________              ______________________  _
   |   |             |  |              |   |             |  |  |  delZ_back
   | + |      +      | +|              | + |      +      | +|  |
   |---|----------------|  --z0        |---|----------------|  -
   |   |             |  |              |   |             |  |
   | + |     i,j,k   | +| (RIGHT)      | + |     i,j,k   | +|
   |   |             |  |              |   |             |  |
   |---|----------------|  --z_frt     |---|----------------|  -
   | + |      +      | +|              | + |      +      | +|  | delZ_front
   ----------------------              ----------------------  -
       |             |                 |---|             |--|
       x0            xright              delX_left         delX_right
                         
______________________________________________________________________*/ 

