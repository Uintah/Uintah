#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/PressureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/DensityBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <vector>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/StaticArray.h>
#include <sstream>
#include <float.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>

using std::vector;
using std::max;
using std::istringstream;
 
using namespace SCIRun;
using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG ICE_NORMAL_COUT:+, ICE_DOING_COUT.....
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
  switchDebug_equilibration_press = false;
  switchDebug_vel_FC              = false;
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
  
}

ICE::~ICE()
{
  delete lb;
  delete d_advector;

}

/* ---------------------------------------------------------------------
 Function~  ICE::problemSetup--
_____________________________________________________________________*/
void  ICE::problemSetup(const ProblemSpecP& prob_spec,GridP& ,
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
  // Find the switches
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {
    d_dbgStartTime = 0.;
    d_dbgStopTime  = 1.;
    d_dbgOutputInterval = 0.0;
    debug_ps->get("dbg_timeStart",     d_dbgStartTime);
    debug_ps->get("dbg_timeStop",      d_dbgStopTime);
    debug_ps->get("dbg_outputInterval",d_dbgOutputInterval);
    d_dbgOldTime      = -d_dbgOutputInterval;
    d_dbgNextDumpTime = 0.0;

    for (ProblemSpecP child = debug_ps->findBlock("debug"); child != 0;
	 child = child->findNextBlock("debug")) {
      map<string,string> debug_attr;
      child->getAttributes(debug_attr);
      if (debug_attr["label"]      == "switchDebugInitialize")
	switchDebugInitialize            = true;
      else if (debug_attr["label"] == "switchDebug_equilibration_press")
	switchDebug_equilibration_press  = true;
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

  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  cfd_ps->require("cfl",d_CFL);
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE");
  cfd_ice_ps->require("max_iteration_equilibration",d_max_iter_equilibration);
  d_advector = AdvectionFactory::create(cfd_ice_ps);
  cout_norm << "cfl = " << d_CFL << endl;
  cout_norm << "max_iteration_equilibration " << d_max_iter_equilibration << endl;
  cout_norm << "Pulled out CFD-ICE block of the input file" << endl;
    
  // Pull out from Time section
  d_initialDt = 10000.0;
  ProblemSpecP time_ps = prob_spec->findBlock("Time");
  time_ps->get("delt_init",d_initialDt);
  cout_norm << "Initial dt = " << d_initialDt << endl;
  cout_norm << "Pulled out Time block of the input file" << endl;

  // Pull out Initial Conditions
  ProblemSpecP mat_ps       =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");  

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    // Extract out the type of EOS and the 
    // associated parameters
     ICEMaterial *mat = scinew ICEMaterial(ps);
     sharedState->registerICEMaterial(mat);
  }     
  cout_norm << "Pulled out InitialConditions block of the input file" << endl;

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
  if (switchDebug_equilibration_press == true) 
    cout_norm << "switchDebug_equilibration_press is ON" << endl;
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
void ICE::scheduleInitialize(const LevelP& level, 
                             SchedulerP& sched)
{

  cout_doing << "Doing ICE::scheduleInitialize " << endl;
  Task* t = scinew Task("ICE::actuallyInitialize",
                  this, &ICE::actuallyInitialize);
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  t->computes(lb->doMechLabel);
  t->computes(lb->vel_CCLabel);
  t->computes(lb->temp_CCLabel);
  t->computes(lb->mass_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->rho_micro_CCLabel);
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->rho_CC_top_cycleLabel);
  t->computes(lb->press_CCLabel, press_matl);
  t->computes(d_sharedState->get_delt_label());

  sched->addTask(t, level->eachPatch(), d_sharedState->allICEMaterials());
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
void ICE::scheduleTimeAdvance(const LevelP& level,
			      SchedulerP& sched)
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

  
  scheduleComputeEquilibrationPressure(sched, patches, press_matl,
                                                       all_matls);

  scheduleComputeFaceCenteredVelocities(sched, patches, ice_matls_sub,
                                                        mpm_matls_sub,
                                                        press_matl, 
                                                        all_matls);

  scheduleAddExchangeContributionToFCVel(sched, patches, all_matls);    

  scheduleMassExchange(sched, patches, all_matls);

  scheduleComputeDelPressAndUpdatePressCC(sched, patches, press_matl,
                                                          ice_matls_sub, 
                                                          mpm_matls_sub,
                                                          all_matls);

  scheduleComputePressFC(sched, patches, press_matl,
                                        all_matls);

  scheduleAccumulateMomentumSourceSinks(sched, patches, press_matl,
                                        ice_matls_sub, all_matls);

  scheduleAccumulateEnergySourceSinks(sched, patches, press_matl,
                                                      all_matls);

  scheduleComputeLagrangianValues(sched, patches,   mpm_matls_sub,
                                                    all_matls);

  scheduleAddExchangeToMomentumAndEnergy(sched, patches, all_matls);

  scheduleAdvectAndAdvanceInTime(sched, patches, all_matls);
  
  if(switchTestConservation) {
    schedulePrintConservedQuantities(sched, patches, press_matl,all_matls); 
  }
  
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeEquilibrationPressure--
_____________________________________________________________________*/
void ICE::scheduleComputeEquilibrationPressure(SchedulerP& sched,
					       const PatchSet* patches,
                                          const MaterialSubset* press_matl,
					       const MaterialSet* ice_matls)
{
  cout_doing << "ICE::scheduleComputeEquilibrationPressure" << endl;
  Task* task = scinew Task("ICE::computeEquilibrationPressure",
                     this, &ICE::computeEquilibrationPressure);
  
  task->requires(Task::OldDW,lb->press_CCLabel, press_matl, Ghost::None);
  task->requires(Task::OldDW,lb->rho_CC_top_cycleLabel,     Ghost::None);
  task->requires(Task::OldDW,lb->temp_CCLabel,              Ghost::None);
  task->requires(Task::OldDW,lb->vel_CCLabel,               Ghost::None);
  
  task->computes(lb->speedSound_CCLabel);
  task->computes(lb->vol_frac_CCLabel);
  task->computes(lb->rho_micro_CCLabel);
  task->computes(lb->rho_CCLabel);
  task->computes(lb->press_equil_CCLabel, press_matl);
  sched->addTask(task, patches, ice_matls);
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
  cout_doing << "ICE::scheduleComputeFaceCenteredVelocities" << endl;
  Task* task = scinew Task("ICE::computeFaceCenteredVelocities",
                     this, &ICE::computeFaceCenteredVelocities);

  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW,lb->press_equil_CCLabel, press_matl,
                                                      Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->rho_micro_CCLabel, /*all_matls*/
                                                      Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->rho_CCLabel,       /*all_matls*/
                                                      Ghost::AroundCells,1);
  task->requires(Task::OldDW,lb->vel_CCLabel,         ice_matls, 
                                                      Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->vel_CCLabel,         mpm_matls, 
                                                      Ghost::AroundCells,1);
  task->requires(Task::OldDW, lb->doMechLabel);


  task->computes(lb->uvel_FCLabel);
  task->computes(lb->vvel_FCLabel);
  task->computes(lb->wvel_FCLabel);
  sched->addTask(task, patches, all_matls);
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
  task->requires(Task::NewDW,lb->rho_micro_CCLabel, Ghost::AroundCells,1);
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
  task->requires(Task::OldDW, lb->temp_CCLabel, Ghost::None);
  task->computes(lb->burnedMass_CCLabel);
  task->computes(lb->releasedHeat_CCLabel);
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
  Task* task = scinew Task("ICE::computeDelPressAndUpdatePressCC",
                     this, &ICE::computeDelPressAndUpdatePressCC);
  
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires( Task::NewDW,lb->press_equil_CCLabel,
                                          press_matl, Ghost::None);
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,  Ghost::AroundCells,1);
  task->requires( Task::NewDW, lb->uvel_FCMELabel,    Ghost::AroundCells,2);
  task->requires( Task::NewDW, lb->vvel_FCMELabel,    Ghost::AroundCells,2);
  task->requires( Task::NewDW, lb->wvel_FCMELabel,    Ghost::AroundCells,2);

  task->requires( Task::NewDW, lb->speedSound_CCLabel,Ghost::None);
  task->requires( Task::NewDW, lb->rho_micro_CCLabel, Ghost::None);
  task->requires( Task::NewDW, lb->rho_CCLabel,       Ghost::None);
  task->requires( Task::OldDW, lb->vel_CCLabel,       ice_matls, 
                                                      Ghost::None); 
  task->requires( Task::NewDW, lb->vel_CCLabel,       mpm_matls, 
                                                      Ghost::None);     
  task->requires(Task::NewDW,lb->burnedMass_CCLabel,  Ghost::None);
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
  cout_doing << "ICE::scheduleAccumulateMomentumSourceSinks" << endl; 
  Task* task = scinew Task("ICE::accumulateMomentumSourceSinks", 
                     this, &ICE::accumulateMomentumSourceSinks);
                     
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW,lb->pressX_FCLabel,   press_matl,    
                                                   Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->pressY_FCLabel,   press_matl,
                                                   Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->pressZ_FCLabel,   press_matl,
                                                   Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->rho_CCLabel,      Ghost::None);
  task->requires(Task::NewDW,lb->vol_frac_CCLabel, Ghost::None);
  task->requires(Task::OldDW,lb->vel_CCLabel,      ice_matls_sub,
                                                   Ghost::None);
  task->requires(Task::OldDW, lb->doMechLabel);
  task->computes(lb->doMechLabel);

  task->computes(lb->mom_source_CCLabel);
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAccumulateEnergySourceSinks--
_____________________________________________________________________*/
void ICE::scheduleAccumulateEnergySourceSinks(SchedulerP& sched,
					      const PatchSet* patches,
                                         const MaterialSubset* press_matl,
					      const MaterialSet* matls)

{
  cout_doing << "ICE::scheduleAccumulateEnergySourceSinks" << endl;
  Task* task = scinew Task("ICE::accumulateEnergySourceSinks",
                     this, &ICE::accumulateEnergySourceSinks);
  
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW, lb->press_CCLabel,     press_matl,Ghost::None);
  task->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,Ghost::None);
  //task->requires(Task::NewDW, lb->delP_MassXLabel,   press_matl,Ghost::None);
  task->requires(Task::NewDW, lb->rho_micro_CCLabel,            Ghost::None);
  task->requires(Task::NewDW, lb->speedSound_CCLabel,           Ghost::None);
  task->requires(Task::NewDW, lb->vol_frac_CCLabel,             Ghost::None);

#ifdef ANNULUSICE
  task->requires(Task::NewDW, lb->rho_CCLabel,                  Ghost::None);
#endif
  
  task->computes(lb->int_eng_source_CCLabel);
  
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE:: scheduleComputeLagrangianValues--
 Note:      Only loop over ICE materials, mom_L for MPM is computed
            prior to this function.  
_____________________________________________________________________*/
void ICE::scheduleComputeLagrangianValues(SchedulerP& sched,
					  const PatchSet* patches,
                                     const MaterialSubset* mpm_matls,
                                     const MaterialSet* ice_matls)
{
  cout_doing << "ICE::scheduleComputeLagrangianValues" << endl;
  Task* task = scinew Task("ICE::computeLagrangianValues",
                      this,&ICE::computeLagrangianValues);

  task->requires(Task::NewDW,lb->rho_CCLabel,             Ghost::None);
  task->requires(Task::OldDW,lb->vel_CCLabel,             Ghost::None);
  task->requires(Task::OldDW,lb->temp_CCLabel,            Ghost::None);
  task->requires(Task::NewDW,lb->mom_source_CCLabel,      Ghost::None);
  task->requires(Task::NewDW,lb->burnedMass_CCLabel,      Ghost::None);
  task->requires(Task::NewDW,lb->releasedHeat_CCLabel,    Ghost::None);
  task->requires(Task::NewDW,lb->int_eng_source_CCLabel,  Ghost::None);
  if (switchDebugLagrangianValues ) {
  task->requires(Task::NewDW,lb->mom_L_CCLabel,     mpm_matls,
                                                          Ghost::None);
  task->requires(Task::NewDW,lb->int_eng_L_CCLabel, mpm_matls,
                                                          Ghost::None);
  }

  task->computes(lb->mom_L_CCLabel);
  task->computes(lb->int_eng_L_CCLabel);
  task->computes(lb->mass_L_CCLabel);
 
  sched->addTask(task, patches, ice_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAddExchangeToMomentumAndEnergy--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeToMomentumAndEnergy(SchedulerP& sched,
						 const PatchSet* patches, 
						 const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleAddExchangeToMomentumAndEnergy" << endl;
  Task* task = scinew Task("ICE::addExchangeToMomentumAndEnergy",
                     this, &ICE::addExchangeToMomentumAndEnergy);;
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,   Ghost::None);
  task->requires(Task::NewDW, lb->mom_L_CCLabel,    Ghost::None);
  task->requires(Task::NewDW, lb->int_eng_L_CCLabel,Ghost::None);
  task->requires(Task::NewDW, lb->vol_frac_CCLabel, Ghost::None);
  task->requires(Task::NewDW, lb->rho_micro_CCLabel,Ghost::None);
 
  task->computes(lb->mom_L_ME_CCLabel);
  task->computes(lb->int_eng_L_ME_CCLabel);
  
  sched->addTask(task, patches, matls);
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
  task->requires(Task::NewDW, lb->rho_micro_CCLabel,   Ghost::AroundCells,1);
  task->requires(Task::OldDW, lb->mass_CCLabel,        Ghost::AroundCells,1);
  task->requires(Task::NewDW, lb->created_vol_CCLabel, Ghost::AroundCells,1);
 
  task->computes(lb->rho_CC_top_cycleLabel);
  task->computes(lb->mass_CCLabel);
  task->computes(lb->sp_vol_CCLabel);
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

  task->requires(Task::NewDW,lb->press_CCLabel,      press_matl,Ghost::None);
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
					 DataWarehouse* old_dw, 
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
    Vector dx       = patch->dCell();
    Vector grav     = d_sharedState->getGravity();
    double cell_vol = dx.x()*dx.y()*dx.z();
    StaticArray<CCVariable<double>   > rho_micro(numMatls);
    StaticArray<CCVariable<double>   > sp_vol_CC(numMatls);
    StaticArray<CCVariable<double>   > mass_CC(numMatls);
    StaticArray<CCVariable<double>   > rho_top_cycle(numMatls);
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
      new_dw->allocate(mass_CC[m],     lb->mass_CCLabel,        indx,patch);
      new_dw->allocate(rho_top_cycle[m],lb->rho_CC_top_cycleLabel,
                                                                indx,patch);
      new_dw->allocate(Temp_CC[m],     lb->temp_CCLabel,        indx,patch);
      new_dw->allocate(speedSound[m],  lb->speedSound_CCLabel,  indx,patch);
      new_dw->allocate(vol_frac_CC[m], lb->vol_frac_CCLabel,    indx,patch);
      new_dw->allocate(vel_CC[m],      lb->vel_CCLabel,         indx,patch);
    }
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      ice_matl->initializeCells(rho_micro[m], sp_vol_CC[m],   rho_top_cycle[m],
                                Temp_CC[m],   speedSound[m], 
                                vol_frac_CC[m], vel_CC[m], 
                                press_CC,  numALLMatls,    patch, new_dw);

      cv[m] = ice_matl->getSpecificHeat();
      setBC(rho_top_cycle[m], "Density",      patch, indx);
      setBC(Temp_CC[m],       "Temperature",  patch, indx);
      setBC(vel_CC[m],        "Velocity",     patch, indx); 

      //__________________________________
      //  Adjust pressure and Temp field if g != 0
      //  so fields are thermodynamically consistent.
      if ((grav.x() !=0 || grav.y() != 0.0 || grav.z() != 0.0))  {
        hydrostaticPressureAdjustment(patch,
                                      rho_micro[SURROUND_MAT], press_CC);

        setBC(press_CC, rho_micro[SURROUND_MAT], "Pressure",patch,0);

        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        double gamma = ice_matl->getGamma();
        ice_matl->getEOS()->computeTempCC(patch, "WholeDomain",
                                     press_CC,   gamma,   cv[m],
					  rho_micro[m],    Temp_CC[m]);
      }

      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
	IntVector c = *iter;
        mass_CC[m][c] = rho_top_cycle[m][c] * cell_vol;
      }
    }  
    for (int m = 0; m < numMatls; m++ ) { 
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex(); 
      new_dw->put(rho_micro[m],     lb->rho_micro_CCLabel,      indx,patch);
      new_dw->put(sp_vol_CC[m],     lb->sp_vol_CCLabel,         indx,patch);
      new_dw->put(mass_CC[m],       lb->mass_CCLabel,           indx,patch);
      new_dw->put(rho_top_cycle[m], lb->rho_CC_top_cycleLabel,  indx,patch);
      new_dw->put(vol_frac_CC[m],   lb->vol_frac_CCLabel,       indx,patch);
      new_dw->put(Temp_CC[m],       lb->temp_CCLabel,           indx,patch);
      new_dw->put(speedSound[m],    lb->speedSound_CCLabel,     indx,patch);
      new_dw->put(vel_CC[m],        lb->vel_CCLabel,            indx,patch);

      if (switchDebugInitialize){
        cout_norm << " Initial Conditions" << endl;       
        ostringstream description;
        description << "Initialization_Mat_" << indx << "_patch_"
		    << patch->getID();
        printData(patch, 1, description.str(), "rho_CC",rho_top_cycle[m]);
        printData(patch, 1, description.str(), "rho_micro_CC",rho_micro[m]);
     // printData(patch, 1, description.str(), "sp_vol_CC", sp_vol_CC[m]);
        printData(patch, 1, description.str(), "Temp_CC",   Temp_CC[m]);
        printData(patch, 1, description.str(), "vol_frac_CC",vol_frac_CC[m]);
        printVector(patch, 1, description.str(), "uvel_CC", 0,  vel_CC[m]);
        printVector(patch, 1, description.str(), "vvel_CC", 1,  vel_CC[m]);
        printVector(patch, 1, description.str(), "wvel_CC", 2,  vel_CC[m]);
      }   
    }
    setBC(press_CC, rho_micro[SURROUND_MAT], "Pressure",patch,0);
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

    StaticArray<double> delVol_frac(numMatls),press_eos(numMatls);
    StaticArray<double> dp_drho(numMatls),dp_de(numMatls);
    StaticArray<CCVariable<double> > vol_frac(numMatls);
    StaticArray<CCVariable<double> > rho_micro(numMatls);
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numMatls);
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<CCVariable<double> > speedSound(numMatls);
    StaticArray<CCVariable<double> > speedSound_new(numMatls);
    StaticArray<constCCVariable<Vector> > vel_CC(numMatls);
    CCVariable<int> n_iters_equil_press;
    constCCVariable<double> press;
    CCVariable<double> press_new;
    StaticArray<double> cv(numMatls);

    old_dw->get(press,         lb->press_CCLabel, 0,patch,Ghost::None, 0); 
    new_dw->allocate(press_new,lb->press_equil_CCLabel, 0,patch);

    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],  lb->temp_CCLabel,         indx,patch,
                                                    Ghost::None, 0);
      old_dw->get(rho_CC[m],lb->rho_CC_top_cycleLabel,indx,patch,
		                                      Ghost::None,0);
      old_dw->get(vel_CC[m],lb->vel_CCLabel,indx,patch,
		                                      Ghost::None,0);

      new_dw->allocate(speedSound_new[m],lb->speedSound_CCLabel,indx, patch);
      new_dw->allocate(rho_micro[m],     lb->rho_micro_CCLabel, indx, patch);
      new_dw->allocate(vol_frac[m],      lb->vol_frac_CCLabel,  indx, patch);
      new_dw->allocate(rho_CC_new[m],    lb->rho_CCLabel,       indx, patch);
      cv[m] = matl->getSpecificHeat();
    }

    press_new.copyData(press);
    //__________________________________
    // Compute rho_micro, speedSound, and volfrac
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
      IntVector c = *iter;
      for (int m = 0; m < numMatls; m++) {
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        double gamma = ice_matl->getGamma();
        rho_micro[m][c] = 
	  ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma,cv[m],
					      Temp[m][c]); 

          ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma,
                                          cv[m], Temp[m][c],
                                          press_eos[m], dp_drho[m], dp_de[m]);

        tmp = dp_drho[m] + dp_de[m] * 
	  (press_eos[m]/(rho_micro[m][c]*rho_micro[m][c]));
        speedSound_new[m][c] = sqrt(tmp);
        vol_frac[m][c] = rho_CC[m][c]/rho_micro[m][c];
      }
    }

   //---- P R I N T   D A T A ------  
    if (switchDebug_equilibration_press) {
    
      new_dw->allocate(n_iters_equil_press, lb->scratchLabel, 0, patch);
#if 0
      ostringstream description;
      description << "TOP_equilibration_patch_" << patch->getID();
      printData( patch, 1, description.str(), "Press_CC_top", press);
     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       description << "TOP_equilibration_Mat_" << indx << "_patch_"
		   <<  patch->getID();
       printData(patch, 1, description.str(), "rho_CC",          rho_CC[m]);
       printData(patch, 1, description.str(), "rho_micro_CC",   rho_micro[m]);
       printData(patch, 0, description.str(), "speedSound", speedSound_new[m]);
       printData(patch, 1, description.str(), "Temp_CC",         Temp[m]);
       printData(patch, 1, description.str(), "vol_frac_CC",     vol_frac[m]);
      }
#endif
    }

  //______________________________________________________________________
  // Done with preliminary calcs, now loop over every cell
    int count, test_max_iter = 0;
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {

      IntVector curcell = *iter;    //So I have a chance at finding bugs -Todd

      double delPress = 0.;
      bool converged  = false;
      count           = 0;
      while ( count < d_max_iter_equilibration && converged == false) {
        count++;
        double A = 0.;
        double B = 0.;
        double C = 0.;

        for (int m = 0; m < numMatls; m++) 
          delVol_frac[m] = 0.;
        //__________________________________
       // evaluate press_eos at cell i,j,k
       for (int m = 0; m < numMatls; m++)  {
         ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
         double gamma = ice_matl->getGamma();

         ice_matl->getEOS()->computePressEOS(rho_micro[m][curcell],gamma,
                                           cv[m], Temp[m][curcell],
                                           press_eos[m], dp_drho[m], dp_de[m]);
       }
       //__________________________________
       // - compute delPress
       // - update press_CC     
       StaticArray<double> Q(numMatls),y(numMatls);     
       for (int m = 0; m < numMatls; m++)   {
         Q[m] =  press_new[curcell] - press_eos[m];
         y[m] =  dp_drho[m] * ( rho_CC[m][curcell]/
                 (vol_frac[m][curcell] * vol_frac[m][curcell]) ); 
         A   +=  vol_frac[m][curcell];
         B   +=  Q[m]/y[m];
         C   +=  1.0/y[m];
       }
       double vol_frac_not_close_packed = 1.;
       delPress = (A - vol_frac_not_close_packed - B)/C;

       press_new[curcell] += delPress;

       //__________________________________
       // backout rho_micro_CC at this new pressure
       for (int m = 0; m < numMatls; m++) {
         ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
         double gamma = ice_matl->getGamma();

         rho_micro[m][curcell] = 
           ice_matl->getEOS()->computeRhoMicro(press_new[curcell],gamma,
                                               cv[m],Temp[m][curcell]);
       }
       //__________________________________
       // - compute the updated volume fractions
       for (int m = 0; m < numMatls; m++)  {
         delVol_frac[m]       = -(Q[m] + delPress)/y[m];
         vol_frac[m][curcell]   = rho_CC[m][curcell]/rho_micro[m][curcell];
       }
       //__________________________________
       // Find the speed of sound 
       // needed by eos and the explicit
       // del pressure function
       for (int m = 0; m < numMatls; m++)  {
          ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
          double gamma = ice_matl->getGamma();
          ice_matl->getEOS()->computePressEOS(rho_micro[m][curcell],gamma,
                                            cv[m],Temp[m][curcell],
                                            press_eos[m],dp_drho[m], dp_de[m]);

          tmp = dp_drho[m] + dp_de[m] * 
                      (press_eos[m]/(rho_micro[m][curcell]*rho_micro[m][curcell]));
          speedSound_new[m][curcell] = sqrt(tmp);
       }
       //__________________________________
       // - Test for convergence 
       //  If sum of vol_frac_CC ~= 1.0 then converged 
       sum = 0.0;
       for (int m = 0; m < numMatls; m++)  {
         sum += vol_frac[m][curcell];
       }
       if (fabs(sum-1.0) < convergence_crit)
         converged = true;

      }   // end of converged

      test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      if(test_max_iter == d_max_iter_equilibration)  {
	ostringstream warning;
	IntVector c = *iter;
	warning << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
		<< "], iter " << count << ", n_passes " << n_passes 
		<< ", Now exiting ";
	string warn = "calc_equilibartion_press: Maximum number of iterations was reached " + warning.str();
	throw InvalidValue(warn);
      }

       for (int m = 0; m < numMatls; m++) {
           ASSERT(( vol_frac[m][c] > 0.0 ) ||
                  ( vol_frac[m][c] < 1.0));
       }
      if ( fabs(sum - 1.0) > convergence_crit)   {
	ostringstream warning;
	IntVector c = *iter;
	warning << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
		<< "], iter " << count << ", n_passes " << n_passes 
		<< ", Now exiting ";
	string warn = "calc_equilibration_press: sum(vol_frac_CC) != 1.0" + 
	  warning.str();
	throw InvalidValue(warn);
      }
      IntVector c = *iter;
      if ( press_new[c] < 0.0 )   {
	ostringstream warning;
	IntVector c = *iter;
	warning << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
		<< "], iter " << count << ", n_passes " << n_passes 
		<< ", Now exiting ";
	string warn = "calc_equilibration_press: press_new[iter*] < 0" 
	  + warning.str();
	throw InvalidValue(warn);
      }

      for (int m = 0; m < numMatls; m++)
      if ( rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) {
	ostringstream warning;
	IntVector c = *iter;
	warning << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
		<< "], iter " << count << ", n_passes " << n_passes 
		<< ", Now exiting ";
	string warn = " calc_equilibartion_press: rho_micro < 0 || vol_frac < 0" + warning.str();
	throw InvalidValue(warn);
      }
      if (switchDebug_equilibration_press) {
        n_iters_equil_press[c] = count;
      }
    }     // end of cell interator

    cerr << " max. iterations in any cell" << test_max_iter << "\t"; 
    //__________________________________
    // update Boundary conditions
    setBC(press_new, rho_micro[SURROUND_MAT], "Pressure",patch,0);    
    //__________________________________
    // carry rho_cc forward 
    // In MPMICE was compute rho_CC_new and 
    // therefore need the machinery here
    for (int m = 0; m < numMatls; m++)   {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      rho_CC_new[m].copyData(rho_CC[m]);
      new_dw->put( vol_frac[m],      lb->vol_frac_CCLabel,   indx, patch);
      new_dw->put( speedSound_new[m],lb->speedSound_CCLabel, indx, patch);
      new_dw->put( rho_micro[m],     lb->rho_micro_CCLabel,  indx, patch);
      new_dw->put( rho_CC_new[m],    lb->rho_CCLabel,        indx, patch);
    }
    new_dw->put(press_new,lb->press_equil_CCLabel,0,patch);

   //---- P R I N T   D A T A ------   
    if (switchDebug_equilibration_press) {
      ostringstream description;
      description << "BOT_equilibration_patch_" << patch->getID();
      printData( patch, 1, description.str(), "Press_CC_equil", press_new);

     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream description;
       description << "BOT_equilibration_Mat_" << indx << "_patch_" 
		   << patch->getID();
       printData( patch, 1, description.str(), "rho_CC",  rho_CC[m]);
       //printData( patch, 1, description.str(), "speedSound",speedSound_new[m]);
       printData( patch, 1, description.str(), "rho_micro_CC", rho_micro[m]);
       printData( patch, 1, description.str(), "vol_frac_CC",  vol_frac[m]);
       //printData( patch, 1, description.str(), "iterations",
       //  n_iters_equil_press);
       
     }
    }
  }  // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::computeFaceCenteredVelocities--
 Purpose~   compute the face centered velocities minus the exchange
            contribution.
_____________________________________________________________________*/
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
      constCCVariable<double> rho_CC, rho_micro_CC;
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
      new_dw->get(rho_micro_CC, lb->rho_micro_CCLabel,indx,patch,
		  Ghost::AroundCells, 1);
      
      
      //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
#if 0
        ostringstream description;
        description << "TOP_vel_FC_Mat_" << indx << "_patch_" 
	            << patch->getID(); 
        printData(  patch, 1, description.str(), "rho_CC",      rho_CC);
        printData(  patch, 1, description.str(), "rho_micro_CC",rho_micro_CC);
        printVector( patch,1, description.str(), "uvel_CC", 0, vel_CC);
        printVector( patch,1, description.str(), "vvel_CC", 1, vel_CC);
        printVector( patch,1, description.str(), "wvel_CC", 2, vel_CC);
#endif
      }
      
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
      
      double term1, term2, term3, press_coeff, rho_micro_FC, rho_FC;
      
      if(doMechOld < -1.5) {
	//__________________________________
	//   B O T T O M   F A C E S 
	int offset=1; // 0=Compute all faces in computational domain
	              // 1=Skip the faces at the border between interior and gc
	for(CellIterator iter=patch->getSFCYIterator(offset);!iter.done();
	    iter++){
	  IntVector curcell = *iter;
	  IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z()); 
	  
	  rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	  rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	  ASSERT(rho_FC > 0.0);
	  //__________________________________
	  // interpolation to the face
	  term1 = (rho_CC[adjcell] * vel_CC[adjcell].y() +
		   rho_CC[curcell] * vel_CC[curcell].y())/(rho_FC);            
	  //__________________________________
	  // pressure term
	  press_coeff = 2.0/(rho_micro_FC);
	  term2 =   delT * press_coeff *
	    (press_CC[curcell] - press_CC[adjcell])/dx.y();                
	  //__________________________________
	  // gravity term
	  term3 =  delT * gravity.y();
	  vvel_FC[curcell] = term1- term2 + term3;
	}

      //__________________________________
      //  L E F T   F A C E 
	for(CellIterator iter=patch->getSFCXIterator(offset);!iter.done();
	    iter++){
	  IntVector curcell = *iter;
	  IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z()); 
	  
	  rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	  rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	  ASSERT(rho_FC > 0.0);
	  //__________________________________
	  // interpolation to the face
	  term1 = (rho_CC[adjcell] * vel_CC[adjcell].x() +
		   rho_CC[curcell] * vel_CC[curcell].x())/(rho_FC);
	  //__________________________________
	  // pressure term
	  press_coeff = 2.0/(rho_micro_FC);
	  
	  term2 =   delT * press_coeff *
	    (press_CC[curcell] - press_CC[adjcell])/dx.x();
	  //__________________________________
	  // gravity term
	  term3 =  delT * gravity.x();
	  uvel_FC[curcell] = term1- term2 + term3;
	}
	
	//__________________________________
	//  B A C K    F A C E
	for(CellIterator iter=patch->getSFCZIterator(offset);!iter.done();
	    iter++){
	  IntVector curcell = *iter;
	  IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1); 
	  
	  rho_micro_FC = rho_micro_CC[adjcell] + rho_micro_CC[curcell];
	  rho_FC       = rho_CC[adjcell]       + rho_CC[curcell];
	  ASSERT(rho_FC > 0.0);
	  //__________________________________
	  // interpolation to the face
	  term1 = (rho_CC[adjcell] * vel_CC[adjcell].z() +
		   rho_CC[curcell] * vel_CC[curcell].z())/(rho_FC);
	  //__________________________________
	  // pressure term
	  press_coeff = 2.0/(rho_micro_FC);
	  
	  term2 =   delT * press_coeff *
	    (press_CC[curcell] - press_CC[adjcell])/dx.z();
	  //__________________________________
	  // gravity term
	  term3 =  delT * gravity.z();
	  wvel_FC[curcell] = term1- term2 + term3;
	}
      }  // if doMech

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()

      new_dw->put(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->put(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->put(wvel_FC, lb->wvel_FCLabel, indx, patch);

   //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream description;
        description <<  "bottom_of_vel_FC_Mat_" << indx << "_patch_" 
		    << patch->getID();
        printData_FC( patch,1, description.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, description.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, description.str(), "wvel_FC", wvel_FC);
      }
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
    double tmp;

    StaticArray<constCCVariable<double> > rho_micro_CC(numMatls);
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
    DenseMatrix beta(numMatls,numMatls),a(numMatls,numMatls);

    DenseMatrix K(numMatls,numMatls), junk(numMatls,numMatls);

    beta.zero();
    a.zero();
    K.zero();
    getExchangeCoefficients( K, junk);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(rho_micro_CC[m], lb->rho_micro_CCLabel, indx, patch, 
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
    int offset=1;   // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc
    for(CellIterator iter=patch->getSFCYIterator(offset);!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z()); 
      for(int m = 0; m < numMatls; m++) {
	for(int n = 0; n < numMatls; n++) {
	  tmp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];

	  beta[m][n] = delT * tmp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);

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
	  b[m] += beta[m][n] * (vvel_FC[n][curcell] - vvel_FC[m][curcell]);
	}
      }
      //__________________________________
      //      S  O  L  V  E  
      //   - backout velocities           
      vector<double> X(numMatls);
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
	vvel_FCME[m][curcell] = vvel_FC[m][curcell] + X[m];
      }
    }

    //__________________________________
    //   L E F T  F A C E-- B  E  T  A      
    //  Note this includes b[m][m]
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    //   - form off diagonal terms of (a)
    for(CellIterator iter=patch->getSFCXIterator(offset);!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z()); 

      for(int m = 0; m < numMatls; m++)  {
	for(int n = 0; n < numMatls; n++)  {
	  tmp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];
	  beta[m][n] = delT * tmp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);

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
	  b[m] += beta[m][n] * (uvel_FC[n][curcell] - uvel_FC[m][curcell]);
	}
      }
      //__________________________________
      //      S  O  L  V  E
      //   - backout velocities
      vector<double> X(numMatls);
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
	uvel_FCME[m][curcell] = uvel_FC[m][curcell] + X[m];
      }   
    }
    //__________________________________
    //  B A C K  F A C E -- B  E  T  A      
    //  Note this includes b[m][m]
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    //   - form off diagonal terms of (a)
    for(CellIterator iter=patch->getSFCZIterator(offset);!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1); 
      for(int m = 0; m < numMatls; m++)  {
	for(int n = 0; n < numMatls; n++) {
	  tmp = (vol_frac_CC[n][adjcell] + vol_frac_CC[n][curcell]) * K[n][m];
	  beta[m][n] = delT * tmp/
	    (rho_micro_CC[m][curcell] + rho_micro_CC[m][adjcell]);

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
	  b[m] += beta[m][n] * (wvel_FC[n][curcell] - wvel_FC[m][curcell]);
	}
      }
      //__________________________________
      //      S  O  L  V  E
      //   - backout velocities 
      vector<double> X(numMatls);
      matrixSolver(numMatls,a,b,X);
      for(int m = 0; m < numMatls; m++) {
	wvel_FCME[m][curcell] = wvel_FC[m][curcell] + X[m];
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
	ostringstream description;
	description << "Exchange_FC_after_BC_Mat_" << indx  << "_patch_" 
		    <<  patch->getID();
	printData_FC( patch,1, description.str(), "uvel_FCME", uvel_FCME[m]);
	printData_FC( patch,1, description.str(), "vvel_FCME", vvel_FCME[m]);
	printData_FC( patch,1, description.str(), "wvel_FCME", wvel_FCME[m]);
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

    CCVariable<double> q_CC,      q_advected;

    constCCVariable<double> pressure;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> press_CC;
    constCCVariable<double> burnedMass;
   
    const IntVector gc(1,1,1);

    new_dw->get(pressure,       lb->press_equil_CCLabel,0,patch,Ghost::None,0);
    new_dw->allocate(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocate(delP_MassX,lb->delP_MassXLabel,    0, patch);
    new_dw->allocate(press_CC,  lb->press_CCLabel,      0, patch);
    new_dw->allocate(q_CC,      lb->q_CCLabel,          0, patch,
		     Ghost::AroundCells,1);
    new_dw->allocate(q_advected,lb->q_advectedLabel,    0, patch);

    StaticArray<constCCVariable<double> > rho_micro(numMatls);

    CCVariable<double> term1, term2, term3;
    new_dw->allocate(term1, lb->term1Label, 0, patch);
    new_dw->allocate(term2, lb->term2Label, 0, patch);
    new_dw->allocate(term3, lb->term3Label, 0, patch);

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
      constCCVariable<double>   speedSound;
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> rho_CC;

      new_dw->get(uvel_FC, lb->uvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,   indx,  patch,
		  Ghost::AroundCells, 2);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,  patch,
		  Ghost::AroundCells,1);
      new_dw->get(rho_CC,      lb->rho_CCLabel,      indx,patch,Ghost::None,0);
      new_dw->get(rho_micro[m],lb->rho_micro_CCLabel,indx,patch,Ghost::None,0);
      new_dw->get(speedSound, lb->speedSound_CCLabel,indx,patch,Ghost::None,0);
      new_dw->get(burnedMass, lb->burnedMass_CCLabel,indx,patch,Ghost::None,0);
      if(ice_matl) {
        old_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }
      if(mpm_matl) {
        new_dw->get(vel_CC,   lb->vel_CCLabel,       indx,patch,Ghost::None,0);
      }

      //__________________________________
      // Advection preprocessing
      // - divide vol_frac_cc/vol

      Advector* advector = d_advector->clone(new_dw,patch);
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch);

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();
	  iter++) {
	IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      //__________________________________
      //   First order advection of q_CC
      advector->advectQ(q_CC,patch,q_advected);
      delete advector;

      //---- P R I N T   D A T A ------  
      if (switchDebug_explicit_press ) {
        ostringstream description;
	description << "middle_of_explicit_Pressure_Mat_" << indx << "_patch_"
		    <<  patch->getID();
        printData_FC( patch,1, description.str(), "uvel_FC", uvel_FC);
        printData_FC( patch,1, description.str(), "vvel_FC", vvel_FC);
        printData_FC( patch,1, description.str(), "wvel_FC", wvel_FC);
      }
      
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
	IntVector c = *iter;
        //__________________________________
        //   Contributions from reactions
        //   to be filled in Be very careful with units
        term1[c] += (burnedMass[c]/delT)/(rho_micro[m][c] * vol);

        //__________________________________
        //   Divergence of velocity * face area
        //   Be very careful with the units
        //   do the volume integral to check them
        //   See journal pg 171
        //   You need to divide by the cell volume
        //
        //  Note that sum(div (theta_k U^f_k) 
        //          =
        //  Advection(theta_k, U^f_k)
        //  This subtle point is discussed on pg
        //  190 of my Journal
        term2[c] -= q_advected[c];

        term3[c] += vol_frac[c] /(rho_micro[m][c] *
				speedSound[c]*speedSound[c]);
      }  //iter loop
    }  //matl loop
    press_CC.copyData(pressure);
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    = (delT * term1[c])/term3[c];
      delP_Dilatate[c] = -term2[c]/term3[c];
      press_CC[c]      = pressure[c] + 
                             delP_MassX[c] + delP_Dilatate[c];    
    }
    setBC(press_CC, rho_micro[SURROUND_MAT], "Pressure",patch,0);

    new_dw->put(delP_Dilatate, lb->delP_DilatateLabel, 0, patch);
    new_dw->put(delP_MassX,    lb->delP_MassXLabel,    0, patch);
    new_dw->put(press_CC,      lb->press_CCLabel,      0, patch);

   //---- P R I N T   D A T A ------  
    if (switchDebug_explicit_press) {
      ostringstream description;
      description << "Bottom_of_explicit_Pressure_patch_" << patch->getID();
      printData( patch, 1,description.str(), "delP_Dilatate", delP_Dilatate);
      //printData( patch, 1,description.str(), "delP_MassX",    delP_MassX);
      printData( patch, 1,description.str(), "Press_CC",      press_CC);
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
    double sum_rho, sum_rho_adj;
    double A;                                 

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

    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(rho_CC[m],lb->rho_CCLabel,indx,patch,Ghost::AroundCells,1);
    }

    //__________________________________
    //  B O T T O M   F A C E
    for(CellIterator iter=patch->getSFCYIterator();!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
      sum_rho     = 0.0;
      sum_rho_adj = 0.0;
      for(int m = 0; m < numMatls; m++) {
	sum_rho      += rho_CC[m][curcell];
	sum_rho_adj  += rho_CC[m][adjcell];
      }

      A =  (press_CC[curcell]/sum_rho) + (press_CC[adjcell]/sum_rho_adj);
      pressY_FC[curcell] = A/((1/sum_rho)+(1.0/sum_rho_adj));
    }
    //__________________________________
    //  L E F T   F A C E
    for(CellIterator iter=patch->getSFCXIterator();!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
      sum_rho     = 0.0;
      sum_rho_adj = 0.0;

      for(int m = 0; m < numMatls; m++) {
	sum_rho      += rho_CC[m][curcell];
	sum_rho_adj  += rho_CC[m][adjcell];
      }

      A =  (press_CC[curcell]/sum_rho) + (press_CC[adjcell]/sum_rho_adj);
      pressX_FC[curcell] = A/((1/sum_rho)+(1.0/sum_rho_adj));
    }
    //__________________________________
    //     B A C K   F A C E 
    for(CellIterator iter=patch->getSFCZIterator();!iter.done();iter++){
      IntVector curcell = *iter;
      IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
      sum_rho     = 0.0;
      sum_rho_adj = 0.0;
      
      for(int m = 0; m < numMatls; m++) {
	sum_rho      += rho_CC[m][curcell];
	sum_rho_adj  += rho_CC[m][adjcell];
      }

      A =  (press_CC[curcell]/sum_rho) + (press_CC[adjcell]/sum_rho_adj);
      pressZ_FC[curcell]=A/((1/sum_rho)+(1.0/sum_rho_adj));
    }

    new_dw->put(pressX_FC,lb->pressX_FCLabel, 0, patch);
    new_dw->put(pressY_FC,lb->pressY_FCLabel, 0, patch);
    new_dw->put(pressZ_FC,lb->pressZ_FCLabel, 0, patch);

   //---- P R I N T   D A T A ------ 
    if (switchDebug_PressFC) {
      ostringstream description;
      description << "press_FC_patch_" <<patch->getID();
      printData_FC( patch,0,description.str(), "press_FC_RIGHT", pressX_FC);
      printData_FC( patch,0,description.str(), "press_FC_TOP",   pressY_FC);
      printData_FC( patch,0,description.str(), "press_FC_FRONT", pressZ_FC);
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
   StaticArray<CCVariable<double> > burnedMass(numMatls);
   StaticArray<CCVariable<double> > releasedHeat(numMatls);
   StaticArray<constCCVariable<double> > rho_CC(numMatls);
   StaticArray<constCCVariable<double> > Temp_CC(numMatls);
   StaticArray<CCVariable<double> > created_vol(numMatls);
   StaticArray<double> cv(numMatls);
   
   int reactant_indx = -1;

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

      // Look for the reactant material
      if (matl->getRxProduct() == Material::reactant)
	reactant_indx = matl->getDWIndex();

      int indx = matl->getDWIndex();
      new_dw->get(rho_CC[m], lb->rho_CCLabel, indx,patch,Ghost::None, 0);
      old_dw->get(Temp_CC[m],     lb->temp_CCLabel,indx,patch,Ghost::None, 0);
      new_dw->allocate(burnedMass[m],  lb->burnedMass_CCLabel,  indx,patch);
      new_dw->allocate(releasedHeat[m],lb->releasedHeat_CCLabel,indx,patch);
      new_dw->allocate(created_vol[m],   lb->created_vol_CCLabel, indx,patch);
      burnedMass[m].initialize(0.0);
      releasedHeat[m].initialize(0.0); 
      created_vol[m].initialize(0.0);

      cv[m] = ice_matl->getSpecificHeat();
    }
    //__________________________________
    // Do the exchange if there is a reactant (reactant_indx >= 0)
    // and the switch is on.
    if(d_massExchange && (reactant_indx >= 0)){       
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector c = *iter;
        double mass_hmx = rho_CC[reactant_indx][c] * vol;
        if (mass_hmx > d_SMALL_NUM)  {
           double burnedMass_tmp = (rho_CC[reactant_indx][c] * vol);  
           // hardwired wipes out all the mass in one 
          // timestep
           burnedMass[reactant_indx][c] =  -burnedMass_tmp;
	   releasedHeat[reactant_indx][c] = -burnedMass_tmp
					      *  cv[reactant_indx]
					      *  Temp_CC[reactant_indx][c];
           // Commented out for now as I'm not sure that this is appropriate
	   // for regular ICE - Jim 7/30/01
//	   created_vol[reactant_indx][c]  =
//			     -burnedMass_tmp/rho_micro_CC[reactant_indx][c];
        }
      }
      //__________________________________
      // Find the ICE matl which is the products of reaction
      // dump all the mass into that matl.
      for(int prods = 0; prods < numICEMatls; prods++) {
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(prods);
	if (ice_matl->getRxProduct() == Material::product) {
	  for(int m = 0; m < numICEMatls; m++) {
	    for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
	      IntVector c = *iter;
	      burnedMass[prods][c]  -= burnedMass[m][c];
              releasedHeat[prods][c] -=
	                       burnedMass[m][c]*cv[m]*Temp_CC[m][c];
           // Commented out for now as I'm not sure that this is appropriate
	   // for regular ICE - Jim 7/30/01
//	      created_vol[prods][c]  -= created_vol[m][c];
	    }
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
      new_dw->put(releasedHeat[m], lb->releasedHeat_CCLabel, indx,patch);
      new_dw->put(created_vol[m],  lb->created_vol_CCLabel,  indx,patch);
    }
    //---- P R I N T   D A T A ------ 
    for(int m = 0; m < numMatls; m++) {
#if 0    // turn off for quality control tests
      if (switchDebugSource_Sink) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream description;
	description <<  "sources/sinks_Mat_" << indx << "_patch_" 
		    <<  patch->getID();
        printData(patch, 0, description.str(),"burnedMass", burnedMass[m]);
        printData(patch, 0, description.str(),"releasedHeat", releasedHeat[m]);
      }
#endif
    }
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
        old_dw->get(vel_CC, lb->vel_CCLabel,  indx,patch,Ghost::None, 0);
        viscosity = ice_matl->getViscosity();
        if(viscosity != 0.0){  
          computeTauX_Components( patch, vel_CC, viscosity, dx, tau_X_FC);
          computeTauY_Components( patch, vel_CC, viscosity, dx, tau_Y_FC);
          computeTauZ_Components( patch, vel_CC, viscosity, dx, tau_Z_FC);
        }
      }
      //__________________________________
      //  accumulate sources
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
	IntVector c = *iter;
        mass = rho_CC[c] * vol;
        right    = c + IntVector(1,0,0);
        left     = c + IntVector(0,0,0);
        top      = c + IntVector(0,1,0);
        bottom   = c + IntVector(0,0,0);
        front    = c + IntVector(0,0,1);
        back     = c + IntVector(0,0,0);

        //__________________________________
        //    X - M O M E N T U M 
        pressure_source = (pressX_FC[right]-pressX_FC[left]) * vol_frac[c];

        viscous_source = (tau_X_FC[right].x() - tau_X_FC[left].x())  *delY*delZ +
                         (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())*delX*delZ +
                         (tau_Z_FC[front].x() - tau_Z_FC[back].x())  *delX*delY;
                                 
        mom_source[c].x( (-pressure_source * delY * delZ +
                               viscous_source +
			          mass * gravity.x()) * delT );
        //__________________________________
        //    Y - M O M E N T U M
         pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c];

        viscous_source = (tau_X_FC[right].y() - tau_X_FC[left].y())  *delY*delZ +
                         (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())*delX*delZ +
                         (tau_Z_FC[front].y() - tau_Z_FC[back].y())  *delX*delY;

        mom_source[c].y( (-pressure_source * delX * delZ +
                               viscous_source +
			          mass * gravity.y()) * delT );
        //__________________________________
        //    Z - M O M E N T U M
        pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c];
        
        viscous_source = (tau_X_FC[right].z() - tau_X_FC[left].z())  *delY*delZ +
                         (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())*delX*delZ +
                         (tau_Z_FC[front].z() - tau_Z_FC[back].z())  *delX*delY;

        mom_source[c].z( (-pressure_source * delX * delY +
			          viscous_source + 
                               mass * gravity.z()) * delT );
                               
      }
      } // if doMechOld
      new_dw->put(mom_source, lb->mom_source_CCLabel, indx, patch);
      new_dw->put(doMechOld,  lb->doMechLabel);

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream description;
	description << "sources/sinks_Mat_" << indx << "_patch_" 
		    <<  patch->getID();
        printVector(patch, 1, description.str(), "xmom_source", 0, mom_source);
        printVector(patch, 1, description.str(), "ymom_source", 1, mom_source);
        printVector(patch, 1, description.str(), "zmom_source", 2, mom_source);
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

    constCCVariable<double> rho_micro_CC;
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
      new_dw->get(rho_micro_CC,lb->rho_micro_CCLabel, indx,patch,
		  Ghost::None, 0);                                            
      new_dw->get(speedSound,  lb->speedSound_CCLabel,indx,patch,
		  Ghost::None, 0); 
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,  indx,patch,
		  Ghost::None, 0);

#ifdef ANNULUSICE
      CCVariable<double> rho_CC;
      new_dw->get(rho_CC,      lb->rho_CCLabel,       indx,patch,
		  Ghost::None, 0);                                            
#endif
      new_dw->allocate(int_eng_source, lb->int_eng_source_CCLabel,indx,patch);

      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      int_eng_source.initialize(0.);
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector c = *iter;
        A = vol * vol_frac[c] * press_CC[c];
        B = rho_micro_CC[c]   * speedSound[c] * speedSound[c];
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
        ostringstream description;
        description <<  "sources/sinks_Mat_" << indx << "_patch_" 
		    <<  patch->getID();
        printData(patch,1,description.str(),"int_eng_source", int_eng_source);
      }

      new_dw->put(int_eng_source,lb->int_eng_source_CCLabel,indx,patch);
    }  // matl loop
  }  // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::computeLagrangianValues--
 Computes lagrangian mass momentum and energy
 Note:      Only loop over ICE materials, mom_L for MPM is computed
            prior to this function
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
     MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
     CCVariable<Vector> mom_L; 
     CCVariable<double> int_eng_L; 
     CCVariable<double> mass_L;
     if(ice_matl)  {               //  I C E
      constCCVariable<double> rho_CC, temp_CC;
      constCCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_source, burnedMass;
      constCCVariable<double> releasedHeat;
      constCCVariable<Vector> mom_source;

      new_dw->get(rho_CC,  lb->rho_CCLabel,     indx,patch,Ghost::None, 0);
      old_dw->get(vel_CC,  lb->vel_CCLabel,     indx,patch,Ghost::None, 0);
      old_dw->get(temp_CC, lb->temp_CCLabel,    indx,patch,Ghost::None, 0);
      new_dw->get(burnedMass,     lb->burnedMass_CCLabel,indx,patch,
		                                             Ghost::None, 0);
      new_dw->get(releasedHeat,   lb->releasedHeat_CCLabel,indx,patch,
		                                             Ghost::None, 0);
      new_dw->get(mom_source,     lb->mom_source_CCLabel,indx,patch,
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
          Vector min_mom_L = vel_CC[c] * min_mass;
	// Todd:  I believe the second term here to be flawed as was
	// int_eng_tmp.  We should create a "burnedMomentum" or something
	// to do this right.  Someday.
          Vector mom_L_tmp = vel_CC[c] * mass;
//                         + vel_CC[c] * burnedMass[c];
                               
           // find the max between mom_L_tmp and min_mom_L
           // but keep the sign of the mom_L_tmp     
           // You need the d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
          double plus_minus_one_x = (mom_L_tmp.x()+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp.x()+d_SMALL_NUM));
          double plus_minus_one_y = (mom_L_tmp.y()+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp.y()+d_SMALL_NUM));
          double plus_minus_one_z = (mom_L_tmp.z()+d_SMALL_NUM)/
                                    (fabs(mom_L_tmp.z()+d_SMALL_NUM));
          
          double mom_L_x = std::max( fabs(mom_L_tmp.x()), min_mom_L.x() );
          double mom_L_y = std::max( fabs(mom_L_tmp.y()), min_mom_L.y() );
          double mom_L_z = std::max( fabs(mom_L_tmp.z()), min_mom_L.z() );
          
          mom_L_x = plus_minus_one_x * mom_L_x;
          mom_L_y = plus_minus_one_y * mom_L_y;
          mom_L_z = plus_minus_one_z * mom_L_z;
 
          mom_L[c] = Vector(mom_L_x,mom_L_y,mom_L_z) + mom_source[c];

          // must have a minimum int_eng   
          double min_int_eng = min_mass * cv * temp_CC[c];
          double int_eng_tmp = mass * cv * temp_CC[c];

          //  Glossary:
          //  int_eng_tmp    = the amount of internal energy for this
          //                   matl in this cell coming into this task
          //  int_eng_source = thermodynamic work = f(delP_Dilatation)
          //  releasedHeat   = enthalpy of reaction gained by the
          //                   product gas, PLUS (OR, MINUS) the
          //                   internal energy of the reactant
          //                   material that was liberated in the
          //                   reaction
          // min_int_eng     = a small amount of internal energy to keep
          //                   the equilibration pressure from going nuts


          int_eng_L[c] = int_eng_tmp +
                             int_eng_source[c] +
                             releasedHeat[c];

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
      }  // if (ice_matl)
      //---- P R I N T   D A T A ------ 
      // Dump out all the matls data
      if (switchDebugLagrangianValues ) {
	 constCCVariable<double> int_eng_L_Debug = int_eng_L; 
	 constCCVariable<Vector> mom_L_Debug = mom_L;
         if(mpm_matl) {
          new_dw->get(int_eng_L_Debug,lb->int_eng_L_CCLabel,indx,patch,Ghost::None,0);
          new_dw->get(mom_L_Debug,lb->mom_L_CCLabel,indx,patch,Ghost::None,0);
        }
        ostringstream description;
	description <<  "Bot_Lagrangian_Values_Mat_" << indx << "_patch_" 
		    << patch->getID();
        printVector(patch,1, description.str(), "xmom_L_CC", 0, mom_L_Debug);
        printVector(patch,1, description.str(), "ymom_L_CC", 1, mom_L_Debug);
        printVector(patch,1, description.str(), "zmom_L_CC", 2, mom_L_Debug);
        printData(patch,1, description.str(), "int_eng_L_CC",int_eng_L_Debug); 
           
      }
      
      if(ice_matl)  {
        new_dw->put(mom_L,     lb->mom_L_CCLabel,     indx,patch);
        new_dw->put(int_eng_L, lb->int_eng_L_CCLabel, indx,patch);
        new_dw->put(mass_L,    lb->mass_L_CCLabel,    indx,patch);
      }
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

    StaticArray<constCCVariable<double> > mass_L(numMatls);
    StaticArray<CCVariable<double> > Temp_CC(numMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numMatls);
    StaticArray<constCCVariable<double> > vol_frac_CC(numMatls);
    StaticArray<constCCVariable<double> > rho_micro_CC(numMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numMatls);
    StaticArray<CCVariable<Vector> > mom_L_ME(numMatls);

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
      new_dw->get(rho_micro_CC[m],lb->rho_micro_CCLabel,indx,patch,
		  Ghost::None,0);           

      new_dw->allocate( mom_L_ME[m],   lb->mom_L_ME_CCLabel,    indx, patch);
      new_dw->allocate(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,indx, patch);
      new_dw->allocate( vel_CC[m],     lb->vel_CCLabel,         indx, patch);
      new_dw->allocate( Temp_CC[m],    lb->temp_CCLabel,        indx, patch);
      mom_L_ME[m].initialize(Vector(0.0, 0.0, 0.0));
      int_eng_L_ME[m].initialize(0.0);
      vel_CC[m].initialize(Vector(0.0, 0.0, 0.0));
      Temp_CC[m].initialize(0.0);
      cv[m] = matl->getSpecificHeat();
    }  

    //__________________________________
    // Convert vars. flux -> primitive 
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      for (int m = 0; m < numMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m]);
        vel_CC[m][c]  =  mom_L[m][c]/mass_L[m][c];
      }  
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      //   Form BETA matrix (a), off diagonal terms
      //  The beta and (a) matrix is common to all momentum exchanges
      for(int m = 0; m < numMatls; m++)  {
        tmp    = rho_micro_CC[m][c];
        for(int n = 0; n < numMatls; n++) {
	  beta[m][n] = delT * vol_frac_CC[n][c] * K[n][m]/tmp;
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
  
      vector<double> X(numMatls);
      multiplyMatrixAndVector(numMatls,a_inverse,b,X);
      for(int m = 0; m < numMatls; m++) {
	  vel_CC[m][c].x( vel_CC[m][c].x() + X[m] );
      }

      //---------- Y - M O M E N T U M
      // -  F O R M   R H S   (b)
      // -  push a copy of (a) into the solver
      // -  Add exchange contribution to orig value
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
      // -  F O R M   R H S   (b)
      // -  push a copy of (a) into the solver
      // -  Adde exchange contribution to orig value
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
        tmp = cv[m]*rho_micro_CC[m][c];
        for(int n = 0; n < numMatls; n++)  {
	  beta[m][n] = delT * vol_frac_CC[n][c] * H[n][m]/tmp;
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
      setBC(vel_CC[m],"Velocity",patch,indx);
      setBC(Temp_CC[m],"Temperature",patch,indx);
    }
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      for (int m = 0; m < numMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c] * mass_L[m][c];
      }  
    } 

    for(int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->put(mom_L_ME[m],    lb->mom_L_ME_CCLabel,    indx, patch);
      new_dw->put(int_eng_L_ME[m],lb->int_eng_L_ME_CCLabel,indx, patch);
    }
  }  // patch loop
}

/* --------------------------------------------------------------------- 
 Function~  ICE::advectAndAdvanceInTime--
 Purpose~
   This function calculates the The cell-centered, time n+1, mass, momentum
   and internal energy

   Need to include kinetic energy 
   Note, ICE doesn't need either sp_vol_CC or mass_CC, but MPMICE does.
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

    int numALLmatls = d_sharedState->getNumMatls();
    int numICEmatls = d_sharedState->getNumICEMatls();

    // These arrays get re-used for each material, and for each
    // advected quantity
    CCVariable<double> q_CC, q_advected;
    const IntVector gc(1,1,1);
    CCVariable<Vector> qV_CC, qV_advected;

    new_dw->allocate(q_CC,       lb->q_CCLabel,       0, patch,
		     Ghost::AroundCells,1);
    new_dw->allocate(q_advected, lb->q_advectedLabel, 0, patch);
    new_dw->allocate(qV_CC,      lb->qV_CCLabel,      0, patch,
		     Ghost::AroundCells,1);
    new_dw->allocate(qV_advected,lb->qV_advectedLabel,0, patch);

    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();

      CCVariable<double> rho_CC, mass_CC, temp, sp_vol_CC;
      constCCVariable<double> rho_micro;
      CCVariable<Vector> vel_CC;
      constCCVariable<Vector> mom_L_ME;
      constCCVariable<double > int_eng_L_ME, mass_L;
      constCCVariable<double > mass_CC_old, createdVol;
      
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      new_dw->get(uvel_FC,lb->uvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(vvel_FC,lb->vvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(wvel_FC,lb->wvel_FCMELabel,indx,patch,Ghost::AroundCells,2);
      new_dw->get(mom_L_ME,  lb->mom_L_ME_CCLabel, indx,patch,
		  Ghost::AroundCells,1);
      new_dw->get(mass_L, lb->mass_L_CCLabel,indx,patch,Ghost::AroundCells,1);
      old_dw->get(mass_CC_old,lb->mass_CCLabel,indx,patch,
		  Ghost::AroundCells,1);
      
      new_dw->get(createdVol, lb->created_vol_CCLabel,
		  indx,patch,Ghost::AroundCells,1);
      new_dw->get(rho_micro,lb->rho_micro_CCLabel,indx,patch,
		  Ghost::AroundCells,1);
      new_dw->get(int_eng_L_ME,lb->int_eng_L_ME_CCLabel,indx,patch,
		  Ghost::AroundCells,1);

      new_dw->allocate(rho_CC,    lb->rho_CC_top_cycleLabel,  indx,patch);
      new_dw->allocate(mass_CC,   lb->mass_CCLabel,           indx,patch);
      new_dw->allocate(sp_vol_CC, lb->sp_vol_CCLabel,         indx,patch);
      new_dw->allocate(temp,      lb->temp_CCLabel,           indx,patch);
      new_dw->allocate(vel_CC,    lb->vel_CCLabel,            indx,patch);
      rho_CC.initialize(0.0);
      mass_CC.initialize(0.0);
      sp_vol_CC.initialize(0.0);
      temp.initialize(0.0);
      vel_CC.initialize(Vector(0.0,0.0,0.0));
      double cv = ice_matl->getSpecificHeat();
      //__________________________________
      //   Advection preprocessing

      Advector* advector = d_advector->clone(new_dw,patch);
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch);

      //__________________________________
      // Advect mass and backout mass_CC and rho_CC
      for(CellIterator iter=patch->getCellIterator(gc); !iter.done();iter++){
	IntVector c = *iter;
        q_CC[c] = mass_L[c] * invvol;
      }

      advector->advectQ(q_CC,patch,q_advected);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
	IntVector c = *iter;
        rho_CC[c]  = (mass_L[c] + q_advected[c]) * invvol;
      }
      setBC(rho_CC,   "Density",              patch,indx);
      // mass_CC is needed for MPMICE
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
	IntVector c = *iter;
        mass_CC[c] = rho_CC[c] * vol;
      }

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
      setBC(vel_CC,   "Velocity",             patch,indx);

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
      setBC(temp,     "Temperature",          patch,indx);

      //__________________________________
      // Advection of specific volume.  Advected quantity is a volume fraction
      if (numICEmatls != numALLmatls)  {
        // I am doing this so that we get a reasonable answer for sp_vol
        // in the extra cells.  This calculation will get overwritten in
        // the interior cells.
        for(CellIterator iter=patch->getExtraCellIterator();!iter.done();
          iter++){
	  IntVector c = *iter;
          sp_vol_CC[c] = 1.0/rho_micro[c];
        }
        for(CellIterator iter=patch->getCellIterator(gc); !iter.done();iter++){
	  IntVector c = *iter;
	  //          q_CC[c] = (mass_L[c]/rho_micro[c])*invvol;
	  q_CC[c] = (mass_CC_old[c]/rho_micro[c]
						 + createdVol[c])*invvol;
        }

	advector->advectQ(q_CC,patch,q_advected);

	// After the following expression, sp_vol_CC is the matl volume
        for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
	  IntVector c = *iter;
          sp_vol_CC[c] = (q_CC[c]*vol + q_advected[c]);
        }
        // Divide by the new mass_CC.
        for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
	  IntVector c = *iter;
	  sp_vol_CC[c] /= mass_CC[c];
        }
      }

      delete advector;

      //---- P R I N T   D A T A ------   
      if (switchDebug_advance_advect ) {
	ostringstream description;
	description << "AFTER_Advection_after_BC_Mat_" << indx << "_patch_"
		    <<  patch->getID();
	printVector( patch,1, description.str(), "xmom_L_CC", 0, mom_L_ME);
	printVector( patch,1, description.str(), "ymom_L_CC", 1, mom_L_ME);
	printVector( patch,1, description.str(), "zmom_L_CC", 2, mom_L_ME);
	printData(   patch,1, description.str(), "int_eng_L_CC",int_eng_L_ME);
	printData(   patch,1, description.str(), "rho_CC",      rho_CC);
	printData(   patch,1, description.str(), "Temp_CC",temp);
	printVector( patch,1, description.str(), "uvel_CC", 0, vel_CC);
	printVector( patch,1, description.str(), "vvel_CC", 1, vel_CC);
	printVector( patch,1, description.str(), "wvel_CC", 2, vel_CC);
      }

      new_dw->put(rho_CC,   lb->rho_CC_top_cycleLabel, indx,patch);
      new_dw->put(mass_CC,  lb->mass_CCLabel,          indx,patch);
      new_dw->put(sp_vol_CC,lb->sp_vol_CCLabel,        indx,patch);
      new_dw->put(vel_CC,   lb->vel_CCLabel,           indx,patch);
      new_dw->put(temp,     lb->temp_CCLabel,          indx,patch);

    }
  }  // patch loop 
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--
 Purpose~   Takes care Pressure_CC
 ---------------------------------------------------------------------  */
void ICE::setBC(CCVariable<double>& press_CC,
		const CCVariable<double>& rho_micro,
		const string& kind, const Patch* patch, const int mat_id)
{
  
  Vector dx = patch->dCell();
  Vector gravity = d_sharedState->getGravity();
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    BoundCondBase *bcs, *sym_bcs;
    BoundCond<double> *new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<double> *>(bcs);
    } else
      continue;
 
    if (sym_bcs != 0) { 
      press_CC.fillFaceFlux(face,0.0,dx);
    }

    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet") 
	press_CC.fillFace(face,new_bcs->getValue());

      if (new_bcs->getKind() == "Neumann") 
	press_CC.fillFaceFlux(face,new_bcs->getValue(),dx);
       
      if ( fabs(gravity.x()) > 0.0  || 
           fabs(gravity.y()) > 0.0  || 
	   fabs(gravity.z()) > 0.0) {
        press_CC.setHydrostaticPressureBC(face, gravity, rho_micro, dx);
      }
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--
 Purpose~   Takes care of Density_CC and Temperature_CC
 ---------------------------------------------------------------------  */
void ICE::setBC(CCVariable<double>& variable, const string& kind, 
		const Patch* patch, const int mat_id)
{
  Vector dx = patch->dCell();
  Vector grav = d_sharedState->getGravity();
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    BoundCondBase *bcs, *sym_bcs;
    BoundCond<double> *new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<double> *>(bcs);
    } else
      continue;
 
    if (sym_bcs != 0) { 
      if (kind == "Density" || kind == "Temperature" || kind == "set_if_sym_BC") {
        variable.fillFaceFlux(face,0.0,dx);
      }
    }   
    
    if (new_bcs != 0) {
      //__________________________________
      //  Density_CC
      if (kind == "Density") {
        if (new_bcs->getKind() == "Dirichlet") 
	  variable.fillFace(face,new_bcs->getValue());

        if (new_bcs->getKind() == "Neumann") 
	  variable.fillFaceFlux(face,new_bcs->getValue(),dx);
      }
 
      //__________________________________
      // Temperature_CC
      if (kind == "Temperature" ){ 
        if (new_bcs->getKind() == "Dirichlet") 
	    variable.fillFace(face,new_bcs->getValue());
           
         // Neumann && gravity                 
        if (new_bcs->getKind() == "Neumann" ) {  

          variable.fillFaceFlux(face,new_bcs->getValue(),dx);
            
          if(fabs(grav.x()) >0.0 ||fabs(grav.y()) >0.0 ||fabs(grav.z()) >0.0) {
            Material *matl = d_sharedState->getMaterial(mat_id);
            ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
            if(ice_matl) {
              double cv     = ice_matl->getSpecificHeat();
              double gamma  = ice_matl->getGamma();
                    
              ice_matl->getEOS()->hydrostaticTempAdjustment(face, 
                                  patch,  grav, gamma,
                                  cv,     dx,   variable);
            }  // if(ice_matl) 
          }  // if(gravity)
        }  // if(Neumann)
      }  //  if(Temperature)
    }  // if(new_bc)
  }  // Patch loop
}


/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--        
 Purpose~   Takes care of Velocity_CC Boundary conditions
 ---------------------------------------------------------------------  */
void ICE::setBC(CCVariable<Vector>& variable, const string& kind, 
		const Patch* patch, const int mat_id) 
{
  IntVector  low, hi;
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    BoundCondBase *bcs,*sym_bcs;
    BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<Vector> *>(bcs);
    } else
      continue;
    //__________________________________
    //  Tangent components Neumann = 0
    //  Normal components = negInterior
    //  It's negInterior since it's on the opposite side of the
    //  plane of symetry
    if (sym_bcs != 0 && (kind == "Velocity" || kind =="set_if_sym_BC") ) {
    
      variable.fillFaceFlux(face,Vector(0.,0.,0.),dx);

      variable.fillFaceNormal(face);
    }
      
    if (new_bcs != 0 && kind == "Velocity") {
      if (new_bcs->getKind() == "Dirichlet") 
	variable.fillFace(face,new_bcs->getValue());
      
      if (new_bcs->getKind() == "Neumann") 
	variable.fillFaceFlux(face,new_bcs->getValue(),dx);
       
      if (new_bcs->getKind() == "NegInterior") {
         variable.fillFaceFlux(face,Vector(0.0,0.0,0.0),dx, -1.0);
      }  
    }  // end velocity loop
  }  // end face loop
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--      
 Purpose~   Takes care of vel_FC.x()
 ---------------------------------------------------------------------  */
void ICE::setBC(SFCXVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    BoundCondBase *bcs, *sym_bcs;
    BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);
    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      //cout<< "Setting uvel_FC symmetry boundary conditons face "<<face<<endl;
      variable.fillFaceFlux(patch, face, 0.0 ,dx,offset);
      
      // Set normal component = 0
      if( face == Patch::xplus || face == Patch::xminus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars"); 
                                                  !iter.done(); iter++) { 
	  IntVector c = *iter;
          //cout<<" now working on uvel_FC "<<c<<endl;
          variable[c] = 0.0;  
        }
      }
    }
    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet" && comp == "x") {
        variable.fillFace(patch, face,new_bcs->getValue().x(),offset);
      }
/*`==========TESTING==========*/ 
#if 1
// fillFaceFlux is broken
// currently vel_FC[gc] = vel_FC[interior] which is WRONG
      if (new_bcs->getKind() == "Neumann" && comp == "x") {
        Vector dx = patch->dCell();
        variable.fillFaceFlux(patch, face,new_bcs->getValue().x(),dx,offset);
      }
#endif
 /*==========TESTING==========`*/
    }
  }
}
/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--      
 Purpose~   Takes care of vel_FC.y()
 ---------------------------------------------------------------------  */
void ICE::setBC(SFCYVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    BoundCondBase *bcs, *sym_bcs;
    BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);
    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      //cout<< "Setting vvel_FC symmetry boundary conditons face "<<face<<endl;
      variable.fillFaceFlux(patch, face, 0.0 ,dx,offset);
      
      // set normal compoent = 0
      if( face == Patch::yminus || face == Patch::yplus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars");
                                                    !iter.done(); iter++) { 
	  IntVector c = *iter;
          //cout<<" now working on vvel_FC"<< c<<endl;
          variable[c] = 0.0;  
        }
      }
    }
    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet" && comp == "y") {
        variable.fillFace(patch, face,new_bcs->getValue().y(),offset);
      }
/*`==========TESTING==========*/ 
#if 1
// fillFaceFlux is broken
// currently vel_FC[gc] = vel_FC[interior] which is WRONG
      if (new_bcs->getKind() == "Neumann" && comp == "y") {
        Vector dx = patch->dCell();
        variable.fillFaceFlux(patch, face,new_bcs->getValue().y(),dx,offset);
      }
#endif
 /*==========TESTING==========`*/
    }
  }
}
/* --------------------------------------------------------------------- 
 Function~  ICE::setBC--      
 Purpose~   Takes care of vel_FC.z()
 ---------------------------------------------------------------------  */
void ICE::setBC(SFCZVariable<double>& variable, const  string& kind, 
		const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    BoundCondBase *bcs, *sym_bcs;
    BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);

    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      variable.fillFaceFlux(patch, face, 0.0 ,dx,offset);
      
      // set normal component = 0
      if( face == Patch::zminus || face == Patch::zplus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars"); 
                                                      !iter.done(); iter++) { 
	  IntVector c = *iter;
          variable[c] = 0.0;  
        }
      }
    }

    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet" && comp == "z") {
        variable.fillFace(patch, face,new_bcs->getValue().z(),offset);
      }
/*`==========TESTING==========*/ 
#if 1
// fillFaceFlux is broken
// currently vel_FC[gc] = vel_FC[interior] which is WRONG
      if (new_bcs->getKind() == "Neumann" && comp == "z") {
        Vector dx = patch->dCell();
        variable.fillFaceFlux(patch, face,new_bcs->getValue().z(),dx,offset);
      }
#endif
 /*==========TESTING==========`*/
    }
  }
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
                          const CCVariable<double>& rho_micro_CC,
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
      press_hydro       = rho_micro_CC[curcell] * 
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
      press_hydro       = rho_micro_CC[curcell] * 
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
      press_hydro       = rho_micro_CC[curcell] * 
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
          
   WARNING: THIS DOESN'T COMPUTE THE SHEAR STRESS ON THE LAST X INTERIOR FACE
            THIS ISN'T BIG DEAL BUT SHOULD BE FIXED.    
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
  // loop over each cell
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
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
//    cout<<"tau_XX: "<<tau_X_FC[cell].x()<<
//          " tau_XY: "<<tau_X_FC[cell].y()<<
//          " tau_XZ: "<<tau_X_FC[cell].z()<<endl;
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
  WARNING: THIS DOESN'T COMPUTE THE SHEAR STRESS ON THE LAST Y INTERIOR FACE
            THIS ISN'T BIG DEAL BUT SHOULD BE FIXED. 
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
  // loop over the bottom then top cell face
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
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
//  cout<<"tau_YX: "<<tau_Y_FC[cell].x()<<
//        " tau_YY: "<<tau_Y_FC[cell].y()<<
//        " tau_YZ: "<<tau_Y_FC[cell].z()<<endl;
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
 WARNING: THIS DOESN'T COMPUTE THE SHEAR STRESS ON THE LAST Z INTERIOR FACE
            THIS ISN'T BIG DEAL BUT SHOULD BE FIXED. 
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
  // loop over the  faces
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){  
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
void ICE::getExchangeCoefficients( DenseMatrix& K,
                                   DenseMatrix& H  )
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


//______________________________________________________________________
//  Snippets of debugging code
#if 0
    int i, j, k;
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector curcell = *iter;       
     i = curcell.x();
     j = curcell.y();
     k = curcell.z();
}
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

