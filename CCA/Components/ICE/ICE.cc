#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
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


#undef  CONVECT
//#define CONVECT

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "ICE_NORMAL_COUT:+,ICE_DOING_COUT:+"
//  ICE_NORMAL_COUT:  dumps out during problemSetup 
//  ICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF
static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);

ICE::ICE(const ProcessorGroup* myworld) 
  : UintahParallelComponent(myworld)
{
  lb   = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  // Turn off all the debuging switches
  switchDebugInitialize           = false;
  switchDebug_EQ_RF_press         = false;
  switchDebug_vel_FC              = false;
  switchDebug_Temp_FC             = false;
  switchDebug_PressDiffRF         = false;
  switchDebug_Exchange_FC         = false;
  switchDebug_explicit_press      = false;
  switchDebug_setupMatrix         = false;
  switchDebug_setupRHS            = false;
  switchDebug_updatePressure      = false;
  switchDebug_computeDelP         = false;
  switchDebug_PressFC             = false;
  switchDebugLagrangianValues     = false;
  switchDebugLagrangianSpecificVol= false;
  switchDebugMomentumExchange_CC  = false;
  switchDebugSource_Sink          = false;
  switchDebug_advance_advect      = false;
  switchTestConservation          = false;

  d_massExchange      = false;    // MODEL REMOVE
  d_RateForm          = false;
  d_EqForm            = false; 
  d_add_heat          = false;
  d_impICE            = false;
  d_delT_knob         = 1.0;
  d_delT_scheme       = "aggressive";

  d_dbgVar1   = 0;     //inputs for debugging                               
  d_dbgVar2   = 0;                                                          
  d_SMALL_NUM = 1.0e-100;                                                   
  d_TINY_RHO  = 1.0e-12;// also defined ICEMaterial.cc and MPMMaterial.cc   
  d_modelInfo = 0;
  d_modelSetup = 0;
}

ICE::~ICE()
{
  delete lb;
  delete MIlb;
  delete d_advector;

  for(vector<ModelInterface*>::iterator iter = d_models.begin();
      iter != d_models.end(); iter++)
    delete *iter;

  releasePort("solver");

  if(d_modelInfo)
    delete d_modelInfo;
  if(d_modelSetup)
    delete d_modelSetup;
}


bool ICE::restartableTimesteps()
{
  // Only implicit ICE will restart timesteps
  return d_impICE;
}

double ICE::recomputeTimestep(double current_dt)
{
  return current_dt * 0.75;
}

/* ---------------------------------------------------------------------
 Function~  ICE::problemSetup--
_____________________________________________________________________*/
void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                        SimulationStateP&   sharedState)
{
  d_sharedState = sharedState;
  lb->delTLabel = sharedState->get_delt_label();

  cout_norm << "In the preprocessor . . ." << endl;
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(dataArchiver == 0){
    cout<<"dataArchiver in ICE is null now exiting; "<<endl;
    exit(1);
  }
  solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver) {
    throw InternalError("ICE:couldn't get solver port");
  } 

  //__________________________________
  // Find the switches
  ProblemSpecP debug_ps = prob_spec->findBlock("Debug");
  if (debug_ps) {
    d_dbgGnuPlot   = false;
    d_dbgStartTime = 0.;
    d_dbgStopTime  = 1.;
    d_dbgOutputInterval = 0.0;
    d_dbgBeginIndx = IntVector(0,0,0);
    d_dbgEndIndx   = IntVector(0,0,0);
    d_dbgSigFigs   = 5;

    debug_ps->get("dbg_GnuPlot",       d_dbgGnuPlot);
    debug_ps->get("dbg_var1",          d_dbgVar1);   
    debug_ps->get("dbg_var2",          d_dbgVar2);  
    debug_ps->get("dbg_timeStart",     d_dbgStartTime);
    debug_ps->get("dbg_timeStop",      d_dbgStopTime);
    debug_ps->get("dbg_outputInterval",d_dbgOutputInterval);
    debug_ps->get("dbg_BeginIndex",    d_dbgBeginIndx);
    debug_ps->get("dbg_EndIndex",      d_dbgEndIndx );
    debug_ps->get("dbg_SigFigs",       d_dbgSigFigs );
    debug_ps->get("dbg_Matls",         d_dbgMatls);
    
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
      else if (debug_attr["label"] == "switchDebug_Temp_FC")
       switchDebug_Temp_FC               = true;
      else if (debug_attr["label"] == "switchDebug_Exchange_FC")
       switchDebug_Exchange_FC          = true;
      else if (debug_attr["label"] == "switchDebug_explicit_press")
       switchDebug_explicit_press       = true;
      else if (debug_attr["label"] == "switchDebug_setupMatrix")
       switchDebug_setupMatrix          = true;
      else if (debug_attr["label"] == "switchDebug_setupRHS")
       switchDebug_setupRHS             = true;
      else if (debug_attr["label"] == "switchDebug_updatePressure")
       switchDebug_updatePressure       = true;
      else if (debug_attr["label"] == "switchDebug_computeDelP")
       switchDebug_computeDelP          = true;
      else if (debug_attr["label"] == "switchDebug_PressFC")
       switchDebug_PressFC              = true;
      else if (debug_attr["label"] == "switchDebugLagrangianValues")
       switchDebugLagrangianValues      = true;
      else if (debug_attr["label"] == "switchDebugLagrangianSpecificVol")
       switchDebugLagrangianSpecificVol = true;
      else if (debug_attr["label"] == "switchDebugMomentumExchange_CC")
       switchDebugMomentumExchange_CC   = true;
      else if (debug_attr["label"] == "switchDebugSource_Sink")
       switchDebugSource_Sink           = true;
      else if (debug_attr["label"] == "switchDebug_advance_advect")
       switchDebug_advance_advect       = true;
      else if (debug_attr["label"] == "switchTestConservation")
        switchTestConservation           = true;
    }
  }
  cout_norm << "Pulled out the debugging switches from input file" << endl;
  cout_norm<< "  debugging starting time "  <<d_dbgStartTime<<endl;
  cout_norm<< "  debugging stopping time "  <<d_dbgStopTime<<endl;
  cout_norm<< "  debugging output interval "<<d_dbgOutputInterval<<endl;
  cout_norm<< "  debugging variable 1 "     <<d_dbgVar1<<endl;
  cout_norm<< "  debugging variable 2 "     <<d_dbgVar2<<endl; 
  for (int i = 0; i<(int) d_dbgMatls.size(); i++) {
    cout_norm << "  d_dbg_matls = " << d_dbgMatls[i] << endl;
  } 
  
  //__________________________________
  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  cfd_ps->require("cfl",d_CFL);
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE"); 
  
  cfd_ice_ps->require("max_iteration_equilibration",d_max_iter_equilibration);
  d_advector = AdvectionFactory::create(cfd_ice_ps, d_advect_type);
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
    string warn="ERROR:\nMust specify EqForm or RateForm in ICE solution";
    throw ProblemSetupException(warn);
  }
  
  cout_norm << "cfl = " << d_CFL << endl;
  cout_norm << "max_iteration_equilibration "<<d_max_iter_equilibration<<endl;
  cout_norm << "Pulled out CFD-ICE block of the input file" << endl;
  //__________________________________
  //  Pull out add heat section
  ProblemSpecP add_heat_ps = cfd_ice_ps->findBlock("ADD_HEAT");
  if(add_heat_ps) {
    d_add_heat = true;
    add_heat_ps->require("add_heat_matls",d_add_heat_matls);
    add_heat_ps->require("add_heat_coeff",d_add_heat_coeff);
    add_heat_ps->require("add_heat_t_start",d_add_heat_t_start);
    add_heat_ps->require("add_heat_t_final",d_add_heat_t_final);
    cout_norm << "HEAT WILL BE ADDED"<<endl;
    cout_norm << "  d_add_heat_t_start: "<< d_add_heat_t_start
              << "  d_add_heat_t_final: "<< d_add_heat_t_final<< endl;
    for (int i = 0; i<(int) d_add_heat_matls.size(); i++) {
      cout_norm << "  d_add_heat_matl " << d_add_heat_matls[i] 
                << "  d_add_heat_coeff "<< d_add_heat_coeff[i]<< endl;
    } 
  }
 
  //__________________________________
  // Pull out implicit solver parameters
  ProblemSpecP impSolver = cfd_ice_ps->findBlock("ImplicitSolver");
  if (impSolver) {
    d_delT_knob = 0.5;      // default value when running implicit
    solver_parameters = solver->readParameters(impSolver, "implicitPressure");
    solver_parameters->setSolveOnExtraCells(false);
    impSolver->require("max_outer_iterations",      d_max_iter_implicit);
    impSolver->require("outer_iteration_tolerance", d_outer_iter_tolerance);
    impSolver->getWithDefault("iters_before_timestep_restart",    
                               d_iters_before_timestep_restart, 5);
    d_impICE = true; 
  }
    
  //__________________________________
  // Pull out TimeStepControl data
  ProblemSpecP tsc_ps = cfd_ice_ps->findBlock("TimeStepControl");
  if (tsc_ps ) {
    tsc_ps ->require("Scheme_for_delT_calc", d_delT_scheme);
    tsc_ps ->require("knob_for_speedSound",  d_delT_knob);
    
    if (d_delT_scheme != "conservative" && d_delT_scheme != "aggressive") {
     string warn="ERROR:\n Scheme_for_delT_calc:  must specify either aggressive or conservative";
     throw ProblemSetupException(warn);
    }
    if (d_delT_knob< 0.0 || d_delT_knob > 1.0) {
     string warn="ERROR:\n knob_for_speedSound:  must be between 0 and 1";
     throw ProblemSetupException(warn);
    }
  } 
  cout_norm << "Scheme for calculating delT: " << d_delT_scheme<< endl;
  cout_norm << "Limiter on speed of sound inside delT calc.: " << d_delT_knob<< endl;    
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

  for (int i = 0; i<(int)d_K_mom.size(); i++) {
    cout_norm << "K_mom = " << d_K_mom[i] << endl;
    if( d_K_mom[i] < 0.0 || d_K_mom[i] > 1e15 ) {
      ostringstream warn;
      warn<<"ERROR\n Momentum exchange coef. is either too big or negative\n";
      throw ProblemSetupException(warn.str());
    }
  }
  for (int i = 0; i<(int)d_K_heat.size(); i++) {
    cout_norm << "K_heat = " << d_K_heat[i] << endl;
    if( d_K_heat[i] < 0.0 || d_K_heat[i] > 1e15 ) {
      ostringstream warn;
      warn<<"ERROR\n Heat exchange coef. is either too big or negative\n";
      throw ProblemSetupException(warn.str());
    }
  }
  cout_norm << "Pulled out exchange coefficients of the input file" << endl;

  //__________________________________
  //  pull out mass exchange
  string mass_exch_in;
  ProblemSpecP mass_exch_ps = exch_ps->get("mass_exchange",mass_exch_in);
  d_massExchange = false;
  if (mass_exch_ps) {
    if (mass_exch_in == "true" || mass_exch_in == "TRUE" || 
        mass_exch_in == "1") {
      d_massExchange = true;
      if (d_RateForm) {
        string warn="ERROR\n RateForm doesn't work with a reaction\n";
        throw ProblemSetupException(warn);
      }
    }
  }

  cout_norm << "Mass exchange = " << d_massExchange << endl;
  
  //__________________________________
  // WARNINGS
  SimulationTime timeinfo(prob_spec); 
  if ( d_impICE && 
       (timeinfo.max_delt_increase  > 10  || d_delT_scheme != "conservative" ) ) {
    cout <<"\n \n W A R N I N G: " << endl;
    cout << " When running implicit ICE you should specify "<<endl;
    cout <<" \t \t <max_delt_increase>    2.0ish  "<<endl;
    cout << "\t \t <Scheme_for_delT_calc> conservative " << endl;
    cout << " to a) prevent rapid fluctuations in the timestep and "<< endl;
    cout << "    b) to prevent outflux Vol > cell volume \n \n" <<endl;
  } 

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
  if (switchDebug_setupMatrix == true) 
    cout_norm << "switchDebug_setupMatrix is ON" << endl;
  if (switchDebug_setupRHS == true) 
    cout_norm << "switchDebug_setupRHS is ON" << endl;
  if (switchDebug_updatePressure == true) 
    cout_norm << "switchDebug_updatePressure is ON" << endl;
  if (switchDebug_PressFC == true) 
    cout_norm << "switchDebug_PressFC is ON" << endl;
  if (switchDebugLagrangianValues == true) 
    cout_norm << "switchDebugLagrangianValues is ON" << endl;
  if (switchDebugLagrangianSpecificVol == true) 
    cout_norm << "switchDebugLagrangianSpecificVol is ON" << endl;
  if (switchDebugSource_Sink == true) 
    cout_norm << "switchDebugSource_Sink is ON" << endl;
  if (switchDebug_advance_advect == true) 
    cout_norm << "switchDebug_advance_advect is ON" << endl;
  if (switchTestConservation == true)
    cout_norm << "switchTestConservation is ON" << endl;

  //__________________________________
  //  Load Model info.
  ModelMaker* modelMaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  if(modelMaker){
    modelMaker->makeModels(prob_spec, grid, sharedState, d_models);
    releasePort("ModelMaker");
    d_modelSetup = scinew ICEModelSetup();
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      (*iter)->problemSetup(grid, sharedState, d_modelSetup);
    }

    d_modelInfo = scinew ModelInfo(d_sharedState->get_delt_label(),
                               lb->modelMass_srcLabel,
                               lb->modelMom_srcLabel,
                               lb->modelEng_srcLabel,
                               lb->modelVol_srcLabel,
                               lb->rho_CCLabel,
                               lb->vel_CCLabel,
                               lb->temp_CCLabel,
                               lb->press_CCLabel,
                               lb->sp_vol_CCLabel);
  }
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
  t->computes(lb->vel_CCLabel);
  t->computes(lb->rho_CCLabel); 
  t->computes(lb->temp_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->rho_micro_CCLabel);
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->press_CCLabel, press_matl);
  t->computes(lb->imp_delPLabel, press_matl); 
  
  sched->addTask(t, level->eachPatch(), d_sharedState->allICEMaterials());

  // The task will have a reference to press_matl
  if (press_matl->removeReference())
    delete press_matl; // shouln't happen, but...

  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleInitialize(sched, level, d_modelInfo);
    }
  }
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
  Task* t = 0;
  if (d_EqForm) {             // EQ 
    cout_doing << "ICE::scheduleComputeStableTimestep " << endl;
    t = scinew Task("ICE::actuallyComputeStableTimestep",
                     this, &ICE::actuallyComputeStableTimestep);
  } else if (d_RateForm) {    // RF
    cout_doing << "ICE::scheduleComputeStableTimestepRF " << endl;
    t = scinew Task("ICE::actuallyComputeStableTimestepRF",
                      this, &ICE::actuallyComputeStableTimestepRF);
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  const MaterialSet* all_matls = d_sharedState->allMaterials(); 
  if (d_EqForm){            // EQ      
    t->requires(Task::NewDW, lb->vel_CCLabel,        gac, 1);  
    t->requires(Task::NewDW, lb->speedSound_CCLabel, gac, 1);
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,     gn,  0);  
  } else if (d_RateForm){   // RATE FORM
    t->requires(Task::NewDW, lb->vel_CCLabel,        gac, 1);
    t->requires(Task::NewDW, lb->speedSound_CCLabel, gac, 1);
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,     gac, 1); 
  }
  t->computes(d_sharedState->get_delt_label());
  sched->addTask(t,level->eachPatch(), all_matls); 
  
  //__________________________________
  //  If model needs to further restrict the timestep
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleComputeStableTimestep(sched, level, d_modelInfo);
    }
  }
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleTimeAdvance--
_____________________________________________________________________*/
void
ICE::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, int, int )
{
  cout_doing << "ICE::scheduleTimeAdvance" << endl;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  
  MaterialSubset* press_matl    = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  MaterialSubset* one_matl = press_matl;
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();

  scheduleComputePressure(                sched, patches, press_matl,
                                                          all_matls);

  if (d_RateForm) {                                                                 
    schedulecomputeDivThetaVel_CC(        sched, patches, ice_matls_sub,        
                                                          mpm_matls_sub,        
                                                          all_matls);           
  }  
  
  scheduleComputeTempFC(                   sched, patches, ice_matls_sub,  
                                                           mpm_matls_sub,         
                                                           all_matls);    
                                                                 
  scheduleMassExchange(                    sched, patches, all_matls);                                                           
  scheduleModelMassExchange(               sched, level,   all_matls);

  if(d_impICE) {        //  I M P L I C I T                                           
    scheduleImplicitPressureSolve(         sched, level,   patches,       
                                                           one_matl,      
                                                           press_matl,    
                                                           ice_matls_sub,  
                                                           mpm_matls_sub, 
                                                           all_matls);
                                                           
    scheduleComputeDel_P(                   sched,  level, patches,  
                                                           one_matl,
                                                           press_matl,
                                                           all_matls);    
                                                           
  }                     //  IMPLICIT AND EXPLICIT
    scheduleComputeVel_FC(                  sched, patches,ice_matls_sub, 
                                                           mpm_matls_sub, 
                                                           press_matl,    
                                                           all_matls,     
                                                           false);        

    scheduleAddExchangeContributionToFCVel( sched, patches,all_matls,
                                                           false); 

  if(!d_impICE){         //  E X P L I C I T
    scheduleComputeDelPressAndUpdatePressCC(sched, patches,press_matl,     
                                                           ice_matls_sub,  
                                                           mpm_matls_sub,  
                                                           all_matls);     
  }
  
  scheduleComputePressFC(                 sched, patches, press_matl,
                                                          all_matls);

  scheduleAccumulateMomentumSourceSinks(  sched, patches, press_matl,
                                                          ice_matls_sub,
                                                          all_matls);

  scheduleAccumulateEnergySourceSinks(    sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          press_matl,
                                                          all_matls);

  scheduleModelMomentumAndEnergyExchange( sched, level,   all_matls);

  scheduleComputeLagrangianValues(        sched, patches, all_matls);

  scheduleAddExchangeToMomentumAndEnergy( sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          press_matl,
                                                          all_matls);
                                                           
  scheduleComputeLagrangianSpecificVolume(sched, patches, ice_matls_sub,
                                                          mpm_matls_sub, 
                                                          press_matl,
                                                          all_matls);

  scheduleAdvectAndAdvanceInTime(         sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          press_matl,
                                                          all_matls);
  if(switchTestConservation) {
    schedulePrintConservedQuantities(     sched, patches, ice_matls_sub,
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
  Task* t = 0;
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
                        // EQ & RATE FORM
  t->requires(Task::OldDW,lb->press_CCLabel, press_matl, Ghost::None);
  t->requires(Task::OldDW,lb->rho_CCLabel,               Ghost::None);
  t->requires(Task::OldDW,lb->temp_CCLabel,              Ghost::None); 
  t->requires(Task::OldDW,lb->sp_vol_CCLabel,            Ghost::None); 
  t->computes(lb->f_theta_CCLabel); 
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->rho_CCLabel);
  t->computes(lb->press_equil_CCLabel, press_matl);
  t->computes(lb->press_CCLabel,       press_matl);  // needed by implicit

  if (d_RateForm) {     // RATE FORM
    t->computes(lb->matl_press_CCLabel);
  }
    
  sched->addTask(t, patches, ice_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeTempFC--
_____________________________________________________________________*/
void ICE::scheduleComputeTempFC(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSubset* ice_matls,
                                const MaterialSubset* mpm_matls,
                                const MaterialSet* all_matls)
{ 
  Task* t;
  cout_doing << "ICE::scheduleComputeTempFC" << endl;
  t = scinew Task("ICE::computeTempFC", this, &ICE::computeTempFC);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW,lb->rho_CCLabel,     /*all_matls*/ gac,1);
  t->requires(Task::OldDW,lb->temp_CCLabel,      ice_matls,  gac,1);
  t->requires(Task::NewDW,lb->temp_CCLabel,      mpm_matls,  gac,1);
  
  t->computes(lb->TempX_FCLabel);
  t->computes(lb->TempY_FCLabel);
  t->computes(lb->TempZ_FCLabel);
  sched->addTask(t, patches, all_matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleComputeVel_FC--
_____________________________________________________________________*/
void ICE::scheduleComputeVel_FC(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSubset* ice_matls,
                                const MaterialSubset* mpm_matls,
                                const MaterialSubset* press_matl,
                                const MaterialSet* all_matls,
                                bool recursion)
{ 
  Task* t = 0;
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleComputeFaceCenteredVelocitiesRF" << endl;
    t = scinew Task("ICE::computeFaceCenteredVelocitiesRF",
              this, &ICE::computeFaceCenteredVelocitiesRF);
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::scheduleComputeVel_FC" << endl;
    t = scinew Task("ICE::computeVel_FC",
              this, &ICE::computeVel_FC, recursion);
  }
                      // EQ  & RATE FORM 
  Ghost::GhostType  gac = Ghost::AroundCells;                      
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW,lb->press_CCLabel,       press_matl, gac,1);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,    /*all_matls*/ gac,1);
  t->requires(Task::NewDW,lb->rho_CCLabel,       /*all_matls*/ gac,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,         ice_matls,  gac,1);
  t->requires(Task::NewDW,lb->vel_CCLabel,         mpm_matls,  gac,1);
  
  if (d_RateForm) {     //RATE FORM
    t->requires(Task::NewDW,lb->DLabel,                        gac, 1);
    t->requires(Task::NewDW,lb->matl_press_CCLabel,            gac, 1);
    t->requires(Task::NewDW,lb->vol_frac_CCLabel,              gac, 1);
    t->requires(Task::NewDW,lb->speedSound_CCLabel,            gac, 1);
    t->computes(lb->press_diffX_FCLabel);
    t->computes(lb->press_diffY_FCLabel);
    t->computes(lb->press_diffZ_FCLabel); 
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
                                           const MaterialSet* matls,
                                           const bool recursion)
{
  cout_doing << "ICE::scheduleAddExchangeContributionToFCVel" << endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel, recursion);

  task->requires(Task::OldDW, lb->delTLabel);  
  task->requires(Task::NewDW,lb->sp_vol_CCLabel,    Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->vol_frac_CCLabel,  Ghost::AroundCells,1);
  task->requires(Task::NewDW,lb->uvel_FCLabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW,lb->vvel_FCLabel,      Ghost::AroundCells,2);
  task->requires(Task::NewDW,lb->wvel_FCLabel,      Ghost::AroundCells,2);

  task->computes(lb->sp_volX_FCLabel);
  task->computes(lb->sp_volY_FCLabel);
  task->computes(lb->sp_volZ_FCLabel); 
  task->computes(lb->uvel_FCMELabel);
  task->computes(lb->vvel_FCMELabel);
  task->computes(lb->wvel_FCMELabel);
  
  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleMassExchange--
 MODEL REMOVE -- entire task
_____________________________________________________________________*/
void ICE::scheduleMassExchange(SchedulerP& sched,
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
 Function~  ICE::scheduleModelMassExchange--
_____________________________________________________________________*/
void ICE::scheduleModelMassExchange(SchedulerP& sched, const LevelP& level,
                                const MaterialSet* matls)
{
  if(d_models.size() != 0){
    cout_doing << "ICE::scheduleModelMassExchange" << endl;
    Task* task = scinew Task("ICE::zeroModelSources",
                          this, &ICE::zeroModelSources);
    task->computes(lb->modelMass_srcLabel);
    task->computes(lb->modelMom_srcLabel);
    task->computes(lb->modelEng_srcLabel);
    task->computes(lb->modelVol_srcLabel);
    for(vector<TransportedVariable*>::iterator iter = d_modelSetup->tvars.begin();
	iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      if(tvar->src)
	task->computes(tvar->src, tvar->matls);
    }
    sched->addTask(task, level->eachPatch(), matls);

    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleMassExchange(sched, level, d_modelInfo);
    }
  }
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
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;  
  task->requires( Task::OldDW, lb->delTLabel);
  task->requires( Task::NewDW, lb->press_equil_CCLabel,
                                          press_matl,  gn);
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2);
  task->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->sp_vol_CCLabel,     gn);
  task->requires( Task::NewDW, lb->rho_CCLabel,        gn);    
  task->requires( Task::NewDW, lb->speedSound_CCLabel, gn);
//__________________________________
  if(d_models.size() == 0){  //MODEL REMOVE
    task->requires( Task::NewDW, lb->burnedMass_CCLabel, gn);
    task->requires( Task::NewDW, lb->created_vol_CCLabel,gn); //MODEL REMOVE
  }
//__________________________________
  if(d_models.size() > 0){
    task->requires(Task::NewDW, lb->modelVol_srcLabel,  gn);
    task->requires(Task::NewDW, lb->modelMass_srcLabel, gn);
  }
  
/*`==========TESTING==========*/
#ifdef LODI_BCS
  task->requires(Task::OldDW, lb->temp_CCLabel,     ice_matls, gn);    
  task->requires(Task::NewDW,MIlb->temp_CCLabel,    mpm_matls, gn);    
  task->requires(Task::NewDW, lb->f_theta_CCLabel,             gn);
#endif 
/*==========TESTING==========`*/
  
  task->modifies(lb->press_CCLabel,        press_matl);
  task->computes(lb->delP_DilatateLabel,   press_matl);
  task->computes(lb->delP_MassXLabel,      press_matl);
  task->computes(lb->term2Label,           press_matl);
  task->computes(lb->term3Label,           press_matl);
  task->computes(lb->sum_rho_CCLabel,      press_matl);  // only one mat subset     
  
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
                     
  Ghost::GhostType  gac = Ghost::AroundCells;
  task->requires(Task::NewDW,lb->press_CCLabel,   press_matl,gac,1);
  task->requires(Task::NewDW,lb->sum_rho_CCLabel, press_matl,gac,1);

  task->computes(lb->pressX_FCLabel, press_matl);
  task->computes(lb->pressY_FCLabel, press_matl);
  task->computes(lb->pressZ_FCLabel, press_matl);

  sched->addTask(task, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleModelMomentumAndEnergyExchange--
_____________________________________________________________________*/
void ICE::scheduleModelMomentumAndEnergyExchange(SchedulerP& sched,
                                           const LevelP& level,
                                           const MaterialSet* /*matls*/)
{
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleMomentumAndEnergyExchange(sched, level, d_modelInfo);
    }
  }
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
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW,lb->pressX_FCLabel,   press_matl,    gac, 1);
  t->requires(Task::NewDW,lb->pressY_FCLabel,   press_matl,    gac, 1);
  t->requires(Task::NewDW,lb->pressZ_FCLabel,   press_matl,    gac, 1);
  t->requires(Task::OldDW,lb->vel_CCLabel,      ice_matls_sub, gac, 2); 
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,   ice_matls_sub, gac, 2);
  t->requires(Task::NewDW,lb->rho_CCLabel,                     gac, 2);
  t->requires(Task::NewDW,lb->vol_frac_CCLabel, Ghost::None);
  if (d_RateForm) {   // RATE FORM
    t->requires(Task::NewDW,lb->press_diffX_FCLabel, gac, 1);
    t->requires(Task::NewDW,lb->press_diffY_FCLabel, gac, 1);
    t->requires(Task::NewDW,lb->press_diffZ_FCLabel, gac, 1);
  }
 
  t->computes(lb->mom_source_CCLabel);
  t->computes(lb->press_force_CCLabel);
  sched->addTask(t, patches, matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAccumulateEnergySourceSinks--
_____________________________________________________________________*/
void ICE::scheduleAccumulateEnergySourceSinks(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSubset* ice_matls,
                                         const MaterialSubset* mpm_matls,
                                         const MaterialSubset* press_matl,
                                         const MaterialSet* matls)

{
  Task* t;              // EQ
    cout_doing << "ICE::scheduleAccumulateEnergySourceSinks" << endl;
    t = scinew Task("ICE::accumulateEnergySourceSinks",
                     this, &ICE::accumulateEnergySourceSinks);
                     
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleAccumulateEnergySourceSinks_RF" << endl;
    t = scinew Task("ICE::accumulateEnergySourceSinks_RF",
                       this, &ICE::accumulateEnergySourceSinks_RF);
  }
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;  
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,gn);
  t->requires(Task::NewDW, lb->speedSound_CCLabel,           gn);
  t->requires(Task::OldDW, lb->temp_CCLabel,      ice_matls, gac,1);
  t->requires(Task::NewDW, lb->rho_CCLabel,                  gac,1);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,               gac,1);
    
  if (d_EqForm) {       //EQ FORM
    t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,gn);
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,             gn);
  }
  if (d_RateForm) {     //RATE FORM
    t->requires(Task::NewDW, lb->f_theta_CCLabel,            gn,0);
    t->requires(Task::OldDW, lb->vel_CCLabel,     ice_matls, gn,0);    
    t->requires(Task::NewDW, lb->vel_CCLabel,     mpm_matls, gn,0);    
    t->requires(Task::NewDW, lb->pressX_FCLabel,  press_matl,gac,1);             
    t->requires(Task::NewDW, lb->pressY_FCLabel,  press_matl,gac,1);             
    t->requires(Task::NewDW, lb->pressZ_FCLabel,  press_matl,gac,1);             
    t->requires(Task::NewDW, lb->uvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->vvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->wvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->press_diffX_FCLabel,        gac,1);     
    t->requires(Task::NewDW, lb->press_diffY_FCLabel,        gac,1);     
    t->requires(Task::NewDW, lb->press_diffZ_FCLabel,        gac,1);
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,           gac,1);          
  }

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
  Task* t = scinew Task("ICE::computeLagrangianValues",
                      this,&ICE::computeLagrangianValues);
  Ghost::GhostType  gn  = Ghost::None; 
  t->requires(Task::NewDW,lb->rho_CCLabel,             gn);
  t->requires(Task::OldDW,lb->vel_CCLabel,             gn);
  t->requires(Task::OldDW,lb->temp_CCLabel,            gn);
  t->requires(Task::NewDW,lb->mom_source_CCLabel,      gn);
  t->requires(Task::NewDW,lb->int_eng_source_CCLabel,  gn);
//__________________________________  
  if(d_models.size() == 0){  //MODEL REMOVE
    t->requires(Task::NewDW,lb->int_eng_comb_CCLabel,  gn);  
    t->requires(Task::NewDW,lb->mom_comb_CCLabel,      gn);  
    t->requires(Task::NewDW,lb->burnedMass_CCLabel,    gn);  
  }
//__________________________________  
  if(d_models.size() > 0){
    t->requires(Task::NewDW, lb->modelMass_srcLabel,   gn);
    t->requires(Task::NewDW, lb->modelMom_srcLabel,    gn);
    t->requires(Task::NewDW, lb->modelEng_srcLabel,    gn);
  }

  t->computes(lb->mom_L_CCLabel);
  t->computes(lb->int_eng_L_CCLabel);
  t->computes(lb->mass_L_CCLabel);
 
  sched->addTask(t, patches, ice_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE:: scheduleComputeLagrangianSpecificVolume--
_____________________________________________________________________*/
void ICE::scheduleComputeLagrangianSpecificVolume(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* ice_matls,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* press_matl,
                                            const MaterialSet* matls)
{
  Task* t = 0;

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

  Ghost::GhostType  gn  = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;       

  t->requires(Task::OldDW, lb->delTLabel);                         
  t->requires(Task::NewDW, lb->rho_CCLabel,         gn);           
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,      gn);           
  t->requires(Task::NewDW, lb->Tdot_CCLabel,        gn);           
  t->requires(Task::NewDW, lb->f_theta_CCLabel,     gn);           
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,    gac,1);        
  if (d_RateForm) {         // RATE FORM
    t->requires(Task::NewDW, lb->uvel_FCMELabel,      gac,1);        
    t->requires(Task::NewDW, lb->vvel_FCMELabel,      gac,1);        
    t->requires(Task::NewDW, lb->wvel_FCMELabel,      gac,1);        
  }
  if (d_EqForm) {         // RATE FORM
    t->requires(Task::NewDW, lb->speedSound_CCLabel,  gn);
    t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,gn);
  }
  t->requires(Task::OldDW, lb->temp_CCLabel,   ice_matls,   gn);   
  t->requires(Task::NewDW, lb->temp_CCLabel,   mpm_matls,   gn);   
//__________________________________          
  if(d_models.size() == 0){
    t->requires(Task::NewDW, lb->created_vol_CCLabel, gn);     //MODEL REMOVE
  }
//__________________________________
  if(d_models.size() > 0){
    t->requires(Task::NewDW, lb->modelVol_srcLabel,    gn);
  }

  t->computes(lb->spec_vol_L_CCLabel);                             
  t->computes(lb->spec_vol_source_CCLabel);                        

  sched->addTask(t, patches, matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAddExchangeToMomentumAndEnergy--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeToMomentumAndEnergy(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSubset* ice_matls,
                               const MaterialSubset* mpm_matls,
                               const MaterialSubset* press_matl,
                               const MaterialSet* all_matls)
{
  Task* t = 0;
  if (d_RateForm) {     //RATE FORM
    cout_doing << "ICE::scheduleAddExchangeToMomentumAndEnergy_RF" << endl;
    t=scinew Task("ICE::addExchangeToMomentumAndEnergyRF",
                  this, &ICE::addExchangeToMomentumAndEnergyRF);
  }
  else if (d_EqForm) {       // EQ 
    cout_doing << "ICE::scheduleAddExchangeToMomentumAndEnergy" << endl;
    t=scinew Task("ICE::addExchangeToMomentumAndEnergy",
                  this, &ICE::addExchangeToMomentumAndEnergy);
  }

  Ghost::GhostType  gn  = Ghost::None; 
  t->requires(Task::OldDW, d_sharedState->get_delt_label());
 
/*`==========TESTING==========*/
#ifdef CONVECT
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  t->requires(Task::NewDW,MIlb->gMassLabel,    mpm_matls,     gac, 1);      
  t->requires(Task::OldDW,MIlb->NC_CCweightLabel, press_matl, gac, 1);
#endif 
/*==========TESTING==========`*/
                                // I C E
  t->requires(Task::OldDW,  lb->temp_CCLabel,  ice_matls, gn); 
                                // A L L  M A T L S
  t->requires(Task::NewDW,  lb->mass_L_CCLabel,           gn);      
  t->requires(Task::NewDW,  lb->mom_L_CCLabel,            gn);      
  t->requires(Task::NewDW,  lb->int_eng_L_CCLabel,        gn);
  t->requires(Task::NewDW,  lb->sp_vol_CCLabel,           gn);      
  t->requires(Task::NewDW,  lb->vol_frac_CCLabel,         gn);      
  if (d_RateForm) {         // RATE FORM
    t->requires(Task::NewDW, lb->f_theta_CCLabel,         gn);      
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,          gn);      
    t->requires(Task::NewDW, lb->rho_CCLabel,             gn);      
    t->requires(Task::NewDW, lb->speedSound_CCLabel,      gn);        
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl, gn);
    t->requires(Task::OldDW, lb->vel_CCLabel,       ice_matls,  gn); 
  }

  t->computes(lb->Tdot_CCLabel);
  t->computes(lb->mom_L_ME_CCLabel);      
  t->computes(lb->eng_L_ME_CCLabel); 
  
  if (mpm_matls->size() > 0){  
    t->modifies(lb->temp_CCLabel, mpm_matls);
    t->modifies(lb->vel_CCLabel,  mpm_matls);
  }
  sched->addTask(t, patches, all_matls);
} 

/* ---------------------------------------------------------------------
 Function~  ICE::scheduleAdvectAndAdvanceInTime--
_____________________________________________________________________*/
void ICE::scheduleAdvectAndAdvanceInTime(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* ice_matls,
                                    const MaterialSubset* mpm_matls,
                                    const MaterialSubset* press_matl,
                                    const MaterialSet* matls)
{
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  Ghost::GhostType  gn   = Ghost::None; 
  cout_doing << "ICE::scheduleAdvectAndAdvanceInTime" << endl;
  Task* task = scinew Task("ICE::advectAndAdvanceInTime",
                     this, &ICE::advectAndAdvanceInTime);
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::NewDW, lb->uvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->vvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->wvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->mom_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,      gac,2);
  task->requires(Task::NewDW, lb->eng_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->spec_vol_L_CCLabel,  gac,2);
  task->requires(Task::NewDW, lb->speedSound_CCLabel,  gn, 0);  
/*`==========TESTING==========*/
#ifdef LODI_BCS  
  task->requires(Task::OldDW, lb->press_CCLabel,    press_matl, gn, 0);      
  task->requires(Task::OldDW, lb->temp_CCLabel,     ice_matls,  gn, 0);      
  task->requires(Task::OldDW, lb->rho_CCLabel,      ice_matls,  gn, 0);      
  task->requires(Task::OldDW, lb->vel_CCLabel,      ice_matls,  gn, 0);      
  task->requires(Task::OldDW, lb->sp_vol_CCLabel,   ice_matls,  gn, 0);      
  task->requires(Task::OldDW, lb->vol_frac_CCLabel, ice_matls,  gn, 0);
#endif
/*==========TESTING==========`*/
  task->modifies(lb->rho_CCLabel,   ice_matls);
  task->modifies(lb->sp_vol_CCLabel,ice_matls);
  task->computes(lb->temp_CCLabel,  ice_matls);
  task->computes(lb->vel_CCLabel,   ice_matls);
  task->computes(lb->machLabel,     ice_matls);  
  
  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    for(vector<TransportedVariable*>::iterator iter = d_modelSetup->tvars.begin();
       iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::OldDW, tvar->var, tvar->matls, gac, 2);
      if(tvar->src)
	task->requires(Task::NewDW, tvar->src, tvar->matls, gac, 2);
      task->computes(tvar->var,   tvar->matls);
    }
  } 
  
  sched->addTask(task, patches, matls);
}
/* ---------------------------------------------------------------------
 Function~  ICE::schedulePrintConservedQuantities--
_____________________________________________________________________*/
void ICE::schedulePrintConservedQuantities(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSubset* ice_matls,
                                      const MaterialSet* all_matls)
{
  cout_doing << "ICE::schedulePrintConservedQuantities" << endl;
  Task* t= scinew Task("ICE::printConservedQuantities",
                 this, &ICE::printConservedQuantities);
  
  Ghost::GhostType  gn  = Ghost::None;                    
  t->requires(Task::NewDW,lb->rho_CCLabel,  ice_matls, gn);
  t->requires(Task::NewDW,lb->vel_CCLabel,  ice_matls, gn);
  t->requires(Task::NewDW,lb->temp_CCLabel, ice_matls, gn);
                               // A L L  M A T L S         
  t->requires(Task::NewDW,lb->mom_L_CCLabel,           gn);         
  t->requires(Task::NewDW,lb->int_eng_L_CCLabel,       gn);    
  t->requires(Task::NewDW,lb->mom_L_ME_CCLabel,        gn);         
  t->requires(Task::NewDW,lb->eng_L_ME_CCLabel,        gn); 
  
  t->computes(lb->mom_exch_errorLabel);
  t->computes(lb->eng_exch_errorLabel);
  t->computes(lb->TotalMassLabel);
  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->TotalIntEngLabel);
  t->computes(lb->CenterOfMassVelocityLabel); //momentum
  sched->addTask(t, patches, all_matls);
}

/* ---------------------------------------------------------------------
 Function~  ICE::actuallyComputeStableTimestep--
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
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    double delt_CFL = 1e3, delt_cond = 1e3, delt;
    double inv_sum_invDelx_sqr = 1.0/( 1.0/(delX * delX) 
                                     + 1.0/(delY * delY) 
                                     + 1.0/(delZ * delZ) );
    constCCVariable<double> speedSound, sp_vol_CC;
    constCCVariable<Vector> vel_CC;
    Ghost::GhostType  gn  = Ghost::None; 
    Ghost::GhostType  gac = Ghost::AroundCells;

    double dCFL = d_CFL;
    static double TIME = 0.;

    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
      Material* matl = d_sharedState->getICEMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx= matl->getDWIndex(); 
      new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gac, 1);
      new_dw->get(vel_CC,     lb->vel_CCLabel,        indx,patch,gac, 1);
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,  0);

     if (d_delT_scheme == "aggressive") {     //      A G G R E S S I V E
        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double speed_Sound = d_delT_knob * speedSound[c];
          double A = dCFL*delX/(speed_Sound + 
                                       fabs(vel_CC[c].x())+d_SMALL_NUM);
          double B = dCFL*delY/(speed_Sound + 
                                       fabs(vel_CC[c].y())+d_SMALL_NUM);
          double C = dCFL*delZ/(speed_Sound + 
                                       fabs(vel_CC[c].z())+d_SMALL_NUM);
          delt_CFL = std::min(A, delt_CFL);
          delt_CFL = std::min(B, delt_CFL);
          delt_CFL = std::min(C, delt_CFL);
        }
//      cout << "  Aggressive delT Based on currant number "<< delt_CFL << endl; 
      } 

      if (d_delT_scheme == "conservative") {  //      C O N S E R V A T I V E
        //__________________________________
        // Use a characteristic velocity
        // to compute a sweptvolume. The
        // swept volume can't exceed the cell volume
        vector<IntVector> adj_offset(3);                   
        adj_offset[0] = IntVector(1, 0, 0);    // X 
        adj_offset[1] = IntVector(0, 1, 0);    // Y 
        adj_offset[2] = IntVector(0, 0, 1);    // Z   

        Vector faceArea;
        faceArea[0] = dx.y() * dx.z();        // X
        faceArea[1] = dx.x() * dx.z();        // Y
        faceArea[2] = dx.x() * dx.y();        // Z

        double vol = dx.x() * dx.y() * dx.z();  
        Vector grav = d_sharedState->getGravity();
        double grav_vel =  Sqrt( dx.x() * fabs(grav.x()) + 
                                 dx.y() * fabs(grav.y()) + 
                                 dx.z() * fabs(grav.z()) ); 
                                 
                                 
        double thermalCond, cv, gamma;
        double viscosity = 0.0;
        if (ice_matl){                         
          thermalCond = ice_matl->getThermalConductivity();
          viscosity   = ice_matl->getViscosity();
          cv          = ice_matl->getSpecificHeat();
          gamma       = ice_matl->getGamma();
        }else{
          thermalCond = mpm_matl->getThermalConductivity();
          cv          = mpm_matl->getSpecificHeat();
          gamma       = mpm_matl->getGamma();
        }
                                     
        double cp    = cv * gamma;       
        double dx_length   = dx.length();
        
        delt_CFL = 1000.0; 
        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          double sumSwept_Vol = 0.0;
          IntVector c = *iter;
          for (int dir = 0; dir <3; dir++) {  //loop over all three directions
            IntVector L = c - adj_offset[dir];
            IntVector R = c + adj_offset[dir];

            double vel_R = vel_CC[R][dir];
            double vel_C = vel_CC[c][dir];
            double vel_L = vel_CC[L][dir];

            double vel_FC_R= 0.5 * (vel_R + vel_C);
            double vel_FC_L= 0.5 * (vel_L + vel_C);

            double c_L = speedSound[L];  
            double c_R = speedSound[R];                    
            double speedSound = max(c_L,c_R );      

            double relative_vel       = fabs(vel_R - vel_L);

            double thermalDiffusivity = thermalCond * sp_vol_CC[c]/cp;
            double diffusion_vel      = std::max(thermalDiffusivity, viscosity)
                                      /dx_length;

            double characteristicVel_R = vel_FC_R 
                                       + d_delT_knob * speedSound 
                                       + relative_vel
                                       + grav_vel
                                       + diffusion_vel; 
            double characteristicVel_L = vel_FC_L 
                                       - d_delT_knob * speedSound 
                                       - relative_vel
                                       - grav_vel
                                       - diffusion_vel;

            double sweptVol_R = characteristicVel_R * faceArea[dir];
            double sweptVol_L = characteristicVel_L * -faceArea[dir]; 
            
            sweptVol_R = std::max( 0.0, sweptVol_R);  // only compute outflow volumes
            sweptVol_L = std::max( 0.0, sweptVol_L);
            sumSwept_Vol += sweptVol_R + sweptVol_L;
          } // dir loop
          
          double delt_tmp = d_CFL *vol/(sumSwept_Vol + d_SMALL_NUM);
          delt_CFL = std::min(delt_CFL, delt_tmp);
        }  // iter loop
//      cout << " Conservative delT based on swept volumes "<< delt_CFL<<endl;
      }  

      
      //__________________________________
      // stability constraint due to heat conduction
      //  I C E  O N L Y
      if (ice_matl) {
        double thermalCond = ice_matl->getThermalConductivity();
        if (thermalCond !=0) {
          double cv    = ice_matl->getSpecificHeat();
          double gamma = ice_matl->getGamma();
          double cp = cv * gamma;

          for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            double inv_thermalDiffusivity = cp/(sp_vol_CC[c] * thermalCond);
            double A =  d_CFL * 0.5 * inv_sum_invDelx_sqr * inv_thermalDiffusivity;
            delt_cond = std::min(A, delt_cond);
          }
        }  //
      }  // ice_matl
    }  // matl loop   
//    cout << "delT based on conduction "<< delt_cond<<endl;

    delt = std::min(delt_CFL, delt_cond);
    delt = std::min(delt, d_initialDt);
    d_initialDt = 10000.0;

    TIME += delt;

    //__________________________________
    //  Bullet proofing
    if(delt < 1e-20) {  
      string warn = " E R R O R \n ICE::ComputeStableTimestep: delT < 1e-20";
      throw InvalidValue(warn);
    }
    new_dw->put(delt_vartype(delt), lb->delTLabel);
  }  // patch loop
  //  update when you should dump debugging data. 
  d_dbgNextDumpTime = d_dbgOldTime + d_dbgOutputInterval;
}

/* --------------------------------------------------------------------- 
 Function~  ICE::actuallyInitialize--
 Purpose~  Initialize CC variables and the pressure  
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
    cout_norm<< "patch low and high index: "<<patch->getID()<<
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
    CCVariable<double>    press_CC, imp_initialGuess;
    StaticArray<double>   cv(numMatls);
    new_dw->allocateAndPut(press_CC,         lb->press_CCLabel, 0,patch);
    new_dw->allocateAndPut(imp_initialGuess, lb->imp_delPLabel, 0,patch);
    press_CC.initialize(0.0);
    imp_initialGuess.initialize(1.0); 

  //__________________________________
  // Note:
  // The press_CC isn't material dependent even though
  // we loop over numMatls below. This is done so we don't need additional
  // machinery to grab the pressure inside a geom_object
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();
      new_dw->allocateAndPut(rho_micro[m],  lb->rho_micro_CCLabel, indx,patch); 
      new_dw->allocateAndPut(sp_vol_CC[m],  lb->sp_vol_CCLabel,    indx,patch); 
      new_dw->allocateAndPut(rho_CC[m],     lb->rho_CCLabel,       indx,patch); 
      new_dw->allocateAndPut(Temp_CC[m],    lb->temp_CCLabel,      indx,patch); 
      new_dw->allocateAndPut(speedSound[m], lb->speedSound_CCLabel,indx,patch); 
      new_dw->allocateAndPut(vol_frac_CC[m],lb->vol_frac_CCLabel,  indx,patch); 
      new_dw->allocateAndPut(vel_CC[m],     lb->vel_CCLabel,       indx,patch); 
    }
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      ice_matl->initializeCells(rho_micro[m],  rho_CC[m],
                                Temp_CC[m],   speedSound[m], 
                                vol_frac_CC[m], vel_CC[m], 
                                press_CC,  numALLMatls,    patch, new_dw);

      cv[m] = ice_matl->getSpecificHeat();
      setBC(press_CC,   rho_micro[SURROUND_MAT], "rho_micro","Pressure", 
                                              patch, d_sharedState, 0, new_dw);
      setBC(rho_CC[m],        "Density",      patch, d_sharedState, indx);
      setBC(rho_micro[m],     "Density",      patch, d_sharedState, indx);
      setBC(Temp_CC[m],       "Temperature",  patch, d_sharedState, indx);
      setBC(speedSound[m],    "zeroNeumann",  patch, d_sharedState, indx); 
      setBC(vel_CC[m],        "Velocity",     patch, indx); 

      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[m][c] = 1.0/rho_micro[m][c];
        vol_frac_CC[m][c] = rho_CC[m][c]*sp_vol_CC[m][c];  // needed for LODI BCs
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
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      ostringstream warn;
      if( !areAllValuesPositive(press_CC, neg_cell) ) {
        warn<<"ERROR ICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " press_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      if( !areAllValuesPositive(rho_CC[m], neg_cell) ) {
        warn<<"ERROR ICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " rho_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      if( !areAllValuesPositive(Temp_CC[m], neg_cell) ) {
        warn<<"ERROR ICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " Temp_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
      if( !areAllValuesPositive(sp_vol_CC[m], neg_cell) ) {
        warn<<"ERROR ICE::actuallyInitialize, mat "<<indx<< " cell "
            <<neg_cell << " sp_vol_CC is negative\n";
        throw ProblemSetupException(warn.str() );
      }
    }   // numMatls loop 

    if (switchDebugInitialize){     
      ostringstream desc;
      desc << "Initialization_patch_"<< patch->getID();
      printData(0, patch, 1, "Initialization", "press_CC", press_CC);         
      for (int m = 0; m < numMatls; m++ ) { 
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();      
        desc << "Initialization_Mat_" << indx << "_patch_"<< patch->getID();
        printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC[m]);
        printData(indx, patch,   1, desc.str(), "rho_micro_CC",rho_micro[m]);
        printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC[m]);
        printData(indx, patch,   1, desc.str(), "Temp_CC",     Temp_CC[m]);
        printData(indx, patch,   1, desc.str(), "vol_frac_CC", vol_frac_CC[m]);
        printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC[m]);;
      }   
    }
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
    double    sum=0., tmp;

    int       numMatls = d_sharedState->getNumICEMatls();
    static int n_passes;                  
    n_passes ++; 

    StaticArray<double> press_eos(numMatls);
    StaticArray<double> dp_drho(numMatls),dp_de(numMatls);
    StaticArray<double> kappa(numMatls);
    StaticArray<CCVariable<double> > vol_frac(numMatls);
    StaticArray<CCVariable<double> > rho_micro(numMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numMatls); 
    StaticArray<CCVariable<double> > speedSound(numMatls);
    StaticArray<CCVariable<double> > speedSound_new(numMatls);
    StaticArray<CCVariable<double> > f_theta(numMatls); 
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls); 
   
    CCVariable<int> n_iters_equil_press;
    constCCVariable<double> press;
    CCVariable<double> press_new, press_copy;
    StaticArray<double> cv(numMatls), gamma(numMatls);
    Ghost::GhostType  gn = Ghost::None;
    
    //__________________________________
    //  Implicit press needs two copies of press 
    old_dw->get(press,                lb->press_CCLabel, 0,patch,gn, 0); 
    new_dw->allocateAndPut(press_new, lb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(press_copy,lb->press_CCLabel,       0,patch);
        
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],      lb->temp_CCLabel,  indx,patch, gn,0);
      old_dw->get(rho_CC[m],    lb->rho_CCLabel,   indx,patch, gn,0);
      old_dw->get(sp_vol_CC[m], lb->sp_vol_CCLabel,indx,patch, gn,0);
      
      new_dw->allocateTemporary(rho_micro[m],  patch);
      new_dw->allocateAndPut(vol_frac[m],   lb->vol_frac_CCLabel,indx, patch);  
      new_dw->allocateAndPut(rho_CC_new[m], lb->rho_CCLabel,     indx, patch);  
      new_dw->allocateAndPut(sp_vol_new[m], lb->sp_vol_CCLabel,  indx, patch); 
      new_dw->allocateAndPut(f_theta[m],    lb->f_theta_CCLabel, indx, patch);  
      new_dw->allocateAndPut(speedSound_new[m], lb->speedSound_CCLabel,
                                                                 indx, patch);
      cv[m] = matl->getSpecificHeat();
      gamma[m] = matl->getGamma();
    }

    press_new.copyData(press);
    //__________________________________
    // Compute rho_micro, speedSound, and volfrac
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
/*`==========TESTING==========*/
// This might be wrong.  Try 1/sp_vol -- Todd 11/22
#if 1
        rho_micro[m][c] = 
         ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[m],cv[m],
                                        Temp[m][c]); 
#endif 
#if 0
        rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
#endif
/*==========TESTING==========`*/

        ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m],
                                         cv[m], Temp[m][c],
                                         press_eos[m], dp_drho[m], dp_de[m]);

        double div = 1./rho_micro[m][c];
        tmp = dp_drho[m] + dp_de[m] * press_eos[m] * div * div;
        speedSound_new[m][c] = sqrt(tmp);
        vol_frac[m][c] = rho_CC[m][c] * div;
      }
    }

   //---- P R I N T   D A T A ------  
    if (switchDebug_EQ_RF_press) {
    
      new_dw->allocateTemporary(n_iters_equil_press,  patch);
#if 1
      ostringstream desc,desc1;
      desc1 << "TOP_equilibration_patch_" << patch->getID();
      printData( 0, patch, 1, desc.str(), "Press_CC_top", press);
     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       desc << "TOP_equilibration_Mat_" << indx << "_patch_"<<patch->getID();
       printData(indx, patch, 1, desc.str(), "rho_CC",       rho_CC[m]);    
       printData(indx, patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);  
       printData(indx, patch, 0, desc.str(), "speedSound",   speedSound_new[m]);
       printData(indx, patch, 1, desc.str(), "Temp_CC",      Temp[m]);       
       printData(indx, patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);   
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
       //cerr << "A=" << A << ", vol_frac...=" << vol_frac_not_close_packed << ", B=" << B << ", C=" << C << '\n';

       press_new[c] += delPress;
       //cerr << "press_new=" << press_new[c] << ", delPress=" << delPress << '\n';

       //__________________________________
       // backout rho_micro_CC at this new pressure
       for (int m = 0; m < numMatls; m++) {
         ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
         rho_micro[m][c] = 
           ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[m],
                                               cv[m],Temp[m][c]);
        //cerr << "rho_micro=" << rho_micro[m][c] << ", press=" << press_new[c] << ", gamma=" << gamma[m] << ", cv=" << cv[m] << ", temp=" << Temp[m][c] << '\n';

        double div = 1./rho_micro[m][c];
       //__________________________________
       // - compute the updated volume fractions
        vol_frac[m][c]   = rho_CC[m][c]*div;
       //cerr << "volfrac=" << vol_frac[m][c] << ", rho=" << rho_CC[m][c] << ", div=" << div << '\n'; 

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
       //cerr << "cell: " << *iter << ", sum=" << sum << '\n';
       if (fabs(sum-1.0) < convergence_crit)
         converged = true;

      }   // end of converged

      test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      if(test_max_iter == d_max_iter_equilibration) {
	throw MaxIteration(c,count,n_passes,"MaxIterations reached");
      }

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
    // compute sp_vol_CC
    for (int m = 0; m < numMatls; m++)   {
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c]; 
      }
    }
    //__________________________________
    // carry rho_cc forward 
    // MPMICE computes rho_CC_new
    // therefore need the machinery here
    for (int m = 0; m < numMatls; m++)   {
      rho_CC_new[m].copyData(rho_CC[m]);
    }

    //__________________________________
    //  compute f_theta  
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      double sumVolFrac_kappa = 0.0;
      for (int m = 0; m < numMatls; m++) {
        kappa[m] = sp_vol_new[m][c]/(speedSound_new[m][c]*speedSound_new[m][c]);
        sumVolFrac_kappa += vol_frac[m][c]*kappa[m];
      }
      for (int m = 0; m < numMatls; m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m]/sumVolFrac_kappa;
      }
    }
    //__________________________________
    // update Boundary conditions
    // make copy of press for implicit calc.   
 /*`==========TESTING==========*/
#ifndef LODI_BCS
    setBC(press_new,   rho_micro[SURROUND_MAT],
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw);
#else 
    setBCPress_LODI( press_new, rho_micro, Temp, f_theta,
                     "rho_micro", "Pressure", patch ,d_sharedState, 0, new_dw); 
#endif
/*==========TESTING==========`*/
    press_copy.copyData(press_new);
   //---- P R I N T   D A T A ------   
    if (switchDebug_EQ_RF_press) {
      ostringstream desc;
      desc << "BOT_equilibration_patch_" << patch->getID();
      printData( 0, patch, 1, desc.str(), "Press_CC_equil", press_new);

     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream desc;
       desc << "BOT_equilibration_Mat_"<< indx << "_patch_"<< patch->getID();
       printData( indx, patch, 1, desc.str(), "rho_CC",       rho_CC[m]);
       printData( indx, patch, 1, desc.str(), "sp_vol_CC",    sp_vol_new[m]); 
       printData( indx, patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);
       printData( indx, patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);
     //printData( indx, patch, 1, desc.str(), "iterations",   n_iters_equil_press);
       
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
    IntVector R = *it;
    IntVector L = R + adj_offset; 

    double rho_FC = rho_CC[L] + rho_CC[R];
    ASSERT(rho_FC > 0.0);
    //__________________________________
    // interpolation to the face
    double term1 = (rho_CC[L] * vel_CC[L][dir] +
                    rho_CC[R] * vel_CC[R][dir])/(rho_FC);            
    //__________________________________
    // pressure term           
    double sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                             (sp_vol_CC[L] + sp_vol_CC[R]); 
    
    double term2 = delT * sp_vol_brack * (press_CC[R] - press_CC[L])/dx;
    
    //__________________________________
    // gravity term
    double term3 =  delT * gravity;
    
    vel_FC[R] = term1- term2 + term3;
  } 
}


/* ---------------------------------------------------------------------
 Function~  ICE::computeTempFace--
 Purpose~   compute the face centered Temperatures.  This is used by
 the HE combustion model
_____________________________________________________________________*/
template<class T> void ICE::computeTempFace(CellIterator it,
                                            IntVector adj_offset,
                                            constCCVariable<double>& rho_CC,
                                            constCCVariable<double>& Temp_CC,
                                            T& Temp_FC)
{
  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset; 

    double rho_FC = rho_CC[L] + rho_CC[R];
    ASSERT(rho_FC > 0.0);
    //__________________________________
    // interpolation to the face
    //  based on continuity of heat flux
    double term1 = (rho_CC[L] * Temp_CC[L] + rho_CC[R] * Temp_CC[R])/(rho_FC);            
    Temp_FC[R] = term1;
  } 
}

//______________________________________________________________________
//                       
void ICE::computeTempFC(const ProcessorGroup*,  
                        const PatchSubset* patches,                     
                        const MaterialSubset* /*matls*/,                
                        DataWarehouse* old_dw,                          
                        DataWarehouse* new_dw)                          
{
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing compute_FC_Temp on patch " 
              << patch->getID() << "\t\t\t ICE" << endl;
            
    int numMatls = d_sharedState->getNumMatls();
    Ghost::GhostType  gac = Ghost::AroundCells; 
    
    // Compute the face centered Temperatures
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      constCCVariable<double> rho_CC, Temp_CC;
      
      if(ice_matl){
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        old_dw->get(Temp_CC,lb->temp_CCLabel,indx, patch, gac, 1); 
      } else {
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        new_dw->get(Temp_CC,lb->temp_CCLabel,indx, patch, gac, 1);
      }
      //---- P R I N T   D A T A ------
  #if 0
      if (switchDebug_Temp_FC ) {
        ostringstream desc;
        desc << "TOP_computeTempFC_Mat_" << indx << "_patch_"<< patch->getID();
        printData(indx, patch, 1, desc.str(), "rho_CC",      rho_CC);
        printData(indx, patch, 1, desc.str(), "Temp_CC",    Temp_CC);
      }
  #endif
      SFCXVariable<double> TempX_FC;
      SFCYVariable<double> TempY_FC;
      SFCZVariable<double> TempZ_FC; 
      new_dw->allocateAndPut(TempX_FC,lb->TempX_FCLabel,indx, patch);   
      new_dw->allocateAndPut(TempY_FC,lb->TempY_FCLabel,indx, patch);   
      new_dw->allocateAndPut(TempZ_FC,lb->TempZ_FCLabel,indx, patch);   
      
      IntVector lowIndex(patch->getSFCXLowIndex());
      TempX_FC.initialize(0.0,lowIndex,patch->getSFCXHighIndex()); 
      TempY_FC.initialize(0.0,lowIndex,patch->getSFCYHighIndex()); 
      TempZ_FC.initialize(0.0,lowIndex,patch->getSFCZHighIndex());
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

      int offset=0;    // 0=Compute all faces in computational domain             
                       // 1=Skip the faces at the border between interior and gc

      //__________________________________
      //  Compute the temperature on each face     
      //  Currently on used by HEChemistry 
      if ( d_massExchange == true || d_models.size() > 0) {        
        computeTempFace<SFCXVariable<double> >(patch->getSFCXIterator(offset),
                                     adj_offset[0], rho_CC,Temp_CC, TempX_FC);

        computeTempFace<SFCYVariable<double> >(patch->getSFCYIterator(offset),
                                     adj_offset[1], rho_CC,Temp_CC, TempY_FC);

        computeTempFace<SFCZVariable<double> >(patch->getSFCZIterator(offset),
                                     adj_offset[2], rho_CC,Temp_CC, TempZ_FC);
      }

      //---- P R I N T   D A T A ------ 
      if (switchDebug_Temp_FC ) {
        ostringstream desc;
        desc << "BOT_computeTempFC_Mat_" << indx << "_patch_"<< patch->getID(); 
        printData_FC( indx, patch,1, desc.str(), "TempX_FC", TempX_FC);
        printData_FC( indx, patch,1, desc.str(), "TempY_FC", TempY_FC);
        printData_FC( indx, patch,1, desc.str(), "TempZ_FC", TempZ_FC);
      }
    } // matls loop
  }  // patch loop
}                       
//______________________________________________________________________
//                       
void ICE::computeVel_FC(const ProcessorGroup*,  
                             const PatchSubset* patches,                
                             const MaterialSubset* /*matls*/,           
                             DataWarehouse* old_dw,                     
                             DataWarehouse* new_dw,
                             bool recursion)                     
{
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing computeVel_FC on patch " 
              << patch->getID() << "\t\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();
    
    
    Vector dx      = patch->dCell();
    Vector gravity = d_sharedState->getGravity();
    
    constCCVariable<double> press_CC;
    Ghost::GhostType  gac = Ghost::AroundCells; 
    //__________________________________
    //  Implicit
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;

    if(recursion) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
      old_dw->get(press_CC,lb->press_CCLabel, 0, patch,gac, 1);
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
      new_dw->get(press_CC,lb->press_CCLabel, 0, patch,gac, 1);
    }
     
    delt_vartype delT;
    pOldDW->get(delT, d_sharedState->get_delt_label());   
     
    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      constCCVariable<double> rho_CC, sp_vol_CC;
      constCCVariable<Vector> vel_CC;
      if(ice_matl){
        pNewDW->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        pOldDW->get(vel_CC, lb->vel_CCLabel, indx, patch, gac, 1); 
      } else {
        pNewDW->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        pNewDW->get(vel_CC, lb->vel_CCLabel, indx, patch, gac, 1);
      }              
      pNewDW->get(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch, gac, 1);
              
      //---- P R I N T   D A T A ------
  #if 1
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc << "TOP_computeVel_FC_Mat_" << indx << "_patch_"<< patch->getID();
        printData(indx, patch, 1, desc.str(), "press_CC",    press_CC); 
        printData(indx, patch, 1, desc.str(), "rho_CC",      rho_CC);
        printData(indx, patch, 1, desc.str(), "sp_vol_CC",   sp_vol_CC);
        printVector(indx,patch,1, desc.str(), "vel_CC",  0,  vel_CC);
      }
  #endif
      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;

      new_dw->allocateAndPut(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(wvel_FC, lb->wvel_FCLabel, indx, patch);   
      
      IntVector lowIndex(patch->getSFCXLowIndex());
      uvel_FC.initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      vvel_FC.initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      wvel_FC.initialize(0.0, lowIndex,patch->getSFCZHighIndex());
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

      int offset=0;    // 0=Compute all faces in computational domain             
                       // 1=Skip the faces at the border between interior and gc
      //__________________________________
      //  Compute vel_FC for each face
      computeVelFace<SFCXVariable<double> >(0,patch->getSFCXIterator(offset),
                                      adj_offset[0],dx[0],delT,gravity[0],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       uvel_FC);

      computeVelFace<SFCYVariable<double> >(1,patch->getSFCYIterator(offset),
                                      adj_offset[1],dx[1],delT,gravity[1],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       vvel_FC);

      computeVelFace<SFCZVariable<double> >(2,patch->getSFCZIterator(offset),
                                      adj_offset[2],dx[2],delT,gravity[2],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       wvel_FC);

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()

      //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc <<"BOT_computeVel_FC_Mat_" << indx << "_patch_"<< patch->getID();
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",  uvel_FC);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",  vvel_FC);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",  wvel_FC); 
      }
    } // matls loop
  }  // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::add_vel_FC_exchange--
 Purpose~   Add the exchange contribution to vel_FC and compute 
            sp_vol_FC for implicit Pressure solve
_____________________________________________________________________*/
template<class V, class T> 
    void ICE::add_vel_FC_exchange( CellIterator iter,
                                       IntVector adj_offset,
                                       int numMatls,
                                       FastMatrix& K,
                                       double delT,
                                       StaticArray<constCCVariable<double> >& vol_frac_CC,
                                       StaticArray<constCCVariable<double> >& sp_vol_CC,
                                       V& vel_FC,
                                       T& sp_vol_FC,
                                       T& vel_FCME)        
                                       
{
  vector<double> b(numMatls), b_sp_vol(numMatls), X(numMatls);
  FastMatrix beta(numMatls, numMatls), a(numMatls, numMatls);
  FastMatrix a_inverse(numMatls, numMatls);
  double tmp, sp_vol_brack;
  
  for(;!iter.done(); iter++){
    IntVector c = *iter;
    IntVector adj = c + adj_offset; 

    //__________________________________
    //   Compute beta and off diagonal term of
    //   Matrix A, this includes b[m][m].
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    
    for(int m = 0; m < numMatls; m++)  {
      for(int n = 0; n < numMatls; n++)  {
        tmp = (vol_frac_CC[n][adj] + vol_frac_CC[n][c]) * K(n,m);

        sp_vol_brack = 2.0 * (sp_vol_CC[m][adj] * sp_vol_CC[m][c])/
                             (sp_vol_CC[m][adj] + sp_vol_CC[m][c]);

        beta(m,n) = 0.5 * sp_vol_brack * delT * tmp;
        a(m,n)    = -beta(m,n);
        b_sp_vol[m] = sp_vol_brack;
      }
    }
    //__________________________________
    // - Form diagonal terms of Matrix (A)
    for(int m = 0; m < numMatls; m++) {
      a(m,m) = 1.;
      for(int n = 0; n < numMatls; n++) {
        a(m,m) +=  beta(m,n);
      }
    }
    //__________________________________
    //  - Form RHS (b) 
    for(int m = 0; m < numMatls; m++)  {
      b[m] = 0.0;
      for(int n = 0; n < numMatls; n++)  {
        b[m] += beta(m,n) * (vel_FC[n][c] - vel_FC[m][c]);
      }
    }
    //__________________________________
    //  - solve and backout velocities
    a_inverse.destructiveInvert(a);
    a_inverse.multiply(b,X);
    
//  a.destructiveSolve(b,X);               // old style
    for(int m = 0; m < numMatls; m++) {
      vel_FCME[m][c] = vel_FC[m][c] + X[m];
    }
    
    //__________________________________
    //  For implicit solve we need sp_vol_FC
    a_inverse.multiply(b_sp_vol,X);
    for(int m = 0; m < numMatls; m++) {    // only needed by implicit Pressure
      sp_vol_FC[m][c] = X[m];
    }
  }  // iterator
}

/*---------------------------------------------------------------------
 Function~  addExchangeContributionToFCVel--
 Purpose~
   This function adds the momentum exchange contribution to the 
   existing face-centered velocities

            
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
 
 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 ---------------------------------------------------------------------  */
void ICE::addExchangeContributionToFCVel(const ProcessorGroup*,  
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw,
                                         const bool recursion)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Add_exchange_contribution_to_FC_vel on patch " <<
      patch->getID() << "\t ICE" << endl;
 
    // change the definition of parent(old/new)DW
    // when implicit
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;
    if(recursion) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    } 

    int numMatls = d_sharedState->getNumMatls();
    delt_vartype delT;
    pOldDW->get(delT, d_sharedState->get_delt_label());

    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    StaticArray<constCCVariable<double> > vol_frac_CC(numMatls);
    StaticArray<constSFCXVariable<double> > uvel_FC(numMatls);
    StaticArray<constSFCYVariable<double> > vvel_FC(numMatls);
    StaticArray<constSFCZVariable<double> > wvel_FC(numMatls);

    StaticArray<SFCXVariable<double> >uvel_FCME(numMatls),sp_vol_XFC(numMatls);  
    StaticArray<SFCYVariable<double> >vvel_FCME(numMatls),sp_vol_YFC(numMatls);  
    StaticArray<SFCZVariable<double> >wvel_FCME(numMatls),sp_vol_ZFC(numMatls);
    
    // lowIndex is the same for all vel_FC
    IntVector lowIndex(patch->getSFCXLowIndex()); 
    
    // Extract the momentum exchange coefficients
    FastMatrix K(numMatls, numMatls), junk(numMatls, numMatls);

    K.zero();
    getExchangeCoefficients( K, junk);
    Ghost::GhostType  gac = Ghost::AroundCells;    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      pNewDW->get(sp_vol_CC[m],    lb->sp_vol_CCLabel,  indx, patch,gac, 1);
      pNewDW->get(vol_frac_CC[m],  lb->vol_frac_CCLabel,indx, patch,gac, 1);  
      new_dw->get(uvel_FC[m],      lb->uvel_FCLabel,    indx, patch,gac, 2);  
      new_dw->get(vvel_FC[m],      lb->vvel_FCLabel,    indx, patch,gac, 2);  
      new_dw->get(wvel_FC[m],      lb->wvel_FCLabel,    indx, patch,gac, 2);  

      new_dw->allocateAndPut(uvel_FCME[m], lb->uvel_FCMELabel, indx, patch);
      new_dw->allocateAndPut(vvel_FCME[m], lb->vvel_FCMELabel, indx, patch);
      new_dw->allocateAndPut(wvel_FCME[m], lb->wvel_FCMELabel, indx, patch);
      
      new_dw->allocateAndPut(sp_vol_XFC[m],lb->sp_volX_FCLabel,indx, patch);   
      new_dw->allocateAndPut(sp_vol_YFC[m],lb->sp_volY_FCLabel,indx, patch);   
      new_dw->allocateAndPut(sp_vol_ZFC[m],lb->sp_volZ_FCLabel,indx, patch); 

      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getSFCXHighIndex());
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getSFCYHighIndex());
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getSFCZHighIndex());
      
      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getSFCXHighIndex());
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getSFCYHighIndex());
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getSFCZHighIndex());     
    }   
    
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
    int offset=0;   // 0=Compute all faces in computational domain
                    // 1=Skip the faces at the border between interior and gc
                                   
    //__________________________________
    //  tack on exchange contribution
    add_vel_FC_exchange<StaticArray<constSFCXVariable<double> >,
                        StaticArray<     SFCXVariable<double> > >
                        (patch->getSFCXIterator(offset), 
                        adj_offset[0],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        uvel_FC,        sp_vol_XFC,  uvel_FCME);
                        
    add_vel_FC_exchange<StaticArray<constSFCYVariable<double> >,
                        StaticArray<     SFCYVariable<double> > >
                        (patch->getSFCYIterator(offset), 
                        adj_offset[1],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        vvel_FC,        sp_vol_YFC,  vvel_FCME);
                        
    add_vel_FC_exchange<StaticArray<constSFCZVariable<double> >,
                        StaticArray<     SFCZVariable<double> > >
                        (patch->getSFCZIterator(offset), 
                        adj_offset[2],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        wvel_FC,        sp_vol_ZFC,  wvel_FCME);

    //_________________________________
    //  For LODI, How to do ?	                             			
    //    Boundary Conditons for Dirichlet and Neumann ONLY
    //    For LODI they are computed above.
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
       printData(    indx, patch,1, desc.str(), "sp_vol_CC", sp_vol_CC[m]);   
       printData_FC( indx, patch,1, desc.str(), "uvel_FCME", uvel_FCME[m]);
       printData_FC( indx, patch,1, desc.str(), "vvel_FCME", vvel_FCME[m]);
       printData_FC( indx, patch,1, desc.str(), "wvel_FCME", wvel_FCME[m]);
      }
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
    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;
    CCVariable<double> press_CC;
    CCVariable<double> term1, term2, term3;
    constCCVariable<double> pressure;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
   
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->getModifiable(press_CC,      lb->press_CCLabel,     0,patch);
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(press_CC,     lb->press_CCLabel,     0, patch);
    new_dw->allocateAndPut(term2,        lb->term2Label,        0, patch);
    new_dw->allocateAndPut(term3,        lb->term3Label,        0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch); 

    new_dw->allocateTemporary(q_advected, patch);
    new_dw->allocateTemporary(term1,      patch);
    new_dw->get(pressure,  lb->press_equil_CCLabel,0,patch,gn,0);

    term1.initialize(0.);
    term2.initialize(0.);
    term3.initialize(0.);
    sum_rho_CC.initialize(0.0); 
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constCCVariable<double> speedSound;
      constCCVariable<double> vol_frac;
      constCCVariable<double> rho_CC;

      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;

      new_dw->get(uvel_FC,     lb->uvel_FCMELabel,     indx,patch,gac, 2);   
      new_dw->get(vvel_FC,     lb->vvel_FCMELabel,     indx,patch,gac, 2);   
      new_dw->get(wvel_FC,     lb->wvel_FCMELabel,     indx,patch,gac, 2);   
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,   indx,patch,gac, 2);   
      new_dw->get(rho_CC,      lb->rho_CCLabel,        indx,patch,gn,0);
      new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel,     indx,patch,gn,0);
      new_dw->get(speedSound,  lb->speedSound_CCLabel, indx,patch,gn,0);
          
      //---- P R I N T   D A T A ------  
      if (switchDebug_explicit_press ) {
        ostringstream desc;
        desc<<"middle_explicit_Pressure_Mat_"<<indx<<"_patch_"<<patch->getID();
        printData(    indx, patch,1, desc.str(), "vol_frac",   vol_frac);
        printData(    indx, patch,1, desc.str(), "speedSound", speedSound);
        printData(    indx, patch,1, desc.str(), "sp_vol_CC",  sp_vol_CC[m]);
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",    uvel_FC);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",    vvel_FC);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",    wvel_FC);
      }
                 
      //__________________________________
      // Advection preprocessing
      // - divide vol_frac_cc/vol
      bool bulletProof_test=true;
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                                    bulletProof_test); 
      //__________________________________
      //   advect vol_frac
      advector->advectQ(vol_frac, patch, q_advected, new_dw);  
      
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        term2[c] -= q_advected[c]; 

      }  //iter loop 
      
      //__________________________________
      //   NO Models   MODEL REMOVE
      if(d_models.size() == 0){
        constCCVariable<double> created_vol; 
        constCCVariable<double> burnedMass;
        new_dw->get(burnedMass,  lb->burnedMass_CCLabel, indx,patch,gn,0);
        new_dw->get(created_vol, lb->created_vol_CCLabel,indx,patch,gn,0); 
                
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
          term1[c] += burnedMass[c] * (sp_vol_CC[m][c]/vol);
           
          term3[c] += (vol_frac[c] + created_vol[c]/vol)*sp_vol_CC[m][c]/
                                                 (speedSound[c]*speedSound[c]);
        }  //iter loop 
      }  // models
      
      //__________________________________
      //   Contributions from models
      if(d_models.size() > 0){
        constCCVariable<double> modelMass_src, modelVol_src;
        new_dw->get(modelMass_src, lb->modelMass_srcLabel, indx, patch, gn, 0);
        new_dw->get(modelVol_src,  lb->modelVol_srcLabel,  indx, patch, gn, 0);
                
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         term1[c] += modelMass_src[c] * (sp_vol_CC[m][c]/vol);
         term3[c] += (vol_frac[c] + modelVol_src[c]/vol)*sp_vol_CC[m][c]/
                                             (speedSound[c]*speedSound[c]);
        }
      }
         
      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      } 
    }  //matl loop
    delete advector;
    press_CC.initialize(0.);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    =  term1[c]/term3[c];
      delP_Dilatate[c] = -term2[c]/term3[c];
      press_CC[c]      = pressure[c] + delP_MassX[c] 
                       + delP_Dilatate[c];
    }
    //____ B U L L E T   P R O O F I N G----
    // This was done to help robustify the equilibration
    // pressure calculation in MPMICE.  Also, in rate form, negative
    // mean pressures are allowed.
    if(d_EqForm){
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
        IntVector c = *iter;
        press_CC[c] = max(1.0e-12, press_CC[c]);  
      }
    }
 /*`==========TESTING==========*/
#ifndef LODI_BCS
    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);
#else 

    //__________________________________
    // TO CLEAN THIS UP FIGURE OUT A WAY TO TEMPLATE SETPRESSLODI
    // FOR EITHER StaticArray<constCCVariable<double> or 
    //           StaticArray<CCVariable<double>
    StaticArray<constCCVariable<double> > Temp_CC(numMatls);
    StaticArray<constCCVariable<double> > f_theta_tmp(numMatls);  
    StaticArray<CCVariable<double> > f_theta(numMatls);
    StaticArray<CCVariable<double> > sp_vol_tmp(numMatls);
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

 
      if(ice_matl){                // I C E
        old_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
        new_dw->get(f_theta_tmp[m], lb->f_theta_CCLabel, indx,patch,gn,0); 
      }
      if(mpm_matl){                // M P M
        new_dw->get(Temp_CC[m],     lb->temp_CCLabel,    indx,patch,gn,0);
        new_dw->get(f_theta_tmp[m], lb->f_theta_CCLabel, indx,patch,gn,0);
      }
      new_dw->allocateTemporary(f_theta[m],     patch);
      new_dw->allocateTemporary(sp_vol_tmp[m],  patch);

      f_theta[m].copyData(f_theta_tmp[m]);
      sp_vol_tmp[m].copyData(sp_vol_CC[m]);
    }
    setBCPress_LODI( press_CC, sp_vol_tmp, Temp_CC, f_theta,
                    "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw); 
#endif
/*==========TESTING==========`*/

                             
   //---- P R I N T   D A T A ------  
    if (switchDebug_explicit_press) {
      ostringstream desc;
      desc << "BOT_explicit_Pressure_patch_" << patch->getID();
//    printData( 0, patch, 1,desc.str(), "term1",         term1);
      printData( 0, patch, 1,desc.str(), "term2",         term2);
      printData( 0, patch, 1,desc.str(), "term3",         term3); 
      printData( 0, patch, 1,desc.str(), "delP_Dilatate", delP_Dilatate);
      printData( 0, patch, 1,desc.str(), "delP_MassX",    delP_MassX);
      printData( 0, patch, 1,desc.str(), "Press_CC",      press_CC);
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
template <class T> void ICE::computePressFace(CellIterator iter, 
                                              IntVector adj_offset,
                                              constCCVariable<double>& sum_rho,
                                              constCCVariable<double>& press_CC,
                                              T& press_FC)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;
    IntVector L = R + adj_offset; 

    press_FC[R] = (press_CC[R] * sum_rho[L] + press_CC[L] * sum_rho[R])/
      (sum_rho[R] + sum_rho[L]);
  }
}
 
//______________________________________________________________________
//
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
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    constCCVariable<double> press_CC;
    constCCVariable<double> sum_rho_CC;
    new_dw->get(press_CC,  lb->press_CCLabel,   0, patch, gac, 1);
    new_dw->get(sum_rho_CC,lb->sum_rho_CCLabel, 0, patch, gac, 1);
    
    SFCXVariable<double> pressX_FC;
    SFCYVariable<double> pressY_FC;
    SFCZVariable<double> pressZ_FC;
    new_dw->allocateAndPut(pressX_FC, lb->pressX_FCLabel, 0, patch);
    new_dw->allocateAndPut(pressY_FC, lb->pressY_FCLabel, 0, patch);
    new_dw->allocateAndPut(pressZ_FC, lb->pressZ_FCLabel, 0, patch);
    
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
         
    //__________________________________
    //  For each face compute the pressure
    computePressFace<SFCXVariable<double> >(patch->getSFCXIterator(),
                                       adj_offset[0], sum_rho_CC, press_CC,
                                       pressX_FC);

    computePressFace<SFCYVariable<double> >(patch->getSFCYIterator(),
                                       adj_offset[1], sum_rho_CC, press_CC,
                                       pressY_FC);

    computePressFace<SFCZVariable<double> >(patch->getSFCZIterator(),
                                       adj_offset[2], sum_rho_CC, press_CC,
                                       pressZ_FC); 
   //---- P R I N T   D A T A ------ 
    if (switchDebug_PressFC) {
      ostringstream desc;
      desc << "press_FC_patch_" <<patch->getID();
      printData_FC( 0, patch,0,desc.str(), "press_FC_RIGHT", pressX_FC);
      printData_FC( 0, patch,0,desc.str(), "press_FC_TOP",   pressY_FC);
      printData_FC( 0, patch,0,desc.str(), "press_FC_FRONT", pressZ_FC);
    }
  }  // patch loop
}


/* ---------------------------------------------------------------------
 Function~  ICE::massExchange--
 // MODEL REMOVE --entire task
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
      new_dw->allocateAndPut(burnedMass[m],  lb->burnedMass_CCLabel,  
                                                              indx,patch);
      new_dw->allocateAndPut(int_eng_comb[m],lb->int_eng_comb_CCLabel,
                                                              indx,patch);
      new_dw->allocateAndPut(created_vol[m], lb->created_vol_CCLabel, 
                                                              indx,patch);
      new_dw->allocateAndPut(mom_comb[m],    lb->mom_comb_CCLabel,    
                                                              indx,patch);
      burnedMass[m].initialize(0.0);
      int_eng_comb[m].initialize(0.0); 
      created_vol[m].initialize(0.0);
      mom_comb[m].initialize(Vector(0.0));
      cv[m] = ice_matl->getSpecificHeat();
    }
    //__________________________________
    // Do the exchange if there is a reactant (react >= 0)
    // and the switch is on.
    if(d_massExchange && (react >= 0)){       
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        double mass_hmx = rho_CC[react][c] * vol;
        if (mass_hmx > d_TINY_RHO)  {
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

#if 0    // turn off for quality control tests
    //---- P R I N T   D A T A ------ 
    for(int m = 0; m < numMatls; m++) {
      if (switchDebugSource_Sink) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc << "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printData(indx, patch, 0, desc.str(),"burnedMass",   burnedMass[m]);
        printData(indx, patch, 0, desc.str(),"int_eng_comb", int_eng_comb[m]);
        printData(indx, patch, 0, desc.str(),"mom_comb",   mom_comb[m]);
      }

    }
#endif
  }   // patch loop
}

/* ---------------------------------------------------------------------
 Function~  ICE::zeroModelMassExchange
 Purpose~   This function initializes the mass exchange quantities to
            zero.  These quantities are subsequently modified by the
            models
 ---------------------------------------------------------------------  */
void ICE::zeroModelSources(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* /*old_dw*/,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double> mass_src, energy_src, vol_src;
      CCVariable<Vector> mom_src;
      
      new_dw->allocateAndPut(mass_src,   lb->modelMass_srcLabel,matl, patch);
      new_dw->allocateAndPut(energy_src, lb->modelEng_srcLabel, matl, patch);
      new_dw->allocateAndPut(mom_src,    lb->modelMom_srcLabel, matl, patch);
      new_dw->allocateAndPut(vol_src,    lb->modelVol_srcLabel, matl, patch);
            
      energy_src.initialize(0.0);
      mass_src.initialize(0.0);
      vol_src.initialize(0.0);
      mom_src.initialize(Vector(0.0, 0.0, 0.0));
      for(vector<TransportedVariable*>::iterator iter = d_modelSetup->tvars.begin();
	  iter != d_modelSetup->tvars.end(); iter++){
	TransportedVariable* tvar = *iter;
	if(tvar->src){
	  CCVariable<double> model_src;
	  new_dw->allocateAndPut(model_src, tvar->src, matl, patch);
	  model_src.initialize(0.0);
	}
      }
    }
  }
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
    double viscous_source,viscosity;
    double include_term;

    delt_vartype delT; 
    old_dw->get(delT, d_sharedState->get_delt_label());
 
    dx      = patch->dCell();
    gravity = d_sharedState->getGravity();
    vol     = dx.x() * dx.y() * dx.z();
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();
    constCCVariable<double>   rho_CC;
    constCCVariable<double>   sp_vol_CC;
    constCCVariable<Vector>   vel_CC;
    constCCVariable<double>   vol_frac;
    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;
    constSFCXVariable<double> press_diffX_FC;
    constSFCYVariable<double> press_diffY_FC;
    constSFCZVariable<double> press_diffZ_FC;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(pressX_FC,lb->pressX_FCLabel, 0, patch, gac, 1);
    new_dw->get(pressY_FC,lb->pressY_FCLabel, 0, patch, gac, 1);
    new_dw->get(pressZ_FC,lb->pressZ_FCLabel, 0, patch, gac, 1);
  //__________________________________
  //  Matl loop 
    for(int m = 0; m < numMatls; m++) {
      Material* matl        = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      indx = matl->getDWIndex();

      new_dw->get(rho_CC,  lb->rho_CCLabel,      indx,patch,gac,2);
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,patch,Ghost::None, 0);
      CCVariable<Vector>   mom_source, press_force;
      new_dw->allocateAndPut(mom_source,  lb->mom_source_CCLabel,  indx, patch);
      new_dw->allocateAndPut(press_force, lb->press_force_CCLabel, indx, patch); 
      press_force.initialize(Vector(0.,0.,0.));
      mom_source.initialize(Vector(0.,0.,0.));
      
      if(d_RateForm){        // R A T E  F O R M 
        new_dw->get(press_diffX_FC,lb->press_diffX_FCLabel,indx,patch,gac, 1);
        new_dw->get(press_diffY_FC,lb->press_diffY_FCLabel,indx,patch,gac, 1);
        new_dw->get(press_diffZ_FC,lb->press_diffZ_FCLabel,indx,patch,gac, 1);
      }
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
        old_dw->get(vel_CC,    lb->vel_CCLabel,     indx,patch,gac,2);
        new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,  indx,patch,gac,2);
        viscosity = ice_matl->getViscosity();
        if(viscosity != 0.0){  
          computeTauX(patch, rho_CC, sp_vol_CC, vel_CC,viscosity,dx, tau_X_FC);
          computeTauY(patch, rho_CC, sp_vol_CC, vel_CC,viscosity,dx, tau_Y_FC);
          computeTauZ(patch, rho_CC, sp_vol_CC, vel_CC,viscosity,dx, tau_Z_FC); 
        }
        include_term = 1.0;
        // This multiplies terms that are only included in the ice_matls
      }
      else{
        include_term = 0.0;
      }
      
      //__________________________________
      //  accumulate sources
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        mass = rho_CC[c] * vol;

        right    = c + IntVector(1,0,0);    left     = c + IntVector(0,0,0);
        top      = c + IntVector(0,1,0);    bottom   = c + IntVector(0,0,0);
        front    = c + IntVector(0,0,1);    back     = c + IntVector(0,0,0);

        //__________________________________
        //    X - M O M E N T U M 
        pressure_source = (pressX_FC[right]-pressX_FC[left]) * vol_frac[c];
        
        press_force[c][0] = -pressure_source * areaX; 
               
        viscous_source=(tau_X_FC[right].x() - tau_X_FC[left].x())  * areaX +
                       (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())* areaY +
                       (tau_Z_FC[front].x() - tau_Z_FC[back].x())  * areaZ;

        mom_source[c].x( (-pressure_source * areaX + 
                           viscous_source +
                           mass * gravity.x() * include_term) * delT ); 

        //__________________________________
        //    Y - M O M E N T U M
        pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c];
         
        press_force[c][1] = -pressure_source * areaY;
        
        viscous_source=(tau_X_FC[right].y() - tau_X_FC[left].y())  * areaX +
                       (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())* areaY +
                       (tau_Z_FC[front].y() - tau_Z_FC[back].y())  * areaZ;

        mom_source[c].y( (-pressure_source * areaY +
                           viscous_source +
                           mass * gravity.y() * include_term) * delT ); 
   
      //__________________________________
      //    Z - M O M E N T U M
        pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c];
        
        press_force[c][2] = -pressure_source * areaZ;
        
        viscous_source=(tau_X_FC[right].z() - tau_X_FC[left].z())  * areaX +
                       (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())* areaY +
                       (tau_Z_FC[front].z() - tau_Z_FC[back].z())  * areaZ;

        mom_source[c].z( (-pressure_source * areaZ +
                           viscous_source + 
                           mass * gravity.z() * include_term) * delT ); 
      }
      //__________________________________
      //  RATE FORM:   Tack on contribution
      //  due to grad press_diff_FC
      if(d_RateForm){
        double press_diff_source;
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          right    = c + IntVector(1,0,0);    left     = c + IntVector(0,0,0);
          top      = c + IntVector(0,1,0);    bottom   = c + IntVector(0,0,0);
          front    = c + IntVector(0,0,1);    back     = c + IntVector(0,0,0);

          //__________________________________
          //    X - M O M E N T U M 
          press_diff_source = (press_diffX_FC[right] - press_diffX_FC[left]);
          mom_source[c].x(mom_source[c].x() -
                        press_diff_source * areaX * include_term * delT);
          //__________________________________
          //    Y - M O M E N T U M 
          press_diff_source = (press_diffY_FC[top] - press_diffY_FC[bottom]);
          mom_source[c].y(mom_source[c].y() -
                        press_diff_source * areaY * include_term * delT);
          //__________________________________
          //    Z - M O M E N T U M 
          press_diff_source = (press_diffZ_FC[front] - press_diffZ_FC[back]);
          mom_source[c].z(mom_source[c].z() -
                        press_diff_source * areaZ * include_term * delT); 
        }
      }

      setBC(press_force, "set_if_sym_BC",patch, indx); 

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc << "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printVector(indx, patch, 1, desc.str(), "mom_source",  0, mom_source);
      //printVector(indx, patch, 1, desc.str(), "press_force", 0, press_force);        
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

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing accumulate_energy_source_sinks on patch " 
         << patch->getID() << "\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx = patch->dCell();
    double A, B, vol=dx.x()*dx.y()*dx.z();
    IntVector right, left, top, bottom, front, back;
    
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();
    
    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> speedSound;
    constCCVariable<double> vol_frac;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;
    constCCVariable<double> matl_press;
    constCCVariable<double> rho_CC;
    constCCVariable<double> Temp_CC;
        
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(press_CC,     lb->press_CCLabel,      0, patch,gn, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,gn, 0);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl); 

      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      new_dw->get(sp_vol_CC,   lb->sp_vol_CCLabel,    indx,patch,gac,1);
      new_dw->get(rho_CC,      lb->rho_CCLabel,       indx,patch,gac,1);
      new_dw->get(speedSound,  lb->speedSound_CCLabel,indx,patch,gn,0);
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,  indx,patch,gn,0);
       
      new_dw->allocateAndPut(int_eng_source, 
                               lb->int_eng_source_CCLabel,indx,patch);
      int_eng_source.initialize(0.0);
     
      //__________________________________
      //  Source due to conduction
      if(ice_matl){
        double thermalCond = ice_matl->getThermalConductivity();
        if(thermalCond != 0.0){ 
          old_dw->get(Temp_CC, lb->temp_CCLabel, indx,patch,gac,1);
          
          SFCXVariable<double> q_X_FC;
          SFCYVariable<double> q_Y_FC;
          SFCZVariable<double> q_Z_FC;
          
          computeQ_conduction_FC( new_dw, patch, 
                                  rho_CC,  sp_vol_CC, Temp_CC, thermalCond,
                                  q_X_FC, q_Y_FC, q_Z_FC);
          
          for(CellIterator iter = patch->getCellIterator(); !iter.done(); 
                                                                    iter++){
            IntVector c = *iter;
            right  = c + IntVector(1,0,0);    left   = c ;    
            top    = c + IntVector(0,1,0);    bottom = c ;    
            front  = c + IntVector(0,0,1);    back   = c ; 
            
            int_eng_source[c]=-((q_X_FC[right] - q_X_FC[left])  *areaX + 
                                (q_Y_FC[top]   - q_Y_FC[bottom])*areaY +
                                (q_Z_FC[front] - q_Z_FC[back])  *areaZ )*delT;
          }
        } 
      }
                                     
      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        A = vol * vol_frac[c] * press_CC[c] * sp_vol_CC[c];
        B = speedSound[c] * speedSound[c];
        int_eng_source[c] += (A/B) * delP_Dilatate[c];
      }

      //__________________________________
      //  User specified source/sink 
      double Time= dataArchiver->getCurrentTime();  
      if (  d_add_heat &&
            Time >= d_add_heat_t_start && 
            Time <= d_add_heat_t_final ) { 
        for (int i = 0; i<(int) d_add_heat_matls.size(); i++) {
          if(m == d_add_heat_matls[i] ){
             for(CellIterator iter = patch->getCellIterator();!iter.done();
                                                                      iter++){            
              IntVector c = *iter;
              if ( vol_frac[c] > 0.001) {
                int_eng_source[c] += d_add_heat_coeff[i]
                                   * delT * rho_CC[c] * vol;
              }
            }  // iter loop
          }  // if right matl
        } 
      }  // if add heat

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc <<  "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printData(indx, patch,1,desc.str(),"int_eng_source", int_eng_source);
      }
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
      constCCVariable<double> int_eng_source;
      constCCVariable<Vector> mom_source;
      constCCVariable<Vector> mom_comb;
      Ghost::GhostType  gn = Ghost::None;
      new_dw->get(rho_CC,         lb->rho_CCLabel,           indx,patch,gn,0);  
      old_dw->get(vel_CC,         lb->vel_CCLabel,           indx,patch,gn,0);  
      old_dw->get(temp_CC,        lb->temp_CCLabel,          indx,patch,gn,0);  
      new_dw->get(mom_source,     lb->mom_source_CCLabel,    indx,patch,gn,0);    
      new_dw->get(int_eng_source, lb->int_eng_source_CCLabel,indx,patch,gn,0);  
      new_dw->allocateAndPut(mom_L,     lb->mom_L_CCLabel,     indx,patch);
      new_dw->allocateAndPut(int_eng_L, lb->int_eng_L_CCLabel, indx,patch);
      new_dw->allocateAndPut(mass_L,    lb->mass_L_CCLabel,    indx,patch);
      double cv = ice_matl->getSpecificHeat();
      //__________________________________
      //  NO mass exchange
      if(d_massExchange == false && d_models.size() == 0) {
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
//   T H R O W   A W A Y   W H E N   M O D E L S   A R E   W O R K I N G

      //__________________________________
      //  WITH mass exchange
      // Note that the mass exchange can't completely
      // eliminate all the mass, momentum and internal E
      // If it does then we'll get erroneous vel, and temps
      // after advection.  Thus there is always a mininum amount
      if(d_massExchange && d_models.size() == 0)  {
       constCCVariable<double> burnedMass;
       new_dw->get(burnedMass,   lb->burnedMass_CCLabel,    indx,patch,gn,0);
       constCCVariable<Vector> mom_comb;
       new_dw->get(mom_comb,     lb->mom_comb_CCLabel,      indx,patch,gn,0);
       constCCVariable<double> int_eng_comb;
       new_dw->get(int_eng_comb, lb->int_eng_comb_CCLabel,  indx,patch,gn,0);
        double massGain = 0.;
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
         IntVector c = *iter;
         massGain += burnedMass[c];
        }
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
           iter++) {
         IntVector c = *iter;
           //  must have a minimum mass
          double mass = rho_CC[c] * vol;
          double min_mass = d_TINY_RHO * vol;

          mass_L[c] = std::max( (mass + burnedMass[c] ), min_mass);

          //  must have a minimum momentum   
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons
            double min_mom_L = vel_CC[c][dir] * min_mass;
            double mom_L_tmp = vel_CC[c][dir] * mass + mom_comb[c][dir];
  
             // Preserve the original sign on momemtum     
             // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp + d_SMALL_NUM)/
                                    (fabs(mom_L_tmp + d_SMALL_NUM));
            
            mom_L[c][dir] = mom_source[c][dir] +
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
         if(massGain > 0.0){
           cout << "Mass gained timestep = " << massGain << endl;
         }
       }  //  if (mass exchange)

      //__________________________________
      //      M O D E L - B A S E D   E X C H A N G E
      //  WITH "model-based" mass exchange
      // Note that the mass exchange can't completely
      // eliminate all the mass, momentum and internal E
      // If it does then we'll get erroneous vel, and temps
      // after advection.  Thus there is always a mininum amount
      if(d_models.size() > 0)  {
       constCCVariable<double> modelMass_src;
       constCCVariable<double> modelEng_src;
       constCCVariable<Vector> modelMom_src;
       new_dw->get(modelMass_src,lb->modelMass_srcLabel,indx, patch, gn, 0);
       new_dw->get(modelMom_src, lb->modelMom_srcLabel, indx, patch, gn, 0);
       new_dw->get(modelEng_src, lb->modelEng_srcLabel, indx, patch, gn, 0);

        double massGain = 0.;
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
         IntVector c = *iter;
         massGain += modelMass_src[c];
        }
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
           iter++) {
         IntVector c = *iter;
           //  must have a minimum mass
          double mass = rho_CC[c] * vol;
          double min_mass = d_TINY_RHO * vol;

          mass_L[c] = std::max( (mass + modelMass_src[c] ), min_mass);

          //  must have a minimum momentum   
          for (int dir = 0; dir <3; dir++) {  //loop over all three directons
            double min_mom_L = vel_CC[c][dir] * min_mass;
            double mom_L_tmp = vel_CC[c][dir] * mass + modelMom_src[c][dir];
  
             // Preserve the original sign on momemtum     
             // Use d_SMALL_NUMs to avoid nans when mom_L_temp = 0.0
            double plus_minus_one = (mom_L_tmp + d_SMALL_NUM)/
                                    (fabs(mom_L_tmp + d_SMALL_NUM));
            
            mom_L[c][dir] = mom_source[c][dir] +
                  plus_minus_one * std::max( fabs(mom_L_tmp), min_mom_L );
          }
          // must have a minimum int_eng   
          double min_int_eng = min_mass * cv * temp_CC[c];
          double int_eng_tmp = mass * cv * temp_CC[c];

          //  Glossary:
          //  int_eng_tmp    = the amount of internal energy for this
          //                   matl in this cell coming into this task
          //  int_eng_source = thermodynamic work = f(delP_Dilatation)
          //  modelEng_src   = enthalpy of reaction gained by the
          //                   product gas, PLUS (OR, MINUS) the
          //                   internal energy of the reactant
          //                   material that was liberated in the
          //                   reaction
          // min_int_eng     = a small amount of internal energy to keep
          //                   the equilibration pressure from going nuts

          int_eng_L[c] = int_eng_tmp +
                             int_eng_source[c] +
                             modelEng_src[c];

          int_eng_L[c] = std::max(int_eng_L[c], min_int_eng);
         }
         if(massGain > 0.0){
          cout << "Mass gained by the models this timestep = " 
               << massGain << endl;
         }
       }  //  if (models.size() > 0)

        //---- P R I N T   D A T A ------ 
        // Dump out all the matls data
        if (switchDebugLagrangianValues ) {
          ostringstream desc;
          desc <<"BOT_Lagrangian_Values_Mat_"<<indx<< "_patch_"<<patch->getID();
          printData(  indx, patch,1, desc.str(), "mass_L_CC",    mass_L);
          printVector(indx, patch,1, desc.str(), "mom_L_CC", 0,  mom_L);
          printData(  indx, patch,1, desc.str(), "int_eng_L_CC", int_eng_L); 

        }
        //____ B U L L E T   P R O O F I N G----
        // catch negative internal energies
        IntVector neg_cell;
        if (!areAllValuesPositive(int_eng_L, neg_cell) ) {
         ostringstream warn;
         warn<<"ICE::computeLagrangianValues, mat "<<indx<<" cell "
             <<neg_cell<<" Negative int_eng_L \n";
         throw InvalidValue(warn.str());
        }
      }  // if (ice_matl)
    }  // end numALLMatl loop
  }  // patch loop
}
/* ---------------------------------------------------------------------
 Function~  ICE::computeLagrangianSpecificVolume--
 ---------------------------------------------------------------------  */
void ICE::computeLagrangianSpecificVolume(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing computeLagrangianSpecificVolume " <<
      patch->getID() << "\t\t\t ICE" << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());

    int numALLMatls = d_sharedState->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    StaticArray<constCCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > vol_frac(numALLMatls);
    StaticArray<constCCVariable<double> > Temp_CC(numALLMatls);
    StaticArray<CCVariable<double> > alpha(numALLMatls);
    constCCVariable<double> rho_CC, f_theta,sp_vol_CC;
    constCCVariable<double> delP, P;
    CCVariable<double> sum_therm_exp;
    vector<double> if_mpm_matl_ignore(numALLMatls);

    new_dw->allocateTemporary(sum_therm_exp,patch);
    new_dw->get(delP, lb->delP_DilatateLabel, 0, patch,gn, 0);
    new_dw->get(P,    lb->press_CCLabel,      0, patch,gn, 0);
    sum_therm_exp.initialize(0.);

    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->get(Tdot[m],    lb->Tdot_CCLabel,    indx,patch, gn,0);
      new_dw->get(vol_frac[m],lb->vol_frac_CCLabel,indx,patch, gac, 1);
      new_dw->allocateTemporary(alpha[m],patch);
      if (ice_matl) {
        old_dw->get(Temp_CC[m], lb->temp_CCLabel,  indx,patch, gn,0);
      }
      if (mpm_matl) {
        new_dw->get(Temp_CC[m],lb->temp_CCLabel,   indx,patch, gn,0);
      }
    }

    //__________________________________
    // Sum of thermal expansion
    // ignore contributions from mpm_matls
    // UNTIL we have temperature dependent EOS's for the solids
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      if (ice_matl) {
       if_mpm_matl_ignore[m]=1.0;
       new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn, 0);
       double cv = ice_matl->getSpecificHeat();
       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          alpha[m][c]=
             ice_matl->getEOS()->getAlpha(Temp_CC[m][c],sp_vol_CC[c],P[c],cv);
          sum_therm_exp[c] += vol_frac[m][c]*alpha[m][c]*Tdot[m][c];
        } 
      } else {
        if_mpm_matl_ignore[m]=0.0;
        alpha[m].initialize(0.0);
      }
     
    }

    //__________________________________ 
    for(int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      CCVariable<double> spec_vol_L, spec_vol_source;
      constCCVariable<double> speedSound;
      new_dw->allocateAndPut(spec_vol_L,     lb->spec_vol_L_CCLabel,
                                                            indx,patch);
      new_dw->allocateAndPut(spec_vol_source,lb->spec_vol_source_CCLabel,
                                                            indx,patch);
      spec_vol_source.initialize(0.);

      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn, 0);
      new_dw->get(rho_CC,     lb->rho_CCLabel,        indx,patch,gn, 0);
      new_dw->get(f_theta,    lb->f_theta_CCLabel,    indx,patch,gn, 0);
      new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gn, 0);

      //__________________________________
      //  compute spec_vol_L * mass
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        spec_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c];
      }
      //__________________________________
      //  MODELS REMOVE
      if(d_models.size() == 0){
        constCCVariable<double> sp_vol_comb;   //MODELS .....REMOVE
        new_dw->get(sp_vol_comb,lb->created_vol_CCLabel,indx,patch,gn, 0); //MODEL REMOVE
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         spec_vol_L[c] += sp_vol_comb[c];
        }
      }      
      //__________________________________
      //   Contributions from models
      if(d_models.size() > 0){
        constCCVariable<double> Modelsp_vol_src;
        new_dw->get(Modelsp_vol_src, lb->modelVol_srcLabel, indx, patch, gn, 0);
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         spec_vol_L[c] += Modelsp_vol_src[c];
        }
      }
      //__________________________________
      //  add the sources to spec_vol_L
      for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        //__________________________________
        //  term1
        double kappa = sp_vol_CC[c]/(speedSound[c]*speedSound[c]);
        double term1 = -vol_frac[m][c] * kappa * vol * delP[c];
        double term2 = delT * vol * (vol_frac[m][c] * alpha[m][c] * Tdot[m][c] -
                                     f_theta[c] * sum_therm_exp[c]);

        // This is actually mass * sp_vol
        double src = term1 + if_mpm_matl_ignore[m] * term2;
        spec_vol_L[c]     += src;
        spec_vol_source[c] = src/(rho_CC[c] * vol);

/*`==========TESTING==========*/
//    do we really want this?  -Todd        
        spec_vol_L[c] = max(spec_vol_L[c], d_TINY_RHO * vol * sp_vol_CC[c]);
/*==========TESTING==========`*/
     }

      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(spec_vol_L, "set_if_sym_BC",patch, d_sharedState, indx);

      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      if (!areAllValuesPositive(spec_vol_L, neg_cell)) {
        cout << "matl            "<< indx << endl;
        cout << "sum_thermal_exp "<< sum_therm_exp[neg_cell] << endl;
        cout << "spec_vol_source "<< spec_vol_source[neg_cell] << endl;
//        cout << "sp_vol_comb     "<< sp_vol_comb[neg_cell] << endl;
        cout << "mass sp_vol_L    "<< spec_vol_L[neg_cell] << endl;
        cout << "mass sp_vol_L_old"
             << (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        ostringstream warn;
        warn<<"ERROR ICE::computeLagrangianSpecificVolumeRF, mat "<<indx
            << " cell " <<neg_cell << " spec_vol_L is negative\n";
        throw InvalidValue(warn.str());
     }
    }  // end numALLMatl loop
  }  // patch loop

}
/*---------------------------------------------------------------------
 Function~  ICE::addExchangeToMomentumAndEnergy--
   This task adds the  exchange contribution to the 
   existing cell-centered momentum and internal energy
            
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

 Steps for each cell;
    1) Comute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for X[*]
    4) Add X[*] to the appropriate Lagrangian data
 - apply Boundary conditions to vel_CC and Temp_CC

 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 ---------------------------------------------------------------------  */
void ICE::addExchangeToMomentumAndEnergy(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t ICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    //Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Scratch Variables
  /*`==========TESTING==========*/
//    StaticArray<CCVariable<double> > scratch1(numALLMatls);  
//    StaticArray<CCVariable<double> > scratch2(numALLMatls);  
//    StaticArray<CCVariable<Vector> > scratchVec(numALLMatls);  
  /*`==========TESTING==========*/

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_temp(numALLMatls);

    vector<double> b(numALLMatls);
    vector<double> sp_vol(numALLMatls);
    vector<double> cv(numALLMatls);
    vector<double> X(numALLMatls);
    double tmp;
    FastMatrix beta(numALLMatls, numALLMatls),acopy(numALLMatls, numALLMatls);
    FastMatrix K(numALLMatls, numALLMatls), H(numALLMatls, numALLMatls);
    FastMatrix a(numALLMatls, numALLMatls), a_inverse(numALLMatls, numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();

#ifdef CONVECT
    FastMatrix cet(2,2),ac(2,2);
    vector<double> RHSc(2),HX(2);
    cet.zero();
    int gm=2;  // gas material from which to get convected heat
    int sm=0;  // solid material that heat goes to
#endif

    getExchangeCoefficients( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      Ghost::GhostType  gn = Ghost::None;
      if(mpm_matl){                 // M P M
        new_dw->get(old_temp[m],     lb->temp_CCLabel,   indx, patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  lb->vel_CCLabel,indx,patch);
        new_dw->getModifiable(Temp_CC[m], lb->temp_CCLabel,indx,patch);
        cv[m] = mpm_matl->getSpecificHeat();
      }
      if(ice_matl){                 // I C E
        old_dw->get(old_temp[m],    lb->temp_CCLabel,    indx, patch,gn,0);
        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch); 
        cv[m] = ice_matl->getSpecificHeat();
      }                             // A L L  M A T L S

      new_dw->get(mass_L[m],        lb->mass_L_CCLabel,   indx, patch,gn, 0);
      new_dw->get(sp_vol_CC[m],     lb->sp_vol_CCLabel,   indx, patch,gn, 0);
      new_dw->get(mom_L[m],         lb->mom_L_CCLabel,    indx, patch,gn, 0);
      new_dw->get(int_eng_L[m],     lb->int_eng_L_CCLabel,indx, patch,gn, 0);
      new_dw->get(vol_frac_CC[m],   lb->vol_frac_CCLabel, indx, patch,gn, 0);
      new_dw->allocateAndPut(Tdot[m],        lb->Tdot_CCLabel,    indx,patch);
      new_dw->allocateAndPut(mom_L_ME[m],    lb->mom_L_ME_CCLabel,indx,patch);
      new_dw->allocateAndPut(int_eng_L_ME[m],lb->eng_L_ME_CCLabel,indx,patch);

  /*`==========TESTING==========*/
//      new_dw->allocateAndPut(scratch1[m],  MIlb->scratch1Label,   indx,patch);
//      new_dw->allocateAndPut(scratch2[m],  MIlb->scratch2Label,   indx,patch);
//      new_dw->allocateAndPut(scratchVec[m],MIlb->scratchVecLabel, indx,patch);
//      scratchVec[m].initialize(Vector(0.,0.,0.));
//      scratch1[m].initialize(0.);
//      scratch2[m].initialize(0.);
  /*`==========TESTING==========*/
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
    if (switchDebugMomentumExchange_CC ) {
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"TOP_addExchangeToMomentumAndEnergy_"<<indx<<"_patch_"
            <<patch->getID();
        printData(   indx, patch,1, desc.str(),"Temp_CC",    Temp_CC[m]);     
        printData(   indx, patch,1, desc.str(),"int_eng_L",  int_eng_L[m]);   
        printData(   indx, patch,1, desc.str(),"mass_L",     mass_L[m]);      
        printVector( indx, patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);            
      }
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      //---------- M O M E N T U M   E X C H A N G E
      //   Form BETA matrix (a), off diagonal terms
      //   beta and (a) matrix are common to all momentum exchanges
      for(int m = 0; m < numALLMatls; m++)  {
        tmp = sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta(m,n) = delT * vol_frac_CC[n][c]  * K(n,m) * tmp;
          a(m,n) = -beta(m,n);
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) = 1.0;
        for(int n = 0; n < numALLMatls; n++) {
          a(m,m) +=  beta(m,n);
        }
      }
      a_inverse.destructiveInvert(a);
      
      for (int dir = 0; dir <3; dir++) {  //loop over all three directons
        for(int m = 0; m < numALLMatls; m++) {
          b[m] = 0.0;
          for(int n = 0; n < numALLMatls; n++) {
           b[m] += beta(m,n) *
             (vel_CC[n][c][dir] - vel_CC[m][c][dir]);
          }
        }
        a_inverse.multiply(b,X);
        for(int m = 0; m < numALLMatls; m++) {
          vel_CC[m][c][dir] =  vel_CC[m][c][dir] + X[m];
        }
      }

      //---------- E N E R G Y   E X C H A N G E     
      for(int m = 0; m < numALLMatls; m++) {
        tmp = sp_vol_CC[m][c] / cv[m];
        for(int n = 0; n < numALLMatls; n++)  {
          beta(m,n) = delT * vol_frac_CC[n][c] * H(n,m)*tmp;
          a(m,n) = -beta(m,n);
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) = 1.;
        for(int n = 0; n < numALLMatls; n++)   {
          a(m,m) +=  beta(m,n);
        }
      }
      // -  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;

       for(int n = 0; n < numALLMatls; n++) {
         b[m] += beta(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
      }
      //     S O L V E, Add exchange contribution to orig value
      a.destructiveSolve(b,X);
      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = Temp_CC[m][c] + X[m];
      }
    }  //end CellIterator loop

#ifdef CONVECT 
    //  Loop over matls
    //  if (mpm_matl)
    //  Loop over cells
    //  find surface and surface normals
    //  choose adjacent cell
    //  find mass weighted average temp in adjacent cell (T_ave)
    //  compute a heat transfer to the container h(T-T_ave)
    //  compute Temp_CC = Temp_CC + h_trans/(mass*cv)
    //  end loop over cells
    //  endif (mpm_matl)
    //  endloop over matls

    Ghost::GhostType  gac = Ghost::AroundCells;
    constNCVariable<double> NC_CCweight, NCsolidMass;
    old_dw->get(NC_CCweight,     MIlb->NC_CCweightLabel,  0,   patch,gac,1);
    Vector dx = patch->dCell();
    double dxlen = dx.length();
    const Level* level=patch->getLevel();

    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int dwindex = matl->getDWIndex();
      if(mpm_matl && dwindex==sm){
        new_dw->get(NCsolidMass,     MIlb->gMassLabel,   dwindex,patch,gac,1);
        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          IntVector nodeIdx[8];
          patch->findNodesFromCell(*iter,nodeIdx);
          double MaxMass = d_SMALL_NUM;
          double MinMass = 1.0/d_SMALL_NUM;
          for (int nN=0; nN<8; nN++) {
            MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                       NCsolidMass[nodeIdx[nN]]);
            MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                       NCsolidMass[nodeIdx[nN]]);
          }
          if ((MaxMass-MinMass)/MaxMass == 1.0 && (MaxMass > d_SMALL_NUM)){
            double gradRhoX = 0.25 *
                   ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                   -
                   ( NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])) / dx.x();
            double gradRhoY = 0.25 *
                   ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                   - 
                   ( NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])) / dx.y();
            double gradRhoZ = 0.25 *                          
                   ((NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                  -
                   ( NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])) / dx.z();

            double absGradRho = sqrt(gradRhoX*gradRhoX +
                                      gradRhoY*gradRhoY +
                                      gradRhoZ*gradRhoZ );

            Vector surNorm(gradRhoX/absGradRho,
                           gradRhoY/absGradRho,
                           gradRhoZ/absGradRho);


            Point this_cell_pos = level->getCellPosition(c);
            Point adja_cell_pos = this_cell_pos + .6*dxlen*surNorm; 

            IntVector q;
            if(patch->findCell(adja_cell_pos, q)){
              cet(0,0)=delT*vol_frac_CC[sm][c]*H(sm,sm)*sp_vol_CC[sm][c]/cv[sm];
              cet(0,1)=delT*vol_frac_CC[gm][q]*H(sm,gm)*sp_vol_CC[sm][c]/cv[sm];
              cet(1,0)=delT*vol_frac_CC[sm][c]*H(gm,sm)*sp_vol_CC[gm][q]/cv[gm];
              cet(1,1)=delT*vol_frac_CC[gm][q]*H(gm,gm)*sp_vol_CC[gm][q]/cv[gm];
              //   Form matrix (a) diagonal terms
              for(int m = 0; m < 2; m++) {
                ac(m,m) = 1.;
                for(int n = 0; n < 2; n++)   {
                  ac(m,m) +=  cet(m,n);
                }
              }
//              scratch1[0][c] = Temp_CC[0][c];
//              scratch1[1][c] = Temp_CC[1][q];
              RHSc[0] = cet(0,0)*(Temp_CC[sm][c] - Temp_CC[sm][c])
                      + cet(0,1)*(Temp_CC[gm][q] - Temp_CC[sm][c]);
              RHSc[1] = cet(1,0)*(Temp_CC[sm][c] - Temp_CC[gm][q])
                      + cet(1,1)*(Temp_CC[gm][q] - Temp_CC[gm][q]);
              ac.destructiveSolve(RHSc,HX);
              Temp_CC[sm][c] += HX[0];
              Temp_CC[gm][q] += HX[1];
//              scratch2[0][c] = Temp_CC[0][c];
//              scratch2[1][c] = Temp_CC[1][q];
//              double Tinf = Temp_CC[1][a];
//              double HX = 0.*(Tinf - Temp_CC[m][c]);
//              Temp_CC[m][c] += HX/cv[m]*mass_L[m][c];
            }
          }  // if a surface cell
        }    // cellIterator
      }      // if mpm_matl
    }        // for ALL matls
#endif
/*`==========TESTING==========*/
#ifndef LODI_BCS
    //__________________________________
    //  Set the Boundary conditions 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int dwindex = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch,dwindex);
      setBC(Temp_CC[m],"Temperature",patch, d_sharedState, dwindex);
    }
#endif 
/*==========TESTING==========`*/
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c] * cv[m] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]          * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
      }
    }

    //---- P R I N T   D A T A ------ 
    if (switchDebugMomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"addExchangeToMomentumAndEnergy_"<<indx<<"_patch_"
            <<patch->getID();
        printVector(indx, patch,1, desc.str(), "mom_L_ME", 0,mom_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"int_eng_L_ME",int_eng_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"Tdot",        Tdot[m]);
        printData(  indx, patch,1, desc.str(),"Temp_CC",     Temp_CC[m]);
      }
    }
  } //patches
}
 
/* --------------------------------------------------------------------- 
 Function~  ICE::advectAndAdvanceInTime--
 Purpose~
   This task calculates the The cell-centered, time n+1, mass, momentum
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
    double vol = dx.x()*dx.y()*dx.z();
    double invvol = 1.0/vol;
    double cv;
    IntVector neg_cell;
    ostringstream warn;

    // These arrays get re-used for each material, and for each
    // advected quantity
    const IntVector gc(1,1,1);
    CCVariable<double>  q_advected, mass_new, mass_advected;
    CCVariable<Vector>  qV_advected; 

    Advector* advector = d_advector->clone(new_dw,patch);
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    new_dw->allocateTemporary(mass_new,     patch);
    new_dw->allocateTemporary(mass_advected,patch);
    new_dw->allocateTemporary(q_advected,   patch);
    new_dw->allocateTemporary(qV_advected,  patch);

    int numALLMatls = d_sharedState->getNumMatls();

    for (int m = 0; m < numALLMatls; m++ ) {
     Material* matl = d_sharedState->getMaterial( m );
     ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
     if(ice_matl){
      int indx = matl->getDWIndex(); 

      CCVariable<double> rho_CC, temp, sp_vol_CC,mach;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_L_ME, mass_L,spec_vol_L,speedSound;
      constCCVariable<Vector> mom_L_ME;
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      new_dw->get(speedSound,  lb->speedSound_CCLabel,    indx,patch,gn,0);
      new_dw->get(uvel_FC,     lb->uvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(vvel_FC,     lb->vvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(wvel_FC,     lb->wvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(mass_L,      lb->mass_L_CCLabel,        indx,patch,gac,2);  

      new_dw->get(mom_L_ME,    lb->mom_L_ME_CCLabel,      indx,patch,gac,2);
      new_dw->get(spec_vol_L,  lb->spec_vol_L_CCLabel,    indx,patch,gac,2);
      new_dw->get(int_eng_L_ME,lb->eng_L_ME_CCLabel,      indx,patch,gac,2);
      new_dw->getModifiable(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch);
      new_dw->getModifiable(rho_CC,    lb->rho_CCLabel,   indx,patch);
      cv = ice_matl->getSpecificHeat();
      new_dw->allocateAndPut(temp,   lb->temp_CCLabel,  indx,patch);          
      new_dw->allocateAndPut(vel_CC, lb->vel_CCLabel,   indx,patch);
      new_dw->allocateAndPut(mach,   lb->machLabel,     indx,patch);  
              
      rho_CC.initialize(0.0);
      temp.initialize(0.0);
      q_advected.initialize(0.0);  
      mass_advected.initialize(0.0);
      vel_CC.initialize(Vector(0.0,0.0,0.0));
      qV_advected.initialize(Vector(0.0,0.0,0.0)); 
    
      //__________________________________
      //   Advection preprocessing
      bool bulletProof_test=true;
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                                    bulletProof_test); 

      //__________________________________
      // Advect mass and backout rho_CC
      advector->advectQ(mass_L,patch,mass_advected, new_dw);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        mass_new[c]  = (mass_L[c] + mass_advected[c]);
        rho_CC[c]    = mass_new[c] * invvol;
      }   
      
/*`==========TESTING==========*/
#ifndef LODI_BCS
      setBC(rho_CC, "Density", patch, d_sharedState, indx);
#endif 
/*==========TESTING==========`*/

      //__________________________________
      // Advect  momentum and backout vel_CC
      advector->advectQ(mom_L_ME,patch,qV_advected, new_dw);

      Vector vel_L_ME, vel_advected;
      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        vel_L_ME = mom_L_ME[c]/mass_L[c];
        
        vel_advected = (qV_advected[c] - vel_L_ME * mass_advected[c])/
                       (mass_new[c]);
        vel_CC[c]    = (vel_L_ME + vel_advected);
//      vel_CC[c] = (mom_L_ME[c] + qV_advected[c])/mass_new[c] ; 
//        if(rho_CC[c] < 1.e-2){
//            vel_CC[c] = Vector(0.,0.,0.);
//        }
      }
      
/*`==========TESTING==========*/
#ifndef LODI_BCS    
      setBC(vel_CC, "Velocity", patch,indx);
#endif 
/*==========TESTING==========`*/

      //__________________________________
      // Advect internal energy and backout Temp_CC
      advector->advectQ(int_eng_L_ME,patch,q_advected, new_dw);

      double Temp_advected, Temp_cv_L_ME, Temp_L_ME;
      if (d_EqForm){         // EQ FORM
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;

          
          Temp_cv_L_ME  = int_eng_L_ME[c]/mass_L[c];
          Temp_L_ME     = Temp_cv_L_ME/cv;
          Temp_advected = (q_advected[c] - Temp_cv_L_ME * mass_advected[c])/
                          (cv * mass_new[c] );
                          
          temp[c]       = Temp_L_ME + Temp_advected;
//        temp[c] = (int_eng_L_ME[c] + q_advected[c])/(mass_new[c]*cv);
        }
      }

      if (d_RateForm){      // RATE FORM
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double KE = 0.5 * vel_CC[c].length() * vel_CC[c].length();

          Temp_cv_L_ME  = int_eng_L_ME[c]/mass_L[c];
          Temp_L_ME     = Temp_cv_L_ME/cv;
          Temp_advected = (q_advected[c] - Temp_cv_L_ME * mass_advected[c])/
                          (cv * mass_new[c]);
                         
          temp[c]       = (Temp_L_ME + Temp_advected - KE/cv);
          
//        double KE = 0.5 * mass_new[c] * vel_CC[c].length() * vel_CC[c].length();
//        temp[c] = (int_eng_L_ME[c] + q_advected[c] - KE)/(mass_new[c] * cv);
        }
      }

/*`==========TESTING==========*/
#ifndef LODI_BCS     
      setBC(temp, "Temperature", patch, d_sharedState, indx);
#endif 
/*==========TESTING==========`*/

      //__________________________________
      // Advection of specific volume
      // Note sp_vol_L[m] is actually sp_vol[m] * mass
      advector->advectQ(spec_vol_L,patch,q_advected, new_dw); 
      
      double sp_vol_tmp, sp_vol_advected;      
      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
        IntVector c = *iter;
        sp_vol_tmp      = spec_vol_L[c]/mass_L[c];        
        sp_vol_advected = (q_advected[c] - sp_vol_tmp * mass_advected[c])/
                          ( mass_new[c]);
        sp_vol_CC[c]    = sp_vol_tmp + sp_vol_advected; 
//      sp_vol_CC[c] = (spec_vol_L[c] + q_advected[c])/mass_new[c];  
      }
      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(sp_vol_CC, "set_if_sym_BC",patch, d_sharedState, indx); 
      
      //__________________________________
      // Advect  model variables 
      if(d_modelSetup && d_modelSetup->tvars.size() > 0){
        for(vector<TransportedVariable*>::iterator iter = d_modelSetup->tvars.begin();
           iter != d_modelSetup->tvars.end(); iter++){
          TransportedVariable* tvar = *iter;
          if(tvar->matls->contains(indx)){
            constCCVariable<double> q_L_CC,q_src;
            CCVariable<double> q_new, q_CC;
	    old_dw->get(q_L_CC, tvar->var, indx, patch, gac, 2);         
	    if(tvar->src) 
	      new_dw->get(q_src,  tvar->src, indx, patch, gac, 2);         
	    new_dw->allocateTemporary(q_new, patch, gac, 2);
            new_dw->allocateAndPut(q_CC, tvar->var, indx, patch);
            
	     if(tvar->src){  // if transported variable has a source
	       for(CellIterator iter(q_L_CC.getLowIndex(), q_L_CC.getHighIndex());
		    !iter.done(); iter++){
		  IntVector c = *iter;                            
		  q_new[c]  = (q_L_CC[c] + q_src[c])*mass_L[c];
	       }
	       advector->advectQ(q_new,patch,q_advected, new_dw);
	     } else {
	       for(CellIterator iter(q_L_CC.getLowIndex(), q_L_CC.getHighIndex());
		    !iter.done(); iter++){
		  IntVector c = *iter;                            
		  q_new[c]  = q_L_CC[c]*mass_L[c];
	       }
	       advector->advectQ(q_new,patch,q_advected, new_dw);
	     }

	     for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
	       IntVector c = *iter;
	       double q_tmp = q_new[c]/mass_L[c];
	       double q_a   = (q_advected[c] - q_tmp * mass_advected[c])/
                             ( mass_new[c]);
	       q_CC[c] = q_tmp + q_a; 
	     }
            
            //  Set Boundary Conditions 
            string Labelname = tvar->var->getName();
	     setBC(q_CC, Labelname,  patch, d_sharedState, indx); 
          }
        }
      }
      //__________________________________
      // Compute Auxilary quantities
      for(CellIterator iter = patch->getExtraCellIterator(); 
                                                             !iter.done(); iter++) {
        IntVector c = *iter;
        mach[c]  = vel_CC[c].length()/speedSound[c];
      }
//______________________ L O D I__________________________________
#ifdef LODI_BCS  
cout << "using LODI BCS" <<endl;
      Ghost::GhostType  gn  = Ghost::None;  
      constCCVariable<double> press_old, rho_old, temp_old, sp_vol_old;
      constCCVariable<double> vol_frac_old;
      constCCVariable<Vector> vel_old;
      CCVariable<double> press_tmp,e;   
      CCVariable<Vector> nu; 
      double gamma = ice_matl->getGamma();

                                //  O L D   D W
      old_dw->get(press_old,    lb->press_CCLabel,    0,   patch,gn,0);
      old_dw->get(temp_old,     lb->temp_CCLabel,     indx,patch,gn,0);
      old_dw->get(rho_old,      lb->rho_CCLabel,      indx,patch,gn,0);
      old_dw->get(vel_old,      lb->vel_CCLabel,      indx,patch,gn,0);
      old_dw->get(sp_vol_old,   lb->sp_vol_CCLabel,   indx,patch,gn,0);
      old_dw->get(vol_frac_old, lb->vol_frac_CCLabel, indx,patch,gn,0);
 
      new_dw->allocateTemporary(press_tmp,  patch);
      new_dw->allocateTemporary(nu, patch);
      new_dw->allocateTemporary(e,   patch);
     
      nu.initialize(Vector(0,0,0)); 
      press_tmp.initialize(0.0);  
      e.initialize(0.0);
      
      StaticArray<CCVariable<Vector> > di(6);
      for (int i = 0; i <= 5; i++){
        new_dw->allocateTemporary(di[i], patch);
        di[i].initialize(Vector(0,0,0));
      }    

      //   T O   D O :  change to faceCellIterator
      for(CellIterator iter = patch->getExtraCellIterator();
                                     !iter.done();iter++){
        IntVector c = *iter;
        e[c] = rho_old[c] * (cv * temp_old[c] +  0.5 * vel_old[c].length2() );                                                 
        press_tmp[c] = vol_frac_old[c] * press_old[c];        
      } 

      //compute dissipation coefficients
      computeNu(nu, press_tmp, patch);
      
      //compute Di at boundary cells
      computeDi(di,rho_old,  press_tmp, vel_old, 
                            speedSound, patch, indx);  
    #if 0
      //--------------------TESTING---------------------//    
      ostringstream desc;
      desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"<<patch->getID();
      printData(   indx, patch,1, desc.str(), "press_tmp",    press_tmp);
      printData(   indx, patch,1, desc.str(), "rho_old",        rho_old);
      printVector( indx, patch,1, desc.str(), "vel_old", 0, vel_old);
      printVector( indx, patch,1, desc.str(), "nu",      0, nu);
      printVector( indx, patch,1, desc.str(), "d1",      0, di[1]);    
      printVector( indx, patch,1, desc.str(), "d2",      0, di[2]);    
      printVector( indx, patch,1, desc.str(), "d3",      0, di[3]);    
      printVector( indx, patch,1, desc.str(), "d4",      0, di[4]);    
      printVector( indx, patch,1, desc.str(), "d5",      0, di[5]);    
     
       //--------------------TESTING---------------------//             
    #endif          
      setBCDensityLODI(rho_CC,di,nu,
                       rho_old, press_tmp, 
                       vel_old, delT, patch, indx);

      setBCVelLODI(vel_CC, di,nu,
                           rho_old, press_tmp, 
                           vel_old, delT, patch, indx);
                     
                       
      setBCTempLODI(temp, di,
                          e, rho_CC, nu,
                          rho_old, press_tmp, vel_old, 
                          delT, cv, gamma, patch, indx);
#endif

      //---- P R I N T   D A T A ------   
      if (switchDebug_advance_advect ) {
       ostringstream desc;
       desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"<<patch->getID(); 
        
       printData(   indx, patch,1, desc.str(), "mass_L",        mass_L); 
       printData(   indx, patch,1, desc.str(), "mass_advected", mass_advected);
       printVector( indx, patch,1, desc.str(), "mom_L_CC", 0, mom_L_ME); 
       printData(   indx, patch,1, desc.str(), "sp_vol_L",    spec_vol_L);
       printData(   indx, patch,1, desc.str(), "int_eng_L_CC",int_eng_L_ME);
       printData(   indx, patch,1, desc.str(), "rho_CC",      rho_CC);
       printData(   indx, patch,1, desc.str(), "Temp_CC",     temp);
       printData(   indx, patch,1, desc.str(), "sp_vol_CC",   sp_vol_CC);
       printVector( indx, patch,1, desc.str(), "vel_CC", 0,   vel_CC);
      }
      //____ B U L L E T   P R O O F I N G----
      if (!areAllValuesPositive(rho_CC, neg_cell)) {
        warn <<"ERROR ICE::advectAndAdvanceInTime, mat "<< indx <<" cell "
             << neg_cell << " negative rho_CC\n ";
        throw InvalidValue(warn.str());
      }
      if (!areAllValuesPositive(temp, neg_cell)) {
        warn <<"ERROR ICE::advectAndAdvanceInTime, mat "<< indx <<" cell "
             << neg_cell << " negative temp_CC\n ";
        throw InvalidValue(warn.str());
      }
      if (!areAllValuesPositive(sp_vol_CC, neg_cell)) {
        warn <<"ERROR ICE::advectAndAdvanceInTime, mat "<< indx <<" cell "
             << neg_cell << " negative sp_vol_CC\n ";        
       throw InvalidValue(warn.str());
      } 
    } // if ice_matl
   }  // for all matls

    delete advector;
  }  // patch loop 
}

/* 
 ======================================================================*
 Function:  hydrostaticPressureAdjustment--
 Notes:      Material 0 is assumed to be the surrounding fluid and 
            we compute the hydrostatic pressure using
            
            press_hydro_ = rho_micro_CC[SURROUNDING_MAT] * grav * some_distance         
_______________________________________________________________________ */
void ICE::hydrostaticPressureAdjustment(const Patch* patch,
                          const CCVariable<double>& rho_micro_CC,
                                CCVariable<double>& press_CC)
{
  Vector dx             = patch->dCell();
  Vector gravity        = d_sharedState->getGravity();
  double press_hydro;
  double dist_from_p_ref;
#if 0  // To reference the hydrostatic pressure from ceiling  
  IntVector HighIndex;
  IntVector L;
  const Level* level= patch->getLevel();
  level->findIndexRange(L, HighIndex);
  int press_ref_x = HighIndex.x() -2;   // we want the interiorCellHighIndex 
  int press_ref_y = HighIndex.y() -2;   // therefore we subtract off 2
  int press_ref_z = HighIndex.z() -2;
#endif  
  int press_ref_x = 0;  // to reference the hydrostatic pressure from the floor   
  int press_ref_y = 0;  
  int press_ref_z = 0;
  //__________________________________
  //  X direction
  if (gravity.x() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      dist_from_p_ref  = fabs((double) (c.x() - press_ref_x)) * dx.x();
      press_hydro      = rho_micro_CC[c] * gravity.x() * dist_from_p_ref;
      press_CC[c] += press_hydro;
    }
  }
  //__________________________________
  //  Y direction
  if (gravity.y() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      dist_from_p_ref = fabs((double) (c.y() - press_ref_y)) * dx.y();
      press_hydro     = rho_micro_CC[c] * gravity.y() * dist_from_p_ref;
      press_CC[c] += press_hydro;
    }
  }
  //__________________________________
  //  Z direction
  if (gravity.z() != 0.)  {
    for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      dist_from_p_ref   = fabs((double) (c.z() - press_ref_z)) * dx.z();
      press_hydro       = rho_micro_CC[c] * gravity.z() * dist_from_p_ref;
      press_CC[c] += press_hydro;
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
void ICE::computeTauX( const Patch* patch,
                       const CCVariable<double>& rho_CC,      
                       const CCVariable<double>& sp_vol_CC,   
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

    double rho_brack = (2.0 * rho_CC[left] * rho_CC[cell])/
                       (rho_CC[left] + rho_CC[cell]);
                       
    double vol_frac_FC = rho_brack * sp_vol_CC[cell]; 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].x()   +
                             vel_CC[IntVector(i,  j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0; 
                                
    double uvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].x()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].x()   +      
                             vel_CC[IntVector(i,  j  ,k  )].x()   +     
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0; 
                              
    double uvel_EC_front  = (vel_CC[IntVector(i-1,j  ,k+1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].x()   +
                             vel_CC[IntVector(i  ,j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x() )/4.0;
                             
    double uvel_EC_back   = (vel_CC[IntVector(i-1,j  ,k-1)].x()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].x()   +
                             vel_CC[IntVector(i  ,j  ,k  )].x()   +
                             vel_CC[IntVector(i-1,j  ,k  )].x())/4.0;
                             
    double vvel_EC_top    = (vel_CC[IntVector(i-1,j+1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j+1,k  )].y()   +
                             vel_CC[IntVector(i  ,j  ,k  )].y()   +
                             vel_CC[IntVector(i-1,j  ,k  )].y())/4.0;
                             
    double vvel_EC_bottom = (vel_CC[IntVector(i-1,j-1,k  )].y()   + 
                             vel_CC[IntVector(i  ,j-1,k  )].y()   +
                             vel_CC[IntVector(i  ,j  ,k  )].y()   +
                             vel_CC[IntVector(i-1,j  ,k  )].y())/4.0;
                             
    double wvel_EC_front  = (vel_CC[IntVector(i-1,j  ,k+1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k+1)].z()   +
                             vel_CC[IntVector(i  ,j  ,k  )].z()   +
                             vel_CC[IntVector(i-1,j  ,k  )].z())/4.0;
                             
    double wvel_EC_back   = (vel_CC[IntVector(i-1,j  ,k-1)].z()   + 
                             vel_CC[IntVector(i  ,j  ,k-1)].z()   +
                             vel_CC[IntVector(i  ,j  ,k  )].z()   +
                             vel_CC[IntVector(i-1,j  ,k  )].z())/4.0;
    //__________________________________
    //  tau_XX
    grad_uvel = (vel_CC[cell].x() - vel_CC[left].x())/delX;
    grad_vvel = (vvel_EC_top      - vvel_EC_bottom)  /delY;
    grad_wvel = (wvel_EC_front    - wvel_EC_back )   /delZ;

    term1 = 2.0 * viscosity * grad_uvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_X_FC[cell].x( vol_frac_FC * (term1 - term2)); 

    //__________________________________
    //  tau_XY
    grad_1 = (uvel_EC_top      - uvel_EC_bottom)  /delY;
    grad_2 = (vel_CC[cell].y() - vel_CC[left].y())/delX;
    tau_X_FC[cell].y(vol_frac_FC * viscosity * (grad_1 + grad_2)); 

    //__________________________________
    //  tau_XZ
    grad_1 = (uvel_EC_front    - uvel_EC_back)    /delZ;
    grad_2 = (vel_CC[cell].z() - vel_CC[left].z())/delX;
    tau_X_FC[cell].z(vol_frac_FC * viscosity * (grad_1 + grad_2)); 

    
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
void ICE::computeTauY( const Patch* patch,
                       const CCVariable<double>& rho_CC,      
                       const CCVariable<double>& sp_vol_CC,   
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
    double rho_brack = (2.0 * rho_CC[bottom] * rho_CC[cell])/
                       (rho_CC[bottom] + rho_CC[cell]);
                       
    double vol_frac_FC = rho_brack * sp_vol_CC[cell]; 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double uvel_EC_left  = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                            vel_CC[IntVector(i-1,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j-1,k  )].x()  +
                            vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                            
    double vvel_EC_right = (vel_CC[IntVector(i+1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i+1,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double vvel_EC_left  = (vel_CC[IntVector(i-1,j  ,k  )].y()  + 
                            vel_CC[IntVector(i-1,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0; 
                            
    double vvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double vvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].y()  +
                            vel_CC[IntVector(i  ,j-1,k  )].y()  +
                            vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                            
    double wvel_EC_front = (vel_CC[IntVector(i  ,j  ,k+1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k+1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k  )].z()  +
                            vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                            
    double wvel_EC_back  = (vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k-1)].z()  +
                            vel_CC[IntVector(i  ,j-1,k  )].z()  +
                            vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                            
    //__________________________________
    //  tau_YY
    grad_uvel = (uvel_EC_right    - uvel_EC_left)      /delX;
    grad_vvel = (vel_CC[cell].y() - vel_CC[bottom].y())/delY;
    grad_wvel = (wvel_EC_front    - wvel_EC_back )     /delZ;

    term1 = 2.0 * viscosity * grad_vvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_Y_FC[cell].y(vol_frac_FC * (term1 - term2)); 
    
    //__________________________________
    //  tau_YX
    grad_1 = (vel_CC[cell].x() - vel_CC[bottom].x())/delY;
    grad_2 = (vvel_EC_right    - vvel_EC_left)      /delX;
    tau_Y_FC[cell].x(vol_frac_FC * viscosity * (grad_1 + grad_2) ); 


    //__________________________________
    //  tau_YZ
    grad_1 = (vvel_EC_front    - vvel_EC_back)      /delZ;
    grad_2 = (vel_CC[cell].z() - vel_CC[bottom].z())/delY;
    tau_Y_FC[cell].z(vol_frac_FC * viscosity * (grad_1 + grad_2)); 
    
//     if (i == 0 && k == 0){    
//       cout<< cell<< " tau_YX: "<<tau_Y_FC[cell].x()<<
//       " tau_YY: "<<tau_Y_FC[cell].y()<<
//       " tau_YZ: "<<tau_Y_FC[cell].z()<<
//        " patch: "<<patch->getID()<<endl;
//     }
  }
}

/*---------------------------------------------------------------------
 Function~  ICE::computeTauZ
 Purpose:   This function computes shear stress tau_zx, ta_zy, tau_zz 
  Note:   - The edge velocities are defined as the average velocity 
            of the 4 cells surrounding that edge, however we only use 2 cells
            to compute it.  When you take the difference of the edge velocities
            there are two common cells that automatically cancel themselves out.
          - The viscosity we're using isn't right if it varies spatially.
 ---------------------------------------------------------------------  */
void ICE::computeTauZ( const Patch* patch,
                       const CCVariable<double>& rho_CC,      
                       const CCVariable<double>& sp_vol_CC,   
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

    double rho_brack = (2.0 * rho_CC[back] * rho_CC[cell])/
                       (rho_CC[back] + rho_CC[cell]);
                       
    double vol_frac_FC = rho_brack * sp_vol_CC[cell]; 

    //__________________________________
    // - find indices of surrounding cells
    // - compute velocities at cell face edges see note above.
    double uvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].x()  + 
                             vel_CC[IntVector(i+1,j,  k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double uvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].x()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].x()  +
                             vel_CC[IntVector(i  ,j  ,k  )].x())/4.0;
                             
    double vvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].y()  + 
                             vel_CC[IntVector(i  ,j+1,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                             
    double vvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].y()  + 
                             vel_CC[IntVector(i  ,j-1,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].y()  +
                             vel_CC[IntVector(i  ,j  ,k  )].y())/4.0;
                             
    double wvel_EC_right  = (vel_CC[IntVector(i+1,j,  k  )].z()  + 
                             vel_CC[IntVector(i+1,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_left   = (vel_CC[IntVector(i-1,j,  k  )].z()  +
                             vel_CC[IntVector(i-1,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_top    = (vel_CC[IntVector(i  ,j+1,k  )].z()  + 
                             vel_CC[IntVector(i  ,j+1,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
                             
    double wvel_EC_bottom = (vel_CC[IntVector(i  ,j-1,k  )].z()  + 
                             vel_CC[IntVector(i  ,j-1,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k-1)].z()  +
                             vel_CC[IntVector(i  ,j  ,k  )].z())/4.0;
    //__________________________________
    //  tau_ZX
    grad_1 = (vel_CC[cell].x() - vel_CC[back].x()) /delZ;
    grad_2 = (wvel_EC_right    - wvel_EC_left)     /delX;
    tau_Z_FC[cell].x(vol_frac_FC * viscosity * (grad_1 + grad_2)); 

    //__________________________________
    //  tau_ZY
    grad_1 = (vel_CC[cell].y() - vel_CC[back].y()) /delZ;
    grad_2 = (wvel_EC_top      - wvel_EC_bottom)   /delY;
    tau_Z_FC[cell].y( vol_frac_FC * viscosity * (grad_1 + grad_2) ); 

    //__________________________________
    //  tau_ZZ
    grad_uvel = (uvel_EC_right    - uvel_EC_left)    /delX;
    grad_vvel = (vvel_EC_top      - vvel_EC_bottom)  /delY;
    grad_wvel = (vel_CC[cell].z() - vel_CC[back].z())/delZ;

    term1 = 2.0 * viscosity * grad_wvel;
    term2 = (2.0/3.0) * viscosity * (grad_uvel + grad_vvel + grad_wvel);
    tau_Z_FC[cell].z( vol_frac_FC * (term1 - term2)); 

//  cout<<"tau_ZX: "<<tau_Z_FC[cell].x()<<
//        " tau_ZY: "<<tau_Z_FC[cell].y()<<
//        " tau_ZZ: "<<tau_Z_FC[cell].z()<<endl;
  }
}

// --------------------------------------------------------------------- 
//
template <class T> 
  void ICE::q_conduction(CellIterator iter, 
                         IntVector adj_offset,
                         const double thermalCond,
                         const double dx,
                         const CCVariable<double>& rho_CC,      
                         const CCVariable<double>& sp_vol_CC,   
                         const CCVariable<double>& Temp_CC,
                         T& q_FC)
{
  //__________________________________
  //  For variable thermalCond use
  //  thermalCond_FC = 2 * k[L] * k[R]/ ( k[R] + k[L])
  double thermalCond_FC = thermalCond;
  
  for(;!iter.done(); iter++){
    IntVector R = *iter;
    IntVector L = R + adj_offset;
    double rho_brack = (2.0 * rho_CC[R] * rho_CC[L])/(rho_CC[R] + rho_CC[L]);
    double vol_frac_FC = rho_brack * sp_vol_CC[R];
    q_FC[R] = -vol_frac_FC * thermalCond_FC* (Temp_CC[R] - Temp_CC[L])/dx;

  }
}


//______________________________________________________________________
//
void ICE::computeQ_conduction_FC(DataWarehouse* new_dw,
                                 const Patch* patch,
                                 const CCVariable<double>& rho_CC,      
                                 const CCVariable<double>& sp_vol_CC,   
                                 const CCVariable<double>& Temp_CC,
                                 const double thermalCond,
                                 SFCXVariable<double>& q_X_FC,
                                 SFCYVariable<double>& q_Y_FC,
                                 SFCZVariable<double>& q_Z_FC)
{
  Vector dx = patch->dCell();
  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);    // X faces
  adj_offset[1] = IntVector(0, -1, 0);    // Y faces
  adj_offset[2] = IntVector(0,  0, -1);   // Z faces

  new_dw->allocateTemporary(q_X_FC, patch, Ghost::AroundCells, 1);
  new_dw->allocateTemporary(q_Y_FC, patch, Ghost::AroundCells, 1);
  new_dw->allocateTemporary(q_Z_FC, patch, Ghost::AroundCells, 1);

  q_X_FC.initialize(0.0);
  q_Y_FC.initialize(0.0);
  q_Z_FC.initialize(0.0);

  //__________________________________
  // For multipatch problems adjust the iter limits
  // on the (left/bottom/back) patches to 
  // include the (right/top/front) faces
  // of the cells at the patch boundary. 
  // We compute q_X[right]-q_X[left] on each patch
  IntVector low,hi;      
  low = patch->getSFCXIterator().begin();    // X Face iterator
  hi  = patch->getSFCXIterator().end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?1:0,
                 patch->getBCType(patch->yplus) ==patch->Neighbor?0:0,
                 patch->getBCType(patch->zplus) ==patch->Neighbor?0:0); 
  CellIterator X_FC_iterLimits(low,hi);
         
  low = patch->getSFCYIterator().begin();   // Y Face iterator
  hi  = patch->getSFCYIterator().end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?0:0,
                 patch->getBCType(patch->yplus) ==patch->Neighbor?1:0,
                 patch->getBCType(patch->zplus) ==patch->Neighbor?0:0); 
  CellIterator Y_FC_iterLimits(low,hi); 
        
  low = patch->getSFCZIterator().begin();   // Z Face iterator
  hi  = patch->getSFCZIterator().end();
  hi +=IntVector(patch->getBCType(patch->xplus) ==patch->Neighbor?0:0,
                 patch->getBCType(patch->yplus) ==patch->Neighbor?0:0,
                 patch->getBCType(patch->zplus) ==patch->Neighbor?1:0); 
  CellIterator Z_FC_iterLimits(low,hi);            
  //__________________________________
  //  For each face compute conduction
  q_conduction<SFCXVariable<double> >(X_FC_iterLimits,
                                     adj_offset[0],  thermalCond, dx.x(),
                                     rho_CC, sp_vol_CC, Temp_CC,
                                     q_X_FC);

  q_conduction<SFCYVariable<double> >(Y_FC_iterLimits,
                                     adj_offset[1], thermalCond, dx.y(),
                                     rho_CC, sp_vol_CC, Temp_CC,
                                     q_Y_FC);
  
  q_conduction<SFCZVariable<double> >(Z_FC_iterLimits,
                                     adj_offset[2],  thermalCond, dx.z(),
                                     rho_CC, sp_vol_CC, Temp_CC,
                                     q_Z_FC); 
}
/*---------------------------------------------------------------------
 Function~  ICE::getExchangeCoefficients--
 ---------------------------------------------------------------------  */
void ICE::getExchangeCoefficients( FastMatrix& K, FastMatrix& H  )
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
        K(i,j) = K(j,i) = *it++;
        H(i,j) = H(j,i) = *it1++;
      }
     }
   } else if (2*num_coeff==(int)d_K_mom.size() && 
             2*num_coeff == (int)d_K_heat.size()){
     // Fill in the whole matrix but skip the diagonal terms
     for (int i = 0; i < numMatls; i++ )  {
      for (int j = 0; j < numMatls; j++) {
        if (i == j) continue;
        K(i,j) = *it++;
        H(i,j) = *it1++;
      }
     }
   } else 
     throw InvalidValue("Number of exchange components don't match.");
  
}
/*---------------------------------------------------------------------
 Function~  ICE::upwindCell--
 purpose:   find the upwind cell in each direction  This is a knock off
            of Bucky's logic
 ---------------------------------------------------------------------  */
IntVector ICE::upwindCell_X(const IntVector& c, 
                            const double& var,              
                            double is_logical_R_face )     
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face); 
  IntVector tmp = c + IntVector(one_or_zero,0,0);
  return tmp;
}

IntVector ICE::upwindCell_Y(const IntVector& c, 
                            const double& var,              
                            double is_logical_R_face )     
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face); 
  IntVector tmp = c + IntVector(0,one_or_zero,0);
  return tmp;
}

IntVector ICE::upwindCell_Z(const IntVector& c, 
                            const double& var,              
                            double is_logical_R_face )     
{
  double  plus_minus_half = 0.5 * (var + d_SMALL_NUM)/fabs(var + d_SMALL_NUM);
  int one_or_zero = int(-0.5 - plus_minus_half + is_logical_R_face); 
  IntVector tmp = c + IntVector(0,0,one_or_zero);
  return tmp;
}
/*---------------------------------------------------------------------
 Function~  ICE::areAllValuesPositive--
 ---------------------------------------------------------------------  */
bool ICE::areAllValuesPositive( CCVariable<double> & src, IntVector& neg_cell )
{ 
  double numCells = 0;
  double sum_src = 0;
  //#if SCI_ASSERTION_LEVEL != 0  // turn off if assertion level = 0
  //    add this when you turn it on (#include <sci_defs.h>)
  IntVector lowIndex  = src.getLowIndex();
  IntVector highIndex = src.getHighIndex();
  for(int i=lowIndex.x();i<highIndex.x();i++) {
    for(int j=lowIndex.y();j<highIndex.y();j++) {
      for(int k=lowIndex.z();k<highIndex.z();k++) {
       sum_src += src[IntVector(i,j,k)]/fabs(src[IntVector(i,j,k)]);
       numCells++;
      }
    }
  }
  //#endif  
  // now find the first cell where the value is < 0   
  if (fabs(sum_src - numCells) > 1.0e-2) {

    for(int i=lowIndex.x();i<highIndex.x();i++) {
      for(int j=lowIndex.y();j<highIndex.y();j++) {
       for(int k=lowIndex.z();k<highIndex.z();k++) {
         if (src[IntVector(i,j,k)] < 0.0) {
           neg_cell = IntVector(i,j,k);
           return false;
         }
       }
      }
    }
  } 
  neg_cell = IntVector(0,0,0); 
  return true;      
} 

ICE::ICEModelSetup::ICEModelSetup()
{
}

ICE::ICEModelSetup::~ICEModelSetup()
{
}

void ICE::ICEModelSetup::registerTransportedVariable(const MaterialSubset* matls,
						     const VarLabel* var,
						     const VarLabel* src)
{
  TransportedVariable* t = scinew TransportedVariable;
  t->matls = matls;
  t->var = var;
  t->src = src;
  t->Lvar = VarLabel::create(var->getName()+"-L", var->typeDescription());
  tvars.push_back(t);
}


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
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

