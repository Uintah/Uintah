#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#define _CPP_CMATH
#endif
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ConservationTest.h>
#include <Packages/Uintah/CCA/Components/ICE/Diffusion.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/TurbulenceFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <float.h>

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
  switchDebugLagrangianTransportedVars = false;
  switchDebugMomentumExchange_CC       = false; 
  switchDebugSource_Sink               = false; 
  switchDebug_advance_advect           = false; 

  d_RateForm            = false;
  d_EqForm              = false; 
  d_add_heat            = false;
  d_impICE              = false;
  d_useCompatibleFluxes = false;
  d_delT_knob         = 1.0;
  d_delT_scheme       = "aggressive";
  d_surroundingMatl_indx = -9;
  d_dbgVar1   = 0;     //inputs for debugging                               
  d_dbgVar2   = 0;                                                          
  d_SMALL_NUM = 1.0e-100;                                                   
  d_TINY_RHO  = 1.0e-12;// also defined ICEMaterial.cc and MPMMaterial.cc   
  d_modelInfo = 0;
  d_modelSetup = 0;
  d_recompile = false;
  d_conservationTest = scinew conservationTest_flags();
  d_conservationTest->onOff = false;
  

  d_customInitialize_basket  = scinew customInitialize_basket();
  d_customBC_var_basket  = scinew customBC_var_basket();
  d_customBC_var_basket->Lodi_var_basket =  scinew Lodi_variable_basket();
  d_customBC_var_basket->Slip_var_basket =  scinew Slip_variable_basket();
}

ICE::~ICE()
{
  cout_doing << "Doing: ICE destructor " << endl;
  delete d_customInitialize_basket;
  delete d_customBC_var_basket->Lodi_var_basket;
  delete d_customBC_var_basket->Slip_var_basket;
  delete d_customBC_var_basket;
  delete d_conservationTest;
  delete lb;
  delete MIlb;
  delete d_advector;
  if(d_turbulence)
    delete d_turbulence;
     
  if (d_press_matl->removeReference()){
    delete d_press_matl;
  }
  cout_doing << "Doing: destorying Model Machinery " << endl;
  // delete transported Lagrangian variables
  vector<TransportedVariable*>::iterator t_iter;
  for( t_iter  = d_modelSetup->tvars.begin();
       t_iter != d_modelSetup->tvars.end(); t_iter++){
       TransportedVariable* tvar = *t_iter;
    VarLabel::destroy(tvar->var_Lagrangian);
  }
  // delete models
  for(vector<ModelInterface*>::iterator iter = d_models.begin();
      iter != d_models.end(); iter++) {
    delete *iter; 
  }
  if(d_modelInfo){
    delete d_modelInfo;
  }
  if(d_modelSetup){
    delete d_modelSetup;
  }
  releasePort("solver");
}

bool ICE::restartableTimesteps()
{
  return true;
}

double ICE::recomputeTimestep(double current_dt)
{
  return current_dt * 0.75;
}

/* _____________________________________________________________________
 Function~  ICE::problemSetup--
_____________________________________________________________________*/
void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                        SimulationStateP&   sharedState)
{
  d_sharedState = sharedState;
  lb->delTLabel = sharedState->get_delt_label();
  d_press_matl = scinew MaterialSubset();
  d_press_matl->add(0);
  d_press_matl->addReference();

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
  //  Custom BC setup
  d_customBC_var_basket->usingLodi = 
        read_LODI_BC_inputs(prob_spec, d_customBC_var_basket->Lodi_var_basket);
  d_customBC_var_basket->usingMicroSlipBCs =
        read_MicroSlip_BC_inputs(prob_spec, d_customBC_var_basket->Slip_var_basket);
  d_customBC_var_basket->usingNG_nozzle = using_NG_hack(prob_spec);
  d_customBC_var_basket->dataArchiver   = dataArchiver;
  d_customBC_var_basket->sharedState    = sharedState;

  //__________________________________
  // read in all the printData switches
  printData_problemSetup( prob_spec);

  //__________________________________
  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  cfd_ps->require("cfl",d_CFL);
  d_canAddICEMaterial=false;
  cfd_ps->get("CanAddICEMaterial",d_canAddICEMaterial);
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE"); 
  
  cfd_ice_ps->require("max_iteration_equilibration",d_max_iter_equilibration);
  
  d_advector = AdvectionFactory::create(cfd_ice_ps, d_useCompatibleFluxes);
  cout_norm << " d_use_compatibleFluxes:  " << d_useCompatibleFluxes<<endl;
  
  // Grab the solution technique
  ProblemSpecP child = cfd_ice_ps->findBlock("solution");
  if(!child){
    throw ProblemSetupException("Cannot find Solution Technique tag for ICE");
  }
  string solution_technique;
  if(!child->getAttribute("technique",solution_technique)){
    throw ProblemSetupException("Nothing specified for solution technique");
  }
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
 //  custom Initialization
  customInitialization_problemSetup(cfd_ice_ps, d_customInitialize_basket);
  
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

  //__________________________________
  //  convective heat transfer
  d_convective = false;
  exch_ps->get("do_convective_heat_transfer", d_convective);
  if(d_convective){
    exch_ps->require("convective_fluid",d_conv_fluid_matlindex);
    exch_ps->require("convective_solid",d_conv_solid_matlindex);
  }

  cout_norm << "Pulled out exchange coefficients of the input file" << endl;

  //__________________________________
  // Set up turbulence models - needs to be done after materials are initialized
  d_turbulence = TurbulenceFactory::create(cfd_ice_ps, sharedState);

  //__________________________________
  //  conservationTest
  ProblemSpecP DA_ps = prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save"); child != 0;
                    child = child->findNextBlock("save")) {
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    if (var_attr["label"] == "TotalMass"){
      d_conservationTest->onOff    = true;
    }
    if (var_attr["label"] == "CenterOfMassVelocity"){
      d_conservationTest->momentum = true;
      d_conservationTest->onOff    = true;
    }
    if (var_attr["label"] == "TotalIntEng"   || 
        var_attr["label"] == "KineticEnergy"){
      d_conservationTest->energy   = true;
      d_conservationTest->onOff    = true;
    }
    if (var_attr["label"] == "eng_exch_error"||
        var_attr["label"] == "mom_exch_error"){
      d_conservationTest->exchange = true;
      d_conservationTest->onOff    = true;
    }
  }

  //__________________________________
  // WARNINGS
  SimulationTime timeinfo(prob_spec); 
  if ( d_impICE && 
       (timeinfo.max_delt_increase  > 10  || d_delT_scheme != "conservative" )){
    cout <<"\n \n W A R N I N G: " << endl;
    cout << " When running implicit ICE you should specify "<<endl;
    cout <<" \t \t <max_delt_increase>    2.0ish  "<<endl;
    cout << "\t \t <Scheme_for_delT_calc> conservative " << endl;
    cout << " to a) prevent rapid fluctuations in the timestep and "<< endl;
    cout << "    b) to prevent outflux Vol > cell volume \n \n" <<endl;
  } 

  cout_norm << "Number of ICE materials: " 
       << d_sharedState->getNumICEMatls()<< endl;

  //__________________________________
  //  Load Model info.
  ModelMaker* modelMaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  if(modelMaker){
    modelMaker->makeModels(prob_spec, grid, sharedState, d_models);
    releasePort("ModelMaker");
    d_modelSetup = scinew ICEModelSetup();
      
    // problem setup for each model  
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      (*iter)->problemSetup(grid, sharedState, d_modelSetup);
    }
    
    //bullet proofing each transported variable must have a boundary condition.
    for(vector<TransportedVariable*>::iterator
                                    iter =  d_modelSetup->tvars.begin();
                                    iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      string Labelname = tvar->var->getName();
      is_BC_specified(prob_spec, Labelname);
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
                               lb->sp_vol_CCLabel,
                               lb->specific_heatLabel,
                               lb->gammaLabel);
  }
}
/*______________________________________________________________________
 Function~  ICE::addMaterial--
 Purpose~   read in the exchange coefficients
 _____________________________________________________________________*/
void ICE::addMaterial(const ProblemSpecP& prob_spec, GridP& grid,
                      SimulationStateP&   sharedState)
{
  d_recompile = true;
  ProblemSpecP mat_ps       =  prob_spec->findBlock("AddMaterialProperties");
  ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");  

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
    ps = ps->findNextBlock("material") ) {
    // Extract out the type of EOS and the associated parameters
    ICEMaterial *mat = scinew ICEMaterial(ps);
    sharedState->registerICEMaterial(mat);
  }

  // Pull out the exchange coefficients
  ProblemSpecP exch_ps = mat_ps->findBlock("exchange_properties");
  if (!exch_ps){
    throw ProblemSetupException("Cannot find exchange_properties tag");
  }
  
  ProblemSpecP exch_co_ps = exch_ps->findBlock("exchange_coefficients");
  d_K_mom.clear();
  d_K_heat.clear();
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

  d_convective = false;
  exch_ps->get("do_convective_heat_transfer", d_convective);
  if(d_convective){
    exch_ps->require("convective_fluid",d_conv_fluid_matlindex);
    exch_ps->require("convective_solid",d_conv_solid_matlindex);
  }
  // problem setup for each model  
  for(vector<ModelInterface*>::iterator iter = d_models.begin();
     iter != d_models.end(); iter++){
    (*iter)->activateModel(grid, sharedState, d_modelSetup);
  }
}
/*______________________________________________________________________
 Function~  ICE::updateExchangeCoefficients--
 Purpose~   read in the exchange coefficients after a material has been
            dynamically added
 _____________________________________________________________________*/
void ICE::updateExchangeCoefficients(const ProblemSpecP& prob_spec, GridP& grid,
                                     SimulationStateP&   sharedState)
{
  cout << "Updating Ex Coefficients" << endl;
  ProblemSpecP mat_ps  =  prob_spec->findBlock("AddMaterialProperties");
  ProblemSpecP exch_ps = mat_ps->findBlock("exchange_properties");
  if (!exch_ps){
    throw ProblemSetupException("Cannot find exchange_properties tag");
  }
  
  ProblemSpecP exch_co_ps = exch_ps->findBlock("exch_coef_after_MPM_add");
  d_K_mom.clear();
  d_K_heat.clear();
  exch_co_ps->require("momentum",d_K_mom);
  exch_co_ps->require("heat",    d_K_heat);

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
}
/*______________________________________________________________________
 Function~  ICE::scheduleInitializeAddedMaterial--
 _____________________________________________________________________*/
void ICE::scheduleInitializeAddedMaterial(const LevelP& level,SchedulerP& sched)
{
                                                                                
  cout_doing << "Doing ICE::scheduleInitializeAddedMaterial " << endl;
  Task* t = scinew Task("ICE::actuallyInitializeAddedMaterial",
                  this, &ICE::actuallyInitializeAddedMaterial);

  int numALLMatls = d_sharedState->getNumMatls();
  MaterialSubset* add_matl = scinew MaterialSubset();
  cout << "Added Material = " << numALLMatls-1 << endl;
  add_matl->add(numALLMatls-1);
  add_matl->addReference();
                                                                                
  t->computes(lb->vel_CCLabel,        add_matl);
  t->computes(lb->rho_CCLabel,        add_matl);
  t->computes(lb->temp_CCLabel,       add_matl);
  t->computes(lb->sp_vol_CCLabel,     add_matl);
  t->computes(lb->vol_frac_CCLabel,   add_matl);
  t->computes(lb->rho_micro_CCLabel,  add_matl);
  t->computes(lb->speedSound_CCLabel, add_matl);
  t->computes(lb->thermalCondLabel,   add_matl);
  t->computes(lb->viscosityLabel,     add_matl);
  t->computes(lb->gammaLabel,         add_matl);
  t->computes(lb->specific_heatLabel, add_matl);

  sched->addTask(t, level->eachPatch(), d_sharedState->allICEMaterials());

  // The task will have a reference to add_matl
  if (add_matl->removeReference()){
    delete add_matl; // shouln't happen, but...
  }
}

/*______________________________________________________________________
 Function~  ICE::actuallyInitializeAddedMaterial--
 _____________________________________________________________________*/
void ICE::actuallyInitializeAddedMaterial(const ProcessorGroup*, 
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse*, 
                                          DataWarehouse* new_dw)
{
  new_dw->unfinalize();
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing InitializeAddedMaterial on patch " << patch->getID() 
         << "\t\t\t ICE" << endl;
    CCVariable<double>  rho_micro, sp_vol_CC, rho_CC, Temp_CC, thermalCond;
    CCVariable<double>  speedSound,vol_frac_CC, cv, gamma, viscosity,dummy;
    CCVariable<Vector>  vel_CC;
    
    //__________________________________
    //  Thermo and transport properties
    int m = d_sharedState->getNumICEMatls() - 1;
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    int indx= ice_matl->getDWIndex();
    cout << "Added Material Index = " << indx << endl;
    new_dw->allocateAndPut(viscosity,     lb->viscosityLabel,    indx,patch);
    new_dw->allocateAndPut(thermalCond,   lb->thermalCondLabel,  indx,patch);
    new_dw->allocateAndPut(cv,            lb->specific_heatLabel,indx,patch);
    new_dw->allocateAndPut(gamma,         lb->gammaLabel,        indx,patch);
    new_dw->allocateAndPut(rho_micro,     lb->rho_micro_CCLabel, indx,patch); 
    new_dw->allocateAndPut(sp_vol_CC,     lb->sp_vol_CCLabel,    indx,patch); 
    new_dw->allocateAndPut(rho_CC,        lb->rho_CCLabel,       indx,patch); 
    new_dw->allocateAndPut(Temp_CC,       lb->temp_CCLabel,      indx,patch); 
    new_dw->allocateAndPut(speedSound,    lb->speedSound_CCLabel,indx,patch); 
    new_dw->allocateAndPut(vol_frac_CC,   lb->vol_frac_CCLabel,  indx,patch); 
    new_dw->allocateAndPut(vel_CC,        lb->vel_CCLabel,       indx,patch);
    new_dw->allocateTemporary(dummy, patch);
    cout << "Done allocateAndPut Index = " << indx << endl;

    gamma.initialize(       ice_matl->getGamma());
    cv.initialize(          ice_matl->getSpecificHeat());    
    viscosity.initialize  ( ice_matl->getViscosity());
    thermalCond.initialize( ice_matl->getThermalConductivity());

    int numALLMatls = d_sharedState->getNumMatls();
    ice_matl->initializeCells(rho_micro,  rho_CC,
                              Temp_CC,    speedSound, 
                              vol_frac_CC, vel_CC, 
                              dummy, numALLMatls, patch, new_dw);

    setBC(rho_CC,     "Density",     patch, d_sharedState, indx, new_dw);
    setBC(rho_micro,  "Density",     patch, d_sharedState, indx, new_dw);
    setBC(Temp_CC,    "Temperature", patch, d_sharedState, indx, new_dw);
    setBC(speedSound, "zeroNeumann", patch, d_sharedState, indx, new_dw); 
    setBC(vel_CC,     "Velocity",    patch, d_sharedState, indx, new_dw); 
            
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      sp_vol_CC[c] = 1.0/rho_micro[c];
      vol_frac_CC[c] = rho_CC[c]*sp_vol_CC[c];  //needed for LODI BCs
    }
  }  // patch loop 
  new_dw->refinalize();
}

/* _____________________________________________________________________
 Function~  ICE::scheduleInitialize--
 Notes:     This task actually schedules several tasks.
_____________________________________________________________________*/
void ICE::scheduleInitialize(const LevelP& level,SchedulerP& sched)
{

  cout_doing << "Doing ICE::scheduleInitialize " << endl;
  Task* t = scinew Task("ICE::actuallyInitialize",
                  this, &ICE::actuallyInitialize);

  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
  t->computes(lb->vel_CCLabel);
  t->computes(lb->rho_CCLabel); 
  t->computes(lb->temp_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->rho_micro_CCLabel);
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->thermalCondLabel);
  t->computes(lb->viscosityLabel);
  t->computes(lb->gammaLabel);
  t->computes(lb->specific_heatLabel);
  t->computes(lb->press_CCLabel,     d_press_matl, oims);
  t->computes(lb->initialGuessLabel, d_press_matl, oims); 
  
  sched->addTask(t, level->eachPatch(), d_sharedState->allICEMaterials());

  //__________________________________
  // Models Initialization
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleInitialize(sched, level, d_modelInfo);
      model->d_dataArchiver = dataArchiver;
    }
  }
 
  //__________________________________
  // Make adjustments to the hydrostatic pressure
  // and temperature fields.  You need to do this
  // after the models have initialized the flowfield
  Vector grav = d_sharedState->getGravity();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  if (grav.length() > 0 ) {
    cout_doing << "Doing ICE::scheduleHydroStaticAdj " << endl;
    Task* t2 = scinew Task("ICE::initializeSubTask_hydrostaticAdj",
                     this, &ICE::initializeSubTask_hydrostaticAdj);
    Ghost::GhostType  gn  = Ghost::None;
    t2->requires(Task::NewDW,lb->gammaLabel,         ice_matls_sub, gn);
    t2->requires(Task::NewDW,lb->specific_heatLabel, ice_matls_sub, gn);
   
    t2->modifies(lb->rho_micro_CCLabel);
    t2->modifies(lb->temp_CCLabel);
    t2->modifies(lb->press_CCLabel, d_press_matl, oims); 

    cout << "ICE: press_matl: " << d_press_matl->size()<< "\n";

    sched->addTask(t2, level->eachPatch(), ice_matls);
  }
  
}

/* _____________________________________________________________________
 Function~  ICE::restartInitialize--
 Purpose:   Set variables that are normally set during the initialization
            phase, but get wiped clean when you restart
_____________________________________________________________________*/
void ICE::restartInitialize()
{
    cout_doing << "Doing restartInitialize "<< "\t\t\t ICE" << endl;
  // disregard initial dt when restarting
  d_initialDt = 10000.0;
  
  //__________________________________
  // Models Initialization
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->d_dataArchiver = dataArchiver;
    }
  }  
  // which matl index is the surrounding matl.
  int numMatls    = d_sharedState->getNumICEMatls();
  for (int m = 0; m < numMatls; m++ ) {
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    if(ice_matl->isSurroundingMatl()) {
      d_surroundingMatl_indx = m;
    } 
  }
  // --------bulletproofing
  Vector grav     = d_sharedState->getGravity();
  if (grav.length() >0.0 && d_surroundingMatl_indx == -9)  {
    throw ProblemSetupException("ERROR ICE::restartInitialize \n"
          "You must have \n" 
          "       <isSurroundingMatl> true </isSurroundingMatl> \n "
          "specified inside the ICE material that is the background matl\n");
  }
}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeStableTimestep--
_____________________________________________________________________*/
void ICE::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  Task* t = 0;
  if (d_EqForm) {             // EQ 
    cout_doing << "ICE::scheduleComputeStableTimestep \t\tL-"
               <<level->getIndex() << endl;
    t = scinew Task("ICE::actuallyComputeStableTimestep",
                     this, &ICE::actuallyComputeStableTimestep);
  } else if (d_RateForm) {    // RF
    cout_doing << "ICE::scheduleComputeStableTimestepRF " << endl;
    t = scinew Task("ICE::actuallyComputeStableTimestepRF",
                      this, &ICE::actuallyComputeStableTimestepRF);
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
                            // COMMON TO EQ AND RATE FORM
  t->requires(Task::NewDW, lb->vel_CCLabel,        gac, 1);  
  t->requires(Task::NewDW, lb->speedSound_CCLabel, gac, 1);
  t->requires(Task::NewDW, lb->thermalCondLabel,   gn,  0);
  t->requires(Task::NewDW, lb->gammaLabel,         gn,  0);
  t->requires(Task::NewDW, lb->specific_heatLabel, gn,  0);
                            
  if (d_EqForm){            // EQ      
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,   gn,  0);   
    t->requires(Task::NewDW, lb->viscosityLabel,   gn,  0);        
  } else if (d_RateForm){   // RATE FORM
    t->requires(Task::NewDW, lb->sp_vol_CCLabel,   gac, 1);   
  }
  t->computes(d_sharedState->get_delt_label());
  sched->addTask(t,level->eachPatch(), ice_matls); 
  
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
/* _____________________________________________________________________
 Function~  ICE::scheduleTimeAdvance--
_____________________________________________________________________*/
void
ICE::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, 
                          int step, int nsteps )
{
  cout_doing << "ICE::scheduleTimeAdvance\t\t\tL-" <<level->getIndex()<< endl;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  

  MaterialSubset* one_matl = d_press_matl;
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();

  double AMR_subCycleProgressVar = double(step)/double(nsteps);
  
  if(d_turbulence){
    // The turblence model is also called directly from
    // accumlateMomentumSourceSinks.  This method just allows other
    // quantities (such as variance) to be computed
    d_turbulence->scheduleTurbulence1(sched, patches, ice_matls);
  }
  vector<PatchSubset*> maxMach_PSS(Patch::numFaces);
  scheduleMaxMach_on_Lodi_BC_Faces(       sched, level,   ice_matls, 
                                                          maxMach_PSS);
                                                          
  scheduleComputeThermoTransportProperties(sched, level,  ice_matls);
  
  scheduleComputePressure(                sched, patches, d_press_matl,
                                                          all_matls);

  if (d_RateForm) {
    schedulecomputeDivThetaVel_CC(        sched, patches, ice_matls_sub,        
                                                          mpm_matls_sub,        
                                                          all_matls);           
  }  

  scheduleComputeTempFC(                   sched, patches, ice_matls_sub,  
                                                           mpm_matls_sub,
                                                           all_matls);    
                                                                 
  scheduleComputeModelSources(             sched, level,   all_matls);

  scheduleUpdateVolumeFraction(            sched, level,   all_matls);


  scheduleComputeVel_FC(                   sched, patches,ice_matls_sub, 
                                                         mpm_matls_sub, 
                                                         d_press_matl,    
                                                         all_matls,     
                                                         false);        

  scheduleAddExchangeContributionToFCVel( sched, patches,ice_matls_sub,
                                                         all_matls,
                                                         false);
                                                          
  if(d_impICE) {        //  I M P L I C I T
  
    scheduleSetupRHS(                     sched, patches,  one_matl, 
                                                           all_matls,
                                                           false);
  
    scheduleImplicitPressureSolve(         sched, level,   patches,
                                                           one_matl,      
                                                           d_press_matl,    
                                                           ice_matls_sub,  
                                                           mpm_matls_sub, 
                                                           all_matls);
                                                           
    scheduleComputeDel_P(                   sched,  level, patches,  
                                                           one_matl,
                                                           d_press_matl,
                                                           all_matls);    
                                                           
  }                    

  if(!d_impICE){         //  E X P L I C I T
    scheduleComputeDelPressAndUpdatePressCC(sched, patches,d_press_matl,     
                                                           ice_matls_sub,  
                                                           mpm_matls_sub,  
                                                           all_matls);     
  }
  
  scheduleComputePressFC(                 sched, patches, d_press_matl,
                                                          all_matls);

  scheduleAccumulateMomentumSourceSinks(  sched, patches, d_press_matl,
                                                          ice_matls_sub,
                                                          mpm_matls_sub,
                                                          all_matls);
  scheduleAccumulateEnergySourceSinks(    sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          d_press_matl,
                                                          all_matls);

  scheduleComputeLagrangianValues(        sched, patches, all_matls);

  scheduleAddExchangeToMomentumAndEnergy( sched, patches, ice_matls_sub,
                                                          mpm_matls_sub,
                                                          d_press_matl,
                                                          all_matls);
                                                           
  scheduleComputeLagrangianSpecificVolume(sched, patches, ice_matls_sub,
                                                          mpm_matls_sub, 
                                                          d_press_matl,
                                                          all_matls);

  scheduleComputeLagrangian_Transported_Vars(sched, patches,
                                                          all_matls);
                                   
  scheduleAdvectAndAdvanceInTime(         sched, patches, AMR_subCycleProgressVar,
                                                          ice_matls_sub,
                                                          mpm_matls_sub,
                                                          d_press_matl,
                                                          all_matls);
                                                          
  scheduleTestConservation(               sched, patches, ice_matls_sub,
                                                          all_matls); 

  if(d_canAddICEMaterial){
    //  This checks to see if the model on THIS patch says that it's
    //  time to add a new material
    scheduleCheckNeedAddMaterial(           sched, level,   all_matls);

    //  This one checks to see if the model on ANY patch says that it's
    //  time to add a new material
    scheduleSetNeedAddMaterialFlag(         sched, level,   all_matls);
  }

}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeThermoTransportProperties--
_____________________________________________________________________*/
void ICE::scheduleComputeThermoTransportProperties(SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* ice_matls)
{ 
  Task* t;
  cout_doing << "ICE::schedulecomputeThermoTransportProperties" << endl;
  t = scinew Task("ICE::computeThermoTransportProperties", 
            this, &ICE::computeThermoTransportProperties);   
  t->computes(lb->viscosityLabel);
  t->computes(lb->thermalCondLabel);
  t->computes(lb->gammaLabel);
  t->computes(lb->specific_heatLabel);
  
  sched->addTask(t, level->eachPatch(), ice_matls);

  //__________________________________
  //  Each model *can* modify the properties
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
                                          iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      if(model-> computesThermoTransportProps() ) {
         model->scheduleModifyThermoTransportProperties(sched,level,ice_matls);
      } 
    }
  }
}

/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW,lb->press_CCLabel, press_matl, oims, gn);
  t->requires(Task::OldDW,lb->rho_CCLabel,               gn);
  t->requires(Task::OldDW,lb->temp_CCLabel,              gn); 
  t->requires(Task::OldDW,lb->sp_vol_CCLabel,            gn);
  t->requires(Task::NewDW,lb->gammaLabel,                gn);
  t->requires(Task::NewDW,lb->specific_heatLabel,        gn);
  
  t->computes(lb->f_theta_CCLabel); 
  t->computes(lb->speedSound_CCLabel);
  t->computes(lb->vol_frac_CCLabel);
  t->computes(lb->sp_vol_CCLabel);
  t->computes(lb->rho_CCLabel);
  t->computes(lb->press_equil_CCLabel, press_matl, oims);
  t->computes(lb->press_CCLabel,       press_matl, oims);  // needed by implicit
 
  if (d_RateForm) {     // RATE FORM
    t->computes(lb->matl_press_CCLabel, press_matl,oims);
  }

  computesRequires_CustomBCs(t, "EqPress", lb, ice_matls->getUnion(),
                            d_customBC_var_basket); 
  
  sched->addTask(t, patches, ice_matls);
}

/* _____________________________________________________________________
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
/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
//  t->requires(Task::OldDW, lb->delTLabel);    For AMR
  t->requires(Task::NewDW,lb->press_CCLabel,       press_matl, oims, gac,1);
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
/* _____________________________________________________________________
 Function~  ICE::scheduleAddExchangeContributionToFCVel--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeContributionToFCVel(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* ice_matls,
                                           const MaterialSet* all_matls,
                                           const bool recursion)
{
  cout_doing << "ICE::scheduleAddExchangeContributionToFCVel" << endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel, recursion);

//  task->requires(Task::OldDW, lb->delTLabel);    FOR AMR 
  Ghost::GhostType  gac = Ghost::AroundCells;
  task->requires(Task::NewDW,lb->sp_vol_CCLabel,    /*all_matls*/gac,1);
  task->requires(Task::NewDW,lb->vol_frac_CCLabel,  /*all_matls*/gac,1);
  task->requires(Task::NewDW,lb->uvel_FCLabel,      /*all_matls*/gac,2);
  task->requires(Task::NewDW,lb->vvel_FCLabel,      /*all_matls*/gac,2);
  task->requires(Task::NewDW,lb->wvel_FCLabel,      /*all_matls*/gac,2);
  
  computesRequires_CustomBCs(task, "velFC_Exchange", lb, ice_matls,
                                d_customBC_var_basket);

  task->computes(lb->sp_volX_FCLabel);
  task->computes(lb->sp_volY_FCLabel);
  task->computes(lb->sp_volZ_FCLabel); 
  task->computes(lb->uvel_FCMELabel);
  task->computes(lb->vvel_FCMELabel);
  task->computes(lb->wvel_FCMELabel);
  
  sched->addTask(task, patches, all_matls);
}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeModelSources--
_____________________________________________________________________*/
void ICE::scheduleComputeModelSources(SchedulerP& sched, 
                                      const LevelP& level,
                                      const MaterialSet* matls)
{
  if(d_models.size() != 0){
    cout_doing << "ICE::scheduleModelMassExchange" << endl;
    Task* task = scinew Task("ICE::zeroModelSources",this, 
                             &ICE::zeroModelSources);
    task->computes(lb->modelMass_srcLabel);
    task->computes(lb->modelMom_srcLabel);
    task->computes(lb->modelEng_srcLabel);
    task->computes(lb->modelVol_srcLabel);
    for(vector<TransportedVariable*>::iterator
                                    iter =  d_modelSetup->tvars.begin();
                                    iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      if(tvar->src){
        task->computes(tvar->src, tvar->matls);
      }
    }
    sched->addTask(task, level->eachPatch(), matls);

    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleComputeModelSources(sched, level, d_modelInfo);
    }
  }
}

/* _____________________________________________________________________
 Function~  ICE::scheduleUpdateVolumeFraction--
_____________________________________________________________________*/
void ICE::scheduleUpdateVolumeFraction(SchedulerP& sched, const LevelP& level,
                                const MaterialSet* matls)
{
  if(d_models.size() != 0){
    cout_doing << "ICE::scheduleUpdateVolumeFraction" << endl;
    Task* task = scinew Task("ICE::updateVolumeFraction",
                          this, &ICE::updateVolumeFraction);
    Ghost::GhostType  gn = Ghost::None;  
    task->requires( Task::NewDW, lb->sp_vol_CCLabel,     gn);
    task->requires( Task::NewDW, lb->rho_CCLabel,        gn);    
    task->requires( Task::NewDW, lb->modelVol_srcLabel,  gn);    
    task->modifies(lb->vol_frac_CCLabel);

    sched->addTask(task, level->eachPatch(), matls);
  }
}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeDelPressAndUpdatePressCC--
_____________________________________________________________________*/
void ICE::scheduleComputeDelPressAndUpdatePressCC(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* press_matl,
                                            const MaterialSubset* ice_matls,
					    const MaterialSubset* /*mpm_matls*/,
                                            const MaterialSet* matls)
{
  cout_doing << "ICE::scheduleComputeDelPressAndUpdatePressCC" << endl;
  Task *task = scinew Task("ICE::computeDelPressAndUpdatePressCC",
                            this, &ICE::computeDelPressAndUpdatePressCC);
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;  
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
//  task->requires( Task::OldDW, lb->delTLabel);    FOR AMR
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2);
  task->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->sp_vol_CCLabel,     gn);
  task->requires( Task::NewDW, lb->rho_CCLabel,        gn);    
  task->requires( Task::NewDW, lb->speedSound_CCLabel, gn);
  //__________________________________
  if(d_models.size() > 0){
    task->requires(Task::NewDW, lb->modelVol_srcLabel,  gn);
    task->requires(Task::NewDW, lb->modelMass_srcLabel, gn);
  }
  
  computesRequires_CustomBCs(task, "update_press_CC", lb, ice_matls,
                             d_customBC_var_basket);
  
  task->modifies(lb->press_CCLabel,        press_matl, oims);
  task->computes(lb->delP_DilatateLabel,   press_matl, oims);
  task->computes(lb->delP_MassXLabel,      press_matl, oims);
  task->computes(lb->term2Label,           press_matl, oims);
  task->computes(lb->term3Label,           press_matl, oims);
  task->computes(lb->sum_rho_CCLabel,      press_matl, oims);
  task->computes(lb->vol_fracX_FCLabel);
  task->computes(lb->vol_fracY_FCLabel);
  task->computes(lb->vol_fracZ_FCLabel);
  
  sched->setRestartable(true);
  sched->addTask(task, patches, matls);
}

/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  task->requires(Task::NewDW,lb->press_CCLabel,   press_matl,oims, gac,1);
  task->requires(Task::NewDW,lb->sum_rho_CCLabel, press_matl,oims, gac,1);

  task->computes(lb->pressX_FCLabel, press_matl, oims);
  task->computes(lb->pressY_FCLabel, press_matl, oims);
  task->computes(lb->pressZ_FCLabel, press_matl, oims);

  sched->addTask(task, patches, matls);
}

/* _____________________________________________________________________
 Function~  ICE::scheduleAccumulateMomentumSourceSinks--
_____________________________________________________________________*/
void
ICE::scheduleAccumulateMomentumSourceSinks(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSubset* press_matl,
					   const MaterialSubset* ice_matls,
					   const MaterialSubset* /*mpm_matls_sub*/,
					   const MaterialSet* matls)
{
  Task* t;
  cout_doing << "ICE::scheduleAccumulateMomentumSourceSinks" << endl; 
  t = scinew Task("ICE::accumulateMomentumSourceSinks", 
            this, &ICE::accumulateMomentumSourceSinks);

                       // EQ  & RATE FORM     
//  t->requires(Task::OldDW, lb->delTLabel);  FOR AMR
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
  t->requires(Task::NewDW,lb->pressX_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->pressY_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->pressZ_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->viscosityLabel,   ice_matls, gac, 2);
  t->requires(Task::OldDW,lb->vel_CCLabel,      ice_matls, gac, 2);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,   ice_matls, gac, 2);
  t->requires(Task::NewDW,lb->rho_CCLabel,                   gac, 2);
  t->requires(Task::NewDW, lb->vol_fracX_FCLabel, ice_matls, gac,2);
  t->requires(Task::NewDW, lb->vol_fracY_FCLabel, ice_matls, gac,2);
  t->requires(Task::NewDW, lb->vol_fracZ_FCLabel, ice_matls, gac,2);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel, Ghost::None);
  
  if (d_RateForm) {   // RATE FORM
    t->requires(Task::NewDW,lb->press_diffX_FCLabel, gac, 1);
    t->requires(Task::NewDW,lb->press_diffY_FCLabel, gac, 1);
    t->requires(Task::NewDW,lb->press_diffZ_FCLabel, gac, 1);
  }

  if(d_turbulence){
    t->requires(Task::NewDW,lb->uvel_FCMELabel,   ice_matls, gac, 3);
    t->requires(Task::NewDW,lb->vvel_FCMELabel,   ice_matls, gac, 3);
    t->requires(Task::NewDW,lb->wvel_FCMELabel,   ice_matls, gac, 3);
    t->computes(lb->turb_viscosity_CCLabel,   ice_matls);
  } 

  t->computes(lb->mom_source_CCLabel);
  t->computes(lb->press_force_CCLabel);
  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
//  t->requires(Task::OldDW, lb->delTLabel);  FOR AMR
  t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims, gn);
  t->requires(Task::NewDW, lb->speedSound_CCLabel,           gn);
  t->requires(Task::OldDW, lb->temp_CCLabel,      ice_matls, gac,1);
  t->requires(Task::NewDW, lb->thermalCondLabel,  ice_matls, gac,1);
  t->requires(Task::NewDW, lb->rho_CCLabel,                  gac,1);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,               gac,1);
  t->requires(Task::NewDW, lb->vol_fracX_FCLabel, ice_matls, gac,2);
  t->requires(Task::NewDW, lb->vol_fracY_FCLabel, ice_matls, gac,2);
  t->requires(Task::NewDW, lb->vol_fracZ_FCLabel, ice_matls, gac,2);

  if (d_EqForm) {       //EQ FORM
    t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,oims, gn);
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,             gn);
  }
  if (d_RateForm) {     //RATE FORM
    t->requires(Task::NewDW, lb->f_theta_CCLabel,            gn,0);
    t->requires(Task::OldDW, lb->vel_CCLabel,     ice_matls, gn,0);    
    t->requires(Task::NewDW, lb->vel_CCLabel,     mpm_matls, gn,0);    
    t->requires(Task::NewDW, lb->pressX_FCLabel,  press_matl,oims, gac,1);
    t->requires(Task::NewDW, lb->pressY_FCLabel,  press_matl,oims, gac,1);
    t->requires(Task::NewDW, lb->pressZ_FCLabel,  press_matl,oims, gac,1);
    t->requires(Task::NewDW, lb->uvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->vvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->wvel_FCMELabel,             gac,1);
    t->requires(Task::NewDW, lb->press_diffX_FCLabel,        gac,1);     
    t->requires(Task::NewDW, lb->press_diffY_FCLabel,        gac,1);     
    t->requires(Task::NewDW, lb->press_diffZ_FCLabel,        gac,1);
    t->requires(Task::NewDW, lb->vol_frac_CCLabel,           gac,1);          
  }
  t->computes(lb->int_eng_source_CCLabel);
  t->computes(lb->heatCond_src_CCLabel);
  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
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
  t->requires(Task::NewDW,lb->specific_heatLabel,      gn); 
  t->requires(Task::NewDW,lb->rho_CCLabel,             gn);
  t->requires(Task::OldDW,lb->vel_CCLabel,             gn);
  t->requires(Task::OldDW,lb->temp_CCLabel,            gn);
  t->requires(Task::NewDW,lb->mom_source_CCLabel,      gn);
  t->requires(Task::NewDW,lb->int_eng_source_CCLabel,  gn);

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

/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

//  t->requires(Task::OldDW, lb->delTLabel);  for AMR                         
  t->requires(Task::NewDW, lb->rho_CCLabel,               gn);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,            gn);    
  t->requires(Task::NewDW, lb->Tdot_CCLabel,              gn);  
  t->requires(Task::NewDW, lb->f_theta_CCLabel,           gn);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,          gac,1);
  t->requires(Task::OldDW, lb->temp_CCLabel,   ice_matls, gn);
  t->requires(Task::NewDW, lb->temp_CCLabel,   mpm_matls, gn); 
  if (d_RateForm) {         // RATE FORM
    t->requires(Task::NewDW, lb->uvel_FCMELabel,      gac,1);
    t->requires(Task::NewDW, lb->vvel_FCMELabel,      gac,1);
    t->requires(Task::NewDW, lb->wvel_FCMELabel,      gac,1);        
  }
  if (d_EqForm) {           // EQ FORM
    t->requires(Task::NewDW, lb->speedSound_CCLabel,           gn);
    t->requires(Task::NewDW, lb->specific_heatLabel,ice_matls, gn);
    t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,oims,gn);
  }
  if(d_models.size() > 0){
    t->requires(Task::NewDW, lb->modelVol_srcLabel,    gn);
  }

  t->computes(lb->sp_vol_L_CCLabel);                             
  t->computes(lb->sp_vol_src_CCLabel);                        

  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
 Function~  ICE:: scheduleComputeTransportedLagrangianValues--
 Purpose:   For each transported variable compute the lagrangian value
            q_L_CC = (q_old + q_src) * mass_L
 Note:      Be care
_____________________________________________________________________*/
void ICE::scheduleComputeLagrangian_Transported_Vars(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
  if(d_models.size() > 0 && d_modelSetup->tvars.size() > 0){
    cout_doing << "ICE::scheduleComputeLagrangian_Transported_Vars" << endl;
    Task* t = scinew Task("ICE::computeLagrangian_Transported_Vars",
                     this,&ICE::computeLagrangian_Transported_Vars);
    Ghost::GhostType  gn  = Ghost::None;

    t->requires(Task::NewDW,lb->mass_L_CCLabel, gn);
    
    // computes and requires for each transported variable
    vector<TransportedVariable*>::iterator t_iter;
    for( t_iter = d_modelSetup->tvars.begin();
        t_iter != d_modelSetup->tvars.end(); t_iter++){
      TransportedVariable* tvar = *t_iter;
                         // require q_old
      t->requires(Task::OldDW, tvar->var,   tvar->matls, gn, 0);

      if(tvar->src){     // require q_src
        t->requires(Task::NewDW, tvar->src, tvar->matls, gn, 0);
      }
      t->computes(tvar->var_Lagrangian, tvar->matls);
    }
    sched->addTask(t, patches, matls);
  }
}
/* _____________________________________________________________________
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
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
//  t->requires(Task::OldDW, d_sharedState->get_delt_label()); for AMR
 
  if(d_convective){
    Ghost::GhostType  gac  = Ghost::AroundCells; 
    t->requires(Task::NewDW,MIlb->gMassLabel,       mpm_matls,  gac, 1);      
    t->requires(Task::OldDW,MIlb->NC_CCweightLabel, press_matl, gac, 1);
  }
                                // I C E
  t->requires(Task::OldDW,  lb->temp_CCLabel,      ice_matls, gn);
  t->requires(Task::NewDW,  lb->specific_heatLabel,ice_matls, gn);
  t->requires(Task::NewDW,  lb->gammaLabel,        ice_matls, gn);
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
    t->requires(Task::NewDW, lb->press_CCLabel,     press_matl, gn, oims);
    t->requires(Task::OldDW, lb->vel_CCLabel,       ice_matls,  gn); 
  }

  computesRequires_CustomBCs(t, "CC_Exchange", lb, ice_matls,
                             d_customBC_var_basket); 

  t->computes(lb->Tdot_CCLabel);
  t->computes(lb->mom_L_ME_CCLabel);      
  t->computes(lb->eng_L_ME_CCLabel); 
  
  if (mpm_matls->size() > 0){  
    t->modifies(lb->temp_CCLabel, mpm_matls);
    t->modifies(lb->vel_CCLabel,  mpm_matls);
  }
  sched->addTask(t, patches, all_matls);
}
/* _____________________________________________________________________ 
 Function~  ICE::scheduleMaxMach_on_Lodi_BC_Faces--
 Purpose    compute the reducton variable max_mach_<face>
            on Lodi boundary faces
_____________________________________________________________________*/
void ICE::scheduleMaxMach_on_Lodi_BC_Faces(SchedulerP& sched, 
                                     const LevelP& level,
                                     const MaterialSet* matls,
                                     vector<PatchSubset*> & /*maxMach_PSS*/)
{ 
  if(d_customBC_var_basket->usingLodi) {
    cout_doing << "ICE::scheduleMaxMach_on_Lodi_BC_Faces" << endl;
    Task* task = scinew Task("ICE::maxMach_on_Lodi_BC_Faces",
                       this, &ICE::maxMach_on_Lodi_BC_Faces);
    Ghost::GhostType  gn = Ghost::None;  
    task->requires( Task::OldDW, lb->vel_CCLabel,        gn);   
    task->requires( Task::OldDW, lb->speedSound_CCLabel, gn);
    
    // Reduction variables with patch subsets don't work with mpi.
    //Lodi_maxMach_patchSubset(level, d_sharedState, maxMach_PSS);
                             
    //__________________________________
    // loop over the Lodi face
    //  add computes for maxMach
    vector<Patch::FaceType>::iterator f ;
         
    for( f = d_customBC_var_basket->Lodi_var_basket->LodiFaces.begin();
         f!= d_customBC_var_basket->Lodi_var_basket->LodiFaces.end(); ++f) {
         
      VarLabel* V_Label = getMaxMach_face_VarLabel(*f);
      task->computes(V_Label, matls->getUnion());
    }
    sched->addTask(task, level->eachPatch(), matls);
  }
}

/* _____________________________________________________________________
 Function~  ICE::scheduleAdvectAndAdvanceInTime--
_____________________________________________________________________*/
void ICE::scheduleAdvectAndAdvanceInTime(SchedulerP& sched,
                                    const PatchSet* patch_set,
                                    const double AMR_subCycleProgressVar,
                                    const MaterialSubset* ice_matlsub,
                                    const MaterialSubset* /*mpm_matls*/,
                                    const MaterialSubset* /*press_matl*/,
                                    const MaterialSet* ice_matls)
{
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  Ghost::GhostType  gn   = Ghost::None;
  
  cout_doing << "ICE::scheduleAdvectAndAdvanceInTime" << endl;
  Task* task = scinew Task("ICE::advectAndAdvanceInTime",
                     this, &ICE::advectAndAdvanceInTime, AMR_subCycleProgressVar);
//  task->requires(Task::OldDW, lb->delTLabel);     for AMR
  task->requires(Task::NewDW, lb->uvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->vvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->wvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->mom_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,      gac,2);
  task->requires(Task::NewDW, lb->eng_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->sp_vol_L_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->specific_heatLabel,  gac,2);  
  task->requires(Task::NewDW, lb->speedSound_CCLabel,  gn, 0);
  
  computesRequires_CustomBCs(task, "Advection", lb, ice_matlsub, 
                             d_customBC_var_basket);
  
  task->modifies(lb->rho_CCLabel);
  task->modifies(lb->sp_vol_CCLabel);
  task->computes(lb->temp_CCLabel);
  task->computes(lb->vel_CCLabel);
  task->computes(lb->machLabel);  

/*`==========TESTING==========*/
#if 0
  task->computes(lb->mass_X_FC_fluxLabel);
  task->computes(lb->mass_Y_FC_fluxLabel);
  task->computes(lb->mass_Z_FC_fluxLabel);

  task->computes(lb->mom_X_FC_fluxLabel);
  task->computes(lb->mom_Y_FC_fluxLabel);
  task->computes(lb->mom_Z_FC_fluxLabel);
  
  task->computes(lb->sp_vol_X_FC_fluxLabel);
  task->computes(lb->sp_vol_Y_FC_fluxLabel);
  task->computes(lb->sp_vol_Z_FC_fluxLabel);
  
  task->computes(lb->int_eng_X_FC_fluxLabel);
  task->computes(lb->int_eng_Y_FC_fluxLabel);
  task->computes(lb->int_eng_Z_FC_fluxLabel);  
  
  if(AMR_subCycleProgressVar> 0){
    task->requires(Task::OldDW, lb->mass_X_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->mass_Y_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->mass_Z_FC_fluxLabel, gn, 0);
    
    task->requires(Task::OldDW, lb->mom_X_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->mom_Y_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->mom_Z_FC_fluxLabel, gn, 0);
    
    task->requires(Task::OldDW, lb->sp_vol_X_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->sp_vol_Y_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->sp_vol_Z_FC_fluxLabel, gn, 0);
    
    task->requires(Task::OldDW, lb->int_eng_X_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->int_eng_Y_FC_fluxLabel, gn, 0);
    task->requires(Task::OldDW, lb->int_eng_Z_FC_fluxLabel, gn, 0);
  }
  // needto do something for the scalar-f variables.
#endif
/*===========TESTING==========`*/  
  
  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;
    
    for(iter = d_modelSetup->tvars.begin();
        iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var_Lagrangian, tvar->matls, gac, 2);
      task->computes(tvar->var,   tvar->matls);
    }
  }
  
  sched->setRestartable(true);
  sched->addTask(task, patch_set, ice_matls);
}
/* _____________________________________________________________________
 Function~  ICE::scheduleTestConservation--
_____________________________________________________________________*/
void ICE::scheduleTestConservation(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSubset* ice_matls,
                                   const MaterialSet* all_matls)
{
  if(d_conservationTest->onOff) {
    cout_doing << "ICE::scheduleTestConservation" << endl;
    Task* t= scinew Task("ICE::TestConservation",
                   this, &ICE::TestConservation);

    Ghost::GhostType  gn  = Ghost::None;
//  t->requires(Task::OldDW, lb->delTLabel);     for AMR                  
    t->requires(Task::NewDW,lb->rho_CCLabel,        ice_matls, gn);
    t->requires(Task::NewDW,lb->vel_CCLabel,        ice_matls, gn);
    t->requires(Task::NewDW,lb->temp_CCLabel,       ice_matls, gn);
    t->requires(Task::NewDW,lb->specific_heatLabel, ice_matls, gn);
    t->requires(Task::NewDW,lb->uvel_FCMELabel,     ice_matls, gn);
    t->requires(Task::NewDW,lb->vvel_FCMELabel,     ice_matls, gn);
    t->requires(Task::NewDW,lb->wvel_FCMELabel,     ice_matls, gn);
    
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
  //__________________________________
  //  Each model *can* test conservation 
  if(d_models.size() != 0){
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
                                          iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleTestConservation(sched,patches,d_modelInfo);
    }
  }
}

/* _____________________________________________________________________
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
    constCCVariable<double> speedSound, sp_vol_CC, thermalCond, viscosity;
    constCCVariable<double> cv, gamma;
    constCCVariable<Vector> vel_CC;
    Ghost::GhostType  gn  = Ghost::None; 
    Ghost::GhostType  gac = Ghost::AroundCells;

    double dCFL = d_CFL;
    delt_CFL = 1000.0; 

    for (int m = 0; m < d_sharedState->getNumICEMatls(); m++) {
      Material* matl = d_sharedState->getICEMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex(); 
      new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gac, 1);
      new_dw->get(vel_CC,     lb->vel_CCLabel,        indx,patch,gac, 1);
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn,  0);
      new_dw->get(viscosity,  lb->viscosityLabel,     indx,patch,gn,  0);
      new_dw->get(thermalCond,lb->thermalCondLabel,   indx,patch,gn,  0);
      new_dw->get(gamma,      lb->gammaLabel,         indx,patch,gn,  0);
      new_dw->get(cv,         lb->specific_heatLabel, indx,patch,gn,  0);
      
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
//      cout << " Aggressive delT Based on currant number "<< delt_CFL << endl;
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
                                   
        double dx_length   = dx.length();

        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          double sumSwept_Vol = 0.0;
          IntVector c = *iter;
          double cp = cv[c] * gamma[c];
          
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

            double thermalDiffusivity = thermalCond[c] * sp_vol_CC[c]/cp;
            double diffusion_vel    = std::max(thermalDiffusivity, viscosity[c])
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
            // only compute outflow volumes
            sweptVol_R = std::max( 0.0, sweptVol_R);  
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
      double thermalCond_test = ice_matl->getThermalConductivity();
      if (thermalCond_test !=0) {

        for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double cp = cv[c] * gamma[c];
          double inv_thermalDiffusivity = cp/(sp_vol_CC[c] * thermalCond[c]);
          double A = d_CFL * 0.5 * inv_sum_invDelx_sqr * inv_thermalDiffusivity;
          delt_cond = std::min(A, delt_cond);
        }
      }  //
    }  // matl loop   
//    cout << "delT based on conduction "<< delt_cond<<endl;

    delt = std::min(delt_CFL, delt_cond);
    delt = std::min(delt, d_initialDt);

    d_initialDt = 10000.0;

    const Level* level = getLevel(patches);
    GridP grid = level->getGrid();
      for(int i=1;i<=level->getIndex();i++) {     // REFINE
        delt *= grid->getLevel(i)->timeRefinementRatio();
      }
    
    //__________________________________
    //  Bullet proofing
    if(delt < 1e-20) {  
      string warn = " E R R O R \n ICE::ComputeStableTimestep: delT < 1e-20";
      throw InvalidValue(warn);
    }
    new_dw->put(delt_vartype(delt), lb->delTLabel);
  }  // patch loop
}

/* _____________________________________________________________________ 
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
    StaticArray<constCCVariable<double> > placeHolder(0);
    StaticArray<CCVariable<double>   > rho_micro(numMatls);
    StaticArray<CCVariable<double>   > sp_vol_CC(numMatls);
    StaticArray<CCVariable<double>   > rho_CC(numMatls); 
    StaticArray<CCVariable<double>   > Temp_CC(numMatls);
    StaticArray<CCVariable<double>   > speedSound(numMatls);
    StaticArray<CCVariable<double>   > vol_frac_CC(numMatls);
    StaticArray<CCVariable<Vector>   > vel_CC(numMatls);
    StaticArray<CCVariable<double>   > cv(numMatls);
    StaticArray<CCVariable<double>   > gamma(numMatls);
    CCVariable<double>    press_CC, imp_initialGuess;
    
    new_dw->allocateAndPut(press_CC,         lb->press_CCLabel,     0,patch);
    new_dw->allocateAndPut(imp_initialGuess, lb->initialGuessLabel, 0,patch);
    press_CC.initialize(0.0);
    imp_initialGuess.initialize(0.0); 

    //__________________________________
    //  Thermo and transport properties
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();
      CCVariable<double> viscosity, thermalCond;
      new_dw->allocateAndPut(viscosity,     lb->viscosityLabel,    indx,patch);
      new_dw->allocateAndPut(thermalCond,   lb->thermalCondLabel,  indx,patch);
      new_dw->allocateAndPut(cv[m],         lb->specific_heatLabel,indx,patch);
      new_dw->allocateAndPut(gamma[m],      lb->gammaLabel,        indx,patch);
      
      gamma[m].initialize(    ice_matl->getGamma());
      cv[m].initialize(       ice_matl->getSpecificHeat());    
      viscosity.initialize  ( ice_matl->getViscosity());
      thermalCond.initialize( ice_matl->getThermalConductivity());
      
      if(ice_matl->isSurroundingMatl()) {
        d_surroundingMatl_indx = m;  //which matl. is the surrounding matl
      } 
    }
    // --------bulletproofing
    if (grav.length() >0.0 && d_surroundingMatl_indx == -9)  {
      throw ProblemSetupException("ERROR ICE::actuallyInitialize \n"
            "You must have \n" 
            "       <isSurroundingMatl> true </isSurroundingMatl> \n "
            "specified inside the ICE material that is the background matl\n");
    }
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

    
    double p_ref = d_sharedState->getRefPress();
    press_CC.initialize(p_ref);
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      ice_matl->initializeCells(rho_micro[m],  rho_CC[m],
                                Temp_CC[m],    speedSound[m], 
                                vol_frac_CC[m], vel_CC[m], 
                                press_CC, numALLMatls, patch, new_dw);
      
      // if specified, overide the initialization             
      customInitialization( patch,rho_CC[m], Temp_CC[m],vel_CC[m], press_CC,
                            ice_matl, d_customInitialize_basket);
                                                    
      setBC(rho_CC[m],     "Density",     patch, d_sharedState, indx, new_dw);
      setBC(rho_micro[m],  "Density",     patch, d_sharedState, indx, new_dw);
      setBC(Temp_CC[m],    "Temperature", patch, d_sharedState, indx, new_dw);
      setBC(speedSound[m], "zeroNeumann", patch, d_sharedState, indx, new_dw); 
      setBC(vel_CC[m],     "Velocity",    patch, d_sharedState, indx, new_dw); 
      setBC(press_CC, rho_micro, placeHolder, d_surroundingMatl_indx, 
            "rho_micro","Pressure", patch, d_sharedState, 0, new_dw);
            
      for (CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[m][c] = 1.0/rho_micro[m][c];
        vol_frac_CC[m][c] = rho_CC[m][c]*sp_vol_CC[m][c];  //needed for LODI BCs
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
    }   // numMatls

    if (switchDebugInitialize){     
      ostringstream desc1;
      desc1 << "Initialization_patch_"<< patch->getID();
      printData(0, patch, 1, desc1.str(), "press_CC", press_CC);         
      for (int m = 0; m < numMatls; m++ ) { 
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();
        ostringstream desc;      
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
/* _____________________________________________________________________
 Function~  ICE::initialize_hydrostaticAdj
 Purpose~   adjust the pressure and temperature fields after both
            ICE and the models have initialized the fields
 _____________________________________________________________________  */
void ICE::initializeSubTask_hydrostaticAdj(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*ice_matls*/,
                                          DataWarehouse* /*old_dw*/,
                                          DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing initialize_hydrostaticAdj on patch "
               << patch->getID() << "\t ICE" << endl;
   
    Ghost::GhostType  gn = Ghost::None;
    int numMatls = d_sharedState->getNumICEMatls();
    //__________________________________
    //  grab rho micro for all matls
    StaticArray<CCVariable<double>   > rho_micro(numMatls);
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->getModifiable(rho_micro[m],lb->rho_micro_CCLabel,  indx, patch);
    }
    
    CCVariable<double> press_CC;
    new_dw->getModifiable(press_CC, lb->press_CCLabel,0, patch);
    
    //_________________________________
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      constCCVariable<double> gamma, cv;
      CCVariable<double> Temp;   
      
      new_dw->get(gamma, lb->gammaLabel,         indx, patch,gn,0);   
      new_dw->get(cv,    lb->specific_heatLabel, indx, patch,gn,0);
      new_dw->getModifiable(Temp, lb->temp_CCLabel,  indx, patch);  
       
      //__________________________________
      //  Adjust pressure and Temp field if g != 0
      //  so fields are thermodynamically consistent.
      StaticArray<constCCVariable<double> > placeHolder(0);
      hydrostaticPressureAdjustment(patch, rho_micro[d_surroundingMatl_indx],
                                    press_CC);

      Patch::FaceType dummy = Patch::invalidFace; // This is a dummy variable
      ice_matl->getEOS()->computeTempCC( patch, "WholeDomain",
					 press_CC, gamma, cv,
					 rho_micro[m], Temp, dummy );

      //__________________________________
      //  Print Data
      if (switchDebugInitialize){     
        ostringstream desc, desc1;
        desc << "hydroStaticAdj_patch_"<< patch->getID();
        printData(0, patch, 1, desc.str(), "press_CC", press_CC);         
        for (int m = 0; m < numMatls; m++ ) { 
          ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
          int indx = ice_matl->getDWIndex();      
          desc1 << "hydroStaticAdj_Mat_" << indx << "_patch_"<< patch->getID();
          printData(indx, patch,   1, desc.str(), "rho_micro_CC",rho_micro[m]);
          printData(indx, patch,   1, desc.str(), "Temp_CC",     Temp);
        }   
      }
    }
  }
} 

/* _____________________________________________________________________
 Function~  ICE::computeThermoTransportProperties
 Purpose~   
 _____________________________________________________________________  */
void ICE::computeThermoTransportProperties(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*ice_matls*/,
                                          DataWarehouse* /*old_dw*/,
                                          DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeThermoTransportProperties on patch "
               << patch->getID() << "\t ICE" << endl;
   
    int numMatls = d_sharedState->getNumICEMatls();
    
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      CCVariable<double> viscosity, thermalCond, gamma, cv;
      
      new_dw->allocateAndPut(thermalCond, lb->thermalCondLabel,  indx, patch);  
      new_dw->allocateAndPut(viscosity,   lb->viscosityLabel,    indx, patch);
      new_dw->allocateAndPut(cv,          lb->specific_heatLabel,indx, patch);
      new_dw->allocateAndPut(gamma,       lb->gammaLabel,        indx, patch); 
      viscosity.initialize  ( ice_matl->getViscosity());
      thermalCond.initialize( ice_matl->getThermalConductivity());
      gamma.initialize  (     ice_matl->getGamma());
      cv.initialize(          ice_matl->getSpecificHeat());
    }

    //__________________________________
    // Is it time to dump printData ?
    // You need to do this in the first task
    // and only on the first patch
    if (patch->getID() == 0) {
      d_dbgTime_to_printData = false;
      double time= dataArchiver->getCurrentTime() + d_SMALL_NUM;
      
      if (time >= d_dbgStartTime && 
          time <= d_dbgStopTime  &&
          time >= d_dbgNextDumpTime) {
        d_dbgTime_to_printData  = true;

        d_dbgNextDumpTime = d_dbgOutputInterval 
                          * ceil(time/d_dbgOutputInterval + d_SMALL_NUM); 
      }
    }
  }
}
/* _____________________________________________________________________ 
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
    StaticArray<constCCVariable<double> > cv(numMatls);
    StaticArray<constCCVariable<double> > gamma(numMatls); 
    StaticArray<constCCVariable<double> > placeHolder(0);   

    CCVariable<int> n_iters_equil_press;
    constCCVariable<double> press;
    CCVariable<double> press_new, press_copy;
    Ghost::GhostType  gn = Ghost::None;
    
    //__________________________________
    //  Implicit press needs two copies of press 
    old_dw->get(press,                lb->press_CCLabel, 0,patch,gn, 0); 
    new_dw->allocateAndPut(press_new, lb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(press_copy,lb->press_CCLabel,       0,patch);
       
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],      lb->temp_CCLabel,      indx,patch, gn,0);
      old_dw->get(rho_CC[m],    lb->rho_CCLabel,       indx,patch, gn,0);
      old_dw->get(sp_vol_CC[m], lb->sp_vol_CCLabel,    indx,patch, gn,0);
      new_dw->get(cv[m],        lb->specific_heatLabel,indx,patch, gn,0);
      new_dw->get(gamma[m],     lb->gammaLabel,        indx,patch, gn,0);
            
      new_dw->allocateTemporary(rho_micro[m],  patch);
      new_dw->allocateAndPut(vol_frac[m],   lb->vol_frac_CCLabel,indx, patch);  
      new_dw->allocateAndPut(rho_CC_new[m], lb->rho_CCLabel,     indx, patch);  
      new_dw->allocateAndPut(sp_vol_new[m], lb->sp_vol_CCLabel,  indx, patch); 
      new_dw->allocateAndPut(f_theta[m],    lb->f_theta_CCLabel, indx, patch);  
      new_dw->allocateAndPut(speedSound_new[m], lb->speedSound_CCLabel,
                                                                 indx, patch);
    }

    press_new.copyData(press);
    
    //__________________________________
    // Compute rho_micro, volfrac
    for (int m = 0; m < numMatls; m++) {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        rho_micro[m][c] = 1.0/sp_vol_CC[m][c];
        vol_frac[m][c] = rho_CC[m][c] * sp_vol_CC[m][c];
      }
    }

   //---- P R I N T   D A T A ------  
    if (switchDebug_EQ_RF_press) {
    
      new_dw->allocateTemporary(n_iters_equil_press,  patch);
      ostringstream desc,desc1;
      desc1 << "TOP_equilibration_patch_" << patch->getID();
      printData( 0, patch, 1, desc1.str(), "Press_CC_top", press);
     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       desc << "TOP_equilibration_Mat_" << indx << "_patch_"<<patch->getID();
       printData(indx, patch, 1, desc.str(), "rho_CC",       rho_CC[m]);    
       printData(indx, patch, 1, desc.str(), "rho_micro_CC", rho_micro[m]);  
       printData(indx, patch, 1, desc.str(), "speedSound",   speedSound_new[m]);
       printData(indx, patch, 1, desc.str(), "Temp_CC",      Temp[m]);       
       printData(indx, patch, 1, desc.str(), "vol_frac_CC",  vol_frac[m]);   
      }
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

        //__________________________________
        // evaluate press_eos at cell i,j,k
        for (int m = 0; m < numMatls; m++)  {
          ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
          ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                              cv[m][c], Temp[m][c],press_eos[m],
                                              dp_drho[m], dp_de[m]);
        }

        //__________________________________
        // - compute delPress
        // - update press_CC     
        double A = 0., B = 0., C = 0.;
        for (int m = 0; m < numMatls; m++)   {
          double Q =  press_new[c] - press_eos[m];
          double div_y =  (vol_frac[m][c] * vol_frac[m][c])
            / (dp_drho[m] * rho_CC[m][c] + d_SMALL_NUM);
          A   +=  vol_frac[m][c];
          B   +=  Q*div_y;
          C   +=  div_y;
        }
        double vol_frac_not_close_packed = 1.0;
        delPress = (A - vol_frac_not_close_packed - B)/C;

        press_new[c] += delPress;

        //__________________________________
        // backout rho_micro_CC at this new pressure
        for (int m = 0; m < numMatls; m++) {
          ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
          rho_micro[m][c] = 
           ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[m][c],
                                          cv[m][c],Temp[m][c],rho_micro[m][c]);

          double div = 1./rho_micro[m][c];
          
          // - updated volume fractions
          vol_frac[m][c]   = rho_CC[m][c]*div;
        }
        //__________________________________
        // - Test for convergence 
        //  If sum of vol_frac_CC ~= vol_frac_not_close_packed then converged 
        sum = 0.0;
        for (int m = 0; m < numMatls; m++)  {
          sum += vol_frac[m][c];
        }
        if (fabs(sum-1.0) < convergence_crit){
          converged = true;
          //__________________________________
          // Find the speed of sound based on converged solution
          for (int m = 0; m < numMatls; m++) {
            ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
            ice_matl->getEOS()->computePressEOS(rho_micro[m][c],gamma[m][c],
                                            cv[m][c],Temp[m][c],
                                            press_eos[m],dp_drho[m], dp_de[m]);

            tmp = dp_drho[m] 
                + dp_de[m] * press_eos[m]/(rho_micro[m][c] * rho_micro[m][c]);
            speedSound_new[m][c] = sqrt(tmp);
          }
        }
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
      if ( fabs(sum - 1.0) > convergence_crit) {  
        throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached vol_frac != 1");
      }
       
      if ( press_new[c] < 0.0 ){ 
        throw MaxIteration(c,count,n_passes,
                         "MaxIteration reached press_new < 0");
      }

      for (int m = 0; m < numMatls; m++){
        if ( rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) { 
          cout << "m = " << m << endl;
          throw MaxIteration(c,count,n_passes,
                      "MaxIteration reached rho_micro < 0 || vol_frac < 0");
        }
      }

      if (switchDebug_EQ_RF_press) {
        n_iters_equil_press[c] = count;
      }
    }     // end of cell interator

    cout_norm << "max. iterations in any cell " << test_max_iter << 
                 " on patch "<<patch->getID()<<endl; 

    //__________________________________
    // carry rho_cc forward 
    // MPMICE computes rho_CC_new
    // therefore need the machinery here
    for (int m = 0; m < numMatls; m++)   {
      rho_CC_new[m].copyData(rho_CC[m]);
    }

    //__________________________________
    // - update Boundary conditions
    // - make copy of press for implicit calc.
    preprocess_CustomBCs("EqPress",old_dw, new_dw, lb,  patch, 
                          999,d_customBC_var_basket);
    
    setBC(press_new,   rho_micro, placeHolder, d_surroundingMatl_indx,
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw, 
          d_customBC_var_basket);
          
    delete_CustomBCs(d_customBC_var_basket);      
   
    press_copy.copyData(press_new);
    
    //__________________________________
    // compute sp_vol_CC
    // - Set BCs on rhoMicro. using press_CC 
    // - backout sp_vol_new 
    for (int m = 0; m < numMatls; m++)   {
/*`==========TESTING==========*/
// This needs to be rethought.
// With a jet inlet, rho_CC is fixed and the pressure is allowed to 
// float.   Backing out rho_micro at the new pressure will eventually
// cause vol_frac to != 1.0.
// 
#if 0
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      
      // only hit boundary faces  
      vector<Patch::FaceType>::const_iterator f;
      for (f  = patch->getBoundaryFaces()->begin(); 
           f != patch->getBoundaryFaces()->end(); ++f){
        Patch::FaceType face = *f;
        
        CellIterator iterLim = patch->getFaceCellIterator(face,"plusEdgeCells");
        for(CellIterator iter=iterLim; !iter.done();iter++) {
          IntVector c = *iter;
          rho_micro[m][c] = 
            ice_matl->getEOS()->computeRhoMicro(press_new[c],gamma[m][c],
                                           cv[m][c],Temp[m][c],rho_micro[m][c]);
        }
      } // face loop
#endif 
/*===========TESTING==========`*/
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c]; 
      }
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
       
     }
    }
  }  // patch loop
}
 
/* _____________________________________________________________________
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


/* _____________________________________________________________________
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
  const Level* level = getLevel(patches);
  
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

      CellIterator XFC_iterator = patch->getSFCXIterator(offset);
      CellIterator YFC_iterator = patch->getSFCYIterator(offset);
      CellIterator ZFC_iterator = patch->getSFCZIterator(offset);
      
      if (level->getIndex() > 0) {  // Finer levels need to hit the ghost cells
        IntVector l, h;
        l = patch->getExtraCellIterator().begin();
        h = patch->getExtraCellIterator().end();
        XFC_iterator = CellIterator(l + IntVector(1,0,0),h);
        YFC_iterator = CellIterator(l + IntVector(0,1,0),h);
        ZFC_iterator = CellIterator(l + IntVector(0,0,1),h);
      }
      
      //__________________________________
      //  Compute the temperature on each face     
      //  Currently on used by HEChemistry 
      if (d_models.size() > 0) {        
        computeTempFace<SFCXVariable<double> >(XFC_iterator, adj_offset[0], 
                                               rho_CC,Temp_CC, TempX_FC);

        computeTempFace<SFCYVariable<double> >(YFC_iterator, adj_offset[1], 
                                               rho_CC,Temp_CC, TempY_FC);

        computeTempFace<SFCZVariable<double> >(ZFC_iterator,adj_offset[2],
                                               rho_CC,Temp_CC, TempZ_FC);
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
  const Level* level = getLevel(patches);
  
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing computeVel_FC on patch " 
              << patch->getID() << "\t\t\t\t ICE" << endl;

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
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    }
     
    new_dw->get(press_CC,lb->press_CCLabel, 0, patch,gac, 1);
    
    delt_vartype delT;
    pOldDW->get(delT, d_sharedState->get_delt_label(),level);   
     
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

      CellIterator XFC_iterator = patch->getSFCXIterator(offset);
      CellIterator YFC_iterator = patch->getSFCYIterator(offset);
      CellIterator ZFC_iterator = patch->getSFCZIterator(offset);

      //__________________________________
      //  Compute vel_FC for each face
      computeVelFace<SFCXVariable<double> >(0, XFC_iterator,
                                       adj_offset[0],dx[0],delT,gravity[0],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       uvel_FC);

      computeVelFace<SFCYVariable<double> >(1, YFC_iterator,
                                       adj_offset[1],dx[1],delT,gravity[1],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       vvel_FC);

      computeVelFace<SFCZVariable<double> >(2, ZFC_iterator,
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

/* _____________________________________________________________________
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
  double b[MAX_MATLS], b_sp_vol[MAX_MATLS];
  double vel[MAX_MATLS], tmp[MAX_MATLS];
  FastMatrix a(numMatls, numMatls);
  
  for(;!iter.done(); iter++){
    IntVector c = *iter;
    IntVector adj = c + adj_offset; 

    //__________________________________
    //   Compute beta and off diagonal term of
    //   Matrix A, this includes b[m][m].
    //  You need to make sure that mom_exch_coeff[m][m] = 0
    
    // - Form diagonal terms of Matrix (A)
    //  - Form RHS (b) 
    for(int m = 0; m < numMatls; m++)  {
      b_sp_vol[m] = 2.0 * (sp_vol_CC[m][adj] * sp_vol_CC[m][c])/
	(sp_vol_CC[m][adj] + sp_vol_CC[m][c]);
      tmp[m] = -0.5 * delT * (vol_frac_CC[m][adj] + vol_frac_CC[m][c]);
      vel[m] = vel_FC[m][c];
    }

    for(int m = 0; m < numMatls; m++)  {
      double betasum = 1;
      double bsum = 0;
      double bm = b_sp_vol[m];
      double vm = vel[m];
      for(int n = 0; n < numMatls; n++)  {
        double b = bm * tmp[n] * K(n,m);
        a(m,n)    = b;
	betasum -= b;
	bsum -= b * (vel[n] - vm);
      }
      a(m,m) = betasum;
      b[m] = bsum;
    }

    //__________________________________
    //  - solve and backout velocities
    
    a.destructiveSolve(b, b_sp_vol);
    //  For implicit solve we need sp_vol_FC
    for(int m = 0; m < numMatls; m++) {
      vel_FCME[m][c] = vel_FC[m][c] + b[m];
      sp_vol_FC[m][c] = b_sp_vol[m];// only needed by implicit Pressure
    }
  }  // iterator
}

/*_____________________________________________________________________
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
 _____________________________________________________________________  */
void ICE::addExchangeContributionToFCVel(const ProcessorGroup*,  
                                         const PatchSubset* patches,
                                         const MaterialSubset* /*matls*/,
                                         DataWarehouse* old_dw, 
                                         DataWarehouse* new_dw,
                                         const bool recursion)
{
  const Level* level = getLevel(patches);
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
    pOldDW->get(delT, d_sharedState->get_delt_label(),level);

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

    CellIterator XFC_iterator = patch->getSFCXIterator(offset);
    CellIterator YFC_iterator = patch->getSFCYIterator(offset);
    CellIterator ZFC_iterator = patch->getSFCZIterator(offset);
                                
    //__________________________________
    //  tack on exchange contribution
    add_vel_FC_exchange<StaticArray<constSFCXVariable<double> >,
                        StaticArray<     SFCXVariable<double> > >
                        (XFC_iterator, 
                        adj_offset[0],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        uvel_FC,        sp_vol_XFC,  uvel_FCME);
                        
    add_vel_FC_exchange<StaticArray<constSFCYVariable<double> >,
                        StaticArray<     SFCYVariable<double> > >
                        (YFC_iterator, 
                        adj_offset[1],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        vvel_FC,        sp_vol_YFC,  vvel_FCME);
                        
    add_vel_FC_exchange<StaticArray<constSFCZVariable<double> >,
                        StaticArray<     SFCZVariable<double> > >
                        (ZFC_iterator, 
                        adj_offset[2],  numMatls,    K, 
                        delT,           vol_frac_CC, sp_vol_CC,
                        wvel_FC,        sp_vol_ZFC,  wvel_FCME);

    //________________________________
    //  Boundary Conditons 
    preprocess_CustomBCs("velFC_Exchange",pOldDW, pNewDW, lb,  patch, 999,
                          d_customBC_var_basket);   
    
    for (int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC<SFCXVariable<double> >(uvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket); 
      setBC<SFCYVariable<double> >(vvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket);
      setBC<SFCZVariable<double> >(wvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket);
    }
    delete_CustomBCs(d_customBC_var_basket);

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

/*_____________________________________________________________________
 Function~  ICE::computeDelPressAndUpdatePressCC--
 Purpose~
   This function calculates the change in pressure explicitly. 
 Note:  Units of delp_Dilatate and delP_MassX are [Pa]
 Reference:  Multimaterial Formalism eq. 1.5
 _____________________________________________________________________  */
void ICE::computeDelPressAndUpdatePressCC(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* old_dw, 
                                          DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    cout_doing << "Doing explicit delPress on patch " << patch->getID() 
         <<  "\t\t\t ICE" << endl;

    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx     = patch->dCell();

    double vol    = dx.x()*dx.y()*dx.z();    
    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;
    CCVariable<double> press_CC;
    CCVariable<double> term1, term2, term3;
    StaticArray<CCVariable<double> > placeHolder(0);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
   
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->getModifiable( press_CC,     lb->press_CCLabel,     0, patch);
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(term2,        lb->term2Label,        0, patch);
    new_dw->allocateAndPut(term3,        lb->term3Label,        0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch); 

    new_dw->allocateTemporary(q_advected, patch);
    new_dw->allocateTemporary(term1,      patch);

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

      SFCXVariable<double> vol_fracX_FC;
      SFCYVariable<double> vol_fracY_FC;
      SFCZVariable<double> vol_fracZ_FC;      

      new_dw->allocateAndPut(vol_fracX_FC, lb->vol_fracX_FCLabel, indx,patch);
      new_dw->allocateAndPut(vol_fracY_FC, lb->vol_fracY_FCLabel, indx,patch);
      new_dw->allocateAndPut(vol_fracZ_FC, lb->vol_fracZ_FCLabel, indx,patch);
      
      // lowIndex is the same for all vel_FC
      IntVector lowIndex(patch->getSFCXLowIndex());
      double nan= getNan();
      vol_fracX_FC.initialize(nan, lowIndex,patch->getSFCXHighIndex());
      vol_fracY_FC.initialize(nan, lowIndex,patch->getSFCYHighIndex());
      vol_fracZ_FC.initialize(nan, lowIndex,patch->getSFCZHighIndex()); 
      
          
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
                                    bulletProof_test, new_dw); 
      //__________________________________
      //   advect vol_frac
      advector->advectQ(vol_frac, patch, q_advected,  
                        vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw); 
      
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        term2[c] -= q_advected[c]; 
      }

      //__________________________________
      // term3 is the same now with or without models
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        term3[c] += vol_frac[c]*sp_vol_CC[m][c]/(speedSound[c]*speedSound[c]);
      }
      
      //__________________________________
      //   term1 contribution from models
      if(d_models.size() > 0){
        constCCVariable<double> modelMass_src, modelVol_src;
        new_dw->get(modelMass_src, lb->modelMass_srcLabel, indx, patch, gn, 0);
        new_dw->get(modelVol_src,  lb->modelVol_srcLabel,  indx, patch, gn, 0);
                
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         term1[c] += modelMass_src[c] * (sp_vol_CC[m][c]/vol);
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

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      delP_MassX[c]    =  term1[c]/term3[c];
      delP_Dilatate[c] = -term2[c]/term3[c];
      press_CC[c]     +=  delP_MassX[c] + delP_Dilatate[c];
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

    //__________________________________
    //  set boundary conditions
    preprocess_CustomBCs("update_press_CC",old_dw, new_dw, lb,  patch, 999,
                          d_customBC_var_basket);
    
    setBC(press_CC, placeHolder, sp_vol_CC, d_surroundingMatl_indx,
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw,
          d_customBC_var_basket);
       
    delete_CustomBCs(d_customBC_var_basket);      
       
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

/* _____________________________________________________________________  
 Function~  ICE::computePressFC--
 Purpose~
    This function calculates the face centered pressure on each of the 
    cell faces for every cell in the computational domain and a single 
    layer of ghost cells. 
  _____________________________________________________________________  */
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

/* _____________________________________________________________________
 Function~  ICE::zeroModelMassExchange
 Purpose~   This function initializes the mass exchange quantities to
            zero.  These quantities are subsequently modified by the
            models
 _____________________________________________________________________  */
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
      for(vector<TransportedVariable*>::iterator
                                    iter = d_modelSetup->tvars.begin();
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

/* _____________________________________________________________________
 Function~  ICE::updateVolumeFraction
 Purpose~   Update the volume fraction to reflect the mass exchange done
            by models
 _____________________________________________________________________  */
void ICE::updateVolumeFraction(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* /*old_dw*/,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gn = Ghost::None;
    int numALLMatls = d_sharedState->getNumMatls();
    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol(numALLMatls);
    StaticArray<constCCVariable<double> > modVolSrc(numALLMatls);
    Vector dx     = patch->dCell();
    double vol     = dx.x() * dx.y() * dx.z();


    for(int m=0;m<matls->size();m++){
      Material* matl        = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->getModifiable(vol_frac[m], lb->vol_frac_CCLabel, indx,patch);
      new_dw->get(rho_CC[m],      lb->rho_CCLabel,             indx,patch,gn,0);
      new_dw->get(sp_vol[m],      lb->sp_vol_CCLabel,          indx,patch,gn,0);
      new_dw->get(modVolSrc[m],   lb->modelVol_srcLabel,       indx,patch,gn,0);
    }

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      double total_vol=0.;
      for(int m=0;m<matls->size();m++){
        total_vol+=(rho_CC[m][c]*vol)*sp_vol[m][c];
      }
      for(int m=0;m<matls->size();m++){
        double new_vol = vol_frac[m][c]*total_vol+modVolSrc[m][c];
        vol_frac[m][c] = max(new_vol/total_vol,0.);
      }
    }

  }
}
/* _____________________________________________________________________
 Function~  ICE::accumulateMomentumSourceSinks--
 Purpose~   This function accumulates all of the sources/sinks of momentum
 _____________________________________________________________________  */
void ICE::accumulateMomentumSourceSinks(const ProcessorGroup*,  
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw, 
                                        DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
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
    double include_term;

    delt_vartype delT; 
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
 
    dx      = patch->dCell();
    gravity = d_sharedState->getGravity();
    vol     = dx.x() * dx.y() * dx.z();
    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();
    
    constCCVariable<double>   rho_CC;
    constCCVariable<double>   sp_vol_CC;
    constCCVariable<double>   viscosity_org;
    constCCVariable<Vector>   vel_CC;
    constCCVariable<double>   vol_frac;
    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;
    constSFCXVariable<double> press_diffX_FC;
    constSFCYVariable<double> press_diffY_FC;
    constSFCZVariable<double> press_diffZ_FC;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;  
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
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,patch,gn, 0);
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
      
      tau_X_FC.initialize(Vector(0.0));
      tau_Y_FC.initialize(Vector(0.0));
      tau_Z_FC.initialize(Vector(0.0));

      if(ice_matl){
        new_dw->get(viscosity_org, lb->viscosityLabel, indx,patch,gac,2); 
        old_dw->get(vel_CC,        lb->vel_CCLabel,    indx,patch,gac,2); 
        new_dw->get(sp_vol_CC,     lb->sp_vol_CCLabel, indx,patch,gac,2); 
        
        //__________________________________
        //  compute the shear stress terms
        double viscosity_test = ice_matl->getViscosity();
        if(viscosity_test != 0.0){
        
          CCVariable<double> viscosity;  // don't alter the original value
          new_dw->allocateTemporary(viscosity, patch, gac, 2);
          viscosity.copyData(viscosity_org);
        
          if(d_turbulence){ 
            d_turbulence->callTurb(new_dw,patch,vel_CC,rho_CC,indx,lb,
                                   d_sharedState, viscosity);
          }//turb
          constSFCXVariable<double> vol_fracX_FC;
          constSFCYVariable<double> vol_fracY_FC;
          constSFCZVariable<double> vol_fracZ_FC;
           
          new_dw->get(vol_fracX_FC, lb->vol_fracX_FCLabel, indx,patch,gac, 2);         
          new_dw->get(vol_fracY_FC, lb->vol_fracY_FCLabel, indx,patch,gac, 2);         
          new_dw->get(vol_fracZ_FC, lb->vol_fracZ_FCLabel, indx,patch,gac, 2); 
                     
          computeTauX(patch, vol_fracX_FC, vel_CC,viscosity,dx, tau_X_FC);
          computeTauY(patch, vol_fracY_FC, vel_CC,viscosity,dx, tau_Y_FC);
          computeTauZ(patch, vol_fracZ_FC, vel_CC,viscosity,dx, tau_Z_FC);
        }
        if(viscosity_test == 0.0 && d_turbulence){
          string warn="ERROR:\n input :viscosity can't be zero when calculate turbulence";
          throw ProblemSetupException(warn);
        }
      }  // ice_matl
      
      // only include term if it's an ice matl
      if (ice_matl) {
        include_term = 1.0;
      }else{
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

      setBC(press_force, "set_if_sym_BC",patch, d_sharedState, indx, new_dw); 

      //---- P R I N T   D A T A ------ 
      if (switchDebugSource_Sink) {
        ostringstream desc;
        desc << "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printVector(indx, patch, 1, desc.str(), "mom_source",  0, mom_source);
      //printVector(indx, patch, 1, desc.str(), "press_force", 0, press_force);
      }
    }  // matls loop
  }  //patches
}

/* _____________________________________________________________________ 
 Function~  ICE::accumulateEnergySourceSinks--
 Purpose~   This function accumulates all of the sources/sinks of energy 
 Currently the kinetic energy isn't included.
 _____________________________________________________________________  */
void ICE::accumulateEnergySourceSinks(const ProcessorGroup*,  
                                  const PatchSubset* patches,
                                  const MaterialSubset* /*matls*/,
                                  DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing accumulate_energy_source_sinks on patch " 
         << patch->getID() << "\t\t ICE" << endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx = patch->dCell();
    double A, B, vol=dx.x()*dx.y()*dx.z();
    
    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> speedSound;
    constCCVariable<double> vol_frac;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;
    constCCVariable<double> matl_press;
    constCCVariable<double> rho_CC;
        
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(press_CC,     lb->press_CCLabel,      0, patch,gn, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,gn, 0);
    
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl); 

      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      CCVariable<double> heatCond_src;
      
      new_dw->get(sp_vol_CC,    lb->sp_vol_CCLabel,     indx,patch,gac,1);
      new_dw->get(rho_CC,       lb->rho_CCLabel,        indx,patch,gac,1);
      new_dw->get(speedSound,   lb->speedSound_CCLabel, indx,patch,gn, 0);
      new_dw->get(vol_frac,     lb->vol_frac_CCLabel,   indx,patch,gn, 0);
       
      new_dw->allocateAndPut(int_eng_source, 
                               lb->int_eng_source_CCLabel,indx,patch);
      new_dw->allocateAndPut(heatCond_src, 
                               lb->heatCond_src_CCLabel,  indx,patch);
      int_eng_source.initialize(0.0);
      heatCond_src.initialize(0.0);
     
      //__________________________________
      //  Source due to conduction ICE only
      if(ice_matl){
        double thermalCond_test = ice_matl->getThermalConductivity();
        if(thermalCond_test != 0.0 ){
          constCCVariable<double> Temp_CC;
          constCCVariable<double> thermalCond;
          new_dw->get(thermalCond, lb->thermalCondLabel, indx,patch,gac,1); 
          old_dw->get(Temp_CC,     lb->temp_CCLabel,     indx,patch,gac,1); 

          constSFCXVariable<double> vol_fracX_FC;
          constSFCYVariable<double> vol_fracY_FC;
          constSFCZVariable<double> vol_fracZ_FC;
          new_dw->get(vol_fracX_FC, lb->vol_fracX_FCLabel,indx,patch,gac, 2);          
          new_dw->get(vol_fracY_FC, lb->vol_fracY_FCLabel,indx,patch,gac, 2);          
          new_dw->get(vol_fracZ_FC, lb->vol_fracZ_FCLabel,indx,patch,gac, 2);  

          bool use_vol_frac = true; // include vol_frac in diffusion calc.
          scalarDiffusionOperator(new_dw, patch, use_vol_frac, Temp_CC,
                                  vol_fracX_FC, vol_fracY_FC, vol_fracZ_FC,
                                  heatCond_src, thermalCond, delT);
        }
      }
                                     
      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      bool includeFlowWork = matl->getIncludeFlowWork();
      if(includeFlowWork){
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          A = vol * vol_frac[c] * press_CC[c] * sp_vol_CC[c];
          B = speedSound[c] * speedSound[c];
          int_eng_source[c] += (A/B) * delP_Dilatate[c] + heatCond_src[c]; 
        }
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

/* _____________________________________________________________________
 Function~  ICE::computeLagrangianValues--
 Computes lagrangian mass momentum and energy
 Note:    Only loop over ICE materials, mom_L, massL and int_eng_L
           for MPM is computed in computeLagrangianValuesMPM()
 _____________________________________________________________________  */
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
      constCCVariable<double> rho_CC, temp_CC, cv, int_eng_source;
      constCCVariable<Vector> vel_CC, mom_source, mom_comb;

      Ghost::GhostType  gn = Ghost::None;
      new_dw->get(cv,             lb->specific_heatLabel,    indx,patch,gn,0);
      new_dw->get(rho_CC,         lb->rho_CCLabel,           indx,patch,gn,0);  
      old_dw->get(vel_CC,         lb->vel_CCLabel,           indx,patch,gn,0);  
      old_dw->get(temp_CC,        lb->temp_CCLabel,          indx,patch,gn,0);  
      new_dw->get(mom_source,     lb->mom_source_CCLabel,    indx,patch,gn,0);
      new_dw->get(int_eng_source, lb->int_eng_source_CCLabel,indx,patch,gn,0);  
      new_dw->allocateAndPut(mom_L,     lb->mom_L_CCLabel,     indx,patch);
      new_dw->allocateAndPut(int_eng_L, lb->int_eng_L_CCLabel, indx,patch);
      new_dw->allocateAndPut(mass_L,    lb->mass_L_CCLabel,    indx,patch);
      //__________________________________
      //  NO mass exchange
      if(d_models.size() == 0) {
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
           iter++) {
         IntVector c = *iter;
          double mass = rho_CC[c] * vol;
          mass_L[c] = mass;
          mom_L[c] = vel_CC[c] * mass + mom_source[c];
          int_eng_L[c] = mass*cv[c] * temp_CC[c] + int_eng_source[c];
        }
      }

      //__________________________________
      //      MODEL - B A S E D   E X C H A N G E
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
          double min_int_eng = min_mass * cv[c] * temp_CC[c];
          double int_eng_tmp = mass * cv[c] * temp_CC[c];

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
/* _____________________________________________________________________
 Function~  ICE::computeLagrangianSpecificVolume--
 _____________________________________________________________________  */
void ICE::computeLagrangianSpecificVolume(const ProcessorGroup*,  
                                          const PatchSubset* patches,
                                          const MaterialSubset* /*matls*/,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing computeLagrangianSpecificVolume " <<
      patch->getID() << "\t\t\t ICE" << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    int numALLMatls = d_sharedState->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    StaticArray<constCCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > vol_frac(numALLMatls);
    StaticArray<constCCVariable<double> > Temp_CC(numALLMatls);
    StaticArray<CCVariable<double> > alpha(numALLMatls);
    constCCVariable<double> rho_CC, f_theta, sp_vol_CC, cv;
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
       new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,    indx,patch,gn, 0);
       new_dw->get(cv,        lb->specific_heatLabel,indx,patch,gn, 0);

       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          alpha[m][c]=
            ice_matl->getEOS()->getAlpha(Temp_CC[m][c],sp_vol_CC[c],P[c],cv[c]);
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
      CCVariable<double> sp_vol_L, sp_vol_src;
      constCCVariable<double> speedSound;
      new_dw->allocateAndPut(sp_vol_L,  lb->sp_vol_L_CCLabel,   indx,patch);
      new_dw->allocateAndPut(sp_vol_src,lb->sp_vol_src_CCLabel, indx,patch);
      sp_vol_src.initialize(0.);

      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn, 0);
      new_dw->get(rho_CC,     lb->rho_CCLabel,        indx,patch,gn, 0);
      new_dw->get(f_theta,    lb->f_theta_CCLabel,    indx,patch,gn, 0);
      new_dw->get(speedSound, lb->speedSound_CCLabel, indx,patch,gn, 0);

      //__________________________________
      //  compute sp_vol_L * mass
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c];
      }
      
      //---- P R I N T   D A T A ------ 
      if (switchDebugLagrangianSpecificVol ) {
        ostringstream desc;
        desc <<"TOP_Lagrangian_sp_vol_Mat_"<<indx<< "_patch_"<<patch->getID();
         printData( indx, patch,1, desc.str(), "rho_CC",     rho_CC);      
         printData( indx, patch,1, desc.str(), "sp_vol_CC",  sp_vol_CC);     
         printData( indx, patch,1, desc.str(), "sp_vol_L",   sp_vol_L);      
      }
      //__________________________________
      //   Contributions from models
      constCCVariable<double> Modelsp_vol_src;
      if(d_models.size() > 0){ 
        new_dw->get(Modelsp_vol_src, lb->modelVol_srcLabel, indx, patch, gn, 0);
        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         sp_vol_L[c] += Modelsp_vol_src[c];
        }
      }
      
      //__________________________________
      //  add the sources to sp_vol_L
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
        sp_vol_L[c]  += src;
        sp_vol_src[c] = src/(rho_CC[c] * vol);

/*`==========TESTING==========*/
//    do we really want this?  -Todd        
        sp_vol_L[c] = max(sp_vol_L[c], d_TINY_RHO * vol * sp_vol_CC[c]);
/*==========TESTING==========`*/
     }

      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(sp_vol_L, "set_if_sym_BC",patch, d_sharedState, indx, new_dw);

      //---- P R I N T   D A T A ------ 
      if (switchDebugLagrangianSpecificVol ) {
        ostringstream desc;
        desc <<"BOT_Lagrangian_sp_vol_Mat_"<<indx<< "_patch_"<<patch->getID();
        printData( indx, patch,1, desc.str(), "sp_vol_L",   sp_vol_L);    
        printData( indx, patch,1, desc.str(), "sp_vol_src", sp_vol_src);  
        if(d_models.size() > 0){
          printData( indx, patch,1, desc.str(), "Modelsp_vol_src", Modelsp_vol_src);
        }
      }
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      if (!areAllValuesPositive(sp_vol_L, neg_cell)) {
        cout << "matl              "<< indx << endl;
        cout << "sum_thermal_exp   "<< sum_therm_exp[neg_cell] << endl;
        cout << "sp_vol_src        "<< sp_vol_src[neg_cell] << endl;
        cout << "mass sp_vol_L     "<< sp_vol_L[neg_cell] << endl;
        cout << "mass sp_vol_L_old "
             << (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        ostringstream warn;
        warn<<"ERROR ICE::computeLagrangianSpecificVolumeRF, mat "<<indx
            << " cell " <<neg_cell << " sp_vol_L is negative\n";
        throw InvalidValue(warn.str());
     }
    }  // end numALLMatl loop
  }  // patch loop

}

/* _____________________________________________________________________
 Function~  ICE::computeLagrangian_Transported_Vars--
 _____________________________________________________________________  */
void ICE::computeLagrangian_Transported_Vars(const ProcessorGroup*,  
                                             const PatchSubset* patches,
                                             const MaterialSubset* /*matls*/,
                                             DataWarehouse* old_dw, 
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << "Doing computeLagrangian_Transported_Vars on patch " 
               << patch->getID() << "\t ICE" << endl;
    Ghost::GhostType  gn  = Ghost::None;
    int numMatls = d_sharedState->getNumICEMatls();
    
    // get mass_L for all ice matls
    StaticArray<constCCVariable<double> > mass_L(numMatls);
    for (int m = 0; m < numMatls; m++ ) {
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(mass_L[m], lb->mass_L_CCLabel,indx, patch,gn,0);
    }
    
    //__________________________________
    //  hit all the transported variables
    vector<TransportedVariable*>::iterator t_iter;
    for( t_iter  = d_modelSetup->tvars.begin();
         t_iter != d_modelSetup->tvars.end(); t_iter++){
      TransportedVariable* tvar = *t_iter;
      
      for (int m = 0; m < numMatls; m++ ) {
        Material* matl = d_sharedState->getICEMaterial( m );
        int indx = matl->getDWIndex();
       
        if(tvar->matls->contains(indx)){  
          constCCVariable<double> q_old,q_src;
          CCVariable<double> q_L_CC;
          old_dw->get(q_old,             tvar->var, indx, patch, gn, 0);
          new_dw->allocateAndPut(q_L_CC, tvar->var_Lagrangian, indx, patch);

          // initialize q_L to q_old
          q_L_CC.copyData(q_old);

          // If there's a source tack it on.     
          if(tvar->src){
            new_dw->get(q_src,  tvar->src, indx, patch, gn, 0);
            for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
              IntVector c = *iter;                            
              q_L_CC[c]  += q_src[c];     // with source
            }
          }
          // Set boundary conditions on q_L_CC
          string Labelname = tvar->var_Lagrangian->getName();
          setBC(q_L_CC, Labelname,  patch, d_sharedState, indx, new_dw);

          // multiply by mass so advection is conserved
          for(CellIterator iter=patch->getExtraCellIterator();
                          !iter.done();iter++){
            IntVector c = *iter;                            
            q_L_CC[c] *= mass_L[m][c];
          }
          
          //---- P R I N T   D A T A ------
          if (switchDebugLagrangianTransportedVars ) {
            ostringstream desc;
            desc <<"BOT_LagrangianTransVars_BC_Mat_" <<indx<<"_patch_" 
                 <<patch->getID();
            printData(  indx, patch,1, desc.str(), tvar->var->getName(), q_old);
            if(tvar->src){
              printData(indx, patch,1, desc.str(), tvar->src->getName(), q_src);
            }
            printData(  indx, patch,1, desc.str(), Labelname, q_L_CC); 
          }   
             
        }  // tvar matl
      }  // ice matl loop
    }  // tvar loop
  }  // patch loop
}

/*_____________________________________________________________________
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
 _____________________________________________________________________  */
void ICE::addExchangeToMomentumAndEnergy(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t ICE" << endl;

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;
    Ghost::GhostType  gn = Ghost::None;
    
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    //Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    StaticArray<CCVariable<double> > cv(numALLMatls);
    StaticArray<CCVariable<double> > Temp_CC(numALLMatls);
    StaticArray<constCCVariable<double> > gamma(numALLMatls);  
    StaticArray<constCCVariable<double> > vol_frac_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numALLMatls);
    StaticArray<constCCVariable<Vector> > mom_L(numALLMatls);
    StaticArray<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    StaticArray<CCVariable<Vector> > mom_L_ME(numALLMatls);
    StaticArray<CCVariable<Vector> > vel_CC(numALLMatls);
    StaticArray<CCVariable<double> > int_eng_L_ME(numALLMatls);
    StaticArray<CCVariable<double> > Tdot(numALLMatls);
    StaticArray<constCCVariable<double> > mass_L(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > old_temp(numALLMatls);

    double b[MAX_MATLS];
    Vector bb[MAX_MATLS];
    vector<double> sp_vol(numALLMatls);

    double tmp;
    FastMatrix beta(numALLMatls, numALLMatls),acopy(numALLMatls, numALLMatls);
    FastMatrix K(numALLMatls, numALLMatls), H(numALLMatls, numALLMatls);
    FastMatrix a(numALLMatls, numALLMatls);
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();

    getExchangeCoefficients( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);
      
      if(mpm_matl){                 // M P M
        new_dw->get(old_temp[m],     lb->temp_CCLabel,   indx, patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  lb->vel_CCLabel,indx,patch);
        new_dw->getModifiable(Temp_CC[m], lb->temp_CCLabel,indx,patch);
        
        cv[m].initialize(mpm_matl->getSpecificHeat());
      }
      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get(old_temp[m],   lb->temp_CCLabel,      indx, patch,gn,0);
        new_dw->get(cv_ice,        lb->specific_heatLabel,indx, patch,gn,0);
        new_dw->get(gamma[m],      lb->gammaLabel,        indx, patch,gn,0);
       
        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch); 
        cv[m].copyData(cv_ice);
      }                             // A L L  M A T L S

      new_dw->get(mass_L[m],        lb->mass_L_CCLabel,   indx, patch,gn, 0);
      new_dw->get(sp_vol_CC[m],     lb->sp_vol_CCLabel,   indx, patch,gn, 0);
      new_dw->get(mom_L[m],         lb->mom_L_CCLabel,    indx, patch,gn, 0);
      new_dw->get(int_eng_L[m],     lb->int_eng_L_CCLabel,indx, patch,gn, 0);
      new_dw->get(vol_frac_CC[m],   lb->vol_frac_CCLabel, indx, patch,gn, 0);
      new_dw->allocateAndPut(Tdot[m],        lb->Tdot_CCLabel,    indx,patch);
      new_dw->allocateAndPut(mom_L_ME[m],    lb->mom_L_ME_CCLabel,indx,patch);
      new_dw->allocateAndPut(int_eng_L_ME[m],lb->eng_L_ME_CCLabel,indx,patch);
    }

    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
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
        printData(   indx, patch,1, desc.str(),"vol_frac_CC",vol_frac_CC[m]);
        printVector( indx, patch,1, desc.str(),"mom_L", 0,   mom_L[m]);
        printVector( indx, patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);
      }
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      //---------- M O M E N T U M   E X C H A N G E
      //   Form BETA matrix (a), off diagonal terms
      //   beta and (a) matrix are common to all momentum exchanges
      for(int m = 0; m < numALLMatls; m++)  {
        tmp = delT*sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta(m,n) = vol_frac_CC[n][c]  * K(n,m) * tmp;
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

      for(int m = 0; m < numALLMatls; m++) {
        Vector sum(0,0,0);
        const Vector& vel_m = vel_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          sum += beta(m,n) *(vel_CC[n][c] - vel_m);
        }
        bb[m] = sum;
      }

      a.destructiveSolve(bb);

      for(int m = 0; m < numALLMatls; m++) {
        vel_CC[m][c] += bb[m];
      }

      //---------- E N E R G Y   E X C H A N G E     
      for(int m = 0; m < numALLMatls; m++) {
        tmp = delT*sp_vol_CC[m][c] / cv[m][c];
        for(int n = 0; n < numALLMatls; n++)  {
          beta(m,n) = vol_frac_CC[n][c] * H(n,m)*tmp;
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
      a.destructiveSolve(b);
      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = Temp_CC[m][c] + b[m];
      }
    }  //end CellIterator loop

  if(d_convective){
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
    FastMatrix cet(2,2),ac(2,2);
    double RHSc[2];
    cet.zero();
    int gm=d_conv_fluid_matlindex;  // gas matl from which to get heat
    int sm=d_conv_solid_matlindex;  // solid matl that heat goes to

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
              cet(0,0)=0.;
              cet(0,1)=delT*vol_frac_CC[gm][q]*H(sm,gm)*sp_vol_CC[sm][c]
                       /cv[sm][c];
              cet(1,0)=delT*vol_frac_CC[sm][c]*H(gm,sm)*sp_vol_CC[gm][q]
                       /cv[gm][q];
              cet(1,1)=0.;

              ac(0,1) = -cet(0,1);
              ac(1,0) = -cet(1,0);

              //   Form matrix (a) diagonal terms
              for(int m = 0; m < 2; m++) {
                ac(m,m) = 1.;
                for(int n = 0; n < 2; n++)   {
                  ac(m,m) +=  cet(m,n);
                }
              }
              
              RHSc[0] = cet(0,1)*(Temp_CC[gm][q] - Temp_CC[sm][c]);
              RHSc[1] = cet(1,0)*(Temp_CC[sm][c] - Temp_CC[gm][q]);
              ac.destructiveSolve(RHSc);
              Temp_CC[sm][c] += RHSc[0];
              Temp_CC[gm][q] += RHSc[1];
            }
          }  // if a surface cell
        }    // cellIterator
      }      // if mpm_matl
    }        // for ALL matls
   }

    /*`==========TESTING==========*/ 
    if(d_customBC_var_basket->usingLodi || 
       d_customBC_var_basket->usingNG_nozzle ||
       d_customBC_var_basket->usingMicroSlipBCs){ 
      StaticArray<CCVariable<double> > temp_CC_Xchange(numALLMatls);
      StaticArray<CCVariable<Vector> > vel_CC_Xchange(numALLMatls);      
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial(m);
        int indx = matl->getDWIndex();
        new_dw->allocateAndPut(temp_CC_Xchange[m],lb->temp_CC_XchangeLabel,indx,patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m], lb->vel_CC_XchangeLabel, indx,patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }
    
    preprocess_CustomBCs("CC_Exchange",old_dw, new_dw, lb, patch, 
                          999,d_customBC_var_basket);
    
/*===========TESTING==========`*/  
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch, d_sharedState, indx, new_dw,
                                                        d_customBC_var_basket);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_sharedState, 
                                         indx, new_dw,  d_customBC_var_basket);
    }
    
    delete_CustomBCs(d_customBC_var_basket);
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c]*cv[m][c] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]           * mass_L[m][c];
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
        printVector( indx, patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);
        printData(  indx, patch,1, desc.str(),"int_eng_L_ME",int_eng_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"Tdot",        Tdot[m]);
        printData(  indx, patch,1, desc.str(),"Temp_CC",     Temp_CC[m]);
      }
    }
  } //patches
}

/* _____________________________________________________________________
 Function~  ICE::maxMach_on_Lodi_BC_Faces
 Purpose~   Find the max mach Number on all lodi faces
 _____________________________________________________________________  */
void ICE::maxMach_on_Lodi_BC_Faces(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* /*matls*/,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing maxMach_on_Lodi_BC_Faces " <<
      patch->getID() << "\t\t\t ICE" << endl;
      
    Ghost::GhostType  gn = Ghost::None;
    int numICEMatls = d_sharedState->getNumICEMatls();
    StaticArray<constCCVariable<Vector> > vel_CC(numICEMatls);
    StaticArray<constCCVariable<double> > speedSound(numICEMatls);
          
    for(int m=0;m < numICEMatls;m++){
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if(ice_matl) {
        old_dw->get(vel_CC[m],      lb->vel_CCLabel,        indx,patch,gn,0);
        old_dw->get(speedSound[m],  lb->speedSound_CCLabel, indx,patch,gn,0);
      }
    }

    //__________________________________
    // Work on the lodi faces for each patch.
    // Every patch has to compute a maxMach
    // even if it isn't on a boundary.  We
    // can't do reduction variables with patch subsets yet.
    vector<Patch::FaceType>::iterator f ;
    
    for( f = d_customBC_var_basket->Lodi_var_basket->LodiFaces.begin();
         f !=d_customBC_var_basket->Lodi_var_basket->LodiFaces.end(); ++f) {
      Patch::FaceType face = *f;
      
      //__________________________________
      // compute maxMach number on this lodi face
      // only ICE matls
      double maxMach = 0.0;
      if (is_LODI_face(patch,face, d_sharedState) ) {
        
        for(int m=0; m < numICEMatls;m++){
          Material* matl = d_sharedState->getMaterial( m );
          ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
          if(ice_matl) {
            for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells");
                                                          !iter.done();iter++) {
              IntVector c = *iter;
              maxMach = Max(maxMach,vel_CC[m][c].length()/speedSound[m][c]);
            }
          }  // icematl
        }  // matl loop
        VarLabel* V_Label = getMaxMach_face_VarLabel(face);
        new_dw->put(max_vartype(maxMach), V_Label);
      }  // is lodi Face
    }  // boundaryFaces
  }  // patches
}

 
/* _____________________________________________________________________ 
Function~  ICE::update_q_CC--
Purpose~   This function tacks on the advection of q_CC. 
_____________________________________________________________________  */
template< class V, class T>
void ICE::update_q_CC(const std::string& desc,
                      CCVariable<T>& Q_CC,
                      V& Q_Lagrangian,
                      const CCVariable<T>& Q_advected,
                      const CCVariable<double>& mass_new,
                      const CCVariable<double>& cv_new,
                      const Patch* patch) 
{
  //__________________________________
  //  all Q quantites except Temperature
  if (desc != "energy"){
    for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      Q_CC[c] = (Q_Lagrangian[c] + Q_advected[c])/mass_new[c] ;
    }
  }
  //__________________________________
  //  Temperature
  if(desc == "energy" ) {
    for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      Q_CC[c] = (Q_Lagrangian[c] + Q_advected[c])/(mass_new[c] * cv_new[c]) ;
    }
  }
} 
/* _____________________________________________________________________ 
 Function~  ICE::advectAndAdvanceInTime--
 Purpose~
   This task calculates the The cell-centered, time n+1, mass, momentum
   and internal energy

   Need to include kinetic energy 
 _____________________________________________________________________  */
void ICE::advectAndAdvanceInTime(const ProcessorGroup* /*pg*/,
                                 const PatchSubset* patches,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const double AMR_subCycleProgressVar)
{
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
 
    cout_doing << "Doing Advect and Advance in Time on patch " << 
      patch->getID() << "\t\t ICE" << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    Vector dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();
    double invvol = 1.0/vol;


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

    int numMatls = d_sharedState->getNumICEMatls();

    for (int m = 0; m < numMatls; m++ ) {
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex(); 

      CCVariable<double> rho_CC, temp, sp_vol_CC,mach;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_L_ME, mass_L,sp_vol_L,speedSound, cv;
      constCCVariable<double> gamma, placeHolder;
      constCCVariable<Vector> mom_L_ME;
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      new_dw->get(gamma,       lb->gammaLabel,            indx,patch,gn,0);
      new_dw->get(speedSound,  lb->speedSound_CCLabel,    indx,patch,gn,0);
      new_dw->get(uvel_FC,     lb->uvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(vvel_FC,     lb->vvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(wvel_FC,     lb->wvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(mass_L,      lb->mass_L_CCLabel,        indx,patch,gac,2);
      new_dw->get(cv,          lb->specific_heatLabel,    indx,patch,gac,2);

      new_dw->get(mom_L_ME,    lb->mom_L_ME_CCLabel,      indx,patch,gac,2);
      new_dw->get(sp_vol_L,    lb->sp_vol_L_CCLabel,      indx,patch,gac,2);
      new_dw->get(int_eng_L_ME,lb->eng_L_ME_CCLabel,      indx,patch,gac,2);
      new_dw->getModifiable(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch);
      new_dw->getModifiable(rho_CC,    lb->rho_CCLabel,   indx,patch);

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
      // common variables that get passed into the advection operators
      advectVarBasket* varBasket = scinew advectVarBasket();
      varBasket->new_dw = new_dw;
      varBasket->old_dw = old_dw;
      varBasket->indx = indx;
      varBasket->patch = patch;
      varBasket->lb  = lb;
      varBasket->doAMR = false;
      varBasket->useCompatibleFluxes = d_useCompatibleFluxes;
      varBasket->AMR_subCycleProgressVar = AMR_subCycleProgressVar;

      //__________________________________
      //   Advection preprocessing
      bool bulletProof_test=true;
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                                    bulletProof_test, new_dw); 

      //__________________________________
      // Advect mass and backout rho_CC
      advector->advectMass(mass_L, mass_advected,  varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        mass_new[c]  = (mass_L[c] + mass_advected[c]);
        rho_CC[c]    = mass_new[c] * invvol;
      }   

      //__________________________________
      // Advect  momentum and backout vel_CC
      varBasket->is_Q_massSpecific   = true;
      varBasket->desc = "mom";
      
      advector->advectQ(mom_L_ME,mass_L,qV_advected, varBasket);

      update_q_CC<constCCVariable<Vector>, Vector>
                 ("velocity",vel_CC, mom_L_ME, qV_advected, mass_new,cv, patch);

      //__________________________________
      //    Jim's tweak
      //for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      //  IntVector c = *iter;
      //  if(rho_CC[c] < 1.e-2){             
      //    vel_CC[c] = Vector(0.,0.,0.); 
      //  }
      //}

      //__________________________________
      // Advection of specific volume
      // Note sp_vol_L[m] is actually sp_vol[m] * mass
      varBasket->is_Q_massSpecific = true;
      varBasket->desc = "sp_vol";
      
      advector->advectQ(sp_vol_L,mass_L, q_advected, varBasket); 

      update_q_CC<constCCVariable<double>, double>
             ("sp_vol",sp_vol_CC, sp_vol_L, q_advected, mass_new, cv,patch);

      //  Set Neumann = 0 if symmetric Boundary conditions
      setBC(sp_vol_CC, "set_if_sym_BC",patch, d_sharedState, indx, new_dw); 

      //__________________________________
      // Advect model variables 
      if(d_models.size() > 0 && d_modelSetup->tvars.size() > 0){
        vector<TransportedVariable*>::iterator t_iter;
        for( t_iter  = d_modelSetup->tvars.begin();
             t_iter != d_modelSetup->tvars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;
          if(tvar->matls->contains(indx)){
            CCVariable<double> q_CC;
            constCCVariable<double> q_L_CC;
            new_dw->allocateAndPut(q_CC, tvar->var,     indx, patch);
            new_dw->get(q_L_CC,   tvar->var_Lagrangian, indx, patch, gac, 2); 
            
            varBasket->is_Q_massSpecific = true;
            advector->advectQ(q_L_CC,mass_L,q_advected, varBasket);  
   
            update_q_CC<constCCVariable<double>, double>
                 ("q_L_CC",q_CC, q_L_CC, q_advected, mass_new, cv, patch);
                  
            //  Set Boundary Conditions 
            string Labelname = tvar->var->getName();
            setBC(q_CC, Labelname,  patch, d_sharedState, indx, new_dw);  
            
            //---- P R I N T   D A T A ------   
            if (switchDebug_advance_advect ) {
              ostringstream desc;
              desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"
                   <<patch->getID();
              string Lag_labelName = tvar->var_Lagrangian->getName();
              printData(indx, patch,1, desc.str(), Lag_labelName, q_L_CC);
              printData(indx, patch,1, desc.str(), Labelname,     q_CC);
            }    
          }
        }
      } 

      //__________________________________
      // A model *can* compute the specific heat
      CCVariable<double> cv_new;
      new_dw->allocateTemporary(cv_new, patch,gac,2);
      cv_new.copyData(cv);
      
      if(d_models.size() != 0){
        for(vector<ModelInterface*>::iterator iter = d_models.begin();
                                              iter != d_models.end(); iter++){ 
          ModelInterface* model = *iter;
          if(model->computesThermoTransportProps() ) {
            model->computeSpecificHeat(cv_new, patch, new_dw, indx);
          }
        }
      }

      //__________________________________
      // Advect internal energy and backout Temp_CC
      varBasket->is_Q_massSpecific = true;
      varBasket->desc = "int_eng";
      advector->advectQ(int_eng_L_ME, mass_L, q_advected, varBasket);
      
      update_q_CC<constCCVariable<double>, double>
            ("energy",temp, int_eng_L_ME, q_advected, mass_new, cv_new, patch);
      
      //__________________________________
      // set the boundary conditions
      preprocess_CustomBCs("Advection",old_dw, new_dw, lb,  patch, 999,
                           d_customBC_var_basket);
       
      setBC(rho_CC, "Density",  placeHolder, placeHolder,
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);
      setBC(vel_CC, "Velocity", 
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);       
      setBC(temp,"Temperature",gamma, cv,
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);
      
      delete_CustomBCs(d_customBC_var_basket);
                               
      //__________________________________
      // Compute Auxilary quantities
      for(CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done(); iter++){
        IntVector c = *iter;
        mach[c]  = vel_CC[c].length()/speedSound[c];
      }

      //---- P R I N T   D A T A ------   
      if (switchDebug_advance_advect ) {
       ostringstream desc;
       desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"<<patch->getID();
       printData(   indx, patch,1, desc.str(), "mass_L",        mass_L); 
       printData(   indx, patch,1, desc.str(), "mass_advected", mass_advected);
       printVector( indx, patch,1, desc.str(), "mom_L_CC", 0, mom_L_ME); 
       printData(   indx, patch,1, desc.str(), "sp_vol_L",    sp_vol_L);
       printData(   indx, patch,1, desc.str(), "int_eng_L_CC",int_eng_L_ME);
       printData(   indx, patch,1, desc.str(), "rho_CC",      rho_CC);
       printData(   indx, patch,1, desc.str(), "Temp_CC",     temp);
       printData(   indx, patch,1, desc.str(), "sp_vol_CC",   sp_vol_CC);
       printVector( indx, patch,1, desc.str(), "vel_CC", 0,   vel_CC);
      }
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      ostringstream warn;
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
      delete varBasket;
    }  // ice_matls loop
    delete advector;
  }  // patch loop
}
/*_______________________________________________________________________
 Function:  TestConservation--
 Purpose:   Test for conservation of mass, momentum, energy.   
            Test to see if the exchange process is conserving
_______________________________________________________________________ */
void ICE::TestConservation(const ProcessorGroup*,  
                           const PatchSubset* patches,
                           const MaterialSubset* /*matls*/,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(),level);
          
  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);
    
    cout_doing << "Doing TestConservation on patch " 
               << patch->getID() << "\t\t\t ICE" << endl;      
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();

    int numICEmatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType  gn  = Ghost::None;
    //__________________________________
    // get face centered velocities to 
    // to compute what's being fluxed through the domain
    StaticArray<constSFCXVariable<double> >uvel_FC(numICEmatls);
    StaticArray<constSFCYVariable<double> >vvel_FC(numICEmatls);
    StaticArray<constSFCZVariable<double> >wvel_FC(numICEmatls);
    for (int m = 0; m < numICEmatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->get(uvel_FC[m], lb->uvel_FCMELabel, indx,patch,gn,0);
      new_dw->get(vvel_FC[m], lb->vvel_FCMELabel, indx,patch,gn,0);
      new_dw->get(wvel_FC[m], lb->wvel_FCMELabel, indx,patch,gn,0);
    }
    
    double total_mass     = 0.0;
    double total_KE       = 0.0;
    double total_int_eng  = 0.0;
    Vector total_mom(0.0, 0.0, 0.0);
    Vector mom_exch_error(0,0,0);
    double eng_exch_error = 0;
    
    //__________________________________
    // conservation of mass  (Always computed)
    constCCVariable<double> rho_CC;
    StaticArray<CCVariable<double> > mass(numICEmatls);
    
    for (int m = 0; m < numICEmatls; m++ ) {
      
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->allocateTemporary(mass[m],patch);
      new_dw->get(rho_CC, lb->rho_CCLabel,   indx, patch, gn,0);
      
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        mass[m][c] = rho_CC[c] * cell_vol;
      }
      
      double mat_mass = 0;
      conservationTest<double>(patch, delT, mass[m], 
                               uvel_FC[m], vvel_FC[m], wvel_FC[m],mat_mass);
      total_mass += mat_mass;
    }
    //__________________________________
    // conservation of momentum
    if(d_conservationTest->momentum){
      CCVariable<Vector> mom;
      constCCVariable<Vector> vel_CC;
      new_dw->allocateTemporary(mom,patch);

      for (int m = 0; m < numICEmatls; m++ ) {

        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();
        new_dw->get(vel_CC, lb->vel_CCLabel,   indx, patch, gn,0);

        for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          mom[c] = mass[m][c] * vel_CC[c];
        }
        
        Vector mat_mom(0,0,0);
        conservationTest<Vector>(patch, delT, mom,
                                  uvel_FC[m],vvel_FC[m],wvel_FC[m], mat_mom);
        total_mom += mat_mom;
      }  
    }
    //__________________________________
    // conservation of internal_energy
    if(d_conservationTest->energy){
      CCVariable<double> int_eng;
      constCCVariable<double> temp_CC;
      constCCVariable<double> cv;
      new_dw->allocateTemporary(int_eng,patch);

      for (int m = 0; m < numICEmatls; m++ ) {

        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();
        new_dw->get(temp_CC, lb->temp_CCLabel,      indx, patch, gn,0);
        new_dw->get(cv,      lb->specific_heatLabel,indx, patch, gn,0);
        
        for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          int_eng[c] = mass[m][c] * cv[c] * temp_CC[c];
        }
        
        double mat_int_eng(0);
        
        conservationTest<double>(patch, delT, int_eng,
                                 uvel_FC[m],vvel_FC[m],wvel_FC[m], mat_int_eng);
        total_int_eng += mat_int_eng;
      }
    }
    //__________________________________
    // conservation of kinetic_energy
    if(d_conservationTest->energy){
      CCVariable<double> KE;
      constCCVariable<Vector> vel_CC;
      new_dw->allocateTemporary(KE,patch);

      for (int m = 0; m < numICEmatls; m++ ) {

        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();
        new_dw->get(vel_CC, lb->vel_CCLabel,indx, patch, gn,0);
        
        for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          double vel_mag = vel_CC[c].length();
          KE[c] = 0.5 * mass[m][c] * vel_mag * vel_mag;
        }
        
        double mat_KE(0);
        conservationTest<double>(patch, delT, KE,
                                  uvel_FC[m],vvel_FC[m],wvel_FC[m], mat_KE);
        total_KE += mat_KE;
      }
    }
    //__________________________________
    // conservation during the exchange process
    if(d_conservationTest->exchange){
      Vector sum_mom_L_CC     = Vector(0.0, 0.0, 0.0);
      Vector sum_mom_L_ME_CC  = Vector(0.0, 0.0, 0.0);
      double sum_int_eng_L_CC = 0.0;
      double sum_eng_L_ME_CC  = 0.0;

      int numALLmatls = d_sharedState->getNumMatls();
      for(int m = 0; m < numALLmatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        constCCVariable<double> int_eng_L_CC, eng_L_ME_CC;
        constCCVariable<Vector> mom_L_CC, mom_L_ME_CC;
        new_dw->get(mom_L_CC,     lb->mom_L_CCLabel,     indx, patch,gn, 0);
        new_dw->get(int_eng_L_CC, lb->int_eng_L_CCLabel, indx, patch,gn, 0);
        new_dw->get(mom_L_ME_CC,  lb->mom_L_ME_CCLabel,  indx, patch,gn, 0);
        new_dw->get(eng_L_ME_CC,  lb->eng_L_ME_CCLabel,  indx, patch,gn, 0); 

        for (CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          sum_mom_L_CC     += mom_L_CC[c];     
          sum_mom_L_ME_CC  += mom_L_ME_CC[c];  
          sum_int_eng_L_CC += int_eng_L_CC[c]; 
          sum_eng_L_ME_CC  += eng_L_ME_CC[c];  
        }
      }
      mom_exch_error = sum_mom_L_CC     - sum_mom_L_ME_CC;
      eng_exch_error = sum_int_eng_L_CC - sum_eng_L_ME_CC;
    } 
    new_dw->put(sum_vartype(total_mass),        lb->TotalMassLabel);
    new_dw->put(sumvec_vartype(total_mom),      lb->CenterOfMassVelocityLabel);
    new_dw->put(sum_vartype(total_int_eng),     lb->TotalIntEngLabel);
    new_dw->put(sum_vartype(total_KE),          lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(mom_exch_error), lb->mom_exch_errorLabel);
    new_dw->put(sum_vartype(eng_exch_error),    lb->eng_exch_errorLabel);
  }  // patch loop
}


/*_____________________________________________________________________
 Function:  hydrostaticPressureAdjustment--
 Notes:     press_hydro_ = rho_micro_CC[SURROUNDING_MAT] * grav * some_distance
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
    for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
                                                             iter++) {
      IntVector c = *iter;
      dist_from_p_ref  = (double) (c.x() - press_ref_x) * dx.x();
      press_hydro      = rho_micro_CC[c] * gravity.x() * dist_from_p_ref;
      press_CC[c] += press_hydro;
    }
  }
  //__________________________________
  //  Y direction
  if (gravity.y() != 0.)  {
    for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
                                                             iter++) {
      IntVector c = *iter;
      dist_from_p_ref = (double) (c.y() - press_ref_y) * dx.y();
      press_hydro     = rho_micro_CC[c] * gravity.y() * dist_from_p_ref;
      press_CC[c] += press_hydro;
    }
  }
  //__________________________________
  //  Z direction
  if (gravity.z() != 0.)  {
    for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); 
                                                             iter++) {
      IntVector c = *iter;
      dist_from_p_ref   = (double) (c.z() - press_ref_z) * dx.z();
      press_hydro       = rho_micro_CC[c] * gravity.z() * dist_from_p_ref;
      press_CC[c] += press_hydro;
    }
  }   
}

/*_____________________________________________________________________
 Function~  ICE::getExchangeCoefficients--
 _____________________________________________________________________  */
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
/*_____________________________________________________________________
 Function~  ICE::upwindCell--
 purpose:   find the upwind cell in each direction  This is a knock off
            of Bucky's logic
 _____________________________________________________________________  */
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
/*_____________________________________________________________________
 Function~  ICE::areAllValuesPositive--
 _____________________________________________________________________  */
bool ICE::areAllValuesPositive( CCVariable<double> & src, IntVector& neg_cell )
{ 
  double numCells = 0;
  double sum_src = 0;
  int sumNan = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  CellIterator iterLim = CellIterator(l,h);
  
  for(CellIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumNan += isnan(src[c]);       // check for nans
    sum_src += src[c]/fabs(src[c]);
    numCells++;
  }

  // now find the first cell where the value is < 0   
  if ( (fabs(sum_src - numCells) > 1.0e-2) || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (src[c] < 0.0 || isnan(src[c]) !=0) {
        neg_cell = c;
        return false;
      }
    }
  } 
  neg_cell = IntVector(0,0,0); 
  return true;      
} 
//______________________________________________________________________
//  Models
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
  t->var_Lagrangian = VarLabel::create(var->getName()+"_L", var->typeDescription());
  tvars.push_back(t);
}

//_____________________________________________________________________
//   Stub functions for AMR
void
ICE::addRefineDependencies( Task*, const VarLabel*, int , int )
{
}

void
ICE::refineBoundaries(const Patch*, CCVariable<double>&,
			    DataWarehouse*, const VarLabel*,
			    int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}
void
ICE::refineBoundaries(const Patch*, CCVariable<Vector>&,
			    DataWarehouse*, const VarLabel*,
			    int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}
void
ICE::refineBoundaries(const Patch*, SFCXVariable<double>&,
			    DataWarehouse*, const VarLabel*,
			    int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

void
ICE::refineBoundaries(const Patch*, SFCYVariable<double>&,
			    DataWarehouse*, const VarLabel*,
			    int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

void
ICE::refineBoundaries(const Patch*, SFCZVariable<double>&,
			    DataWarehouse*, const VarLabel*,
			    int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

bool ICE::needRecompile(double /*time*/, double /*dt*/, const GridP& /*grid*/)
{
  if(d_recompile){
    d_recompile = false;
    return true;
  }
  else{
    return false;
  }
}
//______________________________________________________________________
//      Dynamic material addition
void ICE::scheduleCheckNeedAddMaterial(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* /*ice_matls*/)
{
  if(d_models.size() != 0){
    cout_doing << "ICE::scheduleCheckNeedAddMaterial" << endl;
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
       iter != d_models.end(); iter++){
      ModelInterface* model = *iter;
      model->scheduleCheckNeedAddMaterial(sched, level, d_modelInfo);
    }
  }
}
//__________________________________
void ICE::scheduleSetNeedAddMaterialFlag(SchedulerP& sched,
                                       const LevelP& level,
                                       const MaterialSet* all_matls)
{
  if(d_models.size() != 0){
    cout_doing << "ICE::scheduleSetNeedAddMaterialFlag" << endl;
    Task* t= scinew Task("ICE::setNeedAddMaterialFlag",
                 this, &ICE::setNeedAddMaterialFlag);
    t->requires(Task::NewDW, lb->NeedAddIceMaterialLabel);
    sched->addTask(t, level->eachPatch(), all_matls);
  }
}
//__________________________________
void ICE::setNeedAddMaterialFlag(const ProcessorGroup*,
                                 const PatchSubset* /*patches*/,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)
{
    sum_vartype need_add_flag;
    new_dw->get(need_add_flag, lb->NeedAddIceMaterialLabel);

    if(need_add_flag>0.1){
      d_sharedState->setNeedAddMaterial(1);
    }
    else{
      d_sharedState->setNeedAddMaterial(0);
    }
 }

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif


/*______________________________________________________________________
          S C H E M A T I C   D I A G R A M S

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

