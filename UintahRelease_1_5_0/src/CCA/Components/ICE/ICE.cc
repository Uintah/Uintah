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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#define _CPP_CMATH
#endif
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/impAMRICE.h>
#include <CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <CCA/Components/ICE/ConservationTest.h>
#include <CCA/Components/ICE/Diffusion.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <CCA/Components/ICE/TurbulenceFactory.h>
#include <CCA/Components/ICE/Turbulence.h>
#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/ModelMaker.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Math/FastMatrix.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/Expon.h>
#include <Core/Util/DebugStream.h>

#include   <vector>
#include   <sstream>
#include   <iostream>

#include <cfloat>
#include <sci_defs/hypre_defs.h>

#ifdef HAVE_HYPRE
#include <CCA/Components/Solvers/HypreSolver.h>
#endif

#define SET_CFI_BC 0

using namespace std;
using namespace Uintah;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "ICE_NORMAL_COUT:+,ICE_DOING_COUT:+"
//  ICE_NORMAL_COUT:  dumps out during problemSetup 
//  ICE_DOING_COUT:   dumps when tasks are scheduled and performed
//  default is OFF
static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);
static DebugStream ds_EqPress("DBG_EqPress",false);


ICE::ICE(const ProcessorGroup* myworld, const bool doAMR) :
  UintahParallelComponent(myworld)
{
  lb   = scinew ICELabel();

#ifdef HAVE_HYPRE
  hypre_solver_label = VarLabel::create("hypre_solver_label",
                                        SoleVariable<hypre_solver_structP>::getTypeDescription());
#endif

  d_doAMR               = doAMR;
  d_doRefluxing         = false;
  d_add_heat            = false;
  d_impICE              = false;
  d_useCompatibleFluxes = true;
  
  d_max_iter_equilibration  = 100;
  d_delT_knob               = 1.0;
  d_delT_scheme             = "aggressive";
  d_surroundingMatl_indx    = -9;
  d_dbgVar1   = 0;     //inputs for debugging                               
  d_dbgVar2   = 0;
  d_EVIL_NUM  = -9.99e30;                                                    
  d_SMALL_NUM = 1.0e-100;                                                   
  d_modelInfo = 0;
  d_modelSetup = 0;
  d_recompile               = false;
  d_canAddICEMaterial       = false;
  d_with_mpm                = false;
  d_with_rigid_mpm          = false;
  d_clampSpecificVolume     = false;
  
  d_exchCoeff = scinew ExchangeCoefficients();
  
  d_conservationTest         = scinew conservationTest_flags();
  d_conservationTest->onOff = false;

  d_customInitialize_basket  = scinew customInitialize_basket();
  d_customBC_var_basket  = scinew customBC_var_basket();
  d_customBC_var_basket->Lodi_var_basket =  scinew Lodi_variable_basket();
  d_customBC_var_basket->Slip_var_basket =  scinew Slip_variable_basket();
  d_customBC_var_basket->mms_var_basket  =  scinew mms_variable_basket();
  d_customBC_var_basket->sine_var_basket =  scinew sine_variable_basket();
  d_press_matl    = 0;
  d_press_matlSet = 0;
}

ICE::~ICE()
{
  cout_doing << d_myworld->myrank() << " Doing: ICE destructor " << endl;

#ifdef HAVE_HYPRE
  VarLabel::destroy(hypre_solver_label);
#endif

  delete d_customInitialize_basket;
  delete d_customBC_var_basket->Lodi_var_basket;
  delete d_customBC_var_basket->Slip_var_basket;
  delete d_customBC_var_basket->mms_var_basket;
  delete d_customBC_var_basket->sine_var_basket;  
  delete d_customBC_var_basket;
  delete d_conservationTest;
  delete lb;
  delete d_advector;
  delete d_exchCoeff;

  if(d_turbulence){
    delete d_turbulence;
  }

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      delete *iter;
    }
  }
  
  if (d_press_matl && d_press_matl->removeReference()){
    delete d_press_matl;
  }
  if (d_press_matlSet && d_press_matlSet->removeReference()){
    delete d_press_matlSet;
  }
  if (d_solver_parameters) {
    delete d_solver_parameters;
  }
  //__________________________________
  // MODELS
  cout_doing << d_myworld->myrank() << " Doing: destorying Model Machinery " << endl;
  if(d_modelSetup){
    // delete transported Lagrangian variables
    vector<TransportedVariable*>::iterator t_iter;
    for( t_iter  = d_modelSetup->tvars.begin();
        t_iter != d_modelSetup->tvars.end(); t_iter++){
        TransportedVariable* tvar = *t_iter;
        VarLabel::destroy(tvar->var_Lagrangian);
        VarLabel::destroy(tvar->var_adv);
        delete tvar;
    }
    cout_doing << d_myworld->myrank() << " Doing: destorying refluxing variables " << endl;
    // delete refluxing variables
    vector<AMR_refluxVariable*>::iterator iter;
    for( iter  = d_modelSetup->d_reflux_vars.begin();
         iter != d_modelSetup->d_reflux_vars.end(); iter++){
         AMR_refluxVariable* rvar = *iter;
      VarLabel::destroy(rvar->var_X_FC_flux);
      VarLabel::destroy(rvar->var_Y_FC_flux);
      VarLabel::destroy(rvar->var_Z_FC_flux);
      VarLabel::destroy(rvar->var_X_FC_corr);
      VarLabel::destroy(rvar->var_Y_FC_corr);
      VarLabel::destroy(rvar->var_Z_FC_corr);
    } 
    delete d_modelSetup;
  }
  
  // delete models
  if(d_modelInfo){
    delete d_modelInfo;
  }
  //  releasePort("solver");
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
void ICE::problemSetup(const ProblemSpecP& prob_spec, 
                       const ProblemSpecP& restart_prob_spec,
                       GridP& grid, SimulationStateP&   sharedState)
{
  cout_doing << d_myworld->myrank() << " Doing ICE::problemSetup " << "\t\t\t ICE" << endl;
  d_sharedState = sharedState;
  d_press_matl = scinew MaterialSubset();
  d_press_matl->add(0);
  d_press_matl->addReference();
  
  d_press_matlSet  = scinew MaterialSet();
  d_press_matlSet->add(0);
  d_press_matlSet->addReference();

  d_solver_parameters = 0;
  
  dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!dataArchiver){
    throw InternalError("ICE:couldn't get output port", __FILE__, __LINE__);
  }
  d_solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!d_solver) {
    throw InternalError("ICE:couldn't get solver port", __FILE__, __LINE__);
  }

   d_ref_press = 0.0;

  ProblemSpecP phys_cons_ps = prob_spec->findBlock("PhysicalConstants");
  if(phys_cons_ps){
    phys_cons_ps->require("reference_pressure",d_ref_press);
    phys_cons_ps->require("gravity",d_gravity);
  } else {
    throw ProblemSetupException(                                                
     "\n Could not find the <PhysicalConstants> section in the input file.  This section contains <gravity> and <reference pressure> \n"  
     " This pressure is used during the problem intialization and when\n"       
     " the pressure gradient is interpolated to the MPM particles \n"           
     " you must have it for all MPMICE and multimaterial ICE problems\n",       
     __FILE__, __LINE__);                                                       
  }

  //__________________________________
  // read in all the printData switches
  printData_problemSetup( prob_spec);

  //__________________________________
  // Pull out from CFD-ICE section
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");

  if(!cfd_ps){
    throw ProblemSetupException(                                                                    
     "\n Could not find the <CFD> section in the input file\n",__FILE__, __LINE__);    
  }

  cfd_ps->require("cfl",d_CFL);
  cfd_ps->get("CanAddICEMaterial",d_canAddICEMaterial);
  
  ProblemSpecP cfd_ice_ps = cfd_ps->findBlock("ICE");
  if(!cfd_ice_ps){
    throw ProblemSetupException(                                                                    
     "\n Could not find the <CFD> <ICE> section in the input file\n",__FILE__, __LINE__);    
  }
   
  
  cfd_ice_ps->get("max_iteration_equilibration",d_max_iter_equilibration);
  cfd_ice_ps->get("ClampSpecificVolume",d_clampSpecificVolume);
  
  d_advector = AdvectionFactory::create(cfd_ice_ps, d_useCompatibleFluxes,
                                        d_OrderOfAdvection);
  //__________________________________
  //  Pull out add heat section
  ProblemSpecP add_heat_ps = cfd_ice_ps->findBlock("ADD_HEAT");
  if(add_heat_ps) {
    d_add_heat = true;
    add_heat_ps->require("add_heat_matls",d_add_heat_matls);
    add_heat_ps->require("add_heat_coeff",d_add_heat_coeff);
    add_heat_ps->require("add_heat_t_start",d_add_heat_t_start);
    add_heat_ps->require("add_heat_t_final",d_add_heat_t_final); 
  }
 
 //__________________________________
 //  custom Initialization
  customInitialization_problemSetup(cfd_ice_ps, d_customInitialize_basket, grid);
  
  //__________________________________
  // Pull out implicit solver parameters
  ProblemSpecP impSolver = cfd_ice_ps->findBlock("ImplicitSolver");
  if (impSolver) {
    d_delT_knob = 0.5;      // default value when running implicit
    d_solver_parameters = d_solver->readParameters(impSolver, 
                                                   "implicitPressure",
                                                   sharedState);
    d_solver_parameters->setSolveOnExtraCells(false);
    d_solver_parameters->setRestartTimestepOnFailure(true);
    impSolver->require("max_outer_iterations",      d_max_iter_implicit);
    impSolver->require("outer_iteration_tolerance", d_outer_iter_tolerance);
    impSolver->getWithDefault("iters_before_timestep_restart",    
                               d_iters_before_timestep_restart, 5);
    d_impICE = true;
    Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
    d_subsched = sched->createSubScheduler();
    d_subsched->initialize(3,1);
    d_subsched->setRestartable(true); 
    d_subsched->clearMappings();
    d_subsched->mapDataWarehouse(Task::ParentOldDW, 0);
    d_subsched->mapDataWarehouse(Task::ParentNewDW, 1);
    d_subsched->mapDataWarehouse(Task::OldDW, 2);
    d_subsched->mapDataWarehouse(Task::NewDW, 3);

#ifdef HAVE_HYPRE
    d_subsched->overrideVariableBehavior(hypre_solver_label->getName(),false,
                                         false,false,true,true);
#endif
  
    d_recompileSubsched = true;

    //__________________________________
    // bulletproofing
    double tol;
    ProblemSpecP p = impSolver->findBlock("Parameters");
    p->get("tolerance",tol);
    if(tol>= d_outer_iter_tolerance){
      ostringstream msg;
      msg << "\n ERROR: implicit pressure: The <outer_iteration_tolerance>"
          << " must be greater than the solver tolerance <tolerance> \n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    } 
    
    if(d_doAMR  && d_solver->getName() != "hypreamr"){
      ostringstream msg;
      msg << "\n ERROR: " << d_solver->getName()
          << " cannot be used with an AMR grid \n";
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }

  //__________________________________
  // Pull out TimeStepControl data
  ProblemSpecP tsc_ps = cfd_ice_ps->findBlock("TimeStepControl");
  if (tsc_ps ) {
    tsc_ps ->require("Scheme_for_delT_calc", d_delT_scheme);
    tsc_ps ->require("knob_for_speedSound",  d_delT_knob);
    
    if (d_delT_scheme != "conservative" && d_delT_scheme != "aggressive") {
     string warn="ERROR:\n Scheme_for_delT_calc:  must specify either aggressive or conservative";
     throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
    if (d_delT_knob< 0.0 || d_delT_knob > 1.0) {
     string warn="ERROR:\n knob_for_speedSound:  must be between 0 and 1";
     throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
  } 
  
  //__________________________________
  // Pull out Initial Conditions
  ProblemSpecP mat_ps = 0;

  if (prob_spec->findBlockWithOutAttribute("MaterialProperties")){
    mat_ps = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  }else if (restart_prob_spec){
    mat_ps = 
      restart_prob_spec->findBlockWithOutAttribute("MaterialProperties");
  }
  
  ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");  

  for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
    ps = ps->findNextBlock("material") ) {
    string index("");
    ps->getAttribute("index",index);
    std::stringstream id(index);

    const int DEFAULT_VALUE = -1;

    int index_val = DEFAULT_VALUE;
    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems (redstorm) it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    //cout_norm << "Material attribute = " << index_val << endl;

    // Extract out the type of EOS and the associated parameters
    ICEMaterial *mat = scinew ICEMaterial(ps);
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      sharedState->registerICEMaterial(mat,index_val);
    }else{
      sharedState->registerICEMaterial(mat);
    }
      
    if(mat->isSurroundingMatl()) {
      d_surroundingMatl_indx = mat->getDWIndex();  //which matl. is the surrounding matl
    } 
  } 

  //_________________________________
  // Exchange Coefficients
  proc0cout << "numMatls " << d_sharedState->getNumMatls() << endl;
  
  d_exchCoeff->problemSetup(mat_ps, sharedState);
  
  if (d_exchCoeff->d_heatExchCoeffModel != "constant"){
    proc0cout << "------------------------------Using Variable heat exchange coefficients"<< endl;
  }
  
  //__________________________________
  // Set up turbulence models - needs to be done after materials are initialized
  d_turbulence = TurbulenceFactory::create(cfd_ice_ps, sharedState);

  //__________________________________
  //  conservationTest
  if (dataArchiver->isLabelSaved("TotalMass") ){
    d_conservationTest->mass     = true;
    d_conservationTest->onOff    = true;
  }
  if (dataArchiver->isLabelSaved("TotalMomentum") ){
    d_conservationTest->momentum = true;
    d_conservationTest->onOff    = true;
  }
  if (dataArchiver->isLabelSaved("TotalIntEng")   || 
      dataArchiver->isLabelSaved("KineticEnergy") ){
    d_conservationTest->energy   = true;
    d_conservationTest->onOff    = true;
  }
  if (dataArchiver->isLabelSaved("eng_exch_error") ||
      dataArchiver->isLabelSaved("mom_exch_error") ){
    d_conservationTest->exchange = true;
    d_conservationTest->onOff    = true;
  }

  //__________________________________
  // WARNINGS
  SimulationTime timeinfo(prob_spec); 
  if ( d_impICE &&  timeinfo.max_delt_increase  > 10 && d_myworld->myrank() == 0){
    cout <<"\n \n W A R N I N G: " << endl;
    cout << " When running implicit ICE you should specify "<<endl;
    cout <<" \t \t <max_delt_increase>    2.0ish  "<<endl;
    cout << " to a) prevent rapid fluctuations in the timestep and "<< endl;
    cout << "    b) to prevent outflux Vol > cell volume \n \n" <<endl;
  } 
  
  //__________________________________
  //  Custom BC setup

  d_customBC_var_basket->d_gravity    = d_gravity;
  d_customBC_var_basket->sharedState  = sharedState;
  
  d_customBC_var_basket->usingLodi = 
        read_LODI_BC_inputs(prob_spec,       sharedState, d_customBC_var_basket->Lodi_var_basket);
  d_customBC_var_basket->usingMicroSlipBCs =
        read_MicroSlip_BC_inputs(prob_spec,  d_customBC_var_basket->Slip_var_basket);
  d_customBC_var_basket->using_MMS_BCs =
        read_MMS_BC_inputs(prob_spec,        d_customBC_var_basket->mms_var_basket);
  d_customBC_var_basket->using_Sine_BCs =
        read_Sine_BC_inputs(prob_spec,       d_customBC_var_basket->sine_var_basket);
  //__________________________________
  //  boundary condition warnings
  BC_bulletproofing(prob_spec,sharedState);
  
  //__________________________________
  //  Load Model info.
  // If we are doing a restart, then use the "timestep.xml" 
  ProblemSpecP orig_or_restart_ps = 0;
  if (prob_spec->findBlockWithOutAttribute("MaterialProperties")){
    orig_or_restart_ps = prob_spec;
  }else if (restart_prob_spec){
    orig_or_restart_ps = restart_prob_spec;
  }  
    
  ModelMaker* modelMaker = dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  if(modelMaker){

    modelMaker->makeModels(orig_or_restart_ps, prob_spec, grid, sharedState, d_doAMR);
    d_models = modelMaker->getModels();
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
      is_BC_specified(prob_spec, Labelname, tvar->matls);
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
  
  //__________________________________
  //  Set up data analysis modules
  if(!d_with_mpm){
    d_analysisModules = AnalysisModuleFactory::create(prob_spec, sharedState, dataArchiver);

    if(d_analysisModules.size() != 0){
      vector<AnalysisModule*>::iterator iter;
      for( iter  = d_analysisModules.begin();
           iter != d_analysisModules.end(); iter++){
        AnalysisModule* am = *iter;
        am->problemSetup(prob_spec, grid, sharedState);
      }
    }
  }  // mpm

}
/*______________________________________________________________________
 Function~  ICE::addMaterial--
 Purpose~   read in the exchange coefficients
 _____________________________________________________________________*/
void ICE::addMaterial(const ProblemSpecP& prob_spec, 
                      GridP& grid,
                      SimulationStateP& sharedState)
{
  cout_doing << d_myworld->myrank() << " Doing ICE::addMaterial " << "\t\t\t ICE" << endl;
  d_recompile = true;
  ProblemSpecP mat_ps =   
    prob_spec->findBlockWithAttribute("MaterialProperties","add");

  string attr = "";
  mat_ps->getAttribute("add",attr);
  
  if (attr == "true") {
    ProblemSpecP ice_mat_ps   = mat_ps->findBlock("ICE");  
    
    for (ProblemSpecP ps = ice_mat_ps->findBlock("material"); ps != 0;
         ps = ps->findNextBlock("material") ) {
      ICEMaterial *mat = scinew ICEMaterial(ps);
      sharedState->registerICEMaterial(mat);
    }
    
    d_exchCoeff->problemSetup(mat_ps, sharedState);
    
    // problem setup for each model  
    for(vector<ModelInterface*>::iterator iter = d_models.begin();
        iter != d_models.end(); iter++){
      (*iter)->activateModel(grid, sharedState, d_modelSetup);
    }
  }
}
/*______________________________________________________________________
 Function~  ICE::updateExchangeCoefficients--
 Purpose~   read in the exchange coefficients after a material has been
            dynamically added
 _____________________________________________________________________*/
void ICE::updateExchangeCoefficients(const ProblemSpecP& prob_spec, 
                                     GridP& /*grid*/,
                                     SimulationStateP&  sharedState)
{
  cout << "Updating Ex Coefficients" << endl;
  ProblemSpecP mat_ps  =  
    prob_spec->findBlockWithAttribute("MaterialProperties","add");

  string attr = "";
  mat_ps->getAttribute("add",attr);
  
  if (attr == "true")
    d_exchCoeff->problemSetup(mat_ps, sharedState);
}
/*______________________________________________________________________
 Function~  ICE::outputProblemSpec--
 Purpose~   outputs material state
 _____________________________________________________________________*/
void ICE::outputProblemSpec(ProblemSpecP& root_ps)
{
  cout_doing << d_myworld->myrank() << " Doing ICE::addMaterial " << "\t\t\t ICE" << endl;

  ProblemSpecP root = root_ps->getRootNode();

  ProblemSpecP mat_ps = 0;
  mat_ps = root->findBlockWithOutAttribute("MaterialProperties");

  if (mat_ps == 0)
    mat_ps = root->appendChild("MaterialProperties");

  ProblemSpecP ice_ps = mat_ps->appendChild("ICE");
  for (int i = 0; i < d_sharedState->getNumICEMatls();i++) {
    ICEMaterial* mat = d_sharedState->getICEMaterial(i);
    mat->outputProblemSpec(ice_ps);
  }
  d_exchCoeff->outputProblemSpec(mat_ps);

  ProblemSpecP models_ps = root->appendChild("Models");

  ModelMaker* modelmaker = 
    dynamic_cast<ModelMaker*>(getPort("modelmaker"));
  
  if (modelmaker) {
    modelmaker->outputProblemSpec(models_ps);
  }
}

/*______________________________________________________________________
 Function~  ICE::scheduleInitializeAddedMaterial--
 _____________________________________________________________________*/
void ICE::scheduleInitializeAddedMaterial(const LevelP& level,SchedulerP& sched)
{
  cout_doing << d_myworld->myrank() << " Doing ICE::scheduleInitializeAddedMaterial \t\t\t\tL-"
             <<level->getIndex() << endl;
             
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
    cout_doing << d_myworld->myrank() << " Doing InitializeAddedMaterial on patch " << patch->getID() 
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
            
    SpecificHeat *cvModel = ice_matl->getSpecificHeatModel();
    if(cvModel != 0) {
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        cv[*iter] = cvModel->getSpecificHeat(Temp_CC[*iter]);
        gamma[*iter] = cvModel->getGamma(Temp_CC[*iter]);
      }
    }
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
  cout_doing << d_myworld->myrank() << " Doing ICE::scheduleInitialize \t\t\t\tL-"
             <<level->getIndex() << endl;
  
  Task* t = scinew Task("ICE::actuallyInitialize",
                  this, &ICE::actuallyInitialize);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
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

  if (d_impICE)
    d_solver->scheduleInitialize(level,sched,
                                 d_sharedState->allICEMaterials());
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
  // dataAnalysis 
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }
 
  //__________________________________
  // Make adjustments to the hydrostatic pressure
  // and temperature fields.  You need to do this
  // after the models have initialized the flowfield
  Vector grav = getGravity();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  if (grav.length() > 0 ) {
    cout_doing << d_myworld->myrank() << " Doing ICE::scheduleHydroStaticAdj " << endl;
    Task* t2 = scinew Task("ICE::initializeSubTask_hydrostaticAdj",
                     this, &ICE::initializeSubTask_hydrostaticAdj);
    Ghost::GhostType  gn  = Ghost::None;
    t2->requires(Task::NewDW,lb->gammaLabel,         ice_matls_sub, gn);
    t2->requires(Task::NewDW,lb->specific_heatLabel, ice_matls_sub, gn);
   
    t2->modifies(lb->rho_micro_CCLabel);
    t2->modifies(lb->temp_CCLabel);
    t2->modifies(lb->press_CCLabel, d_press_matl, oims); 

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
  cout_doing << d_myworld->myrank() << " Doing restartInitialize "<< "\t\t\t ICE" << endl;

  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->restartInitialize();
    }
  }
  
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
      d_surroundingMatl_indx = ice_matl->getDWIndex();
    } 
  }
  
  // --------bulletproofing
  Vector grav = getGravity();
  if (grav.length() >0.0 && d_surroundingMatl_indx == -9)  {
    throw ProblemSetupException("ERROR ICE::restartInitialize \n"
          "You must have \n" 
          "       <isSurroundingMatl> true </isSurroundingMatl> \n "
          "specified inside the ICE material that is the background matl\n",
                                __FILE__, __LINE__);
  }
}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeStableTimestep--
_____________________________________________________________________*/
void ICE::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  Task* t = 0;
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeStableTimestep \t\t\t\tL-"
             <<level->getIndex() << endl;
  t = scinew Task("ICE::actuallyComputeStableTimestep",
                   this, &ICE::actuallyComputeStableTimestep);

  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
                            
  t->requires(Task::NewDW, lb->vel_CCLabel,        gac, 1, true);  
  t->requires(Task::NewDW, lb->speedSound_CCLabel, gac, 1, true);
  t->requires(Task::NewDW, lb->thermalCondLabel,   gn,  0, true);
  t->requires(Task::NewDW, lb->gammaLabel,         gn,  0, true);
  t->requires(Task::NewDW, lb->specific_heatLabel, gn,  0, true);   
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,   gn,  0, true);   
  t->requires(Task::NewDW, lb->viscosityLabel,   gn,  0, true);        
  
  t->computes(d_sharedState->get_delt_label(),level.get_rep());
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
ICE::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  MALLOC_TRACE_TAG_SCOPE("ICE::scheduleTimeAdvance()");
  // for AMR, we need to reset the initial Delt otherwise some unsuspecting level will
  // get the init delt when it didn't compute delt on L0.
  
  cout_doing << d_myworld->myrank() << " --------------------------------------------------------L-" 
             <<level->getIndex()<< endl;
  cout_doing << d_myworld->myrank() << " ICE::scheduleTimeAdvance\t\t\t\tL-" <<level->getIndex()<< endl;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  

  MaterialSubset* one_matl = d_press_matl;
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();
  const MaterialSubset* mpm_matls_sub = mpm_matls->getUnion();

  if(d_turbulence){
    // The turblence model is also called directly from
    // accumlateMomentumSourceSinks.  
    d_turbulence->scheduleComputeVariance(sched, patches, ice_matls);
  }

  scheduleMaxMach_on_Lodi_BC_Faces(       sched, level,   ice_matls);
                                                          
  scheduleComputeThermoTransportProperties(sched, level,  ice_matls);
  
  scheduleComputePressure(                sched, patches, d_press_matl,
                                                          all_matls);

  scheduleComputeTempFC(                   sched, patches, ice_matls_sub,  
                                                           mpm_matls_sub,
                                                           all_matls);    
                                                                 
  scheduleComputeModelSources(             sched, level,   all_matls);

  scheduleUpdateVolumeFraction(            sched, level,   d_press_matl,
                                                           all_matls);


  scheduleComputeVel_FC(                   sched, patches,ice_matls_sub, 
                                                          mpm_matls_sub, 
                                                          d_press_matl,    
                                                          all_matls);        

  scheduleAddExchangeContributionToFCVel( sched, patches,ice_matls_sub,
                                                         all_matls,
                                                         false);
                                                          
  if(d_impICE) {        //  I M P L I C I T
  
#ifdef HAVE_HYPRE
    sched->overrideVariableBehavior(hypre_solver_label->getName(),false,false,
                                    false,true,true);
#endif

    scheduleSetupRHS(                     sched, patches,  one_matl, 
                                                           all_matls,
                                                           false,
                                                           "computes");
    
    scheduleCompute_maxRHS(                sched, level,   one_matl,
                                                           all_matls);
    
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
                                   
  scheduleAdvectAndAdvanceInTime(         sched, patches, ice_matls_sub,
                                                          all_matls);
                                                          
  scheduleConservedtoPrimitive_Vars(      sched, patches, ice_matls_sub,
                                                          all_matls,
                                                          "afterAdvection");
}
/* _____________________________________________________________________
 Function~  ICE::scheduleFinalizeTimestep--
  This is called after scheduleTimeAdvance and the scheduleCoarsen
_____________________________________________________________________*/
void
ICE::scheduleFinalizeTimestep( const LevelP& level, SchedulerP& sched)
{
  cout_doing << "----------------------------"<<endl;  
  cout_doing << d_myworld->myrank() << " ICE::scheduleFinalizeTimestep\t\t\t\t\tL-" <<level->getIndex()<< endl;
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();  
  const MaterialSubset* ice_matls_sub = ice_matls->getUnion();


  scheduleConservedtoPrimitive_Vars(      sched, patches, ice_matls_sub,
                                                          all_matls,
                                                          "finalizeTimestep");
                                                          
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
                                                          
  scheduleTestConservation(               sched, patches, ice_matls_sub,
                                                          all_matls);
                                                          
  //_________________________________                                                        
  if(d_canAddICEMaterial){
    //  This checks to see if the model on THIS patch says that it's
    //  time to add a new material
    scheduleCheckNeedAddMaterial(           sched, level,   all_matls);

    //  This one checks to see if the model on ANY patch says that it's
    //  time to add a new material
    scheduleSetNeedAddMaterialFlag(         sched, level,   all_matls);
  }
  cout_doing << "---------------------------------------------------------"<<endl;
}

/* _____________________________________________________________________
 Function~  ICE::scheduleComputeThermoTransportProperties--
_____________________________________________________________________*/
void ICE::scheduleComputeThermoTransportProperties(SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* ice_matls)
{ 
  Task* t;
  cout_doing << d_myworld->myrank() << " ICE::schedulecomputeThermoTransportProperties" 
             << "\t\t\tL-"<< level->getIndex()<< endl;
             
  t = scinew Task("ICE::computeThermoTransportProperties", 
            this, &ICE::computeThermoTransportProperties); 
            
  //if(d_doAMR && level->getIndex() !=0){
  // dummy variable needed to keep the taskgraph in sync
  //t->requires(Task::NewDW,lb->AMR_SyncTaskgraphLabel,Ghost::None,0);
  //}           
  t->requires(Task::OldDW,lb->temp_CCLabel, ice_matls->getUnion(), Ghost::None, 0);  

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
  int levelIndex = getLevel(patches)->getIndex();
  Task* t = 0;
  
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeEquilibrationPressure" 
             << "\t\t\tL-" << levelIndex<< endl;

  if(d_sharedState->getNumMatls() == 1){    
    t = scinew Task("ICE::computeEquilPressure_1_matl",
              this, &ICE::computeEquilPressure_1_matl); 
  } else{
    t = scinew Task("ICE::computeEquilibrationPressure",
              this, &ICE::computeEquilibrationPressure);
  }      


  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, lb->delTLabel, getLevel(patches));  
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
  t->computes(lb->compressibilityLabel);
  t->computes(lb->sumKappaLabel,        press_matl, oims);
  t->computes(lb->press_equil_CCLabel,  press_matl, oims);
  t->computes(lb->sum_imp_delPLabel,    press_matl, oims);  //  initialized for implicit

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
  int levelIndex = getLevel(patches)->getIndex();
  if(d_models.size()>0){
    Task* t;
    cout_doing << d_myworld->myrank() << " ICE::scheduleComputeTempFC" 
               << "\t\t\t\t\tL-"<< levelIndex<< endl;
             
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
}
/* _____________________________________________________________________
 Function~  ICE::scheduleComputeVel_FC--
_____________________________________________________________________*/
void ICE::scheduleComputeVel_FC(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSubset* ice_matls,
                                const MaterialSubset* mpm_matls,
                                const MaterialSubset* press_matl,
                                const MaterialSet* all_matls)
{ 
  int levelIndex = getLevel(patches)->getIndex();
  Task* t = 0;

  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeVel_FC" 
             << "\t\t\t\t\tL-" << levelIndex<< endl;

  t = scinew Task("ICE::computeVel_FC",
            this, &ICE::computeVel_FC);

  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  t->requires(Task::OldDW, lb->delTLabel, getLevel(patches));  
  t->requires(Task::NewDW, lb->press_equil_CCLabel, press_matl, oims, gac,1);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,    /*all_matls*/ gac,1);
  t->requires(Task::NewDW,lb->rho_CCLabel,       /*all_matls*/ gac,1);
  t->requires(Task::OldDW,lb->vel_CCLabel,         ice_matls,  gac,1);
  t->requires(Task::NewDW,lb->vel_CCLabel,         mpm_matls,  gac,1);
  
  t->computes(lb->uvel_FCLabel);
  t->computes(lb->vvel_FCLabel);
  t->computes(lb->wvel_FCLabel);
  t->computes(lb->grad_P_XFCLabel);
  t->computes(lb->grad_P_YFCLabel);
  t->computes(lb->grad_P_ZFCLabel);
  sched->addTask(t, patches, all_matls);
}
/* _____________________________________________________________________
 Function~  ICE::scheduleAddExchangeContributionToFCVel--
_____________________________________________________________________*/
void ICE::scheduleAddExchangeContributionToFCVel(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* ice_matls,
                                           const MaterialSet* all_matls,
                                           bool recursion)
{
  int levelIndex = getLevel(patches)->getIndex();
  cout_doing << d_myworld->myrank() << " ICE::scheduleAddExchangeContributionToFCVel" 
             << "\t\t\tL-" << levelIndex<< endl;
  Task* task = scinew Task("ICE::addExchangeContributionToFCVel",
                     this, &ICE::addExchangeContributionToFCVel, recursion);

  if(recursion) {
    task->requires(Task::ParentOldDW, lb->delTLabel,getLevel(patches));
  } else {
    task->requires(Task::OldDW, lb->delTLabel,getLevel(patches));
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  
  //__________________________________
  // define parent data warehouse
  // change the definition of parent(old/new)DW
  // when using semi-implicit pressure solve
  Task::WhichDW pNewDW = Task::NewDW;
  if(recursion) {
    pNewDW  = Task::ParentNewDW;
  }  
  
  task->requires(pNewDW,     lb->sp_vol_CCLabel,    /*all_matls*/gac,1);
  task->requires(pNewDW,     lb->vol_frac_CCLabel,  /*all_matls*/gac,1);
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
  int levelIndex = level->getIndex();
  if(d_models.size() != 0){
    cout_doing << d_myworld->myrank() << " ICE::scheduleComputeModelSources" 
               << "\t\t\tL-"<< levelIndex<< endl;
    
    
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
void ICE::scheduleUpdateVolumeFraction(SchedulerP& sched, 
                                       const LevelP& level,
                                       const MaterialSubset* press_matl,
                                       const MaterialSet* matls)
{
  int levelIndex =level->getIndex();
  if(d_models.size() != 0){
    cout_doing << d_myworld->myrank() << " ICE::scheduleUpdateVolumeFraction" 
               << "\t\t\tL-"<< levelIndex<< endl;
               
    Task* task = scinew Task("ICE::updateVolumeFraction",
                       this, &ICE::updateVolumeFraction);
    Ghost::GhostType  gn = Ghost::None;  
    task->requires( Task::NewDW, lb->sp_vol_CCLabel,     gn);
    task->requires( Task::NewDW, lb->rho_CCLabel,        gn);    
    task->requires( Task::NewDW, lb->modelVol_srcLabel,  gn);
    task->requires( Task::NewDW, lb->compressibilityLabel,gn);
    task->modifies(lb->sumKappaLabel, press_matl);  
    task->modifies(lb->vol_frac_CCLabel);
    task->modifies(lb->f_theta_CCLabel);
    

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
  int levelIndex = getLevel(patches)->getIndex();
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeDelPressAndUpdatePressCC" 
             << "\t\t\tL-"<< levelIndex<< endl;
  Task *task = scinew Task("ICE::computeDelPressAndUpdatePressCC",
                            this, &ICE::computeDelPressAndUpdatePressCC);
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;  
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  task->requires( Task::OldDW, lb->delTLabel,getLevel(patches));   
  task->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,2);
  task->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);
  task->requires( Task::NewDW, lb->sp_vol_CCLabel,     gn);
  task->requires( Task::NewDW, lb->rho_CCLabel,        gn);    
  task->requires( Task::NewDW, lb->speedSound_CCLabel, gn);
  task->requires( Task::NewDW, lb->sumKappaLabel,      press_matl,oims,gn);
  task->requires( Task::NewDW, lb->press_equil_CCLabel,press_matl,oims,gn);
  //__________________________________
  if(d_models.size() > 0){
    task->requires(Task::NewDW, lb->modelMass_srcLabel, gn);
  }
  
  computesRequires_CustomBCs(task, "update_press_CC", lb, ice_matls,
                             d_customBC_var_basket);
  
  task->computes(lb->press_CCLabel,        press_matl, oims);
  task->computes(lb->delP_DilatateLabel,   press_matl, oims);
  task->computes(lb->delP_MassXLabel,      press_matl, oims);
  task->computes(lb->term2Label,           press_matl, oims);
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
  int levelIndex = getLevel(patches)->getIndex();
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputePressFC" 
             << "\t\t\t\t\tL-"<< levelIndex<< endl;
                                
  Task* task = scinew Task("ICE::computePressFC",
                     this, &ICE::computePressFC);
                     
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
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
  int levelIndex = getLevel(patches)->getIndex();
  Task* t;
  cout_doing << d_myworld->myrank() << " ICE::scheduleAccumulateMomentumSourceSinks" 
             << "\t\t\tL-"<< levelIndex<< endl;
              
  t = scinew Task("ICE::accumulateMomentumSourceSinks", 
            this, &ICE::accumulateMomentumSourceSinks);

                       // EQ  & RATE FORM     
  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
  t->requires(Task::NewDW,lb->pressX_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->pressY_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->pressZ_FCLabel,   press_matl,    oims, gac, 1);
  t->requires(Task::NewDW,lb->viscosityLabel,   ice_matls, gac, 2);
  t->requires(Task::OldDW,lb->vel_CCLabel,      ice_matls, gac, 2);
  t->requires(Task::NewDW,lb->sp_vol_CCLabel,   ice_matls, gac, 2);
  t->requires(Task::NewDW,lb->rho_CCLabel,       gac,2);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel, gac,2);

  if(d_turbulence){
    t->requires(Task::NewDW,lb->uvel_FCMELabel,   ice_matls, gac, 3);
    t->requires(Task::NewDW,lb->vvel_FCMELabel,   ice_matls, gac, 3);
    t->requires(Task::NewDW,lb->wvel_FCMELabel,   ice_matls, gac, 3);
    t->computes(lb->turb_viscosity_CCLabel,   ice_matls);
  } 

  t->computes(lb->mom_source_CCLabel);
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
  int levelIndex = getLevel(patches)->getIndex();
  Task* t;              // EQ
  cout_doing << d_myworld->myrank() << " ICE::scheduleAccumulateEnergySourceSinks" 
             << "\t\t\tL-" << levelIndex << endl;

  t = scinew Task("ICE::accumulateEnergySourceSinks",
            this, &ICE::accumulateEnergySourceSinks);
                     

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));  
  t->requires(Task::NewDW, lb->press_CCLabel,     press_matl,oims, gn);
  t->requires(Task::NewDW, lb->delP_DilatateLabel,press_matl,oims, gn);
  t->requires(Task::NewDW, lb->compressibilityLabel,               gn);
  t->requires(Task::OldDW, lb->temp_CCLabel,      ice_matls, gac,1);
  t->requires(Task::NewDW, lb->thermalCondLabel,  ice_matls, gac,1);
  t->requires(Task::NewDW, lb->rho_CCLabel,                  gac,1);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,               gac,1);
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,             gac,1);

  if(d_with_mpm){
   t->requires(Task::NewDW,lb->TMV_CCLabel,       press_matl,oims, gn);
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
  int levelIndex = getLevel(patches)->getIndex();
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeLagrangianValues" 
             << "\t\t\t\tL-"<< levelIndex<< endl;
             
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
  int levelIndex = getLevel(patches)->getIndex();
  Task* t = 0;
  cout_doing << d_myworld->myrank() << " ICE::scheduleComputeLagrangianSpecificVolume" 
             << "\t\t\tL-"<< levelIndex<< endl;
  t = scinew Task("ICE::computeLagrangianSpecificVolume",
             this,&ICE::computeLagrangianSpecificVolume);

  Ghost::GhostType  gn  = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches));  
  t->requires(Task::NewDW, lb->rho_CCLabel,               gn);
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,            gn);    
  t->requires(Task::NewDW, lb->Tdot_CCLabel,              gn);  
  t->requires(Task::NewDW, lb->f_theta_CCLabel,           gn);
  t->requires(Task::NewDW, lb->compressibilityLabel,      gn);  
  t->requires(Task::NewDW, lb->vol_frac_CCLabel,          gac,1);
  
  t->requires(Task::OldDW, lb->temp_CCLabel,        ice_matls, gn);
  t->requires(Task::NewDW, lb->specific_heatLabel,  ice_matls, gn);
  t->requires(Task::NewDW, lb->temp_CCLabel,        mpm_matls, gn); 

  t->requires(Task::NewDW, lb->delP_DilatateLabel,  press_matl,oims,gn);
  t->requires(Task::NewDW, lb->press_CCLabel,       press_matl,oims,gn);
  if(d_with_mpm){
   t->requires(Task::NewDW,lb->TMV_CCLabel,       press_matl,oims, gn);
  }
    
  if(d_models.size() > 0){
    t->requires(Task::NewDW, lb->modelVol_srcLabel,    gn);
  }

  t->computes(lb->sp_vol_L_CCLabel);                             
  t->computes(lb->sp_vol_src_CCLabel);                        

  sched->setRestartable(true);
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
  int levelIndex = getLevel(patches)->getIndex();
  if(d_models.size() > 0 && d_modelSetup->tvars.size() > 0){
    cout_doing << d_myworld->myrank() << " ICE::scheduleComputeLagrangian_Transported_Vars" 
               << "\t\t\tL-"<<levelIndex<< endl;
               
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
  int levelIndex = getLevel(patches)->getIndex();
  Task* t = 0;

  cout_doing << d_myworld->myrank() << " ICE::scheduleAddExchangeToMomentumAndEnergy" 
             << "\t\t\tL-"<< levelIndex << endl;
  t=scinew Task("ICE::addExchangeToMomentumAndEnergy",
                this, &ICE::addExchangeToMomentumAndEnergy);

  Ghost::GhostType  gn  = Ghost::None;
//  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  t->requires(Task::OldDW, lb->delTLabel,getLevel(patches)); 
 
  if(d_exchCoeff->convective()){
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
                                     const MaterialSet* ice_matls)
{ 
  if(d_customBC_var_basket->usingLodi) {
    cout_doing << d_myworld->myrank() << " ICE::scheduleMaxMach_on_Lodi_BC_Faces" 
               << "\t\t\tL-levelIndex" << endl;
    Task* task = scinew Task("ICE::maxMach_on_Lodi_BC_Faces",
                       this, &ICE::maxMach_on_Lodi_BC_Faces);
    Ghost::GhostType  gn = Ghost::None;  
    task->requires( Task::OldDW, lb->vel_CCLabel,        gn);   
    task->requires( Task::OldDW, lb->speedSound_CCLabel, gn);
                             
    //__________________________________
    // loop over the Lodi face
    //  add computes for maxMach
    vector<Patch::FaceType>::iterator f ;
         
    for( f = d_customBC_var_basket->Lodi_var_basket->LodiFaces.begin();
         f!= d_customBC_var_basket->Lodi_var_basket->LodiFaces.end(); ++f) {
         
      VarLabel* V_Label = getMaxMach_face_VarLabel(*f);
      task->computes(V_Label, ice_matls->getUnion());
    }
    sched->addTask(task, level->eachPatch(), ice_matls);
  }
}
/* _____________________________________________________________________
 Function~  ICE::computesRequires_AMR_Refluxing--
_____________________________________________________________________*/
void ICE::computesRequires_AMR_Refluxing(Task* task, 
                                    const MaterialSet* ice_matls)
{
  cout_doing << d_myworld->myrank() << "      computesRequires_AMR_Refluxing\n";
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
  
  // DON'T require reflux vars from the OldDW.  Since we only require it 
  // between subcycles, we don't want to schedule it.  Otherwise it will
  // just cause excess TG work.  The data will all get to the right place.
  //__________________________________
  // MODELS
  vector<AMR_refluxVariable*>::iterator iter;
  for( iter  = d_modelSetup->d_reflux_vars.begin();
       iter != d_modelSetup->d_reflux_vars.end(); iter++){
       AMR_refluxVariable* rvar = *iter;
       
    task->computes(rvar->var_X_FC_flux); 
    task->computes(rvar->var_Y_FC_flux);
    task->computes(rvar->var_Z_FC_flux);
  } 
}

/* _____________________________________________________________________
 Function~  ICE::scheduleAdvectAndAdvanceInTime--
_____________________________________________________________________*/
void ICE::scheduleAdvectAndAdvanceInTime(SchedulerP& sched,
                                    const PatchSet* patch_set,
                                    const MaterialSubset* ice_matlsub,
                                    const MaterialSet* ice_matls)
{
  int levelIndex = getLevel(patch_set)->getIndex();
  cout_doing << d_myworld->myrank() << " ICE::scheduleAdvectAndAdvanceInTime" 
             << "\t\t\t\tL-"<< levelIndex << endl;
             
  Task* task = scinew Task("ICE::advectAndAdvanceInTime",
                           this, &ICE::advectAndAdvanceInTime);
  task->requires(Task::OldDW, lb->delTLabel,getLevel(patch_set));
  Ghost::GhostType  gac  = Ghost::AroundCells;
  task->requires(Task::NewDW, lb->uvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->vvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->wvel_FCMELabel,      gac,2);
  task->requires(Task::NewDW, lb->mom_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->mass_L_CCLabel,      gac,2);
  task->requires(Task::NewDW, lb->eng_L_ME_CCLabel,    gac,2);
  task->requires(Task::NewDW, lb->sp_vol_L_CCLabel,    gac,2);
                             
  if(d_doRefluxing){            
    computesRequires_AMR_Refluxing(task, ice_matls);
  }
  
  task->computes(lb->mass_advLabel);
  task->computes(lb->mom_advLabel);
  task->computes(lb->eng_advLabel);
  task->computes(lb->sp_vol_advLabel);     
  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;
    
    for(iter = d_modelSetup->tvars.begin();
        iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var_Lagrangian, tvar->matls, gac, 2);
      task->computes(tvar->var_adv,   tvar->matls);
    }
  }
  sched->setRestartable(true);
  sched->addTask(task, patch_set, ice_matls);
}
/* _____________________________________________________________________
 Function~  ICE::scheduleConservedtoPrimitive_Vars--
_____________________________________________________________________*/
void ICE::scheduleConservedtoPrimitive_Vars(SchedulerP& sched,
                                    const PatchSet* patch_set,
                                    const MaterialSubset* ice_matlsub,
                                    const MaterialSet* ice_matls,
                                    const string& where)
{
  ASSERT( where == "afterAdvection" || where == "finalizeTimestep");
  
  int levelIndex = getLevel(patch_set)->getIndex();
  int numLevels = getLevel(patch_set)->getGrid()->numLevels();
  // single level problems we only need to perform this task once
  // immediately after advecton
  if(numLevels == 1 && where == "finalizeTimestep")  
    return;
    
  // On the finest level we only need to perform this task once
  // immediately after advecton
  if(levelIndex + 1 == numLevels && where ==  "finalizeTimestep")
    return;

  // from another taskgraph
  bool fat = false;
  if (where == "finalizeTimestep")
    fat = true;

  //---------------------------  
  cout_doing << d_myworld->myrank() << " ICE::scheduleConservedtoPrimitive_Vars" 
             << "\t\t\tL-"<< levelIndex << endl;

  string name = "ICE::conservedtoPrimitive_Vars:" + where;

  Task* task = scinew Task(name, this, &ICE::conservedtoPrimitive_Vars);
  task->requires(Task::OldDW, lb->delTLabel,getLevel(patch_set));     
  Ghost::GhostType  gn   = Ghost::None;
  task->requires(Task::NewDW, lb->mass_advLabel,      gn,0);
  task->requires(Task::NewDW, lb->mom_advLabel,       gn,0);
  task->requires(Task::NewDW, lb->eng_advLabel,       gn,0);
  task->requires(Task::NewDW, lb->sp_vol_advLabel,    gn,0);
  
  task->requires(Task::NewDW, lb->specific_heatLabel, gn, 0, fat);
  task->requires(Task::NewDW, lb->speedSound_CCLabel, gn, 0, fat);
  task->requires(Task::NewDW, lb->vol_frac_CCLabel,   gn, 0, fat);
  task->requires(Task::NewDW, lb->gammaLabel,         gn, 0, fat);
    
  computesRequires_CustomBCs(task, "Advection", lb, ice_matlsub, 
                             d_customBC_var_basket);
                             
  task->modifies(lb->rho_CCLabel,     fat);
  task->modifies(lb->sp_vol_CCLabel,  fat);               
  if( where == "afterAdvection"){
    task->computes(lb->temp_CCLabel);
    task->computes(lb->vel_CCLabel);
    task->computes(lb->machLabel);
  }
  if( where == "finalizeTimestep"){
    task->modifies(lb->temp_CCLabel,  fat);
    task->modifies(lb->vel_CCLabel,   fat);
    task->modifies(lb->machLabel,     fat);
  } 
  
  //__________________________________
  // Model Variables.
  if(d_modelSetup && d_modelSetup->tvars.size() > 0){
    vector<TransportedVariable*>::iterator iter;
    
    for(iter = d_modelSetup->tvars.begin();
        iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      task->requires(Task::NewDW, tvar->var_adv, tvar->matls, gn,0);
      
      if( where == "afterAdvection"){
        task->computes(tvar->var,   tvar->matls);
      }
      if( where == "finalizeTimestep"){
        task->modifies(tvar->var,   tvar->matls, fat);
      }
      
    }
  }
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
  int levelIndex = getLevel(patches)->getIndex();
  if(d_conservationTest->onOff && levelIndex == 0) {
    cout_doing << d_myworld->myrank() << " ICE::scheduleTestConservation" 
               << "\t\t\t\t\tL-"<< levelIndex<< endl;
    
    Task* t= scinew Task("ICE::TestConservation",
                   this, &ICE::TestConservation);

    Ghost::GhostType  gn  = Ghost::None;
    t->requires(Task::OldDW, lb->delTLabel,getLevel(patches)); 
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
    
    if(d_conservationTest->exchange){
      t->computes(lb->mom_exch_errorLabel);
      t->computes(lb->eng_exch_errorLabel);
    }
    if(d_conservationTest->mass){
      t->computes(lb->TotalMassLabel);
    }
    if(d_conservationTest->energy){
    t->computes(lb->KineticEnergyLabel);
    t->computes(lb->TotalIntEngLabel);
    }
    if(d_conservationTest->momentum){
      t->computes(lb->TotalMomentumLabel);
    }
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
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing Compute Stable Timestep on patch " << patch->getID() 
         << "\t\t ICE \tL-" <<level->getIndex()<< endl;
      
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    double delt_CFL;
    double delt_diff;
    double delt;
    double inv_sum_invDelx_sqr = 1.0/( 1.0/(delX * delX) 
                                     + 1.0/(delY * delY) 
                                     + 1.0/(delZ * delZ) );
    constCCVariable<double> speedSound, sp_vol_CC, thermalCond, viscosity;
    constCCVariable<double> cv, gamma;
    constCCVariable<Vector> vel_CC;
    Ghost::GhostType  gn  = Ghost::None; 
    Ghost::GhostType  gac = Ghost::AroundCells;

    IntVector badCell(0,0,0);
    delt_CFL  = 1000.0; 
    delt_diff = 1000;
    delt      = 1000;

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
          double A = d_CFL*delX/(speed_Sound + 
                                       fabs(vel_CC[c].x())+d_SMALL_NUM);
          double B = d_CFL*delY/(speed_Sound + 
                                       fabs(vel_CC[c].y())+d_SMALL_NUM);
          double C = d_CFL*delZ/(speed_Sound + 
                                       fabs(vel_CC[c].z())+d_SMALL_NUM);
          delt_CFL = std::min(A, delt_CFL);
          delt_CFL = std::min(B, delt_CFL);
          delt_CFL = std::min(C, delt_CFL);
          if (A < 1e-20 || B < 1e-20 || C < 1e-20) {
            if (badCell == IntVector(0,0,0)) {
              badCell = c;
            }
            cout << d_myworld->myrank() << " Bad cell " << c << " (" << patch->getID() << "-" << level->getIndex() << "): " << vel_CC[c]<< endl;
          }
        }
//      cout << " Aggressive delT Based on currant number "<< delt_CFL << endl;
        //__________________________________
        // stability constraint due to diffusion
        //  I C E  O N L Y
        double thermalCond_test = ice_matl->getThermalConductivity();
        double viscosity_test   = ice_matl->getViscosity();
        if (thermalCond_test !=0 || viscosity_test !=0) {

          for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            double cp = cv[c] * gamma[c];
            double inv_thermalDiffusivity = cp/(sp_vol_CC[c] * thermalCond[c]);
            double kinematicViscosity = viscosity[c] * sp_vol_CC[c];
            double inv_diffusionCoeff = min(inv_thermalDiffusivity, 1.0/kinematicViscosity);
            double A = d_CFL * 0.5 * inv_sum_invDelx_sqr * inv_diffusionCoeff;
            delt_diff = std::min(A, delt_diff);
            if (delt_diff < 1e-20 && badCell == IntVector(0,0,0)) {
              badCell = c;
            }
          }
        }  //
//      cout << "delT based on diffusion  "<< delt_diff<<endl;
        delt = std::min(delt_CFL, delt_diff);
      } // aggressive Timestep 


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
        Vector grav = getGravity();
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
            double speedSound = std::max(c_L,c_R );      

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
          delt = std::min(delt, delt_tmp);

          if (delt < 1e-20 && badCell == IntVector(0,0,0)) {
            badCell = c;
          }
        }  // iter loop
//      cout << " Conservative delT based on swept volumes "<< delt<<endl;
      }  
    }  // matl loop   

    const Level* level = getLevel(patches);
    //__________________________________
    //  Bullet proofing
    if(delt < 1e-20) { 
      ostringstream warn;
      warn << "ERROR ICE:(L-"<< level->getIndex()
           << "):ComputeStableTimestep: delT < 1e-20 on cell " << badCell;
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
    new_dw->put(delt_vartype(delt), lb->delTLabel, level);
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
          patch->getExtraCellLowIndex()  << 
          patch->getExtraCellHighIndex() << endl;
  }

  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();
  
  
  //__________________________________
  // find max index of all the ICE matls
  // you could have a 1 matl problem with a starting indx of 2
  int max_indx = -100;
  
  int numICEMatls = d_sharedState->getNumICEMatls();
  for (int m = 0; m < numICEMatls; m++ ){
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
    int indx= ice_matl->getDWIndex();
    max_indx = max(max_indx, indx);
  }
  d_max_iceMatl_indx = max_indx;
  max_indx +=1;   

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing Initialize on patch " << patch->getID() 
         << "\t\t\t\t ICE \tL-" <<L_indx<< endl;
    int numMatls    = d_sharedState->getNumICEMatls();
    int numALLMatls = d_sharedState->getNumMatls();
    Vector grav     = getGravity();
    StaticArray<constCCVariable<double> > placeHolder(0);
    StaticArray<CCVariable<double>   > rho_micro(max_indx);
    StaticArray<CCVariable<double>   > sp_vol_CC(max_indx);
    StaticArray<CCVariable<double>   > rho_CC(max_indx); 
    StaticArray<CCVariable<double>   > Temp_CC(max_indx);
    StaticArray<CCVariable<double>   > speedSound(max_indx);
    StaticArray<CCVariable<double>   > vol_frac_CC(max_indx);
    StaticArray<CCVariable<Vector>   > vel_CC(max_indx);
    StaticArray<CCVariable<double>   > cv(max_indx);
    StaticArray<CCVariable<double>   > gamma(max_indx);
    CCVariable<double>    press_CC, imp_initialGuess, vol_frac_sum;
    
    new_dw->allocateAndPut(press_CC,         lb->press_CCLabel,     0,patch);
    new_dw->allocateAndPut(imp_initialGuess, lb->initialGuessLabel, 0,patch);
    new_dw->allocateTemporary(vol_frac_sum,patch);
    imp_initialGuess.initialize(0.0); 
    vol_frac_sum.initialize(0.0);

    //__________________________________
    //  Thermo and transport properties
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();
      CCVariable<double> viscosity, thermalCond;
      new_dw->allocateAndPut(viscosity,    lb->viscosityLabel,    indx,patch);
      new_dw->allocateAndPut(thermalCond,  lb->thermalCondLabel,  indx,patch);
      new_dw->allocateAndPut(cv[indx],     lb->specific_heatLabel,indx,patch);
      new_dw->allocateAndPut(gamma[indx],  lb->gammaLabel,        indx,patch);

      gamma[indx].initialize( ice_matl->getGamma());
      cv[indx].initialize(    ice_matl->getSpecificHeat());    
      viscosity.initialize  ( ice_matl->getViscosity());
      thermalCond.initialize( ice_matl->getThermalConductivity());
       
    }
    // --------bulletproofing
    if (grav.length() >0.0 && d_surroundingMatl_indx == -9)  {
      throw ProblemSetupException("ERROR ICE::actuallyInitialize \n"
            "You must have \n" 
            "       <isSurroundingMatl> true </isSurroundingMatl> \n "
            "specified inside the ICE material that is the background matl\n",
                                  __FILE__, __LINE__);
    }
  //__________________________________
  // Note:
  // The press_CC isn't material dependent even though
  // we loop over numMatls below. This is done so we don't need additional
  // machinery to grab the pressure inside a geom_object
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx= ice_matl->getDWIndex();
      new_dw->allocateAndPut(rho_micro[indx],  lb->rho_micro_CCLabel, indx,patch); 
      new_dw->allocateAndPut(sp_vol_CC[indx],  lb->sp_vol_CCLabel,    indx,patch); 
      new_dw->allocateAndPut(rho_CC[indx],     lb->rho_CCLabel,       indx,patch); 
      new_dw->allocateAndPut(Temp_CC[indx],    lb->temp_CCLabel,      indx,patch); 
      new_dw->allocateAndPut(speedSound[indx], lb->speedSound_CCLabel,indx,patch); 
      new_dw->allocateAndPut(vol_frac_CC[indx],lb->vol_frac_CCLabel,  indx,patch);
      new_dw->allocateAndPut(vel_CC[indx],     lb->vel_CCLabel,       indx,patch);
    }

    
    double p_ref = getRefPress();
    press_CC.initialize(p_ref);
    for (int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      ice_matl->initializeCells(rho_micro[indx],  rho_CC[indx],
                                Temp_CC[indx],    speedSound[indx], 
                                vol_frac_CC[indx], vel_CC[indx], 
                                press_CC, numALLMatls, patch, new_dw);
      
      // if specified, overide the initialization             
      customInitialization( patch,rho_CC[indx], Temp_CC[indx],vel_CC[indx], press_CC,
                            ice_matl, d_customInitialize_basket);
                                                    
      setBC(rho_CC[indx],     "Density",     patch, d_sharedState, indx, new_dw);
      setBC(rho_micro[indx],  "Density",     patch, d_sharedState, indx, new_dw);
      setBC(Temp_CC[indx],    "Temperature", patch, d_sharedState, indx, new_dw);
      setBC(speedSound[indx], "zeroNeumann", patch, d_sharedState, indx, new_dw); 
      setBC(vel_CC[indx],     "Velocity",    patch, d_sharedState, indx, new_dw); 
      setBC(press_CC, rho_micro, placeHolder, d_surroundingMatl_indx, 
            "rho_micro","Pressure", patch, d_sharedState, 0, new_dw);
            
      SpecificHeat *cvModel = ice_matl->getSpecificHeatModel();
      if(cvModel != 0) {
        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          gamma[indx][c] = cvModel->getGamma(Temp_CC[indx][c]); 
          cv[indx][c]    = cvModel->getSpecificHeat(Temp_CC[indx][c]); 
        }
      } 
            
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_CC[indx][c] = 1.0/rho_micro[indx][c];

        //needed for LODI BCs
        vol_frac_CC[indx][c] = rho_CC[indx][c]*sp_vol_CC[indx][c];
        vol_frac_sum[c] += vol_frac_CC[indx][c];

        double dp_drho, dp_de, c_2, press_tmp;
        ice_matl->getEOS()->computePressEOS(rho_micro[indx][c],gamma[indx][c],
                                          cv[indx][c], Temp_CC[indx][c], press_tmp,
                                          dp_drho, dp_de);

        if( !d_customInitialize_basket->doesComputePressure){
          press_CC[c] = press_tmp;
        }
          
        c_2 = dp_drho + dp_de * press_CC[c]/(rho_micro[indx][c] * rho_micro[indx][c]);
        speedSound[indx][c] = sqrt(c_2);
      }
      //____ B U L L E T   P R O O F I N G----
      IntVector neg_cell;
      ostringstream warn, base;
      base <<"ERROR ICE:(L-"<<L_indx<<"):actuallyInitialize, mat "<< indx <<" cell ";
      
      if( !areAllValuesPositive(press_CC, neg_cell) ) {
        warn << base.str()<< neg_cell << " press_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(rho_CC[indx], neg_cell) ) {
        warn << base.str()<< neg_cell << " rho_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(Temp_CC[indx], neg_cell) ) {
        warn << base.str()<< neg_cell << " Temp_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
      if( !areAllValuesPositive(sp_vol_CC[indx], neg_cell) ) {
        warn << base.str()<< neg_cell << " sp_vol_CC is negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
      }
    }   // numMatls
    
    // make sure volume fractions sum to 1
    if(!d_with_mpm)
    {
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        if(vol_frac_sum[*iter] > 1.0 + 1e-10 || vol_frac_sum[*iter] < 1.0 - 1e-10)
        {
          ostringstream warn, base;
          base <<"ERROR ICE:(L-"<<L_indx<<"):actuallyInitialize";
          warn << base.str() << "Cell: " << *iter << " Volume fractions did not sum to 1. Sum=" << vol_frac_sum[*iter] << "\n";
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
        }
      }
    }
      

    if (switchDebug_Initialize){     
      ostringstream desc1;
      desc1 << "Initialization_patch_"<< patch->getID();
      printData(0, patch, 1, desc1.str(), "press_CC", press_CC);         
      for (int m = 0; m < numMatls; m++ ) { 
        ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
        int indx = ice_matl->getDWIndex();
        ostringstream desc;      
        desc << "Initialization_Mat_" << indx << "_patch_"<< patch->getID();
        printData(indx, patch,   1, desc.str(), "rho_CC",      rho_CC[indx]);
        printData(indx, patch,   1, desc.str(), "rho_micro_CC",rho_micro[indx]);
        printData(indx, patch,   1, desc.str(), "sp_vol_CC",   sp_vol_CC[indx]);
        printData(indx, patch,   1, desc.str(), "Temp_CC",     Temp_CC[indx]);
        printData(indx, patch,   1, desc.str(), "vol_frac_CC", vol_frac_CC[indx]);
        printVector(indx, patch, 1, desc.str(), "vel_CC", 0,   vel_CC[indx]);;
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
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing initialize_hydrostaticAdj on patch "
               << patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;
   
    Ghost::GhostType  gn = Ghost::None;
    int numMatls = d_sharedState->getNumICEMatls();
    //__________________________________
    // adjust the pressure field
    CCVariable<double> rho_micro, press_CC;
    new_dw->getModifiable(press_CC, lb->press_CCLabel,0, patch);
    new_dw->getModifiable(rho_micro,lb->rho_micro_CCLabel,
                                            d_surroundingMatl_indx, patch);
    
    hydrostaticPressureAdjustment(patch, rho_micro, press_CC);
    
    //__________________________________
    //  Adjust Temp field if g != 0
    //  so fields are thermodynamically consistent
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      constCCVariable<double> gamma, cv;
      CCVariable<double> Temp;   
 
      new_dw->get(gamma, lb->gammaLabel,         indx, patch,gn,0);   
      new_dw->get(cv,    lb->specific_heatLabel, indx, patch,gn,0);
      new_dw->getModifiable(Temp,     lb->temp_CCLabel,       indx, patch);  
      new_dw->getModifiable(rho_micro,lb->rho_micro_CCLabel,  indx, patch); 
 
      Patch::FaceType dummy = Patch::invalidFace; // This is a dummy variable
      ice_matl->getEOS()->computeTempCC( patch, "WholeDomain",
                                         press_CC, gamma, cv,
                                         rho_micro, Temp, dummy );

      //__________________________________
      //  Print Data
      if (switchDebug_Initialize){     
        ostringstream desc, desc1;
        desc << "hydroStaticAdj_patch_"<< patch->getID();
        printData(0, patch, 1, desc.str(), "press_CC", press_CC);         
        for (int m = 0; m < numMatls; m++ ) { 
          ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
          int indx = ice_matl->getDWIndex();      
          desc1 << "hydroStaticAdj_Mat_" << indx << "_patch_"<< patch->getID();
          printData(indx, patch,   1, desc.str(), "rho_micro_CC",rho_micro);
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
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{ 

  const Level* level = getLevel(patches);
  int levelIndex = level->getIndex();
 
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << " ---------------------------------------------- L-"<< levelIndex<< endl;
    cout_doing << d_myworld->myrank() << " Doing computeThermoTransportProperties on patch "
               << patch->getID() << "\t ICE \tL-" <<levelIndex<< endl;
   
    int numMatls = d_sharedState->getNumICEMatls();
    
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();

      constCCVariable<double> temp_CC;
      old_dw->get(temp_CC, lb->temp_CCLabel, indx,patch,Ghost::None,0);

      CCVariable<double> viscosity, thermalCond, gamma, cv;
      
      new_dw->allocateAndPut(thermalCond, lb->thermalCondLabel,  indx, patch);  
      new_dw->allocateAndPut(viscosity,   lb->viscosityLabel,    indx, patch);
      new_dw->allocateAndPut(cv,          lb->specific_heatLabel,indx, patch);
      new_dw->allocateAndPut(gamma,       lb->gammaLabel,        indx, patch); 
      viscosity.initialize  ( ice_matl->getViscosity());
      thermalCond.initialize( ice_matl->getThermalConductivity());
      gamma.initialize  (     ice_matl->getGamma());
      cv.initialize(          ice_matl->getSpecificHeat());
      SpecificHeat *cvModel = ice_matl->getSpecificHeatModel();
      if(cvModel != 0) {
        // loop through cells and compute pointwise
        for(CellIterator iter = patch->getCellIterator();!iter.done();iter++) {
          IntVector c = *iter;
          cv[c] = cvModel->getSpecificHeat(temp_CC[c]);
          gamma[c] = cvModel->getGamma(temp_CC[c]);
        }
      }
    }

    

    //__________________________________
    // Is it time to dump printData ?
    // You need to do this in the first task
    // and only on the first patch
    if (levelIndex == 0 && patch->getGridIndex() == 0) {
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
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing calc_equilibration_pressure on patch "<<patch->getID()
               << "\t\t ICE \tL-" <<L_indx<< endl;
    double    converg_coeff = 15;              
    double    convergence_crit = converg_coeff * DBL_EPSILON;
    double    sum=0., tmp;

    int       numMatls = d_sharedState->getNumICEMatls();
    static int n_passes;                  
    n_passes ++; 

    StaticArray<double> press_eos(numMatls);
    StaticArray<double> dp_drho(numMatls),dp_de(numMatls);
    StaticArray<CCVariable<double> > vol_frac(numMatls);
    StaticArray<CCVariable<double> > rho_micro(numMatls);
    StaticArray<CCVariable<double> > rho_CC_new(numMatls);
    StaticArray<CCVariable<double> > sp_vol_new(numMatls); 
    StaticArray<CCVariable<double> > speedSound(numMatls);
    StaticArray<CCVariable<double> > speedSound_new(numMatls);
    StaticArray<CCVariable<double> > f_theta(numMatls); 
    StaticArray<CCVariable<double> > kappa(numMatls);
    StaticArray<constCCVariable<double> > Temp(numMatls);
    StaticArray<constCCVariable<double> > rho_CC(numMatls);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    StaticArray<constCCVariable<double> > cv(numMatls);
    StaticArray<constCCVariable<double> > gamma(numMatls); 
    StaticArray<constCCVariable<double> > placeHolder(0);   

    CCVariable<int> n_iters_equil_press;
    constCCVariable<double> press;
    CCVariable<double> press_new, sumKappa, sum_imp_delP;
    Ghost::GhostType  gn = Ghost::None;
    
    //__________________________________ 
    old_dw->get(press,                   lb->press_CCLabel,       0,patch,gn,0);
    new_dw->allocateAndPut(press_new,    lb->press_equil_CCLabel, 0,patch);
    new_dw->allocateAndPut(sumKappa,     lb->sumKappaLabel,       0,patch);  
    new_dw->allocateAndPut(sum_imp_delP, lb->sum_imp_delPLabel,   0,patch);
       
    sum_imp_delP.initialize(0.0); //-- initialize for implicit pressure
       
    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(Temp[m],      lb->temp_CCLabel,      indx,patch, gn,0);
      old_dw->get(rho_CC[m],    lb->rho_CCLabel,       indx,patch, gn,0);
      old_dw->get(sp_vol_CC[m], lb->sp_vol_CCLabel,    indx,patch, gn,0);
      new_dw->get(cv[m],        lb->specific_heatLabel,indx,patch, gn,0);
      new_dw->get(gamma[m],     lb->gammaLabel,        indx,patch, gn,0);
            
      new_dw->allocateTemporary(rho_micro[m],  patch);
      new_dw->allocateAndPut(vol_frac[m],  lb->vol_frac_CCLabel,   indx,patch);
      new_dw->allocateAndPut(rho_CC_new[m],lb->rho_CCLabel,        indx,patch);
      new_dw->allocateAndPut(sp_vol_new[m],lb->sp_vol_CCLabel,     indx,patch);
      new_dw->allocateAndPut(f_theta[m],   lb->f_theta_CCLabel,    indx,patch);
      new_dw->allocateAndPut(kappa[m],     lb->compressibilityLabel,indx,patch);
      new_dw->allocateAndPut(speedSound_new[m], lb->speedSound_CCLabel,
                                                                   indx,patch);
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
    if (switchDebug_equil_press) {
    
      new_dw->allocateTemporary(n_iters_equil_press,  patch);
      ostringstream desc1;
      desc1 << "TOP_equilibration_patch_" << patch->getID();
      printData( 0, patch, 1, desc1.str(), "Press_CC_top", press);
     for (int m = 0; m < numMatls; m++)  {
       ICEMaterial* matl = d_sharedState->getICEMaterial( m );
       int indx = matl->getDWIndex(); 
       ostringstream desc;
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
      vector<EqPress_dbg> dbgEqPress;
    
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
        
        // Save iteration data for output in case of crash
        if(ds_EqPress.active()){
          EqPress_dbg dbg;
          dbg.delPress     = delPress;
          dbg.press_new    = press_new[c];
          dbg.sumVolFrac   = sum;
          dbg.count        = count;

          for (int m = 0; m < numMatls; m++) {
            EqPress_dbgMatl dmatl;
            dmatl.press_eos   = press_eos[m];
            dmatl.volFrac     = vol_frac[m][c];
            dmatl.rhoMicro    = rho_micro[m][c];
            dmatl.rho_CC      = rho_CC[m][c];
            dmatl.temp_CC     = Temp[m][c];
            dmatl.mat         = m;
            dbg.matl.push_back(dmatl);
          }
          dbgEqPress.push_back(dbg);
        }
      }   // end of converged

      test_max_iter = std::max(test_max_iter, count);

      //__________________________________
      //      BULLET PROOFING
      // ignore BP if a timestep restart has already been requested
      bool tsr = new_dw->timestepRestarted();
      
      string message;
      bool allTestsPassed = true;
      if(test_max_iter == d_max_iter_equilibration && !tsr){
        allTestsPassed = false;
        message += "Max. iterations reached ";
      }
      
      for (int m = 0; m < numMatls; m++) {
        if(( vol_frac[m][c] > 0.0 ) ||( vol_frac[m][c] < 1.0)){
          message += " ( vol_frac[m][c] > 0.0 ) ||( vol_frac[m][c] < 1.0) ";
        }
      }
      
      if ( fabs(sum - 1.0) > convergence_crit && !tsr) {  
        allTestsPassed = false;
        message += " sum (volumeFractions) != 1 ";
      }
      
      if ( press_new[c] < 0.0 && !tsr) {
        allTestsPassed = false;
        message += " Computed pressure is < 0 ";
      }
      
      for( int m = 0; m < numMatls; m++ ) {
        if( (rho_micro[m][c] < 0.0 || vol_frac[m][c] < 0.0) && !tsr ) {
          allTestsPassed = false;
          message += " rho_micro < 0 || vol_frac < 0";
        }
      }
      if(allTestsPassed != true){  // throw an exception of there's a problem
        ostringstream warn;
        warn << "\nICE::ComputeEquilibrationPressure: Cell "<< c << ", L-"<<L_indx <<"\n"
             << message
             <<"\nThis usually means that something much deeper has gone wrong with the simulation. "
             <<"\nCompute equilibration pressure task is rarely the problem. "
             << "For more debugging information set the environmental variable:  \n"
             << "   SCI_DEBUG DBG_EqPress:+\n\n";
             
        warn << "INPUTS: \n"; 
        for (int m = 0; m < numMatls; m++){
          warn<< "\n matl: " << m << "\n"
               << "   rho_CC:     " << rho_CC[m][c] << "\n"
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
            for (int m = 0; m < numMatls; m++){
              warn << "  matl: " << d.matl[m].mat
                   << "  press_eos:  " << d.matl[m].press_eos
                   << "  volFrac:    " << d.matl[m].volFrac
                   << "  rhoMicro:   " << d.matl[m].rhoMicro
                   << "  rho_CC:     " << d.matl[m].rho_CC
                   << "  Temp:       " << d.matl[m].temp_CC << "\n";
            }
          }
        }
        throw InvalidValue(warn.str(), __FILE__, __LINE__); 
      }

      if (switchDebug_equil_press) {
        n_iters_equil_press[c] = count;
      }
      
    } // end of cell interator

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
    preprocess_CustomBCs("EqPress",old_dw, new_dw, lb,  patch, 
                          999,d_customBC_var_basket);
    
    setBC(press_new,   rho_micro, placeHolder, d_surroundingMatl_indx,
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw, 
          d_customBC_var_basket);
          
    delete_CustomBCs(d_customBC_var_basket);      
   
    
    //__________________________________
    // compute sp_vol_CC
    // - Set BCs on rhoMicro. using press_CC 
    // - backout sp_vol_new 
    for (int m = 0; m < numMatls; m++)   {
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_new[m][c] = 1.0/rho_micro[m][c]; 
      }
      
      ICEMaterial* matl = d_sharedState->getICEMaterial(m);
      int indx = matl->getDWIndex();
      setSpecificVolBC(sp_vol_new[m], "SpecificVol", false, rho_CC[m], vol_frac[m],
                       patch,d_sharedState, indx);
    }
    
    //__________________________________
    //  compute f_theta  
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      sumKappa[c] = 0.0;
      for (int m = 0; m < numMatls; m++) {
        kappa[m][c] = sp_vol_new[m][c]/(speedSound_new[m][c]*speedSound_new[m][c]);
        sumKappa[c] += vol_frac[m][c]*kappa[m][c];
      }
      for (int m = 0; m < numMatls; m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m][c]/sumKappa[c];
      }
    }

   //---- P R I N T   D A T A ------   
    if (switchDebug_equil_press) {
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
 Function~  ICE::computeEquilPressure_1_matl--
 Purpose~   Simple EOS evaluation
_____________________________________________________________________*/ 
void ICE::computeEquilPressure_1_matl(const ProcessorGroup*,  
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing computeEquilPressure_1_matl on patch "<<patch->getID()
               << "\t\t ICE \tL-" <<L_indx<< endl;

    CCVariable<double> vol_frac, sp_vol_new; 
    CCVariable<double> speedSound, f_theta, kappa;
    CCVariable<double> press_eq, sumKappa, sum_imp_delP, rho_CC_new;
    constCCVariable<double> Temp,rho_CC, sp_vol_CC, cv, gamma;   
    StaticArray<CCVariable<double> > rho_micro(1);
    
    Ghost::GhostType  gn = Ghost::None;
    ICEMaterial* ice_matl = d_sharedState->getICEMaterial(0);   
    int indx = ice_matl->getDWIndex();
    
    //__________________________________
    old_dw->get(Temp,      lb->temp_CCLabel,      indx,patch, gn,0);
    old_dw->get(rho_CC,    lb->rho_CCLabel,       indx,patch, gn,0);
    old_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,    indx,patch, gn,0);
    new_dw->get(cv,        lb->specific_heatLabel,indx,patch, gn,0);
    new_dw->get(gamma,     lb->gammaLabel,        indx,patch, gn,0);
       
    new_dw->allocateTemporary(rho_micro[0],  patch);

    new_dw->allocateAndPut(press_eq,     lb->press_equil_CCLabel, 0,  patch);
    new_dw->allocateAndPut(sumKappa,     lb->sumKappaLabel,       0,  patch);
    new_dw->allocateAndPut(sum_imp_delP, lb->sum_imp_delPLabel,   0,  patch);
    new_dw->allocateAndPut(kappa,        lb->compressibilityLabel,indx,patch);
    new_dw->allocateAndPut(vol_frac,     lb->vol_frac_CCLabel,   indx,patch);    
    new_dw->allocateAndPut(sp_vol_new,   lb->sp_vol_CCLabel,     indx,patch);     
    new_dw->allocateAndPut(f_theta,      lb->f_theta_CCLabel,    indx,patch);
    new_dw->allocateAndPut(speedSound,   lb->speedSound_CCLabel, indx,patch);       
    sum_imp_delP.initialize(0.0);       

    new_dw->allocateAndPut(rho_CC_new,   lb->rho_CCLabel,        indx,patch);

    //---- P R I N T   D A T A ------   
    if (switchDebug_equil_press) {
      ostringstream desc;
      desc << "TOP_equilibration_patch_" << patch->getID();
      printData( indx, patch, 1, desc.str(), "temp",      Temp);
      printData( indx, patch, 1, desc.str(), "sp_vol_CC", sp_vol_CC);
      printData( indx, patch, 1, desc.str(), "cv",        cv);
      printData( indx, patch, 1, desc.str(), "gamma",     gamma);
    }

    //______________________________________________________________________
    //  Main loop
    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
      IntVector c = *iter;
      vol_frac[c]      = 1.0;
      rho_micro[0][c]  = rho_CC_new[c] = rho_CC[c];
      sp_vol_new[c]    = 1.0/rho_CC[c];
      double dp_drho, dp_de, c_2;
      //__________________________________
      // evaluate EOS
      ice_matl->getEOS()->computePressEOS(rho_micro[0][c],gamma[c],
                                          cv[c], Temp[c], press_eq[c],
                                          dp_drho, dp_de);
                                          
      c_2 = dp_drho + dp_de * press_eq[c]/(rho_micro[0][c] * rho_micro[0][c]);
      speedSound[c] = sqrt(c_2);
      
      //  compute f_theta  
      kappa[c]    = sp_vol_new[c]/(speedSound[c]*speedSound[c]);
      sumKappa[c] = kappa[c];
      f_theta[c]  = 1.0;
    }
    //__________________________________
    // - apply Boundary conditions
    StaticArray<constCCVariable<double> > placeHolder(0);
    preprocess_CustomBCs("EqPress",old_dw, new_dw, lb,  patch, 
                          999,d_customBC_var_basket);
    
    setBC(press_eq,   rho_micro, placeHolder, d_surroundingMatl_indx,
          "rho_micro", "Pressure", patch , d_sharedState, 0, new_dw, 
          d_customBC_var_basket);
          
    delete_CustomBCs(d_customBC_var_basket);      

    //---- P R I N T   D A T A ------   
    if (switchDebug_equil_press) {
      ostringstream desc;
      desc << "BOT_equilibration_patch_" << patch->getID();
      printData( 0,    patch, 1, desc.str(), "Press_CC_equil", press_eq);
    }
  }  // patch loop
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
    
    cout_doing << d_myworld->myrank() << " Doing compute_FC_Temp on patch " 
              << patch->getID() << "\t\t\t ICE \tL-" <<level->getIndex()<< endl;
            
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

      SFCXVariable<double> TempX_FC;
      SFCYVariable<double> TempY_FC;
      SFCZVariable<double> TempZ_FC; 
      new_dw->allocateAndPut(TempX_FC,lb->TempX_FCLabel,indx, patch);   
      new_dw->allocateAndPut(TempY_FC,lb->TempY_FCLabel,indx, patch);   
      new_dw->allocateAndPut(TempZ_FC,lb->TempZ_FCLabel,indx, patch);   
      
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      TempX_FC.initialize(0.0,lowIndex,patch->getExtraSFCXHighIndex()); 
      TempY_FC.initialize(0.0,lowIndex,patch->getExtraSFCYHighIndex()); 
      TempZ_FC.initialize(0.0,lowIndex,patch->getExtraSFCZHighIndex());
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces

      CellIterator XFC_iterator = patch->getSFCXIterator();
      CellIterator YFC_iterator = patch->getSFCYIterator();
      CellIterator ZFC_iterator = patch->getSFCZIterator();
      
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
      computeTempFace<SFCXVariable<double> >(XFC_iterator, adj_offset[0], 
                                             rho_CC,Temp_CC, TempX_FC);

      computeTempFace<SFCYVariable<double> >(YFC_iterator, adj_offset[1], 
                                             rho_CC,Temp_CC, TempY_FC);

      computeTempFace<SFCZVariable<double> >(ZFC_iterator,adj_offset[2],
                                             rho_CC,Temp_CC, TempZ_FC);

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

/* _____________________________________________________________________
 Function~  ICE::computeFaceCenteredVelocities--
 Purpose~   compute the face centered velocities minus the exchange
            contribution.
_____________________________________________________________________*/
template<class T> void ICE::computeVelFace(int dir, 
                                           CellIterator it,
                                           IntVector adj_offset,
                                           double dx,
                                           double delT, double gravity,
                                           constCCVariable<double>& rho_CC,
                                           constCCVariable<double>& sp_vol_CC,
                                           constCCVariable<Vector>& vel_CC,
                                           constCCVariable<double>& press_CC,
                                           T& vel_FC,
                                           T& grad_P_FC,
                                           bool include_acc)
{
  double inv_dx = 1.0/dx;

  double one_or_zero=1.;
  if(!include_acc){
    one_or_zero=0.0;
  }
  
  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset; 

    double rho_FC = rho_CC[L] + rho_CC[R];
#if SCI_ASSERTION_LEVEL >=2
    if (rho_FC <= 0.0) {
      cout << d_myworld->myrank() << " rho_fc <= 0: " << rho_FC << " with L= " << L << " (" 
           << rho_CC[L] << ") R= " << R << " (" << rho_CC[R]<< ")\n";
    }
#endif
    ASSERT(rho_FC > 0.0);

    //__________________________________
    // interpolation to the face
    double term1 = (rho_CC[L] * vel_CC[L][dir] +
                    rho_CC[R] * vel_CC[R][dir])/(rho_FC);            
    //__________________________________
    // pressure term           
    double sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                             (sp_vol_CC[L] + sp_vol_CC[R]); 
                             
    grad_P_FC[R] = (press_CC[R] - press_CC[L]) * inv_dx;
    double term2 = delT * sp_vol_brack * grad_P_FC[R];
     
    //__________________________________
    // gravity term
    double term3 =  delT * gravity;
    
    vel_FC[R] = term1 - one_or_zero*term2 + one_or_zero*term3;
  } 
}
                  
//______________________________________________________________________
//                       
void ICE::computeVel_FC(const ProcessorGroup*,  
                        const PatchSubset* patches,                
                        const MaterialSubset* /*matls*/,           
                        DataWarehouse* old_dw,                     
                        DataWarehouse* new_dw)                     
{
  const Level* level = getLevel(patches);
  
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << d_myworld->myrank() << " Doing computeVel_FC on patch " 
           << patch->getID() << "\t\t\t ICE \tL-"<<level->getIndex()<< endl;

    int numMatls = d_sharedState->getNumMatls();
    
    Vector dx      = patch->dCell();
    Vector gravity = getGravity();
    
    constCCVariable<double> press_CC;
    Ghost::GhostType  gac = Ghost::AroundCells; 
    new_dw->get(press_CC,lb->press_equil_CCLabel, 0, patch,gac, 1);
    
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);   
     
    // Compute the face centered velocities
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl); 
      constCCVariable<double> rho_CC, sp_vol_CC;
      constCCVariable<Vector> vel_CC;
      if(ice_matl){
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        old_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac, 1); 
      } else {
        new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch, gac, 1);
        new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch, gac, 1);
      }              
      new_dw->get(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch, gac, 1);
              
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
      SFCXVariable<double> uvel_FC, grad_P_XFC;
      SFCYVariable<double> vvel_FC, grad_P_YFC;
      SFCZVariable<double> wvel_FC, grad_P_ZFC;

      new_dw->allocateAndPut(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(wvel_FC, lb->wvel_FCLabel, indx, patch);
      // debugging variables
      new_dw->allocateAndPut(grad_P_XFC, lb->grad_P_XFCLabel, indx, patch);
      new_dw->allocateAndPut(grad_P_YFC, lb->grad_P_YFCLabel, indx, patch);
      new_dw->allocateAndPut(grad_P_ZFC, lb->grad_P_ZFCLabel, indx, patch);   
      
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      uvel_FC.initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex());
      vvel_FC.initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex());
      wvel_FC.initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex());
      
      grad_P_XFC.initialize(0.0);
      grad_P_YFC.initialize(0.0);
      grad_P_ZFC.initialize(0.0);
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

      CellIterator XFC_iterator = patch->getSFCXIterator();
      CellIterator YFC_iterator = patch->getSFCYIterator();
      CellIterator ZFC_iterator = patch->getSFCZIterator();

      bool include_acc = true;
      if(mpm_matl && d_with_rigid_mpm){
        include_acc = false;
      }

      //__________________________________
      //  Compute vel_FC for each face
      computeVelFace<SFCXVariable<double> >(0, XFC_iterator,
                                       adj_offset[0],dx[0],delT,gravity[0],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       uvel_FC, grad_P_XFC, include_acc);

      computeVelFace<SFCYVariable<double> >(1, YFC_iterator,
                                       adj_offset[1],dx[1],delT,gravity[1],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       vvel_FC, grad_P_YFC, include_acc);

      computeVelFace<SFCZVariable<double> >(2, ZFC_iterator,
                                       adj_offset[2],dx[2],delT,gravity[2],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       wvel_FC, grad_P_ZFC, include_acc);

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
        printData_FC( indx, patch,1, desc.str(), "grad_P_XFC", grad_P_XFC);
        printData_FC( indx, patch,1, desc.str(), "grad_P_YFC", grad_P_YFC);
        printData_FC( indx, patch,1, desc.str(), "grad_P_ZFC", grad_P_ZFC); 
        
      }
    } // matls loop
  }  // patch loop
}
/* _____________________________________________________________________
 Function~  ICE::updateVelFace--
 - tack on delP to the face centered velocity
_____________________________________________________________________*/
template<class T> void ICE::updateVelFace(int dir, CellIterator it,
                                          IntVector adj_offset,
                                          double dx,
                                          double delT,
                                          constCCVariable<double>& sp_vol_CC,
                                          constCCVariable<double>& imp_delP,
                                          T& vel_FC,
                                          T& grad_dp_FC)
{
  double inv_dx = 1.0/dx;
  
  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset; 

    //__________________________________
    // pressure term           
    double sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                             (sp_vol_CC[L] + sp_vol_CC[R]); 
    
    grad_dp_FC[R] = (imp_delP[R] - imp_delP[L])*inv_dx;
    double term2 = delT * sp_vol_brack * grad_dp_FC[R];
    
    vel_FC[R] -= term2;
  } 
} 
//______________________________________________________________________
//                       
void ICE::updateVel_FC(const ProcessorGroup*,  
                       const PatchSubset* patches,                
                       const MaterialSubset* /*matls*/,           
                       DataWarehouse* old_dw,                     
                       DataWarehouse* new_dw,
                       bool recursion)                     
{
  const Level* level = getLevel(patches);
  
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << d_myworld->myrank() << " Doing updateVel_FC on patch " 
           << patch->getID() << "\t\t\t\tICE \tL-"<<level->getIndex()<< endl;

    int numMatls = d_sharedState->getNumMatls();
    
    Vector dx      = patch->dCell();
    Ghost::GhostType  gac = Ghost::AroundCells; 
    Ghost::GhostType  gn = Ghost::None; 
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;
    
    //__________________________________
    // define parent data warehouse
    if(recursion) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW); 
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    }
 
    delt_vartype delT;
    pOldDW->get(delT, d_sharedState->get_delt_label(),level);   
     
    constCCVariable<double> imp_delP; 
    new_dw->get(imp_delP, lb->imp_delPLabel, 0,   patch,gac, 1);
 
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex(); 
      constCCVariable<double> sp_vol_CC;         
      pNewDW->get(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch, gac, 1);
              
      SFCXVariable<double> uvel_FC, grad_dp_XFC;
      SFCYVariable<double> vvel_FC, grad_dp_YFC;
      SFCZVariable<double> wvel_FC, grad_dp_ZFC;
      
      constSFCXVariable<double> uvel_FC_old;
      constSFCYVariable<double> vvel_FC_old;
      constSFCZVariable<double> wvel_FC_old;

      old_dw->get(uvel_FC_old,  lb->uvel_FCLabel, indx, patch, gn, 0);
      old_dw->get(vvel_FC_old,  lb->vvel_FCLabel, indx, patch, gn, 0);
      old_dw->get(wvel_FC_old,  lb->wvel_FCLabel, indx, patch, gn, 0);
      
      new_dw->allocateAndPut(uvel_FC, lb->uvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(vvel_FC, lb->vvel_FCLabel, indx, patch);
      new_dw->allocateAndPut(wvel_FC, lb->wvel_FCLabel, indx, patch); 
      
      new_dw->allocateAndPut(grad_dp_XFC, lb->grad_dp_XFCLabel, indx, patch);
      new_dw->allocateAndPut(grad_dp_YFC, lb->grad_dp_YFCLabel, indx, patch);
      new_dw->allocateAndPut(grad_dp_ZFC, lb->grad_dp_ZFCLabel, indx, patch);  
      
      uvel_FC.copy(uvel_FC_old);
      vvel_FC.copy(vvel_FC_old);
      wvel_FC.copy(wvel_FC_old);
      
      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces     

      CellIterator XFC_iterator = patch->getSFCXIterator();
      CellIterator YFC_iterator = patch->getSFCYIterator();
      CellIterator ZFC_iterator = patch->getSFCZIterator();

      updateVelFace<SFCXVariable<double> >(0, XFC_iterator,
                                     adj_offset[0],dx[0],delT,
                                     sp_vol_CC,imp_delP,
                                     uvel_FC, grad_dp_XFC);

      updateVelFace<SFCYVariable<double> >(1, YFC_iterator,
                                     adj_offset[1],dx[1],delT,
                                     sp_vol_CC,imp_delP,
                                     vvel_FC, grad_dp_YFC);

      updateVelFace<SFCZVariable<double> >(2, ZFC_iterator,
                                     adj_offset[2],dx[2],delT,
                                     sp_vol_CC,imp_delP,
                                     wvel_FC, grad_dp_ZFC);

      //__________________________________
      // (*)vel_FC BC are updated in 
      // ICE::addExchangeContributionToFCVel()

      //---- P R I N T   D A T A ------ 
      if (switchDebug_vel_FC ) {
        ostringstream desc;
        desc <<"BOT_updateVel_FC_Mat_" << indx << "_patch_"<< patch->getID();
        printData_FC( indx, patch,1, desc.str(), "uvel_FC",  uvel_FC);
        printData_FC( indx, patch,1, desc.str(), "vvel_FC",  vvel_FC);
        printData_FC( indx, patch,1, desc.str(), "wvel_FC",  wvel_FC);
        printData_FC( indx, patch,1, desc.str(), "grad_dp_XFC", grad_dp_XFC);
        printData_FC( indx, patch,1, desc.str(), "grad_dp_YFC", grad_dp_YFC);
        printData_FC( indx, patch,1, desc.str(), "grad_dp_ZFC", grad_dp_ZFC);
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
                                         bool recursion)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing Add_exchange_contribution_to_FC_vel on patch " <<
      patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;
 
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
    IntVector lowIndex(patch->getExtraSFCXLowIndex()); 
    
    // Extract the momentum exchange coefficients
    FastMatrix K(numMatls, numMatls), junk(numMatls, numMatls);

    K.zero();
    getConstantExchangeCoefficients( K, junk);
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

      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCXHighIndex());
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCYHighIndex());
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCZHighIndex());
      
      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex());
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex());
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex());
    }   
    
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
    
    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();
                                
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
    for (int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      preprocess_CustomBCs("velFC_Exchange",pOldDW, pNewDW, lb,  patch, indx,
                            d_customBC_var_basket);
      
      setBC<SFCXVariable<double> >(uvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket); 
      setBC<SFCYVariable<double> >(vvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket);
      setBC<SFCZVariable<double> >(wvel_FCME[m],"Velocity",patch,indx,
                                    d_sharedState, d_customBC_var_basket);
      delete_CustomBCs(d_customBC_var_basket);
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
    cout_doing << d_myworld->myrank() << " Doing explicit delPress on patch " << patch->getID() 
         <<  "\t\t\t ICE \tL-" <<level->getIndex()<< endl;

    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx     = patch->dCell();
    double inv_vol    = 1.0/(dx.x()*dx.y()*dx.z()); 
    
    bool newGrid = d_sharedState->isRegridTimestep();
    Advector* advector = d_advector->clone(new_dw,patch,newGrid );   

    CCVariable<double> q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;
    CCVariable<double> press_CC;
    CCVariable<double> term1, term2;
    constCCVariable<double>sumKappa, press_equil;
    StaticArray<CCVariable<double> > placeHolder(0);
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
   
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(sumKappa,                lb->sumKappaLabel,      0,patch,gn,0);
    new_dw->get(press_equil,             lb->press_equil_CCLabel,0,patch,gn,0);
    new_dw->allocateAndPut( press_CC,    lb->press_CCLabel,      0, patch);   
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel, 0, patch);   
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,    0, patch);
    new_dw->allocateAndPut(term2,        lb->term2Label,         0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,    0, patch); 

    new_dw->allocateTemporary(q_advected, patch);
    new_dw->allocateTemporary(term1,      patch);
    
    term1.initialize(0.);
    term2.initialize(0.);
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

      new_dw->allocateAndPut(vol_fracX_FC, lb->vol_fracX_FCLabel,  indx,patch);
      new_dw->allocateAndPut(vol_fracY_FC, lb->vol_fracY_FCLabel,  indx,patch);
      new_dw->allocateAndPut(vol_fracZ_FC, lb->vol_fracZ_FCLabel,  indx,patch);
      
           
      // lowIndex is the same for all vel_FC
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      double nan= getNan();
      vol_fracX_FC.initialize(nan, lowIndex,patch->getExtraSFCXHighIndex());
      vol_fracY_FC.initialize(nan, lowIndex,patch->getExtraSFCYHighIndex());
      vol_fracZ_FC.initialize(nan, lowIndex,patch->getExtraSFCZHighIndex()); 
      
          
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
      // common variables that get passed into the advection operators
      advectVarBasket* varBasket = scinew advectVarBasket();
      varBasket->doRefluxing = false;  // don't need to reflux here
      
      advector->advectQ(vol_frac, patch, q_advected, varBasket,  
                        vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw);
                        
      delete varBasket; 
      
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        term2[c] -= q_advected[c]; 
      }
      
      //__________________________________
      //   term1 contribution from models
      if(d_models.size() > 0){
        constCCVariable<double> modelMass_src, modelVol_src;
        new_dw->get(modelMass_src, lb->modelMass_srcLabel, indx, patch, gn, 0);

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         term1[c] += modelMass_src[c] * (sp_vol_CC[m][c]* inv_vol);
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
    
    //__________________________________
    //  add delP to press_equil
    //  AMR:  hit the extra cells, BC aren't set an you need a valid pressure there
    // THIS COULD BE TROUBLE
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = press_equil[c];
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      double inv_sumKappa = 1.0/sumKappa[c];
      delP_MassX[c]    =  term1[c] * inv_sumKappa;
      delP_Dilatate[c] = -term2[c] * inv_sumKappa;
      press_CC[c]      =  press_equil[c] + delP_MassX[c] + delP_Dilatate[c];
      press_CC[c]      = max(1.0e-12, press_CC[c]);  // CLAMP
//      delP_Dilatate[c] = press_CC[c] - delP_MassX[c] - press_equil[c];
    }

    //__________________________________
    //  set boundary conditions
    preprocess_CustomBCs("update_press_CC",old_dw, new_dw, lb,  patch, 999,
                          d_customBC_var_basket);
    
    setBC(press_CC, placeHolder, sp_vol_CC, d_surroundingMatl_indx,
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw,
          d_customBC_var_basket);
#if SET_CFI_BC          
    set_CFI_BC<double>(press_CC,patch);
#endif   
    delete_CustomBCs(d_customBC_var_basket);      
       
   //---- P R I N T   D A T A ------  
    if (switchDebug_explicit_press) {
      ostringstream desc;
      desc << "BOT_explicit_Pressure_patch_" << patch->getID();
//    printData( 0, patch, 1,desc.str(), "term1",         term1);
      printData( 0, patch, 1,desc.str(), "term2",         term2);
      printData( 0, patch, 1,desc.str(), "sumKappa",      sumKappa); 
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
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << d_myworld->myrank() << " Doing press_face_MM on patch " << patch->getID() 
         << "\t\t\t ICE \tL-" <<level->getIndex()<< endl;
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
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << d_myworld->myrank() << " Doing zeroModelSources on patch " 
               << patch->getID() << "\t\t\t ICE \tL-" <<level->getIndex()<< endl;
      
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
    }

    for(vector<TransportedVariable*>::iterator
                                  iter = d_modelSetup->tvars.begin();
                                  iter != d_modelSetup->tvars.end(); iter++){
      TransportedVariable* tvar = *iter;
      for(int m=0;m<tvar->matls->size();m++){
        int matl = tvar->matls->get(m);

        if(tvar->src){
          CCVariable<double> model_src;
          new_dw->allocateAndPut(model_src, tvar->src, matl, patch);
          model_src.initialize(0.0);
        }
      }  // matl loop
    }  // transported Variables
  }  // patches loop
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
    CCVariable<double> sumKappa;
    StaticArray<CCVariable<double> > vol_frac(numALLMatls);
    StaticArray<CCVariable<double> > f_theta(numALLMatls);
    StaticArray<constCCVariable<double> > rho_CC(numALLMatls);
    StaticArray<constCCVariable<double> > sp_vol(numALLMatls);
    StaticArray<constCCVariable<double> > modVolSrc(numALLMatls);
    StaticArray<constCCVariable<double> > kappa(numALLMatls);
    new_dw->getModifiable(sumKappa, lb->sumKappaLabel, 0,patch);
    
    Vector dx  = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();
    
    
    for(int m=0;m<matls->size();m++){
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      new_dw->getModifiable(vol_frac[m], lb->vol_frac_CCLabel, indx,patch);
      new_dw->getModifiable(f_theta[m],  lb->f_theta_CCLabel,  indx,patch);
      new_dw->get(rho_CC[m],      lb->rho_CCLabel,             indx,patch,gn,0);
      new_dw->get(sp_vol[m],      lb->sp_vol_CCLabel,          indx,patch,gn,0);
      new_dw->get(modVolSrc[m],   lb->modelVol_srcLabel,       indx,patch,gn,0);
      new_dw->get(kappa[m],       lb->compressibilityLabel,    indx,patch,gn,0);
    }

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      double total_vol=0.;
      for(int m=0;m<matls->size();m++){
        total_vol+=(rho_CC[m][c]*vol)*sp_vol[m][c];
      }

      double sumKappa_tmp = 0.0;
      for(int m=0;m<matls->size();m++){
        double new_vol = vol_frac[m][c]*total_vol+modVolSrc[m][c];
        vol_frac[m][c] = max(new_vol/total_vol,1.e-100);
        sumKappa_tmp += vol_frac[m][c] * kappa[m][c];
      }
      sumKappa[c] = sumKappa_tmp;
      for (int m = 0; m < matls->size(); m++) {
        f_theta[m][c] = vol_frac[m][c]*kappa[m][c]/sumKappa[c];
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

    cout_doing << d_myworld->myrank() << " Doing accumulate_momentum_source_sinks_MM on patch " <<
      patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;

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
    gravity = getGravity();
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
      new_dw->get(vol_frac,lb->vol_frac_CCLabel, indx,patch,gac,2);
      CCVariable<Vector>   mom_source;
      new_dw->allocateAndPut(mom_source,  lb->mom_source_CCLabel,  indx, patch);
      mom_source.initialize(Vector(0.,0.,0.));
      
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
          }         
          computeTauX(patch, vol_frac, vel_CC,viscosity,dx, tau_X_FC);
          computeTauY(patch, vol_frac, vel_CC,viscosity,dx, tau_Y_FC);
          computeTauZ(patch, vol_frac, vel_CC,viscosity,dx, tau_Z_FC);
        }
        if(viscosity_test == 0.0 && d_turbulence){
          string warn="ERROR:\n input :viscosity can't be zero when calculate turbulence";
          throw ProblemSetupException(warn, __FILE__, __LINE__);
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
        
        viscous_source=(tau_X_FC[right].x() - tau_X_FC[left].x())  * areaX +
                       (tau_Y_FC[top].x()   - tau_Y_FC[bottom].x())* areaY +
                       (tau_Z_FC[front].x() - tau_Z_FC[back].x())  * areaZ;             

        mom_source[c].x( (-pressure_source * areaX + 
                           viscous_source +
                           mass * gravity.x() * include_term) * delT ); 

        //__________________________________
        //    Y - M O M E N T U M
        pressure_source = (pressY_FC[top]-pressY_FC[bottom])* vol_frac[c]; 

        viscous_source=(tau_X_FC[right].y() - tau_X_FC[left].y())  * areaX +
                       (tau_Y_FC[top].y()   - tau_Y_FC[bottom].y())* areaY +
                       (tau_Z_FC[front].y() - tau_Z_FC[back].y())  * areaZ;

        mom_source[c].y( (-pressure_source * areaY +
                           viscous_source +
                           mass * gravity.y() * include_term) * delT );    
   
        //__________________________________
        //    Z - M O M E N T U M
        pressure_source = (pressZ_FC[front]-pressZ_FC[back]) * vol_frac[c]; 

        viscous_source=(tau_X_FC[right].z() - tau_X_FC[left].z())  * areaX +
                       (tau_Y_FC[top].z()   - tau_Y_FC[bottom].z())* areaY +
                       (tau_Z_FC[front].z() - tau_Z_FC[back].z())  * areaZ;

        mom_source[c].z( (-pressure_source * areaZ +
                           viscous_source + 
                           mass * gravity.z() * include_term) * delT );
      }

      //---- P R I N T   D A T A ------ 
      if (switchDebug_Source_Sink) {
        ostringstream desc;
        desc << "sources_sinks_Mat_" << indx << "_patch_"<<  patch->getID();
        printVector(indx, patch, 1, desc.str(), "mom_source",  0, mom_source);
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
    cout_doing << d_myworld->myrank() 
               << " Doing accumulate_energy_source_sinks on patch " 
               << patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;

    int numMatls = d_sharedState->getNumMatls();

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    Vector dx = patch->dCell();
    double A, vol=dx.x()*dx.y()*dx.z();
    
    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> kappa;
    constCCVariable<double> vol_frac;
    constCCVariable<double> press_CC;
    constCCVariable<double> delP_Dilatate;
    constCCVariable<double> matl_press;
    constCCVariable<double> rho_CC;
    constCCVariable<double> TMV_CC;

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(press_CC,     lb->press_CCLabel,      0, patch,gn, 0);
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,gn, 0);

    if(d_with_mpm){
      new_dw->get(TMV_CC,     lb->TMV_CCLabel,        0, patch,gn, 0);
    }
    else {
      CCVariable<double>  TMV_create;
      new_dw->allocateTemporary(TMV_create,  patch);
      TMV_create.initialize(vol);
      TMV_CC = TMV_create; // reference created data
    }

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl); 
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl); 

      int indx    = matl->getDWIndex();   
      CCVariable<double> int_eng_source;
      CCVariable<double> heatCond_src;
      
      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,      indx,patch, gac,1);
      new_dw->get(rho_CC,     lb->rho_CCLabel,         indx,patch, gac,1);
      new_dw->get(kappa,      lb->compressibilityLabel,indx,patch, gn, 0);
      new_dw->get(vol_frac,   lb->vol_frac_CCLabel,    indx,patch, gac,1);
       
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

          bool use_vol_frac = true; // include vol_frac in diffusion calc.
          scalarDiffusionOperator(new_dw, patch, use_vol_frac, Temp_CC,
                                  vol_frac, heatCond_src, thermalCond, delT);
        }
      }
                                     
      //__________________________________
      //   Compute source from volume dilatation
      //   Exclude contribution from delP_MassX
      bool includeFlowWork = false;
      if (ice_matl)
        includeFlowWork = ice_matl->getIncludeFlowWork();
      if (mpm_matl)
        includeFlowWork = mpm_matl->getIncludeFlowWork();
      if(includeFlowWork){
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          //          A = vol * vol_frac[c] * kappa[c] * press_CC[c];
          A = TMV_CC[c] * vol_frac[c] * kappa[c] * press_CC[c];
          int_eng_source[c] += A * delP_Dilatate[c] + heatCond_src[c];
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
      if (switchDebug_Source_Sink) {
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
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << d_myworld->myrank() << " Doing Lagrangian mass, momentum and energy on patch " <<
      patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;

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
     double tiny_rho = 1.e-12;
     if(ice_matl)  {               //  I C E
      constCCVariable<double> rho_CC, temp_CC, cv, int_eng_source;
      constCCVariable<Vector> vel_CC, mom_source, mom_comb;
      tiny_rho = ice_matl->getTinyRho();

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
          double min_mass = tiny_rho * vol;

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
#if 0
         if(massGain > 0.0){
           cout << "Mass gained by the models this timestep = " 
                << massGain << "\t L-" <<level->getIndex()<<endl;
         }
#endif
       }  //  if (models.size() > 0)

        //---- P R I N T   D A T A ------ 
        // Dump out all the matls data
        if (switchDebug_LagrangianValues ) {
          ostringstream desc;
          desc <<"BOT_Lagrangian_Values_Mat_"<<indx<< "_patch_"<<patch->getID();
          printData(  indx, patch,1, desc.str(), "mass_L_CC",    mass_L);
          printVector(indx, patch,1, desc.str(), "mom_L_CC", 0,  mom_L);
          printData(  indx, patch,1, desc.str(), "int_eng_L_CC", int_eng_L); 

        }
        //____ B U L L E T   P R O O F I N G----
        // catch negative internal energies
        // ignore BP if timestep restart has already been requested
        IntVector neg_cell;
        bool tsr = new_dw->timestepRestarted();
        
        if (!areAllValuesPositive(int_eng_L, neg_cell) && !tsr ) {
         ostringstream warn;
         int idx = level->getIndex();
         warn<<"ICE:(L-"<<idx<<"):computeLagrangianValues, mat "<<indx<<" cell "
             <<neg_cell<<" Negative int_eng_L: " << int_eng_L[neg_cell] <<  "\n";
         throw InvalidValue(warn.str(), __FILE__, __LINE__);
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

    cout_doing << d_myworld->myrank() << " Doing computeLagrangianSpecificVolume " <<
      patch->getID() << "\t\t ICE \tL-" <<level->getIndex()<< endl;

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
    constCCVariable<double> TMV_CC;
    CCVariable<double> sum_therm_exp;
    vector<double> if_mpm_matl_ignore(numALLMatls);

    new_dw->allocateTemporary(sum_therm_exp,patch);
    new_dw->get(delP, lb->delP_DilatateLabel, 0, patch,gn, 0);
    new_dw->get(P,    lb->press_CCLabel,      0, patch,gn, 0);
    sum_therm_exp.initialize(0.);

    if(d_with_mpm){
      new_dw->get(TMV_CC,     lb->TMV_CCLabel,        0, patch,gn, 0);
    }
    else {
      CCVariable<double>  TMV_create;
      new_dw->allocateTemporary(TMV_create,  patch);
      TMV_create.initialize(vol);
      TMV_CC = TMV_create; // reference created data
    }

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
      constCCVariable<double> kappa;
      new_dw->allocateAndPut(sp_vol_L,  lb->sp_vol_L_CCLabel,   indx,patch);
      new_dw->allocateAndPut(sp_vol_src,lb->sp_vol_src_CCLabel, indx,patch);
      sp_vol_src.initialize(0.);
      double tiny_rho = 1.e-12;
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      if (ice_matl) {
        tiny_rho = ice_matl->getTinyRho();
      }

      new_dw->get(sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,gn, 0);
      new_dw->get(rho_CC,     lb->rho_CCLabel,        indx,patch,gn, 0);
      new_dw->get(f_theta,    lb->f_theta_CCLabel,    indx,patch,gn, 0);
      new_dw->get(kappa,      lb->compressibilityLabel,indx,patch,gn, 0);

      //__________________________________
      //  compute sp_vol_L * mass
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c];
      }

      //---- P R I N T   D A T A ------ 
      if (switchDebug_LagrangianSpecificVol ) {
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
//        double term1 = -vol_frac[m][c] * kappa[c] * TMV_CC[c] * delP[c];
        double term1 = -vol_frac[m][c] * kappa[c] * vol * delP[c];
//        double term2 = delT * TMV_CC[c] *
        double term2 = delT * vol *
                             (vol_frac[m][c] * alpha[m][c] * Tdot[m][c] -
                                               f_theta[c] * sum_therm_exp[c]);

        // This is actually mass * sp_vol
        double src = term1 + if_mpm_matl_ignore[m] * term2;
        sp_vol_L[c]  += src;
        sp_vol_src[c] = src/(rho_CC[c] * vol);
      }

      if(d_clampSpecificVolume){
        for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
/*`==========TESTING==========*/
          sp_vol_L[c] = max(sp_vol_L[c], tiny_rho * vol * sp_vol_CC[c]);
/*==========TESTING==========`*/
        }
      }

      //__________________________________
      // Apply boundary conditions
      setSpecificVolBC(sp_vol_L, "SpecificVol", true ,rho_CC,vol_frac[m],
                       patch,d_sharedState, indx);
      
      //---- P R I N T   D A T A ------ 
      if (switchDebug_LagrangianSpecificVol ) {
        ostringstream desc;
        desc <<"BOT_Lagrangian_sp_vol_Mat_"<<indx<< "_patch_"<<patch->getID();
        printData( indx, patch,1, desc.str(), "sp_vol_L",   sp_vol_L);    
        printData( indx, patch,1, desc.str(), "sp_vol_src", sp_vol_src);  
        if(d_models.size() > 0){
          printData( indx, patch,1, desc.str(), "Modelsp_vol_src", Modelsp_vol_src);
        }
      }
      //____ B U L L E T   P R O O F I N G----
      // ignore BP if timestep restart has already been requested
      IntVector neg_cell;
      bool tsr = new_dw->timestepRestarted();
      
      if (!areAllValuesPositive(sp_vol_L, neg_cell) && !tsr) {
        cout << "\nICE:WARNING......Negative specific Volume"<< endl;
        cout << "cell              "<< neg_cell << " level " <<  level->getIndex() << endl;
        cout << "matl              "<< indx << endl;
        cout << "sum_thermal_exp   "<< sum_therm_exp[neg_cell] << endl;
        cout << "sp_vol_src        "<< sp_vol_src[neg_cell] << endl;
        cout << "mass sp_vol_L     "<< sp_vol_L[neg_cell] << endl;
        cout << "mass sp_vol_L_old "
             << (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        cout << "-----------------------------------"<<endl;
//        ostringstream warn;
//        int L = level->getIndex();
//        warn<<"ERROR ICE:("<<L<<"):computeLagrangianSpecificVolumeRF, mat "<<indx
//            << " cell " <<neg_cell << " sp_vol_L is negative\n";
//        throw InvalidValue(warn.str(), __FILE__, __LINE__);
        new_dw->abortTimestep();
        new_dw->restartTimestep();
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
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing << d_myworld->myrank() << " Doing computeLagrangian_Transported_Vars on patch " 
               << patch->getID() << "\t ICE \tL-" <<level->getIndex()<< endl;
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
          // must use var Labelname not var_Lagrangian
          string Labelname = tvar->var->getName();
          setBC(q_L_CC, Labelname,  patch, d_sharedState, indx, new_dw);

          // multiply by mass so advection is conserved
          for(CellIterator iter=patch->getExtraCellIterator();
                          !iter.done();iter++){
            IntVector c = *iter;                            
            q_L_CC[c] *= mass_L[m][c];
          }
          
          //---- P R I N T   D A T A ------
          if (switchDebug_LagrangianTransportedVars ) {
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
    cout_doing << d_myworld->myrank() << " Doing doCCMomExchange on patch "<< patch->getID()
               <<"\t\t\t ICE \tL-" <<level->getIndex()<< endl;

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

    getConstantExchangeCoefficients( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);
      
      if(mpm_matl){                 // M P M
        CCVariable<double> oldTemp;
        new_dw->getCopy(oldTemp,          lb->temp_CCLabel,indx,patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  lb->vel_CCLabel, indx,patch);
        new_dw->getModifiable(Temp_CC[m], lb->temp_CCLabel,indx,patch);
        old_temp[m] = oldTemp;
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
    if (switchDebug_MomentumExchange_CC ) {
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
      if(d_exchCoeff->d_heatExchCoeffModel != "constant"){
        getVariableExchangeCoefficients( K, H, c, mass_L);
      }
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

  if(d_exchCoeff->convective()){
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
    int gm=d_exchCoeff->conv_fluid_matlindex();  // gas matl from which to get heat
    int sm=d_exchCoeff->conv_solid_matlindex();  // solid matl that heat goes to

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
/*===========TESTING==========`*/ 
 
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      preprocess_CustomBCs("CC_Exchange",old_dw, new_dw, lb, patch, indx, d_customBC_var_basket);
       
      setBC(vel_CC[m], "Velocity",   patch, d_sharedState, indx, new_dw,
                                                        d_customBC_var_basket);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_sharedState, 
                                         indx, new_dw,  d_customBC_var_basket);
#if SET_CFI_BC                                         
//      set_CFI_BC<Vector>(vel_CC[m],  patch);
//      set_CFI_BC<double>(Temp_CC[m], patch);
#endif
      delete_CustomBCs(d_customBC_var_basket);
    }
    
    
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
    if (switchDebug_MomentumExchange_CC ) {
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
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << d_myworld->myrank() << " Doing maxMach_on_Lodi_BC_Faces " <<
      patch->getID() << "\t\t\t ICE \tL-" <<level->getIndex()<< endl;
      
    Ghost::GhostType  gn = Ghost::None;
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<Vector> > vel_CC(numAllMatls);
    StaticArray<constCCVariable<double> > speedSound(numAllMatls);
          
    for(int m=0;m < numAllMatls;m++){
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
        
        for(int m=0; m < numAllMatls;m++){
          Material* matl = d_sharedState->getMaterial( m );
          int indx = matl->getDWIndex();
          ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
          if(ice_matl) {
            Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
            
            for(CellIterator iter=patch->getFaceIterator(face, MEC);
                                                          !iter.done();iter++) {
              IntVector c = *iter;
              maxMach = Max(maxMach,vel_CC[m][c].length()/speedSound[m][c]);
            }
            
            VarLabel* V_Label = getMaxMach_face_VarLabel(face);
            new_dw->put(max_vartype(maxMach), V_Label, level, indx);
          }  // icematl
        }  // matl loop
        
      }  // is lodi Face
    }  // boundaryFaces
  }  // patches
}

 
/* _____________________________________________________________________ 
 Function~  ICE::advectAndAdvanceInTime--
 Purpose~
   This task calculates the The cell-centered, time n+1, mass, momentum
   internal energy, sp_vol
 _____________________________________________________________________  */
void ICE::advectAndAdvanceInTime(const ProcessorGroup* /*pg*/,
                                 const PatchSubset* patches,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  // the advection calculations care about the position of the old dw subcycle
  double AMR_subCycleProgressVar = getSubCycleProgress(old_dw);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
 
    cout_doing << d_myworld->myrank() << " Doing Advect and Advance in Time on patch " 
               << patch->getID() << "\t\t ICE \tL-" <<L_indx
               << " progressVar " << AMR_subCycleProgressVar << endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);

    bool newGrid = d_sharedState->isRegridTimestep();
    Advector* advector = d_advector->clone(new_dw,patch,newGrid );


    CCVariable<double>  q_advected;
    CCVariable<Vector>  qV_advected; 
    new_dw->allocateTemporary(q_advected,   patch);
    new_dw->allocateTemporary(qV_advected,  patch);

    int numMatls = d_sharedState->getNumICEMatls();

    for (int m = 0; m < numMatls; m++ ) {
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex(); 

      CCVariable<double> mass_adv, int_eng_adv, sp_vol_adv;
      CCVariable<Vector> mom_adv;
      constCCVariable<double> int_eng_L_ME, mass_L,sp_vol_L;
      constCCVariable<Vector> mom_L_ME;
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(uvel_FC,     lb->uvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(vvel_FC,     lb->vvel_FCMELabel,        indx,patch,gac,2);  
      new_dw->get(wvel_FC,     lb->wvel_FCMELabel,        indx,patch,gac,2);  

      new_dw->get(mass_L,      lb->mass_L_CCLabel,        indx,patch,gac,2);
      new_dw->get(mom_L_ME,    lb->mom_L_ME_CCLabel,      indx,patch,gac,2);
      new_dw->get(sp_vol_L,    lb->sp_vol_L_CCLabel,      indx,patch,gac,2);
      new_dw->get(int_eng_L_ME,lb->eng_L_ME_CCLabel,      indx,patch,gac,2);

      new_dw->allocateAndPut(mass_adv,    lb->mass_advLabel,   indx,patch);          
      new_dw->allocateAndPut(mom_adv,     lb->mom_advLabel,    indx,patch);
      new_dw->allocateAndPut(int_eng_adv, lb->eng_advLabel,    indx,patch); 
      new_dw->allocateAndPut(sp_vol_adv,  lb->sp_vol_advLabel, indx,patch); 

      mass_adv.initialize(0.0);
      mom_adv.initialize(Vector(0.0,0.0,0.0));
      int_eng_adv.initialize(0.0);
      sp_vol_adv.initialize(0.0);
      q_advected.initialize(0.0);  
      qV_advected.initialize(Vector(0.0,0.0,0.0)); 
      
      //__________________________________
      // common variables that get passed into the advection operators
      advectVarBasket* varBasket = scinew advectVarBasket();
      varBasket->new_dw = new_dw;
      varBasket->old_dw = old_dw;
      varBasket->indx = indx;
      varBasket->patch = patch;
      varBasket->level = level;
      varBasket->lb  = lb;
      varBasket->doRefluxing = d_doRefluxing;
      varBasket->useCompatibleFluxes = d_useCompatibleFluxes;
      varBasket->AMR_subCycleProgressVar = AMR_subCycleProgressVar;

      //__________________________________
      //   Advection preprocessing
      bool bulletProof_test=true;
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                                    bulletProof_test, new_dw); 
      //__________________________________
      // mass
      advector->advectMass(mass_L, q_advected,  varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        mass_adv[c]  = (mass_L[c] + q_advected[c]);
      }   
      //__________________________________
      // momentum
      varBasket->is_Q_massSpecific   = true;
      varBasket->desc = "mom";
      advector->advectQ(mom_L_ME,mass_L,qV_advected, varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        mom_adv[c] = (mom_L_ME[c] + qV_advected[c]) ;
      }
      //__________________________________
      // internal energy
      varBasket->is_Q_massSpecific = true;
      varBasket->desc = "int_eng";
      advector->advectQ(int_eng_L_ME, mass_L, q_advected, varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        int_eng_adv[c] = (int_eng_L_ME[c] + q_advected[c]) ;
      }            
      //__________________________________
      // sp_vol[m] * mass
      varBasket->is_Q_massSpecific = true;
      varBasket->desc = "sp_vol";
      advector->advectQ(sp_vol_L,mass_L, q_advected, varBasket); 

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        sp_vol_adv[c] = (sp_vol_L[c] + q_advected[c]) ;
      }
      //__________________________________
      // Advect model variables 
      if(d_models.size() > 0 && d_modelSetup->tvars.size() > 0){
        vector<TransportedVariable*>::iterator t_iter;
        for( t_iter  = d_modelSetup->tvars.begin();
             t_iter != d_modelSetup->tvars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;
          
          if(tvar->matls->contains(indx)){
            string Labelname = tvar->var->getName();
            CCVariable<double> q_adv;
            constCCVariable<double> q_L_CC;
            new_dw->allocateAndPut(q_adv, tvar->var_adv,     indx, patch);
            new_dw->get(q_L_CC,   tvar->var_Lagrangian, indx, patch, gac, 2); 
            q_adv.initialize(d_EVIL_NUM);
            
            varBasket->desc = Labelname;
            varBasket->is_Q_massSpecific = true;
            advector->advectQ(q_L_CC,mass_L,q_advected, varBasket);  
   
            for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
              IntVector c = *iter;
              q_adv[c] = (q_L_CC[c] + q_advected[c]) ;
            }                   
            
            //---- P R I N T   D A T A ------   
            if (switchDebug_advance_advect ) {
              ostringstream desc;
              desc <<"BOT_Advection_after_BC_Mat_" <<indx<<"_patch_"
                   <<patch->getID();
              string Lag_labelName = tvar->var_Lagrangian->getName();
              printData(indx, patch,1, desc.str(), Lag_labelName, q_L_CC);
              printData(indx, patch,1, desc.str(), Labelname,     q_adv);
            }    
          }
        }
      } 
      
      //---- P R I N T   D A T A ------   
      if (switchDebug_advance_advect ) {
       ostringstream desc;
       desc <<"BOT_Advection_Mat_" <<indx<<"_patch_"<<patch->getID();
       printData(   indx, patch,1, desc.str(), "mass_L",      mass_L); 
       printVector( indx, patch,1, desc.str(), "mom_L_CC", 0, mom_L_ME); 
       printData(   indx, patch,1, desc.str(), "sp_vol_L",    sp_vol_L);
       printData(   indx, patch,1, desc.str(), "int_eng_L_CC",int_eng_L_ME);
       
       printData(   indx, patch,1, desc.str(), "mass_adv",     mass_adv); 
       printVector( indx, patch,1, desc.str(), "mom_adv", 0,   mom_adv); 
       printData(   indx, patch,1, desc.str(), "sp_vol_adv",   sp_vol_adv);
       printData(   indx, patch,1, desc.str(), "int_eng_adv",  int_eng_adv);
      }
 
      delete varBasket;
    }  // ice_matls loop
    delete advector;
  }  // patch loop
}
/* _____________________________________________________________________ 
 Function~  ICE::conservedtoPrimitive_Vars
 Purpose~ This task computes the primitive variables (rho,T,vel,sp_vol,...)
          at time n+1, from the conserved variables mass, momentum, energy...
 _____________________________________________________________________  */
void ICE::conservedtoPrimitive_Vars(const ProcessorGroup* /*pg*/,
                                    const PatchSubset* patches,
                                    const MaterialSubset* /*matls*/,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
 
    cout_doing << d_myworld->myrank() << " Doing conservedtoPrimitive_Vars patch " 
               << patch->getID() << "\t\t ICE \tL-" <<L_indx<< endl;

    Vector dx = patch->dCell();
    double invvol = 1.0/(dx.x()*dx.y()*dx.z());
    Ghost::GhostType  gn  = Ghost::None;
    int numMatls = d_sharedState->getNumICEMatls();

    for (int m = 0; m < numMatls; m++ ) {
      Material* matl = d_sharedState->getICEMaterial( m );
      int indx = matl->getDWIndex();
      
      CCVariable<double> rho_CC, temp_CC, sp_vol_CC,mach;
      CCVariable<Vector> vel_CC;
      constCCVariable<double> int_eng_adv, mass_adv,sp_vol_adv,speedSound, cv;
      constCCVariable<double> gamma, placeHolder, vol_frac;
      constCCVariable<Vector> mom_adv;

      new_dw->get(gamma,       lb->gammaLabel,         indx,patch,gn,0);
      new_dw->get(speedSound,  lb->speedSound_CCLabel, indx,patch,gn,0);
      new_dw->get(vol_frac,    lb->vol_frac_CCLabel,   indx,patch,gn,0);
      new_dw->get(cv,          lb->specific_heatLabel, indx,patch,gn,0);

      new_dw->get(mass_adv,    lb->mass_advLabel,      indx,patch,gn,0); 
      new_dw->get(mom_adv,     lb->mom_advLabel,       indx,patch,gn,0); 
      new_dw->get(sp_vol_adv,  lb->sp_vol_advLabel,    indx,patch,gn,0); 
      new_dw->get(int_eng_adv, lb->eng_advLabel,       indx,patch,gn,0); 
      
      new_dw->getModifiable(sp_vol_CC, lb->sp_vol_CCLabel,indx,patch);
      new_dw->getModifiable(rho_CC,    lb->rho_CCLabel,   indx,patch);

      new_dw->allocateAndPut(temp_CC,lb->temp_CCLabel,  indx,patch);          
      new_dw->allocateAndPut(vel_CC, lb->vel_CCLabel,   indx,patch);
      new_dw->allocateAndPut(mach,   lb->machLabel,     indx,patch);  

      rho_CC.initialize(-d_EVIL_NUM);
      temp_CC.initialize(-d_EVIL_NUM);
      vel_CC.initialize(Vector(0.0,0.0,0.0)); 

      //__________________________________
      // Backout primitive quantities from 
      // the conserved ones.
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        double inv_mass_adv = 1.0/mass_adv[c];
        rho_CC[c]    = mass_adv[c] * invvol;
        vel_CC[c]    = mom_adv[c]    * inv_mass_adv;
        sp_vol_CC[c] = sp_vol_adv[c] * inv_mass_adv;
      }

      //__________________________________
      // model variables 
      if(d_models.size() > 0 && d_modelSetup->tvars.size() > 0){
        vector<TransportedVariable*>::iterator t_iter;
        for( t_iter  = d_modelSetup->tvars.begin();
             t_iter != d_modelSetup->tvars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;
          
          if(tvar->matls->contains(indx)){
            string Labelname = tvar->var->getName();
            CCVariable<double> q_CC;
            constCCVariable<double> q_adv;
            new_dw->allocateAndPut(q_CC, tvar->var,     indx, patch);
            new_dw->get(q_adv,           tvar->var_adv, indx, patch, gn,0);
            q_CC.initialize(0.0);
            
            for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
              IntVector c = *iter;
              q_CC[c] = q_adv[c]/mass_adv[c];
            }
                  
            //  Set Boundary Conditions
            setBC(q_CC, Labelname,  patch, d_sharedState, indx, new_dw);  
            
            //---- P R I N T   D A T A ------   
            if (switchDebug_conserved_primitive ) {
              ostringstream desc;
              desc <<"BOT_convertConservedtoPrimitive_Vars_Mat_" <<indx<<"_patch_"
                   <<patch->getID();
              string adv_LabelName = tvar->var_adv->getName();
              printData(indx, patch,1, desc.str(), adv_LabelName, q_adv);
              printData(indx, patch,1, desc.str(), Labelname,     q_CC);
            }    
          }
        }
      } 

      //__________________________________
      // A model *can* compute the specific heat
      CCVariable<double> cv_new;
      new_dw->allocateTemporary(cv_new, patch,gn,0);
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
      // Backout primitive quantities from 
      // the conserved ones.
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        temp_CC[c] = int_eng_adv[c]/ (mass_adv[c]*cv_new[c]);
      }
      
      //__________________________________
      // set the boundary conditions
      preprocess_CustomBCs("Advection",old_dw, new_dw, lb,  patch, indx,
                           d_customBC_var_basket);
       
      setBC(rho_CC, "Density",  placeHolder, placeHolder,
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);
      setBC(vel_CC, "Velocity", 
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);       
      setBC(temp_CC,"Temperature",gamma, cv,
            patch,d_sharedState, indx, new_dw, d_customBC_var_basket);
            
      setSpecificVolBC(sp_vol_CC, "SpecificVol", false,rho_CC,vol_frac,
                       patch,d_sharedState, indx);     
      delete_CustomBCs(d_customBC_var_basket);
                               
      //__________________________________
      // Compute Auxilary quantities
      for(CellIterator iter = patch->getExtraCellIterator();
                                                        !iter.done(); iter++){
        IntVector c = *iter;
        mach[c]  = vel_CC[c].length()/speedSound[c];
      }
      //---- P R I N T   D A T A ------   
      if (switchDebug_conserved_primitive ) {
       ostringstream desc;
       desc <<"BOT_conservedtoPrimitive_Vars_Mat_" <<indx<<"_patch_"<<patch->getID();
       printData(   indx, patch,1, desc.str(), "mass_adv",    mass_adv);
       printVector( indx, patch,1, desc.str(), "mom_adv", 0,  mom_adv); 
       printData(   indx, patch,1, desc.str(), "sp_vol_adv",  sp_vol_adv);
       printData(   indx, patch,1, desc.str(), "int_eng_adv", int_eng_adv);
       printData(   indx, patch,1, desc.str(), "rho_CC",      rho_CC);
       printData(   indx, patch,1, desc.str(), "temp_CC",     temp_CC);
       printData(   indx, patch,1, desc.str(), "sp_vol_CC",   sp_vol_CC);
       printVector( indx, patch,1, desc.str(), "vel_CC", 0,   vel_CC);
      }
      //____ B U L L E T   P R O O F I N G----
      // ignore BP if timestep restart has already been requested
      IntVector neg_cell;
      bool tsr = new_dw->timestepRestarted();
      
      ostringstream base, warn;
      base <<"ERROR ICE:(L-"<<L_indx<<"):conservedtoPrimitive_Vars, mat "<< indx <<" cell ";
      if (!areAllValuesPositive(rho_CC, neg_cell) && !tsr) {
        warn << base.str() << neg_cell << " negative rho_CC\n ";
        throw InvalidValue(warn.str(), __FILE__, __LINE__);
      }
      if (!areAllValuesPositive(temp_CC, neg_cell) && !tsr) {
        warn << base.str() << neg_cell << " negative temp_CC\n ";
        throw InvalidValue(warn.str(), __FILE__, __LINE__);
      }
      if (!areAllValuesPositive(sp_vol_CC, neg_cell) && !tsr) {
       warn << base.str() << neg_cell << " negative sp_vol_CC\n ";        
       throw InvalidValue(warn.str(), __FILE__, __LINE__);
      } 
    }  // ice_matls loop
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
     
  double total_mass     = 0.0;      
  double total_KE       = 0.0;      
  double total_int_eng  = 0.0;      
  Vector total_mom(0.0, 0.0, 0.0);  
  Vector mom_exch_error(0,0,0);     
  double eng_exch_error = 0;        
          
  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);
    
    cout_doing << d_myworld->myrank() << " Doing TestConservation on patch " 
               << patch->getID() << "\t\t\t ICE \tL-"<<level->getIndex()<< endl;      
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
    
    //__________________________________
    // conservation of mass
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
    }
    
    if(d_conservationTest->mass){    
      for (int m = 0; m < numICEmatls; m++ ) {
        double mat_mass = 0;
        conservationTest<double>(patch, delT, mass[m], 
                                 uvel_FC[m], vvel_FC[m], wvel_FC[m],mat_mass);
        total_mass += mat_mass; 
      }
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
  }  // patch loop
  if(d_conservationTest->mass){
    new_dw->put(sum_vartype(total_mass),        lb->TotalMassLabel);
  }
  if(d_conservationTest->exchange){
    new_dw->put(sumvec_vartype(mom_exch_error), lb->mom_exch_errorLabel);  
    new_dw->put(sum_vartype(eng_exch_error),    lb->eng_exch_errorLabel);  
  }
  if(d_conservationTest->energy){
    new_dw->put(sum_vartype(total_KE),          lb->KineticEnergyLabel);  
    new_dw->put(sum_vartype(total_int_eng),     lb->TotalIntEngLabel);    
  }
  if(d_conservationTest->momentum){
    new_dw->put(sumvec_vartype(total_mom),      lb->TotalMomentumLabel);
  }
}

/*_____________________________________________________________________
 Function:  hydrostaticPressureAdjustment--
 Notes:     press_hydro = rho_micro_CC[SURROUNDING_MAT] * grav * some_distance
_______________________________________________________________________ */
void ICE::hydrostaticPressureAdjustment(const Patch* patch,
                                const CCVariable<double>& rho_micro_CC,
                                CCVariable<double>& press_CC)
{
  Vector gravity = getGravity();
  // find the upper and lower point of the domain.
  const Level* level = patch->getLevel();
  GridP grid = level->getGrid();
  BBox b;
  grid->getSpatialRange(b);
  Vector gridMin = b.min().asVector();
  Vector dx_L0 = grid->getLevel(0)->dCell();
  
  // Pressure reference point is assumed to be 
  //at CELL-CENTER of cell 0,0,0 
  Vector press_ref_pt = gridMin + 1.5*dx_L0;
  
  // Which direction is the gravitational vector pointing
  int dir = -9;
  for (int i = 0; i < 3; i++){
    if ( gravity[i] != 0.0){
      dir = i;
    }
  }
  //__________________________________
  // Tack on the hydrostatic pressure adjustment
  for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    IntVector c = *iter;
    Point here = level->getCellPosition(c);
    Vector dist_from_p_ref = (here.asVector() - press_ref_pt);

    double press_hydro = rho_micro_CC[c] * gravity[dir] * dist_from_p_ref[dir];
    press_CC[c] += press_hydro;
  }
}

/*_____________________________________________________________________
 Function~  ICE::getConstantExchangeCoefficients--
 This routine returns the constant exchange coefficients
 _____________________________________________________________________  */
void ICE::getConstantExchangeCoefficients( FastMatrix& K, FastMatrix& H  )
{
  int numMatls  = d_sharedState->getNumMatls();

  // The vector of exchange coefficients only contains the upper triagonal
  // matrix

  // Check if the # of coefficients = # of upper triangular terms needed
  int num_coeff = ((numMatls)*(numMatls) - numMatls)/2;

  vector<double> d_K_mom = d_exchCoeff->K_mom();
  vector<double> d_K_heat = d_exchCoeff->K_heat();
  vector<double>::iterator it_m=d_K_mom.begin();
  vector<double>::iterator it_h=d_K_heat.begin();

  //__________________________________
  // bulletproofing
  bool test = false;
  string desc;
  if (num_coeff != (int)d_K_mom.size()) {
    test = true;
    desc = "momentum";
  }  
  
  if (num_coeff !=(int)d_K_heat.size() && d_exchCoeff->d_heatExchCoeffModel == "constant") {
    test = true;
    desc = desc + " energy";
  }

  if(test) {   
    ostringstream warn;
    warn << "\nThe number of exchange coefficients (" << desc << ") is incorrect.\n";
    warn << "Here is the correct specification:\n";
    for (int i = 0; i < numMatls; i++ ){
      for (int j = i+1; j < numMatls; j++){
        warn << i << "->" << j << ",\t"; 
      }
      warn << "\n";
      for (int k = 0; k <= i; k++ ){
        warn << "\t";
      }
    } 
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  // Fill in the upper triangular matrix
  // momentum
  for (int i = 0; i < numMatls; i++ )  {
    K(i,i) = 0.0;
    for (int j = i + 1; j < numMatls; j++) {
      K(i,j) = K(j,i) = *it_m++;
    }
  }
  
  // heat
  if(d_exchCoeff->d_heatExchCoeffModel == "constant") {
    for (int i = 0; i < numMatls; i++ )  {
      H(i,i) = 0.0;
      for (int j = i + 1; j < numMatls; j++) {
        H(i,j) = H(j,i) = *it_h++;
      }
    }
  }
  
  
}

/*_____________________________________________________________________
 Function~  ICE::getVariableExchangeCoefficients--
 This routine returns the  exchange coefficients
 _____________________________________________________________________  */
void ICE::getVariableExchangeCoefficients( FastMatrix& ,
                                           FastMatrix& H,
                                           IntVector & c,
                                           StaticArray<constCCVariable<double> >& mass_L  )
{
  int numMatls  = d_sharedState->getNumMatls();

  //__________________________________
  // Momentum  (do nothing for now)
  
  //__________________________________
  // Heat coefficient
  for (int m = 0; m < numMatls; m++ )  {
    H(m,m) = 0.0;
    for (int n = m + 1; n < numMatls; n++) {    
      double massRatioSqr = pow(mass_L[n][c]/mass_L[m][c], 2.0);  

      // 1e5  is the lower limit clamp
      // 1e12 is the upper limit clamp
      if (massRatioSqr < 1e-12){
        H(n,m) = H(m,n) = 1e12;
      }
      else if (massRatioSqr >= 1e-12 && massRatioSqr < 1e-5){
        H(n,m) = H(m,n) = 1./massRatioSqr;
      }
      else if (massRatioSqr >= 1e-5 && massRatioSqr < 1e5){
        H(n,m) = H(m,n) = 1e5;
      }
      else if (massRatioSqr >= 1e5 && massRatioSqr < 1e12){
        H(n,m) = H(m,n) = massRatioSqr;
      }
      else if (massRatioSqr >= 1e12){
        H(n,m) = H(m,n) = 1e12;
      }

    }
  }
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

//______________________________________________________________________
//  Models
ICE::ICEModelSetup::ICEModelSetup()
{
}

ICE::ICEModelSetup::~ICEModelSetup()
{
}

void ICE::ICEModelSetup::registerTransportedVariable(const MaterialSet* matlSet,
                                                     const VarLabel* var,
                                                     const VarLabel* src)
{
  TransportedVariable* t = scinew TransportedVariable;
  t->matlSet = matlSet;
  t->matls   = matlSet->getSubset(0);
  t->var = var;
  t->src = src;
  t->var_Lagrangian = VarLabel::create(var->getName()+"_L", var->typeDescription());
  t->var_adv        = VarLabel::create(var->getName()+"_adv", var->typeDescription());
  tvars.push_back(t);
}

//__________________________________
//  Register scalar flux variables needed
//  by the AMR refluxing task.  We're actually
//  creating the varLabels and putting them is a vector
void ICE::ICEModelSetup::registerAMR_RefluxVariable(const MaterialSet* matlSet,
                                                          const VarLabel* var)
{
  AMR_refluxVariable* t = scinew AMR_refluxVariable;
  t->matlSet = matlSet;
  t->matls   = matlSet->getSubset(0);
  string var_adv_name = var->getName() + "_adv";
  t->var_adv = VarLabel::find(var_adv_name);  //Advected conserved quantity
  if(t->var_adv==NULL){
    throw ProblemSetupException("The refluxing variable name("+var_adv_name +") could not be found",
                                   __FILE__, __LINE__);
  }
  
  t->var = var;
  
  t->var_X_FC_flux = VarLabel::create(var->getName()+"_X_FC_flux", 
                                SFCXVariable<double>::getTypeDescription());
  t->var_Y_FC_flux = VarLabel::create(var->getName()+"_Y_FC_flux", 
                                SFCYVariable<double>::getTypeDescription());
  t->var_Z_FC_flux = VarLabel::create(var->getName()+"_Z_FC_flux", 
                                SFCZVariable<double>::getTypeDescription());
                                
  t->var_X_FC_corr = VarLabel::create(var->getName()+"_X_FC_corr", 
                                SFCXVariable<double>::getTypeDescription());
  t->var_Y_FC_corr = VarLabel::create(var->getName()+"_Y_FC_corr", 
                                SFCYVariable<double>::getTypeDescription());
  t->var_Z_FC_corr = VarLabel::create(var->getName()+"_Z_FC_corr", 
                                SFCZVariable<double>::getTypeDescription());
  d_reflux_vars.push_back(t);
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
  throw InternalError("trying to do AMR iwth the non-AMR component!", __FILE__, __LINE__);
}
void
ICE::refineBoundaries(const Patch*, CCVariable<Vector>&,
                            DataWarehouse*, const VarLabel*,
                            int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!", __FILE__, __LINE__);
}
void
ICE::refineBoundaries(const Patch*, SFCXVariable<double>&,
                            DataWarehouse*, const VarLabel*,
                            int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!", __FILE__, __LINE__);
}

void
ICE::refineBoundaries(const Patch*, SFCYVariable<double>&,
                            DataWarehouse*, const VarLabel*,
                            int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!", __FILE__, __LINE__);
}

void
ICE::refineBoundaries(const Patch*, SFCZVariable<double>&,
                            DataWarehouse*, const VarLabel*,
                            int, double)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!", __FILE__, __LINE__);
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
    cout_doing << d_myworld->myrank() << " ICE::scheduleCheckNeedAddMaterial" << endl;
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
    cout_doing << d_myworld->myrank() << " ICE::scheduleSetNeedAddMaterialFlag" << endl;
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

