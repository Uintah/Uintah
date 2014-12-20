/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

//----- ExplicitSolver.cc ----------------------------------------------
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Components/Arches/EfficiencyCalculator.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/CoalModels/PartVel.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/CQMOM.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>

//NEW TASK STUFF
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Task/SampleTask.h>
#include <CCA/Components/Arches/Task/TemplatedSampleTask.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
//END NEW TASK STUFF

#include <CCA/Components/Arches/ExplicitSolver.h>
#include <Core/Containers/StaticArray.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/WallHTModels/WallModelDriver.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/MiscMath.h>
#include <CCA/Components/Arches/Filter.h>

#ifdef WASATCH_IN_ARCHES
#  include <expression/ExprLib.h>

#  include <CCA/Components/Wasatch/Wasatch.h>
#  include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>
#  include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#  include <CCA/Components/Wasatch/Transport/ParseEquation.h>
#  include <CCA/Components/Wasatch/GraphHelperTools.h>
#  include <CCA/Components/Wasatch/TaskInterface.h>
#  include <CCA/Components/Wasatch/TagNames.h>
#endif // WASATCH_IN_ARCHES


#include <cmath>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

// ****************************************************************************
// Default constructor for ExplicitSolver
// ****************************************************************************
ExplicitSolver::
ExplicitSolver(ArchesLabel* label,
               const MPMArchesLabel* MAlb,
               Properties* props,
               BoundaryCondition* bc,
               TurbulenceModel* turbModel,
               ScaleSimilarityModel* scaleSimilarityModel,
               PhysicalConstants* physConst,
               RadPropertyCalculator* rad_properties, 
               std::map<std::string, boost::shared_ptr<TaskFactoryBase> >& boost_fac_map, 
               const ProcessorGroup* myworld,
               SolverInterface* hypreSolver):
               NonlinearSolver(myworld),
               d_lab(label), d_MAlab(MAlb), d_props(props),
               d_boundaryCondition(bc), d_turbModel(turbModel),
               d_scaleSimilarityModel(scaleSimilarityModel),
               d_physicalConsts(physConst),
               d_hypreSolver(hypreSolver), 
               d_rad_prop_calc(rad_properties),
               _boost_fac_map(boost_fac_map)
{
  d_pressSolver = 0;
  d_momSolver = 0;
  nosolve_timelabels_allocated = false;
  d_printTotalKE = false; 
  d_wall_ht_models = 0; 
}

// ****************************************************************************
// Destructor
// ****************************************************************************
ExplicitSolver::~ExplicitSolver()
{
  delete d_pressSolver;
  delete d_momSolver;
  delete d_eff_calculator; 
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
    delete d_timeIntegratorLabels[curr_level];
  if (nosolve_timelabels_allocated)
    delete nosolve_timelabels;
  if ( d_wall_ht_models != 0 ){ 
    delete d_wall_ht_models; 
  }
}

// ****************************************************************************
// Problem Setup
// ****************************************************************************
void
ExplicitSolver::problemSetup( const ProblemSpecP & params, SimulationStateP & state )
{

  ProblemSpecP db = params->findBlock("ExplicitSolver");
  ProblemSpecP db_parent = params; 

  if ( db->findBlock( "print_total_ke" ) ){ 
    d_printTotalKE = true; 
  }

  db->getWithDefault( "max_ke_allowed", d_ke_limit, 1.0e99 );

  if ( db_parent->findBlock("BoundaryConditions") ){ 
    if ( db_parent->findBlock("BoundaryConditions")->findBlock( "WallHT" ) ){
      ProblemSpecP db_wall_ht = db_parent->findBlock("BoundaryConditions")->findBlock( "WallHT" );  
      d_wall_ht_models = scinew WallModelDriver( d_lab->d_sharedState ); 
      d_wall_ht_models->problemSetup( db_wall_ht ); 
    }
  } 


  d_pressSolver = scinew PressureSolver( d_lab, d_MAlab,
                                         d_boundaryCondition,
                                         d_physicalConsts, d_myworld,
                                         d_hypreSolver );
  d_pressSolver->problemSetup( db, state );

  d_momSolver = scinew MomentumSolver(d_lab, d_MAlab,
                                        d_turbModel, d_boundaryCondition,
                                        d_physicalConsts);
  
  d_momSolver->set_use_wasatch_mom_rhs(this->get_use_wasatch_mom_rhs());
  
  d_momSolver->problemSetup(db);

  const ProblemSpecP params_root = db->getRootNode();
  std::string t_order; 
  ProblemSpecP db_time_int = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator");
  db_time_int->findBlock("ExplicitIntegrator")->getAttribute("order", t_order);


  ProblemSpecP db_vars  = params_root->findBlock("DataArchiver");
  for (ProblemSpecP db_dv = db_vars->findBlock("save"); 
        db_dv !=0; db_dv = db_dv->findNextBlock("save")){

    std::string var_name; 
    db_dv->getAttribute( "label", var_name );

    if ( var_name == "kineticEnergy" || var_name == "totalKineticEnergy" ){
      d_printTotalKE = true;
    }
  }

  //translate order to the older code: 
  if ( t_order == "first" ){ 
    d_timeIntegratorType = "FE"; 
  } else if ( t_order == "second" ){ 
    d_timeIntegratorType = "RK2SSP"; 
  } else if ( t_order == "third" ) {
    d_timeIntegratorType = "RK3SSP"; 
  } else { 
    throw InvalidValue("Error: <ExplicitIntegrator> order attribute must be one of: first, second, third!  Please fix input file.",__FILE__, __LINE__);             
  } 

  if (d_timeIntegratorType == "FE") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::FE));
    numTimeIntegratorLevels = 1;
  }
  else if (d_timeIntegratorType == "RK2") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::OldPredictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::OldCorrector));
    numTimeIntegratorLevels = 2;
  }
  else if (d_timeIntegratorType == "RK2SSP") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Predictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Corrector));
    numTimeIntegratorLevels = 2;
  }
  else if (d_timeIntegratorType == "RK3SSP") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Predictor));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::Intermediate));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::CorrectorRK3));
    numTimeIntegratorLevels = 3;
  }
  else if (d_timeIntegratorType == "BEEmulation") {
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation1));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation2));
    d_timeIntegratorLabels.push_back(scinew TimeIntegratorLabel(d_lab,
                                     TimeIntegratorStepType::BEEmulation3));
    numTimeIntegratorLevels = 3;
  }
  else {
    throw ProblemSetupException("Integrator type is not defined "+d_timeIntegratorType,
                                __FILE__, __LINE__);
  }

  db->getWithDefault("turbModelCalcFreq",d_turbModelCalcFreq,1);
  db->getWithDefault("turbModelCalcForAllRKSteps",d_turbModelRKsteps,true);

  db->getWithDefault("restartOnNegativeDensityGuess",
                     d_restart_on_negative_density_guess,false);
  db->getWithDefault("NoisyDensityGuess",
                     d_noisyDensityGuess, false);
  db->getWithDefault("kineticEnergy_fromFC",d_KE_fromFC,false);
  db->getWithDefault("maxDensityLag",d_maxDensityLag,0.0);

  d_extra_table_lookup = false; 
  if ( db->findBlock("extra_table_lookup")) d_extra_table_lookup = true; 

  d_props->setFilter(d_turbModel->getFilter());
  d_momSolver->setDiscretizationFilter(d_turbModel->getFilter());

  d_mixedModel=d_turbModel->getMixedModel();

  bool check_calculator; 
  d_eff_calculator = scinew EfficiencyCalculator( d_boundaryCondition, d_lab ); 
  check_calculator = d_eff_calculator->problemSetup( db ); 

  if ( !check_calculator ){ 
    proc0cout << "Notice: No efficiency calculators found." << endl;
  } 

}

void 
ExplicitSolver::checkMomBCs( SchedulerP& sched,
                             const LevelP& level, 
                             const MaterialSet* matls)
{

  d_boundaryCondition->sched_checkMomBCs( sched, level, matls ); 

}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int ExplicitSolver::nonlinearSolve(const LevelP& level,
                                   SchedulerP& sched
#                                  ifdef WASATCH_IN_ARCHES
                                   , Wasatch::Wasatch& wasatch, 
                                   ExplicitTimeInt* d_timeIntegrator,
                                   SimulationStateP& state
#                                  endif // WASATCH_IN_ARCHES
                                   )
{

  typedef std::vector<std::string> SVec; 
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);

  sched_setInitialGuess(sched, patches, matls);

  d_boundaryCondition->sched_setAreaFraction(sched, level, matls, 0, false );
  d_turbModel->sched_carryForwardFilterVol(sched, patches, matls); 

  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  if (dqmomFactory.get_quad_nodes() > 0)
    d_doDQMOM = true;
  else
    d_doDQMOM = false; // probably need to sync this better with the bool being set in Arches
  
  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  if (cqmomFactory.get_number_moments() > 0)
    d_doCQMOM = true;
  else
    d_doCQMOM = false;
  

  EqnFactory& eqn_factory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqn_factory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

    EqnBase* eqn = iter->second;
    eqn->sched_initializeVariables(level, sched); 

  }

  //copy the temperature into a radiation temperature variable: 
  d_boundaryCondition->sched_create_radiation_temperature( sched, level, matls, true );

  //========NEW STUFF =================================
  //TIMESTEP INIT: 
  //UtilityFactory
  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  BFM::iterator i_util = _boost_fac_map.find("utility_factory"); 
  TaskFactoryBase::TaskMap init_all_tasks = i_util->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = init_all_tasks.begin(); i != init_all_tasks.end(); i++){ 
    i->second->schedule_timestep_init(level, sched, matls); 
  }

  BFM::iterator i_transport = _boost_fac_map.find("transport_factory"); 
  SVec scalar_rhs_builders = i_transport->second->retrieve_task_subset("scalar_rhs_builders"); 
  for ( SVec::iterator i = scalar_rhs_builders.begin(); i != scalar_rhs_builders.end(); i++){ 
    TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
    tsk->schedule_timestep_init(level, sched, matls); 
  }
  SVec mom_rhs_builders = i_transport->second->retrieve_task_subset("mom_rhs_builders"); 
  for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){ 
    TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
    tsk->schedule_timestep_init(level, sched, matls); 
  }
  BFM::iterator i_property_models = _boost_fac_map.find("property_models"); 
  TaskFactoryBase::TaskMap all_property_models = i_property_models->second->retrieve_all_tasks(); 
  for ( TaskFactoryBase::TaskMap::iterator i = all_property_models.begin(); i != all_property_models.end(); i++){ 
    i->second->schedule_timestep_init(level, sched, matls); 
  }
  //===================END NEW STUFF=======================

  // --------> START RK LOOP <---------
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
  {

    //================ NEW TASK STUFF============================= 
   
    //Build RHS
    for ( TaskFactoryBase::TaskMap::iterator i = init_all_tasks.begin(); i != init_all_tasks.end(); i++){ 
      i->second->schedule_task(level, sched, matls, curr_level); 
    }

    //(scalars)
    for ( SVec::iterator i = scalar_rhs_builders.begin(); i != scalar_rhs_builders.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
      tsk->schedule_task(level, sched, matls, curr_level); 
    }
    
    //(momentum)
    for ( SVec::iterator i = mom_rhs_builders.begin(); i != mom_rhs_builders.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
      tsk->schedule_task(level, sched, matls, curr_level); 
    }

    //Particle Models
    BFM::iterator i_partmod_fac = _boost_fac_map.find("particle_model_factory"); 
    TaskFactoryBase::TaskMap all_particle_models = i_partmod_fac->second->retrieve_all_tasks(); 
    for ( TaskFactoryBase::TaskMap::iterator i = all_particle_models.begin(); 
          i != all_particle_models.end(); i++){ 
      i->second->schedule_task(level, sched, matls, curr_level); 
    }

    //Property Models
    for ( TaskFactoryBase::TaskMap::iterator i = all_property_models.begin(); 
          i != all_property_models.end(); i++ ){ 
      i->second->schedule_task(level, sched, matls, curr_level); 
    }

    //FE update
    SVec scalar_fe_up = i_transport->second->retrieve_task_subset("scalar_fe_update"); 
    SVec mom_fe_up = i_transport->second->retrieve_task_subset("mom_fe_update"); 

    for ( SVec::iterator i = scalar_fe_up.begin(); i != scalar_fe_up.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
      tsk->schedule_task(level, sched, matls, curr_level); 
    }
    for ( SVec::iterator i = mom_fe_up.begin(); i != mom_fe_up.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i); 
      tsk->schedule_task(level, sched, matls, curr_level); 
    }

    //============== END NEW TASK STUFF ==============================

    // Create this timestep labels for properties
    PropertyModelFactory& propFactory = PropertyModelFactory::self();
    PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
        iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;

      if ( curr_level == 0 )
        prop_model->sched_timeStepInit( level, sched ); 

    }
    if (d_doDQMOM) {

      CoalModelFactory& modelFactory = CoalModelFactory::self();
      DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
      DQMOMEqnFactory::EqnMap& weights_eqns = dqmomFactory.retrieve_weights_eqns();
      DQMOMEqnFactory::EqnMap& abscissas_eqns = dqmomFactory.retrieve_abscissas_eqns();

      // Compute the particle velocities at time t w^tu^t/w^t
      d_partVel->schedComputePartVel( level, sched, curr_level );

      // Evaluate DQMOM equations
      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
            iEqn != weights_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level );//compute rhs
      }

      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
            iEqn != abscissas_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level );//compute rhs
      }

      // schedule the models for evaluation
      modelFactory.sched_coalParticleCalculation( level, sched, curr_level );// compute drag, devol, char, etc models..

      // schedule DQMOM linear solve
      d_dqmomSolver->sched_solveLinearSystem( level, sched, curr_level );

      // Evaluate DQMOM equations
      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
            iEqn != weights_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );// add sources and solve equation
      }

      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
            iEqn != abscissas_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );// add sources and solve equation
      }


      // ---- schedule the solution of the transport equations ----


      for( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin();
           iEqn!=dqmom_eqns.end(); ++iEqn ) {

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        //also get the abscissa values
        dqmom_eqn->sched_getUnscaledValues( level, sched );
      }

      // Evaluate DQMOM equations
      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
            iEqn != weights_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );
      }

      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
            iEqn != abscissas_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_updateTransportEqn( level, sched, curr_level );// add sources and solve equation
      }


      // ---- schedule the solution of the transport equations ----


      for( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin();
           iEqn!=dqmom_eqns.end(); ++iEqn ) {

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        //also get the abscissa values
        dqmom_eqn->sched_getUnscaledValues( level, sched );
      }

      // calculate the moments
      bool saveMoments = d_dqmomSolver->getSaveMoments();
      if( saveMoments ) {
        // schedule DQMOM moment calculation
        d_dqmomSolver->sched_calculateMoments( level, sched, curr_level );
      }
    }
    
    if ( d_doCQMOM ) {
      bool doOperatorSplit;
      doOperatorSplit = d_cqmomSolver->getOperatorSplitting();
      CQMOMEqnFactory::EqnMap& moment_eqns = cqmomFactory.retrieve_all_eqns();
      
      if (!doOperatorSplit) {
      //Evaluate CQMOM equations
        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){
         
          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          cqmom_eqn->sched_evalTransportEqn( level, sched, curr_level );
          
        }
        //get new weights and absicissa
        d_cqmomSolver->sched_solveCQMOMInversion( level, sched, curr_level );
        d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
      } else {
        //if operator splitting is turned on use a different CQMOM permutation for each convection direction
        int uVelIndex = d_cqmomSolver->getUVelIndex();
        int vVelIndex = d_cqmomSolver->getVVelIndex();
        int wVelIndex = d_cqmomSolver->getWVelIndex();
        
        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){
          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          if (curr_level == 0)
            cqmom_eqn->sched_initializeVariables( level, sched );
          cqmom_eqn->sched_computeSources( level, sched, curr_level );
        }
        //x-direction - do the CQMOM inversion of this permutation, then do the convection
        if ( uVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion321( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){
          
            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildXConvection( level, sched, curr_level );
          }
        }
        //y-direction
        if ( vVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion312( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){
          
            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildYConvection( level, sched, curr_level );
          }
        }
        //z-direction
        if ( wVelIndex > -1 ) {
          d_cqmomSolver->sched_solveCQMOMInversion213( level, sched, curr_level );
          d_cqmomSolver->sched_momentCorrection( level, sched, curr_level );
          for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
               iEqn != moment_eqns.end(); iEqn++){
          
            CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
            cqmom_eqn->sched_buildZConvection( level, sched, curr_level );
          }
        }

        //combine all 3 fluxes and actually solve eqn with other sources
        for ( CQMOMEqnFactory::EqnMap::iterator iEqn = moment_eqns.begin();
             iEqn != moment_eqns.end(); iEqn++){
          
          CQMOMEqn* cqmom_eqn = dynamic_cast<CQMOMEqn*>(iEqn->second);
          cqmom_eqn->sched_buildSplitRHS( level, sched, curr_level );
          cqmom_eqn->sched_solveTransportEqn( level, sched, curr_level );
        }
      }
    }

    // STAGE 0 
    //d_rad_prop_calc->sched_compute_radiation_properties( level, sched, matls, curr_level, false ); 

    SourceTermFactory& src_factory = SourceTermFactory::self();

    src_factory.sched_computeSources( level, sched, curr_level, 0 ); 

    sched_saveTempCopies(sched, patches, matls,d_timeIntegratorLabels[curr_level]);

    sched_getDensityGuess(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);

    sched_checkDensityGuess(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

      EqnBase* eqn = iter->second;
      //these equations use a density guess
      if ( eqn->get_stage() == 0 )
        eqn->sched_evalTransportEqn( level, sched, curr_level );

    }

    // Property models needed before table lookup:
    PropertyModelBase* hl_model = 0;
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;

      if ( prop_model->getPropType() == "heat_loss" ){
        hl_model = prop_model; 
      } else {
        if ( prop_model->beforeTableLookUp() )
          prop_model->sched_computeProp( level, sched, curr_level );
      }

    }

    if ( hl_model != 0 )
      hl_model->sched_computeProp( level, sched, curr_level ); 


    //1st TABLE LOOKUP
    bool initialize_it  = false;
    bool modify_ref_den = false;
    if ( curr_level == 0 ) initialize_it = true;
    d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, curr_level );

    d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );

    //==================NEW TASK STUFF =========================
    //this is updating the scalar with the latest density from the table lookup
    SVec scalar_ssp = i_transport->second->retrieve_task_subset("scalar_ssp"); 
    SVec mom_ssp = i_transport->second->retrieve_task_subset("mom_ssp"); 

    for ( SVec::iterator i = scalar_ssp.begin(); i != scalar_ssp.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      if ( curr_level > 0 )
        tsk->schedule_task( level, sched, matls, curr_level ); 
    }

    for ( SVec::iterator i = mom_ssp.begin(); i != mom_ssp.end(); i++){ 
      TaskInterface* tsk = i_transport->second->retrieve_task(*i);
      //if ( curr_level > 0 )
      //  tsk->schedule_task( level, sched, matls, curr_level ); 
    }
    //============= END NEW TASK STUFF===============================


    // STAGE 1

    // Property models needed after table lookup:
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;
      if ( !prop_model->beforeTableLookUp() )
        prop_model->sched_computeProp( level, sched, curr_level );

    }

    // Source terms needed after table lookup: 
    src_factory.sched_computeSources( level, sched, curr_level, 1 ); 

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      //these equations do not use a density guess
      if ( eqn->get_stage() == 1 )
        eqn->sched_evalTransportEqn( level, sched, curr_level );
    }

    sched_computeDensityLag(sched, patches, matls,
                           d_timeIntegratorLabels[curr_level],
                           false);

    if (d_maxDensityLag > 0.0)
      sched_checkDensityLag(sched, patches, matls,
                            d_timeIntegratorLabels[curr_level],
                            false);

    // linearizes and solves pressure eqn
    // first computes, hatted velocities and then computes
    // the pressure poisson equation
#ifdef WASATCH_IN_ARCHES
    {
      //____________________________________________________________________________
      // check if wasatch momentum equations were specified
      if ( wasatch.get_wasatch_spec()->findBlock("MomentumEquations") && this->get_use_wasatch_mom_rhs() ) {
        // if momentum equations were specified in Wasatch, then we only care about the partial RHS. Hence, we need to
        // modify the root IDs a bit and generate a new set of rootIDs with wchich we can construct the required taskinterface
        // get the ID of the momentum RHS
        Wasatch::GraphHelper* const gh = wasatch.graph_categories()[Wasatch::ADVANCE_SOLUTION];
        
        gh->rootIDs.erase(gh->exprFactory->get_id(Expr::Tag(d_lab->d_uMomLabel->getName()+ "_rhs",Expr::STATE_NONE) ) );
        gh->rootIDs.erase(gh->exprFactory->get_id(Expr::Tag(d_lab->d_vMomLabel->getName()+ "_rhs",Expr::STATE_NONE) ) );
        gh->rootIDs.erase(gh->exprFactory->get_id(Expr::Tag(d_lab->d_wMomLabel->getName()+ "_rhs",Expr::STATE_NONE) ) );
                
        // manually insert the root ids for the wasatch momentum partial rhs.
        // the wasatch rhs_partial expressions are used to construct the provisional arches
        // velocity fields, i.e. hat(rho u) = (rho u)_n + dt * rhs_partial
        std::set< Expr::ExpressionID > momRootIDs;
        momRootIDs.insert(gh->exprFactory->get_id(Expr::Tag(d_lab->d_uVelRhoHatRHSPartLabel->getName(),Expr::STATE_NONE) ) );
        momRootIDs.insert(gh->exprFactory->get_id(Expr::Tag(d_lab->d_vVelRhoHatRHSPartLabel->getName(),Expr::STATE_NONE) ) );
        momRootIDs.insert(gh->exprFactory->get_id(Expr::Tag(d_lab->d_wVelRhoHatRHSPartLabel->getName(),Expr::STATE_NONE) ) );
        //
        std::stringstream strRKStage;
        strRKStage << curr_level;
        Wasatch::TaskInterface* wasatchMomRHSTask =
        scinew Wasatch::TaskInterface( momRootIDs,
                                       "warches_mom_rhs_partial_task_stage_" + strRKStage.str(),
                                       *(gh->exprFactory),
                                       level, sched, patches, matls,
                                       wasatch.patch_info_map(),
                                       curr_level+1,
                                       state,
                                       wasatch.locked_fields() );
        wasatch.task_interface_list().push_back( wasatchMomRHSTask );
        wasatchMomRHSTask->schedule( curr_level +1 );
        d_momSolver->sched_computeVelHatWarches( level, sched, curr_level );
      }
    }
#endif
    
    d_momSolver->solveVelHat(level, sched, d_timeIntegratorLabels[curr_level] );

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() < 2 )
        eqn->sched_timeAve( level, sched, curr_level );
    }

    // averaging for RKSSP
    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {

      d_props->sched_averageRKProps(sched, patches, matls,
                                    d_timeIntegratorLabels[curr_level]);

      d_props->sched_saveTempDensity(sched, patches, matls,
                                     d_timeIntegratorLabels[curr_level]);

      // Property models before table lookup
      PropertyModelBase* hl_model = 0;
      for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
            iprop != all_prop_models.end(); iprop++){

        PropertyModelBase* prop_model = iprop->second;

        if ( prop_model->getPropType() == "heat_loss" ){
          hl_model = prop_model; 
        } else {
          if ( prop_model->beforeTableLookUp() )
            prop_model->sched_computeProp( level, sched, curr_level );
        }

      }
      if ( hl_model != 0 )
        hl_model->sched_computeProp( level, sched, curr_level ); 

      //TABLE LOOKUP #2
      bool initialize_it  = false;
      bool modify_ref_den = false;
      d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, curr_level );

      // Property models after table lookup
      for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
            iprop != all_prop_models.end(); iprop++){

        PropertyModelBase* prop_model = iprop->second;
        if ( !prop_model->beforeTableLookUp() )
          prop_model->sched_computeProp( level, sched, curr_level );

      }
    }

    //STAGE 2

    // Source terms needed after second table lookup: 
    src_factory.sched_computeSources( level, sched, curr_level, 2 ); 

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 2 )
        eqn->sched_evalTransportEqn( level, sched, curr_level );
    }

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      if ( eqn->get_stage() == 2 )
        eqn->sched_timeAve( level, sched, curr_level );
    }

    if ( d_extra_table_lookup ){ 
      //TABLE LOOKUP #3
      initialize_it  = false;
      modify_ref_den = false;
      d_props->sched_computeProps( level, sched, initialize_it, modify_ref_den, curr_level );
    }

    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {

      sched_computeDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);
      if (d_maxDensityLag > 0.0)
        sched_checkDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);

      d_momSolver->sched_averageRKHatVelocities(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level] );

    }

    d_boundaryCondition->sched_setIntrusionTemperature( sched, level, matls );

    d_boundaryCondition->sched_setIntrusionDensity( sched, level, matls ); 

    if ( d_wall_ht_models != 0 ){ 
      d_wall_ht_models->sched_doWallHT( level, sched, curr_level ); 
    }

    d_props->sched_computeDrhodt(sched, patches, matls,
                                 d_timeIntegratorLabels[curr_level]);

    d_pressSolver->sched_solve(level, sched, d_timeIntegratorLabels[curr_level],
                               false);

    // project velocities using the projection step
    d_momSolver->solve(sched, patches, matls,
                       d_timeIntegratorLabels[curr_level], false);



    if ( d_timeIntegratorLabels[curr_level]->integrator_last_step) { 
      // this is the new efficiency calculator
      d_eff_calculator->sched_computeAllScalarEfficiencies( level, sched ); 
    }


    // Schedule an interpolation of the face centered velocity data
//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
    if (!(this->get_use_wasatch_mom_rhs())) sched_interpolateFromFCToCC(sched, patches, matls,
                                                                        d_timeIntegratorLabels[curr_level]);
//#endif // WASATCH_IN_ARCHES
    if (d_mixedModel) {
      d_scaleSimilarityModel->sched_reComputeTurbSubmodel( sched, level, matls,
                                                           d_timeIntegratorLabels[curr_level]);
    }

#   ifdef WASATCH_IN_ARCHES
    /* hook in construction of task interface for wasatch scalar transport equations here.
     * This is within the RK loop, so we need to pass the stage as well.
     *
     * Note that at this point we should also build Expr::PlaceHolder objects
     * for all "out-edges" in the Wasatch graph (required quantities from Arches)
     * such as advecting velocity, etc.  These need to be consistent with the
     * names given in the input file, which is a bit of a pain.  We will have
     * to trust the user to get those right.
     */
    {
      const Wasatch::Wasatch::EquationAdaptors& adaptors = wasatch.equation_adaptors();
      Wasatch::GraphHelper* const gh = wasatch.graph_categories()[Wasatch::ADVANCE_SOLUTION];
      
      std::vector<std::string> phi;
      std::vector<std::string> phi_rhs;
      
      std::set< Expr::ExpressionID > scalRHSIDs;
      for( Wasatch::Wasatch::EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ) {
        Wasatch::EquationBase* transEq = (*ia)->equation();
        if ( !(transEq->dir_name() == "") ) continue; // skip momentum equations
        std::string solnVarName = transEq->solution_variable_name();
        phi.push_back(solnVarName);
        std::string rhsName =solnVarName + "_rhs";
        phi_rhs.push_back(rhsName);
        //
        scalRHSIDs.insert( gh->exprFactory->get_id(Expr::Tag(rhsName,Expr::STATE_NONE) ) );
      }
      //
      std::stringstream strRKStage;
      strRKStage << curr_level;
      const std::set<std::string>& ioFieldSet = wasatch.locked_fields();

      if (!(scalRHSIDs.empty())) {
        // ADD THE FORCE-ON-GRAPH EXPRESSIONS TO THIS TASK. THERE IS A MAJOR LIMITATION HERE:
        // IF ANY OF THE FORCED DEPENDENCIES ARE SHARED WITH ANOTHER TASKINTERFACE, THEN WE WILL
        // GET MULTIPLE COMPUTES ERRORS FROM UINTAH.
        if ( !( gh->rootIDs.empty() ) ) scalRHSIDs.insert(gh->rootIDs.begin(), gh->rootIDs.end());
        Wasatch::TaskInterface* wasatchRHSTask =
        scinew Wasatch::TaskInterface( scalRHSIDs,
                                      "warches_scalar_rhs_task_stage_" + strRKStage.str(),
                                      *(gh->exprFactory),
                                      level, sched, patches, matls,
                                      wasatch.patch_info_map(),
                                      curr_level+1,
                                      state,
                                      ioFieldSet
                                      );
        
        // jcs need to build a CoordHelper (or graph the one from wasatch?) - see Wasatch::TimeStepper.cc...
        wasatch.task_interface_list().push_back( wasatchRHSTask );
        wasatchRHSTask->schedule( curr_level +1 );
        // note that there is another interface for this if we need some fields from the new DW.
        d_timeIntegrator->sched_fe_update(sched, patches, matls, phi, phi_rhs, curr_level, true);
        if(curr_level>0) d_timeIntegrator->sched_time_ave(sched, patches, matls, phi, curr_level, true);
        //d_timeIntegrator->sched_wasatch_time_ave(sched, patches, matls, phi, phi_rhs, curr_level);
      }
      
    }
#   endif // WASATCH_IN_ARCHES

    d_turbCounter = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    if ((d_turbCounter%d_turbModelCalcFreq == 0)&&
        ((curr_level==0)||((!(curr_level==0))&&d_turbModelRKsteps)))
      d_turbModel->sched_reComputeTurbSubmodel(sched, level, matls,
                                               d_timeIntegratorLabels[curr_level]);


#ifdef WASATCH_IN_ARCHES
    // wee need this so that the values of momenum in the OldDW on the next timestep are correct
    if (wasatch.get_wasatch_spec()->findBlock("MomentumEquations"))
      d_momSolver->sched_computeMomentum( level, sched, curr_level );
#endif

  }

  if ( d_printTotalKE ){ 
   sched_computeKE( sched, patches, matls ); 
   sched_printTotalKE( sched, patches, matls );
  }

  return(0);

}

// ****************************************************************************
// Schedule initialize
// ****************************************************************************
void
ExplicitSolver::sched_setInitialGuess(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  //copies old db to new_db and then uses non-linear
  //solver to compute new values
  Task* tsk = scinew Task( "ExplicitSolver::setInitialGuess",this,
                           &ExplicitSolver::setInitialGuess);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel,      gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel,  gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_turbViscosLabel,  gn, 0);
//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
  if (!(this->get_use_wasatch_mom_rhs())) tsk->requires(Task::OldDW, d_lab->d_CCVelocityLabel, gn, 0);
//#endif // WASATCH_IN_ARCHES
  tsk->requires(Task::OldDW, d_lab->d_densityGuessLabel,  gn, 0);

  if (!(d_MAlab))
    tsk->computes(d_lab->d_cellInfoLabel);
  else
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn, 0);

  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
  if (!(this->get_use_wasatch_mom_rhs())) {
    tsk->computes(d_lab->d_uVelRhoHatLabel);
    tsk->computes(d_lab->d_vVelRhoHatLabel);
    tsk->computes(d_lab->d_wVelRhoHatLabel);
  }
//#endif // WASATCH_IN_ARCHES
  tsk->computes(d_lab->d_densityCPLabel);
//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
  if (!(this->get_use_wasatch_mom_rhs())) tsk->computes(d_lab->d_viscosityCTSLabel);
//#endif // WASATCH_IN_ARCHES
  tsk->computes(d_lab->d_turbViscosLabel);

  //__________________________________
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel, gn, 0);
    tsk->computes(d_lab->d_densityMicroINLabel);
  }

  //__________________________________
  tsk->computes(d_lab->d_densityTempLabel);
  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Schedule Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
// ****************************************************************************
void
ExplicitSolver::sched_interpolateFromFCToCC(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls,
                                            const TimeIntegratorLabel* timelabels)
{
  {
    string taskname =  "ExplicitSolver::interpFCToCC" +
                     timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
                         &ExplicitSolver::interpolateFromFCToCC, timelabels);

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->requires(Task::OldDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
      tsk->requires(Task::OldDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
      tsk->requires(Task::OldDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    // hat velocities are only interpolated for first substep, since they are
    // not really needed anyway
      tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,  gaf, 1);
      tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,  gaf, 1);
      tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,  gaf, 1);

    }


    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,  gn,  0);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_CCVelocityLabel);
      tsk->computes(d_lab->d_velocityDivergenceLabel);
      tsk->computes(d_lab->d_continuityResidualLabel);
      tsk->computes(d_lab->d_CCUVelocityLabel);
      tsk->computes(d_lab->d_CCVVelocityLabel);
      tsk->computes(d_lab->d_CCWVelocityLabel);
    }
    else {
      tsk->modifies(d_lab->d_CCVelocityLabel);
      tsk->modifies(d_lab->d_velocityDivergenceLabel);
      tsk->modifies(d_lab->d_continuityResidualLabel);
      tsk->modifies(d_lab->d_CCUVelocityLabel);
      tsk->modifies(d_lab->d_CCVVelocityLabel);
      tsk->modifies(d_lab->d_CCWVelocityLabel);
    }

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "ExplicitSolver::computeVorticity" +
                     timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
                         &ExplicitSolver::computeVorticity, timelabels);

    Ghost::GhostType  gac = Ghost::AroundCells;

    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);
    tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel,  gac, 1);
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_vorticityXLabel);
      tsk->computes(d_lab->d_vorticityYLabel);
      tsk->computes(d_lab->d_vorticityZLabel);
      tsk->computes(d_lab->d_vorticityLabel);
    }
    else {
      tsk->modifies(d_lab->d_vorticityXLabel);
      tsk->modifies(d_lab->d_vorticityYLabel);
      tsk->modifies(d_lab->d_vorticityZLabel);
      tsk->modifies(d_lab->d_vorticityLabel);
    }

    sched->addTask(tsk, patches, matls);
  }
}
// ****************************************************************************
// Actual interpolation from FC to CC Variable of type Vector
// ** WARNING ** For multiple patches we need ghost information for
//               interpolation
// ****************************************************************************
void
ExplicitSolver::interpolateFromFCToCC(const ProcessorGroup* ,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> divergence;
    CCVariable<double> residual;
    CCVariable<Vector> newCCVel; 
    constCCVariable<double> density;
    constCCVariable<double> drhodt;

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<Vector> CCVel;
    CCVariable<double> ccUVel;
    CCVariable<double> ccVVel;
    CCVariable<double> ccWVel;

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    new_dw->get(newUVel        , d_lab->d_uVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(newVVel        , d_lab->d_vVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(newWVel        , d_lab->d_wVelocitySPBCLabel , indx , patch , gaf , 1);
    new_dw->get(drhodt         , d_lab->d_filterdrhodtLabel  , indx , patch , gn  , 0);
    new_dw->get(density        , d_lab->d_densityCPLabel     , indx , patch , gac , 1);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(newCCVel,      d_lab->d_CCVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(divergence,    d_lab->d_velocityDivergenceLabel,indx, patch);
      new_dw->allocateAndPut(residual,      d_lab->d_continuityResidualLabel,indx, patch);
      new_dw->allocateAndPut(ccUVel,        d_lab->d_CCUVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(ccVVel,        d_lab->d_CCVVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(ccWVel,        d_lab->d_CCWVelocityLabel,     indx, patch);
    }
    else {
      new_dw->getModifiable(newCCVel,       d_lab->d_CCVelocityLabel,      indx, patch);
      new_dw->getModifiable(divergence,     d_lab->d_velocityDivergenceLabel, indx, patch);
      new_dw->getModifiable(residual,       d_lab->d_continuityResidualLabel, indx, patch);
      new_dw->getModifiable(ccUVel,         d_lab->d_CCUVelocityLabel,     indx, patch);
      new_dw->getModifiable(ccVVel,         d_lab->d_CCVVelocityLabel,     indx, patch);
      new_dw->getModifiable(ccWVel,         d_lab->d_CCWVelocityLabel,     indx, patch);
    }
    newCCVel.initialize(Vector(0.0,0.0,0.0));
    divergence.initialize(0.0);
    residual.initialize(0.0);
    ccUVel.initialize(0.0);
    ccVVel.initialize(0.0);
    ccWVel.initialize(0.0);

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;

        }
      }
    }
    // boundary conditions not to compute erroneous values in the case of ramping
    if (xminus) {
      int ii = idxLo.x()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }
    if (xplus) {
      int ii =  idxHi.x()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = newUVel[idx];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }
    if (yminus) {
      int jj = idxLo.y()-1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idxV];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }
    if (yplus) {
      int jj =  idxHi.y()+1;
      for (int kk = idxLo.z(); kk <=  idxHi.z(); kk ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = newVVel[idx];
          double new_w = cellinfo->bfac[kk] * newWVel[idx] +
                         cellinfo->tfac[kk] * newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }
    if (zminus) {
      int kk = idxLo.z()-1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idxW];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }
    if (zplus) {
      int kk =  idxHi.z()+1;
      for (int jj = idxLo.y(); jj <=  idxHi.y(); jj ++) {
        for (int ii = idxLo.x(); ii <=  idxHi.x(); ii ++) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);

          double new_u = cellinfo->wfac[ii] * newUVel[idx] +
                         cellinfo->efac[ii] * newUVel[idxU];
          double new_v = cellinfo->sfac[jj] * newVVel[idx] +
                         cellinfo->nfac[jj] * newVVel[idxV];
          double new_w = newWVel[idx];

          newCCVel[idx] = Vector(new_u,new_v,new_w);
          ccUVel[idx] = new_u;
          ccVVel[idx] = new_v;
          ccWVel[idx] = new_w;
        }
      }
    }

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          IntVector idxxminus(ii-1,jj,kk);
          IntVector idxyminus(ii,jj-1,kk);
          IntVector idxzminus(ii,jj,kk-1);
          double vol =cellinfo->sns[jj]*cellinfo->stb[kk]*cellinfo->sew[ii];

          divergence[idx] = (newUVel[idxU]-newUVel[idx])/cellinfo->sew[ii]+
                            (newVVel[idxV]-newVVel[idx])/cellinfo->sns[jj]+
                            (newWVel[idxW]-newWVel[idx])/cellinfo->stb[kk];

          residual[idx] = (0.5*(density[idxU]+density[idx])*newUVel[idxU]-
                           0.5*(density[idx]+density[idxxminus])*newUVel[idx])/cellinfo->sew[ii]+
                          (0.5*(density[idxV]+density[idx])*newVVel[idxV]-
                           0.5*(density[idx]+density[idxyminus])*newVVel[idx])/cellinfo->sns[jj]+
                          (0.5*(density[idxW]+density[idx])*newWVel[idxW]-
                           0.5*(density[idx]+density[idxzminus])*newWVel[idx])/cellinfo->stb[kk]+
                          drhodt[idx]/vol;
        }
      }
    }
  }
}

// ****************************************************************************
// Actual calculation of vorticity
// ****************************************************************************
void
ExplicitSolver::computeVorticity(const ProcessorGroup* ,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> vorticityX, vorticityY, vorticityZ, vorticity;

    constCCVariable<Vector> CCVel; 

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(CCVel, d_lab->d_CCVelocityLabel, indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(vorticityX, d_lab->d_vorticityXLabel, indx, patch);
      new_dw->allocateAndPut(vorticityY, d_lab->d_vorticityYLabel, indx, patch);
      new_dw->allocateAndPut(vorticityZ, d_lab->d_vorticityZLabel, indx, patch);
      new_dw->allocateAndPut(vorticity,  d_lab->d_vorticityLabel,  indx, patch);
    }
    else {
      new_dw->getModifiable(vorticityX, d_lab->d_vorticityXLabel, indx, patch);
      new_dw->getModifiable(vorticityY, d_lab->d_vorticityYLabel, indx, patch);
      new_dw->getModifiable(vorticityZ, d_lab->d_vorticityZLabel, indx, patch);
      new_dw->getModifiable(vorticity,  d_lab->d_vorticityLabel,  indx, patch);
    }
    vorticityX.initialize(0.0);
    vorticityY.initialize(0.0);
    vorticityZ.initialize(0.0);
    vorticity.initialize(0.0);

    for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {
          IntVector idx(ii,jj,kk);
          IntVector idxU(ii+1,jj,kk);
          IntVector idxV(ii,jj+1,kk);
          IntVector idxW(ii,jj,kk+1);
          IntVector idxxminus(ii-1,jj,kk);
          IntVector idxyminus(ii,jj-1,kk);
          IntVector idxzminus(ii,jj,kk-1);

          // ii,jj,kk velocity component cancels out when computing derivative,
          // so it has been ommited

          vorticityX[idx] = 0.5*(CCVel[idxV].z()-CCVel[idxyminus].z())/cellinfo->sns[jj]
                           -0.5*(CCVel[idxW].y()-CCVel[idxzminus].y())/cellinfo->stb[kk];
          vorticityY[idx] = 0.5*(CCVel[idxW].x()-CCVel[idxzminus].x())/cellinfo->stb[kk]
                           -0.5*(CCVel[idxU].z()-CCVel[idxxminus].z())/cellinfo->sew[ii];
          vorticityZ[idx] = 0.5*(CCVel[idxU].y()-CCVel[idxxminus].y())/cellinfo->sew[ii]
                           -0.5*(CCVel[idxV].x()-CCVel[idxyminus].x())/cellinfo->sns[jj];
          vorticity[idx] = sqrt(vorticityX[idx]*vorticityX[idx]+vorticityY[idx]*vorticityY[idx]
                          + vorticityZ[idx]*vorticityZ[idx]);
        }
      }
    }
  }
}

// ****************************************************************************
// Actual initialize
// ****************************************************************************
void
ExplicitSolver::setInitialGuess(const ProcessorGroup* ,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> denMicro;
    CCVariable<double> denMicro_new;

    Ghost::GhostType  gn = Ghost::None;
    if (d_MAlab) {
      old_dw->get(denMicro, d_lab->d_densityMicroLabel,  indx, patch, gn, 0);
      new_dw->allocateAndPut(denMicro_new, d_lab->d_densityMicroINLabel, indx, patch);
      denMicro_new.copyData(denMicro);
    }

    constCCVariable<int> cellType;
    if (d_MAlab){
      new_dw->get(cellType, d_lab->d_mmcellTypeLabel, indx, patch,gn, 0);
    }else{
      old_dw->get(cellType, d_lab->d_cellTypeLabel,   indx, patch, gn, 0);
    }


    constCCVariable<double> old_density_guess;
    old_dw->get( old_density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);


    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> turb_viscosity; 
    constCCVariable<Vector> ccVel;
    constCCVariable<double> old_volq;

    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel,  indx, patch, gn, 0);
    old_dw->get(turb_viscosity,    d_lab->d_turbViscosLabel,  indx, patch, gn, 0);

//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
    if (!(this->get_use_wasatch_mom_rhs())) old_dw->get(ccVel,     d_lab->d_CCVelocityLabel, indx, patch, gn, 0);
//#endif // WASATCH_IN_ARCHES

  // Create vars for new_dw ***warning changed new_dw to old_dw...check
    CCVariable<int> cellType_new;
    new_dw->allocateAndPut(cellType_new, d_lab->d_cellTypeLabel, indx, patch);
    cellType_new.copyData(cellType);

    PerPatch<CellInformationP> cellInfoP;
    if (!(d_MAlab))
    {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }
    else
    {
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }

    SFCXVariable<double> uVelocity_new;
    new_dw->allocateAndPut(uVelocity_new, d_lab->d_uVelocitySPBCLabel, indx, patch);
    uVelocity_new.copyData(uVelocity); // copy old into new
    SFCYVariable<double> vVelocity_new;
    new_dw->allocateAndPut(vVelocity_new, d_lab->d_vVelocitySPBCLabel, indx, patch);
    vVelocity_new.copyData(vVelocity); // copy old into new
    SFCZVariable<double> wVelocity_new;
    new_dw->allocateAndPut(wVelocity_new, d_lab->d_wVelocitySPBCLabel, indx, patch);
    wVelocity_new.copyData(wVelocity); // copy old into new

//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
    if (!(this->get_use_wasatch_mom_rhs())) {
      SFCXVariable<double> uVelRhoHat_new;
      new_dw->allocateAndPut(uVelRhoHat_new, d_lab->d_uVelRhoHatLabel, indx, patch);
      uVelRhoHat_new.initialize(0.0);     // copy old into new
      SFCYVariable<double> vVelRhoHat_new;
      new_dw->allocateAndPut(vVelRhoHat_new, d_lab->d_vVelRhoHatLabel, indx, patch);
      vVelRhoHat_new.initialize(0.0); // copy old into new
      SFCZVariable<double> wVelRhoHat_new;
      new_dw->allocateAndPut(wVelRhoHat_new, d_lab->d_wVelRhoHatLabel, indx, patch);
      wVelRhoHat_new.initialize(0.0); // copy old into new
    }
//#endif // WASATCH_IN_ARCHES

    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityCPLabel, indx, patch);
    density_new.copyData(density); // copy old into new

    CCVariable<double> density_temp;
    new_dw->allocateAndPut(density_temp, d_lab->d_densityTempLabel, indx, patch);
    density_temp.copyData(density); // copy old into new

//#ifndef WASATCH_IN_ARCHES // UNCOMMENT THIS TO TRIGGER WASATCH MOM_RHS CALC
    if (!(this->get_use_wasatch_mom_rhs())) {
      CCVariable<double> viscosity_new;
      new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityCTSLabel, indx, patch);
      viscosity_new.copyData(viscosity); // copy old into new
    }
//#endif // WASATCH_IN_ARCHES

    CCVariable<double> turb_viscosity_new;
    new_dw->allocateAndPut(turb_viscosity_new, d_lab->d_turbViscosLabel, indx, patch);
    turb_viscosity_new.copyData(turb_viscosity); // copy old into new

  }
}


//______________________________________________________________________
//
void
ExplicitSolver::sched_printTotalKE( SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls )
                                   
{
  string taskname =  "ExplicitSolver::printTotalKE";
  Task* tsk = scinew Task( taskname,
                           this, &ExplicitSolver::printTotalKE );

  tsk->requires(Task::NewDW, d_lab->d_totalKineticEnergyLabel);
  sched->addTask(tsk, patches, matls);
}
//______________________________________________________________________
void
ExplicitSolver::printTotalKE( const ProcessorGroup* ,
                              const PatchSubset* ,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse* new_dw )
{

  sum_vartype tke;
  new_dw->get( tke, d_lab->d_totalKineticEnergyLabel ); 
  double total_kin_energy = tke;

  proc0cout << "Total kinetic energy: " << total_kin_energy << std::endl;

}

//****************************************************************************
// Schedule saving of temp copies of variables
//****************************************************************************
void
ExplicitSolver::sched_saveTempCopies(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::saveTempCopies" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::saveTempCopies,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gn, 0);
  tsk->modifies(d_lab->d_densityTempLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void
ExplicitSolver::saveTempCopies(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse* new_dw,
                               const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> temp_density;
    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,indx, patch);
    new_dw->copyOut(temp_density,       d_lab->d_densityCPLabel,  indx, patch);

  }
}
//****************************************************************************
// Schedule computation of density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::sched_getDensityGuess(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::getDensityGuess" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::getDensityGuess,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
    tsk->requires(old_values_dw, d_lab->d_densityCPLabel,gn, 0);
  }else{
    old_values_dw = Task::NewDW;
  }


  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);


  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);

  //__________________________________
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
    tsk->computes(d_lab->d_densityGuessLabel);
  }else{
    tsk->modifies(d_lab->d_densityGuessLabel);
  }
  tsk->computes(timelabels->negativeDensityGuess);

  // extra mass source terms
  std::vector<std::string> extra_sources;
  extra_sources = d_pressSolver->get_pressure_source_ref();
  SourceTermFactory& factory = SourceTermFactory::self();
  for (vector<std::string>::iterator iter = extra_sources.begin();
      iter != extra_sources.end(); iter++){

    SourceTermBase& src = factory.retrieve_source_term( *iter );
    const VarLabel* srcLabel = src.getSrcLabel();

    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      tsk->requires( Task::OldDW, srcLabel, gn, 0 );
    } else {
      tsk->requires( Task::NewDW, srcLabel, gn, 0 );
    }
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::getDensityGuess(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  double negativeDensityGuess = 0.0;

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<int> cellType;

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }

    if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
      new_dw->allocateAndPut(densityGuess, d_lab->d_densityGuessLabel, indx, patch);
    }else{
      new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,  indx, patch);
    }
    old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel,indx, patch);

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;

    new_dw->get(density,   d_lab->d_densityCPLabel,     indx,patch, gac, 1);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx,patch, gaf, 1);
    new_dw->get(cellType,  d_lab->d_cellTypeLabel,      indx,patch, gn, 0);

    // For adding other source terms as specified in the pressure solver section
    SourceTermFactory& factory = SourceTermFactory::self();
    std::vector<std::string> extra_sources;
    extra_sources = d_pressSolver->get_pressure_source_ref();
    std::vector<constCCVariable<double> > src_values;
    bool have_extra_srcs = false;

    for (std::vector<std::string>::iterator iter = extra_sources.begin();
        iter != extra_sources.end(); iter++){

      SourceTermBase& src = factory.retrieve_source_term( *iter );
      const VarLabel* srcLabel = src.getSrcLabel();
      constCCVariable<double> src_value;

      if ( timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
       old_dw->get( src_value, srcLabel, indx, patch, gn, 0 );
      } else {
       new_dw->get( src_value, srcLabel, indx, patch, gn, 0 );
      }

      src_values.push_back( src_value );

      have_extra_srcs = true;

    }

// Need to skip first timestep since we start with unprojected velocities
//    int currentTimeStep=d_lab->d_sharedState->getCurrentTopLevelTimeStep();
//    if (currentTimeStep > 1) {
//
      if ( have_extra_srcs ){

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

          IntVector currCell   = *iter;
          IntVector xplusCell  = *iter + IntVector(1,0,0);
          IntVector xminusCell = *iter + IntVector(-1,0,0);
          IntVector yplusCell  = *iter + IntVector(0,1,0);
          IntVector yminusCell = *iter + IntVector(0,-1,0);
          IntVector zplusCell  = *iter + IntVector(0,0,1);
          IntVector zminusCell = *iter + IntVector(0,0,-1);

          densityGuess[currCell] -= delta_t * 0.5* (
          ((density[currCell]+density[xplusCell])*uVelocity[xplusCell] -
           (density[currCell]+density[xminusCell])*uVelocity[currCell]) /
          cellinfo->sew[currCell.x()] +
          ((density[currCell]+density[yplusCell])*vVelocity[yplusCell] -
           (density[currCell]+density[yminusCell])*vVelocity[currCell]) /
          cellinfo->sns[currCell.y()] +
          ((density[currCell]+density[zplusCell])*wVelocity[zplusCell] -
           (density[currCell]+density[zminusCell])*wVelocity[currCell]) /
          cellinfo->stb[currCell.z()]);

          for ( std::vector<constCCVariable<double> >::iterator viter = src_values.begin(); viter != src_values.end(); viter++ ){

            densityGuess[currCell] += delta_t*((*viter)[currCell]);

          }

          if (densityGuess[currCell] < 0.0 && d_noisyDensityGuess) {
            proc0cout << "Negative density guess occured at " << currCell << " with a value of " << densityGuess[currCell] << endl;
            negativeDensityGuess = 1.0;
          }
          else if (densityGuess[currCell] < 0.0 && !(d_noisyDensityGuess)) {
            negativeDensityGuess = 1.0;
          }
        }

      } else {

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

          IntVector currCell   = *iter;
          IntVector xplusCell  = *iter + IntVector(1,0,0);
          IntVector xminusCell = *iter + IntVector(-1,0,0);
          IntVector yplusCell  = *iter + IntVector(0,1,0);
          IntVector yminusCell = *iter + IntVector(0,-1,0);
          IntVector zplusCell  = *iter + IntVector(0,0,1);
          IntVector zminusCell = *iter + IntVector(0,0,-1);

          densityGuess[currCell] -= delta_t * 0.5* (
          ((density[currCell]+density[xplusCell])*uVelocity[xplusCell] -
           (density[currCell]+density[xminusCell])*uVelocity[currCell]) /
          cellinfo->sew[currCell.x()] +
          ((density[currCell]+density[yplusCell])*vVelocity[yplusCell] -
           (density[currCell]+density[yminusCell])*vVelocity[currCell]) /
          cellinfo->sns[currCell.y()] +
          ((density[currCell]+density[zplusCell])*wVelocity[zplusCell] -
           (density[currCell]+density[zminusCell])*wVelocity[currCell]) /
          cellinfo->stb[currCell.z()]);

          if (densityGuess[currCell] < 0.0 && d_noisyDensityGuess) {
            cout << "Negative density guess occured at " << currCell << " with a value of " << densityGuess[currCell] << endl;
            negativeDensityGuess = 1.0;
          }
          else if (densityGuess[currCell] < 0.0 && !(d_noisyDensityGuess) && cellType[currCell] == -1 ) {
            negativeDensityGuess = 1.0;
          }
        }
      }

      std::vector<BoundaryCondition::BC_TYPE> bc_types;
      bc_types.push_back( BoundaryCondition::OUTLET );
      bc_types.push_back( BoundaryCondition::PRESSURE );

      d_boundaryCondition->zeroGradientBC( patch, indx, densityGuess, bc_types );

      new_dw->put(sum_vartype(negativeDensityGuess),
                  timelabels->negativeDensityGuess);
  }
}
//****************************************************************************
// Schedule check for negative density guess
//****************************************************************************
void
ExplicitSolver::sched_checkDensityGuess(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::checkDensityGuess" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::checkDensityGuess,
                          timelabels);

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else {
    old_values_dw = Task::NewDW;
  }

  tsk->requires(old_values_dw, d_lab->d_densityCPLabel,Ghost::None, 0);

  tsk->requires(Task::NewDW, timelabels->negativeDensityGuess);

  tsk->modifies(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually check for negative density guess
//****************************************************************************
void
ExplicitSolver::checkDensityGuess(const ProcessorGroup* pc,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  double negativeDensityGuess = 0.0;
  sum_vartype nDG;
  new_dw->get(nDG, timelabels->negativeDensityGuess);

  negativeDensityGuess = nDG;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values)
      old_values_dw = parent_old_dw;
    else
      old_values_dw = new_dw;

    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel, indx, patch);
    if (negativeDensityGuess > 0.0) {
      if (d_restart_on_negative_density_guess) {
        proc0cout << "NOTICE: Negative density guess(es) occured. Timestep restart has been requested under this condition by the user. Restarting timestep." << endl;
        new_dw->abortTimestep();
        new_dw->restartTimestep();
      }
      else {
        proc0cout << "NOTICE: Negative density guess(es) occured. Reverting to old density." << endl;
        old_values_dw->copyOut(densityGuess, d_lab->d_densityCPLabel, indx, patch);
      }
    }
  }
}
//****************************************************************************
// Schedule update of density guess
//****************************************************************************
void
ExplicitSolver::sched_updateDensityGuess(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::updateDensityGuess" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::updateDensityGuess,
                          timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 1);
  tsk->modifies(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute density guess from the continuity equation
//****************************************************************************
void
ExplicitSolver::updateDensityGuess(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw,
                                   const TimeIntegratorLabel*)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> densityGuess;
    constCCVariable<double> density;

    new_dw->getModifiable(densityGuess, d_lab->d_densityGuessLabel,indx, patch);
    new_dw->copyOut(densityGuess,       d_lab->d_densityCPLabel,   indx, patch);
  }
}
//****************************************************************************
// Schedule computing density lag
//****************************************************************************
void
ExplicitSolver::sched_computeDensityLag(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool after_average)
{
  string taskname =  "ExplicitSolver::computeDensityLag" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::computeDensityLag,
                          timelabels, after_average);
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,   gn, 0);

  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      tsk->computes(d_lab->d_densityLagAfterAverage_label);
    }else{
      tsk->computes(d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    tsk->computes(timelabels->densityLag);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually compute deensity lag
//****************************************************************************
void
ExplicitSolver::computeDensityLag(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels,
                                  bool after_average)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                     getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> densityGuess;
    constCCVariable<double> density;

    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(densityGuess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);
    new_dw->get(density, d_lab->d_densityCPLabel,         indx, patch, gn, 0);

    double densityLag = 0.0;
    IntVector idxLo = patch->getExtraCellLowIndex();
    IntVector idxHi = patch->getExtraCellHighIndex();
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          densityLag += Abs(density[currCell] - densityGuess[currCell]);
        }
      }
    }
    if (after_average){
      if ((timelabels->integrator_step_name == "Corrector")||
          (timelabels->integrator_step_name == "CorrectorRK3")){
        new_dw->put(sum_vartype(densityLag), d_lab->d_densityLagAfterAverage_label);
      }else{
        new_dw->put(sum_vartype(densityLag), d_lab->d_densityLagAfterIntermAverage_label);
      }
    }else{
      new_dw->put(sum_vartype(densityLag), timelabels->densityLag);
    }
  }
}
//****************************************************************************
// Schedule check for density lag
//****************************************************************************
void
ExplicitSolver::sched_checkDensityLag(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels,
                                      bool after_average)
{
  string taskname =  "ExplicitSolver::checkDensityLag" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::checkDensityLag,
                          timelabels, after_average);

  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      tsk->requires(Task::NewDW, d_lab->d_densityLagAfterAverage_label);
    }else{
      tsk->requires(Task::NewDW, d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    tsk->requires(Task::NewDW, timelabels->densityLag);
  }

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually check for density lag
//****************************************************************************
void
ExplicitSolver::checkDensityLag(const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels,
                                bool after_average)
{
  double densityLag = 0.0;
  sum_vartype denLag;
  if (after_average){
    if ((timelabels->integrator_step_name == "Corrector")||
        (timelabels->integrator_step_name == "CorrectorRK3")){
      new_dw->get(denLag, d_lab->d_densityLagAfterAverage_label);
    }else{
      new_dw->get(denLag, d_lab->d_densityLagAfterIntermAverage_label);
    }
  }else{
    new_dw->get(denLag, timelabels->densityLag);
  }
  densityLag = denLag;

  for (int p = 0; p < patches->size(); p++) {

    if (densityLag > d_maxDensityLag) {
        if (pc->myrank() == 0)
          proc0cout << "WARNING: density lag " << densityLag
               << " exceeding maximium "<< d_maxDensityLag
               << " specified. Restarting timestep." << endl;
        new_dw->abortTimestep();
        new_dw->restartTimestep();
    }
  }
}
void ExplicitSolver::setInitVelConditionInterface( const Patch* patch,
                                             SFCXVariable<double>& uvel,
                                             SFCYVariable<double>& vvel,
                                             SFCZVariable<double>& wvel )
{

  d_momSolver->setInitVelCondition( patch, uvel, vvel, wvel );

}

void ExplicitSolver::sched_computeKE( SchedulerP& sched, 
                                       const PatchSet* patches, 
                                       const MaterialSet* matls )
{
  string taskname = "ExplicitSolver::computeKE"; 
  Task* tsk = scinew Task( taskname, this, &ExplicitSolver::computeKE ); 
  
  tsk->computes(d_lab->d_totalKineticEnergyLabel); 
  tsk->computes(d_lab->d_kineticEnergyLabel); 

  tsk->requires( Task::NewDW, d_lab->d_uVelocitySPBCLabel, Ghost::None, 0 ); 
  tsk->requires( Task::NewDW, d_lab->d_vVelocitySPBCLabel, Ghost::None, 0 ); 
  tsk->requires( Task::NewDW, d_lab->d_wVelocitySPBCLabel, Ghost::None, 0 ); 

  sched->addTask( tsk, patches, matls ); 

}

void ExplicitSolver::computeKE( const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; 
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    constSFCXVariable<double> u; 
    constSFCYVariable<double> v; 
    constSFCZVariable<double> w; 
    CCVariable<double> ke; 

    new_dw->allocateAndPut( ke, d_lab->d_kineticEnergyLabel, indx, patch ); 
    ke.initialize(0.0);
    new_dw->get( u, d_lab->d_uVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
    new_dw->get( v, d_lab->d_vVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 
    new_dw->get( w, d_lab->d_wVelocitySPBCLabel, indx, patch, Ghost::None, 0 ); 

    double max_ke = 0.0; 
    double sum_ke = 0.0; 

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

      IntVector c = *iter; 

      ke[c] = 0.5 * ( u[c]*u[c] + v[c]*v[c] + w[c]*w[c] ); 

      if ( ke[c] > max_ke ){ 
        max_ke = ke[c]; 
      } 

      sum_ke += ke[c]; 

    }

    new_dw->put(sum_vartype(sum_ke), d_lab->d_totalKineticEnergyLabel);

    if ( sum_ke != sum_ke )
      throw InvalidValue("Error: KE is diverging.",__FILE__,__LINE__);

    if ( sum_ke > d_ke_limit ) { 
      std::stringstream msg; 
      msg << "Error: Your KE has exceeded the max threshold allowed of: " << d_ke_limit << endl;
      throw InvalidValue(msg.str(), __FILE__, __LINE__); 
    }

  }
}
