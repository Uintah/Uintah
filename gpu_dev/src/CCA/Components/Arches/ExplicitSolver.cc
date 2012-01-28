/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

*/


//----- ExplicitSolver.cc ----------------------------------------------
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

#include <CCA/Components/Arches/ExplicitSolver.h>
#include <Core/Containers/StaticArray.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/EnthalpySolver.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ScalarSolver.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
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
#ifdef PetscFilter
#include <CCA/Components/Arches/Filter.h>
#endif

#ifdef WASATCH_IN_ARCHES
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/transport/TransportEquation.h>
#include <CCA/Components/Wasatch/transport/ParseEquation.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/TaskInterface.h>
#include <expression/ExprLib.h>
#include <expression/PlaceHolderExpr.h>
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
               bool calc_Scalar,
               bool calc_enthalpy,
               bool calc_variance,
               const ProcessorGroup* myworld,
               SolverInterface* hypreSolver):
               NonlinearSolver(myworld),
               d_lab(label), d_MAlab(MAlb), d_props(props),
               d_boundaryCondition(bc), d_turbModel(turbModel),
               d_scaleSimilarityModel(scaleSimilarityModel),
               d_calScalar(calc_Scalar),
               d_enthalpySolve(calc_enthalpy),
               d_calcVariance(calc_variance),
               d_physicalConsts(physConst),
               d_hypreSolver(hypreSolver)
{
  d_pressSolver = 0;
  d_momSolver = 0;
  d_scalarSolver = 0;
  d_enthalpySolver = 0;
  nosolve_timelabels_allocated = false;
  d_probe_data = false;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
ExplicitSolver::~ExplicitSolver()
{
  delete d_pressSolver;
  delete d_momSolver;
  delete d_scalarSolver;
  delete d_enthalpySolver;
  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
    delete d_timeIntegratorLabels[curr_level];
  if (nosolve_timelabels_allocated)
    delete nosolve_timelabels;
}

// ****************************************************************************
// Problem Setup
// ****************************************************************************
void
ExplicitSolver::problemSetup(const ProblemSpecP& params)
  // MultiMaterialInterface* mmInterface
{
  ProblemSpecP db = params->findBlock("ExplicitSolver");
  ProblemSpecP test_probe_db = db->findBlock("ProbePoints");
  if ( test_probe_db ) {
    d_probe_data = true;
    IntVector prbPoint;
    for (ProblemSpecP probe_db = db->findBlock("ProbePoints");
         probe_db;
         probe_db = probe_db->findNextBlock("ProbePoints")) {
      probe_db->require("probe_point", prbPoint);
      d_probePoints.push_back(prbPoint);
    }
  }
  d_pressSolver = scinew PressureSolver(d_lab, d_MAlab,
                                          d_boundaryCondition,
                                          d_physicalConsts, d_myworld,
                                          d_hypreSolver);
  d_pressSolver->problemSetup(db);

  d_momSolver = scinew MomentumSolver(d_lab, d_MAlab,
                                        d_turbModel, d_boundaryCondition,
                                        d_physicalConsts);
  d_momSolver->setMMS(d_doMMS);
  d_momSolver->problemSetup(db);

  if (d_calScalar) {
    d_scalarSolver = scinew ScalarSolver(d_lab, d_MAlab,
                                         d_turbModel, d_boundaryCondition,
                                         d_physicalConsts);
    d_scalarSolver->setMMS(d_doMMS);
    d_scalarSolver->problemSetup(db);
  }
  if (d_enthalpySolve) {
    d_enthalpySolver = scinew EnthalpySolver(d_lab, d_MAlab,
                                             d_turbModel, d_boundaryCondition,
                                             d_physicalConsts, d_myworld);
    d_enthalpySolver->setMMS(d_doMMS);
    d_enthalpySolver->problemSetup(db);
  }

  const ProblemSpecP params_root = db->getRootNode();
  std::string t_order; 
  ProblemSpecP db_time_int = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TimeIntegrator");
  db_time_int->findBlock("ExplicitIntegrator")->getAttribute("order", t_order);

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

#ifdef PetscFilter
    d_props->setFilter(d_turbModel->getFilter());
//#ifdef divergenceconstraint
    d_momSolver->setDiscretizationFilter(d_turbModel->getFilter());
//#endif
#endif
  d_dynScalarModel = d_turbModel->getDynScalarModel();
  d_mixedModel=d_turbModel->getMixedModel();
  if (d_enthalpySolve) {
    d_H_air = d_props->getAdiabaticAirEnthalpy();
    d_enthalpySolver->setAdiabaticAirEnthalpy(d_H_air);
  }

  if (d_doMMS) {

    ProblemSpecP params_non_constant = params;
    const ProblemSpecP params_root = params_non_constant->getRootNode();
    ProblemSpecP db_mmsblock=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS");

    if(!db_mmsblock->getAttribute("whichMMS",d_mms))
      d_mms="constantMMS";

    db_mmsblock->getWithDefault("mmsErrorType",d_mmsErrorType,"L2");

    if (d_mms == "constantMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("constantMMS");
      db_whichmms->getWithDefault("cu",cu,1.0);
      db_whichmms->getWithDefault("cv",cv,1.0);
      db_whichmms->getWithDefault("cw",cw,1.0);
      db_whichmms->getWithDefault("cp",cp,1.0);
      db_whichmms->getWithDefault("phi0",phi0,0.5);
    }
    else if (d_mms == "almgrenMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("almgrenMMS");
      db_whichmms->getWithDefault("amplitude",amp,0.0);
      db_whichmms->require("viscosity",d_viscosity);
    }
    else
      throw InvalidValue("current MMS "
                         "not supported: " + d_mms, __FILE__, __LINE__);

  }
}

// ****************************************************************************
// Schedule non linear solve and carry out some actual operations
// ****************************************************************************
int ExplicitSolver::nonlinearSolve(const LevelP& level,
                                   SchedulerP& sched
#                                  ifdef WASATCH_IN_ARCHES
                                   , Wasatch::Wasatch& wasatch, 
                                   ExplicitTimeInt* d_timeIntegrator
#                                  endif // WASATCH_IN_ARCHES
                                   )
{

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  IntVector periodic_vector = level->getPeriodicBoundaries();
  d_3d_periodic = (periodic_vector == IntVector(1,1,1));
  d_turbModel->set3dPeriodic(d_3d_periodic);
  d_props->set3dPeriodic(d_3d_periodic);

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP,
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  d_boundaryCondition->sched_setAreaFraction(sched, patches, matls);

  // Start the iterations

  // check if filter is defined...
#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized())
      d_turbModel->sched_initFilterMatrix(level, sched, patches, matls);
  }
#endif

  // Get a reference to all the DQMOM equations
  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  if (dqmomFactory.get_quad_nodes() > 0)
    d_doDQMOM = true;
  else
    d_doDQMOM = false; // probably need to sync this better with the bool being set in Arches

  for (int curr_level = 0; curr_level < numTimeIntegratorLevels; curr_level ++)
  {

    // Clean up all property models
     PropertyModelFactory& propFactory = PropertyModelFactory::self();
     PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
     for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
         iprop != all_prop_models.end(); iprop++){

       PropertyModelBase* prop_model = iprop->second;
       prop_model->cleanUp();

     }

    if (d_doDQMOM) {

      CoalModelFactory& modelFactory = CoalModelFactory::self();
      DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
      DQMOMEqnFactory::EqnMap& weights_eqns = dqmomFactory.retrieve_weights_eqns();
      DQMOMEqnFactory::EqnMap& abscissas_eqns = dqmomFactory.retrieve_abscissas_eqns();

      // Compute the particle velocities
      d_partVel->schedComputePartVel( level, sched, curr_level );

      // ---- schedule the solution of the transport equations ----

      // Evaluate DQMOM equations

      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = weights_eqns.begin();
            iEqn != weights_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level );
      }

      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = abscissas_eqns.begin();
            iEqn != abscissas_eqns.end(); iEqn++){

        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

        dqmom_eqn->sched_evalTransportEqn( level, sched, curr_level );
      }

      // Clean up after DQMOM equation evaluations & calculate unscaled DQMOM scalar values
      // (also, putting this in its own separate loop makes sure you don't require() before you compute())
      if (curr_level == numTimeIntegratorLevels-1){
        for( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin();
             iEqn!=dqmom_eqns.end(); ++iEqn ) {

          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

          // last time sub-step: so cleanup.
          dqmom_eqn->sched_cleanUp( level, sched );
          //also get the abscissa values
          dqmom_eqn->sched_getUnscaledValues( level, sched );
        }
      }

      // schedule the models for evaluation
      //CoalModelFactory::ModelMap allModels = modelFactory.retrieve_all_models();
      //for (CoalModelFactory::ModelMap::iterator imodel = allModels.begin(); imodel != allModels.end(); imodel++){
      //  imodel->second->sched_computeModel( level, sched, curr_level );
      //}
      modelFactory.sched_coalParticleCalculation( level, sched, curr_level );

      // schedule DQMOM linear solve
      d_dqmomSolver->sched_solveLinearSystem( level, sched, curr_level );

      // calculate the moments
      bool saveMoments = d_dqmomSolver->getSaveMoments();
      if( saveMoments ) {
        // schedule DQMOM moment calculation
        d_dqmomSolver->sched_calculateMoments( level, sched, curr_level );
      }

    }

    sched_saveTempCopies(sched, patches, matls,d_timeIntegratorLabels[curr_level]);

    sched_getDensityGuess(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);

    sched_checkDensityGuess(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);

    d_scalarSolver->solve(sched, patches, matls,
                          d_timeIntegratorLabels[curr_level]);

    EqnFactory& eqn_factory = EqnFactory::self();
    EqnFactory::EqnMap& scalar_eqns = eqn_factory.retrieve_all_eqns();
    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
        eqn->sched_evalTransportEqn( level, sched, curr_level );
    }

    if (d_enthalpySolve)
      d_enthalpySolver->solve(level, sched, patches, matls,
                              d_timeIntegratorLabels[curr_level]);

    if (d_calcVariance) {
      d_turbModel->sched_computeScalarVariance(sched, patches, matls,
                                           d_timeIntegratorLabels[curr_level]);

      d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
                                           d_timeIntegratorLabels[curr_level]);
    }

//    d_props->sched_reComputeProps(sched, patches, matls,
//             d_timeIntegratorLabels[curr_level], false, false);
//    sched_syncRhoF(sched, patches, matls, d_timeIntegratorLabels[curr_level]);
//    sched_updateDensityGuess(sched, patches, matls,
//                                    d_timeIntegratorLabels[curr_level]);
//    d_timeIntegratorLabels[curr_level]->integrator_step_number = TimeIntegratorStepNumber::Second;
//    d_props->sched_reComputeProps(sched, patches, matls,
//             d_timeIntegratorLabels[curr_level], false, false);
//    sched_syncRhoF(sched, patches, matls, d_timeIntegratorLabels[curr_level]);
//    sched_updateDensityGuess(sched, patches, matls,
//                                    d_timeIntegratorLabels[curr_level]);

    // Property models needed before table lookup:
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;
      if ( prop_model->beforeTableLookUp() )
        prop_model->sched_computeProp( level, sched, curr_level );

    }


    string mixmodel = d_props->getMixingModelType();
    if ( mixmodel != "TabProps" && mixmodel != "ClassicTable" && mixmodel != "ColdFlow")
      d_props->sched_reComputeProps(sched, patches, matls,
                                    d_timeIntegratorLabels[curr_level],
                                    true, false );
    else {

      bool initialize_it  = false;
      bool modify_ref_den = true;
      if ( curr_level == 0 ) initialize_it = true;
      d_props->sched_reComputeProps_new( level, sched, d_timeIntegratorLabels[curr_level], initialize_it, modify_ref_den );

    }

    d_boundaryCondition->sched_setIntrusionTemperature( sched, patches, matls );

    // Property models needed after table lookup:
    for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
          iprop != all_prop_models.end(); iprop++){

      PropertyModelBase* prop_model = iprop->second;
      prop_model->sched_computeProp( level, sched, curr_level );

    }

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
      //Transport is constructed above.  Here we only solve if densityGuess is not used.
      if ( !eqn->getDensityGuessBool() )
        eqn->sched_solveTransportEqn( level, sched, curr_level );
    }
    // Clean up after Scalar equation evaluations
    if (curr_level == numTimeIntegratorLevels-1){
      for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
        EqnBase* eqn = iter->second;
        eqn->sched_cleanUp( level, sched );
      }
    }

    sched_computeDensityLag(sched, patches, matls,
                           d_timeIntegratorLabels[curr_level],
                           false);
    if (d_maxDensityLag > 0.0)
      sched_checkDensityLag(sched, patches, matls,
                            d_timeIntegratorLabels[curr_level],
                            false);
//    d_timeIntegratorLabels[curr_level]->integrator_step_number = TimeIntegratorStepNumber::First;
    d_props->sched_computeDenRefArray(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level]);
    // sched_syncRhoF(sched, patches, matls, d_timeIntegratorLabels[curr_level]);

    // linearizes and solves pressure eqn
    // first computes, hatted velocities and then computes
    // the pressure poisson equation
    d_momSolver->solveVelHat(level, sched, d_timeIntegratorLabels[curr_level] );

    for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){
      EqnBase* eqn = iter->second;
        eqn->sched_timeAve( level, sched, curr_level );
    }

    // averaging for RKSSP
    if ((curr_level>0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {
      d_props->sched_averageRKProps(sched, patches, matls,
                                    d_timeIntegratorLabels[curr_level]);
      d_props->sched_saveTempDensity(sched, patches, matls,
                                     d_timeIntegratorLabels[curr_level]);
      if (d_calcVariance) {
        d_turbModel->sched_computeScalarVariance(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);

        d_turbModel->sched_computeScalarDissipation(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);
      }

      if (mixmodel != "TabProps" && mixmodel != "ClassicTable" && mixmodel != "ColdFlow")
        d_props->sched_reComputeProps(sched, patches, matls,
                                      d_timeIntegratorLabels[curr_level],
                                      false, false);
      else {

        bool initialize_it  = false;
        bool modify_ref_den = false;
        d_props->sched_reComputeProps_new( level, sched, d_timeIntegratorLabels[curr_level], initialize_it, modify_ref_den );

      }

      // Property models after table lookup
      for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
            iprop != all_prop_models.end(); iprop++){

        PropertyModelBase* prop_model = iprop->second;
        prop_model->sched_computeProp( level, sched, curr_level );

      }


      sched_computeDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);
      if (d_maxDensityLag > 0.0)
        sched_checkDensityLag(sched, patches, matls,
                              d_timeIntegratorLabels[curr_level],
                              true);
      //sched_syncRhoF(sched, patches, matls, d_timeIntegratorLabels[curr_level]);
      d_momSolver->sched_averageRKHatVelocities(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level] );
    }

    d_boundaryCondition->sched_setIntrusionTemperature( sched, patches, matls );


    d_props->sched_computeDrhodt(sched, patches, matls,
                                 d_timeIntegratorLabels[curr_level]);

    d_pressSolver->sched_solve(level, sched, d_timeIntegratorLabels[curr_level],
                               false);

    // project velocities using the projection step
    d_momSolver->solve(sched, patches, matls,
                       d_timeIntegratorLabels[curr_level], false);



    if (d_extraProjection) {
      d_momSolver->sched_prepareExtraProjection(sched, patches, matls,
                                          d_timeIntegratorLabels[curr_level],
                                          false);
      d_pressSolver->sched_solve(level, sched, d_timeIntegratorLabels[curr_level],
                                 d_extraProjection);

      d_momSolver->solve(sched, patches, matls,
                       d_timeIntegratorLabels[curr_level],
                       d_extraProjection);
    }

    //if (curr_level == numTimeIntegratorLevels - 1) {
    if (d_boundaryCondition->anyArchesPhysicalBC()) {

      d_boundaryCondition->sched_getFlowINOUT(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);
      d_boundaryCondition->sched_correctVelocityOutletBC(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);
    }
    //}
    if ((d_boundaryCondition->anyArchesPhysicalBC())&&
        (d_timeIntegratorLabels[curr_level]->integrator_last_step)) {
      d_boundaryCondition->sched_getScalarFlowRate(sched, patches, matls);
      d_boundaryCondition->sched_getScalarEfficiency(sched, patches, matls);
    }


    // Schedule an interpolation of the face centered velocity data
    sched_interpolateFromFCToCC(sched, patches, matls,
                        d_timeIntegratorLabels[curr_level]);

    // Compute mms error
    if (d_doMMS){
      sched_computeMMSError(sched, patches, matls,
                            d_timeIntegratorLabels[curr_level]);
    }
    if (d_mixedModel) {
      d_scaleSimilarityModel->sched_reComputeTurbSubmodel(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);
    }

    d_turbCounter = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    if ((d_turbCounter%d_turbModelCalcFreq == 0)&&
        ((curr_level==0)||((!(curr_level==0))&&d_turbModelRKsteps)))
      d_turbModel->sched_reComputeTurbSubmodel(sched, patches, matls,
                                            d_timeIntegratorLabels[curr_level]);


    sched_printTotalKE(sched, patches, matls,
                       d_timeIntegratorLabels[curr_level]);
    if ((curr_level==0)&&(!((d_timeIntegratorType == "RK2")||(d_timeIntegratorType == "BEEmulation")))) {
       sched_saveFECopies(sched, patches, matls,
                                       d_timeIntegratorLabels[curr_level]);
    }

#   ifdef WASATCH_IN_ARCHES
    /* hook in construction of task interface for wasatch transport equations here.
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
      
      for( Wasatch::Wasatch::EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ) {
        Wasatch::TransportEquation* transEq = (*ia)->equation();
        std::string solnVarName = transEq->solution_variable_name();
        phi.push_back(solnVarName);
        phi_rhs.push_back(solnVarName + "_rhs");
      }      
      //
      std::stringstream strRKStage;
      strRKStage << curr_level;
      const std::set<std::string>& ioFieldSet = wasatch.io_field_set();              
      Wasatch::TaskInterface* wasatchRHSTask =
        scinew Wasatch::TaskInterface( gh->rootIDs,
                                       "wasatch_task_rhs_stage_" + strRKStage.str(),
                                       *(gh->exprFactory),
                                       level, sched, patches, matls,
                                       wasatch.patch_info_map(),
                                       true,
                                       curr_level+1,
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
#   endif // WASATCH_IN_ARCHES
  }

  // print information at probes provided in input file
  if (d_probe_data)
    sched_probeData(sched, patches, matls);

  return(0);
}

// ****************************************************************************
// No Solve option (used to skip first time step calculation
// so that further time steps will have correct initial condition)
// ****************************************************************************

int ExplicitSolver::noSolve(const LevelP& level,
                            SchedulerP& sched
#                                  ifdef WASATCH_IN_ARCHES
                            , Wasatch::Wasatch& wasatch, 
                            ExplicitTimeInt* d_timeIntegrator
#                                  endif // WASATCH_IN_ARCHES
                            )
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  // use FE timelabels for nosolve
  nosolve_timelabels = scinew TimeIntegratorLabel(d_lab,
                                            TimeIntegratorStepType::FE);
  nosolve_timelabels_allocated = true;

  //initializes and allocates vars for new_dw
  // set initial guess
  // require : old_dw -> pressureSPBC, [u,v,w]velocitySPBC, scalarSP,
  // densityCP, viscosityCTS
  // compute : new_dw -> pressureIN, [u,v,w]velocityIN, scalarIN, densityIN,
  //                     viscosityIN

  sched_setInitialGuess(sched, patches, matls);

  // check if filter is defined...
#ifdef PetscFilter
  if (d_turbModel->getFilter()) {
    // if the matrix is not initialized
    if (!d_turbModel->getFilter()->isInitialized())
      d_turbModel->sched_initFilterMatrix(          level, sched, patches, matls);
  }
#endif

  if (d_calcVariance) {
    d_turbModel->sched_computeScalarVariance(             sched, patches, matls,
                                                          nosolve_timelabels);

    d_turbModel->sched_computeScalarDissipation(          sched, patches, matls,
                                                          nosolve_timelabels);
  }

  EqnFactory& eqn_factory = EqnFactory::self();
  EqnFactory::EqnMap& scalar_eqns = eqn_factory.retrieve_all_eqns();
  for (EqnFactory::EqnMap::iterator iter = scalar_eqns.begin(); iter != scalar_eqns.end(); iter++){

    EqnBase* eqn = iter->second;
    eqn->sched_dummyInit( level, sched );

  }

  string mixmodel = d_props->getMixingModelType();
  if ( mixmodel == "TabProps"  || mixmodel == "ClassicTable" )
    d_props->sched_doTPDummyInit( level, sched );

  d_props->sched_computePropsFirst_mm(                    sched, patches, matls);

  d_props->sched_computeDrhodt(                           sched, patches, matls,
                                                          nosolve_timelabels);

  d_boundaryCondition->sched_setInletFlowRates(           sched, patches, matls);

  sched_dummySolve(                                       sched, patches, matls);

  sched_interpolateFromFCToCC(                            sched, patches, matls,
                                                          nosolve_timelabels);

  if (d_mixedModel) {
    d_scaleSimilarityModel->sched_reComputeTurbSubmodel(  sched, patches, matls,
                                                          nosolve_timelabels);
  }

  d_turbModel->sched_reComputeTurbSubmodel(               sched, patches, matls,
                                                          nosolve_timelabels);

  d_pressSolver->sched_addHydrostaticTermtoPressure(      sched, patches, matls,
                                                          nosolve_timelabels);

  // DQMOM and scalar transport init

  // Get a reference to all the DQMOM equations
  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
  if (dqmomFactory.get_quad_nodes() > 0)
    d_doDQMOM = true;
  else
    d_doDQMOM = false; // probably need to sync this better with the bool being set in Arches

  if (d_doDQMOM) {

    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();

    for (DQMOMEqnFactory::EqnMap::iterator ieqn = dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){

      std::string currname = ieqn->first;
      EqnBase* eqn = ieqn->second;
      eqn->sched_dummyInit( level, sched );

    }

    CoalModelFactory& modelFactory = CoalModelFactory::self();
    CoalModelFactory::ModelMap allModels = modelFactory.retrieve_all_models();
    for (CoalModelFactory::ModelMap::iterator imodel = allModels.begin(); imodel != allModels.end(); imodel++){

      imodel->second->sched_dummyInit( level, sched );

    }
  }

  SourceTermFactory& src_factory = SourceTermFactory::self();
  SourceTermFactory::SourceMap& sources = src_factory.retrieve_all_sources();
  for (SourceTermFactory::SourceMap::iterator iter = sources.begin(); iter != sources.end(); iter++){

    SourceTermBase* src = iter->second;
    src->sched_dummyInit( level, sched );

  }

  PropertyModelFactory& propFactory = PropertyModelFactory::self();
  PropertyModelFactory::PropMap& all_prop_models = propFactory.retrieve_all_property_models();
  for ( PropertyModelFactory::PropMap::iterator iprop = all_prop_models.begin();
      iprop != all_prop_models.end(); iprop++){

    PropertyModelBase* prop_model = iprop->second;
    prop_model->sched_dummyInit( level, sched );

  }


  // Schedule an interpolation of the face centered velocity data
  // to a cell centered vector for used by the viz tools

  // print information at probes provided in input file

  if (d_probe_data)
    sched_probeData(sched, patches, matls);

#   ifdef WASATCH_IN_ARCHES
  {
    
    const Wasatch::Wasatch::EquationAdaptors& adaptors = wasatch.equation_adaptors();
    Wasatch::GraphHelper* const gh = wasatch.graph_categories()[Wasatch::ADVANCE_SOLUTION];      
    
    // create a dummy_init task to allow us to save all wasatch fields. Since there is
    // no time integration, this dummy wasatch task will have no effect on the solution, albeit
    // at the cost of a wasatch calculation during the dummy init step.
    const std::set<std::string>& ioFieldSet = wasatch.io_field_set();              
    Wasatch::TaskInterface* wasatchDummyInitTask =
    scinew Wasatch::TaskInterface( gh->rootIDs,
                                  "wasatch_task_dummy_init",
                                  *(gh->exprFactory),
                                  level, sched, patches, matls,
                                  wasatch.patch_info_map(),
                                  true,
                                  1,
                                  ioFieldSet 
                                  );
    wasatch.task_interface_list().push_back( wasatchDummyInitTask );
    wasatchDummyInitTask->schedule( 1 );
    
    // because of the dummy, dummy_init, we have to manually copy the wasatch 
    // transported variables from the old dw to the new dw.
    std::vector<std::string> phi;      
    for( Wasatch::Wasatch::EquationAdaptors::const_iterator ia=adaptors.begin(); ia!=adaptors.end(); ++ia ) {
      Wasatch::TransportEquation* transEq = (*ia)->equation();
      std::string solnVarName = transEq->solution_variable_name();
      phi.push_back(solnVarName);
    }          
    d_timeIntegrator->sched_dummy_init(sched, patches, matls, phi);
  }
#   endif // WASATCH_IN_ARCHES
  
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
  tsk->requires(Task::OldDW, d_lab->d_scalarSPLabel,      gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_viscosityCTSLabel,  gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_newCCVelocityLabel, gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_areaFractionLabel,  gn, 0);
#ifdef WASATCH_IN_ARCHES
  tsk->requires(Task::OldDW, d_lab->d_areaFractionFXLabel,  gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_areaFractionFYLabel,  gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_areaFractionFZLabel,  gn, 0);
#endif
  tsk->requires(Task::OldDW, d_lab->d_volFractionLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityGuessLabel,  gn, 0);

  if (!(d_MAlab))
    tsk->computes(d_lab->d_cellInfoLabel);
  else
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn, 0);

  tsk->computes(d_lab->d_cellTypeLabel);
  tsk->computes(d_lab->d_uVelocitySPBCLabel);
  tsk->computes(d_lab->d_vVelocitySPBCLabel);
  tsk->computes(d_lab->d_wVelocitySPBCLabel);
  tsk->computes(d_lab->d_uVelRhoHatLabel);
  tsk->computes(d_lab->d_vVelRhoHatLabel);
  tsk->computes(d_lab->d_wVelRhoHatLabel);
  tsk->computes(d_lab->d_densityCPLabel);
  tsk->computes(d_lab->d_scalarSPLabel);
  tsk->computes(d_lab->d_scalarBoundarySrcLabel);
  tsk->computes(d_lab->d_enthalpyBoundarySrcLabel);
  tsk->computes(d_lab->d_umomBoundarySrcLabel);
  tsk->computes(d_lab->d_vmomBoundarySrcLabel);
  tsk->computes(d_lab->d_wmomBoundarySrcLabel);
  tsk->computes(d_lab->d_viscosityCTSLabel);
  tsk->computes(d_lab->d_areaFractionLabel);
#ifdef WASATCH_IN_ARCHES
  tsk->computes(d_lab->d_areaFractionFXLabel);
  tsk->computes(d_lab->d_areaFractionFYLabel);
  tsk->computes(d_lab->d_areaFractionFZLabel);
#endif
  tsk->computes(d_lab->d_volFractionLabel);

  //__________________________________
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmcellTypeLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel, gn, 0);
    tsk->computes(d_lab->d_densityMicroINLabel);
  }

  //__________________________________
  if (d_enthalpySolve){
    tsk->requires(Task::OldDW, d_lab->d_enthalpySPLabel, gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationVolqINLabel,  gn, 0);
    tsk->computes(d_lab->d_enthalpySPLabel);
    tsk->computes(d_lab->d_radiationVolqINLabel);
    tsk->computes(d_lab->d_enthalpyTempLabel);
  }

  //__________________________________
  if (d_dynScalarModel) {
    if (d_calScalar){
      tsk->requires(Task::OldDW, d_lab->d_scalarDiffusivityLabel,   gn, 0);
      tsk->computes(d_lab->d_scalarDiffusivityLabel);
    }
    if (d_enthalpySolve){
      tsk->requires(Task::OldDW, d_lab->d_enthalpyDiffusivityLabel, gn, 0);
      tsk->computes(d_lab->d_enthalpyDiffusivityLabel);
    }
  }

  //__________________________________
  tsk->computes(d_lab->d_scalarTempLabel);
  tsk->computes(d_lab->d_densityTempLabel);

  //__________________________________
  if (d_doMMS) {
    tsk->computes(d_lab->d_uFmmsLabel);
    tsk->computes(d_lab->d_vFmmsLabel);
    tsk->computes(d_lab->d_wFmmsLabel);
  }
  //Helper variable
  tsk->computes(d_lab->d_zerosrcVarLabel);

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

      tsk->computes(d_lab->d_oldCCVelocityLabel);
      tsk->computes(d_lab->d_uVelRhoHat_CCLabel);
      tsk->computes(d_lab->d_vVelRhoHat_CCLabel);
      tsk->computes(d_lab->d_wVelRhoHat_CCLabel);
    }


    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,  gn,  0);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel, gn, 0);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_newCCVelocityLabel);
      tsk->computes(d_lab->d_newCCVelMagLabel);
      tsk->computes(d_lab->d_newCCUVelocityLabel);
      tsk->computes(d_lab->d_newCCVVelocityLabel);
      tsk->computes(d_lab->d_newCCWVelocityLabel);
      tsk->computes(d_lab->d_kineticEnergyLabel);
      tsk->computes(d_lab->d_velocityDivergenceLabel);
      tsk->computes(d_lab->d_velDivResidualLabel);
      tsk->computes(d_lab->d_continuityResidualLabel);
    }
    else {
      tsk->modifies(d_lab->d_newCCVelocityLabel);
      tsk->modifies(d_lab->d_newCCVelMagLabel);
      tsk->modifies(d_lab->d_newCCUVelocityLabel);
      tsk->modifies(d_lab->d_newCCVVelocityLabel);
      tsk->modifies(d_lab->d_newCCWVelocityLabel);
      tsk->modifies(d_lab->d_kineticEnergyLabel);
      tsk->modifies(d_lab->d_velocityDivergenceLabel);
      tsk->modifies(d_lab->d_velDivResidualLabel);
      tsk->modifies(d_lab->d_continuityResidualLabel);
    }
    tsk->computes(timelabels->tke_out);

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
    tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel,  gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel,  gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel,  gac, 1);
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

    constSFCXVariable<double> oldUVel;
    constSFCYVariable<double> oldVVel;
    constSFCZVariable<double> oldWVel;
    constSFCXVariable<double> uHatVel_FCX;
    constSFCYVariable<double> vHatVel_FCY;
    constSFCZVariable<double> wHatVel_FCZ;
    CCVariable<Vector> oldCCVel;
    CCVariable<double> uHatVel_CC;
    CCVariable<double> vHatVel_CC;
    CCVariable<double> wHatVel_CC;
    CCVariable<double> divergence;
    CCVariable<double> div_residual;
    CCVariable<double> residual;
    constCCVariable<double> density;
    constCCVariable<double> drhodt;
    constCCVariable<double> div_constraint;

    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    CCVariable<Vector> newCCVel;
    CCVariable<double> newCCVelMag;
    CCVariable<double> newCCUVel;
    CCVariable<double> newCCVVel;
    CCVariable<double> newCCWVel;
    CCVariable<double> kineticEnergy;

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

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      old_dw->get(oldUVel,     d_lab->d_uVelocitySPBCLabel, indx, patch,   gaf, 1);
      old_dw->get(oldVVel,     d_lab->d_vVelocitySPBCLabel, indx, patch,   gaf, 1);
      old_dw->get(oldWVel,     d_lab->d_wVelocitySPBCLabel, indx, patch,   gaf, 1);
      new_dw->get(uHatVel_FCX, d_lab->d_uVelRhoHatLabel,    indx, patch,  gaf, 1);
      new_dw->get(vHatVel_FCY, d_lab->d_vVelRhoHatLabel,    indx, patch,  gaf, 1);
      new_dw->get(wHatVel_FCZ, d_lab->d_wVelRhoHatLabel,    indx, patch,  gaf, 1);

      new_dw->allocateAndPut(oldCCVel,   d_lab->d_oldCCVelocityLabel, indx, patch);
      new_dw->allocateAndPut(uHatVel_CC, d_lab->d_uVelRhoHat_CCLabel, indx, patch);
      new_dw->allocateAndPut(vHatVel_CC, d_lab->d_vVelRhoHat_CCLabel, indx, patch);
      new_dw->allocateAndPut(wHatVel_CC, d_lab->d_wVelRhoHat_CCLabel, indx, patch);

      oldCCVel.initialize(Vector(0.0,0.0,0.0));
      uHatVel_CC.initialize(0.0);
      vHatVel_CC.initialize(0.0);
      wHatVel_CC.initialize(0.0);
      for (int kk = idxLo.z(); kk <= idxHi.z(); ++kk) {
        for (int jj = idxLo.y(); jj <= idxHi.y(); ++jj) {
          for (int ii = idxLo.x(); ii <= idxHi.x(); ++ii) {

            IntVector idx(ii,jj,kk);
            IntVector idxU(ii+1,jj,kk);
            IntVector idxV(ii,jj+1,kk);
            IntVector idxW(ii,jj,kk+1);

            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = oldUVel[idxU];
            double uhat = uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = oldUVel[idx];
            double uhat = uHatVel_FCX[idx];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = oldVVel[idxV];
            double vhat = vHatVel_FCY[idxV];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = oldVVel[idx];
            double vhat = vHatVel_FCY[idx];
            double old_w = cellinfo->bfac[kk] * oldWVel[idx] +
                           cellinfo->tfac[kk] * oldWVel[idxW];
            double what = cellinfo->bfac[kk] * wHatVel_FCZ[idx] +
                          cellinfo->tfac[kk] * wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = oldWVel[idxW];
            double what = wHatVel_FCZ[idxW];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
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

            double old_u = cellinfo->wfac[ii] * oldUVel[idx] +
                           cellinfo->efac[ii] * oldUVel[idxU];
            double uhat = cellinfo->wfac[ii] * uHatVel_FCX[idx] +
                          cellinfo->efac[ii] * uHatVel_FCX[idxU];
            double old_v = cellinfo->sfac[jj] * oldVVel[idx] +
                           cellinfo->nfac[jj] * oldVVel[idxV];
            double vhat = cellinfo->sfac[jj] * vHatVel_FCY[idx] +
                          cellinfo->nfac[jj] * vHatVel_FCY[idxV];
            double old_w = oldWVel[idx];
            double what = wHatVel_FCZ[idx];

            oldCCVel[idx] = Vector(old_u,old_v,old_w);
            uHatVel_CC[idx] = uhat;
            vHatVel_CC[idx] = vhat;
            wHatVel_CC[idx] = what;
          }
        }
      }
    }

    new_dw->get(newUVel, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(newVVel, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(newWVel, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(drhodt,  d_lab->d_filterdrhodtLabel,  indx, patch, gn, 0);
    new_dw->get(density, d_lab->d_densityCPLabel,     indx, patch, gac, 1);
    new_dw->get(div_constraint,
                         d_lab->d_divConstraintLabel, indx, patch, gn, 0);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(newCCVel,      d_lab->d_newCCVelocityLabel,     indx, patch);
      new_dw->allocateAndPut(newCCVelMag,   d_lab->d_newCCVelMagLabel,       indx, patch);
      new_dw->allocateAndPut(newCCUVel,     d_lab->d_newCCUVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(newCCVVel,     d_lab->d_newCCVVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(newCCWVel,     d_lab->d_newCCWVelocityLabel,    indx, patch);
      new_dw->allocateAndPut(kineticEnergy, d_lab->d_kineticEnergyLabel,     indx, patch);
      new_dw->allocateAndPut(divergence,    d_lab->d_velocityDivergenceLabel,indx, patch);
      new_dw->allocateAndPut(div_residual,  d_lab->d_velDivResidualLabel,    indx, patch);
      new_dw->allocateAndPut(residual,      d_lab->d_continuityResidualLabel,indx, patch);
    }
    else {
      new_dw->getModifiable(newCCVel,       d_lab->d_newCCVelocityLabel,      indx, patch);
      new_dw->getModifiable(newCCVelMag,    d_lab->d_newCCVelMagLabel,        indx, patch);
      new_dw->getModifiable(newCCUVel,      d_lab->d_newCCUVelocityLabel,     indx, patch);
      new_dw->getModifiable(newCCVVel,      d_lab->d_newCCVVelocityLabel,     indx, patch);
      new_dw->getModifiable(newCCWVel,      d_lab->d_newCCWVelocityLabel,     indx, patch);
      new_dw->getModifiable(kineticEnergy,  d_lab->d_kineticEnergyLabel,      indx, patch);
      new_dw->getModifiable(divergence,     d_lab->d_velocityDivergenceLabel, indx, patch);
      new_dw->getModifiable(div_residual,   d_lab->d_velDivResidualLabel,     indx, patch);
      new_dw->getModifiable(residual,       d_lab->d_continuityResidualLabel, indx, patch);
    }
    newCCVel.initialize(Vector(0.0,0.0,0.0));
    newCCUVel.initialize(0.0);
    newCCVVel.initialize(0.0);
    newCCWVel.initialize(0.0);
    kineticEnergy.initialize(0.0);
    divergence.initialize(0.0);
    div_residual.initialize(0.0);
    residual.initialize(0.0);


    double total_kin_energy = 0.0;

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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idxU]*newUVel[idxU]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idxV]*newVVel[idxV]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idx]*newWVel[idx])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idxW]*newWVel[idxW])/2.0;
          total_kin_energy += kineticEnergy[idx];
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
          newCCUVel[idx] = new_u;
          newCCVVel[idx] = new_v;
          newCCWVel[idx] = new_w;
          newCCVelMag[idx] = sqrt(new_u*new_u+new_v*new_v+new_w*new_w);
          if (!d_KE_fromFC)
            kineticEnergy[idx] = (new_u*new_u+new_v*new_v+new_w*new_w)/2.0;
          else
            kineticEnergy[idx] = (newUVel[idx]*newUVel[idx]+
                                  newVVel[idx]*newVVel[idx]+
                                  newWVel[idxW]*newWVel[idxW])/2.0;
          total_kin_energy += kineticEnergy[idx];
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

          div_residual[idx] = divergence[idx]-div_constraint[idx]/vol;

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
    new_dw->put(sum_vartype(total_kin_energy), timelabels->tke_out);
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

    constCCVariable<double> newCCUVel;
    constCCVariable<double> newCCVVel;
    constCCVariable<double> newCCWVel;

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(newCCUVel, d_lab->d_newCCUVelocityLabel, indx, patch, gac, 1);
    new_dw->get(newCCVVel, d_lab->d_newCCVVelocityLabel, indx, patch, gac, 1);
    new_dw->get(newCCWVel, d_lab->d_newCCWVelocityLabel, indx, patch, gac, 1);

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

          vorticityX[idx] = 0.5*(newCCWVel[idxV]-newCCWVel[idxyminus])/cellinfo->sns[jj]
                           -0.5*(newCCVVel[idxW]-newCCVVel[idxzminus])/cellinfo->stb[kk];
          vorticityY[idx] = 0.5*(newCCUVel[idxW]-newCCUVel[idxzminus])/cellinfo->stb[kk]
                           -0.5*(newCCWVel[idxU]-newCCWVel[idxxminus])/cellinfo->sew[ii];
          vorticityZ[idx] = 0.5*(newCCVVel[idxU]-newCCVVel[idxxminus])/cellinfo->sew[ii]
                           -0.5*(newCCUVel[idxV]-newCCUVel[idxyminus])/cellinfo->sns[jj];
          vorticity[idx] = sqrt(vorticityX[idx]*vorticityX[idx]+vorticityY[idx]*vorticityY[idx]
                          + vorticityZ[idx]*vorticityZ[idx]);
        }
      }
    }
  }
}


// ****************************************************************************
// Schedule probe data
// ****************************************************************************
void
ExplicitSolver::sched_probeData(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls)
{
  Task* tsk = scinew Task( "ExplicitSolver::probeData", this,
                           &ExplicitSolver::probeData);
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,   gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,       gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_kineticEnergyLabel,  gn, 0);

  if (d_calcVariance) {
    tsk->requires(Task::NewDW, d_lab->d_scalarVarSPLabel,  gn, 0);
  }

  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_tempINLabel,       gn, 0);

  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel, gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_CCLabel,     gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCXLabel,    gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCYLabel,    gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHT_FCZLabel,    gn, 0);

    tsk->requires(Task::NewDW, d_MAlab->totHtFluxXLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxYLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_MAlab->totHtFluxZLabel,   gn, 0);
  }
  sched->addTask(tsk, patches, matls);
}
// ****************************************************************************
// Actual probe data
// ****************************************************************************
void
ExplicitSolver::probeData(const ProcessorGroup* ,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    double time = d_lab->d_sharedState->getElapsedTime();

  // Get the new velocity
    Ghost::GhostType  gn = Ghost::None;
    constSFCXVariable<double> newUVel;
    constSFCYVariable<double> newVVel;
    constSFCZVariable<double> newWVel;
    constCCVariable<double> newintUVel;
    constCCVariable<double> newintVVel;
    constCCVariable<double> newintWVel;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> pressure;
    constCCVariable<double> mixtureFraction;
    constCCVariable<double> kineticEnergy;

    new_dw->get(newUVel,    d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(newVVel,    d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(newWVel,    d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    new_dw->get(newintUVel, d_lab->d_newCCUVelocityLabel, indx, patch, gn, 0);
    new_dw->get(newintVVel, d_lab->d_newCCVVelocityLabel, indx, patch, gn, 0);
    new_dw->get(newintWVel, d_lab->d_newCCWVelocityLabel, indx, patch, gn, 0);

    new_dw->get(density,         d_lab->d_densityCPLabel,    indx, patch, gn, 0);
    new_dw->get(viscosity,       d_lab->d_viscosityCTSLabel, indx, patch, gn, 0);
    new_dw->get(pressure,        d_lab->d_pressurePSLabel,   indx, patch, gn, 0);
    new_dw->get(mixtureFraction, d_lab->d_scalarSPLabel,     indx, patch, gn, 0);
    new_dw->get(kineticEnergy,   d_lab->d_kineticEnergyLabel,indx, patch, gn, 0);

    constCCVariable<double> mixFracVariance;
    if (d_calcVariance) {
      new_dw->get(mixFracVariance, d_lab->d_scalarVarSPLabel, indx, patch, gn, 0);
    }

    constCCVariable<double> gasfraction;
    constCCVariable<double> tempSolid;
    constCCVariable<double> totalHT;
    constSFCXVariable<double> totalHT_FCX;
    constSFCYVariable<double> totalHT_FCY;
    constSFCZVariable<double> totalHT_FCZ;
    constSFCXVariable<double> totHtFluxX;
    constSFCYVariable<double> totHtFluxY;
    constSFCZVariable<double> totHtFluxZ;
    if (d_MAlab) {
      new_dw->get(gasfraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
      new_dw->get(tempSolid,   d_MAlab->integTemp_CCLabel, indx, patch, gn, 0);
      new_dw->get(totalHT,     d_MAlab->totHT_CCLabel,     indx, patch, gn, 0);
      new_dw->get(totalHT_FCX, d_MAlab->totHT_FCXLabel,    indx, patch, gn, 0);
      new_dw->get(totalHT_FCY, d_MAlab->totHT_FCYLabel,    indx, patch, gn, 0);
      new_dw->get(totalHT_FCZ, d_MAlab->totHT_FCZLabel,    indx, patch, gn, 0);
      new_dw->get(totHtFluxX,  d_MAlab->totHtFluxXLabel,   indx, patch, gn, 0);
      new_dw->get(totHtFluxY,  d_MAlab->totHtFluxYLabel,   indx, patch, gn, 0);
      new_dw->get(totHtFluxZ,  d_MAlab->totHtFluxZLabel,   indx, patch, gn, 0);
    }

    constCCVariable<double> temperature;
    if (d_enthalpySolve)
      new_dw->get(temperature, d_lab->d_tempINLabel, indx, patch, gn, 0);

    for (vector<IntVector>::const_iterator iter = d_probePoints.begin();
         iter != d_probePoints.end(); iter++) {

      if (patch->containsCell(*iter)) {
        cerr.precision(10);
        cerr << "for Intvector: " << *iter << endl;
        cerr << "Density: " << density[*iter] << endl;
        cerr << "Viscosity: " << viscosity[*iter] << endl;
        cerr << "Pressure: " << pressure[*iter] << endl;
        cerr << "MixtureFraction: " << mixtureFraction[*iter] << endl;
        if (d_enthalpySolve)
          cerr<<"Gas Temperature: " << temperature[*iter] << endl;
        cerr << "UVelocity: " << newUVel[*iter] << endl;
        cerr << "VVelocity: " << newVVel[*iter] << endl;
        cerr << "WVelocity: " << newWVel[*iter] << endl;
        cerr << "CCUVelocity: " << newintUVel[*iter] << endl;
        cerr << "CCVVelocity: " << newintVVel[*iter] << endl;
        cerr << "CCWVelocity: " << newintWVel[*iter] << endl;
        cerr << "KineticEnergy: " << kineticEnergy[*iter] << endl;
        if (d_calcVariance) {
          cerr << "MixFracVariance: " << mixFracVariance[*iter] << endl;
        }
        if (d_MAlab) {
          cerr.precision(16);
          cerr << "gas vol fraction: " << gasfraction[*iter] << endl;
          cerr << " Solid Temperature at Location " << *iter << " At time " << time << ","<< tempSolid[*iter] << endl;
          cerr << " Total Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT[*iter] << endl;
          cerr << " Total X-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCX[*iter] << endl;
          cerr << " Total Y-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCY[*iter] << endl;
          cerr << " Total Z-Dir Heat Rate at Location " << *iter << " At time " << time << ","<< totalHT_FCZ[*iter] << endl;
          cerr << " Total X-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxX[*iter] << endl;
          cerr << " Total Y-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxY[*iter] << endl;
          cerr << " Total Z-Dir Heat Flux at Location " << *iter << " At time " << time << ","<< totHtFluxZ[*iter] << endl;
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
  // Get the pressure, velocity, scalars, density and viscosity from the
  // old datawarehouse
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
    constCCVariable<double> scalar;
    constCCVariable<double> enthalpy;
    constCCVariable<double> density;
    constCCVariable<double> viscosity;
    constCCVariable<double> scalardiff;
    constCCVariable<double> enthalpydiff;
    constCCVariable<double> reactscalardiff;
    constCCVariable<Vector> ccVel;
    constCCVariable<Vector> old_areaFraction;
#ifdef WASATCH_IN_ARCHES
    constSFCXVariable<double> old_areaFractionFX;
    constSFCYVariable<double> old_areaFractionFY;
    constSFCZVariable<double> old_areaFractionFZ;
#endif 
    constCCVariable<double> old_volFraction;
    constCCVariable<double> old_volq;

    old_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    old_dw->get(scalar,    d_lab->d_scalarSPLabel,      indx, patch, gn, 0);
    old_dw->get(density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    old_dw->get(viscosity, d_lab->d_viscosityCTSLabel,  indx, patch, gn, 0);
    old_dw->get(ccVel,     d_lab->d_newCCVelocityLabel, indx, patch, gn, 0);
    old_dw->get(old_areaFraction, d_lab->d_areaFractionLabel, indx, patch, gn, 0);
#ifdef WASATCH_IN_ARCHES
    old_dw->get(old_areaFractionFX, d_lab->d_areaFractionFXLabel, indx, patch, gn, 0);
    old_dw->get(old_areaFractionFY, d_lab->d_areaFractionFYLabel, indx, patch, gn, 0);
    old_dw->get(old_areaFractionFZ, d_lab->d_areaFractionFZLabel, indx, patch, gn, 0);
#endif
    old_dw->get(old_volFraction, d_lab->d_volFractionLabel, indx, patch, gn, 0);

    if (d_enthalpySolve){
      old_dw->get(enthalpy, d_lab->d_enthalpySPLabel, indx, patch, gn, 0);
      old_dw->get(old_volq, d_lab->d_radiationVolqINLabel, indx, patch, gn, 0);
      CCVariable<double> new_volq;
      new_dw->allocateAndPut(new_volq, d_lab->d_radiationVolqINLabel, indx, patch);
      new_volq.copyData(old_volq); // copy old into new
    }

    if (d_dynScalarModel) {
      if (d_calScalar)
        old_dw->get(scalardiff,      d_lab->d_scalarDiffusivityLabel,     indx, patch, gn, 0);
      if (d_enthalpySolve)
        old_dw->get(enthalpydiff,    d_lab->d_enthalpyDiffusivityLabel,   indx, patch, gn, 0);
    }

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
    SFCXVariable<double> uVelRhoHat_new;
    new_dw->allocateAndPut(uVelRhoHat_new, d_lab->d_uVelRhoHatLabel, indx, patch);
    uVelRhoHat_new.initialize(0.0);     // copy old into new
    SFCYVariable<double> vVelRhoHat_new;
    new_dw->allocateAndPut(vVelRhoHat_new, d_lab->d_vVelRhoHatLabel, indx, patch);
    vVelRhoHat_new.initialize(0.0); // copy old into new
    SFCZVariable<double> wVelRhoHat_new;
    new_dw->allocateAndPut(wVelRhoHat_new, d_lab->d_wVelRhoHatLabel, indx, patch);
    wVelRhoHat_new.initialize(0.0); // copy old into new

    CCVariable<double> scalar_new;
    CCVariable<double> scalar_temp;
    new_dw->allocateAndPut(scalar_new, d_lab->d_scalarSPLabel, indx, patch);
    scalar_new.copyData(scalar); // copy old into new

    CCVariable<Vector> new_areaFraction;
    new_dw->allocateAndPut(new_areaFraction, d_lab->d_areaFractionLabel, indx, patch);
    new_areaFraction.copyData(old_areaFraction); // copy old into new

#ifdef WASATCH_IN_ARCHES
    SFCXVariable<double> new_areaFractionFX; 
    new_dw->allocateAndPut(new_areaFractionFX, d_lab->d_areaFractionFXLabel, indx, patch); 
    new_areaFractionFX.copyData(new_areaFractionFX);

    SFCYVariable<double> new_areaFractionFY; 
    new_dw->allocateAndPut(new_areaFractionFY, d_lab->d_areaFractionFYLabel, indx, patch); 
    new_areaFractionFY.copyData(new_areaFractionFY);

    SFCZVariable<double> new_areaFractionFZ; 
    new_dw->allocateAndPut(new_areaFractionFZ, d_lab->d_areaFractionFZLabel, indx, patch); 
    new_areaFractionFZ.copyData(new_areaFractionFZ);
#endif 

    CCVariable<double> new_volFraction;
    new_dw->allocateAndPut( new_volFraction, d_lab->d_volFractionLabel, indx, patch );
    new_volFraction.copyData( old_volFraction ); // copy old into new

    new_dw->allocateAndPut(scalar_temp, d_lab->d_scalarTempLabel, indx, patch);
    scalar_temp.copyData(scalar); // copy old into new

    constCCVariable<double> reactscalar;
    CCVariable<double> new_reactscalar;
    CCVariable<double> temp_reactscalar;

    CCVariable<double> new_enthalpy;
    CCVariable<double> temp_enthalpy;
    if (d_enthalpySolve) {
      new_dw->allocateAndPut(new_enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
      new_enthalpy.copyData(enthalpy);

      new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyTempLabel, indx, patch);
      temp_enthalpy.copyData(enthalpy);
    }
    CCVariable<double> density_new;
    new_dw->allocateAndPut(density_new, d_lab->d_densityCPLabel, indx, patch);
    density_new.copyData(density); // copy old into new

    CCVariable<double> density_temp;
    new_dw->allocateAndPut(density_temp, d_lab->d_densityTempLabel, indx, patch);
    density_temp.copyData(density); // copy old into new

    CCVariable<double> viscosity_new;
    new_dw->allocateAndPut(viscosity_new, d_lab->d_viscosityCTSLabel, indx, patch);
    viscosity_new.copyData(viscosity); // copy old into new


    CCVariable<double> scalardiff_new;
    CCVariable<double> enthalpydiff_new;
    CCVariable<double> reactscalardiff_new;
    if (d_dynScalarModel) {
      if (d_calScalar) {
        new_dw->allocateAndPut(scalardiff_new,      d_lab->d_scalarDiffusivityLabel, indx, patch);
        scalardiff_new.copyData(scalardiff); // copy old into new
      }
      if (d_enthalpySolve) {
        new_dw->allocateAndPut(enthalpydiff_new,    d_lab->d_enthalpyDiffusivityLabel, indx, patch);
        enthalpydiff_new.copyData(enthalpydiff); // copy old into new
      }
    }

    if (d_doMMS) {
      SFCXVariable<double> uFmms;
      SFCYVariable<double> vFmms;
      SFCZVariable<double> wFmms;

      SFCXVariable<double> ummsLnError;
      SFCYVariable<double> vmmsLnError;
      SFCZVariable<double> wmmsLnError;

      new_dw->allocateAndPut(uFmms, d_lab->d_uFmmsLabel, indx, patch);
      new_dw->allocateAndPut(vFmms, d_lab->d_vFmmsLabel, indx, patch);
      new_dw->allocateAndPut(wFmms, d_lab->d_wFmmsLabel, indx, patch);

      uFmms.initialize(0.0);
      vFmms.initialize(0.0);
      wFmms.initialize(0.0);
    }
    //Reaction rate term for CO2, read in from table
    CCVariable<double> zerosrcVar;
    new_dw->allocateAndPut(zerosrcVar, d_lab->d_zerosrcVarLabel, indx, patch);
    zerosrcVar.initialize(0.0);

    CCVariable<double> scalarBoundarySrc;
    CCVariable<double> enthalpyBoundarySrc;
    SFCXVariable<double> umomBoundarySrc;
    SFCYVariable<double> vmomBoundarySrc;
    SFCZVariable<double> wmomBoundarySrc;

    new_dw->allocateAndPut(scalarBoundarySrc,   d_lab->d_scalarBoundarySrcLabel,  indx, patch);
    new_dw->allocateAndPut(enthalpyBoundarySrc, d_lab->d_enthalpyBoundarySrcLabel,indx, patch);
    new_dw->allocateAndPut(umomBoundarySrc,     d_lab->d_umomBoundarySrcLabel,    indx, patch);
    new_dw->allocateAndPut(vmomBoundarySrc,     d_lab->d_vmomBoundarySrcLabel,    indx, patch);
    new_dw->allocateAndPut(wmomBoundarySrc,     d_lab->d_wmomBoundarySrcLabel,    indx, patch);

    scalarBoundarySrc.initialize(0.0);
    enthalpyBoundarySrc.initialize(0.0);
    umomBoundarySrc.initialize(0.0);
    vmomBoundarySrc.initialize(0.0);
    wmomBoundarySrc.initialize(0.0);
  }
}


// ****************************************************************************
// Schedule data copy for first time step of Multimaterial algorithm
// ****************************************************************************
void
ExplicitSolver::sched_dummySolve(SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls)
{

  d_boundaryCondition->sched_bcdummySolve( sched, patches, matls );

  Task* tsk = scinew Task( "ExplicitSolver::dataCopy",this,
                           &ExplicitSolver::dummySolve);

  Ghost::GhostType  gn = Ghost::None;

  if (d_extraProjection) {
    tsk->requires(Task::OldDW, d_lab->d_pressureExtraProjectionLabel, gn, 0);
    tsk->computes(d_lab->d_pressureExtraProjectionLabel);
  }

  tsk->requires(Task::OldDW, d_lab->d_divConstraintLabel,gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel,   gn, 0);

  // warning **only works for one scalar
  tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);
  tsk->computes(d_lab->d_pressurePSLabel);
  tsk->computes(d_lab->d_uvwoutLabel);
  tsk->computes(d_lab->d_totalflowINLabel);
  tsk->computes(d_lab->d_totalflowOUTLabel);
  tsk->computes(d_lab->d_netflowOUTBCLabel);
  tsk->computes(d_lab->d_denAccumLabel);
  tsk->computes(d_lab->d_scalarEfficiencyLabel);
  tsk->computes(d_lab->d_enthalpyEfficiencyLabel);
  tsk->computes(d_lab->d_carbonEfficiencyLabel);
  tsk->computes(d_lab->d_sulfurEfficiencyLabel);
  tsk->computes(d_lab->d_CO2FlowRateLabel);
  tsk->computes(d_lab->d_SO2FlowRateLabel);
  tsk->computes(d_lab->d_scalarFlowRateLabel);
  tsk->computes(d_lab->d_divConstraintLabel);
  tsk->computes(d_lab->d_densityGuessLabel);

  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Actual Data Copy for first time step of MPMArches
// ****************************************************************************

void
ExplicitSolver::dummySolve(const ProcessorGroup* ,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    // gets for old dw variables
    constCCVariable<double> div;
    constCCVariable<double> pressure;
    CCVariable<double> div_new;
    CCVariable<double> pressure_new;

    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(div,      d_lab->d_divConstraintLabel, indx, patch, gn, 0);
    old_dw->get(pressure, d_lab->d_pressurePSLabel,    indx, patch, gn, 0);

    new_dw->allocateAndPut(pressure_new, d_lab->d_pressurePSLabel,   indx, patch);
    new_dw->allocateAndPut(div_new,     d_lab->d_divConstraintLabel, indx, patch);
    div_new.copyData(div);
    pressure_new.copyData(pressure);

    CCVariable<double> density_guess;
    new_dw->allocateAndPut(density_guess, d_lab->d_densityGuessLabel, indx, patch);

    constCCVariable<double> pressureExtraProjection;
    CCVariable<double> pressureExtraProjection_new;
    if (d_extraProjection) {
      old_dw->get(pressureExtraProjection,
                             d_lab->d_pressureExtraProjectionLabel, indx, patch,  gn, 0);
      new_dw->allocateAndPut(pressureExtraProjection_new,
                             d_lab->d_pressureExtraProjectionLabel, indx, patch);
      pressureExtraProjection_new.copyData(pressureExtraProjection);
    }

    CCVariable<double> pressureNLSource;
    new_dw->allocateAndPut(pressureNLSource, d_lab->d_presNonLinSrcPBLMLabel, indx, patch);
    pressureNLSource.initialize(0.0);

    proc0cout << "ExplicitSolver.cc: DOING DUMMY SOLVE " << endl;

    double uvwout = 0.0;
    double flowIN = 0.0;
    double flowOUT = 0.0;
    double flowOUToutbc = 0.0;
    double denAccum = 0.0;
    double carbon_efficiency = 0.0;
    double sulfur_efficiency = 0.0;
    double scalar_efficiency = 0.0;
    double enthalpy_efficiency = 0.0;
    double CO2FlowRate = 0.0;
    double SO2FlowRate = 0.0;
    double scalarFlowRate = 0.0;

    new_dw->put(delt_vartype(uvwout),         d_lab->d_uvwoutLabel);
    new_dw->put(delt_vartype(flowIN),         d_lab->d_totalflowINLabel);
    new_dw->put(delt_vartype(flowOUT),        d_lab->d_totalflowOUTLabel);
    new_dw->put(delt_vartype(flowOUToutbc),   d_lab->d_netflowOUTBCLabel);
    new_dw->put(delt_vartype(denAccum),       d_lab->d_denAccumLabel);
    new_dw->put(delt_vartype(carbon_efficiency),   d_lab->d_carbonEfficiencyLabel);
    new_dw->put(delt_vartype(sulfur_efficiency),   d_lab->d_sulfurEfficiencyLabel);
    new_dw->put(delt_vartype(enthalpy_efficiency), d_lab->d_enthalpyEfficiencyLabel);
    new_dw->put(delt_vartype(scalar_efficiency),   d_lab->d_scalarEfficiencyLabel);
    new_dw->put(delt_vartype(CO2FlowRate),         d_lab->d_CO2FlowRateLabel);
    new_dw->put(delt_vartype(SO2FlowRate),         d_lab->d_SO2FlowRateLabel);
    new_dw->put(delt_vartype(scalarFlowRate),      d_lab->d_scalarFlowRateLabel);
  }
}
//______________________________________________________________________
//
void
ExplicitSolver::sched_printTotalKE(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::printTotalKE" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname,
                          this, &ExplicitSolver::printTotalKE,
                          timelabels);

  tsk->requires(Task::NewDW, timelabels->tke_out);
  sched->addTask(tsk, patches, matls);
}
//______________________________________________________________________
void
ExplicitSolver::printTotalKE(const ProcessorGroup* ,
                             const PatchSubset* ,
                             const MaterialSubset*,
                             DataWarehouse*,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels)
{
  sum_vartype tke;
  new_dw->get(tke, timelabels->tke_out);
  double total_kin_energy = tke;
  int me = d_myworld->myrank();
  if (me == 0){
     cerr << "Total kinetic energy " <<  total_kin_energy << endl;
  }
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
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,  gn, 0);

  tsk->modifies(d_lab->d_densityTempLabel);
  tsk->modifies(d_lab->d_scalarTempLabel);

  if (d_enthalpySolve){
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel,    gn, 0);
    tsk->modifies(d_lab->d_enthalpyTempLabel);
  }

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
    CCVariable<double> temp_scalar;
    CCVariable<double> temp_reactscalar;
    CCVariable<double> temp_enthalpy;

    new_dw->getModifiable(temp_density, d_lab->d_densityTempLabel,indx, patch);
    new_dw->getModifiable(temp_scalar,  d_lab->d_scalarTempLabel, indx, patch);

    new_dw->copyOut(temp_density,       d_lab->d_densityCPLabel,  indx, patch);
    new_dw->copyOut(temp_scalar,        d_lab->d_scalarSPLabel,   indx, patch);

    if (d_enthalpySolve) {
      new_dw->getModifiable(temp_enthalpy, d_lab->d_enthalpyTempLabel,indx, patch);
      new_dw->copyOut(temp_enthalpy,       d_lab->d_enthalpySPLabel, indx, patch);
    }
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

            densityGuess[currCell] += (*viter)[currCell];

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
            proc0cout << "Negative density guess occured at " << currCell << " with a value of " << densityGuess[currCell] << endl;
            negativeDensityGuess = 1.0;
          }
          else if (densityGuess[currCell] < 0.0 && !(d_noisyDensityGuess)) {
            negativeDensityGuess = 1.0;
          }
        }
      }

      if ( !d_noisyDensityGuess ) proc0cout << "NOTICE: Set <NoisyDenstyGuess> in <ExplicitSolver> to true to check for negative density guesses" << std::endl;

      // This replaces the ->anyArchesPhysicalBC if statement below when new BCs take over
      if ( d_boundaryCondition->isUsingNewBC() ) {

        std::vector<BoundaryCondition::BC_TYPE> bc_types;
        bc_types.push_back( BoundaryCondition::OUTLET );
        bc_types.push_back( BoundaryCondition::PRESSURE );

        d_boundaryCondition->zeroGradientBC( patch, indx, densityGuess, bc_types );

      }

      if (d_boundaryCondition->anyArchesPhysicalBC()) {
        bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
        bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
        bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
        bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
        bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
        bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
        int outlet_celltypeval = d_boundaryCondition->outletCellType();
        int pressure_celltypeval = d_boundaryCondition->pressureCellType();
        IntVector idxLo = patch->getFortranCellLowIndex();
        IntVector idxHi = patch->getFortranCellHighIndex();
        if (xminus) {
          int colX = idxLo.x();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xminusCell(colX-1, colY, colZ);

              if ((cellType[xminusCell] == outlet_celltypeval)||
                  (cellType[xminusCell] == pressure_celltypeval)) {
                densityGuess[xminusCell] = densityGuess[currCell];
              }
            }
          }
        }
        if (xplus) {
          int colX = idxHi.x();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xplusCell(colX+1, colY, colZ);

              if ((cellType[xplusCell] == outlet_celltypeval)||
                  (cellType[xplusCell] == pressure_celltypeval)) {
                densityGuess[xplusCell] = densityGuess[currCell];
              }
            }
          }
        }
        if (yminus) {
          int colY = idxLo.y();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector yminusCell(colX, colY-1, colZ);

              if ((cellType[yminusCell] == outlet_celltypeval)||
                  (cellType[yminusCell] == pressure_celltypeval)) {
                densityGuess[yminusCell] = densityGuess[currCell];
              }
            }
          }
        }
        if (yplus) {
          int colY = idxHi.y();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector yplusCell(colX, colY+1, colZ);

              if ((cellType[yplusCell] == outlet_celltypeval)||
                  (cellType[yplusCell] == pressure_celltypeval)) {
                densityGuess[yplusCell] = densityGuess[currCell];
              }
            }
          }
        }
        if (zminus) {
          int colZ = idxLo.z();
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector zminusCell(colX, colY, colZ-1);

              if ((cellType[zminusCell] == outlet_celltypeval)||
                  (cellType[zminusCell] == pressure_celltypeval)) {
                densityGuess[zminusCell] = densityGuess[currCell];
              }
            }
          }
        }
        if (zplus) {
          int colZ = idxHi.z();
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector zplusCell(colX, colY, colZ+1);

              if ((cellType[zplusCell] == outlet_celltypeval)||
                  (cellType[zplusCell] == pressure_celltypeval)) {
                densityGuess[zplusCell] = densityGuess[currCell];
              }
            }
          }
        }
      }
   // }

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
        if (pc->myrank() == 0)
          proc0cout << "NOTICE: Negative density guess(es) occured. Timestep restart has been requested under this condition by the user. Restarting timestep." << endl;
        new_dw->abortTimestep();
        new_dw->restartTimestep();
      }
      else {
        if (pc->myrank() == 0)
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
// Schedule syncronizing of rho*f with new density
//****************************************************************************
void
ExplicitSolver::sched_syncRhoF(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::syncRhoF" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::syncRhoF,
                          timelabels);

  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,    Ghost::None, 0);

  tsk->modifies(d_lab->d_scalarSPLabel);
  if (d_enthalpySolve)
    tsk->modifies(d_lab->d_enthalpySPLabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually syncronize of rho*f with new density
//****************************************************************************
void
ExplicitSolver::syncRhoF(const ProcessorGroup*,
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

    constCCVariable<double> densityGuess;
    constCCVariable<double> density;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;

    new_dw->get(densityGuess, d_lab->d_densityGuessLabel, indx, patch, Ghost::None, 0);
    new_dw->get(density,      d_lab->d_densityCPLabel,    indx, patch, Ghost::None, 0);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, indx, patch);

    if (d_enthalpySolve){
      new_dw->getModifiable(enthalpy,    d_lab->d_enthalpySPLabel,    indx, patch);
    }

    IntVector idxLo = patch->getExtraCellLowIndex();
    IntVector idxHi = patch->getExtraCellHighIndex();
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);

          if (density[currCell] > 0.0) {
            scalar[currCell] = scalar[currCell] * densityGuess[currCell] /
                             density[currCell];
          if (scalar[currCell] > 1.0)
            scalar[currCell] = 1.0;
          else if (scalar[currCell] < 0.0)
              scalar[currCell] = 0.0;

          if (d_enthalpySolve)
            enthalpy[currCell] = enthalpy[currCell] * densityGuess[currCell] /
                               density[currCell];
          }
        }
      }
    }
  }
}
//****************************************************************************
// Schedule saving of FE copies of variables
//****************************************************************************
void
ExplicitSolver::sched_saveFECopies(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::saveFECopies" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::saveFECopies,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, gn, 0);

  if (d_enthalpySolve)
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, gn, 0);

  tsk->computes(d_lab->d_scalarFELabel);
  if (d_enthalpySolve)
    tsk->computes(d_lab->d_enthalpyFELabel);

  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actually save temp copies here
//****************************************************************************
void
ExplicitSolver::saveFECopies(const ProcessorGroup*,
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

    CCVariable<double> temp_scalar;
    CCVariable<double> temp_reactscalar;
    CCVariable<double> temp_enthalpy;

    new_dw->allocateAndPut(temp_scalar, d_lab->d_scalarFELabel, indx, patch);
    new_dw->copyOut(temp_scalar,        d_lab->d_scalarSPLabel, indx, patch);

    if (d_enthalpySolve) {
      new_dw->allocateAndPut(temp_enthalpy, d_lab->d_enthalpyFELabel,indx, patch);
      new_dw->copyOut(temp_enthalpy,        d_lab->d_enthalpySPLabel,indx, patch);
    }
  }
}
//****************************************************************************
// Schedule computing mms error
//****************************************************************************
void
ExplicitSolver::sched_computeMMSError(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExplicitSolver::computeMMSError" +
                     timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &ExplicitSolver::computeMMSError,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(Task::OldDW, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gn, 0);

  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,   gn, 0);
  //tsk->requires(Task::NewDW, d_lab->d_pressurePSLabel, gn, 0);

  tsk->requires(Task::NewDW, d_lab->d_uFmmsLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vFmmsLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wFmmsLabel, gn, 0);

  tsk->requires(Task::NewDW, d_lab->d_uFmmsLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vFmmsLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wFmmsLabel, gn, 0);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_ummsLnErrorLabel);
    tsk->computes(d_lab->d_vmmsLnErrorLabel);
    tsk->computes(d_lab->d_wmmsLnErrorLabel);
    tsk->computes(d_lab->d_smmsLnErrorLabel);
    tsk->computes(d_lab->d_gradpmmsLnErrorLabel);
  }
  else{
    tsk->modifies(d_lab->d_ummsLnErrorLabel);
    tsk->modifies(d_lab->d_vmmsLnErrorLabel);
    tsk->modifies(d_lab->d_wmmsLnErrorLabel);
    tsk->modifies(d_lab->d_smmsLnErrorLabel);
    tsk->modifies(d_lab->d_gradpmmsLnErrorLabel);
  }

  tsk->computes(timelabels->ummsLnError);
  tsk->computes(timelabels->vmmsLnError);
  tsk->computes(timelabels->wmmsLnError);
  tsk->computes(timelabels->smmsLnError);
  tsk->computes(timelabels->gradpmmsLnError);
  tsk->computes(timelabels->ummsExactSol);
  tsk->computes(timelabels->vmmsExactSol);
  tsk->computes(timelabels->wmmsExactSol);
  tsk->computes(timelabels->smmsExactSol);
  tsk->computes(timelabels->gradpmmsExactSol);

  sched->addTask(tsk, patches, matls);

}
//****************************************************************************
// Actually compute mms error
//****************************************************************************
void
ExplicitSolver::computeMMSError(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels)
{

  proc0cout << "***START of MMS ERROR CALC***" << endl;
  proc0cout << "  Using Error norm = "  << d_mmsErrorType << endl;

  for (int p = 0; p < patches->size(); p++) {

    DataWarehouse* parent_old_dw;
    if (timelabels->recursion) parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    else parent_old_dw = old_dw;

    delt_vartype delT;
    parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;

    constSFCXVariable<double> uFmms;
    constSFCYVariable<double> vFmms;
    constSFCZVariable<double> wFmms;

    constCCVariable<double> scalar;
    //constCCVariable<double> pressure;

    SFCXVariable<double> ummsLnError;
    SFCYVariable<double> vmmsLnError;
    SFCZVariable<double> wmmsLnError;

    CCVariable<double>   smmsLnError;
    CCVariable<double>   gradpmmsLnError;

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(ummsLnError, d_lab->d_ummsLnErrorLabel, indx, patch);
      new_dw->allocateAndPut(vmmsLnError, d_lab->d_vmmsLnErrorLabel, indx, patch);
      new_dw->allocateAndPut(wmmsLnError, d_lab->d_wmmsLnErrorLabel, indx, patch);
      new_dw->allocateAndPut(smmsLnError, d_lab->d_smmsLnErrorLabel, indx, patch);
      new_dw->allocateAndPut(gradpmmsLnError, d_lab->d_gradpmmsLnErrorLabel, indx, patch);
    }
    else {
      new_dw->getModifiable(ummsLnError, d_lab->d_ummsLnErrorLabel,     indx, patch);
      new_dw->getModifiable(vmmsLnError, d_lab->d_vmmsLnErrorLabel,     indx, patch);
      new_dw->getModifiable(wmmsLnError, d_lab->d_wmmsLnErrorLabel,     indx, patch);
      new_dw->getModifiable(smmsLnError, d_lab->d_smmsLnErrorLabel,     indx, patch);
      new_dw->getModifiable(smmsLnError, d_lab->d_gradpmmsLnErrorLabel, indx, patch);
    }

    ummsLnError.initialize(0.0);
    vmmsLnError.initialize(0.0);
    //wmmsLnError.initialize(0.0);
    //smmsLnError.initialize(0.0);
    //gradpmmsLnError.initialize(0.0);

    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    //new_dw->get(pressure,  d_lab->d_pressurePSLabel,    indx, patch, gn, 0);
    new_dw->get(scalar,    d_lab->d_scalarSPLabel,      indx, patch, gn, 0);
    new_dw->get(uFmms,     d_lab->d_uFmmsLabel,         indx, patch, gn, 0);
    new_dw->get(vFmms,     d_lab->d_vFmmsLabel,         indx, patch, gn, 0);
    new_dw->get(wFmms,     d_lab->d_wFmmsLabel,         indx, patch, gn, 0);

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    //getting current time
    // this might require the time shift??
    // what about currenttime = t + dt?
    double time=d_lab->d_sharedState->getElapsedTime();
    time = time + delT;

    proc0cout << "THE CURRENT TIME IN ERROR CALC IS: " << time << endl;

    double pi = acos(-1.0);

    //__________________________________
    //  Scalar: Cell Centered Error Calculation
    double snumeratordiff = 0.0;
    double sdenomexact = 0.0;

    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double mmsvalue = 0.0;
      double testvalue = 0.0;

      if (d_mms == "constantMMS"){
        mmsvalue = phi0;
      }
      else if (d_mms == "almgrenMMS"){
        // not filled in
      }

      // compute the L-2 or L-infinity error.
      if (d_mmsErrorType == "L2"){
        double diff = scalar[c] - mmsvalue;
        snumeratordiff += diff * diff;
        sdenomexact    += mmsvalue*mmsvalue;
        smmsLnError[c]  = pow(diff * diff/(mmsvalue*mmsvalue),1.0/2.0);
      }
      else if (d_mmsErrorType == "Linf"){

        testvalue = Abs(scalar[c] - mmsvalue);

        if (testvalue > snumeratordiff){
          snumeratordiff = testvalue;
        }
        sdenomexact = 1.0;
        smmsLnError[c] = testvalue;
      }
    }


    //__________________________________
    // X-face Error Calculation
    double unumeratordiff = 0.0;
    double udenomexact = 0.0;

    for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      int colX = c.x();
      int colY = c.y();

      double mmsvalue = 0.0;
      double mmsconvvalue = 0.0;
      double testvalue = 0.0;

      if (d_mms == "constantMMS"){
        mmsvalue = cu;
      }
      else if (d_mms == "almgrenMMS"){

        mmsvalue = 1 - amp * cos(2.0*pi*(cellinfo->xu[colX] - time))
          * sin(2.0*pi*(cellinfo->yy[colY] - time))*exp(-2.0*d_viscosity*time);

        mmsconvvalue = 2*(1-amp*cos(2*pi*(cellinfo->xu[colX]-time))*sin(2*pi*(cellinfo->yy[colY]-time))*exp(-2*d_viscosity*time))*amp*sin(2*pi*(cellinfo->xu[colX]-time))*pi*sin(2*pi*(cellinfo->yy[colY]-time))*exp(-2*d_viscosity*time)-2*amp*cos(2*pi*(cellinfo->xu[colX]-time))*cos(2*pi*(cellinfo->yy[colY]-time))*pi*exp(-2*d_viscosity*time)*(1+amp*sin(2*pi*(cellinfo->xu[colX]-time))*cos(2*pi*(cellinfo->yy[colY]-time))*exp(-2*d_viscosity*time));

      }

      if (d_mmsErrorType == "L2"){
        double diff = uVelocity[c] - mmsvalue;
        unumeratordiff += diff * diff;
        udenomexact    += mmsvalue*mmsvalue;
        ummsLnError[c]  = diff*diff;
      }
      else if (d_mmsErrorType == "Linf"){

        testvalue = Abs(uVelocity[c] - mmsvalue);

        if (testvalue > unumeratordiff){
          unumeratordiff = testvalue;
        }
        udenomexact = 1.0;
        ummsLnError[c] = testvalue;
      }
    }

    //__________________________________
    // Y-face Error Calculation
    double vnumeratordiff = 0.0;
    double vdenomexact = 0.0;

    for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      int colX = c.x();
      int colY = c.y();

      double mmsvalue = 0.0;
      double testvalue = 0.0;

      if (d_mms == "constantMMS"){
        mmsvalue = cv;
      }
      else if (d_mms == "almgrenMMS"){

        mmsvalue = 1 + amp * sin(2.0*pi*(cellinfo->xx[colX] - time))
          * cos(2.0*pi*(cellinfo->yv[colY] - time)) * exp(-2.0*d_viscosity*time);

      }

      if (d_mmsErrorType == "L2"){
        double diff = vVelocity[c] - mmsvalue;
        vnumeratordiff += diff*diff;
        vdenomexact    += mmsvalue*mmsvalue;
        vmmsLnError[c]  = diff*diff;
      }
      else if (d_mmsErrorType == "Linf"){
        testvalue = Abs(vVelocity[c] - mmsvalue);

        if (testvalue > vnumeratordiff){
          vnumeratordiff = testvalue;
        }
        vdenomexact = 1.0;
        vmmsLnError[c] = testvalue;
      }
    }

    //__________________________________
    // Z-face Error Calculation
    double wnumeratordiff = 0.0;
    double wdenomexact = 0.0;

    for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      double mmsvalue  = 0.0;
      double testvalue = 0.0;

      if (d_mms == "constantMMS"){
        mmsvalue = cw;
      }
      else if (d_mms == "almgrenMMS"){
        //nothing for now since sine-cos is in x-y plane
      }
      //__________________________________
      if (d_mmsErrorType == "L2"){
        double diff = wVelocity[c] - mmsvalue;
        wnumeratordiff += diff * diff;
        wdenomexact    += mmsvalue*mmsvalue;
        wmmsLnError[c]  = pow(diff * diff/(mmsvalue*mmsvalue),1.0/2.0);

      }
      else if (d_mmsErrorType == "Linf"){
        testvalue = Abs(wVelocity[c] - mmsvalue);

        if (testvalue > wnumeratordiff){
          wnumeratordiff = testvalue;
        }
        wdenomexact = 1.0;
        wmmsLnError[c] = testvalue;
      }
    }


    //__________________________________
    //
    if (d_mmsErrorType == "L2"){
      new_dw->put(sum_vartype(snumeratordiff), timelabels->smmsLnError);
      new_dw->put(sum_vartype(unumeratordiff), timelabels->ummsLnError);
      proc0cout << "putting vnum =" << vnumeratordiff << "into vmmsLnError" << endl;
      new_dw->put(sum_vartype(vnumeratordiff), timelabels->vmmsLnError);
      new_dw->put(sum_vartype(wnumeratordiff), timelabels->wmmsLnError);

      new_dw->put(sum_vartype(sdenomexact),    timelabels->smmsExactSol);
      new_dw->put(sum_vartype(udenomexact),    timelabels->ummsExactSol);
      new_dw->put(sum_vartype(vdenomexact),    timelabels->vmmsExactSol);
      new_dw->put(sum_vartype(wdenomexact),    timelabels->wmmsExactSol);
    }
    else if (d_mmsErrorType == "Linf"){
      new_dw->put(max_vartype(snumeratordiff), timelabels->smmsLnError);
      new_dw->put(max_vartype(unumeratordiff), timelabels->ummsLnError);
      new_dw->put(max_vartype(vnumeratordiff), timelabels->vmmsLnError);
      new_dw->put(max_vartype(wnumeratordiff), timelabels->wmmsLnError);

      new_dw->put(max_vartype(sdenomexact),    timelabels->smmsExactSol);
      new_dw->put(max_vartype(udenomexact),    timelabels->ummsExactSol);
      new_dw->put(max_vartype(vdenomexact),    timelabels->vmmsExactSol);
      new_dw->put(max_vartype(wdenomexact),    timelabels->wmmsExactSol);
    }
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
