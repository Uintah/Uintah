#include <CCA/Components/Arches/DQMOM.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/LU.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/FileNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/Timers/Timers.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

//===========================================================================

using namespace std;
using namespace Uintah;

DQMOM::DQMOM(ArchesLabel* fieldLabels, std::string which_dqmom):
m_fieldLabels(fieldLabels), m_which_dqmom(which_dqmom)
{

  string varname;

  varname = "normB";
  m_normBLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normX";
  m_normXLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normRes";
  m_normResLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normResNormalizedB";
  m_normRedNormalizedLabelB = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normResNormalizedX";
  m_normRedNormalizedLabelX = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "conditionNumber";
  m_conditionNumberLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
}

DQMOM::~DQMOM()
{
  VarLabel::destroy(m_normBLabel);
  VarLabel::destroy(m_normXLabel);
  VarLabel::destroy(m_normResLabel);
  VarLabel::destroy(m_normRedNormalizedLabelB);
  VarLabel::destroy(m_normRedNormalizedLabelX);
  VarLabel::destroy(m_conditionNumberLabel);

 if( m_solverType == "Optimize" ) {
   delete m_AAopt;
 }

}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void DQMOM::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params;

  if ( m_which_dqmom == "unweightedAbs" )
  {
    m_unmweighted = true;
    throw ProblemSetupException("Error: unweightedAbs not functioning.",__FILE__, __LINE__);
  }
  else
    m_unmweighted = false;

#if defined(VERIFY_LINEAR_SOLVER)
  // grab the name of the file containing the test matrices
  ProblemSpecP db_verify_linear_solver = db->findBlock("Verify_Linear_Solver");
  if( !db_verify_linear_solver )
    throw ProblemSetupException("ERROR: DQMOM: You turned on a compiler flag to perform verification of linear solver, but did not put the corresponding tags in your input file.",__FILE__,__LINE__);
  db_verify_linear_solver->require("A", vls_file_A);
  db_verify_linear_solver->require("X", vls_file_X);
  db_verify_linear_solver->require("B", vls_file_B);
  db_verify_linear_solver->require("R", vls_file_R);
  db_verify_linear_solver->require("normR", vls_file_normR);
  db_verify_linear_solver->require("norms", vls_file_norms); // contains determinant, normResid, normResidNormalized, normB, normX (in that order)
  db_verify_linear_solver->require("dimension",  vls_dimension);
  db_verify_linear_solver->getWithDefault("tolerance", vls_tol, 1);
  m_vls_mat_print = false;
#endif

#if defined(VERIFY_AB_CONSTRUCTION)
  // grab the name of the file containing the test matrices
  ProblemSpecP db_verify_ab_construction = db->findBlock("Verify_AB_Construction");
  if( !db_verify_ab_construction )
    throw ProblemSetupException("ERROR: DQMOM: You turned on a compiler flag to perform verification of A and B construction, but did not put the corresponding tags in your input file.",__FILE__,__LINE__);
  db_verify_ab_construction->require("A", vab_file_A);
  db_verify_ab_construction->require("B", vab_file_B);
  db_verify_ab_construction->require("inputs", vab_file_inputs);
  db_verify_ab_construction->require("moments",vab_file_moments);
  db_verify_ab_construction->require("number_environments", vab_N);
  db_verify_ab_construction->require("number_internal_coordinates", vab_N_xi);
  db_verify_ab_construction->getWithDefault("tolerance", vab_tol, 1);
  m_vab_mat_print = false;
#endif

#if defined(DEBUG_MATRICES)
  m_isFirstTimestep = true;
#endif

  unsigned int moments = 0;
  unsigned int index_length = 0;
  moments = 0;

  m_small_normalizer = 1e-8; // "small" number limit for denominator of norm calculations

  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
    temp_moment_index.resize(0);
    db_moments->get("m", temp_moment_index);

    // put moment index into vector of moment indices:
    momentIndexes.push_back(temp_moment_index);

    // keep track of total number of moments
    ++moments;

    index_length = temp_moment_index.size();
  }

  db->getWithDefault("save_moments", m_save_moments, false);
#if defined(VERIFY_AB_CONSTRUCTION) || defined(VERIFY_LINEAR_SOLVER)
  m_save_moments = false;
#endif
  if( m_save_moments ) {
    throw InvalidValue( "ERROR:Save moments is currently disabled.",__FILE__,__LINE__);
    DQMOM::populateMomentsMap(momentIndexes);
  }

  // This block puts the labels in the same order as the input file, so the moment indices match up OK

  DQMOMEqnFactory & eqn_factory = DQMOMEqnFactory::self();
  m_N_ = eqn_factory.get_quad_nodes();

  for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
    string weight_name = "w_qn";
    string node;
    stringstream out;
    out << alpha;
    node = out.str();
    weight_name += node;
    // store equation:
    EqnBase& temp_weightEqnE = eqn_factory.retrieve_scalar_eqn( weight_name );
    DQMOMEqn& temp_weightEqnD = dynamic_cast<DQMOMEqn&>(temp_weightEqnE);
    weightEqns.push_back( &temp_weightEqnD );
    m_w_small = temp_weightEqnD.getSmallClipPlusTol();
    //d_weight_scaling_constant = temp_weightEqnD.getScalingConstant();
  }

  m_N_xi = 0;
  for (ProblemSpecP db_ic = db->findBlock("Ic"); db_ic != nullptr; db_ic = db_ic->findNextBlock("Ic") ) {
    string ic_name;
    vector<string> modelsList;
    db_ic->getAttribute("label", ic_name);
    for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
      string final_name = ic_name + "_qn";
      string node;
      stringstream out;
      out << alpha;
      node = out.str();
      final_name += node;
      // store equation:
      EqnBase& temp_weightedAbscissaEqnE = eqn_factory.retrieve_scalar_eqn( final_name );
      DQMOMEqn& temp_weightedAbscissaEqnD = dynamic_cast<DQMOMEqn&>(temp_weightedAbscissaEqnE);
      weightedAbscissaEqns.push_back( &temp_weightedAbscissaEqnD );
      //d_weighted_abscissa_scaling_constants.push_back( temp_weightedAbscissaEqnD.getScalingConstant() );
    }
    m_N_xi = m_N_xi + 1;
  }

#if defined(VERIFY_AB_CONSTRUCTION)
  m_N_ = vab_N;
  m_N_xi = vab_N_xi;
#endif

  ProblemSpecP db_linear_solver = db->findBlock("LinearSolver");
  if( db_linear_solver ) {

    db_linear_solver->getWithDefault("tolerance", m_solver_tolerance, 1.0e-5);

    db_linear_solver->getWithDefault("maxConditionNumber", m_maxConditionNumber, 1.0e16);

    db_linear_solver->getWithDefault("calcConditionNumber", m_calcConditionNumber, false);

    db_linear_solver->getWithDefault("type", m_solverType, "LU");

    m_optimize = false;
    m_simplest = false;

    if( m_solverType == "Lapack-invert" ) {
      m_useLapack = true;
    } else if( m_solverType == "Lapack-svd" ){
      m_useLapack = true;
      m_calcConditionNumber = true;
    } else if( m_solverType == "LU" ) {
      m_useLapack = false;
    } else if( m_solverType == "Optimize" ) {
      ProblemSpecP db_optimize = db_linear_solver->findBlock("Optimization");
      if(db_optimize){
        m_optimize = true;
        db_optimize->get("Optimal_abscissas",m_opt_abscissas);
        m_AAopt = scinew DenseMatrix((m_N_xi+1)*m_N_,(m_N_xi+1)*m_N_);
        m_AAopt->zero();
        //if(m_unmweighted == true){
          constructAopt_unw( m_AAopt, m_opt_abscissas );
        //} else {
        //  constructAopt( m_AAopt, m_opt_abscissas );
        //}
        m_AAopt->invert();
      }
    } else if( m_solverType == "Simplest" ) {
        m_simplest = true;
    }else {
      string err_msg = "ERROR: Arches: DQMOM: Unrecognized solver type "+m_solverType+": must be 'Lapack-invert', 'Lapack-svd', or 'LU'.\n";
      throw ProblemSetupException(err_msg,__FILE__,__LINE__);
    }
    if( m_calcConditionNumber == true && m_useLapack == false ) {
      string err_msg = "ERROR: Arches: DQMOM: Cannot perform singular value decomposition without using Lapack!\n";
      throw ProblemSetupException(err_msg,__FILE__,__LINE__);
    }

  } else {

    //default for non-inversion
    m_optimize = false;
    m_solver_tolerance = 1.0e-5;
    m_maxConditionNumber = 1.e16;
    m_calcConditionNumber = false;
    m_solverType = "Simplest";
    m_simplest = true;

    //string err_msg = "ERROR: Arches: DQMOM: Could not find block '<LinearSolver>': this block is required for DQMOM. \n";
    //throw ProblemSetupException(err_msg,__FILE__,__LINE__);
  }

  // Check to make sure number of total moments specified in input file is correct
  if ( moments != (m_N_xi+1)*m_N_ && m_solverType != "Simplest") {
    proc0cout << "ERROR:DQMOM:ProblemSetup: You specified " << moments << " moments, but you need " << (m_N_xi+1)*m_N_ << " moments." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moments specified was incorrect!",__FILE__,__LINE__);
  }

  // Check to make sure number of moment indices matches the number of internal coordinates
  if ( index_length != m_N_xi && m_solverType != "Simplest" ) {
    proc0cout << "ERROR:DQMOM:ProblemSetup: You specified " << index_length << " moment indices, but there are " << m_N_xi << " internal coordinates." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moment indices specified was incorrect! Need ",__FILE__,__LINE__);
  }




}

// *************************************************************
// Populate the moments map (contains a label for each moment)
// *************************************************************
/** @details  This method populates a map containing moment labels; the map
  *           is contained in the ArchesLabel class and consists of:
  *           first, vector<int> (the moment index, e.g. <0,0,1>);
  *           second, VarLabel (with name e.g. "moment_001").
  *           This allows you to save out and visualize each moment
  *           being given to the DQMOM class; but only if you specify
  *           <save_moments>true</save_moments>.
  */
void
DQMOM::populateMomentsMap( std::vector<MomentVector> allMoments )
{
  proc0cout << endl;
  proc0cout << "******* Moment Registration ********" << endl;
  proc0cout << endl;

  for( vector<MomentVector>::iterator iAllMoments = allMoments.begin();
       iAllMoments != allMoments.end(); ++iAllMoments ) {
    string name = "moment_";
    std::stringstream out;
    MomentVector thisMoment = (*iAllMoments);
    for( MomentVector::iterator iMomentIndex = thisMoment.begin();
         iMomentIndex != thisMoment.end(); ++iMomentIndex ) {
      out << (*iMomentIndex);
    }
    name += out.str();
    //vector<int> tempMomentVector = (*iAllMoments);
    const VarLabel* tempVarLabel = VarLabel::create(name, CCVariable<double>::getTypeDescription());

    proc0cout << "Creating label for " << name << endl;

    // actually populate the DQMOMMoments map with moment names
    // e.g. moment_001 &c.
    //m_fieldLabels->DQMOMMoments[tempMomentVector] = tempVarLabel;
    m_fieldLabels->DQMOMMoments[thisMoment] = tempVarLabel;
  }
  proc0cout << endl;
}

// **********************************************
// sched_solveLinearSystem
// **********************************************
void
DQMOM::sched_solveLinearSystem( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "DQMOM::solveLinearSystem";
  Task* tsk = scinew Task(taskname, this, &DQMOM::solveLinearSystem, timeSubStep);

  CoalModelFactory& model_factory = CoalModelFactory::self();
  Task::WhichDW which_dw;

  if (timeSubStep == 0) {
    tsk->computes(m_normBLabel);
    tsk->computes(m_normXLabel);
    tsk->computes(m_normResLabel);
    tsk->computes(m_normRedNormalizedLabelB);
    tsk->computes(m_normRedNormalizedLabelX);
    tsk->computes(m_conditionNumberLabel);
    which_dw = Task::NewDW;
  } else {
    tsk->modifies(m_normBLabel);
    tsk->modifies(m_normXLabel);
    tsk->modifies(m_normResLabel);
    tsk->modifies(m_normRedNormalizedLabelB);
    tsk->modifies(m_normRedNormalizedLabelX);
    tsk->modifies(m_conditionNumberLabel);
    which_dw = Task::OldDW;
  }



  for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {

    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();

    // require weights
    tsk->requires( which_dw, tempLabel, Ghost::None, 0 );

    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      tsk->computes(sourceterm_label);
    } else {
      tsk->modifies(sourceterm_label);
    }

    // require model terms
    vector<string> modelsList = (*iEqn)->getModelsList();
    for ( vector<string>::iterator iModels = modelsList.begin(); iModels != modelsList.end(); ++iModels ) {
      ModelBase& model_base = model_factory.retrieve_model(*iModels);
      const VarLabel* model_label = model_base.getModelLabel();
      tsk->requires( Task::NewDW, model_label, Ghost::None, 0 );
    }
  }

  for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();

    // require weighted abscissas
    tsk->requires(which_dw, tempLabel, Ghost::None, 0);

    // compute or modify source terms
    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      tsk->computes(sourceterm_label);
    } else {
      tsk->modifies(sourceterm_label);
    }

    // require model terms
    vector<string> modelsList = (*iEqn)->getModelsList();
    for ( vector<string>::iterator iModels = modelsList.begin(); iModels != modelsList.end(); ++iModels ) {
      ModelBase& model_base = model_factory.retrieve_model(*iModels);
      const VarLabel* model_label = model_base.getModelLabel();
      tsk->requires( Task::NewDW, model_label, Ghost::None, 0 );
    }
  }

  sched->addTask(tsk, level->eachPatch(), m_fieldLabels->d_sharedState->allArchesMaterials());

}

// **********************************************
// Actually solve systems
// **********************************************
void
DQMOM::solveLinearSystem( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const int timeSubStep )
{
  Timers::Simple timer;
  timer.start();

  Timers::Simple tmp_timer;

#if !defined(VERIFY_LINEAR_SOLVER) && !defined(VERIFY_AB_CONSTRUCTION)
  double total_SolveTime = 0.0;
  double total_SVDTime = 0.0;
  double total_AXBConstructionTime = 0.0;

  bool do_iterative_refinement = false;
#if defined(DEBUG_MATRICES)
  double total_FileWriteTime = 0.0;
  bool b_writefile = true;
#endif
#endif


  CoalModelFactory& model_factory = CoalModelFactory::self();

  // patch loop
  for (int p=0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    int archIndex = 0;
    int matlIndex = m_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    DataWarehouse* which_dw;

    CCVariable<double> normB;
    CCVariable<double> normX;
    CCVariable<double> normRes;
    CCVariable<double> normResNormalizedB;
    CCVariable<double> normResNormalizedX;
    CCVariable<double> conditionNumber;
    if ( timeSubStep == 0 ){
      which_dw = new_dw;
      new_dw->allocateAndPut(normB, m_normBLabel, matlIndex, patch);
      new_dw->allocateAndPut(normX, m_normXLabel, matlIndex, patch);
      new_dw->allocateAndPut( normRes, m_normResLabel, matlIndex, patch );
      new_dw->allocateAndPut( normResNormalizedB, m_normRedNormalizedLabelB, matlIndex, patch );
      new_dw->allocateAndPut( normResNormalizedX, m_normRedNormalizedLabelX, matlIndex, patch );
      new_dw->allocateAndPut( conditionNumber, m_conditionNumberLabel, matlIndex, patch );
    } else {
      which_dw = old_dw;
      new_dw->getModifiable(normB, m_normBLabel, matlIndex, patch);
      new_dw->getModifiable(normX, m_normXLabel, matlIndex, patch);
      new_dw->getModifiable( normRes, m_normResLabel, matlIndex, patch );
      new_dw->getModifiable( normResNormalizedB, m_normRedNormalizedLabelB, matlIndex, patch );
      new_dw->getModifiable( normResNormalizedX, m_normRedNormalizedLabelX, matlIndex, patch );
      new_dw->getModifiable( conditionNumber, m_conditionNumberLabel, matlIndex, patch );
    }
    normB.initialize(0.0);
    normX.initialize(0.0);
    normRes.initialize(0.0);
    normResNormalizedB.initialize(0.0);
    normResNormalizedX.initialize(0.0);
    conditionNumber.initialize(0.0);

    // get/allocate weight source term labels
    for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
         iEqn != weightEqns.end(); iEqn++) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double> tempCCVar;
      if ( timeSubStep == 0 ) {
        new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
      }
      tempCCVar.initialize(0.0);
    }

    // get/allocate weighted abscissa source term labels
    for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
         iEqn != weightedAbscissaEqns.end(); ++iEqn) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double> tempCCVar;
      if ( timeSubStep == 0 ){
        new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
      }
      tempCCVar.initialize(0.0);
    }

    // get weights from data warehouse and put into CCVariable
    vector<constCCVarWrapper_withModels> weightCCVars;
    for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
         iEqn != weightEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();

      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper_withModels tempWrapper;
      which_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );

      // for a given weight, get models from data warehouse and put into vector of constCCVarWrapper
      vector<constCCVarWrapper> modelCCVarsVec;
      vector<string> modelsList = (*iEqn)->getModelsList();
      for( vector<string>::iterator iModels = modelsList.begin();
           iModels != modelsList.end(); ++iModels ) {
        ModelBase& model_base = model_factory.retrieve_model(*iModels);

        // instead of using a CCVariable, use a constCCVarWrapper struct
        constCCVarWrapper tempModelWrapper;
        const VarLabel* model_label = model_base.getModelLabel();
        new_dw->get( tempModelWrapper.data, model_label, matlIndex, patch, Ghost::None, 0);
        modelCCVarsVec.push_back(tempModelWrapper);
      }

      // put the vector into the constCCVarWrapper_withModels
      tempWrapper.models = modelCCVarsVec;

      // put the wrapper into a vector
      weightCCVars.push_back(tempWrapper);
    }

    // get weighted abscissas from data warehouse and put into CCVariable
    vector<constCCVarWrapper_withModels> weightedAbscissaCCVars;
    for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
         iEqn != weightedAbscissaEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      std::string eqn_name = (*iEqn)->getEqnName();

      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper_withModels tempWrapper;
      which_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );

      // for a given weighted abscissa, get models from data warehouse and put into vector of constCCVarWrapper
      vector<constCCVarWrapper> modelCCVarsVec;
      vector<string> modelsList = (*iEqn)->getModelsList();
      for( vector<string>::iterator iModels = modelsList.begin();
           iModels != modelsList.end(); ++iModels ) {
        ModelBase& model_base = model_factory.retrieve_model(*iModels);

        // instead of using a CCVariable, use a constCCVarWrapper struct
        constCCVarWrapper tempModelWrapper;
        const VarLabel* model_label = model_base.getModelLabel();
        new_dw->get( tempModelWrapper.data, model_label, matlIndex, patch, Ghost::None, 0);
        modelCCVarsVec.push_back(tempModelWrapper);
      }

      // put the vector into the constCCVarWrapper_withModels
      tempWrapper.models = modelCCVarsVec;

      // put the wrapper into a vector
      weightedAbscissaCCVars.push_back(tempWrapper);

    }

    vector<CCVariable<double>* > Source_weights_weightedAbscissas ;
    // Weight equations:
    for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
         iEqn != weightEqns.end(); ++iEqn ) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if( new_dw->exists(source_label, matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, source_label, matlIndex, patch);
      }
      Source_weights_weightedAbscissas.push_back(tempCCVar);
    }

    // Weighted abscissa equations:
    for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
         iEqn != weightedAbscissaEqns.end(); ++iEqn) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if (new_dw->exists(source_label, matlIndex, patch)) {
        new_dw->getModifiable(*tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, source_label, matlIndex, patch);
      }
      Source_weights_weightedAbscissas.push_back(tempCCVar);
    }

    // Cell iterator
      int dimension = m_N_*(m_N_xi+1);
      vector<double> weights(m_N_);
      vector<double> weight_models(m_N_);
      vector<double> weightedAbscissas(m_N_xi*m_N_);
      vector<double> models(m_N_xi*m_N_);

    for ( CellIterator iter = patch->getCellIterator();
          !iter.done(); ++iter) {
      IntVector c = *iter;


      // get weights in current cell from CCVariable in constCCVarWrapper, store value in vector
      int jj=0;
      for( vector<constCCVarWrapper_withModels>::iterator iter = weightCCVars.begin();
           iter != weightCCVars.end(); ++iter ) {
        double temp_value = (iter->data)[c];
        weights[jj]=temp_value;

        // now sum the model terms for this weight
        double runningsum = 0;
        for( vector<constCCVarWrapper>::iterator iM = iter->models.begin();
             iM != iter->models.end(); ++iM ) {
          double temp_model_value = (iM->data)[c];
          runningsum += temp_model_value;
        }

        weight_models[jj]=runningsum;
        jj++;
      }


      // get weighted abscissas in current cell from CCVariable in constCCVarWrapper, store value in vector
      jj=0;
      for( vector<constCCVarWrapper_withModels>::iterator iter = weightedAbscissaCCVars.begin();
           iter != weightedAbscissaCCVars.end(); ++iter ) {
        double temp_value = (iter->data)[c];
        weightedAbscissas[jj]=temp_value;

        // now sum the model terms for this weighted abscissa
        double runningsum = 0;
        for( vector<constCCVarWrapper>::iterator iM = iter->models.begin();
             iM != iter->models.end(); ++iM ) {
          double temp_model_value = (iM->data)[c];
          runningsum += temp_model_value;
        }

        models[jj]=runningsum;
        jj++;
      }

#if !defined(VERIFY_LINEAR_SOLVER) && !defined(VERIFY_AB_CONSTRUCTION)


      if (m_optimize == true) {
        ColumnMatrix* BB = scinew ColumnMatrix( dimension );
        ColumnMatrix* XX = scinew ColumnMatrix( dimension );
        BB->zero();

	tmp_timer.reset( true );
        //if(m_unmweighted == true){
          constructBopt_unw( BB, m_opt_abscissas, models );
        //} else {
        //  constructBopt( BB, weights, m_opt_abscissas, models );
        //}
	tmp_timer.stop();

        total_AXBConstructionTime += tmp_timer().seconds();

	tmp_timer.reset( true );
        Mult( (*XX), (*m_AAopt), (*BB) );
	tmp_timer.stop();

        total_SolveTime += tmp_timer().seconds();

        int z=0; // equation loop counter

#if defined(DEBUG_MATRICES)

        if( pc->myrank() == 0 ) {
          if( b_writefile ) {
            char filename[28];
            int currentTimeStep;
            if( m_isFirstTimestep ) {
              currentTimeStep = 0;
            } else {
              currentTimeStep = m_fieldLabels->d_sharedState->getCurrentTopLevelTimeStep();
            }
            int sizeofit;
            ofstream oStream;

	    tmp_timer.reset( true );

            // write Aopt
            sizeofit = sprintf( filename, "Aopt_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              for( int iCol = 0; iCol < dimension; ++iCol ) {
                oStream << scientific << setw(20) << setprecision(20) << " " << (*m_AAopt)[iRow][iCol];
              }
              oStream << endl;
            }
            oStream.close();

            // write X matrix
            sizeofit = sprintf( filename, "X_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              oStream << scientific << setw(20) << setprecision(20) << " " << (*XX)[iRow] << endl;
            }
            oStream.close();

            // write B matrix
            sizeofit = sprintf( filename, "B_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              oStream << scientific << setw(20) << setprecision(20) << " " << (*BB)[iRow] << endl;
            }
            oStream.close();

	    tmp_timer.stop();
            total_FileWriteTime += tmp_timer.seconds();
          }
          b_writefile = false;
        }

#endif

        // Weight equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
             iEqn != weightEqns.end(); ++iEqn ) {
          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = (*XX)[z];
          }
          ++z;
        }
        // Weighted abscissa equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
             iEqn != weightedAbscissaEqns.end(); ++iEqn) {
          // Make sure several critera are met for an acceptable solution
          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = (*XX)[z];
          }
          ++z;
        }

        delete BB;
        delete XX;

      } else if( m_simplest == true ){

	// tmp_timer.reset( true );
	// Something to time - currently nothing constructed
	// tmp_timer.stop();

	// total_AXBConstructionTime += timer().seconds();

	// tmp_timer.reset( true );
	// Something to time - currently nothing solved
	// tmp_timer.stop();

	// total_SolveTime += timer().seconds();

	int z=0; // equation loop counter
	int z2=0; // equation loop counter

	// Weight equations:
	for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
	     iEqn != weightEqns.end(); ++iEqn ) {
	  (*(Source_weights_weightedAbscissas[z]))[c] = weight_models[z];
	  ++z;
	}
          // Weighted abscissa equations:
	for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
	     iEqn != weightedAbscissaEqns.end(); ++iEqn) {
	  (*(Source_weights_weightedAbscissas[z]))[c] = models[z2];
	  ++z;
	  ++z2;
	}
      } else if( m_useLapack == false ) {

        ///////////////////////////////////////////////////////
        // Use the LU solver

        LU A ( dimension );
        vector<double> B( dimension, 0.0 );
        vector<double> Xdoub( dimension, 0.0 );
        vector<long double> Xlong( dimension, 0.0 );
        vector<double> Resid( dimension, 0.0 );

	tmp_timer.reset( true );
        constructLinearSystem( A, B, weights, weightedAbscissas, models);
	tmp_timer.stop();

        total_AXBConstructionTime += tmp_timer().seconds();

        // Save original A before decomposition into LU
        LU Aorig( A );

        // Solve linear system
	tmp_timer.reset( true );
        A.decompose();
        A.back_subs( &B[0], &Xdoub[0] );
	tmp_timer.stop();

        total_SolveTime += tmp_timer().seconds();

        for( unsigned int j=0; j < Xdoub.size(); ++j ) {
          Xlong[j] = Xdoub[j];
        }

        conditionNumber[c] = 0.0;

        // --------------------------------------------
        // If no solution to linear system...
        if( A.isSingular() ) {
          //proc0cout << "WARNING: Arches: DQMOM: A is singular at cell c = " << c << endl;

          // set solution vector = 0
          vector<double>::iterator iXd = Xdoub.begin();
          for( vector<long double>::iterator iXl = Xlong.begin();
               iXl != Xlong.end(); ++iXl, ++iXd ) {
            (*iXl) = 0.0;
            (*iXd) = 0.0;
          }

          // set residual vector = 0
          for( vector<double>::iterator iR = Resid.begin();
               iR != Resid.end(); ++iR ) {
            (*iR) = 0.0;
          }

        // -------------------------------------------------
        // If there is a solution to linear system...
        } else {

	  tmp_timer.reset( true );
          if( do_iterative_refinement ) {
            A.iterative_refinement( Aorig, &B[0], &Xdoub[0], &Xlong[0] );
          }
	  tmp_timer.stop();

          total_SolveTime += tmp_timer().seconds();

          // get residual vector
          Aorig.getResidual( &B[0], &Xlong[0], &Resid[0] );

          // find norm of residual
          double the_normR = A.getNorm( &Resid[0], 0 );
          normRes[c] = the_normR;

          // find norm of residual vector divided by the norm of B
          // R = (B - AX)/(norm(B))
          vector<double> ResidNormalizedB = Resid;
          for( int ii = 0; ii<dimension; ++ii ) {
            if( fabs(B[ii]) > m_small_normalizer) {
              ResidNormalizedB[ii] = Resid[ii] / B[ii];
              // otherwise... well, we'll leave it alone otherwise
            }
          }
          normResNormalizedB[c] = A.getNorm( &ResidNormalizedB[0], 0 );

          // find norm of residual vector divided by the norm of X
          // R = (B - AX)/(norm(X))
          vector<double> ResidNormalizedX = Resid;
          for( int ii=0; ii<dimension; ++ii ) {
            if( fabs(Xlong[ii]) > m_small_normalizer ) {
              ResidNormalizedX[ii] = fabs( Resid[ii] / Xlong[ii] );
              // otherwise... we'll leave it alone
            }
          }
          normResNormalizedX[c] = A.getNorm( &ResidNormalizedX[0], 0 );

          // find norm of RHS vector
          normB[c] = A.getNorm( &B[0], 0 );

          // find norm of solution vector
          normX[c] = A.getNorm( &Xlong[0], 0 );

        }

#if defined(DEBUG_MATRICES)
        if( pc->myrank() == 0 ) {
          if( b_writefile ) {
            char filename[28];
            int currentTimeStep;
            if( m_isFirstTimestep ) {
              currentTimeStep = 0;
            } else {
              currentTimeStep = m_fieldLabels->d_sharedState->getCurrentTopLevelTimeStep();
            }
            int sizeofit;
            ofstream oStream;

	    tmp_timer.reset( true );

            // write A matrix to file:
            sizeofit = sprintf( filename, "A_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              for( int iCol = 0; iCol < dimension; ++iCol ) {
                oStream << " " << Aorig(iRow,iCol);
              }
              oStream << endl;
            }
            oStream.close();

            // Write X (solution vector) to file:
            sizeofit = sprintf( filename, "X_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( vector<long double>::iterator iX = Xlong.begin();
                 iX != Xlong.end(); ++iX) {
              oStream << (*iX) << endl;
            }
            oStream.close();

            // write B matrix to file:
            sizeofit = sprintf( filename, "B_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( vector<double>::iterator iB = B.begin(); iB != B.end(); ++iB ) {
              oStream << (*iB) << endl;
            }
            oStream.close();

            // write Resid vector to file:
            sizeofit = sprintf( filename, "R_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( vector<double>::iterator iR = Resid.begin(); iR != Resid.end(); ++iR ) {
              oStream << (*iR) << endl;
            }
            oStream.close();

	    tmp_timer.stop();
            total_FileWriteTime += tmp_timer.seconds();
          }
          b_writefile = false;
        }

#endif //debug matrices

        int z=0; // equation loop counter

        // check "acceptable solution" criteria, and assign solution values to source terms

        // Weight equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
             iEqn != weightEqns.end(); ++iEqn ) {

          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else if( fabs(normResNormalizedX[c]) > m_solver_tolerance ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( std::isnan( Xlong[z] ) ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( m_calcConditionNumber == true && conditionNumber[c] > m_maxConditionNumber ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
            conditionNumber[c] = -1.0;
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = Xlong[z];
          }
          ++z;
        }

        // Weighted abscissa equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
             iEqn != weightedAbscissaEqns.end(); ++iEqn) {

          // Make sure several critera are met for an acceptable solution
          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else if(  fabs(normResNormalizedX[c]) > m_solver_tolerance ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( std::isnan( Xlong[z] ) ){
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( m_calcConditionNumber == true && conditionNumber[c] > m_maxConditionNumber ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
            conditionNumber[c] = -1.0;
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = Xlong[z];
          }
          ++z;
        }

      } else {

        ///////////////////////////////////////////////////////
        // Use the DenseMatrix solver (which uses LAPACK)

        DenseMatrix* AA = scinew DenseMatrix( dimension, dimension );
        ColumnMatrix* BB = scinew ColumnMatrix( dimension );
        ColumnMatrix* XX = scinew ColumnMatrix( dimension );
        ColumnMatrix* RR = scinew ColumnMatrix( dimension );

        AA->zero();
        BB->zero();

        do_iterative_refinement = false;

	tmp_timer.reset( true );
        constructLinearSystem( AA, BB, weights, weightedAbscissas, models );
	tmp_timer.stop();

        total_AXBConstructionTime += tmp_timer().seconds();

        // save original A before solving
        DenseMatrix* AAorig = AA->clone();

        double conditionNumber_ = 0.0;

        if( m_solverType == "Lapack-svd" ) {

          DenseMatrix* AAsvd = AA->clone();

          // create rr and cc for singular values SparseRowMatrix
          int *cols = scinew int[dimension];
          int *rows = scinew int[dimension+1];
          double *a = scinew double[dimension];

          DenseMatrix* U = scinew DenseMatrix( dimension, dimension );
          SparseRowMatrix* S = scinew SparseRowMatrix( dimension, dimension, rows, cols, dimension, a); // makes an identity matrix
          DenseMatrix* V = scinew DenseMatrix( dimension, dimension );

          DenseMatrix* Ut = scinew DenseMatrix( dimension, dimension );
          DenseMatrix* Sinv = scinew DenseMatrix( dimension, dimension );
          DenseMatrix* Vt = scinew DenseMatrix( dimension, dimension );
          ColumnMatrix* XXsvd = scinew ColumnMatrix( dimension );
          ColumnMatrix* XXsvd2 = scinew ColumnMatrix( dimension );
          Sinv->zero();

	  tmp_timer.reset( true );
          AAsvd->svd( *U, *S, *V );
	  tmp_timer.stop();
          total_SolveTime += tmp_timer().seconds();

          conditionNumber_ = (S->a[0]/S->a[dimension-1]);

          for(int kk=0;kk<dimension;kk++){
            if(S->a[kk] > 1e-20) {
              Sinv->put(kk,kk,1./S->a[kk]);
            } else {
              Sinv->put(kk,kk,0.);
            }
          }

          U->gettranspose(*Ut);
          V->gettranspose(*Vt);

          Mult( *XXsvd, *Ut, *BB );
          Mult( *XXsvd2, *Sinv, *XXsvd );
          Mult( *XX, *Vt, *XXsvd2 );

#if defined(DEBUG_MATRICES)
          if( pc->myrank() == 0 ) {
            if( b_writefile ) {
              char filename[28];
              int currentTimeStep;
              if( m_isFirstTimestep ) {
                currentTimeStep = 0;
              } else {
                currentTimeStep = m_fieldLabels->d_sharedState->getCurrentTopLevelTimeStep();
              }
              int sizeofit;
              ofstream oStream;

	      tmp_timer.reset( true );

              // write U, S, and V matrices to file
              sizeofit = sprintf( filename, "U_%.2d.mat", currentTimeStep );
              oStream.open(filename);
              for( int iRow = 0; iRow < dimension; ++iRow ) {
                for( int iCol = 0; iCol < dimension; ++iCol ) {
                  oStream << scientific << setw(20) << setprecision(20) << " " << (*U)[iRow][iCol];
                }
                oStream << endl;
              }
              oStream.close();

              sizeofit = sprintf( filename, "V_%.2d.mat", currentTimeStep );
              oStream.open(filename);
              for( int iRow = 0; iRow < dimension; ++iRow ) {
                for( int iCol = 0; iCol < dimension; ++iCol ) {
                  oStream << scientific << setw(20) << setprecision(20) << " " << (*V)[iRow][iCol];
                }
                oStream << endl;
              }
              oStream.close();

              sizeofit = sprintf( filename, "S_%.2d.mat", currentTimeStep );
              oStream.open(filename);
              for( int iRow = 0; iRow < dimension; ++iRow ) {
                oStream << scientific << setw(20) << setprecision(20) << " " << S->a[iRow] << endl;
              }
              oStream.close();

	      tmp_timer.stop();
              total_FileWriteTime += tmp_timer().seconds();
            }
          }
#endif

          delete AAsvd;
          delete[] cols;
          delete[] rows;

          delete U;
          delete V;
          delete S;

          delete Ut;
          delete Sinv;
          delete Vt;
          delete XXsvd;
          delete XXsvd2;

        } else if (m_solverType == "Lapack-invert") {

          if( m_calcConditionNumber ) {
            DenseMatrix* AAsvd = AA->clone();

            // create rr and cc for singular values SparseRowMatrix
            int *cols = scinew int[dimension];
            int *rows = scinew int[dimension+1];
            double *a = scinew double[dimension];

            DenseMatrix* U = scinew DenseMatrix( dimension, dimension );
            SparseRowMatrix* S = scinew SparseRowMatrix( dimension, dimension, rows, cols, dimension, a); // makes an identity matrix
            DenseMatrix* V = scinew DenseMatrix( dimension, dimension );

	    tmp_timer.reset( true );
            AAsvd->svd( *U, *S, *V );
	    tmp_timer.stop();

            total_SVDTime += tmp_timer().seconds();

            conditionNumber_ = (S->a[0]/S->a[dimension-1]);

            delete AAsvd;
            delete[] cols;
            delete[] rows;

            delete U;
            delete V;
            delete S;
          }

          // Solve linear system
	  tmp_timer.reset( true );
          bool success = AA->invert();
          if (!success) {
            //proc0cout << "WARNING: Arches: DQMOM: A is singular at cell c = " << c << endl;
          }
          Mult( (*XX), (*AA), (*BB) );
	  tmp_timer.stop();

          total_SolveTime += tmp_timer().seconds();
        }

        conditionNumber[c] = conditionNumber_;

        // get residual vector
        int t_flops, t_memrefs;
        AAorig->mult( (*XX), (*RR), t_flops, t_memrefs );

        for( int yy=0; yy<dimension; ++yy ) {
          double temp = (*RR)[yy] - (*BB)[yy];
          (*RR)[yy] = temp;
        }

        // find norm of residual
        double this_normRes = 0;
        for( int ii=0; ii<dimension; ++ii ) {
          if( fabs((*RR)[ii]) > this_normRes ) {
            this_normRes = fabs( (*RR)[ii] );
          }
        }
        normRes[c] = this_normRes;

        // norm of residual divided by norm of B
        //    Don't divide by small B's!
        //    Otherwise you'll be dividing a small residual by a small B, so the norm will be large
        //    This causes everything to have nonzero source terms, for no good reason
        //    When B is small, just use the non-normalized residual.
        double this_normResNormalizedB = 0;
        for( int ii=0; ii<dimension; ++ii ) {
          if( fabs( (*BB)[ii] ) > m_small_normalizer ) {
            if( fabs( (*RR)[ii] / (*BB)[ii] ) > this_normResNormalizedB ) {
              this_normResNormalizedB = fabs((*RR)[ii] / (*BB)[ii]);
            }
          } else {
            if( fabs( (*RR)[ii] ) > this_normResNormalizedB ) {
              this_normResNormalizedB = fabs((*RR)[ii]);
            }
          }
        }
        normResNormalizedB[c] = this_normResNormalizedB;

        // norm of residual divided by norm of X
        //    Don't divide by small X's!
        //    Otherwise you'll be dividing a small residual by a small X, so the norm will be large
        //    This causes everything to have nonzero source terms, for no good reason
        //    When X is small, just use the non-normalized residual.
        double this_normResNormalizedX = 0;
        for( int ii=0; ii<dimension; ++ii ) {
          if( fabs( (*XX)[ii] ) > m_small_normalizer ) {
            if( fabs( (*RR)[ii] / (*XX)[ii] ) > this_normResNormalizedX ) {
              this_normResNormalizedX = fabs((*RR)[ii] / (*XX)[ii]);
            }
          } else {
            if( fabs( (*RR)[ii] ) > this_normResNormalizedX ) {
              this_normResNormalizedX = fabs((*RR)[ii]);
            }
          }
        }
        normResNormalizedX[c] = this_normResNormalizedX;

        // norm of B
        double this_normB = 0;
        for( int ii=0; ii<dimension; ++ii ) {
          if( fabs((*BB)[ii]) > this_normB ) {
            this_normB = fabs( (*BB)[ii] );
          }
        }
        normB[c] = this_normB;

        // norm of X
        double this_normX = 0;
        for( int ii=0; ii<dimension; ++ii ) {
          if( fabs((*XX)[ii]) > this_normX ) {
            this_normX = fabs( (*XX)[ii] );
          }
        }
        normX[c] = this_normX;

#if defined(DEBUG_MATRICES)
        if( pc->myrank() == 0 ) {
          if( b_writefile ) {
            char filename[28];
            int currentTimeStep;
            if( m_isFirstTimestep ) {
              currentTimeStep = 0;
            } else {
              currentTimeStep = m_fieldLabels->d_sharedState->getCurrentTopLevelTimeStep();
            }
            int sizeofit;
            ofstream oStream;

	    tmp_timer.reset( true );

            // write A matrix to file
            sizeofit = sprintf( filename, "A_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              for( int iCol = 0; iCol < dimension; ++iCol ) {
                oStream << scientific << setw(20) << setprecision(20) << " " << (*AAorig)[iRow][iCol];
              }
              oStream << endl;
            }
            oStream.close();

            if( m_solverType == "Lapack-invert" ) {
              // write inv(A) matrix to file
              sizeofit = sprintf( filename, "Ainv_%.2d.mat", currentTimeStep );
              oStream.open(filename);
              for( int iRow = 0; iRow < dimension; ++iRow ) {
                for( int iCol = 0; iCol < dimension; ++iCol ) {
                  oStream << scientific << setw(20) << setprecision(20) << " " << (*AA)[iRow][iCol];
                }
                oStream << endl;
              }
              oStream.close();
            }

            // write X vector to file
            sizeofit = sprintf( filename, "X_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              oStream << scientific << setw(20) << setprecision(20) << (*XX)[iRow] << endl;
            }
            oStream.close();

            // write B vector to file
            sizeofit = sprintf( filename, "B_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              oStream << scientific << setw(20) << setprecision(20) << (*BB)[iRow] << endl;
            }
            oStream.close();

            // write residual vector to file
            sizeofit = sprintf( filename, "R_%.2d.mat", currentTimeStep );
            oStream.open(filename);
            for( int iRow = 0; iRow < dimension; ++iRow ) {
              oStream << scientific << setw(20) << setprecision(20) << (*RR)[iRow] << endl;
            }
            oStream.close();

	    tmp_timer.stop();
            total_FileWriteTime += tmp_timer().seconds();
          }
          b_writefile = false;
        }
#endif

        // check "acceptable solution" criteria, and assign solution values to source terms

        int z=0; // equation loop counter

        // Weight equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
             iEqn != weightEqns.end(); ++iEqn ) {

          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else if( fabs(normResNormalizedX[c]) > m_solver_tolerance ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( std::isnan( (*XX)[z] ) ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( m_calcConditionNumber == true && conditionNumber[c] > m_maxConditionNumber ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
            conditionNumber[c] = -1.0;
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = (*XX)[z];
          }
          ++z;
        }

        // Weighted abscissa equations:
        for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
             iEqn != weightedAbscissaEqns.end(); ++iEqn) {

          // Make sure several critera are met for an acceptable solution
          if (z >= dimension ) {
            stringstream err_msg;
            err_msg << "ERROR: Arches: DQMOM: Trying to access solution of AX=B system, but had array out of bounds! Accessing element " << z << " of " << dimension << endl;
            throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
          } else if( fabs(normResNormalizedX[c]) > m_solver_tolerance ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( std::isnan( (*XX)[z] ) ){
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
          } else if( m_calcConditionNumber == true && conditionNumber[c] > m_maxConditionNumber ) {
            (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
            conditionNumber[c] = -1.0;
          } else {
            (*(Source_weights_weightedAbscissas[z]))[c] = (*XX)[z];
          }
          ++z;
        }

        delete AA;
        delete BB;
        delete XX;
        delete RR;
        delete AAorig;

      }//end lapack/LU

#else

      // doing verification, so don't need to update transport equations (source terms = 0)

      normB[c] = 0.0;
      normX[c] = 0.0;
      normRes[c] = 0.0;
      normResNormalizedB[c] = 0.0;
      normResNormalizedX[c] = 0.0;

      // set weight transport eqn source terms equal to zero
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
           iEqn != weightEqns.end(); iEqn++) {
        (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
      }

      // set weighted abscissa transport eqn source terms equal to results
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        (*(Source_weights_weightedAbscissas[z]))[c] = 0.0;
      }

#endif //end if not verifying

    }//end for cells
    // to release the pointers
    for(unsigned int i=0; i<Source_weights_weightedAbscissas.size(); ++i ){
      delete Source_weights_weightedAbscissas[i];
    }
  }//end per patch

  // ---------------------------------------------
  // Verification Procedure:
  // ---------------------------------------------

#if defined(VERIFY_LINEAR_SOLVER)
  verifyLinearSolver();
#endif

#if defined(VERIFY_AB_CONSTRUCTION)
  verifyABConstruction();
#endif

  // ---------------------------------------------

#if defined(DEBUG_MATRICES)
  m_isFirstTimestep = false;
#endif


  proc0cout << "Time in DQMOM::solveLinearSystem: " << timer().seconds() << " seconds \n";

#if !defined(VERIFY_AB_CONSTRUCTION) && !defined(VERIFY_LINEAR_SOLVER)
#if defined(DEBUG_MATRICES)
  proc0cout << "    Time for file write: " << total_FileWriteTime << " seconds\n";
#endif

  string msg = "    Time for AX=B construction: ";
  proc0cout << msg << total_AXBConstructionTime << " seconds\n";

  if( m_solverType == "Lapack-invert" ) {
    proc0cout << "    Time for Lapack inversion-multiplication: " << total_SolveTime << " seconds\n";
    if( m_calcConditionNumber ) {
      proc0cout << "    Time for calculation of condition number: " << total_SVDTime << " seconds\n";
    }

  } else if( m_solverType == "Lapack-svd" ) {
    proc0cout << "    Time for Lapack singular value decomposition and solution: " << total_SolveTime << " seconds\n";

  } else if( m_solverType == "LU" ) {
    proc0cout << "    Time for Crout's Method solution: " << total_SolveTime << " seconds\n";

  }else if( m_solverType == "Optimize"){
    proc0cout << " Time for Optimized Method solution: " << total_SolveTime << "seconds\n";
  }else if( m_solverType == "Simplest"){
      proc0cout << "    Time for Simplest Method solution: " << total_SolveTime << "seconds\n";
  }

#endif
}



// **********************************************
// Construct A and B matrices for DQMOM
// **********************************************
void
DQMOM::constructLinearSystem( LU             &A,
                              vector<double> &B,
                              vector<double> &weights,
                              vector<double> &weightedAbscissas,
                              vector<double> &models,
                              int             verbosity)
{
      // FIXME:
      // This construction process needs to be using m_w_small to check for small weights!
  // construct AX=B
  for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
    MomentVector thisMoment = momentIndexes[k];

    // weights
    for ( unsigned int alpha = 0; alpha < m_N_; ++alpha) {
      double prefixA = 1;
      double productA = 1;
      for ( unsigned int i = 0; i < thisMoment.size(); ++i) {
        if (weights[alpha] != 0) {
          // Appendix C, C.9 (A1 matrix)
          prefixA = prefixA - (thisMoment[i]);
          double base = weightedAbscissas[i*(m_N_)+alpha] / weights[alpha];
          double exponent = thisMoment[i];
          productA = productA*( pow(base, exponent) );
        } else {
          prefixA = 0;
          productA = 0;
        }
      }
      A(k,alpha)=prefixA*productA;
      if(verbosity == 1)
        proc0cout << "Setting A(" << k << "," << alpha << ") = " << prefixA*productA << endl;
    } //end weights sub-matrix

    // weighted abscissas
    double totalsumS = 0;
    for( unsigned int j = 0; j < m_N_xi; ++j ) {
      double prefixA    = 1;
      double productA   = 1;

      double prefixS    = 1;
      double productS   = 1;
      double modelsumS  = 0;

      double quadsumS = 0;
      for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
        if (weights[alpha] == 0) {
          prefixA = 0;
          prefixS = 0;
          productA = 0;
          productS = 0;
        } else if ( weightedAbscissas[j*(m_N_)+alpha] == 0 && thisMoment[j] == 0) {
          //FIXME:
          // both prefixes contain 0^(-1)
          prefixA = 0;
          prefixS = 0;
        } else {
          // Appendix C, C.11 (A_j+1 matrix)
          double base = weightedAbscissas[j*(m_N_)+alpha] / weights[alpha];
          double exponent = thisMoment[j] - 1;
          //prefixA = (thisMoment[j])*( pow((weightedAbscissas[j*(m_N_)+alpha]/weights[alpha]),(thisMoment[j]-1)) );
          prefixA = (thisMoment[j])*(pow(base, exponent));
          productA = 1;

          // Appendix C, C.16 (S matrix)
          //prefixS = -(thisMoment[j])*( pow((weightedAbscissas[j*(m_N_)+alpha]/weights[alpha]),(thisMoment[j]-1)));
          prefixS = -(thisMoment[j])*(pow(base, exponent));
          productS = 1;

          // calculate product containing all internal coordinates except j
          for (unsigned int n = 0; n < m_N_xi; ++n) {
            if (n != j) {
              // the if statements checking these same conditions (above) are only
              // checking internal coordinate j, so we need them again for internal
              // coordinate n
              if (weights[alpha] == 0) {
                productA = 0;
                productS = 0;
              //} else if ( weightedAbscissas[n*(m_N_)+alpha] == 0 && thisMoment[n] == 0) {
              //  productA = 0;
              //  productS = 0;
              } else {
                double base2 = weightedAbscissas[n*(m_N_)+alpha]/weights[alpha];
                double exponent2 = thisMoment[n];
                productA = productA*( pow(base2, exponent2));
                productS = productS*( pow(base2, exponent2));
              }//end divide by zero conditionals
            }
          }//end int coord n
        }//end divide by zero conditionals


        modelsumS = - models[j*(m_N_)+alpha];

        int col = (j+1)*m_N_ + alpha;
        A(k,col)=prefixA*productA;
        if(verbosity == 1)
          proc0cout << "Setting A(" << k << "," << col << ") = " << prefixA*productA << endl;

        quadsumS = quadsumS + weights[alpha]*modelsumS*prefixS*productS;
      }//end quad nodes
      totalsumS = totalsumS + quadsumS;
    }//end int coords j sub-matrix

    B[k] = totalsumS;
  } // end moments
}

// **********************************************
// Construct A optimized
// **********************************************
void
DQMOM::constructAopt( DenseMatrix*   &AA,
                      vector<double> &Abscissas)
{
  for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
    MomentVector thisMoment = momentIndexes[k];

    // weights
    for ( unsigned int alpha = 0; alpha < m_N_; ++alpha) {
      double prefixA = 1;
      double productA = 1;
      for ( unsigned int i = 0; i < thisMoment.size(); ++i) {
        // Appendix C, C.9 (A1 matrix)
        prefixA = prefixA - (thisMoment[i]);
        double base = Abscissas[i*(m_N_)+alpha];
        double exponent = thisMoment[i];
        productA = productA*( pow(base, exponent) );
      }

      (*AA)[k][alpha] = prefixA*productA;
    } //end weights sub-matrix

    // weighted abscissas
    for( unsigned int j = 0; j < m_N_xi; ++j ) {
      double prefixA    = 1;
      double productA   = 1;

      for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
        if ( Abscissas[j*(m_N_)+alpha] == 0 && thisMoment[j] == 0) {
          //FIXME:
          // both prefixes contain 0^(-1)
          prefixA = 0;
        } else {
          // Appendix C, C.11 (A_j+1 matrix)
          double base = Abscissas[j*(m_N_)+alpha];
          double exponent = thisMoment[j] - 1;
          prefixA = (thisMoment[j])*(pow(base, exponent));
          productA = 1;

          // calculate product containing all internal coordinates except j
          for (unsigned int n = 0; n < m_N_xi; ++n) {
            if (n != j) {
              // the if statements checking these same conditions (above) are only
              // checking internal coordinate j, so we need them again for internal
              // coordinate n
              double base2 = Abscissas[n*(m_N_)+alpha];
              double exponent2 = thisMoment[n];
              productA = productA*( pow(base2, exponent2));
            }
          }//end int coord n
        }//end divide by zero conditionals

        int col = (j+1)*m_N_ + alpha;
        (*AA)[k][col] = prefixA*productA;
      }//end quad nodes
    }//end int coords j sub-matrix
  } // end moments
}

// **********************************************
// Construct B optimized
// **********************************************
void
DQMOM::constructBopt( ColumnMatrix*  &BB,
                      vector<double> &weights,
                      vector<double> &Abscissas,
                      vector<double> &models)
{
  for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
    MomentVector thisMoment = momentIndexes[k];

    // weighted abscissas
    double totalsumS = 0;
    for( unsigned int j = 0; j < m_N_xi; ++j ) {
      double prefixS    = 1;
      double productS   = 1;
      double modelsumS  = 0;

      double quadsumS = 0;
      for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
        if (weights[alpha] == 0) {
          prefixS = 0;
          productS = 0;
        } else if ( Abscissas[j*(m_N_)+alpha] == 0 && thisMoment[j] == 0) {
          //FIXME:
          // both prefixes contain 0^(-1)
          prefixS = 0;
        } else {
          // Appendix C, C.11 (A_j+1 matrix)
          double base = Abscissas[j*(m_N_)+alpha];
          double exponent = thisMoment[j] - 1;

          // Appendix C, C.16 (S matrix)
          prefixS = -(thisMoment[j])*(pow(base, exponent));
          productS = 1;

          // calculate product containing all internal coordinates except j
          for (unsigned int n = 0; n < m_N_xi; ++n) {
            if (n != j) {
              // the if statements checking these same conditions (above) are only
              // checking internal coordinate j, so we need them again for internal
              // coordinate n
              if (weights[alpha] == 0) {
                productS = 0;
              } else {
                double base2 = Abscissas[n*(m_N_)+alpha];
                double exponent2 = thisMoment[n];
                productS = productS*( pow(base2, exponent2));
              }//end divide by zero conditionals
            }
          }//end int coord n
        }//end divide by zero conditionals


        modelsumS = - models[j*(m_N_)+alpha];
        quadsumS = quadsumS + weights[alpha]*modelsumS*prefixS*productS;
      }//end quad nodes
      totalsumS = totalsumS + quadsumS;
    }//end int coords j sub-matrix

    (*BB)[k] = totalsumS;
  } // end moments
}


// **********************************************
// Construct A optimized for unweighted abscissas
// **********************************************
void
DQMOM::constructAopt_unw( DenseMatrix*   &AA,
                          vector<double> &Abscissas)
{
  for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
    MomentVector thisMoment = momentIndexes[k];

    // weights
    for ( unsigned int alpha = 0; alpha < m_N_; ++alpha) {
      double prefixA = 1;
      double productA = 1;
      for ( unsigned int i = 0; i < thisMoment.size(); ++i) {
        // Appendix C, C.9 (A1 matrix)
        //prefixA = prefixA - (thisMoment[i]);
        double base = Abscissas[i*(m_N_)+alpha];
        double exponent = thisMoment[i];
        productA = productA*( pow(base, exponent) );
      }

      (*AA)[k][alpha] = prefixA*productA;
    } //end weights sub-matrix

    // weighted abscissas
    for( unsigned int j = 0; j < m_N_xi; ++j ) {
      double prefixA    = 1;
      double productA   = 1;

      for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
        if ( Abscissas[j*(m_N_)+alpha] == 0 && thisMoment[j] == 0) {
          //FIXME:
          // both prefixes contain 0^(-1)
          prefixA = 0;
        } else {
          // Appendix C, C.11 (A_j+1 matrix)
          double base = Abscissas[j*(m_N_)+alpha];
          double exponent = thisMoment[j] - 1;
          prefixA = (thisMoment[j])*(pow(base, exponent));
          productA = 1;

          // calculate product containing all internal coordinates except j
          for (unsigned int n = 0; n < m_N_xi; ++n) {
            if (n != j) {
              // the if statements checking these same conditions (above) are only
              // checking internal coordinate j, so we need them again for internal
              // coordinate n
              double base2 = Abscissas[n*(m_N_)+alpha];
              double exponent2 = thisMoment[n];
              productA = productA*( pow(base2, exponent2));
            }
          }//end int coord n
        }//end divide by zero conditionals

        int col = (j+1)*m_N_ + alpha;
        (*AA)[k][col] = prefixA*productA;
      }//end quad nodes
    }//end int coords j sub-matrix
  } // end moments
}

// **********************************************
// Construct B optimized for unweighted abscissas
// **********************************************
void
DQMOM::constructBopt_unw( ColumnMatrix*  &BB,
                          vector<double> &Abscissas,
                          vector<double> &models)
{
  for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
    MomentVector thisMoment = momentIndexes[k];

    double totalsumS = 0;
    for( unsigned int j = 0; j < m_N_xi; ++j ) {
      double prefixS    = 1;
      double productS   = 1;
      double modelsumS  = 0;

      double quadsumS = 0;
      for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
        if ( Abscissas[j*(m_N_)+alpha] == 0 && thisMoment[j] == 0) {
          //FIXME:
          // both prefixes contain 0^(-1)
          prefixS = 0;
        } else {
          // Appendix C, C.11 (A_j+1 matrix)
          double base = Abscissas[j*(m_N_)+alpha];
          double exponent = thisMoment[j] - 1;

          // Appendix C, C.16 (S matrix)
          prefixS = -(thisMoment[j])*(pow(base, exponent));
          productS = 1;
          // calculate product containing all internal coordinates except j
          for (unsigned int n = 0; n < m_N_xi; ++n) {
            if (n != j) {
              // the if statements checking these same conditions (above) are only
              // checking internal coordinate j, so we need them again for internal
              // coordinate n
              double base2 = Abscissas[n*(m_N_)+alpha];
              double exponent2 = thisMoment[n];
              productS = productS*( pow(base2, exponent2));
            }
          }//end int coord n
        }//end divide by zero conditionals

        modelsumS = - models[j*(m_N_)+alpha];
        //quadsumS = quadsumS + weights[alpha]*modelsumS*prefixS*productS;
        quadsumS = quadsumS + modelsumS*prefixS*productS;
      }//end quad nodes
      totalsumS = totalsumS + quadsumS;
    }//end int coords j sub-matrix

    (*BB)[k] = totalsumS;
  } // end moments
}




// **********************************************
// Construct A and B matrices for DQMOM
// **********************************************
void
DQMOM::constructLinearSystem( DenseMatrix*   &AA,
                              ColumnMatrix*  &BB,
                              vector<double> &weights,
                              vector<double> &weightedAbscissas,
                              vector<double> &models,
                              int             verbosity)
{
  // construct AX=B
  unsigned int indicesSize = momentIndexes.size();
  for (unsigned int k = 0; k < indicesSize; ++k) {
    MomentVector thisMoment = momentIndexes[k];

    // preprocessing - start with powers
    double d_powers[m_N_][m_N_xi];  // a^(b-1)
    double powers[m_N_][m_N_xi];    // a^b
    double rightPartialProduct[m_N_][m_N_xi], leftPartialProduct[m_N_][m_N_xi];
    for (unsigned int m = 0; m < m_N_; m++) {
      if (weights[m] != 0) {
        for (unsigned int n = 0; n < m_N_xi; n++) {

          //TODO should we worry about 0^0, based on former seq code, only fear is resolved
          double base = weightedAbscissas[n * m_N_ + m] / weights[m];
          double exponent = thisMoment[n] - 1;
          d_powers[m][n] = pow(base, exponent);
          powers[m][n] = d_powers[m][n] * base;
        }
      } else {
        for (unsigned int n = 0; n < m_N_xi; n++) {
          d_powers[m][n] = powers[m][n] = 0;
          rightPartialProduct[m][n] = leftPartialProduct[m][n] = 0;
        }
        rightPartialProduct[m][0] = leftPartialProduct[m][m_N_xi - 1] = 1;
      }
    }
    // now partial products to eliminate innermost for loop
    for (unsigned int m = 0; m < m_N_; m++) {
      if (weights[m] != 0) {
        rightPartialProduct[m][0] = 1;
        leftPartialProduct[m][m_N_xi - 1] = 1;
        for (unsigned int n = 1; n < m_N_xi; n++) {
          rightPartialProduct[m][n] = rightPartialProduct[m][n - 1] * powers[m][n - 1];
          leftPartialProduct[m][m_N_xi - 1 - n] = leftPartialProduct[m][m_N_xi - 1 - n + 1] * powers[m][m_N_xi - 1 - n + 1];
        }
      }
    }
    //end preprocessing

    // weights
    for (unsigned int alpha = 0; alpha < m_N_; ++alpha) {
      double prefixA = 1;
      double productA = 1;

      // preprocessing eliminates conditional
      unsigned int momentSize = thisMoment.size();
      for (unsigned int i = 0; i < momentSize; ++i) {
        // Appendix C, C.9 (A1 matrix)
        prefixA = prefixA - (thisMoment[i]);
        productA = productA * (powers[alpha][i]);
      }
      (*AA)[k][alpha] = prefixA * productA;
      if (verbosity == 1) {
        proc0cout << "Setting A(" << k << "," << alpha << ") = " << prefixA * productA << endl;
      }
    }  //end weights sub-matrix

    // weighted abscissas
    double totalsumS = 0;
    for (unsigned int j = 0; j < m_N_xi; ++j) {
      double prefixA = 1;
      double productA = 1;

      double prefixS = 1;
      double productS = 1;
      double modelsumS = 0;

      double quadsumS = 0;
      for (unsigned int alpha = 0; alpha < m_N_; ++alpha) {

        if (weights[alpha] == 0) {
          prefixA = 0;
          prefixS = 0;
          productA = 0;
          productS = 0;
        } else if (weightedAbscissas[j * (m_N_) + alpha] == 0 && thisMoment[j] == 0) {
          // FIXME: both prefixes contain 0^(-1)
          prefixA = 0;
          prefixS = 0;
        } else {

          // Appendix C, C.11 (A_j+1 matrix)
          prefixA = (thisMoment[j]) * (d_powers[alpha][j]);
          productA = 1;

          // Appendix C, C.16 (S matrix)
          prefixS = -(thisMoment[j]) * (d_powers[alpha][j]);
          productS = 1;

          // calculate product containing all internal coordinates except j
          // use partial products to do this quickly w/o a for loop
          productA = productS = rightPartialProduct[alpha][j] * leftPartialProduct[alpha][j];
        } //end divide by zero conditionals

        modelsumS = -models[j * (m_N_) + alpha];

        int col = (j + 1) * m_N_ + alpha;
        (*AA)[k][col] = prefixA * productA;
        if (verbosity == 1) {
          proc0cout << "Setting A(" << k << "," << col << ") = " << prefixA * productA << endl;
        }

        quadsumS = quadsumS + weights[alpha] * modelsumS * prefixS * productS;
      } //end quad nodes

      totalsumS = totalsumS + quadsumS;
    } //end int coords j sub-matrix

    (*BB)[k] = totalsumS;
  } // end moments
}



// **********************************************
// schedule the calculation of moments
// **********************************************
void
DQMOM::sched_calculateMoments( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  std::cout << " WARNING: The calculateMoments feature is non-functional " << std::endl;
  
  //string taskname = "DQMOM::calculateMoments";
  //Task* tsk = scinew Task(taskname, this, &DQMOM::calculateMoments);

  //if( timeSubStep == 0 )
    //proc0cout << "Requesting DQMOM moment labels" << endl;

  //// computing/modifying the actual moments
  //for( ArchesLabel::MomentMap::iterator iMoment = m_fieldLabels->DQMOMMoments.begin();
       //iMoment != m_fieldLabels->DQMOMMoments.end(); ++iMoment ) {
    //if( timeSubStep == 0 )
      //tsk->computes( iMoment->second );
    //else
      //tsk->modifies( iMoment->second );
  //}

  //// require the weights and weighted abscissas
  //for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    //// require weights
    //const VarLabel* tempLabel;
    //tempLabel = (*iEqn)->getTransportEqnLabel();
    //tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );
  //}
  //for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    //// require weighted abscissas
    //const VarLabel* tempLabel;
    //tempLabel = (*iEqn)->getTransportEqnLabel();
    //tsk->requires(Task::NewDW, tempLabel, Ghost::None, 0);
  //}

  //sched->addTask(tsk, level->eachPatch(), m_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// actually calculate the moments
// **********************************************
/** @details  This calculates the value of each of the $\f(m_N_{\xi}+1)N$\f moments
  *           given to the DQMOM class. For a given moment index $\f k = k_1, k_2, \dots $\f,
  *           the moment is calculated as:
  *           \f[ m_{k} = \sum_{\alpha=1}^{N} w_{\alpha} \prod_{j=1}^{m_N_{\xi}} \xi_{j}^{k_j} \f]
  */
void
DQMOM::calculateMoments( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{
  //// patch loop
  //for( int p=0; p < patches->size(); ++p ) {
    //const Patch* patch = patches->get(p);

    //int archIndex = 0;
    //int matlIndex = m_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    //// get weights from data warehouse and store their corresponding CCVariables
    //vector<constCCVarWrapper> weightCCVars;
    //for( vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
         //iEqn != weightEqns.end(); ++iEqn ) {
      //const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();

      //// instead of using a CCVariable, use a constCCVarWrapper struct
      //constCCVarWrapper tempWrapper;
      //new_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );

      //// put the wrapper into a vector
      //weightCCVars.push_back(tempWrapper);
    //}

    //// get weighted abscissas from data warehouse and store their corresponding CCVariables
    //vector<constCCVarWrapper> weightedAbscissaCCVars;
    //for( vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
         //iEqn != weightedAbscissaEqns.end(); ++iEqn ) {
      //const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();

      //// instead of using a CCVariable, use a constCCVarWrapper struct
      //constCCVarWrapper tempWrapper;
      //new_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );

      //// put the wrapper into vector
      //weightedAbscissaCCVars.push_back(tempWrapper);
    //}

    //// Cell iterator
    //for ( CellIterator iter = patch->getCellIterator();
          //!iter.done(); ++iter) {
      //IntVector c = *iter;

      //vector<double> weights;
      //vector<double> weightedAbscissas;

      //// get weights in current cell from CCVariable in constCCVarWrapper, store value in vector
      //for( vector<constCCVarWrapper>::iterator iter = weightCCVars.begin();
           //iter != weightCCVars.end(); ++iter ) {
        //double temp_value = (iter->data)[c];
        //weights.push_back(temp_value);
      //}

      //// get weighted abscissas in current cell from CCVariable in constCCVarWrapper, store value in vector
      //for( vector<constCCVarWrapper>::iterator iter = weightedAbscissaCCVars.begin();
           //iter != weightedAbscissaCCVars.end(); ++iter ) {
        //double temp_value = (iter->data)[c];
        //weightedAbscissas.push_back(temp_value);
      //}

      //// moment index k = {k1, k2, k3, ...}
      //// moment k = \displaystyle{ sum_{alpha=1}^{N}{ w_{\alpha} \prod_{j=1}^{m_N_xi}{ \langle \xi_{j} \rangle_{\alpha}^{k_j} } } }
      //for( vector<MomentVector>::iterator iAllMoments = momentIndexes.begin();
           //iAllMoments != momentIndexes.end(); ++iAllMoments ) {

        //MomentVector thisMoment = (*iAllMoments);

        //// Grab the corresponding moment from the DQMOMMoment map (to get the VarLabel associated with this moment)
        //const VarLabel* moment_label;
        //ArchesLabel::MomentMap::iterator iMoment = m_fieldLabels->DQMOMMoments.find( thisMoment );
        //if( iMoment != m_fieldLabels->DQMOMMoments.end() ) {
          //// grab the corresponding label
          //moment_label = iMoment->second;
        //} else {
          //string index;
          //std::stringstream out;
          //for( MomentVector::iterator iMomentIndex = thisMoment.begin();
               //iMomentIndex != thisMoment.end(); ++iMomentIndex ) {
            //out << (*iMomentIndex);
            //index += out.str();
          //}
          //string errmsg = "ERROR: DQMOM: calculateMoments: could not find moment index " + index + " in DQMOMMoment map!\nIf you are running verification, you must turn off calculation of moments using <calculate_moments>false</calculate_moments>";
          //throw InvalidValue( errmsg,__FILE__,__LINE__);
        //}

        //// Associate a CCVariable<double> with this moment
        //CCVariable<double> moment_k;
        //if( new_dw->exists( moment_label, matlIndex, patch ) ) {
          //// running getModifiable once for each cell
          //new_dw->getModifiable( moment_k, moment_label, matlIndex, patch );
        //} else {
          //// only run for first cell - so this doesn't wipe out previous calculations
          //new_dw->allocateAndPut( moment_k, moment_label, matlIndex, patch );
          //moment_k.initialize(0.0);
        //}

        //// Calculate the value of the moment
        //double temp_moment_k = 0.0;
        //double running_weights_sum = 0.0; // this is the denominator of p_alpha
        //double running_product = 0.0; // this is w_alpha * xi_1_alpha^k1 * xi_2_alpha^k2 * ...

        //bool is_zeroth_moment = true;
        //for( MomentVector::iterator iM = thisMoment.begin(); iM != thisMoment.end(); ++iM ) {
          //if ( (*iM) != 0 ) {
            //is_zeroth_moment = false;
            //break;
          //}
        //}

        //for( unsigned int alpha = 0; alpha < m_N_; ++alpha ) {
          //if( weights[alpha] < m_w_small ) {
            //running_product = 0.0;
          //} else {
            //double weight = weights[alpha]*d_weight_scaling_constant;
            //running_weights_sum += weight;
            //running_product = weight;
            //for( unsigned int j = 0; j < m_N_xi; ++j ) {

              //double base = (weightedAbscissas[j*(m_N_)+alpha]/weights[alpha])*(d_weighted_abscissa_scaling_constants[j*(m_N_)+alpha]); // don't need 1/d_weight_scaling_constant
                                                                                                                                    //// because it's already in the weighted abscissa!
              //// calculating moments about the mean... so find the mean
              //if( thisMoment[j] != 0 && thisMoment[j] != 1 ) {
                //double mean = 0.0;
                //double mean_numerator = 0.0;
                //double mean_divisor = 0.0;
                //for( unsigned int alpha2 = 0; alpha2 < m_N_; ++alpha2 ) {
                  //mean_numerator += weightedAbscissas[j*(m_N_)+alpha2]*d_weighted_abscissa_scaling_constants[j*(m_N_)+alpha2];
                  //mean_divisor += weights[alpha2];
                //}
                //if (mean_divisor != 0 ) {
                  //mean = mean_numerator/mean_divisor;
                //}
                //base -= mean;
              //}
              //double exponent = thisMoment[j];
              //running_product *= pow( base, exponent );
            //}
          //}
          //temp_moment_k += running_product;
          //running_product = 0.0;
        //}

        //if (running_weights_sum == 0) {
          //moment_k[c] = 0.0;
        //} else if (is_zeroth_moment) {
          //moment_k[c] = temp_moment_k; // don't normalize zeroth moment! otherwise it's always 1...
        //} else {
          //moment_k[c] = temp_moment_k/running_weights_sum; // normalize environment weight to get environment probability
        //}

      //}//end all moments

    //}//end cells
  //}//end patches
}



#if defined(VERIFY_LINEAR_SOLVER)
// **********************************************
// Verify linear solver
// **********************************************
void
DQMOM::verifyLinearSolver()
{
  double tol = vls_tol;

  // -----------------------------------
  // Assemble verification objects

  // assemble A
  LU verification_A(vls_dimension);
  getMatrixFromFile( verification_A, vls_file_A );

  DenseMatrix* verification_AA = scinew DenseMatrix( vls_dimension, vls_dimension );
  getMatrixFromFile( verification_AA, vls_dimension, vls_file_A );

  // assemble B
  vector<double> verification_B(vls_dimension);
  getVectorFromFile( verification_B, vls_file_B );

  ColumnMatrix* verification_BB = scinew ColumnMatrix( vls_dimension );
  getVectorFromFile( verification_BB, vls_dimension, vls_file_B );

  // assemble actual solution
  vector<double> verification_X(vls_dimension);
  getVectorFromFile( verification_X, vls_file_X );

  ColumnMatrix* verification_XX = scinew ColumnMatrix( vls_dimension );
  getVectorFromFile( verification_XX, vls_dimension, vls_file_X );

  // assemble actual residual
  vector<double> verification_R(vls_dimension);
  getVectorFromFile( verification_R, vls_file_R );

  ColumnMatrix* verification_RR = scinew ColumnMatrix( vls_dimension );
  getVectorFromFile( verification_RR, vls_dimension, vls_file_R );

  // assemble actual normalized residual
  vector<double> verification_normR(vls_dimension);
  getVectorFromFile( verification_normR, vls_file_normR );

  ColumnMatrix* verification_normalizedRR = scinew ColumnMatrix( vls_dimension );
  getVectorFromFile( verification_normalizedRR, vls_dimension, vls_file_normR);

  // assemble norms (determinant, normRes, normResNormalized, normX)
  vector<double> verification_norms(5);
  getVectorFromFile( verification_norms, vls_file_norms );

  vector<string> verification_normnames;
  verification_normnames.push_back("determinant");
  verification_normnames.push_back("normRes");
  verification_normnames.push_back("normResNormalized");
  verification_normnames.push_back("normB");
  verification_normnames.push_back("normX");


  // ---------------------------------------------------------------------
  // Print verification objects picked up by verifyLinearSolver() method
  // (But only print them ONCE!)
  if( !m_vls_mat_print ) {
    // print A
    proc0cout << endl << endl;
    proc0cout << "***************************************************************************************" << endl;
    proc0cout << "                      DUMPING LINEAR SOLVER VERIFICATION OBJECTS..." << endl;

    proc0cout << endl << endl;
    proc0cout << "Matrix Verification_A (LU):" << endl;
    proc0cout << "---------------------------" << endl;
    verification_A.dump();


    proc0cout << endl << endl;
    proc0cout << "Matrix Verification_AA (DenseMatrix):" << endl;
    proc0cout << "-------------------------------------" << endl;
    for( int yy=0; yy < vls_dimension; ++yy ) {
      for( int zz=0; zz < vls_dimension; ++zz ) {
        proc0cout << setw(9) << setprecision(4) << (*verification_AA)[yy][zz] << "  ";
      }
      proc0cout << endl;
    }


    proc0cout << endl << endl;
    proc0cout << "RHS Vector Verification_B (vector<double>):" << endl;
    proc0cout << "-------------------------------------------" << endl;
    for(vector<double>::iterator iB = verification_B.begin();
        iB != verification_B.end(); ++iB ) {
      proc0cout << (*iB) << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "RHS Vector Verification_BB (ColumnMatrix): " << endl;
    proc0cout << "-------------------------------------------" << endl;
    for( int yy=0; yy < vls_dimension; ++yy ) {
      proc0cout << (*verification_BB)[yy] << endl;
    }


    proc0cout << endl << endl;
    proc0cout << "Solution Vector Verification_X (vector<double>):" << endl;
    proc0cout << "------------------------------------------------" << endl;
    for(vector<double>::iterator iX = verification_X.begin();
        iX != verification_X.end(); ++iX ) {
      proc0cout << (*iX) << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "Solution Vector Verification_XX (ColumnMatrix): " << endl;
    proc0cout << "------------------------------------------------" << endl;
    for( int yy=0; yy < vls_dimension; ++yy ) {
      proc0cout << (*verification_XX)[yy] << endl;
    }


    proc0cout << endl << endl;
    proc0cout << "Residual Vector Verification_R (vector<double>):" << endl;
    proc0cout << "------------------------------------------------" << endl;
    for(vector<double>::iterator iR = verification_R.begin();
        iR != verification_R.end(); ++iR ) {
      proc0cout << (*iR) << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "Residual Vector Verification_RR (ColumnMatrix): " << endl;
    proc0cout << "------------------------------------------------" << endl;
    for( int yy=0; yy < vls_dimension; ++yy ) {
      proc0cout << (*verification_RR)[yy] << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "Normalized Residual Vector normR:" << endl;
    proc0cout << "-----------------------------" << endl;
    for(vector<double>::iterator iNR = verification_normR.begin();
        iNR != verification_normR.end(); ++iNR ) {
      proc0cout << (*iNR) << endl;
    }


    proc0cout << endl << endl;
    proc0cout << "Determinant/Norms Vector Norms:" << endl;
    proc0cout << "-------------------------------" << endl;
    vector<string>::iterator iNN = verification_normnames.begin();
    for(vector<double>::iterator iN = verification_norms.begin();
        iN != verification_norms.end(); ++iN, ++iNN) {
      proc0cout << (*iNN) << " = " << (*iN) << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "                     ENDING DUMP OF LINEAR SOLVER VERIFICATION OBJECTS..." << endl;
    proc0cout << "***************************************************************************************" << endl;
    proc0cout << endl;

    m_vls_mat_print = true;

  }


  // --------------------------------------------------------------------
  // Begin the actual verification procedure
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "***************************************************************************************" << endl;
  proc0cout << "                      BEGINNING VERIFICATION FOR LINEAR SOLVER..." << endl;
  proc0cout << endl;
  proc0cout << "Using verification procedure for DQMOM linear solver." << endl;
  proc0cout << endl;
  proc0cout << endl;

  // --------------------------------------------------------------------
  // 1. Decompose
  // (For now, verify these as ONE step;
  //  these should be verified separately
  //  to make bugs easier to locate...)
  proc0cout << "Step 1: LU decomposition          " << endl;
  proc0cout << "----------------------------------" << endl;

  LU verification_A_original( verification_A );
  DenseMatrix* AAorig = verification_AA->clone();


  // --------------------------------------------------------------------
  // 2. Back-substitute - back-substitute the decomposed A using the RHS
  //    vector from file. Store in a NEW solution vector.
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 2A: LU back-substitution                             " << endl;
  proc0cout << "Comparing calculated solution to verification solution..." << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  vector<double> X(vls_dimension);
  verification_A.decompose();
  verification_A.back_subs( &verification_B[0], &X[0] );
  compare( verification_X, X, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 2B: DenseMatrix::solve()" << endl;
  proc0cout << "Comparing calculated solution to verification solution..." << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  ColumnMatrix* XX = scinew ColumnMatrix( vls_dimension );
  verification_AA->solve( (*verification_BB), (*XX), 1 );
  compare( verification_XX, XX, vls_dimension, tol );


  // --------------------------------------------------------------------
  // 3. Get determinant - calculate determinant from LU class and compare
  //    to determinant from file.
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 3: Verifying determinant calculation   " << endl;
  proc0cout << "--------------------------------------------" << endl;
  double determinant = verification_A.getDeterminant();
  compare( verification_norms[0], determinant, tol );


  // --------------------------------------------------------------------
  // 4. Get residual - Calculate residual from A, B, X from file and
  //    compare to residual (from file).
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 4A: Verifying LU residual calculation      " << endl;
  proc0cout << "Comparing calculated residual to verification residual..." << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  vector<double> R(vls_dimension);
  verification_A_original.getResidual( &verification_B[0], &X[0], &R[0] );
  compare( verification_R, R, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 4B: Verifying ColumnMatrix residual calculation      " << endl;
  proc0cout << "Comparing calculated residual to verification residual..." << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  ColumnMatrix* RR = scinew ColumnMatrix( vls_dimension );
  int tflops, tmemrefs;
  AAorig->mult( (*XX), (*RR), tflops, tmemrefs );
  Sub( (*RR), (*verification_BB), (*RR) );
  compare( verification_RR, RR, vls_dimension, tol );


  // ---------------------------------------------------------------------
  // 5. Get norm of residual - compare calculated norm of (verification)
  //    residual to norm of residual (from file).
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 5A: Verifying LU's norm of residal " << endl;
  proc0cout << "---------------------------------------------------" << endl;
  double normRes = verification_A.getNorm( &verification_R[0], 1 );
  compare( verification_norms[1], normRes, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 5B: Verifying ColumnMatrix norm of residal " << endl;
  proc0cout << "---------------------------------------------------" << endl;
  double this_normRes = 0;
  for( int ii=0; ii<vls_dimension; ++ii ) {
    if( fabs((*RR)[ii]) > this_normRes ) {
      this_normRes = (*RR)[ii];
    }
  }
  compare( verification_norms[1], this_normRes, tol );


  // -----------------------------------------------------------------
  // 6. Normalize residual by B - compare calculated verification residual
  //    normalized by RHS vector B to the same quantity from file.
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 6: Verifying LU normalization of residual by B " << endl;
  proc0cout << "-----------------------------------------------------" << endl;
  vector<double> Rnormalized(vls_dimension);
  for(vector<double>::iterator iR = R.begin(), iRN = Rnormalized.begin(), iB = verification_B.begin();
      iR != R.end(); ++iR, ++iRN, ++iB) {
    (*iRN) = (*iR)/(*iB);
  }
  compare( verification_normR, Rnormalized, tol );


  // -------------------------------------------------------------------
  // 7. Get norm of normalized residual vector
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 7A: Verifying LU calculation of norm of " << endl;
  proc0cout << "         normalized residual " << endl;
  proc0cout << "---------------------------------------------------" << endl;
  double normResNormalized = verification_A.getNorm( &verification_normR[0], 1 );
  compare( verification_norms[2], normResNormalized, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 7B: Verifying ColumnMatrix calculation of norm of " << endl;
  proc0cout << "         normalized residual " << endl;
  proc0cout << "---------------------------------------------------" << endl;
  double this_normResNormalizedB = 0;
  for( int ii=0; ii<vls_dimension; ++ii ) {
    if( fabs( (*RR)[ii] / (*verification_BB)[ii] ) > this_normResNormalizedB ) {
      this_normResNormalizedB = ((*RR)[ii] / (*verification_BB)[ii]);
    }
  }
  compare( verification_norms[2], this_normResNormalizedB, tol );



  // -------------------------------------------------------------------
  // 8. Get norm of RHS vector B
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 8A: Verifying LU norm of RHS vector B  " << endl;
  proc0cout << "----------------------------------------" << endl;
  double normB = verification_A.getNorm( &verification_B[0], 1 );
  compare( verification_norms[3], normB, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 8B: Verifying ColumnMatrix norm of RHS vector B  " << endl;
  proc0cout << "----------------------------------------" << endl;
  double this_normB = 0;
  for( int ii=0; ii<vls_dimension; ++ii ) {
    if( fabs((*verification_BB)[ii]) > this_normB ) {
      this_normB = (*verification_BB)[ii];
    }
  }
  compare( verification_norms[3], this_normB, tol );


  // -------------------------------------------------------------------
  // 9. Get norm of solution vector X
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 8A: Verifying norm of solution vector X  " << endl;
  proc0cout << "----------------------------------------------" << endl;
  double normX = verification_A.getNorm( &verification_X[0], 1 );
  compare( verification_norms[4], normX, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 8B: Verifying norm of solution vector X  " << endl;
  proc0cout << "----------------------------------------------" << endl;
  double this_normX = 0;
  for( int ii=0; ii<vls_dimension; ++ii ) {
    if( fabs((*XX)[ii]) > this_normX ) {
      this_normX = (*XX)[ii];
    }
  }
  compare( verification_norms[4], this_normX, tol );




  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "                      ENDING VERIFICATION FOR LINEAR SOLVER..." << endl;
  proc0cout << "***************************************************************************************" << endl;
  proc0cout << endl;
  proc0cout << endl;

  delete verification_AA;
  delete verification_BB;
  delete verification_XX;
  delete verification_RR;
  delete verification_normalizedRR;
  delete XX;
  delete RR;

}
#endif



#if defined(VERIFY_AB_CONSTRUCTION)
// **********************************************
// Verify construction of A and B
// **********************************************
void
DQMOM::verifyABConstruction()
{
  vab_dimension = (vab_N_xi + 1)*vab_N;
  double tol = vab_tol;

  // assemble A
  LU verification_A( vab_dimension );
  getMatrixFromFile( verification_A, vab_file_A );

  DenseMatrix* verification_AA = scinew DenseMatrix( vab_dimension, vab_dimension );
  getMatrixFromFile( verification_AA, vab_dimension, vab_file_A );

  // assemble B
  vector<double> verification_B(vab_dimension);
  getVectorFromFile( verification_B, vab_file_B );

  ColumnMatrix* verification_BB = scinew ColumnMatrix( vab_dimension );
  getVectorFromFile( verification_BB, vab_dimension, vab_file_B );

  // assemble inputs
  ifstream vab_inputs( vab_file_inputs.c_str() );
  if(vab_inputs.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: Verification of A and B construction procedure could not find the file containing weight/weighted abscissa/model inputs: You specified " << vab_file_inputs << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  // assemble weights
  vector<double> weights(vab_N);
  getVectorFromFile( weights, vab_inputs );

  // assemble weighted abscissas
  vector<double> weightedAbscissas(vab_N*vab_N_xi);
  getVectorFromFile( weightedAbscissas, vab_inputs );
  // weight the abscissas
  for(int m=0; m<vab_N_xi; ++m) {
    for(int n=0; n<vab_N; ++n) {
      weightedAbscissas[m*(m_N_)+n] = weightedAbscissas[m*(m_N_)+n]*weights[n];
    }
  }

  // assemble weighted abscissas
  vector<double> models(vab_N*vab_N_xi);
  getVectorFromFile( models, vab_inputs );

  // assemble moment indices
  momentIndexes.resize(0);// get rid of moment indices from file
  momentIndexes.resize(vab_dimension);
  getMomentsFromFile( momentIndexes, vab_file_moments );

  // ---------------------------------------------------------------------
  // Print verification objects picked up by verifyABConstruction() method
  if( !m_vab_mat_print ) {
    // print A
    proc0cout << endl << endl;
    proc0cout << "***************************************************************************************" << endl;
    proc0cout << "                      DUMPING AB CONSTRUCTION VERIFICATION OBJECTS..." << endl;
    proc0cout << endl << endl;

    proc0cout << "Matrix A:" << endl;
    proc0cout << "-----------------------------" << endl;
    verification_A.dump();

    proc0cout << endl << endl;

    proc0cout << "RHS Vector B:" << endl;
    proc0cout << "-----------------------------" << endl;
    for(vector<double>::iterator iB = verification_B.begin();
        iB != verification_B.end(); ++iB ) {
      proc0cout << (*iB) << endl;
    }

    proc0cout << endl << endl;

    proc0cout << "Input Weights:" << endl;
    proc0cout << "-----------------------------" << endl;
    for(vector<double>::iterator iW = weights.begin();
        iW != weights.end(); ++iW ) {
      proc0cout << (*iW) << endl;
    }

    proc0cout << endl << endl;

    proc0cout << "Input Weighted Abscissas:" << endl;
    proc0cout << "-----------------------------" << endl;
    for(vector<double>::iterator iWA = weightedAbscissas.begin();
        iWA != weightedAbscissas.end(); ++iWA ) {
      proc0cout << (*iWA) << endl;
    }

    proc0cout << endl << endl;

    proc0cout << "Input Models:" << endl;
    proc0cout << "-----------------------------" << endl;
    for(vector<double>::iterator iM = models.begin();
        iM != models.end(); ++iM ) {
      proc0cout << (*iM) << endl;
    }

    proc0cout << endl << endl;

    proc0cout << "Input Moments:" << endl;
    proc0cout << "------------------------------" << endl;
    for(vector<MomentVector>::iterator iAllMoments = momentIndexes.begin();
        iAllMoments != momentIndexes.end(); ++iAllMoments ) {
      for(MomentVector::iterator iThisMoment = iAllMoments->begin();
          iThisMoment != iAllMoments->end(); ++iThisMoment ) {
        proc0cout << (*iThisMoment) << " ";
      }
      proc0cout << endl;
    }

    proc0cout << endl << endl;
    proc0cout << "                     ENDING DUMP OF AB CONSTRUCTION VERIFICATION OBJECTS..." << endl;
    proc0cout << "***************************************************************************************" << endl;
    proc0cout << endl;

    m_vab_mat_print = true;

  }


  // -------------------------------------------------------------
  // Begin the actual verfication procedure

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "***************************************************************************************" << endl;
  proc0cout << "                      BEGINNING VERIFICATION FOR AB CONSTRUCTION..." << endl;
  proc0cout << endl;
  proc0cout << "Using verification procedure for DQMOM linear solver." << endl;
  proc0cout << endl;


  // construct A and B
  LU A( vab_dimension );
  vector<double> B( vab_dimension );
  constructLinearSystem( A, B, weights, weightedAbscissas, models );

  DenseMatrix* AA = scinew DenseMatrix( vab_dimension, vab_dimension );
  ColumnMatrix* BB = scinew ColumnMatrix( vab_dimension );
  constructLinearSystem( AA, BB, weights, weightedAbscissas, models );

  // --------------------------------------------------------------
  // 1. Compare constructed A to verification A (from file)
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 1A: Compare constructed A to assembled A (from file) " << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  proc0cout << "Constructed LU A:" << endl;
  A.dump();
  compare( verification_A, A, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 1B: Compare constructed A to assembled A (from file) " << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  proc0cout << "Constructed DenseMatrix A:" << endl;
  for( int yy=0; yy < vab_dimension; ++yy ) {
    for( int zz=0; zz < vab_dimension; ++zz ) {
      proc0cout << setw(6) << setprecision(4) << (*AA)[yy][zz] << "\t";
    }
    proc0cout << endl;
  }
  proc0cout << endl;
  compare( verification_AA, AA, vab_dimension, tol );


  // --------------------------------------------------------------
  // 2. Compare constructed B to verification B (from file)
  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 2A: Compare constructed vector<double> B to assembled B (from file) " << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  proc0cout << "Vector<double> B:" << endl;
  for(vector<double>::iterator iB = B.begin(); iB != B.end(); ++iB ) {
    proc0cout << (*iB) << endl;
  }
  compare( verification_B, B, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "Step 2B: Compare constructed ColumnMatrix B to assembled B (from file) " << endl;
  proc0cout << "---------------------------------------------------------" << endl;
  proc0cout << "ColumnMatrix B:" << endl;
  for( int yy=0; yy < vab_dimension; ++yy ) {
    proc0cout << (*BB)[yy] << endl;
  }
  compare( verification_BB, BB, vab_dimension, tol );

  proc0cout << endl;
  proc0cout << endl;
  proc0cout << "                      ENDING VERIFICATION FOR AB CONSTRUCTION..." << endl;
  proc0cout << "***************************************************************************************" << endl;
  proc0cout << endl;
  proc0cout << endl;

  delete verification_AA;
  delete verification_BB;
  delete AA;
  delete BB;
}
#endif

#if defined(VERIFY_LINEAR_SOLVER) || defined(VERIFY_AB_CONSTRUCTION)
void
DQMOM::compare( vector<double> vector1, vector<double> vector2, double tolerance )
{
  const int size1 = vector1.size();
  const int size2 = vector2.size();
  if( size1 != size2 ) {
    proc0cout << "You specified vector 1 (length = " << size1 << ") and vector 2 (length = " << size2 << ")." << endl;
    string err_msg = "ERROR: DQMOM: Compare: Cannot compare vectors of dissimilar size.";
    throw InvalidValue( err_msg,__FILE__,__LINE__);
  }

  proc0cout << " >>> " << endl;
  int mismatches = 0;
  int element = 0;
  for(vector<double>::iterator ivec1 = vector1.begin(), ivec2 = vector2.begin();
      ivec1 != vector1.end(); ++ivec1, ++ivec2, ++element) {
    double pdiff = fabs( ( (*ivec1)-(*ivec2) )/(*ivec1) );
    if( pdiff > tolerance ) {
      proc0cout << " >>> Element " << element << " mismatch: "
                << "\tPercent Diff = "    << setw(3)  << setprecision(2) << pdiff
                << "\tVector1 (Verif) = " << setw(12) << setprecision(4) << (*ivec1)
                << "\tVector2 (Calc) = "  << setw(12) << setprecision(4) << (*ivec2) << endl;
      ++mismatches;
    }
  }

  if( mismatches > 0 ) {
    proc0cout << " >>> " << endl;
    proc0cout << " >>> !!! COMPARISON FAILED !!! " << endl;
    proc0cout << " >>> " << endl;
  }

  proc0cout << " >>> Summary of vector comparison: found " << mismatches << " mismatches in " << element << " elements." << endl;
  proc0cout << " >>> " << endl;
  proc0cout << endl;
}


void
DQMOM::compare( ColumnMatrix* &vector1, ColumnMatrix* &vector2, int dimension, double tolerance )
{
  proc0cout << " >>> " << endl;
  int mismatches = 0;
  int element = 0;
  for( int yy=0; yy<dimension; ++yy ) {
    double pdiff = fabs( ( (*vector1)[yy] - (*vector2)[yy] )/( (*vector1)[yy] ) );
    if( pdiff > tolerance ) {
      proc0cout << " >>> Element " << element << " mismatch: "
                << "\tPercent Diff = "    << setw(3)  << setprecision(2) << pdiff
                << "\tVector1 (Verif) = " << setw(12) << setprecision(4) << (*vector1)[yy]
                << "\tVector2 (Calc) = "  << setw(12) << setprecision(4) << (*vector2)[yy] << endl;
      ++mismatches;
    }
  }

  if( mismatches > 0 ) {
    proc0cout << " >>> " << endl;
    proc0cout << " >>> !!! COMPARISON FAILED !!! " << endl;
    proc0cout << " >>> " << endl;
  }

  proc0cout << " >>> Summary of vector comparison: found " << mismatches << " mismatches in " << element << " elements." << endl;
  proc0cout << " >>> " << endl;
  proc0cout << endl;
}


void
DQMOM::compare( double x1, double x2, double tolerance )
{
  proc0cout << " >>> " << endl;
  int mismatches = 0;

  double pdiff = fabs( (x1-x2)/x1 );
  if( pdiff > tolerance ) {
    proc0cout << " >>> Element mismatch: "
              << "\tPercent Diff = "  << setw(3)  << setprecision(2) << pdiff
              << "\tX1 (Verif) = "    << setw(12) << setprecision(4) << x1
              << "\tX2 (Calc) = "     << setw(12) << setprecision(4) << x2 << endl;
    ++mismatches;
  }

  if( mismatches > 0 ) {
    proc0cout << " >>> " << endl;
    proc0cout << " >>> !!! COMPARISON FAILED !!! " << endl;
    proc0cout << " >>> " << endl;
  }

  proc0cout << " >>> Summary of scalar comparison: found " << mismatches << " mismatches." << endl;
  proc0cout << " >>> " << endl;
}



void
DQMOM::compare( LU matrix1, LU matrix2, double tolerance )
{
  const int size1 = matrix1.getDimension();
  const int size2 = matrix2.getDimension();
  if( size1 != size2 ) {
    proc0cout << "You specified matrix 1 (length = " << size1 << ") and matrix 2 (length = " << size2 << ")." << endl;
    string err_msg = "ERROR: DQMOM: Compare: Cannot compare matrices of dissimilar size.";
    throw InvalidValue( err_msg,__FILE__,__LINE__);
  }

  proc0cout << " >>> " << endl;
  int mismatches = 0;
  int element = 0;
  for( int row=0; row < size1; ++row ) {
    for( int col=0; col < size1; ++col ) {
      double pdiff = fabs( (matrix1(row,col)-matrix2(row,col))/matrix1(row,col) );
      if( pdiff > tolerance ) {
        proc0cout << " >>> Element (row "    << setw(2)  << setprecision(0) << row+1 << ", col " << setw(2) << setprecision(0) << col+1 << ") mismatch: "
                  << "\tPercent Diff = "     << setw(3)  << setprecision(2) << pdiff
                  << "\tMatrix 1 (Verif) = " << setw(12) << setprecision(4) << matrix1(row,col)
                  << "\tMatrix 2 (Calc) = "  << setw(12) << setprecision(4) << matrix2(row,col) << endl;
        ++mismatches;
      }
      ++element;
    }
  }

  if( mismatches > 0 ) {
    proc0cout << " >>> " << endl;
    proc0cout << " >>> !!! COMPARISON FAILED !!! " << endl;
    proc0cout << " >>> " << endl;
  }

  proc0cout << " >>> Summary of matrix comparison: found " << mismatches << " mismatches in " << element << " elements." << endl;
  proc0cout << " >>> " << endl;
  proc0cout << endl;
}


void
DQMOM::compare( DenseMatrix* &matrix1, DenseMatrix* &matrix2, int dimension, double tolerance )
{
  proc0cout << " >>> " << endl;
  int mismatches = 0;
  int element = 0;
  for( int row=0; row < dimension; ++row ) {
    for( int col=0; col < dimension; ++col ) {
      double pdiff = fabs( ( (*matrix1)[row][col] - (*matrix2)[row][col] )/( (*matrix1)[row][col] ) );
      if( pdiff > tolerance ) {
        proc0cout << " >>> Element (row "    << setw(2)  << setprecision(0) << row+1 << ", col " << setw(2) << setprecision(0) << col+1 << ") mismatch: "
                  << "\tPercent Diff = "     << setw(3)  << setprecision(2) << pdiff
                  << "\tMatrix 1 (Verif) = " << setw(12) << setprecision(4) << (*matrix1)[row][col]
                  << "\tMatrix 2 (Calc)  = " << setw(12) << setprecision(4) << (*matrix2)[row][col] << endl;
        ++mismatches;
      }
      ++element;
    }
  }

  if( mismatches > 0 ) {
    proc0cout << " >>> " << endl;
    proc0cout << " >>> !!! COMPARISON FAILED !!! " << endl;
    proc0cout << " >>> " << endl;
  }

  proc0cout << " >>> Summary of matrix comparison: found " << mismatches << " mismatches in " << element << " elements." << endl;
  proc0cout << " >>> " << endl;
  proc0cout << endl;
}


void
DQMOM::tokenizeInput( const string& str, vector<string>& tokens, const string& delimiters )
{
  // see http://oopweb.com/CPP/Documents/CPPHOWTO/Volume/C++Programming-HOWTO-7.html
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

void
DQMOM::getMatrixFromFile( LU& matrix, string filename )
{
  ifstream ifile( filename.c_str() );
  if( ifile.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: getMatrixFromFile: Verification procedure could not find the file you specified: " << filename << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  for( unsigned int jj=0; jj<matrix.getDimension(); ++jj) {
    string s1;

    // grab entire row
    do {
      getline( ifile, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    vector<string> elementsOfA;
    tokenizeInput( s1, elementsOfA, " " );

    if( elementsOfA.size() != matrix.getDimension() ) {
      ostringstream err_msg;
      err_msg << "ERROR: DQMOM: getMatrixFromFile: Verification procedure found incorrect number of elements in matrix, file " << filename << ": found " << elementsOfA.size() << " elements, needed " << matrix.getDimension() << " elements." << endl;
      throw InvalidValue( err_msg.str(),__FILE__,__LINE__);
    }

    int kk = 0;
    for( vector<string>::iterator iE = elementsOfA.begin();
         iE != elementsOfA.end(); ++iE, ++kk ) {
      double d = 0.0;
      stringstream ss( (*iE) );
      ss >> d;
      matrix(jj,kk) = d;
    }
  }
}

void
DQMOM::getMatrixFromFile( DenseMatrix* &matrix, int dimension, string filename )
{
  ifstream ifile( filename.c_str() );
  if( ifile.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: getMatrixFromFile: Verification procedure could not find the file you specified: " << filename << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  for( unsigned int jj=0; jj<dimension; ++jj) {
    string s1;

    // grab entire row
    do {
      getline( ifile, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    vector<string> elementsOfA;
    tokenizeInput( s1, elementsOfA, " " );

    if( elementsOfA.size() != dimension ) {
      ostringstream err_msg;
      err_msg << "ERROR: DQMOM: getMatrixFromFile: Verification procedure found incorect number of elements in matrix, file " << filename << ": found " << elementsOfA.size() << " elements, needed " << dimension << " elements." << endl;
      throw InvalidValue( err_msg.str(),__FILE__,__LINE__);
    }

    int kk=0;
    for( vector<string>::iterator iE = elementsOfA.begin();
         iE != elementsOfA.end(); ++iE, ++kk ) {
      double d = 0.0;
      stringstream ss( (*iE) );
      ss >> d;
      (*matrix)[jj][kk] = d;
    }
  }
}

/** @details  This method opens an ifstream to a file containing moment indices.
  *           This allows for verification of the construction of A and B using the
  *           correct moment indices, independent of the moment indeces found in the
  *           input file.
  *           This allows the verification procedure to be completely independent
  *           of the UPS input file, and depend only upon the verification input
  *           files.
  *
  *           The number of moment indices is equal to the number of elements
  *           in "moments".
  */
void
DQMOM::getMomentsFromFile( vector<MomentVector>& moments, string filename )
{
  ifstream ifile( filename.c_str() );
  if( ifile.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: getMomentsFromFile: Verification procedure could not find file you specified: " << filename << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  for( vector<MomentVector>::iterator i1 = moments.begin(); i1 != moments.end(); ++i1 ) {
    string s1;

    // grab entire row
    do {
      getline( ifile, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    vector<string>elementsOfA;
    tokenizeInput( s1, elementsOfA, " " );

    vector<int> temp_vector;
    for( vector<string>::iterator iE = elementsOfA.begin();
         iE != elementsOfA.end(); ++iE ) {
      int d = 0;
      stringstream ss( (*iE) );
      ss >> d;
      temp_vector.push_back(d);
    }
    (*i1) = temp_vector;
  }
}

/** @details  This method opens an ifstream to file "filename", then reads the file
  *           one line at a time.  Each line becomes an element of the vector "vec",
  *           unless the line begins with a '#' character (indicates comment) or
  *           the line is empty.
  *
  *           The number of lines read is equal to the size of the vector "vec".
  */
void
DQMOM::getVectorFromFile( vector<double>& vec, string filename )
{
  ifstream jfile( filename.c_str() );
  if( jfile.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: getVectorFromFile: Verification procedure could not find file you specified: " << filename << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  for( vector<double>::iterator iB = vec.begin();
       iB != vec.end(); ++iB ) {
    string s1;

    do {
      getline( jfile, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    double d = 0.0;
    stringstream ss(s1);
    ss >> d;
    (*iB) = d;
  }
}

void
DQMOM::getVectorFromFile( ColumnMatrix* &vec, int dimension, string filename )
{
  ifstream jfile( filename.c_str() );
  if( jfile.fail() ) {
    ostringstream err_msg;
    err_msg << "ERROR: DQMOM: getVectorFromFile: Verification procedure could not find file you specified: " << filename << endl;
    throw FileNotFound(err_msg.str(),__FILE__,__LINE__);
  }

  for( int yy=0; yy<dimension; ++yy ) {
    string s1;

    do {
      getline( jfile, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    double d = 0.0;
    stringstream ss(s1);
    ss >> d;
    (*vec)[yy] = d;
  }
}

/** @details  This method uses an already open ifstream "filestream" and reads the
  *           file one line at a time.  The filestream is passed, rather than a file
  *           name, in the case that one file contains multiple vectors.
  *
  *           This allows the "getVectorFromFile" method to be called several times
  *           for the same file without the place in the file being lost.
  *           (If you pass the same ifstream, the ifstream remembers its
  *            current location in the file.)
  *
  *           Each line becomes an element of the vector "vec",
  *           unless the line begins with a '#' character (indicates comment) or
  *           the line is empty.
  *
  *           The number of lines read is equal to the size of vector "vec".
  */
void
DQMOM::getVectorFromFile( vector<double>& vec, ifstream& filestream )
{
  for( vector<double>::iterator iB = vec.begin();
       iB != vec.end(); ++iB ) {
    string s1;

    do {
      getline( filestream, s1 );
    }
    while( s1[0] == '\n' || s1[0] == '#' );

    double d = 0.0;
    stringstream ss(s1);
    ss >> d;
    (*iB) = d;
  }
}

#endif
