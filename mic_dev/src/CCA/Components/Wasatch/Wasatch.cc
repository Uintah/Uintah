/*
 * The MIT License
 *
 * Copyright (c) 2010-2012 The University of Utah
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
#include <fstream>
#include <limits>
//-- Uintah framework includes --//
#include <sci_defs/uintah_defs.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

//-- SpatialOps includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Expressions/StableTimestep.h>
#include <CCA/Components/Wasatch/CoordinateHelper.h>

#include <spatialops/structured/FVStaggered.h>
#ifdef ENABLE_THREADS
#include <spatialops/SpatialOpsTools.h>
#include <expression/SchedulerBase.h>
#endif

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- TabProps --//
#include <tabprops/TabPropsConfig.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <CCA/Components/Wasatch/WasatchMaterial.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/TaskInterface.h>
#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/Properties.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Expressions/BasicExprBuilder.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/RadiationSource.h>
#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>
#include <CCA/Components/Wasatch/Expressions/NullExpression.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>

#include <CCA/Components/Wasatch/Transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>

#include <CCA/Components/Wasatch/BCHelperTools.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/FieldClippingTools.h>
#include <CCA/Components/Wasatch/OldVariable.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>
#include <CCA/Components/Wasatch/ParticlesHelper.h>
#include <CCA/Components/Wasatch/WasatchParticlesHelper.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/Expressions/CellType.h>
using std::endl;

namespace Wasatch{

  //--------------------------------------------------------------------

  Wasatch::Wasatch( const Uintah::ProcessorGroup* myworld )
    : Uintah::UintahParallelComponent( myworld ),
      buildTimeIntegrator_ ( true ),
      buildWasatchMaterial_( true ),
      nRKStages_(1),
      isPeriodic_ (true ),
      doRadiation_(false),
      doParticles_(false),
      timeIntegrator_(TimeIntegrator("FE"))
  {
    proc0cout << std::endl
              << "-------------------------------------------------------------" << std::endl
              << "Wasatch was built against:" << std::endl
              << "  SpatialOps HASH: " << SOPS_REPO_HASH << std::endl
              << "             DATE: " << SOPS_REPO_DATE << std::endl
              << "     ExprLib HASH: " << EXPR_REPO_HASH << std::endl
              << "             DATE: " << EXPR_REPO_DATE << std::endl
              << "    TabProps HASH: " << TabPropsVersionHash << std::endl
              << "             DATE: " << TabPropsVersionDate << std::endl
              << "-------------------------------------------------------------" << std::endl
              << std::endl;

    materials_   = NULL;
    timeStepper_ = NULL;
    linSolver_   = NULL;

    cellType_ = scinew CellType();
    
    isRestarting_ = false;

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::d_combineMemory = false;

    const bool log = false;
    graphCategories_[ INITIALIZATION     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ TIMESTEP_SELECTION ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ ADVANCE_SOLUTION   ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ POSTPROCESSING     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );

    OldVariable::self().sync_with_wasatch( this );
    ReductionHelper::self().sync_with_wasatch( this );
    particlesHelper_ = scinew WasatchParticlesHelper();
    particlesHelper_->sync_with_wasatch(this);
  }

  //--------------------------------------------------------------------

  Wasatch::~Wasatch()
  {
    for( PatchInfoMap::iterator i=patchInfoMap_.begin(); i!=patchInfoMap_.end(); ++i ){
      delete i->second.operators;
    }

    for( EquationAdaptors::iterator i=adaptors_.begin(); i!=adaptors_.end(); ++i ){
      delete *i;
    }

    for( std::list<const TaskInterface*>::iterator i=taskInterfaceList_.begin(); i!=taskInterfaceList_.end(); ++i ){
      delete *i;
    }

    for( std::map<int, const Uintah::PatchSet*>::iterator i=patchesForOperators_.begin(); i!=patchesForOperators_.end(); ++i ){
      delete i->second;
    }

    if( buildTimeIntegrator_ ) delete timeStepper_;

    for( GraphCategories::iterator igc=graphCategories_.begin(); igc!=graphCategories_.end(); ++igc ){
      delete igc->second->exprFactory;
      delete igc->second;
    }
    
    for( BCHelperMapT::iterator it=bcHelperMap_.begin(); it != bcHelperMap_.end(); ++it ){
      delete it->second;
    }
    delete cellType_;
    delete particlesHelper_;
  }

  //--------------------------------------------------------------------
  
  void force_expressions_on_graph( Expr::TagList& exprTagList,
                                   GraphHelper* const graphHelper )
  {    
    Expr::Tag exprtag;
    for( Expr::TagList::iterator exprtag=exprTagList.begin();
         exprtag!=exprTagList.end();
         ++exprtag )
    {
      const Expr::ExpressionID exprID = graphHelper->exprFactory->get_id(*exprtag);
      graphHelper->rootIDs.insert( exprID );
    }    
  }
  
  //--------------------------------------------------------------------

  void force_expressions_on_graph( Uintah::ProblemSpecP forceOnGraphParams,
                                   GraphHelper* const gh )
  {
    for( Uintah::ProblemSpecP exprParams = forceOnGraphParams->findBlock("NameTag");
         exprParams != 0;
         exprParams = exprParams->findNextBlock("NameTag") )
    {
      const Expr::ExpressionID exprID = gh->exprFactory->get_id(parse_nametag(exprParams));
      gh->rootIDs.insert( exprID );
    }
  }

  //--------------------------------------------------------------------
  
  void check_periodicity_extra_cells( const Uintah::ProblemSpecP& params,
                                      Uintah::IntVector& extraCells,
                                      bool& isPeriodic)
  {
    // disallow different periodicities on multiple levels
    std::vector<Uintah::IntVector> levelPeriodicityVectors;
    // disallow specification of extraCells and different periodicities on multiple levels
    int nlevels = 0;
    std::ostringstream msg;
    bool foundExtraCells = false;
    Uintah::ProblemSpecP gridspec = params->findBlock("Grid");
    for( Uintah::ProblemSpecP level = gridspec->findBlock("Level");
         level != 0;
         level = gridspec->findNextBlock("Level") ){
      nlevels++;
      Uintah::IntVector periodicDirs(0,0,0);
      level->get("periodic", periodicDirs);
      levelPeriodicityVectors.push_back(periodicDirs);
      
      for( Uintah::ProblemSpecP box = level->findBlock("Box");
           box != 0;
           box = level->findNextBlock("Box") )
      {
        // note that a [0,0,0] specification gets added by default,
        // so we will check to ensure that something other than
        // [0,0,0] has not been specified.
        if( box->findBlock("extraCells") ){
          //Uintah::IntVector extraCells;
          box->get("extraCells",extraCells);
          if( extraCells != Uintah::IntVector(0,0,0) ){
            foundExtraCells = true;
          }
        }
      }
    }
    
    // check for different periodicities on different levels
    std::vector<Uintah::IntVector>::iterator periodicityIter = levelPeriodicityVectors.begin();
    int xPeriodSum = 0, yPeriodSum = 0, zPeriodSum = 0;
    while( periodicityIter != levelPeriodicityVectors.end() ){
      Uintah::IntVector& pvector = *periodicityIter;
      xPeriodSum += pvector.x();
      yPeriodSum += pvector.y();
      zPeriodSum += pvector.z();
      ++periodicityIter;
    }
    
    if( ( xPeriodSum !=0 && xPeriodSum < nlevels ) ||
        ( yPeriodSum !=0 && yPeriodSum < nlevels ) ||
        ( zPeriodSum !=0 && zPeriodSum < nlevels ) )
    {
      msg << endl
      << "  Specification of different periodicities for different levels is not supported in Wasatch." << endl
      << "  Please revise your input file." << endl
      << endl;
      throw std::runtime_error( msg.str() );
    }
    
    Uintah::IntVector periodicityVector = levelPeriodicityVectors[0];
    const bool isXPeriodic = periodicityVector.x() == 1;
    const bool isYPeriodic = periodicityVector.y() == 1;
    const bool isZPeriodic = periodicityVector.z() == 1;
    isPeriodic = isXPeriodic || isYPeriodic || isZPeriodic;
#   ifdef WASATCH_IN_ARCHES
    // we are only allowing for a single extra cell :(
    // make sure that extra cell and periodicity are consistent
    proc0cout << "periodicity = " << isPeriodic << std::endl;
    
    if( !foundExtraCells && !isPeriodic ){
      msg << endl
      << "  Specification of 'extraCells' is required when wasatch-in-arches is enabled." << endl
      << "  Please add an 'extraCells' block to your input file" << endl
      << endl;
      throw std::runtime_error( msg.str() );
    }
#   else
    if( foundExtraCells ){
      msg << endl
      << "  Specification of 'extraCells' is forbidden in Wasatch. The number of extraCells is automatically determined." << endl
      << "  Please remove it from your input file." << endl
      << endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
        
    extraCells = Uintah::IntVector( (isXPeriodic) ? 0 : 1,
                                    (isYPeriodic) ? 0 : 1,
                                    (isZPeriodic) ? 0 : 1 );

  }

  //--------------------------------------------------------------------
  
  void assign_unique_boundary_names( Uintah::ProblemSpecP bcProbSpec )
  {
    if( !bcProbSpec ) return;
    int i=0;
    std::string strFaceID;
    std::set<std::string> faceNameSet;
    for( Uintah::ProblemSpecP faceSpec = bcProbSpec->findBlock("Face");
         faceSpec != 0; faceSpec=faceSpec->findNextBlock("Face"), ++i ){
      
      std::string faceName = "none";
      faceSpec->getAttribute("name",faceName);
      
      strFaceID = number_to_string(i);
      
      if( faceName=="none" || faceName=="" ){
        faceName ="Face_" + strFaceID;
        faceSpec->setAttribute("name",faceName);
      }
      else{
        if( faceNameSet.find(faceName) != faceNameSet.end() ){
          bool fndInc = false;
          int j = 1;
          while( !fndInc ){
            if( faceNameSet.find( faceName + "_" + number_to_string(j) ) != faceNameSet.end() )
              j++;
            else
              fndInc = true;
          }
          // rename this face
          std::cout << "WARNING: I found a duplicate face label " << faceName;
          faceName = faceName + "_" + number_to_string(j);
          std::cout << " in your Boundary condition specification. I will rename it to " << faceName << std::endl;
          faceSpec->replaceAttributeValue("name", faceName);
        }
      }
      faceNameSet.insert(faceName);
    }
  }

  //--------------------------------------------------------------------

  
  void Wasatch::preGridProblemSetup(const Uintah::ProblemSpecP& params,
                           Uintah::GridP& grid,
                           Uintah::SimulationStateP& state)
  {
    Uintah::IntVector extraCells;
    check_periodicity_extra_cells( params, extraCells, isPeriodic_);
    grid->setExtraCells(extraCells);
  }
  
  //--------------------------------------------------------------------

  void Wasatch::problemSetup( const Uintah::ProblemSpecP& params,
                              const Uintah::ProblemSpecP& ,  /* jcs not sure what this param is for */
                              Uintah::GridP& grid,
                              Uintah::SimulationStateP& sharedState )
  {
    wasatchSpec_ = params->findBlock("Wasatch");
    if( !wasatchSpec_ ) return;

    //
    // Check whether we are solving for particles
    //
    doParticles_ = wasatchSpec_->findBlock("ParticleTransportEquations");
    if( doParticles_ ){
      particlesHelper_->problem_setup(wasatchSpec_->findBlock("ParticleTransportEquations"), sharedState);
    }

    // setup names for all the boundary condition faces that do NOT have a name or that have duplicate names
    if( params->findBlock("Grid") ){
      Uintah::ProblemSpecP bcProbSpec = params->findBlock("Grid")->findBlock("BoundaryConditions");
      assign_unique_boundary_names( bcProbSpec );
    }
    
    sharedState_ = sharedState;
    // TSAAD: keep the line of code below for future use. at this time, there is no apparent use for
    // it. it doesn't do anything.
    //    dynamic_cast<Uintah::Scheduler*>(getPort("scheduler"))->setPositionVar(pPosLabel);
    double deltMin, deltMax;
    params->findBlock("Time")->require("delt_min", deltMin);
    params->findBlock("Time")->require("delt_max", deltMax);
    const bool useAdaptiveDt = std::abs(deltMax - deltMin) > 2.0*std::numeric_limits<double>::epsilon();
    
    // Multithreading in ExprLib and SpatialOps
    if( wasatchSpec_->findBlock("FieldParallelThreadCount") ){
#    ifdef ENABLE_THREADS
      int spatialOpsThreads=0;
      wasatchSpec_->get( "FieldParallelThreadCount", spatialOpsThreads );
      SpatialOps::set_hard_thread_count(NTHREADS);
      SpatialOps::set_soft_thread_count( spatialOpsThreads );
      proc0cout << "-> Wasatch is running with " << SpatialOps::get_soft_thread_count()
      << " / " << SpatialOps::get_hard_thread_count()
      << " data-parallel threads (SpatialOps)" << std::endl;
#    else
      proc0cout << "NOTE: cannot specify thread counts unless SpatialOps is built with multithreading" << std::endl;
#    endif
    }
    if( wasatchSpec_->findBlock("TaskParallelThreadCount") ){
#    ifdef ENABLE_THREADS
      int exprLibThreads=0;
      wasatchSpec_->get( "TaskParallelThreadCount", exprLibThreads );
      Expr::set_hard_thread_count( NTHREADS );
      Expr::set_soft_thread_count( exprLibThreads );
      proc0cout << "-> Wasatch is running with " << Expr::get_soft_thread_count()
      << " / " << Expr::get_hard_thread_count()
      << " task-parallel threads (ExprLib)" << std::endl;
#    else
      proc0cout << "NOTE: cannot specify thread counts unless SpatialOps is built with multithreading" << std::endl;
#    endif
    }

    // register expressions that calculate coordinates
    register_coordinate_expressions(graphCategories_, isPeriodic_);
    
    //
    // extract the density tag for scalar transport equations and momentum equations
    // and perform error handling
    //
    Expr::Tag densityTag = Expr::Tag();
    bool isConstDensity = true;
    {
      Uintah::ProblemSpecP momEqnParams      = wasatchSpec_->findBlock("MomentumEquations");
      Uintah::ProblemSpecP particleEqnParams = wasatchSpec_->findBlock("ParticleTransportEquations");
      Uintah::ProblemSpecP densityParams     = wasatchSpec_->findBlock("Density");
      Uintah::ProblemSpecP transEqnParams    = wasatchSpec_->findBlock("TransportEquation");
      if( transEqnParams || momEqnParams || particleEqnParams ){
        if( !densityParams ) {
          std::ostringstream msg;
          msg << "ERROR: You must include a 'Density' block in your input file when solving transport equations" << endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        if( densityParams->findBlock("NameTag") ){
          densityTag = parse_nametag( densityParams->findBlock("NameTag") );
          isConstDensity = false;
        }
        else{
          double densVal = 1.0; std::string densName;
          Uintah::ProblemSpecP constDensParam = densityParams->findBlock("Constant");
          constDensParam->getAttribute( "value", densVal );
          constDensParam->getAttribute( "name", densName );
          densityTag = Expr::Tag( densName, Expr::STATE_NONE );
          graphCategories_[INITIALIZATION  ]->exprFactory->register_expression( new Expr::ConstantExpr<SVolField>::Builder(densityTag,densVal) );
          graphCategories_[ADVANCE_SOLUTION]->exprFactory->register_expression( new Expr::ConstantExpr<SVolField>::Builder(densityTag,densVal) );
        }
      }
    }

    // PARSE BC FUNCTORS
    Uintah::ProblemSpecP bcParams = params->findBlock("Grid")->findBlock("BoundaryConditions");
    if (bcParams) {
      for( Uintah::ProblemSpecP faceBCParams=bcParams->findBlock("Face");
           faceBCParams != 0;
           faceBCParams=faceBCParams->findNextBlock("Face") )
      {
        for( Uintah::ProblemSpecP bcTypeParams=faceBCParams->findBlock("BCType");
             bcTypeParams != 0;
             bcTypeParams=bcTypeParams->findNextBlock("BCType") )
        {
          std::string functorName;
          
          Uintah::ProblemSpecP valSpec = bcTypeParams->findBlock("value");
          if   (valSpec) bcTypeParams->get( "value", functorName );
          else           bcTypeParams->getAttribute("value",functorName);
          
          Uintah::ProblemSpec::InputType theInputType = bcTypeParams->getInputType(functorName);
          // if the value of this bc is of type string, then it is a functor. add to the list of functors
          if( theInputType == Uintah::ProblemSpec::STRING_TYPE ){
            
            std::string phiName;
            bcTypeParams->getAttribute("label",phiName);
            
            BCFunctorMap::iterator iter = bcFunctorMap_.find(phiName);
            // check if we already have an entry for phiname
            if( iter != bcFunctorMap_.end() ){
              (*iter).second.insert(functorName);
            }
            else{
              BCFunctorMap::mapped_type functorSet;
              functorSet.insert( functorName );
              bcFunctorMap_.insert( BCFunctorMap::value_type(phiName,functorSet) );
            }
          }          
        }
      }
    }
    
    // PARSE IO FIELDS
    Uintah::ProblemSpecP archiverParams = params->findBlock("DataArchiver");
    for( Uintah::ProblemSpecP saveLabelParams=archiverParams->findBlock("save");
         saveLabelParams != 0;
         saveLabelParams=saveLabelParams->findNextBlock("save") )
    {
      std::string saveTheLabel;
      saveLabelParams->getAttribute("label",saveTheLabel);
      lockedFields_.insert(saveTheLabel);
    }

    //
    // Material
    //
    if( buildWasatchMaterial_ ){
      Uintah::WasatchMaterial* mat= scinew Uintah::WasatchMaterial();
      sharedState->registerWasatchMaterial(mat);
    }

    // we are able to get the solver port from here
    linSolver_ = dynamic_cast<Uintah::SolverInterface*>(getPort("solver"));
    if( !linSolver_ ){
      throw Uintah::InternalError("Wasatch: couldn't get solver port", __FILE__, __LINE__);
    }
    else if( linSolver_ ){
      proc0cout << "Detected solver: " << linSolver_->getName() << std::endl;
      const bool needPressureSolve = wasatchSpec_->findBlock("MomentumEquations") && !(wasatchSpec_->findBlock("MomentumEquations")->findBlock("DisablePressureSolve"));
      if ( (linSolver_->getName()).compare("hypre") != 0 && needPressureSolve) {
        std::ostringstream msg;
        msg << "  Invalid solver specified: "<< linSolver_->getName() << std::endl
        << "  Wasatch currently works with hypre solver only. Please change your solver type." << std::endl
        << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }
    
    std::string timeIntName;
    wasatchSpec_->get("TimeIntegrator",timeIntName);
    timeIntegrator_ = TimeIntegrator(timeIntName);
    nRKStages_ = timeIntegrator_.nStages;

    //
    //  Parse geometry pieces. NOTE: This must take place before create_expressions_from_input
    //  because some input expressions will use the intrusion geometries (e.g. particle initialization)
    //
    parse_embedded_geometry(wasatchSpec_,graphCategories_);
    
    //
    // create expressions explicitly defined in the input file.  These
    // are typically associated with, e.g. initial conditions.
    //
    create_expressions_from_input( wasatchSpec_, graphCategories_ );
    setup_property_evaluation( wasatchSpec_, graphCategories_, lockedFields_ );

    //
    // get the turbulence params, if any, and parse them.
    //
    Uintah::ProblemSpecP turbulenceModelParams = wasatchSpec_->findBlock("Turbulence");
    TurbulenceParameters turbParams;
    parse_turbulence_input(turbulenceModelParams, turbParams);

    //
    // get the variable density model params, if any, and parse them.
    //
    Uintah::ProblemSpecP varDenModelParams = wasatchSpec_->findBlock("VariableDensity");
    VarDenParameters varDenParams;
    parse_varden_input(varDenModelParams, varDenParams);

    //
    // Build transport equations.  This registers all expressions as
    // appropriate for solution of each transport equation.
    //
    for( Uintah::ProblemSpecP transEqnParams=wasatchSpec_->findBlock("TransportEquation");
         transEqnParams != 0;
         transEqnParams=transEqnParams->findNextBlock("TransportEquation") )
    {
      adaptors_.push_back( parse_scalar_equation( transEqnParams, turbParams, densityTag, isConstDensity, graphCategories_ ) );
    }

    //
    // Build coupled transport equations scalability test for wasatch.
    //
    for( Uintah::ProblemSpecP scalEqnParams=wasatchSpec_->findBlock("ScalabilityTest");
         scalEqnParams != 0;
         scalEqnParams=scalEqnParams->findNextBlock("ScalabilityTest") )
    {
      try{
        // note - parse_scalability_test returns a vector of equation adaptors
        EquationAdaptors scalEqnAdaptors = parse_scalability_test( scalEqnParams, graphCategories_ );
        adaptors_.insert( adaptors_.end(), scalEqnAdaptors.begin(), scalEqnAdaptors.end() );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up scalability test equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    //
    // Build momentum transport equations.  This registers all expressions
    // required for solution of each momentum equation.
    //
    for( Uintah::ProblemSpecP momEqnParams=wasatchSpec_->findBlock("MomentumEquations");
         momEqnParams != 0;
         momEqnParams=momEqnParams->findNextBlock("MomentumEquations") )
    {
      try{
          // note - parse_momentum_equations returns a vector of equation adaptors
          const EquationAdaptors adaptors = parse_momentum_equations( momEqnParams,
                                                                      turbParams,
                                                                      varDenParams,
                                                                      useAdaptiveDt,
                                                                      isConstDensity,
                                                                      densityTag,
                                                                      graphCategories_,
                                                                      *linSolver_,
                                                                      sharedState );
        adaptors_.insert( adaptors_.end(), adaptors.begin(), adaptors.end() );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
            << "Problems setting up momentum transport equations.  Details follow:" << endl
            << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    //
    // get the 2D variable density, osicllating (and periodic) mms params, if any, and parse them.
    //
    Uintah::ProblemSpecP varDenOscillatingMMSParams = wasatchSpec_->findBlock("VarDenOscillatingMMS");
    if (varDenOscillatingMMSParams) {
      const bool computeContinuityResidual = wasatchSpec_->findBlock("MomentumEquations")->findBlock("ComputeMassResidual");
      parse_var_den_oscillating_mms(wasatchSpec_, varDenOscillatingMMSParams, computeContinuityResidual, graphCategories_);
    }
    
    //
    // Build moment transport equations.  This registers all expressions
    // required for solution of each momentum equation.
    //
    for( Uintah::ProblemSpecP momEqnParams=wasatchSpec_->findBlock("MomentTransportEquation");
         momEqnParams != 0;
         momEqnParams=momEqnParams->findNextBlock("MomentTransportEquation") )
    {
      // note - parse_moment_transport_equations returns a vector of equation adaptors
      try{
        //For the Multi-Environment mixing model, the entire Wasatch Block must be passed to find values for initial moments
        const EquationAdaptors adaptors =
            parse_moment_transport_equations( momEqnParams, wasatchSpec_, isConstDensity,
                                              graphCategories_ );
        adaptors_.insert( adaptors_.end(), adaptors.begin(), adaptors.end() );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up moment transport equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    //
    // Build poisson equations
    for( Uintah::ProblemSpecP poissonEqnParams=wasatchSpec_->findBlock("PoissonEquation");
         poissonEqnParams != 0;
         poissonEqnParams=poissonEqnParams->findNextBlock("PoissonEquation") )
    {
      try{
        parse_poisson_equation( poissonEqnParams, graphCategories_,
                                *linSolver_, sharedState );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up momentum transport equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    
    if( wasatchSpec_->findBlock("Radiation") ){
      parse_radiation_solver( wasatchSpec_->findBlock("Radiation"),
                              *graphCategories_[ADVANCE_SOLUTION],
                              *linSolver_, sharedState, locked_fields() );
    }

    if( buildTimeIntegrator_ ){
      timeStepper_ = scinew TimeStepper( sharedState_, graphCategories_, timeIntegrator_ );
    }    
    
    //
    // force additional expressions on the graph
    //
    for( Uintah::ProblemSpecP forceOnGraphParams=wasatchSpec_->findBlock("ForceOnGraph");
         forceOnGraphParams != 0;
         forceOnGraphParams=forceOnGraphParams->findNextBlock("ForceOnGraph") )
    {
      std::vector<std::string> taskListNames;
      forceOnGraphParams->getAttribute("tasklist", taskListNames);
      std::vector<std::string>::iterator taskListIter = taskListNames.begin();
      for( ; taskListIter != taskListNames.end(); ++taskListIter ){
        force_expressions_on_graph( forceOnGraphParams, graphCategories_[select_tasklist(*taskListIter)] );
      }
    }

    parse_cleave_requests    ( wasatchSpec_, graphCategories_ );
    parse_attach_dependencies( wasatchSpec_, graphCategories_ );
    //
    // get the variable density mms params, if any, and parse them.
    //
    Uintah::ProblemSpecP VarDenMMSParams = wasatchSpec_->findBlock("VariableDensityMMS");
    if( VarDenMMSParams ){
      const bool computeContinuityResidual = wasatchSpec_->findBlock("MomentumEquations")->findBlock("ComputeMassResidual");
      parse_var_den_mms(wasatchSpec_, VarDenMMSParams, computeContinuityResidual, graphCategories_);
    }
    
    // radiation
    if( params->findBlock("RMCRT") ){
      doRadiation_ = true;
      Uintah::ProblemSpecP radSpec = params->findBlock("RMCRT");
      Uintah::ProblemSpecP radPropsSpec=wasatchSpec_->findBlock("RadProps");
      Uintah::ProblemSpecP RMCRTBenchSpec=wasatchSpec_->findBlock("RMCRTBench");

      Expr::Tag absorptionCoefTag;
      Expr::Tag temperatureTag;
      
      if( radPropsSpec ){
        Uintah::ProblemSpecP greyGasSpec = radPropsSpec->findBlock("GreyGasAbsCoef");
        absorptionCoefTag = parse_nametag(greyGasSpec->findBlock("NameTag"));
        temperatureTag = parse_nametag(greyGasSpec->findBlock("Temperature")->findBlock("NameTag"));
        
      }
      else if( RMCRTBenchSpec ){
        
        const TagNames& tagNames = TagNames::self();
        
        absorptionCoefTag = tagNames.absorption;
        temperatureTag    = tagNames.temperature;
        
        // check which benchmark we are using:
        std::string benchName;
        RMCRTBenchSpec->getAttribute("benchmark", benchName);
        if( benchName == "BurnsChriston" ){
          // register constant temperature expression:
          graphCategories_[ADVANCE_SOLUTION]->exprFactory->register_expression(new Expr::ConstantExpr<SVolField>::Builder(temperatureTag,64.804));
          // register Burns-Christon trilinear abskg
          graphCategories_[ADVANCE_SOLUTION]->exprFactory->register_expression(new BurnsChristonAbskg<SVolField>::Builder(absorptionCoefTag,tagNames.xsvolcoord, tagNames.ysvolcoord, tagNames.zsvolcoord));
        }
      }
      
      const Expr::ExpressionID exprID = graphCategories_[ADVANCE_SOLUTION]->exprFactory->
          register_expression( new RadiationSource::Builder( tag_list( TagNames::self().radiationsource,
                                                                       TagNames::self().radvolq,
                                                                       TagNames::self().radvrflux ),
                                                             temperatureTag,
                                                             absorptionCoefTag,
                                                             TagNames::self().celltype,
                                                             radSpec,
                                                             sharedState_,
                                                             grid ) );
      graphCategories_[ADVANCE_SOLUTION]->exprFactory->cleave_from_parents ( exprID );
      graphCategories_[ADVANCE_SOLUTION]->exprFactory->cleave_from_children( exprID );
      graphCategories_[ADVANCE_SOLUTION]->rootIDs.insert( exprID );
    }
    //
    //
    // Build particle transport equations.  This registers all expressions
    // required for solution of each particle equation.
    //
    Uintah::ProblemSpecP particleEqnSpec = wasatchSpec_->findBlock("ParticleTransportEquations");
    // note - parse_particle_transport_equations returns a vector of equation adaptors
    if( particleEqnSpec ){
      try{
        const EquationAdaptors adaptors = parse_particle_transport_equations( particleEqnSpec, wasatchSpec_, graphCategories_);
        adaptors_.insert( adaptors_.end(), adaptors.begin(), adaptors.end() );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up particle transport equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    //
    // process any reduction variables specified through the input file
    //
    ReductionHelper::self().parse_reduction_spec( wasatchSpec_ );
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleInitialize( const Uintah::LevelP& level,
                                    Uintah::SchedulerP& sched )
  {
    // accessing the sharedState_->allWasatchMaterials() must be done after
    // problemSetup. The sharedstate class will create this material set
    // in postgridsetup, which is called after problemsetup. This is dictated
    // by Uintah.
    if( buildWasatchMaterial_ ){
      set_wasatch_materials(sharedState_->allWasatchMaterials());
      if( doParticles_ ){
        particlesHelper_->set_materials(get_wasatch_materials());
      }
    }
    else{
      if( doParticles_ ){
        particlesHelper_->set_materials(sharedState_->allMaterials());
      }
    }

    setup_patchinfo_map( level, sched );

    const Uintah::PatchSet* const allPatches   = get_patchset( USE_FOR_TASKS, level, sched );
    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );
    
#   ifndef WASATCH_IN_ARCHES // this is a bit annoying... when warches is turned on, disable any linearsolver calls from Wasatch
    if( linSolver_ ) {
      linSolver_->scheduleInitialize( level, sched, 
                                      sharedState_->allMaterials() );
    }
#   endif

    GraphHelper* const icGraphHelper = graphCategories_[ INITIALIZATION ];

    Expr::ExpressionFactory& exprFactory = *icGraphHelper->exprFactory;

    if( doParticles_ ){
      particlesHelper_->schedule_initialize(level,sched);
    }
    
    //bcHelper_ = scinew BCHelper(localPatches, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_);
    bcHelperMap_[level->getID()] = scinew BCHelper(localPatches, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_);
    
    // handle intrusion boundaries
    if( wasatchSpec_->findBlock("EmbeddedGeometry") ){
      apply_intrusion_boundary_conditions( *bcHelperMap_[level->getID()] );
    }

    //_______________________________________
    // set the time
    Expr::TagList timeTags;
    timeTags.push_back( TagNames::self().time     );
    timeTags.push_back( TagNames::self().dt       );
    timeTags.push_back( TagNames::self().timestep );
    timeTags.push_back( TagNames::self().rkstage  );
    exprFactory.register_expression( scinew SetCurrentTime::Builder(timeTags), true );
    
    //_____________________________________________
    // Build the initial condition expression graph
    if( !icGraphHelper->rootIDs.empty() ){
      
      // -----------------------------------------------------------------------
      // INITIAL BOUNDARY CONDITIONS TREATMENT
      // -----------------------------------------------------------------------
      typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;

      proc0cout << "------------------------------------------------\n"
                << "SETTING INITIAL BOUNDARY CONDITIONS:\n"
                << "------------------------------------------------\n";

      for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
        EqnTimestepAdaptorBase* const adaptor = *ia;
        EquationBase* transEq = adaptor->equation();
        std::string eqnLabel = transEq->solution_variable_name();
        //______________________________________________________
        // set up initial boundary conditions on this transport equation
        try{
          transEq->setup_boundary_conditions( *bcHelperMap_[level->getID()], graphCategories_);
          proc0cout << "Setting Initial BCs for transport equation '" << eqnLabel << "'\n";
          transEq->apply_initial_boundary_conditions( *icGraphHelper, *bcHelperMap_[level->getID()]);
        }
        catch( std::runtime_error& e ){
          std::ostringstream msg;
          msg << e.what() << "\nERORR while setting initial boundary conditions on equation '" << eqnLabel << "'\n";
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      }
      proc0cout << "------------------------------------------------\n";

      // -----------------------------------------------------------------------
      try{
        TaskInterface* const task = scinew TaskInterface( icGraphHelper->rootIDs,
                                                          "initialization",
                                                          *icGraphHelper->exprFactory,
                                                          level, sched,
                                                          allPatches,
                                                          materials_,
                                                          patchInfoMap_,
                                                          1,
                                                          sharedState_,
                                                          lockedFields_ );
        //_______________________________________________________
        // create the TaskInterface and schedule this task for
        // execution.  Note that field dependencies are assigned
        // within the TaskInterface object.
        task->schedule();
        taskInterfaceList_.push_back( task );
      }
      catch( std::exception& err ){
        std::ostringstream msg;
        msg << "ERROR SETTING UP GRAPH FOR INITIALIZATION" << std::endl
            << err.what() << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    
    // Compute the cell type only when radiation is present. This may change in the future.
    if( doRadiation_ ) cellType_->schedule_compute_celltype( allPatches, materials_, sched );

    if(doParticles_ ) particlesHelper_->schedule_sync_particle_position( level, sched, true );
    
    proc0cout << "Wasatch: done creating initialization task(s)" << std::endl;
  }

  //--------------------------------------------------------------------
  
  void Wasatch::restartInitialize()
  {
    isRestarting_ = true;

    // accessing the sharedState_->allWasatchMaterials() must be done after
    // problemSetup. The sharedstate class will create this material set
    // in postgridsetup, which is called after problemsetup. This is dictated
    // by Uintah.
    if( buildWasatchMaterial_ ){
      set_wasatch_materials( sharedState_->allWasatchMaterials() );
      if( doParticles_ ){
        particlesHelper_->set_materials(get_wasatch_materials());
      }
    } 
  }

  //--------------------------------------------------------------------

  void Wasatch::setup_patchinfo_map( const Uintah::LevelP& level,
                                     Uintah::SchedulerP& sched )
  {
    //_______________________________________________________________
    // Set up the operators associated with the local patches.  We
    // only need to do this once, so we choose to do it here.  It
    // could just as well be done on any other schedule callback that
    // has access to the levels (patches).
    //
    // Also save off the timestep label information.
    //
    const Uintah::PatchSet* patches = get_patchset( USE_FOR_OPERATORS, level, sched );

    for( int ipss=0; ipss<patches->size(); ++ipss ){
      const Uintah::PatchSubset* pss = patches->getSubset(ipss);
      for( int ip=0; ip<pss->size(); ++ip ){
        SpatialOps::OperatorDatabase* const opdb = scinew SpatialOps::OperatorDatabase();
        const Uintah::Patch* const patch = pss->get(ip);

        //tsaad: register an patch container as an operator for easy access to the Uintah patch
        // inside of an expression.
        opdb->register_new_operator<UintahPatchContainer>(scinew UintahPatchContainer(patch) );
        
        opdb->register_new_operator<TimeIntegrator>(scinew TimeIntegrator(timeIntegrator_.name) );
        
        build_operators( *patch, *opdb );
        PatchInfo& pi = patchInfoMap_[patch->getID()];
        pi.operators = opdb;
        pi.patchID = patch->getID();
//        std::cout << "Set up operators for Patch ID: " << patch->getID()
//                  << " on level " << level->getID()
//                  << " and process " << Uintah::Parallel::getMPIRank() << std::endl;
      }
    }
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    GraphHelper* const tsGraphHelper = graphCategories_[ TIMESTEP_SELECTION ];
    const Uintah::PatchSet* const localPatches = get_patchset(USE_FOR_TASKS,level,sched);

    if( tsGraphHelper->rootIDs.size() > 0 ){
      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      TaskInterface* const task = scinew TaskInterface( tsGraphHelper->rootIDs,
                                                        "compute timestep",
                                                        *tsGraphHelper->exprFactory,
                                                        level, sched,
                                                        localPatches,
                                                        materials_,
                                                        patchInfoMap_,
                                                        1, sharedState_, lockedFields_ );
      task->schedule(1);
      taskInterfaceList_.push_back( task );
    }
    else{ // default

      proc0cout << "Scheduling Task 'compute timestep' COMPUTES 'delT' in NEW data warehouse" << endl;

      Uintah::Task* task = scinew Uintah::Task( "compute timestep", this, &Wasatch::computeDelT );

      // jcs it appears that for reduction variables we cannot specify the patches - only the materials.
      	task->computes( sharedState_->get_delt_label(),
                      level.get_rep() );
      //              materials_->getUnion() );
      // jcs why can't we specify a material here?  It doesn't seem to be working if I do.
      
      const GraphHelper* slnGraphHelper = graphCategories_[ADVANCE_SOLUTION];
      const TagNames& tagNames = TagNames::self();
      const bool useStableDT = slnGraphHelper->exprFactory->have_entry( tagNames.stableTimestep );
      // since the StableDT expression is only registered on the time_advance graph,
      // make the necessary checks before adding a requires for that
      if( sharedState_->getCurrentTopLevelTimeStep() > 0 ){
        if( useStableDT ){
          task->requires(Uintah::Task::NewDW, Uintah::VarLabel::find(tagNames.stableTimestep.name()),  Uintah::Ghost::None, 0);
        }
      }
                  
      sched->addTask( task, localPatches, materials_ );
    }

    proc0cout << "Wasatch: done creating timestep task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void
  Wasatch::scheduleTimeAdvance( const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched )
  {
    const Uintah::PatchSet* const allPatches = get_patchset( USE_FOR_TASKS, level, sched );
    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );
    const GraphHelper* advSolGraphHelper = graphCategories_[ ADVANCE_SOLUTION ];

    if( isRestarting_ ){
      setup_patchinfo_map( level, sched );
      
      
      if( doParticles_ ){
        particlesHelper_->schedule_restart_initialize(level,sched);
        particlesHelper_->schedule_find_boundary_particles(level,sched);
      }

      //bcHelper_ = scinew BCHelper(localPatches, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_);
      bcHelperMap_[level->getID()] = scinew BCHelper(localPatches, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_);
    }
    
    if( doParticles_ ){
      particlesHelper_->schedule_find_boundary_particles(level,sched);
    }
    
    for( int iStage=1; iStage<=nRKStages_; iStage++ ){
      // jcs why do we need this instead of getting the level?
      // jcs notes:
      //
      //   eachPatch() returns a PatchSet that will result in the task
      //       being executed asynchronously accross all patches.  This
      //       can improve performance but will deadlock if any global MPI
      //       calls occur.
      //
      //   allPatches() returns a PatchSet that results in the task being
      //       executed together across all patches.  This is required if
      //       any global MPI syncronizations occurr (e.g. in a linear
      //       solve)
      //    also need to set a flag on the task: task->setType(Task::OncePerProc);
      
      // set up any "old" variables that have been requested.
      OldVariable::self().setup_tasks( allPatches, materials_, sched, iStage );

      // Compute the cell type only when radiation is present. This may change in the future.
      if( doRadiation_ ) cellType_->schedule_carry_forward(allPatches,materials_,sched);

      // -----------------------------------------------------------------------
      // BOUNDARY CONDITIONS TREATMENT
      // -----------------------------------------------------------------------
      proc0cout << "------------------------------------------------" << std::endl
      << "SETTING BOUNDARY CONDITIONS:" << std::endl;
      proc0cout << "------------------------------------------------" << std::endl;

      typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
      
      for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
        EqnTimestepAdaptorBase* const adaptor = *ia;
        EquationBase* transEq = adaptor->equation();
        std::string eqnLabel = transEq->solution_variable_name();
        //______________________________________________________
        // set up boundary conditions on this transport equation
        try{
          // only verify boundary conditions on the first stage!
          if( isRestarting_ && iStage < 2 ) transEq->setup_boundary_conditions(*bcHelperMap_[level->getID()], graphCategories_);
          proc0cout << "Setting BCs for transport equation '" << eqnLabel << "'" << std::endl;
          transEq->apply_boundary_conditions(*advSolGraphHelper, *bcHelperMap_[level->getID()]);
        }
        catch( std::runtime_error& e ){
          std::ostringstream msg;
          msg << e.what()
          << std::endl
          << "ERORR while setting boundary conditions on equation '" << eqnLabel << "'"
          << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      }
      proc0cout << "------------------------------------------------" << std::endl;
      
      //
      // process clipping on fields - must be done AFTER all bcs are applied
      //
      process_field_clipping( wasatchSpec_, graphCategories_, localPatches );

      if( buildTimeIntegrator_ ){
        create_timestepper_on_patches( allPatches, materials_, level, sched, iStage );
      }

      proc0cout << "Wasatch: done creating solution task(s)" << std::endl;
      
      // pass the bc Helper to pressure expressions on all patches
      bcHelperMap_[level->getID()]->synchronize_pressure_expression();
    }

    
    // post processing
    GraphHelper* const postProcGH = graphCategories_[ POSTPROCESSING ];
    Expr::ExpressionFactory& postProcFactory = *postProcGH->exprFactory;
    if( !postProcGH->rootIDs.empty() ){
      TaskInterface* const task = scinew TaskInterface( postProcGH->rootIDs,
                                                       "postprocessing",
                                                       postProcFactory,
                                                       level, sched,
                                                       allPatches,
                                                       materials_,
                                                       patchInfoMap_,
                                                       1,
                                                       sharedState_,
                                                       lockedFields_ );
      task->schedule(1);
      taskInterfaceList_.push_back( task );
    }
    proc0cout << "Wasatch: done creating post-processing task(s)" << std::endl;

    // ensure that any "CARRY_FORWARD" variable has an initialization provided for it.
    if( buildTimeIntegrator_ ){ // make sure that we have a timestepper created - this is needed for wasatch-in-arches
      const Expr::ExpressionFactory* const icFactory = graphCategories_[INITIALIZATION]->exprFactory;
      typedef std::list< TaskInterface* > TIList;
      bool isOk = true;
      Expr::TagList missingTags;
      const TIList& tilist = timeStepper_->get_task_interfaces();
      for( TIList::const_iterator iti=tilist.begin(); iti!=tilist.end(); ++iti ){
        const Expr::TagList tags = (*iti)->collect_tags_in_task();
        for( Expr::TagList::const_iterator itag=tags.begin(); itag!=tags.end(); ++itag ){
          if( itag->context() == Expr::CARRY_FORWARD ){
            if( !icFactory->have_entry(*itag) ) missingTags.push_back( *itag );
          }
        }
      }
      if( !isOk ){
        std::ostringstream msg;
        msg << "ERORR: The following fields were marked 'CARRY_FORWARD' but were not initialized." << std::endl
            << "       Ensure that all of these fields are present on the initialization graph:" << std::endl;
        for( Expr::TagList::const_iterator it=missingTags.begin(); it!=missingTags.end(); ++it ){
          msg << "         " << *it << std::endl;
        }
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    
    //_________________________
    // After the time advance graphs have all finished executing, it is time
    // to synchronize the Wasatch particle position varibles with Uintah's.
    // Recall that Uintah requires that particle position be specified as a
    // Uintah::Point whereas Wasatch uses x, y, and z variables, separately.
    if( doParticles_ ){
      particlesHelper_->schedule_transfer_particle_ids(level,sched);
      particlesHelper_->schedule_sync_particle_position(level,sched);
      particlesHelper_->schedule_relocate_particles(level,sched);
      particlesHelper_->schedule_add_particles(level,sched);
    }
    
    if (isRestarting_) isRestarting_ = false;
  }

  //--------------------------------------------------------------------

  void
  Wasatch::create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                          const Uintah::MaterialSet* const materials,
                                          const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched,
                                          const int rkStage )
  {
    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];
    Expr::ExpressionFactory& exprFactory = *gh->exprFactory;

    if( adaptors_.size() == 0 && gh->rootIDs.empty()) return; // no equations registered.

    //_____________________________________________________________
    // create an expression to set the current time as a field that
    // will be available to all expressions if needed.
    Expr::ExpressionID timeID;
    if( rkStage==1 && !exprFactory.have_entry(TagNames::self().time) ){
      Expr::TagList timeTags;
      timeTags.push_back( TagNames::self().time     );
      timeTags.push_back( TagNames::self().dt     );
      timeTags.push_back( TagNames::self().timestep );
      timeTags.push_back( TagNames::self().rkstage  );
      timeID = exprFactory.register_expression( scinew SetCurrentTime::Builder(timeTags), true );
    }
    else{
      timeID = exprFactory.get_id(TagNames::self().time);
    }

    //___________________________________________
    // Plug in each equation that has been set up
    if( rkStage==1 ){
      for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
        const EqnTimestepAdaptorBase* const adaptor = *ia;
        try{
          adaptor->hook( *timeStepper_ );
        }
        catch( std::exception& e ){
          std::ostringstream msg;
          msg << "Problems plugging transport equation for '"
          << adaptor->equation()->solution_variable_name()
          << "' into the time integrator" << std::endl
          << e.what() << std::endl;
          proc0cout << msg.str() << endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      }
    }

    //____________________________________________________________________
    // create all of the required tasks on the timestepper.  This involves
    // the task(s) that compute(s) the RHS for each transport equation and
    // the task that updates the variables from time "n" to "n+1"
    timeStepper_->create_tasks( timeID, patchInfoMap_, localPatches,
                                materials, level, sched,
                                rkStage, lockedFields_ );
  }

  //--------------------------------------------------------------------

  void
  Wasatch::computeDelT( const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches,
                        const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* old_dw,
                        Uintah::DataWarehouse* new_dw )
  {
    Uintah::delt_vartype deltat = 1.0;
    double val = 9999999999999.0;
    
    const GraphHelper* slnGraphHelper = graphCategories_[ADVANCE_SOLUTION];
    const TagNames& tagNames = TagNames::self();
    const bool useStableDT = slnGraphHelper->exprFactory->have_entry( tagNames.stableTimestep );
    if( sharedState_->getCurrentTopLevelTimeStep() > 0 ){
      if( useStableDT ){
        //__________________
        // loop over patches
        for( int ip=0; ip<patches->size(); ++ip ){
          // grab the stable timestep value calculated by the StableDT expression
          Uintah::PerPatch<double*> tempDtP;
          new_dw->get(tempDtP, Uintah::VarLabel::find(tagNames.stableTimestep.name()), 0, patches->get(ip));          
          val = std::min( val, *tempDtP );
        }
      }
      else {
        // FOR FIXED dt: (min = max in input file)
        // if this is not the first timestep, then grab dt from the olddw.
        // This will avoid Uintah's message that it is setting dt to max dt/min dt
        old_dw->get( deltat, sharedState_->get_delt_label() );
      }
    }
    
    if( useStableDT ){
      new_dw->put(Uintah::delt_vartype(val),sharedState_->get_delt_label(),
                  Uintah::getLevel(patches) );
    }
    else{
      new_dw->put( deltat,
                  sharedState_->get_delt_label(),
                  Uintah::getLevel(patches) );
    }
  }

  //------------------------------------------------------------------

  const Uintah::PatchSet*
  Wasatch::get_patchset( const PatchsetSelector pss,
                         const Uintah::LevelP& level,
                         Uintah::SchedulerP& sched )
  {
    switch ( pss ) {

      case USE_FOR_TASKS:
        // return sched->getLoadBalancer()->getPerProcessorPatchSet(level);
        return level->eachPatch();
        break;

      case USE_FOR_OPERATORS: {
        const int levelID = level->getID();
        const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
        const Uintah::PatchSubset* const localPatches = allPatches->getSubset( d_myworld->myrank() );

        std::map< int, const Uintah::PatchSet* >::iterator ip = patchesForOperators_.find( levelID );

        if( ip != patchesForOperators_.end() ) return ip->second;

        Uintah::PatchSet* patches = new Uintah::PatchSet;
        // jcs: this results in "normal" scheduling and WILL NOT WORK FOR LINEAR SOLVES
        //      in that case, we need to use "gang" scheduling: addAll( localPatches )
        patches->addEach( localPatches->getVector() );
        //     const std::set<int>& procs = sched->getLoadBalancer()->getNeighborhoodProcessors();
        //     for( std::set<int>::const_iterator ip=procs.begin(); ip!=procs.end(); ++ip ){
        //       patches->addEach( allPatches->getSubset( *ip )->getVector() );
        //     }
        patchesForOperators_[levelID] = patches;
        return patches;
      }
    }
    return NULL;
  }

 //------------------------------------------------------------------

 void
 Wasatch::scheduleCoarsen( const Uintah::LevelP& /*coarseLevel*/,
                           Uintah::SchedulerP& /*sched*/ )
 {
   // do nothing for now
 }

 //------------------------------------------------------------------

 void
 Wasatch::scheduleRefineInterface( const Uintah::LevelP& /*fineLevel*/,
                                   Uintah::SchedulerP& /*scheduler*/,
                                   bool, bool )
 {
   // do nothing for now
 }
//------------------------------------------------------------------

} // namespace Wasatch
