/*
 * The MIT License
 *
 * Copyright (c) 2010-2018 The University of Utah
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
#include <sci_defs/wasatch_defs.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManager.h>
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
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Expressions/StableTimestep.h>
#include <CCA/Components/Wasatch/CoordinateHelper.h>
#include <CCA/Components/Wasatch/DualTimeMatrixManager.h>

//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggered.h>
#ifdef ENABLE_THREADS
#  include <spatialops/SpatialOpsTools.h>
#  include <expression/SchedulerBase.h>
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
#include <CCA/Components/Wasatch/Expressions/NullExpression.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseParticleEquations.h>
#include <CCA/Components/Wasatch/Transport/PreconditioningParser.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/FieldClippingTools.h>
#include <CCA/Components/Wasatch/OldVariable.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>
#include <CCA/Components/Wasatch/ParticlesHelper.h>
#include <CCA/Components/Wasatch/WasatchParticlesHelper.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/Expressions/CellType.h>
#include <CCA/Components/Wasatch/Expressions/TestNestedExpression.h>

#ifdef HAVE_POKITT
//-- includes for coal models --//
#include <CCA/Components/Wasatch/Coal/CoalEquation.h>
#include <CCA/Components/Wasatch/Transport/SetupCoalModels.h>
#endif

using std::endl;
WasatchCore::FlowTreatment WasatchCore::Wasatch::flowTreatment_;
bool WasatchCore::Wasatch::needPressureSolve_ = false;
bool WasatchCore::Wasatch::hasDualTime_ = false;
namespace WasatchCore{

  //--------------------------------------------------------------------

  Wasatch::Wasatch( const Uintah::ProcessorGroup* myworld,
                    const Uintah::MaterialManagerP materialManager )
    : Uintah::ApplicationCommon( myworld, materialManager ),
      buildTimeIntegrator_ ( true ),
      buildWasatchMaterial_( true ),
      nRKStages_(1),
      isPeriodic_ (true ),
      doRadiation_(false),
      doParticles_(false),
      totalDualTimeIterations_(0),
      timeIntegrator_(TimeIntegrator("FE")),
      subsched_(nullptr),
      compileDualTimeSched_(true),
      dtLabel_     (nullptr),
      tLabel_      (nullptr),
      tStepLabel_  (nullptr),
      rkStageLabel_(nullptr)
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

    materials_   = nullptr;
    timeStepper_ = nullptr;

    isRestarting_ = false;

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::s_combine_memory = false;

    const bool log = false;
    graphCategories_[ INITIALIZATION     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ TIMESTEP_SELECTION ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ ADVANCE_SOLUTION   ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ POSTPROCESSING     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );

    dualTimeMatrixInfo_ = scinew DualTimeMatrixInfo();

    OldVariable::self().sync_with_wasatch( this );
    ReductionHelper::self().sync_with_wasatch( this );
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
    
//    for( DTIntegratorMapT::iterator it=dualTimeIntegrators_.begin(); it != dualTimeIntegrators_.end(); ++it ){
//      delete it->second;
//    }
//
//    for( auto it=dualTimeMatrixManagers_.begin(); it != dualTimeMatrixManagers_.end(); ++it ){
//      delete it->second;
//    }

    for( auto it=dualTimePatchMap_.begin(); it != dualTimePatchMap_.end(); ++it ){
      delete it->second.first;
      delete it->second.second;
    }

    delete dualTimeMatrixInfo_;

    if( doRadiation_ ){
      delete rmcrt_;
      delete cellType_;
    }
    if( doParticles_ ) delete particlesHelper_;
    
    Uintah::VarLabel::destroy(dtLabel_);
    Uintah::VarLabel::destroy(tLabel_);
    Uintah::VarLabel::destroy(tStepLabel_);
    Uintah::VarLabel::destroy(rkStageLabel_);
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
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("NameTag") )
    {
      const Expr::ExpressionID exprID = gh->exprFactory->get_id(parse_nametag(exprParams));
      gh->rootIDs.insert( exprID );
    }
  }

  //--------------------------------------------------------------------
  
  void check_periodicity_extra_cells( const Uintah::ProblemSpecP& uintahSpec,
                                      Uintah::IntVector& extraCells,
                                      bool& isPeriodic)
  {
    // disallow different periodicities on multiple levels
    std::vector<Uintah::IntVector> levelPeriodicityVectors;
    // disallow specification of extraCells and different periodicities on multiple levels
    int nlevels = 0;
    std::ostringstream msg;
    bool foundExtraCells = false;
    Uintah::ProblemSpecP gridspec = uintahSpec->findBlock("Grid");
    for( Uintah::ProblemSpecP level = gridspec->findBlock("Level");
         level != nullptr;
         level = gridspec->findNextBlock("Level") ){
      nlevels++;
      Uintah::IntVector periodicDirs(0,0,0);
      level->get("periodic", periodicDirs);
      levelPeriodicityVectors.push_back(periodicDirs);
      
      for( Uintah::ProblemSpecP box = level->findBlock("Box");
           box != nullptr;
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
    if( foundExtraCells ){
      msg << endl
      << "  Specification of 'extraCells' is forbidden in Wasatch. The number of extraCells is automatically determined." << endl
      << "  Please remove it from your input file." << endl
      << endl;
      throw std::runtime_error( msg.str() );
    }
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
         faceSpec != nullptr; faceSpec=faceSpec->findNextBlock("Face"), ++i ){
      
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

  
  void Wasatch::preGridProblemSetup(const Uintah::ProblemSpecP& uintahSpec,
                                    Uintah::GridP& grid)
  {
    Uintah::IntVector extraCells;
    check_periodicity_extra_cells( uintahSpec, extraCells, isPeriodic_);
    grid->setExtraCells(extraCells);
  }
  
  //--------------------------------------------------------------------

  void Wasatch::problemSetup( const Uintah::ProblemSpecP& uintahSpec,
                              const Uintah::ProblemSpecP& restartSpec,
                              Uintah::GridP& grid )
  {
    wasatchSpec_ = uintahSpec->findBlock("Wasatch");
    if( !wasatchSpec_ ) return;

    //
    // Check whether we are solving for particles
    //
    doParticles_ = wasatchSpec_->findBlock("ParticleTransportEquations");
    if( doParticles_ ){
      particlesHelper_ = scinew WasatchParticlesHelper();
      particlesHelper_->sync_with_wasatch(this);
      particlesHelper_->problem_setup( uintahSpec, wasatchSpec_->findBlock("ParticleTransportEquations") );
    }

    // setup names for all the boundary condition faces that do NOT have a name or that have duplicate names
    if( uintahSpec->findBlock("Grid") ){
      Uintah::ProblemSpecP bcProbSpec = uintahSpec->findBlock("Grid")->findBlock("BoundaryConditions");
      assign_unique_boundary_names( bcProbSpec );
    }
    
    // TSAAD: keep the line of code below for future use. at this time, there is no apparent use for
    // it. it doesn't do anything.
    //    m_scheduler->setPositionVar(pPosLabel);
    double deltMin, deltMax;
    uintahSpec->findBlock("Time")->require("delt_min", deltMin);
    uintahSpec->findBlock("Time")->require("delt_max", deltMax);
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
        
        std::string flowTreatment;
        densityParams->getAttribute( "method", flowTreatment );
        Wasatch::set_flow_treatment( flowTreatment );
        
        if( densityParams->findBlock("NameTag") ){
          densityTag = parse_nametag( densityParams->findBlock("NameTag") );
          if     ( Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE ) densityTag.reset_context(Expr::STATE_DYNAMIC);
          else if( Wasatch::flow_treatment() == WasatchCore::LOWMACH      ) densityTag.reset_context(Expr::STATE_N      );
          isConstDensity = false;
        }
        else{
          double densVal = 1.0; std::string densName;
          Uintah::ProblemSpecP constDensParam = densityParams->findBlock("Constant");
          constDensParam->getAttribute( "value", densVal );
          constDensParam->getAttribute( "name", densName );
          densityTag = Expr::Tag( densName, Expr::STATE_NONE );
          graphCategories_[INITIALIZATION  ]->exprFactory->register_expression( scinew Expr::ConstantExpr<SVolField>::Builder(densityTag,densVal) );
          graphCategories_[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew Expr::ConstantExpr<SVolField>::Builder(densityTag,densVal) );
        }
      }
    }

    // PARSE BC FUNCTORS
    Uintah::ProblemSpecP bcParams = uintahSpec->findBlock("Grid")->findBlock("BoundaryConditions");
    if( bcParams ){
      for( Uintah::ProblemSpecP faceBCParams=bcParams->findBlock("Face");
           faceBCParams != nullptr;
           faceBCParams=faceBCParams->findNextBlock("Face") )
      {
        for( Uintah::ProblemSpecP bcTypeParams=faceBCParams->findBlock("BCType");
             bcTypeParams != nullptr;
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
    Uintah::ProblemSpecP archiverParams = uintahSpec->findBlock("DataArchiver");
    for( Uintah::ProblemSpecP saveLabelParams=archiverParams->findBlock("save");
         saveLabelParams != nullptr;
         saveLabelParams=saveLabelParams->findNextBlock("save") )
    {
      std::string iof;
      saveLabelParams->getAttribute("label",iof);
      make_field_persistent(iof);
    }

    //
    // Material
    //
    if( buildWasatchMaterial_ ){
      Uintah::WasatchMaterial* mat= scinew Uintah::WasatchMaterial();
      m_materialManager->registerMaterial( "Wasatch", mat);
    }

    // we are able to get the solver port from here
    proc0cout << "Detected solver: " << m_solver->getName() << std::endl;
    const bool hasMomentum = wasatchSpec_->findBlock("MomentumEquations");
    if( hasMomentum ){
      std::string densMethod;
      wasatchSpec_->findBlock("Density")->getAttribute("method",densMethod);
      const bool isCompressible = densMethod == "COMPRESSIBLE";
      
      bool needPressureSolve = true;
      if( wasatchSpec_->findBlock("MomentumEquations")->findBlock("DisablePressureSolve") ) needPressureSolve = false;
      if( isCompressible ) needPressureSolve = false;
      
      Wasatch::need_pressure_solve( needPressureSolve );
      
      if( needPressureSolve && (m_solver->getName()).compare("hypre") != 0 ){
        std::ostringstream msg;
        msg << "  Invalid solver specified: "<< m_solver->getName() << std::endl
            << "  Wasatch currently works with hypre solver only. Please change your solver type." << std::endl
            << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }
    
    std::string timeIntName;
    wasatchSpec_->get("TimeIntegrator",timeIntName);
    timeIntegrator_ = TimeIntegrator(timeIntName);
    nRKStages_ = timeIntegrator_.nStages;

    // parse dual time specification, set coordinate tags for matrix assembly
    if( wasatchSpec_->findBlock("DualTime") ){
      Uintah::ProblemSpecP dualTimeSpec = wasatchSpec_->findBlock("DualTime");
      timeIntegrator_.has_dual_time(true);

      if ( dualTimeSpec->findAttribute("blockimplicit") ){
        dualTimeSpec->getAttribute( "blockimplicit", dualTimeMatrixInfo_->doBlockImplicit );
      }

      if ( dualTimeSpec->findAttribute("localcflvnn") )
        dualTimeSpec->getAttribute( "localcflvnn", dualTimeMatrixInfo_->doLocalCflVnn );

      if ( dualTimeSpec->findAttribute("precondition") )
        dualTimeSpec->getAttribute( "precondition", dualTimeMatrixInfo_->doPreconditioning );

      if ( dualTimeSpec->findAttribute("doImplicitInviscid" ) )
        dualTimeSpec->getAttribute( "doImplicitInviscid", dualTimeMatrixInfo_->doImplicitInviscid );

      if ( dualTimeSpec->findAttribute("cfl") )
        dualTimeSpec->getAttribute( "cfl", dualTimeMatrixInfo_->cfl );

      if ( dualTimeSpec->findAttribute("vnn") )
        dualTimeSpec->getAttribute( "vnn", dualTimeMatrixInfo_->vnn );

      if ( dualTimeSpec->findAttribute("maxvalue") )
        dualTimeSpec->getAttribute( "maxvalue", dualTimeMatrixInfo_->dsMax );

      if ( dualTimeSpec->findAttribute("minvalue") )
        dualTimeSpec->getAttribute( "minvalue", dualTimeMatrixInfo_->dsMin );

      if ( dualTimeSpec->findAttribute("ds") )
        dualTimeSpec->getAttribute( "ds", dualTimeMatrixInfo_->constantDs );

      if ( dualTimeSpec->findAttribute("iterations") )
        dualTimeSpec->getAttribute( "iterations", dualTimeMatrixInfo_->maxIterations );

      if ( dualTimeSpec->findAttribute("tolerance") )
        dualTimeSpec->getAttribute( "tolerance", dualTimeMatrixInfo_->tolerance );

      if ( dualTimeSpec->findAttribute("lograte") )
        dualTimeSpec->getAttribute( "lograte", dualTimeMatrixInfo_->logIterationRate );

      // set a few tags
      const TagNames& tagNames = TagNames::self();
      dualTimeMatrixInfo_->soundSpeed = tagNames.soundspeed;
      dualTimeMatrixInfo_->timeStepSize = tagNames.dt;
      dualTimeMatrixInfo_->xCoord = TagNames::self().xsvolcoord;
      dualTimeMatrixInfo_->yCoord = TagNames::self().ysvolcoord;
      dualTimeMatrixInfo_->zCoord = TagNames::self().zsvolcoord;

      has_dual_time(true);
    }

    //
    //  Parse geometry pieces. NOTE: This must take place before create_expressions_from_input
    //  because some input expressions will use the intrusion geometries (e.g. particle initialization)
    //
    parse_embedded_geometry(wasatchSpec_,graphCategories_);
    
    //
    // create expressions explicitly defined in the input file.  These
    // are typically associated with, e.g. initial conditions, source terms, or post-processing.
    //
    create_expressions_from_input( uintahSpec, graphCategories_ );
    
    //
    // setup property evaluations
    //
    setup_property_evaluation( wasatchSpec_, graphCategories_, persistentFields_ );

    //
    // get the turbulence params, if any, and parse them.
    //
    Uintah::ProblemSpecP turbulenceModelParams = wasatchSpec_->findBlock("Turbulence");
    TurbulenceParameters turbParams;
    parse_turbulence_input(turbulenceModelParams, turbParams);

    //
    // Build transport equations.  This registers all expressions as
    // appropriate for solution of each transport equation.
    //
    for( Uintah::ProblemSpecP transEqnParams=wasatchSpec_->findBlock("TransportEquation");
         transEqnParams != nullptr;
         transEqnParams=transEqnParams->findNextBlock("TransportEquation") )
    {
      adaptors_.push_back( parse_scalar_equation( transEqnParams,
                                                  wasatchSpec_,
                                                  turbParams,
                                                  densityTag,
                                                  graphCategories_,
                                                  *dualTimeMatrixInfo_,
                                                  persistentFields_ ) );
    }

    //
    // Build species transport equations
    //
    Uintah::ProblemSpecP specEqnParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    Uintah::ProblemSpecP momEqnParams = wasatchSpec_->findBlock("MomentumEquations");
    if( specEqnParams ){
      EquationAdaptors specEqns = parse_species_equations( specEqnParams,
                                                           wasatchSpec_,
                                                           momEqnParams,
                                                           turbParams,
                                                           densityTag,
                                                           graphCategories_,
                                                           *dualTimeMatrixInfo_,
                                                           dualTimeMatrixInfo_->doBlockImplicit );
      adaptors_.insert( adaptors_.end(), specEqns.begin(), specEqns.end() );
    }

    //
    // Build coupled transport equations scalability test for wasatch.
    //
    for( Uintah::ProblemSpecP scalEqnParams=wasatchSpec_->findBlock("ScalabilityTest");
         scalEqnParams != nullptr;
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
         momEqnParams != nullptr;
         momEqnParams=momEqnParams->findNextBlock("MomentumEquations") )
    {
      try{
          // note - parse_momentum_equations returns a vector of equation adaptors
          const EquationAdaptors adaptors = parse_momentum_equations( wasatchSpec_,
                                                                      turbParams,
                                                                      useAdaptiveDt,
                                                                      doParticles_,
                                                                      densityTag,
                                                                      graphCategories_,
                                                                      *m_solver,
                                                                      m_materialManager,
                                                                      *dualTimeMatrixInfo_,
                                                                      persistentFields_ );
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

    {
      PreconditioningParser precondParser( wasatchSpec_, graphCategories_ );
    }

    //
    // get the 2D variable density, oscillating (and periodic) mms params, if any, and parse them.
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
         momEqnParams != nullptr;
         momEqnParams=momEqnParams->findNextBlock("MomentTransportEquation") )
    {
      // note - parse_moment_transport_equations returns a vector of equation adaptors
      try{
        //For the Multi-Environment mixing model, the entire Wasatch Block must be passed to find values for initial moments
        const EquationAdaptors adaptors =
            parse_moment_transport_equations( momEqnParams, wasatchSpec_, graphCategories_ );
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
         poissonEqnParams != nullptr;
         poissonEqnParams=poissonEqnParams->findNextBlock("PoissonEquation") )
    {
      try{
        parse_poisson_equation( poissonEqnParams, graphCategories_,
                                *m_solver, m_materialManager );
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
                              *m_solver, m_materialManager, persistent_fields() );
    }

    if( buildTimeIntegrator_ ){
      timeStepper_ = scinew TimeStepper( graphCategories_, timeIntegrator_ );
    }

    // -----------------------------------------------------------------------
    // nested expression tester
    //
    if( wasatchSpec_->findBlock("TestNestedExpression") ){
      Uintah::ProblemSpecP testParams = wasatchSpec_->findBlock("TestNestedExpression");
      test_nested_expression( testParams,
                              graphCategories_,
                              persistentFields_ );
    }    
    
    //
    // force additional expressions on the graph
    //
    for( Uintah::ProblemSpecP forceOnGraphParams=wasatchSpec_->findBlock("ForceOnGraph");
         forceOnGraphParams != nullptr;
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
    if( uintahSpec->findBlock("RMCRT") ){

      //---------------------------------------------------------------------------------------------------------------------------
      // Added for temporal scheduling support when using RMCRT - APH 05/24/17
      //---------------------------------------------------------------------------------------------------------------------------
      // For RMCRT there will be 2 task graphs - put the radiation tasks in TG-1, otherwise tasks go into TG-0, or both TGs
      //   TG-0 == carry forward and/or non-radiation timesteps
      //   TG-1 == RMCRT radiation timestep
      m_scheduler->setNumTaskGraphs(2);
      Uintah::ProblemSpecP radFreqSpec = uintahSpec;
      radFreqSpec->getWithDefault( "calc_frequency",  radCalcFrequency_, 1 );
      //---------------------------------------------------------------------------------------------------------------------------

      doRadiation_ = true;

      cellType_ = scinew CellType();
      rmcrt_    = scinew Uintah::Ray( Uintah::TypeDescription::double_type );

      Uintah::ProblemSpecP radSpec        = uintahSpec->findBlock("RMCRT");
      Uintah::ProblemSpecP radPropsSpec   = wasatchSpec_->findBlock("RadProps");
      Uintah::ProblemSpecP RMCRTBenchSpec = wasatchSpec_->findBlock("RMCRTBench");

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
                                                             rmcrt_,
                                                             radSpec,
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
        if( flow_treatment() == COMPRESSIBLE ){
          const EquationAdaptors adaptors = parse_particle_transport_equations<SVolField,SVolField,SVolField>( particleEqnSpec, wasatchSpec_, useAdaptiveDt, graphCategories_);
          adaptors_.insert( adaptors_.end(), adaptors.begin(), adaptors.end() );
        }
        else {
          const EquationAdaptors adaptors = parse_particle_transport_equations<XVolField,YVolField,ZVolField>( particleEqnSpec, wasatchSpec_, useAdaptiveDt, graphCategories_);
          adaptors_.insert( adaptors_.end(), adaptors.begin(), adaptors.end() );
        }
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up particle transport equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

#   ifdef HAVE_POKITT
    Uintah::ProblemSpecP coalSpec = wasatchSpec_->findBlock("Coal");
    // setup tar and soot transport equations kinetic models if specified in the input file
    if( wasatchSpec_->findBlock("TarAndSootEquations") || coalSpec ){
      EquationAdaptors tarEqns = parse_tar_and_soot_equations( wasatchSpec_,
                                                               turbParams,
                                                               densityTag,
                                                               graphCategories_ );
      adaptors_.insert( adaptors_.end(), tarEqns.begin(), tarEqns.end() );
    }

    if( coalSpec ){
        if( !(particleEqnSpec->findBlock("ParticleTemperature")) ){
          std::ostringstream msg;
          msg << endl
          << "Implementation of coal models requires a 'ParticleTransportEquations' block\n"
          << "with a ParticleTemperature sub-block."
          << endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }

        // setup coal models
        proc0cout << "Setting up coal models" << std::endl;
        SetupCoalModels* scm = scinew SetupCoalModels( particleEqnSpec,
                                                       wasatchSpec_,
                                                       coalSpec,
                                                       graphCategories_ ,
                                                       persistentFields_ );
        EquationAdaptors coalEqns = scm->get_adaptors();
        adaptors_.insert( adaptors_.end(), coalEqns.begin(), coalEqns.end() );
    }
#   endif // HAVE_POKITT

    //
    // process any reduction variables specified through the input file
    //
    ReductionHelper::self().parse_reduction_spec( wasatchSpec_ );
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleInitialize( const Uintah::LevelP& level,
                                    Uintah::SchedulerP& sched )
  {
    // Accessing the m_materialManager->allMaterials( "Wasatch" ) must
    // be done after problemSetup. The material manager class will
    // create this material set in postgridsetup, which is called
    // after problemsetup. This is dictated by Uintah.
    if( buildWasatchMaterial_ ){
      set_wasatch_materials(m_materialManager->allMaterials( "Wasatch" ));
      if( doParticles_ ){
        particlesHelper_->set_materials(get_wasatch_materials());
      }
    }
    else{
      if( doParticles_ ){
        particlesHelper_->set_materials(m_materialManager->allMaterials());
      }
    }

    setup_patchinfo_map( level, sched );

    const Uintah::PatchSet* const allPatches   = get_patchset( USE_FOR_TASKS, level, sched );
//  const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );
    
    GraphHelper* const icGraphHelper = graphCategories_[ INITIALIZATION ];

    Expr::ExpressionFactory& exprFactory = *icGraphHelper->exprFactory;

    if( doParticles_ ){
      particlesHelper_->schedule_initialize(level,sched);
    }
    
    bcHelperMap_[level->getID()] = scinew WasatchBCHelper(level, sched, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_, wasatchSpec_);
    
    // handle intrusion boundaries
    if( wasatchSpec_->findBlock("EmbeddedGeometry") ){
      apply_intrusion_boundary_conditions( *bcHelperMap_[level->getID()] );
    }

    //_______________________________________
    // set the time at the initial condition
    Expr::TagList timeTags;
    timeTags.push_back( TagNames::self().time     );
    timeTags.push_back( TagNames::self().timestep );
    scheduleSetInitialTime(level, sched);
    typedef Expr::PlaceHolder<SpatialOps::SingleValueField>  PlcHolder;
    exprFactory.register_expression( scinew PlcHolder::Builder(timeTags), true );
    
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

      process_field_clipping( wasatchSpec_, graphCategories_, allPatches );
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
                                                          persistentFields_ );
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
    if( doRadiation_ ) cellType_->schedule_compute_celltype( rmcrt_, allPatches, materials_, sched );

    if( doParticles_ ) particlesHelper_->schedule_sync_particle_position( level, sched, true );
    
    if( needPressureSolve_ ) m_solver->scheduleInitialize(level, sched, materials_);
    
    proc0cout << "Wasatch: done creating initialization task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleRestartInitialize( const Uintah::LevelP& level,
                                           Uintah::SchedulerP& sched )
  {
    if( needPressureSolve_ ) m_solver->scheduleRestartInitialize(level, sched, materials_);
  }

  //--------------------------------------------------------------------
  
  void Wasatch::restartInitialize()
  {
    isRestarting_ = true;

    // Accessing the m_materialManager->allMaterials( "Wasatch" ) must
    // be done after problemSetup. The material manager will create
    // this material set in postgridsetup, which is called after
    // problemsetup. This is dictated by Uintah.
    if( buildWasatchMaterial_ ){
      set_wasatch_materials( m_materialManager->allMaterials( "Wasatch" ) );
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

        //tsaad: register a patch container as an operator for easy access to the Uintah patch
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

  void Wasatch::scheduleComputeStableTimeStep( const Uintah::LevelP& level,
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
                                                        1, persistentFields_ );
      task->schedule(1);
      taskInterfaceList_.push_back( task );
    }
    else{ // default

      // int timeStep = m_materialManager->getCurrentTopLevelTimeStep();

      // This method is called at both initialization and
      // otherwise. At initialization the old DW, i.e. DW(0) will not
      // exist. As such, get the time step from the new DW,
      // i.e. DW(1).  Otherwise for a normal time step get the time
      // step from the new DW.
      Uintah::timeStep_vartype timeStep(0);
      if( sched->get_dw(0) && sched->get_dw(0)->exists( getTimeStepLabel() ) )
        sched->get_dw(0)->get( timeStep, getTimeStepLabel() );
      else if( sched->get_dw(1) && sched->get_dw(1)->exists( getTimeStepLabel() ) )
        sched->get_dw(1)->get( timeStep, getTimeStepLabel() );

      proc0cout << "Scheduling Task 'compute timestep' COMPUTES 'delT' in NEW data warehouse" << endl;

      Uintah::Task* task = scinew Uintah::Task( "compute timestep", this, &Wasatch::computeDelT );

      // This method is called at both initialization and
      // otherwise. At initialization the old DW, i.e. DW(0) will not
      // exist so require the value from the new DW.  Otherwise for a
      // normal time step require the time step from the old DW.
      if(sched->get_dw(0) ) {
        task->requires( Uintah::Task::OldDW, getTimeStepLabel() );
        task->requires( Uintah::Task::OldDW, getSimTimeLabel() );
      }
      else if(sched->get_dw(1) ) {
        task->requires( Uintah::Task::NewDW, getTimeStepLabel() );
        task->requires( Uintah::Task::NewDW, getSimTimeLabel() );
      }
      
      // jcs it appears that for reduction variables we cannot specify the patches - only the materials.
      task->computes( getDelTLabel(),
                      level.get_rep() );
      //              materials_->getUnion() );
      // jcs why can't we specify a material here?  It doesn't seem to be working if I do.
      
      const GraphHelper* slnGraphHelper = graphCategories_[ADVANCE_SOLUTION];
      const TagNames& tagNames = TagNames::self();
      const bool useStableDT = slnGraphHelper->exprFactory->have_entry( tagNames.stableTimestep );
      // since the StableDT expression is only registered on the time_advance graph,
      // make the necessary checks before adding a requires for that

      if( timeStep > 0 ){
        if( useStableDT ){
          task->requires(Uintah::Task::NewDW, Uintah::VarLabel::find(tagNames.stableTimestep.name()),  Uintah::Ghost::None, 0);
        }
      }
                  
      sched->addTask( task, localPatches, materials_ );
    }

    proc0cout << "Wasatch: done creating timestep task(s)" << std::endl;
  }

  //--------------------------------------------------------------------
  
  void Wasatch::scheduleComputeDualTimeResidual(const Uintah::LevelP& level,
                                                Uintah::SchedulerP& subsched)
  {
    
    Uintah::Task* t = scinew Uintah::Task("Wasatch::computeDualTimeResidual", this, &Wasatch::computeDualTimeResidual, level, subsched.get_rep());
    
    t->requires( Uintah::Task::NewDW, Uintah::VarLabel::find("convergence"), Uintah::Ghost::None, 0);
    t->computes(Uintah::VarLabel::find("DualtimeResidual"));
    subsched->addTask(t, level->eachPatch(), materials_);
  }

  //--------------------------------------------------------------------

  void Wasatch::computeDualTimeResidual(const Uintah::ProcessorGroup* pg,
                               const Uintah::PatchSubset* patches,
                               const Uintah::MaterialSubset* matls,
                               Uintah::DataWarehouse* subOldDW,
                               Uintah::DataWarehouse* subNewDW,
                               Uintah::LevelP level, Uintah::Scheduler* sched)
  {
    const Uintah::VarLabel* const convMeasureLabel = Uintah::VarLabel::find("DualtimeResidual"); // this is the value reduced by Uintah
    const Uintah::VarLabel* const convLabel = Uintah::VarLabel::find("convergence"); // this is the value computed by the dual time integrator in ExprLib

    //__________________
    // set the error norm in the dw for each patch
    for( int ip=0; ip<patches->size(); ++ip ){
      // grab the convergence measure computed by the dual time integrator
      Uintah::PerPatch<double> val_;
      subNewDW->get(val_, convLabel, 0, patches->get(ip));
      const double val = val_.get();
      subNewDW->put(Uintah::max_vartype(val), convMeasureLabel);
    }
  }
  //--------------------------------------------------------------------
  void
  Wasatch::scheduleTimeAdvance( const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched )
  {    
    const Uintah::PatchSet* const allPatches = get_patchset( USE_FOR_TASKS, level, sched );
    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );
    const GraphHelper* advSolGraphHelper = graphCategories_[ ADVANCE_SOLUTION ];

    const Uintah::PatchSet * perproc_patches = m_loadBalancer->getPerProcessorPatchSet( level );

    if ( timeIntegrator_.has_dual_time() ) {
      subsched_ = sched->createSubScheduler();
      subsched_->initialize(3,1); // initialize(number of OldDWs, number of NewDWs)

      
      // if we are in dual time, then the TimeAdvance will drive the dual time integrator
      Uintah::Task* dualTimeTask = scinew Uintah::Task("Wasatch::dualTimeAdvance",
                                                       this, &Wasatch::dualTimeAdvance,
                                                       level, sched.get_rep(),
                                                       advSolGraphHelper->exprFactory);
      dualTimeTask->hasSubScheduler();

      // we need the "outer" timestep
      dualTimeTask->requires( Uintah::Task::OldDW, getTimeStepLabel() );
      dualTimeTask->requires( Uintah::Task::OldDW, getDelTLabel() );
      
      Expr::TagList timeTags;
      timeTags.push_back( TagNames::self().time     );
      timeTags.push_back( TagNames::self().dt     );
      timeTags.push_back( TagNames::self().timestep );
      timeTags.push_back( TagNames::self().rkstage  );
      typedef Expr::PlaceHolder<SpatialOps::SingleValueField>  PlcHolder;
      advSolGraphHelper->exprFactory->register_expression(scinew PlcHolder::Builder(timeTags));
      
      // figure out how to deal with the convergence criterion (reduction)
      Uintah::VarLabel* convLabel = Uintah::VarLabel::create( "DualtimeResidual", Uintah::max_vartype::getTypeDescription() );
      dualTimeTask->requires( Uintah::Task::NewDW, convLabel );

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
          if( isRestarting_ ) transEq->setup_boundary_conditions(*bcHelperMap_[level->getID()], graphCategories_);
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
      
      // create and schedule the Wasatch RHS tasks as well as the dualtime integrators
      subsched_->clearMappings();
      
      subsched_->mapDataWarehouse(Uintah::Task::ParentOldDW, 0);
      subsched_->mapDataWarehouse(Uintah::Task::ParentNewDW, 1);
      subsched_->mapDataWarehouse(Uintah::Task::OldDW, 2);
      subsched_->mapDataWarehouse(Uintah::Task::NewDW, 3);

      // updates the current time. This should happen on the outer timestep since time will be
      // frozen in the dual time iteration
      scheduleUpdateCurrentTime(level, subsched_, 1);
      
      // create all the Wasatch RHS tasks for the transport equations
      create_dual_timestepper_on_patches( level->eachPatch(), materials_, level, subsched_);
      // compute the residual using a reduction
      scheduleComputeDualTimeResidual(level, subsched_);
      
      Uintah::GridP grid = level->getGrid();
      subsched_->advanceDataWarehouse(grid); // needed for compiling the subscheduler tasks
    
      //----------------------------------------------------------------------------------------------
      /* tsaad: This section below stinks. we should figure out a way to find the dependencies of the
       dualTimeTask without invoking the subscheduler - we should try to get these from the ExprLib graph
       directly.
       */
      subsched_->compile(); // compile subscheduler
      
      // Now get all the dependencies for the dualtimetask from the subscheduler and add them to the dualTimeTask.
      // These are the dependencies that the dual time task needs from the parent scheduler
      const std::set<const Uintah::VarLabel*, Uintah::VarLabel::Compare>& initialRequires = subsched_->getInitialRequiredVars();
      for (std::set<const Uintah::VarLabel*>::const_iterator it=initialRequires.begin(); it!=initialRequires.end(); ++it)
      {
        dualTimeTask->requires(Uintah::Task::OldDW, *it, Uintah::Ghost::AroundCells, 1);
      }

      const std::set<const Uintah::VarLabel*, Uintah::VarLabel::Compare>& computedVars = subsched_->getComputedVars();
      for (std::set<const Uintah::VarLabel*>::const_iterator it=computedVars.begin(); it!=computedVars.end(); ++it)
      {
        std::string varname = (*it)->getName();
        if (varname == "dt" || varname == "rkstage" || varname == "timestep" || varname == "time" )  continue;
        dualTimeTask->computes(*it);
      }
      //----------------------------------------------------------------------------------------------
      
      // add the dualtimetask to the parent scheduler
      sched->addTask(dualTimeTask, perproc_patches, m_materialManager->allMaterials());
      
      subsched_->compile(); // here we need to recompile the subscheduler for reasons mysterious to me - but this seems to make dualtime work with mpi!
    } else {
      if( isRestarting_ ){
        setup_patchinfo_map( level, sched );
        
        
        if( doParticles_ ){
          particlesHelper_->schedule_restart_initialize(level,sched);
          particlesHelper_->schedule_find_boundary_particles(level,sched);
        }
        
        bcHelperMap_[level->getID()] = scinew WasatchBCHelper(level, sched, materials_, patchInfoMap_, graphCategories_,  bcFunctorMap_, wasatchSpec_);
      }
      
      if( doParticles_ ){
        particlesHelper_->schedule_find_boundary_particles(level,sched);
      }

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
             if( isRestarting_ ) transEq->setup_boundary_conditions(*bcHelperMap_[level->getID()], graphCategories_);
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

        //
        // process clipping on fields - must be done AFTER all bcs are applied
        //
        process_field_clipping( wasatchSpec_, graphCategories_, localPatches );
        
        if( buildTimeIntegrator_ ){
          scheduleUpdateCurrentTime(level, sched, iStage);
          create_timestepper_on_patches( allPatches, materials_, level, sched, iStage );
        }
        
        proc0cout << "Wasatch: done creating solution task(s)" << std::endl;
        
        // pass the bc Helper to pressure expressions on all patches
        if (flow_treatment() != COMPRESSIBLE) {
          if (wasatchSpec_->findBlock("MomentumEquations")) {
            const bool needPressureSolve = !(wasatchSpec_->findBlock("MomentumEquations")->findBlock("DisablePressureSolve"));
            if (needPressureSolve) {
              bcHelperMap_[level->getID()]->synchronize_pressure_expression();
            }
          }
        }
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
                                                         persistentFields_ );
        task->schedule(1);
        taskInterfaceList_.push_back( task );
      }
      proc0cout << "Wasatch: done creating post-processing task(s)" << std::endl;
      
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
      
      if( isRestarting_ ) isRestarting_ = false;
      
    }
  }

  //--------------------------------------------------------------------

  void Wasatch::dualTimeAdvance(const Uintah::ProcessorGroup* pg,
                                const Uintah::PatchSubset* patches,
                                const Uintah::MaterialSubset* matls,
                                Uintah::DataWarehouse* parentOldDW,
                                Uintah::DataWarehouse* parentNewDW,
                                Uintah::LevelP level, Uintah::Scheduler* sched,
                                Expr::ExpressionFactory* const factory)
  {
    using namespace Uintah;
    GridP grid = level->getGrid();
    
    //__________________________________
    //  turn off parentDW scrubbing
    subsched_->setParentDWs(parentOldDW, parentNewDW);
    
    parentOldDW->setScrubbing(DataWarehouse::ScrubNone);
    parentNewDW->setScrubbing(DataWarehouse::ScrubNone);
    
    subsched_->clearMappings();
    subsched_->mapDataWarehouse(Uintah::Task::ParentOldDW, 0);
    subsched_->mapDataWarehouse(Uintah::Task::ParentNewDW, 1);
    subsched_->mapDataWarehouse(Uintah::Task::OldDW, 2);
    subsched_->mapDataWarehouse(Uintah::Task::NewDW, 3);

    DataWarehouse* subOldDW = subsched_->get_dw(2);
    DataWarehouse* subNewDW = subsched_->get_dw(3);

    //__________________________________
    //  Move data from parentOldDW to subSchedNewDW.
    const std::set<const Uintah::VarLabel*, Uintah::VarLabel::Compare>& initialRequires = subsched_->getInitialRequiredVars();
    for( std::set<const Uintah::VarLabel*>::const_iterator it=initialRequires.begin(); it!=initialRequires.end(); ++it ){

      Uintah::TypeDescription::Type vartype = (*it)->typeDescription()->getType();

      // avoid reduction and sole vars
      if (vartype != TypeDescription::ReductionVariable &&
          vartype != TypeDescription::SoleVariable){
        subNewDW->transferFrom(parentOldDW, *it, patches, matls, true);
      }
    }
    
    subsched_->advanceDataWarehouse(grid);

    //__________________
    // Iterate
    int c = 1;
    Uintah::max_vartype residual = 999999.9;

    const Uintah::VarLabel* const ptcResLabel = Uintah::VarLabel::find("DualtimeResidual"); // this is the value reduced by Uintah
        
    do {
      subOldDW = subsched_->get_dw(2);
      subNewDW = subsched_->get_dw(3);
      
      for( std::set<const Uintah::VarLabel*>::const_iterator it=initialRequires.begin(); it!=initialRequires.end(); ++it ){

        Uintah::TypeDescription::Type vartype = (*it)->typeDescription()->getType();

        // avoid reduction and sole vars
        if (vartype != TypeDescription::ReductionVariable &&
            vartype != TypeDescription::SoleVariable){
          subNewDW->transferFrom(subOldDW, *it, patches, matls, true);
        }
      }
      
      subOldDW->setScrubbing(DataWarehouse::ScrubNone);
      subNewDW->setScrubbing(DataWarehouse::ScrubNone);
      subsched_->execute();

      // now grab the maximum value of the error norm across all patches
      subNewDW->get(residual, ptcResLabel);
      
      // advance the dws
      subsched_->advanceDataWarehouse(grid);
      
      c++;
      if( !(c % dualTimeMatrixInfo_->logIterationRate) ) proc0cout << "    dual time: iteration " << c << "    residual " << residual << std::endl;
    } while(c <= dualTimeMatrixInfo_->maxIterations && residual >= dualTimeMatrixInfo_->tolerance);
    
    totalDualTimeIterations_ += c - 1;

    // int timeStep = m_materialManager->getCurrentTopLevelTimeStep();
    
    Uintah::timeStep_vartype timeStep;
    parentOldDW->get( timeStep, getTimeStepLabel() );

    proc0cout << " Dual time iterations = " << c-1 << ". Residual = " << residual << ". Average iterations = " << (double) totalDualTimeIterations_/timeStep << std::endl;
    
    // move dependencies to the parent DW
    const std::set<const Uintah::VarLabel*, Uintah::VarLabel::Compare>& computedVars = subsched_->getComputedVars();
    for (std::set<const Uintah::VarLabel*>::const_iterator it=computedVars.begin(); it!=computedVars.end(); ++it ){

      Uintah::TypeDescription::Type vartype = (*it)->typeDescription()->getType();

      // avoid reduction and sole vars
      if (vartype != TypeDescription::ReductionVariable &&
          vartype != TypeDescription::SoleVariable){
        std::string varname = (*it)->getName();
        if (varname == "dt" || varname == "time" || varname == "rkstage" || varname == "timestep" ) continue;
        parentNewDW->transferFrom(subNewDW, *it, patches, matls, true);
      }
    }
    
  }

  //---------------------------------------------------------------------------------
  
  void
  Wasatch::scheduleSetInitialTime( const Uintah::LevelP& level,
                                   Uintah::SchedulerP& sched )
  {
    //________________________________________________________
    // add a task to populate a "field" with the current time.
    // This is required by the time integrator.
    //    const Uintah::PatchSet* localPatches = m_loadBalancer->getPerProcessorPatchSet(level);
    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_TASKS, level, sched );
    {
      // add a task to update current simulation time
      Uintah::Task* updateCurrentTimeTask =
      scinew Uintah::Task( "set initial time",
                           this,
                           &Wasatch::set_initial_time );

      updateCurrentTimeTask->requires( Uintah::Task::NewDW, getTimeStepLabel() );
      updateCurrentTimeTask->requires( Uintah::Task::NewDW, getSimTimeLabel() );
      
      const Uintah::TypeDescription* perPatchTD = Uintah::PerPatch<double>::getTypeDescription();
      tLabel_     = (!tLabel_      ) ? Uintah::VarLabel::create( TagNames::self().time.name(), perPatchTD )     : tLabel_    ;
      tStepLabel_ = (!tStepLabel_  ) ? Uintah::VarLabel::create( TagNames::self().timestep.name(), perPatchTD ) : tStepLabel_;
      
      updateCurrentTimeTask->computes( tLabel_     );
      updateCurrentTimeTask->computes( tStepLabel_ );
        
      sched->addTask( updateCurrentTimeTask, localPatches, materials_ );
    }
  }
  
  //------------------------------------------------------------------
  
  void
  Wasatch::set_initial_time( const Uintah::ProcessorGroup* const pg,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             Uintah::DataWarehouse* const oldDW,
                             Uintah::DataWarehouse* const newDW )
  {
    // grab the timestep
    // const double simTime = m_materialManager->getElapsedSimTime();
    // const double timeStep = m_materialManager->getCurrentTopLevelTimeStep();

    Uintah::timeStep_vartype timeStep(0);
    newDW->get( timeStep, getTimeStepLabel() );

    Uintah::simTime_vartype simTime(0);
    newDW->get( simTime, getSimTimeLabel() );

    typedef Uintah::PerPatch<double> perPatchT;
    perPatchT tstep( timeStep );
    perPatchT time ( simTime  );
    
    for( int p=0; p < patches->size(); p++ ){
      const Uintah::Patch* patch = patches->get(p);
      newDW->put( tstep,   tStepLabel_, 0, patch );
      newDW->put( time,    tLabel_, 0, patch );
    }
  }
  
  //---------------------------------------------------------------------------------
  
  void
  Wasatch::scheduleUpdateCurrentTime( const Uintah::LevelP& level,
                                      Uintah::SchedulerP& sched,
                                      const int rkStage)
  {
    //_____________________________________________________________
    // create an expression to set the current time as a field that
    // will be available to all expressions if needed.
    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];
    Expr::ExpressionFactory& exprFactory = *gh->exprFactory;

    Expr::ExpressionID timeID;
    if( rkStage==1 && !exprFactory.have_entry(TagNames::self().time) ){
      Expr::TagList timeTags;
      timeTags.push_back( TagNames::self().time     );
      timeTags.push_back( TagNames::self().dt     );
      timeTags.push_back( TagNames::self().timestep );
      timeTags.push_back( TagNames::self().rkstage  );
      typedef Expr::PlaceHolder<SpatialOps::SingleValueField>  PlcHolder;
      timeID = exprFactory.register_expression(scinew PlcHolder::Builder(timeTags));
    }
    else{
      timeID = exprFactory.get_id(TagNames::self().time);
    }

    //________________________________________________________
    // add a task to populate a "field" with the current time.
    // This is required by the time integrator.
//    const Uintah::PatchSet* localPatches =  m_loadBalancer->getPerProcessorPatchSet(level);
    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_TASKS, level, sched );
    {
      // add a task to update current simulation time
      Uintah::Task* updateCurrentTimeTask =
      scinew Uintah::Task( "update current time",
                           this,
                           &Wasatch::update_current_time,
                          rkStage );
      updateCurrentTimeTask->requires( (has_dual_time() ? Uintah::Task::ParentOldDW : Uintah::Task::OldDW), getTimeStepLabel() );
      updateCurrentTimeTask->requires( (has_dual_time() ? Uintah::Task::ParentOldDW : Uintah::Task::OldDW), getSimTimeLabel() );
      updateCurrentTimeTask->requires( (has_dual_time() ? Uintah::Task::ParentOldDW : Uintah::Task::OldDW), getDelTLabel() );
      
      const Uintah::TypeDescription* perPatchTD = Uintah::PerPatch<double>::getTypeDescription();
      dtLabel_      = (!dtLabel_     ) ? Uintah::VarLabel::create( TagNames::self().dt.name(), perPatchTD )       : dtLabel_     ;
      tLabel_       = (!tLabel_      ) ? Uintah::VarLabel::create( TagNames::self().time.name(), perPatchTD )     : tLabel_      ;
      tStepLabel_   = (!tStepLabel_  ) ? Uintah::VarLabel::create( TagNames::self().timestep.name(), perPatchTD ) : tStepLabel_  ;
      rkStageLabel_ = (!rkStageLabel_) ? Uintah::VarLabel::create( TagNames::self().rkstage.name(), perPatchTD )  : rkStageLabel_;
      if( rkStage < 2 ){
        updateCurrentTimeTask->computes( dtLabel_      );
        updateCurrentTimeTask->computes( tLabel_       );
        updateCurrentTimeTask->computes( tStepLabel_   );
        updateCurrentTimeTask->computes( rkStageLabel_ );
      }
      else {
        updateCurrentTimeTask->modifies( dtLabel_      );
        updateCurrentTimeTask->modifies( tLabel_       );
        updateCurrentTimeTask->modifies( tStepLabel_   );
        updateCurrentTimeTask->modifies( rkStageLabel_ );
      }
      
      sched->addTask( updateCurrentTimeTask, localPatches, materials_ );
    }
  }

  //------------------------------------------------------------------
  
  void
  Wasatch::update_current_time( const Uintah::ProcessorGroup* const pg,
                                const Uintah::PatchSubset* const patches,
                                const Uintah::MaterialSubset* const materials,
                                Uintah::DataWarehouse* const oldDW,
                                Uintah::DataWarehouse* const newDW,
                                const int rkStage )
  {
    Uintah::DataWarehouse* whichDW = has_dual_time() ? oldDW->getOtherDataWarehouse(Uintah::Task::ParentOldDW) : oldDW;

    // grab the timestep
    // const double simTime = m_materialManager->getElapsedSimTime();
    // const double timeStep = m_materialManager->getCurrentTopLevelTimeStep();
    
    Uintah::timeStep_vartype timeStep;
    whichDW->get( timeStep, getTimeStepLabel() );

    Uintah::simTime_vartype simTime;
    whichDW->get( simTime, getSimTimeLabel() );

    Uintah::delt_vartype deltat;
    whichDW->get( deltat, getDelTLabel() );

    const Expr::Tag timeTag = TagNames::self().time;
    double rks = (double) rkStage;
    double* timeCor = timeIntegrator_.timeCorrection;
    
    typedef Uintah::PerPatch<double> perPatchT;
    perPatchT dt     (deltat );
    perPatchT tstep  (timeStep );
    perPatchT time   (simTime + timeCor[rkStage -1 ] * deltat );
    perPatchT rkstage( rks );

    for( int p=0; p < patches->size(); p++ ){
      const Uintah::Patch* patch = patches->get(p);
      newDW->put( dt,      dtLabel_,      0, patch );
      newDW->put( tstep,   tStepLabel_,   0, patch );
      newDW->put( time,    tLabel_,       0, patch );
      newDW->put( rkstage, rkStageLabel_, 0, patch );
    }
  }
  
  //---------------------------------------------------------------------------------
  
  void
  Wasatch::create_dual_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                               const Uintah::MaterialSet* const materials,
                                               const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];

    if( adaptors_.size() == 0 && gh->rootIDs.empty() ) return; // no equations registered.
    
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

    //____________________________________________________________________
    // create all of the required tasks on the timestepper.  This involves
    // the task(s) that compute(s) the RHS for each transport equation and
    // the task that updates the variables from time "n" to "n+1"
    timeStepper_->create_dualtime_tasks( patchInfoMap_, localPatches,
                                         materials, level, sched,
                                         dualTimePatchMap_, persistentFields_,
                                         *dualTimeMatrixInfo_ );
  }

  
  //---------------------------------------------------------------------------------
  
  void
  Wasatch::create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                          const Uintah::MaterialSet* const materials,
                                          const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched,
                                          const int rkStage )
  {
    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];

    if( adaptors_.size() == 0 && gh->rootIDs.empty()) return; // no equations registered.

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
    timeStepper_->create_tasks( patchInfoMap_, localPatches,
                                materials, level, sched,
                                rkStage, persistentFields_ );
  }

  //--------------------------------------------------------------------

  void
  Wasatch::computeDelT( const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches,
                        const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* oldDW,
                        Uintah::DataWarehouse* newDW )
  {
    // int timeStep = m_materialManager->getCurrentTopLevelTimeStep();
    
    // This method is called at both initialization and otherwise. At
    // initialization the old DW will not exist so get the value from
    // the new DW.  Otherwise for a normal time step get the time step
    // from the old DW.
    Uintah::timeStep_vartype timeStep(0);
    if( oldDW && oldDW->exists( getTimeStepLabel() ) )
      oldDW->get( timeStep, getTimeStepLabel() );
    else if( newDW && newDW->exists( getTimeStepLabel() ) )
      newDW->get( timeStep, getTimeStepLabel() );

    Uintah::delt_vartype deltat = 1.0;
    double val = 9999999999999.0;
    
    const GraphHelper* slnGraphHelper = graphCategories_[ADVANCE_SOLUTION];
    const TagNames& tagNames = TagNames::self();
    const bool useStableDT = slnGraphHelper->exprFactory->have_entry( tagNames.stableTimestep );
    if( timeStep > 0 ){
      if( useStableDT ){
        //__________________
        // loop over patches
        for( int ip=0; ip<patches->size(); ++ip ){
          // grab the stable timestep value calculated by the StableDT expression
          Uintah::PerPatch<double> tempDtP;
          newDW->get(tempDtP, Uintah::VarLabel::find(tagNames.stableTimestep.name()), 0, patches->get(ip));          
          val = std::min( val, tempDtP.get() );
        }
      }
      else {
        // FOR FIXED dt: (min = max in input file)
        // if this is not the first timestep, then grab dt from the olddw.
        // This will avoid Uintah's message that it is setting dt to max dt/min dt
        oldDW->get( deltat, getDelTLabel() );
      }
    }
    
    if( useStableDT ){
      newDW->put(Uintah::delt_vartype(val),getDelTLabel(),
                 Uintah::getLevel(patches) );
    }
    else{
      newDW->put( deltat,
                  getDelTLabel(),
                  Uintah::getLevel(patches) );
    }

    // The component specifies task graph index for next
    // timestep.
    if (doRadiation_) {
      // Setup the correct task graph for execution for the NEXT time
      // step.  Also do radiation solve on time step 1.
      int task_graph_index =
        ((((timeStep+1) % radCalcFrequency_ == 0) || ((timeStep+1) == 1)) ?
         Uintah::RMCRTCommon::TG_RMCRT :
         Uintah::RMCRTCommon::TG_CARRY_FORWARD);
      
      setTaskGraphIndex( task_graph_index );
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
        // return  m_loadBalancer->getPerProcessorPatchSet(level);
        return level->eachPatch();
        break;

      case USE_FOR_OPERATORS: {
        const int levelID = level->getID();
        const Uintah::PatchSet* const allPatches = m_loadBalancer->getPerProcessorPatchSet(level);
        const Uintah::PatchSubset* const localPatches = allPatches->getSubset( d_myworld->myRank() );

        std::map< int, const Uintah::PatchSet* >::iterator ip = patchesForOperators_.find( levelID );

        if( ip != patchesForOperators_.end() ) return ip->second;

        Uintah::PatchSet* patches = new Uintah::PatchSet;
        // jcs: this results in "normal" scheduling and WILL NOT WORK FOR LINEAR SOLVES
        //      in that case, we need to use "gang" scheduling: addAll( localPatches )
        patches->addEach( localPatches->getVector() );
        //     const std::set<int>& procs =  m_loadBalancer->getNeighborhoodProcessors();
        //     for( std::set<int>::const_iterator ip=procs.begin(); ip!=procs.end(); ++ip ){
        //       patches->addEach( allPatches->getSubset( *ip )->getVector() );
        //     }
        patchesForOperators_[levelID] = patches;
        return patches;
      }
    }
    return nullptr;
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

} // namespace WasatchCore
