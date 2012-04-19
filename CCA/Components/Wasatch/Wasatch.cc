/*

  The MIT License

  Copyright (c) 2010-2012 Institute for Clean & Secure Energy (ICSE), University of Utah.

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

#include <fstream>

//-- Uintah framework includes --//
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
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FVStaggeredBCTools.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- TabProps --//
#include <tabprops/TabPropsConfig.h>

//-- Wasatch includes --//
#include "Wasatch.h"
#include "WasatchMaterial.h"
#include "CoordHelper.h"
#include "FieldAdaptor.h"
#include "StringNames.h"
#include "TaskInterface.h"
#include "TimeStepper.h"
#include "Properties.h"
#include "Operators/Operators.h"
#include "Expressions/BasicExprBuilder.h"
#include "Expressions/SetCurrentTime.h"
#include "transport/ParseEquation.h"
#include "transport/TransportEquation.h"
#include "BCHelperTools.h"
#include "ParseTools.h"

using std::endl;

namespace Wasatch{

  //--------------------------------------------------------------------

  Wasatch::Wasatch( const Uintah::ProcessorGroup* myworld )
    : Uintah::UintahParallelComponent( myworld ),
      buildTimeIntegrator_ ( true ),
      buildWasatchMaterial_( true ),
      nRKStages_(1)
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

    isRestarting_ = false;

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::d_combineMemory = false;

    const bool log = false;
    graphCategories_[ INITIALIZATION     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ TIMESTEP_SELECTION ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ ADVANCE_SOLUTION   ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );

    icCoordHelper_  = new CoordHelper( *(graphCategories_[INITIALIZATION]->exprFactory) );
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

    for( std::list<const Uintah::PatchSet*>::iterator i=patchSetList_.begin(); i!=patchSetList_.end(); ++i ){
      delete *i;
    }

    delete icCoordHelper_;
    if (buildTimeIntegrator_) delete timeStepper_;

    for( GraphCategories::iterator igc=graphCategories_.begin(); igc!=graphCategories_.end(); ++igc ){
      delete igc->second->exprFactory;
      delete igc->second;
    }
  }

  //--------------------------------------------------------------------

  void force_expressions_on_graph( Uintah::ProblemSpecP forceOnGraphParams,
                                   GraphCategories& gc,
                                   const std::string taskListName ) {
    for( Uintah::ProblemSpecP exprParams = forceOnGraphParams->findBlock("AnyExpression");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("AnyExpression") ){

      const Expr::Tag anyExpressionTag = parse_nametag( exprParams->findBlock("NameTag") );

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << endl
        << __FILE__ << " : " << __LINE__ << endl;
      }

      GraphHelper* const graphHelper = gc[cat];

      const Expr::ExpressionID anyExpressionID = graphHelper->exprFactory->get_id(anyExpressionTag);
      graphHelper->rootIDs.insert( anyExpressionID );
    }
  }

  //--------------------------------------------------------------------

  void Wasatch::problemSetup( const Uintah::ProblemSpecP& params,
                              const Uintah::ProblemSpecP& ,  /* jcs not sure what this param is for */
                              Uintah::GridP& grid,
                              Uintah::SimulationStateP& sharedState )
  {
    sharedState_ = sharedState;

    // disallow specification of extraCells
    {
      std::ostringstream msg;
      bool foundExtraCells = false;
      Uintah::ProblemSpecP grid = params->findBlock("Grid");
      for( Uintah::ProblemSpecP level = grid->findBlock("Level");
           level != 0;
           level = grid->findNextBlock("Level") ){
        for( Uintah::ProblemSpecP box = level->findBlock("Box");
             box != 0;
             box = level->findNextBlock("Box") ){
          // note that a [0,0,0] specification gets added by default,
          // so we will check to ensure that something other than
          // [0,0,0] has not been specified.
          if( box->findBlock("extraCells") ){
            Uintah::IntVector extraCells;
            box->get("extraCells",extraCells);
            if( extraCells != Uintah::IntVector(0,0,0) ){
              foundExtraCells = true;
              std::string boxLabel;
              box->get("label",boxLabel);
              msg << "box '" << boxLabel << "' has extraCells specified." << endl;
            }
          }
        }
      }
      if( foundExtraCells ){
        msg << endl
            << "  Specification of 'extraCells' is forbidden in Wasatch." << endl
            << "  Please remove it from your input file" << endl
            << endl;
        throw std::runtime_error( msg.str() );
      }
    }

    // ADD BLOCK FOR IO FIELDS
    Uintah::ProblemSpecP archiverParams = params->findBlock("DataArchiver");
    for( Uintah::ProblemSpecP saveLabelParams=archiverParams->findBlock("save");
        saveLabelParams != 0;
        saveLabelParams=saveLabelParams->findNextBlock("save") ){
      std::string saveTheLabel;
      saveLabelParams->getAttribute("label",saveTheLabel);
      ioFieldSet_.insert(saveTheLabel);
    }

    Uintah::ProblemSpecP wasatchParams = params->findBlock("Wasatch");
    if (!wasatchParams) return;

    //
    // Material
    //
    if (buildWasatchMaterial_) {
      Uintah::WasatchMaterial* mat= scinew Uintah::WasatchMaterial();
      sharedState->registerWasatchMaterial(mat);
    }

    // we are able to get the solver port from here
    linSolver_ = dynamic_cast<Uintah::SolverInterface*>(getPort("solver"));
    if(!linSolver_) {
      throw Uintah::InternalError("Wasatch: couldn't get solver port", __FILE__, __LINE__);
    } else if (linSolver_) {
      proc0cout << "Detected solver: " << linSolver_->getName() << std::endl;
      if ( (linSolver_->getName()).compare("hypre") != 0 && wasatchParams->findBlock("MomentumEquations") ) {
        std::ostringstream msg;
        msg << "  Invalid solver specified: "<< linSolver_->getName() << std::endl
        << "  Wasatch currently works with hypre solver only. Please change your solver type." << std::endl
        << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }

    //
    std::string timeIntegrator;
    wasatchParams->get("TimeIntegrator",timeIntegrator);
    if (timeIntegrator=="RK3SSP") nRKStages_ = 3;

    //
    // create expressions explicitly defined in the input file.  These
    // are typically associated with, e.g. initial conditions.
    //
    create_expressions_from_input( wasatchParams, graphCategories_ );
    setup_property_evaluation( wasatchParams, graphCategories_ );

    //
    // extract the density tag for scalar transport equations and momentum equations
    // and perform error handling
    //
    Uintah::ProblemSpecP momEqnParams   = wasatchParams->findBlock("MomentumEquations");
    Uintah::ProblemSpecP densityParams  = wasatchParams->findBlock("Density");
    bool existSrcTerm=false;

    for( Uintah::ProblemSpecP transEqnParams=wasatchParams->findBlock("TransportEquation");
        transEqnParams != 0;
        transEqnParams=transEqnParams->findNextBlock("TransportEquation") ){
      existSrcTerm = (existSrcTerm || transEqnParams->findBlock("SourceTermExpression") );
    }
    Uintah::ProblemSpecP transEqnParams = wasatchParams->findBlock("TransportEquation");

    Expr::Tag densityTag = Expr::Tag();
    bool isConstDensity = true;

    if (transEqnParams || momEqnParams) {
      if( !densityParams ) {
        std::ostringstream msg;
        msg << "ERROR: You must include a 'Density' block in your input file when solving transport equations" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      const bool existDensity = densityParams->findBlock("NameTag");
      densityParams->get("IsConstant",isConstDensity);

      if( !isConstDensity || existSrcTerm || momEqnParams) {
        if( !existDensity ) {
          std::ostringstream msg;
          msg << "ERROR: For variable density cases or when source terms exist in transport equations (scalar, momentum, etc...), the density expression tag" << endl
              << "       must be provided in the <Density> block" << endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        densityTag = parse_nametag( densityParams->findBlock("NameTag") );
      }
    }

    //
    // Build transport equations.  This registers all expressions as
    // appropriate for solution of each transport equation.
    //
    for( Uintah::ProblemSpecP transEqnParams=wasatchParams->findBlock("TransportEquation");
         transEqnParams != 0;
         transEqnParams=transEqnParams->findNextBlock("TransportEquation") ){
      adaptors_.push_back( parse_equation( transEqnParams, densityTag, isConstDensity, graphCategories_ ) );
    }

    //
    // Build coupled transport equations scalability test for wasatch.
    //
    for( Uintah::ProblemSpecP scalEqnParams=wasatchParams->findBlock("ScalabilityTest");
        scalEqnParams != 0;
        scalEqnParams=scalEqnParams->findNextBlock("ScalabilityTest") ) {
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
    for( Uintah::ProblemSpecP momEqnParams=wasatchParams->findBlock("MomentumEquations");
        momEqnParams != 0;
        momEqnParams=momEqnParams->findNextBlock("MomentumEquations") ){
      // note - parse_momentum_equations returns a vector of equation adaptors
      try{
        EquationAdaptors momentumAdaptors = parse_momentum_equations( momEqnParams, densityTag, graphCategories_, *linSolver_);
        adaptors_.insert( adaptors_.end(), momentumAdaptors.begin(), momentumAdaptors.end() );
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
    // Build moment transport equations.  This registers all expressions
    // required for solution of each momentum equation.
    //
    for( Uintah::ProblemSpecP momEqnParams=wasatchParams->findBlock("MomentTransportEquation");
        momEqnParams != 0;
        momEqnParams=momEqnParams->findNextBlock("MomentTransportEquation") ){
      // note - parse_moment_transport_equations returns a vector of equation adaptors
      try{
        EquationAdaptors momentAdaptors = parse_moment_transport_equations( momEqnParams, graphCategories_);
        adaptors_.insert( adaptors_.end(), momentAdaptors.begin(), momentAdaptors.end() );
      }
      catch( std::runtime_error& err ){
        std::ostringstream msg;
        msg << endl
        << "Problems setting up moment transport equations.  Details follow:" << endl
        << err.what() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    if( buildTimeIntegrator_ ){
      timeStepper_ = scinew TimeStepper( sharedState_->get_delt_label(),
                                         *graphCategories_[ ADVANCE_SOLUTION ] );
    }

    //
    // force additional expressions on the graph
    //
    for( Uintah::ProblemSpecP forceOnGraphParams=wasatchParams->findBlock("ForceOnGraph");
        forceOnGraphParams != 0;
        forceOnGraphParams=forceOnGraphParams->findNextBlock("ForceOnGraph") ){
      std::string taskListName;
      forceOnGraphParams->getAttribute("tasklist", taskListName);
      force_expressions_on_graph(forceOnGraphParams, graphCategories_, taskListName);
    }
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleInitialize( const Uintah::LevelP& level,
                                    Uintah::SchedulerP& sched )
  {
    // accessing the sharedState_->allWasatchMaterials() must be done after
    // problemSetup. The sharedstate class will create this material set
    // in postgridsetup, which is called after problemsetup. This is dictated
    // by Uintah.
    if (buildWasatchMaterial_) {
      set_wasatch_materials(sharedState_->allWasatchMaterials());
    }

    setup_patchinfo_map( level, sched );

    const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_TASKS, level, sched );

    GraphHelper* const icGraphHelper = graphCategories_[ INITIALIZATION ];

    Expr::ExpressionFactory& exprFactory = *icGraphHelper->exprFactory;

    //_______________________________________
    // set the time
    Expr::TagList timeTags;
    timeTags.push_back( Expr::Tag( StringNames::self().time, Expr::STATE_NONE ) );
    timeTags.push_back( Expr::Tag( StringNames::self().timestep, Expr::STATE_NONE ) );
    const Expr::Tag timeTag( StringNames::self().time, Expr::STATE_NONE );
    exprFactory.register_expression( scinew SetCurrentTime::Builder( timeTags, sharedState_, 1 ), true );
    //_____________________________________________
    // Build the initial condition expression graph
    if( !icGraphHelper->rootIDs.empty() ){

      TaskInterface* const task = scinew TaskInterface( icGraphHelper->rootIDs,
                                                        "initialization",
                                                        *icGraphHelper->exprFactory,
                                                        level,
                                                        sched,
                                                        localPatches,
                                                        materials_,
                                                        patchInfoMap_,
                                                        true, 1, ioFieldSet_ );

      // set coordinate values as required by the IC graph.
      icCoordHelper_->create_task( sched, localPatches, materials_ );

      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      task->schedule( icCoordHelper_->field_tags(), 1 );
      taskInterfaceList_.push_back( task );

      // -----------------------------------------------------------------------
      // INITIAL BOUNDARY CONDITIONS TREATMENT
      // -----------------------------------------------------------------------
      const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );
      const GraphHelper* icGraphHelper = graphCategories_[ INITIALIZATION ];
      typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;

      for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
        EqnTimestepAdaptorBase* const adaptor = *ia;
        TransportEquation* transEq = adaptor->equation();
        std::string eqnLabel = transEq->solution_variable_name();
        //______________________________________________________
        // set up boundary conditions on this transport equation
        try{
          proc0cout << "Setting Initial BCs for transport equation '" << eqnLabel << "'" << std::endl;
          transEq->setup_initial_boundary_conditions(*icGraphHelper, localPatches, patchInfoMap_, materials_->getUnion());
        }
        catch( std::runtime_error& e ){
          std::ostringstream msg;
          msg << e.what()
          << std::endl
          << "ERORR while setting initial boundary conditions on equation '" << eqnLabel << "'"
          << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      }

    }
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
     {
       const Uintah::PatchSet* patches = get_patchset( USE_FOR_OPERATORS, level, sched );

       for( int ipss=0; ipss<patches->size(); ++ipss ){
         const Uintah::PatchSubset* pss = patches->getSubset(ipss);
         for( int ip=0; ip<pss->size(); ++ip ){
           SpatialOps::OperatorDatabase* const opdb = scinew SpatialOps::OperatorDatabase();
           const Uintah::Patch* const patch = pss->get(ip);
           build_operators( *patch, *opdb );
           PatchInfo& pi = patchInfoMap_[patch->getID()];
           pi.operators = opdb;
           pi.patchID = patch->getID();
           //std::cout << "Set up operators for Patch ID: " << patch->getID() << " on process " << Uintah::Parallel::getMPIRank() << std::endl;
         }
       }
     }
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    std::cout << "Scheduling compute stable timestep\n";
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
                                                        level,
                                                        sched,
                                                        localPatches,
                                                        materials_,
                                                        patchInfoMap_,
                                                        true, 1, ioFieldSet_ );
      task->schedule(1);
      taskInterfaceList_.push_back( task );
    }
    else{ // default

      proc0cout << "Task 'compute timestep' COMPUTES 'delT' in NEW data warehouse" << endl;

      Uintah::Task* task = scinew Uintah::Task( "compute timestep", this, &Wasatch::computeDelT );

      // jcs it appears that for reduction variables we cannot specify the patches - only the materials.
      	task->computes( sharedState_->get_delt_label(),
                      level.get_rep() );
      //              materials_->getUnion() );
      // jcs why can't we specify a metrial here?  It doesn't seem to be working if I do.

      sched->addTask( task, localPatches, materials_ );
    }

    proc0cout << "Wasatch: done creating timestep task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void
  Wasatch::scheduleTimeAdvance( const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched )
  {
    if( isRestarting_ ){
      setup_patchinfo_map( level, sched );
      isRestarting_ = false;
    }

    for (int iStage=1; iStage<=nRKStages_; iStage++) {
      // jcs why do we need this instead of getting the level?
      const Uintah::PatchSet* const allPatches = get_patchset( USE_FOR_TASKS, level, sched );
      const Uintah::PatchSet* const localPatches = get_patchset( USE_FOR_OPERATORS, level, sched );

      if( buildTimeIntegrator_ ){
        create_timestepper_on_patches( allPatches, materials_, level, sched, iStage );
      }

      proc0cout << "Wasatch: done creating solution task(s)" << std::endl;

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


      // -----------------------------------------------------------------------
      // BOUNDARY CONDITIONS TREATMENT
      // -----------------------------------------------------------------------
      const GraphHelper* advSolGraphHelper = graphCategories_[ ADVANCE_SOLUTION ];
      typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;

      for( EquationAdaptors::const_iterator ia=adaptors_.begin(); ia!=adaptors_.end(); ++ia ){
        EqnTimestepAdaptorBase* const adaptor = *ia;
        TransportEquation* transEq = adaptor->equation();
        std::string eqnLabel = transEq->solution_variable_name();
        //______________________________________________________
        // set up boundary conditions on this transport equation
        try{
          proc0cout << "Setting BCs for transport equation '" << eqnLabel << "'" << std::endl;
          transEq->setup_boundary_conditions(*advSolGraphHelper, localPatches, patchInfoMap_, materials_->getUnion());
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
    }

    // ensure that any "CARRY_FORWARD" variable has an initialization provided for it.
    {
      const Expr::ExpressionFactory* const icFactory = graphCategories_[INITIALIZATION]->exprFactory;
      typedef std::list< TaskInterface* > TIList;
      bool isOk = true;
      Expr::TagList missingTags;
      const TIList& tilist = timeStepper_->get_task_interfaces();
      for( TIList::const_iterator iti=tilist.begin(); iti!=tilist.end(); ++iti ){
        const Expr::TagList tags = (*iti)->collect_tags_in_task();
        for( Expr::TagList::const_iterator itag=tags.begin(); iti!=tags.end(); ++iti ){
          if( itag->context_ == Expr::CARRY_FORWARD ){
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

  }

  //--------------------------------------------------------------------

  void
  Wasatch::create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                          const Uintah::MaterialSet* const materials,
                                          const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched,
                                          const int rkStage )
  {
    if( adaptors_.size() == 0 ) return; // no equations registered.

    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];
    Expr::ExpressionFactory& exprFactory = *gh->exprFactory;

    //_____________________________________________________________
    // create an expression to set the current time as a field that
    // will be available to all expressions if needed.
    const Expr::Tag timeTag (StringNames::self().time,Expr::STATE_NONE);
    Expr::ExpressionID timeID;
    if( rkStage==1 ){
      Expr::TagList timeTags;
      timeTags.push_back( Expr::Tag( StringNames::self().time, Expr::STATE_NONE ) );
      timeTags.push_back( Expr::Tag( StringNames::self().timestep, Expr::STATE_NONE ) );
      const Expr::Tag timeTag( StringNames::self().time, Expr::STATE_NONE );
      timeID = exprFactory.register_expression( scinew SetCurrentTime::Builder( timeTags, sharedState_, rkStage), true );
    } else {
      timeID = exprFactory.get_id(timeTag);
    }

    //___________________________________________
    // Plug in each equation that has been set up
    if (rkStage==1) {
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
    timeStepper_->create_tasks( timeID,
                                patchInfoMap_,
                                localPatches,
                                materials,
                                level,
                                sched,
                                rkStage, ioFieldSet_ );
  }

  //--------------------------------------------------------------------

  void
  Wasatch::computeDelT( const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches,
                        const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* old_dw,
                        Uintah::DataWarehouse* new_dw )
  {
    const double deltat = 1.0; // jcs should get this from an input file possibly?

//       proc0cout << std::endl
//                 << "Wasatch: executing 'Wasatch::computeDelT()' on all patches"
//                 << std::endl;
      new_dw->put( Uintah::delt_vartype(deltat),
                  sharedState_->get_delt_label(),
                  Uintah::getLevel(patches) );
      //                   material );
      // jcs it seems that we cannot specify a material here.  Why not?
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
    case USE_FOR_OPERATORS:

      const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
      const Uintah::PatchSubset* const localPatches = allPatches->getSubset( d_myworld->myrank() );
      Uintah::PatchSet* patches = new Uintah::PatchSet;
      // jcs: this results in "normal" scheduling and WILL NOT WORK FOR LINEAR SOLVES
      //      in that case, we need to use "gang" scheduling: addAll( localPatches )
      patches->addEach( localPatches->getVector() );
      //     const std::set<int>& procs = sched->getLoadBalancer()->getNeighborhoodProcessors();
      //     for( std::set<int>::const_iterator ip=procs.begin(); ip!=procs.end(); ++ip ){
      //       patches->addEach( allPatches->getSubset( *ip )->getVector() );
      //     }
      patchSetList_.push_back( patches );
      return patches;
    }
    return NULL;
  }

 //------------------------------------------------------------------

 void
 Wasatch::scheduleCoarsen(const Uintah::LevelP& /*coarseLevel*/,
                          Uintah::SchedulerP& /*sched*/)
 {
   // do nothing for now
 }

 //------------------------------------------------------------------

 void
 Wasatch::scheduleRefineInterface(const Uintah::LevelP& /*fineLevel*/,
                                  Uintah::SchedulerP& /*scheduler*/,
                                  bool, bool)
 {
   // do nothing for now
 }
//------------------------------------------------------------------

} // namespace Wasatch
