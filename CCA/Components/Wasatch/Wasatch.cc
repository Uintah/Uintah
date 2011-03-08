/*

  The MIT License

  Copyright (c) Institute for Clean & Secure Energy (ICSE), University of Utah.

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

//-- Wasatch includes --//
#include "Wasatch.h"
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
#include "BCHelperTools.h"

namespace Wasatch{

  //--------------------------------------------------------------------

  Wasatch::Wasatch( const Uintah::ProcessorGroup* myworld )
    : Uintah::UintahParallelComponent( myworld )
  {
    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::d_combineMemory = false;

    const bool log = false;
    graphCategories_[ INITIALIZATION     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ TIMESTEP_SELECTION ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ ADVANCE_SOLUTION   ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );

    icCoordHelper_  = new CoordHelper( *(graphCategories_[INITIALIZATION  ]->exprFactory) );
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
    delete timeStepper_;

    for( GraphCategories::iterator igc=graphCategories_.begin(); igc!=graphCategories_.end(); ++igc ){
      delete igc->second->exprFactory;
      delete igc->second;
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

    Uintah::ProblemSpecP wasatchParams = params->findBlock("Wasatch");

    //
    // Material
    //
    Uintah::SimpleMaterial* mymaterial = scinew Uintah::SimpleMaterial();  
    sharedState->registerSimpleMaterial(mymaterial);

    // we are able to get the solver port from here
    linSolver_ = dynamic_cast<Uintah::SolverInterface*>(getPort("solver"));
    if(!linSolver_) {
      throw Uintah::InternalError("Wasatch: couldn't get solver port", __FILE__, __LINE__);
    } else if (linSolver_) {
      std::cout << "Detected solver port... \n";
    }
    
    //
    // create expressions explicitly defined in the input file.  These
    // are typically associated with, e.g. initial conditions.
    //
    create_expressions_from_input( wasatchParams, graphCategories_ );

    setup_property_evaluation( wasatchParams, *graphCategories_[ADVANCE_SOLUTION] );

    //
    // Build transport equations.  This registers all expressions as
    // appropriate for solution of each transport equation.
    //
    for( Uintah::ProblemSpecP momEqnParams=wasatchParams->findBlock("TransportEquation");
         momEqnParams != 0;
         momEqnParams=momEqnParams->findNextBlock("TransportEquation") ){
      adaptors_.push_back( parse_equation( momEqnParams, graphCategories_ ) );
    }
    
    //
    // Build momentum transport equations.  This registers all expressions 
    // required for solution of each momentum equation.
    //
    for( Uintah::ProblemSpecP transEqnParams=wasatchParams->findBlock("MomentumEquations");
        transEqnParams != 0;
        transEqnParams=transEqnParams->findNextBlock("MomentumEquations") ){
      // note - parse_momentum_equations returns a vector of equation adaptors
      try{
        EquationAdaptors momentumAdaptors = parse_momentum_equations( transEqnParams, graphCategories_, *linSolver_);
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
    
    

    timeStepper_ = scinew TimeStepper( sharedState_->get_delt_label(),
                                       *graphCategories_[ ADVANCE_SOLUTION ]->exprFactory );
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleInitialize( const Uintah::LevelP& level,
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
    const Uintah::PatchSet* const localPatches = get_patchset( level, sched );
    const Uintah::MaterialSet* const materials = sharedState_->allMaterials();

    for( int ipss=0; ipss<localPatches->size(); ++ipss ){
        const Uintah::PatchSubset* pss = localPatches->getSubset(ipss);
        for( int ip=0; ip<pss->size(); ++ip ){
          SpatialOps::OperatorDatabase* const opdb = scinew SpatialOps::OperatorDatabase();
          const Uintah::Patch* const patch = pss->get(ip);
          build_operators( *patch, *opdb );
          PatchInfo& pi = patchInfoMap_[patch->getID()];
          pi.operators = opdb;
          pi.patchID = patch->getID();
        }
      }

    GraphHelper* const icGraphHelper = graphCategories_[ INITIALIZATION ];

    Expr::ExpressionFactory& exprFactory = *icGraphHelper->exprFactory;

    //_______________________________________
    // set the time
    exprFactory.register_expression( Expr::Tag(StringNames::self().time,Expr::STATE_NONE),
                                     scinew SetCurrentTime::Builder(sharedState_) );
    
    //_____________________________________________
    // Build the initial condition expression graph
    if( !icGraphHelper->rootIDs.empty() ){

      TaskInterface* const task = scinew TaskInterface( icGraphHelper->rootIDs,
                                                        "initialization",
                                                        *icGraphHelper->exprFactory,
                                                        sched,
                                                        localPatches,
                                                        materials,
                                                        patchInfoMap_,
                                                        true );

      // set coordinate values as required by the IC graph.
      icCoordHelper_->create_task( sched, localPatches, materials );

      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      task->schedule( icCoordHelper_->field_tags() );
      taskInterfaceList_.push_back( task );
    }
    if( d_myworld->myrank() == 0 )
      std::cout << "Wasatch: done creating initialization task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    GraphHelper* const tsGraphHelper = graphCategories_[ TIMESTEP_SELECTION ];

    // jcs: was getting patch set this way (from discussions with Justin).
    const Uintah::PatchSet* const localPatches = get_patchset(level,sched);

    const Uintah::MaterialSet* materials = sharedState_->allMaterials();

    if( tsGraphHelper->rootIDs.size() > 0 ){

      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      TaskInterface* const task = scinew TaskInterface( tsGraphHelper->rootIDs,
                                                        "compute timestep",
                                                        *tsGraphHelper->exprFactory,
                                                        sched,
                                                        localPatches,
                                                        materials,
                                                        patchInfoMap_,
                                                        true );
      task->schedule();
      taskInterfaceList_.push_back( task );
    }
    else{ // default

      if( d_myworld->myrank() == 0 )
        cout << "Task 'compute timestep' COMPUTES 'delT' in NEW data warehouse" << endl;

      Uintah::Task* task = scinew Uintah::Task( "compute timestep", this, &Wasatch::computeDelT );

      // jcs it appears that for reduction variables we cannot specify the patches - only the materials.
      task->computes( sharedState_->get_delt_label(),
                      level.get_rep() );
      //              materials->getUnion() );
      // jcs why can't we specify a metrial here?  It doesn't seem to be working if I do.

      sched->addTask( task, localPatches, sharedState_->allMaterials() );
    }

    if( d_myworld->myrank() == 0 )
      std::cout << "Wasatch: done creating timestep task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void
  Wasatch::scheduleTimeAdvance( const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched )
  {
    // jcs why do we need this instead of getting the level?
    const Uintah::PatchSet* const localPatches = get_patchset( level, sched );

    const Uintah::MaterialSet* const materials = sharedState_->allMaterials();

    create_timestepper_on_patches( localPatches, materials, sched );
    
    if( d_myworld->myrank() == 0 )
      std::cout << "Wasatch: done creating solution task(s)" << std::endl;

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
    const GraphHelper* gh = graphCategories_[ ADVANCE_SOLUTION ];
    build_bcs( adaptors_, *gh, localPatches, patchInfoMap_, materials->getUnion() );
  }
  
  //--------------------------------------------------------------------

  void
  Wasatch::create_timestepper_on_patches( const Uintah::PatchSet* const localPatches,
                                          const Uintah::MaterialSet* const materials,
                                          Uintah::SchedulerP& sched )
  {
    if( adaptors_.size() == 0 ) return; // no equations registered.

    GraphHelper* const gh = graphCategories_[ ADVANCE_SOLUTION ];
    Expr::ExpressionFactory& exprFactory = *gh->exprFactory;

    //_____________________________________________________________
    // create an expression to set the current time as a field that
    // will be available to all expressions if needed.
    const Expr::ExpressionID timeID =
      exprFactory.register_expression( Expr::Tag(StringNames::self().time,Expr::STATE_NONE),
                                       scinew SetCurrentTime::Builder(sharedState_) );

    //___________________________________________
    // Plug in each equation that has been set up
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
        cout << msg.str() << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
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
                                sched );

//     //________________________________________________________
//     // add a task to populate a "field" with the current time.
//     // This is required by the time integrator in general since
//     // some things (e.g. boundary conditions) may be prescribed
//     // functions of time.
//     {
//       TaskInterface* const timeTask = scinew TaskInterface( timeID, "set time", exprFactory, sched, localPatches, materials, patchInfoMap_, true );
//       timeTask->schedule( sched );
//       taskInterfaceList_.push_back( timeTask );
//     }
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

//       std::cout << std::endl
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
  Wasatch::get_patchset( const Uintah::LevelP& level,
                         Uintah::SchedulerP& sched )
  {
//     return sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    return level->eachPatch();

//     const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
//     const Uintah::PatchSubset* const localPatches = allPatches->getSubset( d_myworld->myrank() );
//     Uintah::PatchSet* patches = new Uintah::PatchSet;
//     // jcs: this results in "normal" scheduling and WILL NOT WORK FOR LINEAR SOLVES
//     //      in that case, we need to use "gang" scheduling: addAll( localPatches )
//     patches->addEach( localPatches->getVector() );
// //     const std::set<int>& procs = sched->getLoadBalancer()->getNeighborhoodProcessors();
// //     for( std::set<int>::const_iterator ip=procs.begin(); ip!=procs.end(); ++ip ){
// //       patches->addEach( allPatches->getSubset( *ip )->getVector() );
// //     }
//     patchSetList_.push_back( patches );
//     return patches;
  }

  //------------------------------------------------------------------

} // namespace Wasatch
