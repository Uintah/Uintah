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
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/ProblemSetupException.h>


//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>


//-- ExprLib includes --//
#include <expression/ExprLib.h>


//-- Wasatch includes --//
#include "Wasatch.h"
#include "FieldAdaptor.h"
#include "StringNames.h"
#include "TaskInterface.h"
#include "TimeStepper.h"
#include "Properties.h"
#include "Operators/Operators.h"
#include "Expressions/BasicExprBuilder.h"
#include "Expressions/SetCurrentTime.h"
#include "Expressions/Coordinate.h"
#include "transport/ParseEquation.h"

namespace Wasatch{

  //--------------------------------------------------------------------

  Wasatch::Wasatch( const Uintah::ProcessorGroup* myworld )
    : Uintah::UintahParallelComponent( myworld )
  {
    needCoords_ = false;
    xSVolCoord_ = ySVolCoord_ = zSVolCoord_ = false;
    xXVolCoord_ = yXVolCoord_ = zXVolCoord_ = false;
    xYVolCoord_ = yYVolCoord_ = zYVolCoord_ = false;
    xZVolCoord_ = yZVolCoord_ = zZVolCoord_ = false;


    const bool log = true;
    graphCategories_[ INITIALIZATION     ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ TIMESTEP_SELECTION ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
    graphCategories_[ ADVANCE_SOLUTION   ] = scinew GraphHelper( scinew Expr::ExpressionFactory(log) );
  }

  //--------------------------------------------------------------------

  Wasatch::~Wasatch()
  {
    // wipe out the patchInfoMap_ stuff
    for( PatchInfoMap::iterator i=patchInfoMap_.begin(); i!=patchInfoMap_.end(); ++i ){
      delete i->second.operators;
    }

    for( GraphCategories::iterator igc=graphCategories_.begin(); igc!=graphCategories_.end(); ++igc ){
      delete igc->second->exprFactory;
    }

    for( EquationAdaptors::iterator i=adaptors_.begin(); i!=adaptors_.end(); ++i ){
      delete *i;
    }

    delete timeStepper_;
  }

  //--------------------------------------------------------------------

  void Wasatch::problemSetup( const Uintah::ProblemSpecP& params, 
                              const Uintah::ProblemSpecP& ,  /* jcs not sure what this param is for */
                              Uintah::GridP& grid, 
                              Uintah::SimulationStateP& sharedState )
  {
    sharedState_ = sharedState;

    Uintah::ProblemSpecP wasatchParams = params->findBlock("Wasatch");

    //
    // Material
    //
    Uintah::SimpleMaterial* mymaterial = scinew Uintah::SimpleMaterial();  
    sharedState->registerSimpleMaterial(mymaterial);

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
    for( Uintah::ProblemSpecP transEqnParams=wasatchParams->findBlock("TransportEquation");
         transEqnParams != 0;
         transEqnParams=transEqnParams->findNextBlock("TransportEquation") ){
      adaptors_.push_back( parse_equation( transEqnParams, graphCategories_ ) );
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
    const Uintah::PatchSet* localPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);

    for( int ip=0; ip<localPatches->size(); ++ip ){

      const Uintah::PatchSubset* const patches = localPatches->getSubset(ip);

      for( int ipss=0; ipss<patches->size(); ++ipss ){

        SpatialOps::OperatorDatabase* const opdb = scinew SpatialOps::OperatorDatabase();
        const Uintah::Patch* const patch = patches->get(ipss);
        build_operators( *patch, *opdb );
        PatchInfo& pi = patchInfoMap_[patch->getID()];
        pi.operators   = opdb;
        pi.patchID = patch->getID();
      }
    }

    GraphHelper* const icGraphHelper = graphCategories_[ INITIALIZATION ];

    //_____________________________________________________________
    // build expressions to set coordinates.  If any initialization
    // expressions require the coordinates, then this will trigger
    icGraphHelper->exprFactory->register_expression( Expr::Tag("XSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(this,XDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("YSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(this,YDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("ZSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(this,ZDIR) );

    icGraphHelper->exprFactory->register_expression( Expr::Tag("XXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(this,XDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("YXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(this,YDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("ZXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(this,ZDIR) );

    icGraphHelper->exprFactory->register_expression( Expr::Tag("XYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(this,XDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("YYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(this,YDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("ZYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(this,ZDIR) );

    icGraphHelper->exprFactory->register_expression( Expr::Tag("XZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(this,XDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("YZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(this,YDIR) );
    icGraphHelper->exprFactory->register_expression( Expr::Tag("ZZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(this,ZDIR) );

    //_____________________________________________
    // Build the initial condition expression graph
    if( !icGraphHelper->rootIDs.empty() ){

      Expr::ExpressionTree* const graph = scinew Expr::ExpressionTree( *icGraphHelper->exprFactory, -1, "initialization" );

      for( IDSet::const_iterator iid=icGraphHelper->rootIDs.begin(); iid!=icGraphHelper->rootIDs.end(); ++iid ){
        graph->insert_tree( *iid );
      }

      // at this point, if we needed coordinate information we will
      // have called back to set that fact. Schedule the coordinate
      // calculation prior to the initialization task.
      if( needCoords_ ){
        Uintah::Task* task = scinew Uintah::Task( "coordinates", this, &Wasatch::set_grid_variables );
        register_coord_fields( task, localPatches, sharedState_->allMaterials() );
        sched->addTask( task, localPatches, sharedState_->allMaterials() );
      }

      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      TaskInterface* const task = scinew TaskInterface( graph, patchInfoMap_ );
      task->schedule( sched, localPatches, sharedState_->allMaterials() );

      //________________
      // jcs diagnostics
      std::cout << "writing tree file to 'initialization.dot'" << std::endl;
      std::ofstream fout("initialization.dot");
      graph->write_tree(fout);
    }

    std::cout << "Wasatch: done creating initialization task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void Wasatch::scheduleComputeStableTimestep( const Uintah::LevelP& level,
                                               Uintah::SchedulerP& sched )
  {
    GraphHelper* const tsGraphHelper = graphCategories_[ TIMESTEP_SELECTION ];

    // jcs: was getting patch set this way (from discussions with Justin).
    // sched->getLoadBalancer()->getPerProcessorPatchSet(level),
    const Uintah::PatchSet* patches = level->eachPatch();
    const Uintah::MaterialSet* materials = sharedState_->allMaterials();

    if( tsGraphHelper->rootIDs.size() > 0 ){

      Expr::ExpressionTree* const graph = scinew Expr::ExpressionTree( *tsGraphHelper->exprFactory, -1, "initialization" );

      for( IDSet::const_iterator iid=tsGraphHelper->rootIDs.begin(); iid!=tsGraphHelper->rootIDs.end(); ++iid ){
        graph->insert_tree( *iid );
      }

      //_______________________________________________________
      // create the TaskInterface and schedule this task for
      // execution.  Note that field dependencies are assigned
      // within the TaskInterface object.
      TaskInterface* const task = scinew TaskInterface( graph, patchInfoMap_ );
      task->schedule( sched, patches, materials );

      //________________
      // jcs diagnostics
      std::cout << "writing tree file to 'deltat.dot'" << std::endl;
      std::ofstream fout("detat.dot");
      graph->write_tree(fout);
    }
    else{ // default

      cout << "Task 'compute timestep' COMPUTES 'delT' in NEW data warehouse" << endl;
      Uintah::Task* task = scinew Uintah::Task( "compute timestep", this, &Wasatch::computeDelT );

      // jcs it appears that for reduction variables we cannot specify the patches - only the materials.
      task->computes( sharedState_->get_delt_label(),
                      level.get_rep() );
      //              materials->getUnion() );
      // jcs why can't we specify a metrial here?  It doesn't seem to be working if I do.

      sched->addTask( task, level->eachPatch(), sharedState_->allMaterials() );

    }

    std::cout << "Wasatch: done creating timestep task(s)" << std::endl;
  }

  //--------------------------------------------------------------------

  void
  Wasatch::scheduleTimeAdvance( const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched )
  {
    create_timestepper_on_patches( sched->getLoadBalancer()->getPerProcessorPatchSet(level),
                                   sharedState_->allMaterials(),
                                   sched );

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
//       Expr::ExpressionTree* timeTree = scinew Expr::ExpressionTree( timeID, exprFactory, -1, "set time" );
//       TaskInterface* const timeTask = scinew TaskInterface( timeTree, patchInfoMap_ );
//       timeTask->schedule( sched, localPatches, materials );
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


  void
  Wasatch::register_coord_fields( Uintah::Task* const task,
                                  const Uintah::PatchSet* const ps,
                                  const Uintah::MaterialSet* const ms )
  {
    const Uintah::PatchSubset*    const pss = ps->getUnion();
    const Uintah::MaterialSubset* const mss = ms->getUnion();

    const Uintah::Task::DomainSpec domain = Uintah::Task::NormalDomain;

    if( xSVolCoord_ ){
      xSVol_=Uintah::VarLabel::create("XSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( xSVol_, pss, domain, mss, domain );
    }
    if( ySVolCoord_ ){
      ySVol_=Uintah::VarLabel::create("YSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( ySVol_, pss, domain, mss, domain );
    }
    if( zSVolCoord_ ){
      zSVol_=Uintah::VarLabel::create("ZSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( zSVol_, pss, domain, mss, domain );
    }

    if( xXVolCoord_ ){
      xXVol_=Uintah::VarLabel::create("XXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( xXVol_, pss, domain, mss, domain );
    }
    if( yXVolCoord_ ){
      yXVol_=Uintah::VarLabel::create("YXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( yXVol_, pss, domain, mss, domain );
    }
    if( zXVolCoord_ ){
      zSVol_=Uintah::VarLabel::create("ZXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( zXVol_, pss, domain, mss, domain );
    }

    if( xYVolCoord_ ){
      xYVol_=Uintah::VarLabel::create("XYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( xYVol_, pss, domain, mss, domain );
    }
    if( yYVolCoord_ ){
      yYVol_=Uintah::VarLabel::create("YYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( yYVol_, pss, domain, mss, domain );
    }
    if( zYVolCoord_ ){
      zYVol_=Uintah::VarLabel::create("ZYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( zYVol_, pss, domain, mss, domain );
    }

    if( xZVolCoord_ ){
      xZVol_=Uintah::VarLabel::create("XZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( xZVol_, pss, domain, mss, domain );
    }
    if( yZVolCoord_ ){
      yZVol_=Uintah::VarLabel::create("YZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( yZVol_, pss, domain, mss, domain );
    }
    if( zZVolCoord_ ){
      zZVol_=Uintah::VarLabel::create("ZZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( zZVol_, pss, domain, mss, domain );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void set_coord( FieldT& field, const Uintah::Patch* p, const double shift, const int idir )
  {
    IntVector low, hi;
    for( int k=low[2]; k<low[2]; ++k ){
      for( int j=low[1]; j<low[1]; ++j ){
        for( int i=low[0]; i<low[0]; ++i ){
          const IntVector index(i,j,k);
          const SCIRun::Vector xyz = p->getCellPosition(index).vector();
          field[index] = xyz[idir] + shift;
        }
      }
    }
  }

  //------------------------------------------------------------------

  void
  Wasatch::set_grid_variables( const Uintah::ProcessorGroup* const pg,
                               const Uintah::PatchSubset* const patches,
                               const Uintah::MaterialSubset* const materials,
                               Uintah::DataWarehouse* const oldDW,
                               Uintah::DataWarehouse* const newDW )
  {
    for( int ip=0; ip<patches->size(); ++ip ){
      const Uintah::Patch* const patch = patches->get(ip);
      const PatchInfoMap::const_iterator ipim = patchInfoMap_.find(patch->getID());
      ASSERT( ipim!=patchInfoMap_.end() );

      for( int im=0; im<materials->size(); ++im ){

        const int material = materials->get(im);

        // populate the field.

        const SCIRun::Vector spacing = patch->dCell();

        double shift = 0.0;
        if( xSVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, xSVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( ySVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, ySVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( zSVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, zSVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }

        shift = -spacing[0]*0.5;  // shift x by -dx/2
        if( xXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, xXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( yXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, yXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( zXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, zXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }

        shift = -spacing[1]*0.5;
        if( xYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, xYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }
        if( yYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, yYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }
        if( zYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, zYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }

        shift = -spacing[1]*0.5;
        if( xYVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, xZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }
        if( yZVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, yZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }
        if( zZVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, zZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }

      }  // material loop
    } // patch loop
  }

  //------------------------------------------------------------------

} // namespace Wasatch
