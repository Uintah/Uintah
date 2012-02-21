//-- Wasatch Includes --//

#include "TimeStepper.h"
#include "TaskInterface.h"
#include "CoordHelper.h"
#include "StringNames.h"
#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>

//-- ExprLib includes --//
#include <expression/FieldManager.h>  // for field type mapping
#include <expression/ExpressionTree.h>

//-- SpatialOps includes --//
#include <spatialops/FieldExpressions.h>

//-- Uintah Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

using std::endl;
namespace SO=SpatialOps::structured;


namespace Wasatch{

  //==================================================================

  template<typename FieldT>
  void
  set_soln_field_requirements( Uintah::Task* const task,
                               const std::set< TimeStepper::FieldInfo<FieldT> >& fields,
                               const Uintah::PatchSubset* const pss,
                               const Uintah::MaterialSubset* const mss,
                               const int rkStage )
  {
    typedef typename std::set< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld = fields.begin(); ifld!=fields.end(); ++ifld ){
      if (rkStage==1) task->computes( ifld->varLabel );
      else            task->modifies( ifld->varLabel );
      // jcs for some reason this one does not work:
      //       task->computes( ifld->varLabel,
      //                       pss, Uintah::Task::NormalDomain,
      //                       mss, Uintah::Task::NormalDomain );
      task->requires( Uintah::Task::OldDW,
                      ifld->varLabel,
                      pss, Uintah::Task::NormalDomain,
                      mss, Uintah::Task::NormalDomain,
                      get_uintah_ghost_type<FieldT>(),
                      get_n_ghost<FieldT>() );
      task->requires( Uintah::Task::NewDW,
                      ifld->rhsLabel,
                      pss, Uintah::Task::NormalDomain,
                      mss, Uintah::Task::NormalDomain,
                      get_uintah_ghost_type<FieldT>(),
                      get_n_ghost<FieldT>() );
    }
  }

  //==================================================================

  template<typename FieldT>
  void
  do_update( const std::set< TimeStepper::FieldInfo<FieldT> >& fields,
             const Uintah::Patch* const patch,
             const int material,
             Uintah::DataWarehouse* const oldDW,
             Uintah::DataWarehouse* const newDW,
             const double deltat,
             const int rkStage )
  {
    typedef std::set< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld=fields.begin(); ifld!=fields.end(); ++ifld ){

      typedef typename SelectUintahFieldType<FieldT>::const_type ConstUintahField;
      typedef typename SelectUintahFieldType<FieldT>::type       UintahField;

      const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<FieldT>();
      const int ng = get_n_ghost<FieldT>();

      UintahField phiNew;
      ConstUintahField phiOld, rhs;
      if (rkStage==1) {
        newDW->allocateAndPut( phiNew, ifld->varLabel, material, patch, gt, ng );  // note that these fields do have ghost info.
      } else {
        newDW->           getModifiable( phiNew, ifld->varLabel, material, patch, gt, ng );
      }
      oldDW->           get( phiOld, ifld->varLabel, material, patch, gt, ng );
      newDW->           get( rhs,    ifld->rhsLabel, material, patch, gt, ng );

      //______________________________________
      // forward Euler or RK3SSP timestep at each point:
      FieldT* const fnew = wrap_uintah_field_as_spatialops<FieldT>(phiNew,patch);
      const FieldT* const fold = wrap_uintah_field_as_spatialops<FieldT>(phiOld,patch);
      const FieldT* const frhs = wrap_uintah_field_as_spatialops<FieldT>(rhs,patch);
      using namespace SpatialOps;
      if (rkStage==1) {
        *fnew <<= *fold + deltat * *frhs;
      } else if (rkStage==2) {
        *fnew <<= 0.25 * *fnew + 0.75 * *fold + 0.25*deltat * *frhs;
      } else if (rkStage==3) {
        *fnew <<= 2.0/3.0* *fnew + 1.0/3.0* *fold + 2.0/3.0*deltat * *frhs;
      }
      delete fnew; delete fold; delete frhs;
    }
  }

  //==================================================================

  TimeStepper::TimeStepper( const Uintah::VarLabel* const deltaTLabel,
                            GraphHelper& solnGraphHelper )
    : solnGraphHelper_( &solnGraphHelper ),
      deltaTLabel_( deltaTLabel ),
      coordHelper_( new CoordHelper( *(solnGraphHelper_->exprFactory) ) )
  {}

  //------------------------------------------------------------------

  TimeStepper::~TimeStepper()
  {
    delete coordHelper_;
    for( std::list<TaskInterface*>::iterator i=taskInterfaceList_.begin(); i!=taskInterfaceList_.end(); ++i ){
      delete *i;
    }
    for( std::vector<Uintah::VarLabel*>::iterator i=createdVarLabels_.begin(); i!=createdVarLabels_.end(); ++i ){
      Uintah::VarLabel::destroy( *i );
    }
  }

  //------------------------------------------------------------------

  // jcs this should be done on a single patch, since the PatchInfo is for a single patch.
  void
  TimeStepper::create_tasks( const Expr::ExpressionID timeID,
                             const PatchInfoMap& patchInfoMap,
                             const Uintah::PatchSet* const patches,
                             const Uintah::MaterialSet* const materials,
                             const Uintah::LevelP& level,
                             Uintah::SchedulerP& sched,
                             const int rkStage,
                             const std::set<std::string>& ioFieldSet )
  {
    // for now we will assume that we are computing things on ALL materials
    const Uintah::MaterialSubset* const mss = materials->getUnion();
    std::stringstream strRKStage;
    strRKStage << rkStage;

    //________________________________________________________
    // add a task to populate a "field" with the current time.
    // This is required by the time integrator.
    {
      TaskInterface* const timeTask = scinew TaskInterface( timeID,
                                                            "set_time",
                                                            *(solnGraphHelper_->exprFactory),
                                                            level, sched, patches, materials,
                                                            patchInfoMap,
                                                            true, 1, ioFieldSet );
      taskInterfaceList_.push_back( timeTask );
      timeTask->schedule( coordHelper_->field_tags(), rkStage );
      // add a task to update current simulation time
      Uintah::Task* updateCurrentTimeTask = scinew Uintah::Task( "update current time", this, &TimeStepper::update_current_time, timeTask->get_time_tree(), rkStage );
      updateCurrentTimeTask->requires( Uintah::Task::OldDW, deltaTLabel_ );
      sched->addTask( updateCurrentTimeTask, patches, materials );
    }

    //_________________________________________________________________
    // Schedule the task to compute the RHS for the transport equations
    //

    try{
      // jcs for multistage integrators, we may need to keep the same
      //     field manager list for all of the stages?  Otherwise we
      //     will have all sorts of name clashes?

      TaskInterface* rhsTask = scinew TaskInterface( solnGraphHelper_->rootIDs,
                                                     "rhs_" + strRKStage.str(),
                                                     *(solnGraphHelper_->exprFactory),
                                                     level, sched, patches, materials,
                                                     patchInfoMap,
                                                     true,
                                                     rkStage, ioFieldSet );

      taskInterfaceList_.push_back( rhsTask );
      if(rkStage==1) coordHelper_->create_task( sched, patches, materials );
      rhsTask->schedule( coordHelper_->field_tags(), rkStage ); // must be scheduled after coordHelper_
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << "*************************************************" << endl
          << "Error building ExpressionTree for RHS evaluation." << endl << endl
          << e.what() << endl
          << "*************************************************" << endl << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    //_____________________________________________________
    // add a task to advance each solution variable in time
    {
      Uintah::Task* updateTask = scinew Uintah::Task( "update solution vars_" + strRKStage.str(), this, &TimeStepper::update_variables, rkStage );

      const Uintah::PatchSubset* const pss = patches->getUnion();
      set_soln_field_requirements<SO::SVolField>( updateTask, scalarFields_, pss, mss, rkStage );
      set_soln_field_requirements<SO::XVolField>( updateTask, xVolFields_,   pss, mss, rkStage );
      set_soln_field_requirements<SO::YVolField>( updateTask, yVolFields_,   pss, mss, rkStage );
      set_soln_field_requirements<SO::ZVolField>( updateTask, zVolFields_,   pss, mss, rkStage );

      // we require the timestep value
      updateTask->requires( Uintah::Task::OldDW, deltaTLabel_ );
      /* jcs if we specify this, then things fail:
                            patches, Uintah::Task::NormalDomain,
                            mss, Uintah::Task::NormalDomain,
                            Uintah::Ghost::None, 0 );
      */

      sched->addTask( updateTask, patches, materials );
    }

  }

  //------------------------------------------------------------------

  void
  TimeStepper::update_current_time( const Uintah::ProcessorGroup* const pg,
                                    const Uintah::PatchSubset* const patches,
                                    const Uintah::MaterialSubset* const materials,
                                    Uintah::DataWarehouse* const oldDW,
                                    Uintah::DataWarehouse* const newDW,
                                    Expr::ExpressionTree::TreePtr timeTree,
                                    const int rkStage )
  {
    // grab the timestep
    Uintah::delt_vartype deltat;
    oldDW->get( deltat, deltaTLabel_ );

    const Expr::Tag TimeTag (StringNames::self().time,Expr::STATE_NONE);
    SetCurrentTime& settimeexpr = dynamic_cast<SetCurrentTime&>( timeTree->get_expression( TimeTag ) );
    settimeexpr.set_integrator_stage(rkStage);
    settimeexpr.set_deltat(deltat);
  }

  //------------------------------------------------------------------

  void
  TimeStepper::update_variables( const Uintah::ProcessorGroup* const pg,
                                 const Uintah::PatchSubset* const patches,
                                 const Uintah::MaterialSubset* const materials,
                                 Uintah::DataWarehouse* const oldDW,
                                 Uintah::DataWarehouse* const newDW,
                                 const int rkStage )
  {
    //__________________
    // loop over patches
    for( int ip=0; ip<patches->size(); ++ip ){

      const Uintah::Patch* const patch = patches->get(ip);

      //____________________
      // loop over materials
      for( int im=0; im<materials->size(); ++im ){

        const int material = materials->get(im);

        // grab the timestep
        Uintah::delt_vartype deltat;
        //jcs this doesn't work:
        //newDW->get( deltat, deltaTLabel_, Uintah::getLevel(patches), material );
        oldDW->get( deltat, deltaTLabel_ );

        //____________________________________________
        // update variables on this material and patch
				// jcs note that we could do this in parallel
        do_update<SO::SVolField>( scalarFields_, patch, material, oldDW, newDW, deltat, rkStage );
        do_update<SO::XVolField>( xVolFields_,   patch, material, oldDW, newDW, deltat, rkStage );
        do_update<SO::YVolField>( yVolFields_,   patch, material, oldDW, newDW, deltat, rkStage );
        do_update<SO::ZVolField>( zVolFields_,   patch, material, oldDW, newDW, deltat, rkStage );

      } // material loop
    } // patch loop
  }

  //------------------------------------------------------------------

} // namespace Wasatch
