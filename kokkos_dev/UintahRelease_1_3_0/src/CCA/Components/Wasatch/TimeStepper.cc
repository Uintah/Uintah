//-- Wasatch Includes --//
#include "TimeStepper.h"
#include "TaskInterface.h"
#include "CoordHelper.h"

//-- ExprLib includes --//
#include <expression/FieldManager.h>  // for field type mapping
#include <expression/ExpressionTree.h>


//-- Uintah Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>


namespace SO=SpatialOps::structured;


namespace Wasatch{

  template<typename FieldT> struct IteratorSelector;

  template<> struct IteratorSelector<SO::SVolField>
  {
    typedef Uintah::CellIterator type;
    static type getBegin( const Uintah::Patch* patch ){ return patch->getCellIterator(); }
  };
  template<> struct IteratorSelector<SO::XVolField>
  {
    typedef Uintah::CellIterator type;
    static type getBegin( const Uintah::Patch* patch ){ return patch->getSFCXIterator(); }
  };
  template<> struct IteratorSelector<SO::YVolField>
  {
    typedef Uintah::CellIterator type;
    static type getBegin( const Uintah::Patch* patch ){ return patch->getSFCYIterator(); }
  };
  template<> struct IteratorSelector<SO::ZVolField>
  {
    typedef Uintah::CellIterator type;
    static type getBegin( const Uintah::Patch* patch ){ return patch->getSFCZIterator(); }
  };

  //==================================================================

  template<typename FieldT>
  void
  set_field_requirements( Uintah::Task* const task,
                          const std::vector< TimeStepper::FieldInfo<FieldT> >& fields,
                          const Uintah::PatchSubset* const pss,
                          const Uintah::MaterialSubset* const mss )
  {
    typedef typename std::vector< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld = fields.begin(); ifld!=fields.end(); ++ifld ){
      cout << "timestepper COMPUTES '" << ifld->varLabel->getName() << "' in NEW DW" << endl
           << "            REQUIRES '" << ifld->varLabel->getName() << "' in OLD DW" << endl
           << "            REQUIRES '" << ifld->rhsLabel->getName() << "' in NEW DW" << endl
           << endl;
      task->computes( ifld->varLabel );
      // jcs for some reason this one does not work:
      //       task->computes( ifld->varLabel,
      //                       pss, Uintah::Task::NormalDomain,
      //                       mss, Uintah::Task::NormalDomain );
      task->requires( Uintah::Task::OldDW,
                      ifld->varLabel,
                      pss, Uintah::Task::NormalDomain,
                      mss, Uintah::Task::NormalDomain,
                      getUintahGhostType<FieldT>(),
                      getNGhost<FieldT>() );
      task->requires( Uintah::Task::NewDW,
                      ifld->rhsLabel,
                      pss, Uintah::Task::NormalDomain,
                      mss, Uintah::Task::NormalDomain,
                      getUintahGhostType<FieldT>(),
                      getNGhost<FieldT>() );
    }
  }

  //==================================================================

  template<typename FieldT>
  void
  do_update( const std::vector< TimeStepper::FieldInfo<FieldT> >& fields,
             const Uintah::Patch* const patch,
             const int material,
             Uintah::DataWarehouse* const oldDW,
             Uintah::DataWarehouse* const newDW,
             const double deltat )
  {
    typedef std::vector< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld=fields.begin(); ifld!=fields.end(); ++ifld ){

      typedef typename SelectUintahFieldType<FieldT>::const_type ConstUintahField;
      typedef typename SelectUintahFieldType<FieldT>::type       UintahField;

      const Uintah::Ghost::GhostType gt = getUintahGhostType<FieldT>();
      const int ng = getNGhost<FieldT>();

      UintahField phiNew;
      ConstUintahField phiOld, rhs;
      newDW->allocateAndPut( phiNew, ifld->varLabel, material, patch, gt, ng );  // note that these fields do have ghost info.
      oldDW->           get( phiOld, ifld->varLabel, material, patch, gt, ng );
      newDW->           get( rhs,    ifld->rhsLabel, material, patch, gt, ng );

      //______________________________________
      // forward Euler timestep at each point:
      typedef IteratorSelector<FieldT> CellIter;
      for( typename CellIter::type iter=CellIter::getBegin(patch); !iter.done(); ++iter ){
        phiNew[*iter] = phiOld[*iter] + deltat * rhs[*iter];
      }
    }
  }

  //==================================================================

  TimeStepper::TimeStepper( const Uintah::VarLabel* const deltaTLabel,
                            Expr::ExpressionFactory& factory )
    : factory_( &factory ),
      deltaTLabel_( deltaTLabel ),
      coordHelper_( new CoordHelper( factory ) )
  {}

  //------------------------------------------------------------------

  // jcs this should be done on a single patch, since the PatchInfo is for a single patch.
  void
  TimeStepper::create_tasks( const Expr::ExpressionID timeID,
                             const PatchInfoMap& patchInfoMap,
                             const Uintah::PatchSet* const patches,
                             const Uintah::MaterialSet* const materials,
                             Uintah::SchedulerP& sched )
  {
    // for now we will assume that we are computing things on ALL patches and ALL materials
    const Uintah::PatchSubset*    const pss = patches  ->getUnion();
    const Uintah::MaterialSubset* const mss = materials->getUnion();

    //_________________________________________________________________
    // Schedule the task to compute the RHS for the transport equations
    //
    try{
      // jcs for multistage integrators, we may need to keep the same
      //     field manager list for all of the stages?  Otherwise we
      //     will have all sorts of name clashes?
      Expr::ExpressionTree* rhsTree = scinew Expr::ExpressionTree( rhsIDs_, *factory_, -1, "rhs" );
      TaskInterface* rhsTask = scinew TaskInterface( rhsTree, patchInfoMap );
      coordHelper_->create_task( sched, patches, materials );
      rhsTask->schedule( sched, patches, materials, coordHelper_->field_tags() );
      
      // jcs hacked diagnostics - problems in parallel.
      const std::string fname("rhs.dot");
      std::cout << "writing RHS tree to '" << fname << "'" << std::endl;
      std::ofstream fout(fname.c_str());
      rhsTree->write_tree(fout);
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << "*************************************************" << endl
          << "Error building ExpressionTree for RHS evaluation." << endl << endl
          << e.what() << endl
          << "*************************************************" << endl << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    //________________________________________________________
    // add a task to populate a "field" with the current time.
    // This is required by the time integrator.
    {
      Expr::ExpressionTree* timeTree = scinew Expr::ExpressionTree( timeID, *factory_, -1, "set time" );
      TaskInterface* const timeTask = scinew TaskInterface( timeTree, patchInfoMap );
      timeTask->schedule( sched, patches, materials, coordHelper_->field_tags() );
    }

    //_____________________________________________________
    // add a task to advance each solution variable in time
    {
      Uintah::Task* updateTask = scinew Uintah::Task( "update solution vars", this, &TimeStepper::update_variables );
      
      set_field_requirements<SO::SVolField>( updateTask, scalarFields_, pss, mss );
      set_field_requirements<SO::XVolField>( updateTask, xVolFields_,   pss, mss );
      set_field_requirements<SO::YVolField>( updateTask, yVolFields_,   pss, mss );
      set_field_requirements<SO::ZVolField>( updateTask, zVolFields_,   pss, mss );

      // we require the timestep value
      updateTask->requires( Uintah::Task::NewDW, deltaTLabel_ );
      /* jcs if we specify this, then things fail:
                            pss, Uintah::Task::NormalDomain,
                            mss, Uintah::Task::NormalDomain,
                            Uintah::Ghost::None, 0 );
      */

      sched->addTask( updateTask, patches, materials );
    }

  }

  //------------------------------------------------------------------

  void
  TimeStepper::update_variables( const Uintah::ProcessorGroup* const pg,
                                 const Uintah::PatchSubset* const patches,
                                 const Uintah::MaterialSubset* const materials,
                                 Uintah::DataWarehouse* const oldDW,
                                 Uintah::DataWarehouse* const newDW )
  {
    //__________________
    // loop over patches
    for( int ip=0; ip<patches->size(); ++ip ){

      const Uintah::Patch* const patch = patches->get(ip);

      //____________________
      // loop over materials
      for( int im=0; im<materials->size(); ++im ){
        
        const int material = materials->get(im);

//         std::cout << std::endl
//                   << "Wasatch: executing 'TimeStepper::update_variables()' on patch "
//                   << patch->getID() << " and material " << material
//                   << std::endl;

        // grab the timestep
        Uintah::delt_vartype deltat;
        //jcs this doesn't work:
        //newDW->get( deltat, deltaTLabel_, Uintah::getLevel(patches), material );
        newDW->get( deltat, deltaTLabel_ );

//         cout << "TimeStepper::update_variables() : dt = " << deltat << endl;

        //____________________________________________
        // update variables on this material and patch
        do_update<SO::SVolField>( scalarFields_, patch, material, oldDW, newDW, deltat );
        do_update<SO::XVolField>( xVolFields_,   patch, material, oldDW, newDW, deltat );
        do_update<SO::YVolField>( yVolFields_,   patch, material, oldDW, newDW, deltat );
        do_update<SO::ZVolField>( zVolFields_,   patch, material, oldDW, newDW, deltat );

      } // material loop
    } // patch loop
  }

  //------------------------------------------------------------------

} // namespace Wasatch
