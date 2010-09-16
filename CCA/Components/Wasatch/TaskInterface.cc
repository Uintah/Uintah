//-- SpatialOps library includes --//
#include <spatialops/OperatorDatabase.h>


//-- expression library includes --//
#include <expression/ExprLib.h>


//-- Uintah includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ComputeSet.h>


//-- Wasatch includes --//
#include "TaskInterface.h"


#include <stdexcept>

using std::cout;
using std::endl;

namespace Wasatch{

  //------------------------------------------------------------------

  TaskInterface::TaskInterface( Expr::ExpressionTree* tree,
                                const PatchInfoMap& info,
                                Expr::FieldManagerList* fml )
    : tree_( tree ),
      taskName_( tree->name() ),
      patchInfoMap_( info ),

      builtFML_( fml==NULL ),
      fml_( builtFML_ ? new Expr::FieldManagerList( tree->name() ) : fml ),
      uintahTask_( new Uintah::Task( tree->name(), this, &TaskInterface::execute ) )
  {
    hasBeenScheduled_ = false;
    tree->register_fields( *fml_ );
  }

  //------------------------------------------------------------------

  TaskInterface::~TaskInterface()
  {
    // tasks are deleted by the scheduler that they are assigned to.
    // This means that we don't need to delete uintahTask_
    if( builtFML_ ) delete fml_;
    delete tree_;
  }

  //------------------------------------------------------------------

  void
  TaskInterface::schedule( Uintah::SchedulerP& scheduler,
                           const Uintah::PatchSet* const patches,
                           const Uintah::MaterialSet* const materials )
  {
    add_fields_to_task( patches, materials );
    scheduler->addTask( uintahTask_, patches, materials );
    hasBeenScheduled_ = true;
  }

  //------------------------------------------------------------------

  void
  TaskInterface::add_fields_to_task( const Uintah::PatchSet* const patches,
                                     const Uintah::MaterialSet* const materials )
  {
    // this is done once when the task is scheduled.  The purpose of
    // this method is to collect the fields from the ExpressionTree
    // and then advertise their usage to Uintah.

    ASSERT( !hasBeenScheduled_ );

    // for now we will assume that we are computing things on ALL patches and ALL materials
    const Uintah::PatchSubset*    const pss = patches->getUnion();
    const Uintah::MaterialSubset* const mss = materials->getUnion();

    //
    // root nodes in the dependency tree are always "COMPUTES" fields
    //
    // bottom nodes in the dependency tree are either "COMPUTES" or "REQUIRES" fields
    //
    // intermediate nodes in the tree can be either scratch fields or
    //              COMPUTES fields, depending on whether we want them
    //              output or not.  Currently, we don't have any
    //              scratch fields.  Not sure where those would be
    //              added.
    //

    //______________________________
    // cycle through each field type
    for( Expr::FieldManagerList::iterator ifm=fml_->begin(); ifm!=fml_->end(); ++ifm ){

      typedef Expr::FieldManagerBase::PropertyMap PropertyMap;
      PropertyMap& properties = (*ifm)->properties();
      PropertyMap::iterator ipm = properties.find("UintahInfo");
      assert( ipm != properties.end() );
      Expr::IDInfoMap& infomap = boost::any_cast< boost::reference_wrapper<Expr::IDInfoMap> >(ipm->second);

      //______________________________________
      // cycle through each field of this type
      for( Expr::IDInfoMap::iterator ii=infomap.begin(); ii!=infomap.end(); ++ii ){

        Expr::FieldInfo& fieldInfo = ii->second;

        // set field mode 
        // 
        // jcs this is a hack.  On initialization, it takes a STATE_N
        // to a new DW and on timestepping it does the same for the
        // solution variable because we must wrap it with an
        // expression.  Need to figure out a more robust way of
        // handling this.
        {
          const Expr::Tag fieldTag(fieldInfo.varlabel->getName(), fieldInfo.context );

          if( tree_->has_expression( fieldTag ) )
            fieldInfo.mode = Expr::COMPUTES;
          else
            fieldInfo.mode = Expr::REQUIRES;
        }

        // jcs : old dw is (should be) read only.
        Uintah::Task::WhichDW dw = Uintah::Task::NewDW;
        if( fieldInfo.useOldDataWarehouse ) dw = Uintah::Task::OldDW;

        cout << "Task '" << taskName_ << "' "; // jcs diagnostic
        switch( fieldInfo.mode ){

        case Expr::COMPUTES:
          cout << "COMPUTES";  // jcs diagnostic
          ASSERT( dw == Uintah::Task::NewDW );
          // jcs note that we need ghost information on the computes fields as well!
          uintahTask_->computes( fieldInfo.varlabel,
                                 pss, Uintah::Task::NormalDomain,
                                 mss, Uintah::Task::NormalDomain );
          break;

        case Expr::REQUIRES:
          cout << "REQUIRES";  // jcs diagnostic
          uintahTask_->requires( dw,
                                 fieldInfo.varlabel,
                                 pss, Uintah::Task::NormalDomain,
                                 mss, Uintah::Task::NormalDomain,
                                 fieldInfo.ghostType, fieldInfo.nghost );
          break;

        case Expr::MODIFIES:
          cout << "MODIFIES"; // jcs diagnostic
          ASSERT( dw == Uintah::Task::NewDW );
          uintahTask_->modifies( fieldInfo.varlabel,
                                 pss, Uintah::Task::NormalDomain,
                                 mss, Uintah::Task::NormalDomain );
          break;
        } // switch
        
        //==================== <diagnostics> ====================
        cout << " '"  << fieldInfo.varlabel->getName() << "' ("
             << fieldInfo.context << ") in ";
        if( fieldInfo.useOldDataWarehouse ) cout << "OLD";
        else cout << "NEW";
        cout << " data warehouse"
             << " with " << fieldInfo.nghost << " ghosts"
             << endl;
        //==================== </diagnostics> ====================


      } // field loop
    } // field type loop
  }

  //------------------------------------------------------------------

  void
  TaskInterface::bind_fields_operators( const Expr::AllocInfo& info,
                                        const SpatialOps::OperatorDatabase& opDB )
  {
    // wrap fields from Uintah in the strongly typed field managers
    // and then bind them to the tree.  Then bind all operators
    // associated with this patch and task
    //
    // This is called each time a task is executed.
    fml_->allocate_fields( info );
    tree_->bind_fields( *fml_ );
    tree_->bind_operators( opDB );
  }

  //------------------------------------------------------------------

  void
  TaskInterface::execute( const Uintah::ProcessorGroup* const pg,
                          const Uintah::PatchSubset* const patches,
                          const Uintah::MaterialSubset* const materials,
                          Uintah::DataWarehouse* const oldDW,
                          Uintah::DataWarehouse* const newDW )
  {
    //
    // execute on each patch
    //
    // NOTE: in principle this patch loop could be done in parallel
    //       IFF the expression tree was generated for each patch.
    //       Otherwise we would have binding clashes between different
    //       threads.
    //
    for( int ip=0; ip<patches->size(); ++ip ){

      const Uintah::Patch* const patch = patches->get(ip);
      const PatchInfoMap::const_iterator ipim = patchInfoMap_.find(patch->getID());
      ASSERT( ipim!=patchInfoMap_.end() );
      const SpatialOps::OperatorDatabase& opdb = *ipim->second.operators;

      for( int im=0; im<materials->size(); ++im ){

        const int material = materials->get(im);
        try{
          cout << endl
               << "Wasatch: executing '" << taskName_
               << "' for patch " << patch->getID()
               << " and material " << material
               << endl;
          bind_fields_operators( Expr::AllocInfo( oldDW, newDW, material, patch ),
                                 opdb );
          tree_->execute_tree();
        }
        catch( exception& e ){
          cout << e.what() << endl;
          throw std::runtime_error( "Error" );
        }
      }
    }
  }

  //------------------------------------------------------------------

} // namespace Wasatch
