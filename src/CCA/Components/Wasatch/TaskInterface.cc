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
#include <CCA/Components/Wasatch/Expressions/Pressure.h>

#include <stdexcept>

using std::cout;
using std::endl;

namespace Wasatch{

  class TreeTaskExecute
  {
    typedef Expr::ExpressionTree::TreePtr  TreePtr;

    TreePtr masterTree_;

    typedef std::pair< TreePtr, Uintah::Task* > TreeTaskPair;
    typedef std::map< int, TreeTaskPair > PatchTreeMap;

    Uintah::SchedulerP& scheduler_;
    const Uintah::PatchSet* const patches_;
    const Uintah::MaterialSet* const materials_;

    const bool createUniqueTreePerPatch_;

    const std::string taskName_;        ///< the name of the task
    const PatchInfoMap& patchInfoMap_;  ///< information for each individual patch.
    Expr::FieldManagerList& fml_;       ///< the FieldManagerList for this TaskInterface

    bool hasBeenScheduled_;
    PatchTreeMap patchTreeMap_;

    /** main execution driver - the callback function exposed to Uintah. */
    void execute( const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const );

  public:

    TreeTaskExecute( TreePtr tree,
                     const std::string taskName,
                     Uintah::SchedulerP& scheduler,
                     const Uintah::PatchSet* const patches,
                     const Uintah::MaterialSet* const materials,
                     const PatchInfoMap& info,
                     const bool createUniqueTreePerPatch,
                     Expr::FieldManagerList& fml );

    ~TreeTaskExecute();
    
    void schedule( const std::vector<Expr::Tag>& newDWFields );

  };


  //------------------------------------------------------------------

  TreeTaskExecute::TreeTaskExecute( TreePtr tree,
                                    const std::string taskName,
                                    Uintah::SchedulerP& sched,
                                    const Uintah::PatchSet* const patches,
                                    const Uintah::MaterialSet* const materials,
                                    const PatchInfoMap& info,
                                    const bool createUniqueTreePerPatch,
                                    Expr::FieldManagerList& fml )
    : masterTree_( tree ),
      scheduler_( sched ),
      patches_( patches ),
      materials_( materials ),
      createUniqueTreePerPatch_( createUniqueTreePerPatch ),
      taskName_( taskName ),
      patchInfoMap_( info ),
      fml_( fml )
  {
    hasBeenScheduled_ = false;

    if( createUniqueTreePerPatch_ ){
      for( int ipss=0; ipss!=patches_->size(); ++ipss ){
        const Uintah::PatchSubset* const pss = patches_->getSubset(ipss);
        for( int ip=0; ip<pss->size(); ++ip ){
          const Uintah::Patch* const patch = pss->get(ip);
 //           cout << "Setting up tree '" << taskName_ << "' on patch (" << patch->getID() << ")" << endl;

          TreePtr tree( new Expr::ExpressionTree( masterTree_->get_roots(),
                                                  masterTree_->get_expression_factory(),
                                                  patch->getID(),
                                                  masterTree_->name() ) );
          tree->register_fields( fml_ );
          patchTreeMap_[ patch->getID() ] = make_pair( tree, scinew Uintah::Task( taskName_, this, &TreeTaskExecute::execute ) );
        }
      }
    }
    else{
      masterTree_->register_fields( fml_ );
      patchTreeMap_[ -1 ] = make_pair( masterTree_, scinew Uintah::Task( taskName_, this, &TreeTaskExecute::execute ) );
    }

    std::ostringstream fnam;
    fnam << tree->name() << ".dot";
    std::ofstream fout( fnam.str().c_str() );
    tree->write_tree(fout);
  }

  //------------------------------------------------------------------

  TreeTaskExecute::~TreeTaskExecute()
  {
    // jcs do we need to delete the Uintah::Task?
    //     for( PatchTreeMap::iterator i=patchTreeMap_.begin(); i!=patchTreeMap_.end(); ++i ){
    //       delete i->second.first;
    //     }
  }

  //------------------------------------------------------------------

  void
  add_fields_to_task( Uintah::Task& task,
                      const Expr::ExpressionTree& tree,
                      Expr::FieldManagerList& fml,
                      const Uintah::PatchSubset* const patches,
                      const Uintah::MaterialSubset* const materials,
                      const std::vector<Expr::Tag>& newDWFields )
  {
    // this is done once when the task is scheduled.  The purpose of
    // this method is to collect the fields from the ExpressionTree
    // and then advertise their usage to Uintah.

    //
    // root nodes in the dependency tree are always "COMPUTES" fields
    //
    // bottom nodes in the dependency tree are either "COMPUTES" or "REQUIRES" fields
    //
    // intermediate nodes in the tree can be either scratch fields or
    //    COMPUTES fields, depending on whether we want them output or
    //    not.  Currently, we don't have any scratch fields.  Not sure
    //    where those would be added.
    //

    //______________________________
    // cycle through each field type
    for( Expr::FieldManagerList::iterator ifm=fml.begin(); ifm!=fml.end(); ++ifm ){

      typedef Expr::FieldManagerBase::PropertyMap PropertyMap;
      PropertyMap& properties = (*ifm)->properties();
      PropertyMap::iterator ipm = properties.find("UintahInfo");
      assert( ipm != properties.end() );
      Expr::IDInfoMap& infomap = boost::any_cast< boost::reference_wrapper<Expr::IDInfoMap> >(ipm->second);

      //______________________________________
      // cycle through each field of this type
      for( Expr::IDInfoMap::iterator ii=infomap.begin(); ii!=infomap.end(); ++ii ){

        Expr::FieldInfo& fieldInfo = ii->second;

        //________________
        // set field mode 
        {
          const Expr::Tag fieldTag(fieldInfo.varlabel->getName(), fieldInfo.context );

          if( tree.has_expression( fieldTag ) ){
            if( tree.get_expression(fieldTag).is_placeholder() ){
              fieldInfo.mode = Expr::REQUIRES;
              if( find( newDWFields.begin(), newDWFields.end(), fieldTag ) == newDWFields.end() )
                fieldInfo.useOldDataWarehouse = true;
            }
            else
              fieldInfo.mode = Expr::COMPUTES;
          }
          else{
            fieldInfo.mode = Expr::REQUIRES;
          }
        }

        // jcs : old dw is (should be) read only.
        Uintah::Task::WhichDW dw = Uintah::Task::NewDW;
        if( fieldInfo.useOldDataWarehouse ) dw = Uintah::Task::OldDW;

//         cout << "Task '" << tree.name() << "' "; // jcs diagnostic
        switch( fieldInfo.mode ){

        case Expr::COMPUTES:
//           cout << "COMPUTES";  // jcs diagnostic
          ASSERT( dw == Uintah::Task::NewDW );
          // jcs note that we need ghost information on the computes fields as well!
          task.computes( fieldInfo.varlabel,
                         patches, Uintah::Task::NormalDomain,
                         materials, Uintah::Task::NormalDomain );
          break;

        case Expr::REQUIRES:
//           cout << "REQUIRES";  // jcs diagnostic
          task.requires( dw,
                         fieldInfo.varlabel,
                         patches, Uintah::Task::NormalDomain,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost );
          break;

        case Expr::MODIFIES:
//           cout << "MODIFIES"; // jcs diagnostic
          ASSERT( dw == Uintah::Task::NewDW );
          task.modifies( fieldInfo.varlabel,
                         patches, Uintah::Task::NormalDomain,
                         materials, Uintah::Task::NormalDomain );
          break;
        } // switch
        
        //==================== <diagnostics> ====================
//         cout << " '"  << fieldInfo.varlabel->getName() << "' ("
//              << fieldInfo.context << ") in ";
//         if( fieldInfo.useOldDataWarehouse ) cout << "OLD";
//         else cout << "NEW";
//         cout << " data warehouse"
//              << " with " << fieldInfo.nghost << " ghosts on patches "
//              << *patches
//              << endl;
        //==================== </diagnostics> ====================


      } // field loop
    } // field type loop
  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::schedule( const std::vector<Expr::Tag>& newDWFields )
  {
    ASSERT( !hasBeenScheduled_ );

    const PatchTreeMap::iterator iptm = patchTreeMap_.begin();
    ASSERT( iptm != patchTreeMap_.end() );

    Uintah::Task* const task = iptm->second.second;
    TreePtr tree = iptm->second.first;

    const Uintah::MaterialSubset* const mss = materials_->getUnion();

    const Uintah::PatchSubset* const pss = patches_->getUnion();
    add_fields_to_task( *task, *tree, fml_, pss, mss, newDWFields );
    // jcs eachPatch vs. allPatches (gang schedule vs. independent...)
    scheduler_->addTask( task, patches_, materials_ );

    hasBeenScheduled_ = true;
  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::execute( const Uintah::ProcessorGroup* const pg,
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

      // resolve the tree
      TreePtr tree;
      if( createUniqueTreePerPatch_ ){
        PatchTreeMap::iterator iptm = patchTreeMap_.find( patch->getID() );
        ASSERT( iptm != patchTreeMap_.end() );
        tree = iptm->second.first;
      }
      else{
        tree = patchTreeMap_[ -1 ].first;
      }

      for( int im=0; im<materials->size(); ++im ){

        const int material = materials->get(im);
        try{
//           cout << endl
//                << "Wasatch: executing graph '" << taskName_
//                << "' for patch " << patch->getID()
//                << " and material " << material
//                << endl;

//     fml_->dump_fields(cout);
          fml_.allocate_fields( Expr::AllocInfo( oldDW, newDW, material, patch, pg ) );

          tree->bind_fields( fml_ );
          tree->bind_operators( opdb );          
          tree->execute_tree();
          // get pressure expression and build coefficient matrix for pressure
          //const Expr::Tag pressuret("pressure", Expr::STATE_NONE );
          //if( tree->has_expression( pressuret ) ){
          //  Pressure& pressureExpr = dynamic_cast<Pressure&> (tree->get_expression(pressuret));
          //  pressureExpr.bind_uintah_vars(newDW, patch, material);
          //}
          //
//           cout << "Wasatch: done executing graph '" << taskName_ << "'" << endl;
          fml_.deallocate_fields();
        }
        catch( exception& e ){
          cout << e.what() << endl;
          throw std::runtime_error( "Error" );
        }
      }
    }
  }







  //------------------------------------------------------------------

  TaskInterface::TaskInterface( const Expr::ExpressionID& root,
                                const std::string taskName,
                                Expr::ExpressionFactory& factory,
                                Uintah::SchedulerP& sched,
                                const Uintah::PatchSet* const patches,
                                const Uintah::MaterialSet* const materials,
                                const PatchInfoMap& info,
                                const bool createUniqueTreePerPatch,
                                Expr::FieldManagerList* fml )
    : builtFML_( fml==NULL ),
      fml_( builtFML_ ? scinew Expr::FieldManagerList( taskName ) : fml )
  {
    typedef Expr::ExpressionTree::TreeList TreeList;
    Expr::ExpressionTree::TreePtr tree( new Expr::ExpressionTree( root, factory, -1, taskName ) );
    TreeList treeList = tree->split_tree();
    for( TreeList::iterator itr=treeList.begin(); itr!=treeList.end(); ++itr ){
      cout << "creating task executor for tree " << (*itr)->name() << endl;
      execList_.push_back( new TreeTaskExecute( *itr, taskName, sched, patches, materials, info, createUniqueTreePerPatch, *fml_ ) );
    }
  }

  //------------------------------------------------------------------

  TaskInterface::TaskInterface( const IDSet& roots,
                                const std::string taskName,
                                Expr::ExpressionFactory& factory,
                                Uintah::SchedulerP& sched,
                                const Uintah::PatchSet* patches,
                                const Uintah::MaterialSet* const materials,
                                const PatchInfoMap& info,
                                const bool createUniqueTreePerPatch,
                                Expr::FieldManagerList* fml )
    : builtFML_( fml==NULL ),
      fml_( builtFML_ ? scinew Expr::FieldManagerList( taskName ) : fml )
  {
    typedef Expr::ExpressionTree::TreeList TreeList;
    Expr::ExpressionTree::TreePtr tree( new Expr::ExpressionTree( roots, factory, -1, taskName ) );
    TreeList treeList = tree->split_tree();
    for( TreeList::iterator itr=treeList.begin(); itr!=treeList.end(); ++itr ){
      cout << "creating task executor for tree " << (*itr)->name() << endl;
      execList_.push_back( new TreeTaskExecute( *itr, taskName, sched, patches, materials, info, createUniqueTreePerPatch, *fml_ ) );
    }
  }

  //------------------------------------------------------------------

  TaskInterface::~TaskInterface()
  {
    // tasks are deleted by the scheduler that they are assigned to.
    // This means that we don't need to delete uintahTask_
    if( builtFML_ ) delete fml_;

    for( ExecList::iterator iex=execList_.begin(); iex!=execList_.end(); ++iex ){
      delete *iex;
    }
  }

  //------------------------------------------------------------------

  void
  TaskInterface::schedule( const std::vector<Expr::Tag>& newDWFields )
  {
    for( ExecList::iterator iex=execList_.begin(); iex!=execList_.end(); ++iex ){
      (*iex)->schedule( newDWFields );
    }
  }

  //------------------------------------------------------------------

  void
  TaskInterface::schedule()
  {
    std::vector<Expr::Tag> newDWFields;
    this->schedule( newDWFields );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
