/*
 * Copyright (c) 2012 The University of Utah
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

//-- SpatialOps library includes --//
#include <spatialops/OperatorDatabase.h>


//-- expression library includes --//
#include <expression/ExprLib.h>
#include <expression/ExpressionTree.h>


//-- Uintah includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Level.h>
#include <Core/Parallel/Parallel.h>


//-- Wasatch includes --//
#include "TaskInterface.h"
#include <CCA/Components/Wasatch/Expressions/Pressure.h>


#include <stdexcept>
#include <fstream>

using std::endl;

//#define WASATCH_TASK_DIAGNOSTICS
//#define WASATCH_TASK_FIELD_DIAGNOSTICS

namespace Wasatch{

  /**
   *  \class TreeTaskExecute
   *  \ingroup WasatchGraph
   *  \author James C. Sutherland
   *  \brief Handles execution of a Expr::ExpressionTree object on a set of patches.
   */
  class TreeTaskExecute
  {
  public:
    typedef Expr::ExpressionTree::TreePtr  TreePtr;

  private:
    TreePtr masterTree_;

    typedef std::pair< TreePtr, Uintah::Task* > TreeTaskPair;
    typedef std::map< int, TreeTaskPair > PatchTreeMap;

    Uintah::SchedulerP& scheduler_;
    const Uintah::PatchSet* const patches_;
    const Uintah::MaterialSet* const materials_;

    const bool createUniqueTreePerPatch_;

    const std::string taskName_;        ///< the name of the task
    const PatchInfoMap& patchInfoMap_;  ///< information for each individual patch.
    Expr::FieldManagerList* const fml_; ///< the FieldManagerList for this TaskInterface

    const bool hasPressureExpression_;

    bool hasBeenScheduled_;
    PatchTreeMap patchTreeMap_;

    /** \brief main execution driver - the callback function exposed to Uintah. */
    void execute( const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const,
                  const int rkStage );

  public:

    /**
     *  \brief Construct a TreeTaskExecute object.
     *  \param tree - the TreePtr that this object is associated with
     *  \param taskName - the name of this task
     *  \param scheduler - the scheduler that this task is associated with
     *  \param patches 	- the list of patches that this TreeTaskExecute object is to be executed on.
     *  \param materials - the list of materials that this task is to be associated with.
     *  \param info	- the PatchInfoMap object that holds patch-specific information (like operators).
     *  \param createUniqueTreePerPatch - if true, then a unique tree will be created per patch (recommended).
     */
    TreeTaskExecute( TreePtr tree,
                     const std::string taskName,
                     const Uintah::LevelP& level,
                     Uintah::SchedulerP& scheduler,
                     const Uintah::PatchSet* const patches,
                     const Uintah::MaterialSet* const materials,
                     const PatchInfoMap& info,
                     const bool createUniqueTreePerPatch,
                     const int rkStage,
                     const std::set<std::string>& ioFieldSet );

    ~TreeTaskExecute();

    void schedule( Expr::TagSet newDWFields, const int rkStage );

    PatchTreeMap get_patch_tree_map() {return patchTreeMap_;}

    TreePtr get_tree() const{ return masterTree_; }

  };

  //------------------------------------------------------------------

  TreeTaskExecute::TreeTaskExecute( TreePtr tree,
                                    const std::string taskName,
                                    const Uintah::LevelP& level,
                                    Uintah::SchedulerP& sched,
                                    const Uintah::PatchSet* const patches,
                                    const Uintah::MaterialSet* const materials,
                                    const PatchInfoMap& info,
                                    const bool createUniqueTreePerPatch,
                                    const int rkStage,
                                    const std::set<std::string>& ioFieldSet)
    : masterTree_( tree ),
      scheduler_( sched ),
      patches_( patches ),
      materials_( materials ),
      createUniqueTreePerPatch_( createUniqueTreePerPatch ),
      taskName_( taskName ),
      patchInfoMap_( info ),
      fml_( scinew Expr::FieldManagerList(taskName) ),
      hasPressureExpression_( tree->computes_field( pressure_tag() ) )
  {
    hasBeenScheduled_ = false;
    if( Uintah::Parallel::getMPIRank() == 0 ){
      std::ostringstream fnam;
      fnam << tree->name() << ".dot";
      std::ofstream fout( fnam.str().c_str() );
      tree->write_tree(fout);
    }

    if( createUniqueTreePerPatch_ ){

      // only set up trees on the patches that we own on this process.
      const Uintah::PatchSet*  perproc_patchset = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
      const Uintah::PatchSubset* const localPatches = perproc_patchset->getSubset(Uintah::Parallel::getMPIRank());

      Uintah::Task* tsk = scinew Uintah::Task( taskName_, this, &TreeTaskExecute::execute, rkStage );

      for( int ip=0; ip<localPatches->size(); ++ip ){
        const Uintah::Patch* const patch = localPatches->get(ip);
        TreePtr tree( new Expr::ExpressionTree( *masterTree_ ) );
        tree->set_patch_id( patch->getID() );

        std::set<std::string>::iterator ioFieldSetIter = ioFieldSet.begin();

        while (ioFieldSetIter != ioFieldSet.end()) {
          const Expr::Tag fieldStateN   (*ioFieldSetIter, Expr::STATE_N   );
          const Expr::Tag fieldStateNONE(*ioFieldSetIter, Expr::STATE_NONE);

          if (tree->has_field(fieldStateN)) {
            if (tree->has_expression(tree->get_id(fieldStateN))) {
              tree->set_expr_is_persistent( fieldStateN, true, *fml_);
            }
          }

          if (tree->has_field(fieldStateNONE)) {
            if (tree->has_expression(tree->get_id(fieldStateNONE))) {
              tree->set_expr_is_persistent( fieldStateNONE, true, *fml_);
            }
          }

          ioFieldSetIter++;
        }

        tree->register_fields( *fml_ );
        patchTreeMap_[ patch->getID() ] = std::make_pair( tree, tsk );
      }
    }
    else{
      masterTree_->register_fields( *fml_ );
      patchTreeMap_[ -1 ] = std::make_pair( masterTree_, scinew Uintah::Task( taskName_, this, &TreeTaskExecute::execute, rkStage ) );
    }
  }

  //------------------------------------------------------------------

  TreeTaskExecute::~TreeTaskExecute()
  {
    delete fml_;
//     // jcs do we need to delete the Uintah::Task?
//         for( PatchTreeMap::iterator i=patchTreeMap_.begin(); i!=patchTreeMap_.end(); ++i ){
//           delete i->second.second;
//         }
  }

  //------------------------------------------------------------------

  /**
   *  \ingroup WasatchGraph
   *  \brief adds requisite fields to the given task.
   *  \param task - the task to add fields to
   *  \param tree - the ExpressionTree that is being wrapped as a task
   *  \param fml  - the FieldManagerList that manages the fields associated with this ExpressionTree and task.
   *  \param patches - the patches to associate with this task
   *  \param materials - the materials to associate with this task
   *  \param newDWFields - any fields specified in this TagSet will be taken from the new DataWarehouse instead of the old DataWarehouse.
   *
   *  This function analyzes the ExpressionTree to identify what
   *  fields are required for this task, and then advertises them to
   *  Uintah.  The COMPUTES/REQUIRES is automatically deduced.
   */
  void
  add_fields_to_task( Uintah::Task& task,
                      const Expr::ExpressionTree& tree,
                      Expr::FieldManagerList& fml,
                      const Uintah::PatchSubset* const patches,
                      const Uintah::MaterialSubset* const materials,
                      const Expr::TagSet& newDWFields,
                      const int rkStage )
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

#   ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
    proc0cout << "Field requirements for task '" << tree.name() << "'" << endl
              << std::setw(10) << "Mode " << std::left << std::setw(20) << "Field Name"
              << "DW  #Ghost PatchID" << endl
              << "-----------------------------------------------------------------------" << endl;
#   endif

    //______________________________
    // cycle through each field type
    for( Expr::FieldManagerList::iterator ifm=fml.begin(); ifm!=fml.end(); ++ifm ){

      typedef Expr::FieldManagerBase::PropertyMap PropertyMap;
      PropertyMap& properties = ifm->second->properties();
      PropertyMap::iterator ipm = properties.find("UintahInfo");
      assert( ipm != properties.end() );
      Expr::IDInfoMap& infomap = boost::any_cast< boost::reference_wrapper<Expr::IDInfoMap> >(ipm->second);

      //______________________________________
      // cycle through each field of this type
      for( Expr::IDInfoMap::iterator ii=infomap.begin(); ii!=infomap.end(); ++ii ){

        Expr::FieldInfo& fieldInfo = ii->second;
        const Expr::Tag& fieldTag = ii->first;

        // see if this field is required by the given tree
        if( !tree.has_field(fieldTag) ){
          continue;
        }

        // Use the old DW on the first RK stage.  Thereafter,
        // we modify the values already in the new DW.
        fieldInfo.useOldDataWarehouse = (rkStage < 1);

        if( fieldTag.context() == Expr::CARRY_FORWARD ){
          fieldInfo.mode = Expr::COMPUTES;
          fieldInfo.useOldDataWarehouse = false;
          task.requires( Uintah::Task::OldDW,
                         fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost );

#         ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
          proc0cout << std::setw(10) << "(REQUIRES)"
                    << std::setw(20) << std::left << fieldInfo.varlabel->getName()
                    << "OLD   "
                    << std::left << std::setw(5) << fieldInfo.nghost
                    << *patches << endl;
#         endif
        }

        //________________
        // set field mode
        if( tree.computes_field( fieldTag ) ){

          // if the field uses dynamic allocation, then the uintah task should not be aware of this field
          // jcs the const_cast is a hack because of the lack of const on the is_persistent method...
          if( ! tree.is_persistent(fieldTag) ){
            continue;
          }

          fieldInfo.mode = (rkStage==1) ? Expr::COMPUTES : Expr::MODIFIES;
        }
        else if( fieldTag.context() == Expr::STATE_N ){
          fieldInfo.mode = Expr::REQUIRES;
          fieldInfo.useOldDataWarehouse = (rkStage < 2);
        }
        else{
          fieldInfo.mode = Expr::REQUIRES;
          if( newDWFields.find( fieldTag ) != newDWFields.end() )
            fieldInfo.useOldDataWarehouse = false;
        }
        if( tree.name()!="set_time" &&
            tree.name()!="initialization" &&
            (fieldInfo.varlabel->getName()=="time" ||
             fieldInfo.varlabel->getName()=="timestep")){
          fieldInfo.mode = Expr::REQUIRES;
        }

        const Uintah::Task::WhichDW dw = ( fieldInfo.useOldDataWarehouse ) ? Uintah::Task::OldDW : Uintah::Task::NewDW;

        switch( fieldInfo.mode ){

        case Expr::COMPUTES:
#         ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
          proc0cout << std::setw(10) << "COMPUTES";
#         endif
          ASSERT( dw == Uintah::Task::NewDW );
          task.computes( fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain );
          break;

        case Expr::REQUIRES:
#         ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
          proc0cout << std::setw(10) << "REQUIRES";
#         endif
          task.requires( dw,
                         fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost );
          break;

        case Expr::MODIFIES:
#         ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
          proc0cout << std::setw(10) << "MODIFIES";
#         endif
          ASSERT( dw == Uintah::Task::NewDW );
          // jcs it appears that we need to set a "requires" so that
          // the proper ghost inforation is incoporated since
          // "modifies" does not allow us to do that.
          task.requires( dw, fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost );
          task.modifies( fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain );
          break;

        } // switch

#       ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
        proc0cout << std::setw(20) << std::left << fieldInfo.varlabel->getName();
        if( fieldInfo.useOldDataWarehouse ){ proc0cout << "OLD   "; }
        else{ proc0cout << "NEW   "; }
        proc0cout << std::left << std::setw(5) << fieldInfo.nghost
                  << *patches << endl;
#       endif

      } // field loop
    } // field type loop

#   ifdef WASATCH_TASK_FIELD_DIAGNOSTICS
    proc0cout << endl;
#   endif

  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::schedule( Expr::TagSet newDWFields, const int rkStage )
  {
    ASSERT( !hasBeenScheduled_ );

#   ifdef WASATCH_TASK_DIAGNOSTICS
    proc0cout << "Scheduling task '" << taskName_ << "'" << endl;
#   endif

    const PatchTreeMap::iterator iptm = patchTreeMap_.begin();
    ASSERT( iptm != patchTreeMap_.end() );

    Uintah::Task* const task = iptm->second.second;
    TreePtr tree = iptm->second.first;

    const Uintah::MaterialSubset* const mss = materials_->getUnion();
    const Uintah::PatchSubset* const pss = patches_->getUnion();

    // augment newDWFields to include any fields that are a result of tree cleaving
    {
      for( Expr::ExpressionTree::ExprFieldMap::const_iterator imp=tree->field_map().begin(); imp!=tree->field_map().end(); ++imp ){
        const Expr::FieldDeps::FldHelpers& fh = imp->second->field_helpers();
        for( Expr::FieldDeps::FldHelpers::const_iterator ifld=fh.begin(); ifld!=fh.end(); ++ifld ){
          const Expr::FieldDeps::FieldHelperBase& fhb = **ifld;
          const Expr::Tag& tag = fhb.tag();
          if( tree->get_expression( tag ).is_placeholder() ) continue;
          newDWFields.insert( tag );
        }
      }
    }

    add_fields_to_task( *task, *tree, *fml_, pss, mss, newDWFields, rkStage );

    // jcs eachPatch vs. allPatches (gang schedule vs. independent...)
    scheduler_->addTask( task, patches_, materials_ );

    if( hasPressureExpression_ ){
      Pressure& pexpr = dynamic_cast<Pressure&>( tree->get_expression( pressure_tag() ) );
      pexpr.schedule_solver( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );
      pexpr.declare_uintah_vars( *task, pss, mss, rkStage );
      pexpr.schedule_set_pressure_bcs( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );            
    }

    hasBeenScheduled_ = true;
  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::execute( const Uintah::ProcessorGroup* const pg,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            Uintah::DataWarehouse* const oldDW,
                            Uintah::DataWarehouse* const newDW,
                            const int rkStage )
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
#         ifdef WASATCH_TASK_DIAGNOSTICS
          proc0cout << endl
              << "Wasatch: executing graph '" << taskName_
              << "' for patch " << patch->getID()
              << " and material " << material
              << endl;
          if( Uintah::Parallel::getMPIRank() == 0 ){
            fml_->dump_fields(std::cout);
          }
#         endif
          fml_->allocate_fields( Expr::AllocInfo( oldDW, newDW, material, patch, pg ) );

          if( hasPressureExpression_ ){
            Pressure& pexpr = dynamic_cast<Pressure&>( tree->get_expression( pressure_tag() ) );
            pexpr.set_patch(patches->get(ip));
            pexpr.bind_uintah_vars( newDW, patch, material, rkStage );
          }

          tree->bind_fields( *fml_ );
          tree->bind_operators( opdb );
          tree->execute_tree();
#         ifdef WASATCH_TASK_DIAGNOSTICS
          proc0cout << "Wasatch: done executing graph '" << taskName_ << "'" << endl;
#         endif
          fml_->deallocate_fields();
        }
        catch( std::exception& e ){
          proc0cout << e.what() << endl;
          throw std::runtime_error( "Error" );
        }
      }
    }
  }


  //------------------------------------------------------------------

  TaskInterface::TaskInterface( const IDSet& roots,
                                const std::string taskName,
                                Expr::ExpressionFactory& factory,
                                const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched,
                                const Uintah::PatchSet* patches,
                                const Uintah::MaterialSet* const materials,
                                const PatchInfoMap& info,
                                const bool createUniqueTreePerPatch,
                                const int rkStage,
                                const std::set<std::string>& ioFieldSet,
                                Expr::FieldManagerList* fml )
    : builtFML_( fml==NULL ),
      fml_( builtFML_ ? scinew Expr::FieldManagerList( taskName ) : fml )
  {
    typedef Expr::ExpressionTree::TreeList TreeList;
    Expr::ExpressionTree::TreePtr tree( new Expr::ExpressionTree( roots, factory, -1, taskName ) );
    TreeList treeList = tree->split_tree();
    if( Uintah::Parallel::getMPIRank() == 0 ){
      if( treeList.size() > 1 ){
        std::ostringstream fnam;
        fnam << tree->name() << "_original.dot";
        proc0cout << "writing pre-cleave tree to " << fnam.str() << endl;
        std::ofstream fout( fnam.str().c_str() );
        tree->write_tree(fout);
      }
    }
    for( TreeList::iterator itr=treeList.begin(); itr!=treeList.end(); ++itr ){
      Expr::ExpressionTree::TreePtr tr = *itr;
      execList_.push_back( new TreeTaskExecute( tr, tr->name(), level, sched, patches, materials, info, createUniqueTreePerPatch, rkStage, ioFieldSet ) );
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
  TaskInterface::schedule( const Expr::TagSet& newDWFields, const int rkStage )
  {
    for( ExecList::iterator iex=execList_.begin(); iex!=execList_.end(); ++iex ){
      (*iex)->schedule( newDWFields, rkStage );
    }
  }

  //------------------------------------------------------------------

  void
  TaskInterface::schedule( const int rkStage )
  {
    Expr::TagSet newDWFields;
    this->schedule( newDWFields, rkStage );
  }

  //------------------------------------------------------------------

  Expr::ExpressionTree::TreePtr
  TaskInterface::get_time_tree()
  {
    typedef Expr::ExpressionTree::TreePtr TreePtr;
    typedef std::pair< TreePtr, Uintah::Task* > TreeTaskPair;
    typedef std::map< int, TreeTaskPair > PatchTreeMap;
    for( ExecList::iterator iex=execList_.begin(); iex!=execList_.end(); ++iex ){
      Wasatch::TreeTaskExecute* taskexec = *iex;
      PatchTreeMap ptmap= taskexec->get_patch_tree_map();
      const PatchTreeMap::iterator iptm = ptmap.begin();
      TreePtr tree = iptm->second.first;
      if (tree->name()=="set_time") {
        return tree;
      }
    }
    throw std::runtime_error( "TaskInterface::get_time_tree() could not resolve a valid tree");
  }

  //------------------------------------------------------------------

  Expr::TagList
  TaskInterface::collect_tags_in_task() const
  {
    Expr::TagList tags;
    for( ExecList::const_iterator itte=execList_.begin(); itte!=execList_.end(); ++itte ){
      const Expr::ExpressionTree::TreePtr tree = (*itte)->get_tree();
      const Expr::ExpressionTree::ExprFieldMap& fieldMap = tree->field_map();
      for( Expr::ExpressionTree::ExprFieldMap::const_iterator ifm=fieldMap.begin(); ifm!=fieldMap.end(); ++ifm ){
        const Expr::TagList tl = tree->get_expression_factory().get_labels( ifm->first );
        tags.insert( tags.end(), tl.begin(), tl.end() );
      }
    }
    return tags;
  }

  //------------------------------------------------------------------

} // namespace Wasatch
