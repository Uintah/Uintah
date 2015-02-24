/*
 * The MIT License
 *
 * Copyright (c) 2012-2014 The University of Utah
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

#include <boost/foreach.hpp>

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
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <sci_defs/cuda_defs.h>

//-- Wasatch includes --//
#include "TaskInterface.h"
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>
#include <CCA/Components/Wasatch/Expressions/Coordinate.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/RadiationSource.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>
#include <CCA/Components/Wasatch/CoordinateHelper.h>
#include <CCA/Components/Wasatch/OldVariable.h>

#include <stdexcept>
#include <fstream>

// getting ready for debugstream
#include <Core/Util/DebugStream.h>

using std::endl;

typedef Expr::ExpressionTree::TreeList TreeList;
typedef Expr::ExpressionTree::TreePtr  TreePtr;
typedef std::map<int,TreePtr> TreeMap;

/*
 usage:
 To enable a debug stream, use the + identifier
  tcsh: setenv SCI_DEBUG WASATCH_TASKS:+
  bash: export SCI_DEBUG=WASATCH_TASKS:+

 To disable a debug stream, use the - identifier
  tcsh: setenv SCI_DEBUG WASATCH_TASKS:-
  bash: export SCI_DEBUG=WASATCH_TASKS:-

 To enable multiple debug flags, use a comma to separate them
  tcsh: setenv SCI_DEBUG WASATCH_TASKS:+, WASATCH_FIELDS:+
  bash: export SCI_DEBUG=WASATCH_TASKS:+, WASATCH_FIELDS:+
 
 To enable one flag and disable another that was previously enabled, either
 define a new flag excluding the unwanted flag, or redefine SCI_DEBUG with a -
 after the unwanted flag
   tcsh: setenv SCI_DEBUG WASATCH_TASKS:-, WASATCH_FIELDS:+
   bash: export SCI_DEBUG=WASATCH_TASKS:-, WASATCH_FIELDS:+
 
 */

static SCIRun::DebugStream dbgt("WASATCH_TASKS", false);  // task diagnostics
static SCIRun::DebugStream dbgf("WASATCH_FIELDS", false); // field diagnostics
#define dbg_tasks_on  dbgt.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_fields_on dbgf.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_tasks  if( dbg_tasks_on  ) dbgt
#define dbg_fields if( dbg_fields_on ) dbgf

namespace Wasatch{

  /**
   *  \class TreeTaskExecute
   *  \ingroup WasatchGraph
   *  \author James C. Sutherland
   *  \brief Handles execution of a Expr::ExpressionTree object on a set of patches.
   */
  class TreeTaskExecute
  {
    struct Info{
      SpatialOps::OperatorDatabase* operators;
      Uintah::Task* task;
      TreePtr tree;
    };
    typedef std::map< int, Info > PatchTreeTaskMap;

    Uintah::SchedulerP& scheduler_;
    const Uintah::PatchSet* const patches_;
    const Uintah::MaterialSet* const materials_;

    const std::string taskName_;        ///< the name of the task
    Expr::FieldManagerList* const fml_; ///< the FieldManagerList for this TaskInterface

    bool hasPressureExpression_, hasBeenScheduled_;
    PatchTreeTaskMap patchTreeMap_;

    /** \brief main execution driver - the callback function exposed to Uintah. */
    void execute( Uintah::Task::CallBackEvent event,
                  const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const,
                  void* stream,  // for GPU tasks, this is the associated stream
                  const int rkStage );

  public:

    /**
     *  \brief Construct a TreeTaskExecute object.
     *  \param trees - the trees that this object is associated with (one per patch)
     *  \param taskName - the name of this task
     *  \param scheduler - the scheduler that this task is associated with
     *  \param patches 	- the list of patches that this TreeTaskExecute object is to be executed on.
     *  \param materials - the list of materials that this task is to be associated with.
     *  \param info	- the PatchInfoMap object that holds patch-specific information (like operators).
     *  \param rkStage - the stage of the RK integrator that this is associated with
     *  \param ioFieldSet - the set of fields that are requested for IO.  This prevents these fields from being recycled internally.
     */
    TreeTaskExecute( TreeMap& trees,
                     const std::string taskName,
                     Uintah::SchedulerP& scheduler,
                     const Uintah::PatchSet* const patches,
                     const Uintah::MaterialSet* const materials,
                     const PatchInfoMap& info,
                     const int rkStage,
                     Uintah::SimulationStateP state,
                     const std::set<std::string>& ioFieldSet,
                     const bool lockAllFields=false);

    ~TreeTaskExecute();

    void schedule( Expr::TagSet newDWFields, const int rkStage );

    PatchTreeTaskMap& get_patch_tree_map() {return patchTreeMap_;}

  };

  //------------------------------------------------------------------

  TreeTaskExecute::TreeTaskExecute( TreeMap& treeMap,
                                    const std::string taskName,
                                    Uintah::SchedulerP& sched,
                                    const Uintah::PatchSet* const patches,
                                    const Uintah::MaterialSet* const materials,
                                    const PatchInfoMap& patchInfoMap,
                                    const int rkStage,
                                    Uintah::SimulationStateP state,
                                    const std::set<std::string>& ioFieldSet,
                                    const bool lockAllFields)
    : scheduler_( sched ),
      patches_( patches ),
      materials_( materials ),
      taskName_( taskName ),
      fml_( scinew Expr::FieldManagerList(taskName) )
  {
    assert( treeMap.size() > 0 );

    hasPressureExpression_ = false;
    hasBeenScheduled_ = false;

#  ifdef HAVE_CUDA
   const int patchID = treeMap.begin()->first;
   TreePtr tree      = treeMap.begin()->second;
   bool isGPUTask = tree->is_homogeneous_gpu( patchID );
   // turn off GPU task for the "initialization" task graph
   if( !( isGPUTask && Uintah::Parallel::usingDevice() ) || ( taskName == "initialization") ) {
     tree->flip_gpu_runnable( patchID, false );
     isGPUTask = false;
   }
#  endif

    Uintah::Task* tsk = scinew Uintah::Task( taskName, this, &TreeTaskExecute::execute, rkStage );

#  ifdef HAVE_CUDA
   if( isGPUTask && Uintah::Parallel::usingDevice() && taskName != "initialization" && state->getCurrentTopLevelTimeStep() != 0 )
     tsk->usesDevice(true);
#  endif

    BOOST_FOREACH( TreeMap::value_type& vt, treeMap ){

      const int patchID = vt.first;
      TreePtr tree = vt.second;

      if( !hasPressureExpression_ ){
        if( tree->computes_field( pressure_tag() ) )
          hasPressureExpression_ = true;
      }

      tree->register_fields( *fml_ );

      BOOST_FOREACH( const std::string& iof, ioFieldSet ){

        const Expr::Tag fieldStateN   ( iof, Expr::STATE_N    );
        const Expr::Tag fieldStateNONE( iof, Expr::STATE_NONE );
        if( tree->has_field(fieldStateN) ){
          if( tree->has_expression(tree->get_id(fieldStateN)) ){
            tree->set_expr_is_persistent( fieldStateN, *fml_ );
          }
        }
        if( tree->has_field(fieldStateNONE) ){
          if( tree->has_expression(tree->get_id(fieldStateNONE)) ){
            tree->set_expr_is_persistent( fieldStateNONE, *fml_ );
          }
        }

      } // loop over persistent fields

      // uncomment the next line to force Uintah to manage all fields:
      if (lockAllFields) tree->lock_fields(*fml_);

      tree->register_fields( *fml_ );

      PatchInfoMap::const_iterator ipim = patchInfoMap.find(patchID);
      assert( ipim != patchInfoMap.end() );
      Info info;
      info.operators = ipim->second.operators;
      info.task = tsk;
      info.tree = tree;
      patchTreeMap_[patchID] = info;

    } // loop over trees

  }

  //------------------------------------------------------------------

  TreeTaskExecute::~TreeTaskExecute()
  {
    delete fml_;
    // Tasks are deleted by the scheduler that they are assigned to.
    // This means that we don't need to delete the task created here.
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
    dbg_fields << "Field requirements for task '" << tree.name() << "'" << endl
               << std::setw(10) << "Mode " << std::left << std::setw(20) << "Field Name"
               << "DW  #Ghost PatchID" << endl
               << "-----------------------------------------------------------------------" << endl;

    //______________________________
    // cycle through each field type
    for( Expr::FieldManagerList::iterator ifm=fml.begin(); ifm!=fml.end(); ++ifm ){

      typedef Expr::FieldManagerBase::PropertyMap PropertyMap;
      PropertyMap& properties = ifm->second->properties();
      PropertyMap::iterator ipm = properties.find("UintahInfo");
      assert( ipm != properties.end() );
      Expr::IDUintahInfoMap& infomap = boost::any_cast< boost::reference_wrapper<Expr::IDUintahInfoMap> >(ipm->second);

      //______________________________________
      // cycle through each field of this type
      for( Expr::IDUintahInfoMap::iterator ii=infomap.begin(); ii!=infomap.end(); ++ii ){

        Expr::UintahFieldAllocInfo& fieldInfo = *(ii->second);
        const Expr::Tag& fieldTag = ii->first;

        dbg_fields << "examining field: " << fieldTag << " for stage " << rkStage << std::endl;

        // see if this field is required by the given tree
        if( !tree.has_field(fieldTag) ){
          dbg_fields << "  - not required by this tree" << std::endl;
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

          dbg_fields << std::setw(10) << "(REQUIRES)"
                     << std::setw(20) << std::left << fieldInfo.varlabel->getName()
                     << "OLD   "
                     << std::left << std::setw(5) << fieldInfo.nghost
                     << *patches << endl;
        }

        //________________
        // set field mode
        if( tree.computes_field( fieldTag ) ){

          // if the field uses dynamic allocation, then the uintah task should not be aware of this field
          if( ! tree.is_persistent(fieldTag) ){
            dbg_fields << " - field is not persistent -> hiding from Uintah." << std::endl;
            continue;
          }

          fieldInfo.mode = (rkStage==1) ? Expr::COMPUTES : Expr::MODIFIES;
        }
        else if( fieldTag.context() == Expr::STATE_N ){
          fieldInfo.mode = Expr::REQUIRES;
          fieldInfo.useOldDataWarehouse = (rkStage < 2);
        }
        else if( fieldTag.context() == Expr::STATE_NP1 ){
          fieldInfo.mode = Expr::REQUIRES;
          fieldInfo.useOldDataWarehouse = false;
        }
        else{
          fieldInfo.mode = Expr::REQUIRES;
          if( newDWFields.find( fieldTag ) != newDWFields.end() )
            fieldInfo.useOldDataWarehouse = false;
        }
        if( tree.name()!="set_time" &&
            tree.name()!="initialization" &&
            (fieldInfo.varlabel->getName()=="time" ||
             fieldInfo.varlabel->getName()=="dt"   ||
             fieldInfo.varlabel->getName()=="timestep")){
          fieldInfo.mode = Expr::REQUIRES;
        }

        // this was needed for Warches. When adding a placeholder expression with STATE_N,
        // we must use the newdw in the initialization task graph.
        if (tree.name() == "initialization" && fieldTag.context() == Expr::STATE_N) {
          fieldInfo.useOldDataWarehouse = false;
        }

        const Uintah::Task::WhichDW dw = ( fieldInfo.useOldDataWarehouse ) ? Uintah::Task::OldDW : Uintah::Task::NewDW;

        switch( fieldInfo.mode ){

        case Expr::COMPUTES:
          dbg_fields << std::setw(10) << "COMPUTES";
          ASSERT( dw == Uintah::Task::NewDW );
          task.computesWithScratchGhost( fieldInfo.varlabel,
                                         materials, Uintah::Task::NormalDomain,
                                         fieldInfo.ghostType, fieldInfo.nghost );

          break;

        case Expr::REQUIRES:
          dbg_fields << std::setw(10) << "REQUIRES";
          task.requires( dw,
                         fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost );
          break;

        case Expr::MODIFIES:
          dbg_fields << std::setw(10) << "MODIFIES";
          ASSERT( dw == Uintah::Task::NewDW );
          task.modifiesWithScratchGhost( fieldInfo.varlabel,
                         patches, Uintah::Task::ThisLevel,
                         materials, Uintah::Task::NormalDomain,
                         fieldInfo.ghostType, fieldInfo.nghost);
          break;

        } // switch

        dbg_fields << std::setw(20) << std::left << fieldInfo.varlabel->getName();
        if( fieldInfo.useOldDataWarehouse ){ dbg_fields << "OLD   "; }
        else{ dbg_fields << "NEW   "; }
        dbg_fields << std::left << std::setw(5) << fieldInfo.nghost << *patches << endl;

      } // field loop
    } // field type loop

    dbg_fields << endl;

  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::schedule( Expr::TagSet newDWFields, const int rkStage )
  {
    ASSERT( !hasBeenScheduled_ );

    dbg_tasks << "Scheduling task '" << taskName_ << "'" << endl;

    const PatchTreeTaskMap::iterator iptm = patchTreeMap_.begin();
    ASSERT( iptm != patchTreeMap_.end() );

    Uintah::Task* const task = iptm->second.task;
    TreePtr tree = iptm->second.tree;
    const int patchID = iptm->first;

    const Uintah::MaterialSubset* const mss = materials_->getUnion();
    const Uintah::PatchSubset* const pss = patches_->getUnion();

    Expr::ExpressionFactory& factory = tree->get_expression_factory();

    // augment newDWFields to include any fields that are a result of tree cleaving
    {
      for( Expr::ExpressionTree::ExprFieldMap::const_iterator imp=tree->field_map().begin(); imp!=tree->field_map().end(); ++imp ){
        const Expr::FieldDeps::FldHelpers& fh = imp->second->field_helpers();
        for( Expr::FieldDeps::FldHelpers::const_iterator ifld=fh.begin(); ifld!=fh.end(); ++ifld ){
          const Expr::FieldDeps::FieldHelperBase& fhb = **ifld;
          const Expr::Tag& tag = fhb.tag();
          if( factory.retrieve_expression( tag, patchID, true ).is_placeholder() ) continue;
          newDWFields.insert( tag );
        }
      }
    }

    add_fields_to_task( *task, *tree, *fml_, pss, mss, newDWFields, rkStage );

    // jcs eachPatch vs. allPatches (gang schedule vs. independent...)
    scheduler_->addTask( task, patches_, materials_ );

    if( hasPressureExpression_ ){
      Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( pressure_tag(), patchID, true ) );
      pexpr.declare_uintah_vars( *task, pss, mss, rkStage );
      pexpr.schedule_solver( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );      
      pexpr.schedule_set_pressure_bcs( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );            
    }
    
    if (tree->computes_field(TagNames::self().radiationsource)) {
      RadiationSource& radExpr = dynamic_cast<RadiationSource&>( factory.retrieve_expression(TagNames::self().radiationsource,patchID,true) );
      radExpr.schedule_ray_tracing( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );
    }

    Expr::Tag ptag;
    for( Expr::TagList::iterator ptag=PoissonExpression::poissonTagList.begin();
        ptag!=PoissonExpression::poissonTagList.end();
        ++ptag ){
      if (tree->computes_field( *ptag )) {
        PoissonExpression& pexpr = dynamic_cast<PoissonExpression&>( factory.retrieve_expression(*ptag,patchID,true) );
        pexpr.schedule_solver( Uintah::getLevelP(pss), scheduler_, materials_, rkStage, tree->name()=="initialization" );
        pexpr.declare_uintah_vars( *task, pss, mss, rkStage );
        pexpr.schedule_set_poisson_bcs( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );                      
      }      
    }
    
    // go through reduction variables that are computed in this Wasatch Task
    // and insert a Uintah task immediately after.
    ReductionHelper::self().schedule_tasks(Uintah::getLevelP(pss), scheduler_, materials_, tree, patchID, rkStage);

    hasBeenScheduled_ = true;
  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::execute( Uintah::Task::CallBackEvent event,
                            const Uintah::ProcessorGroup* const pg,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            Uintah::DataWarehouse* const oldDW,
                            Uintah::DataWarehouse* const newDW,
                            void* stream,
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
    if( event == Uintah::Task::CPU || event == Uintah::Task::GPU ){
      // preventing postGPU / preGPU callbacks to execute the tree again
      for( int ip=0; ip<patches->size(); ++ip ){

        const Uintah::Patch* const patch = patches->get(ip);
        const int patchID = patch->getID();
        PatchTreeTaskMap::iterator iptm = patchTreeMap_.find(patchID);
        ASSERT( iptm != patchTreeMap_.end() );
        const TreePtr tree = iptm->second.tree;

  #     ifdef HAVE_CUDA
        // set the stream for the Tree and the underlying expressions in it
        if( event == Uintah::Task::GPU ) tree->set_cuda_stream( *(cudaStream_t*)stream );
  #     endif

        Expr::ExpressionFactory& factory = tree->get_expression_factory();
        const SpatialOps::OperatorDatabase& opdb = *iptm->second.operators;

        for( int im=0; im<materials->size(); ++im ){

          const int material = materials->get(im);
          try{
            dbg_tasks << endl
                      << "Wasatch: executing graph '" << taskName_
                      << "' for patch " << patch->getID()
                      << " and material " << material
                      << endl;
            if( dbg_tasks_on ) fml_->dump_fields(std::cout);

            fml_->allocate_fields( Expr::AllocInfo( oldDW, newDW, material, patch, pg ) );

            if( hasPressureExpression_ ){
              Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( pressure_tag(), patchID, true ) );
              pexpr.set_patch(patches->get(ip));
              pexpr.set_RKStage(rkStage);
              pexpr.bind_uintah_vars( newDW, patch, material, rkStage );
            }

            Expr::Tag ptag;
            for( Expr::TagList::iterator ptag=PoissonExpression::poissonTagList.begin();
                ptag!=PoissonExpression::poissonTagList.end();
                ++ptag ){
              if (tree->computes_field( *ptag )) {
                PoissonExpression& pexpr = dynamic_cast<PoissonExpression&>( factory.retrieve_expression( *ptag, patchID, true ) );
                pexpr.set_patch(patches->get(ip));
                pexpr.set_RKStage(rkStage);
              }
            }

            // Pass patch information to the coordinate expressions
            typedef std::map<Expr::Tag, std::string> CoordMapT;
            const CoordMapT& coordMap = CoordinateNames::self().coordinate_map();
            // OldVariable& oldVar = OldVariable::self();
            BOOST_FOREACH( const CoordMapT::value_type& coordPair, coordMap )
            {
              const Expr::Tag coordTag = coordPair.first;
              const std::string coordFieldT = coordPair.second;

              if (!(tree->computes_field(coordTag))) continue;

              if (coordFieldT == "SVOL")
              {
                Coordinates<SVolField>& coordExpr = dynamic_cast<Coordinates<SVolField>&>( factory.retrieve_expression( coordTag, patchID, true ) );
                coordExpr.set_patch(patches->get(ip));
                // In case we want to copy coordinates instead of recomputing them, uncomment the following line
  //              oldVar.add_variable<SVolField>( ADVANCE_SOLUTION, coordTag, true);
              }
              if (coordFieldT == "XVOL")
              {
                Coordinates<XVolField>& coordExpr = dynamic_cast<Coordinates<XVolField>&>( factory.retrieve_expression( coordTag, patchID, true ) );
                coordExpr.set_patch(patches->get(ip));
                // In case we want to copy coordinates instead of recomputing them, uncomment the following line
  //              oldVar.add_variable<XVolField>( ADVANCE_SOLUTION, coordTag, true);
              }
              if (coordFieldT == "YVOL")
              {
                Coordinates<YVolField>& coordExpr = dynamic_cast<Coordinates<YVolField>&>( factory.retrieve_expression( coordTag, patchID, true ) );
                coordExpr.set_patch(patches->get(ip));
                // In case we want to copy coordinates instead of recomputing them, uncomment the following line
  //              oldVar.add_variable<YVolField>( ADVANCE_SOLUTION, coordTag, true);
              }
              if (coordFieldT == "ZVOL")
              {
                Coordinates<ZVolField>& coordExpr = dynamic_cast<Coordinates<ZVolField>&>( factory.retrieve_expression( coordTag, patchID, true ) );
                coordExpr.set_patch(patches->get(ip));
                // In case we want to copy coordinates instead of recomputing them, uncomment the following line
  //              oldVar.add_variable<ZVolField>( ADVANCE_SOLUTION, coordTag, true);
              }
            }


            tree->bind_fields( *fml_ );
            tree->bind_operators( opdb );
            tree->execute_tree();

            dbg_tasks << "Wasatch: done executing graph '" << taskName_ << "'" << endl;
            fml_->deallocate_fields();
          }
          catch( std::exception& e ){
            proc0cout << e.what() << endl;
            throw std::runtime_error( "Error" );
          }
        }
      }
    } // event : GPU, CPU
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
                                const int rkStage,
                                Uintah::SimulationStateP state,
                                const std::set<std::string>& ioFieldSet,
                                const bool lockAllFields)
  {
    // only set up trees on the patches that we own on this process.
    const Uintah::PatchSet*  perproc_patchset = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    const Uintah::PatchSubset* const localPatches = perproc_patchset->getSubset(Uintah::Parallel::getMPIRank());

    std::vector<TreeMap> trLstTrns(localPatches->size());

    for( int ip=0; ip<localPatches->size(); ++ip ){
      const int patchID = localPatches->get(ip)->getID();
      TreePtr tree( scinew Expr::ExpressionTree(roots,factory,patchID,taskName) );
      const TreeList treeList = tree->split_tree();

      if( ip==0 ){
        trLstTrns.resize( treeList.size() );
        if( Uintah::Parallel::getMPIRank() == 0 ){
          if( treeList.size() > 1 ){
            std::ostringstream fnam;
            fnam << tree->name() << "_original.dot";
            proc0cout << "writing pre-cleave tree to " << fnam.str() << endl;
            std::ofstream fout( fnam.str().c_str() );
            tree->write_tree(fout);
          }
          BOOST_FOREACH( TreePtr tr, treeList ){
            std::ostringstream fnam;
            fnam << tr->name() << ".dot";
            std::ofstream fout( fnam.str().c_str() );
            tr->write_tree(fout);
          }
        }
      }

      // Transpose the storage so that we have a vector with each entry in the
      // vector containing the map of patch IDs to each tree
      for( size_t i=0; i<treeList.size(); ++i ){
        trLstTrns[i][patchID] = treeList[i];
      }

    } // patch loop

    // create a TreeTaskExecute for each tree (on all patches)
    BOOST_FOREACH( TreeMap& tl, trLstTrns ){
      execList_.push_back( scinew TreeTaskExecute( tl, tl.begin()->second->name(),
                                                   sched, patches, materials,
                                                   info, rkStage, state, ioFieldSet, lockAllFields ) );
    }

  }

  //------------------------------------------------------------------

  TaskInterface::~TaskInterface()
  {
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

  Expr::TagList
  TaskInterface::collect_tags_in_task() const
  {
    Expr::TagList tags;
    BOOST_FOREACH( TreeTaskExecute* taskexec, execList_ ){
      TreePtr tree = taskexec->get_patch_tree_map().begin()->second.tree;
      BOOST_FOREACH( const Expr::ExpressionTree::ExprFieldMap::value_type& vt, tree->field_map() ){
        const Expr::TagList& tl = tree->get_expression_factory().get_labels( vt.first );
        tags.insert( tags.end(), tl.begin(), tl.end() );
      }
    }
    return tags;
  }

  //------------------------------------------------------------------

} // namespace Wasatch
