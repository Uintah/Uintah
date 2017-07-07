/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

//-- SpatialOps library includes --//
#include <spatialops/OperatorDatabase.h>


//-- expression library includes --//
#include <expression/ExprLib.h>
#include <expression/FieldRequest.h>
#include <expression/ExpressionTree.h>


//-- Uintah includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
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
#include <CCA/Components/Wasatch/Expressions/DORadSolver.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/Expressions/RadiationSource.h>
#include <CCA/Components/Wasatch/ReductionHelper.h>
#include <CCA/Components/Wasatch/CoordinateHelper.h>
#include <CCA/Components/Wasatch/ParticlesHelper.h>
#include <CCA/Components/Wasatch/WasatchParticlesHelper.h>
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

 To enable multiple debug flags, put quotation marks around the flags and use a comma 
 to separate them (no spaces after commas)
  tcsh: setenv SCI_DEBUG "WASATCH_TASKS:+ WASATCH_FIELDS:+"
  bash: export SCI_DEBUG="WASATCH_TASKS:+,WASATCH_FIELDS:+"

 To enable one flag and disable another that was previously enabled, either
 define a new flag excluding the unwanted flag, or redefine SCI_DEBUG with a -
 after the unwanted flag (no spaces after commas)
   tcsh: setenv SCI_DEBUG "WASATCH_TASKS:- WASATCH_FIELDS:+"
   bash: export SCI_DEBUG="WASATCH_TASKS:-,WASATCH_FIELDS:+"

 */

static Uintah::DebugStream dbgt("WASATCH_TASKS", false);  // task diagnostics
static Uintah::DebugStream dbgf("WASATCH_FIELDS", false); // field diagnostics
#define dbg_tasks_on  dbgt.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_fields_on dbgf.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_tasks  if( dbg_tasks_on  ) dbgt
#define dbg_fields if( dbg_fields_on ) dbgf

namespace WasatchCore{

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
    void execute( Uintah::DetailedTask* dtask,
                  Uintah::Task::CallBackEvent event,
                  const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const,
                  void* old_TaskGpuDW,
                  void* new_TaskGpuDW,
                  void* stream,  // for GPU tasks, this is the associated stream
                  int deviceID,
                  const int rkStage);

# ifdef HAVE_CUDA

    /**
     *  \class GPULoadBalancer
     *  \ingroup WasatchGraph
     *  \brief Handles the selection and assignment of GPU device index for a given task.
     *
     *     - Supports selection and assignment of device for on-node multi-GPU system
     *     - Used for assigning device indices for heterogeneous CPU-GPU tasks.
     *     - Works with an assumption that all the GPUs have the same computational resources.
     */

    class GPULoadBalancer
    {
      int gpuDeviceID_;    ///< current asssigned GPU
      int gpuDeviceCount_; ///< total available GPUs

      GPULoadBalancer() :
        gpuDeviceID_(0),
        gpuDeviceCount_(0)
      {
        ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
        gpuDeviceCount_ = CDI.get_device_count();
      }

      ~GPULoadBalancer(){}

      inline static GPULoadBalancer& self()
      {
        static GPULoadBalancer gpulb;
        return gpulb;
      }

    public:

      /** \brief returns the device Index */
      inline static int get_device_index(){
        GPULoadBalancer& gpulb = self();
        gpulb.gpuDeviceID_ = (++gpulb.gpuDeviceID_) % gpulb.gpuDeviceCount_;
        return gpulb.gpuDeviceID_;
      }
    };

# endif

    /**
     *  \class ExecMutex
     *  \brief Scoped lock.
     */
    class ExecMutex
    {
#   ifdef ENABLE_THREADS
      const boost::mutex::scoped_lock lock;
      inline boost::mutex& get_mutex() const{ static boost::mutex m; return m; }
    public:
      ExecMutex() : lock( get_mutex() ) {}
      ~ExecMutex() {}
#   else
    public:
      ExecMutex(){}
      ~ExecMutex(){}
#   endif
    };

  public:

    /**
     *  \brief Construct a TreeTaskExecute object.
     *  \param treeMap - the trees that this object is associated with (one per patch)
     *  \param taskName - the name of this task
     *  \param scheduler - the scheduler that this task is associated with
     *  \param patches  - the list of patches that this TreeTaskExecute object is to be executed on.
     *  \param materials - the list of materials that this task is to be associated with.
     *  \param info     - the PatchInfoMap object that holds patch-specific information (like operators).
     *  \param rkStage - the stage of the RK integrator that this is associated with
     *  \param state
     *  \param ioFieldSet - the set of fields that are requested for IO.  This prevents these fields from being recycled internally.
     *  \param lockAllFields if true, then all fields will be marked persistent.
     *         Otherwise, memory will be reclaimed when possible.
     */

    TreeTaskExecute( TreeMap& treeMap,
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
    
    Expr::FieldManagerList& get_fml(){return *fml_;}

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
                                    const bool lockAllFields )
    : scheduler_( sched ),
      patches_( patches ),
      materials_( materials ),
      taskName_( taskName ),
      fml_( scinew Expr::FieldManagerList(taskName) )
  {
    assert( treeMap.size() > 0 );
    hasPressureExpression_ = false;
    hasBeenScheduled_ = false;

    Uintah::Task* tsk = scinew Uintah::Task( taskName, this, &TreeTaskExecute::execute, rkStage );
    BOOST_FOREACH( TreeMap::value_type& vt, treeMap ){

      const int patchID = vt.first;
      TreePtr tree = vt.second;

#     ifdef HAVE_CUDA
      const bool isHomogeneous = tree->is_homogeneous_gpu();
      bool gpuTurnedOff = !isHomogeneous;
      if( !(isHomogeneous && Uintah::Parallel::usingDevice()) || (taskName == "initialization") ) {
        // Force everything to CPU for initialization & also for heterogeneous tasks.
        // For heterogeneous graphs, ExprLib will control GPU execution.
        tree->turn_off_gpu_runnable();
        gpuTurnedOff = true;
	
        // Get the best device available
        tree->set_device_index( GPULoadBalancer::get_device_index(), *fml_ );
      }

      // Flag the task as Uintah GPU::Task, if it is homogeneous GPU graph
      if( !gpuTurnedOff && Uintah::Parallel::usingDevice() && taskName != "initialization" ){
        tsk->usesDevice( true );
      }
#     endif

      if( !hasPressureExpression_ ){
        if( tree->computes_field( TagNames::self().pressure ) && taskName != "initialization" ){
          hasPressureExpression_ = true;
        }
      }
      tree->register_fields( *fml_ );

      BOOST_FOREACH( const std::string& iof, ioFieldSet ){

        const Expr::Tag fieldStateN   ( iof, Expr::STATE_N    );
        const Expr::Tag fieldStateNONE( iof, Expr::STATE_NONE );
        const Expr::Tag fieldStateNP1 ( iof, Expr::STATE_NP1  );
        
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
        if( tree->has_field(fieldStateNP1) ){
          if( tree->has_expression(tree->get_id(fieldStateNP1)) ){
            tree->set_expr_is_persistent( fieldStateNP1, *fml_ );
          }
        }

        // the error norm computed by the dual time integrator needs to be also persistent
        const Expr::Tag normTag = Expr::Tag(iof + "_err_norm", Expr::STATE_NONE);
        if (tree->has_field(normTag) ) {
          tree->set_expr_is_persistent(normTag, *fml_);
        }

      } // loop over persistent fields

      // force Uintah to manage all fields:
      if( lockAllFields ) tree->lock_fields(*fml_);

#     ifdef HAVE_CUDA
      // For Heterogeneous case only
      if( taskName != "initialization" && tree->is_homogeneous_gpu() && Uintah::Parallel::usingDevice() ){
        // For a heterogeneous task, restore the GPU runnable property for the expressions.
        if (gpuTurnedOff) tree->restore_gpu_runnable();
      }
#     endif

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
   *  \param rkStage - the current Runge-Kutta stage
   *
   *  This function analyzes the ExpressionTree to identify what
   *  fields are required for this task, and then advertises them to
   *  Uintah.  The COMPUTES/REQUIRES is automatically deduced.
   */
  void
  add_fields_to_task( Uintah::Task& task,
                      Expr::ExpressionTree& tree,
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

        // for the initialization tree, make sure ALL fields are persistent. This is needed
        // in case an initial condition for a transport equation is "derived" from other quantities.
        // In this case, ExprLib will mark the expression as local non persistent
        if (tree.name() == "initialization") {
          tree.set_expr_is_persistent(fieldTag, fml);
        }
        // look for particle variables that are managed by uintah
        if (tree.name()!="initialization") {
          if (fieldInfo.varlabel->typeDescription()->getType() == Uintah::TypeDescription::ParticleVariable) {
            if (tree.is_persistent(fieldTag)) {
              // if a particle variable is managed by uintah, then pass it on to the particles helper
              // for use in relocation
              Uintah::ParticlesHelper::mark_for_relocation(fieldTag.name());
              Uintah::ParticlesHelper::needs_boundary_condition(fieldTag.name());
            }
          }
        }

        dbg_fields << "examining field: " << fieldTag << " for stage " << rkStage << std::endl;

        // see if this field is required by the given tree
        if( !tree.has_field(fieldTag) ){
          dbg_fields << "  - not required by this tree" << std::endl;
          continue;
        }

        // Use the old DW on the first RK stage.  Thereafter,
        // we modify the values already in the new DW.
        fieldInfo.useOldDataWarehouse = (rkStage < 1);

        //________________
        // set field mode
        fieldInfo.useParentOldDataWarehouse = false;
        const bool hasDualTime = WasatchCore::Wasatch::has_dual_time();
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
          fieldInfo.useOldDataWarehouse = true;
          fieldInfo.useParentOldDataWarehouse = hasDualTime? true : false;
        }
        else if( fieldTag.context() == Expr::STATE_NP1 ){
          fieldInfo.mode = Expr::REQUIRES;
          fieldInfo.useOldDataWarehouse = false;
        }
        else if( fieldTag.context() == Expr::STATE_DYNAMIC ){
          fieldInfo.mode = Expr::REQUIRES;
          fieldInfo.useOldDataWarehouse = (rkStage < 2);
        }
        else{
          fieldInfo.mode = Expr::REQUIRES;
          if( newDWFields.find( fieldTag ) != newDWFields.end() )
            fieldInfo.useOldDataWarehouse = false;
        }

        
        // here's what's happening here:
        /*
         _____________________________________________________________________________
         | INTEGRATOR  |    STATE_N            |  STATE_NONE |  STATE_DYNAMIC          |
         |-------------|-----------------------|-------------|-------------------------|
         | REGULAR     | Task::OldDW           | Task::NewDW | stage==1? OldDW : NewDW |
         |-------------|-----------------------|-------------|-------------------------|
         | DUAL TIME   | Task::ParentOldDW     | Task::NewDW | Task::OldDW             |
         |_____________|_______________________|_____________|_________________________|
         */
        const Uintah::Task::WhichDW dw = ( fieldInfo.useOldDataWarehouse ) ? ( (fieldTag.context() == Expr::STATE_DYNAMIC) ? Uintah::Task::OldDW : ( hasDualTime ? Uintah::Task::ParentOldDW : Uintah::Task::OldDW) )  : Uintah::Task::NewDW;
//        const Uintah::Task::WhichDW dw = ( fieldInfo.useOldDataWarehouse ) ? Uintah::Task::ParentOldDW : Uintah::Task::NewDW;
        
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
          // tsaad: To be able to modify ghost cells, we will need:
          //  1. requires with ghost cells
          //  2. a modifies
          // To avoid pitfalls of misusing modifies in Wasatch, we use modifiesWithScratchGhost
          // which calls both a requires and a modifies.
          task.modifiesWithScratchGhost( fieldInfo.varlabel,
                                         patches, Uintah::Task::ThisLevel,
                                         materials, Uintah::Task::NormalDomain,
                                         fieldInfo.ghostType, fieldInfo.nghost);
          break;

        } // switch

        dbg_fields << std::setw(20) << std::left << fieldInfo.varlabel->getName();
        if( fieldInfo.useOldDataWarehouse ){ dbg_fields << " OLD   "; }
        else{ dbg_fields << " NEW   "; }
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
        const Expr::TagList tags = imp->second->get_tags();
        for (Expr::TagList::const_iterator iTag = tags.begin(); iTag != tags.end(); ++iTag) {
          const Expr::Tag& tag = *iTag;
          if( factory.retrieve_expression( tag, patchID, true ).is_placeholder() ) continue;
          newDWFields.insert( tag );
        }
      }
    }

    add_fields_to_task( *task, *tree, *fml_, pss, mss, newDWFields, rkStage );

    //---------------------------------------------------------------------------------------------------------------------------
    // Added for temporal scheduling support when using RMCRT - APH 05/30/17
    //---------------------------------------------------------------------------------------------------------------------------
    if (tree->computes_field(TagNames::self().radiationsource)) {
      // For RMCRT there will be 2 task graphs - put the radiation tasks in TG-1, otherwise tasks go into TG-0, or both TGs
      //   TG-0 == carry forward and/or non-radiation timesteps
      //   TG-1 == RMCRT radiation timestep
      scheduler_->addTask(task, patches_, materials_, Uintah::RMCRTCommon::TG_RMCRT);
    }
    //---------------------------------------------------------------------------------------------------------------------------
    // jcs eachPatch vs. allPatches (gang schedule vs. independent...)
    else {
      scheduler_->addTask(task, patches_, materials_);
    }

    if( hasPressureExpression_ && Wasatch::flow_treatment() != WasatchCore::COMPRESSIBLE && Wasatch::need_pressure_solve() ){
      Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( TagNames::self().pressure, patchID, true ) );
      pexpr.declare_uintah_vars( *task, pss, mss, rkStage );
      pexpr.schedule_solver( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );
      pexpr.schedule_set_pressure_bcs( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );
    }

    if( tree->computes_field(TagNames::self().radiationsource) ){
      RadiationSource& radExpr = dynamic_cast<RadiationSource&>( factory.retrieve_expression(TagNames::self().radiationsource,patchID,true) );
      radExpr.schedule_ray_tracing( Uintah::getLevelP(pss), scheduler_, materials_, rkStage );
    }

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

    BOOST_FOREACH( const Expr::Tag& tag, DORadSolver::intensityTags ){
      if( !tree->computes_field(tag) ) continue;
      std::cout << "preliminary stuff for " << tag << " ... " << std::flush;
      DORadSolver& rad = dynamic_cast<DORadSolver&>( factory.retrieve_expression(tag,patchID,true) );
      rad.schedule_solver( Uintah::getLevelP(pss), scheduler_, materials_, rkStage, tree->name()=="initialization" );
      rad.declare_uintah_vars( *task, pss, mss, rkStage );
      std::cout << "done" << std::endl;
    }

    // go through reduction variables that are computed in this Wasatch Task
    // and insert a Uintah task immediately after.
    ReductionHelper::self().schedule_tasks(Uintah::getLevelP(pss), scheduler_, materials_, tree, patchID, rkStage);

    hasBeenScheduled_ = true;
  }

  //------------------------------------------------------------------

  void
  TreeTaskExecute::execute( Uintah::DetailedTask* dtask,
                            Uintah::Task::CallBackEvent event,
                            const Uintah::ProcessorGroup* const pg,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            Uintah::DataWarehouse* const oldDW,
                            Uintah::DataWarehouse* const newDW,
                            void* old_TaskGpuDW,
                            void* new_TaskGpuDW,
                            void* stream,
                            int deviceID,
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

    ExecMutex lock; // thread-safe

    const bool isGPUTask = (event == Uintah::Task::GPU);

    // preventing postGPU / preGPU callbacks to execute the tree again
    for( int ip=0; ip<patches->size(); ++ip ){

      const Uintah::Patch* const patch = patches->get(ip);
      const int patchID = patch->getID();
      PatchTreeTaskMap::iterator iptm = patchTreeMap_.find(patchID);
      ASSERT( iptm != patchTreeMap_.end() );
      const TreePtr tree = iptm->second.tree;

#     ifdef HAVE_CUDA
      if( isGPUTask ){ // homogeneous GPU task
        dbg_tasks << endl
            << "Executing -  Wasatch as Homogeneous GPU Task : " << taskName_
            << " on patch : " << patch->getID()
            << endl;

        // set the device index passed from Uintah to the Expression tree
        // Currently it is not yet fixed as the callback is not providing deviceID
        tree->set_device_index( deviceID, *fml_);
      }
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


          Uintah::ParticleSubset* const pset = newDW->haveParticleSubset(material, patch) ?
              newDW->getParticleSubset(material, patch) :
              ( oldDW ? (oldDW->haveParticleSubset(material, patch) ? oldDW->getParticleSubset(material, patch) : nullptr ) : nullptr );

          AllocInfo ainfo( oldDW, newDW, material, patch, pset, pg, isGPUTask );
          fml_->allocate_fields( ainfo );

          if( hasPressureExpression_ && Wasatch::flow_treatment() != WasatchCore::COMPRESSIBLE && Wasatch::need_pressure_solve() ){
            Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( TagNames::self().pressure, patchID, true ) );
            pexpr.bind_uintah_vars( newDW, patch, material, rkStage );
          }

          BOOST_FOREACH( const Expr::Tag& ptag, PoissonExpression::poissonTagList ){
            if( tree->computes_field( ptag ) ){
              PoissonExpression& pexpr = dynamic_cast<PoissonExpression&>( factory.retrieve_expression( ptag, patchID, true ) );
              pexpr.bind_uintah_vars( newDW, patch, material, rkStage );
            }
          }

          // In case we want to copy coordinates instead of recomputing them, uncomment the following lines
          //            OldVariable& oldVar = OldVariable::self();
          //            typedef std::map<Expr::Tag, std::string> CoordMapT;
          //            BOOST_FOREACH( const CoordMapT::value_type& coordPair, CoordinateNames::coordinate_map() ){
          //              const Expr::Tag& coordTag = coordPair.first;
          //              const std::string& coordFieldT = coordPair.second;
          //
          //              if( ! tree->computes_field(coordTag) ) continue;
          //
          //              if     ( coordFieldT == "SVOL" ) oldVar.add_variable<SVolField>( ADVANCE_SOLUTION, coordTag, true );
          //              else if( coordFieldT == "XVOL" ) oldVar.add_variable<XVolField>( ADVANCE_SOLUTION, coordTag, true );
          //              else if( coordFieldT == "YVOL" ) oldVar.add_variable<YVolField>( ADVANCE_SOLUTION, coordTag, true );
          //              else if( coordFieldT == "ZVOL" ) oldVar.add_variable<ZVolField>( ADVANCE_SOLUTION, coordTag, true );
          //            }

          BOOST_FOREACH( const Expr::Tag& tag, DORadSolver::intensityTags ){
            if( tree->computes_field( tag ) ){
              DORadSolver& rad = dynamic_cast<DORadSolver&>( factory.retrieve_expression(tag,patchID, true ) );
              std::cout << "Binding vars for " << tag << " ..." << std::flush;
              rad.bind_uintah_vars( newDW, patch, material, rkStage );
              std::cout << "done\n";
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
                               DTIntegratorMapT& dualTimeIntegrators,
                               const std::vector<std::string> & varNames,
                               const std::vector<Expr::Tag>   & rhsTags,
                               Uintah::SimulationStateP state,
                               const std::set<std::string>& ioFieldSet,
                               const bool lockAllFields)
  {
    // only set up trees on the patches that we own on this process.
    const Uintah::PatchSet*  perproc_patchset = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    const Uintah::PatchSubset* const localPatches = perproc_patchset->getSubset(Uintah::Parallel::getMPIRank());
    
    typedef std::map<int,TreeMap> TreeMapTranspose;
    TreeMapTranspose trLstTrns;
    
    const TagNames& tags = TagNames::self();
    
    for( int ip=0; ip<localPatches->size(); ++ip ){
      const int patchID = localPatches->get(ip)->getID();
      TreePtr tree( scinew Expr::ExpressionTree(roots,factory,patchID,taskName) );
      
      // tsaad: for the moment, just use a fixed point integrator with BDF order 1
      dualTimeIntegrators[patchID] = scinew Expr::DualTime::FixedPointBDFDualTimeIntegrator<SVolField>(tree.get(),
                                                                                                       &factory,
                                                                                                       patchID, "Wasatch Dual Time Integrator",
                                                                                                       tags.dt,
                                                                                                       tags.ds,
                                                                                                       tags.timestep,
                                                                                                       1,
                                                                                                       Expr::STATE_DYNAMIC);
      
      Expr::DualTime::BDFDualTimeIntegrator& dtIntegrator = *dualTimeIntegrators[patchID];
      dtIntegrator.set_dual_time_step_expression(factory.get_id(tags.ds));

      for( size_t i=0; i<varNames.size(); i++ ){
        dtIntegrator.add_variable<SVolField>(varNames[i], rhsTags[i]);
      }
      dtIntegrator.prepare_for_integration<SVolField>();
      
      const TreeList treeList = (dtIntegrator.get_tree()).split_tree();
      
      // write out graph information.
      if( Uintah::Parallel::getMPIRank() == 0 && ip == 0 ){
        const bool writeTreeDetails = dbg_tasks_on;
        if( treeList.size() > 1 ){
          std::ostringstream fnam;
          fnam << tree->name() << "_original.dot";
          proc0cout << "writing pre-cleave tree to " << fnam.str() << endl;
          std::ofstream fout( fnam.str().c_str() );
          tree->write_tree(fout,false,writeTreeDetails);
        }
        BOOST_FOREACH( TreePtr tr, treeList ){
          std::ostringstream fnam;
          fnam << tr->name() << ".dot";
          std::ofstream fout( fnam.str().c_str() );
          tr->write_tree(fout,false,writeTreeDetails);
        }
      }
      
      // Transpose the storage so that we have a vector with each entry in the
      // vector containing the map of patch IDs to each tree
      for( size_t i=0; i<treeList.size(); ++i ){
        trLstTrns[i][patchID] = treeList[i];
      }
      
    } // patch loop
    
    //__________________________________________________________________________
    // create a TreeTaskExecute for each tree (on all patches)
    
    // set persistency on ALL transported variables. Important for Uintah.
    std::set<std::string> persistentFields = ioFieldSet;
    BOOST_FOREACH(const std::string& var, varNames)
    {
      persistentFields.insert(var);
    }

    BOOST_FOREACH( TreeMapTranspose::value_type& tlpair, trLstTrns ){
      TreeMap& tl = tlpair.second;
      TreeTaskExecute* tskExec = scinew TreeTaskExecute( tl, tl.begin()->second->name(),
                                                        sched, patches, materials,
                                                        info, 0, state,
                                                        persistentFields, lockAllFields );
      execList_.push_back( tskExec );
      
      for( int ip=0; ip<localPatches->size(); ++ip ){
        const int patchID = localPatches->get(ip)->getID();
        dualTimeIntegrators[patchID]->set_fml(tskExec->get_fml());
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
                                const int rkStage,
                                Uintah::SimulationStateP state,
                                const std::set<std::string>& ioFieldSet,
                                const bool lockAllFields)
  {
    // only set up trees on the patches that we own on this process.
    const Uintah::PatchSet*  perproc_patchset = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    const Uintah::PatchSubset* const localPatches = perproc_patchset->getSubset(Uintah::Parallel::getMPIRank());

    typedef std::map<int,TreeMap> TreeMapTranspose;
    TreeMapTranspose trLstTrns;

    for( int ip=0; ip<localPatches->size(); ++ip ){
      const int patchID = localPatches->get(ip)->getID();
      TreePtr tree( scinew Expr::ExpressionTree(roots,factory,patchID,taskName) );
      const TreeList treeList = tree->split_tree();

      // write out graph information.
      if( Uintah::Parallel::getMPIRank() == 0 && ip == 0 ){
        const bool writeTreeDetails = dbg_tasks_on;
        if( treeList.size() > 1 ){
          std::ostringstream fnam;
              fnam << tree->name() << "_original.dot";
          proc0cout << "writing pre-cleave tree to " << fnam.str() << endl;
          std::ofstream fout( fnam.str().c_str() );
          tree->write_tree(fout,false,writeTreeDetails);
        }
        BOOST_FOREACH( TreePtr tr, treeList ){
          std::ostringstream fnam;
              fnam << tr->name() << ".dot";
          std::ofstream fout( fnam.str().c_str() );
          tr->write_tree(fout,false,writeTreeDetails);
        }
      }

      // Transpose the storage so that we have a vector with each entry in the
      // vector containing the map of patch IDs to each tree
      for( size_t i=0; i<treeList.size(); ++i ){
        trLstTrns[i][patchID] = treeList[i];
      }
      
    } // patch loop

    // create a TreeTaskExecute for each tree (on all patches)
    BOOST_FOREACH( TreeMapTranspose::value_type& tlpair, trLstTrns ){
      TreeMap& tl = tlpair.second;
      execList_.push_back( scinew TreeTaskExecute( tl, tl.begin()->second->name(),
                                                   sched, patches, materials,
                                                   info, rkStage, state,
                                                   ioFieldSet, lockAllFields ) );
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

} // namespace WasatchCore
