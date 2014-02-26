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

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/TaskInterface.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>


//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>
#include <expression/PlaceHolderExpr.h>

//-- SpatialOps includes --//
#include <spatialops/Nebo.h>

//-- Uintah Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/SimulationState.h>
#include <sci_defs/cuda_defs.h>


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
                      pss, Uintah::Task::ThisLevel,
                      mss, Uintah::Task::NormalDomain,
                      get_uintah_ghost_type<FieldT>(),
                      get_n_ghost<FieldT>() );
      task->requires( Uintah::Task::NewDW,
                      ifld->rhsLabel,
                      pss, Uintah::Task::ThisLevel,
                      mss, Uintah::Task::NormalDomain,
                      get_uintah_ghost_type<FieldT>(),
                      get_n_ghost<FieldT>() );
    }
  }

  //==================================================================

  template<typename FieldT>
  void
  do_update( const Uintah::Task::CallBackEvent event,
             const std::set< TimeStepper::FieldInfo<FieldT> >& fields,
             const Uintah::Patch* const patch,
             const int material,
             Uintah::DataWarehouse* const oldDW,
             Uintah::DataWarehouse* const newDW,
             const double deltat,
             const int rkStage,
             void* stream,
             const TimeIntegrator& timeInt )
  {
    const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<FieldT>();
    const int ng = get_n_ghost<FieldT>();

    typedef std::set< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld=fields.begin(); ifld!=fields.end(); ++ifld ){

      typedef typename SelectUintahFieldType<FieldT>::const_type ConstUintahField;
      typedef typename SelectUintahFieldType<FieldT>::type       UintahField;

      double* phiNewDevice = NULL;                           // Device Variable
      UintahField phiNew;

#     ifdef HAVE_CUDA
      const char* philabel = ifld->varLabel->getName().c_str();
      if( event == Uintah::Task::GPU || event == Uintah::Task::postGPU ){
        newDW->allocateTemporary( phiNew, patch, gt, ng );
        Uintah::GPUGridVariable<double> myDeviceVar;
        newDW->getGPUDW()->getModifiable( myDeviceVar, philabel, patch->getID(), material );
        phiNewDevice = const_cast<double*>( myDeviceVar.getPointer() );
      } else{
        if( rkStage==1 ) newDW->allocateAndPut( phiNew, ifld->varLabel, material, patch, gt, ng );  // note that these fields do have ghost info.
        else             newDW->getModifiable ( phiNew, ifld->varLabel, material, patch, gt, ng );
      }
#     else
        if( rkStage==1 ) newDW->allocateAndPut( phiNew, ifld->varLabel, material, patch, gt, ng );  // note that these fields do have ghost info.
        else             newDW->getModifiable ( phiNew, ifld->varLabel, material, patch, gt, ng );
#     endif

      ConstUintahField phiOld, rhs;
      double* phiOldDevice = NULL;
      double* rhsDevice    = NULL;
      SpatialOps::MemoryType mtype = SpatialOps::LOCAL_RAM;
      const unsigned short int deviceIndex = 0;

#     ifdef HAVE_CUDA
      const char* rhslabel = ifld->rhsLabel->getName().c_str();
      if( event == Uintah::Task::GPU || event == Uintah::Task::postGPU ){
        mtype = SpatialOps::EXTERNAL_CUDA_GPU;

        oldDW->get( phiOld, ifld->varLabel, material, patch, gt, ng );
        Uintah::GPUGridVariable<double> myOldDeviceVar;
        oldDW->getGPUDW()->get( myOldDeviceVar, philabel, patch->getID(), material );
        phiOldDevice = const_cast<double*>( myOldDeviceVar.getPointer() );

        newDW->get( rhs,    ifld->rhsLabel, material, patch, gt, ng );
        Uintah::GPUGridVariable<double> myrhsDeviceVar;
        newDW->getGPUDW()->get( myrhsDeviceVar, rhslabel, patch->getID(), material );
        rhsDevice = const_cast<double*>( myrhsDeviceVar.getPointer() );
      }else{
        oldDW->get( phiOld, ifld->varLabel, material, patch, gt, ng );
        newDW->get( rhs,    ifld->rhsLabel, material, patch, gt, ng );
      }
#     else
      oldDW->get( phiOld, ifld->varLabel, material, patch, gt, ng );
      newDW->get( rhs,    ifld->rhsLabel, material, patch, gt, ng );
#     endif
      //______________________________________
      // forward Euler or RK3SSP timestep at each point:
      FieldT*       const fnew = wrap_uintah_field_as_spatialops<FieldT>(phiNew, patch, mtype, deviceIndex, phiNewDevice);
      const FieldT* const frhs = wrap_uintah_field_as_spatialops<FieldT>(rhs,    patch, mtype, deviceIndex, rhsDevice);
      const FieldT* const fold = wrap_uintah_field_as_spatialops<FieldT>(phiOld, patch, mtype, deviceIndex, phiOldDevice);
      using namespace SpatialOps;
      const double a = timeInt.alpha[rkStage-1];
      const double b = timeInt.beta[rkStage-1];

#     ifdef HAVE_CUDA
      if( event == Uintah::Task::GPU || event == Uintah::Task::postGPU ) fnew->set_stream( *(cudaStream_t*)stream );
#     endif

      if( rkStage==1 ) *fnew <<= *fold + deltat * *frhs; // for the first stage, no need to do an extra multiplication
      else             *fnew <<= a * *fold + b * (*fnew  + deltat * *frhs);

      // Clean up the spatialops fields we created.
      delete fnew; delete fold; delete frhs;
    }
  }

  //==================================================================

  TimeStepper::TimeStepper( Uintah::SimulationStateP sharedState,
                            GraphCategories& grafCat,
                            const TimeIntegrator timeInt )
    : sharedState_        ( sharedState ),
      solnGraphHelper_    ( grafCat[ADVANCE_SOLUTION] ),
      postProcGraphHelper_( grafCat[POSTPROCESSING] ),
      timeInt_            ( timeInt )
  {}

  //------------------------------------------------------------------

  TimeStepper::~TimeStepper()
  {
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

    // need to explicitly make all RHS fields persistent.  This avoids the situation
    // where they may be internal nodes in a graph and could thus turn into "temporary"
    // fields, leading to non-exposure to Uintah and bad things...
    std::set<std::string> persistentFields( ioFieldSet );
    for( ScalarFields::const_iterator i=scalarFields_.begin(); i!=scalarFields_.end(); ++i )  persistentFields.insert( i->rhsLabel->getName() );
    for(   XVolFields::const_iterator i=  xVolFields_.begin(); i!=  xVolFields_.end(); ++i )  persistentFields.insert( i->rhsLabel->getName() );
    for(   YVolFields::const_iterator i=  yVolFields_.begin(); i!=  yVolFields_.end(); ++i )  persistentFields.insert( i->rhsLabel->getName() );
    for(   ZVolFields::const_iterator i=  zVolFields_.begin(); i!=  zVolFields_.end(); ++i )  persistentFields.insert( i->rhsLabel->getName() );


    //________________________________________________________
    // add a task to populate a "field" with the current time.
    // This is required by the time integrator.
    {
      // add a task to update current simulation time
      Uintah::Task* updateCurrentTimeTask =
          scinew Uintah::Task( "update current time",
                               this,
                               &TimeStepper::update_current_time,
                               solnGraphHelper_->exprFactory,
                               rkStage );
      updateCurrentTimeTask->requires( Uintah::Task::OldDW, sharedState_->get_delt_label() );
      sched->addTask( updateCurrentTimeTask, patches, materials );

      IDSet ids; ids.insert(timeID);
      TaskInterface* const timeTask =
          scinew TaskInterface( ids,
                                "set_time",
                                *(solnGraphHelper_->exprFactory),
                                level, sched, patches, materials,
                                patchInfoMap,
                                1, sharedState_, persistentFields );
      taskInterfaceList_.push_back( timeTask );
      timeTask->schedule( rkStage );
    }

    //_________________________________________________________________
    // Schedule the task to compute the RHS for the transport equations
    //
    bool rhsDeviceTask;
    try{
      // jcs for multistage integrators, we may need to keep the same
      //     field manager list for all of the stages?  Otherwise we
      //     will have all sorts of name clashes?

      TaskInterface* rhsTask = scinew TaskInterface( solnGraphHelper_->rootIDs,
                                                     "rhs_" + strRKStage.str(),
                                                     *(solnGraphHelper_->exprFactory),
                                                     level, sched, patches, materials,
                                                     patchInfoMap,
                                                     rkStage, sharedState_, persistentFields );

      taskInterfaceList_.push_back( rhsTask );
      rhsTask->schedule( rkStage ); // must be scheduled after coordHelper_
#     ifdef HAVE_CUDA
        //if( rhsTask->get_task_event() == Uintah::Task::GPU ) rhsDeviceTask = true;
#     endif
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << "*************************************************" << endl
          << "Error building ExpressionTree for RHS evaluation." << endl
          << " root nodes: ";
          for( IDSet::const_iterator id = solnGraphHelper_->rootIDs.begin(); id!=solnGraphHelper_->rootIDs.end(); ++id ){
            msg << solnGraphHelper_->exprFactory->get_labels(*id);
          }
      msg << endl << e.what() << endl
          << "*************************************************" << endl << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    //_____________________________________________________
    // add a task to advance each solution variable in time
    {
      Uintah::Task* updateTask = scinew Uintah::Task( "update solution vars_" + strRKStage.str(), this, &TimeStepper::update_variables, rkStage );

#     ifdef HAVE_CUDA
        //if( rhsDeviceTask ) updateTask->usesDevice( true );
#     endif

      const Uintah::PatchSubset* const pss = patches->getUnion();
      set_soln_field_requirements<SO::SVolField>( updateTask, scalarFields_, pss, mss, rkStage );
      set_soln_field_requirements<SO::XVolField>( updateTask, xVolFields_,   pss, mss, rkStage );
      set_soln_field_requirements<SO::YVolField>( updateTask, yVolFields_,   pss, mss, rkStage );
      set_soln_field_requirements<SO::ZVolField>( updateTask, zVolFields_,   pss, mss, rkStage );

      // we require the timestep value
      updateTask->requires( Uintah::Task::OldDW, sharedState_->get_delt_label() );
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
                                    Expr::ExpressionFactory* const factory,
                                    const int rkStage )
  {
    // grab the timestep
    Uintah::delt_vartype deltat;
    oldDW->get( deltat, sharedState_->get_delt_label() );

    const Expr::Tag timeTag = TagNames::self().time;
    //__________________
    // loop over patches
    for( int ip=0; ip<patches->size(); ++ip ){
      SetCurrentTime& settimeexpr = dynamic_cast<SetCurrentTime&>(
          factory->retrieve_expression( timeTag, patches->get(ip)->getID(), false ) );
      settimeexpr.set_integrator_stage( rkStage );
      settimeexpr.set_deltat( deltat );
      settimeexpr.set_time( sharedState_->getElapsedTime() );
      settimeexpr.set_timestep( sharedState_->getCurrentTopLevelTimeStep() );
    }
  }

  //------------------------------------------------------------------

  void
  TimeStepper::update_variables( const Uintah::Task::CallBackEvent event,
                                 const Uintah::ProcessorGroup* const pg,
                                 const Uintah::PatchSubset* const patches,
                                 const Uintah::MaterialSubset* const materials,
                                 Uintah::DataWarehouse* const oldDW,
                                 Uintah::DataWarehouse* const newDW,
                                 void* stream,
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
        oldDW->get( deltat, sharedState_->get_delt_label() );

        //____________________________________________
        // update variables on this material and patch
        // jcs note that we could do this in parallel
        do_update<SO::SVolField>( event, scalarFields_, patch, material, oldDW, newDW, deltat, rkStage, stream, timeInt_ );
        do_update<SO::XVolField>( event, xVolFields_,   patch, material, oldDW, newDW, deltat, rkStage, stream, timeInt_ );
        do_update<SO::YVolField>( event, yVolFields_,   patch, material, oldDW, newDW, deltat, rkStage, stream, timeInt_ );
        do_update<SO::ZVolField>( event, zVolFields_,   patch, material, oldDW, newDW, deltat, rkStage, stream, timeInt_ );

      } // material loop
    } // patch loop
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  TimeStepper::add_equation( const std::string& solnVarName,
                             const Expr::ExpressionID& rhsID )
  {
    const std::string rhsName = solnGraphHelper_->exprFactory->get_labels(rhsID)[0].name();
    const Uintah::TypeDescription* typeDesc = get_uintah_field_type_descriptor<FieldT>();
    const Uintah::IntVector ghostDesc       = get_uintah_ghost_descriptor<FieldT>();
    Uintah::VarLabel* const solnVarLabel = Uintah::VarLabel::create( solnVarName, typeDesc, ghostDesc );
    Uintah::VarLabel* const rhsVarLabel  = Uintah::VarLabel::create( rhsName,     typeDesc, ghostDesc );
    std::set< FieldInfo<FieldT> >& fields = field_info_selctor<FieldT>();
    fields.insert( FieldInfo<FieldT>( solnVarName, solnVarLabel, rhsVarLabel ) );
    createdVarLabels_.push_back( solnVarLabel );
    createdVarLabels_.push_back( rhsVarLabel );

    typedef Expr::PlaceHolder<FieldT>  FieldExpr;
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_N  )),true );
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_NP1)),true );
    
    postProcGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_NP1)),true );
  }

  //------------------------------------------------------------------

  template<>
  inline std::set< TimeStepper::FieldInfo<SpatialOps::structured::SVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::SVolField>()
  {
    return scalarFields_;
  }
  template<>
  inline std::set<TimeStepper::FieldInfo<SpatialOps::structured::XVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::XVolField>()
  {
    return xVolFields_;
  }
  template<>
  inline std::set<TimeStepper::FieldInfo<SpatialOps::structured::YVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::YVolField>()
  {
    return yVolFields_;
  }
  template<>
  inline std::set<TimeStepper::FieldInfo<SpatialOps::structured::ZVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::ZVolField>()
  {
    return zVolFields_;
  }

  //------------------------------------------------------------------

  template void TimeStepper::add_equation<SpatialOps::structured::SVolField>( const std::string&, const Expr::ExpressionID& );
  template void TimeStepper::add_equation<SpatialOps::structured::XVolField>( const std::string&, const Expr::ExpressionID& );
  template void TimeStepper::add_equation<SpatialOps::structured::YVolField>( const std::string&, const Expr::ExpressionID& );
  template void TimeStepper::add_equation<SpatialOps::structured::ZVolField>( const std::string&, const Expr::ExpressionID& );

} // namespace Wasatch
