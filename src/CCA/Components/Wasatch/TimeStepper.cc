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
#include <CCA/Components/Wasatch/CoordinateHelper.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <CCA/Components/Wasatch/Expressions/TimeAdvance.h>


//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

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
  create_time_advance_expressions( const std::set< TimeStepper::FieldInfo<FieldT> >& fields,
                                   GraphHelper* gh,
                                   const TimeIntegrator timeInt )
  {
    typedef typename TimeAdvance<FieldT>::Builder TimeAdvBuilder;
    typedef typename std::set< TimeStepper::FieldInfo<FieldT> > Fields;
    for( typename Fields::const_iterator ifld = fields.begin(); ifld!=fields.end(); ++ifld ){
      if (!gh->exprFactory->have_entry(ifld->solnVarTag)) {
        const Expr::ExpressionID id = gh->exprFactory->register_expression( scinew TimeAdvBuilder(ifld->solnVarTag, ifld->rhsTag, timeInt ) );
        gh->rootIDs.insert(id);
        //      gh->exprFactory->cleave_from_children(id);
      }
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
    std::stringstream strRKStage;
    strRKStage << rkStage;

    // need to explicitly make all RHS fields persistent.  This avoids the situation
    // where they may be internal nodes in a graph and could thus turn into "temporary"
    // fields, leading to non-exposure to Uintah and bad things...
    std::set<std::string> persistentFields( ioFieldSet );
    for( ScalarFields::const_iterator i=scalarFields_.begin(); i!=scalarFields_.end(); ++i )  persistentFields.insert( i->rhsTag.name() );
    for(   XVolFields::const_iterator i=  xVolFields_.begin(); i!=  xVolFields_.end(); ++i )  persistentFields.insert( i->rhsTag.name() );
    for(   YVolFields::const_iterator i=  yVolFields_.begin(); i!=  yVolFields_.end(); ++i )  persistentFields.insert( i->rhsTag.name() );
    for(   ZVolFields::const_iterator i=  zVolFields_.begin(); i!=  zVolFields_.end(); ++i )  persistentFields.insert( i->rhsTag.name() );


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
    try{
      // jcs for multistage integrators, we may need to keep the same
      //     field manager list for all of the stages?  Otherwise we
      //     will have all sorts of name clashes?

      // plug in time advance expression
      if (rkStage == 1) {
        create_time_advance_expressions<SO::SVolField>(scalarFields_, solnGraphHelper_, timeInt_);
        create_time_advance_expressions<SO::XVolField>(xVolFields_  , solnGraphHelper_, timeInt_);
        create_time_advance_expressions<SO::YVolField>(yVolFields_  , solnGraphHelper_, timeInt_);
        create_time_advance_expressions<SO::ZVolField>(zVolFields_  , solnGraphHelper_, timeInt_);
      }
      
      TaskInterface* rhsTask = scinew TaskInterface( solnGraphHelper_->rootIDs,
                                                     "rhs_" + strRKStage.str(),
                                                     *(solnGraphHelper_->exprFactory),
                                                     level, sched, patches, materials,
                                                     patchInfoMap,
                                                     rkStage, sharedState_, persistentFields );

      taskInterfaceList_.push_back( rhsTask );
      rhsTask->schedule( rkStage ); // must be scheduled after coordHelper_
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
      settimeexpr.set_deltat  ( deltat );
      settimeexpr.set_time    ( sharedState_->getElapsedTime() );
      settimeexpr.set_timestep( sharedState_->getCurrentTopLevelTimeStep() );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  TimeStepper::add_equation( const std::string& solnVarName,
                             const Expr::ExpressionID& rhsID )
  {
    const std::string rhsName = solnGraphHelper_->exprFactory->get_labels(rhsID)[0].name();

    const Expr::Tag solnVarTag(solnVarName,Expr::STATE_NONE);
    const Expr::Tag rhsVarTag (rhsName,    Expr::STATE_NONE);
    
    std::set< FieldInfo<FieldT> >& fields = field_info_selctor<FieldT>();
    fields.insert( FieldInfo<FieldT>( solnVarName, solnVarTag, rhsVarTag ) );

    typedef Expr::PlaceHolder<FieldT>  FieldExpr;
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_N      )), true );
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_NP1    )), true );
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_DYNAMIC)), true );
    
    postProcGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_NP1)), true );    
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
