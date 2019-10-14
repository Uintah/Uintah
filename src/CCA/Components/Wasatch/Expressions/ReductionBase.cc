
#include "ReductionBase.h"
//-- Boost includes --//
#include <boost/foreach.hpp>

namespace so = SpatialOps;

ReductionBase::
ReductionBase( const Expr::Tag& resultTag,
               const Expr::Tag& srcTag,
               const ReductionEnum reductionName,
               const bool printVar )
: Expr::Expression<so::SingleValueField>(),
  srcTag_( srcTag ),
  reductionName_( reductionName ),
  printVar_(printVar)
{
  const std::string redVarName = resultTag.name() + "_uintah";
  Uintah::VarLabel* reductionVarLabel;
  switch (reductionName_) {
    case ReduceMin: reductionVarLabel = Uintah::VarLabel::create( redVarName, UintahReduceMin::getTypeDescription() ); break;
    case ReduceMax: reductionVarLabel = Uintah::VarLabel::create( redVarName, UintahReduceMax::getTypeDescription() ); break;
    case ReduceSum: reductionVarLabel = Uintah::VarLabel::create( redVarName, UintahReduceSum::getTypeDescription() ); break;
    default: break;
  }
  rkRedVarLbls_.push_back( reductionVarLabel );
  rkRedVarLbls_.push_back( Uintah::VarLabel::create(redVarName+"_2",reductionVarLabel->typeDescription() ) );
  rkRedVarLbls_.push_back( Uintah::VarLabel::create(redVarName+"_3",reductionVarLabel->typeDescription() ) );  
  thisVarLabel_ = Uintah::VarLabel::create( resultTag.name(), Uintah::PerPatch<double>::getTypeDescription() );
}

//--------------------------------------------------------------------

ReductionBase::
~ReductionBase()
{
  BOOST_FOREACH(Uintah::VarLabel* var, rkRedVarLbls_) {
    Uintah::VarLabel::destroy(var);
  }
  Uintah::VarLabel::destroy( thisVarLabel_ );
}

//--------------------------------------------------------------------

void
ReductionBase::
evaluate()
{}

//--------------------------------------------------------------------

void
ReductionBase::
schedule_set_reduction_vars( const Uintah::LevelP& level,
                             Uintah::SchedulerP sched,
                             const Uintah::MaterialSet* const materials,
                             const int RKStage )
{
  dbg_red << "scheduling set reduction vars for RK Stage: " << RKStage << std::endl;
  // create a Uintah task to populate the reduction variable with values calculated by this expression
  Uintah::Task* reductionVarTask = scinew Uintah::Task( "set reduction variables", this, &ReductionBase::populate_reduction_variable, RKStage );
  // we require this perpatch variable
  reductionVarTask->requires( Uintah::Task::NewDW, thisVarLabel_,
                              WasatchCore::get_uintah_ghost_type<so::SingleValueField>() );
  // we compute the reduction on this per patch variable
  reductionVarTask->computes( rkRedVarLbls_[RKStage - 1] ); // RKStage starts at 1
  sched->addTask( reductionVarTask,
                  sched->getLoadBalancer()->getPerProcessorPatchSet(level),
                  materials );
  
  dbg_red << "scheduling get reduction vars \n";
  // create a uintah task to get the reduced variables and put them back into this expression
  Uintah::Task* getReductionVarTask =
      scinew Uintah::Task( "get reduction variables",
                           this, &ReductionBase::get_reduction_variable,
                           RKStage );
  // get the reduction variable value
  getReductionVarTask->requires( Uintah::Task::NewDW, rkRedVarLbls_[RKStage - 1] );
  // modify the value for the perpatch expression
  getReductionVarTask->modifies( thisVarLabel_ );
  sched->addTask( getReductionVarTask,
                  sched->getLoadBalancer()->getPerProcessorPatchSet(level),
                  materials );
}

//--------------------------------------------------------------------

void
ReductionBase::
populate_reduction_variable( const Uintah::ProcessorGroup* const pg,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             Uintah::DataWarehouse* const oldDW,
                             Uintah::DataWarehouse* const newDW,
                             const int RKStage )
{
  dbg_red << "populating reduction variables \n";
  for( int ip=0; ip<patches->size(); ++ip ){
    const Uintah::Patch* const patch = patches->get(ip);
    for( int im=0; im<materials->size(); ++im ){
      Uintah::PerPatch<double> val;
      if( newDW->exists(thisVarLabel_,im,patch) ){
        newDW->get( val, thisVarLabel_, im, patch );        
        dbg_red << this->get_tag().name() << " patch " << patch->getID() << " val = " << val.get() << std::endl;
        Uintah::ReductionVariableBase* redcVar = NULL;
        
        switch (reductionName_) {
          case ReduceMin: redcVar = scinew UintahReduceMin(val.get()); break;
          case ReduceMax: redcVar = scinew UintahReduceMax(val.get()); break;
          case ReduceSum: redcVar = scinew UintahReduceSum(val.get()); break;
          default: break;
        }
        
        if (redcVar) {
          newDW->put(*redcVar,rkRedVarLbls_[RKStage - 1]);
          delete redcVar;
        }
        
      } else {
        std::cout << "Warning: reduction variable " << this->get_tag().name() << " not found! \n";
      }
    }
  }
}

//--------------------------------------------------------------------

void
ReductionBase::
get_reduction_variable( const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW,
                        const int RKStage)
{
  double reducedValue = 3.3;
  dbg_red << "getting reduction variables \n";
  // grab the reduced variable
  switch (reductionName_) {
    case ReduceMin:
    {
      UintahReduceMin val;
      newDW->get( val, rkRedVarLbls_[RKStage - 1] );
      reducedValue = val;
    }
      break;
    case ReduceMax:
    {
      UintahReduceMax val;
      newDW->get( val, rkRedVarLbls_[RKStage - 1] );
      reducedValue = val;
    }
      break;
    case ReduceSum:
    {
      UintahReduceSum val;
      newDW->get( val, rkRedVarLbls_[RKStage - 1] );
      reducedValue = val;
    }
      break;      
    default:
      break;
  }

  
  for( int ip=0; ip<patches->size(); ++ip ){
    const Uintah::Patch* const patch = patches->get(ip);
    for( int im=0; im<materials->size(); ++im ){
      
      if (newDW->exists(thisVarLabel_,im,patch)) {
        // now put the reduced variable back into this expression
        Uintah::PerPatch<double> perPatchVal;
        newDW->get(perPatchVal, thisVarLabel_, im, patch );
        perPatchVal.setData(reducedValue);
        //*( (double *) perPatchVal.getBasePointer()) = reducedValue;
        dbg_red << this->get_tag().name() << " expression value on patch " << patch->getID() << " after reduction = " << this->value()[0] << std::endl;
      } else {
        std::cout << "variable not found \n";
      }
    }
  }
  
  if ( printVar_ ) proc0cout << this->get_tag().name() << " = " << reducedValue << std::endl;
  
}

std::map<Expr::Tag, bool > ReductionBase::reductionTagList;
