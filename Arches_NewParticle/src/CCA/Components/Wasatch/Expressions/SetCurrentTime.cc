#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>

#include <Core/Grid/SimulationState.h>

namespace Wasatch{

  SetCurrentTime::SetCurrentTime( const Uintah::SimulationStateP sharedState,
                                  const Expr::ExpressionID& id,
                                  const Expr::ExpressionRegistry& reg  )
    : Expr::Expression<double>( id, reg ),
      state_( sharedState )
  {}

  //--------------------------------------------------------------------

  SetCurrentTime::~SetCurrentTime()
  {}

  //--------------------------------------------------------------------

  void
  SetCurrentTime::evaluate()
  {
    double& result = this->value();
    result = state_->getElapsedTime();
  }

  //--------------------------------------------------------------------

  SetCurrentTime::Builder::Builder( const Uintah::SimulationStateP sharedState )
    : state_( sharedState )
  {}

  //--------------------------------------------------------------------

  Expr::ExpressionBase*
  SetCurrentTime::Builder::build( const Expr::ExpressionID& id,
                                  const Expr::ExpressionRegistry& reg ) const
  {
    return new SetCurrentTime( state_, id, reg );
  }

} // namespace Wasatch
