#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>

namespace Wasatch{

  SetCurrentTime::SetCurrentTime( const Uintah::SimulationStateP sharedState,
                                  const int RKStage )
    : Expr::Expression<double>(),
      state_( sharedState ),
      RKStage_( RKStage )
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
    if (RKStage_ == 2) result += deltat_;
    if (RKStage_ == 3) result += deltat_*0.5;
    //std::cout<<"Current Simulation Time: " << result << " STAGE = "<< RKStage_	 << std::endl;
  }

  //--------------------------------------------------------------------

  SetCurrentTime::Builder::Builder( const Expr::Tag& result,
                                    const Uintah::SimulationStateP sharedState,
                                    const int RKStage )
    : ExpressionBuilder(result),
      state_( sharedState ),
      RKStage_(RKStage)
  {}

  //--------------------------------------------------------------------

  Expr::ExpressionBase*
  SetCurrentTime::Builder::build() const
  {
    return new SetCurrentTime( state_, RKStage_ );
  }

} // namespace Wasatch
