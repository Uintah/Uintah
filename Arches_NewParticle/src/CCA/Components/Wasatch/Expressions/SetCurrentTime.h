#ifndef SetCurrentTime_Expr_h
#define SetCurrentTime_Expr_h

#include <expression/Expr_Expression.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>


namespace Wasatch{

/**
 *  \class SetCurrentTime
 */
class SetCurrentTime
 : public Expr::Expression<double>
{
  const Uintah::SimulationStateP state_;

  SetCurrentTime( const Uintah::SimulationStateP sharedState,
                  const Expr::ExpressionID& id,
                  const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Uintah::SimulationStateP state_;
  public:
    Builder( const Uintah::SimulationStateP sharedState );
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
  };

  ~SetCurrentTime();

  void advertise_dependents( Expr::ExprDeps& exprDeps ){}

  void bind_fields( const Expr::FieldManagerList& fml ){}

  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}

  void evaluate();

};




} // namespace Wasatch

#endif // SetCurrentTime_Expr_h
