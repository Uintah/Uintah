#ifndef SetCurrentTime_Expr_h
#define SetCurrentTime_Expr_h

#include <expression/Expression.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>


namespace Wasatch{

/**
 *  \class 	SetCurrentTime
 *  \ingroup 	Expressions
 *  \author 	James C. Sutherland
 *
 *  \brief Provides a simple expression to set the current simulation
 *         time.  May be needed for time-varying BCs, etc.
 */
class SetCurrentTime
 : public Expr::Expression<double>
{
  const Uintah::SimulationStateP state_;
  int RKStage_;
  double deltat_;

  SetCurrentTime( const Uintah::SimulationStateP sharedState,
                  const int RKStage );

public:
  int RKStage;

  class Builder : public Expr::ExpressionBuilder
  {
    const Uintah::SimulationStateP state_;
    const int RKStage_;

  public:
    Builder( const Expr::Tag& result,
             const Uintah::SimulationStateP sharedState,
             const int RKStage );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~SetCurrentTime();

  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}
  void evaluate();
  void set_integrator_stage( const int RKStage ){RKStage_ = RKStage;}
  void set_deltat( const double deltat ) {deltat_ = deltat;}
};




} // namespace Wasatch

#endif // SetCurrentTime_Expr_h
