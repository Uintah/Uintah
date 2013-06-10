#ifndef StableTimestep_Expr_h
#define StableTimestep_Expr_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>

/**
 *  \class 	StableTimestep
 *  \author 	Tony Saad
 *  \date 	 June 2013
 *  \ingroup	Expressions
 *
 *  \brief Calculates a stable timestep based on the momemntum equations using
 common CFD criteria.
 *
 */
class StableTimestep
 : public Expr::Expression<double>
{
  const Expr::Tag rhoTag_, viscTag_, uTag_, vTag_, wTag_;
  bool doX_, doY_, doZ_, isViscous_;
  const SVolField* rho_;
  const SVolField* visc_;
  const XVolField* u_;
  const YVolField* v_;
  const ZVolField* w_;
  
  double invDx_, invDy_, invDz_;
  
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type X2SOpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Y2SOpT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Z2SOpT;
  const X2SOpT* x2SInterp_;
  const Y2SOpT* y2SInterp_;
  const Z2SOpT* z2SInterp_;

  // gradient operators are only here to extract spacing information out of them
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;

  /* declare operators associated with this expression here */

    StableTimestep( const Expr::Tag& rhoTag,
                const Expr::Tag& viscTag,
                const Expr::Tag& uTag,
                const Expr::Tag& vTag,
                const Expr::Tag& wTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a StableTimestep expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& rhoTag,
             const Expr::Tag& viscTag,
             const Expr::Tag& uTag,
             const Expr::Tag& vTag,
             const Expr::Tag& wTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag rhoTag_, viscTag_, uTag_, vTag_, wTag_;
  };

  ~StableTimestep();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  double get_stable_dt();
};

#endif // StableTimestep_Expr_h
