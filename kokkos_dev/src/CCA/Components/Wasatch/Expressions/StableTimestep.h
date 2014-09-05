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
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  const Expr::Tag rhoTag_, viscTag_, uTag_, vTag_, wTag_;
  double invDx_, invDy_, invDz_; // 1/dx, 1/dy, 1/dz
  const bool doX_, doY_, doZ_, isViscous_;
  const bool is3dconvdiff_;
  const SVolField* rho_;
  const SVolField* visc_;
  const XVolField* u_;
  const YVolField* v_;
  const ZVolField* w_;
  
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type X2SOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type Y2SOpT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type Z2SOpT;
  const X2SOpT* x2SInterp_;
  const Y2SOpT* y2SInterp_;
  const Z2SOpT* z2SInterp_;

  // gradient operators are declared here to extract grid-spacing information
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, XVolField, SVolField >::type GradXT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, YVolField, SVolField >::type GradYT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, ZVolField, SVolField >::type GradZT;
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;

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
};

#endif // StableTimestep_Expr_h
