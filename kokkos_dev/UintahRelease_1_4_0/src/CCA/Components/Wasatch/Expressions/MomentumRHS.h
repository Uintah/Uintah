#ifndef MomentumRHS_Expr_h
#define MomentumRHS_Expr_h

#include <expression/Expr_Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>

/**
 *  \class MomRHS
 *  \ingroup Expressions
 *
 *  \brief Calculates the full momentum RHS
 *
 *  \tparam FieldT the type of field for the momentum RHS (nominally
 *          XVolField, YVolField, ZVolField).
 *
 *  The momentum RHS is split into two contributions:
 *   - the pressure term
 *   - the convective, diffusive, and body force terms
 *  These are calculated in the MomRHSPart and Pressure expressions, respectively.
 */
template< typename FieldT >
class MomRHS
 : public Expr::Expression<FieldT>
{
  typedef SpatialOps::structured::SVolField PFieldT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, PFieldT, FieldT >::type Grad;

  const Expr::Tag pressuret_, rhsPartt_, emptyTag_;

  const FieldT *rhsPart_;
  const PFieldT *pressure_;

  const Grad* gradOp_;

  MomRHS( const Expr::Tag& pressure,
          const Expr::Tag& partRHS,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag pressuret_, rhspt_;
  public:
    /**
     *  \param pressure the expression to compute the pressure as a scalar volume field
     *
     *  \param partRHS the expression to compute the other terms in
     *         the momentum RHS (body force, divergence of convection
     *         and stress)
     */
    Builder( const Expr::Tag& pressure,
             const Expr::Tag& partRHS );

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;
  };

  ~MomRHS();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // MomentumRHS_Expr_h
