#ifndef PrimVar_Expr_h
#define PrimVar_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \class 	PrimVar
 *  \ingroup 	Expressions
 *  \author 	James C. Sutherland
 *
 *  \brief given \f$\rho \phi\f$ and \f$\rho\f$, this calculates \f$\phi\f$.
 *
 *  \tparam FieldT - the type of field for \f$\rho\phi\f$ and \f$\phi\f$.
 *  \tparam DensT  - the type of field for \f$\rho\f$.
 *
 *   Note: it is currently assumed that \f$\rho\f$ is an SVolField
 *         type.  Therefore, no interpolation of the density occurs in
 *         that case.  In other cases, the density is interpolated to
 *         the location of \f$\rho\phi\f$.
 */
template< typename FieldT,
          typename DensT >
class PrimVar
 : public Expr::Expression<FieldT>
{
  const Expr::Tag rhophit_, rhot_;

  typedef typename OperatorTypeBuilder< Interpolant, DensT, FieldT >::type  InterpT;

  const DensT* rho_;
  const FieldT* rhophi_;
  const InterpT* interpOp_;

  PrimVar( const Expr::Tag& rhoPhiTag,
           const Expr::Tag& rhoTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoPhiTag,
             const Expr::Tag& rhoTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag rhophit_, rhot_;
  };

  ~PrimVar();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};



template< typename FieldT >
class PrimVar<FieldT,FieldT>
 : public Expr::Expression<FieldT>
{
  const Expr::Tag rhophit_, rhot_;

  const FieldT* rho_;
  const FieldT* rhophi_;

  PrimVar( const Expr::Tag& rhoPhiTag,
           const Expr::Tag& rhoTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoPhiTag,
             const Expr::Tag& rhoTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
  const Expr::Tag rhophit_, rhot_;
  };

  ~PrimVar();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

#endif // PrimVar_Expr_h
