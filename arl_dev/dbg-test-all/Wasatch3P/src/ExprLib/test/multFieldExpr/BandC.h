#ifndef BandC_Expr_h
#define BandC_Expr_h

#include <expression/Expression.h>


class BandC
  : public Expr::Expression<SpatialOps::SingleValueField>
{
  BandC();

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& bc ) : ExpressionBuilder(bc) {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new BandC(); }
  };

  ~BandC();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // BandC_Expr_h
