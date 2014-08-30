#ifndef expressions_h
#define expressions_h

#include <expression/Expression.h>

//====================================================================

class TestExpr : public Expr::Expression<SpatialOps::SingleValueField>
{
public:
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}

  void bind_fields( const Expr::FieldManagerList& fml )
  {}

  void evaluate(){
    using namespace SpatialOps;
    this->value() <<= k_;
  }

  ~TestExpr(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{
      return new TestExpr(k_);
    }
    Builder( const Expr::Tag& name, const double value )
    : ExpressionBuilder(name),
      k_(value)
    {}
    ~Builder(){}
  private:
    const double k_;
  };

protected:
  TestExpr( const double value )
    : Expr::Expression<SpatialOps::SingleValueField>(),
      k_( value )
  {}
  const double k_;
};

//====================================================================

#endif
