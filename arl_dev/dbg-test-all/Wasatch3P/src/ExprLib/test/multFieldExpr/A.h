#ifndef A_Expr_h
#define A_Expr_h

#include <expression/ExprLib.h>


/**
 *  \class A
 */
class A
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  const SpatialOps::SingleValueField* b_;
  const SpatialOps::SingleValueField* c_;
  Expr::Tag btag_, ctag_;

  A( const Expr::Tag& b,
     const Expr::Tag& c );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag b_, c_;
  public:
    Builder( const Expr::Tag& a, const Expr::Tag& b, const Expr::Tag& c )
    : ExpressionBuilder(a),
      b_(b), c_(c)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new A(b_,c_); }
  };

  ~A(){}

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



A::A( const Expr::Tag& b,
      const Expr::Tag& c )
  : Expr::Expression<SpatialOps::SingleValueField>(),
    btag_( b ),
    ctag_( c )
{}

//--------------------------------------------------------------------

void
A::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( btag_ );
  exprDeps.requires_expression( ctag_ );
}

//--------------------------------------------------------------------

void
A::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fm = fml.field_manager<SpatialOps::SingleValueField>();
  b_ = &fm.field_ref( btag_ );
  c_ = &fm.field_ref( ctag_ );
}

//--------------------------------------------------------------------

void
A::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

void
A::evaluate()
{
  using namespace SpatialOps;
  SingleValueField& a = this->value();
  a <<= *b_ + *c_;
}


#endif // A_Expr_h
