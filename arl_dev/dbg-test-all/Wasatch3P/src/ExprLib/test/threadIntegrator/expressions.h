#ifndef expressions_h
#define expressions_h

#include <expression/Expression.h>

//====================================================================

template<typename FieldT>
class C1 : public Expr::Expression<FieldT>
{
public:

  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( Expr::Tag( "C1", this->get_tag().context() ) );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    c1_ = &fml.field_manager<FieldT>().field_ref( Expr::Tag("C1", this->get_tag().context()) );
  }

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& rhs = this->value();
    rhs <<= -k_*(*c1_);
  }


  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const double k )
    : ExpressionBuilder(result),
      k_(k)
    {}
    Expr::ExpressionBase* build() const{ return new C1<FieldT>( k_ ); }
  private:
    const double k_;
  };

private:

  C1( const double k )
    : Expr::Expression<FieldT>(),
      k_(k)
  {
#   ifdef ENABLE_CUDA
    this->set_gpu_runnable( true );
#   endif
  }

  const double k_;
  const FieldT *c1_;
};


//====================================================================


template<typename FieldT>
class C2 : public Expr::Expression<FieldT>
{
public:

  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( Expr::Tag( "C1", this->get_tag().context() ) );
  }

  void bind_fields( const Expr::FieldManagerList& fml )
  {
    c1_ = &fml.field_manager<FieldT>().field_ref( Expr::Tag("C1", this->get_tag().context()) );
  }

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& rhs = this->value();
    rhs <<= k_ * *c1_;
  }

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const{ return new C2<FieldT>( k_ ); }
    Builder( const Expr::Tag& result, const double k ) : ExpressionBuilder(result), k_(k) {}
  private:
    const double k_;
  };

private:

  C2( const double k )
    : Expr::Expression<FieldT>(),
      k_(k)
  {
#   ifdef ENABLE_CUDA
    this->set_gpu_runnable( true );
#   endif
}

  const double k_;
  const FieldT *c1_;
};

//====================================================================

#endif // expressions_h
