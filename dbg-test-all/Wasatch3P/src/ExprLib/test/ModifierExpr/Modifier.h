#ifndef Modifier_Expr_h
#define Modifier_Expr_h

#include <expression/Expression.h>

template< typename FieldT >
class Modifier
 : public Expr::Expression<FieldT>
{
  const Expr::Tag tag_;
  const FieldT* f_;

  Modifier( const Expr::Tag tag ) : tag_(tag)
  {
#   ifdef ENABLE_CUDA
    this->set_gpu_runnable( true );
#   endif
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag, const Expr::Tag& tag ) : ExpressionBuilder(resultTag), tag_(tag) {}
    Expr::ExpressionBase* build() const{ return new Modifier(tag_); }
  private:
    const Expr::Tag tag_;
  };

  ~Modifier(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){ exprDeps.requires_expression(tag_);   }
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
Modifier<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  f_ = &fm.field_ref(tag_);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Modifier<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= result + *f_;
}

#endif // Modifier_Expr_h
