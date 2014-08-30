#ifndef GradTExpr_h
#define GradTExpr_h

#include <expression/Expression.h>

/**
 *  @class  GradTExpr
 *  @author James C. Sutherland
 *  @date   May, 2007
 *  @brief  Calculates the temperature gradient.
 */
template< typename GradOp >
class GradTExpr
  : public Expr::Expression< typename GradOp::DestFieldType >
{
  typedef typename GradOp::SrcFieldType  ScalarFieldT; ///< the scalar field type implied by the gradient operator
  typedef typename GradOp::DestFieldType GradFieldT;   ///< the flux field type implied by the gradient operator

  GradTExpr( const Expr::Tag& tempTag );

  const Expr::Tag tempT_;

  const GradOp* gradOp_;
  const ScalarFieldT* temp_;

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

  virtual ~GradTExpr();

  /** @brief Constructs a GradTExpr object. */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const
    { return new GradTExpr(tempT_); }

    Builder( const Expr::Tag& gradTTag,
             const Expr::Tag& tempTag )
    : Expr::ExpressionBuilder(gradTTag),
      tempT_(tempTag)
    {}
    ~Builder(){}
  private:
    const Expr::Tag tempT_;
  };

};


//====================================================================


//--------------------------------------------------------------------
template<typename GradOp>
GradTExpr<GradOp>::
GradTExpr( const Expr::Tag& tempTag  )
  : Expr::Expression<GradFieldT>(),
    tempT_( tempTag )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}
//--------------------------------------------------------------------
template<typename GradOp>
void
GradTExpr<GradOp>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tempT_ );
}
//--------------------------------------------------------------------
template<typename GradOp>
void
GradTExpr<GradOp>::
bind_fields( const Expr::FieldManagerList& fml )
{
  temp_ = &fml.template field_ref<ScalarFieldT>( tempT_ );
}
//--------------------------------------------------------------------
template<typename GradOp>
void
GradTExpr<GradOp>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<GradOp>();
}
//--------------------------------------------------------------------
template<typename GradOp>
void
GradTExpr<GradOp>::
evaluate()
{
  gradOp_->apply_to_field( *temp_, this->value() );
}
//--------------------------------------------------------------------
template<typename GradOp>
GradTExpr<GradOp>::
~GradTExpr()
{}
//--------------------------------------------------------------------


//====================================================================


#endif
