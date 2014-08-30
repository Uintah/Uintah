#ifndef HeatFluxExpr_h
#define HeatFluxExpr_h

#include <sstream>

#include <expression/Expression.h>

//====================================================================


/**
 *  @class  HeatFluxExpr
 *  @author James C. Sutherland
 *  @date   May, 2007
 *
 * @brief Implements the expression for the Fourier heat flux,
 * \f$\mathbf{q}=-\lambda\nabla T \f$
 *
 *  @par Template Parameters
 *
 *   \li \b GradOp The type of operator for use in calculating
 *   \f$\nabla T\f$.  Currently, this mandates usage of the SpatialOps
 *   library.  However, we could try to make a general interface.
 *   Right now, the concrete operator type is obtained through a
 *   singleton class in SpatialOps.
 */
template< typename GradOp >
class HeatFluxExpr
  : public Expr::Expression< typename GradOp::DestFieldType >
{
  typedef typename GradOp::SrcFieldType   ScalarFieldT; ///< scalar field type implied by the gradient operator
  typedef typename GradOp::DestFieldType  GradFieldT;   ///< the flux field type implied by the gradient operator

  Expr::Tag thermCondT_, gradTT_;

  const GradFieldT *lambda_, *gradT_;

  HeatFluxExpr( const Expr::Tag& thermCondTag,
                const Expr::Tag& gradTTag );

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

  virtual ~HeatFluxExpr();


  /**
   *  @struct Builder
   *  @brief Constructs HeatFluxExpr objects.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& heatFluxTag,
             const Expr::Tag& thermCondTag,
             const Expr::Tag& gradTTag );
    ~Builder(){}
  private:
    const Expr::Tag tct_, gtt_;
  };

};


//====================================================================


//--------------------------------------------------------------------
template<typename GradOp>
HeatFluxExpr<GradOp>::
HeatFluxExpr( const Expr::Tag& thermCondTag,
              const Expr::Tag& gradTTag )
  : Expr::Expression<GradFieldT>(),
    thermCondT_( thermCondTag ),
    gradTT_    ( gradTTag     )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}
//--------------------------------------------------------------------
template<typename GradOp>
void
HeatFluxExpr<GradOp>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  using Expr::Tag;
  exprDeps.requires_expression( thermCondT_ );
  exprDeps.requires_expression( gradTT_     );
}
//--------------------------------------------------------------------
template<typename GradOp>
void
HeatFluxExpr<GradOp>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<GradFieldT>::type& fm = fml.template field_manager<GradFieldT>();
  lambda_ = &fm.field_ref( thermCondT_ );
  gradT_  = &fm.field_ref( gradTT_     );
}
//--------------------------------------------------------------------
template<typename GradOp>
void
HeatFluxExpr<GradOp>::
evaluate()
{
  using namespace SpatialOps;
  GradFieldT& result = this->value();
  result <<= -1.0* (*lambda_) * (*gradT_);
}
//--------------------------------------------------------------------
template<typename GradOp>
HeatFluxExpr<GradOp>::
~HeatFluxExpr()
{}

//--------------------------------------------------------------------

template<typename GradOp>
Expr::ExpressionBase*
HeatFluxExpr<GradOp>::Builder::
build() const
{
  return new HeatFluxExpr( tct_, gtt_ );
}
//--------------------------------------------------------------------
template<typename GradOp>
HeatFluxExpr<GradOp>::Builder::
Builder( const Expr::Tag& heatFluxTag,
         const Expr::Tag& thermCondTag,
         const Expr::Tag& gradTTag )
  : ExpressionBuilder( heatFluxTag ),
    tct_(thermCondTag), gtt_(gradTTag)
{}
//--------------------------------------------------------------------

#endif
