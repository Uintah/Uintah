#ifndef MonosurfaceGrowth_Expr_h
#define MonosurfaceGrowth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expr_Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class MonosurfaceGrowth
 *  \author Tony Saad
 *  \date July, 2011
 *
 *  \brief Implements the moment integral of the monosurface growth model of the form
 *         \f$ G = g_0(\mathbf{x},S,t)r^2 \f$.
 *
 *  \tparam FieldT the type of field that this expression evaluates
 *
 *   In the context of the moments method, this expression is part of
 *   the convection term in internal coordinates.  Let
 *   \f$\eta\equiv\eta(r;\mathbf{x},t),\f$ where \f$r\f$ is an internal
 *   coordinate and \f$\mathbf{x}\f$ and \f$ t \f$ are spatial and
 *   time coordinates. The convection term in internal coordinates
 *   takes the form
 *    \f[
 *      \frac{\partial G\eta}{\partial r} =
 *      g_0(\mathbf{x},S,t)\frac{\partial r^2 \eta}{\partial r}.
 *    \f]
 *   Upon integration, we have
 *    \f[
 *      \int_{-\infty}^{\infty) r^k g_0 \frac{\partial r^2 \eta}{\partial r}
 *      \, \mathrm{d}r = k g_0 m_{k+1},
 *    \f]
 *   where \f$ k \f$ corresponds to the \f$k^{th}\f$ moment.
 */
template< typename FieldT >
class MonosurfaceGrowth
 : public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag phiTag_, growthCoefTag_;  
  const double growthCoefVal_;  
  const double momentOrder_; // this is the order of the moment equation in which the growth rate is used  
  const bool isConstCoef_;
  const FieldT* phi_; // this will correspond to m(k+1)
  const FieldT* growthCoef_; // this will correspond to the coefficient in the growth rate term

  MonosurfaceGrowth( const Expr::Tag phiTag,
                     const Expr::Tag growthCoefTag,
                     const double momentOrder,
                     const Expr::ExpressionID& id,
                     const Expr::ExpressionRegistry& reg  );

  MonosurfaceGrowth( const Expr::Tag phiTag,
                     const double growthCoefVal,
                     const double momentOrder,
                     const Expr::ExpressionID& id,
                     const Expr::ExpressionRegistry& reg  );
  

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder(const Expr::Tag phiTag, 
            const Expr::Tag growthCoefTag, 
            const double momentOrder )
      : phit_(phiTag),
        growthcoeft_(growthCoefTag),
		    growthcoefval_(0.0),
        momentorder_(momentOrder),
				isconstcoef_( false )
    {}
    
    Builder( const Expr::Tag phiTag, const double growthCoefVal, const double momentOrder )
      : phit_(phiTag),
        growthcoefval_(growthCoefVal),
        momentorder_(momentOrder),
				isconstcoef_( true )    
    {}
    

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      if (isconstcoef_) return new MonosurfaceGrowth<FieldT>( phit_, growthcoefval_, momentorder_, id, reg);
      else 										return new MonosurfaceGrowth<FieldT>( phit_, growthcoeft_, momentorder_, id, reg);
    }

  private:
    const Expr::Tag phit_, growthcoeft_; 
    const double growthcoefval_;
    const double momentorder_;
    const bool isconstcoef_;
  };

  ~MonosurfaceGrowth();

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



template< typename FieldT >
MonosurfaceGrowth<FieldT>::
MonosurfaceGrowth( const Expr::Tag phiTag,
                   const Expr::Tag growthCoefTag,
                   const double momentOrder,
                   const Expr::ExpressionID& id,
                   const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FieldT>(id,reg),
    phiTag_(phiTag),
    growthCoefTag_(growthCoefTag),
    growthCoefVal_(0.0),
    momentOrder_(0.0),
    isConstCoef_( false )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MonosurfaceGrowth<FieldT>::
MonosurfaceGrowth( const Expr::Tag phiTag,
                   const double growthCoefVal,
                   const double momentOrder,
                   const Expr::ExpressionID& id,
                   const Expr::ExpressionRegistry& reg  )
: Expr::Expression<FieldT>(id,reg),
  phiTag_(phiTag),
  growthCoefTag_("NULL", Expr::INVALID_CONTEXT),
  growthCoefVal_(growthCoefVal),
  momentOrder_(momentOrder),
  isConstCoef_( true )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MonosurfaceGrowth<FieldT>::
~MonosurfaceGrowth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonosurfaceGrowth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( !isConstCoef_ ) exprDeps.requires_expression( growthCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonosurfaceGrowth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  phi_ = &fm.field_ref( phiTag_ );
  if( !isConstCoef_ ) growthCoef_ = &fm.field_ref( growthCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonosurfaceGrowth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonosurfaceGrowth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if( isConstCoef_ ){
    result <<= momentOrder_ * growthCoefVal_ * *phi_;  // G = g0 * m_{k+1}
  }
  else{ // constant coefficient
    result <<= momentOrder_ * *growthCoef_ * *phi_;  // G =  g0 * m_{k+1}
  }
}

//--------------------------------------------------------------------

#endif // MonosurfaceGrowth_Expr_h
