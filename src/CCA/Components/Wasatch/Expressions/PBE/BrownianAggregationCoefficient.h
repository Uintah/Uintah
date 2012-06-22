#ifndef BrownianAggregationCoefficient_Expr_h
#define BrownianAggregationCoefficient_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class BrownianAggregationCoefficient
 *  \author Alex Abboud
 *  \date June 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief Calculates the coefficent used for brownian diffusion
 *  \f$ 2 k_B T / 3 \rho \f$
 *  K_B boltzmann const, T temeprature, \rho density
 */

template< typename FieldT >
class BrownianAggregationCoefficient
: public Expr::Expression<FieldT>
{
  const Expr::Tag densityTag_;
  const double coefVal_;
  const FieldT* density_; 
  
  BrownianAggregationCoefficient( const Expr::Tag& densityTag,
                                  const double coefVal );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& densityTag,
             const double coefVal )
    : ExpressionBuilder(result),
    densityt_(densityTag),
    coefval_(coefVal)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new BrownianAggregationCoefficient<FieldT>( densityt_, coefval_);
    }
    
  private:
    const Expr::Tag densityt_ ;
    const double coefval_;
  };
  
  ~BrownianAggregationCoefficient();
  
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
BrownianAggregationCoefficient<FieldT>::
BrownianAggregationCoefficient( const Expr::Tag& densityTag,
                                const double coefVal )
: Expr::Expression<FieldT>(),
densityTag_(densityTag),
coefVal_(coefVal)
{}

//--------------------------------------------------------------------

template< typename FieldT >
BrownianAggregationCoefficient<FieldT>::
~BrownianAggregationCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BrownianAggregationCoefficient<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( densityTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
BrownianAggregationCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  density_ = &fm.field_ref( densityTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
BrownianAggregationCoefficient<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BrownianAggregationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<=  coefVal_ / *density_ ;
}

//--------------------------------------------------------------------

#endif // BrownianAggregationCoefficient_Expr_h
