#ifndef PrecipitationMonosurfaceCoefficient_Expr_h
#define PrecipitationMonosurfaceCoefficient_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/*
 *  \ingroup WasatchExpressions
 *  \class PrecipitationMonosurfaceCoefficient
 *  \author Alex Abboud
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 
 *  \brief calculates the expression containing the coefficient used in a
 *  precipitation reaction with monosurface nucleation growth
 *  g0 = B D d^3 exp(    
 *  g(r) = r^2
 */
template< typename FieldT >
class PrecipitationMonosurfaceCoefficient
: public Expr::Expression<FieldT>
{
  const Expr::Tag superSatTag_;
  const double growthCoefVal_;
  const double expConst_;
  const FieldT* superSat_; //field from table of supersaturation
  
  PrecipitationMonosurfaceCoefficient( const Expr::Tag& superSatTag,
                                       const double growthCoefVal,
                                       const double expConst);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& superSatTag,
            const double growthCoefVal,
            const double expConst)
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    growthcoefval_(growthCoefVal),
    expconst_(expConst)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new PrecipitationMonosurfaceCoefficient<FieldT>( supersatt_,  growthcoefval_, expconst_ );
    }
    
  private:
    const Expr::Tag supersatt_;
    const double growthcoefval_;
    const double expconst_;
  };
  
  ~PrecipitationMonosurfaceCoefficient();
  
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
PrecipitationMonosurfaceCoefficient<FieldT>::
PrecipitationMonosurfaceCoefficient( const Expr::Tag& superSatTag,
                                     const double growthCoefVal,
                                     const double expConst)
: Expr::Expression<FieldT>(),
superSatTag_(superSatTag),
growthCoefVal_(growthCoefVal),
expConst_(expConst)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationMonosurfaceCoefficient<FieldT>::
~PrecipitationMonosurfaceCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationMonosurfaceCoefficient<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationMonosurfaceCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  superSat_ = &fm.field_ref( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationMonosurfaceCoefficient<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationMonosurfaceCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= growthCoefVal_ * exp(expConst_ /  log(*superSat_ + 1.01) );  // this is g0
  //note: + 1.01 added to precent NaN for now
}

//--------------------------------------------------------------------

#endif // PrecipitationMonosurfaceCoefficient_Expr_h
