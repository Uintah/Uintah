#ifndef PrecipitationRCritical_Expr_h
#define PrecipitationRCritical_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationRCritical
 *  \author Alex Abboud
 *  \date February 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the critical radius for Nucleation
 *  \f$ R^* = R_0 / \ln (S) \f$
 *
 */
template< typename FieldT >
class PrecipitationRCritical
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag superSatTag_;
  const double rKnotVal_;
  const FieldT* superSat_; //field from table of supersaturation
  const FieldT* eqConc_;   //field form table of equilibrium concentration
  
  PrecipitationRCritical( const Expr::Tag& superSatTag,
                          const double rKnotVal_);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const double rKnotVal)
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    rknotval_(rKnotVal)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new PrecipitationRCritical<FieldT>( supersatt_, rknotval_ );
    }
    
  private:
    const Expr::Tag supersatt_;
    const double rknotval_;
  };
  
  ~PrecipitationRCritical();
  
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
PrecipitationRCritical<FieldT>::
PrecipitationRCritical( const Expr::Tag& superSatTag,
                        const double rKnotVal)
: Expr::Expression<FieldT>(),
  superSatTag_(superSatTag),
  rKnotVal_(rKnotVal)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationRCritical<FieldT>::
~PrecipitationRCritical()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationRCritical<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationRCritical<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  superSat_ = &fm.field_ref( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationRCritical<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationRCritical<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( *superSat_ > 1.0, rKnotVal_ / log(*superSat_ ) )
                 ( 0.0 ); //this is r*
}

//--------------------------------------------------------------------

#endif // PrecipitationRCritical_Expr_h
