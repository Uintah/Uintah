
#ifndef Birth_Expr_h
#define Birth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class Birth
 *  \author Alex Abboud	
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 
 *  \brief Implements any type of birth term
 *  
 */
template< typename FieldT >
class Birth
: public Expr::Expression<FieldT>
{
  const Expr::Tag birthTypeTag_, birthCoefTag_;  //this will correspond to proper tags for constant calc & momnet dependency
  const double momentOrder_; // this is the order of the moment equation in which the Birth rate is used
  const double constCoef_;
  const FieldT* birthTypeField_; // this will correspond to m(k + x) x depends on which Birth model
  const FieldT* birthCoef_; // this will correspond to the coefficient in the Birth rate term
  
  Birth( const Expr::Tag& birthTypeTag,
         const Expr::Tag& birthCoefTag,
         const double momentOrder, 
         const double constCoef);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& birthTypeTag,
            const Expr::Tag& birthCoefTag,
            const double momentOrder,
            const double constCoef)
    : ExpressionBuilder(result),
    birthtypet_ (birthTypeTag),
    birthcoeft_ (birthCoefTag),
    momentorder_(momentOrder),
    constcoef_  (constCoef)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new Birth<FieldT>( birthtypet_, birthcoeft_, momentorder_, constcoef_ );
    }
    
  private:
    const Expr::Tag birthtypet_, birthcoeft_;
    const double momentorder_;
    const double constcoef_;
  };
  
  ~Birth();
  
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
Birth<FieldT>::
Birth( const Expr::Tag& birthTypeTag,
       const Expr::Tag& birthCoefTag,
       const double momentOrder, 
       const double constCoef)
: Expr::Expression<FieldT>(),
  birthTypeTag_(birthTypeTag),
  birthCoefTag_(birthCoefTag),
  momentOrder_ (momentOrder),
  constCoef_   (constCoef)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Birth<FieldT>::
~Birth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( birthTypeTag_ );
  if ( birthCoefTag_ != Expr::Tag () )
    exprDeps.requires_expression( birthCoefTag_ );  
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();

  birthTypeField_ = &fm.field_ref( birthTypeTag_ );
  if ( birthCoefTag_ != Expr::Tag () )
    birthCoef_ = &fm.field_ref( birthCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Birth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  //add iterator in other loops for S check
  
  if ( birthCoefTag_ != Expr::Tag () ) {
      result <<= constCoef_ * *birthCoef_ * *birthTypeField_; 
  } else {
      result <<= constCoef_ * *birthTypeField_; 
  }
}

#endif
