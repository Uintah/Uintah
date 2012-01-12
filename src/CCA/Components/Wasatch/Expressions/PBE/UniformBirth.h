#ifndef UniformBirth_Expr_h
#define UniformBirth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class UniformBirth
 *  \author Alex Abboud
 *  \date January, 2012
 *
 *  \tparam FieldT the type of field.
 
 *  Nucleation Source term for use in QMOM

 */
template< typename FieldT >
class UniformBirth
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag phiTag_;
  const double momentOrder_; // this is the order of the moment equation in which the growth rate is used
  const double sig; 
  const FieldT* phi_; // this will correspond to m(k+1)
  
  UniformBirth( const Expr::Tag& phiTag,
                const double birthrate,
                const double sig,
		            const double momentOrder );
  
 public:
  class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::Tag& result,
	       const Expr::Tag& phiTag,
	       const double momentOrder, 
         const double sig     )
	: ExpressionBuilder(result),
	phit_(phiTag),
  sig_(sig),    
	momentorder_(momentOrder)
	  {}
    
      ~Builder(){}
    
      Expr::ExpressionBase* build() const
	{
	  return new UniformBirth<FieldT>( phit_, momentorder_ );
	  
	}
    
    private:
      const Expr::Tag phit_;
      const double momentorder_;
    };
  
  ~UniformBirth();
  
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
UniformBirth<FieldT>::
UniformBirth( const Expr::Tag& phiTag,
		          const double momentOrder )
: Expr::Expression<FieldT>(),
  phiTag_(phiTag),
  momentOrder_(momentOrder),
{}

//--------------------------------------------------------------------

template< typename FieldT >
UniformBirth<FieldT>::
~UniformBirth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformBirth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
//  if( !isUniBirth_ ) exprDeps.requires_expression( growthCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformBirth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  /* add additional code here to bind any fields required by this expression */
  phi_ = &fm.field_ref( phiTag_ );

}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformBirth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformBirth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  double rknot = 5.1;
  //rknot = surf_eng*mol_val/R/T/log(S)
  double B = 1.0;
  //B = J*exp(-16pi/3*(SurfEng/K_boltz/T)^3 * (MolecVol/log(S)/N_A)^2)

  //int_a^b  B * r^k  dr
  double a;
  double b;
  a = rknot - sig;
  b = rknot + sig;
  result <<= B/(momentOrder_+1)*(pow(b,momentOrder_+1)-pow(a,momentOrder_+1)); 
}

//--------------------------------------------------------------------

#endif // UniformBirth_Expr_h


