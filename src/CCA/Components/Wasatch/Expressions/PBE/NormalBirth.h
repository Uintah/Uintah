#ifndef NormalBirth_Expr_h
#define NormalBirth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class NormalBirth
 *  \author Alex Abboud
 *  \date January, 2012
 *
 *  \tparam FieldT the type of field.
 
 *  Nucleation Source term for use in QMOM

 */
template< typename FieldT >
class NormalBirth
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag phiTag_, growthCoefTag_;
  //  const double growthCoefVal_;
  const double momentOrder_; // this is the order of the moment equation in which the growth rate is used
  const double sig; 
  const bool isUniBirth_;
  const FieldT* phi_; // this will correspond to m(k+1)
  //  const FieldT* growthCoef_; // this will correspond to the coefficient in the growth rate term
  
  NormalBirth( const Expr::Tag& phiTag,
		    const Expr::Tag& growthCoefTag,
		    const double momentOrder );
  
  NormalBirth( const Expr::Tag& phiTag,
		    const double growthCoefVal,
		    const double momentOrder );
  
  
 public:
  class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::Tag& result,
	       const Expr::Tag& phiTag,
	       const Expr::Tag& growthCoefTag,
	       const double momentOrder )
	: ExpressionBuilder(result),
	phit_(phiTag),
	growthcoeft_(growthCoefTag),
	growthcoefval_(0.0),
	momentorder_(momentOrder),
	isconstcoef_( false )
	  {}
    
      Builder( const Expr::Tag& result,
	       const Expr::Tag& phiTag,
	       const double growthCoefVal,
	       const double momentOrder )
	: ExpressionBuilder(result),
	phit_(phiTag),
	growthcoefval_(growthCoefVal),
	momentorder_(momentOrder),
	isconstcoef_( true )
	  {}
    
      ~Builder(){}
    
      Expr::ExpressionBase* build() const
	{
	  if (isconstcoef_) return new NormalBirth<FieldT>( phit_, growthcoefval_, momentorder_ );
	  else              return new NormalBirth<FieldT>( phit_, growthcoeft_,   momentorder_ );
	}
    
    private:
      const Expr::Tag phit_, growthcoeft_;
      const double growthcoefval_;
      const double momentorder_;
      const bool isconstcoef_;
    };
  
  ~NormalBirth();
  
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
NormalBirth<FieldT>::
NormalBirth( const Expr::Tag& phiTag,
		  const Expr::Tag& growthCoefTag,
		  const double momentOrder )
: Expr::Expression<FieldT>(),
  phiTag_(phiTag),
  momentOrder_(momentOrder),
{}


//--------------------------------------------------------------------

template< typename FieldT >
NormalBirth<FieldT>::
~NormalBirth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
NormalBirth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );

}

//--------------------------------------------------------------------

template< typename FieldT >
void
NormalBirth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  /* add additional code here to bind any fields required by this expression */
  phi_ = &fm.field_ref( phiTag_ );

}

//--------------------------------------------------------------------

template< typename FieldT >
void
NormalBirth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
NormalBirth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
 
  double rknot = 5.1;
  //rknot = surf_eng*mol_val/R/T/log(S)
  
  std::vector <double> x;
  double dx;
  double npts = 10;
  //set up in constuction ?
  x = vector<double>(npts);
  dx = (RMax-Rmin)/npts;
  x[0] = Rmin;
  for(i =1; i<npts; i++) {
    x[i] = x[i-1] + dx;
  }  

  double IntVal = 0.0;
  for(i =0, i<npts-1, i++) {
    //.399 ~ 1/sqrt(2pi)
    IntVal = IntVal + dx/2/.399*( pow(x[i],momentOrder_)* exp(-sig/2 * (x[i] - rknot) * (x[i] - knot)) +
                                  pow(x[i+1],momentOrder_) * exp(-sig/2 * (x[i+1] - rknot) * (x[i+1] - knot)) );
  }
  result <<= IntVal;
  
}

//--------------------------------------------------------------------

#endif // NormalBirth_Expr_h


