#ifndef PointBirth_Expr_h
#define PointBirth_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PointBirth
 *  \author Alex Abboud
 *  \date January, 2012
 *
 *  \tparam FieldT the type of field.
 
 *  Nucleation Source term for use in QMOM

 */
template< typename FieldT >
class PointBirth
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag phiTag_;
  const double momentOrder_; // this is the order of the moment equation in which the growth rate is used
  const FieldT* phi_; // this will correspond to m(k+1)
  const double birthRate_;

  PointBirth( const Expr::Tag& phiTag,
              const double birthRate,
              const double momentOrder );

 public:
  class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::Tag& result,
               const Expr::Tag& phiTag,
               const double birthRate,
	             const double momentOrder )
	: ExpressionBuilder(result),
	phit_(phiTag),
  birthrate_(birthRate),    
	momentorder_(momentOrder)
	  {}
    
    
      ~Builder(){}
    
      Expr::ExpressionBase* build() const
      {
        return new PointBirth<FieldT>( phit_, birthrate_, momentorder_ );
      }
    
    private:
      const Expr::Tag phit_;
      const double birthrate_;
      const double momentorder_;
    };
  
  ~PointBirth();
  
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
PointBirth<FieldT>::
PointBirth( const Expr::Tag& phiTag,
            const double birthRate,
            const double momentOrder )
: Expr::Expression<FieldT>(),
  phiTag_(phiTag),
  birthRate_(birthRate),
  //momentOrder_(0.0)
  momentOrder_(momentOrder)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PointBirth<FieldT>::
~PointBirth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PointBirth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PointBirth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  /* add additional code here to bind any fields required by this expression */
  phi_ = &fm.field_ref( phiTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PointBirth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PointBirth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  double rknot = 5.1;
  //rknot = surf_eng*mol_val/R/T/log(S)
  double B = birthRate_;
  //B = J*exp(-16pi/3*(SurfEng/K_boltz/T)^3 * (MolecVol/log(S)/N_A)^2) 

  result <<= B * pow(rknot,momentOrder_);
 }

//--------------------------------------------------------------------

#endif // PointBirth_Expr_h


