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
 *
 *  \brief Nucleation Source term for use in QMOM
 *  uses Dirac function for birht term b = del(r-r*)
 *  TODO: (alex) add latex math equations
 */
template< typename FieldT >
class PointBirth
: public Expr::Expression<FieldT>
{
  // be careful on what phi (or moment) you pass to this expression
  const Expr::Tag phiTag_, rStarTag_;
  const double momentOrder_;  // this is the order of the moment equation in which the growth rate is used
  const FieldT* phi_;         // this checks is m_k =/= 0
  const FieldT* rStar_;
  const double constRStar_;

  PointBirth( const Expr::Tag& phiTag,
              const Expr::Tag& rStarTag,
              const double momentOrder,
              const double constRStar);

 public:
  class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::Tag& result,
               const Expr::Tag& phiTag,
               const Expr::Tag& rStarTag,
	              const double momentOrder,
               const double constRStar)
	: ExpressionBuilder(result),
  	phit_(phiTag),
   rstart_(rStarTag),    
  	momentorder_(momentOrder),
   constrstar_(constRStar)
	  {}
    
      ~Builder(){}
    
      Expr::ExpressionBase* build() const
      {
        return new PointBirth<FieldT>( phit_, rstart_, momentorder_, constrstar_ );
      }
    
    private:
      const Expr::Tag phit_, rstart_;
      const double constrstar_;
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
            const Expr::Tag& rStarTag,
            const double momentOrder,
            const double constRStar)
: Expr::Expression<FieldT>(),
  phiTag_(phiTag),
  rStarTag_(rStarTag),
  momentOrder_(momentOrder),
  constRStar_(constRStar)
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
  if (rStarTag_ != Expr::Tag () ) {
    exprDeps.requires_expression( rStarTag_ );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PointBirth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  phi_ = &fm.field_ref( phiTag_ );
  if (rStarTag_ != Expr::Tag () )
    rStar_ = &fm.field_ref( rStarTag_ );  
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
  
  if (rStarTag_ != Expr::Tag () ) {
    result <<= pow(*rStar_, momentOrder_); 
  } else {
    result <<= pow(constRStar_, momentOrder_);
  }
 }

//--------------------------------------------------------------------

#endif // PointBirth_Expr_h


