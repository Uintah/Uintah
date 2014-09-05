#ifndef MomentClosure_Expr_h
#define MomentClosure_Expr_h

#include <expression/Expr_Expression.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/FieldExpressionsExtended.h>

/**
 *  \class QuadratureClosure
 *  \author Tony Saad
 *  \todo add documentation
 */
template< typename FieldT >
class QuadratureClosure
 : public Expr::Expression<FieldT>
{

  const Expr::TagList weightsTagList_; // these are the tags of all the known moments  
  const Expr::TagList abscissaeTagList_; // these are the tags of all the known moments    
  const double momentOrder_; // order of this unclosed moment. this will be used int he quadrature
  
  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  
  QuadratureClosure( const Expr::TagList weightsTagList_,
                     const Expr::TagList abscissaeTagList_,
                     const double momentOrder,
                     const Expr::ExpressionID& id,
                     const Expr::ExpressionRegistry& reg  );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList weightsTagList, const Expr::TagList abscissaeTagList, const double momentOrder )
      : weightstaglist_(weightsTagList),
        abscissaetaglist_(abscissaeTagList),
        momentorder_(momentOrder)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const 
    {
      return new QuadratureClosure<FieldT>(weightstaglist_,abscissaetaglist_, momentorder_, id, reg);
    }

  private:
    const Expr::TagList weightstaglist_; // these are the tags of all the known moments  
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments    
    const double momentorder_;
  };

  ~QuadratureClosure();

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
QuadratureClosure<FieldT>::
QuadratureClosure( const Expr::TagList weightsTagList,
                   const Expr::TagList abscissaeTagList,
                   const double momentOrder,
                   const Expr::ExpressionID& id,
                   const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FieldT>(id,reg),
    weightsTagList_(weightsTagList),
    abscissaeTagList_(abscissaeTagList),
    momentOrder_(momentOrder)
{}

//--------------------------------------------------------------------

template< typename FieldT >
QuadratureClosure<FieldT>::
~QuadratureClosure()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuadratureClosure<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  exprDeps.requires_expression( weightsTagList_ );
  exprDeps.requires_expression( abscissaeTagList_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuadratureClosure<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& volfm = fml.template field_manager<FieldT>();
  weights_.clear();
  abscissae_.clear();
  for (Expr::TagList::const_iterator iweight=weightsTagList_.begin(); iweight!=weightsTagList_.end(); iweight++) {
    weights_.push_back(&volfm.field_ref(*iweight));
  }  
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }  
}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuadratureClosure<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuadratureClosure<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result = 0.0;
  typename FieldVec::const_iterator abscissaeIterator = abscissae_.begin();
  for( typename FieldVec::const_iterator weightsIterator=weights_.begin(); 
       weightsIterator!=weights_.end();
       ++weightsIterator, ++abscissaeIterator) {
    result <<= result + (**weightsIterator) * pow(**abscissaeIterator,momentOrder_);
  }
}

#endif // MomentClosure_Expr_h
