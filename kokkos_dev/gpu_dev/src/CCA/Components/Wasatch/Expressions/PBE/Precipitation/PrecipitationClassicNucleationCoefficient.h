#ifndef PrecipitationClassicNucleationCoefficient_Expr_h
#define PrecipitationClassicNucleationCoefficient_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**                                                                                                                   
 *  \ingroup WasatchExpressions                                                                                       
 *  \class PrecipitationClassicNucleationCoefficient                                                                                                 
 *  \author Alex Abboud                                                                                               
 *  \date January, 2012                                                                                               
 *                                                                                                                    
 *  \tparam FieldT the type of field.                                                                                 
 *                                                                                                                    
 *  \brief Nucleation Coeffcient Source term for use in QMOM                                                                     
 *  classic nucleation refers to this value as                                                                       
 *  \f$ B_0 = \exp ( 16 \pi /3 ( \gamma /K_B T)^3( \nu /N_A/ \ln(S)^2  \f$                                                                                                                
 */
template< typename FieldT >
class PrecipitationClassicNucleationCoefficient
: public Expr::Expression<FieldT>
{
  const Expr::Tag phiTag_, superSatTag_;            
  const FieldT* phi_; // this will correspond to m(k+1)                                                               
  const FieldT* superSat_;
  const double expConst_;
  
  PrecipitationClassicNucleationCoefficient( const Expr::Tag& superSatTag,
                                             const double expConst);
  
  public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const double expConst )
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    expconst_(expConst)
    {}

    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new PrecipitationClassicNucleationCoefficient<FieldT>( supersatt_, expconst_);
    }
    
  private:
    const Expr::Tag supersatt_;
    const double expconst_;
  };
  
  ~PrecipitationClassicNucleationCoefficient();
  
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
PrecipitationClassicNucleationCoefficient<FieldT>::
PrecipitationClassicNucleationCoefficient( const Expr::Tag& superSatTag,
                                           const double expConst)
: Expr::Expression<FieldT>(),
superSatTag_(superSatTag),
expConst_(expConst)
{}

//--------------------------------------------------------------------                                                

template< typename FieldT >
PrecipitationClassicNucleationCoefficient<FieldT>::
~PrecipitationClassicNucleationCoefficient()
{}

//--------------------------------------------------------------------                                                

template< typename FieldT >
void
PrecipitationClassicNucleationCoefficient<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
}

//--------------------------------------------------------------------                                                

template< typename FieldT >
void
PrecipitationClassicNucleationCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  superSat_ = &fm.field_ref( superSatTag_ );
}

//--------------------------------------------------------------------       
template< typename FieldT >
void
PrecipitationClassicNucleationCoefficient<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------                                                

template< typename FieldT >
void
PrecipitationClassicNucleationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  typename FieldT::const_interior_iterator superSatIterator = superSat_->interior_begin();
  typename FieldT::interior_iterator resultsIterator = result.interior_begin();
  
  while (superSatIterator!=superSat_->interior_end() ) {
    if (*superSatIterator > 1.1 ) {  //set value to 0 if S is too small
      *resultsIterator = exp(expConst_ / log(*superSatIterator) / log(*superSatIterator) );
    } else {
      *resultsIterator = 0.0;
    }
    ++superSatIterator;
    ++resultsIterator;
  }
}

//--------------------------------------------------------------------                                                

#endif // PrecipitationClassicNucleationCoefficient_Expr_h                                                                                           
