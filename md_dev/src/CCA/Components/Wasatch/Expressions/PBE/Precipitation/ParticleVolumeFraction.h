
#ifndef ParticleVolumeFraction_Expr_h
#define ParticleVolumeFraction_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
/**
 *  \ingroup WasatchExpressions
 *  \class ParticleVolumeFraction
 *  \author Alex Abboud	 
 *  \date June 2012
 *  \brief Calcualtes the particle volume fraction
 *  \f$ \phi = \sum \frac{4 \pi}{3} <r>^3 N/V \f$
 *  <r> = m_1/m_0
 *  \phi = \sum 4 \pi/3 (m_1 * CF)^3 / m_^2
 */
template< typename FieldT >
class ParticleVolumeFraction
: public Expr::Expression<FieldT>
{
  
  const Expr::TagList zerothMomentTagList_;            //list of all m0s
  const Expr::TagList firstMomentTagList_;             //list of all m1s
  const double convFac_; 														   //Conversion factor for consistent units

  typedef std::vector<const FieldT*> FieldVec;
  FieldVec zerothMoments_;
  FieldVec firstMoments_;
  
  ParticleVolumeFraction( const Expr::TagList zerothMomentTagList_,
                          const Expr::TagList firstMomentTagList_,
                          const double convFac);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& zerothMomentTagList,
             const Expr::TagList& firstMomentTagList,
             const double convFac)
    : ExpressionBuilder(result),
    zerothmomenttaglist_(zerothMomentTagList),
    firstmomenttaglist_(firstMomentTagList),
    convfac_(convFac)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new ParticleVolumeFraction<FieldT>( zerothmomenttaglist_, firstmomenttaglist_, convfac_ );
    }
    
  private:
    const Expr::TagList zerothmomenttaglist_;
    const Expr::TagList firstmomenttaglist_;
    const double convfac_;
  };
  
  ~ParticleVolumeFraction();
  
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
ParticleVolumeFraction<FieldT>::
ParticleVolumeFraction( const Expr::TagList zerothMomentTagList,
                        const Expr::TagList firstMomentTagList, 
                        const double convFac)
: Expr::Expression<FieldT>(),
zerothMomentTagList_(zerothMomentTagList),
firstMomentTagList_(firstMomentTagList),
convFac_(convFac)
{}

//--------------------------------------------------------------------

template< typename FieldT >
ParticleVolumeFraction<FieldT>::
~ParticleVolumeFraction()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleVolumeFraction<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( zerothMomentTagList_ );
  exprDeps.requires_expression( firstMomentTagList_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleVolumeFraction<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  zerothMoments_.clear();
  firstMoments_.clear();
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  
  for (Expr::TagList::const_iterator iM0=zerothMomentTagList_.begin(); iM0 != zerothMomentTagList_.end(); iM0++) {
    zerothMoments_.push_back(&fm.field_ref(*iM0) ); 
  }
  for (Expr::TagList::const_iterator iM1=firstMomentTagList_.begin(); iM1 != firstMomentTagList_.end(); iM1++) {
    firstMoments_.push_back(&fm.field_ref(*iM1)); 
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleVolumeFraction<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleVolumeFraction<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0 ;
  
  typename FieldVec::const_iterator firstMomentIterator = firstMoments_.begin();
  for ( typename FieldVec::const_iterator zerothMomentIterator=zerothMoments_.begin();
       zerothMomentIterator!=zerothMoments_.end();
       ++zerothMomentIterator, ++firstMomentIterator) {
    result <<= result + 4.0/3.0 * PI * ( **firstMomentIterator * convFac_ ) * ( **firstMomentIterator * convFac_ ) *
                       ( **firstMomentIterator * convFac_ ) / **zerothMomentIterator / **zerothMomentIterator;
                                   
  }
}  

#endif

