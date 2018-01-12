#ifndef Mv_Expr_h
#define Mv_Expr_h

#include <expression/Expression.h>


namespace CPD{

/**
 *  \class Mv
 *
 *   Calculating volatile mass based on
 *      - laible bridge  (l)
 *      - side chains    (delta_i)
 *
 *   ltag      : liable bridge
 *   deltaitag : taglist of side chains
 *
 *
 *
 */
template< typename FieldT >
class Mv
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELD( FieldT, l_ )
  DECLARE_FIELD( FieldT, tar_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, deltai_ )

  const Coal::CoalComposition& coalcomp_;
  const double ml0_;
  const SpeciesSum& specSum_;

  Mv( const Expr::Tag& ltag,
	  const Expr::Tag& tarTag,
      const Expr::TagList& deltaitag,
      const CPDInformation& cpd,
      const Coal::CoalComposition& coalcomp );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& mvTag,
             const Expr::Tag& ltag,
             const Expr::Tag& tarRag,
             const Expr::TagList& deltaitag,
             const CPDInformation& cpd,
             const Coal::CoalComposition& coalcomp );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag ltag_;
    const Expr::Tag tarTag_;
    const Expr::TagList deltaitag_;
    const Coal::CoalComposition& coalcomp_;
    const CPDInformation& cpd_;

  };

  ~Mv(){}
  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
Mv<FieldT>::
Mv( const Expr::Tag& ltag,
	const Expr::Tag& tarTag,
    const Expr::TagList& deltaitag,
    const CPDInformation& cpd,
    const Coal::CoalComposition& coalcomp  )
: Expr::Expression<FieldT>(),
  coalcomp_ ( coalcomp ),
  ml0_      ( cpd.l0_molecular_weight() ),
  specSum_  ( SpeciesSum::self() )
{
  this->set_gpu_runnable(true);

  l_   = this->template create_field_request<FieldT>( ltag   );
  tar_ = this->template create_field_request<FieldT>( tarTag );
  this->template create_field_vector_request<FieldT>( deltaitag, deltai_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
Mv<FieldT>::
evaluate()
{
  FieldT& result = this->value();
  using namespace SpatialOps;

  const FieldT& ell = l_->  field_ref();
  const FieldT& tar = tar_->field_ref();

  result <<= ell + tar;

  SpecContributeVec::const_iterator iCi = specSum_.get_vec_comp().begin();
  const SpecContributeVec::const_iterator iCie = specSum_.get_vec_comp().end();

  for( ; iCi != iCie; ++iCi ){
    const double mw = iCi->first;
    const VecI& contributingIndices = iCi->second;

    for( size_t i=0; i<contributingIndices.size(); ++i ){
      const size_t index = contributingIndices[i] - 1;
      const FieldT& deltai = deltai_[index]->field_ref();
      result <<= result + (deltai);
    }
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
Mv<FieldT>::
Builder::Builder( const Expr::Tag& mvTag,
                  const Expr::Tag& ltag,
                  const Expr::Tag& tarTag,
                  const Expr::TagList& deltaitag,
                  const CPDInformation& cpd,
                  const Coal::CoalComposition& coalcomp)
 : ExpressionBuilder(mvTag),
   ltag_      ( ltag ),
   tarTag_    ( tarTag ),
   deltaitag_ ( deltaitag ),
   coalcomp_  ( coalcomp ),
   cpd_       ( cpd )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
Mv<FieldT>::
Builder::build() const
{
  return new Mv<FieldT>( ltag_, tarTag_, deltaitag_, cpd_, coalcomp_ );
}


} // namespace
#endif // Mv_Expr_h
