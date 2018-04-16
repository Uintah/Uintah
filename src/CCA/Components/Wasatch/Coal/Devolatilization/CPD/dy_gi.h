#ifndef dy_gi_h
#define dy_gi_h

#include <expression/Expression.h>

namespace CPD{

/**
 *  \ingroup CPD
 *  \class dy_gi
 *  \brief This expression adds the result of \f$g_i\f$, and calculates
 *         the rate of production of each component.
 */
template <typename FieldT>
class dy_gi
  : public Expr::Expression<FieldT>
{
  typedef std::vector<int> VecI;

  const SpeciesSum& specSum_;
  DECLARE_VECTOR_OF_FIELDS( FieldT, gi_ )

  dy_gi( const Expr::TagList& giRHStag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param giRHStag   : gas species production rate in CPD model
     */
    Builder( const Expr::TagList& dyiTag,
             const Expr::TagList& giRHStag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::TagList giRHStag_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template<typename FieldT>
dy_gi<FieldT>::
dy_gi( const Expr::TagList& giRHStag )
  : Expr::Expression<FieldT>(),
    specSum_( SpeciesSum::self() )
{
  this->set_gpu_runnable(true);
  this->template create_field_vector_request<FieldT>( giRHStag, gi_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
void
dy_gi<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typename Expr::Expression<FieldT>::ValVec& dyi = this->get_value_vec();

  const SpecContributeVec& spSum = specSum_.get_vec_comp();
  for( size_t i=0; i<dyi.size(); ++i ){

    //const double mw = spSum[i].first; // g/s
    const VecI& contributingIndices = spSum[i].second;

    FieldT& dy = *dyi[i];
    dy <<= 0.0;
    for( size_t j=0; j<contributingIndices.size(); ++j ){
      const size_t index = contributingIndices[j];
      const FieldT& g = gi_[index - 1]->field_ref();
      dy <<= dy - g;
    }
  }
}
//--------------------------------------------------------------------

template<typename FieldT>
dy_gi<FieldT>::
Builder::Builder( const Expr::TagList& dyiTag,
                  const Expr::TagList& giRHStag )
 : ExpressionBuilder(dyiTag),
   giRHStag_( giRHStag )
{}

//--------------------------------------------------------------------

template<typename FieldT>
Expr::ExpressionBase*
dy_gi<FieldT>::
Builder::build() const
{
  return new dy_gi<FieldT>( giRHStag_ );
}

} // namespace CPD

#endif // y_gi_h
