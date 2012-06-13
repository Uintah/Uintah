
#ifndef MultiEnvSource_Expr_h
#define MultiEnvSource_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class MultiEnvSource
 *  \author Alex Abboud
 *  \date June 2012
 *  \brief Calculates teh source term for each moment equation that is due to the subgrid scale environments
 *  \f$ S_\mix = -\frac{dw_1/dt}{w2} (\phi_1 - \phi_2 ) - \frac{dw_3/dt}{w_2} ( \phi_3 - \phi_2 )  \f$
 *  requires the initial moment value
 *  and the tag list of weights and derivatives of weights
 */
template< typename FieldT >
class MultiEnvSource
: public Expr::Expression<FieldT>
{
  typedef std::vector<const FieldT*> FieldTVec;
  FieldTVec weightsAndDerivs_;
  const Expr::TagList weightAndDerivativeTags_; //this tag list has wieghts and derivatives [w0 dw0/dt w1 dw1/dt w2 dw2/dt]
  const Expr::Tag phiTag_;                      //tag for this moment in 2nd env
  const FieldT* phi_;
  const double initialMoment_;

  MultiEnvSource( const Expr::TagList weightAndDerivativeTags,
                  const Expr::Tag phiTag,
                  const double initialMoment);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightAndDerivativeTags,
             const Expr::Tag& phiTag,
             const double initialMoment)
    : ExpressionBuilder(result),
    weightandderivtaglist_(weightAndDerivativeTags),
    phit_(phiTag),
    initialmoment_(initialMoment)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new MultiEnvSource<FieldT>( weightandderivtaglist_, phit_, initialmoment_ );
    }

  private:
    const Expr::TagList weightandderivtaglist_;
    const Expr::Tag phit_;
    const double initialmoment_;
  };

  ~MultiEnvSource();

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
MultiEnvSource<FieldT>::
MultiEnvSource( const Expr::TagList weightAndDerivativeTags,
               const Expr::Tag phiTag,
                const double initialMoment)
: Expr::Expression<FieldT>(),
weightAndDerivativeTags_(weightAndDerivativeTags),
phiTag_(phiTag),
initialMoment_(initialMoment)
{}

//--------------------------------------------------------------------

template< typename FieldT >
MultiEnvSource<FieldT>::
~MultiEnvSource()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvSource<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( weightAndDerivativeTags_ );
  exprDeps.requires_expression( phiTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvSource<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldManagerSelector<FieldT>::type& fm = fml.field_manager<FieldT>();

  weightsAndDerivs_.clear();
  for( Expr::TagList::const_iterator iW=weightAndDerivativeTags_.begin();
      iW!=weightAndDerivativeTags_.end();
      ++iW ){
    weightsAndDerivs_.push_back( &fm.field_ref(*iW) );
  }
  phi_ = &fm.field_ref( phiTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvSource<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvSource<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  const int wdSize = 6;
  const FieldT* sampleField = weightsAndDerivs_[0];
  typename FieldT::const_interior_iterator sampleIterator = sampleField->interior_begin();
  typename FieldT::const_interior_iterator phiIter = phi_->interior_begin();
  typename FieldT::interior_iterator resultsIter = result.interior_begin();

  std::vector<typename FieldT::const_interior_iterator> weightsAndDerivsIters;
  for (int i = 0; i<wdSize; i++) {
    typename FieldT::const_interior_iterator thisIterator = weightsAndDerivs_[i]->interior_begin();
    weightsAndDerivsIters.push_back(thisIterator);
  }

  while (sampleIterator!=sampleField->interior_end() ) {
    *resultsIter = - *weightsAndDerivsIters[1] / *weightsAndDerivsIters[2] * ( initialMoment_ - *phiIter ) - *weightsAndDerivsIters[5] / *weightsAndDerivsIters[2] * ( initialMoment_ - *phiIter );
    //increment iterators
    for (int i = 0; i< wdSize; i++) {
      weightsAndDerivsIters[i] += 1;
    }
    ++phiIter;
    ++resultsIter;
    ++sampleIterator;
  }

}

#endif
