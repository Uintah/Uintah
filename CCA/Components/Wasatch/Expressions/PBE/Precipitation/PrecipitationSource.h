#ifndef PrecipitationSource_Expr_h
#define PrecipitationSource_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationSource
 *  \author Alex Abboud
 *  \date March 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief This adds up the non diffusive/non convective source terms
 *  of the various populations for the 3rd moment and multiplies by
 *  the correct scaling factors to use as the reaction extent extent source term
 *  \f$ S = w_2 * \frac{1}{\eta_{scale}} \sum \nu (B + G - D) \f$
 *  also has the optional w_2 for modifying by the middle weight in a multi environment model
 */
template< typename FieldT >
class PrecipitationSource
: public Expr::Expression<FieldT>
{
  const Expr::TagList sourceTagList_;      ///< these are the tags of all the known sources
  const Expr::Tag etaScaleTag_;            ///< this expression value can be read table header and takign inverse
  const Expr::Tag densityTag_;             ///< rho to multiply source term by, since scalar solution is for dphirho/dt
  const Expr::Tag envWeightTag_;           // weight tag for middle environment of multi mix model (optional)
  const std::vector< double > molecVols_;  ///< \nu in the source evaluation

  typedef std::vector<const FieldT*> FieldVec;
  FieldVec sources_;
  const FieldT* etaScale_;
  const FieldT* density_;
  const FieldT* envWeight_;

  PrecipitationSource( const Expr::TagList sourceTagList_,
                       const Expr::Tag etaScaleTag_,
                       const Expr::Tag densityTag_,
                       const Expr::Tag envWeightTag_,
                       const std::vector<double> molecVols_);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& sourceTagList,
             const Expr::Tag& etaScaleTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& envWeightTag,
             const std::vector<double> molecVols)
    : ExpressionBuilder(result),
      sourcetaglist_  (sourceTagList),
      etascalet_      (etaScaleTag),
      densityt_       (densityTag),
      envweightt_     (envWeightTag),
      molecvols_      (molecVols)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new PrecipitationSource<FieldT>( sourcetaglist_, etascalet_, densityt_, envweightt_, molecvols_ );
    }

  private:
    const Expr::TagList sourcetaglist_;    // these are the tags of all the known source
    const Expr::Tag etascalet_;          // eta scaling tag
    const Expr::Tag densityt_;           //density tag
    const Expr::Tag envweightt_;         //middle environment weight tag
    const std::vector<double> molecvols_;  // vector for scaling source term
  };

  ~PrecipitationSource();

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
PrecipitationSource<FieldT>::
PrecipitationSource( const Expr::TagList sourceTagList,
                     const Expr::Tag etaScaleTag,
                     const Expr::Tag densityTag,
                    const Expr::Tag envWeightTag,
                     const std::vector<double> molecVols)
: Expr::Expression<FieldT>(),
  sourceTagList_  (sourceTagList),
  etaScaleTag_    (etaScaleTag),
  densityTag_     (densityTag),
  envWeightTag_   (envWeightTag),
  molecVols_      (molecVols)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationSource<FieldT>::
~PrecipitationSource()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationSource<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( sourceTagList_ );
  exprDeps.requires_expression( etaScaleTag_ );
  exprDeps.requires_expression( densityTag_ );
  if ( envWeightTag_ != Expr::Tag() )
    exprDeps.requires_expression( envWeightTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationSource<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  sources_.clear();
  for (Expr::TagList::const_iterator isource=sourceTagList_.begin(); isource!=sourceTagList_.end(); isource++) {
    sources_.push_back(&fm.field_ref(*isource) );
  }
  etaScale_ = &fm.field_ref( etaScaleTag_ );
  density_ = &fm.field_ref( densityTag_ );
  if ( envWeightTag_ != Expr::Tag() )
    envWeight_ = &fm.field_ref( envWeightTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationSource<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationSource<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  
  const size_t nSources_ = molecVols_.size();
  typename FieldVec::const_iterator sourceIterator = sources_.begin();
  
  for (size_t i = 0; i < nSources_; i++) {
    if (envWeightTag_ != Expr::Tag () ) {
      result <<= result + 4.0/3.0*PI * molecVols_[i] * **sourceIterator * *density_ * *envWeight_ / *etaScale_;
    } else {
      result <<= result + 4.0/3.0*PI * molecVols_[i] * **sourceIterator * *density_ / *etaScale_;
    }
    ++sourceIterator;
  }
}

#endif // PrecipitationSource_Expr_h

