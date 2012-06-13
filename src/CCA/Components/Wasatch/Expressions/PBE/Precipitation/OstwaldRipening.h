#ifndef OstwaldRipening_Expr_h
#define OstwaldRipening_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class OstwaldRipening
 *  \author Alex Abboud
 *  \date February 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the source term associated with Oswalt Ripening
 *  where \f$ G = g_0 * ( S- \bar{S} )/r \f$ this is the second term
 *  \f$ \bar{S} = \exp ( 2 \nu \gamma /R T r) \f$
 *  \f$ OR = - g_0 * k * \int \exp( r_0 /r ) r^{k-2} N(r) dr \f$
 *  with the qudarature approximation used so that
 *  \f$ \int \exp( r_0 /r ) r^{k-2} N(r) dr \approx \sum_i exp( r0 / r_i ) r_i^{k-2} \f$
 *
 *  when \f$ r < r_{cutoff} \f$
 *  swap to using \f$ r^2 \f$ correlation
 *  \f$ r_{cutoff} \f$ set by input file
 *  this is for numerical stability
 */
template< typename FieldT >
class OstwaldRipening
: public Expr::Expression<FieldT>
{
  const Expr::Tag growthCoefTag_;        //same g0 used for growth expr
  const Expr::TagList weightsTagList_;   // these are the tags of all the known moments
  const Expr::TagList abscissaeTagList_; // these are the tags of all the known moments
  const double momentOrder_;             // order of this moment
  const double expCoef_;                 // exponential coefficient (r0 = 2 nu gamma/R T )
  const double rCutOff_;                 // size to swap r correlation 1/r to r^2
  const double constCoef_;							 // this is the same constant coefficient used for the growth expression
  const int nPts_;                       // number of qudarature nodes in closure

  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  const FieldT* growthCoef_;

  OstwaldRipening( const Expr::Tag growthCoefTag,
                   const Expr::TagList weightsTagList_,
                   const Expr::TagList abscissaeTagList_,
                   const double momentOrder,
                   const double expCoef,
                   const double rCutOff,
                   const double constCoef,
                   const int nPts);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& growthCoefTag,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const double momentOrder,
             const double expCoef,
             const double rCutOff,
             const double constCoef,
             const int nPts)
    : ExpressionBuilder(result),
    growthcoeft_     (growthCoefTag),
    weightstaglist_  (weightsTagList),
    abscissaetaglist_(abscissaeTagList),
    momentorder_     (momentOrder),
    expcoef_         (expCoef),
    rcutoff_         (rCutOff),
    constcoef_       (constCoef),
    npts_            (nPts)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new OstwaldRipening<FieldT>( growthcoeft_, weightstaglist_,abscissaetaglist_, momentorder_, expcoef_, rcutoff_, constcoef_, npts_ );
    }

  private:
    const Expr::Tag growthcoeft_;          //growth coefficient g0 expr
    const Expr::TagList weightstaglist_;   // these are the tags of all the known moments
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known moments
    const double momentorder_;
    const double expcoef_;
    const double rcutoff_;
    const double constcoef_;
    const int npts_;
  };

  ~OstwaldRipening();

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
OstwaldRipening<FieldT>::
OstwaldRipening( const Expr::Tag growthCoefTag,
                 const Expr::TagList weightsTagList,
                 const Expr::TagList abscissaeTagList,
                 const double momentOrder,
                 const double expCoef,
                 const double rCutOff,
                 const double constCoef,
                 const int nPts)
  : Expr::Expression<FieldT>(),
  growthCoefTag_   (growthCoefTag),
  weightsTagList_  (weightsTagList),
  abscissaeTagList_(abscissaeTagList),
  momentOrder_     (momentOrder),
  expCoef_         (expCoef),
  rCutOff_         (rCutOff),
  constCoef_       (constCoef),
  nPts_            (nPts)
  {}

//--------------------------------------------------------------------

template< typename FieldT >
OstwaldRipening<FieldT>::
~OstwaldRipening()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( growthCoefTag_ );
  exprDeps.requires_expression( weightsTagList_ );
  exprDeps.requires_expression( abscissaeTagList_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldManagerSelector<FieldT>::type& volfm = fml.template field_manager<FieldT>();
  weights_.clear();
  abscissae_.clear();
  for (Expr::TagList::const_iterator iweight=weightsTagList_.begin(); iweight!=weightsTagList_.end(); iweight++) {
    weights_.push_back(&volfm.field_ref(*iweight));
  }
  for (Expr::TagList::const_iterator iabscissa=abscissaeTagList_.begin(); iabscissa!=abscissaeTagList_.end(); iabscissa++) {
    abscissae_.push_back(&volfm.field_ref(*iabscissa));
  }

  growthCoef_ = &fml.template field_manager<FieldT>().field_ref( growthCoefTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  typename FieldT::const_interior_iterator growthCoefIter = growthCoef_->interior_begin();
  typename FieldT::interior_iterator resultsIterator = result.interior_begin();

  std::vector<typename FieldT::const_interior_iterator> weightsIterators;
  std::vector<typename FieldT::const_interior_iterator> abscissaeIterators;
  for (int i=0; i < nPts_; i++) {
    typename FieldT::const_interior_iterator thisIterator = weights_[i]->interior_begin();
    weightsIterators.push_back(thisIterator);

    typename FieldT::const_interior_iterator otherIterator = abscissae_[i]->interior_begin();
    abscissaeIterators.push_back(otherIterator);
  }

  double SumVal;
  while (growthCoefIter!=growthCoef_->interior_end() ) {
    SumVal = 0.0;
    for (int i = 0; i < nPts_; i++) {
      if (*abscissaeIterators[0] > rCutOff_ ) {
        SumVal += constCoef_ * momentOrder_ * *growthCoefIter * *weightsIterators[i] * pow(*abscissaeIterators[i], momentOrder_ - 2 ) * exp(expCoef_ / *abscissaeIterators[i] );
      } else {
        SumVal += constCoef_ * momentOrder_ * *growthCoefIter * *weightsIterators[i] * pow(*abscissaeIterators[i], momentOrder_ + 1 ) * exp(expCoef_ / *abscissaeIterators[i] );
      }
    }
    *resultsIterator = -( SumVal ); //this term is negative when appearing on RHS

    for (int i = 0; i < nPts_; i++) {
      ++weightsIterators[i];
      ++abscissaeIterators[i];
    }
    ++resultsIterator;
    ++growthCoefIter;
  }
}

#endif // OstwaldRipening_Expr_h
