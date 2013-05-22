/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef OstwaldRipening_Expr_h
#define OstwaldRipening_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class OstwaldRipening
 *  \author Alex Abboud
 *  \date February 2012, revised Feb 2013
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the source term associated with Oswalt Ripening
 *  here \f$ \bar{S} = exp ( 2 \sigma \nu / RT / r) \f$
 *  Then with the quadrature approxiamtion \f$ \bar{S} \approx \sum_i w_i exp( 2 \sigma \nu / RT / r_i) \f$
 *  this term is then subtracted from the current supersaturation in the growth coefficient expressions if needed
 *  when \f$ r < r_{cutoff} \f$
 *  use \f$ \bar{S} = 0 \f$ for that abcissae
 */
template< typename FieldT >
class OstwaldRipening
: public Expr::Expression<FieldT>
{
  const Expr::TagList weightsTagList_;   // these are the tags of all the known moments
  const Expr::TagList abscissaeTagList_; // these are the tags of all the known moments
  const Expr::Tag moment0Tag_;
  const double expCoef_;                 // exponential coefficient (r0 = 2 nu gamma/R T )
  const double rCutOff_;                 // size to swap r correlation 1/r to r^2

  typedef std::vector<const FieldT*> FieldVec;
  FieldVec weights_;
  FieldVec abscissae_;
  const FieldT* moment0_;

  OstwaldRipening( const Expr::TagList weightsTagList_,
                   const Expr::TagList abscissaeTagList_,
                   const Expr::Tag& moment0Tag_,
                   const double expCoef,
                   const double rCutOff);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& weightsTagList,
             const Expr::TagList& abscissaeTagList,
             const Expr::Tag& moment0Tag,
             const double expCoef,
             const double rCutOff )
    : ExpressionBuilder(result),
    weightstaglist_  (weightsTagList),
    abscissaetaglist_(abscissaeTagList),
    moment0t_        (moment0Tag),
    expcoef_         (expCoef),
    rcutoff_         (rCutOff)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new OstwaldRipening<FieldT>( weightstaglist_, abscissaetaglist_, moment0t_, expcoef_, rcutoff_ );
    }

  private:
    const Expr::TagList weightstaglist_;   // these are the tags of all the known weights
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known absicase
    const Expr::Tag moment0t_;
    const double expcoef_;
    const double rcutoff_;
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
OstwaldRipening( const Expr::TagList weightsTagList,
                 const Expr::TagList abscissaeTagList,
                 const Expr::Tag& moment0Tag,
                 const double expCoef,
                 const double rCutOff)
  : Expr::Expression<FieldT>(),
  weightsTagList_  (weightsTagList),
  abscissaeTagList_(abscissaeTagList),
  moment0Tag_      (moment0Tag),
  expCoef_         (expCoef),
  rCutOff_         (rCutOff)
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
  exprDeps.requires_expression( weightsTagList_ );
  exprDeps.requires_expression( abscissaeTagList_ );
  exprDeps.requires_expression( moment0Tag_);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
//  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  
  
  const typename Expr::FieldMgrSelector<FieldT>::type& volfm = fml.template field_manager<FieldT>();
  moment0_ = &volfm.field_ref( moment0Tag_ );
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
OstwaldRipening<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  result <<= 0.0;
  typename FieldVec::const_iterator abscissaeIterator = abscissae_.begin();
  for( typename FieldVec::const_iterator weightsIterator=weights_.begin();
      weightsIterator!=weights_.end();
      ++weightsIterator, ++abscissaeIterator) {
    result <<= result + cond(**abscissaeIterator > rCutOff_, (**weightsIterator) / *moment0_ * exp( expCoef_ / **abscissaeIterator ) )
                            ( 0.0 );
  }
}

#endif // OstwaldRipening_Expr_h
