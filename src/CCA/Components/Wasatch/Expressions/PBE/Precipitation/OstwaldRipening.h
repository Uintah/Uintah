/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
 *  add in tolman length variation such that \f$ \nu = \nu_{bulk} * (1 - \delta_T / r) \f$
 *  \f$ \delta_T \f$ is the tolman length, this is multiplied by the exponential coefficient in the code 
 *  Then with the quadrature approxiamtion \f$ \bar{S} \approx \sum_i w_i exp( 2 \sigma \nu / RT / r_i) \f$
 *  this term is then subtracted from the current supersaturation in the growth coefficient expressions if needed
 *  when \f$ r < r_{cutoff} \f$
 *  use \f$ \bar{S} = 0 \f$ for that abcissae
 */
template< typename FieldT >
class OstwaldRipening
: public Expr::Expression<FieldT>
{
//  const Expr::TagList weightsTagList_;   // these are the tags of all the known moments
//  const Expr::TagList abscissaeTagList_; // these are the tags of all the known moments
//  const Expr::Tag moment0Tag_;
  const double expCoef_;                 // exponential coefficient (r0 = 2 nu gamma/R T )
  const double tolmanLength_;            // tolman length
  const double rCutOff_;                 // size to swap r correlation 1/r to r^2

  DECLARE_VECTOR_OF_FIELDS(FieldT, weights_)
  DECLARE_VECTOR_OF_FIELDS(FieldT, abscissae_)
  DECLARE_FIELD(FieldT, m0_)
  

  OstwaldRipening( const Expr::TagList weightsTagList_,
                   const Expr::TagList abscissaeTagList_,
                   const Expr::Tag& moment0Tag_,
                   const double expCoef,
                   const double tolmanLength,
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
             const double tolmanLength,
             const double rCutOff )
    : ExpressionBuilder(result),
    weightstaglist_  (weightsTagList),
    abscissaetaglist_(abscissaeTagList),
    moment0t_        (moment0Tag),
    expcoef_         (expCoef),
    tolmanlength_    (tolmanLength),
    rcutoff_         (rCutOff)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      return new OstwaldRipening<FieldT>( weightstaglist_, abscissaetaglist_, moment0t_, expcoef_, tolmanlength_, rcutoff_ );
    }

  private:
    const Expr::TagList weightstaglist_;   // these are the tags of all the known weights
    const Expr::TagList abscissaetaglist_; // these are the tags of all the known absicase
    const Expr::Tag moment0t_;
    const double expcoef_;
    const double tolmanlength_;
    const double rcutoff_;
  };

  ~OstwaldRipening();
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
                 const double tolmanLength,
                 const double rCutOff)
: Expr::Expression<FieldT>(),
  expCoef_         (expCoef),
  tolmanLength_    (tolmanLength),
  rCutOff_         (rCutOff)
{
  this->set_gpu_runnable( true );
  this->template create_field_vector_request<FieldT>(weightsTagList, weights_);
  this->template create_field_vector_request<FieldT>(abscissaeTagList, abscissae_);
   m0_ = this->template create_field_request<FieldT>(moment0Tag);
}

//--------------------------------------------------------------------

template< typename FieldT >
OstwaldRipening<FieldT>::
~OstwaldRipening()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
OstwaldRipening<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& m0 = m0_->field_ref();
  result <<= 0.0;
  SpatFldPtr<FieldT> surfaceEnergyModification = SpatialFieldStore::get<FieldT>( result );
  
  for (size_t i=0; i<weights_.size(); ++i) {
    const FieldT& w = weights_[i]->field_ref();
    const FieldT& a = abscissae_[i]->field_ref();
    *surfaceEnergyModification <<=  1.0 - tolmanLength_ / a ;
    result <<= cond( m0 > 0.0 , result + cond(a > rCutOff_, (w) / m0 * exp( *surfaceEnergyModification *  expCoef_ / a ) )
                    ( 0.0 ) )
    (0.0);
  }
}

#endif // OstwaldRipening_Expr_h
