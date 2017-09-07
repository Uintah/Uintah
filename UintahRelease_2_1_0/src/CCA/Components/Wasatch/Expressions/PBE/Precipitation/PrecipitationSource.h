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

#ifndef PrecipitationSource_Expr_h
#define PrecipitationSource_Expr_h

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
//  const Expr::TagList sourceTagList_;      ///< these are the tags of all the known sources
//  const Expr::Tag etaScaleTag_;            ///< this expression value can be read table header and takign inverse
//  const Expr::Tag densityTag_;             ///< rho to multiply source term by, since scalar solution is for dphirho/dt
//  const Expr::Tag envWeightTag_;           // weight tag for middle environment of multi mix model (optional)
  const std::vector< double > molecVols_;  ///< \f$\nu\f$ in the source evaluation
  const bool hasEnvWeight_;
  DECLARE_VECTOR_OF_FIELDS(FieldT, sources_)
  DECLARE_FIELDS(FieldT, etaScale_, density_, envWeight_)
  
  PrecipitationSource( const Expr::TagList& sourceTagList_,
                       const Expr::Tag& etaScaleTag_,
                       const Expr::Tag& densityTag_,
                       const Expr::Tag& envWeightTag_,
                       const std::vector<double>& molecVols_);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::TagList& sourceTagList,
             const Expr::Tag& etaScaleTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& envWeightTag,
             const std::vector<double>& molecVols)
    : ExpressionBuilder(result),
      sourcetaglist_(sourceTagList),
      etascalet_    (etaScaleTag),
      densityt_     (densityTag),
      envweightt_   (envWeightTag),
      molecvols_    (molecVols)
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
  void evaluate();

};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
PrecipitationSource<FieldT>::
PrecipitationSource( const Expr::TagList& sourceTagList,
                     const Expr::Tag& etaScaleTag,
                     const Expr::Tag& densityTag,
                     const Expr::Tag& envWeightTag,
                     const std::vector<double>& molecVols)
: Expr::Expression<FieldT>(),
  molecVols_    (molecVols),
  hasEnvWeight_(envWeightTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  this->template create_field_vector_request<FieldT>(sourceTagList, sources_);
   etaScale_ = this->template create_field_request<FieldT>(etaScaleTag);
   density_ = this->template create_field_request<FieldT>(densityTag);
  if (hasEnvWeight_)  envWeight_ = this->template create_field_request<FieldT>(envWeightTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationSource<FieldT>::
~PrecipitationSource()
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
//  typename FieldVec::const_iterator sourceIterator = sources_.begin();
  
  const FieldT& etaScale = etaScale_->field_ref();
  const FieldT& rho = density_->field_ref();
  
  for (size_t i = 0; i < nSources_; i++) {
    const FieldT& src = sources_[i]->field_ref();
    if ( hasEnvWeight_ ) {
      const FieldT& envW = envWeight_->field_ref();
      result <<= cond( etaScale > 0.0, result + 4.0/3.0*PI * molecVols_[i] * src * rho * envW / etaScale )
                     (0.0);
    } else {
      result <<= cond( etaScale > 0.0, result + 4.0/3.0*PI * molecVols_[i] * src * rho / etaScale )
                     (0.0);
    }
  }
}

//--------------------------------------------------------------------

#endif // PrecipitationSource_Expr_h

