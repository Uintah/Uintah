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

#ifndef MultiEnvMixingModel_Expr_h
#define MultiEnvMixingModel_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class MultiEnvMixingModel
 *  \author Alex Abboud, Tony Saad
 *  \date June 2012
 *  \tparam FieldT the type of field.
 *  \brief Implements a basic three absciassae multi-environment mixing model.
 *  This expression sets \f$w_1\f$ at \f$\eta = 0\f$ and \f$w_3\f$ at \f$\eta = 1\f$
 *  where \f$\eta\f$ is the average mixture fraction.
 *  closure of this is that \f$w_2 = <Z>\f$
 *  for precipitation, reaction only occurs at w_2
 *  this returns a vector of weights
 *  with a vector of dw/dt base on scalr diss
 *  [w1 dw1/dt w2 dw2/dt w3 dw3/dt]
 */
template< typename FieldT >
class MultiEnvMixingModel
: public Expr::Expression<FieldT>
{

//  const Expr::Tag mixFracTag_, scalarVarTag_, scalarDissTag_;    //this will correspond to proper tags for mix frac & sclar var
//  const FieldT* mixFrac_; 											 // mixture fraction from grid
//  const FieldT* scalarVar_; 										 // sclar variance form grid
//  const FieldT* scalarDiss_;
  
  DECLARE_FIELDS(FieldT, mixFrac_, scalarVar_, scalarDiss_)
  const double maxDt_;
  
  MultiEnvMixingModel( const Expr::Tag& mixFracTag_,
                       const Expr::Tag& scalarVarTag_,
                       const Expr::Tag& scalarDissTag_,
                       const double& maxDt_ );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& results,
             const Expr::Tag& mixFracTag,
             const Expr::Tag& scalarVarTag,
             const Expr::Tag& scalarDissTag,
             const double& maxDt)
    : ExpressionBuilder(results),
    mixfract_(mixFracTag),
    scalarvart_(scalarVarTag),
    scalardisst_(scalarDissTag),
    maxdt_(maxDt)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new MultiEnvMixingModel<FieldT>( mixfract_, scalarvart_, scalardisst_, maxdt_ );
    }

  private:
    const Expr::Tag mixfract_, scalarvart_, scalardisst_;
    double maxdt_;
  };

  ~MultiEnvMixingModel();

  void evaluate();
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
MultiEnvMixingModel<FieldT>::
MultiEnvMixingModel( const Expr::Tag& mixFracTag,
                     const Expr::Tag& scalarVarTag,
                     const Expr::Tag& scalarDissTag,
                     const double& maxDt)
: Expr::Expression<FieldT>(),
  maxDt_(maxDt)
{
  this->set_gpu_runnable( true );
   mixFrac_ = this->template create_field_request<FieldT>(mixFracTag);
   scalarVar_ = this->template create_field_request<FieldT>(scalarVarTag);
   scalarDiss_ = this->template create_field_request<FieldT>(scalarDissTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
MultiEnvMixingModel<FieldT>::
~MultiEnvMixingModel()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvMixingModel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef typename Expr::Expression<FieldT>::ValVec ResultsVec;

  ResultsVec& results = this->get_value_vec();

  const FieldT& mixFrac = mixFrac_->field_ref();
  const FieldT& scalarVar = scalarVar_->field_ref();
  const FieldT& scalarDiss = scalarDiss_->field_ref();
  
  double small = 1.0e-10;
  // w1
  *results[0] <<= cond( mixFrac <= small, 1.0  )
                      ( mixFrac >= 1.0-small, 0.0  )
                      ( scalarVar/ mixFrac );
  
  // dw1/dt
  *results[1] <<= cond( mixFrac <= small || mixFrac >= 1.0-small, 0.0 )
                      ( - scalarDiss/ mixFrac > - *results[0]/maxDt_, - scalarDiss/ mixFrac )
                      ( - *results[0]/maxDt_ );

  // w3
  *results[4] <<= cond( mixFrac <= small, 0.0 )
                      ( mixFrac >= 1.0-small, 1.0 )
                      ( - scalarVar / ( mixFrac - 1.0 ) );

  // dw3/dt
  *results[5] <<= cond( mixFrac <= small || mixFrac >= 1.0-small, 0.0 )
                      ( - scalarDiss / (1.0 - mixFrac) > - *results[4]/maxDt_, - scalarDiss / (1.0 - mixFrac) )
                      ( - *results[4]/maxDt_);
  
  //weight 2 last, sicne stability requires w1&3 calc
  // w2
  *results[2] <<= cond( mixFrac <= small || mixFrac >= 1.0-small, 0.0 )
                      ( 1.0 + scalarVar / (mixFrac * mixFrac - mixFrac) );
  
  // dw2/dt
  *results[3] <<= cond( mixFrac <= small || mixFrac >= 1.0-small, 0.0 )
                      ( scalarDiss / (mixFrac - mixFrac * mixFrac) < (*results[0] + *results[4])/maxDt_ , scalarDiss / (mixFrac - mixFrac * mixFrac) )
                      ( (*results[0] + *results[4])/maxDt_ );
}

#endif
