/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef MultiEnvSource_Expr_h
#define MultiEnvSource_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class MultiEnvSource
 *  \authors Alex Abboud, Tony Saad
 *  \date June 2012
 *  \brief Calculates teh source term for each moment equation that is due to the subgrid scale environments
 *  \f$ S_{mix} = -\frac{dw_1/dt}{w2} (\phi_1 - \phi_2 ) - \frac{dw_3/dt}{w_2} ( \phi_3 - \phi_2 )  \f$
 *  requires the initial moment value
 *  and the tag list of weights and derivatives of weights
 */
template< typename FieldT >
class MultiEnvSource
: public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, weightsAndDerivs_)
  DECLARE_FIELD(FieldT, phi_)
  
  const double initialMoment_;

  MultiEnvSource( const Expr::TagList& weightAndDerivativeTags,
                  const Expr::Tag& phiTag,
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
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
MultiEnvSource<FieldT>::
MultiEnvSource( const Expr::TagList& weightAndDerivativeTags,
                const Expr::Tag& phiTag,
                const double initialMoment)
: Expr::Expression<FieldT>(),
  initialMoment_(initialMoment)
{
  this->set_gpu_runnable( true );
   phi_ = this->template create_field_request<FieldT>(phiTag);
  this->template create_field_vector_request<FieldT>(weightAndDerivativeTags, weightsAndDerivs_);
}

//--------------------------------------------------------------------

template< typename FieldT >
MultiEnvSource<FieldT>::
~MultiEnvSource()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MultiEnvSource<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  double small = 1.0e-10;
  result <<= cond(  weightsAndDerivs_[2]->field_ref() <= small || weightsAndDerivs_[2]->field_ref() >= 1.0-small, 0.0 )
                 ( - (weightsAndDerivs_[1]->field_ref() + weightsAndDerivs_[5]->field_ref()) / weightsAndDerivs_[2]->field_ref() * (initialMoment_ - phi_->field_ref()));
}

#endif
