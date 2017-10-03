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

#ifndef Growth_Expr_h
#define Growth_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class Growth
 *  \authors Tony Saad, Alex Abboud
 *  \date January, 2012. (Originally created: July, 2011).
 *  \note modified and merged by Alex Abboud to generalize growth
 *  \tparam FieldT the type of field.
 *
 *  \brief Implements any type of growth term
 *  relies on the proper phi_tag from parsing
 */
template< typename FieldT >
class Growth
: public Expr::Expression<FieldT>
{

//  const Expr::Tag phiTag_, growthCoefTag_;  // this will correspond to proper tags for constant calc & momnet dependency
  const double momentOrder_;    ///< this is the order of the moment equation in which the growth model is used
  const double constCoef_;
  const bool doGrowth_;
//  const FieldT* phi_;           ///< this will correspond to m(k + i), i depends on which growth model is used
//  const FieldT* growthCoef_;    ///< this will correspond to the coefficient in the growth rate term
  DECLARE_FIELDS(FieldT, phi_, growthCoef_)

  Growth( const Expr::Tag& phiTag,
          const Expr::Tag& growthCoefTag,
          const double momentOrder,
          const double constCoef);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& growthCoefTag,
             const double momentOrder,
             const double constCoef)
    : ExpressionBuilder(result),
    phit_(phiTag),
    growthcoeft_(growthCoefTag),
    momentorder_(momentOrder),
    constcoef_(constCoef)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new Growth<FieldT>( phit_, growthcoeft_, momentorder_, constcoef_ );
    }

  private:
    const Expr::Tag phit_, growthcoeft_;
    const double momentorder_;
    const double constcoef_;
  };

  ~Growth();

  void evaluate();
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
Growth<FieldT>::
Growth( const Expr::Tag& phiTag,
        const Expr::Tag& growthCoefTag,
        const double momentOrder,
        const double constCoef)
  : Expr::Expression<FieldT>(),
    momentOrder_(momentOrder),
    constCoef_(constCoef),
    doGrowth_(growthCoefTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
   phi_ = this->template create_field_request<FieldT>(phiTag);
  if (doGrowth_)  growthCoef_ = this->template create_field_request<FieldT>(growthCoefTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
Growth<FieldT>::
~Growth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
Growth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if ( doGrowth_ ) {
    if (momentOrder_ != 0 ) {
      result <<= constCoef_ * momentOrder_ * growthCoef_->field_ref() * phi_->field_ref(); //this corresponds to source of G
    } else {
      result <<= 0.0; //zero growth for m_0
    }
  } else {
    if (momentOrder_ != 0 ) {
      result <<= constCoef_ * momentOrder_ * phi_->field_ref(); //this corresponds to source of G
    } else {
      result <<= 0.0; //zero growth for m_0
    }
  }
}

#endif
