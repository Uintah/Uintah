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

#ifndef PrecipitationMonosurfaceCoefficient_Expr_h
#define PrecipitationMonosurfaceCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationMonosurfaceCoefficient
 *  \author Alex Abboud
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the expression containing the coefficient used in a
 *  precipitation reaction with monosurface nucleation growth
 *  \f$ g_0 = \beta_A D d^3 \exp ( - \Delta G / K_B T ) \f$
 *  \f$ \Delta G = \frac{\beta_L \gamma^2 d^2}{4 \beta_A k_B T \ln (S)} \f$
 *  \f$ g(r) = r^2 \f$
 */
template< typename FieldT >
class PrecipitationMonosurfaceCoefficient
: public Expr::Expression<FieldT>
{
  const double growthCoefVal_;
  const double expConst_;
  DECLARE_FIELD(FieldT, superSat_)

  PrecipitationMonosurfaceCoefficient( const Expr::Tag& superSatTag,
                                       const double growthCoefVal,
                                       const double expConst);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& superSatTag,
            const double growthCoefVal,
            const double expConst)
    : ExpressionBuilder(result),
      supersatt_(superSatTag),
      growthcoefval_(growthCoefVal),
      expconst_(expConst)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new PrecipitationMonosurfaceCoefficient<FieldT>( supersatt_,  growthcoefval_, expconst_ );
    }

  private:
    const Expr::Tag supersatt_;
    const double growthcoefval_;
    const double expconst_;
  };

  ~PrecipitationMonosurfaceCoefficient();
  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
PrecipitationMonosurfaceCoefficient<FieldT>::
PrecipitationMonosurfaceCoefficient( const Expr::Tag& superSatTag,
                                     const double growthCoefVal,
                                     const double expConst)
: Expr::Expression<FieldT>(),
  growthCoefVal_(growthCoefVal),
  expConst_(expConst)
{
  this->set_gpu_runnable( true );
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationMonosurfaceCoefficient<FieldT>::
~PrecipitationMonosurfaceCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationMonosurfaceCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  const FieldT& S = superSat_->field_ref();
  FieldT& result = this->value();
  result <<= cond( S > 1.0, growthCoefVal_ * exp(expConst_ /  log(S) ) )
                 ( 0.0 );
}

//--------------------------------------------------------------------

#endif // PrecipitationMonosurfaceCoefficient_Expr_h
