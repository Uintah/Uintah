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

#ifndef PrecipitationBulkDiffusionCoefficient_Expr_h
#define PrecipitationBulkDiffusionCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationBulkDiffusionCoefficient
 *  \author Alex Abboud
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the expression containing the coefficient used in a
 *  precipitation reaction with bulk diffusion growth
 *  \f$ g_0 = \nu D C_{eq} (S-1) \f$ or \f$ (S - \bar{S}) \f$
 *  \f$ g(r) = 1/r \f$
 *
 */
template< typename FieldT >
class PrecipitationBulkDiffusionCoefficient
: public Expr::Expression<FieldT>
{
  const double growthCoefVal_, sMin_;
  const bool doSBar_;
//  const FieldT* superSat_; //field from table of supersaturation
//  const FieldT* eqConc_;   //field form table of equilibrium concentration
//  const FieldT* sBar_;     //S Bar term for ostwald ripening

  DECLARE_FIELDS(FieldT, superSat_, eqConc_, sBar_)
  
  PrecipitationBulkDiffusionCoefficient( const Expr::Tag& superSatTag,
                                         const Expr::Tag& eqConcTag,
                                         const Expr::Tag& sBarTag,
                                         const double growthCoefVal,
                                         const double sMin);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& superSatTag,
            const Expr::Tag& eqConcTag,
            const Expr::Tag& sBarTag,
            const double growthCoefVal,
            const double sMin)
    : ExpressionBuilder(result),
      supersatt_(superSatTag),
      eqconct_(eqConcTag),
      sbart_(sBarTag),
      growthcoefval_(growthCoefVal),
      smin_(sMin)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new PrecipitationBulkDiffusionCoefficient<FieldT>( supersatt_, eqconct_, sbart_, growthcoefval_, smin_ );
    }

  private:
    const Expr::Tag supersatt_, eqconct_, sbart_;
    const double growthcoefval_, smin_;
  };

  ~PrecipitationBulkDiffusionCoefficient();
  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
PrecipitationBulkDiffusionCoefficient<FieldT>::
PrecipitationBulkDiffusionCoefficient( const Expr::Tag& superSatTag,
                                       const Expr::Tag& eqConcTag,
                                       const Expr::Tag& sBarTag,
                                       const double growthCoefVal,
                                       const double sMin)
: Expr::Expression<FieldT>(),
  growthCoefVal_(growthCoefVal),
  sMin_(sMin),
  doSBar_(sBarTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
   eqConc_ = this->template create_field_request<FieldT>(eqConcTag);
  if (doSBar_)  sBar_ = this->template create_field_request<FieldT>(sBarTag);  
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationBulkDiffusionCoefficient<FieldT>::
~PrecipitationBulkDiffusionCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationBulkDiffusionCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& S = superSat_->field_ref();
  const FieldT& eqConc = eqConc_->field_ref();
  
  if ( doSBar_ ) {
    const FieldT& sBar = sBar_->field_ref();
    result <<= cond( S > sMin_, growthCoefVal_ * eqConc * ( S - sBar) )
                   (0.0);
  } else {
    result <<= cond( S > sMin_, growthCoefVal_ * eqConc * ( S - 1.0 ) )
                   (0.0);
  }         
}

//--------------------------------------------------------------------

#endif // PrecipitationBulkDiffusionCoefficient_Expr_h
