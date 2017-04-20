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

#ifndef KineticGrowthCoefficient_Expr_h
#define KineticGrowthCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class KineticGrowthCoefficient
 *  \author Alex Abboud
 *  \date February 2013
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the expression containing the coefficient used in a
 *  precipitation reaction with kinetic limited growth at small supersaturations
 *  \f$ g_0 = K_A (S - 1)^2 \f$ or \f$ (S - \bar{S} )^2 \f$ here \f$ K_A \f$ is a fitted coefficient
 *  \f$ g(r) = 1 \f$
 *
 */
template< typename FieldT >
class KineticGrowthCoefficient
: public Expr::Expression<FieldT>
{
  const double growthCoefVal_, sMax_, sMin_;
  const bool doSBar_;
//  const FieldT* superSat_; //field from table of supersaturation
//  const FieldT* sBar_;     //S bar calculatino for ostwald ripening
  DECLARE_FIELDS(FieldT, sBar_, superSat_)
  
  KineticGrowthCoefficient( const Expr::Tag& superSatTag,
                            const Expr::Tag& sBarTag,
                            const double growthCoefVal,
                            const double sMax, 
                            const double sMin);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const Expr::Tag& sBarTag,
             const double growthCoefVal,
             const double sMax,
            const double sMin)
    : ExpressionBuilder(result),
    supersatt_    (superSatTag),
    sbart_        (sBarTag),
    growthcoefval_(growthCoefVal),
    smax_         (sMax),
    smin_         (sMin)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new KineticGrowthCoefficient<FieldT>( supersatt_, sbart_, growthcoefval_, smax_, smin_ );
    }
    
  private:
    const Expr::Tag supersatt_, sbart_;
    const double growthcoefval_, smax_, smin_;
  };
  
  ~KineticGrowthCoefficient();
  void evaluate();
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
KineticGrowthCoefficient<FieldT>::
KineticGrowthCoefficient( const Expr::Tag& superSatTag,
                          const Expr::Tag& sBarTag,
                          const double growthCoefVal,
                          const double sMax,
                          const double sMin)
: Expr::Expression<FieldT>(),
  growthCoefVal_(growthCoefVal),
  sMax_         (sMax),
  sMin_         (sMin),
  doSBar_       (sBarTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
  if (doSBar_)  sBar_ = this->template create_field_request<FieldT>(sBarTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
KineticGrowthCoefficient<FieldT>::
~KineticGrowthCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
KineticGrowthCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  const FieldT& S = superSat_->field_ref();
  FieldT& result = this->value();
  
  if ( doSBar_ ) {
    const FieldT& sBar = sBar_->field_ref();
    result <<= cond( S < sMax_ && S > sMin_ , growthCoefVal_  * ( S - sBar ) * ( S - sBar )  )
                   (0.0);  
  } else {
    result <<= cond( S < sMax_ && S > sMin_ , growthCoefVal_ * ( S - 1.0) * ( S - 1.0) )
                   (0.0);  
  }
}

//--------------------------------------------------------------------

#endif // KineticGrowthCoefficient_Expr_h
