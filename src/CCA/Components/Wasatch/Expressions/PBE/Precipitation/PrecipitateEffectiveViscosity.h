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

#ifndef PrecipitateEffectiveViscosity_Expr_h
#define PrecipitateEffectiveViscosity_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitateEffectiveViscosity
 *  \author Alex Abboud	 
 *  \date June 2012
 *  \brief Modifies Einstein Viscosity -a first order taylor series expansion derived from kinetic theory
 *  to account for shear thinning effect
 *  \f$ \mu_r = \mu / \mu_0 = ( 1 + 2.5 \lambda \phi ) |S|^n \f$
 *  Best used for particle volume fractionss < 0.10
 */
template< typename FieldT >
class PrecipitateEffectiveViscosity
: public Expr::Expression<FieldT>
{
  
//  const Expr::Tag volumeFractionTag_;  //Tag for particle volume fraction
//  const Expr::Tag strainMagnitudeTag_; //Tag for strain magnitude
  const double corrFac_; 							 //correction factor \lambda
  const double baseViscosity_;         // \mu_0
  const double power_;                 //power law exponent (n)
  const double minStrain_;             //cutoff so that vscosity does not go too high
  DECLARE_FIELD(FieldT, volumeFraction_)
  DECLARE_FIELD(FieldT, strainMagnitude_)
  
  PrecipitateEffectiveViscosity( const Expr::Tag& volumeFractionTag,
                                 const Expr::Tag& strainMagnitudeTag,
                                 const double corrFac,
                                 const double baseViscosity,
                                 const double power,
                                 const double minStrain);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& volumeFractionTag,
             const Expr::Tag& strainMagnitudeTag,
             const double corrFac,
             const double baseViscosity,
             const double power,
             const double minStrain)
    : ExpressionBuilder(result),
      volumefractiont_(volumeFractionTag),
      strainmagnitudet_(strainMagnitudeTag),
      corrfac_(corrFac),
      baseviscosity_(baseViscosity),
      power_(power),
      minstrain_(minStrain)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new PrecipitateEffectiveViscosity<FieldT>( volumefractiont_, strainmagnitudet_, corrfac_, baseviscosity_, power_, minstrain_ );
    }
    
  private:
    const Expr::Tag volumefractiont_;
    const Expr::Tag strainmagnitudet_;
    const double corrfac_;
    const double baseviscosity_;
    const double power_;
    const double minstrain_;
  };
  
  ~PrecipitateEffectiveViscosity();
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
PrecipitateEffectiveViscosity<FieldT>::
PrecipitateEffectiveViscosity( const Expr::Tag& volumeFractionTag,
                               const Expr::Tag& strainMagnitudeTag,
                               const double corrFac,
                               const double baseViscosity,
                               const double power,
                               const double minStrain)
: Expr::Expression<FieldT>(),
  corrFac_(corrFac),
  baseViscosity_(baseViscosity),
  power_(power),
  minStrain_(minStrain)
{
  this->set_gpu_runnable( true );
   volumeFraction_ = this->template create_field_request<FieldT>(volumeFractionTag);
   strainMagnitude_ = this->template create_field_request<FieldT>(strainMagnitudeTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitateEffectiveViscosity<FieldT>::
~PrecipitateEffectiveViscosity()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitateEffectiveViscosity<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  
  FieldT& result = this->value();
  const FieldT& vf = volumeFraction_->field_ref();
  const FieldT& sMag = strainMagnitude_->field_ref();
  //multiplying |S| by 2.0 & raising to 1/2, since wasatch turb model returns SijSij
  result <<= cond( vf < 1e-10 , baseViscosity_) 
                 ( 1.0 > (1 + 2.5 * corrFac_ * vf ) * pow(2.0 * sMag, power_/2 ), baseViscosity_)
                 ( sqrt(2.0* sMag) < minStrain_ && 1.0 > (1 + 2.5 * corrFac_ * vf ) * pow( minStrain_ , power_ ),  baseViscosity_)
                 ( sqrt(2.0* sMag) < minStrain_  , ( 1 + 2.5 * corrFac_ * vf ) * pow( minStrain_ , power_ ) * baseViscosity_ ) 
                 ( (1 + 2.5 * corrFac_ * vf ) * pow(2.0 * sMag, power_/2 ) * baseViscosity_ );
}

#endif
