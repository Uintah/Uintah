/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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
  const Expr::Tag superSatTag_, sBarTag_;
  const double growthCoefVal_, sMax_, sMin_;
  const FieldT* superSat_; //field from table of supersaturation
  const FieldT* sBar_;     //S bar calculatino for ostwald ripening
  
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
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
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
  superSatTag_  (superSatTag),
  sBarTag_      (sBarTag),
  growthCoefVal_(growthCoefVal),
  sMax_         (sMax),
  sMin_         (sMin)
{
  this->set_gpu_runnable( true );
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
  if ( sBarTag_ != Expr::Tag() )
    exprDeps.requires_expression( sBarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
KineticGrowthCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  superSat_ = &fm.field_ref( superSatTag_ );
  if ( sBarTag_ != Expr::Tag() )
    sBar_ = &fm.field_ref( sBarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
KineticGrowthCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if ( sBarTag_ != Expr::Tag() ) {
    result <<= cond( *superSat_ < sMax_ && *superSat_ > sMin_ , growthCoefVal_  * ( *superSat_ - *sBar_ ) * ( *superSat_ - *sBar_ )  )
                   (0.0);  
  } else {
    result <<= cond( *superSat_ < sMax_ && *superSat_ > sMin_ , growthCoefVal_ * ( *superSat_ - 1.0) * ( *superSat_ - 1.0) )
                   (0.0);  
  }
}

//--------------------------------------------------------------------

#endif // KineticGrowthCoefficient_Expr_h
