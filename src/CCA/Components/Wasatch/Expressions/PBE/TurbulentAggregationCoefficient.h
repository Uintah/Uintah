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

#ifndef TurbulentAggregationCoefficient_Expr_h
#define TurbulentAggregationCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class TurbulentAggregationCoefficient
 *  \author Alex Abboud
 *  \date June 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief Calculates the coefficent used for Turbulent aggregation
 *  \f$ (4/3)*(3 \pi /10)^{1/2} (\epsilon / \nu ) ^{1/2} \f$
 *  \f$ \epsilon \f$ is the energy dissipation, \f$ \nu \f$ is the kinematic viscosity
 */
template< typename FieldT >
class TurbulentAggregationCoefficient
: public Expr::Expression<FieldT>
{
  const double coefVal_;
  DECLARE_FIELDS(FieldT, kinVisc_, dissipation_)
  
  TurbulentAggregationCoefficient( const Expr::Tag& kinViscTag,
                                   const Expr::Tag& dissipationTag,
                                   const double coefVal );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& kinViscTag,
             const Expr::Tag& dissipationTag,
             const double coefVal )
    : ExpressionBuilder(result),
      kinvisct_(kinViscTag),
      dissipationt_(dissipationTag),
      coefval_(coefVal)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new TurbulentAggregationCoefficient<FieldT>( kinvisct_, dissipationt_, coefval_);
    }
    
  private:
    const Expr::Tag kinvisct_ ;
    const Expr::Tag dissipationt_;
    const double coefval_;
  };
  
  ~TurbulentAggregationCoefficient();
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
TurbulentAggregationCoefficient<FieldT>::
TurbulentAggregationCoefficient( const Expr::Tag& kinViscTag,
                                 const Expr::Tag& dissipationTag,
                                 const double coefVal )
: Expr::Expression<FieldT>(),
  coefVal_(coefVal)
{
  this->set_gpu_runnable( true );
   kinVisc_ = this->template create_field_request<FieldT>(kinViscTag);
   dissipation_ = this->template create_field_request<FieldT>(dissipationTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
TurbulentAggregationCoefficient<FieldT>::
~TurbulentAggregationCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TurbulentAggregationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& mu = kinVisc_->field_ref();
  const FieldT& eps = dissipation_->field_ref();
  result <<= cond( mu > 0.0, coefVal_ * sqrt( eps / mu ) )
                 (0.0);
}

//--------------------------------------------------------------------

#endif // TurbulentAggregationCoefficient_Expr_h
