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

#ifndef BrownianAggregationCoefficient_Expr_h
#define BrownianAggregationCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class BrownianAggregationCoefficient
 *  \author Alex Abboud
 *  \date June 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief Calculates the coefficent used for brownian diffusion
 *  \f$ 2 k_B T / 3 \rho \f$
 *  \f$k_B\f$ boltzmann const, \f$T\f$ temeprature, \f$\rho\f$ density
 */

template< typename FieldT >
class BrownianAggregationCoefficient
: public Expr::Expression<FieldT>
{
  const double coefVal_;
  DECLARE_FIELD(FieldT, density_)
  
  BrownianAggregationCoefficient( const Expr::Tag& densityTag,
                                  const double coefVal );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& densityTag,
             const double coefVal )
    : ExpressionBuilder(result),
    densityt_(densityTag),
    coefval_(coefVal)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new BrownianAggregationCoefficient<FieldT>( densityt_, coefval_);
    }
    
  private:
    const Expr::Tag densityt_ ;
    const double coefval_;
  };
  
  ~BrownianAggregationCoefficient();
  void evaluate();
  
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
BrownianAggregationCoefficient<FieldT>::
BrownianAggregationCoefficient( const Expr::Tag& densityTag,
                                const double coefVal )
: Expr::Expression<FieldT>(),
  coefVal_(coefVal)
{
  this->set_gpu_runnable( true );
   density_ = this->template create_field_request<FieldT>(densityTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
BrownianAggregationCoefficient<FieldT>::
~BrownianAggregationCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BrownianAggregationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( density_->field_ref() > 0.0, coefVal_ / density_->field_ref() )
                 (0.0);
}

//--------------------------------------------------------------------

#endif // BrownianAggregationCoefficient_Expr_h
