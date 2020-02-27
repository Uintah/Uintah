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

#ifndef TanhFunction_Expr_h
#define TanhFunction_Expr_h

#include <expression/Expression.h>

/**
 *  @class TanhFunction
 *  @author Josh McConnell
 *  @date February, 2020
 *  @brief Implements a hyperbolic tangent function of a single independent variable,
 *         \f$ y = a \tanh( b (x-c) ) + d\f$, where a, b, c, and d are constants.
 */

template< typename FieldT>
class TanhFunction
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, x_ )
  const double &a_, &b_, &c_, &d_;

  TanhFunction( const Expr::Tag& indepVarTag,
                  const double&    a,
                  const double&    b,
                  const double&    c,
                  const double&    d )
  : Expr::Expression<FieldT>(),
  a_( a ),
  b_( b ),
  c_( c ),
  d_( d )
{
  this->set_gpu_runnable(true);
  x_ = this->template create_field_request<FieldT>( indepVarTag );
}

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a TanhFunction expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag resultTag,
             const Expr::Tag indepVarTag,
             const double    a,
             const double    b,
             const double    c,
             const double    d )
    : ExpressionBuilder( resultTag ),
    indepVarTag_( indepVarTag ),
    a_( a ),
    b_( b ),
    c_( c ),
    d_( d )
    {}

    Expr::ExpressionBase* build() const
    {
      return new TanhFunction<FieldT>( indepVarTag_, a_, b_, c_, d_ );
    }

  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_, c_, d_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();
    const FieldT& x = this->x_->field_ref();
    result <<= a_ * tanh( b_ * (x - c_) ) + d_;
  }
};

#endif // TanhFunction_Expr_h
