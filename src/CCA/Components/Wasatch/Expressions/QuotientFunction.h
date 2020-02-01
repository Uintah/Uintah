/*
 * The MIT License
 *
 * Copyright (c) 2010-2018 The University of Utah
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

// todo: move this expreesion to ExprLib rather than have it in Wasatch.
#include <expression/Expression.h>

#ifndef QuotientFunction_h
#define QuotientFunction_h

/**
 *  \class  QuotientFunction
 *  \author Josh McConnell
 *  \date   November 2018
 *
 *  \brief Computes \f$ f(x,y) = \frac{x}{y}\f$
 */

template<typename FieldT>
class QuotientFunction : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS(FieldT, numerator_, denominator_ )

    QuotientFunction( const Expr::Tag& numeratorTag,
                      const Expr::Tag& denominatorTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a QuotientFunction expression
     *  @param resultTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& numeratorTag,
             const Expr::Tag& denominatorTag );

    Expr::ExpressionBase* build() const{
      return new QuotientFunction( numeratorTag_, denominatorTag_ );
    }

  private:
    const Expr::Tag numeratorTag_, denominatorTag_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
QuotientFunction<FieldT>::
QuotientFunction( const Expr::Tag& numeratorTag,
                  const Expr::Tag& denominatorTag )
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(true);

  numerator_   = this->template create_field_request<FieldT>( numeratorTag   );
  denominator_ = this->template create_field_request<FieldT>( denominatorTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
QuotientFunction<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& numerator   = numerator_  ->field_ref();
  const FieldT& denominator = denominator_->field_ref();

  result <<= numerator / denominator;
}

//--------------------------------------------------------------------

template<typename FieldT>
QuotientFunction<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& numeratorTag,
                  const Expr::Tag& denominatorTag )
  : ExpressionBuilder( resultTag ),
    numeratorTag_  ( numeratorTag   ),
    denominatorTag_( denominatorTag )
{}

//====================================================================
#endif // QuotientFunction_h

