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

#ifndef TimeDerivativeExpr_h
#define TimeDerivativeExpr_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename ValT >
class TimeDerivative : public Expr::Expression<ValT>
{
public:
  
  /**
   *  \class   TimeDerivative
   *  \author  Tony Saad
   *  \date    March, 2013
   *  \ingroup Expressions
   *
   *  \brief Calculates the time derivative of any field.
   *
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const Expr::Tag& newVarTag,
            const Expr::Tag& oldVarTag,
            const Expr::Tag& timestepTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag newvartag_;
    const Expr::Tag oldvartag_;
    const Expr::Tag timesteptag_;
  };
  
  void evaluate();
  
private:
  typedef typename SpatialOps::SingleValueField TimeField;
  DECLARE_FIELDS(ValT, newvar_, oldvar_)
  DECLARE_FIELD (TimeField, dt_)
  
  TimeDerivative( const Expr::Tag& newVarTag,
                 const Expr::Tag& oldVarTag,
                 const Expr::Tag& timestepTag );
  
};

//====================================================================

//--------------------------------------------------------------------

template<typename ValT>
TimeDerivative<ValT>::
TimeDerivative( const Expr::Tag& newVarTag,
                const Expr::Tag& oldVarTag,
                const Expr::Tag& dtTag )
: Expr::Expression<ValT>()
{
   newvar_ = this->template create_field_request<ValT>(newVarTag);
   oldvar_ = this->template create_field_request<ValT>(oldVarTag);
   dt_ = this->template create_field_request<TimeField>(dtTag);
}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= (newvar_->field_ref() - oldvar_->field_ref())/ dt_->field_ref();
}

//--------------------------------------------------------------------

template< typename ValT >
TimeDerivative<ValT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& newVarTag,
        const Expr::Tag& oldVarTag,
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
  newvartag_  ( newVarTag ),
  oldvartag_  ( oldVarTag ),
  timesteptag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
TimeDerivative<ValT>::Builder::build() const
{
  return new TimeDerivative<ValT>( newvartag_, oldvartag_, timesteptag_ );
}

//--------------------------------------------------------------------

#endif // TimeDerivativeExpr_h
