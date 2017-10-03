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

#ifndef OutflowBC_Expr_h
#define OutflowBC_Expr_h

#include <expression/Expression.h>
#include "BoundaryConditionBase.h"
#include <CCA/Components/Wasatch/TagNames.h>

template< typename FieldT >
class OutflowBC
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  OutflowBC( const Expr::Tag& momTag )
  {
    this->set_gpu_runnable(false);
    u_  = this->template create_field_request<FieldT>(momTag);
    dt_ = this->template create_field_request<SpatialOps::SingleValueField>(WasatchCore::TagNames::self().dt);
  }
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& momTag )
    : ExpressionBuilder(resultTag),
      momTag_ (momTag)
    {}
    Expr::ExpressionBase* build() const{ return new OutflowBC(momTag_); }
  private:
    const Expr::Tag momTag_;
  };
  
  ~OutflowBC(){}

  void evaluate();
  
private:
  DECLARE_FIELD(FieldT, u_)
  DECLARE_FIELD(SpatialOps::SingleValueField, dt_)
};

#endif // OutflowBC_Expr_h
