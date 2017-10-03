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

#ifndef ParticleWallBC_Expr_h
#define ParticleWallBC_Expr_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include "BoundaryConditionBase.h"

/**
 *  \class 	ParticleWallBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    July, 2014
 *
 *  \brief Provides an expression to set Wall boundary conditions on particle variables
 *
 */
class ParticleWallBC
: public WasatchCore::BoundaryConditionBase<ParticleField>
{
public:
  ParticleWallBC(const double restCoef,
                const bool transverseVelocity):
  restCoef_(restCoef),
  transverseVelocity_(transverseVelocity)
  {}
  
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param result Tag of the resulting expression.
     * @param bcValue   constant boundary condition value.
     */
    Builder( const Expr::Tag& resultTag,
             const double restCoef,
             const bool transverseVelocity) :
    ExpressionBuilder(resultTag),
    restCoef_(restCoef),
    transverseVelocity_(transverseVelocity)
    {}
    Expr::ExpressionBase* build() const{ return new ParticleWallBC(restCoef_, transverseVelocity_); }
  private:
    const double restCoef_;
    const bool transverseVelocity_;
  };
  
  ~ParticleWallBC(){}
  void evaluate();
private:
  const double restCoef_;
  const bool transverseVelocity_;
};

#endif // ParticleWallBC_Expr_h
