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

#ifndef MomentumRHS_Expr_h
#define MomentumRHS_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

/**
 *  \class MomRHS
 *  \ingroup Expressions
 *  \author Tony Saad, James C. Sutherland
 *  \brief Calculates the full momentum RHS
 *
 *  \tparam FieldT the type of field for the momentum RHS (nominally
 *          XVolField, YVolField, ZVolField).
 *
 *  The momentum RHS is split into two contributions:
 *   - the pressure term
 *   - the convective, diffusive, and body force terms
 *  These are calculated in the MomRHSPart and Pressure expressions, respectively.
 */
template< typename FieldT, typename DirT>
class MomRHS
 : public Expr::Expression<FieldT>
{
  typedef SpatialOps::SVolField PFieldT;
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<FieldT, DirT>::Gradient, PFieldT, FieldT >::type Grad;

  DECLARE_FIELDS(FieldT, rhsPart_, volfrac_)
  DECLARE_FIELD(PFieldT, pressure_)
  const bool hasP_, hasIntrusion_;
  const Grad* gradOp_;

  MomRHS( const Expr::Tag& pressure,
          const Expr::Tag& partRHS,
          const Expr::Tag& volFracTag);

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag pressuret_, rhspartt_, volfract_;
  public:
    /**
     *  \param result the result of this expression
     *  \param pressure the expression to compute the pressure as a scalar volume field
     *  \param partRHS the expression to compute the other terms in
     *         the momentum RHS (body force, divergence of convection
     *         and strain)
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& pressure,
             const Expr::Tag& partRHS,
             const Expr::Tag& volFracTag);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~MomRHS();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // MomentumRHS_Expr_h
