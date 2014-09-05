/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>

/**
 *  \class MomRHS
 *  \ingroup Expressions
 *
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
template< typename FieldT >
class MomRHS
 : public Expr::Expression<FieldT>
{
  typedef SpatialOps::structured::SVolField PFieldT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, PFieldT, FieldT >::type Grad;

  const Expr::Tag pressuret_, rhspartt_, volfract_, emptyTag_;

  const FieldT  *rhsPart_;
  const FieldT  *volfrac_;
  const PFieldT *pressure_;

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
     *         and stress)
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& pressure,
             const Expr::Tag& partRHS,
             const Expr::Tag& volFracTag);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~MomRHS();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // MomentumRHS_Expr_h
