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

#ifndef ContinuityResidual_Expr_h
#define ContinuityResidual_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class ContinuityResidual
 *  \ingroup Expressions
 *
 *  \brief calculates \f$ \nabla\cdot\mathbf{u} \f$
 *
 *  \tparam FieldT the field type for the ContinuityResidual (nominally the scalar volume field)
 *  \tparam Vel1T  the field type for the first velocity component
 *  \tparam Vel2T  the field type for the second velocity component
 *  \tparam Vel3T  the field type for the third velocity component
 */
template< typename FieldT,
          typename Vel1T,
          typename Vel2T,
          typename Vel3T >
class ContinuityResidual
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELD(FieldT, drhodt_)
  DECLARE_FIELD(Vel1T, u1_)
  DECLARE_FIELD(Vel2T, u2_)
  DECLARE_FIELD(Vel3T, u3_)
  const bool constDen_, doX_, doY_, doZ_, is3d_;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, FieldT >::type Vel1GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, FieldT >::type Vel2GradT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, Vel3T, FieldT >::type Vel3GradT;

  const Vel1GradT* vel1GradOp_;
  const Vel2GradT* vel2GradOp_;
  const Vel3GradT* vel3GradOp_;

  ContinuityResidual( const Expr::Tag&     drhodtTag,
                      const Expr::TagList& velTags    );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     *  \param vel3tag the velocity corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag&     drhodtTag,
             const Expr::TagList& velTags    );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag     drhodtTag_;
    const Expr::TagList velTags_;
  };

  ~ContinuityResidual();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // ContinuityResidual_Expr_h
