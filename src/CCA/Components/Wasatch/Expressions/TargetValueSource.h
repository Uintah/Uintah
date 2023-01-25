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

#ifndef TargetValueSource_Expr_h
#define TargetValueSource_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/FieldTypes.h>


/**
 *  \class 	TargetValueSource
 *  \ingroup 	Expressions
 *
 *  \brief Defines a source term to be added to a scalar equation in order to achieve a target scalar value.
 *
 *  \tparam FieldT the type of field for the momentum RHS (nominally
 *          XVolField, YVolField, ZVolField).
 *
 \f[
    (\rho u)^{n+1} = (\rho u)^{n} + \Delta F^n + \Delta t S
 \f]
 We can achieve a target velocity by setting
 \f[
 (\rho u)^{n+1} = \rho^{n+1} U
 \f]
Then, we solve for S
 \f[
    S = \frac{\rho^{n+1} U - (\rho u)^{n}}{\Delta t} - F^n
 \f]
 In practice, insteady of accessin F in Wasatch, we grab the momentum RHS from the
 old timestep and subtract from that the old value of S, that is:
 \f[
 S = \frac{\rho^{n+1} U - (\rho u)^{n}}{\Delta t} - ( mom_old_RHS - S_old)
 \f]
 */
template< typename FieldT >
class TargetValueSource
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::BasicOpTypes<FieldT>  OpTypes;
  const bool constValTarget_;

  
  typedef typename SpatialOps::SingleValueField TimeField;
  DECLARE_FIELD ( TimeField, dt_ )
  DECLARE_FIELDS( FieldT, phi_, targetphi_, phiRHS_, volFrac_ )
  const double targetphivalue_;
  TargetValueSource( const Expr::Tag& phiTag,
            const Expr::Tag& phiRHSTag,
            const Expr::Tag& volFracTag,
            const Expr::Tag& targetPhiTag,
            const double targetPhiValue);
  

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag phit_;
    const Expr::Tag phirhst_;
    const Expr::Tag volfract_;
    const Expr::Tag targetphit_;
    const double targetphivalue_;
    
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& phiRHSTag,
             const Expr::Tag& volFracTag,
            const Expr::Tag& targetPhiTag,
             const double targetVelocity);

    Expr::ExpressionBase* build() const;
  };

  ~TargetValueSource();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // TargetValueSource_Expr_h
