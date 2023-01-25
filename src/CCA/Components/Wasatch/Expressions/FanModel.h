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

#ifndef FanModel_Expr_h
#define FanModel_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

#include <CCA/Components/Wasatch/FieldTypes.h>


/**
 *  \class 	FanModel
 *  \ingroup 	Expressions
 *
 *  \brief Defines a source term to added to a momentum equation in order to achieve a target velocity in a region of the domain.
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
class FanModel
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::BasicOpTypes<FieldT>  OpTypes;
  
  // interpolant for density: svol to fieldT
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  DensityInterpT;  
  const DensityInterpT* densityInterpOp_;

  
  typedef typename SpatialOps::SingleValueField TimeField;
  DECLARE_FIELD ( TimeField, dt_ )
  DECLARE_FIELD ( SVolField, rho_ )
  DECLARE_FIELDS( FieldT, mom_, momRHS_, fanSourceOld_, volFrac_ )
  const double targetVel_;
  FanModel( const Expr::Tag& rhoTag,
            const Expr::Tag& momTag,
            const Expr::Tag& momRHSTag,
            const Expr::Tag& fanSrcOldTag,
            const Expr::Tag& volFracTag,
            const double targetVelocity);
  

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag rhot_;
    const Expr::Tag momt_;
    const Expr::Tag momrhst_;
    const Expr::Tag volfract_;
    const Expr::Tag fansrcoldt_;
    const double targetvelocity_;
    
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& rhoTag,
             const Expr::Tag& momTag,
             const Expr::Tag& momRHSTag,
             const Expr::Tag& volFracTag,
             const double targetVelocity);

    Expr::ExpressionBase* build() const;
  };

  ~FanModel();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // FanModel_Expr_h
