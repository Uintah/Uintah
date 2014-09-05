/**
 *  \file   MonolithicRHS.h
 *
 *  \date   Apr 10, 2012
 *  \author James C. Sutherland
 *
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



#ifndef MonolithicRHS_Expr_h
#define MonolithicRHS_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/FVStaggeredFieldTypes.h>

/**
 *  \class MonolithicRHS
 *
 *  Calculates
 *  \f[
 *    -\nabla\cdot(\rho \phi \mathbf{u}) + \nabla\cdot(D_\phi \nabla\phi) + s_\phi
 *  \f]
 *  which is the RHS of the transport equation for \f$\rho\phi\f$.
 */
template< typename FieldT >
class MonolithicRHS
: public Expr::Expression<FieldT>
{
  const Expr::Tag dCoefTag_, xconvFluxTag_, yconvFluxTag_, zconvFluxTag_, phiTag_, srcTag_;

  typedef typename FaceTypes<FieldT>::XFace  XFaceT;
  typedef typename FaceTypes<FieldT>::YFace  YFaceT;
  typedef typename FaceTypes<FieldT>::ZFace  ZFaceT;

  const FieldT* dCoef_;
  const FieldT* phi_;
  const FieldT* src_;

  const XFaceT* convFluxX_;  ///< x-direction convective flux
  const YFaceT* convFluxY_;  ///< y-direction convective flux
  const ZFaceT* convFluxZ_;  ///< z-direction convective flux

  typedef SpatialOps::structured::BasicOpTypes<FieldT>  OpTypes;

  typedef typename OpTypes::InterpC2FX InterpX;
  typedef typename OpTypes::InterpC2FY InterpY;
  typedef typename OpTypes::InterpC2FZ InterpZ;
  typedef typename OpTypes::GradX      GradX;
  typedef typename OpTypes::GradY      GradY;
  typedef typename OpTypes::GradZ      GradZ;
  typedef typename OpTypes::DivX       DivX;
  typedef typename OpTypes::DivY       DivY;
  typedef typename OpTypes::DivZ       DivZ;

  const InterpX *interpX_;
  const InterpY *interpY_;
  const InterpZ *interpZ_;
  const GradX   *gradX_;
  const GradY   *gradY_;
  const GradZ   *gradZ_;
  const DivX    *divX_;
  const DivY    *divY_;
  const DivZ    *divZ_;

  MonolithicRHS( const Expr::Tag& dCoefTag,
                 const Expr::Tag& xconvFluxTag,
                 const Expr::Tag& yconvFluxTag,
                 const Expr::Tag& zconvFluxTag,
                 const Expr::Tag& phiTag,
                 const Expr::Tag& srcTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a MonolithicRHS expression
     *  @param resultTag the Expr::Tag for the value that this expression computes
     *  @param dCoefTag the Expr::Tag for the diffusion coefficient
     *  @param convFluxTag the Expr::Tag for the convective flux of \f$\rho*\phi\f$
     *  @param phiTag the Expr::Tag for the primitive variable
     *  @param srcTag the Expr::Tag for the source term
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& dCoefTag,
//             const Expr::Tag& xconvFluxTag,
//             const Expr::Tag& yconvFluxTag,
//             const Expr::Tag& zconvFluxTag,
             const Expr::Tag& phiTag,
             const Expr::Tag& srcTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag dCoefTag_, xconvFluxTag_, yconvFluxTag_, zconvFluxTag_, phiTag_, srcTag_;
  };

  ~MonolithicRHS();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // MonolithicRHS_Expr_h
