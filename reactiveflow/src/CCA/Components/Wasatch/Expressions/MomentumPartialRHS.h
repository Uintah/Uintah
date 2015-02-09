/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#ifndef MomentumPartialRHS_Expr_h
#define MomentumPartialRHS_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>


/**
 *  \class 	MomRHSPart
 *  \ingroup 	Expressions
 *
 *  \brief Calculates the RHS of a momentum equation excluding the pressure gradient term.
 *
 *  \tparam FieldT the type of field for the momentum RHS (nominally
 *          XVolField, YVolField, ZVolField).
 *
 *  \f[
 *    \frac{\partial \rho u_i}{\partial t} =
 *         - \nabla\cdot (\rho u_i \mathbf{u})
 *         - \nabla\cdot \tau_{*i}
 *         - \frac{\partial p}{\partial x_i}
 *         - \rho g_i
 *  \f]
 *
 *  where \f$\tau_{*i}\f$ is row of the Strain tensor corresponding to
 *  the component of momentum this equation is describing.  We define
 *
 *  \f[
 *     F_i \equiv -\frac{\partial \rho u_i u_j}{\partial x_j}
 *                -\frac{\partial \tau_{ij}}{\partial x_j}
 *                -\rho g_i
 *  \f]
 *  so that the momentum equations are written as
 *  \f[
 *    \frac{\partial \rho u_i}{\partial t} = F_i -\frac{\partial p}{\partial x_i}
 *  \f]
 *  This expression calculates \f$F_i\f$.
 */
template< typename FieldT >
class MomRHSPart
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::FaceTypes<FieldT>::XFace  XFluxT;
  typedef typename SpatialOps::FaceTypes<FieldT>::YFace  YFluxT;
  typedef typename SpatialOps::FaceTypes<FieldT>::ZFace  ZFluxT;

  typedef typename SpatialOps::BasicOpTypes<FieldT>  OpTypes;

  typedef typename OpTypes::DivX  DivX;
  typedef typename OpTypes::DivY  DivY;
  typedef typename OpTypes::DivZ  DivZ;
  
  // interpolant for density: svol to fieldT
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  DensityInterpT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,XFluxT>::type  SVol2XFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,YFluxT>::type  SVol2YFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,ZFluxT>::type  SVol2ZFluxInterpT;

  const Expr::Tag cfluxXt_, cfluxYt_, cfluxZt_, viscTag_, tauXt_, tauYt_, tauZt_, densityt_, bodyForcet_, srcTermt_, emptyTag_;
  const Expr::Tag volfract_;

  const XFluxT    *cFluxX_, *tauX_;
  const YFluxT    *cFluxY_, *tauY_;
  const ZFluxT    *cFluxZ_, *tauZ_;
  const SVolField *density_, *visc_;
  const FieldT    *bodyForce_;
  const FieldT    *srcTerm_;

  const FieldT* volfrac_;
  
  const DivX* divXOp_;
  const DivY* divYOp_;
  const DivZ* divZOp_;

  const SVol2XFluxInterpT* sVol2XFluxInterpOp_;
  const SVol2YFluxInterpT* sVol2YFluxInterpOp_;
  const SVol2ZFluxInterpT* sVol2ZFluxInterpOp_;
  
  const DensityInterpT* densityInterpOp_;
  
  const bool is3dconvdiff_;

  MomRHSPart( const Expr::Tag& convFluxX,
              const Expr::Tag& convFluxY,
              const Expr::Tag& convFluxZ,
              const Expr::Tag& viscTag,
              const Expr::Tag& tauX,
              const Expr::Tag& tauY,
              const Expr::Tag& tauZ,
              const Expr::Tag& densityTag,
              const Expr::Tag& bodyForceTag,
              const Expr::Tag& srcTermTag,
              const Expr::Tag& volFracTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag cfluxXt_, cfluxYt_, cfluxZt_, viscTag_, tauXt_, tauYt_, tauZt_, densityt_, bodyForcet_, srcTermt_;
    const Expr::Tag volfract_;
    
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& convFluxX,
             const Expr::Tag& convFluxY,
             const Expr::Tag& convFluxZ,
             const Expr::Tag& viscTag,
             const Expr::Tag& tauX,
             const Expr::Tag& tauY,
             const Expr::Tag& tauZ,
             const Expr::Tag& densityTag,
             const Expr::Tag& bodyForceTag,
             const Expr::Tag& srcTermTag,
             const Expr::Tag& volFracTag );

    Expr::ExpressionBase* build() const;
  };

  ~MomRHSPart();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // MomentumPartialRHS_Expr_h
