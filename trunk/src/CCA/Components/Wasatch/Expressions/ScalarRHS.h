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

#ifndef ScalarRHS_h
#define ScalarRHS_h

#include <map>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

#include "RHSTerms.h"


#include <CCA/Components/Wasatch/PatchInfo.h>


/**
 *  \ingroup 	Expressions
 *  \class 	ScalarRHS
 *  \author 	James C. Sutherland
 *
 *  \brief Support for a basic scalar transport equation involving
 *         any/all of advection, diffusion and reaction.
 *
 *  \tparam FieldT - the type of field for the RHS.
 *
 *  The ScalarRHS Expression defines a template class for basic
 *  transport equations.  Each equation is templated on an interpolant
 *  and divergence operator, from which the field types are deduced.
 *
 *  The user provides expressions to calculate the advecting velocity,
 *  diffusive fluxes and/or source terms.  This will then calculate
 *  the full RHS for use with the time integrator.
 *
 *  The general form of the ScalarRHS is assumed to be:
 *  \f[
 *    -\frac{\partial \alpha_x \rho \phi u_x}{\partial x}
 *    -\frac{\partial \alpha_y \rho \phi u_y}{\partial y}
 *    -\frac{\partial \alpha_z \rho \phi u_z}{\partial z}
 *    -\frac{\partial \alpha_x J_{\phi_x}}{\partial x}
 *    -\frac{\partial \alpha_y J_{\phi_y}}{\partial y}
 *    -\frac{\partial \alpha_z J_{\phi_z}}{\partial z}
 *    + \alpha_V s_\phi
 *  \f]
 *  where \f$\alpha_i\f$ are area fractions and \f$\alpha_V\f$ is a volume fraction.
 *
 *  This implementation also accommodates constant density form of the RHS:
 *  \f[
 *    \frac{\partial \phi}{\partial t} =
 *     \frac{1}{\rho} \left[
 *     - \nabla\cdot \phi \vec{u}
 *     - \nabla\cdot\vec{V}^d_\phi
 *     + \frac{1}{\rho} s_\phi
 *    \right]
 *  \f]
 *  where \f$\vec{V}^d_\phi = \frac{\vec{J}_\phi}{\rho}\f$ is the diffusion velocity.
 *
 *  It also supports the weak form of the RHS:
 *  \f[
 *    \frac{\partial \phi}{\partial t} =
 *     \frac{1}{\rho}\left[
 *     - \phi \frac{\partial \rho}{\partial t}
 *     - \nabla\cdot\rho\phi\vec{u}
 *     - \nabla\cdot\vec{J}_\phi
 *     + s_\phi
 *    \right]
 *  \f]
 *  both with and without embedded boundaries.
 */
template< typename FieldT >
class ScalarRHS : public Expr::Expression<FieldT>
{
protected:

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  DensityInterpT;

  typedef typename SpatialOps::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFluxT; ///< The type of field for the x-face variables.
  typedef typename FaceTypes::YFace YFluxT; ///< The type of field for the y-face variables.
  typedef typename FaceTypes::ZFace ZFluxT; ///< The type of field for the z-face variables.

  typedef typename SpatialOps::BasicOpTypes<FieldT> OpTypes;
  typedef typename OpTypes::DivX   DivX; ///< Divergence operator (surface integral) in the x-direction
  typedef typename OpTypes::DivY   DivY; ///< Divergence operator (surface integral) in the y-direction
  typedef typename OpTypes::DivZ   DivZ; ///< Divergence operator (surface integral) in the z-direction

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  SVolToFieldTInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,XVolField,XFluxT>::type  XVolToXFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,YVolField,YFluxT>::type  YVolToYFluxInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ZVolField,ZFluxT>::type  ZVolToZFluxInterpT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,XFluxT,FieldT>::type  XFluxToFieldInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,YFluxT,FieldT>::type  YFluxToFieldInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ZFluxT,FieldT>::type  ZFluxToFieldInterpT;

public:



  /**
   *  \class Builder
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief builder for ScalarRHS objects.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \brief Constructs a builder for a ScalarRHS object.
     *  \param result the tag for the ScalarRHS
     *  \param fieldInfo the FieldTagInfo object that holds information for the
     *   various expressions that form the RHS.
     *  \param densityTag density tag for cases that we have constant density and a source term.
     *  \param isConstDensity a boolean o show if density is constant or not.
     *  \param isStrongForm true (default) for the strong form of the governing equation, false otherwise.
     *  \param divrhouTag the Tag for divrhou.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const Expr::Tag& densityTag,
             const bool isConstDensity,
             const bool isStrongForm=true,
             const Expr::Tag divrhouTag=Expr::Tag() );

    /**
     *  \brief Constructs a builder for a ScalarRHS object. This is being
     *         used by ScalarTransportEquation.
     *  \param result the value of this expression
     *  \param fieldInfo the FieldTagInfo object that holds information for the
     *   various expressions that form the RHS.
     *  \param srcTags extra source terms to attach to this RHS.
     *  \param densityTag density tag for cases that we have constant density and a source term.
     *  \param isConstDensity a boolean to show if density is constant or not.
     *  \param isStrongForm true (default) for the strong form of the governing equation, false otherwise.
     *  \param divrhouTag the Tag for divrhou.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const Expr::TagList srcTags,
             const Expr::Tag& densityTag,
             const bool isConstDensity,
             const bool isStrongForm=true,
             const Expr::Tag divrhouTag=Expr::Tag() );
    virtual ~Builder(){}
    virtual Expr::ExpressionBase* build() const;
  protected:
    const FieldTagInfo info_;
    const Expr::TagList srcT_;
    const Expr::Tag densityT_, divrhouTag_;
    const bool isConstDensity_, isStrongForm_;
  };

  virtual void evaluate();
  virtual void sensitivity( const Expr::Tag& varTag );
  virtual void bind_operators( const SpatialOps::OperatorDatabase& opDB );

protected:

  const Expr::Tag convTagX_, convTagY_, convTagZ_;
  const Expr::Tag diffTagX_, diffTagY_, diffTagZ_;

  const bool haveConvection_, haveDiffusion_;
  const bool doXConv_, doYConv_, doZConv_;
  const bool doXDiff_, doYDiff_, doZDiff_;
  const bool doXDir_, doYDir_, doZDir_;

  const Expr::Tag volFracTag_, xAreaFracTag_, yAreaFracTag_, zAreaFracTag_;
  const bool haveVolFrac_, haveXAreaFrac_, haveYAreaFrac_, haveZAreaFrac_;

  const Expr::Tag densityTag_;

  DECLARE_FIELDS( SVolField, rho_, volfrac_, divrhou_, phi_ )
  DECLARE_FIELDS( SVolField, xSVol_, ySVol_, zSVol_ )
  DECLARE_FIELD( XVolField, xareafrac_ )
  DECLARE_FIELD( YVolField, yareafrac_ )
  DECLARE_FIELD( ZVolField, zareafrac_ )

  WasatchCore::UintahPatchContainer* patchContainer_;

  // Operators
  const SVolToFieldTInterpT* volFracInterpOp_;
  const XVolToXFluxInterpT* xAreaFracInterpOp_;
  const YVolToYFluxInterpT* yAreaFracInterpOp_;
  const ZVolToZFluxInterpT* zAreaFracInterpOp_;

  const XFluxToFieldInterpT* xInterpOp_;
  const YFluxToFieldInterpT* yInterpOp_;
  const ZFluxToFieldInterpT* zInterpOp_;

  const bool isConstDensity_, strongForm_;
  const DensityInterpT* densityInterpOp_;

  // things requried for weak form:
  const Expr::Tag phiTag_, divrhouTag_;

  Expr::TagList srcTags_;

  const DivX* divOpX_;
  const DivY* divOpY_;
  const DivZ* divOpZ_;

  DECLARE_FIELDS( XFluxT, xConvFlux_, xDiffFlux_ )
  DECLARE_FIELDS( YFluxT, yConvFlux_, yDiffFlux_ )
  DECLARE_FIELDS( ZFluxT, zConvFlux_, zDiffFlux_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, srcTerms_ )

  ScalarRHS( const FieldTagInfo& fieldTags,
             const Expr::TagList srcTags,
             const Expr::Tag& densityTag,
             const Expr::Tag& divrhouTag,
             const bool isConstDensity,
             const bool isStrongForm = true );

  virtual ~ScalarRHS();
};

#endif
