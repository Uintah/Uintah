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

#ifndef ScalarEOSCoupling_h
#define ScalarEOSCoupling_h

#include <map>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

#include "RHSTerms.h"


/**
 *  \ingroup 	Expressions
 *  \class 	  ScalarEOSCoupling
 *  \author 	Tony Saad
 *  \date     August, 2015
 *
 *  \brief This expression computes the scalar-EOS (equation of state) coupling into the pressu
 poisson equation.
 *
 *  \tparam FieldT - the type of field for the RHS.
 *
 */
template< typename FieldT >
class ScalarEOSCoupling : public Expr::Expression<FieldT>
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

public:



  /**
   *  \class Builder
   *  \author Tony Saad
   *  \date   August, 2015
   *
   *  \brief builder for ScalarEOSCoupling objects.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \brief Constructs a builder for a ScalarEOSCoupling object.
     *  \param result the tag for the ScalarEOSCoupling
     *  \param fieldInfo the FieldTagInfo object that holds information for the various expressions that form the RHS.
     *  \param rhoStarTag density tag for the (n+1) time level.
     *  \param dRhoDPhiTag is the tag for the Jacobian of the equation of state. This is usually tabulated.
     *  \param isStrongForm true (default) for the strong form of the governing equation, false otherwise.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const Expr::Tag& rhoStarTag,
             const Expr::Tag& dRhoDPhiTag,
             const bool isStrongForm=true );

    /**
     *  \brief Constructs a builder for a ScalarEOSCoupling object. This is being
     *         used by ScalarTransportEquation.
     *  \param result the tag for the ScalarEOSCoupling
     *  \param fieldInfo the FieldTagInfo object that holds information for the RHS side.
     *  \param rhoStarTag density tag for the (n+1) time level.
     *  \param dRhoDPhiTag is the tag for the Jacobian of the equation of state. This is usually tabulated.
     *  \param isStrongForm true (default) for the strong form of the governing equation, false otherwise.
     *  \param srcTags extra source terms to attach to this RHS.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const Expr::TagList srcTags,
             const Expr::Tag& rhoStarTag,
             const Expr::Tag& dRhoDPhiTag,
             const bool isStrongForm=true );
    virtual ~Builder(){}
    virtual Expr::ExpressionBase* build() const;
  protected:
    const FieldTagInfo info_;
    const Expr::TagList srcT_;
    const Expr::Tag rhoStarTag_, dRhoDPhiTag_;
    const bool isStrongForm_;
  };

  virtual void evaluate();
  virtual void bind_operators( const SpatialOps::OperatorDatabase& opDB );

protected:

  const Expr::Tag diffTagX_, diffTagY_, diffTagZ_, dRhoDPhiTag_;
  const Expr::TagList srcTags_;

  const bool haveDiffusion_;
  const bool doXDiff_, doYDiff_, doZDiff_;
  const bool doXDir_, doYDir_, doZDir_;

  const Expr::Tag volFracTag_, xAreaFracTag_, yAreaFracTag_, zAreaFracTag_;
  const bool haveVolFrac_, haveXAreaFrac_, haveYAreaFrac_, haveZAreaFrac_;
  const bool isStrongForm_;

  DECLARE_FIELDS( SVolField, rhoStar_, dRhoDPhi_, volfrac_ )

  DECLARE_FIELD( XVolField, xareafrac_ )
  DECLARE_FIELD( YVolField, yareafrac_ )
  DECLARE_FIELD( ZVolField, zareafrac_ )
  DECLARE_FIELD( XFluxT, xDiffFlux_ )
  DECLARE_FIELD( YFluxT, yDiffFlux_ )
  DECLARE_FIELD( ZFluxT, zDiffFlux_ )

  DECLARE_VECTOR_OF_FIELDS( FieldT, srcTerms_ )

  // Operators
  const SVolToFieldTInterpT* volFracInterpOp_;
  const XVolToXFluxInterpT* xAreaFracInterpOp_;
  const YVolToYFluxInterpT* yAreaFracInterpOp_;
  const ZVolToZFluxInterpT* zAreaFracInterpOp_;
  const DensityInterpT*     densityInterpOp_;
  const DivX*               divOpX_;
  const DivY*               divOpY_;
  const DivZ*               divOpZ_;

  ScalarEOSCoupling( const FieldTagInfo& fieldTags,
                     const Expr::TagList& srcTags,
                     const Expr::Tag& rhoStarTag,
                     const Expr::Tag& dRhoDPhiTag,
                     const bool isStrongForm = true );

  virtual ~ScalarEOSCoupling();
};

#endif
