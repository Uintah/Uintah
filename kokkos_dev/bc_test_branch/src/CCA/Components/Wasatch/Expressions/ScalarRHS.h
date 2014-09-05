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

#ifndef ScalarRHS_h
#define ScalarRHS_h

#include <map>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include "RHSTerms.h"


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
 *  Note: In the case that we are solving a scalar transport equation with
 *        constant density we move out the density by devision from all
 *        terms except the source term. For the source term in this case we
 *        divide the specified source term expression by density here in
 *        ScalarRHS.
 *        So, you should be careful with the cases that source terms are
 *        NOT defined in the INPUT FILE but they will be added to the RHS
 *        automatically during the solution process and they are almost
 *        impossible to track (e.g. in ODT solver)
 *
 */
template< typename FieldT >
class ScalarRHS : public Expr::Expression<FieldT>
{
protected:

  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  DensityInterpT;

  typedef typename SpatialOps::structured::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFluxT; ///< The type of field for the x-face variables.
  typedef typename FaceTypes::YFace YFluxT; ///< The type of field for the y-face variables.
  typedef typename FaceTypes::ZFace ZFluxT; ///< The type of field for the z-face variables.

  typedef typename SpatialOps::structured::BasicOpTypes<FieldT> OpTypes;
  typedef typename OpTypes::DivX   DivX; ///< Divergence operator (surface integral) in the x-direction
  typedef typename OpTypes::DivY   DivY; ///< Divergence operator (surface integral) in the y-direction
  typedef typename OpTypes::DivZ   DivZ; ///< Divergence operator (surface integral) in the z-direction

  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FieldT>::type  SVolToFieldTInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,XVolField,XFluxT>::type  XVolToXFluxInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,YVolField,YFluxT>::type  YVolToYFluxInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,ZVolField,ZFluxT>::type  ZVolToZFluxInterpT;

public:



  /**
   *  \class Builder
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief builder for ScalarRHS objecst.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:

    /**
     *  \brief Constructs a builder for a ScalarRHS object.
     *
     *  \param fieldInfo the FieldTagInfo object that holds
     *         information for the various expressions that form the
     *         RHS.
     *
     *  \param densityTag density tag for cases that we have contsant density and a source term.
     *
     *  \param isConstDensity a boolean o show if density is constant or not.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const Expr::Tag& densityTag,
             const Expr::Tag& volFracTag,
             const Expr::Tag& xAreaFracTag,
             const Expr::Tag& yAreaFracTag,
             const Expr::Tag& zAreaFracTag,
             const bool isConstDensity);

    /**
     *  \brief Constructs a builder for a ScalarRHS object. This is being
     *         used by ScalarTransportEqu.
     *
     *  \param result the value of this expression
     *
     *  \param fieldInfo the FieldTagInfo object that holds
     *         information for the various expressions that form the
     *         RHS.
     *
     *  \param srcTags extra source terms to attach to this RHS.
     *
     *  \param densityTag density tag for cases that we have contsant density and a source term.
     *
     *  \param isConstDensity a boolean o show if density is constant or not.
     */
    Builder( const Expr::Tag& result,
             const FieldTagInfo& fieldInfo,
             const std::vector<Expr::Tag>& srcTags,
             const Expr::Tag& densityTag,
             const Expr::Tag& volFracTag,
             const Expr::Tag& xAreaFracTag,
             const Expr::Tag& yAreaFracTag,
             const Expr::Tag& zAreaFracTag,
             const bool isConstDensity);
    virtual ~Builder(){}
    virtual Expr::ExpressionBase* build() const;
  protected:
    const FieldTagInfo info_;
    const std::vector<Expr::Tag> srcT_;
    const Expr::Tag volfracT_;
    const Expr::Tag xareafracT_;
    const Expr::Tag yareafracT_;
    const Expr::Tag zareafracT_;
    const Expr::Tag densityT_;
    const bool isConstDensity_;
  };

  virtual void evaluate();
  virtual void advertise_dependents( Expr::ExprDeps& exprDeps );
  virtual void bind_fields( const Expr::FieldManagerList& fml );
  virtual void bind_operators( const SpatialOps::OperatorDatabase& opDB );

protected:

  const Expr::Tag convTagX_, convTagY_, convTagZ_;
  const Expr::Tag diffTagX_, diffTagY_, diffTagZ_;

  const bool haveConvection_, haveDiffusion_;
  const bool doXDir_, doYDir_, doZDir_;

  const Expr::Tag volFracTag_, xAreaFracTag_, yAreaFracTag_, zAreaFracTag_;
  const bool haveVolFrac_, haveXAreaFrac_, haveYAreaFrac_, haveZAreaFrac_;

  const Expr::Tag densityTag_;

  const SVolField* rho_;
  const SVolField* volfrac_;
  const XVolField* xareafrac_;
  const YVolField* yareafrac_;
  const ZVolField* zareafrac_;
  const SVolToFieldTInterpT* volFracInterpOp_;
  const XVolToXFluxInterpT* xAreaFracInterpOp_;
  const YVolToYFluxInterpT* yAreaFracInterpOp_;
  const ZVolToZFluxInterpT* zAreaFracInterpOp_;

  const bool isConstDensity_;
  const DensityInterpT* densityInterpOp_;

  std::vector<Expr::Tag> srcTags_;

  const DivX* divOpX_;
  const DivY* divOpY_;
  const DivZ* divOpZ_;

  const XFluxT *xConvFlux_, *xDiffFlux_;
  const YFluxT *yConvFlux_, *yDiffFlux_;
  const ZFluxT *zConvFlux_, *zDiffFlux_;

  typedef std::vector<const FieldT*> SrcVec;
  SrcVec srcTerm_;

  void nullify_fields();

  static Expr::Tag resolve_field_tag( const FieldSelector field,
                                      const FieldTagInfo& info );

  ScalarRHS( const FieldTagInfo& fieldTags,
             const std::vector<Expr::Tag>& srcTags,
             const Expr::Tag densityTag,
             const Expr::Tag volFracTag,
             const Expr::Tag xAreaFracTag,
             const Expr::Tag yAreaFracTag,
             const Expr::Tag zAreaFracTag,
             const bool  isConstDensity );

  virtual ~ScalarRHS();
};

#endif
