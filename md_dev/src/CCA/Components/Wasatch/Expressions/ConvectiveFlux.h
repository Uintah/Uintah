/*
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

/* ----------------------------------------------------------------------------------------------
   %%%%%%      %%%%%%    %%      %%  %%      %%  %%%%%%%%    %%%%%%  %%%%%%%%%%  %%      %%%%%%    %%      %%
 %%          %%      %%  %%%%    %%  %%      %%  %%        %%            %%      %%    %%      %%  %%%%    %%
 %%          %%      %%  %%  %%  %%    %%  %%    %%%%%%    %%            %%      %%    %%      %%  %%  %%  %%
 %%          %%      %%  %%  %%  %%    %%  %%    %%        %%            %%      %%    %%      %%  %%  %%  %%
 %%          %%      %%  %%    %%%%    %%  %%    %%        %%            %%      %%    %%      %%  %%    %%%%
   %%%%%%      %%%%%%    %%      %%      %%      %%%%%%%%    %%%%%%      %%      %%      %%%%%%    %%      %%
 ---------------------------------------------------------------------------------------------- */
#ifndef ConvectiveFlux_h
#define ConvectiveFlux_h

//-- ExprLib includes --//
#include <expression/Expression.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

/**
 *  \ingroup Expressions
 *  \class   ConvectiveFlux
 *  \author  Tony Saad
 *  \date    July, 2010
 *
 *  \brief Creates an expression for the convective flux of a scalar
 *  given a velocity field \f$\mathbf{u}\f$ using central interpolation.
 *  We write the convective flux in conservation form as
    \f$ J_i = \rho \varphi u_i = \phi u_i \f$
 *  where \f$i=1,2,3\f$ is the coordinate direction. This requires
 *  knowledge of the velocity field.
 *
 *  Here, we are constructing the convective flux J_i, therefore, it
 *  is convenient to set \f$ \rho \varphi \equiv \phi\f$
 *
 *  \tparam PhiInterpT The type of operator used in forming
 *          \f$\frac{\partial \phi}{\partial x}\f$
 *
 *  \tparam VelInterpT The type of operator used in interpolating the
 *          velocity from volume to face fields
 */
template< typename PhiInterpT, typename VelInterpT > // scalar interpolant and velocity interpolant
class ConvectiveFlux
  : public Expr::Expression<typename PhiInterpT::DestFieldType>
{
  // PhiInterpT: an interpolant from staggered or non-staggered volume field to staggered or non-staggered face field
  typedef typename PhiInterpT::SrcFieldType  PhiVolT;  ///< source field is a scalar volume
  typedef typename PhiInterpT::DestFieldType PhiFaceT; ///< destination field is scalar face

  // VelInterpT: an interpolant from Staggered volume field to scalar face field
  typedef typename VelInterpT::SrcFieldType  VelVolT;  ///< source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;
  // the destination field of VelInterpT should be a PhiFaceT

  const Expr::Tag phiTag_, velTag_;
  const PhiVolT* phi_;
  const VelVolT* vel_;
  PhiInterpT* phiInterpOp_;
  const VelInterpT* velInterpOp_;

public:
  ConvectiveFlux( const Expr::Tag& phiTag,
                  const Expr::Tag& velTag );
  ~ConvectiveFlux();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag phiT_, velT_;

  public:
    /**
     *  \brief Construct a convective flux given an expression for
     *         \f$\phi\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field.
     *         This is located at cell centroids.
     *
     *  \param velTag the Expr::Tag for the velocity field.
     *         The velocity field is a face field.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& velTag )
      : ExpressionBuilder(result),
        phiT_(phiTag), velT_(velTag)
    {}
    Expr::ExpressionBase* build() const;
    ~Builder(){}
  };
};

/**
 *  \ingroup WasatchExpressions
 *  \class   ConvectiveFluxLimiter
 *  \author  Tony Saad
 *  \date    January, 2011
 *
 *  \brief Creates an expression for the convective flux of a scalar
 *  given a velocity field \f$\mathbf{u}\f$. We write the convective
 *  flux in conservation form as \f$ J_i = \rho \varphi u_i = \phi u_i
 *  \f$ where \f$i=1,2,3\f$ is the coordinate direction. This requires
 *  knowledge of the velocity field. The ConvectiveFluxLimiter is used for
 *  all interpolants that are dependent on a velocity field such as the
 *  Upwind scheme or any of the flux limiters such as Superbee.
 *
 *  Here, we are constructing the convective flux J_i, therefore, it
 *  is convenient to set \f$ \rho \varphi \equiv \phi\f$
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b PhiInterpT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$.  Interpolates from
 *       staggered or non-staggered volume field to staggered or
 *       non-staggered face field
 *  <li> \b VelInterpT The type of operator used in interpolating the
 *        velocity from volume to face fields. Interpolates from
 *        staggered volume field to scalar face field
 *  </ul>
 */
template< typename LimiterInterpT, typename PhiInterpLowT,
          typename PhiInterpHiT  , typename VelInterpT >
class ConvectiveFluxLimiter
  : public Expr::Expression<typename PhiInterpHiT::DestFieldType>
{
  typedef typename PhiInterpHiT::SrcFieldType  PhiVolT;  ///< source field is a scalar volume
  typedef typename PhiInterpHiT::DestFieldType PhiFaceT; ///< destination field is scalar face

  typedef typename VelInterpT::SrcFieldType  VelVolT;  ///< source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;

  const Expr::Tag phiTag_, velTag_, volFracTag_;
  const Wasatch::ConvInterpMethods limiterType_;

  const PhiVolT* phi_;
  const VelVolT* vel_;
  const PhiVolT* volFrac_;

  const bool isUpwind_;
  const bool isCentral_;
  const bool hasEmbeddedBoundary_;

  LimiterInterpT* psiInterpOp_;
  PhiInterpLowT*  phiInterpLowOp_;
  PhiInterpHiT*   phiInterpHiOp_;
  const VelInterpT* velInterpOp_;

  ConvectiveFluxLimiter( const Expr::Tag& phiTag,
                         const Expr::Tag& velTag,
                         Wasatch::ConvInterpMethods limiterType,
                         const Expr::Tag& volFracTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag phiT_, velT_, volFracT_;
    Wasatch::ConvInterpMethods limiterType_;
  public:
    /**
     *  \brief Construct an convective flux limiter given an expression
     *         for \f$\phi\f$.
     *
     *  \param phiTag the Expr::Tag for the scalar field.  This is
     *         located at cell centroids.
     *
     *  \param velTag the Expr::Tag for the velocity field.  The
     *         velocity field is a face field.
     *
     *  \param limiterType the type of flux limiter to use.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& velTag,
             Wasatch::ConvInterpMethods limiterType,
             const Expr::Tag volFracTag = Expr::Tag() )
      : ExpressionBuilder(result),
        phiT_( phiTag ), velT_( velTag ), volFracT_ ( volFracTag ),
        limiterType_( limiterType )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ConvectiveFluxLimiter<LimiterInterpT, PhiInterpLowT, PhiInterpHiT, VelInterpT>( phiT_, velT_, limiterType_, volFracT_ );
    }
  };

  ~ConvectiveFluxLimiter();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // /ConvectiveFlux_h
