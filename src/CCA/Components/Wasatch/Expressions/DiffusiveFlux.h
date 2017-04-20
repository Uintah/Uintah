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

#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

//-- ExprLib includes --//
#include <expression/Expression.h>

//-- SpatialOps includes --//
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveFlux
 *  \authors James C. Sutherland, Amir Biglari
 *  \date July,2011. (Originally created: June, 2010).
 *
 *  \brief Calculates a generic diffusive flux, \f$J = -\rho (\Gamma + \Gamma_T)
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is located at 
 *         the same location as \f$\phi\f$.
 *
 *  \tparam FluxT the type for the diffusive flux which is actually the ScalarT faces.
 */
template< typename FluxT >
class DiffusiveFlux
  : public Expr::Expression< FluxT >
{
  typedef typename SpatialOps::VolType<FluxT>::VolField ScalarT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,  FluxT>::type  InterpT;

  const bool isTurbulent_, isConstCoef_;

  const double coefVal_;

  const GradT* gradOp_;
  const InterpT* interpOp_;

  DECLARE_FIELDS(ScalarT, phi_, turbDiff_, rho_, coef_)
  
  DiffusiveFlux( const Expr::Tag& rhoTag,
                 const Expr::Tag& turbDiffTag,
                 const Expr::Tag& phiTag,
                 const Expr::Tag& coefTag );

  DiffusiveFlux( const Expr::Tag& rhoTag,
                 const Expr::Tag& turbDiffTag,
                 const Expr::Tag& phiTag,
                 const double coefVal );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = - \rho (\Gamma + \Gamma_T) \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the same location as 
   *         \f$\phi\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a DiffusiveFlux::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *
     *  \param result the diffusive flux
     *
     *  \param phiTag the Expr::Tag for the scalar field.
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the scalar field).
     *
     *  \param turbDiffTag the Expr::Tag for the turbulent diffusivity which will be interpolated to FluxT field.
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& coefTag,
             const Expr::Tag& turbDiffTag = Expr::Tag(),
             const Expr::Tag rhoTag = Expr::Tag() )
    : ExpressionBuilder(result),
      phit_     ( phiTag      ),
      coeft_    ( coefTag     ),
      rhot_     ( rhoTag      ),
      turbDifft_( turbDiffTag ),
      coefVal_  ( -1.0        )
    {}

    /**
     *  \brief Construct a DiffusiveFlux::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *
     *  \param result the diffusive flux
     *
     *  \param phiTag the Expr::Tag for the scalar field.
     *
     *  \param coefVal the (constant) value for the diffusion coefficient
     *         (located at same points as the scalar field).
     *
     *  \param turbDiffTag the Expr::Tag for the turbulent diffusivity which will be interpolated to FluxT field.
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const double coefVal,
             const Expr::Tag& turbDiffTag = Expr::Tag(),
             const Expr::Tag rhoTag = Expr::Tag() )
    : ExpressionBuilder(result),
      phit_     ( phiTag      ),
      coeft_    (             ),
      rhot_     ( rhoTag      ),
      turbDifft_( turbDiffTag ),
      coefVal_  ( coefVal     )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag phit_,coeft_,rhot_,turbDifft_;
    const double coefVal_;
  };

  ~DiffusiveFlux();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // DiffusiveFlux_Expr_h
