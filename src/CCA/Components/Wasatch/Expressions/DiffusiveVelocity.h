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

#ifndef DiffusiveVelocity_Expr_h
#define DiffusiveVelocity_Expr_h

#include <expression/Expression.h>

//-- SpatialOps includes --//
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveVelocity
 *  \author Amir Biglari, James C. Sutherland
 *  \date   Aug, 2011
 *
 *  \brief Calculates a generic diffusive velocity, \f$V = -(\Gamma + \Gamma_T)
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is
 *         located at the same location as \f$\phi\f$.
 *
 *  \tparam VelT the type for the diffusive velocity
 */
template< typename VelT >
class DiffusiveVelocity : public Expr::Expression<VelT>
{
  typedef typename SpatialOps::VolType<VelT>::VolField ScalarT;

  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  VelT>::type  GradT;
  typedef typename SpatialOps::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,  VelT>::type  InterpT;

  const bool isTurbulent_, isConstCoef_;

  const GradT* gradOp_;
  const InterpT* interpOp_;
  
  DECLARE_FIELDS(ScalarT, phi_, turbDiff_, coef_)
  const double coefVal_;  ///< the value of the diffusion coefficient if constant

  DiffusiveVelocity( const Expr::Tag& turbDiffTag,
                     const Expr::Tag& phiTag,
                     const Expr::Tag& coefTag );
  DiffusiveVelocity( const Expr::Tag& turbDiffTag,
                     const Expr::Tag& phiTag,
                     const double coefVal );

public:
  /**
   *  \brief Builder for a diffusive velocity \f$ V = -\Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$\phi\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a DiffusiveVelocity::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *  \param result the diffusive velocity tag
     *  \param phiTag the Expr::Tag for the scalar field.
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the scalar field).
     *  \param turbDiffTag the Expr::Tag for the turbulent diffusivity which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& coefTag,
             const Expr::Tag& turbDiffTag = Expr::Tag() )
      : ExpressionBuilder(result),
        phit_(phiTag), coeft_( coefTag ), turbDifft_( turbDiffTag ), coefVal_(0.0)
    {}

    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const double coefVal,
             const Expr::Tag& turbDiffTag = Expr::Tag() )
      : ExpressionBuilder(result),
        phit_(phiTag), coeft_(), turbDifft_( turbDiffTag ), coefVal_(coefVal)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag phit_, coeft_, turbDifft_;
    const double coefVal_;
  };

  ~DiffusiveVelocity();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  void sensitivity( const Expr::Tag& varTag );

};
#endif // DiffusiveVelocity_Expr_h
