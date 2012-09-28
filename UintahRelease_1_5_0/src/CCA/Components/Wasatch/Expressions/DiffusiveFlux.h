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

#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

//-- ExprLib includes --//
#include <expression/Expression.h>

//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


/**
 *  \ingroup Expressions
 *  \class  DiffusiveFlux
 *  \author James C. Sutherland, Amir Biglari
 *  \date July, 2011. (Originally created: June, 2010).
 *
 *  \brief Calculates a simple diffusive flux of the form
 *         \f$ J_i = -\rho (\Gamma + \Gamma_T) \frac{\partial \phi}{\partial x_i} \f$
 *         where \f$i=1,2,3\f$ is the coordinate direction.
 *         This requires knowledge of a the velocity field.
 *
 *  \tparam ScalarT the type for the scalar primary variable.
 *
 *  \tparam FluxT the type for the diffusive flux which is actually the scalarT faces.
 *
 *  Note that this requires the diffusion coefficient, \f$\Gamma\f$,
 *  to be evaluated at the same location as \f$J_i\f$ and
 *  \f$\frac{\partial \phi}{\partial x_i}\f$.
 */
template< typename ScalarT, typename FluxT >
class DiffusiveFlux
  : public Expr::Expression< FluxT >
{
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FluxT>::type  SVolInterpT;

  const bool  isConstCoef_, isTurbulent_;
  const Expr::Tag phiTag_, coefTag_, rhoTag_, turbDiffTag_;
  const double coefVal_;

  const GradT*          gradOp_;
  const SVolInterpT* sVolInterpOp_;

  const ScalarT*   phi_;
  const SVolField* turbDiff_;
  const SVolField* rho_;
  const FluxT*     coef_;

  DiffusiveFlux( const Expr::Tag& rhoTag,
                 const Expr::Tag& turbDiffTag,
                 const Expr::Tag& phiTag,
                 const Expr::Tag& coefTag );

  DiffusiveFlux( const Expr::Tag& rhoTag,
                 const Expr::Tag& turbDiffTag,
                 const Expr::Tag& phiTag,
                 const double coefTag );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\rho \Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$J\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a diffusive flux given expressions for
     *         \f$\phi\f$, \f$\Gamma_T\f$ and \f$\Gamma\f$
     *
     *  \param phiTag the Expr::Tag for the scalar field
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the flux field).
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
        isConstCoef_( false ),
        phit_     ( phiTag      ),
        coeft_    ( coefTag     ),
        rhot_     ( rhoTag      ),
        turbDifft_( turbDiffTag ),
        coef_     ( 0.0         )
    {}

    /**
     *  \brief Construct a diffusive flux given expressions for
     *         \f$\phi\f$ and \f$\Gamma_T\f$ and a constant value for \f$\Gamma\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coef the value (constant in space and time) for the
     *         diffusion coefficient.
     *
     *  \param turbDiffTag the Expr::Tag for the turbulent diffusivity which will be interpolated to FluxT field.
     *
     *  \param rhoTag the Expr::Tag for the density which will be interpolated to FluxT field.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const double coef,
             const Expr::Tag& turbDiffTag = Expr::Tag(),
             const Expr::Tag rhoTag = Expr::Tag() )
      : ExpressionBuilder(result),
        isConstCoef_( true        ),
        phit_       ( phiTag      ),
        coeft_      (             ),
        rhot_       ( rhoTag      ),
        turbDifft_  ( turbDiffTag ),
        coef_       ( coef        )
    {}

    Expr::ExpressionBase* build() const
    {
      if( isConstCoef_ ) return new DiffusiveFlux<ScalarT, FluxT>( rhot_, turbDifft_, phit_, coef_  );
      else               return new DiffusiveFlux<ScalarT, FluxT>( rhot_, turbDifft_, phit_, coeft_ );
    }
  private:
    const bool isConstCoef_;
    const Expr::Tag phit_,coeft_,rhot_,turbDifft_;
    const double coef_;
  };

  ~DiffusiveFlux();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};




/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveFlux2
 *  \authors James C. Sutherland, Amir Biglari
 *  \date July,2011. (Originally created: June, 2010).
 *
 *  \brief Calculates a generic diffusive flux, \f$J = -\rho (\Gamma + \Gamma_T)
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is located at 
 *         the same location as \f$\phi\f$.
 *
 *  \tparam ScalarT the type for the scalar primary variable.
 *
 *  \tparam FluxT the type for the diffusive flux which is actually the scalarT faces.
 *
 *  \tparam InterpT The type of operator used in interpolating
 *       \f$\Gamma\f$ from the location of \f$\phi\f$ to the location
 *       of \f$\frac{\partial \phi}{\partial x}\f$
 */
template< typename ScalarT,
          typename FluxT >
class DiffusiveFlux2
  : public Expr::Expression< FluxT >
{
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Gradient,   ScalarT,  FluxT>::type  GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,ScalarT,  FluxT>::type  InterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::Interpolant,SVolField,FluxT>::type  SVolInterpT;

  const bool isTurbulent_;
  const Expr::Tag phiTag_, coefTag_, rhoTag_, turbDiffTag_;

  const GradT* gradOp_;
  const InterpT* interpOp_;
  const SVolInterpT* sVolInterpOp_;

  const ScalarT*   phi_;
  const SVolField* turbDiff_;
  const SVolField* rho_;
  const ScalarT*   coef_;

  DiffusiveFlux2( const Expr::Tag& rhoTag,
                  const Expr::Tag& turbDiffTag,
                  const Expr::Tag& phiTag,
                  const Expr::Tag& coefTag );

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
     *  \brief Construct a DiffusiveFlux2::Builder object for
     *         registration with an Expr::ExpressionFactory.
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
      turbDifft_( turbDiffTag )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new DiffusiveFlux2<ScalarT,FluxT>( rhot_, turbDifft_, phit_, coeft_ ); }
  private:
    const Expr::Tag phit_,coeft_,rhot_,turbDifft_;
  };

  ~DiffusiveFlux2();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // DiffusiveFlux_Expr_h
