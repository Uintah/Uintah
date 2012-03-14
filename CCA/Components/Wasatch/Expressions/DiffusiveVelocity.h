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

#ifndef DiffusiveVelocity_Expr_h
#define DiffusiveVelocity_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveVelocity
 *  \author Amir Biglari
 *  \date   Aug, 2011
 *
 *  \brief Calculates a simple diffusive velocity of the form
 *         \f$ V_i = -\Gamma \frac{\partial \phi}{\partial x_i} \f$
 *         where \f$i=1,2,3\f$ is the coordinate direction.
 *         This requires knowledge of a the velocity field. This
 *         expression is suitable mostly for the cases that we
 *         have constant density or Temperature equation etc. to be
 *         used instead of diffusive flux expression
 *
 *  Note that this requires the diffusion coefficient, \f$\Gamma\f$,
 *  to be evaluated at the same location as \f$V_i\f$ and
 *  \f$\frac{\partial \phi}{\partial x_i}\f$.
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b GradT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *  </ul>
 *
 */
template< typename GradT >
class DiffusiveVelocity
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType VelT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const bool isConstCoef_;
  const Expr::Tag phiTag_, coefTag_;
  const double coefVal_;

  const GradT* gradOp_;
  const ScalarT* phi_;
  const VelT* coef_;

  DiffusiveVelocity( const Expr::Tag phiTag,
                     const Expr::Tag coefTag );

  DiffusiveVelocity( const Expr::Tag phiTag,
                     const double coefTag );

public:
  /**
   *  \brief Builder for a diffusive velocity \f$ V = -\Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$J\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a diffusive velocity given expressions for
     *         \f$\phi\f$ and \f$\Gamma\f$
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the diffusive velocity field).
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& coefTag )
      : ExpressionBuilder(result),
        isConstCoef_( false ),
        phit_ ( phiTag  ),
        coeft_( coefTag ),
        coef_ ( 0.0     )
    {}

    /**
     *  \brief Construct a diffusive velocity given an expression for
     *         \f$\phi\f$ and a constant value for \f$\Gamma\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coef the value (constant in space and time) for the
     *         diffusion coefficient.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const double coef )
      : ExpressionBuilder(result),
        isConstCoef_( true ),
        phit_ ( phiTag      ),
        coeft_( Expr::Tag() ),
        coef_ ( coef        )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const
    {
      if( isConstCoef_ ) return new DiffusiveVelocity<GradT>( phit_, coef_ );
      else               return new DiffusiveVelocity<GradT>( phit_, coeft_);
    }
  private:
    const bool isConstCoef_;
    const Expr::Tag phit_, coeft_;
    const double coef_;
  };

  ~DiffusiveVelocity();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};




/**
 *  \ingroup WasatchExpressions
 *  \class  DiffusiveVelocity2
 *  \author Amir Biglari
 *  \date   Aug, 2011
 *
 *  \brief Calculates a generic diffusive velocity, \f$V = -\Gamma
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is
 *         located at the same location as \f$\phi\f$.
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b GradT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *  <li> \b InterpT The type of operator used in interpolating
 *       \f$\Gamma\f$ from the location of \f$\phi\f$ to the location
 *       of \f$\frac{\partial \phi}{\partial x}\f$
 *  </ul>
 */
template< typename GradT,
          typename InterpT >
class DiffusiveVelocity2
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType VelT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const Expr::Tag phiTag_, coefTag_;

  const GradT* gradOp_;
  const InterpT* interpOp_;

  const ScalarT* phi_;
  const ScalarT* coef_;

  DiffusiveVelocity2( const Expr::Tag& phiTag,
                      const Expr::Tag& coefTag );

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
     *  \brief Construct a DiffusiveVelocity2::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *
     *  \param phiTag the Expr::Tag for the scalar field.
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the scalar field).
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& phiTag,
             const Expr::Tag& coefTag )
      : ExpressionBuilder(result),
        phit_(phiTag), coeft_( coefTag )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new DiffusiveVelocity2<GradT,InterpT>( phit_, coeft_ ); }
  private:
    const Expr::Tag phit_,coeft_;
  };

  ~DiffusiveVelocity2();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // DiffusiveVelocity_Expr_h
