/**
 * \file Functions.h
 *
 * Copyright (c) 2011 The University of Utah
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
#ifndef Expr_Function_h
#define Expr_Function_h

#include <cmath>

#include <expression/Expression.h>

namespace Expr{


/**
 *  @class ConstantExpr
 *  @author James C. Sutherland
 *  @date  April, 2009
 *  @brief Sets a field to a constant
 */
template< typename ValT >
class ConstantExpr : public Expression<ValT>
{
public:

  /**
   *  @brief Builds a ConstantExpr Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    Builder( const Tag& name,  ///< the name of the field to set
             const double a    ///< The constant value
             )
    : ExpressionBuilder(name), a_(a) {}
    ~Builder(){}
    ExpressionBase* build() const{ return new ConstantExpr(a_); }
  private:
    const double a_;
  };

  void advertise_dependents( ExprDeps& exprDeps ){}
  void bind_fields( const FieldManagerList& fml ){}
  void evaluate()
  {
    using SpatialOps::operator<<=;
    this->value() <<= val_;
  }

protected:
  ~ConstantExpr(){}
  ConstantExpr( const double value )
  : Expression<ValT>(),
    val_( value )
  {
    this->set_gpu_runnable(true);
  }

  const double val_;
};

//====================================================================

/**
 *  @class FunctionExpr1D
 *  @author James C. Sutherland
 *  @date  April, 2009
 *  @brief Base class for simple functions of a single independent variable.
 */
template< typename ValT >
class FunctionExpr1D : public Expression<ValT>
{
public:

  void advertise_dependents( ExprDeps& exprDeps )
  {
    if( ivarIsExpression_ )  exprDeps.requires_expression( ivarTag_ );
  }

  void bind_fields( const FieldManagerList& fml )
  {
    x_ = &fml.field_ref<ValT>( ivarTag_ );
  }

protected:
  virtual ~FunctionExpr1D(){}
  /**
   *  This constructor should only be called by children.
   *
   *  @param indepVarTag Specifies the independent variable for this Expression.
   *  @param ivarIsExpression If true, the independent variable will
   *         be treated as an expression - otherwise the independent
   *         variable will be treated as a "static" field.
   */
  FunctionExpr1D( const Tag& indepVarTag,
                  const bool ivarIsExpression )
  : Expression<ValT>(),
    ivarTag_( indepVarTag ),
    ivarIsExpression_( ivarIsExpression )
  {}
  const ValT* x_;
  const Tag ivarTag_;
  const bool ivarIsExpression_;
};

//====================================================================

/**
 *  @class FunctionExpr2D
 *  @author James C. Sutherland
 *  @date  April, 2009
 *  @brief Base class for simple functions of a single independent variable.
 */
template< typename ValT >
class FunctionExpr2D : public Expression<ValT>
{
public:

  void advertise_dependents( ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( ivar1Tag_ );
    exprDeps.requires_expression( ivar2Tag_ );
  }

  void bind_fields( const FieldManagerList& fml )
  {
    const typename FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
    x1_ = &fm.field_ref( ivar1Tag_ );
    x2_ = &fm.field_ref( ivar2Tag_ );
  }

protected:
  virtual ~FunctionExpr2D(){}
  /**
   *  This constructor should only be called by children.
   *
   *  @param indepVar1Tag Specifies the independent variable for this Expression.
   *  @param indepVar2Tag Specifies the independent variable for this Expression.
   */
  FunctionExpr2D( const Tag& indepVar1Tag,
                  const Tag& indepVar2Tag )
  : Expression<ValT>(),
    ivar1Tag_( indepVar1Tag ), ivar2Tag_( indepVar2Tag )
  {}
  const ValT* x1_;
  const ValT* x2_;
  const Tag ivar1Tag_, ivar2Tag_;
};

//====================================================================

/**
 *  @class LinearFunction
 *  @date   April, 2009
 *  @author James C. Sutherland
 *  @brief Implements a linear function of a single variable.
 */
template< typename ValT >
class LinearFunction : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a LinearFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    /**
     * @param depVarTag        The dependent variable set by this function
     * @param indepVarTag      The independent variable to use for this function
     * @param slope            The slope of the line
     * @param intercept        The intercept of the line
     * @param ivarIsExpression true -> treat the independent variable
     *        as an Expression, otherwise treat as a "static" variable.
     */
    Builder( const Tag& depVarTag,
             const Tag& indepVarTag,
             const double slope,
             const double intercept,
             const bool ivarIsExpression=true )
    : ExpressionBuilder( depVarTag ),
      a_( slope ),
      b_( intercept ),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ~Builder(){}
    ExpressionBase* build() const
    {
      return new LinearFunction<ValT>( ivarTag_, a_, b_, ivarIsExpression_ );
    }

  private:
    const double a_, b_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const ValT& x = *(this->x_);
    phi <<= x*a_ + b_;
  }

private:
  LinearFunction( const Tag& indepVarTag,
                  const double slope,
                  const double intercept,
                  const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    a_( slope ),
    b_( intercept )
  {
    this->set_gpu_runnable(true);
  }

  const double a_, b_;
};

//====================================================================

/**
 *  @class GaussianFunction
 *  @author James C. Sutherland
 *  @date April, 2009
 *  @brief Implements a gaussian function of a single independent variable.
 *
 * The gaussian function is written as
 *  \f[
 *    f(x) = y_0 + a \exp\left( \frac{\left(x-x_0\right)^2 }{2\sigma^2} \right)
 *  \f]
 * where
 *  - \f$x_0\f$ is the mean (center of the gaussian)
 *  - \f$\sigma\f$ is the standard deviation (width of the gaussian)
 *  - \f$a\f$ is the amplitude of the gaussian
 *  - \f$y_0\f$ is the baseline value.
 */
template< typename ValT >
class GaussianFunction : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a GaussianFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    Builder( const Tag& depVarTag,   ///<   dependent variable tag
             const Tag& indepVarTag, ///< independent variable tag
             const double a,         ///< Amplitude of the Gaussian spike
             const double stddev,    ///< Standard deviation
             const double mean,      ///< Mean of the function
             const double yo=0.0,    ///< baseline value
             const bool ivarIsExpression=true ///< true if independent variable comes from an expression
             )
    : ExpressionBuilder(depVarTag),
      a_(a),
      sigma_(stddev),
      mean_(mean),
      yo_(yo),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ~Builder(){}
    ExpressionBase* build() const
    {
      return new GaussianFunction<ValT>( ivarTag_, a_, sigma_, mean_, yo_, ivarIsExpression_ );
    }

  private:
    const double a_, sigma_, mean_, yo_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const double denom = 1.0/(2.0*sigma_*sigma_);
    const ValT& x = *(this->x_);
    phi <<= yo_ + a_ * exp( -denom * (x-mean_)*(x-mean_) );
  }


private:

  GaussianFunction( const Tag& indepVarTag,
                    const double a,
                    const double stddev,
                    const double mean,
                    const double yo,
                    const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    a_( a ), sigma_( stddev ), mean_( mean ), yo_( yo )
  {
    this->set_gpu_runnable(true);
  }

  const double a_, sigma_, mean_, yo_;
};

//====================================================================

/**
 *  @class GaussianFunction2D
 *  @author James C. Sutherland
 *  @date Aug, 2014
 *  @brief Implements a 2D Gaussian function of a single independent variable.
 *
 * The gaussian function is written as
 *  \f[
 *    f(x,y) = f_0 + a \exp\left( -\frac{\left(x-x_0\right)^2 }{2\sigma_x^2} - \frac{\left(y-y_0\right)^2 }{2\sigma_y^2} \right)
 *  \f]
 * where
 *  - \f$(x_0,y_0)\f$ is the mean (center of the Gaussian)
 *  - \f$\sigma_x\f$ and \f$\sigma_y\f$ is the standard deviation (width of the Gaussian)
 *  - \f$a\f$ is the amplitude of the Gaussian
 *  - \f$f_0\f$ is the baseline value.
 */
template< typename FieldT >
class GaussianFunction2D : public FunctionExpr2D<FieldT>
{
public:

  /**
   *  @brief Builds a GaussianFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    Builder( const Tag depVarTag,    ///<   dependent variable tag
             const Tag indepVar1Tag, ///< independent variable tag
             const Tag indepVar2Tag, ///< independent variable tag
             const double a,          ///< Amplitude of the Gaussian spike
             const double stddev1,    ///< Standard deviation on var1
             const double stddev2,    ///< Standard deviation on var2
             const double mean1,      ///< Mean of the function in x1
             const double mean2,      ///< Mean of the function in x2
             const double fo=0.0      ///< baseline value
             )
    : ExpressionBuilder( depVarTag ),
      a_( a ), sigma1_(stddev1), sigma2_(stddev2), mean1_(mean1), mean2_(mean2), fo_(fo),
      ivar1Tag_( indepVar1Tag ), ivar2Tag_( indepVar2Tag )
    {}
    ~Builder(){}
    ExpressionBase* build() const{
      return new GaussianFunction2D( ivar1Tag_, ivar2Tag_, a_, sigma1_, sigma2_, mean1_, mean2_, fo_ );
    }
  private:
    const double a_, sigma1_, sigma2_, mean1_, mean2_, fo_;
    const Tag ivar1Tag_, ivar2Tag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& f = this->value();
    const FieldT& x1 = *this->x1_;
    const FieldT& x2 = *this->x2_;
    f <<= fo_ + a_ * exp( - ( x1 - mean1_)*( x1 - mean1_) * den1_
                          - ( x2 - mean2_)*( x2 - mean2_) * den2_
                        );
  }

private:

  GaussianFunction2D( const Tag& indepVar1Tag,
                      const Tag& indepVar2Tag,
                      const double a,
                      const double stddev1,
                      const double stddev2,
                      const double mean1,
                      const double mean2,
                      const double fo )
  : FunctionExpr2D<FieldT>( indepVar1Tag, indepVar2Tag ),
    a_( a ), mean1_(mean1), mean2_(mean2), fo_(fo),
    den1_( 1.0/(stddev1*stddev1*2.0) ),
    den2_( 1.0/(stddev2*stddev2*2.0) )
  {
    this->set_gpu_runnable(true);
  }
  const double a_, mean1_, mean2_, fo_;
  const double den1_, den2_;
};

//====================================================================

/**
 *  @class SinFunction
 *  @author James C. Sutherland
 *  @date April, 2009
 *  @brief Implements a sin function of a single independent variable,
 *         \f$ y = a \sin( b x ) + c\f$, where a, b, and c are constants.
 */
template< typename ValT >
class SinFunction : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a SinFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    /**
     *  @brief Build a SinFunction, \f$ y = a*\sin(b x) + c\f$.
     *
     *  @param depVarTag   the Expr::Tag for the dependent variable (y).
     *  @param indepVarTag the Expr::Tag for the independent variable (x).
     *  @param a the amplitude
     *  @param b the frequency
     *  @param c the offset
     *  @param ivarIsExpression [default=true] if true, then the
     *         independent variable will be treated as an expression
     *         that must be evaluated prior to executing this
     *         expression.
     */
    Builder( const Tag& depVarTag,
             const Tag& indepVarTag,
             const double a=1.0,
             const double b=1.0,
             const double c=0.0,
             const bool ivarIsExpression=true )
    : ExpressionBuilder(depVarTag),
      a_(a), b_(b), c_(c),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ~Builder(){}
    ExpressionBase* build() const
    {
      return new SinFunction<ValT>( ivarTag_, a_, b_, c_, ivarIsExpression_ );
    }

  private:
    const double a_, b_, c_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const ValT& x = *(this->x_);
    phi <<= a_ * sin( b_ * x ) + c_;
  }

private:

  SinFunction( const Tag& indepVarTag,
               const double a,
               const double b,
               const double c,
               const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    a_(a), b_(b), c_(c)
  {
    this->set_gpu_runnable(true);
  }

  const double a_, b_, c_;
};

//====================================================================

/**
 *  @class DoubleTanhFunction
 *  @author James C. Sutherland
 *  @date April, 2009
 *  @brief Implements a double hyperbolic tangent function for a single
 *         independent variable. The double Tanh expression can represent
 *         a smooth square pulse.
 *
 *  The double Tanh function implemented here is given as
 *    \f[
 *       f(x) = \frac{A}{2} \left(1+\tanh\left(\frac{x-L_1}{w}\right)\right) \left(1-\frac{1}{2}\tanh\left(\frac{x-L_2}{w}\right)\right)
 *     \f]
 *  where w is the width of the transition, A is the amplitude of the
 *  transition, \f$L_1\f$ is the midpoint for the upward transition,
 *  and \f$L_2\f$ is the midpoint for the downward transition. This
 */
template< typename ValT >
class DoubleTanhFunction : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a DoubleTanhFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    /**
     *  Builds a DoubleTanhFunction object
     *  \f[
     *     f(x) = \frac{A}{2} \left(1+\tanh\left(\frac{x-L_1}{w}\right)\right) \left(1-\frac{1}{2}\tanh\left(\frac{x-L_2}{w}\right)\right)
     *  \f]
     */
    Builder( const Tag& depVarTag,   ///< The dependent variable
             const Tag& indepVarTag, ///< The independent variable
             const double L1,        ///< The midpoint for the upward transition
             const double L2,        ///< The midpoint for the downward transition
             const double w,         ///< The width of the transition
             const double A=1.0,     ///< The amplitude of the transition
             const bool ivarIsExpression=true ///< If true, the independent variable will be treated as an expression - otherwise the independent variable will be treated as a "static" field.
             )
    : ExpressionBuilder(depVarTag),
      L1_(L1), L2_(L2), w_(w), A_(A),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ~Builder(){}
    ExpressionBase* build() const
    {
      return new DoubleTanhFunction<ValT>( ivarTag_, L1_, L2_, w_, A_, ivarIsExpression_ );
    }

  private:
    const double L1_, L2_, w_, A_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const ValT& x = *(this->x_);
    phi <<= (0.5*A_) * ( 1.0 + tanh( (x-L1_)/w_ ) ) * ( 1.0-0.5*(1.0+tanh( (x-L2_)/w_ ) ) );
  }


private:

  DoubleTanhFunction( const Tag& indepVarTag,
                      const double L1,
                      const double L2,
                      const double w,
                      const double A,
                      const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    L1_(L1), L2_(L2), w_(w), A_(A)
  {
    this->set_gpu_runnable(true);
  }

  const double L1_, L2_, w_, A_;
};

//====================================================================

/**
 *  @class ParabolicFunction
 *  @author Tony Saad
 *  @date March, 2010
 *  @brief Implements a parabolic function of a single independent variable,
 *         \f$ y = a x^2 + b x + c\f$, where a, b, and c are constants.
 */
template< typename ValT >
class ParabolicFunction : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a SinFunction Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    /**
     *  @brief Build a SinFunction, \f$ y = a x^2 + b x + c \f$.
     *
     *  @param depVarTag the Expr::Tag for the dependent variable (x).
     *  @param indepVarTag the Expr::Tag for the independent variable (x).
     *  @param a the coefficient of \f$ x^2 \f$
     *  @param b the coefficient of \f$ x \f$
     *  @param c the constant
     *  @param ivarIsExpression [default=true] if true, then the
     *         independent variable will be treated as an expression
     *         that must be evaluated prior to executing this
     *         expression.
     */
    Builder( const Tag& depVarTag,
             const Tag& indepVarTag,
             const double a=1.0,
             const double b=1.0,
             const double c=0.0,
             const bool ivarIsExpression=true )
    : ExpressionBuilder( depVarTag ),
      a_(a), b_(b), c_(c),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ~Builder(){}
    ExpressionBase* build() const
    {
      return new ParabolicFunction<ValT>( ivarTag_, a_, b_, c_, ivarIsExpression_ );
    }

  private:
    const double a_, b_, c_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const ValT& x = *(this->x_);
    phi <<= (a_ * x) * x + b_*x + c_;
  }


private:

  ParabolicFunction( const Tag& indepVarTag,
                     const double a,
                     const double b,
                     const double c,
                     const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    a_(a), b_(b), c_(c)
  {
    this->set_gpu_runnable(true);
  }

  const double a_, b_, c_;
};

//====================================================================

//====================================================================

/**
 *  @class CoaxialJet
 *  @author Naveen Punati
 *  @date April, 2009
 *  @brief Implements a coaxial jet configuration for a single
 *         independent variable.
 *
 */
template< typename ValT >
class CoaxialJet : public FunctionExpr1D<ValT>
{
public:

  /**
   *  @brief Builds a CoaxialJet Expression.
   */
  struct Builder : public ExpressionBuilder
  {
    Builder( const Tag& depVarTag,
             const Tag& indepVarTag,
             const double A1,
             const double A2,
             const double w,
             const double L1,
             const double L2,
             const double L3,
             const double L4,
             const double L5,
             const double L6,
             const double B,
             const bool ivarIsExpression=true )
    : ExpressionBuilder(depVarTag),
      A1_(A1),A2_(A2),w_(w),L1_(L1), L2_(L2), L3_(L3), L4_(L4),L5_(L5), L6_(L6), B_(B),
      ivarTag_( indepVarTag ),
      ivarIsExpression_( ivarIsExpression )
    {}

    ExpressionBase* build() const
    {
      return new CoaxialJet<ValT>( ivarTag_, A1_,A2_, w_,L1_, L2_, L3_, L4_, L5_, L6_, B_, ivarIsExpression_ );
    }

  private:
    const double L1_, L2_, L3_,L4_,L5_,L6_,w_, A1_,A2_, B_;
    const Tag ivarTag_;
    const bool ivarIsExpression_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ValT& phi = this->value();
    const ValT& x = *(this->x_);
    phi <<= (0.5*A2_) * ( 1.0 + tanh( (x-L1_)/w_ ) ) * ( 1.0-0.5*(1.0+tanh( (x-L2_)/w_ ) ) )
          + (0.5*A1_) * ( 1.0 + tanh( (x-L3_)/w_ ) ) * ( 1.0-0.5*(1.0+tanh( (x-L4_)/w_ ) ) )
          + (0.5*A2_) * ( 1.0 + tanh( (x-L5_)/w_ ) ) * ( 1.0-0.5*(1.0+tanh( (x-L6_)/w_ ) ) )
          + B_;
  }


private:

  CoaxialJet( const Tag& indepVarTag,
              const double A1, const double A2,
              const double w,
              const double L1, const double L2, const double L3,
              const double L4, const double L5, const double L6,
              const double B,
              const bool ivarIsExpression )
  : FunctionExpr1D<ValT>( indepVarTag, ivarIsExpression ),
    A1_(A1),A2_(A2),w_(w), L1_(L1), L2_(L2), L3_(L3),L4_(L4),L5_(L5),L6_(L6), B_(B)
  {
    this->set_gpu_runnable(true);
  }

  const double L1_, L2_,L3_,L4_,L5_,L6_, w_, A1_,A2_, B_;
};

//====================================================================

} // namespace Expr

#endif // Expr_Function_h
