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

#ifndef Wasatch_Taylor_Vortex
#define Wasatch_Taylor_Vortex

#include <expression/Expression.h>


/**
 *  \class VelocityX
 *  \author Amir Biglari
 *  \date October, 2010
 *  \brief Implements the taylor vortex velocity field in x direction
 *
 *  The taylor vortex velocity field in x direction is given as
 *  \f[
 *    u(x,y,t)=1 - A \cos(x - t) \sin(y - t) \exp\left(-2 \nu t \right)
 *  \f]
 *  where
 *   - \f$A\f$ is the amplitude of the function
 *   - \f$t\f$ is the time variable
 *   - \f$\nu\f$ is kinematic viscosity
 */
template< typename FieldT >
class VelocityX : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
public:

  /**
   *  \brief Builds a Taylor Vortex velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,///< x-velocity tag
             const Expr::Tag& xTag,  ///< x coordinate
             const Expr::Tag& yTag,  ///< y coordinate
             const Expr::Tag& tTag,  ///< time
             const double A=1.0,     ///< Amplitude of the function
             const double nu=0.1     ///< Kinematic viscosity of the fluid
             );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double A_, nu_;
    const Expr::Tag xt_, yt_, tt_;
  };

  void evaluate();

private:

  VelocityX( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double A,
             const double nu );
  const double a_, nu_;
  
  DECLARE_FIELDS(FieldT, x_, y_)
  DECLARE_FIELD(TimeField, t_)
};

//====================================================================

/**
 *  \class VelocityY
 *  \author Amir Biglari
 *  \date October, 2010
 *  \brief Implements the taylor vortex velocity field in y direction
 *
 *  The taylor vortex velocity field in y direction is given as
 *  \f[
 *    v(x,y,t)=1 + A \sin(x - t) \cos(y - t) \exp\left(-2 \nu t \right)
 *  \f]
 *  where
 *   - \f$A\f$ is the amplitude of the function
 *   - \f$t\f$ is the time variable
 *   - \f$\nu\f$ is kinematic viscosity
 */
template< typename FieldT >
class VelocityY : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
public:

  /**
   *  \brief Builds a Taylor Vortex Velocity Function in y direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result, ///< y-velocity tag
             const Expr::Tag& xTag,  ///< x-coordinate
             const Expr::Tag& yTag,  ///< y-coordinate
             const Expr::Tag& tTag,  ///< time
             const double A=1.0,    ///< Amplitude of the function
             const double nu=0.1    ///< Kinematic viscosity of the fluid
             );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double A_, nu_;
    const Expr::Tag xt_, yt_, tt_;
  };

  void evaluate();

private:

  VelocityY( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double A,
             const double nu );
  const double a_, nu_;
  DECLARE_FIELDS(FieldT, x_, y_)
  DECLARE_FIELD(TimeField, t_)
};

//====================================================================

/**
 *  \class GradP
 *  \author Amir Biglari, Tony Saad
 *  \date October, 2010
 *  \brief Implements the taylor vortex pressure field gradient in x direction
 *
 *  The taylor vortex pressure field is given as
 *  \f[
 *    p(x,y,t)= \frac{-A^{2}}{4}\left[\cos(2(x-t))+\cos(2(y-t))\right]\exp\left(-4\nu t\right)
 *  \f]
 *  So, the gradient in x direction will be
 *  \f[
 *    \frac{\partial p}{\partial x}(x,y,t)= \frac{A^{2}}{2}\sin(2(x-t))\exp\left(-4\nu t\right)
 *  \f]
 *  where
 *   - \f$A\f$ is the amplitude of the function
 *   - \f$t\f$ is the time variable
 *   - \f$\nu\f$ is kinematic viscosity
 */
template< typename FieldT >
class GradP : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
public:

  /**
   *  \brief Builds an Expression for Taylor Vortex Pressure Function gradient in x direction.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,///< dp/dx tag
             const Expr::Tag& xTag,  ///< x-coordinate
             const Expr::Tag& tTag,  ///< time
             const double A=1.0,     ///< Amplitude of the function
             const double nu=0.1     ///< Kinematic viscosity of the fluid
             );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double A_, nu_;
    const Expr::Tag xt_, tt_;
  };

  void evaluate();

private:

  GradP( const Expr::Tag& xTag,
          const Expr::Tag& tTag,
          const double A,
          const double nu );
  const double a_, nu_;
  DECLARE_FIELDS(FieldT, x_)
  DECLARE_FIELD(TimeField, t_)
};

//====================================================================

/**
 *  \class VelX3D
 *  \author Tony Saad
 *  \date June, 2011
 *  \brief Implements the generalized Taylor-Green vortex three dimensional
velocity field. This is usually used as an initial condition for the velocity.
 *
 *  The taylor vortex velocity field in x direction is given as
 *  \f[
 *    u(x,y,z)= \frac{2}{\sqrt{3}} \sin(\theta + \frac{2\pi}{3}) \sin x \cos y \cos z
 *  \f]
 *  \f[
 *    v(x,y,z)= \frac{2}{\sqrt{3}} \sin(\theta - \frac{2\pi}{3}) \sin y \cos x \cos z
 *  \f]
 *  \f[
 *    w(x,y,z)= \frac{2}{\sqrt{3}} \sin(\theta) \sin z \cos x \cos y
 *  \f]
 *  where
 *   - \f$\theta\f$ is some angle specified by the user
 *  Note: that we implement only one expression for this velocity field. By merely
 *  shuffling the coordinates, we can generate all velocity components. This
 *  should be in processing the user input.
 *
 *  Citation:
 *    Brachet et. al., Small-scale structure of the Taylor-Green vortex,
 *    J. Fluid Mech, vol. 130, no. 41, p. 1452, 1983.
 */
template< typename FieldT >
class TaylorGreenVel3D : public Expr::Expression<FieldT>
{
public:

  /**
   *  \brief Builds a Taylor Vortex velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,///< velocity tag
             const Expr::Tag& xTag,  ///< coordinate 1
             const Expr::Tag& yTag,  ///< coordinate 2
             const Expr::Tag& zTag,  ///< coordinate 3
             const double angle=0.1  ///< \f$\theta\f$
            );
    Expr::ExpressionBase* build() const;
  private:
    const double angle_;
    const Expr::Tag xt_, yt_, zt_;
  };

  void evaluate();

private:

  TaylorGreenVel3D( const Expr::Tag& xTag,
                    const Expr::Tag& yTag,
                    const Expr::Tag& zTag,
                    const double angle );
  const double angle_;
  DECLARE_FIELDS(FieldT, x_, y_, z_)
};

//====================================================================


#endif // Wasatch_Taylor_Vortex
