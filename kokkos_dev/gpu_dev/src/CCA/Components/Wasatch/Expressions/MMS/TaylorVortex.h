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
 *   - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class VelocityX : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds a Taylor Vortex velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
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

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  VelocityX( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double A,
             const double nu );
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;
  const double* t_;
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
 *   - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class VelocityY : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds a Taylor Vortex Velocity Function in y direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
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

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  VelocityY( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const Expr::Tag& tTag,
             const double A,
             const double nu );
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;
  const double* t_;
};

//====================================================================

/**
 *  \class GradPX
 *  \author Amir Biglari
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
 *   - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class GradPX : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds an Expression for Taylor Vortex Pressure Function gradient in x direction.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
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

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  GradPX( const Expr::Tag& xTag,
          const Expr::Tag& yTag,
          const Expr::Tag& tTag,
          const double A,
          const double nu );
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;
  const double* t_;
};

//====================================================================

/**
 *  \class GradPY
 *  \author Amir Biglari
 *  \date October, 2010
 *  \brief Implements the taylor vortex pressure field gradient in y direction
 *
 *  The taylor vortex pressure field is given as
 *  \f[
 *    p(x,y,t)= \frac{-A^{2}}{4}\left[\cos(2(x-t))+\cos(2(y-t))\right]\exp\left(-4\nu t\right)
 *  \f]
 *  So, the gradient in y direction will be
 *  \f[
 *    \frac{\partial p}{\partial x}(x,y,t)= \frac{A^{2}}{2}\sin(2(y-t))\exp\left(-4\nu t\right)
 *  \f]
 *  where
 *   - \f$A\f$ is the amplitude of the function
 *   - \f$t\f$ is the time variable
 *   - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class GradPY : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds an Expression for Taylor vortex pressure gradient in y direction.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
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

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  GradPY( const Expr::Tag& xTag,
          const Expr::Tag& yTag,
          const Expr::Tag& tTag,
          const double A,
          const double nu );
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;
  const double* t_;
};

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
 */
template< typename ValT >
class TaylorGreenVel3D : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds a Taylor Vortex velocity function in x direction Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,  ///< x coordinate
             const Expr::Tag& yTag,  ///< y coordinate
             const Expr::Tag& zTag,  ///< z
             const double angle=0.1    ///< Kinematic viscosity of the fluid
            );
    Expr::ExpressionBase* build() const;
  private:
    const double angle_;
    const Expr::Tag xt_, yt_, zt_;
  };

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  TaylorGreenVel3D( const Expr::Tag& xTag,
                    const Expr::Tag& yTag,
                    const Expr::Tag& zTag,
                    const double angle );
  const double angle_;
  const Expr::Tag xTag_, yTag_, zTag_;
  const ValT* x_;
  const ValT* y_;
  const ValT* z_;
};

//====================================================================


#endif // Wasatch_Taylor_Vortex
