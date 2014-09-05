#ifndef Wasatch_Taylor_Vortex
#define Wasatch_Taylor_Vortex

#include <cmath>

#include <expression/Expr_Expression.h>

//namespace Expr{

	
//====================================================================
	
/**
 *  @class VelocityX
 *  @author Amir Biglari
 *  @date October, 2010
 *  @brief Implements the taylor vortex velocity field in x direction
 *
 * The taylor vortex velocity field in x direction is given as
 *  \f[
 *    u(x,y,t)=1 - A \cos(x - t) \sin(y - t) \exp\left(-2 \nu t \right)
 *  \f]
 * where
 *  - \f$A\f$ is the amplitude of the function
 *  - \f$t\f$ is the time variable 
 *  - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class VelocityX : public Expr::Expression<ValT>
{
public:
		
  /**
   *  @brief Builds a Taylor Vortex Velocity Function in x dirextion Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag xTag,
             const Expr::Tag yTag,
             const Expr::Tag tTag,
             const double A=1.0,         ///< Amplitude of the function
             const double nu=0.1 );		 ///< Kinematic viscosity of the fluid
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
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
             const double nu,
             const Expr::ExpressionID& id,
             const Expr::ExpressionRegistry& reg);
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;	
  const double* t_;
};
	
//====================================================================
	
/**
 *  @class VelocityY
 *  @author Amir Biglari
 *  @date October, 2010
 *  @brief Implements the taylor vortex velocity field in y direction
 *
 * The taylor vortex velocity field in y direction is given as
 *  \f[
 *    v(x,y,t)=1 + A \sin(x - t) \cos(y - t) \exp\left(-2 \nu t \right)
 *  \f]
 * where
 *  - \f$A\f$ is the amplitude of the function
 *  - \f$t\f$ is the time variable 
 *  - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class VelocityY : public Expr::Expression<ValT>
{
public:
		
  /**
   *  @brief Builds a Taylor Vortex Velocity Function in y dirextion Expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag xTag,
             const Expr::Tag yTag,
             const Expr::Tag tTag,
             const double A=1.0,         ///< Amplitude of the function
             const double nu=0.1 );		 ///< Kinematic viscosity of the fluid
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
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
             const double nu,
             const Expr::ExpressionID& id,
             const Expr::ExpressionRegistry& reg);
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;	
  const double* t_;
};
	
//====================================================================
	
/**
 *  @class GradPX
 *  @author Amir Biglari
 *  @date October, 2010
 *  @brief Implements the taylor vortex pressure field gradient in x direction
 *
 * The taylor vortex pressure field is given as
 *  \f[
 *    p(x,y,t)= \frac{-A^{2}}{4}\left[\cos(2(x-t))+\cos(2(y-t))\right]\exp\left(-4\nu t\right)
 *  \f]
 * So, the gradient in x direction will be
 *  \f[
 *    \frac{\partial p}{\partial x}(x,y,t)= \frac{A^{2}}{2}\sin(2(x-t))\exp\left(-4\nu t\right)
 *  \f]
 * where
 *  - \f$A\f$ is the amplitude of the function
 *  - \f$t\f$ is the time variable 
 *  - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class GradPX : public Expr::Expression<ValT>
{
public:
		
  /**
   *  @brief Builds an Expression for Taylor Vortex Pressure Function gradient in x direction.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag xTag,
             const Expr::Tag yTag,
             const Expr::Tag tTag,
             const double A=1.0,         ///< Amplitude of the function
             const double nu=0.1 );		 ///< Kinematic viscosity of the fluid
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
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
          const double nu,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg);
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;	
  const double* t_;
};
	
//====================================================================

/**
 *  @class GradPY
 *  @author Amir Biglari
 *  @date October, 2010
 *  @brief Implements the taylor vortex pressure field gradient in y direction
 *
 * The taylor vortex pressure field is given as
 *  \f[
 *    p(x,y,t)= \frac{-A^{2}}{4}\left[\cos(2(x-t))+\cos(2(y-t))\right]\exp\left(-4\nu t\right)
 *  \f]
 * So, the gradient in y direction will be
 *  \f[
 *    \frac{\partial p}{\partial x}(x,y,t)= \frac{A^{2}}{2}\sin(2(y-t))\exp\left(-4\nu t\right)
 *  \f]
 * where
 *  - \f$A\f$ is the amplitude of the function
 *  - \f$t\f$ is the time variable 
 *  - \f$\nu\f$ is kinematic viscousity
 */
template< typename ValT >
class GradPY : public Expr::Expression<ValT>
{
public:
		
  /**
   *  @brief Builds an Expression for Taylor Vortex Pressure Function gradient in y direction.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag xTag,
             const Expr::Tag yTag,
             const Expr::Tag tTag,
             const double A=1.0,         ///< Amplitude of the function
             const double nu=0.1 );		 ///< Kinematic viscosity of the fluid
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const;
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
          const double nu,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg);
  const double A_, nu_;
  const Expr::Tag xTag_, yTag_, tTag_;
  const ValT* x_;
  const ValT* y_;	
  const double* t_;
};
	
//====================================================================
	
	



//####################################################################
//
//                          IMPLEMENTATION
//
//####################################################################




	
//--------------------------------------------------------------------
	
template<typename ValT>
VelocityX<ValT>::
VelocityX( const Expr::Tag& xtag,
           const Expr::Tag& ytag,
           const Expr::Tag& ttag,
           const double A,
           const double nu,
           const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg )
  : Expr::Expression<ValT>( id, reg ),
    A_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{
  std::cout << xTag_ << " : " << yTag_ << " : " << tTag_ << std::endl;
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityX<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityX<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ValT>& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
	
  const Expr::FieldManager<double>& timeFM = fml.template field_manager<double>();
  t_ = &timeFM.field_ref( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityX<ValT>::
evaluate()
{
  ValT& phi = this->value();
  phi <<= 1.0 - A_ * cos( *x_ - *t_ ) * sin( *y_ - *t_ ) * exp( -2.0 * nu_ * *t_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
VelocityX<ValT>::Builder::
Builder( const Expr::Tag xtag,
         const Expr::Tag ytag,
         const Expr::Tag ttag,
         const double A,
         const double nu)
  : A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
Expr::ExpressionBase*
VelocityX<ValT>::Builder::
build( const Expr::ExpressionID& id,
       const Expr::ExpressionRegistry& reg ) const
{
  return new VelocityX<ValT>( xt_, yt_, tt_, A_, nu_, id, reg );
}
	
//--------------------------------------------------------------------

//====================================================================
	
//--------------------------------------------------------------------
	
template<typename ValT>
VelocityY<ValT>::
VelocityY( const Expr::Tag& xtag,
           const Expr::Tag& ytag,
           const Expr::Tag& ttag,
           const double A,
           const double nu,
           const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg )
  : Expr::Expression<ValT>( id, reg ),
    A_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityY<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityY<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ValT>& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
		
  const Expr::FieldManager<double>& timeFM = fml.template field_manager<double>();
  t_ = &timeFM.field_ref( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
VelocityY<ValT>::
evaluate()
{
  ValT& phi = this->value();
  phi <<= 1.0 + A_ * sin( *x_ - *t_ ) * cos( *y_ - *t_ ) * exp( -2.0*nu_ * *t_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
VelocityY<ValT>::Builder::
Builder( const Expr::Tag xtag,
         const Expr::Tag ytag,
         const Expr::Tag ttag,
         const double A,
         const double nu )
  : A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
Expr::ExpressionBase*
VelocityY<ValT>::Builder::
build( const Expr::ExpressionID& id,
       const Expr::ExpressionRegistry& reg ) const
{
  return new VelocityY<ValT>( xt_, yt_, tt_, A_, nu_, id, reg );
}
	
//--------------------------------------------------------------------

//====================================================================
	
//--------------------------------------------------------------------
	
template<typename ValT>
GradPX<ValT>::
GradPX( const Expr::Tag& xtag,
        const Expr::Tag& ytag,
        const Expr::Tag& ttag,
        const double A,
        const double nu,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg )
  : Expr::Expression<ValT>( id, reg ),
    A_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPX<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPX<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ValT>& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
		
  const Expr::FieldManager<double>& timeFM = fml.template field_manager<double>();
  t_ = &timeFM.field_ref( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPX<ValT>::
evaluate()
{
  ValT& phi = this->value();
  phi <<= -1.0*(A_*A_/2.0) * sin( 2.0*( *x_ - *t_ ) ) * exp( -4.0*nu_ * *t_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
GradPX<ValT>::Builder::
Builder( const Expr::Tag xtag,
         const Expr::Tag ytag,
         const Expr::Tag ttag,
         const double A,
         const double nu )
  : A_(A), 
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
Expr::ExpressionBase*
GradPX<ValT>::Builder::
build( const Expr::ExpressionID& id,
       const Expr::ExpressionRegistry& reg ) const
{
  return new GradPX<ValT>( xt_, yt_, tt_, A_, nu_, id, reg );
}
	
//--------------------------------------------------------------------

//====================================================================
	
//--------------------------------------------------------------------
	
template<typename ValT>
GradPY<ValT>::
GradPY( const Expr::Tag& xtag,
        const Expr::Tag& ytag,
        const Expr::Tag& ttag,
        const double A,
        const double nu,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg )
  : Expr::Expression<ValT>( id, reg ),
    A_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPY<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPY<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ValT>& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
		
  const Expr::FieldManager<double>& timeFM = fml.template field_manager<double>();
  t_ = &timeFM.field_ref( tTag_ );
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
void
GradPY<ValT>::
evaluate()
{
  ValT& phi = this->value();
  phi <<= -1.0*(A_*A_/2.0) * sin( 2.0*(*y_-*t_) ) * exp(-4.0 * nu_ * *t_);
}
	
//--------------------------------------------------------------------
	
template< typename ValT >
GradPY<ValT>::Builder::
Builder( const Expr::Tag xtag,
         const Expr::Tag ytag,
         const Expr::Tag ttag,
         const double A,
         const double nu )
  : A_(A), 
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}
	
//--------------------------------------------------------------------
	
template< typename ValT >
Expr::ExpressionBase*
GradPY<ValT>::Builder::
build( const Expr::ExpressionID& id,
       const Expr::ExpressionRegistry& reg ) const
{
  return new GradPY<ValT>( xt_, yt_, tt_, A_, nu_, id, reg );
}
	
//--------------------------------------------------------------------
	
//} // namespace Expr

#endif // Wasatch_Taylor_Vortex
