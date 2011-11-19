#include <cmath>

#include "TaylorVortex.h"

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
{}

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
  using namespace SpatialOps;
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
  using namespace SpatialOps;
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
  using namespace SpatialOps;
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
  using namespace SpatialOps;
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

//==========================================================================

//--------------------------------------------------------------------

template<typename ValT>
TaylorGreenVel3D<ValT>::
TaylorGreenVel3D( const Expr::Tag& xtag,
          const Expr::Tag& ytag,
          const Expr::Tag& ztag,
          const double angle,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg )
: Expr::Expression<ValT>( id, reg ),
  angle_(angle), xTag_( xtag ), yTag_( ytag ), zTag_( ztag )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
TaylorGreenVel3D<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( zTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TaylorGreenVel3D<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ValT>& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
  z_ = &fm.field_ref( zTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TaylorGreenVel3D<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= (2/sqrt(3.0))*sin(angle_) * sin(*x_) * cos( *y_ ) * cos(*z_);
}

//--------------------------------------------------------------------

template< typename ValT >
TaylorGreenVel3D<ValT>::Builder::
Builder( const Expr::Tag xtag,
        const Expr::Tag ytag,
        const Expr::Tag ztag,
        const double angle)
: angle_(angle),
  xt_( xtag ),
  yt_( ytag ),
  zt_( ztag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
TaylorGreenVel3D<ValT>::Builder::
build( const Expr::ExpressionID& id,
      const Expr::ExpressionRegistry& reg ) const
{
  return new TaylorGreenVel3D<ValT>( xt_, yt_, zt_, angle_, id, reg );
}

//--------------------------------------------------------------------

//==========================================================================

// Explicit template instantiation for supported versions of this expression
#include <CCA/Components/Wasatch/FieldTypes.h>
using namespace Wasatch;

#define DECLARE_TAYLOR_MMS( VOL ) 	\
  template class VelocityX< VOL >;	\
  template class VelocityY< VOL >;	\
  template class TaylorGreenVel3D< VOL >;	\
  template class GradPX< VOL >;		\
  template class GradPY< VOL >;

DECLARE_TAYLOR_MMS( SVolField );
DECLARE_TAYLOR_MMS( XVolField );
DECLARE_TAYLOR_MMS( YVolField );
DECLARE_TAYLOR_MMS( ZVolField );
//==========================================================================
