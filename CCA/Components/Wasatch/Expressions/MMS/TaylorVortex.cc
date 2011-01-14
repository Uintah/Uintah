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


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <CCA/Components/Wasatch/FieldTypes.h>
using namespace Wasatch;

template class VelocityX< SVolField >;
template class VelocityX< XVolField >;
template class VelocityX< YVolField >;
template class VelocityX< ZVolField >;

template class VelocityY< SVolField >;
template class VelocityY< XVolField >;
template class VelocityY< YVolField >;
template class VelocityY< ZVolField >;

template class GradPX< SVolField >;
template class GradPX< XVolField >;
template class GradPX< YVolField >;
template class GradPX< ZVolField >;

template class GradPY< SVolField >;
template class GradPY< XVolField >;
template class GradPY< YVolField >;
template class GradPY< ZVolField >;
//==========================================================================
