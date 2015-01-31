/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#include <cmath>

#include "TaylorVortex.h"
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

template<typename ValT>
VelocityX<ValT>::
VelocityX( const Expr::Tag& xtag,
           const Expr::Tag& ytag,
           const Expr::Tag& ttag,
           const double A,
           const double nu )
  : Expr::Expression<ValT>(),
    a_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{
  this->set_gpu_runnable(true);
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
  const typename Expr::FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );

  t_ = &fml.template field_ref<TimeField>( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
VelocityX<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= 1.0 - a_ * cos( 2.0*PI * *x_ - *t_ ) * sin( 2.0*PI * *y_ - *t_ ) * exp( -2.0 * nu_ * *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
VelocityX<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ytag,
         const Expr::Tag& ttag,
         const double A,
         const double nu)
  : ExpressionBuilder(result),
    A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
VelocityX<ValT>::Builder::
build() const
{
  return new VelocityX<ValT>( xt_, yt_, tt_, A_, nu_ );
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
           const double nu )
  : Expr::Expression<ValT>(),
    a_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{
  this->set_gpu_runnable( true );
}

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
  const typename Expr::FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );

  t_ = &fml.template field_ref<TimeField>( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
VelocityY<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= 1.0 + a_ * sin( 2.0*PI * *x_ - *t_ ) * cos( 2.0*PI * *y_ - *t_ ) * exp( -2.0*nu_ * *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
VelocityY<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ytag,
         const Expr::Tag& ttag,
         const double A,
         const double nu )
  : ExpressionBuilder(result),
    A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
VelocityY<ValT>::Builder::
build() const
{
  return new VelocityY<ValT>( xt_, yt_, tt_, A_, nu_ );
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
        const double nu )
  : Expr::Expression<ValT>(),
    a_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{
  this->set_gpu_runnable( true );
}

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
  const typename Expr::FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );

  t_ = &fml.template field_ref<TimeField>( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
GradPX<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= -1.0*(a_*a_/2.0) * sin( 2.0*( 2.0*PI * *x_ - *t_ ) ) * exp( -4.0*nu_ * *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
GradPX<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ytag,
         const Expr::Tag& ttag,
         const double A,
         const double nu )
  : ExpressionBuilder(result),
    A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
GradPX<ValT>::Builder::
build() const
{
  return new GradPX<ValT>( xt_, yt_, tt_, A_, nu_ );
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
        const double nu )
  : Expr::Expression<ValT>(),
    a_(A), nu_(nu), xTag_( xtag ), yTag_( ytag ), tTag_( ttag )
{
  this->set_gpu_runnable( true );
}

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
  const typename Expr::FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );

  t_ = &fml.template field_ref<TimeField>( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
GradPY<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= -1.0*(a_*a_/2.0) * sin( 2.0*(2.0*PI * *y_-*t_) ) * exp(-4.0 * nu_ * *t_);
}

//--------------------------------------------------------------------

template< typename ValT >
GradPY<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ytag,
         const Expr::Tag& ttag,
         const double A,
         const double nu )
  : ExpressionBuilder(result),
    A_(A),
    nu_(nu),
    xt_( xtag ),
    yt_( ytag ),
    tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
GradPY<ValT>::Builder::
build() const
{
  return new GradPY<ValT>( xt_, yt_, tt_, A_, nu_ );
}

//--------------------------------------------------------------------

//==========================================================================

//--------------------------------------------------------------------

template<typename ValT>
TaylorGreenVel3D<ValT>::
TaylorGreenVel3D( const Expr::Tag& xtag,
                  const Expr::Tag& ytag,
                  const Expr::Tag& ztag,
                  const double angle )
: Expr::Expression<ValT>(),
  angle_(angle), xTag_( xtag ), yTag_( ytag ), zTag_( ztag )
{
  this->set_gpu_runnable( true );
}

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
  const typename Expr::FieldMgrSelector<ValT>::type& fm = fml.template field_manager<ValT>();
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
  phi <<= (2/sqrt(3.0))*sin(angle_) * sin(2.0*PI * *x_) * cos( 2.0*PI * *y_ ) * cos(2.0*PI * *z_);
}

//--------------------------------------------------------------------

template< typename ValT >
TaylorGreenVel3D<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ytag,
         const Expr::Tag& ztag,
         const double angle)
: ExpressionBuilder(result),
  angle_(angle),
  xt_( xtag ),
  yt_( ytag ),
  zt_( ztag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
TaylorGreenVel3D<ValT>::Builder::
build() const
{
  return new TaylorGreenVel3D<ValT>( xt_, yt_, zt_, angle_ );
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
