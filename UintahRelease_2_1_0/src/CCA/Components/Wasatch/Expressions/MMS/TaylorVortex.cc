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

#include <cmath>

#include <CCA/Components/Wasatch/Expressions/MMS/TaylorVortex.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

template<typename FieldT>
VelocityX<FieldT>::
VelocityX( const Expr::Tag& xtag,
           const Expr::Tag& ytag,
           const Expr::Tag& ttag,
           const double A,
           const double nu )
  : Expr::Expression<FieldT>(),
    a_(A), nu_(nu)
{
  this->set_gpu_runnable(true);
  
   x_ = this->template create_field_request<FieldT>(xtag);
   y_ = this->template create_field_request<FieldT>(ytag);
   t_ = this->template create_field_request<TimeField>(ttag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VelocityX<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const TimeField& t = t_->field_ref();
  phi <<= 1.0 - a_ * cos( 2.0*PI * x - t ) * sin( 2.0*PI * y - t ) * exp( -2.0 * nu_ * t );
}

//--------------------------------------------------------------------

template< typename FieldT >
VelocityX<FieldT>::Builder::
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

template< typename FieldT >
Expr::ExpressionBase*
VelocityX<FieldT>::Builder::
build() const
{
  return new VelocityX<FieldT>( xt_, yt_, tt_, A_, nu_ );
}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------

template<typename FieldT>
VelocityY<FieldT>::
VelocityY( const Expr::Tag& xtag,
           const Expr::Tag& ytag,
           const Expr::Tag& ttag,
           const double A,
           const double nu )
  : Expr::Expression<FieldT>(),
    a_(A), nu_(nu)
{
  this->set_gpu_runnable( true );
  
   x_ = this->template create_field_request<FieldT>(xtag);
   y_ = this->template create_field_request<FieldT>(ytag);
   t_ = this->template create_field_request<TimeField>(ttag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VelocityY<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const TimeField& t = t_->field_ref();
  phi <<= 1.0 + a_ * sin( 2.0*PI * x - t ) * cos( 2.0*PI * y - t ) * exp( -2.0*nu_ * t );
}

//--------------------------------------------------------------------

template< typename FieldT >
VelocityY<FieldT>::Builder::
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

template< typename FieldT >
Expr::ExpressionBase*
VelocityY<FieldT>::Builder::
build() const
{
  return new VelocityY<FieldT>( xt_, yt_, tt_, A_, nu_ );
}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------

template<typename FieldT>
GradP<FieldT>::
GradP( const Expr::Tag& xtag,
        const Expr::Tag& ttag,
        const double A,
        const double nu )
  : Expr::Expression<FieldT>(),
    a_(A), nu_(nu)
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<FieldT>(xtag);
  t_ = this->template create_field_request<TimeField>(ttag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
GradP<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const FieldT& x = x_->field_ref();
  const TimeField& t = t_->field_ref();
  phi <<= -(a_*a_/2.0) * sin( 2.0*( 2.0*PI * x - t ) ) * exp( -4.0*nu_ * t );
}

//--------------------------------------------------------------------

template< typename FieldT >
GradP<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xtag,
         const Expr::Tag& ttag,
         const double A,
         const double nu )
  : ExpressionBuilder(result),
    A_(A),
    nu_(nu),
    xt_( xtag ),
    tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
GradP<FieldT>::Builder::
build() const
{
  return new GradP<FieldT>( xt_, tt_, A_, nu_ );
}

//--------------------------------------------------------------------

//==========================================================================

//--------------------------------------------------------------------

template<typename FieldT>
TaylorGreenVel3D<FieldT>::
TaylorGreenVel3D( const Expr::Tag& xtag,
                  const Expr::Tag& ytag,
                  const Expr::Tag& ztag,
                  const double angle )
: Expr::Expression<FieldT>(),
  angle_(angle)
{
  this->set_gpu_runnable( true );
   x_ = this->template create_field_request<FieldT>(xtag);
   y_ = this->template create_field_request<FieldT>(ytag);
   z_ = this->template create_field_request<FieldT>(ztag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TaylorGreenVel3D<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const FieldT& z = z_->field_ref();
  phi <<= (2/sqrt(3.0))*sin(angle_) * sin(2.0*PI * x) * cos( 2.0*PI * y ) * cos(2.0*PI * z);
}

//--------------------------------------------------------------------

template< typename FieldT >
TaylorGreenVel3D<FieldT>::Builder::
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

template< typename FieldT >
Expr::ExpressionBase*
TaylorGreenVel3D<FieldT>::Builder::
build() const
{
  return new TaylorGreenVel3D<FieldT>( xt_, yt_, zt_, angle_ );
}

//--------------------------------------------------------------------

//==========================================================================

// Explicit template instantiation for supported versions of this expression
#include <CCA/Components/Wasatch/FieldTypes.h>
using namespace WasatchCore;

#define DECLARE_TAYLOR_MMS( VOL ) 	\
  template class VelocityX< VOL >;	\
  template class VelocityY< VOL >;	\
  template class TaylorGreenVel3D< VOL >;	\
  template class GradP< VOL >;

DECLARE_TAYLOR_MMS( SVolField );
DECLARE_TAYLOR_MMS( XVolField );
DECLARE_TAYLOR_MMS( YVolField );
DECLARE_TAYLOR_MMS( ZVolField );
//==========================================================================
