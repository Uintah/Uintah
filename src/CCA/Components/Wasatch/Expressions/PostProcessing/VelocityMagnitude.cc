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

#include <CCA/Components/Wasatch/Expressions/PostProcessing/VelocityMagnitude.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

// ###################################################################
//
//               Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
VelocityMagnitude( const Expr::Tag& vel1tag,
                   const Expr::Tag& vel2tag,
                   const Expr::Tag& vel3tag )
: Expr::Expression<FieldT>(),
  doX_( vel1tag != Expr::Tag() ),
  doY_( vel2tag != Expr::Tag() ),
  doZ_( vel3tag != Expr::Tag() ),
  is3d_( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
  if(doX_)  u_ = this->template create_field_request<Vel1T>(vel1tag);
  if(doY_)  v_ = this->template create_field_request<Vel2T>(vel2tag);
  if(doZ_)  w_ = this->template create_field_request<Vel3T>(vel3tag);
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
~VelocityMagnitude()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ )  interpVel1T2FieldTOp_ = opDB.retrieve_operator<InterpVel1T2FieldT>();
  if( doY_ )  interpVel2T2FieldTOp_ = opDB.retrieve_operator<InterpVel2T2FieldT>();
  if( doZ_ )  interpVel3T2FieldTOp_ = opDB.retrieve_operator<InterpVel3T2FieldT>();
}

//--------------------------------------------------------------------

template<>
void
VelocityMagnitude<SVolField,SVolField,SVolField,SVolField>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}


//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& velMag = this->value();

  if( is3d_ ){ // inline the 3D calculation for better performance:
    const Vel1T& u = u_->field_ref();
    const Vel2T& v = v_->field_ref();
    const Vel3T& w = w_->field_ref();
    velMag <<= sqrt(
        (*interpVel1T2FieldTOp_)(u) * (*interpVel1T2FieldTOp_)(u) +
        (*interpVel2T2FieldTOp_)(v) * (*interpVel2T2FieldTOp_)(v) +
        (*interpVel3T2FieldTOp_)(w) * (*interpVel3T2FieldTOp_)(w)
      );
  }
  else{ // 1D and 2D are assembled in pieces (slower):
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( velMag );
    if( doX_ ) velMag <<=          (*interpVel1T2FieldTOp_)(u_->field_ref()) * (*interpVel1T2FieldTOp_)(u_->field_ref());
    else                        velMag <<= 0.0;
    if( doY_ ) velMag <<= velMag + (*interpVel2T2FieldTOp_)(v_->field_ref()) * (*interpVel2T2FieldTOp_)(v_->field_ref());
    if( doZ_ ) velMag <<= velMag + (*interpVel3T2FieldTOp_)(w_->field_ref()) * (*interpVel3T2FieldTOp_)(w_->field_ref());
    velMag <<= sqrt(velMag);
  }
}

//--------------------------------------------------------------------

template<>
void
VelocityMagnitude<SVolField,SVolField,SVolField,SVolField>::
evaluate()
{
  using namespace SpatialOps;
  typedef SVolField FieldT;
  FieldT& velMag = this->value();
  
  if( is3d_ ){ // inline the 3D calculation for better performance:
    const FieldT& u = u_->field_ref();
    const FieldT& v = v_->field_ref();
    const FieldT& w = w_->field_ref();
    velMag <<= sqrt(u*u + v*v + w*w);
  }
  else{ // 1D and 2D are assembled in pieces (slower):
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( velMag );
    if( doX_ ) velMag <<= u_->field_ref() * u_->field_ref();
    else       velMag <<= 0.0;
    if( doY_ ) velMag <<= velMag + v_->field_ref() * v_->field_ref();;
    if( doZ_ ) velMag <<= velMag + w_->field_ref() * w_->field_ref();;
    velMag <<= sqrt(velMag);
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag,
                 const Expr::Tag& vel3tag )
: ExpressionBuilder(result),
  v1t_( vel1tag ), v2t_( vel2tag ), v3t_( vel3tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class VelocityMagnitude< SpatialOps::SVolField,
                                  SpatialOps::XVolField,
                                  SpatialOps::YVolField,
                                  SpatialOps::ZVolField >;

template class VelocityMagnitude< SpatialOps::SVolField,
                                  SpatialOps::SVolField,
                                  SpatialOps::SVolField,
                                  SpatialOps::SVolField >;

//==========================================================================
