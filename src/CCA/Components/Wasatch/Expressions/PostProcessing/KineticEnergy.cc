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

#include "KineticEnergy.h"

#include <spatialops/structured/FVStaggered.h>

// ###################################################################
//
//               KineticEnergy Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>::
KineticEnergy( const Expr::Tag& vel1Tag,
               const Expr::Tag& vel2Tag,
               const Expr::Tag& vel3Tag )
: VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>(vel1Tag, vel2Tag, vel3Tag)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>::
~KineticEnergy()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& kE = this->value();
    
  if( this->is3d_ ){ // inline the 3D calculation for better performance:
    const Vel1T& u = this->u_->field_ref();
    const Vel2T& v = this->v_->field_ref();
    const Vel3T& w = this->w_->field_ref();
    kE <<= 0.5 * (
        (*this->interpVel1T2FieldTOp_)(u) * (*this->interpVel1T2FieldTOp_)(u) +
        (*this->interpVel2T2FieldTOp_)(v) * (*this->interpVel2T2FieldTOp_)(v) +
        (*this->interpVel3T2FieldTOp_)(w) * (*this->interpVel3T2FieldTOp_)(w)
      );
  }
  else{ // 1D and 2D are assembled in pieces (slower):
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( kE );
    if( this->doX_ ) kE <<=      (*this->interpVel1T2FieldTOp_)(this->u_->field_ref()) * (*this->interpVel1T2FieldTOp_)(this->u_->field_ref());
    else                              kE <<= 0.0;
    if( this->doY_ ) kE <<= kE + (*this->interpVel2T2FieldTOp_)(this->v_->field_ref()) * (*this->interpVel2T2FieldTOp_)(this->v_->field_ref());
    if( this->doZ_ ) kE <<= kE + (*this->interpVel3T2FieldTOp_)(this->w_->field_ref()) * (*this->interpVel3T2FieldTOp_)(this->w_->field_ref());
    kE <<= 0.5 * kE;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>::
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
KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new KineticEnergy<FieldT,Vel1T,Vel2T,Vel3T>( v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------

// ###################################################################
//
//              TotalKineticEnergy Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
TotalKineticEnergy( const Expr::Tag& resultTag,
                    const Expr::Tag& vel1tag,
                    const Expr::Tag& vel2tag,
                    const Expr::Tag& vel3tag )
: Expr::Expression<SpatialOps::SingleValueField>(),
  doX_( vel1tag != Expr::Tag() ),
  doY_( vel2tag != Expr::Tag() ),
  doZ_( vel3tag != Expr::Tag() ),
  is3d_( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
  if(doX_) this->template create_field_request(vel1tag, u_);
  if(doY_) this->template create_field_request(vel2tag, v_);
  if(doZ_) this->template create_field_request(vel3tag, w_);
}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
~TotalKineticEnergy()
{}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
void
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  SpatialOps::SingleValueField& tKE = this->value();
  
  if (is3d_) {
    const Vel1T& u = this->u_->field_ref();
    const Vel2T& v = this->v_->field_ref();
    const Vel3T& w = this->w_->field_ref();

    tKE <<= 0.5 * ( field_sum_interior(u * u)
                  + field_sum_interior(v * v)
                  + field_sum_interior(w * w) );
  } else {
    tKE <<= 0.0;
    if( doX_ ) tKE <<= tKE + field_sum_interior(u_->field_ref() * u_->field_ref());
    if( doY_ ) tKE <<= tKE + field_sum_interior(v_->field_ref() * v_->field_ref());
    if( doZ_ ) tKE <<= tKE + field_sum_interior(w_->field_ref() * w_->field_ref());
    tKE <<= 0.5*tKE;
  }
}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& vel1Tag,
                  const Expr::Tag& vel2Tag,
                  const Expr::Tag& vel3Tag )
: ExpressionBuilder(result),
  resultTag_(result),
  v1t_( vel1Tag ),
  v2t_( vel2Tag ),
  v3t_( vel3Tag )
{}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new TotalKineticEnergy<Vel1T,Vel2T,Vel3T>( resultTag_, v1t_, v2t_, v3t_ );
}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

template class KineticEnergy< SpatialOps::SVolField,
                              SpatialOps::XVolField,
                              SpatialOps::YVolField,
                              SpatialOps::ZVolField >;

template class TotalKineticEnergy< SpatialOps::XVolField,
                                   SpatialOps::YVolField,
                                   SpatialOps::ZVolField >;
//==========================================================================
