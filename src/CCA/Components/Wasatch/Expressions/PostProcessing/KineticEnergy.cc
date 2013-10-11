/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

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
    kE <<= 0.5 * (
        (*this->interpVel1T2FieldTOp_)(*this->vel1_) * (*this->interpVel1T2FieldTOp_)(*this->vel1_) +
        (*this->interpVel2T2FieldTOp_)(*this->vel2_) * (*this->interpVel2T2FieldTOp_)(*this->vel2_) +
        (*this->interpVel3T2FieldTOp_)(*this->vel3_) * (*this->interpVel3T2FieldTOp_)(*this->vel3_)
      );
  }
  else{ // 1D and 2D are assembled in pieces (slower):
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( kE );
    if( this->vel1t_ != Expr::Tag() ) kE <<=      (*this->interpVel1T2FieldTOp_)(*this->vel1_) * (*this->interpVel1T2FieldTOp_)(*this->vel1_);
    else                              kE <<= 0.0;
    if( this->vel2t_ != Expr::Tag() ) kE <<= kE + (*this->interpVel2T2FieldTOp_)(*this->vel2_) * (*this->interpVel2T2FieldTOp_)(*this->vel2_);
    if( this->vel3t_ != Expr::Tag() ) kE <<= kE + (*this->interpVel3T2FieldTOp_)(*this->vel3_) * (*this->interpVel3T2FieldTOp_)(*this->vel3_);
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
: Expr::Expression<SpatialOps::structured::SingleValueField>(),
  vel1t_( vel1tag ),
  vel2t_( vel2tag ),
  vel3t_( vel3tag ),
  is3d_( vel1t_ != Expr::Tag() && vel2t_ != Expr::Tag() && vel3t_ != Expr::Tag() )
{
  this->set_gpu_runnable( true );
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
  if( vel3t_ != Expr::Tag() )  exprDeps.requires_expression( vel3t_ );
}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
void
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<Vel1T>::type& v1fm = fml.template field_manager<Vel1T>();
  const typename Expr::FieldMgrSelector<Vel2T>::type& v2fm = fml.template field_manager<Vel2T>();
  const typename Expr::FieldMgrSelector<Vel3T>::type& v3fm = fml.template field_manager<Vel3T>();
  
  if( vel1t_ != Expr::Tag() )  vel1_ = &v1fm.field_ref( vel1t_ );
  if( vel2t_ != Expr::Tag() )  vel2_ = &v2fm.field_ref( vel2t_ );
  if( vel3t_ != Expr::Tag() )  vel3_ = &v3fm.field_ref( vel3t_ );
}

//--------------------------------------------------------------------

template< typename Vel1T, typename Vel2T, typename Vel3T >
void
TotalKineticEnergy<Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  SpatialOps::structured::SingleValueField& tKE = this->value();
  
  if (is3d_) {
    tKE <<= 0.5 * ( field_sum_interior(*vel1_ * *vel1_)
                  + field_sum_interior(*vel2_ * *vel2_)
                  + field_sum_interior(*vel3_ * *vel3_) );
  } else {
    tKE <<= 0.0;
    if( vel1t_ != Expr::Tag() ) tKE <<= tKE + field_sum_interior(*vel1_ * *vel1_);
    if( vel2t_ != Expr::Tag() ) tKE <<= tKE + field_sum_interior(*vel2_ * *vel2_);
    if( vel3t_ != Expr::Tag() ) tKE <<= tKE + field_sum_interior(*vel3_ * *vel3_);
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

template class KineticEnergy< SpatialOps::structured::SVolField,
                              SpatialOps::structured::XVolField,
                              SpatialOps::structured::YVolField,
                              SpatialOps::structured::ZVolField >;

template class TotalKineticEnergy< SpatialOps::structured::XVolField,
                                   SpatialOps::structured::YVolField,
                                   SpatialOps::structured::ZVolField >;
//==========================================================================
