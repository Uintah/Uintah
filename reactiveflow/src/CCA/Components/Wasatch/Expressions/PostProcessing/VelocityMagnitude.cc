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

#include "VelocityMagnitude.h"

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
  vel1t_( vel1tag ),
  vel2t_( vel2tag ),
  vel3t_( vel3tag ),
  is3d_( vel1t_ != Expr::Tag() && vel2t_ != Expr::Tag() && vel3t_ != Expr::Tag() )
{
  vel1_=NULL;  vel2_=NULL;  vel3_=NULL;
  this->set_gpu_runnable( true );
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
  if( vel3t_ != Expr::Tag() )  exprDeps.requires_expression( vel3t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
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

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() )  interpVel1T2FieldTOp_ = opDB.retrieve_operator<InterpVel1T2FieldT>();
  if( vel2t_ != Expr::Tag() )  interpVel2T2FieldTOp_ = opDB.retrieve_operator<InterpVel2T2FieldT>();
  if( vel3t_ != Expr::Tag() )  interpVel3T2FieldTOp_ = opDB.retrieve_operator<InterpVel3T2FieldT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
VelocityMagnitude<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& velMag = this->value();

  if( is3d_ ){ // inline the 3D calculation for better performance:
    velMag <<= sqrt(
        (*interpVel1T2FieldTOp_)(*vel1_) * (*interpVel1T2FieldTOp_)(*vel1_) +
        (*interpVel2T2FieldTOp_)(*vel2_) * (*interpVel2T2FieldTOp_)(*vel2_) +
        (*interpVel3T2FieldTOp_)(*vel3_) * (*interpVel3T2FieldTOp_)(*vel3_)
      );
  }
  else{ // 1D and 2D are assembled in pieces (slower):
    SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( velMag );
    if( vel1t_ != Expr::Tag() ) velMag <<=          (*interpVel1T2FieldTOp_)(*vel1_) * (*interpVel1T2FieldTOp_)(*vel1_);
    else                        velMag <<= 0.0;
    if( vel2t_ != Expr::Tag() ) velMag <<= velMag + (*interpVel2T2FieldTOp_)(*vel2_) * (*interpVel2T2FieldTOp_)(*vel2_);
    if( vel3t_ != Expr::Tag() ) velMag <<= velMag + (*interpVel3T2FieldTOp_)(*vel3_) * (*interpVel3T2FieldTOp_)(*vel3_);
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
//==========================================================================
