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

#include "Vorticity.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
Vorticity( const Expr::Tag& vel1tag,
           const Expr::Tag& vel2tag )
: Expr::Expression<FieldT>(),
  vel1t_( vel1tag ),
  vel2t_( vel2tag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
~Vorticity()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  if( vel1t_ != Expr::Tag() )  vel1_ = &fml.field_ref<Vel1T>( vel1t_ );
  if( vel2t_ != Expr::Tag() )  vel2_ = &fml.field_ref<Vel2T>( vel2t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() ){
    vel1GradTOp_              = opDB.retrieve_operator<Vel1GradT>();
    interpVel1FaceT2FieldTOp_ = opDB.retrieve_operator<InterpVel1FaceT2FieldT>();
  }
  if( vel2t_ != Expr::Tag() ){
    vel2GradTOp_              = opDB.retrieve_operator<Vel2GradT>();
    interpVel2FaceT2FieldTOp_ = opDB.retrieve_operator<InterpVel2FaceT2FieldT>();
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& vorticity = this->value();

  if( vel1t_ != Expr::Tag() && vel2t_ != Expr::Tag() ){
    // fully inlined:
    vorticity <<= (*interpVel1FaceT2FieldTOp_)( (*vel1GradTOp_)(*vel1_) )
                - (*interpVel2FaceT2FieldTOp_)( (*vel2GradTOp_)(*vel2_) );
  }
  else{
    if( vel1t_ != Expr::Tag() ) vorticity <<=  (*interpVel1FaceT2FieldTOp_)( (*vel1GradTOp_)(*vel1_) );
    if( vel2t_ != Expr::Tag() ) vorticity <<= -(*interpVel2FaceT2FieldTOp_)( (*vel2GradTOp_)(*vel2_) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag )
: ExpressionBuilder(result),
  v1t_( vel1tag ), v2t_( vel2tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Expr::ExpressionBase*
Vorticity<FieldT,Vel1T,Vel2T>::Builder::build() const
{
  return new Vorticity<FieldT,Vel1T,Vel2T>( v1t_, v2t_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class Vorticity< SpatialOps::SVolField,
                          SpatialOps::YVolField,
                          SpatialOps::XVolField >;

template class Vorticity< SpatialOps::SVolField,
                          SpatialOps::ZVolField,
                          SpatialOps::YVolField >;

template class Vorticity< SpatialOps::SVolField,
                          SpatialOps::XVolField,
                          SpatialOps::ZVolField >;

//==========================================================================
