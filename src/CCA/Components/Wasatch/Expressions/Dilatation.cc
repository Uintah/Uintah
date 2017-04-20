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

#include <CCA/Components/Wasatch/Expressions/Dilatation.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
Dilatation( const Expr::TagList& velTags )
  : Expr::Expression<FieldT>(),
    doX_ (velTags[0] != Expr::Tag()),
    doY_ (velTags[1] != Expr::Tag()),
    doZ_ (velTags[2] != Expr::Tag()),
    is3d_( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
  
  if (doX_)  vel1_ = this->template create_field_request<Vel1T>(velTags[0]);
  if (doY_)  vel2_ = this->template create_field_request<Vel2T>(velTags[1]);
  if (doZ_)  vel3_ = this->template create_field_request<Vel3T>(velTags[2]);
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
~Dilatation()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if (doX_) vel1GradOp_ = opDB.retrieve_operator<Vel1GradT>();
  if (doY_) vel2GradOp_ = opDB.retrieve_operator<Vel2GradT>();
  if (doZ_) vel3GradOp_ = opDB.retrieve_operator<Vel3GradT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& dil = this->value();

  dil <<= 0.0; // avoid potential garbage in extra/ghost cells

  if( is3d_ ){ // fully inline for 3D
    const Vel1T& u = vel1_->field_ref();
    const Vel2T& v = vel2_->field_ref();
    const Vel3T& w = vel3_->field_ref();
    dil <<= (*vel1GradOp_)(u) + (*vel2GradOp_)(v) + (*vel3GradOp_)(w);
  }
  else{ // for 2D and 1D, assemble in pieces
    if( doX_ ) dil <<=       (*vel1GradOp_)(vel1_->field_ref());
    if( doY_ ) dil <<= dil + (*vel2GradOp_)(vel2_->field_ref());
    if( doZ_ ) dil <<= dil + (*vel3GradOp_)(vel3_->field_ref());
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::TagList& velTags )
  : ExpressionBuilder(result),
    velTags_(velTags)
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
Dilatation<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new Dilatation<FieldT,Vel1T,Vel2T,Vel3T>( velTags_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class Dilatation< SpatialOps::SVolField,
                           SpatialOps::XVolField,
                           SpatialOps::YVolField,
                           SpatialOps::ZVolField >;

template class Dilatation< SpatialOps::SVolField,
                           SpatialOps::SVolField,
                           SpatialOps::SVolField,
                           SpatialOps::SVolField >;

//==========================================================================
