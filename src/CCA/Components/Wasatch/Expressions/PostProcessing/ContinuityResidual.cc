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

#include <CCA/Components/Wasatch/Expressions/PostProcessing/ContinuityResidual.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
ContinuityResidual( const Expr::Tag&     drhodtTag,
                    const Expr::TagList& velTags    )
  : Expr::Expression<FieldT>(),
    constDen_ ( drhodtTag == Expr::Tag() ),
    doX_      ( velTags[0] != Expr::Tag() ),
    doY_      ( velTags[1] != Expr::Tag() ),
    doZ_      ( velTags[2] != Expr::Tag() ),
    is3d_     ( doX_ && doY_ && doZ_ )
{
  this->set_gpu_runnable( true );
  if (!constDen_)  drhodt_ = this->template create_field_request<FieldT>(drhodtTag);
  if (doX_)  u1_ = this->template create_field_request<Vel1T>(velTags[0]);
  if (doY_)  u2_ = this->template create_field_request<Vel2T>(velTags[1]);
  if (doZ_)  u3_ = this->template create_field_request<Vel3T>(velTags[2]);
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
~ContinuityResidual()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ )  vel1GradOp_ = opDB.retrieve_operator<Vel1GradT>();
  if( doY_ )  vel2GradOp_ = opDB.retrieve_operator<Vel2GradT>();
  if( doZ_ )  vel3GradOp_ = opDB.retrieve_operator<Vel3GradT>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& cont = this->value();

  cont <<= 0.0; // avoid potential garbage in extra/ghost cells

  if (!constDen_) cont <<= drhodt_->field_ref();
  
  if( is3d_ ){ // fully inline for 3D
    const Vel1T& u1 = u1_->field_ref();
    const Vel2T& u2 = u2_->field_ref();
    const Vel3T& u3 = u3_->field_ref();
    cont <<= cont + (*vel1GradOp_)(u1) + (*vel2GradOp_)(u2) + (*vel3GradOp_)(u3);
  }
  else{ // for 2D and 1D, assemble in pieces
    if( doX_ ) cont <<= cont + (*vel1GradOp_)(u1_->field_ref());
    if( doY_ ) cont <<= cont + (*vel2GradOp_)(u2_->field_ref());
    if( doZ_ ) cont <<= cont + (*vel3GradOp_)(u3_->field_ref());
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag&     drhodtTag,
                  const Expr::TagList& velTags )
  : ExpressionBuilder(result),
    drhodtTag_(drhodtTag),
    velTags_(velTags)
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::Builder::build() const
{
  return new ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>( drhodtTag_, velTags_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class ContinuityResidual< SpatialOps::SVolField,
                                   SpatialOps::XVolField,
                                   SpatialOps::YVolField,
                                   SpatialOps::ZVolField >;
//==========================================================================
