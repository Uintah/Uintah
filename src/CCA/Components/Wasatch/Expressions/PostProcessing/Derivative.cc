/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/PostProcessing/Derivative.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

// ###################################################################
//
//               Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
Derivative<SrcT, DestT, DirT>::
Derivative( const Expr::Tag& srctag )
: Expr::Expression<DestT>()
{
  this->set_gpu_runnable( true );
   src_ = this->template create_field_request<SrcT>(srctag);
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
Derivative<SrcT, DestT, DirT>::
~Derivative()
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
void
Derivative<SrcT, DestT, DirT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  derivativeOp_ = opDB.retrieve_operator<DerivativeT>();
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
void
Derivative<SrcT, DestT, DirT>::
evaluate()
{
  using SpatialOps::operator<<=;
  DestT& destResult = this->value();
  const SrcT& src = src_->field_ref();
  destResult <<= (*derivativeOp_)( src );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
Derivative<SrcT, DestT, DirT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& srctag )
: ExpressionBuilder(result),
  srct_( srctag )
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT, typename DirT >
Expr::ExpressionBase*
Derivative<SrcT, DestT, DirT>::Builder::build() const
{
  return new Derivative<SrcT, DestT, DirT>( srct_ );
}

//--------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

// for staggered sources, the destination is always SVOLFIELD and the direction doesn't matter
#define INSTANTIATE_VARIANTS(VOLT)\
template class Derivative< VOLT, SpatialOps::SVolField, SpatialOps::XDIR >;

INSTANTIATE_VARIANTS(SpatialOps::XVolField);
INSTANTIATE_VARIANTS(SpatialOps::YVolField);
INSTANTIATE_VARIANTS(SpatialOps::ZVolField);

// for SVOL sources, the we currently support SVOL destinations. In that case, the direction must be specified 
template class Derivative< SpatialOps::SVolField, SpatialOps::SVolField, SpatialOps::XDIR >;
template class Derivative< SpatialOps::SVolField, SpatialOps::SVolField, SpatialOps::YDIR >;
template class Derivative< SpatialOps::SVolField, SpatialOps::SVolField, SpatialOps::ZDIR >;
//==========================================================================
