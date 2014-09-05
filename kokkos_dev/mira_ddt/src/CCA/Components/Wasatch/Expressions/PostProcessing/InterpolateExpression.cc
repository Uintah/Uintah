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

#include "InterpolateExpression.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

// ###################################################################
//
//               Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
InterpolateExpression( const Expr::Tag& srctag )
: Expr::Expression<DestT>(),
  srct_( srctag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
~InterpolateExpression()
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( srct_ != Expr::Tag() )   exprDeps.requires_expression( srct_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  if( srct_ != Expr::Tag() )  src_ = &fml.template field_ref<SrcT>( srct_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( srct_ != Expr::Tag() ) interpSrcT2DestTOp_ = opDB.retrieve_operator<InterpSrcT2DestT>();
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
evaluate()
{
  using SpatialOps::operator<<=;
  DestT& destResult = this->value();
  destResult <<= (*interpSrcT2DestTOp_)( *src_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& srctag )
: ExpressionBuilder(result),
  srct_( srctag )
{}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
Expr::ExpressionBase*
InterpolateExpression<SrcT, DestT>::Builder::build() const
{
  return new InterpolateExpression<SrcT, DestT>( srct_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class InterpolateExpression< SpatialOps::SVolField,
                                      SpatialOps::SVolField >;

template class InterpolateExpression< SpatialOps::SVolField,
                                      SpatialOps::XVolField >;

template class InterpolateExpression< SpatialOps::SVolField,
                                      SpatialOps::YVolField >;

template class InterpolateExpression< SpatialOps::SVolField,
                                      SpatialOps::ZVolField >;

template class InterpolateExpression< SpatialOps::XVolField,
                                      SpatialOps::SVolField >;

template class InterpolateExpression< SpatialOps::XVolField,
                                      SpatialOps::XVolField >; 

template class InterpolateExpression< SpatialOps::XVolField,
                                      SpatialOps::YVolField >;

template class InterpolateExpression< SpatialOps::XVolField,
                                      SpatialOps::ZVolField >;

template class InterpolateExpression< SpatialOps::YVolField,
                                      SpatialOps::SVolField >;

template class InterpolateExpression< SpatialOps::YVolField,
                                      SpatialOps::XVolField >;

template class InterpolateExpression< SpatialOps::YVolField,
                                      SpatialOps::YVolField >;

template class InterpolateExpression< SpatialOps::YVolField,
                                      SpatialOps::ZVolField >;

template class InterpolateExpression< SpatialOps::ZVolField,
                                      SpatialOps::SVolField >;

template class InterpolateExpression< SpatialOps::ZVolField,
                                      SpatialOps::XVolField >;

template class InterpolateExpression< SpatialOps::ZVolField,
                                      SpatialOps::YVolField >;

template class InterpolateExpression< SpatialOps::ZVolField,
                                      SpatialOps::ZVolField >;
//==========================================================================
