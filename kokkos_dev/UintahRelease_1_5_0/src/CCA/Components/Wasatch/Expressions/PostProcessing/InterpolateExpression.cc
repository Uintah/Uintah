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

template< typename SrcT, typename DestT >
InterpolateExpression<SrcT, DestT>::
InterpolateExpression( const Expr::Tag& srctag )
: Expr::Expression<DestT>(),
srct_( srctag )
{}

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
  if( srct_ != Expr::Tag() )  src_ = &fml.template field_manager<SrcT>().field_ref( srct_ );
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( srct_ != Expr::Tag() )
    InpterpSrcT2DestTOp_ = opDB.retrieve_operator<InpterpSrcT2DestT>();
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
evaluate()
{
  using SpatialOps::operator<<=;
  DestT& destResult = this->value();
  InpterpSrcT2DestTOp_->apply_to_field(*src_, destResult);
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
template class InterpolateExpression< SpatialOps::structured::SVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::SVolField,
                                      SpatialOps::structured::XVolField >;

template class InterpolateExpression< SpatialOps::structured::SVolField,
                                      SpatialOps::structured::YVolField >;

template class InterpolateExpression< SpatialOps::structured::SVolField,
                                      SpatialOps::structured::ZVolField >;

template class InterpolateExpression< SpatialOps::structured::XVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::XVolField,
                                      SpatialOps::structured::XVolField >; 

template class InterpolateExpression< SpatialOps::structured::XVolField,
                                      SpatialOps::structured::YVolField >;

template class InterpolateExpression< SpatialOps::structured::XVolField,
                                      SpatialOps::structured::ZVolField >;

template class InterpolateExpression< SpatialOps::structured::YVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::YVolField,
                                      SpatialOps::structured::XVolField >;

template class InterpolateExpression< SpatialOps::structured::YVolField,
                                      SpatialOps::structured::YVolField >;

template class InterpolateExpression< SpatialOps::structured::YVolField,
                                      SpatialOps::structured::ZVolField >;

template class InterpolateExpression< SpatialOps::structured::ZVolField,
                                      SpatialOps::structured::SVolField >;

template class InterpolateExpression< SpatialOps::structured::ZVolField,
                                      SpatialOps::structured::XVolField >;

template class InterpolateExpression< SpatialOps::structured::ZVolField,
                                      SpatialOps::structured::YVolField >;

template class InterpolateExpression< SpatialOps::structured::ZVolField,
                                      SpatialOps::structured::ZVolField >;
//==========================================================================
