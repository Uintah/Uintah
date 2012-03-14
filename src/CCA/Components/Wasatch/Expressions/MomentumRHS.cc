/*
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

#include "MomentumRHS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>


template< typename FieldT >
MomRHS<FieldT>::
MomRHS( const Expr::Tag& pressure,
        const Expr::Tag& partRHS )
  : Expr::Expression<FieldT>(),
    pressuret_( pressure ),
    rhsPartt_( partRHS ),
   emptyTag_( Expr::Tag() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHS<FieldT>::
~MomRHS()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( pressuret_ != emptyTag_ )    exprDeps.requires_expression( pressuret_ );;
  exprDeps.requires_expression( rhsPartt_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  rhsPart_ = &fm.field_ref( rhsPartt_ );

  const Expr::FieldManager<PFieldT>& pfm = fml.template field_manager<PFieldT>();
  if( pressuret_ != emptyTag_ )    pressure_ = &pfm.field_ref( pressuret_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<Grad>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result = 0.0;
  if ( pressuret_ != emptyTag_ ){
    gradOp_->apply_to_field( *pressure_, result );
    result <<= -result;
  }
  result += *rhsPart_;
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHS<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& pressure,
                  const Expr::Tag& partRHS )
  : ExpressionBuilder(result),
    pressuret_( pressure ),
    rhspt_( partRHS )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHS<FieldT>::Builder::build() const
{
  return new MomRHS<FieldT>( pressuret_, rhspt_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHS< SpatialOps::structured::XVolField >;
template class MomRHS< SpatialOps::structured::YVolField >;
template class MomRHS< SpatialOps::structured::ZVolField >;
//==================================================================
