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

#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>


template< typename FieldT >
MomRHS<FieldT>::
MomRHS( const Expr::Tag& pressure,
        const Expr::Tag& partRHS,
        const Expr::Tag& volFracTag )
  : Expr::Expression<FieldT>(),
    pressuret_( pressure ),
    rhspartt_( partRHS ),
    volfract_( volFracTag ),
    emptyTag_( Expr::Tag() )
{
  this->set_gpu_runnable( true );
}

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
  exprDeps.requires_expression( rhspartt_ );
  if( volfract_ != emptyTag_ )    exprDeps.requires_expression( volfract_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHS<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  rhsPart_ = &fm.field_ref( rhspartt_ );
  if( volfract_  != emptyTag_ )  volfrac_  = &fm.field_ref( volfract_ );
  if( pressuret_ != emptyTag_ )  pressure_ = &fml.field_ref<PFieldT>( pressuret_ );
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

  if ( pressuret_ != emptyTag_ ){
    if( volfract_ != emptyTag_ )  result <<= *volfrac_ * ( *rhsPart_ - (*gradOp_)(*pressure_) );
    else                          result <<= *rhsPart_ - (*gradOp_)(*pressure_);
  }
  else{
    if( volfract_ != emptyTag_ ) result <<= *volfrac_ * *rhsPart_;
    else                         result <<= *rhsPart_;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHS<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& pressure,
                  const Expr::Tag& partRHS,
                  const Expr::Tag& volFracTag )
  : ExpressionBuilder(result),
    pressuret_( pressure ),
    rhspartt_( partRHS ),
    volfract_( volFracTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHS<FieldT>::Builder::build() const
{
  return new MomRHS<FieldT>( pressuret_, rhspartt_, volfract_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHS< SpatialOps::XVolField >;
template class MomRHS< SpatialOps::YVolField >;
template class MomRHS< SpatialOps::ZVolField >;
//==================================================================
