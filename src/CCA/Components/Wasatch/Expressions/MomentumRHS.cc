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

#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>


template< typename FieldT, typename DirT>
MomRHS<FieldT, DirT>::
MomRHS( const Expr::Tag& pressureTag,
        const Expr::Tag& partRHSTag,
        const Expr::Tag& volFracTag )
  : Expr::Expression<FieldT>(),
    hasP_(pressureTag != Expr::Tag()),
    hasIntrusion_(volFracTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
  
  if( hasP_ )  pressure_ = this->template create_field_request<PFieldT>(pressureTag);
   rhsPart_ = this->template create_field_request<FieldT>(partRHSTag);
  if( hasIntrusion_ )  volfrac_ = this->template create_field_request<FieldT>(volFracTag);
}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
MomRHS<FieldT, DirT>::
~MomRHS()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
void
MomRHS<FieldT, DirT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<Grad>();
}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
void
MomRHS<FieldT, DirT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT&  rhsPart = rhsPart_->field_ref();

  if( hasP_ ){
    const PFieldT& p = pressure_->field_ref();
    if( hasIntrusion_ )  result <<= volfrac_->field_ref() * ( rhsPart - (*gradOp_)(p) );
    else                 result <<= rhsPart - (*gradOp_)(p);
  }
  else{
    if( hasIntrusion_ ) result <<= volfrac_->field_ref() * rhsPart;
    else                result <<= rhsPart;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename DirT>
MomRHS<FieldT, DirT>::
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

template< typename FieldT, typename DirT>
Expr::ExpressionBase*
MomRHS<FieldT, DirT>::Builder::build() const
{
  return new MomRHS<FieldT, DirT>( pressuret_, rhspartt_, volfract_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHS< SpatialOps::SVolField, SpatialOps::XDIR >;
template class MomRHS< SpatialOps::SVolField, SpatialOps::YDIR >;
template class MomRHS< SpatialOps::SVolField, SpatialOps::ZDIR >;
template class MomRHS< SpatialOps::XVolField, SpatialOps::NODIR >;
template class MomRHS< SpatialOps::YVolField, SpatialOps::NODIR >;
template class MomRHS< SpatialOps::ZVolField, SpatialOps::NODIR >;
//==================================================================
