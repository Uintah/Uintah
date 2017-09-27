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

#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>
#include <spatialops/particles/ParticleOperators.h>
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
: Expr::Expression<DestT>()
{
  this->set_gpu_runnable( true );
   src_ = this->template create_field_request<SrcT>(srctag);
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
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpSrcT2DestTOp_ = opDB.retrieve_operator<InterpSrcT2DestT>();
}

//--------------------------------------------------------------------

template< typename SrcT, typename DestT >
void
InterpolateExpression<SrcT, DestT>::
evaluate()
{
  using SpatialOps::operator<<=;
  DestT& destResult = this->value();
  const SrcT& src = src_->field_ref();
  destResult <<= (*interpSrcT2DestTOp_)( src );
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
#define INSTANTIATE_VARIANTS(VOLT)\
template class InterpolateExpression< VOLT, \
                                      SpatialOps::SVolField >;\

INSTANTIATE_VARIANTS(SpatialOps::XVolField);
INSTANTIATE_VARIANTS(SpatialOps::YVolField);
INSTANTIATE_VARIANTS(SpatialOps::ZVolField);

template class InterpolateExpression< SpatialOps::SVolField, SpatialOps::XVolField >;
template class InterpolateExpression< SpatialOps::SVolField, SpatialOps::YVolField >;
template class InterpolateExpression< SpatialOps::SVolField, SpatialOps::ZVolField >;
//==========================================================================

// ###################################################################
//
//               Implementation
//
// ###################################################################

//--------------------------------------------------------------------

template< typename DestT >
InterpolateParticleExpression<DestT>::
InterpolateParticleExpression( const Expr::Tag& srctag,
                              const Expr::Tag& particleSizeTag,
                              const Expr::TagList& particlePositionTags)
: Expr::Expression<DestT>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants
  src_   = this->template create_field_request<ParticleField>(srctag                 );
  psize_ = this->template create_field_request<ParticleField>(particleSizeTag        );
  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);
}

//--------------------------------------------------------------------

template< typename DestT >
InterpolateParticleExpression<DestT>::
~InterpolateParticleExpression()
{}

//--------------------------------------------------------------------

template< typename DestT >
void
InterpolateParticleExpression<DestT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  p2CellOp_ = opDB.retrieve_operator<P2CellOpT>();
  pPerCellOp_ = opDB.retrieve_operator<PPerCellOpT>();
}

//--------------------------------------------------------------------

template< typename DestT >
void
InterpolateParticleExpression<DestT>::
evaluate()
{
  using namespace SpatialOps;
  DestT& result = this->value();
  
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();
  const ParticleField& psize = psize_->field_ref();
  const ParticleField& src = src_->field_ref();
  
  SpatFldPtr<DestT> nParticlesPerCell = SpatialFieldStore::get<DestT>( result );
  *nParticlesPerCell <<= 0.0;
  pPerCellOp_->set_coordinate_information( &px,&py,&pz,&psize );
  pPerCellOp_->apply_to_field( *nParticlesPerCell );
  
  p2CellOp_->set_coordinate_information( &px,&py,&pz,&psize );
  p2CellOp_->apply_to_field( src, result );
  result <<= cond( *nParticlesPerCell > 0.0, result/ *nParticlesPerCell )
                 ( result );
}

//--------------------------------------------------------------------

template<typename DestT >
InterpolateParticleExpression<DestT>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& srctag,
                 const Expr::Tag& particleSizeTag,
                 const Expr::TagList& particlePositionTags)
: ExpressionBuilder(result),
srct_( srctag ),
pSizeTag_(particleSizeTag),
pPosTags_(particlePositionTags)
{}

//--------------------------------------------------------------------

template< typename DestT >
Expr::ExpressionBase*
InterpolateParticleExpression<DestT>::Builder::build() const
{
  return new InterpolateParticleExpression<DestT>( srct_, pSizeTag_, pPosTags_);
}

//--------------------------------------------------------------------

template class InterpolateParticleExpression< SpatialOps::SVolField >;

//==========================================================================
