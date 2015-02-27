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

#include <CCA/Components/Wasatch/Expressions/PostProcessing/ContinuityResidual.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
ContinuityResidual( const Expr::Tag&     drhodtTag,
                    const Expr::TagList& velTags    )
  : Expr::Expression<FieldT>(),
    drhodtTag_( drhodtTag),
    vel1t_    ( velTags[0] ),
    vel2t_    ( velTags[1] ),
    vel3t_    ( velTags[2] ),
    is3d_     ( vel1t_ != Expr::Tag() && vel2t_ != Expr::Tag() && vel3t_ != Expr::Tag() )
{
  this->set_gpu_runnable( true );
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( drhodtTag_ != Expr::Tag() )  exprDeps.requires_expression( drhodtTag_ );
  if( vel1t_     != Expr::Tag() )  exprDeps.requires_expression( vel1t_     );
  if( vel2t_     != Expr::Tag() )  exprDeps.requires_expression( vel2t_     );
  if( vel3t_     != Expr::Tag() )  exprDeps.requires_expression( vel3t_     );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& svfm = fml.template field_manager<FieldT>();
  const typename Expr::FieldMgrSelector<Vel1T>::type& v1fm = fml.template field_manager<Vel1T>();
  const typename Expr::FieldMgrSelector<Vel2T>::type& v2fm = fml.template field_manager<Vel2T>();
  const typename Expr::FieldMgrSelector<Vel3T>::type& v3fm = fml.template field_manager<Vel3T>();

  if( drhodtTag_ != Expr::Tag() )  drhodt_ = &svfm.field_ref( drhodtTag_ );
  if( vel1t_ != Expr::Tag() )  vel1_ = &v1fm.field_ref( vel1t_ );
  if( vel2t_ != Expr::Tag() )  vel2_ = &v2fm.field_ref( vel2t_ );
  if( vel3t_ != Expr::Tag() )  vel3_ = &v3fm.field_ref( vel3t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T, typename Vel3T >
void
ContinuityResidual<FieldT,Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() )  vel1GradOp_ = opDB.retrieve_operator<Vel1GradT>();
  if( vel2t_ != Expr::Tag() )  vel2GradOp_ = opDB.retrieve_operator<Vel2GradT>();
  if( vel3t_ != Expr::Tag() )  vel3GradOp_ = opDB.retrieve_operator<Vel3GradT>();
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

  if (drhodtTag_ != Expr::Tag()) cont <<= *drhodt_;
  
  if( is3d_ ){ // fully inline for 3D
    cont <<= cont + (*vel1GradOp_)(*vel1_) + (*vel2GradOp_)(*vel2_) + (*vel3GradOp_)(*vel3_);
  }
  else{ // for 2D and 1D, assemble in pieces
    if( vel1t_ != Expr::Tag() ) cont <<= cont + (*vel1GradOp_)(*vel1_);
    if( vel2t_ != Expr::Tag() ) cont <<= cont + (*vel2GradOp_)(*vel2_);
    if( vel3t_ != Expr::Tag() ) cont <<= cont + (*vel3GradOp_)(*vel3_);
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
