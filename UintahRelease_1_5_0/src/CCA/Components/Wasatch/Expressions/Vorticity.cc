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

#include "Vorticity.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
Vorticity( const Expr::Tag& vel1tag,
           const Expr::Tag& vel2tag )
: Expr::Expression<FieldT>(),
  vel1t_( vel1tag ),
  vel2t_( vel2tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
~Vorticity()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( vel1t_ != Expr::Tag() )  exprDeps.requires_expression( vel1t_ );
  if( vel2t_ != Expr::Tag() )  exprDeps.requires_expression( vel2t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<Vel1T>::type& v1fm = fml.template field_manager<Vel1T>();
  const typename Expr::FieldMgrSelector<Vel2T>::type& v2fm = fml.template field_manager<Vel2T>();

  if( vel1t_ != Expr::Tag() )  vel1_ = &v1fm.field_ref( vel1t_ );
  if( vel2t_ != Expr::Tag() )  vel2_ = &v2fm.field_ref( vel2t_ );
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( vel1t_ != Expr::Tag() )  {
    Vel1GradTOp_               = opDB.retrieve_operator<Vel1GradT>();
    InpterpVel1FaceT2FieldTOp_ = opDB.retrieve_operator<InpterpVel1FaceT2FieldT>();
  }
  if( vel2t_ != Expr::Tag() )  {
    Vel2GradTOp_               = opDB.retrieve_operator<Vel2GradT>();
    InpterpVel2FaceT2FieldTOp_ = opDB.retrieve_operator<InpterpVel2FaceT2FieldT>();
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
void
Vorticity<FieldT,Vel1T,Vel2T>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& vorticity = this->value();

  SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( vorticity );

  vorticity <<= 0.0;

  if( vel1t_ != Expr::Tag() ){
    SpatFldPtr<Vel1FaceT> tmp1 = SpatialFieldStore::get<Vel1FaceT>( vorticity );
    Vel1GradTOp_->apply_to_field( *vel1_, *tmp1 );

    *tmp <<= 0.0;
    InpterpVel1FaceT2FieldTOp_->apply_to_field( *tmp1, *tmp );

    vorticity <<= *tmp;
  }

  if( vel2t_ != Expr::Tag() ){
    SpatFldPtr<Vel2FaceT> tmp2 = SpatialFieldStore::get<Vel2FaceT>( vorticity );
    Vel2GradTOp_->apply_to_field( *vel2_, *tmp2 );

    *tmp <<= 0.0;
    InpterpVel2FaceT2FieldTOp_->apply_to_field( *tmp2, *tmp );

    vorticity <<= vorticity - *tmp;
  }
}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Vorticity<FieldT,Vel1T,Vel2T>::
Builder::Builder( const Expr::Tag& result,
                 const Expr::Tag& vel1tag,
                 const Expr::Tag& vel2tag )
: ExpressionBuilder(result),
v1t_( vel1tag ), v2t_( vel2tag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename Vel1T, typename Vel2T >
Expr::ExpressionBase*
Vorticity<FieldT,Vel1T,Vel2T>::Builder::build() const
{
  return new Vorticity<FieldT,Vel1T,Vel2T>( v1t_, v2t_ );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class Vorticity< SpatialOps::structured::SVolField,
                          SpatialOps::structured::YVolField,
                          SpatialOps::structured::XVolField >;

template class Vorticity< SpatialOps::structured::SVolField,
                          SpatialOps::structured::ZVolField,
                          SpatialOps::structured::YVolField >;

template class Vorticity< SpatialOps::structured::SVolField,
                          SpatialOps::structured::XVolField,
                          SpatialOps::structured::ZVolField >;

//==========================================================================
