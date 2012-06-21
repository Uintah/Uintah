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

#include "Stress.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Stress( const Expr::Tag& viscTag,
        const Expr::Tag& vel1Tag,
        const Expr::Tag& vel2Tag )
  : Expr::Expression<StressT>(),
    visct_     ( viscTag ),
    vel1t_     ( vel1Tag ),
    vel2t_     ( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
~Stress()
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( visct_ );
  exprDeps.requires_expression( vel1t_ );
  exprDeps.requires_expression( vel2t_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ViscT>::type& viscfm = fml.template field_manager<ViscT>();
  const typename Expr::FieldMgrSelector<Vel1T>::type& vel1fm = fml.template field_manager<Vel1T>();
  const typename Expr::FieldMgrSelector<Vel2T>::type& vel2fm = fml.template field_manager<Vel2T>();

  visc_ = &viscfm.field_ref( visct_ );
  vel1_ = &vel1fm.field_ref( vel1t_ );
  vel2_ = &vel2fm.field_ref( vel2t_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  viscInterpOp_ = opDB.retrieve_operator<ViscInterpT>();
  vel1GradOp_   = opDB.retrieve_operator<Vel1GradT  >();
  vel2GradOp_   = opDB.retrieve_operator<Vel2GradT  >();
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
evaluate()
{
  using namespace SpatialOps;
  StressT& stress = this->value();
  stress <<= 0.0;

  SpatFldPtr<StressT> tmp = SpatialFieldStore::get<StressT>( stress );
  *tmp <<= 0.0;

  vel1GradOp_->apply_to_field( *vel1_, stress ); // dui/dxj
  vel2GradOp_->apply_to_field( *vel2_, *tmp   ); // duj/dxi

  stress <<= stress + *tmp; // dui/dxj + duj/dxi
  
  viscInterpOp_->apply_to_field( *visc_, *tmp );
  
  stress <<= - stress * *tmp; // - mu * (dui/dxj + duj/dxi)
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& viscTag,
                  const Expr::Tag& vel1Tag,
                  const Expr::Tag& vel2Tag,
                  const Expr::Tag& dilTag )
  : ExpressionBuilder(result),
    visct_    ( viscTag ),
    vel1t_    ( vel1Tag ),
    vel2t_    ( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Expr::ExpressionBase*
Stress<StressT,Vel1T,Vel2T,ViscT>::Builder::build() const
{
  return new Stress<StressT,Vel1T,Vel2T,ViscT>( visct_, vel1t_, vel2t_ );
}


//====================================================================


template< typename StressT, typename VelT, typename ViscT >
Stress<StressT,VelT,VelT,ViscT>::
Stress( const Expr::Tag& viscTag,
        const Expr::Tag& velTag,
        const Expr::Tag& dilTag )
  : Expr::Expression<StressT>(),
    visct_    ( viscTag ),
    velt_     ( velTag  ),
    dilt_     ( dilTag  )
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Stress<StressT,VelT,VelT,ViscT>::
~Stress()
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
Stress<StressT,VelT,VelT,ViscT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( visct_ );
  exprDeps.requires_expression( velt_  );
  exprDeps.requires_expression( dilt_  );
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
Stress<StressT,VelT,VelT,ViscT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ViscT>::type& viscfm = fml.template field_manager<ViscT>();
  const typename Expr::FieldMgrSelector<VelT >::type& velfm  = fml.template field_manager<VelT >();

  visc_ = &viscfm.field_ref( visct_ );
  vel_  = &velfm. field_ref( velt_  );
  dil_  = &viscfm.field_ref( dilt_  );
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
Stress<StressT,VelT,VelT,ViscT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  viscInterpOp_ = opDB.retrieve_operator<ViscInterpT>();
  velGradOp_    = opDB.retrieve_operator<VelGradT   >();
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
Stress<StressT,VelT,VelT,ViscT>::
evaluate()
{
  using namespace SpatialOps;

  StressT& stress = this->value();
  stress <<= 0.0;

  SpatFldPtr<StressT> velgrad    = SpatialFieldStore::get<StressT>( stress );
  SpatFldPtr<StressT> dilatation = SpatialFieldStore::get<StressT>( stress );
  //*velgrad <<= 0.0;
  //*dilatation <<= 0.0;
  viscInterpOp_->apply_to_field( *visc_, stress ); // stress = mu
  viscInterpOp_->apply_to_field( *dil_, *dilatation );
  velGradOp_   ->apply_to_field( *vel_, *velgrad    );
  stress <<= ( stress * -2.0*( *velgrad ) ) + 2.0/3.0 * stress * *dilatation; // stress = -2 (mu + mu_turbulent) * du/dx + 2/3 (mu + mu_turbulent) * div(u)
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Stress<StressT,VelT,VelT,ViscT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& viscTag,
                  const Expr::Tag& vel1Tag,
                  const Expr::Tag& vel2Tag,
                  const Expr::Tag& dilTag )
  : ExpressionBuilder(result),
    visct_( viscTag ),
    velt_ ( vel1Tag ),
    dilt_ ( dilTag  )
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Expr::ExpressionBase*
Stress<StressT,VelT,VelT,ViscT>::Builder::build() const
{
  return new Stress<StressT,VelT,VelT,ViscT>( visct_, velt_, dilt_ );
}

//====================================================================


//====================================================================
// Explicit template instantiation
#include <spatialops/structured/FVStaggered.h>
#define DECLARE_STRESS( VOL )	\
  template class Stress< SpatialOps::structured::FaceTypes<VOL>::XFace,	\
                         VOL,						\
                         SpatialOps::structured::XVolField,             \
                         SpatialOps::structured::SVolField >;           \
  template class Stress< SpatialOps::structured::FaceTypes<VOL>::YFace,	\
                         VOL,						\
                         SpatialOps::structured::YVolField,		\
                         SpatialOps::structured::SVolField >;           \
  template class Stress< SpatialOps::structured::FaceTypes<VOL>::ZFace,	\
                         VOL,						\
                         SpatialOps::structured::ZVolField,		\
                         SpatialOps::structured::SVolField >;

DECLARE_STRESS( SpatialOps::structured::XVolField );
DECLARE_STRESS( SpatialOps::structured::YVolField );
DECLARE_STRESS( SpatialOps::structured::ZVolField );
//====================================================================
