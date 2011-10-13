#include "Stress.h"

#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Stress( const Expr::Tag viscTag,
        const Expr::Tag vel1Tag,
        const Expr::Tag vel2Tag,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<StressT>(id,reg),
    visct_( viscTag ),
    vel1t_( vel1Tag ),
    vel2t_( vel2Tag )
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
  const Expr::FieldManager<ViscT>& viscfm = fml.template field_manager<ViscT>();
  const Expr::FieldManager<Vel1T>& vel1fm = fml.template field_manager<Vel1T>();
  const Expr::FieldManager<Vel2T>& vel2fm = fml.template field_manager<Vel2T>();

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
  SpatFldPtr<StressT> tmp = SpatialFieldStore<StressT>::self().get( stress );

  vel1GradOp_->apply_to_field( *vel1_, stress ); // dui/dxj
  vel2GradOp_->apply_to_field( *vel2_, *tmp   ); // duj/dxi
  
  stress += *tmp; // dui/dxj + duj/dxi
  
  viscInterpOp_->apply_to_field( *visc_, *tmp );
  stress <<= -stress * *tmp; // -mu * (dui/dxj + duj/dxi)
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Builder::Builder( const Expr::Tag viscTag,
                  const Expr::Tag vel1Tag,
                  const Expr::Tag vel2Tag,
                  const Expr::Tag dilTag )
  : visct_( viscTag ),
    vel1t_( vel1Tag ),
    vel2t_( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Expr::ExpressionBase*
Stress<StressT,Vel1T,Vel2T,ViscT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new Stress<StressT,Vel1T,Vel2T,ViscT>( visct_, vel1t_, vel2t_, id, reg );
}


//====================================================================


template< typename StressT, typename VelT, typename ViscT >
Stress<StressT,VelT,VelT,ViscT>::
Stress( const Expr::Tag viscTag,
        const Expr::Tag velTag,
        const Expr::Tag dilTag,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<StressT>(id,reg),
    visct_( viscTag ),
    velt_ ( velTag  ),
    dilt_ ( dilTag  )
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
  const Expr::FieldManager<ViscT>& viscfm = fml.template field_manager<ViscT>();
  const Expr::FieldManager<VelT >& velfm  = fml.template field_manager<VelT >();

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
  SpatFldPtr<StressT> velgrad    = SpatialFieldStore<StressT>::self().get( stress );
  SpatFldPtr<StressT> dilatation = SpatialFieldStore<StressT>::self().get( stress );

  velGradOp_   ->apply_to_field( *vel_, *velgrad    );
  viscInterpOp_->apply_to_field( *visc_, stress     );
  viscInterpOp_->apply_to_field( *dil_, *dilatation );

  stress <<= ( stress * -2.0*( *velgrad ) ) + 2.0/3.0 * stress * *dilatation;
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Stress<StressT,VelT,VelT,ViscT>::
Builder::Builder( const Expr::Tag viscTag,
                  const Expr::Tag vel1Tag,
                  const Expr::Tag vel2Tag,
                  const Expr::Tag dilTag )
  : visct_( viscTag ),
    velt_ ( vel1Tag ),
    dilt_ ( dilTag  )
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Expr::ExpressionBase*
Stress<StressT,VelT,VelT,ViscT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new Stress<StressT,VelT,VelT,ViscT>( visct_, velt_, dilt_, id, reg );
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
