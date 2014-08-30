#ifndef RHS_Expr_h
#define RHS_Expr_h

#include <expression/Expression.h>

template< typename FieldT >
class RHSExpr
  : public Expr::Expression< FieldT >
{

  typedef typename SpatialOps::FaceTypes<FieldT>::XFace XFluxT;
  typedef typename SpatialOps::FaceTypes<FieldT>::YFace YFluxT;
  typedef typename SpatialOps::FaceTypes<FieldT>::ZFace ZFluxT;

  typedef typename SpatialOps::BasicOpTypes<FieldT>::DivX DivX;
  typedef typename SpatialOps::BasicOpTypes<FieldT>::DivY DivY;
  typedef typename SpatialOps::BasicOpTypes<FieldT>::DivZ DivZ;

  const Expr::Tag xft_, yft_, zft_;
  const bool varyX_, varyY_, varyZ_;

  const XFluxT *xflux_;
  const YFluxT *yflux_;
  const ZFluxT *zflux_;

  const DivX  *xdiv_;
  const DivY  *ydiv_;
  const DivZ  *zdiv_;

  RHSExpr( const Expr::Tag& xFlux,
           const Expr::Tag& yFlux,
           const Expr::Tag& zFlux );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag xFluxt_, yFluxt_, zFluxt_;
  public:
    Builder( const Expr::Tag& rhsTag,
             const Expr::Tag& xFlux,
             const Expr::Tag& yFlux,
             const Expr::Tag& zFlux );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~RHSExpr();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
RHSExpr<FieldT>::
RHSExpr( const Expr::Tag& xFlux,
         const Expr::Tag& yFlux,
         const Expr::Tag& zFlux )
  : Expr::Expression<FieldT>(),
    xft_( xFlux ),
    yft_( yFlux ),
    zft_( zFlux ),
    varyX_( xft_ != Expr::Tag() ),
    varyY_( yft_ != Expr::Tag() ),
    varyZ_( zft_ != Expr::Tag() )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}

//--------------------------------------------------------------------

template< typename FieldT >
RHSExpr<FieldT>::
~RHSExpr()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
RHSExpr<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( varyX_ ) exprDeps.requires_expression( xft_ );
  if( varyY_ ) exprDeps.requires_expression( yft_ );
  if( varyZ_ ) exprDeps.requires_expression( zft_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RHSExpr<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  if( varyX_ ) xflux_ = &fml.template field_manager<XFluxT>().field_ref( xft_ );
  if( varyY_ ) yflux_ = &fml.template field_manager<YFluxT>().field_ref( yft_ );
  if( varyZ_ ) zflux_ = &fml.template field_manager<ZFluxT>().field_ref( zft_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RHSExpr<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  if( varyX_ ) xdiv_ = opDB.retrieve_operator<DivX>();
  if( varyY_ ) ydiv_ = opDB.retrieve_operator<DivY>();
  if( varyZ_ ) zdiv_ = opDB.retrieve_operator<DivZ>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RHSExpr<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  using namespace SpatialOps;

  if( varyX_ ){
    if( varyY_ ){
      if( varyZ_ ){
        result <<= -(*xdiv_)(*xflux_) - (*ydiv_)(*yflux_) - (*zdiv_)(*zflux_);
      }
      else{
        result <<= -(*xdiv_)(*xflux_) - (*ydiv_)(*yflux_);
      }
    }
    else{
      if( varyZ_ ) result <<= -(*xdiv_)(*xflux_) - (*zdiv_)(*zflux_);
      else         result <<= -(*xdiv_)(*xflux_);
    }
  }
  else if( varyY_ ){
    if( varyZ_ ) result <<= -(*ydiv_)(*yflux_) - (*zdiv_)(*zflux_);
    else         result <<= -(*ydiv_)(*yflux_);
  }
  else if( varyZ_ ){
    result <<= -(*zdiv_)(*zflux_);
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
RHSExpr<FieldT>::
Builder::Builder( const Expr::Tag& rhsTag,
                  const Expr::Tag& xFlux,
                  const Expr::Tag& yFlux,
                  const Expr::Tag& zFlux )
  : ExpressionBuilder(rhsTag),
    xFluxt_( xFlux ),
    yFluxt_( yFlux ),
    zFluxt_( zFlux )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
RHSExpr<FieldT>::
Builder::build() const
{
  return new RHSExpr<FieldT>( xFluxt_, yFluxt_, zFluxt_ );
}

#endif // RHS_Expr_h
