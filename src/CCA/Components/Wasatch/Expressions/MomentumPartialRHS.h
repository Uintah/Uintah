#ifndef MomentumPartialRHS_Expr_h
#define MomentumPartialRHS_Expr_h

#include <expression/Expr_Expression.h>

#include <spatialops/structured/FVStaggered.h>


/**
 *  \class MomRHSPart
 *  \brief Calculates the RHS of a momentum equation excluding the pressure gradient term.
 *
 */
template< typename FieldT >
class MomRHSPart
 : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::structured::FaceTypes<FieldT>::XFace  XFluxT;
  typedef typename SpatialOps::structured::FaceTypes<FieldT>::YFace  YFluxT;
  typedef typename SpatialOps::structured::FaceTypes<FieldT>::ZFace  ZFluxT;

  typedef typename SpatialOps::structured::BasicOpTypes<FieldT>  OpTypes;

  typedef typename OpTypes::DivX  DivX;
  typedef typename OpTypes::DivY  DivY;
  typedef typename OpTypes::DivZ  DivZ;
  

  const Expr::Tag cfluxXt_, cfluxYt_, cfluxZt_, tauXt_, tauYt_, tauZt_, bodyForcet_, emptyTag_;

  const XFluxT *cFluxX_, *tauX_;
  const YFluxT *cFluxY_, *tauY_;
  const ZFluxT *cFluxZ_, *tauZ_;
  const FieldT *bodyForce_;

  const DivX* divXOp_;
  const DivY* divYOp_;
  const DivZ* divZOp_;

  MomRHSPart( const Expr::Tag& convFluxX,
              const Expr::Tag& convFluxY,
              const Expr::Tag& convFluxZ,
              const Expr::Tag& tauX,
              const Expr::Tag& tauY,
              const Expr::Tag& tauZ,
              const Expr::Tag& bodyForce,
              const Expr::ExpressionID& id,
              const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag cfluxXt_, cfluxYt_, cfluxZt_, tauXt_, tauYt_, tauZt_, bodyForcet_;
  public:
    Builder( const Expr::Tag& convFluxX,
             const Expr::Tag& convFluxY,
             const Expr::Tag& convFluxZ,
             const Expr::Tag& tauX,
             const Expr::Tag& tauY,
             const Expr::Tag& tauZ,
             const Expr::Tag& bodyForce );

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;
  };

  ~MomRHSPart();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};
#endif // MomentumPartialRHS_Expr_h
