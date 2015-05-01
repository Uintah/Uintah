#ifndef MonolithicRHS_Expr_h
#define MonolithicRHS_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

/**
 *  \class MonolithicRHS
 *
 *  Calculates
 *  \f[
 *    -\nabla\cdot(\rho \phi \mathbf{u}) + \nabla\cdot(D_\phi \nabla\phi) + s_\phi
 *  \f]
 *  which is the RHS of the transport equation for \f$\rho\phi\f$.
 */
template< typename FieldT >
class MonolithicRHS
: public Expr::Expression<FieldT>
{
  const bool doX_, doY_, doZ_, doSrc_, is3d_;

  typedef typename FaceTypes<FieldT>::XFace  XFaceT;
  typedef typename FaceTypes<FieldT>::YFace  YFaceT;
  typedef typename FaceTypes<FieldT>::ZFace  ZFaceT;

  DECLARE_FIELDS(FieldT, dCoef_, phi_, src_)
  DECLARE_FIELD(XFaceT, convFluxX_)
  DECLARE_FIELD(YFaceT, convFluxY_)
  DECLARE_FIELD(ZFaceT, convFluxZ_)

  typedef SpatialOps::BasicOpTypes<FieldT>  OpTypes;

  typedef typename OpTypes::InterpC2FX InterpX;
  typedef typename OpTypes::InterpC2FY InterpY;
  typedef typename OpTypes::InterpC2FZ InterpZ;
  typedef typename OpTypes::GradX      GradX;
  typedef typename OpTypes::GradY      GradY;
  typedef typename OpTypes::GradZ      GradZ;
  typedef typename OpTypes::DivX       DivX;
  typedef typename OpTypes::DivY       DivY;
  typedef typename OpTypes::DivZ       DivZ;

  const InterpX *interpX_;
  const InterpY *interpY_;
  const InterpZ *interpZ_;
  const GradX   *gradX_;
  const GradY   *gradY_;
  const GradZ   *gradZ_;
  const DivX    *divX_;
  const DivY    *divY_;
  const DivZ    *divZ_;

  MonolithicRHS( const Expr::Tag& dCoefTag,
                 const Expr::Tag& xconvFluxTag,
                 const Expr::Tag& yconvFluxTag,
                 const Expr::Tag& zconvFluxTag,
                 const Expr::Tag& phiTag,
                 const Expr::Tag& srcTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a MonolithicRHS expression
     *  @param resultTag the Expr::Tag for the value that this expression computes
     *  @param dCoefTag the Expr::Tag for the diffusion coefficient
     *  @param xconvFluxTag the Expr::Tag for the convective flux of \f$\rho*\phi\f$
     *  @param yconvFluxTag the Expr::Tag for the convective flux of \f$\rho*\phi\f$
     *  @param zconvFluxTag the Expr::Tag for the convective flux of \f$\rho*\phi\f$
     *  @param phiTag the Expr::Tag for the primitive variable
     *  @param srcTag the Expr::Tag for the source term
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& dCoefTag,
             const Expr::Tag& xconvFluxTag,
             const Expr::Tag& yconvFluxTag,
             const Expr::Tag& zconvFluxTag,
             const Expr::Tag& phiTag,
             const Expr::Tag& srcTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag dCoefTag_, xconvFluxTag_, yconvFluxTag_, zconvFluxTag_, phiTag_, srcTag_;
  };

  ~MonolithicRHS();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // MonolithicRHS_Expr_h
