#ifndef MonolithicRHS_Expr_h
#define MonolithicRHS_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>

/**
 *  \class MonolithicRHS
 */
template< typename FieldT >
class MonolithicRHS
 : public Expr::Expression<FieldT>
{
  const Expr::Tag tcondTag_, tempTag_;
  const FieldT* tcond_;
  const FieldT* temp_;

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

  MonolithicRHS( const Expr::Tag& tcondTag,
                 const Expr::Tag& tempTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a MonolithicRHS expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& tcondTag,
             const Expr::Tag& tempTag );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag tcondTag_, tempTag_;
  };

  ~MonolithicRHS();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // MonolithicRHS_Expr_h
