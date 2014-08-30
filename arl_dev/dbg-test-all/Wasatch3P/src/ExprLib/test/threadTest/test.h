#ifndef test_h
#define test_h

#include "defs.h"

#include <expression/ExprLib.h>

//====================================================================

class RHSExpr : public Expr::Expression<VolT>
{
  const Expr::Tag fluxTag_, srcTag_;
  const bool doFlux_, doSrc_;
  const XFluxT* flux_;
  const VolT*   src_;
  const XDivT* div_;

  RHSExpr( const Expr::Tag& fluxTag,
           const Expr::Tag& srcTag );

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& rhsTag,
             const Expr::Tag& fluxTag,
             const Expr::Tag& srcTag );
  private:
    const Expr::Tag fluxTag_, srcTag_;
  };

};

//====================================================================

class FluxExpr : public Expr::Expression<XFluxT>
{
  const Expr::Tag phiTag_;
  const double diffCoef_;
  const VolT* phi_;
  const XGradT* grad_;

  FluxExpr( const Expr::Tag& var,
            const double diffCoef );

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& fluxTag,
             const Expr::Tag& var,
             const double diffCoef=1.0 );
  private:
    const Expr::Tag tag_;
    const double coef_;
  };
};

//====================================================================

class BusyWork : public Expr::Expression<VolT>
{
  const Expr::Tag phiTag_;
  const VolT* phi_;
  const int nvar_;

  BusyWork( const Expr::Tag& var, const int nvar );

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& result,
             const Expr::Tag& var,
             const int nvar );
  private:
    const Expr::Tag tag_;
    const int nvar_;
  };

};

//====================================================================

class CoupledBusyWork : public Expr::Expression<VolT>
{
  const Expr::Tag phiTag_;
  const int nvar_;

  typedef std::vector<const VolT*> FieldVecT;
  FieldVecT phi_;

  typedef std::vector<VolT::const_iterator> IterVec;
  IterVec iterVec_;
  std::vector<double> tmpVec_;

  CoupledBusyWork( const Expr::Tag& var,
                   const int nvar );

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& result,
             const Expr::Tag& var,
             const int nvar );
  private:
    const Expr::Tag tag_;
    const int nvar_;
  };

};

//====================================================================

#endif
