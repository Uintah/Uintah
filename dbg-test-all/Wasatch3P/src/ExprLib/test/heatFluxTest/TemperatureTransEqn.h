#ifndef TemperatureTransEqn_h
#define TemperatureTransEqn_h

#include <expression/TransportEquation.h>
#include <expression/ExprPatch.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>


class TemperatureTransEqn : public Expr::TransportEquation
{
  typedef Expr::ExprPatch PatchT;
  const Expr::Tag tempTag_, thermCondTag_, gradTTag_, heatFluxTag_;
  const PatchT& patch_;

public:

  typedef SpatialOps::SVolField  FieldT;

  TemperatureTransEqn( const Expr::Tag& temperature,
                       const Expr::Tag& lambda,
                       const Expr::Tag& gradT,
                       const Expr::Tag& heatFlux,
                       Expr::ExpressionFactory& exprFactory,
                       PatchT& Patch );

  ~TemperatureTransEqn();

  void setup_boundary_conditions( Expr::ExpressionFactory& );

  Expr::ExpressionID initial_condition( Expr::ExpressionFactory& exprFactory );

};



#endif
