#include <spatialops/structured/FVStaggered.h>

#include <expression/ExprLib.h>

#include "TemperatureTransEqn.h"
#include "TemperatureRHS.h"
#include "HeatFluxExpr.h"
#include "GradTExpr.h"

namespace so = SpatialOps;

typedef so::SVolField    CellField;
typedef so::SSurfXField  XSideField;

typedef so::BasicOpTypes<CellField>  OpTypes;
typedef OpTypes::GradX               GradXC2F;
typedef OpTypes::DivX                DivX;
typedef OpTypes::InterpC2FX          InterpC2F;

typedef Expr::ExprPatch  PatchT;

typedef TemperatureRHS    <DivX      >    TRHS;
typedef HeatFluxExpr      <GradXC2F  >    HeatFlux_X;
typedef Expr::ConstantExpr<XSideField>    TCond;
typedef GradTExpr         <GradXC2F  >    GradT_X;

//--------------------------------------------------------------------

TemperatureTransEqn::
TemperatureTransEqn( const Expr::Tag& temperature,
                     const Expr::Tag& lambda,
                     const Expr::Tag& gradT,
                     const Expr::Tag& heatFlux,
                     Expr::ExpressionFactory& exprFactory,
                     PatchT& patch )
  : Expr::TransportEquation( exprFactory,
                             temperature.name(),
                             new TRHS::Builder( Expr::Tag("TRHS", Expr::STATE_N), heatFlux ) ),
    tempTag_     ( temperature ),
    thermCondTag_( lambda      ),
    gradTTag_    ( gradT       ),
    heatFluxTag_ ( heatFlux    ),

    patch_( patch )
{
  exprFactory.register_expression( new HeatFlux_X::Builder(heatFluxTag_, thermCondTag_,gradTTag_) );
  exprFactory.register_expression( new TCond     ::Builder(thermCondTag_,20.0) );
  exprFactory.register_expression( new GradT_X   ::Builder(gradTTag_,    tempTag_) );
}

//--------------------------------------------------------------------

TemperatureTransEqn::
~TemperatureTransEqn()
{}

//--------------------------------------------------------------------

void
TemperatureTransEqn::
setup_boundary_conditions( Expr::ExpressionFactory& factory )
{
  // dirichlet BC --- use the interpolant operator to set temperature on the boundary face.
  typedef so::SVolField  FieldT;
  typedef so::ConstValEval BCEval;    // basic functor for constant functions.
  typedef so::BoundaryConditionOp<InterpC2F,BCEval>  TBC;

  const Expr::FieldManagerList& fml = patch_.field_manager_list();
  const FieldT& temp = fml.field_manager<FieldT>().field_ref(tempTag_);

  const so::IntVec last( temp.window_with_ghost().extent() - so::IntVec(1,1,1) );

  const double T0 = 5.0;
  const double TL = 15.0;

  TBC Tbc0( so::IntVec(0, 0, 0),
            so::MINUS_SIDE,
            BCEval(T0),
            patch_.operator_database() );

  TBC TbcL( so::IntVec(last[0]-1,last[1],last[2]),
            so::PLUS_SIDE,
            BCEval(TL),
            patch_.operator_database() );

  // grab the temperature expression
  const Expr::Tag tempLabel( solnVarName_, Expr::STATE_N );
  const Expr::ExpressionID tempID = factory.get_id(tempLabel);
  Expr::Expression<FieldT>& texpr = dynamic_cast<Expr::Expression<FieldT>&>
    ( factory.retrieve_expression( tempLabel, patch_.id() ) );

  // load the BC evaluators onto the expression
  texpr.process_after_evaluate( Tbc0, Tbc0.is_gpu_runnable() );
  texpr.process_after_evaluate( TbcL, TbcL.is_gpu_runnable() );
}

//--------------------------------------------------------------------

Expr::ExpressionID
TemperatureTransEqn::
initial_condition( Expr::ExpressionFactory& exprFactory )
{
  return exprFactory.get_id( Expr::Tag( solnVarName_, Expr::STATE_N ) );
}

//--------------------------------------------------------------------
