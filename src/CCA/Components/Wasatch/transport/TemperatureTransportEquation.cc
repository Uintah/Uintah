//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/transport/TemperatureTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <CCA/Components/Wasatch/StringNames.h>

#include <CCA/Components/Wasatch/Expressions/HeatFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>


//-- ExprLib Includes --//
#include <expression/ExprLib.h>


namespace Wasatch{

  //------------------------------------------------------------------

  TemperatureTransportEquation::
  TemperatureTransportEquation( Expr::ExpressionFactory& solnExprFactory,
                                Uintah::ProblemSpecP params )
    : Expr::TransportEquation( StringNames::self().temperature,
                               get_rhs_id( solnExprFactory ) )
  {
    // register all relevant solver expressions

    typedef OpTypes<FieldT>  Ops;

    using Expr::Tag;
    using Expr::STATE_NONE;
    using Expr::STATE_N;

    const StringNames& sName = StringNames::self();

    const Tag
      tempTag( sName.temperature, STATE_N ),
      heatFluxXtag( sName.xHeatFlux, STATE_NONE ),
      heatFluxYtag( sName.yHeatFlux, STATE_NONE ),
      heatFluxZtag( sName.zHeatFlux, STATE_NONE ),
      tcondTag( sName.thermalConductivity, STATE_NONE );

    // fourier heat flux
    solnExprFactory.register_expression( heatFluxXtag, new HeatFlux< Ops::GradX, Ops::InterpC2FX >::Builder( tcondTag, tempTag ) );
    solnExprFactory.register_expression( heatFluxYtag, new HeatFlux< Ops::GradY, Ops::InterpC2FY >::Builder( tcondTag, tempTag ) );
    solnExprFactory.register_expression( heatFluxZtag, new HeatFlux< Ops::GradZ, Ops::InterpC2FZ >::Builder( tcondTag, tempTag ) );

    // species heat flux

    // ...
  }

  //------------------------------------------------------------------

  TemperatureTransportEquation::~TemperatureTransportEquation()
  {}

  //------------------------------------------------------------------

  void
  TemperatureTransportEquation::
  setup_boundary_conditions( Expr::ExpressionFactory& exprFactory )
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  TemperatureTransportEquation::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    const StringNames& sName = StringNames::self();
    return icFactory.get_registry().get_id( Expr::Tag( sName.temperature, Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  Expr::ExpressionID
  TemperatureTransportEquation::get_rhs_id( Expr::ExpressionFactory& factory )
  {
    const StringNames& sName = StringNames::self();

    ScalarRHS::FieldTagInfo info;
    using Expr::Tag;  using Expr::STATE_NONE;

    //    info[ ScalarRHS::CONVECTIVE_FLUX_X ] = ???

    info[ ScalarRHS::DIFFUSIVE_FLUX_X ] = Tag( sName.xHeatFlux, STATE_NONE );
    info[ ScalarRHS::DIFFUSIVE_FLUX_Y ] = Tag( sName.yHeatFlux, STATE_NONE );
    info[ ScalarRHS::DIFFUSIVE_FLUX_Z ] = Tag( sName.zHeatFlux, STATE_NONE );

    return factory.register_expression( Tag(sName.temperature+"_rhs",STATE_NONE),
                                        new ScalarRHS::Builder(info) );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
