//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/transport/TemperatureTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <CCA/Components/Wasatch/StringNames.h>

#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>


//-- ExprLib Includes --//
#include <expression/ExprLib.h>


namespace Wasatch{

  //------------------------------------------------------------------

  TemperatureTransportEquation::
  TemperatureTransportEquation( Expr::ExpressionFactory& solnExprFactory )
    : Wasatch::TransportEquation( StringNames::self().temperature,
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

    typedef DiffusiveVelocity2< Ops::GradX, Ops::InterpC2FX >::Builder HeatFluxX;
    typedef DiffusiveVelocity2< Ops::GradY, Ops::InterpC2FY >::Builder HeatFluxY;
    typedef DiffusiveVelocity2< Ops::GradZ, Ops::InterpC2FZ >::Builder HeatFluxZ;

    // fourier heat flux
    solnExprFactory.register_expression( heatFluxXtag, scinew HeatFluxX( tcondTag, tempTag ) );
    solnExprFactory.register_expression( heatFluxYtag, scinew HeatFluxY( tcondTag, tempTag ) );
    solnExprFactory.register_expression( heatFluxZtag, scinew HeatFluxZ( tcondTag, tempTag ) );

    // species heat flux

    // ...
  }

  //------------------------------------------------------------------

  TemperatureTransportEquation::~TemperatureTransportEquation()
  {}

  //------------------------------------------------------------------
  
  void 
  TemperatureTransportEquation::
  setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                            const Uintah::PatchSet* const localPatches,
                            const PatchInfoMap& patchInfoMap,
                            const Uintah::MaterialSubset* const materials)
  {}
  
  //------------------------------------------------------------------

  void 
  TemperatureTransportEquation::
  setup_boundary_conditions( const GraphHelper& graphHelper,
                                 const Uintah::PatchSet* const localPatches,
                                 const PatchInfoMap& patchInfoMap,
                                 const Uintah::MaterialSubset* const materials)
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

    ScalarRHS<FieldT>::FieldTagInfo info;
    using Expr::Tag;  using Expr::STATE_NONE;

    //    info[ ScalarRHS::CONVECTIVE_FLUX_X ] = ???

    info[ ScalarRHS<FieldT>::DIFFUSIVE_FLUX_X ] = Tag( sName.xHeatFlux, STATE_NONE );
    info[ ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Y ] = Tag( sName.yHeatFlux, STATE_NONE );
    info[ ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Z ] = Tag( sName.zHeatFlux, STATE_NONE );

    //
    // Because of the forms that the ScalarRHS expression builders are defined, 
    // we need a density tag and a boolean variable to be passed into this expression
    // builder. So we just define an empty tag and a false boolean to be passed into 
    // the builder of ScalarRHS in order to prevent any errors in ScalarRHS
    
    const Expr::Tag densT = Expr::Tag();
    const bool tempConstDens = false;
    return factory.register_expression( Tag(sName.temperature+"_rhs",STATE_NONE),
                                        scinew ScalarRHS<FieldT>::Builder(info,densT,tempConstDens) );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
