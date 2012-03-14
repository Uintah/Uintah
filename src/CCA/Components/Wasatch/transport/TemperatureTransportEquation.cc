/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
    solnExprFactory.register_expression( scinew HeatFluxX( heatFluxXtag, tcondTag, tempTag ) );
    solnExprFactory.register_expression( scinew HeatFluxY( heatFluxYtag, tcondTag, tempTag ) );
    solnExprFactory.register_expression( scinew HeatFluxZ( heatFluxZtag, tcondTag, tempTag ) );

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
    return icFactory.get_id( Expr::Tag( sName.temperature, Expr::STATE_N ) );
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
    const Tag rhsTag(sName.temperature+"_rhs",STATE_NONE);
    //const Expr::Tag volFracTag = Expr::Tag();
    const Expr::Tag emptyTag = Expr::Tag();
    return factory.register_expression( scinew ScalarRHS<FieldT>::Builder(rhsTag,info,densT, emptyTag, emptyTag,emptyTag,emptyTag,tempConstDens) );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
