/*
 * The MIT License
 *
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

//-- Wasatch includes --//
#include "ScalabilityTestTransportEquation.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalabilityTestSrc.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Expressions/MonolithicRHS.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace Wasatch{

  //------------------------------------------------------------------

  template< typename FieldT >
  void setup_diffusive_velocity_expression( const std::string dir,
                                            const std::string thisPhiName,
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;

    Expr::Tag diffFluxTag;  // we will populate this.

    diffFluxTag = Expr::Tag( thisPhiName + "_diffFlux_" + dir, Expr::STATE_NONE );
    const Expr::Tag phiTag( thisPhiName, Expr::STATE_N );

    Expr::ExpressionBuilder* builder = NULL;

    if( dir=="X" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradX>::Builder Flux;
      builder = scinew Flux(diffFluxTag,  phiTag, 1.0 );
    } else if( dir=="Y" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradY>::Builder Flux;
      builder = scinew Flux( diffFluxTag, phiTag, 1.0 );
    } else if( dir=="Z" ){
      typedef typename DiffusiveVelocity<typename MyOpTypes::GradZ>::Builder Flux;
      builder = scinew Flux( diffFluxTag, phiTag, 1.0 );
    }

    if( builder == NULL ){
      std::ostringstream msg;
      msg << "Could not build a diffusive velocity expression for '" << thisPhiName << "'" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    factory.register_expression( builder );

    FieldSelector fs;
    if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
    else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
    else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for diffusive flux expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    info[ fs ] = diffFluxTag;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::
  ScalabilityTestTransportEquation( const std::string thisPhiName,
                                    const Expr::ExpressionID rhsID )
  : Wasatch::TransportEquation( thisPhiName, rhsID,
                                get_staggered_location<FieldT>() )
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::~ScalabilityTestTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     const Uintah::PatchSet* const localPatches,
                                     const PatchInfoMap& patchInfoMap,
                                     const Uintah::MaterialSubset* const materials )
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  setup_boundary_conditions( const GraphHelper& graphHelper,
                             const Uintah::PatchSet* const localPatches,
                             const PatchInfoMap& patchInfoMap,
                             const Uintah::MaterialSubset* const materials)
                             {}

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalabilityTestTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_id( Expr::Tag( this->solution_variable_name(), Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalabilityTestTransportEquation<FieldT>::
  get_rhs_expr_id(std::string thisPhiName,
                  Expr::ExpressionFactory& factory,
                  Uintah::ProblemSpecP params )
  {
    FieldTagInfo info;

    //_________________
    // Diffusive Velocities
    bool doDiffusion = true;
    params->get( "DoDiffusion", doDiffusion);
    if (doDiffusion) {
      setup_diffusive_velocity_expression<FieldT>( "X", thisPhiName, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Y", thisPhiName, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Z", thisPhiName, factory, info );
    }
    //__________________
    // Convective Fluxes
    bool doConvection = true;
    params->get( "DoConvection", doConvection);
    if (doConvection) {
      throw Uintah::ProblemSetupException( "convection is disabled for the scalability test", __FILE__, __LINE__ );
      setup_convective_flux_expression<FieldT>( "X",
                                                Expr::Tag(thisPhiName,Expr::STATE_N),
                                                Expr::Tag(), // convective flux (empty to build it)
                                                Expr::Tag(), // volume fraction
                                                CENTRAL,
                                                parse_nametag( params->findBlock("X-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
      setup_convective_flux_expression<FieldT>( "Y",
                                                Expr::Tag(thisPhiName,Expr::STATE_N),
                                                Expr::Tag(), // convective flux (empty to build it)
                                                Expr::Tag(), // volume fraction
                                                CENTRAL,
                                                parse_nametag( params->findBlock("Y-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
      setup_convective_flux_expression<FieldT>( "Z",
                                                Expr::Tag(thisPhiName,Expr::STATE_N),
                                                Expr::Tag(), // convective flux (empty to build it)
                                                Expr::Tag(), // volume fraction
                                                CENTRAL,
                                                parse_nametag( params->findBlock("Z-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
    }
    //_____________
    // Source Terms
    Expr::Tag srcTag;
    bool doSrc = true;
    params->get( "DoSourceTerm", doSrc);
    if (doSrc) {
      srcTag = Expr::Tag( thisPhiName + "_src", Expr::STATE_NONE );
      info[SOURCE_TERM] = srcTag;

      int nEqs=0;
      params->get( "NumberOfEquations", nEqs );

      std::string basePhiName;
      params->get( "SolutionVariable", basePhiName );
      const Expr::Tag basePhiTag ( basePhiName, Expr::STATE_N );

      params->get( "SolutionVariable", basePhiName );

      typedef typename ScalabilityTestSrc<FieldT>::Builder coupledSrcTerm;
      factory.register_expression( scinew coupledSrcTerm( srcTag, basePhiTag, nEqs) );
    }

    bool monolithic = false;
    if( params->findBlock("MonolithicRHS") )  params->get("MonolithicRHS",monolithic);
    if( monolithic ){
      proc0cout << "ScalabilityTestTransportEquation " << thisPhiName << " MONOLITHIC RHS ACTIVE - diffusion is always on!" << endl;
      const Expr::Tag dcoefTag( thisPhiName+"DiffCoeff", Expr::STATE_NONE );
      factory.register_expression( scinew typename Expr::ConstantExpr<FieldT>::Builder( dcoefTag, 1.0 ) );
      return factory.register_expression(
          scinew typename MonolithicRHS<FieldT>::
          Builder( Expr::Tag(thisPhiName+"_rhs", Expr::STATE_NONE),
                   dcoefTag,
//                   info[CONVECTIVE_FLUX_X],
//                   info[CONVECTIVE_FLUX_Y],
//                   info[CONVECTIVE_FLUX_Z],
                   Expr::Tag( thisPhiName, Expr::STATE_N ),
                   info[SOURCE_TERM] ) );
    }
    else{
      const Expr::Tag densT = Expr::Tag();
      const Expr::Tag emptyTag = Expr::Tag();
      const bool tempConstDens = false;
      return factory.register_expression(
          scinew typename ScalarRHS<FieldT>::Builder( Expr::Tag( thisPhiName + "_rhs", Expr::STATE_NONE ),
                                                      info,
                                                      densT,
                                                      emptyTag, emptyTag, emptyTag, emptyTag,
                                                      tempConstDens) );
    }
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class ScalabilityTestTransportEquation< SVolField >;
  //==================================================================


} // namespace Wasatch
