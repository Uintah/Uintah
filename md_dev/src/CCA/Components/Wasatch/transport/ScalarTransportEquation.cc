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
#include "ScalarTransportEquation.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/transport/ParseEquation.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/structured/IndexTriplet.h>

//-- Wasatch includes --//
#include "ParseEquation.h"

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace Wasatch{

  //------------------------------------------------------------------

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalarTransportEquation<FieldT>::
  ScalarTransportEquation( const std::string solnVarName,
                           Uintah::ProblemSpecP params,
                           const Expr::Tag densityTag,
                           const bool isConstDensity,
                           const Expr::ExpressionID rhsID )
    : Wasatch::TransportEquation( solnVarName, rhsID,
                                  get_staggered_location<FieldT>() ),
      isConstDensity_( isConstDensity ),
      densityTag_( densityTag )
  {

    // defining the primary variable ans solution variable tags regarding to the type of
    // the equations that we are solving and throwing appropriate error messages regarding
    // to the input file arguments.
    params->get("StrongForm",isStrong_);

    const bool existPrimVar = params->findBlock("PrimitiveVariable");

    if (isConstDensity_) {
      solnVarTag_ = Expr::Tag::Tag( solnVarName, Expr::STATE_N );
      primVarTag_ = solnVarTag_;

      if (existPrimVar) {
        std::ostringstream msg;
        msg << "ERROR: For constant density cases the primitive variable will be the same as the solution variable. So, you don't need to specify it. Please remove the \"PrimitiveVariable\" block from the \"TransportEquation\" block in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

    }
    else {
      solnVarTag_ = Expr::Tag::Tag( solnVarName, Expr::STATE_N );

      if (isStrong_ && !existPrimVar) {
        std::ostringstream msg;
        msg << "ERROR: When you are solving a transport equation with constant density in its strong form, you need to specify your primitive and solution variables separately. Please include the \"PrimitiveVariable\" block in your input file in the \"TransportEquation\" block." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      else if (isStrong_ && existPrimVar ) {
        const std::string primVarName = get_primvar_name( params );
        primVarTag_ = Expr::Tag::Tag( primVarName, Expr::STATE_NONE );
      }
      else if (!isStrong_ && existPrimVar ) {
        std::ostringstream msg;
        msg << "ERROR: For solving the transport equations in weak form, the primitive variable will be the same as the solution variable. So, you don't need to specify it. Please remove the \"PrimitiveVariable\" block from the \"TransportEquation\" block in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      else {
        primVarTag_ = solnVarTag_;
      }
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalarTransportEquation<FieldT>::~ScalarTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::
  setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     const Uintah::PatchSet* const localPatches,
                                     const PatchInfoMap& patchInfoMap,
                                     const Uintah::MaterialSubset* const materials)
  {

    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    const Expr::Tag phiTag( this->solution_variable_name(), Expr::STATE_N );
    if( factory.have_entry(phiTag) ){
      process_boundary_conditions<FieldT>( phiTag,
                                           this->solution_variable_name(),
                                           this->staggered_location(),
                                           graphHelper,
                                           localPatches,
                                           patchInfoMap,
                                           materials );
    }
  }


  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::
  setup_boundary_conditions( const GraphHelper& graphHelper,
                             const Uintah::PatchSet* const localPatches,
                             const PatchInfoMap& patchInfoMap,
                             const Uintah::MaterialSubset* const materials )
  {
    // see BCHelperTools.cc
    process_boundary_conditions<FieldT>( Expr::Tag( this->solution_variable_name(),Expr::STATE_N ),
                                         this->solution_variable_name(),
                                         this->staggered_location(),
                                         graphHelper,
                                         localPatches,
                                         patchInfoMap,
                                         materials );
    // see BCHelperTools.cc
    process_boundary_conditions<FieldT>( Expr::Tag( this->solution_variable_name()+"_rhs",Expr::STATE_NONE ),
                                        this->solution_variable_name() + "_rhs",
                                        this->staggered_location(),
                                        graphHelper,
                                        localPatches,
                                        patchInfoMap,
                                        materials );

  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalarTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    if (isStrong_ && !isConstDensity_) {
      // register expression to calculate the initial condition of the solution variable from the initial
      // conditions on primitive variable and density in the cases that we are solving for e.g. rho*phi
      typedef ExprAlgebra<SVolField> ExprAlgbr;
      Expr::TagList theTagList;
      theTagList.push_back(primVarTag_);
      theTagList.push_back(Expr::Tag(densityTag_.name(),Expr::STATE_NONE));
      return icFactory.register_expression( new typename ExprAlgbr::Builder( solnVarTag_,
                                                                             theTagList,
                                                                             ExprAlgbr::PRODUCT ) );
    }
    return icFactory.get_id( Expr::Tag( this->solution_variable_name(), Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  std::string
  ScalarTransportEquation<FieldT>::get_solnvar_name( Uintah::ProblemSpecP params )
  {
    std::string solnVarName;
    params->get("SolutionVariable",solnVarName);
    return solnVarName;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  std::string
  ScalarTransportEquation<FieldT>::get_primvar_name( Uintah::ProblemSpecP params )
  {
    std::string primVarName;
    params->get("PrimitiveVariable",primVarName);
    return primVarName;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalarTransportEquation<FieldT>::get_rhs_expr_id( const Expr::Tag densityTag,
                                                    const bool isConstDensity,
                                                    Expr::ExpressionFactory& factory,
                                                    Uintah::ProblemSpecP params,
                                                    TurbulenceParameters turbulenceParams)
  {
    FieldTagInfo info;

    //______________________________________________________________________
    // Setting up the tags for solution variable and primitive variable. Also,
    // getting information about the equation format that we are solving (
    // strong form, weak form, Constant density or variable density) and
    // throwing errors with respect to input file definition.
    const std::string solnVarName = get_solnvar_name( params );
    std::string primVarName;

    Expr::Tag primVarTag, solnVarTag;
    //Expr::Tag advVelocityTagX, advVelocityTagY, advVelocityTagZ;
    bool isStrong;

    params->get("StrongForm",isStrong);

    solnVarTag = Expr::Tag( solnVarName, Expr::STATE_N );

    if (isConstDensity || !isStrong) {
      primVarTag = solnVarTag;
    }
    else {
      const std::string primVarName = get_primvar_name( params );
      primVarTag = Expr::Tag( primVarName, Expr::STATE_NONE );

      factory.register_expression( new typename PrimVar<FieldT,SVolField>::Builder( primVarTag, solnVarTag, densityTag));
    }

    //_____________
    // volume fraction for embedded boundaries Terms
    Expr::Tag volFracTag = Expr::Tag();
    if (params->findBlock("VolumeFractionExpression")) {
      volFracTag = parse_nametag( params->findBlock("VolumeFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag xAreaFracTag = Expr::Tag();
    if (params->findBlock("XAreaFractionExpression")) {
      xAreaFracTag = parse_nametag( params->findBlock("XAreaFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag yAreaFracTag = Expr::Tag();
    if (params->findBlock("YAreaFractionExpression")) {
      yAreaFracTag = parse_nametag( params->findBlock("YAreaFractionExpression")->findBlock("NameTag") );
    }

    Expr::Tag zAreaFracTag = Expr::Tag();
    if (params->findBlock("ZAreaFractionExpression")) {
      zAreaFracTag = parse_nametag( params->findBlock("ZAreaFractionExpression")->findBlock("NameTag") );
    }
    
    
    Expr::Tag turbDiffTag = Expr::Tag();
    // TURBULENCE
    bool enableTurbulenceModel = !(params->findBlock("DisableTurbulenceModel"));
    if (turbulenceParams.turbulenceModelName != NONE && enableTurbulenceModel ) { 
      Expr::Tag turbViscTag = turbulent_viscosity_tag();//Expr::Tag( "TurbulentViscosity", Expr::STATE_NONE );
      turbDiffTag = turbulent_diffusivity_tag();//Expr::Tag( "TurbulentDiffusivity", Expr::STATE_NONE );
      
      if( !factory.have_entry( turbDiffTag ) ){
        typedef typename TurbulentDiffusivity::Builder TurbDiffT;
        factory.register_expression( scinew TurbDiffT(turbDiffTag, densityTag, turbulenceParams.turbulentSchmidt, turbViscTag ) );
      }      
    }
    // END TURBULENCE
    
    //_________________
    // Diffusive Fluxes
    if (!isConstDensity) {
      for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
           diffFluxParams != 0;
           diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFluxExpression") ){

        setup_diffusive_flux_expression<FieldT>( diffFluxParams, densityTag, primVarTag, isStrong, turbDiffTag, factory, info );
      }
    }
    else {
      for( Uintah::ProblemSpecP diffVelParams=params->findBlock("DiffusiveFluxExpression");
          diffVelParams != 0;
          diffVelParams=diffVelParams->findNextBlock("DiffusiveFluxExpression") ){

        setup_diffusive_velocity_expression<FieldT>( diffVelParams, primVarTag, turbDiffTag, factory, info );
      }
    }


    //__________________
    // Convective Fluxes
    if (isStrong) {
      for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
           convFluxParams != 0;
           convFluxParams=convFluxParams->findNextBlock("ConvectiveFluxExpression") ){
        setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag, volFracTag, factory, info );
      }
    }
    else {
      // Here we shoulld use diffusive flux for scalaRHS in weak form
      std::ostringstream msg;
      msg << "ERROR: This part is not written for weak form yet." << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    //_____________
    // Source Terms
    std::vector<Expr::Tag> srcTags;
    for( Uintah::ProblemSpecP sourceTermParams=params->findBlock("SourceTermExpression");
         sourceTermParams != 0;
         sourceTermParams=sourceTermParams->findNextBlock("SourceTermExpression") ){

      const Expr::Tag srcTag = parse_nametag( sourceTermParams->findBlock("NameTag") );
      srcTags.push_back( srcTag );

    }

    //_____________
    // Right Hand Side
    if (isStrong){
      const Expr::Tag rhsTag( solnVarName+"_rhs", Expr::STATE_NONE );
      return factory.register_expression( scinew typename ScalarRHS<FieldT>::Builder(rhsTag, info, srcTags, densityTag, volFracTag, xAreaFracTag, yAreaFracTag, zAreaFracTag, isConstDensity) );
    }
    else{
      // Here we shoulld use diffusive flux for scalaRHS in weak form
      std::ostringstream msg;
      msg << "ERROR: This part is not written for weak form yet." << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
  }

  //------------------------------------------------------------------

  //==================================================================
  // explicit template instantiation
  template class ScalarTransportEquation< SVolField >;

} // namespace Wasatch
