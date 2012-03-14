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

//-- Wasatch includes --//
#include "ScalarTransportEquation.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Expressions/Multiplier.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/structured/IndexTriplet.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace Wasatch{

  //------------------------------------------------------------------

  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const bool isStrong,
                                        Expr::ExpressionFactory& factory,
                                        typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffFluxTag;  // we will populate this.

    std::string dir;
    diffFluxParams->get("Direction",dir);

    // see if we have an expression set for the diffusive flux.
    Uintah::ProblemSpecP nameTagParam = diffFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      diffFluxTag = parse_nametag( nameTagParam );
    }
    else{ // build an expression for the diffusive flux.

      diffFluxTag = Expr::Tag( primVarName+"_diffFlux_"+dir, Expr::STATE_NONE );

      Expr::ExpressionBuilder* builder = NULL;

      if( dir=="X" ){
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux<typename MyOpTypes::GradX::SrcFieldType, typename MyOpTypes::GradX::DestFieldType>::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          // calling the appropriate form of DiffusiveFlux expression according to constant or variable density.
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          /**
           *  \todo need to ensure that the type that the user gives
           *        for the diffusion coefficient field matches the
           *        type implied here.  Alternatively, we don't let
           *        the user specify the type for the diffusion
           *        coefficient.  But there is the matter of what
           *        independent variable is used when calculating the
           *        coefficient...  Arrrgghh.
           */
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradX::SrcFieldType, typename MyOpTypes::GradX::DestFieldType >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
      }
      else if( dir=="Y" ){
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux< typename MyOpTypes::GradY::SrcFieldType, typename MyOpTypes::GradY::DestFieldType >::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradY::SrcFieldType, typename MyOpTypes::GradY::DestFieldType >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
      }
      else if( dir=="Z") {
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux< typename MyOpTypes::GradZ::SrcFieldType, typename MyOpTypes::GradZ::DestFieldType >::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradZ::SrcFieldType, typename MyOpTypes::GradZ::DestFieldType >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( diffFluxTag, primVarTag, coef, densityTag );
        }
      }

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive flux expression for '" << primVarName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      factory.register_expression( builder );

    }

      typename ScalarRHS<FieldT>::FieldSelector fs;
      if     ( dir=="X" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_X;
      else if( dir=="Y" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Y;
      else if( dir=="Z" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive flux expression" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      info[ fs ] = diffFluxTag;
  }

  //------------------------------------------------------------------

  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            Expr::ExpressionFactory& factory,
                                            typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffVelTag;  // we will populate this.

    std::string dir;
    diffVelParams->get("Direction",dir);

    // see if we have an expression set for the diffusive velocity.
    Uintah::ProblemSpecP nameTagParam = diffVelParams->findBlock("NameTag");
    if( nameTagParam ){
      diffVelTag = parse_nametag( nameTagParam );
    }
    else{ // build an expression for the diffusive velocity.

      diffVelTag = Expr::Tag( primVarName+"_diffVelocity_"+dir, Expr::STATE_NONE );

      Expr::ExpressionBuilder* builder = NULL;

      if( dir=="X" ){
        if( diffVelParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveVelocity<typename MyOpTypes::GradX>::Builder Velocity;
          double coef;
          diffVelParams->get("ConstantDiffusivity",coef);
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
        else if( diffVelParams->findBlock("DiffusionCoefficient") ){
          /**
           *  \todo need to ensure that the type that the user gives
           *        for the diffusion coefficient field matches the
           *        type implied here.  Alternatively, we don't let
           *        the user specify the type for the diffusion
           *        coefficient.  But there is the matter of what
           *        independent variable is used when calculating the
           *        coefficient...  Arrrgghh.
           */
          typedef typename DiffusiveVelocity2< typename MyOpTypes::GradX, typename MyOpTypes::InterpC2FX >::Builder Velocity;
          const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
      }
      else if( dir=="Y" ){
        if( diffVelParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveVelocity< typename MyOpTypes::GradY >::Builder Velocity;
          double coef;
          diffVelParams->get("ConstantDiffusivity",coef);
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
        else if( diffVelParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveVelocity2< typename MyOpTypes::GradY, typename MyOpTypes::InterpC2FY >::Builder Velocity;
          const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
      }
      else if( dir=="Z") {
        if( diffVelParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveVelocity< typename MyOpTypes::GradZ >::Builder Velocity;
          double coef;
          diffVelParams->get("ConstantDiffusivity",coef);
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
        else if( diffVelParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveVelocity2< typename MyOpTypes::GradZ, typename MyOpTypes::InterpC2FZ >::Builder Velocity;
          const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Velocity( diffVelTag, primVarTag, coef );
        }
      }

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive velocity expression for '" << primVarName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      factory.register_expression( builder );

    }

    typename ScalarRHS<FieldT>::FieldSelector fs;
    if     ( dir=="X" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_X;
    else if( dir=="Y" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Y;
    else if( dir=="Z" ) fs=ScalarRHS<FieldT>::DIFFUSIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for diffusive velocity expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    info[ fs ] = diffVelTag;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag solnVarTag,
                                         Expr::ExpressionFactory& factory,
                                         typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef SpatialOps::structured::XVolField XVolField;  ///< field type for x-staggered volume
    typedef SpatialOps::structured::YVolField YVolField;  ///< field type for y-staggered volume
    typedef SpatialOps::structured::ZVolField ZVolField;  ///< field type for z-staggered volume

    typedef OpTypes<FieldT> Ops;
    const std::string& solnVarName = solnVarTag.name();


    Expr::Tag convFluxTag;
    Expr::Tag advVelocityTag;

    // get the direction
    std::string dir;
    convFluxParams->get("Direction",dir);

    // get the interpolation method (UPWIND, CENTRAL, etc...)
    std::string interpMethod;
    convFluxParams->get("Method",interpMethod);
    const Wasatch::ConvInterpMethods convInterpMethod = Wasatch::get_conv_interp_method(interpMethod);

    // get the tag for the advective velocity
    Uintah::ProblemSpecP advVelocityTagParam = convFluxParams->findBlock( "AdvectiveVelocity" );

    if (advVelocityTagParam) {
      advVelocityTag = parse_nametag( advVelocityTagParam->findBlock( "NameTag" ) );
    }
    else{
      // advective velocity is not specified - either take on default velocity
      // from momentum or throw exception
      std::ostringstream msg;
      msg << "ERROR: no advective velocity set for transport equation '" << solnVarName << "'" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    // see if we have an expression set for the advective flux.
    Uintah::ProblemSpecP nameTagParam = convFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      convFluxTag = parse_nametag( nameTagParam );

      // if no expression was specified, build one for the convective flux.
    } else {
      convFluxTag = Expr::Tag( solnVarName + "_convective_flux_" + dir, Expr::STATE_NONE );
      Expr::ExpressionBuilder* builder = NULL;

      if( dir=="X" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,XVolField,typename FaceTypes<FieldT>::XFace>::type VelInterpOpT;

        switch (convInterpMethod) {

          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FX, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(convFluxTag, solnVarTag, advVelocityTag);
            break;

          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FXUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;

          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FXLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;
        }
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,YVolField,typename FaceTypes<FieldT>::YFace>::type VelInterpOpT;

        switch (convInterpMethod) {

          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FY, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(convFluxTag, solnVarTag, advVelocityTag);
            break;

          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FYUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;

          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FYLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;
        }
      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,ZVolField,typename FaceTypes<FieldT>::ZFace>::type VelInterpOpT;

        switch (convInterpMethod) {

          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FZ, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(convFluxTag, solnVarTag, advVelocityTag);
            break;

          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FZUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;

          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FZLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(convFluxTag, solnVarTag, advVelocityTag, convInterpMethod);
            break;
        }
      }

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "ERROR: Could not build a convective flux expression for '" << solnVarName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );

      }

      factory.register_expression( builder );
    }

    typename ScalarRHS<FieldT>::FieldSelector fs;
    if      ( dir=="X" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_X;
    else if ( dir=="Y" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_Y;
    else if ( dir=="Z" ) fs = ScalarRHS<FieldT>::CONVECTIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for convective flux expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    info[ fs ] = convFluxTag;
  }

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

    // defining the primary variable ans solutioan variable tags regarding to the type of
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
      const Expr::ExpressionID phiID = factory.get_id(phiTag);
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
      typedef typename Multiplier<FieldT,SVolField>::Builder  Mult;
      return icFactory.register_expression( new Mult( solnVarTag_, primVarTag_,
                                                      Expr::Tag(densityTag_.name(),Expr::STATE_NONE) ) );
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
                                                    Uintah::ProblemSpecP params )
  {
    typename ScalarRHS<FieldT>::FieldTagInfo info;

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

    //_________________
    // Diffusive Fluxes
    if (!isConstDensity) {
      for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
           diffFluxParams != 0;
           diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFluxExpression") ){

        setup_diffusive_flux_expression<FieldT>( diffFluxParams, densityTag, primVarTag, isStrong, factory, info );
      }
    }
    else {
      for( Uintah::ProblemSpecP diffVelParams=params->findBlock("DiffusiveFluxExpression");
          diffVelParams != 0;
          diffVelParams=diffVelParams->findNextBlock("DiffusiveFluxExpression") ){

        setup_diffusive_velocity_expression<FieldT>( diffVelParams, primVarTag, factory, info );
      }
    }


    //__________________
    // Convective Fluxes
    if (isStrong) {
      for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
           convFluxParams != 0;
           convFluxParams=convFluxParams->findNextBlock("ConvectiveFluxExpression") ){
        setup_convective_flux_expression<FieldT>( convFluxParams, solnVarTag, factory, info );
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
  // Explicit template instantiation
#define INSTANTIATE( FIELDT )                                   \
    template class ScalarTransportEquation< FIELDT >;           \
                                                                \
    template void setup_diffusive_flux_expression<FIELDT>(      \
       Uintah::ProblemSpecP diffFluxParams,                     \
       const Expr::Tag densityTag,                              \
       const Expr::Tag primVarTag,                              \
       const bool isStrong,                                     \
       Expr::ExpressionFactory& factory,                        \
       ScalarRHS<FIELDT>::FieldTagInfo& info );                 \
                                                                \
    template void setup_diffusive_velocity_expression<FIELDT>(  \
       Uintah::ProblemSpecP diffVelParams,                      \
       const Expr::Tag primVarTag,                              \
       Expr::ExpressionFactory& factory,                        \
       ScalarRHS<FIELDT>::FieldTagInfo& info );                 \
                                                                \
    template void setup_convective_flux_expression<FIELDT>(     \
       Uintah::ProblemSpecP convFluxParams,                     \
       const Expr::Tag solnVarName,                             \
       Expr::ExpressionFactory& factory,                        \
       ScalarRHS<FIELDT>::FieldTagInfo& info );

  INSTANTIATE( SVolField );
  INSTANTIATE( XVolField );
  INSTANTIATE( YVolField );
  INSTANTIATE( ZVolField );
  //==================================================================


} // namespace Wasatch
