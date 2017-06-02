/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>

//-- includes for the expressions built here --//
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>

//-- Uintah includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/Parallel.h>

//-- Expression Library includes --//
#include <expression/ExpressionFactory.h>

#include <iostream>

namespace WasatchCore{
  
  //-----------------------------------------------------------------
  
  template< typename FieldT >
  void setup_convective_flux_expression( const std::string& dir,
                                        const Expr::Tag& solnVarTag,
                                        Expr::Tag convFluxTag,
                                        const ConvInterpMethods convMethod,
                                        const Expr::Tag& advVelocityTag,
                                        Expr::ExpressionFactory& factory,
                                        FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> Ops;
    typedef typename FaceTypes<FieldT>::XFace XFace;
    typedef typename FaceTypes<FieldT>::YFace YFace;
    typedef typename FaceTypes<FieldT>::ZFace ZFace;
    
    if( advVelocityTag == Expr::Tag() ){
      std::ostringstream msg;
      msg << "ERROR: no advective velocity set for transport equation '" << solnVarTag.name() << "'" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    if( convFluxTag == Expr::Tag() ){
      const TagNames& tagNames = TagNames::self();
      convFluxTag = Expr::Tag( solnVarTag.name() + tagNames.convectiveflux + dir, Expr::STATE_NONE );
      // make new Tag for solnVar by adding the appropriate suffix ( "_*" or nothing ). This
      // is because we need the ScalarRHS at time step n+1 for our pressure projection method
      
      Expr::ExpressionBuilder* builder = nullptr;
      
      const std::string interpMethod = get_conv_interp_method( convMethod );
      if( dir=="X" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FXLimiter,
          typename Ops::InterpC2FXUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   XFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,SVolField,XFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FXLimiter,
          typename Ops::InterpC2FXUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   XFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,XVolField,XFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }
        
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FYLimiter,
          typename Ops::InterpC2FYUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   YFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,SVolField,YFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FYLimiter,
          typename Ops::InterpC2FYUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   YFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,YVolField,YFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }
        
      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION USING " << interpMethod << std::endl;
        if (Wasatch::flow_treatment() == WasatchCore::COMPRESSIBLE) {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FZLimiter,
          typename Ops::InterpC2FZUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   ZFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,SVolField,ZFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        } else {
          typedef typename ConvectiveFluxLimiter<
          typename Ops::InterpC2FZLimiter,
          typename Ops::InterpC2FZUpwind,
          typename OperatorTypeBuilder<Interpolant,FieldT,   ZFace>::type, // scalar interp type
          typename OperatorTypeBuilder<Interpolant,ZVolField,ZFace>::type  // velocity interp type
          >::Builder ConvFluxLim;
          builder = scinew ConvFluxLim( convFluxTag, solnVarTag, advVelocityTag, convMethod, info[VOLUME_FRAC] );
        }
      }
      
      if( builder == nullptr ){
        std::ostringstream msg;
        msg << "ERROR: Could not build a convective flux expression for '"
        << solnVarTag.name() << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      factory.register_expression( builder );
    }
    
    FieldSelector fs;
    if     ( dir=="X" ) fs = CONVECTIVE_FLUX_X;
    else if( dir=="Y" ) fs = CONVECTIVE_FLUX_Y;
    else if( dir=="Z" ) fs = CONVECTIVE_FLUX_Z;
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for convective flux expression on " << solnVarTag.name() << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    info[ fs ] = convFluxTag;
  }
  
  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag& solnVarTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info )
  {
    Expr::Tag convFluxTag, advVelocityTag, advVelocityCorrectedTag;
    
    std::string dir, interpMethod;
    convFluxParams->getAttribute("direction",dir);
    convFluxParams->getAttribute("method",interpMethod);
    
    // get the tag for the advective velocity
    Uintah::ProblemSpecP advVelocityTagParam = convFluxParams->findBlock( "AdvectiveVelocity" );
    if( advVelocityTagParam ){
      advVelocityTag = parse_nametag( advVelocityTagParam->findBlock( "NameTag" ) );
    }
    
    // see if we have an expression set for the advective flux.
    Uintah::ProblemSpecP nameTagParam = convFluxParams->findBlock("NameTag");
    if( nameTagParam ) convFluxTag = parse_nametag( nameTagParam );
    
    setup_convective_flux_expression<FieldT>( dir,
                                              solnVarTag, convFluxTag,
                                              get_conv_interp_method(interpMethod),
                                              advVelocityTag,
                                              factory,
                                              info );
  }
  
  //-----------------------------------------------------------------
  
  template< typename FluxT >
  Expr::ExpressionBuilder*
  build_diff_flux_expr( Uintah::ProblemSpecP diffFluxParams,
                       const Expr::Tag& diffFluxTag,
                       const Expr::Tag& primVarTag,
                       const Expr::Tag& densityTag,
                       const Expr::Tag& turbDiffTag )
  {
    typedef typename DiffusiveFlux<FluxT>::Builder Flux;
    bool isDiffusiveVelocity = false;
    if( diffFluxParams->findAttribute("isvelocity") ){
      diffFluxParams->getAttribute("isvelocity",isDiffusiveVelocity);
    }
    
    if( diffFluxParams->findAttribute("coefficient") ){
      double coef;
      diffFluxParams->getAttribute("coefficient",coef);
      if (isDiffusiveVelocity) return scinew typename DiffusiveVelocity<FluxT>::Builder( diffFluxTag, primVarTag, coef, turbDiffTag );
      return scinew Flux( diffFluxTag, primVarTag, coef, turbDiffTag, densityTag );
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
      const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
      if (isDiffusiveVelocity) return scinew typename DiffusiveVelocity<FluxT>::Builder( diffFluxTag, primVarTag, coef, turbDiffTag );
      return scinew Flux( diffFluxTag, primVarTag, coef, turbDiffTag, densityTag );
    }
    else {
      std::ostringstream msg;
      msg << "You mus provide a coefficient for your diffusive flux expressions. Please revise your input file." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    return nullptr;
  }
  
  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                       const Expr::Tag densityTag,
                                       const Expr::Tag primVarTag,
                                       const Expr::Tag turbDiffTag,
                                       Expr::ExpressionFactory& factory,
                                       FieldTagInfo& info )
  {
    typedef typename FaceTypes<FieldT>::XFace XFaceT;
    typedef typename FaceTypes<FieldT>::YFace YFaceT;
    typedef typename FaceTypes<FieldT>::ZFace ZFaceT;
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffFluxTag;  // we will populate this.
    
    std::string direction;
    diffFluxParams->getAttribute("direction",direction);
    
    const bool singleDirection = (direction == "X" || direction == "Y" || direction == "Z");
    // see if we have an expression set for the diffusive flux.
    Uintah::ProblemSpecP nameTagParam = diffFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      if( singleDirection ) diffFluxTag = parse_nametag( nameTagParam );
      else{
        std::ostringstream msg;
        msg << "You cannot build a diffusive flux expression with a specified nametag for '" << primVarName << "' in multiple directions" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      FieldSelector fs;
      if     ( direction == "X" ) fs=DIFFUSIVE_FLUX_X;
      else if( direction == "Y" ) fs=DIFFUSIVE_FLUX_Y;
      else if( direction == "Z" ) fs=DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive flux expression" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      info[ fs ] = diffFluxTag;
    }
    else{ // build an expression for the diffusive flux.
      
      for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
        std::string dir(1,*it);
        const TagNames& tagNames = TagNames::self();
        diffFluxTag = Expr::Tag( primVarName + tagNames.diffusiveflux + dir, Expr::STATE_NONE );
        // make new Tags for density and primVar by adding the appropriate suffix ( "_*" or nothing ). This
        // is because we need the ScalarRHS at time step n+1 for our pressure projection method
        
        Expr::ExpressionBuilder* builder = nullptr;
        if     ( dir=="X" ) builder = build_diff_flux_expr<XFaceT>(diffFluxParams,diffFluxTag,primVarTag,densityTag,turbDiffTag);
        else if( dir=="Y" ) builder = build_diff_flux_expr<YFaceT>(diffFluxParams,diffFluxTag,primVarTag,densityTag,turbDiffTag);
        else if( dir=="Z" ) builder = build_diff_flux_expr<ZFaceT>(diffFluxParams,diffFluxTag,primVarTag,densityTag,turbDiffTag);
        
        if( builder == nullptr ){
          std::ostringstream msg;
          msg << "Could not build a diffusive flux expression for '" << primVarName << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        factory.register_expression( builder );
        
        FieldSelector fs;
        if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
        else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
        else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
        else{
          std::ostringstream msg;
          msg << "Invalid direction selection for diffusive flux expression" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        info[ fs ] = diffFluxTag;
      }
    }
  }
  
  //------------------------------------------------------------------
  
  template< typename VelT >
  Expr::ExpressionBuilder*
  build_diff_vel_expr( Uintah::ProblemSpecP diffVelParams,
                      const Expr::Tag& diffVelTag,
                      const Expr::Tag& primVarTag,
                      const Expr::Tag& turbDiffTag )
  {
    typedef typename DiffusiveVelocity<VelT>::Builder Velocity;
    
    if( diffVelParams->findAttribute("coefficient") ){
      double coef;
      diffVelParams->getAttribute("coefficient",coef);
      return scinew Velocity( diffVelTag, primVarTag, coef, turbDiffTag );
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
      const Expr::Tag coef = parse_nametag( diffVelParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
      return scinew Velocity( diffVelTag, primVarTag, coef, turbDiffTag );
    }
    return nullptr;
  }
  
  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            const Expr::Tag turbDiffTag,
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info )
  {
    typedef typename FaceTypes<FieldT>::XFace XFaceT;
    typedef typename FaceTypes<FieldT>::YFace YFaceT;
    typedef typename FaceTypes<FieldT>::ZFace ZFaceT;
    
    const std::string& primVarName = primVarTag.name();
    Expr::Tag diffVelTag;  // we will populate this.
    
    std::string direction;
    diffVelParams->getAttribute("direction",direction);
    
    const bool singleDirection = (direction == "X" || direction == "Y" || direction == "Z");
    // see if we have an expression set for the diffusive velocity.
    Uintah::ProblemSpecP nameTagParam = diffVelParams->findBlock("NameTag");
    if( nameTagParam ){
      if( singleDirection ) diffVelTag = parse_nametag( nameTagParam );
      else {
        std::ostringstream msg;
        msg << "You cannot build a diffusive velocity expression with a specified nametag for '" << primVarName << "' in multiple directions" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      FieldSelector fs;
      if     ( direction == "X" ) fs=DIFFUSIVE_FLUX_X;
      else if( direction == "Y" ) fs=DIFFUSIVE_FLUX_Y;
      else if( direction == "Z" ) fs=DIFFUSIVE_FLUX_Z;
      else{
        std::ostringstream msg;
        msg << "Invalid direction selection for diffusive velocity expression" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      info[ fs ] = diffVelTag;
      
    }
    else { // build an expression for the diffusive velocity.
      
      for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
        std::string dir(1,*it);
        diffVelTag = Expr::Tag( primVarName+"_diffVelocity_"+dir, Expr::STATE_NONE );
        
        Expr::ExpressionBuilder* builder = nullptr;
        if     ( dir=="X" )  builder = build_diff_vel_expr<XFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        else if( dir=="Y" )  builder = build_diff_vel_expr<YFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        else if( dir=="Z" )  builder = build_diff_vel_expr<ZFaceT>(diffVelParams,diffVelTag,primVarTag,turbDiffTag);
        
        if( builder == nullptr ){
          std::ostringstream msg;
          msg << "Could not build a diffusive velocity expression for '"
          << primVarName << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        factory.register_expression( builder );
        
        FieldSelector fs;
        if     ( dir=="X" ) fs=DIFFUSIVE_FLUX_X;
        else if( dir=="Y" ) fs=DIFFUSIVE_FLUX_Y;
        else if( dir=="Z" ) fs=DIFFUSIVE_FLUX_Z;
        else{
          std::ostringstream msg;
          msg << "Invalid direction selection for diffusive velocity expression" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        info[ fs ] = diffVelTag;
      }
    }
  }
  
  //------------------------------------------------------------------
  
  //==================================================================
  // explicit template instantiation
#define INSTANTIATE_DIFFUSION( FIELDT )                         \
\
template void setup_diffusive_flux_expression<FIELDT>(      \
Uintah::ProblemSpecP diffFluxParams,                     \
const Expr::Tag densityTag,                              \
const Expr::Tag primVarTag,                              \
const Expr::Tag turbDiffTag,                             \
Expr::ExpressionFactory& factory,                        \
FieldTagInfo& info );                                    \
\
template void setup_diffusive_velocity_expression<FIELDT>(  \
Uintah::ProblemSpecP diffVelParams,                      \
const Expr::Tag primVarTag,                              \
const Expr::Tag turbDiffTag,                             \
Expr::ExpressionFactory& factory,                        \
FieldTagInfo& info );
  
#define INSTANTIATE_CONVECTION( FIELDT )                        \
template void setup_convective_flux_expression<FIELDT>(     \
const std::string& dir,                                 \
const Expr::Tag& solnVarTag,                            \
Expr::Tag convFluxTag,                                  \
const ConvInterpMethods convMethod,                     \
const Expr::Tag& advVelocityTag,                        \
Expr::ExpressionFactory& factory,                       \
FieldTagInfo& info );                                   \
\
template void setup_convective_flux_expression<FIELDT>(     \
Uintah::ProblemSpecP convFluxParams,                    \
const Expr::Tag& solnVarName,                           \
Expr::ExpressionFactory& factory,                       \
FieldTagInfo& info );
  
  // diffusive fluxes only for scalars.
  INSTANTIATE_DIFFUSION ( SVolField )
  
  // convective fluxes are supported for momentum as well.
  INSTANTIATE_CONVECTION( SVolField )
  
  //-----------------------------------------------------------------
  
} // namespace WasatchCore
