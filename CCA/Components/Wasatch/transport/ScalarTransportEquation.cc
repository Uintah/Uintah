//-- Wasatch includes --//
#include "ScalarTransportEquation.h"
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

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
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const std::string& phiName,
                                        Expr::ExpressionFactory& factory,
                                        typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef OpTypes<FieldT> MyOpTypes;
    
    Expr::Tag diffFluxTag;  // we will populate this.
    
    std::string dir;
    diffFluxParams->get("Direction",dir);
    
    // see if we have an expression set for the diffusive flux.
    Uintah::ProblemSpecP nameTagParam = diffFluxParams->findBlock("NameTag");
    if( nameTagParam ){
      diffFluxTag = parse_nametag( nameTagParam );
    }
    else{ // build an expression for the diffusive flux.
      
      diffFluxTag = Expr::Tag( phiName+"_diffFlux_"+dir, Expr::STATE_NONE );
      const Expr::Tag phiTag( phiName, Expr::STATE_N );
      
      Expr::ExpressionBuilder* builder = NULL;
      
      if( dir=="X" ){
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux<typename MyOpTypes::GradX>::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = scinew Flux( phiTag, coef );
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
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradX, typename MyOpTypes::InterpC2FX >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( phiTag, coef );
        }
      }
      else if( dir=="Y" ){
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux<typename MyOpTypes::GradY>::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = scinew Flux( phiTag, coef );
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradY, typename MyOpTypes::InterpC2FY >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( phiTag, coef );
        }
      }
      else if( dir=="Z") {
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          typedef typename DiffusiveFlux<typename MyOpTypes::GradZ>::Builder Flux;
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = scinew Flux( phiTag, coef );
        }
        else if( diffFluxParams->findBlock("DiffusionCoefficient") ){
          typedef typename DiffusiveFlux2< typename MyOpTypes::GradZ, typename MyOpTypes::InterpC2FZ >::Builder Flux;
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
          builder = scinew Flux( phiTag, coef );
        }
      }
      
      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive flux expression for '" << phiName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      factory.register_expression( diffFluxTag, builder );
      
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

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const std::string& phiName,
                                         Expr::ExpressionFactory& factory,
                                         typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    typedef SpatialOps::structured::XVolField XVolField;  ///< field type for x-staggered volume
    typedef SpatialOps::structured::YVolField YVolField;  ///< field type for y-staggered volume
    typedef SpatialOps::structured::ZVolField ZVolField;  ///< field type for z-staggered volume

    typedef OpTypes<FieldT> Ops;
    
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
      msg << "ERROR: no advective velocity set for transport equation '" << phiName << "'" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    // see if we have an expression set for the advective flux.
    Uintah::ProblemSpecP nameTagParam = convFluxParams->findBlock("NameTag");
    if( nameTagParam ){      
      convFluxTag = parse_nametag( nameTagParam );

      // if no expression was specified, build one for the convective flux.  
    } else {             
      convFluxTag = Expr::Tag( phiName + "_convective_flux_" + dir, Expr::STATE_NONE );
      const Expr::Tag phiTag( phiName, Expr::STATE_N );
      Expr::ExpressionBuilder* builder = NULL;
      
      if( dir=="X" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,XVolField,typename FaceTypes<FieldT>::XFace>::type VelInterpOpT;

        switch (convInterpMethod) {
            
          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FX, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            break;
            
          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FXUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(phiTag, advVelocityTag, convInterpMethod);
            break;
            
          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FXLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(phiTag, advVelocityTag, convInterpMethod);          
            break;
        }
      }
      else if( dir=="Y" ){
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,YVolField,typename FaceTypes<FieldT>::YFace>::type VelInterpOpT;
        
        switch (convInterpMethod) {
            
          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FY, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            break;
            
          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FYUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(phiTag, advVelocityTag, convInterpMethod);
            break;
            
          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FYLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(phiTag, advVelocityTag, convInterpMethod);          
            break;
        }        
      }
      else if( dir=="Z") {
        proc0cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION USING " << interpMethod << std::endl;
        typedef typename OperatorTypeBuilder<Interpolant,ZVolField,typename FaceTypes<FieldT>::ZFace>::type VelInterpOpT;
        
        switch (convInterpMethod) {
            
          case CENTRAL: // for central and upwind, use specified interpolants
            typedef typename ConvectiveFlux< typename Ops::InterpC2FZ, VelInterpOpT >::Builder convFluxCent;
            builder = scinew convFluxCent(phiTag, advVelocityTag);
            break;
            
          case UPWIND:
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FZUpwind, VelInterpOpT >::Builder convFluxUpw;
            builder = scinew convFluxUpw(phiTag, advVelocityTag, convInterpMethod);
            break;
            
          default: // for all other limiter types
            typedef typename ConvectiveFluxLimiter< typename Ops::InterpC2FZLimiter, VelInterpOpT >::Builder convFluxLim;
            builder = scinew convFluxLim(phiTag, advVelocityTag, convInterpMethod);          
            break;
        }        
      }
      
      if( builder == NULL ){        
        std::ostringstream msg;
        msg << "ERROR: Could not build a convective flux expression for '" << phiName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        
      }
      
      factory.register_expression( convFluxTag, builder );
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
  ScalarTransportEquation( const std::string phiName,
                           const Expr::ExpressionID rhsID )
    : Wasatch::TransportEquation( phiName, rhsID, get_staggered_location<FieldT>() )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalarTransportEquation<FieldT>::~ScalarTransportEquation()
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >  
  void ScalarTransportEquation<FieldT>::setup_boundary_conditions(const GraphHelper& graphHelper,
                                         const Uintah::PatchSet* const localPatches,
                                         const PatchInfoMap& patchInfoMap,
                                         const Uintah::MaterialSubset* const materials)
  {
    build_bcs( Expr::Tag( this->solution_variable_name(),
                         Expr::STATE_N ),
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
    return icFactory.get_registry().get_id( Expr::Tag( this->solution_variable_name(),
                                                       Expr::STATE_N ) );
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  std::string
  ScalarTransportEquation<FieldT>::get_phi_name( Uintah::ProblemSpecP params )
  {
    std::string phiName;
    params->get("SolutionVariable",phiName);
    return phiName;
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionID
  ScalarTransportEquation<FieldT>::get_rhs_expr_id( Expr::ExpressionFactory& factory,
                                                    Uintah::ProblemSpecP params )
  {
    typename ScalarRHS<FieldT>::FieldTagInfo info;
    
    const std::string phiName = get_phi_name( params );
    
    //_________________
    // Diffusive Fluxes
    for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
         diffFluxParams != 0;
         diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFluxExpression") ){
      
      setup_diffusive_flux_expression<FieldT>( diffFluxParams, phiName, factory, info );
      
    }
    
    //__________________
    // Convective Fluxes
    for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
         convFluxParams != 0;
         convFluxParams=convFluxParams->findNextBlock("ConvectiveFluxExpression") ){
    
      setup_convective_flux_expression<FieldT>( convFluxParams, phiName, factory, info );
      
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
    
    return factory.register_expression( Expr::Tag( phiName+"_rhs", Expr::STATE_NONE ),
                                        scinew typename ScalarRHS<FieldT>::Builder(info,srcTags) );
  }
  
  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class ScalarTransportEquation< SVolField >;
  template class ScalarTransportEquation< XVolField >;
  template class ScalarTransportEquation< YVolField >;
  template class ScalarTransportEquation< ZVolField >;
  //==================================================================


} // namespace Wasatch
