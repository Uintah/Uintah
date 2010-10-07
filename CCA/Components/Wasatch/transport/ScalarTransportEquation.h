#ifndef Wasatch_ScalarTransportEquation_h
#define Wasatch_ScalarTransportEquation_h

//-- ExprLib includes --//
#include <expression/TransportEquation.h>
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredTypes.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/transport/ScalarTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>


namespace Wasatch{

  /**
   *  \class ScalarTransportEquation
   *  \date June, 2010
   *  \author James C. Sutherland
   *
   *  \brief Support for a generic scalar transport equation.
   *
   *  Sets up solution for a transport equation of the form:
   *
   *  \f[
   *    \frac{\partial \rho \phi}{\partial t} =
   *    - \frac{\partial \rho \phi u_x }{\partial x} 
   *    - \frac{\partial \rho \phi u_y }{\partial y} 
   *    - \frac{\partial \rho \phi u_z }{\partial z} 
   *    - \frac{\partial J_{\phi,x}}{\partial x}
   *    - \frac{\partial J_{\phi,y}}{\partial y}
   *    - \frac{\partial J_{\phi,z}}{\partial z}
   *    + s_\phi
   *  \f]
   *
   *  Any or all of the terms in the RHS above may be activated
   *  through the input file.
   *
   *  \par Notes & Restrictions
   *
   *  - Currently, only basic forms for the scalar diffusive flux are
   *    supported.  Specifically, either an expression for the
   *    diffusion coefficient, \f$\Gamma_\phi\f$ is required or the
   *    diffusion coefficient must be a constant value.  See
   *    DiffusiveFlux and DiffusiveFlux2 classes.
   *
   *  - Source terms can only be added if the expression to evaluate
   *    them has been constructed elsewhere.
   *
   *  \todo Need to hook in parser support for boundary and initial conditions.
   */
  template<typename FieldT>
  class ScalarTransportEquation : public Expr::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a ScalarTransportEquation
     *  \param phiName the name of the solution variable for this ScalarTransportEquation
     *  \param id the Expr::ExpressionID for the RHS expression for this ScalarTransportEquation
     *
     *  Note that the static member methods get_rhs_expr_id and
     *  get_phi_name can be useful to obtain the appropriate input
     *  arguments here.
     */
    ScalarTransportEquation( const std::string phiName,
                             const Expr::ExpressionID id );

    ~ScalarTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( Expr::ExpressionFactory& factory );

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     * \brief Parse the input file to determine the rhs expression id.
     *        Also registers convective flux, diffusive flux, and
     *        source term expressions.
     *
     *  \param solnExprFactory the Expr::ExpressionFactory object that
     *         terms associated with the RHS of this transport
     *         equation should be registered on.
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation.  Scope should be within the TransportEquation tag.
     */
    static Expr::ExpressionID get_rhs_expr_id( Expr::ExpressionFactory& factory, Uintah::ProblemSpecP params );

    /**
     *  \brief Parse the input file to get the name of this ScalarTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_phi_name( Uintah::ProblemSpecP params );
  
  private:

  };


// ###################################################################
//
//                          Implementation
//
// ###################################################################


  
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
    
    info[ fs ] = diffFluxTag;
  }
    
  //------------------------------------------------------------------

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const std::string& phiName,
                                         Expr::ExpressionFactory& factory,
                                         typename ScalarRHS<FieldT>::FieldTagInfo& info )
  {
    // typedef the fields for the interpolant - cell to face, X, Y, or Z
//    typedef OpTypes<FieldT> MyOpTypes;
//    typedef typename MyOpTypes::InterpC2FX InterpC2FX;
//    typedef typename MyOpTypes::InterpC2FY InterpC2FY;
//    typedef typename MyOpTypes::InterpC2FZ InterpC2FZ;
    typedef SpatialOps::structured::XVolField XVolField;  ///< field type for x-staggered volume
    typedef SpatialOps::structured::YVolField YVolField;  ///< field type for y-staggered volume
    typedef SpatialOps::structured::ZVolField ZVolField;  ///< field type for z-staggered volume
    
    Expr::Tag convFluxTag;
    Expr::Tag advVelocityTag;
    
    // get the direction
    std::string dir;
    convFluxParams->get("Direction",dir);
    
    // get the interpolation method (UPWIND, CENTRAL, etc...)
    std::string interpMethod;
    convFluxParams->get("Method",interpMethod);
    
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
        cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN X DIRECTION "<< std::endl;
        
        if (interpMethod=="UPWIND") {
          cout << "SETTING UP UPWIND CONVECTION INTERPOLANT IN X DIRECTION "<< std::endl;
          typedef UpwindInterpolant< FieldT, typename FaceTypes<FieldT>::XFace > UpwindInterpOpT;
          typedef typename OperatorTypeBuilder<Interpolant,XVolField,typename FaceTypes<FieldT>::XFace>::type VelInterpOpT;
          typedef typename ConvectiveFluxUpwind<UpwindInterpOpT,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);
          
        } else if (interpMethod=="CENTRAL") {      
          cout << "SETTING UP CENTRAL CONVECTION INTERPOLANT IN X DIRECTION "<< std::endl;
          typedef typename OperatorTypeBuilder<Interpolant,FieldT,typename FaceTypes<FieldT>::XFace>::type InterpPhiVol2PhiFX;
          typedef typename OperatorTypeBuilder<Interpolant,XVolField,typename FaceTypes<FieldT>::XFace>::type VelInterpOpT;
          typedef typename ConvectiveFlux<InterpPhiVol2PhiFX,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);          
        }
      
      } else if( dir=="Y" ){        
        cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Y DIRECTION "<< std::endl;
        
        if (interpMethod=="UPWIND") {    
          cout << "SETTING UP UPWIND CONVECTION INTERPOLANT IN Y DIRECTION "<< std::endl;
          typedef UpwindInterpolant< FieldT, typename FaceTypes<FieldT>::YFace > UpwindInterpOpT;
          typedef typename OperatorTypeBuilder<Interpolant,YVolField,typename FaceTypes<FieldT>::YFace>::type VelInterpOpT;
          typedef typename ConvectiveFluxUpwind<UpwindInterpOpT,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);
          
        } else if (interpMethod == "CENTRAL") {
          cout << "SETTING UP CENTRAL CONVECTION INTERPOLANT IN Y DIRECTION "<< std::endl;
          typedef typename OperatorTypeBuilder<Interpolant,YVolField,typename FaceTypes<FieldT>::YFace>::type VelInterpOpT;
          typedef typename OperatorTypeBuilder<Interpolant,FieldT,typename FaceTypes<FieldT>::YFace>::type InterpPhiVol2PhiFY;
          typedef typename ConvectiveFlux<InterpPhiVol2PhiFY,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);                    
        }
        
      } else if( dir=="Z") {        
        cout << "SETTING UP CONVECTIVE FLUX EXPRESSION IN Z DIRECTION "<< std::endl;
        
        if (interpMethod=="UPWIND") {         
          cout << "SETTING UP UPWIND CONVECTION INTERPOLANT IN Z DIRECTION "<< std::endl;
          typedef UpwindInterpolant< FieldT, typename FaceTypes<FieldT>::ZFace > UpwindInterpOpT;
          typedef typename OperatorTypeBuilder<Interpolant,ZVolField,typename FaceTypes<FieldT>::ZFace>::type VelInterpOpT;
          typedef typename ConvectiveFluxUpwind<UpwindInterpOpT,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);
          
        } else if (interpMethod=="CENTRAL") {
          cout << "SETTING UP CENTRAL CONVECTION INTERPOLANT IN Z DIRECTION "<< std::endl;
          typedef typename OperatorTypeBuilder<Interpolant,ZVolField,typename FaceTypes<FieldT>::ZFace>::type VelInterpOpT;
          typedef typename OperatorTypeBuilder<Interpolant,FieldT,typename FaceTypes<FieldT>::ZFace>::type InterpPhiVol2PhiFZ;          
          typedef typename ConvectiveFlux<InterpPhiVol2PhiFZ,VelInterpOpT>::Builder theConvectiveFlux;
          builder = scinew theConvectiveFlux(phiTag, advVelocityTag);                    
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
    
    info[ fs ] = convFluxTag;
  }
  
  //------------------------------------------------------------------

  template< typename FieldT >
  ScalarTransportEquation<FieldT>::
  ScalarTransportEquation( const std::string phiName,
                          const Expr::ExpressionID rhsID )
  : Expr::TransportEquation( phiName, rhsID )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  ScalarTransportEquation<FieldT>::~ScalarTransportEquation()
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalarTransportEquation<FieldT>::setup_boundary_conditions( Expr::ExpressionFactory& exprFactory )
  {}
  
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
    for( Uintah::ProblemSpecP sourceTermParams=params->findBlock("SourceTerm");
        sourceTermParams != 0;
        sourceTermParams=sourceTermParams->findNextBlock("SourceTerm") ){
      
      //      setup_source_term_expression( sourceTermParams, phiName, factory, info );
      
      std::ostringstream msg;
      msg << "Source term support is not yet implemented on scalar transport equation." << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    return factory.register_expression( Expr::Tag( phiName+"_rhs", Expr::STATE_NONE ),
                                        scinew typename ScalarRHS<FieldT>::Builder(info) );
  }
  
  //------------------------------------------------------------------
  
} // namespace Wasatch
#endif // Wasatch_ScalarTransportEquation_h
