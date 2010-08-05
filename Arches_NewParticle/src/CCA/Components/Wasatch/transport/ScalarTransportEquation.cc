//-- Uintah Includes --//
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/transport/ScalarTransportEquation.h>

#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

#include <CCA/Components/Wasatch/ParseTools.h>

#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>


//-- ExprLib Includes --//
#include <expression/ExprLib.h>


namespace Wasatch{

  typedef OpTypes<ScalarVolField> MyOpTypes;

  void
  setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                   const std::string& phiName,
                                   Expr::ExpressionFactory& factory,
                                   ScalarRHS::FieldTagInfo& info )
  {
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
        typedef DiffusiveFlux<MyOpTypes::GradX>::Builder Flux;
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = new Flux( phiTag, coef );
        }
        else if( diffFluxParams->findBlock("DiffusiveCoefficient") ){
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusiveCoefficient")->findBlock("NameTag") );
          builder = new Flux( phiTag, coef );
        }
      }
      else if( dir=="Y" ){
        typedef DiffusiveFlux<MyOpTypes::GradY>::Builder Flux;
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = new Flux( phiTag, coef );
        }
        else if( diffFluxParams->findBlock("DiffusiveCoefficient") ){
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusiveCoefficient")->findBlock("NameTag") );
          builder = new Flux( phiTag, coef );
        }
      }
      else if( dir=="Z") {
        typedef DiffusiveFlux<MyOpTypes::GradZ>::Builder Flux;
        if( diffFluxParams->findBlock("ConstantDiffusivity") ){
          double coef;
          diffFluxParams->get("ConstantDiffusivity",coef);
          builder = new Flux( phiTag, coef );
        }
        else if( diffFluxParams->findBlock("DiffusiveCoefficient") ){
          const Expr::Tag coef = parse_nametag( diffFluxParams->findBlock("DiffusiveCoefficient")->findBlock("NameTag") );
          builder = new Flux( phiTag, coef );
        }
      }

      if( builder == NULL ){
        std::ostringstream msg;
        msg << "Could not build a diffusive flux expression for '" << phiName << "'" << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      factory.register_expression( diffFluxTag, builder );

    }

    ScalarRHS::FieldSelector fs;
    if     ( dir=="X" ) fs=ScalarRHS::DIFFUSIVE_FLUX_X;
    else if( dir=="Y" ) fs=ScalarRHS::DIFFUSIVE_FLUX_Y;
    else if( dir=="Z" ) fs=ScalarRHS::DIFFUSIVE_FLUX_Z;
    
    info[ fs ] = diffFluxTag;
  }

  //------------------------------------------------------------------

  std::string
  get_phi_name( Uintah::ProblemSpecP params )
  {
    std::string phiName;
    params->get("SolutionVariable",phiName);
    return phiName;
  }

  //------------------------------------------------------------------

  Expr::ExpressionID
  get_rhs_expr_id( Expr::ExpressionFactory& factory,
                   Uintah::ProblemSpecP params )
  {
    ScalarRHS::FieldTagInfo info;

    const std::string phiName = get_phi_name( params );

    //_________________
    // Diffusive Fluxes
    for( Uintah::ProblemSpecP diffFluxParams=params->findBlock("DiffusiveFluxExpression");
         diffFluxParams != 0;
         diffFluxParams=params->findNextBlock("DiffusiveFluxExpression") ){

      setup_diffusive_flux_expression( diffFluxParams, phiName, factory, info );

    }

    //__________________
    // Convective Fluxes
    for( Uintah::ProblemSpecP convFluxParams=params->findBlock("ConvectiveFluxExpression");
         convFluxParams != 0;
         convFluxParams=params->findNextBlock("ConvectiveFluxExpression") ){

//       setup_convective_flux_expression( convFluxParams, phiName, factory, info );

    }

    //_____________
    // Source Terms
    for( Uintah::ProblemSpecP sourceTermParams=params->findBlock("SourceTerm");
         sourceTermParams != 0;
         sourceTermParams=params->findNextBlock("SourceTerm") ){

      //      setup_source_term_expression( sourceTermParams, phiName, factory, info );

    }


    return factory.register_expression( Expr::Tag( phiName+"_rhs", Expr::STATE_NONE ),
                                        new ScalarRHS::Builder(info) );
  }

  //------------------------------------------------------------------

  //==================================================================

  //------------------------------------------------------------------

  ScalarTransportEquation::
  ScalarTransportEquation( Expr::ExpressionFactory& solnExprFactory,
                           Uintah::ProblemSpecP params )
    : Expr::TransportEquation( get_phi_name( params ),
                               get_rhs_expr_id( solnExprFactory, params ) )
  {}

  //------------------------------------------------------------------

  ScalarTransportEquation::~ScalarTransportEquation()
  {}

  //------------------------------------------------------------------

  void
  ScalarTransportEquation::
  setup_boundary_conditions( Expr::ExpressionFactory& exprFactory )
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ScalarTransportEquation::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_registry().get_id( Expr::Tag( this->solution_variable_name(),
                                                       Expr::STATE_N ) );
  }

  //------------------------------------------------------------------

} // namespace Wasatch
