//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>


//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/BasicExprBuilder.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/MMS/TaylorVortex.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/StringNames.h>

#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationBulkDiffusionCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationMonosurfaceCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationClassicNucleationCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationRCritical.h>


//-- ExprLib includes --//
#include <expression/ExprLib.h>


#include <string>
#define PI 3.1415926535897932384626433832795

using std::endl;

namespace Wasatch{

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_basic_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );

    Expr::ExpressionBuilder* builder = NULL;

    std::string exprType;
    Uintah::ProblemSpecP valParams = params->get("value",exprType);
    if( params->findBlock("Constant") ){
      double val;  params->get("Constant",val);
      typedef typename Expr::ConstantExpr<FieldT>::Builder Builder;
      builder = scinew Builder( tag, val );
    }
    else if( params->findBlock("LinearFunction") ){
      double slope, intercept;
      Uintah::ProblemSpecP valParams = params->findBlock("LinearFunction");
      valParams->getAttribute("slope",slope);
      valParams->getAttribute("intercept",intercept);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::LinearFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, slope, intercept );
    }

    else if ( params->findBlock("SineFunction") ) {
      double amplitude, frequency, offset;
      Uintah::ProblemSpecP valParams = params->findBlock("SineFunction");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("frequency",frequency);
      valParams->getAttribute("offset",offset);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::SinFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, amplitude, frequency, offset);
    }

    else if ( params->findBlock("ParabolicFunction") ) {
      double a, b, c;
      Uintah::ProblemSpecP valParams = params->findBlock("ParabolicFunction");
      valParams->getAttribute("a",a);
      valParams->getAttribute("b",b);
      valParams->getAttribute("c",c);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::ParabolicFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, a, b, c);
    }

    else if ( params->findBlock("GaussianFunction") ) {
      double amplitude, deviation, mean, baseline;
      Uintah::ProblemSpecP valParams = params->findBlock("GaussianFunction");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("deviation",deviation);
      valParams->getAttribute("mean",mean);
      valParams->getAttribute("baseline",baseline);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::GaussianFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, amplitude, deviation, mean, baseline);
    }

    else if ( params->findBlock("DoubleTanhFunction") ) {
      double amplitude, width, midpointUp, midpointDown;
      Uintah::ProblemSpecP valParams = params->findBlock("DoubleTanhFunction");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("width",width);
      valParams->getAttribute("midpointUp",midpointUp);
      valParams->getAttribute("midpointDown",midpointDown);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::DoubleTanhFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, midpointUp, midpointDown, width, amplitude);
    }

    else if ( params->findBlock("SineTime") ) {
      const Expr::Tag timeVarTag( "time", Expr::STATE_NONE );
      typedef typename SineTime<FieldT>::Builder Builder;
      builder = scinew Builder( tag, timeVarTag );
    }
    
    else if ( params->findBlock("ExprAlgebra") ) {
      std::string algebraicOperation;
      Uintah::ProblemSpecP valParams = params->findBlock("ExprAlgebra");
      const Expr::Tag field1Tag = parse_nametag( valParams->findBlock("Field1")->findBlock("NameTag") );
      const Expr::Tag field2Tag = parse_nametag( valParams->findBlock("Field2")->findBlock("NameTag") );
      valParams->getAttribute("algebraicOperation",algebraicOperation);
      typedef typename ExprAlgebra<FieldT>::Builder Builder;
      builder = scinew Builder( tag, field1Tag, field2Tag, algebraicOperation );      
    }    

    return builder;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_taylor_vortex_mms_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );

    const StringNames& sName = StringNames::self();

    Expr::ExpressionBuilder* builder = NULL;

    std::string exprType;
    Uintah::ProblemSpecP valParams = params->get("value",exprType);

    if( params->findBlock("VelocityX") ){
      double amplitude,viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("VelocityX");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      const Expr::Tag timeVarTag( sName.time, Expr::STATE_NONE );
      typedef typename VelocityX<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, timeVarTag, amplitude, viscosity );
    }

    else if( params->findBlock("VelocityY") ){
      double amplitude, viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("VelocityY");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      const Expr::Tag timeVarTag( sName.time, Expr::STATE_NONE );
      typedef typename VelocityY<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, timeVarTag, amplitude, viscosity );
    }

    else if( params->findBlock("GradPX") ){
      double amplitude, viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("GradPX");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      const Expr::Tag timeVarTag( sName.time, Expr::STATE_NONE );
      typedef typename GradPX<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, timeVarTag, amplitude, viscosity );
    }

    else if( params->findBlock("GradPY") ){
      double amplitude, viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("GradPY");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      const Expr::Tag timeVarTag( sName.time, Expr::STATE_NONE );
      typedef typename GradPY<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, timeVarTag, amplitude, viscosity );
    }

    else if( params->findBlock("TGVel3D") ){
      double angle;
      std::string velComponent;
      Uintah::ProblemSpecP valParams = params->findBlock("TGVel3D");
      valParams->getAttribute("angle",angle);
      valParams->get("VelComponent", velComponent);
      const Expr::Tag XCoordinate = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag YCoordinate = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      const Expr::Tag ZCoordinate = parse_nametag( valParams->findBlock("ZCoordinate")->findBlock("NameTag") );
      typedef typename TaylorGreenVel3D<FieldT>::Builder Builder;
      // shuffle the x, y, and z coordinates based on the velocity component
      if (velComponent=="X") {
        angle += 2*PI/3.0;
        builder = scinew Builder( tag, XCoordinate, YCoordinate, ZCoordinate, angle );
      } else if (velComponent=="Y") {
        angle -= 2*PI/3.0;
        builder = scinew Builder( tag, YCoordinate, XCoordinate, ZCoordinate, angle );
      } else if (velComponent=="Z") {
        builder = scinew Builder( tag, ZCoordinate, XCoordinate, YCoordinate, angle );
      }
    }

    return builder;
  }
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_physics_coefficient_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    Expr::ExpressionBuilder* builder = NULL;
    std::string exprType;
    Uintah::ProblemSpecP valParams = params->get("value",exprType);
      
    if (params->findBlock("PrecipitationBulkDiffusionCoefficient") ) {      
      double coef, MolecularVolume, DiffusionCoefficient;     
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationBulkDiffusionCoefficient");
      coefParams -> getAttribute("Molec_Vol",MolecularVolume);
      coefParams -> getAttribute("Diff_Coef",DiffusionCoefficient);
      coef = MolecularVolume*DiffusionCoefficient;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag eqTag  = parse_nametag( coefParams->findBlock("EquilibriumConcentration")->findBlock("NameTag") );
      typedef typename PrecipitationBulkDiffusionCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, eqTag, coef); 
    }

    else if (params->findBlock("PrecipitationMonosurfaceCoefficient") ) {      
      double coef, expcoef, MolecularDiameter, DiffusionCoefficient, SurfaceEnergy , T;
      const double K_B = 1.3806488e-23;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationMonosurfaceCoefficient");
      coefParams -> getAttribute("Molec_D",MolecularDiameter);
      coefParams -> getAttribute("Diff_Coef",DiffusionCoefficient);
      coefParams -> getAttribute("Surf_Eng", SurfaceEnergy);
      coefParams -> getAttribute("Temperature",T);
      coef = DiffusionCoefficient * PI / MolecularDiameter / MolecularDiameter /MolecularDiameter;
      expcoef = - SurfaceEnergy * SurfaceEnergy * MolecularDiameter * MolecularDiameter * PI / K_B / K_B / T / T;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      typedef typename PrecipitationMonosurfaceCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, coef, expcoef); 
    } 

    else if (params->findBlock("PrecipitationClassicNucleationCoefficient") ) {
      double expcoef, SurfaceEnergy, MolecularVolume, T;
      const double K_B = 1.3806488e-23;
      const double N_A = 6.023e23;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationClassicNucleationCoefficient");
      coefParams -> getAttribute("Molec_Vol",MolecularVolume);
      coefParams -> getAttribute("Surf_Eng",SurfaceEnergy);
      coefParams -> getAttribute("Temperature",T);
      expcoef = -16 * PI / 3 * SurfaceEnergy * SurfaceEnergy * SurfaceEnergy / K_B / K_B / K_B / T / T / T * MolecularVolume * MolecularVolume / N_A / N_A;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") ); 
      typedef typename PrecipitationClassicNucleationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, expcoef);
    }

    else if (params->findBlock("PrecipitationSimpleRStarValue") ) {
      double RKnot, coef, CFCoef;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationSimpleRStarValue");
      coefParams -> getAttribute("R0", RKnot);
      coefParams -> getAttribute("Conversion_Fac", CFCoef);
      coef = RKnot*CFCoef;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") ); 
      typedef typename PrecipitationRCritical<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, coef);
    }

    else if (params->findBlock("PrecipitationClassicRStarValue") ) {
      double SurfaceEnergy, MolecularVolume, T, CFCoef, coef;
      const double R = 8.314;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationClassicRStarValue");
      coefParams -> getAttribute("Surf_Eng", SurfaceEnergy);
      coefParams -> getAttribute("Conversion_Fac", CFCoef);
      coefParams -> getAttribute("Temperature", T);
      coefParams -> getAttribute("Molec_Vol",MolecularVolume);
      coef = 2.0*SurfaceEnergy*MolecularVolume/R/T*CFCoef;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") ); 
      typedef typename PrecipitationRCritical<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, coef);
      //Note: both RStars are same basic form, same builder, but different coefficient parse
    }
    return builder;
  }

  //------------------------------------------------------------------

  void
  create_expressions_from_input( Uintah::ProblemSpecP parser,
                                 GraphCategories& gc )
  {
    Expr::ExpressionBuilder* builder = NULL;

    for( Uintah::ProblemSpecP exprParams = parser->findBlock("BasicExpression");
         exprParams != 0;
         exprParams = exprParams->findNextBlock("BasicExpression") ){

      std::string fieldType, taskListName;
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);

      switch( get_field_type(fieldType) ){
      case SVOL : builder = build_basic_expr< SVolField >( exprParams );  break;
      case XVOL : builder = build_basic_expr< XVolField >( exprParams );  break;
      case YVOL : builder = build_basic_expr< YVolField >( exprParams );  break;
      case ZVOL : builder = build_basic_expr< ZVolField >( exprParams );  break;
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }


    for( Uintah::ProblemSpecP exprParams = parser->findBlock("TaylorVortexMMS");
         exprParams != 0;
         exprParams = exprParams->findNextBlock("TaylorVortexMMS") ){

      std::string fieldType, taskListName;
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);

//       std::cout << "Creating TaylorVortexMMS for variable '" << tag.name()
//                 << "' with state " << tag.context()
//                 << " on task list '" << taskListName << "'"
//                 << std::endl;

      switch( get_field_type(fieldType) ){
      case SVOL : builder = build_taylor_vortex_mms_expr< SVolField >( exprParams );  break;
      case XVOL : builder = build_taylor_vortex_mms_expr< XVolField >( exprParams );  break;
      case YVOL : builder = build_taylor_vortex_mms_expr< YVolField >( exprParams );  break;
      case ZVOL : builder = build_taylor_vortex_mms_expr< ZVolField >( exprParams );  break;
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }
    
    
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("PhysicsCoefficient");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("PhysicsCoefficient") ){
      
      std::string fieldType, taskListName;
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);
      
      switch( get_field_type(fieldType) ){
        case SVOL : builder = build_physics_coefficient_expr< SVolField >( exprParams );  break;
        case XVOL : builder = build_physics_coefficient_expr< XVolField >( exprParams );  break;
        case YVOL : builder = build_physics_coefficient_expr< YVolField >( exprParams );  break;
        case ZVOL : builder = build_physics_coefficient_expr< ZVolField >( exprParams );  break;
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << fieldType << "'" << endl
          << __FILE__ << " : " << __LINE__ << endl;
      }
      
      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << endl
        << __FILE__ << " : " << __LINE__ << endl;
      }
      
      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }
    
  }

  //------------------------------------------------------------------

}
