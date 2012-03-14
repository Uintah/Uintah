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

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>


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

#include <CCA/Components/Wasatch/Expressions/VelocityMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/Vorticity.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>

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
      bool hasOstwaldRipening = false;
      if (coefParams->findBlock("OstwaldRipening") )
        hasOstwaldRipening = true;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag eqTag  = parse_nametag( coefParams->findBlock("EquilibriumConcentration")->findBlock("NameTag") );
      typedef typename PrecipitationBulkDiffusionCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, eqTag, coef, hasOstwaldRipening);
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

  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_post_processing_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );

    Expr::ExpressionBuilder* builder = NULL;
    if( params->findBlock("VelocityMagnitude") ){
      Uintah::ProblemSpecP valParams = params->findBlock("VelocityMagnitude");
      Expr::Tag xVelTag = Expr::Tag();
      if (valParams->findBlock("XVelocity"))
        xVelTag = parse_nametag( valParams->findBlock("XVelocity")->findBlock("NameTag") );
      Expr::Tag yVelTag = Expr::Tag();
      if (valParams->findBlock("YVelocity"))
        yVelTag = parse_nametag( valParams->findBlock("YVelocity")->findBlock("NameTag") );
      Expr::Tag zVelTag = Expr::Tag();
      if (valParams->findBlock("ZVelocity"))
        Expr::Tag zVelTag = parse_nametag( valParams->findBlock("ZVelocity")->findBlock("NameTag") );
      typedef typename VelocityMagnitude<SVolField, XVolField, YVolField, ZVolField>::Builder Builder;
      builder = scinew Builder(tag, xVelTag, yVelTag, zVelTag);
    }

    else if( params->findBlock("Vorticity") ){
      Uintah::ProblemSpecP valParams = params->findBlock("Vorticity");
      std::string vorticityComponent;
      valParams->require("Component",vorticityComponent);

      Expr::Tag vel1Tag = Expr::Tag();
      if (valParams->findBlock("Vel1"))
        vel1Tag = parse_nametag( valParams->findBlock("Vel1")->findBlock("NameTag") );
      Expr::Tag vel2Tag = Expr::Tag();
      if (valParams->findBlock("Vel2"))
        vel2Tag = parse_nametag( valParams->findBlock("Vel2")->findBlock("NameTag") );
      if (vorticityComponent == "X") {
        typedef typename Vorticity<SVolField, ZVolField, YVolField>::Builder Builder;
        builder = scinew Builder(tag, vel1Tag, vel2Tag);
      } else if (vorticityComponent == "Y") {
        typedef typename Vorticity<SVolField, XVolField, ZVolField>::Builder Builder;
        builder = scinew Builder(tag, vel1Tag, vel2Tag);
      } else if (vorticityComponent == "Z") {
        typedef typename Vorticity<SVolField, YVolField, XVolField>::Builder Builder;
        builder = scinew Builder(tag, vel1Tag, vel2Tag);
      }
    }

    else if( params->findBlock("InterpolateExpression") ){
      Uintah::ProblemSpecP valParams = params->findBlock("InterpolateExpression");
      std::string srcFieldType;
      valParams->getAttribute("type",srcFieldType);
      Expr::Tag srcTag = Expr::Tag();
      srcTag = parse_nametag( valParams->findBlock("NameTag") );

      switch( get_field_type(srcFieldType) ){
        case SVOL : {
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << srcFieldType << "'" << ". Trying to interpolate SVOL to SVOL is redundant." << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );

        }
        case XVOL : {
          typedef typename InterpolateExpression<XVolField, SVolField>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        case YVOL : {
          typedef typename InterpolateExpression<YVolField, SVolField>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        case ZVOL : {
          typedef typename InterpolateExpression<ZVolField, SVolField>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << srcFieldType << "'" << "while parsing an InterpolateExpression." << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    return builder;
  }

  //------------------------------------------------------------------

  void
  create_expressions_from_input( Uintah::ProblemSpecP parser,
                                 GraphCategories& gc )
  {
    Expr::ExpressionBuilder* builder = NULL;

    //___________________________________
    // parse and build basid expressions
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
        msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }

    //________________________________________
    // parse and build Taylor-Green Vortex MMS
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
        msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }

    //___________________________________________________
    // parse and build physical coefficients expressions
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
          msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }

    //___________________________________________________
    // parse and build post-processing expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("PostProcessingExpression");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("PostProcessingExpression") ){

      std::string fieldType, taskListName;
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);

      switch( get_field_type(fieldType) ){
        case SVOL : builder = build_post_processing_expr< SVolField >( exprParams );  break;
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << fieldType << "'. Postprocessing expressions are setup with SVOLFields as destination fields only." << std::endl
              << "You were trying to register a postprocessing expression with a non cell centered destination field. Please revise you input file." << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "timestep_size"    )   cat = TIMESTEP_SELECTION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      GraphHelper* const graphHelper = gc[cat];
      graphHelper->exprFactory->register_expression( builder );
    }

  }

  //------------------------------------------------------------------

}
