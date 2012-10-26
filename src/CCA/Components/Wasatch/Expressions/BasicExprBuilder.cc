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
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/WallDistance.h>

#include <CCA/Components/Wasatch/StringNames.h>

#include <CCA/Components/Wasatch/Expressions/PBE/BrownianAggregationCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/TurbulentAggregationCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/MultiEnvMixingModel.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationBulkDiffusionCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationMonosurfaceCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationClassicNucleationCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationRCritical.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitationSource.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/ParticleVolumeFraction.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/PrecipitateEffectiveViscosity.h>

#include <CCA/Components/Wasatch/Expressions/VelocityMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/Vorticity.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>

// BC Expressions Includes
#include "BoundaryConditions/ConstantBC.h"
#include "BoundaryConditions/LinearBC.h"
#include "BoundaryConditions/ParabolicBC.h"
#include "BoundaryConditions/BoundaryConditionBase.h"

//-- ExprLib includes --//
#include <expression/ExprLib.h>


#include <string>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

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
      

       Expr::TagList srcFieldTagList;
       
       for( Uintah::ProblemSpecP exprParams = valParams->findBlock("NameTag");
           exprParams != 0;
           exprParams = exprParams->findNextBlock("NameTag") )
       {
          srcFieldTagList.push_back( parse_nametag( exprParams ) );
       }

       valParams->getAttribute("algebraicOperation",algebraicOperation);

      // for now, only support parsing for fields of same type.  In the future,
      // we could extend parsing support for differing source field types.
      typedef ExprAlgebra<FieldT> AlgExpr;
      typename AlgExpr::OperationType optype;
      if      (algebraicOperation == "SUM"       ) optype = AlgExpr::SUM;
      else if (algebraicOperation == "DIFFERENCE") optype = AlgExpr::DIFFERENCE;
      else if (algebraicOperation == "PRODUCT"   ) optype = AlgExpr::PRODUCT;
      else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: The operator " << algebraicOperation
        << " is not supported in ExprAlgebra." << std::endl;
        throw std::invalid_argument( msg.str() );
      }
      builder = scinew typename AlgExpr::Builder( tag, srcFieldTagList, optype );
    }

    else if ( params->findBlock("Cylinder") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("Cylinder");
      double radius, insideValue, outsideValue;
      std::vector<double> origin;
      valParams->getAttribute("radius",radius);
      valParams->getAttribute("insideValue",insideValue);
      valParams->getAttribute("outsideValue",outsideValue);
      valParams->get("Origin",origin);
      const Expr::Tag field1Tag = parse_nametag( valParams->findBlock("Coordinate1")->findBlock("NameTag") );
      const Expr::Tag field2Tag = parse_nametag( valParams->findBlock("Coordinate2")->findBlock("NameTag") );
      typedef typename CylinderPatch<FieldT>::Builder Builder;
      builder = scinew Builder( tag, field1Tag, field2Tag, origin, insideValue, outsideValue, radius );
    }
    
    else if( params->findBlock("WallDistanceFunction") ){
      Uintah::ProblemSpecP valParams = params->findBlock("WallDistanceFunction");
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename WallDistance::Builder Builder;
      builder = scinew Builder( tag, indepVarTag );
    }
    
    else if( params->findBlock("ReadFromFile") ){            
      std::string fieldType;
      params->getAttribute("type",fieldType);
      
      Uintah::ProblemSpecP valParams = params->findBlock("ReadFromFile");
      std::string fileName;      
      valParams->get("FileName",fileName);      
      
      const Expr::Tag xTag("X" + fieldType, Expr::STATE_NONE);
      const Expr::Tag yTag("Y" + fieldType, Expr::STATE_NONE);
      const Expr::Tag zTag("Z" + fieldType, Expr::STATE_NONE);      
      
      typedef typename ReadFromFileExpression<FieldT>::Builder Builder;      
      builder = scinew Builder( tag, xTag, yTag, zTag, fileName );      
    }
    
    else if ( params->findBlock("StepFunction") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("StepFunction");
      double transitionPoint, lowValue, highValue;
      valParams->getAttribute("transitionPoint",transitionPoint);
      valParams->getAttribute("lowValue",lowValue);
      valParams->getAttribute("highValue",highValue);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename StepFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, transitionPoint, lowValue, highValue );
    }
    
    else if ( params->findBlock("PlusProfile") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("PlusProfile");
      double xStart, yStart, xWidth, yWidth, lowValue, highValue;
      valParams->getAttribute("xStart",xStart);
      valParams->getAttribute("yStart",yStart);
      valParams->getAttribute("xWidth",xWidth);
      valParams->getAttribute("xWidth",yWidth);
      valParams->getAttribute("lowValue",lowValue);
      valParams->getAttribute("highValue",highValue);
      const Expr::Tag xTag = parse_nametag( valParams->findBlock("Coordinate1")->findBlock("NameTag") );
      const Expr::Tag yTag = parse_nametag( valParams->findBlock("Coordinate2")->findBlock("NameTag") );
      typedef typename PlusProfile<FieldT>::Builder Builder;
      builder = scinew Builder( tag, xTag, yTag, xStart, yStart, xWidth, yWidth, lowValue, highValue );
    }
    
    else if ( params->findBlock("ExponentialVortex") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("ExponentialVortex");
      double x0, y0, G, R, U, V;
      std::string velocityComponent;
      
      valParams->getAttribute("x0",x0);
      valParams->getAttribute("y0",y0);
      valParams->getAttribute("G",G);
      valParams->getAttribute("R",R);
      valParams->getAttribute("U",U);
      valParams->getAttribute("V",V);      
      valParams->getAttribute("velocityComponent",velocityComponent);
      
      typedef ExponentialVortex<FieldT> ExpVortex;
      typename ExpVortex::VelocityComponent velComponent;            
      if      (velocityComponent == "X1") velComponent = ExpVortex::X1;
      else if (velocityComponent == "X2") velComponent = ExpVortex::X2;
      else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: The velocity component " << velocityComponent
        << " is not supported in the ExponentialVortex expression." << std::endl;
        throw std::invalid_argument( msg.str() );
      }      
      
      const Expr::Tag xTag = parse_nametag( valParams->findBlock("Coordinate1")->findBlock("NameTag") );
      const Expr::Tag yTag = parse_nametag( valParams->findBlock("Coordinate2")->findBlock("NameTag") );

      builder = scinew typename ExpVortex::Builder( tag, xTag, yTag, x0, y0, G, R, U, V, velComponent );
    }    
    
    else if ( params->findBlock("LambsDipole") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("LambsDipole");
      double x0, y0, G, R, U;
      std::string velocityComponent;
      
      valParams->getAttribute("x0",x0);
      valParams->getAttribute("y0",y0);
      valParams->getAttribute("G",G);
      valParams->getAttribute("R",R);
      valParams->getAttribute("U",U);
      valParams->getAttribute("velocityComponent",velocityComponent);
      
      typedef LambsDipole<FieldT> Dipole;
      typename Dipole::VelocityComponent velComponent;            
      if      (velocityComponent == "X1") velComponent = Dipole::X1;
      else if (velocityComponent == "X2") velComponent = Dipole::X2;
      else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: The velocity component " << velocityComponent
        << " is not supported in the ExponentialVortex expression." << std::endl;
        throw std::invalid_argument( msg.str() );
      }      
      
      const Expr::Tag xTag = parse_nametag( valParams->findBlock("Coordinate1")->findBlock("NameTag") );
      const Expr::Tag yTag = parse_nametag( valParams->findBlock("Coordinate2")->findBlock("NameTag") );
      builder = scinew typename Dipole::Builder( tag, xTag, yTag, x0, y0, G, R, U, velComponent );
    }        
    
    else if( params->findBlock("RandomField") ){
      Uintah::ProblemSpecP valParams = params->findBlock("RandomField");
      double low, high, seed;
      valParams->getAttribute("low",low);
      valParams->getAttribute("high",high);
      valParams->getAttribute("seed",seed);
      typedef typename RandomField<FieldT>::Builder Builder;
      builder = scinew Builder( tag, low, high, seed );
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
  build_physics_coefficient_expr( Uintah::ProblemSpecP params , Uintah::ProblemSpecP wasatchParams )
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
    
    else if (params->findBlock("BrownianAggregationCoefficient") ) {
      const double K_B = 1.3806488e-23;
      double T, coef;
      Uintah::ProblemSpecP coefParams = params->findBlock("BrownianAggregationCoefficient");
      coefParams -> getAttribute("Temperature", T);
      double ConvFac = 1.0;
      if (coefParams->getAttribute("Conversion_Fac", ConvFac) )
        coefParams->getAttribute("Conversion_Fac", ConvFac);
      coef = 2.0 * K_B * T / 3.0 * ConvFac ;
      const Expr::Tag densityTag = parse_nametag( coefParams->findBlock("Density")->findBlock("NameTag") );
      typedef typename BrownianAggregationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, densityTag, coef);
    }
    
    else if (params->findBlock("TurbulentAggregationCoefficient") ) {
      Uintah::ProblemSpecP coefParams = params->findBlock("TurbulentAggregationCoefficient");  
      const Expr::Tag kinematicViscosityTag = parse_nametag( coefParams->findBlock("KinematicViscosity")->findBlock("NameTag") );
      const Expr::Tag energyDissipationTag = parse_nametag( coefParams->findBlock("EnergyDissipation")->findBlock("NameTag") );
      double coef;
      double ConvFac = 1.0;
      if (coefParams->getAttribute("Conversion_Fac", ConvFac) )
        coefParams->getAttribute("Conversion_Fac", ConvFac);
      coef = (4.0 / 3.0) * sqrt(3.0 * PI / 10.0) * ConvFac;
      typedef typename TurbulentAggregationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, kinematicViscosityTag, energyDissipationTag, coef);
    }
    
    else if (params->findBlock("PrecipitateEffectiveViscosity") ) {
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitateEffectiveViscosity");
      double corrFac, power, baseViscos, minStrain;
      const Expr::Tag volFracTag = parse_nametag( coefParams->findBlock("VolumeFraction")->findBlock("NameTag") );
      const Expr::Tag strainMagTag = parse_nametag( coefParams->findBlock("StrainMagnitude")->findBlock("NameTag") );
      coefParams -> getAttribute("Correction_Fac", corrFac);
      coefParams -> getAttribute("Power", power);
      coefParams -> getAttribute("BaseViscosity", baseViscos);
      coefParams -> getAttribute("MinStrain", minStrain);
      typedef typename PrecipitateEffectiveViscosity<FieldT>::Builder Builder;
      builder= scinew Builder(tag, volFracTag, strainMagTag, corrFac, baseViscos, power, minStrain);
    }
    
    else if (params->findBlock("ParticleVolumeFraction") ) {
      Uintah::ProblemSpecP coefParams = params->findBlock("ParticleVolumeFraction");
      double convFac;
      coefParams -> get("Conversion_Fac", convFac);
      std::string basePhiName;
      Expr::Tag momentTag;
      Expr::TagList zerothMomentTags;
      Expr::TagList firstMomentTags;
      //assumes all moment transport eqns are for particles
      for ( Uintah::ProblemSpecP momentParams=wasatchParams->findBlock("MomentTransportEquation");
           momentParams != 0;
           momentParams = momentParams->findNextBlock("MomentTransportEquation") ) {
        momentParams ->get("PopulationName",basePhiName);
        if (momentParams->findBlock("MultiEnvMixingModel") ) {
          momentTag = Expr::Tag("m_" + basePhiName + "_0_ave", Expr::STATE_NONE);
          zerothMomentTags.push_back(momentTag);
          momentTag = Expr::Tag("m_" + basePhiName + "_1_ave", Expr::STATE_NONE);
          firstMomentTags.push_back(momentTag);
        } else {
          momentTag = Expr::Tag("m_" + basePhiName + "_0", Expr::STATE_N);
          zerothMomentTags.push_back(momentTag);
          momentTag = Expr::Tag("m_" + basePhiName + "_1", Expr::STATE_N);
          firstMomentTags.push_back(momentTag);
        }
      }
      typedef typename ParticleVolumeFraction<FieldT>::Builder Builder;
      builder = scinew Builder(tag, zerothMomentTags, firstMomentTags, convFac);
    }

    else if (params->findBlock("MultiEnvMixingModel") ) {
      std::stringstream wID;
      std::string baseName;
      std::string stateType;
      Uintah::ProblemSpecP multiEnvParams = params->findBlock("MultiEnvMixingModel");
      Expr::TagList multiEnvWeightsTags;
      params->findBlock("NameTag")->getAttribute("name",baseName);
      params->findBlock("NameTag")->getAttribute("state",stateType);
      const int numEnv = 3;
      //create the expression for weights and derivatives if block found  
      for (int i=0; i<numEnv; i++) {
        wID.str(std::string());
        wID << i;
        if (stateType == "STATE_N") {
          multiEnvWeightsTags.push_back(Expr::Tag("w_" + baseName + "_" + wID.str(), Expr::STATE_N) );
          multiEnvWeightsTags.push_back(Expr::Tag("dwdt_" + baseName + "_" + wID.str(), Expr::STATE_N) );
        } else if (stateType == "STATE_NONE") {
          multiEnvWeightsTags.push_back(Expr::Tag("w_" + baseName + "_" + wID.str(), Expr::STATE_NONE) );
          multiEnvWeightsTags.push_back(Expr::Tag("dwdt_" + baseName + "_" + wID.str(), Expr::STATE_NONE) );
        }
      }
      
      const Expr::Tag mixFracTag = parse_nametag( multiEnvParams->findBlock("MixtureFraction")->findBlock("NameTag") );
      const Expr::Tag scalarVarTag = parse_nametag( multiEnvParams->findBlock("ScalarVariance")->findBlock("NameTag") );
      const Expr::Tag scalarDissTag = parse_nametag( multiEnvParams->findBlock("ScalarDissipation")->findBlock("NameTag") );
      builder = scinew typename MultiEnvMixingModel<FieldT>::Builder(multiEnvWeightsTags, mixFracTag, scalarVarTag, scalarDissTag);
    }
    
    else if (params->findBlock("PrecipitationSource") ) {
      //this loops over all possible non-convective/non-diffusive rhs terms and creates a taglist
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationSource");
      std::vector<double> Molec_Volumes;
      Expr::TagList sourceTagList;
      Expr::Tag sourceTag;
      Expr::Tag midEnvWeightTag; //tag for central weight
      double molecVol;
      std::string modelType;
      std::string basePhiName;

      const Expr::Tag etaScaleTag = parse_nametag( coefParams->findBlock("EtaScale")->findBlock("NameTag") );
      const Expr::Tag densityTag = parse_nametag( coefParams->findBlock("Density")->findBlock("NameTag") );
      
      if (coefParams->findBlock("MultiEnvWeight") ) {
        midEnvWeightTag = parse_nametag( coefParams->findBlock("MultiEnvWeight")->findBlock("NameTag") );
      }

      for ( Uintah::ProblemSpecP momentParams=wasatchParams->findBlock("MomentTransportEquation");
            momentParams != 0;
            momentParams = momentParams->findNextBlock("MomentTransportEquation") ) {
        momentParams->get("MolecVol", molecVol);
        momentParams->get("PopulationName", basePhiName);

        for (Uintah::ProblemSpecP growthParams=momentParams->findBlock("GrowthExpression");
             growthParams != 0;
             growthParams = growthParams->findNextBlock("GrowthExpression") ) {
          Molec_Volumes.push_back(molecVol);
          growthParams->get("GrowthModel", modelType);
          sourceTag = Expr::Tag( "m_" + basePhiName + "_3_growth_" + modelType, Expr::STATE_NONE);
          sourceTagList.push_back(sourceTag);
          if (growthParams->findBlock("OstwaldRipening") ){
            Molec_Volumes.push_back(molecVol);
            sourceTag = Expr::Tag( "m_" + basePhiName + "_3_Ostwald_Ripening", Expr::STATE_NONE);
            sourceTagList.push_back(sourceTag);
          }
        }
        for (Uintah::ProblemSpecP birthParams=momentParams->findBlock("BirthExpression");
             birthParams != 0;
             birthParams = birthParams->findNextBlock("BirthExpression") ) {
          Molec_Volumes.push_back(molecVol);
          birthParams->get("BirthModel", modelType);
          sourceTag = Expr::Tag("m_" + basePhiName + "_3_birth_" + modelType, Expr::STATE_NONE);
          sourceTagList.push_back(sourceTag);
        }
      }
      typedef typename PrecipitationSource<FieldT>::Builder Builder;
      builder = scinew Builder(tag, sourceTagList, etaScaleTag, densityTag, midEnvWeightTag, Molec_Volumes);
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
        zVelTag = parse_nametag( valParams->findBlock("ZVelocity")->findBlock("NameTag") );

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
      std::string destFieldType;
      valParams->getAttribute("type",srcFieldType);
      Expr::Tag srcTag = Expr::Tag();
      srcTag = parse_nametag( valParams->findBlock("NameTag") );

      switch( get_field_type(srcFieldType) ){
        case SVOL : {
          typedef typename InterpolateExpression<SVolField, FieldT>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        case XVOL : {
          typedef typename InterpolateExpression<XVolField, FieldT>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        case YVOL : {
          typedef typename InterpolateExpression<YVolField, FieldT>::Builder Builder;
          builder = scinew Builder(tag, srcTag);
          break;
        }
        case ZVOL : {
          typedef typename InterpolateExpression<ZVolField, FieldT>::Builder Builder;
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
  
  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_bc_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    
    Expr::ExpressionBuilder* builder = NULL;
    
    std::string exprType;
    Uintah::ProblemSpecP valParams = params->get("value",exprType);
    if( params->findBlock("Constant") ){
      double val;  params->get("Constant",val);
      typedef typename ConstantBC<FieldT>::Builder Builder;
      builder = scinew Builder( tag, val );
    }

    else if( params->findBlock("LinearFunction") ){
      double slope, intercept;
      Uintah::ProblemSpecP valParams = params->findBlock("LinearFunction");
      valParams->getAttribute("slope",slope);
      valParams->getAttribute("intercept",intercept);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename LinearBC<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, slope, intercept );
    }

    else if ( params->findBlock("ParabolicFunction") ) {
      double a=0.0, b=0.0, c=0.0, x0=0.0, f0=0.0, h=0.0;
      Uintah::ProblemSpecP valParams = params->findBlock("ParabolicFunction");
      
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      
      std::string parabolaType;
      valParams->getAttribute("type", parabolaType);

      if (parabolaType.compare("CENTERED") == 0) {
        valParams = valParams->findBlock("Centered");
        valParams->getAttribute("x0",x0);
        valParams->getAttribute("f0",f0);
        valParams->getAttribute("h",h);
        a = - f0/(h*h);
        b = 0.0;
        c = f0;
      } else if (parabolaType.compare("GENERAL") == 0) {
        valParams = valParams->findBlock("General");
        valParams->getAttribute("a",a);
        valParams->getAttribute("b",b);
        valParams->getAttribute("c",c);
      }
      
      typedef typename ParabolicBC<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, a, b, c, x0);
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
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("PrecipitationBasicExpression");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("PrecipitationBasicExpression") ){

      std::string fieldType, taskListName;
      exprParams->getAttribute("type",fieldType);
      exprParams->require("TaskList",taskListName);

      switch( get_field_type(fieldType) ){
        case SVOL : builder = build_physics_coefficient_expr< SVolField >( exprParams , parser);  break;
        case XVOL : builder = build_physics_coefficient_expr< XVolField >( exprParams , parser);  break;
        case YVOL : builder = build_physics_coefficient_expr< YVolField >( exprParams , parser);  break;
        case ZVOL : builder = build_physics_coefficient_expr< ZVolField >( exprParams , parser);  break;
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
        case XVOL : builder = build_post_processing_expr< XVolField >( exprParams );  break;
        case YVOL : builder = build_post_processing_expr< YVolField >( exprParams );  break;
        case ZVOL : builder = build_post_processing_expr< ZVolField >( exprParams );  break;        
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
    
    //___________________________________________________
    // parse and build boundary condition expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("BCExpression");
        exprParams != 0;
        exprParams = exprParams->findNextBlock("BCExpression") ){
      
      std::string fieldType;
      exprParams->getAttribute("type",fieldType);
      
      // get the list of tasks
      std::string taskNames;
      exprParams->require("TaskList", taskNames);
      std::stringstream ss(taskNames);
      std::istream_iterator<std::string> begin(ss);
      std::istream_iterator<std::string> end;
      std::vector<std::string> taskNamesList(begin,end);      
      std::vector<std::string>::iterator taskNameIter = taskNamesList.begin();
      
      // iterate through the list of tasks to which this expression is to be added
      while (taskNameIter != taskNamesList.end()) {
        std::string taskName = *taskNameIter;

        switch( get_field_type(fieldType) ){
          case SVOL : builder = build_bc_expr< SVolField >( exprParams );  break;
          case XVOL : builder = build_bc_expr< XVolField >( exprParams );  break;
          case YVOL : builder = build_bc_expr< YVolField >( exprParams );  break;
          case ZVOL : builder = build_bc_expr< ZVolField >( exprParams );  break;
          default:
            std::ostringstream msg;
            msg << "ERROR: unsupported field type '" << fieldType << "' while trying to register BC expression.." << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        Category cat = INITIALIZATION;
        if     ( taskName == "initialization"   )   cat = INITIALIZATION;
        else if( taskName == "advance_solution" )   cat = ADVANCE_SOLUTION;
        else{
          std::ostringstream msg;
          msg << "ERROR: unsupported task list '" << taskName << "' while parsing BCExpression." << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        GraphHelper* const graphHelper = gc[cat];
        graphHelper->exprFactory->register_expression( builder );
        
        ++taskNameIter;
      }
    }
  }

  //------------------------------------------------------------------

}
