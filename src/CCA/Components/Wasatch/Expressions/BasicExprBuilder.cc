/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/BasicExprBuilder.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/MMS/TaylorVortex.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Varden2DMMS.h>
#include <CCA/Components/Wasatch/Expressions/MMS/Functions.h>
#include <CCA/Components/Wasatch/Expressions/MMS/VardenMMS.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <CCA/Components/Wasatch/Expressions/TimeDerivative.h>
#include <CCA/Components/Wasatch/Expressions/GeometryBased.h>
#include <CCA/Components/Wasatch/Expressions/FanModel.h>
#include <CCA/Components/Wasatch/Expressions/TargetValueSource.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/WallDistance.h>
#include <CCA/Components/Wasatch/OldVariable.h>

#include <CCA/Components/Wasatch/TagNames.h>

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
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/CylindricalDiffusionCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/KineticGrowthCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/HomogeneousNucleationCoefficient.h>
#include <CCA/Components/Wasatch/Expressions/PBE/Precipitation/CriticalSurfaceEnergy.h>

#include <CCA/Components/Wasatch/Expressions/PostProcessing/Vorticity.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/KineticEnergy.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/VelocityMagnitude.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/InterpolateExpression.h>

#include <CCA/Components/Wasatch/Expressions/Particles/ParticleInitialization.h>


// BC Expressions Includes
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/TurbulentInletBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/VardenMMSBCs.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>


#include <string>
#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

using std::endl;

namespace WasatchCore{
  
  //------------------------------------------------------------------
  // Special parser for particle expressions
  
  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_basic_particle_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );    
    Expr::ExpressionBuilder* builder = nullptr;

    if( params->findBlock("Constant") ){
      double val;  params->get("Constant",val);
      typedef typename Expr::ConstantExpr<FieldT>::Builder Builder;
      builder = scinew Builder( tag, val );
    } else if (params->findBlock("ParticlePositionIC")) {
      Uintah::ProblemSpecP valParams = params->findBlock("ParticlePositionIC");
      // parse coordinate
      std::string coord;
      valParams->getAttribute("coordinate",coord);
      // check what type of bounds we are using: specified or patch based?
      std::string boundsType;
      valParams->getAttribute("bounds",boundsType);
      const bool usePatchBounds = (boundsType == "PATCHBASED");
      double lo = -99999.0, hi = 99999.0;

      if (valParams->findBlock("Bounds")) {
        valParams->findBlock("Bounds")->getAttribute("low", lo);
        valParams->findBlock("Bounds")->getAttribute("high", hi);
      }
      
      if (valParams->findBlock("Uniform")) {
        bool transverse = false;
        valParams->findBlock("Uniform")->getAttribute("transversedir", transverse);
        builder = scinew ParticleUniformIC::Builder( tag, lo, hi, transverse, coord, usePatchBounds );
      } else if (valParams->findBlock("Random")) {
        int seed = 0;
        valParams->findBlock("Random")->getAttribute("seed",seed);
        builder = scinew ParticleRandomIC::Builder( tag, coord, lo, hi, seed, usePatchBounds );
      } else if ( valParams->findBlock("Geometry") ) {
        std::vector <Uintah::GeometryPieceP > geomObjects;
        Uintah::ProblemSpecP geomBasedSpec = valParams->findBlock("Geometry");
        int seed = 1;
        geomBasedSpec->getAttribute("seed",seed);
        // parse all intrusions
  
        for( Uintah::ProblemSpecP intrusionParams = geomBasedSpec->findBlock("geom_object");
             intrusionParams != nullptr;
             intrusionParams = intrusionParams->findNextBlock("geom_object") ) {
          Uintah::GeometryPieceFactory::create(intrusionParams,geomObjects);
        }
        builder = scinew typename ParticleGeometryBased::Builder(tag, coord, seed, geomObjects);
      }

      
    } else if ( params->findBlock("RandomField") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("RandomField");
      double lo, hi, seed;
      valParams->getAttribute("low",lo);
      valParams->getAttribute("high",hi);
      valParams->getAttribute("seed",seed);
      const std::string coord="";
      const bool usePatchBounds = false;
      builder = scinew ParticleRandomIC::Builder( tag, coord, lo, hi, seed, usePatchBounds );
    } else if( params->findBlock("LinearFunction") ){
      double slope, intercept;
      Uintah::ProblemSpecP valParams = params->findBlock("LinearFunction");
      valParams->getAttribute("slope",slope);
      valParams->getAttribute("intercept",intercept);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::LinearFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, slope, intercept );
    } else if ( params->findBlock("SineFunction") ) {
      double amplitude, frequency, offset;
      Uintah::ProblemSpecP valParams = params->findBlock("SineFunction");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("frequency",frequency);
      valParams->getAttribute("offset",offset);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename Expr::SinFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, amplitude, frequency, offset);
    } else {
      std::ostringstream msg;
      msg << "ERROR: unsupported BasicExpression for Particles. Note that not all BasicExpressions are supported by particles. Please revise your input file." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    return builder;
  }

  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_basic_expr( Uintah::ProblemSpecP params, Uintah::ProblemSpecP uintahSpec  )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    
    const TagNames& tagNames = TagNames::self();

    Expr::ExpressionBuilder* builder = nullptr;
    
    //    std::string exprType;
    //    Uintah::ProblemSpecP valParams = params->get( "value", exprType );
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
      
      
      double a=0.0, b=0.0, c=0.0, x0=0.0, h=0.0;
      Uintah::ProblemSpecP valParams = params->findBlock("ParabolicFunction");
      
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      
      std::string parabolaType;
      valParams->getAttribute("type", parabolaType);
      
      if( parabolaType.compare("CENTERED") == 0 ){
        double f0 = 0.0;
        valParams = valParams->findBlock("Centered");
        valParams->getAttribute("x0",x0);
        valParams->getAttribute("f0",f0);
        valParams->getAttribute("h",h);
        a = - f0/(h*h);
        b = 0.0;
        c = f0;
      } else if( parabolaType.compare("GENERAL") == 0 ){
        valParams = valParams->findBlock("General");
        valParams->getAttribute("a",a);
        valParams->getAttribute("b",b);
        valParams->getAttribute("c",c);
      }
      
      typedef typename Expr::ParabolicFunction<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, a, b, c, x0 );
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
          exprParams != nullptr;
          exprParams = exprParams->findNextBlock("NameTag") ) {
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
    
    else if ( params->findBlock("RayleighTaylor") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("RayleighTaylor");
      double transitionPoint, lowValue, highValue, frequency, amplitude;
      std::string x1, x2;
      valParams->getAttribute("transitionPoint",transitionPoint);
      valParams->getAttribute("lowValue",lowValue);
      valParams->getAttribute("highValue",highValue);
      valParams->getAttribute("frequency",frequency);
      valParams->getAttribute("amplitude",amplitude);
      
      valParams->getAttribute("x1",x1);
      valParams->getAttribute("x2",x2);
      const Expr::Tag x1Tag(x1,Expr::STATE_NONE);
      const Expr::Tag x2Tag(x2,Expr::STATE_NONE);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename RayleighTaylor<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, x1Tag, x2Tag, transitionPoint, lowValue, highValue, frequency, amplitude );
    }

    else if ( params->findBlock("VarDen1DMMSMixFracSrc") ) {
      Uintah::ProblemSpecP valParams = params->findBlock("VarDen1DMMSMixFracSrc");
      double D, rho0, rho1;
      valParams->getAttribute("D",    D);
      valParams->getAttribute("rho0", rho0);
      valParams->getAttribute("rho1", rho1);
      const Expr::Tag xTag = parse_nametag( valParams->findBlock("Coordinate")->findBlock("NameTag") );
      typedef typename VarDen1DMMSMixFracSrc<SVolField>::Builder Builder;
      builder = scinew Builder( tag, xTag, tagNames.time, tagNames.dt, D, rho0, rho1, false );
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
    
    else if( params->findBlock("TimeDerivative") ){
      Uintah::ProblemSpecP valParams = params->findBlock("TimeDerivative");
      const Expr::Tag srcTag = parse_nametag( valParams->findBlock("NameTag") );
      // create an old variable
      OldVariable& oldVar = OldVariable::self();
      oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, srcTag);
      
      Expr::Tag srcOldTag = Expr::Tag( srcTag.name() + "_old", Expr::STATE_NONE );
      const TagNames& tagNames = TagNames::self();
      typedef typename TimeDerivative<FieldT>::Builder Builder;
      builder = scinew Builder( tag, srcTag, srcOldTag, tagNames.dt );
    }
    
    else if ( params->findBlock("BurnsChristonAbskg") ){
      typedef BurnsChristonAbskg<FieldT> BurnsChristonAbskgExpr;
      std::string fieldType;
      params->getAttribute("type",fieldType);
      const Expr::Tag xTag("X" + fieldType, Expr::STATE_NONE);
      const Expr::Tag yTag("Y" + fieldType, Expr::STATE_NONE);
      const Expr::Tag zTag("Z" + fieldType, Expr::STATE_NONE);
      
      builder = scinew typename BurnsChristonAbskgExpr::Builder( tag, xTag, yTag, zTag  );
    }

    else if ( params->findBlock("GeometryBased") ) {
      std::multimap <Uintah::GeometryPieceP, double > geomObjectsMap;
      double outsideValue = 1.0;
      Uintah::ProblemSpecP geomBasedSpec = params->findBlock("GeometryBased");
      geomBasedSpec->getAttribute("value", outsideValue);
      // parse all intrusions
      std::vector<Uintah::GeometryPieceP> geomObjects;
      
      for( Uintah::ProblemSpecP intrusionParams = geomBasedSpec->findBlock("Intrusion");
          intrusionParams != nullptr;
          intrusionParams = intrusionParams->findNextBlock("Intrusion") ) {
        Uintah::GeometryPieceFactory::create(intrusionParams->findBlock("geom_object"),geomObjects);
        double insideValue = 0.0;
        intrusionParams->getAttribute("value", insideValue);
        geomObjectsMap.insert(std::pair<Uintah::GeometryPieceP, double>(geomObjects.back(), insideValue)); // set a value inside the geometry object
      }
      builder = scinew typename GeometryBased<FieldT>::Builder(tag, geomObjectsMap, outsideValue);
    }
    
    else if ( params->findBlock("Bubbles") ) {
      std::vector<Uintah::GeometryPieceP> geomObjects;
      std::multimap <Uintah::GeometryPieceP, double > geomObjectsMap;

      Uintah::ProblemSpecP bubblesSpec = params->findBlock("Bubbles");
      std::vector<double> layout;
      bubblesSpec->get("Layout", layout);

      if (layout[0]<= 0 || layout[1]<= 0 || layout[2]<=0) {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: You cannot set 0 or negative values in the Bubbles layout. Please revise your inputfile." << std::endl;
        throw std::invalid_argument( msg.str() );
      }
      
      double insideValue;
      bubblesSpec->getAttribute("insidevalue", insideValue);
      
      double outsideValue;
      bubblesSpec->getAttribute("outsidevalue", outsideValue);
      
      double r;
      bubblesSpec->getAttribute("radius", r);
      
      // now get the domain's boundaries
      std::vector<double> low, high;
      uintahSpec->findBlock("Grid")->findBlock("Level")->findBlock("Box")->get("lower", low);
      uintahSpec->findBlock("Grid")->findBlock("Level")->findBlock("Box")->get("upper", high);
      
      const double wx = (high[0] - low[0])/layout[0];
      const double wy = (high[1] - low[1])/layout[1];
      const double wz = (high[2] - low[2])/layout[2];
      
      const double cx0 = low[0] + 0.5*wx;
      const double cy0 = low[1] + 0.5*wy;
      const double cz0 = low[2] + 0.5*wz;
      double cx,cy,cz;
      for (int nz = 0; nz < layout[2] ; ++nz) {
        cz = cz0 + nz * wz;
        for (int ny = 0; ny < layout[1]; ++ny) {
          cy = cy0 + ny * wy;
          for (int nx = 0; nx < layout[0]; ++nx) {
            cx = cx0 + nx * wx;
            geomObjects.push_back( scinew Uintah::SphereGeometryPiece(Uintah::Point(cx,cy,cz), r) );
          }
        }
      }

      Uintah::UnionGeometryPiece* bubblesUnited = scinew Uintah::UnionGeometryPiece(geomObjects);
      
      geomObjectsMap.insert(std::pair<Uintah::GeometryPieceP, double>(bubblesUnited, insideValue)); // set a value inside the geometry object
      
      builder = scinew typename GeometryBased<FieldT>::Builder(tag, geomObjectsMap, outsideValue);
    }


    return builder;
  }
  
  //------------------------------------------------------------------
  
  template<typename FieldT>
  Expr::ExpressionBuilder*
  build_taylor_vortex_mms_expr( Uintah::ProblemSpecP params )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    
    const TagNames& tagNames = TagNames::self();

    Expr::ExpressionBuilder* builder = nullptr;
    
    //std::string exprType;
    //bool valParams = params->get("value",exprType);
    
    if( params->findBlock("VelocityX") ){
      double amplitude,viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("VelocityX");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      typedef typename VelocityX<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, tagNames.time, amplitude, viscosity );
    }
    
    else if( params->findBlock("VelocityY") ){
      double amplitude, viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("VelocityY");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag1 = parse_nametag( valParams->findBlock("XCoordinate")->findBlock("NameTag") );
      const Expr::Tag indepVarTag2 = parse_nametag( valParams->findBlock("YCoordinate")->findBlock("NameTag") );
      typedef typename VelocityY<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag1, indepVarTag2, tagNames.time, amplitude, viscosity );
    }
    
    else if( params->findBlock("GradP") ){
      double amplitude, viscosity;
      Uintah::ProblemSpecP valParams = params->findBlock("GradP");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("viscosity",viscosity);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("Coordinate")->findBlock("NameTag") );
      typedef typename GradP<FieldT>::Builder Builder;
      builder = scinew Builder( tag, indepVarTag, tagNames.time, amplitude, viscosity );
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
  build_precipitation_expr( Uintah::ProblemSpecP params , Uintah::ProblemSpecP wasatchParams )
  {
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    Expr::ExpressionBuilder* builder = nullptr;

    const double kB = 1.3806488e-23;
    const double nA = 6.023e23;
    const double R = 8.314;

    if (params->findBlock("PrecipitationBulkDiffusionCoefficient") ) {
      double coef, molecularVolume, diffusionCoefficient, sMin;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationBulkDiffusionCoefficient");
      coefParams -> getAttribute("Molec_Vol",molecularVolume);
      coefParams -> getAttribute("Diff_Coef",diffusionCoefficient);
      sMin = 0.0;
      if (coefParams->getAttribute("S_Min", sMin) )
        coefParams->getAttribute("S_Min", sMin);
      coef = molecularVolume*diffusionCoefficient;
      Expr::Tag sBarTag;
      if (coefParams->findBlock("SBar") ) 
        sBarTag = parse_nametag( coefParams->findBlock("SBar")->findBlock("NameTag") );
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag eqTag  = parse_nametag( coefParams->findBlock("EquilibriumConcentration")->findBlock("NameTag") );
      typedef typename PrecipitationBulkDiffusionCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, eqTag, sBarTag, coef, sMin);
    }
    
    else if (params->findBlock("CylindricalDiffusionCoefficient") ) {
      double coef, molecularVolume, diffusionCoefficient, sMin;
      Uintah::ProblemSpecP coefParams = params->findBlock("CylindricalDiffusionCoefficient");
      coefParams -> getAttribute("Molec_Vol",molecularVolume);
      coefParams -> getAttribute("Diff_Coef",diffusionCoefficient);
      sMin = 0.0;
      if (coefParams->getAttribute("S_Min", sMin) )
        coefParams->getAttribute("S_Min", sMin);
      coef = molecularVolume*diffusionCoefficient* 7.0/6.0/log(0.5);
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag eqTag  = parse_nametag( coefParams->findBlock("EquilibriumConcentration")->findBlock("NameTag") );
      Expr::Tag sBarTag;
      if (coefParams->findBlock("SBar") ) 
        sBarTag = parse_nametag( coefParams->findBlock("SBar")->findBlock("NameTag") );
      typedef typename CylindricalDiffusionCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder( tag, saturationTag, eqTag, sBarTag, coef, sMin);
    }
    
    else if (params->findBlock("KineticGrowthCoefficient") ) {
      double coef, sMax, sMin;
      Uintah::ProblemSpecP coefParams = params->findBlock("KineticGrowthCoefficient");
      coefParams -> getAttribute("K_A",coef);
      sMin = 0.0;
      sMax = 1e10;
      if( coefParams->getAttribute("S_Max",sMax) )
        coefParams->getAttribute("S_Max",sMax);
      if( coefParams->getAttribute("S_Min",sMin) )
        coefParams->getAttribute("S_Min",sMin);
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      Expr::Tag sBarTag;
      if (coefParams->findBlock("SBar") ) 
        sBarTag = parse_nametag( coefParams->findBlock("SBar")->findBlock("NameTag") );
      typedef typename KineticGrowthCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder( tag, saturationTag, sBarTag, coef, sMax, sMin);
    }
    
    else if (params->findBlock("PrecipitationMonosurfaceCoefficient") ) {
      double coef, expcoef, molecularDiameter, diffusionCoefficient, surfaceEnergy , T;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationMonosurfaceCoefficient");
      coefParams -> getAttribute("Molec_D",molecularDiameter);
      coefParams -> getAttribute("Diff_Coef",diffusionCoefficient);
      coefParams -> getAttribute("Surf_Eng", surfaceEnergy);
      coefParams -> getAttribute("Temperature",T);
      coef = diffusionCoefficient * PI / molecularDiameter / molecularDiameter /molecularDiameter;
      expcoef = - surfaceEnergy * surfaceEnergy * molecularDiameter * molecularDiameter * PI / kB / kB / T / T;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      typedef typename PrecipitationMonosurfaceCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, coef, expcoef);
    }
    
    else if (params->findBlock("PrecipitationClassicNucleationCoefficient") ) {
      double expcoef, SurfaceEnergy, MolecularVolume, T;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationClassicNucleationCoefficient");
      coefParams -> getAttribute("Molec_Vol",MolecularVolume);
      coefParams -> getAttribute("Surf_Eng",SurfaceEnergy);
      coefParams -> getAttribute("Temperature",T);
      expcoef = -16 * PI / 3 * SurfaceEnergy * SurfaceEnergy * SurfaceEnergy / kB / kB / kB / T / T / T * MolecularVolume * MolecularVolume / nA / nA;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      typedef typename PrecipitationClassicNucleationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, expcoef);
    }
    
    else if (params->findBlock("HomogeneousNucleationCoefficient") ) {
      double molecularVolume, T, D, sRatio;
      double surfaceEnergy = 1.0;
      Uintah::ProblemSpecP coefParams = params->findBlock("HomogeneousNucleationCoefficient");
      coefParams -> getAttribute("Molar_Vol",molecularVolume);
      if (coefParams->getAttribute("Surf_Eng",surfaceEnergy) )
        coefParams -> getAttribute("Surf_Eng",surfaceEnergy);
      coefParams -> getAttribute("Temperature",T);
      coefParams -> getAttribute("Diff_Coef",D);
      coefParams -> getAttribute("S_Ratio", sRatio);
      molecularVolume = molecularVolume/6.02214129e23; //convert molar to molecular volume in this term
      Expr::Tag surfaceEngTag;
      if ( coefParams->findBlock("SurfaceEnergy") )
        surfaceEngTag = parse_nametag( coefParams->findBlock("SurfaceEnergy")->findBlock("NameTag") );
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag eqConcTag = parse_nametag( coefParams->findBlock("EquilibriumConcentration")->findBlock("NameTag") );
      typedef typename HomogeneousNucleationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, eqConcTag,  surfaceEngTag, molecularVolume, surfaceEnergy, T, D, sRatio);
    }
    
    else if (params->findBlock("PrecipitationSimpleRStarValue") ) {
      double rknot, coef, cfCoef;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationSimpleRStarValue");
      coefParams -> getAttribute("R0", rknot);
      coefParams -> getAttribute("Conversion_Fac", cfCoef);
      coef = rknot*cfCoef;
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      const Expr::Tag surfaceEngTag; //dummy tag since this uses same function as classic rStar
      typedef typename PrecipitationRCritical<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, surfaceEngTag, coef);
    }
    
    else if (params->findBlock("PrecipitationClassicRStarValue") ) {
      double surfaceEnergy, molecularVolume, T, cfCoef, coef;
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationClassicRStarValue");
      coefParams -> getAttribute("Surf_Eng", surfaceEnergy);
      coefParams -> getAttribute("Conversion_Fac", cfCoef);
      coefParams -> getAttribute("Temperature", T);
      coefParams -> getAttribute("Molec_Vol",molecularVolume);
      coef = 2.0*surfaceEnergy*molecularVolume/R/T*cfCoef;
      Expr::Tag surfaceEngTag;
      if (coefParams->findBlock("SurfaceEnergy") ) {
        surfaceEngTag = parse_nametag( coefParams->findBlock("SurfaceEnergy")->findBlock("NameTag") ) ;
        coef = 2.0*molecularVolume/R/T*cfCoef;  //calcualte coefficient without the surface energy
      }
      
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      typedef typename PrecipitationRCritical<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, surfaceEngTag, coef);
      //Note: both RStars are same basic form, same builder, but different coefficient parse
    }
    
    else if(params->findBlock("CriticalSurfaceEnergy") ) {
      double bulkSurfaceEnergy, T, molarVolume, coef, tolmanL;
      Uintah::ProblemSpecP coefParams = params->findBlock("CriticalSurfaceEnergy");
      coefParams -> getAttribute("Temperature",T);
      coefParams -> getAttribute("Bulk_Surf_Eng",bulkSurfaceEnergy);
      coefParams -> getAttribute("Molar_Vol",molarVolume);
      coefParams -> getWithDefault("TolmanLength",tolmanL,0.2);
      const Expr::Tag saturationTag = parse_nametag( coefParams->findBlock("Supersaturation")->findBlock("NameTag") );
      double r1 = pow(3.0*molarVolume/nA/4.0/PI,1.0/3.0); //convert molar vol to molec radius
      coef = 4.0 * tolmanL * R * T*bulkSurfaceEnergy* r1/molarVolume;
      typedef typename CriticalSurfaceEnergy<FieldT>::Builder Builder;
      builder = scinew Builder(tag, saturationTag, bulkSurfaceEnergy, coef);
    }
    
    else if (params->findBlock("BrownianAggregationCoefficient") ) {
      double T, coef;
      Uintah::ProblemSpecP coefParams = params->findBlock("BrownianAggregationCoefficient");
      coefParams -> getAttribute("Temperature", T);
      double ConvFac = 1.0;
      if (coefParams->getAttribute("Conversion_Fac", ConvFac) )
        coefParams->getAttribute("Conversion_Fac", ConvFac);
      coef = 2.0 * kB * T / 3.0 * ConvFac ;
      const Expr::Tag densityTag = parse_nametag( coefParams->findBlock("Density")->findBlock("NameTag") );
      typedef typename BrownianAggregationCoefficient<FieldT>::Builder Builder;
      builder = scinew Builder(tag, densityTag, coef);
    }
    
    else if (params->findBlock("TurbulentAggregationCoefficient") ) {
      Uintah::ProblemSpecP coefParams = params->findBlock("TurbulentAggregationCoefficient");
      const Expr::Tag kinematicViscosityTag = parse_nametag( coefParams->findBlock("KinematicViscosity")->findBlock("NameTag") );
      const Expr::Tag energyDissipationTag = parse_nametag( coefParams->findBlock("EnergyDissipation")->findBlock("NameTag") );
      double coef;
      double convFac = 1.0;
      if (coefParams->getAttribute("Conversion_Fac", convFac) )
        coefParams->getAttribute("Conversion_Fac", convFac);
      coef = (4.0 / 3.0) * sqrt(3.0 * PI / 10.0) * convFac;
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
           momentParams != nullptr;
           momentParams = momentParams->findNextBlock("MomentTransportEquation") ) {
        momentParams ->get("PopulationName",basePhiName);
        if (momentParams->findBlock("MultiEnvMixingModel") ) {
          momentTag = Expr::Tag("m_" + basePhiName + "_0_ave", Expr::STATE_NONE);
          zerothMomentTags.push_back(momentTag);
          momentTag = Expr::Tag("m_" + basePhiName + "_1_ave", Expr::STATE_NONE);
          firstMomentTags.push_back(momentTag);
        }
        else {
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
      std::string baseName, stateType;
      double maxDt;
      Uintah::ProblemSpecP multiEnvParams = params->findBlock("MultiEnvMixingModel");
      multiEnvParams -> getAttribute("MaxDt",maxDt);
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
      builder = scinew typename MultiEnvMixingModel<FieldT>::Builder(multiEnvWeightsTags, mixFracTag, scalarVarTag, scalarDissTag, maxDt);
    }
    
    else if (params->findBlock("PrecipitationSource") ) {
      //this loops over all possible non-convective/non-diffusive rhs terms and creates a taglist
      Uintah::ProblemSpecP coefParams = params->findBlock("PrecipitationSource");
      std::vector<double> molecVolumes;
      Expr::TagList sourceTagList;
      Expr::Tag sourceTag;
      Expr::Tag midEnvWeightTag; //tag for central weight
      double molecVol;
      std::string modelType, basePhiName;
      
      const Expr::Tag etaScaleTag = parse_nametag( coefParams->findBlock("EtaScale")->findBlock("NameTag") );
      const Expr::Tag densityTag = parse_nametag( coefParams->findBlock("Density")->findBlock("NameTag") );
      
      if (coefParams->findBlock("MultiEnvWeight") ) {
        midEnvWeightTag = parse_nametag( coefParams->findBlock("MultiEnvWeight")->findBlock("NameTag") );
      }
      
      for( Uintah::ProblemSpecP momentParams=wasatchParams->findBlock("MomentTransportEquation");
           momentParams != nullptr;
           momentParams = momentParams->findNextBlock("MomentTransportEquation") ){
        momentParams->get("MolecVol", molecVol);
        momentParams->get("PopulationName", basePhiName);
        
        for( Uintah::ProblemSpecP growthParams=momentParams->findBlock("GrowthExpression");
             growthParams != nullptr;
             growthParams = growthParams->findNextBlock("GrowthExpression") ){
          molecVolumes.push_back(molecVol);
          growthParams->get("GrowthModel", modelType);
          sourceTag = Expr::Tag( "m_" + basePhiName + "_3_growth_" + modelType, Expr::STATE_NONE);
          sourceTagList.push_back(sourceTag);
        }
        for( Uintah::ProblemSpecP birthParams=momentParams->findBlock("BirthExpression");
             birthParams != nullptr;
             birthParams = birthParams->findNextBlock("BirthExpression") ){
          molecVolumes.push_back(molecVol);
          birthParams->get("BirthModel", modelType);
          sourceTag = Expr::Tag("m_" + basePhiName + "_3_birth_" + modelType, Expr::STATE_NONE);
          sourceTagList.push_back(sourceTag);
        }
        for( Uintah::ProblemSpecP deathParams=momentParams->findBlock("Dissolution");
            deathParams != nullptr;
            deathParams = deathParams->findNextBlock("Dissolution") ){
          molecVolumes.push_back(molecVol);
          sourceTag = Expr::Tag( "m_" + basePhiName + "_3_death", Expr::STATE_NONE);
          sourceTagList.push_back(sourceTag);
        }
      }
      typedef typename PrecipitationSource<FieldT>::Builder Builder;
      builder = scinew Builder(tag, sourceTagList, etaScaleTag, densityTag, midEnvWeightTag, molecVolumes);
    }
    return builder;
  }

  //------------------------------------------------------------------
  
  template<typename FieldT>
  std::list<Expr::ExpressionBuilder*>
  build_bc_expr( Uintah::ProblemSpecP params, Uintah::ProblemSpecP WasatchSpec )
  {
    std::list<Expr::ExpressionBuilder*> builders;
    const Expr::Tag tag = parse_nametag( params->findBlock("NameTag") );
    const TagNames& tagNames = TagNames::self();
    
    if( params->findBlock("Constant") ){
      double val;  params->get("Constant",val);
      typedef typename ConstantBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder( tag, val ) );
    }
    
    else if( params->findBlock("LinearFunction") ){
      double slope, intercept;
      Uintah::ProblemSpecP valParams = params->findBlock("LinearFunction");
      valParams->getAttribute("slope",slope);
      valParams->getAttribute("intercept",intercept);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename LinearBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder( tag, indepVarTag, slope, intercept ) );
    }
    
    else if( params->findBlock("ParabolicFunction") ){
      double a=0.0, b=0.0, c=0.0, x0=0.0, h=0.0;
      Uintah::ProblemSpecP valParams = params->findBlock("ParabolicFunction");
      
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      
      std::string parabolaType;
      valParams->getAttribute("type", parabolaType);
      
      if( parabolaType.compare("CENTERED") == 0 ){
        double f0 = 0.0;
        valParams = valParams->findBlock("Centered");
        valParams->getAttribute("x0",x0);
        valParams->getAttribute("f0",f0);
        valParams->getAttribute("h",h);
        a = - f0/(h*h);
        b = 0.0;
        c = f0;
      } else if( parabolaType.compare("GENERAL") == 0 ){
        valParams = valParams->findBlock("General");
        valParams->getAttribute("a",a);
        valParams->getAttribute("b",b);
        valParams->getAttribute("c",c);
      }
      
      typedef typename ParabolicBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder( tag, indepVarTag, a, b, c, x0) );
    }
    
    else if( params->findBlock("PowerLawFunction") ) {
      double x0, phic, R, n;
      Uintah::ProblemSpecP valParams = params->findBlock("PowerLawFunction");
      valParams->getAttribute("x0",x0);
      valParams->getAttribute("PhiCenter",phic);
      valParams->getAttribute("HalfHeight",R);
      valParams->getAttribute("n",n);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename PowerLawBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder( tag, indepVarTag,x0, phic, R, n) );
    }

    else if ( params->findBlock("GaussianFunction") ) {
      double amplitude, deviation, mean, baseline;
      Uintah::ProblemSpecP valParams = params->findBlock("GaussianFunction");
      valParams->getAttribute("amplitude",amplitude);
      valParams->getAttribute("deviation",deviation);
      valParams->getAttribute("mean",mean);
      valParams->getAttribute("baseline",baseline);
      const Expr::Tag indepVarTag = parse_nametag( valParams->findBlock("NameTag") );
      typedef typename GaussianBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder( tag, indepVarTag, amplitude, deviation, mean, baseline) );
    }

    else if( params->findBlock("VarDenMMSVelocity") ){
      std::string side;
      Uintah::ProblemSpecP valParams = params->findBlock("VarDenMMSVelocity");
      valParams->getAttribute("side",side);
      
      typedef VarDen1DMMSVelocity<FieldT> VarDenMMSVExpr;
      SpatialOps::BCSide bcSide;
      if      (side == "PLUS"  ) bcSide = SpatialOps::PLUS_SIDE;
      else if (side == "MINUS" ) bcSide = SpatialOps::MINUS_SIDE;
      else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: The boundary side " << side
        << " is not supported in VarDen1DMMSVelocity expression." << std::endl;
        throw std::invalid_argument( msg.str() );
      }
      builders.push_back( scinew typename VarDenMMSVExpr::Builder( tag, tagNames.time, bcSide ) );
    }

    else if( params->findBlock("VarDenMMSMomentum") ){
      std::string side;
      double rho0=1.29985, rho1=0.081889;
      Uintah::ProblemSpecP valParams = params->findBlock("VarDenMMSMomentum");
      valParams->getAttribute("side",side);
      if (WasatchSpec->findBlock("TwoStreamMixing")){
      Uintah::ProblemSpecP densityParams =WasatchSpec->findBlock("TwoStreamMixing") ;
      densityParams->getAttribute("rho0",rho0);
      densityParams->getAttribute("rho1",rho1);
      }
      typedef VarDen1DMMSMomentum<FieldT> VarDenMMSMomExpr;
      SpatialOps::BCSide bcSide;
      if      (side == "PLUS"  ) bcSide = SpatialOps::PLUS_SIDE;
      else if (side == "MINUS" ) bcSide = SpatialOps::MINUS_SIDE;
      else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR: The boundary side " << side
        << " is not supported in VarDen1DMMSMomentum expression." << std::endl;
        throw std::invalid_argument( msg.str() );
      }
      builders.push_back( scinew typename VarDenMMSMomExpr::Builder( tag, tagNames.time, rho0, rho1, bcSide ) );
    }

    else if( params->findBlock("VarDenMMSMixtureFraction") ){
      Uintah::ProblemSpecP valParams = params->findBlock("VarDenMMSMixtureFraction");      
      typedef VarDen1DMMSMixtureFraction<FieldT> VarDen1DMMSMixtureFractionExpr;
      builders.push_back( scinew typename VarDen1DMMSMixtureFractionExpr::Builder( tag, tagNames.time ) );
    }

    else if( params->findBlock("VarDenMMSDensity") ){
      double rho0=1.29985, rho1=0.081889;
      Uintah::ProblemSpecP valParams = params->findBlock("VarDenMMSDensity");
      if (WasatchSpec->findBlock("TwoStreamMixing")){
      Uintah::ProblemSpecP densityParams = WasatchSpec->findBlock("TwoStreamMixing");
      densityParams->getAttribute("rho0",rho0);
      densityParams->getAttribute("rho1",rho1);
      }
      typedef VarDen1DMMSDensity<FieldT> VarDen1DMMSDensityExpr;
      builders.push_back( scinew typename VarDen1DMMSDensityExpr::Builder( tag, tagNames.time, rho0, rho1 ) );
    }

    else if( params->findBlock("VarDenMMSSolnVar") ){
      double rho0=1.29985, rho1=0.081889;
      Uintah::ProblemSpecP valParams = params->findBlock("VarDenMMSSolnVar");
      if (WasatchSpec->findBlock("TwoStreamMixing")){
      Uintah::ProblemSpecP densityParams = WasatchSpec->findBlock("TwoStreamMixing");
      densityParams->getAttribute("rho0",rho0);
      densityParams->getAttribute("rho1",rho1);
      }
      typedef VarDen1DMMSSolnVar<FieldT> VarDen1DMMSSolnVarExpr;
      builders.push_back( scinew typename VarDen1DMMSSolnVarExpr::Builder( tag, tagNames.time, rho0, rho1 ) );
    }
    
    else if( params->findBlock("TurbulentInlet") ){
      std::string inputFileName;
      std::string velDir;
      int period=1;
      double timePeriod;
      Uintah::ProblemSpecP valParams = params->findBlock("TurbulentInlet");
      valParams->get("InputFile",inputFileName);
      valParams->getAttribute("component",velDir);
      
      bool hasPeriod = valParams->getAttribute("period",period);
      bool hasTimePeriod = valParams->getAttribute("timeperiod",timePeriod);
      if( hasTimePeriod ) period = 0;
      
      if( hasPeriod && hasTimePeriod ){
        std::ostringstream msg;
        msg << "ERROR: When specifying a TurbulentInletBC, you cannot specify both timeperiod AND period. Please revise your input file." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      typedef typename TurbulentInletBC<FieldT>::Builder Builder;
      builders.push_back( scinew Builder(tag,inputFileName, velDir,period, timePeriod) );
    }
    
    return builders;
  }
  
  //------------------------------------------------------------------
  
  void
  create_expressions_from_input( Uintah::ProblemSpecP uintahSpec,
                                 GraphCategories& gc )
  {
    Expr::ExpressionBuilder* builder = nullptr;
    
    Uintah::ProblemSpecP parser = uintahSpec->findBlock("Wasatch");
    //___________________________________
    // parse and build basic expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("BasicExpression");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("BasicExpression") ){
      
      std::string fieldType;
      exprParams->getAttribute("type",fieldType);
      
      switch( get_field_type(fieldType) ){
        case SVOL : builder = build_basic_expr< SVolField >( exprParams, uintahSpec );  break;
        case XVOL : builder = build_basic_expr< XVolField >( exprParams, uintahSpec );  break;
        case YVOL : builder = build_basic_expr< YVolField >( exprParams, uintahSpec );  break;
        case ZVOL : builder = build_basic_expr< ZVolField >( exprParams, uintahSpec );  break;
        case PARTICLE : builder = build_basic_particle_expr< ParticleField >( exprParams );  break;
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      const Category cat = parse_tasklist(exprParams,false);
      gc[cat]->exprFactory->register_expression( builder );
    }

    //________________________________________
    // parse and build FAN Models
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("FanModel");
        exprParams != nullptr;
        exprParams = exprParams->findNextBlock("FanModel") ){

      // get the name of this fan model
      std::string fanName;
      exprParams->getAttribute("name",fanName);
      
      // get the target velocities, u, v, w. Here we now only support constant velocities.
      std::vector<double> targetVelocities;
      exprParams->get("TargetVelocities", targetVelocities);
      

      // get the density tagname
      Expr::Tag densityTag;
      Uintah::ProblemSpecP densityParams = parser->findBlock("Density");
      if( densityParams->findBlock("NameTag") ){
        densityTag = parse_nametag( densityParams->findBlock("NameTag") );
      }
      else{
        std::string densName;
        Uintah::ProblemSpecP constDensParam = densityParams->findBlock("Constant");
        constDensParam->getAttribute( "name", densName );
        densityTag = Expr::Tag( densName, Expr::STATE_NONE );
      }

      
      // now obtain all relevant momentum names and tags so that we can add the fan source accordingly
      Uintah::ProblemSpecP momentumSpec  = parser->findBlock("MomentumEquations");
      std::string xmomname, ymomname, zmomname;
      const Uintah::ProblemSpecP doxmom = momentumSpec->get( "X-Momentum", xmomname );
      const Uintah::ProblemSpecP doymom = momentumSpec->get( "Y-Momentum", ymomname );
      const Uintah::ProblemSpecP dozmom = momentumSpec->get( "Z-Momentum", zmomname );

      // get the geometry of the fan
      std::multimap <Uintah::GeometryPieceP, double > geomObjectsMap;
      double outsideValue = 0.0;
      std::vector<Uintah::GeometryPieceP> geomObjects;
      Uintah::GeometryPieceFactory::create(exprParams->findBlock("geom_object"),geomObjects);
      double insideValue = 1.0;
      geomObjectsMap.insert(std::pair<Uintah::GeometryPieceP, double>(geomObjects.back(), insideValue)); // set a value inside the geometry object
      
      const TagNames tNames = TagNames::self();
      OldVariable& oldVar = OldVariable::self();
      
      if( doxmom ){
        typedef XVolField FieldT;
        const double targetVelocity =targetVelocities[0];
        std::cout << "target vels" << targetVelocity << std::endl;
        // declare fan source term tag
        const Expr::Tag fanSourceTag(fanName + "_source_x", Expr::STATE_NONE);
        
        // need to use the old momentum RHS tag
        Expr::Tag momRHSOldTag(xmomname + "_rhs_old",Expr::STATE_NONE);
        const Expr::Tag momOldTag(xmomname, Expr::STATE_DYNAMIC);
        
        // now create an XVOL geometry expression using GeometryBased
        const Expr::Tag volFracTag(fanName+"_location_x",Expr::STATE_NONE);
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename GeometryBased<FieldT>::Builder(volFracTag, geomObjectsMap, outsideValue));
        
        // now create the xmomentum source term
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename FanModel<FieldT>::Builder(fanSourceTag,  densityTag, momOldTag, momRHSOldTag, volFracTag, targetVelocity));
        
        // create an old variable
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, fanSourceTag);
        Expr::Tag momRHSTag(xmomname + "_rhs", Expr::STATE_NONE);
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, momRHSTag);
      }
      
      if( doymom ){
        typedef YVolField FieldT;
        const double targetVelocity =targetVelocities[1];
        std::cout << "target vels" << targetVelocity << std::endl;

        // declare fan source term tag
        const Expr::Tag fanSourceTag(fanName + "_source_y", Expr::STATE_NONE);
        
        // need to use the old momentum RHS tag
        Expr::Tag momRHSOldTag(ymomname + "_rhs_old",Expr::STATE_NONE);
        const Expr::Tag momOldTag(ymomname, Expr::STATE_DYNAMIC);
        
        // now create a YVOL geometry expression using GeometryBased
        const Expr::Tag volFracTag(fanName+"_location_y",Expr::STATE_NONE);
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename GeometryBased<FieldT>::Builder(volFracTag, geomObjectsMap, outsideValue));
        
        // now create the xmomentum source term
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename FanModel<FieldT>::Builder(fanSourceTag,  densityTag, momOldTag, momRHSOldTag, volFracTag, targetVelocity));
        
        // create an old variable
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, fanSourceTag);
        Expr::Tag momRHSTag(ymomname + "_rhs", Expr::STATE_NONE);
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, momRHSTag);
      }
      
      if( dozmom ){
        typedef ZVolField FieldT;
        const double targetVelocity =targetVelocities[2];
        // declare fan source term tag
        const Expr::Tag fanSourceTag(fanName + "_source_z", Expr::STATE_NONE);
        
        // need to use the old momentum RHS tag
        Expr::Tag momRHSOldTag(zmomname + "_rhs_old",Expr::STATE_NONE);
        const Expr::Tag momOldTag(zmomname, Expr::STATE_DYNAMIC);
        
        // now create a ZVOL geometry expression using GeometryBased
        const Expr::Tag volFracTag(fanName+"_location_z",Expr::STATE_NONE);
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename GeometryBased<FieldT>::Builder(volFracTag, geomObjectsMap, outsideValue));
        
        // now create the xmomentum source term
        gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename FanModel<FieldT>::Builder(fanSourceTag,  densityTag, momOldTag, momRHSOldTag, volFracTag, targetVelocity));
        
        // create an old variable
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, fanSourceTag);
        Expr::Tag momRHSTag(zmomname + "_rhs", Expr::STATE_NONE);
        oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, momRHSTag);
      }
    }

    //________________________________________
    // parse and build TargetValueSource Models
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("TargetValueSource");
        exprParams != nullptr;
        exprParams = exprParams->findNextBlock("TargetValueSource") ){
      
      typedef SVolField FieldT;
      // get the name of this target source
      std::string targetValueSrcName;
      exprParams->getAttribute("name",targetValueSrcName);
      const Expr::Tag targetValueSourceTag(targetValueSrcName + "_source", Expr::STATE_NONE);

      // get the name of the target field (usually a scalar, always state_dynamic - use old value)
      std::string targetFieldName;
      exprParams->getAttribute("targetfieldname",targetFieldName);
      const Expr::Tag targetFieldTag(targetFieldName, Expr::STATE_DYNAMIC);
      const Expr::Tag targetFieldRHSTag(targetFieldName + "_rhs", Expr::STATE_NONE);
      const Expr::Tag targetFieldRHSOldTag(targetFieldName + "_rhs_old", Expr::STATE_NONE);
      
      // get the name of the target field (usually a scalar, always state_dynamic - use old value)
      double targetValue;
      if (exprParams->getAttribute("targetvalue",targetValue))
        exprParams->getAttribute("targetvalue",targetValue);

      // if the target value is defined by another expression, get the nametag of that expression
      Expr::Tag targetValueExpressionTag;
      if (exprParams->findBlock("NameTag"))
        targetValueExpressionTag = parse_nametag( exprParams->findBlock("NameTag") );

     
      // get the geometry of the fan
      std::multimap <Uintah::GeometryPieceP, double > geomObjectsMap;
      double outsideValue = 0.0;
      std::vector<Uintah::GeometryPieceP> geomObjects;
      Uintah::GeometryPieceFactory::create(exprParams->findBlock("geom_object"),geomObjects);
      double insideValue = 1.0;
      geomObjectsMap.insert(std::pair<Uintah::GeometryPieceP, double>(geomObjects.back(), insideValue)); // set a value inside the geometry object
      // now create an XVOL geometry expression using GeometryBased
      const Expr::Tag volFracTag(targetValueSrcName + "_location",Expr::STATE_NONE);
      gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename GeometryBased<FieldT>::Builder(volFracTag, geomObjectsMap, outsideValue));

      
      const TagNames tNames = TagNames::self();
      OldVariable& oldVar = OldVariable::self();

      // create an old variable
      oldVar.add_variable<FieldT>( ADVANCE_SOLUTION, targetFieldRHSTag);

      // now create the xmomentum source term
      gc[ADVANCE_SOLUTION]->exprFactory->register_expression(scinew typename TargetValueSource<FieldT>::Builder(targetValueSourceTag, targetFieldTag, targetFieldRHSOldTag, volFracTag, targetValueExpressionTag, targetValue));
      
    }
    //________________________________________
    // parse and build Taylor-Green Vortex MMS
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("TaylorVortexMMS");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("TaylorVortexMMS") ){
      
      std::string fieldType;
      exprParams->getAttribute("type",fieldType);
      
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
      
      const Category cat = parse_tasklist(exprParams,false);
      gc[cat]->exprFactory->register_expression( builder );
    }
    
    //___________________________________________________
    // parse and build physical coefficients expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("PrecipitationBasicExpression");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("PrecipitationBasicExpression") ){
      
      std::string fieldType;
      exprParams->getAttribute("type",fieldType);
      
      switch( get_field_type(fieldType) ){
        case SVOL : builder = build_precipitation_expr< SVolField >( exprParams , parser);  break;
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      const Category cat = parse_tasklist(exprParams,false);
      gc[cat]->exprFactory->register_expression( builder );
    }
    
    //___________________________________________________
    // parse and build post-processing expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("PostProcessingExpression");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("PostProcessingExpression") ){
      
      std::string fieldType;
      exprParams->getAttribute("type",fieldType);
      
      const Expr::Tag tag = parse_nametag( exprParams->findBlock("NameTag") );
    
      if( exprParams->findBlock("VelocityMagnitude") ){
        Uintah::ProblemSpecP valParams = exprParams->findBlock("VelocityMagnitude");
        
        Expr::Tag xVelTag = Expr::Tag();
        if (valParams->findBlock("XVelocity"))
          xVelTag = parse_nametag( valParams->findBlock("XVelocity")->findBlock("NameTag") );
        
        Expr::Tag yVelTag = Expr::Tag();
        if (valParams->findBlock("YVelocity"))
          yVelTag = parse_nametag( valParams->findBlock("YVelocity")->findBlock("NameTag") );
        
        Expr::Tag zVelTag = Expr::Tag();
        if (valParams->findBlock("ZVelocity"))
          zVelTag = parse_nametag( valParams->findBlock("ZVelocity")->findBlock("NameTag") );
        
        typedef VelocityMagnitude<SVolField, XVolField, YVolField, ZVolField>::Builder Builder;
        builder = scinew Builder(tag, xVelTag, yVelTag, zVelTag);
      }
      
      else if( exprParams->findBlock("Vorticity") ){
        Uintah::ProblemSpecP valParams = exprParams->findBlock("Vorticity");
        std::string vorticityComponent;
        valParams->require("Component",vorticityComponent);
        
        Expr::Tag vel1Tag = Expr::Tag();
        if (valParams->findBlock("Vel1"))
          vel1Tag = parse_nametag( valParams->findBlock("Vel1")->findBlock("NameTag") );
        Expr::Tag vel2Tag = Expr::Tag();
        if (valParams->findBlock("Vel2"))
          vel2Tag = parse_nametag( valParams->findBlock("Vel2")->findBlock("NameTag") );
        if (vorticityComponent == "X") {
          typedef Vorticity<SVolField, ZVolField, YVolField>::Builder Builder;
          builder = scinew Builder(tag, vel1Tag, vel2Tag);
        } else if (vorticityComponent == "Y") {
          typedef Vorticity<SVolField, XVolField, ZVolField>::Builder Builder;
          builder = scinew Builder(tag, vel1Tag, vel2Tag);
        } else if (vorticityComponent == "Z") {
          typedef Vorticity<SVolField, YVolField, XVolField>::Builder Builder;
          builder = scinew Builder(tag, vel1Tag, vel2Tag);
        }
      }
      
      else if( exprParams->findBlock("KineticEnergy") ) {
        Uintah::ProblemSpecP keSpec = exprParams->findBlock("KineticEnergy");
        
        Expr::Tag xVelTag = Expr::Tag();
        if (keSpec->findBlock("XVelocity"))
          xVelTag = parse_nametag( keSpec->findBlock("XVelocity")->findBlock("NameTag") );
        
        Expr::Tag yVelTag = Expr::Tag();
        if (keSpec->findBlock("YVelocity"))
          yVelTag = parse_nametag( keSpec->findBlock("YVelocity")->findBlock("NameTag") );
        
        Expr::Tag zVelTag = Expr::Tag();
        if (keSpec->findBlock("ZVelocity"))
          zVelTag = parse_nametag( keSpec->findBlock("ZVelocity")->findBlock("NameTag") );
        
        bool totalKE=false;
        keSpec->getAttribute("total",totalKE);
        if (totalKE) {
          typedef TotalKineticEnergy<XVolField, YVolField, ZVolField>::Builder Builder;
          builder = scinew Builder(tag, xVelTag, yVelTag, zVelTag);
        } else {
          typedef KineticEnergy<SVolField, XVolField, YVolField, ZVolField>::Builder Builder;
          builder = scinew Builder(tag, xVelTag, yVelTag, zVelTag);
        }
      }  else if( exprParams->findBlock("InterpolateExpression") ){
        Uintah::ProblemSpecP valParams = exprParams->findBlock("InterpolateExpression");
        std::string srcFieldType;
        valParams->getAttribute("type",srcFieldType);
        const Expr::Tag srcTag = parse_nametag( valParams->findBlock("NameTag") );
        
        switch( get_field_type(srcFieldType) ){
          case XVOL : {
            typedef InterpolateExpression<XVolField, SVolField>::Builder Builder;
            builder = scinew Builder(tag, srcTag);
            break;
          }
          case YVOL : {
            typedef InterpolateExpression<YVolField, SVolField>::Builder Builder;
            builder = scinew Builder(tag, srcTag);
            break;
          }
          case ZVOL : {
            typedef InterpolateExpression<ZVolField, SVolField>::Builder Builder;
            builder = scinew Builder(tag, srcTag);
            break;
          }
          case PARTICLE : {
            typedef InterpolateParticleExpression<SVolField>::Builder Builder;
            Uintah::ProblemSpecP pInfoSpec = valParams->findBlock("ParticleInfo");
            std::string psize, px, py, pz;
            
            pInfoSpec->getAttribute("size",psize);
            const Expr::Tag psizeTag(psize,Expr::STATE_NP1);
            
            pInfoSpec->getAttribute("px",px);
            const Expr::Tag pxTag(px,Expr::STATE_NP1);
            
            pInfoSpec->getAttribute("py",py);
            const Expr::Tag pyTag(py,Expr::STATE_NP1);
            
            pInfoSpec->getAttribute("pz",pz);
            const Expr::Tag pzTag(pz,Expr::STATE_NP1);
            
            const Expr::TagList pPosTags = tag_list(pxTag,pyTag,pzTag);
            builder = scinew Builder(tag, srcTag, psizeTag, pPosTags );
            break;
          }
          default:
            std::ostringstream msg;
            msg << "ERROR: unsupported field type '" << srcFieldType << "'" << "while parsing an InterpolateExpression." << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      }

      const Category cat = parse_tasklist(exprParams,false);
      gc[cat]->exprFactory->register_expression( builder );
    }
    
    //___________________________________________________
    // parse and build boundary condition expressions
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("BCExpression");
         exprParams != nullptr;
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
      while( taskNameIter != taskNamesList.end() ){
        std::string taskName = *taskNameIter;
        std::list<Expr::ExpressionBuilder*> builders;
        switch( get_field_type(fieldType) ){
          case SVOL : builders = build_bc_expr< SVolField >( exprParams,parser );  break;
          case XVOL : builders = build_bc_expr< XVolField >( exprParams,parser );  break;
          case YVOL : builders = build_bc_expr< YVolField >( exprParams,parser );  break;
          case ZVOL : builders = build_bc_expr< ZVolField >( exprParams,parser );  break;
          default:
            std::ostringstream msg;
            msg << "ERROR: unsupported field type '" << fieldType << "' while trying to register BC expression.." << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        Category cat = INITIALIZATION;
        if     ( taskName == "initialization"   ) cat = INITIALIZATION;
        else if( taskName == "advance_solution" ) cat = ADVANCE_SOLUTION;
        else{
          std::ostringstream msg;
          msg << "ERROR: unsupported task list '" << taskName << "' while parsing BCExpression." << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
        GraphHelper* const graphHelper = gc[cat];
        BOOST_FOREACH( Expr::ExpressionBuilder* builder, builders ){
          graphHelper->exprFactory->register_expression( builder );
        }
        
        ++taskNameIter;
      }
    }
    
    // This is a special parser for turbulent inlets
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("TurbulentInlet");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("TurbulentInlet") ){
      
      std::string inputFileName, baseName;
      int period=1;
      double timePeriod;
      exprParams->get("InputFile",inputFileName);
      exprParams->get("BaseName",baseName);
      
      bool hasPeriod = exprParams->getAttribute("period",period);
      bool hasTimePeriod = exprParams->getAttribute("timeperiod",timePeriod);
      if (hasTimePeriod) period = 0;
      
      if (hasPeriod && hasTimePeriod) {
        std::ostringstream msg;
        msg << "ERROR: When specifying a TurbulentInletBC, you cannot specify both timeperiod AND period. Please revise your input file." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      Expr::Tag xVelTag("x-" + baseName, Expr::STATE_NONE);
      typedef TurbulentInletBC<XVolField>::Builder xBuilder;
      
      Expr::Tag yVelTag("y-" + baseName, Expr::STATE_NONE);
      typedef TurbulentInletBC<YVolField>::Builder yBuilder;
      
      Expr::Tag zVelTag("z-" + baseName, Expr::STATE_NONE);
      typedef TurbulentInletBC<ZVolField>::Builder zBuilder;
      
      GraphHelper* const initGraphHelper = gc[INITIALIZATION];
      initGraphHelper->exprFactory->register_expression( scinew xBuilder(xVelTag, inputFileName, "X", period, timePeriod) );
      initGraphHelper->exprFactory->register_expression( scinew yBuilder(yVelTag, inputFileName, "Y", period, timePeriod) );
      initGraphHelper->exprFactory->register_expression( scinew zBuilder(zVelTag, inputFileName, "Z", period, timePeriod) );
      
      GraphHelper* const slnGraphHelper = gc[ADVANCE_SOLUTION];
      slnGraphHelper->exprFactory->register_expression( scinew xBuilder(xVelTag, inputFileName, "X", period, timePeriod) );
      slnGraphHelper->exprFactory->register_expression( scinew yBuilder(yVelTag, inputFileName, "Y", period, timePeriod) );
      slnGraphHelper->exprFactory->register_expression( scinew zBuilder(zVelTag, inputFileName, "Z", period, timePeriod) );
    }

    //_________________________________________________
    // This is a special parser for variable density MMS
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("VarDenOscillatingMMS");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("VarDenOscillatingMMS") ){
      
      const TagNames& tagNames = TagNames::self();

      double rho0, rho1, uf, vf, k, w, d;
      exprParams->getAttribute("rho0",rho0);
      exprParams->getAttribute("rho1",rho1);
      exprParams->getAttribute("uf",uf);
      exprParams->getAttribute("vf",vf);
      exprParams->getAttribute("k",k);
      exprParams->getAttribute("w",w);
      exprParams->getAttribute("d",d);
      
      std::string x1="X", x2="Y";
      if (exprParams->findAttribute("x1"))
        exprParams->getAttribute("x1",x1);
      if (exprParams->findAttribute("x2"))
        exprParams->getAttribute("x2",x2);

      Expr::Tag x1Tag, x2Tag;

      if      (x1 == "X")  x1Tag = tagNames.xsvolcoord;
      else if (x1 == "Y")  x1Tag = tagNames.ysvolcoord;
      else if (x1 == "Z")  x1Tag = tagNames.zsvolcoord;

      if      (x2 == "X")  x2Tag = tagNames.xsvolcoord;
      else if (x2 == "Y")  x2Tag = tagNames.ysvolcoord;
      else if (x2 == "Z")  x2Tag = tagNames.zsvolcoord;
      
      Expr::ExpressionFactory& icFactory = *gc[INITIALIZATION  ]->exprFactory;
      Expr::ExpressionFactory& factory   = *gc[ADVANCE_SOLUTION]->exprFactory;

      std::string mixFracName;
      exprParams->get("Scalar", mixFracName);
      const Expr::Tag mixFracTag     ( mixFracName        , Expr::STATE_NONE );
      const Expr::Tag mixFracExactTag(mixFracName+"_exact", Expr::STATE_NONE );
      typedef VarDenOscillatingMMSMixFrac<SVolField>::Builder MixFracBuilder;
      icFactory.register_expression( scinew MixFracBuilder( mixFracTag     , x1Tag, x2Tag, tagNames.time, rho0, rho1, w, k, uf, vf, false ) );
      icFactory.register_expression( scinew MixFracBuilder( mixFracExactTag, x1Tag, x2Tag, tagNames.time, rho0, rho1, w, k, uf, vf, false ) );
      factory  .register_expression( scinew MixFracBuilder( mixFracExactTag, x1Tag, x2Tag, tagNames.time, rho0, rho1, w, k, uf, vf, true  ) );

      const Expr::Tag diffCoefTag = parse_nametag(exprParams->findBlock("DiffusionCoefficient")->findBlock("NameTag"));

      std::string densityName;
      parser->findBlock("Density")->findBlock("NameTag")->getAttribute( "name", densityName );
      const Expr::Tag initDensityTag = Expr::Tag(densityName, Expr::STATE_NONE);
      const Expr::Tag densityTag     = Expr::Tag(densityName, Expr::STATE_NP1 );

      typedef DiffusiveConstant<SVolField>::Builder diffCoefBuilder;
      icFactory.register_expression( scinew diffCoefBuilder( diffCoefTag, initDensityTag, d ) );
      factory  .register_expression( scinew diffCoefBuilder( diffCoefTag , densityTag   , d ) );
    }  
    
    //___________________________________________________
    // parse and build initial conditions for moment transport
    int nEnv = 0;
    if (parser->findBlock("MomentTransportEquation")) {
      parser->findBlock("MomentTransportEquation")->get( "NumberOfEnvironments", nEnv );
    }
    int nEqs = 2*nEnv; // we need the number of equations so that we only build the necessary number of moments for initialization
    for( Uintah::ProblemSpecP exprParams = parser->findBlock("MomentInitialization");
         exprParams != nullptr;
         exprParams = exprParams->findNextBlock("MomentInitialization") ){
      
      std::string populationName;
      
      exprParams->get("PopulationName", populationName);
      std::vector<double> initialMoments;
      
      exprParams->get("Values", initialMoments, nEqs); // get only the first nEqs moments
      
      assert( (int) initialMoments.size() == nEqs );
      
      const int nMoments = initialMoments.size();
      typedef Expr::ConstantExpr<SVolField>::Builder Builder;
      GraphHelper* const graphHelper = gc[INITIALIZATION];
      for (int i=0; i<nMoments; i++) {
        double val = initialMoments[i];
        std::stringstream ss;
        ss << i;
        Expr::Tag thisMomentTag("m_" + populationName + "_" + ss.str(), Expr::STATE_NONE);
        graphHelper->exprFactory->register_expression( scinew Builder( thisMomentTag, val ) );
      }
    }
  }
  
  //------------------------------------------------------------------
  
}
