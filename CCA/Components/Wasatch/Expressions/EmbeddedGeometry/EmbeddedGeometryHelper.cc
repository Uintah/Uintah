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
#include "EmbeddedGeometryHelper.h"
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/StringNames.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/OscillatingCylinder.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/GeometryPieceWrapper.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/AreaFraction.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

namespace Wasatch{

  Expr::Tag xvol_frac_tag() {
    return Expr::Tag("xvolFraction",Expr::STATE_NONE);
  }
  
  Expr::Tag yvol_frac_tag() {
    return Expr::Tag("yvolFraction",Expr::STATE_NONE);
  }
  
  Expr::Tag zvol_frac_tag() {
    return Expr::Tag("zvolFraction",Expr::STATE_NONE);
  }

  Expr::Tag svol_frac_tag() {
    return Expr::Tag("svolFraction",Expr::STATE_NONE);
  }

  void
  parse_embedded_geometry( Uintah::ProblemSpecP parser,
                                GraphCategories& gc )
  {
    if (parser->findBlock("EmbeddedGeometry")) {

      Expr::ExpressionBuilder* volFracBuilder = NULL;
      GraphHelper* const initgh = gc[INITIALIZATION];
      GraphHelper* const solngh = gc[ADVANCE_SOLUTION];
      
      Uintah::ProblemSpecP geomParams = parser->findBlock("EmbeddedGeometry");

      // we only allow the ENTIRE geometry to be inverted, not per intrusion
      bool inverted = geomParams->findBlock("Inverted");

      Uintah::ProblemSpecP geomExprParams = geomParams->findBlock("GeometryExpression");      
      if (geomExprParams) {        
        if (geomExprParams->findBlock("OscillatingCylinder")) {
          Uintah::ProblemSpecP valParams = geomExprParams->findBlock("OscillatingCylinder");
          double radius, insideValue, outsideValue, frequency, amplitude;
          valParams->getAttribute("radius",radius);
          valParams->getAttribute("insideValue",insideValue);
          valParams->getAttribute("outsideValue",outsideValue);
          valParams->getAttribute("frequency",frequency);
          valParams->getAttribute("amplitude",amplitude);
          std::vector<double> origin;
          valParams->get("Origin", origin, 2); // get only the first nEqs moments
          std::vector<double> oscillatingdir;
          valParams->get("OscillatingDir", oscillatingdir, 2); // get only the first nEqs moments
          if (oscillatingdir[0]!=0) oscillatingdir[0] /= oscillatingdir[0];
          if (oscillatingdir[1]!=0) oscillatingdir[1] /= oscillatingdir[1];
          std::string axis;
          valParams->get("Axis",axis);
          typedef OscillatingCylinder::Builder Builder;
          volFracBuilder = scinew Builder( svol_frac_tag(), axis, origin, oscillatingdir, insideValue, outsideValue, radius,frequency, amplitude );
        }
        
      } else {
        // parse all intrusions
        std::vector<Uintah::GeometryPieceP> geomObjects;
        for( Uintah::ProblemSpecP intrusionParams = geomParams->findBlock("Intrusion");
            intrusionParams != 0;
            intrusionParams = intrusionParams->findNextBlock("Intrusion") )
        {
          Uintah::GeometryPieceFactory::create(intrusionParams->findBlock("geom_object"),geomObjects);
        }
        typedef GeometryPieceWrapper::Builder svolfracBuilder;        
        volFracBuilder = scinew svolfracBuilder( svol_frac_tag(), geomObjects, inverted );
      }
      
      // register the volume fractions
//      initgh->exprFactory->register_expression( volFracBuilder );
      solngh->exprFactory->register_expression( volFracBuilder );
      
      // register the area fractions
      typedef AreaFraction<XVolField>::Builder xvolfracBuilder;
      typedef AreaFraction<YVolField>::Builder yvolfracBuilder;
      typedef AreaFraction<ZVolField>::Builder zvolfracBuilder;      
//      initgh->exprFactory->register_expression( scinew xvolfracBuilder( xvol_frac_tag(), svol_frac_tag() ) );
//      initgh->exprFactory->register_expression( scinew yvolfracBuilder( yvol_frac_tag(), svol_frac_tag() ) );
//      initgh->exprFactory->register_expression( scinew zvolfracBuilder( zvol_frac_tag(), svol_frac_tag() ) );
      solngh->exprFactory->register_expression( scinew xvolfracBuilder( xvol_frac_tag(), svol_frac_tag() ) );
      solngh->exprFactory->register_expression( scinew yvolfracBuilder( yvol_frac_tag(), svol_frac_tag() ) );
      solngh->exprFactory->register_expression( scinew zvolfracBuilder( zvol_frac_tag(), svol_frac_tag() ) );
    }
  }
  
  //------------------------------------------------------------------
  
}
