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
#include <expression/PlaceHolderExpr.h>

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
  
  VolFractionNames::VolFractionNames() :
    svolfrac_("svolFraction"),
    xvolfrac_("xvolFraction"),
    yvolfrac_("yvolFraction"),
    zvolfrac_("zvolFraction")
  {}
  
  //------------------------------------------------------------------
  
  VolFractionNames&
  VolFractionNames::self()
  {
    static VolFractionNames s;
    return s;
  }

  //------------------------------------------------------------------

  void
  parse_embedded_geometry( Uintah::ProblemSpecP parser,
                                GraphCategories& gc )
  {
    if (parser->findBlock("EmbeddedGeometry")) {
      
      VolFractionNames& vNames = VolFractionNames::self();
      
      Expr::ExpressionBuilder* volFracBuilder = NULL;
      Expr::ExpressionBuilder* volFracBuilderInit = NULL;
      GraphHelper* const initgh = gc[INITIALIZATION];
      GraphHelper* const solngh = gc[ADVANCE_SOLUTION];
      
      Uintah::ProblemSpecP geomParams = parser->findBlock("EmbeddedGeometry");

      // we only allow the ENTIRE geometry to be inverted, not per intrusion
      bool inverted = geomParams->findBlock("Inverted");

      // check if we have external volume fractions
      if ( geomParams->findBlock("External") ) {
        Uintah::ProblemSpecP externalParams = geomParams->findBlock("External");
        std::string svolfracname, xvolfracname, yvolfracname, zvolfracname;
        externalParams->get("SVolFraction",svolfracname);
        externalParams->get("XVolFraction",xvolfracname);
        externalParams->get("YVolFraction",yvolfracname);
        externalParams->get("ZVolFraction",zvolfracname);
        VolFractionNames& vNames = VolFractionNames::self();
        vNames.set_svol_frac_name(svolfracname);
        vNames.set_xvol_frac_name(xvolfracname);
        vNames.set_yvol_frac_name(yvolfracname);
        vNames.set_zvol_frac_name(zvolfracname);
        // volume fraction expressions have been specified external and should
        // be registered outside, therefore, we return from this call.
        return;
      }
      
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
          volFracBuilder = scinew Builder( vNames.svol_frac_tag(), axis, origin, oscillatingdir, insideValue, outsideValue, radius,frequency, amplitude );
          volFracBuilderInit = scinew Builder( vNames.svol_frac_tag(), axis, origin, oscillatingdir, insideValue, outsideValue, radius,frequency, amplitude );
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
        volFracBuilderInit = scinew svolfracBuilder( vNames.svol_frac_tag(), geomObjects, inverted );
        volFracBuilder = scinew svolfracBuilder( vNames.svol_frac_tag(), geomObjects, inverted );
      }
      
      // register the volume fractions
      initgh->exprFactory->register_expression( volFracBuilderInit );
      solngh->exprFactory->register_expression( volFracBuilder );
      
      // register the area fractions
      typedef AreaFraction<XVolField>::Builder xvolfracBuilder;
      typedef AreaFraction<YVolField>::Builder yvolfracBuilder;
      typedef AreaFraction<ZVolField>::Builder zvolfracBuilder;
      
      initgh->exprFactory->register_expression( scinew xvolfracBuilder( vNames.xvol_frac_tag(), vNames.svol_frac_tag() ) );
      initgh->exprFactory->register_expression( scinew yvolfracBuilder( vNames.yvol_frac_tag(), vNames.svol_frac_tag() ) );
      initgh->exprFactory->register_expression( scinew zvolfracBuilder( vNames.zvol_frac_tag(), vNames.svol_frac_tag() ) );
      
      solngh->exprFactory->register_expression( scinew xvolfracBuilder( vNames.xvol_frac_tag(), vNames.svol_frac_tag() ) );
      solngh->exprFactory->register_expression( scinew yvolfracBuilder( vNames.yvol_frac_tag(), vNames.svol_frac_tag() ) );
      solngh->exprFactory->register_expression( scinew zvolfracBuilder( vNames.zvol_frac_tag(), vNames.svol_frac_tag() ) );
      
      // force on graph
      //initgh->rootIDs.insert( initgh->exprFactory->get_id( svol_frac_tagN() ) );
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.xvol_frac_tag() ) );
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.yvol_frac_tag() ) );
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.zvol_frac_tag() ) );
    }
  }
  
  //------------------------------------------------------------------
  
}
