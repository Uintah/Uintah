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

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>

//-- Wasatch includes --//
#include "EmbeddedGeometryHelper.h"
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/OscillatingCylinder.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/GeometryPieceWrapper.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/AreaFraction.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <CCA/Components/Wasatch/OldVariable.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

namespace WasatchCore{
  
  EmbeddedGeometryHelper::EmbeddedGeometryHelper() :
    svolfrac_(""),
    xvolfrac_(""),
    yvolfrac_(""),
    zvolfrac_(""),
    hasEmbeddedGeometry_( false ),
    hasMovingGeometry_  ( false ),
    doneSetup_          ( false )
  {}
  
  //------------------------------------------------------------------

  template<>
  Expr::Tag
  EmbeddedGeometryHelper::vol_frac_tag<SVolField>() const
  {
    check_state();
    return (svolfrac_=="") ? Expr::Tag() : Expr::Tag(svolfrac_,Expr::STATE_NONE);
  }

  //------------------------------------------------------------------
  
  template<>
  Expr::Tag
  EmbeddedGeometryHelper::vol_frac_tag<XVolField>() const
  {
    check_state();
    return (xvolfrac_=="") ? Expr::Tag() : Expr::Tag(xvolfrac_,Expr::STATE_NONE);
  }

  //------------------------------------------------------------------
  
  template<>
  Expr::Tag
  EmbeddedGeometryHelper::vol_frac_tag<YVolField>() const
  {
    check_state();
    return (yvolfrac_=="") ? Expr::Tag() : Expr::Tag(yvolfrac_,Expr::STATE_NONE);
  }

  //------------------------------------------------------------------
  
  template<>
  Expr::Tag  
  EmbeddedGeometryHelper::vol_frac_tag<ZVolField>() const
  {
    check_state();
    return (zvolfrac_=="") ? Expr::Tag() : Expr::Tag(zvolfrac_,Expr::STATE_NONE);
  }

  //------------------------------------------------------------------

  EmbeddedGeometryHelper&
  EmbeddedGeometryHelper::self()
  {
    static EmbeddedGeometryHelper s;
    return s;
  }

  //------------------------------------------------------------------

  void
  parse_embedded_geometry( Uintah::ProblemSpecP parser,
                           GraphCategories& gc )
  {
    EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    vNames.set_state(true);    
    vNames.set_vol_frac_names("","","","");

    // first parse any common geometry. This MUST COME FIRST BEFORE parsing the embedded geometry
    if (parser->findBlock("CommonGeometry")) {
      // parse all intrusions
      Uintah::ProblemSpecP geomParams = parser->findBlock("CommonGeometry");
      std::vector<Uintah::GeometryPieceP> geomObjects;
      for( Uintah::ProblemSpecP intrusionParams = geomParams->findBlock("geom_object");
          intrusionParams != nullptr;
          intrusionParams = intrusionParams->findNextBlock("geom_object") )
      {
        Uintah::GeometryPieceFactory::create(intrusionParams,geomObjects);
      }
    }
    
    if( parser->findBlock("EmbeddedGeometry") ){
      vNames.set_has_embedded_geometry(true);
      
      Expr::ExpressionBuilder* volFracBuilder = nullptr;
      Expr::ExpressionBuilder* volFracBuilderInit = nullptr;
      GraphHelper* const initgh = gc[INITIALIZATION];
      GraphHelper* const solngh = gc[ADVANCE_SOLUTION];
      
      Uintah::ProblemSpecP geomParams = parser->findBlock("EmbeddedGeometry");

      // we only allow the ENTIRE geometry to be inverted, not per intrusion
      bool inverted = geomParams->findBlock("Inverted");
      
      bool movingGeom = geomParams->findBlock("MovingGeometry");
      vNames.set_has_moving_geometry(movingGeom);

      // check if we have external volume fractions
      if ( geomParams->findBlock("External") ) {
        Uintah::ProblemSpecP externalParams = geomParams->findBlock("External");
        std::string svolfracname, xvolfracname, yvolfracname, zvolfracname;
        externalParams->get("SVolFraction",svolfracname);
        externalParams->get("XVolFraction",xvolfracname);
        externalParams->get("YVolFraction",yvolfracname);
        externalParams->get("ZVolFraction",zvolfracname);
        vNames.set_vol_frac_names(svolfracname, xvolfracname, yvolfracname, zvolfracname);
        // volume fraction expressions have been specified external and should
        // be registered outside, therefore, we return from this call.
        return;
      }
      
      // if no external geometry has been specified, then we parse user-specified intrusions
      vNames.set_vol_frac_names("svolFraction", "xvolFraction", "yvolFraction", "zvolFraction");
      
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
          if (oscillatingdir[0]!=0.0) oscillatingdir[0] = 1.0;
          if (oscillatingdir[1]!=0.0) oscillatingdir[1] = 1.0;
          std::string axis;
          valParams->get("Axis",axis);
          typedef OscillatingCylinder::Builder Builder;
          volFracBuilderInit = scinew Builder( vNames.vol_frac_tag<SVolField>(), axis, origin, oscillatingdir, insideValue, outsideValue, radius,frequency, amplitude );
          if (movingGeom) volFracBuilder = scinew Builder( vNames.vol_frac_tag<SVolField>(), axis, origin, oscillatingdir, insideValue, outsideValue, radius,frequency, amplitude );
        }
        
      } else {
        // parse all intrusions
        std::vector<Uintah::GeometryPieceP> geomObjects;
        for( Uintah::ProblemSpecP intrusionParams = geomParams->findBlock("Intrusion");
            intrusionParams != nullptr;
            intrusionParams = intrusionParams->findNextBlock("Intrusion") )
        {
          Uintah::GeometryPieceFactory::create(intrusionParams->findBlock("geom_object"),geomObjects);
        }
        typedef GeometryPieceWrapper::Builder svolfracBuilder;        
        volFracBuilderInit = scinew svolfracBuilder( vNames.vol_frac_tag<SVolField>(), geomObjects, inverted );
        if (movingGeom) volFracBuilder = scinew svolfracBuilder( vNames.vol_frac_tag<SVolField>(), geomObjects, inverted );
      }
      
      // register the volume fractions
      initgh->exprFactory->register_expression( volFracBuilderInit );
      if (movingGeom) solngh->exprFactory->register_expression( volFracBuilder );
      
      // register the area fractions
      typedef AreaFraction<XVolField>::Builder xvolfracBuilder;
      typedef AreaFraction<YVolField>::Builder yvolfracBuilder;
      typedef AreaFraction<ZVolField>::Builder zvolfracBuilder;
      
      initgh->exprFactory->register_expression( scinew xvolfracBuilder( vNames.vol_frac_tag<XVolField>(), vNames.vol_frac_tag<SVolField>() ) );
      initgh->exprFactory->register_expression( scinew yvolfracBuilder( vNames.vol_frac_tag<YVolField>(), vNames.vol_frac_tag<SVolField>() ) );
      initgh->exprFactory->register_expression( scinew zvolfracBuilder( vNames.vol_frac_tag<ZVolField>(), vNames.vol_frac_tag<SVolField>() ) );

      if (movingGeom) {
        // when the geometry is moving, then recalculate volume fractions at every timestep
        solngh->exprFactory->register_expression( scinew xvolfracBuilder( vNames.vol_frac_tag<XVolField>(), vNames.vol_frac_tag<SVolField>() ) );
        solngh->exprFactory->register_expression( scinew yvolfracBuilder( vNames.vol_frac_tag<YVolField>(), vNames.vol_frac_tag<SVolField>() ) );
        solngh->exprFactory->register_expression( scinew zvolfracBuilder( vNames.vol_frac_tag<ZVolField>(), vNames.vol_frac_tag<SVolField>() ) );
      } else {
        // when the geometry is not moving, copy the volume fractions from the previous timestep
        OldVariable& oldVar = OldVariable::self();
        oldVar.add_variable<SVolField>( ADVANCE_SOLUTION, vNames.vol_frac_tag<SVolField>(), true);
        oldVar.add_variable<XVolField>( ADVANCE_SOLUTION, vNames.vol_frac_tag<XVolField>(), true);
        oldVar.add_variable<YVolField>( ADVANCE_SOLUTION, vNames.vol_frac_tag<YVolField>(), true);
        oldVar.add_variable<ZVolField>( ADVANCE_SOLUTION, vNames.vol_frac_tag<ZVolField>(), true);
      }
      
      // force on initial conditions graph
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.vol_frac_tag<XVolField>() ) );
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.vol_frac_tag<YVolField>() ) );
      initgh->rootIDs.insert( initgh->exprFactory->get_id( vNames.vol_frac_tag<ZVolField>() ) );
    }
  }
  
  //------------------------------------------------------------------
  
  void apply_intrusion_boundary_conditions(WasatchBCHelper& bcHelper)
  {
    EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    BndCondSpec svolFracSpec = {vNames.vol_frac_tag<SVolField>().name(), "none", 0, NEUMANN, DOUBLE_TYPE};
    bcHelper.add_boundary_condition(svolFracSpec);
    bcHelper.apply_boundary_condition<SVolField>( vNames.vol_frac_tag<SVolField>(), INITIALIZATION );
    
    BndCondSpec xvolFracSpec = {vNames.vol_frac_tag<XVolField>().name(), "none", 0, NEUMANN, DOUBLE_TYPE};
    bcHelper.add_boundary_condition(xvolFracSpec);
    bcHelper.apply_boundary_condition<XVolField>( vNames.vol_frac_tag<XVolField>(), INITIALIZATION );
    
    BndCondSpec yvolFracSpec = {vNames.vol_frac_tag<YVolField>().name(), "none", 0, NEUMANN, DOUBLE_TYPE};
    bcHelper.add_boundary_condition(yvolFracSpec);
    bcHelper.apply_boundary_condition<YVolField>( vNames.vol_frac_tag<YVolField>(), INITIALIZATION );
    
    BndCondSpec zvolFracSpec = {vNames.vol_frac_tag<ZVolField>().name(), "none", 0, NEUMANN, DOUBLE_TYPE};
    bcHelper.add_boundary_condition(zvolFracSpec);
    bcHelper.apply_boundary_condition<ZVolField>( vNames.vol_frac_tag<ZVolField>(), INITIALIZATION );
  }

  //------------------------------------------------------------------
}
