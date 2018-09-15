/**
 *  \file   TransportEquation.cc
 *  \date   Nov 13, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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
 *
 */



#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

namespace WasatchCore{

  //---------------------------------------------------------------------------

  TransportEquation::
  TransportEquation( GraphCategories& gc,
                     const std::string& solnVarName,
                     const Direction stagLoc )
  : EquationBase::EquationBase( gc, solnVarName, stagLoc ),
    flowTreatment_  ( Wasatch::flow_treatment()        ),
    isConstDensity_ ( flowTreatment_ == INCOMPRESSIBLE )
  {}

  //---------------------------------------------------------------------------

  void TransportEquation::setup()
  {
    FieldTagInfo tagInfo;
    Expr::TagList sourceTags;

    EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ){
      EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
      tagInfo[VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
      tagInfo[AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
      tagInfo[AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
      tagInfo[AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();
    }

    setup_diffusive_flux ( tagInfo );
    setup_convective_flux( tagInfo );
    setup_source_terms   ( tagInfo, sourceTags );

    // now build the RHS given the tagInfo that has been populated
    rhsExprID_ = setup_rhs( tagInfo, sourceTags );
    assert( rhsExprID_ != Expr::ExpressionID::null_id() );
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }

  //---------------------------------------------------------------------------

} // namespace WasatchCore
