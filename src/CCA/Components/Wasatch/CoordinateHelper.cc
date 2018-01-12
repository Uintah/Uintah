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

#include <boost/foreach.hpp>

//-- Wasatch includes --//
#include "CoordinateHelper.h"
#include "TagNames.h"

//-- Uintah includes --//
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>
#include <CCA/Components/Wasatch/Expressions/Coordinate.h>
#include <CCA/Components/Wasatch/OldVariable.h>

namespace WasatchCore{
  
  CoordinateNames::CoordinateNames()
  {
    using namespace std;
    const TagNames& tagNames = TagNames::self();
    
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.xsvolcoord, "SVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.ysvolcoord, "SVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.zsvolcoord, "SVOL"));
    
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.xxvolcoord, "XVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.yxvolcoord, "XVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.zxvolcoord, "XVOL"));

    coordMap_.insert(pair<Expr::Tag, string>(tagNames.xyvolcoord, "YVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.yyvolcoord, "YVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.zyvolcoord, "YVOL"));
    
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.xzvolcoord, "ZVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.yzvolcoord, "ZVOL"));
    coordMap_.insert(pair<Expr::Tag, string>(tagNames.zzvolcoord, "ZVOL"));
  }
  
  //------------------------------------------------------------------
  
  const CoordinateNames&
  CoordinateNames::self()
  {
    static CoordinateNames s;
    return s;
  }
  
  const CoordinateNames::CoordMap&
  CoordinateNames::coordinate_map()
  {
    return self().coordMap_;
  }

  //------------------------------------------------------------------

  void register_coordinate_expressions( GraphCategories& gc,
                                        const bool isPeriodic )
  {
    using namespace std;
    const GraphHelper& initgh = *gc[INITIALIZATION  ];
    const GraphHelper& slngh  = *gc[ADVANCE_SOLUTION];
    
    const CoordinateNames::CoordMap& coordMap = CoordinateNames::coordinate_map();
    // make logical decisions based on the specified boundary types
    BOOST_FOREACH( const CoordinateNames::CoordMap::value_type& coordPair, coordMap )
    {
      const Expr::Tag& coordTag = coordPair.first;
      const string& coordFieldT = coordPair.second;
      
      // OldVariable& oldVar = OldVariable::self();
      Expr::ExpressionID coordID;
      if (coordFieldT == "SVOL") {
        initgh         .exprFactory->register_expression( scinew Coordinates<SVolField>::Builder( coordTag ) );
        coordID = slngh.exprFactory->register_expression( scinew Coordinates<SVolField>::Builder( coordTag ) );
      }
      
      if (coordFieldT == "XVOL") {
        initgh         .exprFactory->register_expression( scinew Coordinates<XVolField>::Builder( coordTag ) );
        coordID = slngh.exprFactory->register_expression( scinew Coordinates<XVolField>::Builder( coordTag ) );
      }
      
      if (coordFieldT == "YVOL") {
        initgh         .exprFactory->register_expression( scinew Coordinates<YVolField>::Builder( coordTag ) );
        coordID = slngh.exprFactory->register_expression( scinew Coordinates<YVolField>::Builder( coordTag ) );
      }
      
      if (coordFieldT == "ZVOL") {
        initgh         .exprFactory->register_expression( scinew Coordinates<ZVolField>::Builder( coordTag ) );
        coordID = slngh.exprFactory->register_expression( scinew Coordinates<ZVolField>::Builder( coordTag ) );
      }
      
      slngh.exprFactory->cleave_from_parents(coordID);
    }
  }
} // namespace WasatchCore
