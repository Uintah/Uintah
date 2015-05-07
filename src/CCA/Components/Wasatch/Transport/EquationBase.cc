/**
 *  \file   EquationBase.cc
 *  \date   Nov 13, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2015 The University of Utah
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



#include <CCA/Components/Wasatch/Transport/EquationBase.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

namespace Wasatch{

  //---------------------------------------------------------------------------

  EquationBase::
  EquationBase( GraphCategories& gc,
                const std::string solnVarName,
                const Direction direction,
                Uintah::ProblemSpecP params )
  : direction_          ( direction ),
    params_             ( params ),
    gc_                 ( gc ),
    solnVarName_        ( solnVarName ),
    solnVarTag_         ( solnVarName, Expr::STATE_DYNAMIC ),
    rhsTag_             ( solnVarName + "_rhs", Expr::STATE_NONE )
  {}


  //---------------------------------------------------------------------------

  Expr::ExpressionID EquationBase::get_rhs_id() const
  {
    assert( rhsExprID_ != Expr::ExpressionID::null_id() );
    return rhsExprID_;
  }

  //---------------------------------------------------------------------------
  
  std::string
  EquationBase::dir_name() const
  {
    switch (direction_) {
      case XDIR: return "x";
      case YDIR: return "y";
      case ZDIR: return "z";
      case NODIR:
      default: return "";
    }
  }

  //---------------------------------------------------------------------------

} // namespace Wasatch
