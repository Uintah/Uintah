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

//-- Wasatch includes --//
#include "TagNames.h"

namespace Wasatch{
  
  //------------------------------------------------------------------
  
  TagNames::TagNames() :
  
  time    ( "time", Expr::STATE_NONE     ),
  timestep( "timestep", Expr::STATE_NONE ),
  
  xsvolcoord( "XSVOL", Expr::STATE_NONE ),
  ysvolcoord( "YSVOL", Expr::STATE_NONE ),
  zsvolcoord( "ZSVOL", Expr::STATE_NONE ),
  xxvolcoord( "XXVOL", Expr::STATE_NONE ),
  yxvolcoord( "YXVOL", Expr::STATE_NONE ),
  zxvolcoord( "ZXVOL", Expr::STATE_NONE ),
  xyvolcoord( "XYVOL", Expr::STATE_NONE ),
  yyvolcoord( "YYVOL", Expr::STATE_NONE ),
  zyvolcoord( "ZYVOL", Expr::STATE_NONE ),
  xzvolcoord( "XZVOL", Expr::STATE_NONE ),
  yzvolcoord( "YZVOL", Expr::STATE_NONE ),
  zzvolcoord( "ZZVOL", Expr::STATE_NONE ),
  
  // momentum related variables
  
  pressure  ("pressure",   Expr::STATE_NONE  ),
  dilatation("dilatation", Expr::STATE_NONE),
  tauxx("tau_xx", Expr::STATE_NONE),
  tauxy("tau_xy", Expr::STATE_NONE),
  tauxz("tau_xz", Expr::STATE_NONE),
  tauyx("tau_yx", Expr::STATE_NONE),
  tauyy("tau_yy", Expr::STATE_NONE),
  tauyz("tau_yz", Expr::STATE_NONE),
  tauzx("tau_zx", Expr::STATE_NONE),
  tauzy("tau_zy", Expr::STATE_NONE),
  tauzz("tau_zz", Expr::STATE_NONE),
  
  // turbulence related
  turbulentviscosity( "TurbulentViscosity", Expr::STATE_NONE            ),
  straintensormag   ( "StrainTensorMagnitude", Expr::STATE_NONE         ),
  vremantensormag   ( "VremanTensorMagnitude", Expr::STATE_NONE         ),
  waletensormag     ( "WaleTensorMagnitude", Expr::STATE_NONE           ),
  dynamicsmagcoef   ( "DynamicSmagorinskyCoefficient", Expr::STATE_NONE )
  
  {}
  
  //------------------------------------------------------------------
  
  const TagNames&
  TagNames::self()
  {
    static const TagNames s;
    return s;
  }
  
  //------------------------------------------------------------------
  
} // namespace Wasatch
