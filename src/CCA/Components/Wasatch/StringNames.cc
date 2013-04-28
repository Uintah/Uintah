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
#include "StringNames.h"

namespace Wasatch{

  //------------------------------------------------------------------

  StringNames::StringNames() :

    time("time"),
    timestep("timestep"),

    xsvolcoord("XSVOL"),  ysvolcoord("YSVOL"),  zsvolcoord("ZSVOL"),
    xxvolcoord("XXVOL"),  yxvolcoord("YXVOL"),  zxvolcoord("ZXVOL"),
    xyvolcoord("XYVOL"),  yyvolcoord("YYVOL"),  zyvolcoord("ZYVOL"),
    xzvolcoord("XZVOL"),  yzvolcoord("YZVOL"),  zzvolcoord("ZZVOL"),

    // energy related variables
    temperature("temperature"),
    e0("total internal energy"),
    rhoE0("rhoE0"),
    enthalpy("enthalpy"),
    xHeatFlux("heatFlux_x"),
    yHeatFlux("heatFlux_y"),
    zHeatFlux("heatFlux_z"),

    // species related variables
    species("species"),
    rhoyi("rhoy"),
    mixtureFraction("mixture fraction"),

    // thermochemistry related variables
    heatCapacity("heat capacity"),
    thermalConductivity("thermal conductivity"),
    viscosity("viscosity"),

    // momentum related variables
    xvel("x-velocity"),
    yvel("y-velocity"),
    zvel("z-velocity"),
    xmom("x-momentum"),
    ymom("y-momentum"),
    zmom("z-momentum"),
    pressure("pressure"),
    dilatation("dilatation"),
    tauxx("tau_xx"),
    tauxy("tau_xy"),
    tauxz("tau_xz"),
    tauyx("tau_yx"),
    tauyy("tau_yy"),
    tauyz("tau_yz"),
    tauzx("tau_zx"),
    tauzy("tau_zy"),
    tauzz("tau_zz"),
  
    // turbulence related
    turbulentviscosity("TurbulentViscosity"),
    straintensormag("StrainTensorMagnitude"),
    vremantensormag("VremanTensorMagnitude"),
    waletensormag("WaleTensorMagnitude"),
    dynamicsmagcoef("DynamicSmagorinskyCoefficient")

  {}

  //------------------------------------------------------------------

  const StringNames&
  StringNames::self()
  {
    static const StringNames s;
    return s;
  }

  //------------------------------------------------------------------

} // namespace Wasatch
