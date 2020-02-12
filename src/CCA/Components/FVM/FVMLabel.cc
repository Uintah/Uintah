/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/FVM/FVMLabel.h>

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

using namespace Uintah;

FVMLabel::FVMLabel()
{

  ccESPotential       = VarLabel::create("cc.esPotential",
	                      CCVariable<double>::getTypeDescription());
  ccESPotentialMatrix = VarLabel::create("cc.esPotentialMatrix",
	                      CCVariable<Stencil7>::getTypeDescription());
  ccRHS_ESPotential   = VarLabel::create("cc.rhsEsPotential",
	                      CCVariable<double>::getTypeDescription());
  ccConductivity      = VarLabel::create("cc.Conductivity",
                        CCVariable<double>::getTypeDescription());
  ccGridConductivity  = VarLabel::create("cc.GridConductivity",
                          CCVariable<double>::getTypeDescription());
  ccCurrent           = VarLabel::create("cc.Current",
                          CCVariable<Vector>::getTypeDescription());
  ccPosCharge         = VarLabel::create("cc.PosCharge",
                          CCVariable<double>::getTypeDescription());
  ccNegCharge         = VarLabel::create("cc.NegCharge",
                          CCVariable<double>::getTypeDescription());
  ccTotalCharge       = VarLabel::create("cc.TotalCharge",
                            CCVariable<double>::getTypeDescription());
  ccPosChargeDensity  = VarLabel::create("cc.Charge_Density1",
                          CCVariable<double>::getTypeDescription());
  ccNegChargeDensity  = VarLabel::create("cc.Charge_Density2",
                          CCVariable<double>::getTypeDescription());
  ccPermittivity      = VarLabel::create("cc.Permittivity",
                          CCVariable<double>::getTypeDescription());

  fcxConductivity     = VarLabel::create("fcx.Conductivity",
                        SFCXVariable<double>::getTypeDescription());
  fcyConductivity     = VarLabel::create("fcy.Conductivity",
                        SFCYVariable<double>::getTypeDescription());
  fczConductivity     = VarLabel::create("fcz.Conductivity",
                        SFCZVariable<double>::getTypeDescription());

  fcxConcentration    = VarLabel::create("fcx.Concentration",
                        SFCXVariable<double>::getTypeDescription());
  fcyConcentration    = VarLabel::create("fcy.Concentration",
                        SFCYVariable<double>::getTypeDescription());
  fczConcentration    = VarLabel::create("fcz.Concentration",
                        SFCZVariable<double>::getTypeDescription());

  fcxMass             = VarLabel::create("fcx.Mass",
                        SFCXVariable<double>::getTypeDescription());
  fcyMass             = VarLabel::create("fcy.Mass",
                        SFCYVariable<double>::getTypeDescription());
  fczMass             = VarLabel::create("fcz.Mass",
                        SFCZVariable<double>::getTypeDescription());
}

FVMLabel::~FVMLabel()
{
  VarLabel::destroy(ccESPotential);
  VarLabel::destroy(ccESPotentialMatrix);
  VarLabel::destroy(ccRHS_ESPotential);
  VarLabel::destroy(ccConductivity);
  VarLabel::destroy(ccGridConductivity);
  VarLabel::destroy(ccCurrent);
  VarLabel::destroy(ccPosCharge);
  VarLabel::destroy(ccNegCharge);
  VarLabel::destroy(ccTotalCharge);
  VarLabel::destroy(ccPosChargeDensity);
  VarLabel::destroy(ccNegChargeDensity);
  VarLabel::destroy(ccPermittivity);
  VarLabel::destroy(fcxConductivity);
  VarLabel::destroy(fcyConductivity);
  VarLabel::destroy(fczConductivity);
  VarLabel::destroy(fcxConcentration);
  VarLabel::destroy(fcyConcentration);
  VarLabel::destroy(fczConcentration);
  VarLabel::destroy(fcxMass);
  VarLabel::destroy(fcyMass);
  VarLabel::destroy(fczMass);
}

