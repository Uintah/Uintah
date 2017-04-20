/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_FVM_FVMLABEL_H
#define UINTAH_CCA_COMPONENTS_FVM_FVMLABEL_H

#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah{
  class FVMLabel {
    public:
	    FVMLabel();
	    ~FVMLabel();

      const VarLabel* ccESPotential;
      const VarLabel* ccESPotentialMatrix;
      const VarLabel* ccRHS_ESPotential;
      const VarLabel* ccConductivity;
      const VarLabel* ccGridConductivity;
      const VarLabel* ccCurrent;
      const VarLabel* ccPosCharge;
      const VarLabel* ccNegCharge;
      const VarLabel* ccTotalCharge;
      const VarLabel* ccPosChargeDensity;
      const VarLabel* ccNegChargeDensity;
      const VarLabel* ccPermittivity;
      const VarLabel* fcxConductivity;
      const VarLabel* fcyConductivity;
      const VarLabel* fczConductivity;
      const VarLabel* fcxConcentration;
      const VarLabel* fcyConcentration;
      const VarLabel* fczConcentration;
      const VarLabel* fcxMass;
      const VarLabel* fcyMass;
      const VarLabel* fczMass;
  };
} // End Uintah namespace

#endif
