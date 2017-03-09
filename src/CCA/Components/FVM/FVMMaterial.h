/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_FVM_FVMMATERIAL_H
#define UINTAH_CCA_COMPONENTS_FVM_FVMMATERIAL_H

#include <Core/Grid/Material.h>
#include <Core/Grid/Patch.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <vector>

namespace Uintah {
  class FVMMaterial : public Material {
    public:
      FVMMaterial(ProblemSpecP& ps, SimulationStateP& shared_state);
      ~FVMMaterial();

      void initializeConductivity(CCVariable<double>& conductivity, const Patch* patch);
      void initializePermitivityAndCharge(CCVariable<double>& permitivity,
                                          CCVariable<double>& charge,
                                          const Patch* patch);

    private:
       std::vector<GeometryObject*> d_geom_objs;

       // Prevent copying of this class
       // copy constructor
       FVMMaterial(const FVMMaterial &fvmm);
       FVMMaterial& operator=(const FVMMaterial &fvmm);
  };

} // End namespace Uintah

#endif // UINTAH_CCA_COMPONENTS_FVM_FVMMATERIAL_H
