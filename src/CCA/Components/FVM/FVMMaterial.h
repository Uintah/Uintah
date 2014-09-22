/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef __FVM_MATERIAL_H__
#define __FVM_MATERIAL_H__


#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/Material.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {
  using namespace SCIRun;
 
class FVMMaterial : public Material {
  public:
    FVMMaterial(ProblemSpecP&, 
                SimulationStateP& sharedState);

    ~FVMMaterial();


    double getDiffusivity() const;
		double getConcentration() const;

  private:
    double d_diffusivity;
		double d_concentration;

    FVMMaterial(const FVMMaterial &fvmm);
    FVMMaterial& operator=(const FVMMaterial &fvmm);        

};
} // End namespace Uintah

#endif // __FVM_MATERIAL_H__





