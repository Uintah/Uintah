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

#ifndef UINTAH_CCA_COMPONENTS_FVM_FVMBOUNDCOND_H
#define UINTAH_CCA_COMPONENTS_FVM_FVMBOUNDCOND_H

#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

namespace Uintah{

  class Patch;
  
  class FVMBoundCond{
    public:
      FVMBoundCond();
      ~FVMBoundCond();

      void setConductivityBC(const Patch* patch, int dwi, CCVariable<double>& conductivity);

      void setESBoundaryConditions(const Patch* patch, int dwi,
    		                           CCVariable<Stencil7>& A, CCVariable<double>& rhs,
    		                           constSFCXVariable<double>& fcx_conductivity,
    		                           constSFCYVariable<double>& fcy_conductivity,
    		                           constSFCZVariable<double>& fcz_conductivity);

      void setESPotentialBC(const Patch* patch, int dwi,
                            CCVariable<double>& es_potential,
                            constSFCXVariable<double>& fcx_conductivity,
                            constSFCYVariable<double>& fcy_conductivity,
                            constSFCZVariable<double>& fcz_conductivity);

      void setESPotentialBC(const Patch* patch, int dwi,
                            CCVariable<double> &es_potential);

      void setG1BoundaryConditions(const Patch* patch, int dwi,
                                   CCVariable<Stencil7>& A,
                                   CCVariable<double>& rhs);
  };
}
#endif
