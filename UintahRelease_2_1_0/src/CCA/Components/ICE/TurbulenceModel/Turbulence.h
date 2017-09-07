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

#ifndef _TURBULENCE_H
#define _TURBULENCE_H

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class ICELabel;
  class Material;
  class Patch;


  class Turbulence {

  public:
    Turbulence();
    Turbulence(ProblemSpecP& ps, SimulationStateP& sharedState);
    virtual ~Turbulence();

    virtual void computeTurbViscosity(DataWarehouse* new_dw,
                                      const Patch* patch,
                                      const ICELabel* lb,
                                      constCCVariable<Vector>& vel_CC,
                                      constSFCXVariable<double>& uvel_FC,
                                      constSFCYVariable<double>& vvel_FC,
                                      constSFCZVariable<double>& wvel_FC,
                                      constCCVariable<double>& rho_CC,
                                      const int indx,
                                      SimulationStateP&  d_sharedState,
                                      CCVariable<double>& turb_viscosity) = 0;

    virtual void scheduleComputeVariance(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls) = 0;

    void callTurb(DataWarehouse* new_dw,
                 const Patch* patch,
                 constCCVariable<Vector>& vel_CC,
                 constCCVariable<double>& rho_CC,
                 constCCVariable<double>& vol_frac_CC,
                 const int indx,
                 ICELabel* lb,
                 SimulationStateP&  d_sharedState,
                 constCCVariable<double>& molecularVis,
                 CCVariable<double>& tot_viscosity);

    template<class T>
    void setZeroNeumannBC_CC( const Patch* patch,
                              CCVariable<T>& var,
                              const int NGC);
  protected:
    SimulationStateP d_sharedState;
    double d_filter_width;

    struct FilterScalar {
      std::string name;
      double scale;
      const VarLabel* scalar;
      const VarLabel* scalarVariance;
      Material* matl;
      MaterialSet* matl_set;
    };
    std::vector<FilterScalar*> filterScalars;

  };// End class Turbulence

}// End namespace Uintah

#endif
