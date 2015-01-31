/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef UINTAH_SMAGORINSKYMODEL_H
#define UINTAH_SMAGORINSKYMODEL_H

#include <CCA/Components/ICE/TurbulenceModel/Turbulence.h>
#include <Core/Containers/StaticArray.h>
#include <cmath>

namespace Uintah {

  class Smagorinsky_Model : public Turbulence {

  public:
    Smagorinsky_Model(ProblemSpecP& ps, SimulationStateP& sharedState);
    Smagorinsky_Model();
    
    virtual ~Smagorinsky_Model();
    
    friend class DynamicModel;
    
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
                                      CCVariable<double>& turb_viscosity);    
                                         
    virtual void scheduleComputeVariance(SchedulerP& sched, 
                                         const PatchSet* patches,
                                         const MaterialSet* matls);
    

  private:
//    double d_filter_width;
    double d_model_constant;

    void computeStrainRate(const Patch* patch,
                           const SFCXVariable<double>& uvel_FC,
                           const SFCYVariable<double>& vvel_FC,
                           const SFCZVariable<double>& wvel_FC,
                           const int indx,
                           SimulationStateP&  d_sharedState,
                           DataWarehouse* new_dw,
                           SCIRun::StaticArray<CCVariable<double> >& SIJ);
                           
    void computeVariance(const ProcessorGroup*, 
                         const PatchSubset* patch,  
                         const MaterialSubset* matls,
                         DataWarehouse*, 
                         DataWarehouse*,
                         FilterScalar*);
    
    };// End class

}// End namespace Uintah
#endif
