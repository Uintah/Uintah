/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_DYNAMICMODEL_H
#define UINTAH_DYNAMICMODEL_H

#include <CCA/Components/ICE/Turbulence.h>
#include <CCA/Components/ICE/SmagorinskyModel.h>
#include <Core/Containers/StaticArray.h>
#include <cmath>

namespace Uintah {


  class DynamicModel : public Turbulence{

  public:
  
    // ------ constructors
    DynamicModel(ProblemSpecP& ps, SimulationStateP& sharedState);
    
    // ------ destructor
    virtual ~DynamicModel();
    
    virtual void computeTurbViscosity(DataWarehouse* new_dw,
                                            const Patch* patch,
                                            const CCVariable<Vector>& vel_CC,
                                            const SFCXVariable<double>& uvel_FC,
                                            const SFCYVariable<double>& vvel_FC,
                                            const SFCZVariable<double>& wvel_FC,
                                            const CCVariable<double>& rho_CC,
                                            const int indx,
                                            SimulationStateP&  d_sharedState,
                                            CCVariable<double>& turb_viscosity);
             
    virtual void scheduleComputeVariance(SchedulerP& sched, 
                                         const PatchSet* patches,
                                         const MaterialSet* matls);

  private:
    
    void computeSmagCoeff(DataWarehouse* new_dw,
                                         const Patch* patch,
                                         const CCVariable<Vector>& vel_CC,
                                         const SFCXVariable<double>& uvel_FC,
                                         const SFCYVariable<double>& vvel_FC,
                                         const SFCZVariable<double>& wvel_FC,
                                         const int indx,
                                         SimulationStateP&  d_sharedState,
                                         CCVariable<double>& term,
                                         CCVariable<double>& meanSIJ);
       
    template <class T>
    void applyFilter(const Patch* patch,
                           CCVariable<T>& var,
                           CCVariable<T>& var_hat);
       
    void applyFilter(const Patch* patch,
                           SCIRun::StaticArray<CCVariable<double> >& var,
                           SCIRun::StaticArray<CCVariable<double> >& var_hat);
                                                                
    Smagorinsky_Model d_smag;
    
    double filter_width;
    double d_test_filter_width;
    double d_model_constant;
//    double d_turbPr; // turbulent prandtl number   

    void computeVariance(const ProcessorGroup*, 
                         const PatchSubset* patch,  
                         const MaterialSubset* matls,
                         DataWarehouse*, 
                         DataWarehouse*,
                         FilterScalar*);
    
    };// End class

}// End namespace Uintah
#endif
