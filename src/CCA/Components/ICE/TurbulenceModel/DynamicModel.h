/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_DYNAMICMODEL_H
#define UINTAH_DYNAMICMODEL_H

#include <CCA/Components/ICE/TurbulenceModel/Turbulence.h>
#include <CCA/Components/ICE/TurbulenceModel/SmagorinskyModel.h>

namespace Uintah {


  class DynamicModel : public Turbulence{

  public:
  
    DynamicModel(ProblemSpecP& ps, MaterialManagerP& materialManager);
    virtual ~DynamicModel();
    
    virtual void computeTurbViscosity(DataWarehouse* new_dw,
                                      const Patch* patch,
                                      const ICELabel* lb,               
                                      constCCVariable<Vector>& vel_CC,         
                                      constSFCXVariable<double>& uvel_FC,      
                                      constSFCYVariable<double>& vvel_FC,      
                                      constSFCZVariable<double>& wvel_FC,      
                                      constCCVariable<double>& rho_CC,         
                                      const int indx,                           
                                      MaterialManagerP&  d_materialManager,         
                                      CCVariable<double>& turb_viscosity);      
             
    virtual void scheduleComputeVariance(SchedulerP& sched, 
                                         const PatchSet* patches,
                                         const MaterialSet* matls);

  private:
    
    void computeSmagCoeff(DataWarehouse* new_dw,
                          const Patch* patch,
                          const ICELabel* lb,                          
                          constCCVariable<Vector>& vel_CC,                  
                          constSFCXVariable<double>& uvel_FC,               
                          constSFCYVariable<double>& vvel_FC,               
                          constSFCZVariable<double>& wvel_FC,               
                          const int indx,                                    
                          MaterialManagerP&  d_materialManager,                  
                          CCVariable<double>& term,                          
                          CCVariable<double>& meanSIJ);                      

    template <class T, class V> 
    void applyFilter(const Patch* patch, 
                     T& var,
                     CCVariable<V>& var_hat);      
       
    void applyFilter(const Patch* patch,
                     std::vector<CCVariable<double> >& var,           
                     std::vector<CCVariable<double> >& var_hat);      
                                                                
    Smagorinsky_Model d_smag;
    
    double filter_width;
    double d_model_constant;   

    void computeVariance(const ProcessorGroup*, 
                         const PatchSubset* patch,  
                         const MaterialSubset* matls,
                         DataWarehouse*, 
                         DataWarehouse*,
                         FilterScalar*);
    
    };// End class

}// End namespace Uintah
#endif
