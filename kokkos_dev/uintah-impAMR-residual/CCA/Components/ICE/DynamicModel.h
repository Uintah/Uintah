#ifndef UINTAH_DYNAMICMODEL_H
#define UINTAH_DYNAMICMODEL_H

#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Packages/Uintah/CCA/Components/ICE/SmagorinskyModel.h>
#include <Core/Containers/StaticArray.h>
#include <math.h>

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
             
    virtual void scheduleTurbulence1(SchedulerP& sched, const PatchSet* patches,
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
