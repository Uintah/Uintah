#ifndef UINTAH_SMAGORINSKYMODEL_H
#define UINTAH_SMAGORINSKYMODEL_H

#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Core/Containers/StaticArray.h>
#include <math.h>

namespace Uintah {


  class Smagorinsky_Model : public Turbulence {

  public:
    //----- constructors
    Smagorinsky_Model(ProblemSpecP& ps);
    Smagorinsky_Model();
    
    //----- destructor
    virtual ~Smagorinsky_Model();
    
    friend class DynamicModel;
    
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
  
  private:

    double filter_width;
    double d_model_constant;
//    double d_turbPr; // turbulent prandtl number
    
    void computeStrainRate(const Patch* patch,
                           const SFCXVariable<double>& uvel_FC,
                           const SFCYVariable<double>& vvel_FC,
                           const SFCZVariable<double>& wvel_FC,
                           const int indx,
                           SimulationStateP&  d_sharedState,
                           StaticArray<CCVariable<double> >& SIJ);
    
    };// End class

}// End namespace Uintah
#endif
