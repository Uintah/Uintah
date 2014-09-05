#ifndef __HARD_SPHERE_GAS_H__
#define __HARD_SPHERE_GAS_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState

   Hard sphere equation of state approximates the non-ideal properties of
   a real gas by accounting for a fixed volume of gas occupied by the 
   molecules themselves:

        RT
   P = -----   
        v-b

   P = pressure (Pa)
   R = gas constant = (gamma-1)*cv  (J/kg/K)
   gamma = heat capacity ratio = cp/cv (dimensionless) 
   cv = const volume specific heat (J/kg/K)
   cp = const pressure specific heat (J/kg/K)
   v = specific volume (m^3/kg)
   b = molecular volume (m^3/kg)

GENERAL INFORMATION

   HardSphereGas.h

   Chuck Wight
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2007 SCI Group

KEYWORDS
   Equation_of_State Hard Sphere

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class HardSphereGas : public EquationOfState {
  public:

   HardSphereGas(ProblemSpecP& ps);
   virtual ~HardSphereGas();

   virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual double computeRhoMicro(double press,double gamma,
                                   double cv, double Temp, double rho_guess);

    virtual void computePressEOS(double rhoM, double gamma,
                                 double cv, double Temp,
                                 double& press, double& dp_drho,
                                 double& dp_de);

    virtual void computeTempCC(const Patch* patch,
                               const string& comp_domain,
                               const CCVariable<double>& press, 
                               const CCVariable<double>& gamma,
                               const CCVariable<double>& cv,
                               const CCVariable<double>& rho_micro, 
                               CCVariable<double>& Temp,
                               Patch::FaceType face);

    virtual double getAlpha(double Temp,double sp_vol,double P, double cv);

    virtual void hydrostaticTempAdjustment(Patch::FaceType face, 
                                     const Patch* patch,
                                     const vector<IntVector>* bound_ptr,
                                     Vector& gravity,
                                     const CCVariable<double>& gamma,
                                     const CCVariable<double>& cv,
                                     const Vector& dx,
                                     CCVariable<double>& Temp_CC);
      private:
        double   b;

  };
} // End namespace Uintah
      
#endif  // __HARD_SPHERE_GAS_H__


