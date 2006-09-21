#ifndef __TST_GAS_H__
#define __TST_GAS_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   TSTGas.h

   Liping Xue
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2006 

KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
   Twe-Sim-Tassone gas 
  
WARNING
  
****************************************/

  class TSTGas : public EquationOfState {
  public:

   TSTGas(ProblemSpecP& ps);
   virtual ~TSTGas();

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
                                     const vector<IntVector>& bound,
                                     Vector& gravity,
                                     const CCVariable<double>& gamma,
                                     const CCVariable<double>& cv,
                                     const Vector& dx,
                                     CCVariable<double>& Temp_CC);
    double getGasConstant() const;

  private:
    double a;
    double b;
    double u;
    double w;
    double d_gas_constant;
  };
} // End namespace Uintah
      
#endif  // __TST_GAS_H__


