#ifndef __IDEAL_GAS_H__
#define __IDEAL_GAS_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   IdealGas.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

      class IdealGas : public EquationOfState {
      public:

       IdealGas(ProblemSpecP& ps);
       virtual ~IdealGas();
        
        // Per cell

        virtual double computeRhoMicro(double press,double gamma,
                                       double cv, double Temp);
         
        virtual void computePressEOS(double rhoM, double gamma,
                                     double cv, double Temp,
                                     double& press, double& dp_drho,
                                     double& dp_de);

        virtual void computeTempCC(const Patch* patch,
                               const string& comp_domain,
                               const CCVariable<double>& press, 
                               const double& gamma,
                               const double& cv,
                               const CCVariable<double>& rho_micro, 
                               CCVariable<double>& Temp,
                               Patch::FaceType face=Patch::xplus);
       
        virtual double getAlpha(double Temp,double sp_vol,double P, double cv);

        virtual void hydrostaticTempAdjustment(Patch::FaceType face,
                                          const Patch* patch,
                                          Vector& gravity,
                                          const double& gamma,
                                          const double& cv,
                                          const Vector& dx,
                                          CCVariable<double>& Temp_CC);
      };
} // End namespace Uintah
      
#endif  // __IDEAL_GAS_H__


