#ifndef __HARLOW_H__
#define __HARLOW_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   Harlow.h

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

      class Harlow : public EquationOfState {
      public:
        
        Harlow(ProblemSpecP& ps);
        virtual ~Harlow();
        
        // Per cell
        virtual double computeRhoMicro(double press,double gamma,
                                    double cv, double Temp);
        
        virtual void computePressEOS(double rhoM, double gamma,
                                  double cv, double Temp,
                                  double& press, double& dp_drho,
                                  double& dp_de);
        //per patch                          
        virtual void computeTempCC(const Patch* patch,
                                const CCVariable<double>& press, 
                                const double& gamma,
                                const double& cv,
                                const CCVariable<double>& rho_micro, 
                                CCVariable<double>& Temp);


        double getGasConstant() const;

      private:
       double d_gas_constant;
      };
} // End namespace Uintah

#endif  // __HARLOW_H__



