#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>

namespace Uintah {

  class Task;
  class Patch;
  class VarLabel;
  class ICEMaterial;

/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   EquationOfState.h

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

      class EquationOfState {
      public:
        EquationOfState();
        virtual ~EquationOfState();
        
        // Per cell

         virtual double computeRhoMicro(double press,double gamma,
                                        double cv, double Temp) =0;

         virtual void computePressEOS(double rhoM, double gamma,
                                      double cv, double Temp,
                                      double& press, double& dp_drho, 
                                      double& dp_de) = 0;

        virtual void computeTempCC(const Patch* patch,
                               const string& comp_domain,
                               const CCVariable<double>& press, 
                               const double& gamma,
                               const double& cv,
                               const CCVariable<double>& rho_micro, 
                               CCVariable<double>& Temp,
                               Patch::FaceType face=Patch::xplus) = 0;
       

        virtual void hydrostaticTempAdjustment(Patch::FaceType face,
                                          const Patch* patch,
                                          Vector& gravity,
                                          const double& gamma,
                                          const double& cv,
                                          const Vector& dx,
                                          CCVariable<double>& Temp_CC)=0;
      };
} // End namespace Uintah
      
#endif  // __EQUATION_OF_STATE_H__

