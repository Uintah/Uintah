#ifndef __JWL_H__
#define __JWL_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   A version of the JWL equation of state, as described in
   (for example) "Sideways plate push test for Detonating Solid Explosives"
   Craig M. Tarver, et al, Propellants, Explosives, Pyrotechnics, 21,
   238-246, 1996.

GENERAL INFORMATION

   JWL.h

   Jim Guilkey
   Department of Mechanical Engineerng
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State JWL

DESCRIPTION
   Long description...
  
WARNING
****************************************/

      class JWL : public EquationOfState {
      public:

        JWL(ProblemSpecP& ps);
        virtual ~JWL();
        
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
       

        virtual double getAlpha(double Temp,double sp_vol, double P, double cv);
         
        virtual void hydrostaticTempAdjustment(Patch::FaceType face,
                                               const Patch* patch,
                                               Vector& gravity,
                                               const double& gamma,
                                               const double& cv,
                                               const Vector& dx,
                                               CCVariable<double>& Temp_CC);

      private:
        double   A;   // Pascals
        double   B;   // Pascals
        double   R1;
        double   R2;
        double   om;
        double   rho0;  // kg/m^3
      };
} // End namespace Uintah
      
#endif  // __JWL_H__


