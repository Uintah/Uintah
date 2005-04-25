#ifndef __JWL_H__
#define __JWL_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
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
                                      double cv, double Temp, double rho_guess);
         
        virtual void computePressEOS(double rhoM, double gamma,
                                     double cv, double Temp,
                                     double& press, double& dp_drho,
                                     double& dp_de);

        virtual void computeTempCC(const Patch* patch,
                                   const string& comp_domain,
                                   const CCVariable<double>& press, 
                                   const CCVariable<double>&,
                                   const CCVariable<double>& cv,
                                   const CCVariable<double>& rhoM, 
                                   CCVariable<double>& Temp,
                                   Patch::FaceType face);
       

        virtual double getAlpha(double Temp,double sp_vol, double P, double cv);
         
        virtual void hydrostaticTempAdjustment(Patch::FaceType, 
                                               const Patch*,
                                               const vector<IntVector>&,
                                               Vector&,
                                               const CCVariable<double>&,
                                               const CCVariable<double>&,
                                               const Vector&,
                                               CCVariable<double>&);

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


