#ifndef __Murnahan_H__
#define __Murnahan_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   A version of the Murnahan equation of state, as described in
   "JWL++:  A Simple Reactive Flow Code Package for Detonation"
   P. Clark Souers, Steve Anderson, James Mercer, Estella McGuire and
   Peter Vitello, Propellants, Explosives and Pyrotechnics, 25, 54-58, 2000.

GENERAL INFORMATION

   Murnahan.h

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

      class Murnahan : public EquationOfState {
      public:

        Murnahan(ProblemSpecP& ps);
        virtual ~Murnahan();
        
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
        double   n;
        double   K;     // 1/Pascals
        double   rho0;  // kg/m^3
        double   P0;    // Pascals
      };
} // End namespace Uintah
      
#endif  // __Murnahan_H__
