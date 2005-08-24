#ifndef __Gruneisen_H__
#define __Gruneisen_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   A version of the Gruneisen equation of state, as described in
   in the Phosphorous Compendium, Nov 02,
   4of6_Phosphorous_EOS/LLNL_Gruneisen_WP_EOS.pdf

GENERAL INFORMATION

   Gruneisen.h

   Jim Guilkey
   Department of Mechanical Engineerng
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State Gruneisen

DESCRIPTION
   EOS of the form P = P0 + A*(rho/rho0 -1) + B(T-T0)
  
WARNING
****************************************/

      class Gruneisen : public EquationOfState {
      public:

    Gruneisen(ProblemSpecP& ps, ICEMaterial* ice_matl);
        virtual ~Gruneisen();
        
        virtual double computeRhoMicro(double press,double gamma,
                                      double cv, double Temp, double rho_guess);
         
        virtual void computePressEOS(double rhoM, double gamma,
                                     double cv, double Temp,
                                     double& press, double& dp_drho,
                                     double& dp_de);

        virtual void computeTempCC(const Patch* patch,
                                   const string& comp_domain,
                                   const CCVariable<double>& P, 
                                   const CCVariable<double>&,
                                   const CCVariable<double>&,
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
        // Units are typical only, any consistent units will work.
        double   A;     // Pascals
        double   B;     // Pascals/K
        double   rho0;  // kg/m^3
        double   P0;    // Pascals
        double   T0;    // Kelvin
      };
} // End namespace Uintah
      
#endif  // __Gruneisen_H_
