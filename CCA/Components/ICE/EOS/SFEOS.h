#ifndef __SFEOS_H__
#define __SFEOS_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
GENERAL INFORMATION

   SFEOS.h

   Jim Guilkey
   Department of Mechanical Engineerng
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

   Martin Denison
   Reaction Engineering International

KEYWORDS
   Equation_of_State Soil Foam

DESCRIPTION
   This is an implementation of the Soil and Foam loading pressure curve as EOS
   The curve exists as it does in the Soil and Foam constitutive model as 
   linear curve segements of pressure versus volumetric strain.
  
WARNING
****************************************/

      class SFEOS : public EquationOfState {
      public:

        SFEOS(ProblemSpecP& ps);
        virtual ~SFEOS();

        virtual void outputProblemSpec(ProblemSpecP& ps);
        
        virtual double computeRhoMicro(double press,double gamma,
                                      double cv, double Temp, double rho_guess);
         
        virtual void computePressEOS(double rhoM, double gamma,
                                     double cv, double Temp,
                                     double& press, double& dp_drho,
                                     double& dp_de);

        virtual void computeTempCC(const Patch* patch,
                                   const string& comp_domain,
                                   const CCVariable<double>&, 
                                   const CCVariable<double>&,
                                   const CCVariable<double>&,
                                   const CCVariable<double>&, 
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
        double eps[10], p[10], lnp[10];
        double slope[9];
        double   rho0;  // kg/m^3

      };
} // End namespace Uintah
      
#endif  // __TILLOTSON_H__
