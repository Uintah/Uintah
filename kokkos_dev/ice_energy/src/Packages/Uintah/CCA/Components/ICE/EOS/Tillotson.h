#ifndef __TILLOTSON_H__
#define __TILLOTSON_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
GENERAL INFORMATION

   Tillotson.h

   Jim Guilkey
   Department of Mechanical Engineerng
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State Tillotson

DESCRIPTION
   This is an implementation of the Tillotson EOS as described in
   "Selected Topics in Shock Wave Physics and Equation of State Modeling"
   by G. Roger Gathers.  This is a model that is often used for soil.
  
WARNING
****************************************/

      class Tillotson : public EquationOfState {
      public:

    Tillotson(ProblemSpecP& ps, ICEMaterial* ice_matl);
        virtual ~Tillotson();
        
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
        double   a;   // non-dimensional
        double   b;   // non-dimensional
        double   A;   // Pascals
        double   B;   // Pascals
        double   E0;  // J/kg
        double   Es;  // J/kg
        double   Esp; // J/kg
        double   alpha;
        double   beta;
        double   rho0;  // kg/m^3

double Pofrho(double rho)
{
  double press;
  double E = .00001034777*Es;
  double eta=rho/rho0;
  double mu=(eta-1.);
  double etasq=eta*eta;
  double musq=mu*mu;

  if(eta>=1.){
     press   = (E*rho)*(a + b/(E/(E0*etasq)+1.)) + A*mu + B*musq;
   }
   else{
     double AA=A*0.;
     double expterm=exp(-alpha*((1./eta-1.)*(1./eta-1.)));
                                                                                
     press = a*E*rho
           + (b*E*rho/(E/(E0*etasq)+1)+AA*mu*exp(-beta*(1./eta-1.)))*expterm;
   }
   return press;
};
      };
} // End namespace Uintah
      
#endif  // __TILLOTSON_H__
