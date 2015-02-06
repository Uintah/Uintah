/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __TILLOTSON_H__
#define __TILLOTSON_H__

#include <Core/Grid/Variables/CCVariable.h>
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

        Tillotson(ProblemSpecP& ps);
        virtual ~Tillotson();

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
                                               Iterator&,
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
