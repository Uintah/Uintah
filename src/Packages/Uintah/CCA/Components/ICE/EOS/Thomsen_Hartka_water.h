#ifndef __Thomsen_Hartka_water_H__
#define __Thomsen_Hartka_water_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
GENERAL INFORMATION

   Equation of state for 'cold water' in the 1-100 atm pressure range.

   g(T,P) = .(co + b*To)*T*ln(T/To) + (co + b*To)*(T . To) + (1/2)*b*(T . To)^2 
   + vo*[P .(1/2)*ko*P^2] + .*vo*P*[(T . To)^2 + a*P*(T.To) + (1/3)*(a^2)*(P^2)]

   a = 2*10^-7          (K/Pa)
   b = 2.6              (J/kgK^2)
   co = 4205.7          (J/kgK)
   ko = 5*10^-10        (1/Pa)
   To = 277             (K)
   L = 8*10^-6          (1/K^2)
   vo = 1.00008*10^-3   (m^3/kg)

   Reference: 
   Adrian Bejan, 1988 Advanced Engineering Thermodynamics, pgs. 724-725.
   
   Original Reference:
   Thomsen, J.S. and Hartka, T.J., 1962, 
   Strange Carnot cycles; thermodynamics of a system 
   with a density extremum, Am. J. Phys. (30) 26-33.

KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class Thomsen_Hartka_water : public EquationOfState {
  public:

   Thomsen_Hartka_water(ProblemSpecP& ps);
   virtual ~Thomsen_Hartka_water();

   virtual void outputProblemSpec(ProblemSpecP& ps);

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

    virtual double getAlpha(double Temp,double sp_vol,double P, double cv);

    virtual void hydrostaticTempAdjustment(Patch::FaceType face, 
                                           const Patch* patch,
                                           const vector<IntVector>* bound_ptr,
                                           Vector& gravity,
                                           const CCVariable<double>& gamma,
                                           const CCVariable<double>& cv,
                                           const Vector& dx,
                                           CCVariable<double>& Temp_CC);
     private:
       double   d_a;
       double   d_b;
       double   d_co;
       double   d_ko;
       double   d_To;
       double   d_L;
       double   d_vo;
  };
} // End namespace Uintah
      
#endif  // __Thomsen_Hartka_water_H__


