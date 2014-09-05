/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __BirchMurnaghan_H__
#define __BirchMurnaghan_H__

#include <Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   A version of the Birch-Murnaghan equation of state, as described in:
    Birch, Francis. "Finite Elastic Strin of Cubic Crystals." Physical Review, 71 (11), 1947, 809-824.

   The Debye temperature model and specific heat model are described in:
    Sewell, T.D. and Menikoff, R. "Complete Equation of State for Beta-HMX and Implications for Initiation".
    Shock Compression o fCondensed Matter, 2003.
GENERAL INFORMATION

   BirchMurnaghan.h

   Joseph Peterson
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Equation_of_State Birch Murnaghan

DESCRIPTION
   
   EOS:
    P(V)=3/(2*K) * [v^(-7/3)-v^(-5/3)] * [1 + 3/4*(n-4) * (v^(-2/3) - 1)]

    where v is the reduced volume, V/V0.


   Debye Model:
    theta(V) = (V0/V)^a*exp(b*(V0-V)/V)


   Specific Heat Model:
    Cv(T)=T^3/(c0 + c1*T + c2*T^2 + c3*T^3)

    where T is the unitless temperature.
  
WARNING
****************************************/

      class BirchMurnaghan : public EquationOfState {
      public:

        BirchMurnaghan(ProblemSpecP& ps);
        virtual ~BirchMurnaghan();

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
       

        virtual double getAlpha(double Temp,double sp_vol, double P, double cv);
         
        virtual void hydrostaticTempAdjustment(Patch::FaceType, 
                                               const Patch*,
                                               Iterator& ,
                                               Vector&,
                                               const CCVariable<double>&,
                                               const CCVariable<double>&,
                                               const Vector&,
                                               CCVariable<double>&);

      private:
        // Functions to compute Pressure and Pressure derivative
        /// @param v the current relative volume
        double computeP(double);
        
        /// @param v the current relative volume
        double computedPdrho(double);


        // Equation of State Parameters
        double   n;     // Bulk modulus pressure derivative
        double   K;     // Compressibility, 1/Bulk Modulus  (1/Pa)
        double   rho0;  // kg/m^3
        double   P0;    // Pascals

        // Specific Heat Model Parameters
        bool useSpecificHeatModel;
        double a;
        double b;
        double c0;   // K*kg/J
        double c1;   // K*kg/J
        double c2;   // K*kg/J
        double c3;   // K*kg/J
      };
} // End namespace Uintah
      
#endif  // __BirchMurnaghan_H__
