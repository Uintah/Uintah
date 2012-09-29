/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __KNAUSS_SEA_WATER_H__
#define __KNAUSS_SEA_WATER_H__

#include <Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   KnaussSeaWater.h

   Jim  Guilkey
   Schlumberger Technology Corporation
   Perforating Research

   
KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class KnaussSeaWater : public EquationOfState {
  public:

   KnaussSeaWater(ProblemSpecP& ps);
   virtual ~KnaussSeaWater();

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
                                     Iterator &bound_ptr,
                                     Vector& gravity,
                                     const CCVariable<double>& gamma,
                                     const CCVariable<double>& cv,
                                     const Vector& dx,
                                     CCVariable<double>& Temp_CC);
     private:
                      // Typical values
       double   d_a;  // -0.15 kg/(m^3 C)
       double   d_b;  //  0.78 kg/(m^3 PSU) PSU = Practical Salinity Unit
       double   d_k;  //  4.5e-7 kg/(m^3 Pa)
       double   d_T0; //  10 C = 283.15 K
       double   d_P0; //  101325 Pa
       double   d_S;  //  For 3% KCl, take d_S = 30
       double   d_S0; // 35 PSU
       double   d_rho0; // 1027 kg/m^3
  };
} // End namespace Uintah
      
#endif  // __KNAUSS_SEA_WATER_H__


