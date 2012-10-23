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

#ifndef __KumariDass_H__
#define __KumariDass_H__

#include <Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   KumariDaas 
   
   A version of the KumariDass equation of state, as described in
   M. Kumari and N. Dass. "An equation of state applied to 50 solids." 
   Condensed Matter, 2:3219, 1990. 
   .

GENERAL INFORMATION

   KumariDass.h

   Joseph R. Peterson
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Equation_of_State KumariDass

DESCRIPTION
   The Kumari-Dass equation of state is used for solids and has been applied 
   to various types.  It has similar accuracy to the Vinet equation of state.
   The upside is that unlike many other solid equations of state (e.g. JWL,
   Murnaghan, Birch-Murnaghan) it does not change sign at a relative volume
   of 1.
 
   It is based on the assumption that the bulk modulus, (B), varies with pressure (P)
   as:

   B=B_0+B_0'/lambda*(1-e^(lambda*P))
 
   where (B_0') if the derivative that is pressure independent, and (lambda) is 
   a softening parameter on the linearity of the bulk modulus.
 
WARNING
****************************************/

      class KumariDass : public EquationOfState {
      public:

        KumariDass(ProblemSpecP& ps);
        virtual ~KumariDass();

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
                                   const CCVariable<double>&,
                                   const CCVariable<double>& cv,
                                   const CCVariable<double>& rhoM, 
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

          double lambda;    // Linearity softening parameter
          double B0;        // Reference bulk modulus
          double B0prime;   // Constant pressure bulk modulus pressure 
                            //  derivative at the reference density
          double rho0;      // Reference density

      };
    
} // End namespace Uintah
      
#endif  // __KumariDass_H__


