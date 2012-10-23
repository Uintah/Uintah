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

#ifndef __JWL_H__
#define __JWL_H__

#include <Core/Grid/Variables/CCVariable.h>
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
        double   A;   // Pascals
        double   B;   // Pascals
        double   R1;
        double   R2;
        double   om;
        double   rho0;  // kg/m^3
	
	/* The following used only in JWL::computeRhoMicro 

	   Modified by:
	   Changwei Xiong
	   Department of Chemistry 
	   University of Utah
	*/
        typedef struct { 
	  double   Pressure;
	  double   Temperature;
	  double   SpecificHeat;
	  double   IL, IR;
        } IterationVariables;

	double func(double rhoM, IterationVariables *);
	double deri(double rhoM, IterationVariables *);
	void   setInterval(double f, double rhoM, IterationVariables *);
      };
} // End namespace Uintah
      
#endif  // __JWL_H__


