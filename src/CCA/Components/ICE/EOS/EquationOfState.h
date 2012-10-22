/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __EQUATION_OF_STATE_H__
#define __EQUATION_OF_STATE_H__

#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah {

  class Patch;
  class ICEMaterial;

/**************************************

CLASS
   EquationOfState
   
   Short description...

GENERAL INFORMATION

   EquationOfState.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Equation_of_State

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class EquationOfState {
  public:
    EquationOfState();
    virtual ~EquationOfState();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    // Per cell

     virtual double computeRhoMicro(double press,double gamma,
                                    double cv, double Temp, double rho_guess)=0;

     virtual void computePressEOS(double rhoM, double gamma,
                                  double cv, double Temp,
                                  double& press, double& dp_drho, 
                                  double& dp_de) = 0;

    virtual void computeTempCC(const Patch* patch,
                               const string& comp_domain,
                               const CCVariable<double>& press, 
                               const CCVariable<double>& gamma,
                               const CCVariable<double>& cv,
                               const CCVariable<double>& rho_micro, 
                               CCVariable<double>& Temp,
                               Patch::FaceType face)=0;

     virtual double getAlpha(double temp,double sp_v,double P, double cv)=0;


     virtual void hydrostaticTempAdjustment(Patch::FaceType face, 
                                            const Patch* patch,
                                            Iterator& bound_ptr,
                                            Vector& gravity,
                                            const CCVariable<double>& gamma,
                                            const CCVariable<double>& cv,
                                            const Vector& dx,
                                            CCVariable<double>& Temp_CC)=0;
  };
} // End namespace Uintah
      
#endif  // __EQUATION_OF_STATE_H__

