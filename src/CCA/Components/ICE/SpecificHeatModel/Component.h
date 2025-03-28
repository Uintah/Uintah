/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef _COMPONENTSPECIFICHEAT_H_
#define _COMPONENTSPECIFICHEAT_H_

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/MaterialManagerP.h>

namespace Uintah {

// Citation:
//   'Coefficients for Calculating Thermodynamic and Transport Properties of Individual Species', 
//    NASA Report TM-4513, by B.J. McBride, S. Gordon and M.A. Reno, October 1993.

class ComponentCv : public SpecificHeat {
public:
  ComponentCv(ProblemSpecP& ps);
  ~ComponentCv();

  virtual void outputProblemSpec(ProblemSpecP& ice_ps);

  virtual double getGamma(double T);

  virtual double getInternalEnergy(double T);

  //__________________________________
  //        Wrappers around the implentation
  virtual void
  computeSpecificHeat(CellIterator        & iter,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & cv)
  {
    computeSpecificHeat_impl<CCVariable<double>>( iter, temp_CC, cv);       
  }
             
  virtual void
  computeSpecificHeat(CellIterator             & iter,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & cv)
  {
    computeSpecificHeat_impl<constCCVariable<double>>( iter, temp_CC, cv);   
  }
  
protected:
  double d_fractionCO2;
  double d_fractionH2O;
  double d_fractionCO;
  double d_fractionH2;
  double d_fractionO2;
  double d_fractionN2;
  double d_fractionOH;
  double d_fractionNO;
  double d_fractionO;
  double d_fractionH;

  double d_sum;  // sum of fractions
  double d_gamma;  // mixture adiabatic index
  double d_massPerMole; // mixture mass per mole

  template< class CCVar>
  void computeSpecificHeat_impl( CellIterator      & iter,       
                                 CCVar             & temp_CC,
                                 CCVariable<double>& cv);    
  
};

}

#endif /* _COMPONENTSPECIFICHEAT_H_ */

