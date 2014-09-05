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

#ifndef _POLYNOMIALSPECIFICHEAT_H_
#define _POLYNOMIALSPECIFICHEAT_H_

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>

//  A generalized Polynomial form for specific heat that asymptotes to a upper bound
//   where:
//
//   Cv(T) = SUM(a_i * x^i)/x^m 
//           from i=0 to m
//           where m is the maximum order
//
// Inputs include the maximum order, coefficients up to that order (in logical order
//  in the input file) and an optional minimum and maximum temperature to clamp
//  specific heat dependence range.


namespace Uintah {

class PolynomialCv : public SpecificHeat {
public:
  PolynomialCv(ProblemSpecP& ps);
  ~PolynomialCv();

  virtual void outputProblemSpec(ProblemSpecP& ice_ps);

  virtual double getSpecificHeat(double T);

  virtual double getGamma(double T);

  virtual double getInternalEnergy(double T);

protected:
  
  double d_Tmin;  // Clamp minimum temperature; default 0
  double d_Tmax;  // Clamp maximum temperature; default 1e6

  int d_maxOrder;                    // Maximum order of the polynomial
  std::vector<double> d_coefficient; // Coefficient to each monomer order with order=index

};

}

#endif /* _POLYNOMIALSPECIFICHEAT_H_ */


