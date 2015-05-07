/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef _DEBYESPECIFICHEAT_H_
#define _DEBYESPECIFICHEAT_H_

#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {

// A standard Debye specific heat model designed for solids based on estimating
// phonon contributions to the vibrational modes in the solid lattice.
//
// Citation:
//  'Zur Theorie der spezifischen Waerme' by Peter Debye, Annalen der Physik, 1912, 39(4): 789
//
// And for English versions, check any standard physical chemistry or solid-state physics textbook.
// 

class DebyeCv : public SpecificHeat {
public:
  DebyeCv(ProblemSpecP& ps);
  ~DebyeCv();

  virtual void outputProblemSpec(ProblemSpecP& ice_ps);

  virtual double getSpecificHeat(double T);

  virtual double getGamma(double T);

  virtual double getInternalEnergy(double T);

protected:
  double d_T_D; // Debye Temperature
  double d_N;   // Number of atoms in a molecule 
};

}

#endif /* _DEBYESPECIFICHEAT_H_ */


