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

#ifndef _SPECIFICHEAT_H_
#define _SPECIFICHEAT_H_

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Uintah {

class DataWarehouse;
class ICELabel;
class Material;
class Patch;


class SpecificHeat {
public:
  SpecificHeat(ProblemSpecP& ps);
  virtual ~SpecificHeat();

  virtual void outputProblemSpec(ProblemSpecP& ice_ps) = 0;

  

  virtual double getGamma(double T) = 0;

  virtual double getInternalEnergy(double T) = 0;
  
  virtual void
  computeSpecificHeat(CellIterator        & iter,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & cv) = 0;
             
  virtual void
  computeSpecificHeat(CellIterator             & iter,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & cv) = 0;

protected:
};

}

#endif /* _SPECIFICHEAT_H_ */

