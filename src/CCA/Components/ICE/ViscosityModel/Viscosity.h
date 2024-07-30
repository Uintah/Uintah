/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef ICE_VISCOSITY_H
#define ICE_VISCOSITY_H

#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

class DataWarehouse;
class ICELabel;
class Material;
class Patch;


class Viscosity {
public:
            // constructors
  Viscosity( ProblemSpecP& ps);

  Viscosity( ProblemSpecP & boxps,
             const GridP  & grid );

            // destructor
  virtual ~Viscosity();

  virtual void
  outputProblemSpec(ProblemSpecP& vModels_ps) = 0;

            // methods compute the viscosity
  virtual void
  computeDynViscosity(const Patch         * patch,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & mu) = 0;

  virtual void
  computeDynViscosity(const Patch              * patch,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & mu) = 0;

  virtual void
  initialize (const Level * level ) = 0;

protected:
};

}

#endif /* _Viscosity_H_ */

