/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_MPM_REACTIONDIFFUSION_CONDUCTIVITYEQUATION_H
#define UINTAH_CCA_COMPONENTS_MPM_REACTIONDIFFUSION_CONDUCTIVITYEQUATION_H

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
//#include <Core/Util/Handle.h>

namespace Uintah{
/*************************************************
 *
 * CLASS
 *   ConductivityEquation
 *
 *   This class computes the conductivity of a
 *   particle based on concentration levels.
 *
 *   The is a base class that can be extended to
 *   implement new conductivity equations.
 *
 *************************************************/


  class ConductivityEquation{
    public:
      ConductivityEquation(ProblemSpecP& ps);

      virtual ~ConductivityEquation();

      virtual double computeConductivity(double conductivity);

      virtual void outputProblemSpec(ProblemSpecP& ps);

  };
}
#endif // End of UINTAH_CCA_COMPONENTS_MPM_REACTIONDIFFUSION_CONDUCTIVITYEQUATION_H
