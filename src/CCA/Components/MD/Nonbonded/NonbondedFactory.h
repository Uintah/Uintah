/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_NONBONDEDFACTORY_h
#define UINTAH_MD_NONBONDEDFACTORY_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Integrators/Integrator.h>
#include <CCA/Components/MD/Nonbonded/Nonbonded.h>

namespace Uintah {

  class ProcessorGroup;
  class MDSystem;

  /**
   *  @class NonbondedFactory
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   June, 2013
   *
   *  @brief
   *
   *  @param
   */
  class NonbondedFactory {

    public:

      /**
       * @brief Simply create the appropriate Nonbonded object.
       *         This method has a switch for all known Non-bonded types.
       * @param ps The ProblemSpec handle with which to get non-bonded properties from the input file.
       * @param system The MD system handle to pass off to the appropriate Nonbonded constructor.
       */
      static Nonbonded* create(const ProblemSpecP&,
                               coordinateSystem*,
                               MDLabel*,
                               forcefieldInteractionClass,
                               interactionModel);
  };
}  // End namespace Uintah

#endif
