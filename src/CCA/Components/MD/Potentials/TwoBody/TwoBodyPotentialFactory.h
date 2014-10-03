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

#ifndef UINTAHMD_ELECTROSTATICSFACTORY_h
#define UINTAHMD_ELECTROSTATICSFACTORY_h

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <CCA/Components/MD/Forcefields/definedForcefields.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/Potentials/TwoBody/definedTwoBodyPotentials.h>
#include <string>
#include <vector>

namespace Uintah {

  class MDSystem;

  /**
   *  @class ElectrostaticsFactory
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   February, 2013
   *
   *  @brief
   *
   *  @param
   */
  class NonbondedTwoBodyFactory {

    public:

      /**
       * @brief Simply create the appropriate NonbondedTwoBodyPotential object.
       *         This method has a switch for all known NonbondedTwoBody types.
       * @param ps The ProblemSpec handle with which to get properties from the input file.
       * @param system The MD system handle to pass off to the appropriate Electrostatics constructor.
       */
      static NonbondedTwoBodyPotential* create(      Forcefield*,
                                               const std::string&,
                                               const std::vector<std::string>&,
                                               const std::string& label,
                                               const std::string& defaultComment = "");
  };
}  // End namespace Uintah

#endif
