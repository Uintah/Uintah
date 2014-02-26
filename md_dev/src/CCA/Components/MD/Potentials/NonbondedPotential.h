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

/*
 * NonbondedPotential.h
 *
 *  Created on: Jan 26, 2014
 *      Author: jbhooper
 */

#ifndef NONBONDEDPOTENTIAL_H_
#define NONBONDEDPOTENTIAL_H_

#include <CCA/Components/MD/Potentials/Potential.h>
#include <Core/Geometry/Vector.h>

#include <string>
#include <sstream>
#include <iomanip>

namespace UintahMD {

  class NonbondedPotential : public Potential {
    public:
		// returns potential energy of NonbondedPotential with input parameters
      virtual const std::string getPotentialBaseType() const = 0;
      virtual const std::string getPotentialDescriptor() const = 0;
      const std::string getPotentialSuperType() const {
        return d_potentialSuperType;
      }
    private:
      static const std::string d_potentialSuperType;
  };

//  const std::string NonbondedPotential::d_potentialSuperType = "Nonbonded::";
}
#endif /* NONBONDEDPOTENTIAL_H_ */
