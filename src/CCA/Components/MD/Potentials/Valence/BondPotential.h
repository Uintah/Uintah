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
 * BondPotential.h
 *
 *  Created on: Feb 20, 2014
 *      Author: jbhooper
 */

#ifndef BONDPOTENTIAL_H_
#define BONDPOTENTIAL_H_

#include <CCA/Components/MD/Potentials/Potential.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <string>

namespace Uintah {

  class BondPotential : public Potential {

    public:
      // Specific potential identifier functions
      const std::string getPotentialSuperType() {
        return d_potentialSuperType;
      }
      virtual const std::string getPotentialBaseType() const = 0;
      virtual const std::string getPotentialSubType() const = 0;

      //  Inherited from class Potential (potential.h)  VVV
      virtual const std::string getPotentialDescriptor() const = 0;
      virtual void fillEnergyAndForce(SCIRun::Vector& force, double& energy, SCIRun::Vector& bondVector) const = 0;
      virtual void fillEnergyAndForce(SCIRun::Vector& force, double& energy, SCIRun::Point& P1, SCIRun::Point& P2) const = 0;
      virtual void fillForce(SCIRun::Vector& force, SCIRun::Vector& bondVector) const = 0;
      virtual void fillForce(SCIRun::Vector& force, SCIRun::Point& P1, SCIRun::Point& P2) const = 0;
      virtual void fillEnergy(double& energy, SCIRun::Vector& bondVector) const = 0;
      virtual void fillEnergy(double& energy, SCIRun::Point& P1, SCIRun::Point& P2) const = 0;

      virtual ~BondPotential() = 0; // Force implementation of potential destructor even if automatic

    private:
      static const std::string d_potentialSuperType;

  };

//  const std::string BondPotential::d_potentialSuperType = "Bond::";

}



#endif /* BONDPOTENTIAL_H_ */
