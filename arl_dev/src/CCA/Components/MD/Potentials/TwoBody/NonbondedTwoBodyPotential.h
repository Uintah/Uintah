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
 * NonbondedTwoBodyPotential.h
 *
 *  Created on: Jan 26, 2014
 *      Author: jbhooper
 */

#ifndef UINTAH_MD_NONBONDED_TWOBODY_H_
#define UINTAH_MD_NONBONDED_TWOBODY_H_

#include <CCA/Components/MD/Potentials/NonbondedPotential.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <string>

namespace Uintah {

  using namespace SCIRun;

    class NonbondedTwoBodyPotential : public NonbondedPotential {
    public:
      NonbondedTwoBodyPotential() {}
      virtual void fillEnergyAndForce(SCIRun::Vector& force,
                                      double& energy,
                                      const SCIRun::Vector& offSet) const = 0;
      virtual void fillEnergyAndForce(SCIRun::Vector& force,
                                      double& energy,
                                      const SCIRun::Point& point1,
                                      const SCIRun::Point& point2) const = 0;

      virtual void fillEnergy(double& energy,
                              const SCIRun::Vector& offSet) const = 0;
      virtual void fillEnergy(double& energy,
                              const SCIRun::Point& point1,
                              const SCIRun::Point& point2) const = 0;

      virtual void fillForce(SCIRun::Vector& force,
                             const SCIRun::Vector& offSet) const = 0;
      virtual void fillForce(SCIRun::Vector& force,
                             const SCIRun::Point& point1,
                             const SCIRun::Point& point2) const = 0;

      virtual const std::string getPotentialDescriptor() const = 0;
      virtual ~NonbondedTwoBodyPotential() { }

      const std::string getPotentialBaseType() const {
        return d_potentialBaseType;
      }

      // Returns potential specific label
      virtual const std::string getLabel() const = 0;

    private:
      static const std::string d_potentialBaseType;
  };

}

#endif /* UINTAH_MD_NONBONDED_TWOBODY_H_ */
