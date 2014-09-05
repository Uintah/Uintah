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
 * Buckingham.h
 *
 *  Created on: Jan 31, 2014
 *      Author: jbhooper
 */

#ifndef BUCKINGHAM_H_
#define BUCKINGHAM_H_

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/Potentials/NonbondedPotential.h>

#include <string>

namespace Uintah {

  using namespace SCIRun;

  class Buckingham : public NonbondedTwoBodyPotential {

    public:
      //Buckingham() {}
      Buckingham(double, double, double, const std::string&, const std::string& defaultComment = "");
      ~Buckingham() {}

      void fillEnergyAndForce(SCIRun::Vector& force,
                              double& energy,
                              const SCIRun::Vector& offSet) const;

      inline void fillEnergyAndForce(SCIRun::Vector& force,
                                     double& energy,
                                     const SCIRun::Point& P1,
                                     const SCIRun::Point& P2) const {
        fillEnergyAndForce(force, energy, P2 - P1);
        return;
      }

      void fillEnergy(double& energy,
                      const SCIRun::Vector& offSet) const;

      inline void fillEnergy(double& energy,
                             const SCIRun::Point& P1,
                             const SCIRun::Point& P2) const {
        fillEnergy(energy, P2 - P1);
        return;
      }

      void fillForce(SCIRun::Vector& force,
                     const SCIRun::Vector& offSet) const;

      void fillForce(SCIRun::Vector& force,
                     const SCIRun::Point& P1,
                     const SCIRun::Point& P2) const {
        fillForce(force, P2 - P1);
        return;
      }

      inline const std::string getPotentialDescriptor() const {
        return this->getPotentialSuperType() + this->getPotentialBaseType() + d_potentialSubtype;
      }

      inline const std::string getComment() const {
        return d_comment;
      }

      inline const std::string getLabel() const {
        return d_label;
      }

    private:
      static const std::string d_potentialSubtype;
      const std::string& d_comment;
      const std::string& d_label;
      double Rmin, epsilon, lambda;
      double A, B, C;
  };

}

#endif /* BUCKINGHAM_H_ */
