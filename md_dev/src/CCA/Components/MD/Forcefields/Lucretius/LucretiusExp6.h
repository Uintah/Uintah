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
 * LucretiusExp6.h
 *
 *  Created on: Jan 31, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSEXP6_H_
#define LUCRETIUSEXP6_H_

#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>
#include <assert.h>

namespace UintahMD {
  using namespace SCIRun;

  class LucretiusExp6 : public NonbondedTwoBodyPotential {

    public:
      LucretiusExp6() {}
      LucretiusExp6(double _Rmin,
                    double _eps,
                    double _lambda) : Rmin(_Rmin), epsilon(_eps), lambda(_lambda) {
        A = 6.0 * epsilon * exp(lambda) / (lambda - 6);
        B = lambda / Rmin;
        C = epsilon * lambda * pow(Rmin, 6.0) / (lambda - 6);
        D = 0.00005 * pow((12.0 / B), 12.0);  // D(12/(B*r_ij))^12 term; D = 5e-5 kCal/mol

      }
      ~LucretiusExp6() {
      }

      void fillEnergyAndForce(SCIRun::Vector& force,
                              double& energy,
                              const SCIRun::Vector& offSet) const;

      inline void fillEnergyAndForce(SCIRun::Vector& force,
                                     double& energy,
                                     const SCIRun::Vector& P1,
                                     const SCIRun::Vector& P2) const {
        fillEnergyAndForce(force, energy, P2 - P1);
        return;
      }

      void fillEnergy(double& energy,
                      const SCIRun::Vector& offSet) const;
      inline void fillEnergy(SCIRun::Vector& force,
                             double& energy,
                             const SCIRun::Vector& P1,
                             const SCIRun::Vector& P2) const {
        fillEnergy(energy, P2 - P1);
        return;
      }

      void fillForce(SCIRun::Vector& force,
                     const SCIRun::Vector& offSet) const;

      void fillForce(SCIRun::Vector& force,
                     const SCIRun::Vector& P1,
                     const SCIRun::Vector& P2) const {
        fillForce(force, P2 - P1);
        return;
      }

      std::string& getPotentialDescriptor() const;

    private:
      static const std::string d_potentialName = "Buckingham_Variant";
      double Rmin, epsilon, lambda;
      double A, B, C, D;
  };
}

#endif /* LUCRETIUSEXP6_H_ */
