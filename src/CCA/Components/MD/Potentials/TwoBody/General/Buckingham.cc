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
 * Buckingham.cc
 *
 *  Created on: Feb 1, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Potentials/TwoBody/General/Buckingham.h>

namespace Uintah {

  const std::string Buckingham::d_potentialSubtype = "Buckingham";

  Buckingham::Buckingham(double _Rmin,
                         double _eps,
                         double _lambda,
                         const std::string& _label,
                         const std::string& _comment)
                        :Rmin(_Rmin),
                         epsilon(_eps),
                         lambda(_lambda),
                         d_label(_label),
                         d_comment(_comment) {
    A = 6.0 * epsilon * exp(lambda) / (lambda - 6);
    B = lambda / Rmin;
    C = epsilon * lambda * pow(Rmin, 6.0) / (lambda - 6);

  }

  void Buckingham::fillEnergyAndForce(SCIRun::Vector& force,
                                      double& energy,
                                      const SCIRun::Vector& R_ij) const {
    double r2 = R_ij.length2();
    double r = sqrt(r2);
    double rinv = 1.0 / r;
    double expNegBR = exp(-B * r);
    double C_over_R6 = C / (r2 * r2 * r2);

    energy = A * expNegBR - C_over_R6;
    force = (R_ij*rinv)*(A*B*expNegBR - rinv*6.0*C_over_R6);
    return;
  }

  void Buckingham::fillEnergy(double& energy,
                              const SCIRun::Vector& R_ij) const {
    double r2 = R_ij.length2();
    double r = sqrt(r2);
    double expNegBR = exp(-B * r);
    double C_over_R6 = C / (r2 * r2 * r2);

    energy = A * expNegBR - C_over_R6;

    return;
  }

  void Buckingham::fillForce(SCIRun::Vector& force,
                             const SCIRun::Vector& R_ij) const {
    double r2 = R_ij.length2();
    double r = sqrt(r2);
    double rinv = 1.0 / r;
    double expNegBR = exp(-B * r);
    double C_over_R6 = C / (r2 * r2 * r2);

    force = (R_ij*rinv)*(A*B*expNegBR - rinv*6.0*C_over_R6);
    return;
  }
}

