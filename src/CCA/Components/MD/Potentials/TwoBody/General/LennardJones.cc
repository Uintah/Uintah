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
 * LennardJones.cc
 *
 *  Created on: Jan 30, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Potentials/TwoBody/General/LennardJones.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>

namespace Uintah {

  const int LennardJonesPotential::sc_maxIntegralPower = 15;
  const std::string LennardJonesPotential::d_potentialSubtype = "Lennard-Jones";

  LennardJonesPotential::LennardJonesPotential(double _sigma,
                                               double _eps,
                                               const std::string& _label,
                                               size_t _repulsive,
                                               size_t _attractive,
                                               const std::string& _comment)
                                              :sigma(_sigma),
                                               epsilon(_eps),
                                               d_label(_label),
                                               m(_repulsive),
                                               n(_attractive),
                                               d_comment(_comment) {
    ASSERTRANGE(m, 0, sc_maxIntegralPower);
    ASSERTRANGE(n, 0, sc_maxIntegralPower);
    A = 4.0 * epsilon * pow(sigma, static_cast<double>(m));
    C = 4.0 * epsilon * pow(sigma, static_cast<double>(n));
    std::stringstream descriptor;
    double logm = log10(m);
    double logn = log10(n);
    int mDigits = static_cast<int>(logm) + 1;
    int nDigits = static_cast<int>(logn) + 1;
    descriptor << "_" << std::setw(mDigits) << m << "-" << std::setw(nDigits) << n;  // Append _m-n to potential subtype
    d_potentialDescriptor = this->getPotentialSuperType() + this->getPotentialBaseType() + d_potentialSubtype + descriptor.str();
  }

  double LennardJonesPotential::RToPower(double r,
                                         double r2,
                                         int b) const {
    // Calculates r^b using pre-calculated single and squared powers of r

    int power = b;
    double answer = 1.0;
    if (power % 2 != 0) {
      answer *= r;
      --power;
    }

    while (power > 0) {
      answer *= r2;
      power -= 2;
    }

    return (answer);
  }

  void LennardJonesPotential::fillEnergyAndForce(SCIRun::Vector& force,
                                                 double& energy,
                                                 const SCIRun::Vector& R_ij) const {
    // R_ij is the vector from the source point (i, should be in patch) to
    //   the target point (j, not necessarily in patch).

    // u(R) = A/|R|^m - C/|R|^n
    // F(R) = -R (-mA/|R|^(m+1) + C/|R|^(n+1)) = R (mA/|R|^m - nC/|R|^n)/|R|^2

    double r2 = R_ij.length2();
    double r = sqrt(r2);

    double r_to_m = RToPower(r, r2, m);
    double r_to_n = RToPower(r, r2, n);
//    double r_to_m = pow(r,m);
//    double r_to_n = pow(r,n);
    double mTerm = A/r_to_m;
    double nTerm = C/r_to_n;

    // R = vector; r = |R| = magnitude of vector
    // U(|R|) = A/r^m - C/r^n
    energy = mTerm - nTerm;
    // F(|R|) = -grad(U|R|) = (R/r)*-(m * A/r^(m+1) + n * C/(r^n+1))
    //                      = (R/r^2)*(-m * A/r^m + n * C/r^n)
    force = R_ij * (-m * mTerm + n * nTerm) / r2;
//    std::cerr << std::setprecision(10) << "Distance: " << r << " A: " << A << " C: " << C << " Energy: " << energy << " Force: " << force << std::endl;
    return;
  }

  void LennardJonesPotential::fillEnergy(double& energy,
                                         const SCIRun::Vector& R_ij) const {
    double r2 = R_ij.length2();
    double r = sqrt(r2);

    double r_to_m = RToPower(r, r2, m);
    double r_to_n = RToPower(r, r2, n);

    energy = A / r_to_m - C / r_to_n;
    return;
  }

  void LennardJonesPotential::fillForce(SCIRun::Vector& force,
                                        const SCIRun::Vector& R_ij) const {
    double r2 = R_ij.length2();
    double r = sqrt(r2);

    double r_to_m = RToPower(r, r2, m);
    double r_to_n = RToPower(r, r2, n);

    force = R_ij * (-m * A / r_to_m + n * C / r_to_n) / r2;
    return;
  }

}  // Namespace:  Uintah_MD

