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
 * LucretiusExp6.cc
 *
 *  Created on: Feb 1, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Potentials/TwoBody/Lucretius/LucretiusExp6.h>

using namespace Uintah;
using namespace SCIRun;

const std::string LucretiusExp6::d_potentialSubtype = "Lucretius_Exp6";

bool LucretiusExp6::findAlternativeRepresentation(const double A,
                                             const double B,
                                             const double C,
                                             double& Rmin,
                                             double& epsilon,
                                             double& lambda) {
  // FIXME
  return (false);
}

LucretiusExp6::LucretiusExp6(double _first,
                             double _second,
                             double _third,
                             const std::string& _label,
                             const std::string& _comment)
                            :A(_first),
                             B(_second),
                             C(_third),
                             d_label(_label),
                             d_comment(_comment) {

  // Determine Rmin, epsilon, and lambda
    REL_format = this->findAlternativeRepresentation(A,B,C,Rmin,epsilon,lambda);

    D = 0.0005 * pow((12.0 / B), 12.0);  // D(12/(B*r_ij))^12 term; D = 5e-5 kCal/mol
    d_potentialDescriptor = this->getPotentialSuperType() + this->getPotentialBaseType() + d_potentialSubtype;
  }

void LucretiusExp6::fillEnergyAndForce(SCIRun::Vector& force,
                                       double& energy,
                                       const SCIRun::Vector& R_ij) const {
  double r2 = R_ij.length2();
  double r = sqrt(r2);
  double rinv = 1.0 / r;
  double r6 = r2 * r2 * r2;
  double r12 = r6 * r6;
  double expBetaR = exp(-B * r);
  double C_over_R6 = C / (r6);
  double D_over_R12 = D / (r12);

  energy = A * expBetaR - C_over_R6 + D / r12;
  force = R_ij * (A * B * expBetaR + rinv * (D_over_R12 - C_over_R6)) * rinv;
  return;
}

void LucretiusExp6::fillEnergy(double& energy,
                               const SCIRun::Vector& R_ij) const {
  double r2 = R_ij.length2();
  double r = sqrt(r2);
  double r6 = r2 * r2 * r2;
  double r12 = r6 * r6;
  double expBetaR = exp(-B * r);
  double C_over_R6 = C / (r6);
  double D_over_R12 = D / (r12);

  energy = A * expBetaR - C_over_R6 + D / r12;
  return;
}

void LucretiusExp6::fillForce(SCIRun::Vector& force,
                              const SCIRun::Vector& R_ij) const {
  double r2 = R_ij.length2();
  double r = sqrt(r2);
  double rinv = 1.0 / r;
  double r6 = r2 * r2 * r2;
  double r12 = r6 * r6;
  double expBetaR = exp(-B * r);
  double C_over_R6 = C / (r2 * r2 * r2);
  double D_over_R12 = D / (r12);

  force = R_ij * (A * B * expBetaR + rinv * (D_over_R12 - C_over_R6)) * rinv;
  return;
}

