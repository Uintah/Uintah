/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include "MieGruneisenEOSEnergy.h"
#include <Core/Math/DEIntegrator.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Exceptions/InvalidValue.h>
#include <cmath>
#include <iostream>

using namespace Uintah;
using namespace std;

MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(ProblemSpecP& ps)
{
  ps->require("C_0",d_const.C_0);
  ps->require("Gamma_0",d_const.Gamma_0);
  ps->require("S_alpha",d_const.S_1);
  ps->getWithDefault("S_2",d_const.S_2,0.0);
  ps->getWithDefault("S_3",d_const.S_3,0.0);
} 
         
MieGruneisenEOSEnergy::MieGruneisenEOSEnergy(const MieGruneisenEOSEnergy* cm)
{
  d_const.C_0 = cm->d_const.C_0;
  d_const.Gamma_0 = cm->d_const.Gamma_0;
  d_const.S_1 = cm->d_const.S_1;
  d_const.S_2 = cm->d_const.S_2;
  d_const.S_3 = cm->d_const.S_3;
} 
         
MieGruneisenEOSEnergy::~MieGruneisenEOSEnergy()
{
}
         
void MieGruneisenEOSEnergy::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP eos_ps = ps->appendChild("equation_of_state");
  eos_ps->setAttribute("type","mie_gruneisen");

  eos_ps->appendElement("C_0",d_const.C_0);
  eos_ps->appendElement("Gamma_0",d_const.Gamma_0);
  eos_ps->appendElement("S_alpha",d_const.S_1);
  eos_ps->appendElement("S_2",d_const.S_2);
  eos_ps->appendElement("S_3",d_const.S_3);
}

//////////
// Calculate the pressure using the Mie-Gruneisen equation of state
double 
MieGruneisenEOSEnergy::computePressure(const MPMMaterial* matl,
                                 const PlasticityState* state,
                                 const Matrix3& ,
                                 const Matrix3& ,
                                 const double& )
{
  // Get the current density
  double rho = state->density;

  // Get original density
  double rho_0 = matl->getInitialDensity();
   
  // Calc. eta
  double eta = 1. - rho_0/rho;

  // Retrieve specific internal energy e
  double e = state->energy;

  // Calculate the pressure, See:
  // Steinberg, D.,
  // Equation of State and Strength Properties of Selected materials,
  // 1991, Lawrence Livermore National Laboratory.
  // UCRL-MA-106439

  double p;
  if(eta >= 0.0) {
    double denom = 
               (1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta)
              *(1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta);
    p = rho_0*d_const.Gamma_0*e 
      + rho_0*(d_const.C_0*d_const.C_0)*eta*(1. - .5*d_const.Gamma_0*eta)/denom;
  }
  else{
    p = rho_0*d_const.Gamma_0*e
      + rho_0*(d_const.C_0*d_const.C_0)*eta;
  }

  return -p;
}


double 
MieGruneisenEOSEnergy::computeIsentropicTemperatureRate(const double T,
                                                        const double rho_0,
                                                        const double rho_cur,
                                                        const double Dtrace)
{
  double dTdt = -T*d_const.Gamma_0*rho_0*Dtrace/rho_cur;

  return dTdt;
}

double 
MieGruneisenEOSEnergy::eval_dp_dJ(const MPMMaterial* matl,
                            const double& detF, 
                            const PlasticityState* state)
{
  double rho_0 = matl->getInitialDensity();
  double J = detF;
  double rho_cur = rho_0/J;
  return computeBulkModulus(rho_0,rho_cur);
#if 0
  double C_0 = d_const.C_0;
  double S_1 = d_const.S_1;
  double S_2 = d_const.S_2;
  double S_3 = d_const.S_3;
  double Gamma_0 = d_const.Gamma_0;

  double J = detF;

  double eta = 1.0-J;
  double denom = (1.0 - S_1*eta - S_2*eta*eta - S_3*eta*eta*eta);
  double numer = -rho_0*C_0*C_0*((1.0 - Gamma_0*eta)*denom 
       + 2.0*eta*(1.0 - Gamma_0*eta/2.0)*(S_1 + 2.0*S_2*eta + 3.0*S_3*eta*eta));
//  double denom3 = (denom*denom*denom);

//  if (denom3 == 0.0) {
//    cout << "rh0_0 = " << rho_0 << " J = " << J 
//           << " numer = " << numer << endl;
//    denom3 = 1.0e-5;
//  }

  return (numer/denom);
#endif
}

// Compute bulk modulus
double 
MieGruneisenEOSEnergy::computeBulkModulus(const double& rho_orig,
                                          const double& rho_cur)
{
  // Calculate J 
  double J = rho_orig/rho_cur;

  // Calc. eta = 1 - J  (Note that J = 1 - eta)
  double eta = 1. - J;

  // Calculate the pressure
  double C0sq = d_const.C_0*d_const.C_0;
  double K = C0sq*rho_orig;
  if(eta >= 0.0) {
    double S1 = d_const.S_1;
    double S2 = d_const.S_2;
    double S3 = d_const.S_3;
    double etaSq = eta*eta;
    double etaCb = eta*eta*eta;
    double Jsq = J*J;
    double numer = 1.0 + S1 - J*S1 + 3.0*S2 - 6.0*J*S2 + 3.0*Jsq*S2 + 
            5.0*etaCb*S3 - eta*d_const.Gamma_0*(1.0 + etaSq*S2 + 2.0*etaCb*S3);
    double denom = 1.0 - S1*eta - S2*etaSq - S3*etaCb;
    K *= numer/(denom*denom*denom);
  }
  else{
    K /= (J*J);
  }
  return K;
}

// Compute pressure (option 1) - no internal energy contribution
// Compression part:
//  p = -((C0^2 Eta (2 - Eta Gamma0) rho0)/(2 (1 - Eta S1 - Eta^2 S2 - Eta^3 S3)^2))
// Tension part:
//  p = -((C0^2 Eta rho0)/(1 - Eta))
double 
MieGruneisenEOSEnergy::computePressure(const double& rho_orig,
                                       const double& rho_cur)
{
  // Calculate J
  double J = rho_orig/rho_cur;

  // Calc. eta = 1 - J  (Note that J = 1 - eta)
  double eta = 1. - J;

  // Calculate the pressure
  double p = 0.0;
  if(eta >= 0.0) {
    double etaMax = 1.0 - 1.0e-16; // Hardcoded to take care of machine precision issues
    eta = (eta > etaMax) ? etaMax : eta;
    p = pCompression(rho_orig, eta);
  }
  else{
    p = pTension(rho_orig, eta);
  }

  return p;
}

// Compute pressure (option 2) - no internal energy contribution
// Compression part:
//  p = -((C0^2 Eta (2 - Eta Gamma0) rho0)/(2 (1 - Eta S1 - Eta^2 S2 - Eta^3 S3)^2))
//  dp_dJ = (C0^2 rho0 (1 + S1 - (1 - Eta) S1 + 3 S2 - 6 (1 - Eta) S2 + 
//     3 (1 - Eta)^2 S2 + 5 Eta^3 S3 - Eta Gamma0 (1 + Eta^2 S2 + 2 Eta^3 S3)))/
//       (1 - Eta S1 - Eta^2 S2 - Eta^3 S3)^3
//  K = dp_dJ
// Tension part:
//  p = -((C0^2 Eta rho0)/(1 - Eta))
//  dp_dJ = (C0^2 rho0)/(1 - Eta)^2
void 
MieGruneisenEOSEnergy::computePressure(const double& rho_orig,
                                       const double& rho_cur,
                                       double& pressure,
                                       double& dp_drho,
                                       double& csquared)
{
  // Calculate J and dJ_drho
  double J = rho_orig/rho_cur;
  double dJ_drho = -J/rho_cur;
  double dp_dJ = 0.0;

  // Calc. eta = 1 - J  (Note that J = 1 - eta)
  double eta = 1. - J;

  // Calculate the pressure
  if(eta >= 0.0) {
    double etaMax = 1.0 - 1.0e-16; // Hardcoded to take care of machine precision issues
    eta = (eta > etaMax) ? etaMax : eta;
    pressure = pCompression(rho_orig, eta);
    dp_dJ = dpdJCompression(rho_orig, eta);
  }
  else{
    pressure = pTension(rho_orig, eta);
    dp_dJ = dpdJTension(rho_orig, eta);
  }
  dp_drho = dp_dJ*dJ_drho;
  csquared = dp_dJ/rho_cur;

  if (isnan(pressure) || fabs(dp_dJ) < 1.0e-30) {
    ostringstream desc;
    desc << "pressure = " << -pressure << " rho_cur = " << rho_cur 
         << " dp_drho = " << -dp_drho << " c^2 = " << csquared << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
  return;
}

// Compute density given pressure (tension +ve)
double 
MieGruneisenEOSEnergy::computeDensity(const double& rho_orig,
                                      const double& pressure)
{
  double eta = 0.0;
  double C0sq = d_const.C_0*d_const.C_0;
  double bulk = C0sq*rho_orig;
  if (fabs(pressure) < 0.1*bulk) {
    // Use Newton's method for small pressures (less than 1 GPa hardcoded)
    // (should use p_ref instead for this this work under non-SI units - TO DO)

    if (pressure < 0.0) {
      // Compressive deformations
      const double J0 = 0.8;
      const double tolerance = 1.0e-3;
      const int maxIter = 10;
      pFuncPtr pFunc = &MieGruneisenEOSEnergy::pCompression;
      dpdJFuncPtr dpdJFunc = &MieGruneisenEOSEnergy::dpdJCompression;
      eta = findEtaNewton(pFunc, dpdJFunc, 
                          rho_orig, pressure, J0, tolerance, maxIter);
    } else {
      // Tensile deformations
      const double J0 = 1.5;
      const double tolerance = 1.0e-3;
      const int maxIter = 10;
      pFuncPtr pFunc = &MieGruneisenEOSEnergy::pTension;
      dpdJFuncPtr dpdJFunc = &MieGruneisenEOSEnergy::dpdJTension;
      eta = findEtaNewton(pFunc, dpdJFunc, 
                          rho_orig, pressure, J0, tolerance, maxIter);
    }
  } else {
    // Use Ridder's method for other pressures
    if (pressure < 0.0) {
      double etamin = 0.0;
      double etamax = 1.0 - 1.0e-16; // Hardcoded for machine precision issues
                                     // Needs to be resolved (TO DO)
      const double tolerance = 1.0e-3;
      const int maxIter = 100;
      pFuncPtr pFunc = &MieGruneisenEOSEnergy::pCompression;
      eta = findEtaRidder(pFunc, rho_orig, pressure,
                          etamin, etamax, tolerance, maxIter);
    } else {
      double etamin = -5.0; // Hardcoded: Needs to be resolved (TO DO)
      double etamax = 0.0; 
      const double tolerance = 1.0e-3;
      const int maxIter = 100;
      pFuncPtr pFunc = &MieGruneisenEOSEnergy::pTension;
      eta = findEtaRidder(pFunc, rho_orig, pressure,
                          etamin, etamax, tolerance, maxIter);
    }
  }
  double J = 1.0 - eta;
  double rho = rho_orig/J;  // **TO DO** Infinity check

  return rho;
}

// Private method: Find root of p(eta) - p0 = 0 using Ridder's method
double
MieGruneisenEOSEnergy::findEtaRidder(pFuncPtr pFunc,
                                     const double& rho_orig,
                                     const double& p0,
                                     double& etamin,
                                     double& etamax,
                                     const double& tolerance,
                                     const int& maxIter)
{
  double eta = 1.0 - 1.0e-16; // Hardcoded to take care of machine precision issues
  double pp = (this->*pFunc)(rho_orig, eta);
  //if (0.1*fabs(pp) < fabs(p0)) return eta;
  if (0.1*fabs(pp) < fabs(p0)) {
     pp = copysign(0.1*fabs(pp), p0);
  } else {
     pp = p0;
  }

  //etamin = 1.0 - Jmax;
  double pmin = (this->*pFunc)(rho_orig, etamin);
  double fmin = pmin - pp;
  if (fmin == 0.0) {
    return etamin;
  }

  //etamax = 1.0 - Jmin;
  double pmax = (this->*pFunc)(rho_orig, etamax);
  double fmax = pmax - pp;
  if (fmax == 0.0) {
    return etamax;
  }

  int count = 1;
  double fnew = 0.0;
  while (count < maxIter) {

    // compute mid point
    double etamid = 0.5*(etamin + etamax); 
    double pmid = (this->*pFunc)(rho_orig, etamid);
    double fmid = pmid - pp;

    double ss = sqrt(fmid*fmid - fmin*fmax);
    if (ss == 0.0) {
      return eta;
    }

    // compute new point
    double dx = (etamid - etamin)*fmid/ss;
    if ((fmin - fmax) < 0.0) {
      dx = -dx;
    }
    double etanew = etamid + dx; 
    double pnew = (this->*pFunc)(rho_orig, etanew);
    fnew = pnew - pp;

    // Test for convergence
    if (count > 1) {
      //if abs(etanew - eta) < tolerance*max(abs(etanew),1.0)
      if (fabs(fnew) < tolerance*fabs(pp)) {
        return etanew;
      }
    }
    eta = etanew;

    // Re-bracket the root as tightly as possible
    if (fmid*fnew > 0.0) {
      if (fmin*fnew < 0.0) {
        etamax = etanew; 
        fmax = fnew;
      } else {
        etamin = etanew; 
        fmin = fnew;
      }
    } else {
      etamin = etamid; 
      fmin = fmid; 
      etamax = etanew; 
      fmax = fnew;
    }

    count++;

  }
  ostringstream desc;
  desc << "**ERROR** Ridder algorithm did not converge" 
       << " pressure = " << p0 << " pp = " << pp 
       << " eta = " << eta << endl;
  throw ConvergenceFailure(desc.str(), maxIter, fabs(fnew), tolerance*fabs(pp), __FILE__, __LINE__);

  return -1;
}

// Private method: Find root of p(eta) - p0 = 0 using Newton's method
double
MieGruneisenEOSEnergy::findEtaNewton(pFuncPtr pFunc,
                                     dpdJFuncPtr dpdJFunc,
                                     const double& rho_orig,
                                     const double& p0,
                                     const double& J0, 
                                     const double& tolerance,
                                     const int& maxIter)
{
  double p = 0.0;
  double dp_dJ = 0.0;
  double J = J0;
  double eta = 1.0 - J;

  double f = 0.0;
  double fPrime = 0.0;
  int iter = 0;

  do {

    // Calculate p
    p = (this->*pFunc)(rho_orig, eta);

    // Calculate dp/dJ
    dp_dJ = (this->*dpdJFunc)(rho_orig, eta);

    // f(J) and f'(J) calc
    f = p - p0;
    fPrime = dp_dJ;
    J -= f/fPrime;

    // Update eta
    eta = 1.0 - J;

    ++iter;
  } while (fabs(f) > tolerance && iter < maxIter);

  if (iter >= maxIter) {
    ostringstream desc;
    desc << "**ERROR** Newton algorithm did not converge" 
         << " pressure = " << p0 << " eta = " << eta << endl;
    throw ConvergenceFailure(desc.str(), maxIter, fabs(f), tolerance, __FILE__, __LINE__);
  }

  return eta;
}

// Private method: Compute p for compressive volumetric deformations
double 
MieGruneisenEOSEnergy::pCompression(const double& rho_orig,
                                    const double& eta)
{
  // Calc eta^2 and eta^3
  double etaSq = eta*eta;
  double etaCb = eta*eta*eta;

  // Calculate p
  double numer = rho_orig*d_const.C_0*d_const.C_0*eta*(1.0 - 0.5*d_const.Gamma_0*eta);
  double denom = 1.0 - d_const.S_1*eta - d_const.S_2*etaSq - d_const.S_3*etaCb;
  double p = -numer/(denom*denom);

  return p;
}

// Private method: Compute dp/dJ for compressive volumetric deformations
double 
MieGruneisenEOSEnergy::dpdJCompression(const double& rho_orig,
                                       const double& eta)
{
  // Calc eta^2 and eta^3
  double etaSq = eta*eta;
  double etaCb = eta*eta*eta;

  // Calculate dp/dJ
  double J = 1 - eta;
  double numer = 1.0 +  eta*d_const.S_1 + 3.0*(1.0 - 2.0*J + J*J)*d_const.S_2 + 
                 5.0*etaCb*d_const.S_3 - eta*d_const.Gamma_0*(1.0 + etaSq*d_const.S_2 + 
                 2.0*etaCb*d_const.S_3);
  double denom = 1.0 - d_const.S_1*eta - d_const.S_2*etaSq - d_const.S_3*etaCb;
  double dp_dJ = d_const.C_0*d_const.C_0*rho_orig*numer/(denom*denom*denom);

  return dp_dJ;
}

// Private method: Compute p for tensile volumetric deformations
double 
MieGruneisenEOSEnergy::pTension(const double& rho_orig,
                                const double& eta)
{
  // Calculate p
  double p = -rho_orig*d_const.C_0*d_const.C_0*eta/(1.0 - eta);
  
  return p;
}

// Private method: Compute dp/dJ for tensile volumetric deformations
double 
MieGruneisenEOSEnergy::dpdJTension(const double& rho_orig,
                                   const double& eta)
{
  // Calculate dp/dJ
  double J = 1 - eta;
  double dp_dJ = (rho_orig*d_const.C_0*d_const.C_0)/(J*J);

  return dp_dJ;
}

// Compute strain energy
//   An exact integral does not exist and numerical integration is needed
//   Use double exponential integration because the function is smooth
//   (even though we use it only in a limited region)
//   **WARNING** Requires well behaved EOS that does not blow up to 
//               infinity in the middle of the domain
double 
MieGruneisenEOSEnergy::computeStrainEnergy(const double& rho_orig,
                                           const double& rho_cur)
{
  // Calculate J 
  double J = rho_orig/rho_cur;

  // Calc. eta = 1 - J  (Note that J = 1 - eta)
  double eta = 1. - J;

  // Calculate the pressure
  double U = 0.0;
  double C0sq = d_const.C_0*d_const.C_0;
  if(eta >= 0.0) {
    int evals;
    double error;
    U = DEIntegrator<MieGruneisenEOSEnergy>::Integrate(this, 0, eta, 1.0e-6, evals, error);
    U *= rho_orig*C0sq;
  }
  else{
    U = C0sq*rho_orig*(J - 1.0 - log(J)); 
  }
  return U;
}

// Special operator for computing energy
double
MieGruneisenEOSEnergy::operator()(double eta) const
{
  // Calculate the pressure
  double etaSq = eta*eta;
  double etaCb = eta*eta*eta;
  double numer = eta*(1.0 - 0.5*d_const.Gamma_0*eta);
  double denom = 1.0 - d_const.S_1*eta - d_const.S_2*etaSq - d_const.S_3*etaCb;

  return -numer/(denom*denom);
}

