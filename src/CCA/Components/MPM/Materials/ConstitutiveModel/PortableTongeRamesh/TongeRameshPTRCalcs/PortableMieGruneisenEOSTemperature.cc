/*
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC § 105.
 *
 * However, because the project utilizes code licensed from contributors and other
 * third parties, it therefore is licensed under the MIT License.
 * http://opensource.org/licenses/mit-license.php.
 *
 * Under that license, permission is granted free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the conditions that any
 * appropriate copyright notices and this permission notice are
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "PortableMieGruneisenEOSTemperature.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

namespace PTR{

PortableMieGruneisenEOSTemperature::PortableMieGruneisenEOSTemperature()
{
}
        
PortableMieGruneisenEOSTemperature::PortableMieGruneisenEOSTemperature(const CMData cm)
{
  d_const = cm;
} 
         
PortableMieGruneisenEOSTemperature::PortableMieGruneisenEOSTemperature(const PortableMieGruneisenEOSTemperature* cm)
{
  d_const = cm->d_const;
} 

PortableMieGruneisenEOSTemperature::~PortableMieGruneisenEOSTemperature()
{
}
         
//////////
// Calculate the mean stress (1/3tr(sigma)) using the Mie-Gruneisen equation of state
double 
PortableMieGruneisenEOSTemperature::computePressure(const PState state) const
{
  // Get the current density
  double rho = state.density;

  // Get original density
  double rho_0 = state.initialDensity;
   
  // Calc. eta
  double eta = 1. - rho_0/rho;

  // Retrieve specific internal energy e (energy per unit mass)
  double e_theta = d_const.C_v*(state.temperature - d_const.theta_0);
  double e_c = computeStrainEnergy(rho_0,rho)/rho_0; // strain energy is energy per unit original volume
  double e = e_theta+e_c;

  // Calculate the shock speed:

  double Us;
  if(eta >= 0.0) {
    double denom = (1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta);
    Us = d_const.C_0/denom;
  }
  else{
    Us = d_const.C_0;
  }

  double p_H = rho_0 * Us * Us * eta;
  double p = p_H*(1-0.5*d_const.Gamma_0 *eta) + rho_0*d_const.Gamma_0 * e;

  return -p;
 }


double 
PortableMieGruneisenEOSTemperature::computeIsentropicTemperatureRate(const double T,
                                                        const double rho_0,
                                                        const double rho_cur,
                                                        const double Dtrace) const
{
  double dTdt = -T*d_const.Gamma_0*rho_0*Dtrace/rho_cur;
  return dTdt;
}

// TO DO update this to be the isentropic bulk modulus, it is
// currently the cold bulk modulus.
double 
PortableMieGruneisenEOSTemperature::eval_dp_dJ(const double rho_orig,
                                               const double detF, 
                                               const PState state) const
{
  double rho_cur = rho_orig/detF;
  return computeBulkModulus(rho_orig, rho_cur);
}

// Compute bulk modulus (cold bulk modulus)
double 
PortableMieGruneisenEOSTemperature::computeBulkModulus(const double rho_0,
                                                       const double rho)  const
{
  const double C_0(d_const.C_0);
  const double S_1(d_const.S_1);
  const double S_2(d_const.S_2);
  const double S_3(d_const.S_3);
  const double Gamma_0(d_const.Gamma_0);
  double J = rho_0/rho;
  double eta = 1. - J;

  // Compute the change in shock speed with J:
  double dUs_dJ(0), Us(C_0);
  if(eta>0){
    double denom = 1-S_1*eta-S_2*eta*eta-S_3*eta*eta*eta;
    
    dUs_dJ = -C_0 * (S_1 + 2*S_2*eta + 3*S_3*eta*eta*eta)/(denom*denom);
    Us = C_0/denom;
  }

  // Compute dP_h/dJ:
  double dPh_dJ = 2*rho_0*Us*dUs_dJ*eta - rho_0*Us*Us;
  double dPc_dJ;
  if(fabs(Gamma_0)>0){
    double Ph = rho_0*Us*Us*eta;
    double e_c = computeStrainEnergy(rho_0,rho)/rho_0;
    dPc_dJ = dPh_dJ*(1-0.5*Gamma_0*eta)-0.5*Gamma_0*Ph
      +0.5*Gamma_0*Gamma_0*Ph*eta-rho_0*Gamma_0*Gamma_0*e_c;
  } else {
    dPc_dJ = dPh_dJ;
  }

  return -dPc_dJ;
}

double 
PortableMieGruneisenEOSTemperature::computeIsentropicBulkModulus(const double rho_0,
                                                                 const double rho,
                                                                 const double theta
                                                                 )  const
{
  const double Gamma_0(d_const.Gamma_0);
  const double c_eta(d_const.C_v);

  double K_therm = computeBulkModulus(rho_0,rho);
  double K_ise   = K_therm + rho_0*Gamma_0*c_eta*(Gamma_0*theta);
  return K_ise;
}
// Compute pressure (option 1) - no internal energy contribution
// Compression part:
//  p = -((C0^2 Eta (2 - Eta Gamma0) rho0)/(2 (1 - Eta S1 - Eta^2 S2 - Eta^3 S3)^2))
// Tension part:
//  p = -((C0^2 Eta rho0)/(1 - Eta))
double 
PortableMieGruneisenEOSTemperature::computePressure(const double rho_orig,
                                                    const double rho_cur) const
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
PortableMieGruneisenEOSTemperature::computePressure(const double rho_orig,
                                                    const double rho_cur,
                                                    double *pressure,
                                                    double *dp_drho,
                                                    double *csquared) const
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
    *pressure = pCompression(rho_orig, eta);
    dp_dJ = dpdJCompression(rho_orig, eta);
  }
  else{
    *pressure = pTension(rho_orig, eta);
    dp_dJ = dpdJTension(rho_orig, eta);
  }
  *dp_drho = dp_dJ*dJ_drho;
  *csquared = dp_dJ/rho_cur;

  if (std::isnan(*pressure) || std::fabs(dp_dJ) < 1.0e-30) {
    std::ostringstream desc;
    desc << "pressure = " << -(*pressure) << " rho_cur = " << rho_cur 
         << " dp_drho = " << -(*dp_drho) << " c^2 = " << (*csquared) << std::endl
         << "File: " << __FILE__ << ", Line: " << __LINE__;
    throw std::runtime_error(desc.str());
  }
  return;
}

// Private method: Compute p for compressive volumetric deformations
// This will return the cold pressure
double 
PortableMieGruneisenEOSTemperature::pCompression(const double rho_0,
                                                 const double eta) const
{
  double J(1-eta);
  double rho(rho_0/J);
  double e_c(0);

  if(fabs(d_const.Gamma_0)>0){
    // strain energy is energy per unit original volume
    e_c = computeStrainEnergy(rho_0,rho)/rho_0;
  }

  // Calculate the shock speed:

  double denom = (1.-d_const.S_1*eta-d_const.S_2*eta*eta-d_const.S_3*eta*eta*eta);
  double Us = d_const.C_0/denom;

  double p_H = rho_0 * Us * Us * eta;
  double p = p_H*(1-0.5*d_const.Gamma_0 *eta) + rho_0*d_const.Gamma_0 * e_c;

  return -p;
}

// Private method: Compute dp/dJ for compressive volumetric deformations
double 
PortableMieGruneisenEOSTemperature::dpdJCompression(const double rho_orig,
                                                    const double eta) const
{
  double J = 1-eta;
  double rho_cur = rho_orig/J;

  return computeBulkModulus(rho_orig, rho_cur);
}

// Private method: Compute p for tensile volumetric deformations
double 
PortableMieGruneisenEOSTemperature::pTension(const double rho_0,
                                             const double eta) const
{
  double J(1-eta);
  double rho(rho_0/J);
  double e_c(0);

  if(fabs(d_const.Gamma_0)>0){
    // strain energy is energy per unit original volume
    e_c = computeStrainEnergy(rho_0,rho)/rho_0;
  }

  // Calculate the shock speed:

  double Us = d_const.C_0;

  double p_H = rho_0 * Us * Us * eta;
  double p = p_H*(1-0.5*d_const.Gamma_0 *eta) + rho_0*d_const.Gamma_0 * e_c;

  return -p;
}

// Private method: Compute dp/dJ for tensile volumetric deformations
double 
PortableMieGruneisenEOSTemperature::dpdJTension(const double rho_orig,
                                                const double eta) const
{
  double J = 1-eta;
  double rho_cur = rho_orig/J;

  return computeBulkModulus(rho_orig, rho_cur);
}

// Compute strain energy
//   An exact integral does not exist and numerical integration is needed
//   Use double exponential integration because the function is smooth
//   (even though we use it only in a limited region)
//   **WARNING** Requires well behaved EOS that does not blow up to 
//               infinity in the middle of the domain 
double 
PortableMieGruneisenEOSTemperature::computeStrainEnergy(const double rho_0,
                                                        const double rho) const
{
  // The strain energy is the "cold" internal energy per unit original volume
  // Calculate J 
  double J = rho_0/rho;

  // Calc. eta = 1 - J  (Note that J = 1 - eta)
  double eta = 1. - J;

  double e_c = 0;
  if(eta<=0){
    // Use analytic solution:
    e_c = 0.5*d_const.C_0*d_const.C_0*eta*eta;
  } else {
    const double C0(d_const.C_0);
    const double S1(d_const.S_1);
    const double S2(d_const.S_2);
    const double S3(d_const.S_3);
    const double G0(d_const.Gamma_0);
    const double Cv(d_const.C_v);
    const double T0(d_const.theta_0);

    // Power series expansion using 10 terms: (Computed in mathematica)
#define Power(x, y)     (pow((double)(x), (double)(y)))
    double CvThetaH = (Power(C0,2)*Power(eta,2))/2. +
      ((-3*Power(C0,2)*G0 + 4*Power(C0,2)*S1)*
       Power(eta,3))/6. + (Power(C0,2)*
                         (Power(G0,2) - 3*G0*S1 + 3*Power(S1,2) + 2*S2)*Power(eta,4))/4. - 
      (Power(C0,2)*(5*Power(G0,3) - 24*Power(G0,2)*S1 + 54*G0*Power(S1,2) - 
                    48*Power(S1,3) + 36*G0*S2 - 72*S1*S2 - 24*S3)*Power(eta,5))/60. + 
      (Power(C0,2)*(3*Power(G0,4) - 20*Power(G0,3)*S1 + 
                    72*Power(G0,2)*Power(S1,2) - 144*G0*Power(S1,3) + 120*Power(S1,4) + 
                    48*Power(G0,2)*S2 - 216*G0*S1*S2 + 288*Power(S1,2)*S2 + 
                    72*Power(S2,2) - 72*G0*S3 + 144*S1*S3)*Power(eta,6))/144. - 
      (Power(C0,2)*(7*Power(G0,5) - 60*Power(G0,4)*S1 + 
                    300*Power(G0,3)*Power(S1,2) - 960*Power(G0,2)*Power(S1,3) + 
                    1800*G0*Power(S1,4) - 1440*Power(S1,5) + 200*Power(G0,3)*S2 - 
                    1440*Power(G0,2)*S1*S2 + 4320*G0*Power(S1,2)*S2 - 
                    4800*Power(S1,3)*S2 + 1080*G0*Power(S2,2) - 2880*S1*Power(S2,2) - 
                    480*Power(G0,2)*S3 + 2160*G0*S1*S3 - 2880*Power(S1,2)*S3 - 1440*S2*S3
                    )*Power(eta,7))/1680. + (Power(C0,2)*
                                           (2*Power(G0,6) - 21*Power(G0,5)*S1 + 135*Power(G0,4)*Power(S1,2) - 
                                            600*Power(G0,3)*Power(S1,3) + 1800*Power(G0,2)*Power(S1,4) - 
                                            3240*G0*Power(S1,5) + 2520*Power(S1,6) + 90*Power(G0,4)*S2 - 
                                            900*Power(G0,3)*S1*S2 + 4320*Power(G0,2)*Power(S1,2)*S2 - 
                                            10800*G0*Power(S1,3)*S2 + 10800*Power(S1,4)*S2 + 
                                            1080*Power(G0,2)*Power(S2,2) - 6480*G0*S1*Power(S2,2) + 
                                            10800*Power(S1,2)*Power(S2,2) + 1440*Power(S2,3) - 
                                            300*Power(G0,3)*S3 + 2160*Power(G0,2)*S1*S3 - 
                                            6480*G0*Power(S1,2)*S3 + 7200*Power(S1,3)*S3 - 3240*G0*S2*S3 + 
                                            8640*S1*S2*S3 + 1080*Power(S3,2))*Power(eta,8))/2880. - 
      (Power(C0,2)*(9*Power(G0,7) - 112*Power(G0,6)*S1 + 
                    882*Power(G0,5)*Power(S1,2) - 5040*Power(G0,4)*Power(S1,3) + 
                    21000*Power(G0,3)*Power(S1,4) - 60480*Power(G0,2)*Power(S1,5) + 
                    105840*G0*Power(S1,6) - 80640*Power(S1,7) + 588*Power(G0,5)*S2 - 
                    7560*Power(G0,4)*S1*S2 + 50400*Power(G0,3)*Power(S1,2)*S2 - 
                    201600*Power(G0,2)*Power(S1,3)*S2 + 453600*G0*Power(S1,4)*S2 - 
                    423360*Power(S1,5)*S2 + 12600*Power(G0,3)*Power(S2,2) - 
                    120960*Power(G0,2)*S1*Power(S2,2) + 
                    453600*G0*Power(S1,2)*Power(S2,2) - 604800*Power(S1,3)*Power(S2,2) + 
                    60480*G0*Power(S2,3) - 201600*S1*Power(S2,3) - 2520*Power(G0,4)*S3 + 
                    25200*Power(G0,3)*S1*S3 - 120960*Power(G0,2)*Power(S1,2)*S3 + 
                    302400*G0*Power(S1,3)*S3 - 302400*Power(S1,4)*S3 - 
                    60480*Power(G0,2)*S2*S3 + 362880*G0*S1*S2*S3 - 
                    604800*Power(S1,2)*S2*S3 - 120960*Power(S2,2)*S3 + 
                    45360*G0*Power(S3,2) - 120960*S1*Power(S3,2))*Power(eta,9))/90720. + 
      (Power(C0,2)*(5*Power(G0,8) - 72*Power(G0,7)*S1 + 
                    672*Power(G0,6)*Power(S1,2) - 4704*Power(G0,5)*Power(S1,3) + 
                    25200*Power(G0,4)*Power(S1,4) - 100800*Power(G0,3)*Power(S1,5) + 
                    282240*Power(G0,2)*Power(S1,6) - 483840*G0*Power(S1,7) + 
                    362880*Power(S1,8) + 448*Power(G0,6)*S2 - 7056*Power(G0,5)*S1*S2 + 
                    60480*Power(G0,4)*Power(S1,2)*S2 - 
                    336000*Power(G0,3)*Power(S1,3)*S2 + 
                    1209600*Power(G0,2)*Power(S1,4)*S2 - 2540160*G0*Power(S1,5)*S2 + 
                    2257920*Power(S1,6)*S2 + 15120*Power(G0,4)*Power(S2,2) - 
                    201600*Power(G0,3)*S1*Power(S2,2) + 
                    1209600*Power(G0,2)*Power(S1,2)*Power(S2,2) - 
                    3628800*G0*Power(S1,3)*Power(S2,2) + 
                    4233600*Power(S1,4)*Power(S2,2) + 161280*Power(G0,2)*Power(S2,3) - 
                    1209600*G0*S1*Power(S2,3) + 2419200*Power(S1,2)*Power(S2,3) + 
                    201600*Power(S2,4) - 2352*Power(G0,5)*S3 + 30240*Power(G0,4)*S1*S3 - 
                    201600*Power(G0,3)*Power(S1,2)*S3 + 
                    806400*Power(G0,2)*Power(S1,3)*S3 - 1814400*G0*Power(S1,4)*S3 + 
                    1693440*Power(S1,5)*S3 - 100800*Power(G0,3)*S2*S3 + 
                    967680*Power(G0,2)*S1*S2*S3 - 3628800*G0*Power(S1,2)*S2*S3 + 
                    4838400*Power(S1,3)*S2*S3 - 725760*G0*Power(S2,2)*S3 + 
                    2419200*S1*Power(S2,2)*S3 + 120960*Power(G0,2)*Power(S3,2) - 
                    725760*G0*S1*Power(S3,2) + 1209600*Power(S1,2)*Power(S3,2) + 
                    483840*S2*Power(S3,2))*Power(eta,10))/403200.;
    e_c = exp(G0*eta)*CvThetaH+Cv*T0*(1-exp(G0*eta));
  }
#undef Power
  // correct for finite reference temperature
  double gamma_0 = d_const.Gamma_0;
  double theta_0 = d_const.theta_0;
  double c_v = d_const.C_v;
  
  double tempCorrection = c_v*theta_0 * (1.0 - exp( gamma_0*eta )); 
  e_c += tempCorrection;

  return e_c*rho_0;               // Strain energy is per unit original volume.
}

} // End namespace PTR
