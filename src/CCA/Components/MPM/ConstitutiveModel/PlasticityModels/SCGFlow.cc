/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/SCGFlow.h>

#include <cmath>

#include <iostream>

#include <Core/Exceptions/InvalidValue.h>


using namespace Uintah;
using namespace std;

SCGFlow::SCGFlow(ProblemSpecP& ps)
{
  ps->require("mu_0",d_CM.mu_0); 
  ps->require("A",d_CM.A); 
  ps->require("B",d_CM.B); 
  ps->require("sigma_0",d_CM.sigma_0); 
  ps->require("beta",d_CM.beta); 
  ps->require("n",d_CM.n); 
  ps->require("epsilon_p0",d_CM.epsilon_p0); 
  ps->require("Y_max",d_CM.Y_max); 
  ps->require("T_m0",d_CM.T_m0); 
  ps->require("a",d_CM.a); 
  ps->require("Gamma_0",d_CM.Gamma_0); 

  // Compute C1 and C2
  d_CM.C1 = 0.0;
  d_CM.C2 = 0.0;
  ps->get("C1",d_CM.C1); 
  ps->get("C2",d_CM.C2); 
  double C1 = d_CM.C1;
  double C2 = d_CM.C2;
  if (C1 == 0.0 || C2 == 0.0) {
    ps->require("dislocation_density", d_CM.dislocationDensity);
    ps->require("length_of_dislocation_segment",
                d_CM.lengthOfDislocationSegment);
    ps->require("distance_between_Peierls_valleys",
                d_CM.distanceBetweenPeierlsValleys);
    ps->require("length_of_Burger_vector", d_CM.lengthOfBurgerVector);
    ps->require("Debye_frequency", d_CM.debyeFrequency);
    ps->require("width_of_kink_loop", d_CM.widthOfKinkLoop);
    ps->require("drag_coefficient", d_CM.dragCoefficient);
    double rho = d_CM.dislocationDensity;
    double L = d_CM.lengthOfDislocationSegment;
    double a = d_CM.distanceBetweenPeierlsValleys;
    double b = d_CM.lengthOfBurgerVector;
    double nu = d_CM.debyeFrequency;
    double w = d_CM.widthOfKinkLoop;
    double D = d_CM.dragCoefficient;
    d_CM.C1 = rho*L*a*b*b*nu/(2*w*w);
    d_CM.C2 = D/(rho*b*b);
  }
  ps->require("energy_to_form_kink_pair",d_CM.kinkPairEnergy);
  ps->require("Boltzmann_constant",d_CM.boltzmannConstant);
  ps->require("Peierls_stress",d_CM.peierlsStress);
}
         
SCGFlow::SCGFlow(const SCGFlow* cm)
{
  d_CM.mu_0 = cm->d_CM.mu_0;
  d_CM.A = cm->d_CM.A;
  d_CM.B = cm->d_CM.B;
  d_CM.sigma_0 = cm->d_CM.sigma_0;
  d_CM.beta = cm->d_CM.beta;
  d_CM.n = cm->d_CM.n;
  d_CM.epsilon_p0 = cm->d_CM.epsilon_p0;
  d_CM.Y_max = cm->d_CM.Y_max;
  d_CM.T_m0 = cm->d_CM.T_m0;
  d_CM.a = cm->d_CM.a;
  d_CM.Gamma_0 = cm->d_CM.Gamma_0;
  d_CM.C1 = cm->d_CM.C1; 
  d_CM.C2 = cm->d_CM.C2; 
  d_CM.dislocationDensity = cm->d_CM.dislocationDensity;
  d_CM.lengthOfDislocationSegment = cm->d_CM.lengthOfDislocationSegment;
  d_CM.distanceBetweenPeierlsValleys = cm->d_CM.distanceBetweenPeierlsValleys;
  d_CM.lengthOfBurgerVector = cm->d_CM.lengthOfBurgerVector;
  d_CM.debyeFrequency = cm->d_CM.debyeFrequency;
  d_CM.widthOfKinkLoop = cm->d_CM.widthOfKinkLoop;
  d_CM.dragCoefficient = cm->d_CM.dragCoefficient;
  d_CM.kinkPairEnergy = cm->d_CM.kinkPairEnergy;
  d_CM.boltzmannConstant = cm->d_CM.boltzmannConstant;
  d_CM.peierlsStress = cm->d_CM.peierlsStress;
}
         
SCGFlow::~SCGFlow()
{
}

void SCGFlow::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP flow_ps = ps->appendChild("flow_model");
  flow_ps->setAttribute("type","steinberg_cochran_guinan");

  flow_ps->appendElement("mu_0",d_CM.mu_0); 
  flow_ps->appendElement("A",d_CM.A); 
  flow_ps->appendElement("B",d_CM.B); 
  flow_ps->appendElement("sigma_0",d_CM.sigma_0); 
  flow_ps->appendElement("beta",d_CM.beta); 
  flow_ps->appendElement("n",d_CM.n); 
  flow_ps->appendElement("epsilon_p0",d_CM.epsilon_p0); 
  flow_ps->appendElement("Y_max",d_CM.Y_max); 
  flow_ps->appendElement("T_m0",d_CM.T_m0); 
  flow_ps->appendElement("a",d_CM.a); 
  flow_ps->appendElement("Gamma_0",d_CM.Gamma_0); 

  // Compute C1 and C2
  flow_ps->appendElement("C1",d_CM.C1); 
  flow_ps->appendElement("C2",d_CM.C2); 
  double C1 = d_CM.C1;
  double C2 = d_CM.C2;
  if (C1 == 0.0 || C2 == 0.0) {
    flow_ps->appendElement("dislocation_density", d_CM.dislocationDensity);
    flow_ps->appendElement("length_of_dislocation_segment",
                d_CM.lengthOfDislocationSegment);
    flow_ps->appendElement("distance_between_Peierls_valleys",
                d_CM.distanceBetweenPeierlsValleys);
    flow_ps->appendElement("length_of_Burger_vector", 
                              d_CM.lengthOfBurgerVector);
    flow_ps->appendElement("Debye_frequency", d_CM.debyeFrequency);
    flow_ps->appendElement("width_of_kink_loop", d_CM.widthOfKinkLoop);
    flow_ps->appendElement("drag_coefficient", d_CM.dragCoefficient);
  }
  flow_ps->appendElement("energy_to_form_kink_pair",d_CM.kinkPairEnergy);
  flow_ps->appendElement("Boltzmann_constant",d_CM.boltzmannConstant);
  flow_ps->appendElement("Peierls_stress",d_CM.peierlsStress);
}

         
void 
SCGFlow::addInitialComputesAndRequires(Task* ,
                                          const MPMMaterial* ,
                                          const PatchSet*)
{
}

void 
SCGFlow::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet*)
{
}

void 
SCGFlow::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*,
                                   bool ,
                                   bool )
{
}

void 
SCGFlow::addParticleState(std::vector<const VarLabel*>& ,
                             std::vector<const VarLabel*>& )
{
}

void 
SCGFlow::allocateCMDataAddRequires(Task* ,
                                      const MPMMaterial* ,
                                      const PatchSet* ,
                                      MPMLabel* )
{
}

void 
SCGFlow::allocateCMDataAdd(DataWarehouse* ,
                              ParticleSubset* ,
                              map<const VarLabel*, 
                                 ParticleVariableBase*>* ,
                              ParticleSubset* ,
                              DataWarehouse* )
{
}



void 
SCGFlow::initializeInternalVars(ParticleSubset* ,
                                   DataWarehouse* )
{
}

void 
SCGFlow::getInternalVars(ParticleSubset* ,
                            DataWarehouse* ) 
{
}

void 
SCGFlow::allocateAndPutInternalVars(ParticleSubset* ,
                                       DataWarehouse* ) 
{
}

void
SCGFlow::allocateAndPutRigid(ParticleSubset* ,
                                DataWarehouse* )
{
}

void
SCGFlow::updateElastic(const particleIndex )
{
}

void
SCGFlow::updatePlastic(const particleIndex , const double& )
{
}

double 
SCGFlow::computeFlowStress(const PlasticityState* state,
                              const double& ,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Calculate mu/mu_0
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate sigma_A <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep + d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double sigma_A = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  // Calculate the thermal part of the yield stress using 
  // the Hoge and Mukherjee model
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double sigma_T = computeThermallyActivatedYieldStress(epdot, T, 1.0e-6);

  double sigma = (sigma_T + sigma_A)*mu_mu_0;
  return sigma;
}

double 
SCGFlow::computeThermallyActivatedYieldStress(const double& epdot,
                                                 const double& T,
                                                 const double& tolerance)
{
  // If the strain rate is very small return 0.0
  if (epdot < tolerance) return 0.0;

  // Get the Hoge and Mukherjee model constants
  double C1 = d_CM.C1;
  double C2 = d_CM.C2;
  double U_k = d_CM.kinkPairEnergy;
  double kappa = d_CM.boltzmannConstant;
  double sigma_P = d_CM.peierlsStress;
  double U_k_kappaT = 2.0*U_k/(kappa*T);
  double Z4 = C2*C1;

  // Find the maximum plastic strain rate and minumum plastic
  // strains at which this part of the calculation is valid
  double tau = sigma_P;
  double Z0 = 1.0-tau/sigma_P;
  double Z1 = U_k_kappaT*Z0;
  double Z2 = exp(Z0*Z1);
  double Z5 = Z2*tau;
  double Z6 = Z4 + Z5;
  double epdot_max =  C1*tau/Z6;
  if (epdot > epdot_max) return tau;

  tau = 0.0;
  Z0 = 1.0-tau/sigma_P;
  Z1 = U_k_kappaT*Z0;
  Z2 = exp(Z0*Z1);
  Z5 = Z2*tau;
  Z6 = Z4 + Z5;
  double epdot_min = C1*tau/Z6;
  if (epdot < epdot_min) return tau;

  // Do Newton iteration
  double f = 0.0;
  double fPrime = 0.0;
  tau = 0.5*sigma_P;
  double tauOld = tau;
  double count = 0;
  do {
    ++count;
    Z0 = 1.0-tau/sigma_P;
    Z1 = U_k_kappaT*Z0;
    Z2 = exp(Z0*Z1);
    Z5 = Z2*tau;
    Z6 = Z4 + Z5;
    f = epdot - C1*tau/Z6;
    fPrime = -C1*(2.0*Z5*Z1*tau + sigma_P*Z4)/(sigma_P*Z6*Z6);

    tauOld = tau;
    tau -= f/fPrime;

    if (isnan(tau)) {
      //cout << "iter = " << count << " epdot = " << epdot 
      //     << " T = " << T << endl;
      //cout << "iter = " << count << " Z0 = " << Z0 << " Z1 = " << Z1
      //   << " Z2 = " << Z2 << " Z4 = " << Z4 << " Z5 = " << Z5
      //   << " Z6 = " << Z6 << " C1 = " << C1 << " C2 = " << C2 << endl;
      //cout << "iter = " << count 
      //     << " f = " << fabs(f) << " fPrime = " << fPrime 
      //   << " tau = " << tau << " tolerance = " << tolerance
      //   << " tau-tauOld = " << fabs(tau-tauOld) << endl;
      break;
    }
    if (fabs(tau-tauOld) < tolerance*tau) break;

  } while (fabs(f) > tolerance);
  
  /* The equation is not appropriate for Newton iterations.
     Do bisection instead. */
  if (isnan(tau)) {
    double tau_hi = sigma_P;
    double tau_lo = tolerance; 
    tau = 0.5*(tau_hi + tau_lo);
    while ((tau_hi - tau_lo) > tolerance) {

      // Compute f(tau_lo)
      Z0 = 1.0-tau_lo/sigma_P;
      Z1 = U_k_kappaT*Z0;
      Z2 = exp(Z0*Z1);
      Z5 = Z2*tau_lo;
      Z6 = Z4 + Z5;
      double f_lo = epdot - C1*tau_lo/Z6;

      // Compute f(tau)
      Z0 = 1.0-tau/sigma_P;
      Z1 = U_k_kappaT*Z0;
      Z2 = exp(Z0*Z1);
      Z5 = Z2*tau;
      Z6 = Z4 + Z5;
      f = epdot - C1*tau/Z6;

      // Check closeness
      if (f_lo*f > 0.0) 
        tau_lo = tau;
      else
        tau_hi = tau;

      // Compute new value of tau
      tau = 0.5*(tau_hi + tau_lo);
    }
  }

  if (isnan(tau)) {
    cout << "iter = " << count << " epdot = " << epdot 
         << " T = " << T << endl;
    cout << "iter = " << count << " Z0 = " << Z0 << " Z1 = " << Z1
       << " Z2 = " << Z2 << " Z4 = " << Z4 << " Z5 = " << Z5
       << " Z6 = " << Z6 << " C1 = " << C1 << " C2 = " << C2 << endl;
    cout << "iter = " << count 
         << " f = " << fabs(f) << " fPrime = " << fPrime 
       << " tau = " << tau << " tolerance = " << tolerance
       << " tau-tauOld = " << fabs(tau-tauOld) << endl;
  }
  tau = (tau > sigma_P) ? sigma_P : tau;
  tau = (tau < 0.0) ? 0.0 : tau;

  return tau;
}

double 
SCGFlow::computeEpdot(const PlasticityState* state,
                         const double& ,
                         const double& ,
                         const MPMMaterial* ,
                         const particleIndex )
{
  // Get the needed data
  double tau = state->yieldStress;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double mu = state->shearModulus;

  // Get the Hoge and Mukerhjee model constants
  double mu_0 = d_CM.mu_0;
  double C1 = d_CM.C1;
  double C2 = d_CM.C2;
  double U_k = d_CM.kinkPairEnergy;
  double kappa = d_CM.boltzmannConstant;
  double sigma_P = d_CM.peierlsStress;

  // Compute the sigma_A and sigma_T
  double f_ep = 1.0 + d_CM.beta*(ep + d_CM.epsilon_p0);
  ASSERT(f_ep >= 0.0);
  double sigma_A = Min(d_CM.sigma_0*pow(f_ep, d_CM.n), d_CM.Y_max);
  double sigma_T = tau*(mu_0/mu) - sigma_A;
  
  double t1 = 1.0 - sigma_T/sigma_P;
  double t2 = (1.0/C1)*exp(2.0*U_k*t1*t1/(kappa*T)) + C2/sigma_T;;
  double epdot = 1.0/t2;

  return epdot;
}


void 
SCGFlow::computeTangentModulus(const Matrix3& stress,
                                  const PlasticityState* state,
                                  const double& ,
                                  const MPMMaterial* ,
                                  const particleIndex idx,
                                  TangentModulusTensor& Ce,
                                  TangentModulusTensor& Cep)
{
  throw InternalError("Empty Function: SCGFlow::computeTangentModulus", __FILE__, __LINE__); 
}

void
SCGFlow::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTPressure(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
SCGFlow::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                           const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Calculate mu/mu_0
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep + d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = pow(Ya, d_CM.n-1.0);

  double dsigma_dep = d_CM.sigma_0*mu_mu_0*d_CM.n*d_CM.beta*Y;
  return dsigma_dep;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
SCGFlow::computeShearModulus(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);
  double P = -state->pressure;
  double mu = d_CM.mu_0*(1.0 + d_CM.A*P/eta - 
              d_CM.B*(state->temperature - 300.0));
  return mu;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
SCGFlow::computeMeltingTemp(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  double power = 2.0*(d_CM.Gamma_0 - d_CM.a - 1.0/3.0);
  double Tm = d_CM.T_m0*exp(2.0*d_CM.a*(1.0 - 1.0/eta))*
              pow(eta,power);
  return Tm;
}

/*  ** WARNING ** NOT COMPLETE
    This assumes that the SCG shear modulus model is being used 
    The strain rate dependent term in the Steinberg-Lund version
    of the model has not been included and should be for correctness.*/
double
SCGFlow::evalDerivativeWRTTemperature(const PlasticityState* state,
                                         const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep + d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  return -Y*d_CM.B;
}

double
SCGFlow::evalDerivativeWRTPressure(const PlasticityState* state,
                                      const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep + d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  double eta = state->density/state->initialDensity;
  return Y*d_CM.A/pow(eta,1.0/3.0);
}

/*! The yield function is given by
    \f[
     Y = [Y_t(epdot,T) + Y_a(ep)]*\mu(p,T)/\mu_0
    \f]
    The derivative wrt epdot is
    \f[
      dY/depdot = dY_t/depdot*\mu/\mu_0
    \f]

    The equation for Y_t in terms of epdot can be expressed as
    \f[
      A(1 - B3 Y_t)^2 - ln(B1 Y_t - B2) + ln(Y_t) = 0
    \f]
    where \f$ A = 2*U_k/(\kappa T) \f$, \f$ B1 = C1/epdot \f$ \n
    \f$ B2 = C1 C2 \f$, and \f$ B3 = 1/Y_p \f$.\n

    The solution of this equation is 
    \f[
      Y_t(epdot,T) = exp(RootOf(Z + A - 2 A B3 exp(Z) (1 - B3 exp(Z))
       - ln(B1 exp(Z) - B2) = 0))
    \f]
    The root is determined using a Newton iterative technique.

    The derivative is given by
    \f[
      dY_t/depdot = -B1 X1^2/[X4(2 X2 - 1) - 2 X3(C1(1-B3 X1) + B3 X4)]
    \f]
    where \f$ X1 = exp(Z) \f$, \f$ X2 = A B3 X1 \f$, \f$ X3 = X2 X1 \f$, \n
    \f$ X4 = B2 epdot \f$.
*/
double
SCGFlow::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                        const particleIndex )
{
  // Get the current state data
  double epdot = state->plasticStrain;
  double T = state->temperature;
  double mu = state->shearModulus;
  double mu_0 = d_CM.mu_0;

  // Compute the value of tau at two strain rates
  double tau_1 = computeThermallyActivatedYieldStress(epdot, T, 1.0e-6);
  double tau_2 = computeThermallyActivatedYieldStress(1.01*epdot, T, 1.0e-6);

  // Compute the slope delta(tau)/delta(epdot)
  double dYt_depdot = (tau_2 - tau_1)/0.01;
  double dY_depdot = dYt_depdot*mu/mu_0;
  return dY_depdot;

  /*
  // ** WARNING ** After a lot of trial I have found that derivatives
  // wrt strain rate do not exist at many points.  So I'm going to 
  // return zero for now until I find a better way of doing things.
  // BB 5/2/05
  return 0.0;

  // Get the Hoge and Mukherjee model constants
  double C1 = d_CM.C1;
  double C2 = d_CM.C2;
  double U_k = d_CM.kinkPairEnergy;
  double kappa = d_CM.boltzmannConstant;
  double sigma_P = d_CM.peierlsStress;
  double U_k_kappa = U_k/kappa;

  // Find the maximum plastic strain rate and minumum plastic
  // strains at which this part of the calculation is valid
  double tau = sigma_P;
  double Z0 = 1.0-tau/sigma_P;
  double Z1 = U_k_kappa*Z0/T;
  double Z2 = exp(2.0*Z0*Z1);
  double Z4 = C2*C1;
  double Z5 = Z2*tau;
  double Z6 = Z4 + Z5;
  double epdot_max =  C1*tau/Z6;
  if (epdot > epdot_max) return 0.0;

  // If the strain rate is very small return 0.0
  if (epdot < 1.0e-6) return 0.0;

  // Compute constants
  double A = 2.0*U_k_kappa/T;
  double B1 = C1/epdot;
  double B2 = C1*C2; 
  double B3 = 1.0/sigma_P;
  double X4 = B2*epdot;

  // Solve Z + A - 2 A B3 exp(Z) (1 - B3 exp(Z)) - ln(B1 exp(Z) - B2) = 0)
  double Z = log(1.1*C2*epdot);
  double g = 0.0;
  double Dg = 0.0;
  double X1 = 0.0, X2 = 0.0, X3 = 0.0, X5 = 0.0, X6 = 0.0, X7 = 0.0;
  double count = 0;
  do {

    ++count;

    // Compute variables
    X1 = exp(Z);
    X2 = A*B3*X1; 
    X3 = X2*X1;
    X5 = B1*X1;
    X6 = B3*X3;
    X7 = X5 - B2;
    if (X7 <= 0.0) {
      g = Z + A - 2.0*X2 + X6 - ln(C1) - 1.0;
      Dg = 1.0 - 2.0*X2 + 2.0*X6;
    } else {
      g = Z + A - 2.0*X2 + X6 - log(X7);
      Dg = 1.0 - 2.0*X2 + 2.0*X6 - X5/X7;
    }
    Z -= g/Dg;

    if (isnan(g) || isnan(Z) || idx == 4924) {
      cout << "iter = " << count << " g = " << g << " Dg = " << Dg 
           << " Z = " << Z << " epdot = " << epdot << " T = " << T
           << " A = " << A << " B1 = " << B1 << " B2 = " << B2
           << " B3 = " << B3 << " X4 = " << X4 << " X1 = " << X1
           << " X2 = " << X2 << " X3 = " << X3 << " X5 = " << X5
           << " X6 = " << X6 << " X7 = " << X7 << endl;
    }
  } while (fabs(g) > 1.0e-3);

  // Compute derivative
  X1 = exp(Z);
  X2 = A*B3*X1; 
  X3 = X2*X1;
  double denom = X4*(2.0*X2 - 1.0) - 2.0*X3*(C1*(1.0-B3*X1) + B3*X4);
  if (denom == 0.0) {
    cout << " denom = " << denom << endl;
  }
  double dYt_depdot = -B1*X1*X1/denom;
  double dY_depdot = dYt_depdot*mu/mu_0;
  return dY_depdot;
  */
}
