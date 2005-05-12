#include "PTWPlastic.h"
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


PTWPlastic::PTWPlastic(ProblemSpecP& ps)
{
  ps->require("theta",d_CM.theta);
  ps->require("p",d_CM.p);
  ps->require("s0",d_CM.s0);
  ps->require("sinf",d_CM.sinf);
  ps->require("kappa",d_CM.kappa);
  ps->require("gamma",d_CM.gamma);
  ps->require("y0",d_CM.y0);
  ps->require("yinf",d_CM.yinf);
  ps->require("y1",d_CM.y1);
  ps->require("y2",d_CM.y2);
  ps->require("beta",d_CM.beta);
  ps->require("M",d_CM.M);
}
         
PTWPlastic::PTWPlastic(const PTWPlastic* cm)
{
  d_CM.theta = cm->d_CM.theta;
  d_CM.p = cm->d_CM.p;
  d_CM.s0 = cm->d_CM.s0;
  d_CM.sinf = cm->d_CM.sinf;
  d_CM.kappa = cm->d_CM.kappa;
  d_CM.gamma = cm->d_CM.gamma;
  d_CM.y0 = cm->d_CM.y0;
  d_CM.yinf = cm->d_CM.yinf;
  d_CM.y1 = cm->d_CM.y1;
  d_CM.y2 = cm->d_CM.y2;
  d_CM.beta = cm->d_CM.beta;
  d_CM.M = cm->d_CM.M;
}
         
PTWPlastic::~PTWPlastic()
{
}
         
double 
PTWPlastic::computeFlowStress(const PlasticityState* state,
                              const double& delT,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex idx)
{
  // Retrieve plastic strain and strain rate
  double epdot = state->plasticStrainRate;
  epdot = (epdot <= 0.0) ? 1.0e-8 : epdot;
  
  double ep = state->plasticStrain;

  // Check if temperature is correct
  double T = state->temperature;
  double Tm = state->meltingTemp;
  ASSERT(T > 0.0); ASSERT(!(T > Tm));

  // Check if shear modulus is correct
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Get the current mass density
  double rho = state->density;
  
  // Convert the atomic mass to kg
  double Mkg = d_CM.M*1.66043998903379e-27;
  // Compute invxidot - the time required for a transverse wave to cross 
  // an atom
  if (mu < 0.0 || rho < 0.0) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: mu = " << mu 
         << " rho = " << rho << endl;
  }
  
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*Mkg),(1.0/3.0))*sqrt(mu/rho);

  // Compute the dimensionless plastic strain rate
  double edot = epdot/xidot;
  if (!(xidot > 0.0) || !(edot > 0.0)) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: xidot = " << xidot 
         << " edot = " << edot << endl;
  }

  // Compute the dimensionless temperature
  double That = T/Tm;

  // Calculate the dimensionless Arrhenius factor
  double arrhen = d_CM.kappa*That*log(d_CM.gamma/edot);

  // Calculate the saturation hardening flow stress in the thermally 
  // activated glide regime
  double tauhat_s = d_CM.s0 - (d_CM.s0 - d_CM.sinf)*erf(arrhen);

  // Calculate the yield stress in the thermally activated glide regime
  double tauhat_y = d_CM.y0 - (d_CM.y0 - d_CM.yinf)*erf(arrhen);

  // The overdriven shock regime
  if (epdot > 1.0e8) {

    // Calculate the saturation hardening flow stress in the overdriven 
    // shock regime
    double shock_tauhat_s = d_CM.s0*pow(edot/d_CM.gamma,d_CM.beta);

    // Calculate the yield stress in the overdriven shock regime
    double shock_tauhat_y_jump = d_CM.y1*pow(edot/d_CM.gamma,d_CM.y2);
    double shock_tauhat_y = min(shock_tauhat_y_jump,shock_tauhat_s);

    // Calculate the saturation stress and yield stress
    tauhat_s = max(tauhat_s, shock_tauhat_s);
    tauhat_y = max(tauhat_y, shock_tauhat_y);
  }

  // Compute the dimensionless flow stress
  double tauhat = tauhat_s;
  if (tauhat_s != tauhat_y) {
    double A = (d_CM.s0 - tauhat_y)/d_CM.p;
    double B = tauhat_s - tauhat_y;
    double D = exp(B/A);
    double C = D - 1.0;
    double F = C/D;
    double E = d_CM.theta/(A*C);
    double exp_EEp = exp(-E*ep);
    tauhat = tauhat_s + A*log(1.0 - F*exp_EEp);
  }
  double sigma = 2.0*tauhat*mu;
  return sigma;
}

// **WARNING** We compute these values only for the smooth part of the
//             yield surface (i.e., no overdriven shock regime included)
// (The derivative was computed using Maple)
double 
PTWPlastic::computeEpdot(const PlasticityState* state,
                         const double& delT,
                         const double& tolerance,
                         const MPMMaterial* ,
                         const particleIndex idx)
{
  // Get the needed data
  double tau = state->yieldStress;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tm = state->meltingTemp;
  double That = T/Tm;
  double mu = state->shearModulus;
  double rho = state->density;

  // Do Newton iteration
  double epdot = 1.0;
  double f = 0.0;
  double fPrime = 0.0;
  do {
    evalFAndFPrime(tau, epdot, ep, rho, That, mu, delT, f, fPrime);
    epdot -= f/fPrime;
  } while (fabs(f) > tolerance);

  return epdot;
}

// **WARNING** We compute these values only for the smooth part of the
//             yield surface (i.e., no overdriven shock regime included)
// (The derivative was computed using Maple)
void 
PTWPlastic::evalFAndFPrime(const double& tau,
                           const double& epdot,
                           const double& ep,
                           const double& rho,
                           const double& That,
                           const double& mu,
                           const double& delT,
                           double& f,
                           double& fPrime)
{
  double Mkg = d_CM.M*1.66043998903379e-27;
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*Mkg),(1.0/3.0))*sqrt(mu/rho);

  // Compute the dimensionless plastic strain rate
  double edot = epdot/xidot;
  if (!(xidot > 0.0)) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: xidot = " << xidot 
         << " edot = " << edot << endl;
  }

  // Calculate the dimensionless Arrhenius factor
  double upsilon = d_CM.kappa*That*log(d_CM.gamma/edot);

  // Set up the other constants
  double alpha = d_CM.s0 - d_CM.sinf;
  double beta = d_CM.y0 - d_CM.yinf;
  double delta = d_CM.s0 - d_CM.y0;
  double phi = alpha - beta;
  double zeta = erf(upsilon);
  double eta = phi*zeta;
  double lambda = beta*zeta;
  double sigma = delta + lambda;
  double Xi = delta - eta ;
  double iota = d_CM.p*Xi/sigma;
  double Phi = phi + beta;
  double Lambda = eta + Xi;
  double Delta = exp(iota) - 1.0;
  double Theta = d_CM.theta*d_CM.p*ep/(sigma*Delta);
  double Omega = exp(-Theta)/exp(iota);
  double Z1 = -Delta*Omega + 1.0;
  double Z2 = Phi*Lambda*d_CM.p;
  double Z3 = Theta*sigma;
  double Z4 = Theta*Z2;
  double Z5 = log(Z1);
  double Z6 = 2*d_CM.kappa*That/(d_CM.p*epdot);
  double Z7 = exp(-upsilon*upsilon)*Z6;
  double Z8 = Z1*sigma;
  double Z9 = Omega*Delta;
  double Z10 = Z5*Z8;
  double Z11 = Z3*Z9; 
  double Z12 = Z7/(Z1*sigma*sqrt(M_PI)); 
  double Z13 = Phi - phi;

  double tauhat = d_CM.s0 - alpha*zeta + sigma*Z5/d_CM.p;
  double dtauhatdpsi = Z12*(Z13*(Z11-Z10) - Z4*(Z9+Omega) +
                       (alpha*d_CM.p*Z8 - Omega*Z2));

  f = tau - tauhat*2.0*mu;
  fPrime = - dtauhatdpsi*2.0*mu;
}

/*! The evolving internal variable is \f$q = \epsilon_p\f$.  If the 
  evolution equation for internal variables is of the form 
  \f$ \dot q = \gamma h (\sigma, q) \f$, then 
  \f[
  \dot q = \frac{d\epsilon_p}{dt} = \dot\epsilon_p .
  \f] 
  If \f$\dot\epsilon_p = \gamma\f$, then \f$ h = 1 \f$.
  Also, \f$ f_q = \frac{\partial f}{\partial \epsilon_p} \f$.
  For the von Mises yield condition, \f$(f)\f$, 
  \f$ f_q = \frac{\partial \sigma}{\partial \epsilon_p} \f$
  where \f$\sigma\f$ is the PTW flow stress.
*/
void 
PTWPlastic::computeTangentModulus(const Matrix3& stress,
                                  const PlasticityState* state,
                                  const double& ,
                                  const MPMMaterial* ,
                                  const particleIndex idx,
                                  TangentModulusTensor& Ce,
                                  TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress and strain rate
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Get f_q1 = dsigma/dep (h = 1, therefore f_q.h = f_q)
  double f_q1 = evalDerivativeWRTPlasticStrain(state, idx);

  // Form the elastic-plastic tangent modulus
  Matrix3 Cr, rC;
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      Cr(ii,jj) = 0.0;
      rC(ii,jj) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cr(ii,jj) += Ce(ii,jj,kk,ll)*rr(kk,ll);
          rC(ii,jj) += rr(kk,ll)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii,jj)*rr(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
            Cr(ii,jj)*rC(kk,ll)/(-f_q1 + rCr);
        }  
      }  
    }  
  }  
}

void
PTWPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
PTWPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                           const particleIndex idx)
{
  // Retrieve plastic strain and strain rate
  double epdot = state->plasticStrainRate;
  epdot = (epdot <= 0.0) ? 1.0e-8 : epdot;
  double ep = state->plasticStrain;

  // Check if temperature is correct
  double T = state->temperature;
  double Tm = state->meltingTemp;
  ASSERT(T > 0.0);
  ASSERT(!(T > Tm));

  // Check if shear modulus is correct
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Get the current mass density
  double rho = state->density;

  // Compute invxidot - the time required for a transverse wave to cross at atom
  double Mkg = d_CM.M*1.66043998903379e-27;
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*Mkg),(1.0/3.0))*sqrt(mu/rho);

  // Compute the dimensionless plastic strain rate
  double edot = epdot/xidot;

  // Compute the dimensionless temperature
  double That = T/Tm;

  // Calculate the dimensionless Arrhenius factor
  double arrhen = d_CM.kappa*That*log(d_CM.gamma/edot);

  // Calculate the saturation hardening flow stress in the thermally 
  // activated glide regime
  double thermal_tauhat_s = d_CM.s0 - (d_CM.s0 - d_CM.sinf)*erf(arrhen);

  // Calculate the saturation hardening flow stress in the overdriven 
  // shock regime
  double shock_tauhat_s = d_CM.s0*pow(edot/d_CM.gamma,d_CM.beta);

  // Calculate the yield stress in the thermally activated glide regime
  double thermal_tauhat_y = d_CM.y0 - (d_CM.y0 - d_CM.yinf)*erf(arrhen);

  // Calculate the yield stress in the overdriven shock regime
  double shock_tauhat_y_jump = d_CM.y1*pow(edot/d_CM.gamma,d_CM.y2);
  double shock_tauhat_y = min(shock_tauhat_y_jump,shock_tauhat_s);

  // Calculate the saturation stress and yield stress
  double tauhat_s = max(thermal_tauhat_s, shock_tauhat_s);
  double tauhat_y = max(thermal_tauhat_y, shock_tauhat_y);

  // Compute the derivative of the flow stress
  double deriv = 0.0; // Assume no strain hardening at high rates
  if (tauhat_s != tauhat_y) {
    double A = (d_CM.s0 - tauhat_y)/d_CM.p;
    double B = tauhat_s - tauhat_y;
    double D = exp(B/A);
    double C = D - 1.0;
    double F = C/D;
    double E = d_CM.theta/(A*C);

    // The following can lead to a division by zero if ep = 0
    //double F_exp_Eep = F/exp(E*ep);
    //deriv = A*E*F_exp_Eep/(1.0 - F_exp_Eep);

    // Alternative approach
    double exp_Eep_F = exp(E*ep)/F;
    deriv = A*E/(exp_Eep_F - 1.0);
  }
  deriv *= 2.0*mu;
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
PTWPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->initialShearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
PTWPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
PTWPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
                                         const particleIndex )
{
  // Get the state data
  double mu = state->shearModulus;
  double rho = state->density;
  double epdot = state->plasticStrainRate;
  epdot = (epdot <= 0.0) ? 1.0e-8 : epdot;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tm = state->meltingTemp;
  double That = T/Tm;

  // Compute the dimensionless plastic strain rate
  double Mkg = d_CM.M*1.66043998903379e-27;
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*Mkg),(1.0/3.0))*sqrt(mu/rho);
  double edot = epdot/xidot;
  if (!(xidot > 0.0)) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: xidot = " << xidot 
         << " edot = " << edot << endl;
  }

  // Set up constants
  double alpha = d_CM.s0 - d_CM.sinf;
  double beta = d_CM.y0 - d_CM.yinf;
  double delta = d_CM.s0 - d_CM.y0;
  double phi = alpha - beta;
  double X_1 = log(d_CM.gamma/edot);
  double upsilon = d_CM.kappa*That*X_1;
  double zeta =  erf(upsilon);
  double lambda =  beta*zeta;
  double eta =  phi*zeta;
  double sigma =  delta + lambda;
  double Xi =  -lambda+sigma-eta;
  double iota =  d_CM.p*Xi/sigma;
  double Phi =  phi+beta;
  double Delta =  exp(iota)-1.0;
  double Theta =  d_CM.theta*d_CM.p*ep/(sigma*Delta);
  double Omega =  exp(-Theta)/exp(iota);
  double Z1 =  -Delta*Omega+1.0;
  double Z3 =  Theta*sigma;
  double Z5 =  log(Z1);
  double Z7 =  exp(-upsilon*upsilon);
  double Z8 =  Z1*sigma;
  double Z9 =  Omega*Delta;
  double Z10 =  Z5*Z8;
  double Z11 =  Z3*Z9;
  double Z12 =  Z7/(Z1*sigma);
  double X_3 =  Theta*Phi;
  double X_4 =  Omega*d_CM.p;
  double X_5 =  Z9*X_3;
  double X_6 =  phi*zeta;
  double X_7 =  lambda*X_5;
  double X_8 = lambda*X_4;

  // Derivative of tauhat wrt T
  double dtauhat_dT = (2.0*d_CM.kappa*X_1*Z12)/(d_CM.p*sqrt(M_PI))*
    (Phi*(-d_CM.p*Z11 - Z10 + X_8 + Z11 + zeta*X_5 + (-Z3 - sigma)*X_4) +
     Theta*Z9*(X_6 + lambda)*phi + X_3*X_8 + d_CM.p*(X_7 + alpha*Z8) -
     2.0*X_5*X_6 - X_7 + (-Z11 + Z10)*phi);

  // Derivative of sigma_y wrt T
  return (dtauhat_dT*2.0*mu);
}

double
PTWPlastic::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                        const particleIndex )
{
  // Get the state data
  double mu = state->shearModulus;
  double rho = state->density;
  double epdot = state->plasticStrainRate;
  epdot = (epdot <= 0.0) ? 1.0e-8 : epdot;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tm = state->meltingTemp;
  double That = T/Tm;

  // Compute the dimensionless plastic strain rate
  double Mkg = d_CM.M*1.66043998903379e-27;
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*Mkg),(1.0/3.0))*sqrt(mu/rho);
  double edot = epdot/xidot;
  if (!(xidot > 0.0)) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: xidot = " << xidot 
         << " edot = " << edot << endl;
  }

  // Calculate the dimensionless Arrhenius factor
  double upsilon = d_CM.kappa*That*log(d_CM.gamma/edot);

  // Set up the other constants
  double alpha = d_CM.s0 - d_CM.sinf;
  double beta = d_CM.y0 - d_CM.yinf;
  double delta = d_CM.s0 - d_CM.y0;
  double phi = alpha - beta;
  double zeta = erf(upsilon);
  double eta = phi*zeta;
  double lambda = beta*zeta;
  double sigma = delta + lambda;
  double Xi = delta - eta ;
  double iota = d_CM.p*Xi/sigma;
  double Phi = phi + beta;
  double Lambda = eta + Xi;
  double Delta = exp(iota) - 1.0;
  double Theta = d_CM.theta*d_CM.p*ep/(sigma*Delta);
  double Omega = exp(-Theta)/exp(iota);
  double Z1 = -Delta*Omega + 1.0;
  double Z2 = Phi*Lambda*d_CM.p;
  double Z3 = Theta*sigma;
  double Z4 = Theta*Z2;
  double Z5 = log(Z1);
  double Z6 = 2*d_CM.kappa*That/(d_CM.p*epdot);
  double Z7 = exp(-upsilon*upsilon)*Z6;
  double Z8 = Z1*sigma;
  double Z9 = Omega*Delta;
  double Z10 = Z5*Z8;
  double Z11 = Z3*Z9; 
  double Z12 = Z7/(Z1*sigma*sqrt(M_PI)); 
  double Z13 = Phi - phi;

  // Derivative of tauhat wrt epdot
  double dtauhatdpsi = Z12*(Z13*(Z11-Z10) - Z4*(Z9+Omega) +
                       (alpha*d_CM.p*Z8 - Omega*Z2));

  // Also calculate the slope in the overdriven shock regime
  if (epdot > 1.0e8) {
    double dtau_dpsi_OD = d_CM.beta*d_CM.s0*
	    pow(edot/d_CM.gamma, d_CM.beta)/epdot;
    dtauhatdpsi = max(dtauhatdpsi, dtau_dpsi_OD);
  }

  // Derivative of sigma_y wrt epdot
  return (dtauhatdpsi*2.0*mu);
}

//------------------------------------------------------------------------------
//  Methods needed by Uintah Computational Framework
//------------------------------------------------------------------------------
void 
PTWPlastic::addInitialComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet*) const
{
}

void 
PTWPlastic::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*) const
{
}

void 
PTWPlastic::addParticleState(std::vector<const VarLabel*>& from,
                             std::vector<const VarLabel*>& to)
{
}

void 
PTWPlastic::allocateCMDataAddRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patch,
                                      MPMLabel* lb) const
{
}

void 
PTWPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
                              ParticleSubset* addset,
                              map<const VarLabel*, 
                                ParticleVariableBase*>* newState,
                              ParticleSubset* delset,
                              DataWarehouse* old_dw)
{
}

void 
PTWPlastic::initializeInternalVars(ParticleSubset* pset,
                                   DataWarehouse* new_dw)
{
}

void 
PTWPlastic::getInternalVars(ParticleSubset* pset,
                            DataWarehouse* old_dw) 
{
}

void 
PTWPlastic::allocateAndPutInternalVars(ParticleSubset* pset,
                                       DataWarehouse* new_dw) 
{
}

void
PTWPlastic::allocateAndPutRigid(ParticleSubset* pset,
                                DataWarehouse* new_dw)
{
}

void
PTWPlastic::updateElastic(const particleIndex idx)
{
}

void
PTWPlastic::updatePlastic(const particleIndex idx, const double& )
{
}

