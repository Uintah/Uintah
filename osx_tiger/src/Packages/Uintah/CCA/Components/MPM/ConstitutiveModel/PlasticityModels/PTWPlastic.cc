
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
  ASSERT(epdot > 0.0);
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
  
  // Compute invxidot - the time required for a transverse wave to cross at atom
  if (mu < 0.0 || rho < 0.0) {
    cerr << "**ERROR** PTWPlastic::computeFlowStress: mu = " << mu 
         << " rho = " << rho << endl;
  }
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*d_CM.M),(1.0/3.0))*sqrt(mu/rho);

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
    double exp_EEp = 1.0/exp(E*ep);
    tauhat = tauhat_s + A*log(1.0 - F*exp_EEp);
  }
  double sigma = 2.0*tauhat*mu;
  return sigma;
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
  ASSERT(epdot > 0.0);
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
  double xidot = 0.5*pow(4.0*M_PI*rho/(3.0*d_CM.M),(1.0/3.0))*sqrt(mu/rho);

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

  // Compute the dimensionless flow stress
  double A = (d_CM.s0 - tauhat_y)/d_CM.p;
  double B = tauhat_s - tauhat_y;
  double D = exp(B/A);
  double C = D - 1.0;
  double F = C/D;
  double E = d_CM.theta/(A*C);
  double F_exp_Eep = F/exp(E*ep);
  double deriv = A*E*F_exp_Eep/(1.0 - F_exp_Eep);
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
PTWPlastic::evalDerivativeWRTTemperature(const PlasticityState* ,
					 const particleIndex )
{
  ostringstream desc;
  desc << "**PTW Deriv WRT Temp not implemented." << endl;
  throw InvalidValue(desc.str());
}

double
PTWPlastic::evalDerivativeWRTStrainRate(const PlasticityState* ,
                                        const particleIndex )
{
  ostringstream desc;
  desc << "**PTW Deriv WRT Strain Rate not implemented." << endl;
  throw InvalidValue(desc.str());
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

