#include "JohnsonCookPlastic.h"
#include <math.h>

using namespace Uintah;
using namespace SCIRun;


JohnsonCookPlastic::JohnsonCookPlastic(ProblemSpecP& ps)
{
  ps->require("A",d_CM.A);
  ps->require("B",d_CM.B);
  ps->require("C",d_CM.C);
  ps->require("n",d_CM.n);
  ps->require("m",d_CM.m);
  d_CM.epdot_0 = 1.0;
  ps->get("epdot_0", d_CM.epdot_0);
  d_CM.TRoom = 294;
  ps->get("T_r",d_CM.TRoom);
  d_CM.TMelt = 1594;
  ps->get("T_m",d_CM.TMelt);
}
         
JohnsonCookPlastic::JohnsonCookPlastic(const JohnsonCookPlastic* cm)
{
  d_CM.A = cm->d_CM.A;
  d_CM.B = cm->d_CM.B;
  d_CM.C = cm->d_CM.C;
  d_CM.n = cm->d_CM.n;
  d_CM.m = cm->d_CM.m;
  d_CM.epdot_0 = cm->d_CM.epdot_0;
  d_CM.TRoom = cm->d_CM.TRoom;
  d_CM.TMelt = cm->d_CM.TMelt;
}
         
JohnsonCookPlastic::~JohnsonCookPlastic()
{
}
         
void 
JohnsonCookPlastic::addInitialComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet*) const
{
}

void 
JohnsonCookPlastic::addComputesAndRequires(Task* ,
                                    const MPMMaterial* ,
                                    const PatchSet*) const
{
}

void 
JohnsonCookPlastic::addParticleState(std::vector<const VarLabel*>& ,
                                     std::vector<const VarLabel*>& )
{
}

void 
JohnsonCookPlastic::allocateCMDataAddRequires(Task* ,
                                              const MPMMaterial* ,
                                              const PatchSet* ,
                                              MPMLabel* ) const
{
}

void JohnsonCookPlastic::allocateCMDataAdd(DataWarehouse* ,
                                           ParticleSubset* ,
                                           map<const VarLabel*, 
                                           ParticleVariableBase*>* ,
                                           ParticleSubset* ,
                                           DataWarehouse* )
{
}


void 
JohnsonCookPlastic::initializeInternalVars(ParticleSubset* ,
                                           DataWarehouse* )
{
}

void 
JohnsonCookPlastic::getInternalVars(ParticleSubset* ,
                                    DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutInternalVars(ParticleSubset* ,
                                               DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutRigid(ParticleSubset* ,
                                        DataWarehouse* ) 
{
}

void
JohnsonCookPlastic::updateElastic(const particleIndex )
{
}

void
JohnsonCookPlastic::updatePlastic(const particleIndex , const double& )
{
}

double 
JohnsonCookPlastic::computeFlowStress(const PlasticityState* state,
                                      const double& delT,
                                      const double& tolerance,
                                      const MPMMaterial* matl,
                                      const particleIndex idx)
{
  double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tr = matl->getRoomTemperature();
  double Tm = state->meltingTemp;

  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_CM.C);
  else
    strainRatePart = 1.0 + d_CM.C*log(epdot);
  d_CM.TRoom = Tr;  d_CM.TMelt = Tm;
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : ((T-Tr)/(Tm-Tr)); 
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));
  double sigy = strainPart*strainRatePart*tempPart;
  if (isnan(sigy)) {
    cout << "**ERROR** JohnsonCook: sig_y == nan " << endl; 
  }
  return sigy;
}

double 
JohnsonCookPlastic::computeEpdot(const PlasticityState* state,
                                 const double& delT,
                                 const double& tolerance,
                                 const MPMMaterial* matl,
                                 const particleIndex idx)
{
  // All quantities should be at the beginning of the 
  // time step
  double tau = state->yieldStress;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double Tr = matl->getRoomTemperature();
  double Tm = state->meltingTemp;

  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  d_CM.TRoom = Tr;  d_CM.TMelt = Tm;
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : ((T-Tr)/(Tm-Tr)); 
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double fac1 = tau/(strainPart*tempPart);
  double fac2 = (1.0/d_CM.C)*(fac1-1.0);
  double epdot = exp(fac2)*d_CM.epdot_0;
  if (isnan(epdot)) {
    cout << "**ERROR** JohnsonCook: epdot == nan " << endl; 
  }
  return epdot;
}
 
/*! In this case, \f$\dot{\epsilon_p}\f$ is the time derivative of 
    \f$\epsilon_p\f$.  Hence, the evolution law of the internal variables
    \f$ \dot{q_\alpha} = \gamma h_\alpha \f$ requires 
    \f$\gamma = \dot{\epsilon_p}\f$ and \f$ h_\alpha = 1\f$. */
void 
JohnsonCookPlastic::computeTangentModulus(const Matrix3& stress,
                                          const PlasticityState* state,
                                          const double& ,
                                          const MPMMaterial* matl,
                                          const particleIndex idx,
                                          TangentModulusTensor& Ce,
                                          TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Get f_q = dsigma/dep (h = 1, therefore f_q.h = f_q)
  double f_q = evalDerivativeWRTPlasticStrain(state, idx);

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
                             Cr(ii,jj)*rC(kk,ll)/(-f_q + rCr);
        }  
      }  
    }  
  }  
}

void
JohnsonCookPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                                const particleIndex idx,
                                                Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}


double
JohnsonCookPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                                   const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain rate part
  double strainRatePart = (epdot < 1.0) ? 
	  (pow((1.0 + epdot),d_CM.C)) : (1.0 + d_CM.C*log(epdot));

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-d_CM.TRoom)/(Tm-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double D = strainRatePart*tempPart;

  double deriv =  (ep > 0.0) ?  (d_CM.B*d_CM.n*D*pow(ep,d_CM.n-1)) : 0.0;
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/dep == nan " << endl; 
  }
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
JohnsonCookPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
                                                 const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-d_CM.TRoom)/(Tm-d_CM.TRoom);

  double F = strainPart*strainRatePart;
  double deriv = - m*F*pow(Tstar,m)/(T-d_CM.TRoom);
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/dT == nan " << endl; 
  }
  return deriv;
}

double
JohnsonCookPlastic::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                                const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double T = state->temperature;
  double Tm = state->meltingTemp;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-d_CM.TRoom)/(Tm-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double E = strainPart*tempPart;

  double deriv = 0.0;
  if (epdot < 1.0) 
    deriv = E*d_CM.C*pow((1.0 + epdot),(d_CM.C-1.0));
  else
    deriv = E*d_CM.C/epdot;
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/depdot == nan " << endl; 
  }
  return deriv;

}


