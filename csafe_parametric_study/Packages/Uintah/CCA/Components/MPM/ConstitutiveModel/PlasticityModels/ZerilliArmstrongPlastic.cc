#include "ZerilliArmstrongPlastic.h"
#include <math.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>


using namespace Uintah;
using namespace SCIRun;
////////////////////////////////////////////////////////////////////////////////

ZerilliArmstrongPlastic::ZerilliArmstrongPlastic(ProblemSpecP& ps)
{
  cout << "I am in constructor " << endl;
  ps->require("bcc_or_fcc", d_CM.bcc_or_fcc);
  if (d_CM.bcc_or_fcc == "fcc")
  {
    d_CM.c1 = 0.0;
    d_CM.c5 = 0.0;
    d_CM.n = 0.0;
    ps->require("c2",d_CM.c2);
    ps->require("c3",d_CM.c3);
    ps->require("c4",d_CM.c4);
  }
  else
  {
    if (d_CM.bcc_or_fcc == "bcc")
    {
      d_CM.c2 = 0.0;
      ps->require("c1",d_CM.c1);
      ps->require("c3",d_CM.c3);
      ps->require("c4",d_CM.c4);
      ps->require("c5",d_CM.c5);
      ps->require("n",d_CM.n);
    }  
    else
    {
      throw ProblemSetupException("Material must be of the type BCC or FCC");
    }
  }
}
         
ZerilliArmstrongPlastic::ZerilliArmstrongPlastic(const 
						 ZerilliArmstrongPlastic* cm)
{
  d_CM.bcc_or_fcc = cm->d_CM.bcc_or_fcc;
  d_CM.c1 = cm->d_CM.c1;
  d_CM.c2 = cm->d_CM.c2;
  d_CM.c3 = cm->d_CM.c3;
  d_CM.c4 = cm->d_CM.c4;
  d_CM.c5 = cm->d_CM.c5;
  d_CM.n = cm->d_CM.n;
  cout << "BCC or FCC" << d_CM.bcc_or_fcc << endl;
  cout << d_CM.c1 << endl;
  cout << d_CM.c2 << endl;
  cout << d_CM.c3 << endl;
  cout << d_CM.c4 << endl;
  cout << "Constructor is called" << endl;
}
         
ZerilliArmstrongPlastic::~ZerilliArmstrongPlastic()
{
}
         
void 
ZerilliArmstrongPlastic::addInitialComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet*) const
{
}

void 
ZerilliArmstrongPlastic::addComputesAndRequires(Task* ,
                                    const MPMMaterial* ,
                                    const PatchSet*) const
{
}

void 
ZerilliArmstrongPlastic::addParticleState(std::vector<const VarLabel*>& ,
                                          std::vector<const VarLabel*>& )
{
}

void 
ZerilliArmstrongPlastic::allocateCMDataAddRequires(Task* ,
                                              const MPMMaterial* ,
                                              const PatchSet* ,
                                              MPMLabel* ) const
{
}

void ZerilliArmstrongPlastic::allocateCMDataAdd(DataWarehouse* ,
                                           ParticleSubset* ,
                                           map<const VarLabel*, 
                                           ParticleVariableBase*>* ,
                                           ParticleSubset* ,
                                           DataWarehouse* )
{
}


void 
ZerilliArmstrongPlastic::initializeInternalVars(ParticleSubset* ,
                                                DataWarehouse* )
{
}

void 
ZerilliArmstrongPlastic::getInternalVars(ParticleSubset* ,
                                         DataWarehouse* ) 
{
}

void 
ZerilliArmstrongPlastic::allocateAndPutInternalVars(ParticleSubset* ,
                                                    DataWarehouse* ) 
{
}

void 
ZerilliArmstrongPlastic::allocateAndPutRigid(ParticleSubset* ,
                                             DataWarehouse* ) 
{
}

void
ZerilliArmstrongPlastic::updateElastic(const particleIndex )
{
}

void
ZerilliArmstrongPlastic::updatePlastic(const particleIndex , 
				       const double& )
{
}

double 
ZerilliArmstrongPlastic::computeFlowStress(const PlasticityState* state,
                                      const double& delT,
                                      const double& tolerance,
                                      const MPMMaterial* matl,
                                      const particleIndex idx)
{
  double epdot = state->plasticStrainRate;
  double ep = state->plasticStrain;
  double T = state->temperature;
  double flowStress;
  cout << "Temp " << T << endl;
  cout << "ep " << ep << endl;
  cout << "epdot  " << epdot << endl;
  ASSERT(ep >= 0.0);
  //check for square root
  ASSERT(epdot > 0.0);
  //check for log
  if (d_CM.bcc_or_fcc == "fcc")
  {
    flowStress = d_CM.c2*sqrt(ep) + exp(-d_CM.c3*T + 
                 d_CM.c4*T*log(epdot)) + 65.0e6;
  }
  else
  {
    flowStress = d_CM.c1*exp(-d_CM.c3*T + d_CM.c4*T*log(epdot)) + 
                 d_CM.c5*pow(ep,d_CM.n) + 65.0e6; 
  }
  cout << "Flowstress is" << flowStress <<  endl;					     
  return flowStress;
  
}


void 
ZerilliArmstrongPlastic::computeTangentModulus(const Matrix3& stress,
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
ZerilliArmstrongPlastic::evalDerivativeWRTScalarVars(
						const PlasticityState* state,
						const particleIndex idx,
						Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}


double
ZerilliArmstrongPlastic::evalDerivativeWRTPlasticStrain(
						const PlasticityState* state,
						const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double deriv;  
  if (d_CM.bcc_or_fcc == "fcc")
  {
    deriv = (0.5*d_CM.c2*exp(-d_CM.c3*T + 
				  d_CM.c4*T*log(epdot)))/sqrt(ep);  
  }
  else
  {
    deriv = d_CM.n*d_CM.c5*pow(ep,((d_CM.n)-1));
  }
  
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
ZerilliArmstrongPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
ZerilliArmstrongPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
ZerilliArmstrongPlastic::evalDerivativeWRTTemperature(
						const PlasticityState* state,
                                                const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double deriv;
  
  if (d_CM.bcc_or_fcc == "fcc")
  {
    deriv = d_CM.c2*sqrt(ep)*exp(-d_CM.c3*T + 
		 d_CM.c4*T*log(epdot))*(-d_CM.c3 + d_CM.c4*log(epdot));    
  }
  else
  {
    deriv = d_CM.c1*exp(-d_CM.c3*T + d_CM.c4*T*log(epdot))*(-d_CM.c3 
                   + d_CM.c4*log(epdot));
  }
  
  return deriv;
}

double
ZerilliArmstrongPlastic::evalDerivativeWRTStrainRate(
						const PlasticityState* state,
                                                const particleIndex idx)
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  double deriv;

  if (d_CM.bcc_or_fcc == "fcc")
  {
    deriv = (d_CM.c2*d_CM.c4*T*sqrt(ep)*exp(-d_CM.c3*T + 
		   d_CM.c4*T*log(epdot)))/epdot;  
  }
  else
  {
    deriv = (d_CM.c1*d_CM.c4*T*exp(-d_CM.c3*T + 
					  d_CM.c4*T*log(epdot)))/epdot;    

  }
  
  return deriv;

}


