#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookPlastic.h>
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
  d_CM.TRoom = 294;
  d_CM.TMelt = 1594;

  // Initialize internal variable labels for evolution
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
			ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
			ParticleVariable<double>::getTypeDescription());
}
	 
JohnsonCookPlastic::~JohnsonCookPlastic()
{
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
}
	 
void 
JohnsonCookPlastic::addInitialComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pPlasticStrainLabel, matlset);
}

void 
JohnsonCookPlastic::addComputesAndRequires(Task* task,
				    const MPMMaterial* matl,
				    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset,Ghost::None);
  task->computes(pPlasticStrainLabel_preReloc, matlset);
}

void 
JohnsonCookPlastic::addParticleState(std::vector<const VarLabel*>& from,
				     std::vector<const VarLabel*>& to)
{
  from.push_back(pPlasticStrainLabel);
  to.push_back(pPlasticStrainLabel_preReloc);
}

void 
JohnsonCookPlastic::initializeInternalVars(ParticleSubset* pset,
				           DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pPlasticStrain_new, pPlasticStrainLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) pPlasticStrain_new[*iter] = 0.0;
}

void 
JohnsonCookPlastic::getInternalVars(ParticleSubset* pset,
                                    DataWarehouse* old_dw) 
{
  old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
}

void 
JohnsonCookPlastic::allocateAndPutInternalVars(ParticleSubset* pset,
                                               DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pPlasticStrain_new, pPlasticStrainLabel_preReloc,pset);
}

void
JohnsonCookPlastic::updateElastic(const particleIndex idx)
{
  pPlasticStrain_new[idx] = pPlasticStrain[idx];
}

void
JohnsonCookPlastic::updatePlastic(const particleIndex idx, const double& )
{
  pPlasticStrain_new[idx] = pPlasticStrain_new[idx];
}

double
JohnsonCookPlastic::getUpdatedPlasticStrain(const particleIndex idx)
{
  return pPlasticStrain_new[idx];
}

double 
JohnsonCookPlastic::computeFlowStress(const Matrix3& rateOfDeformation,
                                      const double& temperature,
                                      const double& delT,
                                      const double& tolerance,
                                      const MPMMaterial* matl,
                                      const particleIndex idx)
{
  double plasticStrain = pPlasticStrain[idx];
  double plasticStrainRate = sqrt(rateOfDeformation.NormSquared()*2.0/3.0);
  plasticStrain += plasticStrainRate*delT;
  pPlasticStrain_new[idx] = plasticStrain;

  return evaluateFlowStress(plasticStrain, plasticStrainRate, 
                            temperature, matl, tolerance);
}

double 
JohnsonCookPlastic::evaluateFlowStress(const double& ep, 
				       const double& epdot,
				       const double& T,
                                       const MPMMaterial* matl,
                                       const double& )
{
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_CM.C);
  else
    strainRatePart = 1.0 + d_CM.C*log(epdot);
  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  ASSERT(T < Tm);
  d_CM.TRoom = Tr;  d_CM.TMelt = Tm;
  double m = d_CM.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));
  return (strainPart*strainRatePart*tempPart);
}

/*! In this case, \f$\dot{\epsilon_p}\f$ is the time derivative of 
    \f$\epsilon_p\f$.  Hence, the evolution law of the internal variables
    \f$ \dot{q_\alpha} = \gamma h_\alpha \f$ requires 
    \f$\gamma = \dot{\epsilon_p}\f$ and \f$ h_\alpha = 1\f$. */
void 
JohnsonCookPlastic::computeTangentModulus(const Matrix3& sig,
                                          const Matrix3& D, 
                                          double T,
                                          double ,
                                          const particleIndex idx,
                                          const MPMMaterial* matl,
                                          TangentModulusTensor& Ce,
				          TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = sig - one*(sig.Trace()/3.0);
  Matrix3 Ddev = D - one*(D.Trace()/3.0);

  // Calculate the equivalent stress
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the dircetion of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Calculate f_q (h = 1, therefore f_q.h = f_q)
  double ep = pPlasticStrain_new[idx];
  double epdot = sqrt(Ddev.NormSquared()/1.5);
  double strainPart = d_CM.n*d_CM.B*
                      pow(ep,d_CM.n-1);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_CM.C);
  else
    strainRatePart = 1.0 + d_CM.C*log(epdot);
  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  double m = d_CM.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));
  double f_q = strainPart*strainRatePart*tempPart;

  // Form the elastic-plastic tangent modulus
  Matrix3 Cr, rC;
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      Cr(ii1,jj1) = 0.0;
      rC(ii1,jj1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cr(ii1,jj1) += Ce(ii,jj,kk,ll)*rr(kk1,ll+1);
          rC(ii1,jj1) += rr(kk1,ll+1)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii,jj)*rr(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
                             Cr(ii1,jj1)*rC(kk1,ll+1)/(-f_q + rCr);
	}  
      }  
    }  
  }  
}

double
JohnsonCookPlastic::evalDerivativeWRTTemperature(double epdot,
						 double T,
                                                 const particleIndex idx)
{
  double ep = pPlasticStrain_new[idx];

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(d_CM.TMelt-d_CM.TRoom);

  double F = strainPart*strainRatePart;
  double deriv = - m*F*pow(Tstar,m)/(T-d_CM.TRoom);
  return deriv;
}

double
JohnsonCookPlastic::evalDerivativeWRTStrainRate(double epdot,
						double T,
                                                const particleIndex idx)
{
  double ep = pPlasticStrain_new[idx];

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(d_CM.TMelt-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double E = strainPart*tempPart;

  double deriv = E*d_CM.C/epdot;
  return deriv;

}

double
JohnsonCookPlastic::evalDerivativeWRTPlasticStrain(double epdot,
						   double T,
                                                   const particleIndex idx)
{
  double ep = pPlasticStrain_new[idx];

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T-d_CM.TRoom)/(d_CM.TMelt-d_CM.TRoom);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));

  double D = strainRatePart*tempPart;

  double deriv = d_CM.B*d_CM.n*D*pow(ep,d_CM.n)/ep;
  return deriv;
}


void
JohnsonCookPlastic::evalDerivativeWRTScalarVars(double epdot,
						double T,
						const particleIndex idx,
						Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(epdot, T, idx);
  derivs[1] = evalDerivativeWRTTemperature(epdot, T, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(epdot, T, idx);
}
