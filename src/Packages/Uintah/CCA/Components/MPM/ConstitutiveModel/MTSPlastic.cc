
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MTSPlastic.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


MTSPlastic::MTSPlastic(ProblemSpecP& ps)
{
  ps->require("sigma_a",d_CM.sigma_a);
  ps->require("mu_0",d_CM.mu_0); //b1
  ps->require("D",d_CM.D); //b2
  ps->require("T_0",d_CM.T_0); //b3
  ps->require("koverbcubed",d_CM.koverbcubed);
  ps->require("g_0i",d_CM.g_0i);
  ps->require("g_0e",d_CM.g_0e); // g0
  ps->require("edot_0i",d_CM.edot_0i);
  ps->require("edot_0e",d_CM.edot_0e); //edot
  ps->require("p_i",d_CM.p_i);
  ps->require("q_i",d_CM.q_i);
  ps->require("p_e",d_CM.p_e); //p
  ps->require("q_e",d_CM.q_e); //q
  ps->require("sigma_i",d_CM.sigma_i);
  ps->require("a_0",d_CM.a_0);
  ps->require("a_1",d_CM.a_1);
  ps->require("a_2",d_CM.a_2);
  ps->require("a_3",d_CM.a_3);
  ps->require("theta_IV",d_CM.theta_IV);
  ps->require("alpha",d_CM.alpha);
  ps->require("edot_es0",d_CM.edot_es0);
  ps->require("g_0es",d_CM.g_0es); //A
  ps->require("sigma_es0",d_CM.sigma_es0);

  // Initialize internal variable labels for evolution
  pMTSLabel = VarLabel::create("p.mtStress",
	ParticleVariable<double>::getTypeDescription());
  pMTSLabel_preReloc = VarLabel::create("p.mtStress+",
	ParticleVariable<double>::getTypeDescription());
}
	 
MTSPlastic::~MTSPlastic()
{
  VarLabel::destroy(pMTSLabel);
  VarLabel::destroy(pMTSLabel_preReloc);
}
	 
void 
MTSPlastic::addInitialComputesAndRequires(Task* task,
					  const MPMMaterial* matl,
					  const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pMTSLabel, matlset);
}

void 
MTSPlastic::addComputesAndRequires(Task* task,
				   const MPMMaterial* matl,
				   const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pMTSLabel, matlset,Ghost::None);
  task->computes(pMTSLabel_preReloc, matlset);
}

void 
MTSPlastic::addParticleState(std::vector<const VarLabel*>& from,
			     std::vector<const VarLabel*>& to)
{
  from.push_back(pMTSLabel);
  to.push_back(pMTSLabel_preReloc);
}

void 
MTSPlastic::allocateCMDataAddRequires(Task* task,
				      const MPMMaterial* matl,
				      const PatchSet* patch,
				      MPMLabel* lb) const
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,pMTSLabel, Ghost::None);
}

void 
MTSPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
			      ParticleSubset* addset,
			      map<const VarLabel*, 
                                ParticleVariableBase*>* newState,
			      ParticleSubset* delset,
			      DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
 
  ParticleVariable<double> pMTS;
  constParticleVariable<double> o_MTS;

  new_dw->allocateTemporary(pMTS,addset);

  old_dw->get(o_MTS,pMTSLabel,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o = delset->begin(); o != delset->end(); o++, n++) {
    pMTS[*n] = o_MTS[*o];
  }

  (*newState)[pMTSLabel]=pMTS.clone();

}



void 
MTSPlastic::initializeInternalVars(ParticleSubset* pset,
				   DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pMTS_new, pMTSLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
    pMTS_new[*iter] = 0.0;
  }
}

void 
MTSPlastic::getInternalVars(ParticleSubset* pset,
			    DataWarehouse* old_dw) 
{
  old_dw->get(pMTS, pMTSLabel, pset);
}

void 
MTSPlastic::allocateAndPutInternalVars(ParticleSubset* pset,
				       DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pMTS_new, pMTSLabel_preReloc, pset);
}

void
MTSPlastic::allocateAndPutRigid(ParticleSubset* pset,
                                DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pMTS_new, pMTSLabel_preReloc, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
     pMTS_new[*iter] = 0.0;
  }
}

void
MTSPlastic::updateElastic(const particleIndex idx)
{
  pMTS_new[idx] = pMTS[idx];
}

void
MTSPlastic::updatePlastic(const particleIndex idx, const double& )
{
  pMTS_new[idx] = pMTS_new[idx];
}

double 
MTSPlastic::computeFlowStress(const double& plasticStrainRate,
                              const double& ,
                              const double& T,
                              const double& delT,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex idx)
{
  // Calculate strain rate and incremental strain
  double edot = plasticStrainRate;
  double delEps = edot*delT;

  // Check if temperature is correct
  if (T <= 0.0) {
    ostringstream desc;
    desc << "**MTS ERROR** Absolute temperature <= 0." << endl;
    desc << "T = " << T << " edot = " << edot << endl;
    throw InvalidValue(desc.str());
  }

  // Calculate mu and mu/mu_0
  double expT0_T = exp(d_CM.T_0/T) - 1.0;
  ASSERT(expT0_T != 0);
  double mu = d_CM.mu_0 - d_CM.D/expT0_T;
  if (mu <= 0.0) {
    ostringstream desc;
    desc << "**MTS ERROR** Shear modulus <= 0." << endl;
    desc << "T = " << T << " mu0 = " << d_CM.mu_0 << " T0 = " << d_CM.T_0
         << " exp(To/T) = " << exp(d_CM.T_0/T) << " D = " << d_CM.D << endl;
    throw InvalidValue(desc.str());
  }
  //double mu_mu_0 = mu/d_CM.mu_0;
  //cout << "mu = " << mu << " mu/mu_0 = " << mu_mu_0 << endl;

  // Calculate S_i
  double CC = d_CM.koverbcubed*T/mu;
  //cout << "CC = " << CC << endl;
  double S_i = 0.0;
  if (d_CM.p_i > 0.0) {
    double CCi = CC/d_CM.g_0i;
    double logei = log(d_CM.edot_0i/edot);
    double logei_q = 1.0 - pow((CCi*logei),(1.0/d_CM.q_i));
    if (!(logei_q > 0.0)) {
      ostringstream desc;
      desc << "**MTS ERROR** 1 - [Gi log(edoti/edot)]^q <= 0." << endl;
      desc << "T = " << T << " mu = " << mu << " CCi = " << CCi
           << " edot = " << edot << " logei = " << logei 
           << " logei_q = " << endl;
      throw InvalidValue(desc.str());
    }
    S_i = pow(logei_q,(1.0/d_CM.p_i));
    //S_i = pow((1.0-pow((CCi*logei),(1.0/d_CM.q_i))),(1.0/d_CM.p_i));
    //cout << "CC_i = " << CCi << " loge_i = " << logei << endl;
  }
  //cout << "S_i = " << S_i << endl;

  // Calculate S_e
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/edot);
  double logee_q = 1.0 - pow((CCe*logee),(1.0/d_CM.q_e));
  if (!(logee_q > 0.0)) {
    ostringstream desc;
    desc << "**MTS ERROR** 1 - [Ge log(edote/edot)]^q <= 0." << endl;
    desc << "T = " << T << " mu = " << mu << " CCe = " << CCe
	 << " edot = " << edot << " logee = " << logee 
	 << " logee_q = " << endl;
    throw InvalidValue(desc.str());
  }
  double S_e = pow(logee_q,(1.0/d_CM.p_e));
  //double S_e = pow((1.0-pow((CCe*logee),(1.0/d_CM.q_e))),(1.0/d_CM.p_e));
  //cout << "CC_e = " << CCe << " loge_e = " << logee << endl;
  //cout << "S_e = " << S_e << endl;

  // Calculate theta_0
  double theta_0 = d_CM.a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot)
    - d_CM.a_3*T;
  //cout << "theta_0 = " << theta_0 << endl;

  // Calculate sigma_es
  double CCes = CC/d_CM.g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = d_CM.sigma_es0*powees;
  //cout << "CC_es = " << CCes << " (e/es)^CC = " << powees << endl;
  //cout << "sigma_es = " << sigma_es << endl;

  // Calculate X and FX
  double X = pMTS[idx]/sigma_es;
  double FX = tanh(d_CM.alpha*X);
  //cout << "X = " << X << " FX = " << FX << endl;

  // Calculate theta
  double theta = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;
  //cout << "theta = " << theta << endl;

  // Calculate the flow stress
  double sigma_e = pMTS[idx] + delEps*theta; 
  pMTS_new[idx] = sigma_e;
  //cout << "sigma_e = " << sigma_e << endl;
  double sigma = d_CM.sigma_a + S_i*d_CM.sigma_i + S_e*sigma_e;
  //cout << "MTS::edot = " << edot << " delEps = " << delEps 
  //     << " sigma_Y = " << sigma << endl;
  return sigma;
}

/*! The evolving internal variable is \f$q = \hat\sigma_e\f$.  If the 
  evolution equation for internal variables is of the form 
  \f$ \dot q = \gamma h (\sigma, q) \f$, then 
  \f[
  \dot q = \frac{d\hat\sigma_e}{dt} 
  = \frac{d\hat\sigma_e}{d\epsilon} \frac{d\epsilon}{dt}
  = \theta \dot\epsilon .
  \f] 
  If \f$\dot\epsilon = \gamma\f$, then \f$ \theta = h \f$.
  Also, \f$ f_q = \frac{\partial f}{\partial \hat\sigma_e} \f$.
  For the von Mises yield condition, \f$(f)\f$, 
  \f$ f_q = \frac{\partial \sigma}{\partial \hat\sigma_e} \f$
  where \f$\sigma\f$ is the MTS flow stress.
*/
void 
MTSPlastic::computeTangentModulus(const Matrix3& sig,
				  const double& plasticStrainRate, 
				  const double& , 
				  double T,
				  double ,
				  const particleIndex idx,
				  const MPMMaterial* ,
				  TangentModulusTensor& Ce,
				  TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = sig - one*(sig.Trace()/3.0);

  // Calculate the equivalent stress and strain rate
  double sigeqv = sqrt(sigdev.NormSquared()); 
  double edot = plasticStrainRate;

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Calculate mu and mu/mu_0
  double mu_mu_0 = 1.0 - d_CM.D/(d_CM.mu_0*(exp(d_CM.T_0/T) - 1.0)); 
  double mu = mu_mu_0*d_CM.mu_0;

  // Calculate theta_0
  double theta_0 = d_CM.a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot)
    - d_CM.a_3*T;

  // Calculate sigma_es
  double CC = d_CM.koverbcubed*T/mu;
  double CCes = CC/d_CM.g_0es;
  double logees = log(edot/d_CM.edot_es0);
  double sigma_es = d_CM.sigma_es0*exp(CCes*logees);

  // Calculate X and FX
  double X = pMTS_new[idx]/sigma_es;
  double FX = tanh(d_CM.alpha*X);

  // Calculate theta
  double theta = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;
  double h = theta;

  // Calculate f_q (h = theta, therefore f_q.h = f_q.theta)
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/edot);
  double S_e = pow((1.0-pow((CCe*logee),(1.0/d_CM.q_e))),(1.0/d_CM.p_e));
  double f_q = mu_mu_0*S_e;

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
      rCr += rC(ii1,jj1)*rr(ii1,jj1);
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
	    Cr(ii1,jj1)*rC(kk1,ll+1)/(-f_q*h + rCr);
	}  
      }  
    }  
  }  
}


double
MTSPlastic::evalDerivativeWRTTemperature(double edot,
                                         double ,
                                         double T,
					 const particleIndex idx)
{
  double sigma_e = pMTS_new[idx];

  // Calculate exp(T0/T)
  double expT0T = exp(d_CM.T_0/T);

  // Calculate mu and mu/mu_0
  double mu = d_CM.mu_0 - d_CM.D/(expT0T - 1.0); 
  if (!(mu > 0.0)) {
    ostringstream desc;
    desc << "**MTS Deriv Edot ERROR** Shear modulus <= 0." << endl;
    desc << "T = " << T << " mu0 = " << d_CM.mu_0 << " T0 = " << d_CM.T_0
         << " exp(To/T) = " << expT0T << " D = " << d_CM.D << endl;
    throw InvalidValue(desc.str());
  }
  double mu_mu_0 = mu/d_CM.mu_0;
  double CC = d_CM.koverbcubed/mu;

  // Materials with (sigma_i != 0) are treated specially
  double numer1i = 0.0;
  double ratio2i = 0.0;
  if (d_CM.p_i > 0.0) {

    // Calculate log(edot0/edot)
    double logei = log(d_CM.edot_0i/edot);

    // Calculate E = k/(g0*mu*b^3) log(edot_0/edot)
    double E_i = (CC/d_CM.g_0i)*logei;

    // Calculate (E*T)^{1/q}
    double ET_i = pow(E_i*T, 1.0/d_CM.q_i);

    // Calculate (1-(E*T)^{1/q})^{1/p)
    double ETpq_i = pow(1.0-ET_i,1.0/d_CM.p_i);

    // Calculate Ist term numer
    numer1i = ETpq_i*d_CM.sigma_i;

    // Calculate second term denom
    double denom2i = T*(1.0-ET_i);

    // Calculate second term numerator
    double numer2i = ET_i*ETpq_i*d_CM.sigma_i/(d_CM.p_i*d_CM.q_i);

    // Calculate the ratios
    ratio2i = numer2i/denom2i;
  }

  // Calculate log(edot0/edot)
  double logee = log(d_CM.edot_0e/edot);

  // Calculate E = k/(g0*mu*b^3) log(edot_0/edot)
  double E_e = (CC/d_CM.g_0e)*logee;

  // Calculate (E*T)^{1/q}
  double ET_e = pow(E_e*T, 1.0/d_CM.q_e);

  // Calculate (1-(E*T)^{1/q})^{1/p)
  double ETpq_e = pow(1.0-ET_e,1.0/d_CM.p_e);

  // Calculate Ist term denom
  double denom1 = (expT0T - 1.0)*(expT0T - 1.0)*T*T*d_CM.mu_0;

  // Calculate Ist term numer
  double numer1e = ETpq_e*sigma_e;

  // Calculate the total (first term)
  double numer1 = - d_CM.D*d_CM.T_0*expT0T*(numer1i + numer1e);

  // Calculate the first term
  double first = numer1/denom1;

  // Calculate second term denom
  double denom2e = T*(1.0-ET_e);

  // Calculate second term numerator
  double numer2e = ET_e*ETpq_e*sigma_e/(d_CM.p_e*d_CM.q_e);

  // Calculate the ratios
  double ratio2e = numer2e/denom2e;

  // Calculate the total (second term)
  double second = - mu_mu_0*(ratio2i + ratio2e);

  return (first+second);
}

double
MTSPlastic::evalDerivativeWRTStrainRate(double edot,
                                        double ,
                                        double T,
                                        const particleIndex idx)
{
  double sigma_e = pMTS_new[idx];

  // Calculate mu and mu/mu_0
  double mu_mu_0 = 1.0 - d_CM.D/(d_CM.mu_0*(exp(d_CM.T_0/T) - 1.0)); 
  if (!(mu_mu_0 > 0.0)) {
    ostringstream desc;
    desc << "**MTS Deriv Edot ERROR** Shear modulus <= 0." << endl;
    throw InvalidValue(desc.str());
  }
  double mu = mu_mu_0*d_CM.mu_0;
  double CC = d_CM.koverbcubed*T/mu;

  // Materials with (sigma_i != 0) are treated specially
  double ratioi = 0.0;
  if (d_CM.p_i > 0.0) {

    // Calculate (mu/mu0)*sigma_hat
    double M_i = mu_mu_0*d_CM.sigma_i;

    // Calculate CC/g0
    double K_i = CC/d_CM.g_0i;

    // Calculate log(edot0/edot)
    double logei = log(d_CM.edot_0i/edot);

    // Calculate (K*log(edot0/edot))^{1/q}
    double Klogi = pow(K_i*logei, 1.0/d_CM.q_i);
 
    // Calculate the denominator
    double denomi = edot*logei*(1.0-Klogi);

    // Calculate the numerator
    double numeri = (M_i*Klogi)/
      (d_CM.p_i*d_CM.q_i)*pow(1.0-Klogi,1.0/d_CM.p_i);

    // Calculate the ratios
    ratioi = numeri/denomi;
  }
  // Calculate (mu/mu0)*sigma_hat
  double M_e = mu_mu_0*sigma_e;

  // Calculate CC/g0
  double K_e = CC/d_CM.g_0e;

  // Calculate log(edot0/edot)
  double logee = log(d_CM.edot_0e/edot);

  // Calculate (K*log(edot0/edot))^{1/q}
  double Kloge = pow(K_e*logee, 1.0/d_CM.q_e);
 
  // Calculate the denominator
  double denome = edot*logee*(1.0-Kloge);

  // Calculate the numerator
  double numere = (M_e*Kloge)/
    (d_CM.p_e*d_CM.q_e)*pow(1.0-Kloge, 1.0/d_CM.p_e);

  // Calculate the ratios
  double ratioe = numere/denome;

  return (ratioi+ratioe);
}

double
MTSPlastic::evalDerivativeWRTPlasticStrain(double edot,
                                           double ep,
                                           double T,
					   const particleIndex idx)
{
  //double sigma_e = pMTS_new[idx];
  double dsigY_dsig_e = evalDerivativeWRTSigmaE(edot, ep, T, idx);

  // Calculate theta_0
  double theta_0 = d_CM.a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot)
    - d_CM.a_3*T;

  // Calculate mu and mu/mu_0
  double mu = d_CM.mu_0 - d_CM.D/(exp(d_CM.T_0/T) - 1.0); 
  if (!(mu > 0.0)) {
    ostringstream desc;
    desc << "**MTS Deriv Edot ERROR** Shear modulus <= 0." << endl;
    throw InvalidValue(desc.str());
  }

  // Calculate sigma_es
  double CC = d_CM.koverbcubed*T/mu;
  double CCes = CC/d_CM.g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = d_CM.sigma_es0*powees;

  // Calculate X and FX
  double X = pMTS_new[idx]/sigma_es;
  double FX = tanh(d_CM.alpha*X);

  // Calculate theta
  double dsig_e_dep = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;
  
  return (dsigY_dsig_e*dsig_e_dep);
}

double
MTSPlastic::evalDerivativeWRTSigmaE(double edot,
                                    double ,
                                    double T,
				    const particleIndex )
{
  //double sigma_e = pMTS_new[idx];

  // Calculate mu and mu/mu_0
  double mu_mu_0 = 1.0 - d_CM.D/(d_CM.mu_0*(exp(d_CM.T_0/T) - 1.0)); 
  if (!(mu_mu_0 > 0.0)) {
    ostringstream desc;
    desc << "**MTS Deriv T ERROR** Shear modulus <= 0." << endl;
    throw InvalidValue(desc.str());
  }
  double mu = mu_mu_0*d_CM.mu_0;

  // Calculate S_e
  double CC = d_CM.koverbcubed*T/mu;
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/edot);
  double S_e = pow((1.0-pow((CCe*logee),(1.0/d_CM.q_e))),(1.0/d_CM.p_e));
  S_e = mu_mu_0*S_e;
  return S_e;
}


void
MTSPlastic::evalDerivativeWRTScalarVars(double edot,
                                        double ep,
                                        double T,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(edot, ep, T, idx);
  derivs[1] = evalDerivativeWRTTemperature(edot, ep, T, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(edot, ep, T, idx);
}
