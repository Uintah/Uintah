
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MTSPlastic.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


MTSPlastic::MTSPlastic(ProblemSpecP& ps)
{
  ps->require("sigma_a",d_const.sigma_a);
  ps->require("mu_0",d_const.mu_0); //b1
  ps->require("D",d_const.D); //b2
  ps->require("T_0",d_const.T_0); //b3
  ps->require("koverbcubed",d_const.koverbcubed);
  ps->require("g_0i",d_const.g_0i);
  ps->require("g_0e",d_const.g_0e); // g0
  ps->require("edot_0i",d_const.edot_0i);
  ps->require("edot_0e",d_const.edot_0e); //edot
  ps->require("p_i",d_const.p_i);
  ps->require("q_i",d_const.q_i);
  ps->require("p_e",d_const.p_e); //p
  ps->require("q_e",d_const.q_e); //q
  ps->require("sigma_i",d_const.sigma_i);
  ps->require("a_0",d_const.a_0);
  ps->require("a_1",d_const.a_1);
  ps->require("a_2",d_const.a_2);
  ps->require("a_3",d_const.a_3);
  ps->require("theta_IV",d_const.theta_IV);
  ps->require("alpha",d_const.alpha);
  ps->require("edot_es0",d_const.edot_es0);
  ps->require("g_0es",d_const.g_0es); //A
  ps->require("sigma_es0",d_const.sigma_es0);

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
MTSPlastic::initializeInternalVars(ParticleSubset* pset,
				   DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pMTS_new, pMTSLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) pMTS_new[*iter] = 0.0;
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
MTSPlastic::computeFlowStress(const Matrix3& rateOfDeformation,
                              const Matrix3& ,
                              const double& T,
                              const double& delT,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex idx)
{
  //cout << "sigma_a " << d_const.sigma_a << endl;
  //cout << "mu_0 " << d_const.mu_0 << endl; //b1
  //cout << "D " << d_const.D << endl; //b2
  //cout << "T_0 " << d_const.T_0 << endl; //b3
  //cout << "koverbcubed " << d_const.koverbcubed << endl;
  //cout << "g_0i " << d_const.g_0i << endl;
  //cout << "g_0e " << d_const.g_0e << endl; // g0
  //cout << "edot_0i " << d_const.edot_0i << endl;
  //cout << "edot_0e " << d_const.edot_0e << endl; //edot
  //cout << "p_i " << d_const.p_i << endl;
  //cout << "q_i " << d_const.q_i << endl;
  //cout << "p_e " << d_const.p_e << endl; //p
  //cout << "q_e " << d_const.q_e << endl; //q
  //cout << "sigma_i " << d_const.sigma_i << endl;
  //cout << "a_0 " << d_const.a_0 << endl;
  //cout << "a_1 " << d_const.a_1 << endl;
  //cout << "a_2 " << d_const.a_2 << endl;
  //cout << "a_3 " << d_const.a_3 << endl;
  //cout << "theta_IV " << d_const.theta_IV << endl;
  //cout << "alpha " << d_const.alpha << endl;
  //cout << "edot_es0 " << d_const.edot_es0 << endl;
  //cout << "g_0es " << d_const.g_0es << endl; //A
  //cout << "sigma_es0 " << d_const.sigma_es0 << endl;

  // Calculate strain rate and incremental strain
  double edot = sqrt(rateOfDeformation.NormSquared()/1.5);
  if (edot < 0.00001) return 0.0;

  double delEps = edot*delT;
  //cout << "edot = " << edot << " delEps = " << delEps << endl;

  // Calculate mu and mu/mu_0
  double mu_mu_0 = 1.0 - d_const.D/(d_const.mu_0*(exp(d_const.T_0/T) - 1.0)); 
  double mu = mu_mu_0*d_const.mu_0;
  //cout << "mu = " << mu << " mu/mu_0 = " << mu_mu_0 << endl;

  // Calculate S_i
  double CC = d_const.koverbcubed*T/mu;
  //cout << "CC = " << CC << endl;
  double S_i = 0.0;
  if (d_const.p_i > 0.0) {
    double CCi = CC/d_const.g_0i;
    double logei = log(d_const.edot_0i/edot);
    S_i = pow((1.0-pow((CCi*logei),(1.0/d_const.q_i))),(1.0/d_const.p_i));
    //cout << "CC_i = " << CCi << " loge_i = " << logei << endl;
  }
  //cout << "S_i = " << S_i << endl;

  // Calculate S_e
  double CCe = CC/d_const.g_0e;
  double logee = log(d_const.edot_0e/edot);
  double S_e = pow((1.0-pow((CCe*logee),(1.0/d_const.q_e))),(1.0/d_const.p_e));
  //cout << "CC_e = " << CCe << " loge_e = " << logee << endl;
  //cout << "S_e = " << S_e << endl;

  // Calculate theta_0
  double theta_0 = d_const.a_0 + d_const.a_1*log(edot) + d_const.a_2*sqrt(edot)
    - d_const.a_3*T;
  //cout << "theta_0 = " << theta_0 << endl;

  // Calculate sigma_es
  double CCes = CC/d_const.g_0es;
  double logees = log(edot/d_const.edot_es0);
  double sigma_es = d_const.sigma_es0*exp(CCes*logees);
  //cout << "CC_es = " << CCes << " loge_es = " << logees << endl;
  //cout << "sigma_es = " << sigma_es << endl;

  // Calculate X and FX
  double X = pMTS[idx]/sigma_es;
  double FX = tanh(d_const.alpha*X);
  //cout << "X = " << X << " FX = " << FX << endl;

  // Calculate theta
  double theta = theta_0*(1.0 - FX) + d_const.theta_IV*FX;
  //cout << "theta = " << theta << endl;

  // Calculate the flow stress
  double sigma_e = pMTS[idx] + delEps*theta; 
  pMTS_new[idx] = sigma_e;
  //cout << "sigma_e = " << sigma_e << endl;
  double sigma = d_const.sigma_a + S_i*d_const.sigma_i + S_e*sigma_e;
  //cout << "sigma = " << sigma << endl;
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

  // Calculate the equivalent stress and strain rate
  double sigeqv = sqrt(sigdev.NormSquared()); 
  double edot = sqrt(D.NormSquared()/1.5);

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Calculate mu and mu/mu_0
  double mu_mu_0 = 1.0 - d_const.D/(d_const.mu_0*(exp(d_const.T_0/T) - 1.0)); 
  double mu = mu_mu_0*d_const.mu_0;

  // Calculate theta_0
  double theta_0 = d_const.a_0 + d_const.a_1*log(edot) + d_const.a_2*sqrt(edot)
    - d_const.a_3*T;

  // Calculate sigma_es
  double CC = d_const.koverbcubed*T/mu;
  double CCes = CC/d_const.g_0es;
  double logees = log(edot/d_const.edot_es0);
  double sigma_es = d_const.sigma_es0*exp(CCes*logees);

  // Calculate X and FX
  double X = pMTS_new[idx]/sigma_es;
  double FX = tanh(d_const.alpha*X);

  // Calculate theta
  double theta = theta_0*(1.0 - FX) + d_const.theta_IV*FX;
  double h = theta;

  // Calculate f_q (h = theta, therefore f_q.h = f_q.theta)
  double CCe = CC/d_const.g_0e;
  double logee = log(d_const.edot_0e/edot);
  double S_e = pow((1.0-pow((CCe*logee),(1.0/d_const.q_e))),(1.0/d_const.p_e));
  double f_q = mu_mu_0*S_e;

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
	    Cr(ii,jj)*rC(kk,ll)/(-f_q*h + rCr);
	}  
      }  
    }  
  }  
}
