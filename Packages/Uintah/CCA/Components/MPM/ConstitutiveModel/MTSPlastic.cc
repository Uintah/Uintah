
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MTSPlastic.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;


MTSPlastic::MTSPlastic(ProblemSpecP& ps)
{
  ps->require("s_a",d_const.s_a);
  ps->require("koverbcubed",d_const.koverbcubed);
  ps->require("edot0",d_const.edot0);
  ps->require("g0",d_const.g0);
  ps->require("q",d_const.q);
  ps->require("p",d_const.p);
  ps->require("alpha",d_const.alpha);
  ps->require("edot_s0",d_const.edot_s0);
  ps->require("A",d_const.A);
  ps->require("s_s0",d_const.s_s0);
  ps->require("a0",d_const.a0);
  ps->require("a1",d_const.a1);
  ps->require("a2",d_const.a2);
  ps->require("b1",d_const.b1);
  ps->require("b2",d_const.b2);
  ps->require("b3",d_const.b3);
  ps->require("mu_0",d_const.mu_0);

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
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
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
  double edot = sqrt(rateOfDeformation.NormSquared()*2.0/3.0);
  double delEps = edot*delT;
  double theta_0 = d_const.a0 + d_const.a1*log(edot) + d_const.a2*sqrt(edot);
  double mu = d_const.b1 - d_const.b2/(exp(d_const.b3/T) - 1.0);
  double CC = d_const.koverbcubed*T/mu;
  double s_s = d_const.s_s0*pow((edot/d_const.edot_s0),(CC/d_const.A));
  double X = (pMTS[idx] - d_const.s_a)/(s_s - d_const.s_a);
  double FX = tanh(d_const.alpha*X)/tanh(d_const.alpha);
  double theta = theta_0*(1.0 - FX);
  double s_thermal = pMTS[idx] - d_const.s_a;
  double p1 = pow((log(d_const.edot0/edot)*(CC/d_const.g0)),(1.0/d_const.q));
  double sigma = d_const.s_a + (mu/d_const.mu_0)*s_thermal*pow((1.0 - p1),(1.0/d_const.p));
  pMTS_new[idx] = pMTS[idx] + delEps*theta;
  return sigma;
}

