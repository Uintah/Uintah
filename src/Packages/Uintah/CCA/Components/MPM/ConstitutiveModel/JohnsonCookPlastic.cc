
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/JohnsonCookPlastic.h>
#include <math.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;


JohnsonCookPlastic::JohnsonCookPlastic(ProblemSpecP& ps)
{
  ps->require("A",d_initialData.A);
  ps->require("B",d_initialData.B);
  ps->require("C",d_initialData.C);
  ps->require("n",d_initialData.n);
  ps->require("m",d_initialData.m);

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
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
    pPlasticStrain_new[*iter] = 0.0;
  }
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
JohnsonCookPlastic::computeFlowStress(const Matrix3& rateOfDeformation,
                                      const Matrix3& ,
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
  double strainPart = d_initialData.A + d_initialData.B*pow(ep,d_initialData.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0) 
    strainRatePart = pow((1.0 + epdot),d_initialData.C);
  else
    strainRatePart = 1.0 + d_initialData.C*log(epdot);
  double Tr = matl->getRoomTemperature();
  double Tm = matl->getMeltTemperature();
  double m = d_initialData.m;
  double Tstar = (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? 1.0 : (1.0-pow(Tstar,m));
  //if (Tstar < 0.0) {
  //  cerr << " ep = " << ep << " Strain Part = " << strainPart << endl;
  //  cerr << "epdot = " << epdot << " Strain Rate Part = " << strainRatePart << endl;
  //  cerr << "Tstar = " << Tstar << " T = " << T << " Tr = " << Tr << " Tm = " << Tm << endl;
  //  cerr << "Forcing Tstar to be 0.0" << endl;
  //  Tstar = 0.0;
  //}
  //double tm = pow(Tstar,m);
  //double tempPart = 1.0 - tm;
  return (strainPart*strainRatePart*tempPart);
}

