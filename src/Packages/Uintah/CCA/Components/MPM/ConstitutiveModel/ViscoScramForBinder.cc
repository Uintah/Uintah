
#include "ViscoScramForBinder.h"
#include "ConstitutiveModelFactory.h"
#include "MPMMaterial.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // just added
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/NotFinished.h>
#include <Core/Math/MinMax.h>

#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

ViscoScramForBinder::ViscoScramForBinder(ProblemSpecP& ps, 
                                         MPMLabel* Mlb, 
                                         int n8or27)
{
  lb = Mlb;

  d_useModifiedEOS = false;
  ps->get("useModifiedEOS",d_useModifiedEOS);

  // Modulus data
  ps->require("bulk_modulus", d_initialData.bulkModulus);
  d_initialData.numMaxwellElements = 22;
  ps->get("num_maxwell_elements", d_initialData.numMaxwellElements);
  int nn = d_initialData.numMaxwellElements;
  d_initialData.shearModulus = new double[nn];
  for (int ii = 0; ii < nn; ++ii) {
    char* buf = new char[16];
    sprintf(buf,"shear_modulus%.2d",(ii+1));
    std::string shear(buf);
    ps->require(shear, d_initialData.shearModulus[ii]); 
  }

  // Time-temperature data for relaxtion time calculation
  d_initialData.reducedTemperature_WLF = 19.0;
  ps->get("T0", d_initialData.reducedTemperature_WLF);
  d_initialData.constantA1_WLF = -6.5;
  ps->get("A1", d_initialData.constantA1_WLF);
  d_initialData.constantA2_WLF = 120.0;
  ps->get("A2", d_initialData.constantA2_WLF);
  d_initialData.constantB1_RelaxTime = 1.5;
  ps->get("B1", d_initialData.constantB1_RelaxTime);
  d_initialData.constantB2_RelaxTime = 7;
  ps->get("B2", d_initialData.constantB2_RelaxTime);

  // Crack data for SCRAM
  d_doCrack = false;
  d_initialData.initialSize_Crack = 0.0;
  ps->get("a", d_initialData.initialSize_Crack);
  if (d_initialData.initialSize_Crack > 0.0)
    d_doCrack = true;
  d_initialData.powerValue_Crack = 10;
  ps->get("m", d_initialData.powerValue_Crack);
  d_initialData.initialRadius_Crack = 0.00003;
  ps->get("c0", d_initialData.initialRadius_Crack);
  d_initialData.maxGrowthRate_Crack = 300.0;
  ps->get("vmax", d_initialData.maxGrowthRate_Crack);
  d_initialData.stressIntensityF_Crack = 5.0e5;
  ps->get("K0", d_initialData.stressIntensityF_Crack);
  d_initialData.frictionCoeff_Crack = 0.35;
  ps->get("mu_s", d_initialData.frictionCoeff_Crack);

  // State data
  pStatedataLabel = 
    VarLabel::create("p.statedata_vsb",
		     ParticleVariable<Statedata>::getTypeDescription());
  pStatedataLabel_preReloc = 
    VarLabel::create("p.statedata_vsb+",
		     ParticleVariable<Statedata>::getTypeDescription());

  // Interpolation range
  d_8or27 = n8or27;
  if(d_8or27==8){
    NGN=1;
  } else if(d_8or27==27){
    NGN=2;
  }
}

ViscoScramForBinder::~ViscoScramForBinder()
{
  // Delete local stuff
  delete d_initialData.shearModulus;

  // Destructor
  VarLabel::destroy(pStatedataLabel);
  VarLabel::destroy(pStatedataLabel_preReloc);
}

void 
ViscoScramForBinder::initializeCMData(const Patch* patch,
				      const MPMMaterial* matl,
				      DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 one, zero(0.0); one.Identity();
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Statedata> pStatedata;
  new_dw->allocateAndPut(pStatedata, pStatedataLabel, pset);
  ParticleVariable<Matrix3> deformationGradient, pstress;
  new_dw->allocateAndPut(deformationGradient,lb->pDeformationMeasureLabel,pset);
  new_dw->allocateAndPut(pstress,            lb->pStressLabel,            pset);
  
  if (d_doCrack) {
    ParticleVariable<double> pCrackRadius;
    new_dw->allocateAndPut(pCrackRadius, lb->pCrackRadiusLabel, pset);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
      // Initialize state data
      pStatedata[*iter].numElements = d_initialData.numMaxwellElements;
      for(int ii = 0; ii < d_initialData.numMaxwellElements; ii++){
	(pStatedata[*iter].sigDev).push_back(zero);
      }
      // Initialize other stuff
      deformationGradient[*iter] = one;
      pstress[*iter] = zero;
      pCrackRadius[*iter] = d_initialData.initialRadius_Crack;
    }
  } else {
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
      // Initialize state data
      pStatedata[*iter].numElements = d_initialData.numMaxwellElements;
      for(int ii = 0; ii < d_initialData.numMaxwellElements; ii++){
	(pStatedata[*iter].sigDev).push_back(zero);
      }
      // Initialize other stuff
      deformationGradient[*iter] = one;
      pstress[*iter] = zero;
    }
  }
  computeStableTimestep(patch, matl, new_dw);
}

void 
ViscoScramForBinder::addParticleState(std::vector<const VarLabel*>& from,
				      std::vector<const VarLabel*>& to)
{
  from.push_back(pStatedataLabel);
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  to.push_back(pStatedataLabel_preReloc);
  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);

  if (d_doCrack) {
    from.push_back(lb->pCrackRadiusLabel);
    to.push_back(lb->pCrackRadiusLabel_preReloc);
  }
}

void 
ViscoScramForBinder::computeStableTimestep(const Patch* patch,
					   const MPMMaterial* matl,
					   DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<Statedata> pStatedata;
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pStatedata, pStatedataLabel,  pset);
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  int N = d_initialData.numMaxwellElements;
  double G = 0.0;
  for (int ii = 0; ii < N; ++ii) G += d_initialData.shearModulus[ii];
  double k = d_initialData.bulkModulus;

  // Compute wave speed at each particle, store the maximum
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    particleIndex idx = *iter;

    double c_dil = sqrt((k + 4.*G/3.)*pvolume[idx]/pmass[idx]);
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;

  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void 
ViscoScramForBinder::computeStressTensor(const PatchSubset* patches,
					 const MPMMaterial* matl,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  // Constants
  double onethird = (1.0/3.0);
  Matrix3 one, zero(0.); one.Identity();

  // Put the constitutive model constants into form
  // similar to that in the Mas et al. and Bennett et al. papers.
  int NN = d_initialData.numMaxwellElements;
  double kk =  d_initialData.bulkModulus;
  double* G_n = new double[NN];
  double GG = 0.0;
  for (int ii = 0; ii < NN; ++ii) {
    G_n[ii] = d_initialData.shearModulus[ii]; 
    GG += G_n[ii];
  }
  double T0 = d_initialData.reducedTemperature_WLF;
  double A1 = d_initialData.constantA1_WLF;
  double A2 = d_initialData.constantA2_WLF;
  double B1 = d_initialData.constantB1_RelaxTime;
  int B2 = d_initialData.constantB2_RelaxTime;
  double mm = d_initialData.powerValue_Crack;
  double vmax = d_initialData.maxGrowthRate_Crack;
  double K0 = d_initialData.stressIntensityF_Crack;
  double mu_s = d_initialData.frictionCoeff_Crack;
   
  // Data required from and computed for DW
  constParticleVariable<double>  pMass;
  constParticleVariable<double>  pVolume;
  constParticleVariable<double>  pTemperature;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pVelocity;
  constParticleVariable<Vector>  pSize;
  constParticleVariable<Matrix3> pDefGrad; 
  constParticleVariable<Matrix3> pStress; 
  constNCVariable<Vector>        gVelocity;

  ParticleVariable<double>       pVolume_new;    
  ParticleVariable<Matrix3>      pDefGrad_new;
  ParticleVariable<Matrix3>      pStress_new;
  ParticleVariable<Statedata>    pStatedata;

  constParticleVariable<double>  pCrackRadius;
  ParticleVariable<double>       pCrackRadius_new;

  // Get the datawarehouse index and set ghost type
  int dwi = matl->getDWIndex();
  Ghost::GhostType gac = Ghost::AroundCells;

  // Get time increment
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  // Loop thru patches
  for(int p=0;p<patches->size();p++){

    // Get current patch info
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get variables from datawarehouse
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pMass, lb->pMassLabel, pset);
    old_dw->get(pVolume, lb->pVolumeLabel, pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);
    old_dw->get(pX, lb->pXLabel, pset);
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    if(d_8or27==27) old_dw->get(pSize, lb->pSizeLabel, pset);
    old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress, lb->pStressLabel, pset);
    new_dw->get(gVelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);
    old_dw->copyOut(pStatedata, pStatedataLabel, pset);

    // Allocate and make ready for put into datawarehouse
    new_dw->allocateAndPut(pVolume_new, lb->pVolumeDeformedLabel, pset);
    new_dw->allocateAndPut(pDefGrad_new, 
			   lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new, lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pStatedata, pStatedataLabel_preReloc, pset);
    ASSERTEQ(pset, pStatedata.getParticleSubset());

    if (d_doCrack) {
      old_dw->get(pCrackRadius, lb->pCrackRadiusLabel, pset);
      new_dw->allocateAndPut(pCrackRadius_new, 
			     lb->pCrackRadiusLabel_preReloc, pset);
    }

#ifdef FRACTURE
    // for Fracture -----------------------------------------------------------
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    
    constNCVariable<Vector> Gvelocity;
    new_dw->get(Gvelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
    // ------------------------------------------------------------------------
#endif

    double strainEnergy = 0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 velGrad(0.0);
    Matrix3 defGradInc; defGradInc.Identity();
    ParticleSubset::iterator iter = pset->begin();

    // Loop thru particles
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the velocity gradient
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      if (d_8or27 == 27)
	patch->findCellAndShapeDerivatives27(pX[idx], ni, d_S, pSize[idx]);
      else
	patch->findCellAndShapeDerivatives(pX[idx], ni, d_S);
      Vector gvel;
      velGrad.set(0.0);
      for(int k = 0; k < d_8or27; k++) {
#ifdef FRACTURE
	// for Fracture ------------------------------------------------------
	if(pgCode[idx][k]==1)
	  gvel = gVelocity[ni[k]];
	else if(pgCode[idx][k]==2)
	  gvel = Gvelocity[ni[k]];
	else {
	  cout << "Unknown velocity field in "
               << "ViscoScramForBinder::computeStressTensor:"
	       << pgCode[idx][k] << endl;
	  exit(1);
	}
	// -------------------------------------------------------------------
#else
	gvel = gVelocity[ni[k]];
#endif
	for (int j = 0; j<3; j++){
	  double temp = d_S[k][j] * oodx[j];
	  for (int i = 0; i<3; i++) {
	    velGrad(i+1,j+1)+=gvel[i] * temp;
	  }
	}
      }

      // Calculate rate of deformation D, deviatoric rate DPrime,
      // and effective deviatoric strain rate eDotPrimeEff
      Matrix3 DD = (velGrad + velGrad.Transpose())*0.5;
      Matrix3 eDot = DD - one*onethird*DD.Trace();

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + one)
      // Update the deformation gradient tensor to its time n+1 value.
      defGradInc = velGrad * delT + one;
      double Jinc = defGradInc.Determinant();
      pDefGrad_new[idx] = defGradInc * pDefGrad[idx];

      // Update the volume
      pVolume_new[idx]=Jinc*pVolume[idx];

      // For the stress update we need the relaxation times tau(ii)
      // First calculate a_T
      double TDiff = pTemperature[idx] - T0;
      double a_T = pow(10.0, (A1*TDiff/(A2 + TDiff))); 

      // Then calculate relaxation times and store in an array
      // (Note that shear moduli are already in the array Gi)
      double* Tau_n = new double[NN];
      for (int ii = 0; ii < NN; ++ii) {
        Tau_n[ii] = B1*a_T*pow(10.0, (B2-ii));
      }

      // Store the deviatoric stress in each element in an array
      // and calculate the sum
      Matrix3* S_n = new Matrix3[NN];
      Matrix3 sigPrime(0.0);
      for (int ii = 0; ii < NN; ++ii) {
        S_n[ii] = pStatedata[idx].sigDev[ii];
        sigPrime += S_n[ii];
      }

      // old total stress norm, effective stress, hydrostaic pressure
      double p = -onethird * pStress[idx].Trace();

      // Deviatoric stress integration
      Matrix3* S_n_new = new Matrix3[NN];
      if (d_doCrack) {

        // Calculate updated crack radius
	// Decide tension or compression (tension +ve) and calculate the
	// effective stress
	int compflag = (p < 0.0) ? -1 : 0;
	double sigma = sqrt(sigPrime.NormSquared() - compflag*(3*p*p));

	// Modification to include friction on crack faces
	double xmup   = (1 + compflag)*
                        sqrt(45.0/(2.0*(3.0 - 2.0*mu_s*mu_s)))*mu_s;
	double a      = xmup*p*sqrt(pCrackRadius[idx]);
	double b      = 1.0 + a/K0;
	double termm  = sqrt(1.0 + (M_PI*a*b)/K0);

        // Calculate crack growth rate and new crack radius
        // using fourth-order Runge-Kutta
        double K0 = K0*termm;
        double cc = pCrackRadius[idx];
        double KK = sqrt(M_PI*cc)*sigma;
        double KPrime = K0*sqrt(1.0+2.0/mm);
        double cDot = 0.0;
        double* rkc = new double[4];
        for (int ii = 0; ii < 4; ++ii) rkc[ii] = 0.0;
        if (KK < KPrime) {
          double K1 = KPrime*pow((1.0+mm/2.0), (1.0/mm));
          cDot = vmax*pow((KK/K1), mm);
          pCrackRadius_new[idx] = doRungeKuttaForCrack(
                                    &ViscoScramForBinder::crackGrowthEqn1, 
                                    cc, (double) delT, K0, sigma, rkc);
        } else {
          double K0overK = K0/KK;
          cDot = vmax*(1.0-K0overK*K0overK);
          pCrackRadius_new[idx] = doRungeKuttaForCrack(
                                    &ViscoScramForBinder::crackGrowthEqn2, 
                                    cc, (double) delT, K0, sigma, rkc);
        }

        // Deviatoric stress integration
        doRungeKuttaForStress(&ViscoScramForBinder::stressEqnWithCrack,
			      S_n, (double) delT, rkc, cc,
			      G_n, Tau_n, eDot, cDot, S_n_new); 
        delete rkc;

      } else {

        // Deviatoric stress integration
        double* rkc = new double[4];
        for (int ii = 0; ii < 4; ++ii) rkc[ii] = 0.0;
        doRungeKuttaForStress(&ViscoScramForBinder::stressEqnWithoutCrack,
			      S_n, (double) delT, rkc, 0.0,
			      G_n, Tau_n, eDot, 0.0, S_n_new); 
        delete rkc;
      }
      delete G_n;
      delete Tau_n;
      delete S_n;

      // Calculate the total deviatoric stress
      // and Update Maxwell element Deviatoric Stresses
      Matrix3 sigPrime_new(0.0);
      for (int ii = 0; ii < NN; ++ii) {
        sigPrime_new += S_n_new[ii];
        pStatedata[idx].sigDev[ii] = S_n_new[ii];
      }
      delete S_n_new;
      pStatedata[idx].numElements = NN;

      // Update the Cauchy stress
      double ekkdot = DD.Trace();
      p = onethird*(pStress[idx].Trace()) + ekkdot*kk*delT;
      pStress_new[idx] = one*p + sigPrime_new;

      // Compute the strain energy for all the particles
      Matrix3 sigAv = (pStress_new[idx]+pStress[idx])*0.5;
      strainEnergy += (DD(1,1)*sigAv(1,1) + DD(2,2)*sigAv(2,2) +
	               DD(3,3)*sigAv(3,3) + 2.*(DD(1,2)*sigAv(1,2) +
		       DD(1,3)*sigAv(1,3) + DD(2,3)*sigAv(2,3))) * 
                      pVolume_new[idx]*delT;

      // Compute wave speed at each particle, store the maximum
      Vector pVelocity_idx = pVelocity[idx];
      double c_dil = sqrt((kk + 4.*GG/3.)*pVolume_new[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVelocity_idx.x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pVelocity_idx.y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pVelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new),lb->delTLabel);
    new_dw->put(sum_vartype(strainEnergy), lb->StrainEnergyLabel);
  }
}

void 
ViscoScramForBinder::computeStressTensor(const PatchSubset* ,
				const MPMMaterial* ,
				DataWarehouse* ,
				DataWarehouse* ,
				Solver* ,
				const bool )
{
}

// Solve an ordinary differential equation of the form
// dy/dt = f(y,t)
// using a fourth-order Runge-Kutta method
// between t=T and t=T+delT (h = delT)
// For the crack : need two extra variables K0 and sigma
double
ViscoScramForBinder::doRungeKuttaForCrack(double (ViscoScramForBinder::*fptr)
                                                 (double, double, double),
                                  double y, 
                                  double h,
                                  double K0,
                                  double sigma,
                                  double* kk) 
{
  double k1 = (this->*fptr)(y, K0, sigma);
  double k2 = (this->*fptr)(y+0.5*h*k1, K0, sigma);
  double k3 = (this->*fptr)(y+0.5*h*k2, K0, sigma);
  double k4 = (this->*fptr)(y+h*k3, K0, sigma);

  kk[0] = k1; kk[1] = k2; kk[2] = k3; kk[3] = k4;

  return (y + (h/6.0)*(k1+2.0*(k2+k3)+k4));
}

// Function for calculating crack growth rate
// and crack size
double 
ViscoScramForBinder::crackGrowthEqn1(double c, double K0, double sigma) 
{
  double m = d_initialData.powerValue_Crack;
  double vmax = d_initialData.maxGrowthRate_Crack;
  double K = sqrt(M_PI*c)*sigma;
  double KPrime = K0*sqrt(1.0+2.0/m);
  double K1 = KPrime*pow((1.0+m/2.0), (1.0/m));
  return vmax*pow((K/K1), m);
}

double 
ViscoScramForBinder::crackGrowthEqn2(double c, double K0, double sigma) 
{
  double vmax = d_initialData.maxGrowthRate_Crack;
  double K = sqrt(M_PI*c)*sigma;
  double K0overK = K0/K;
  return vmax*(1.0-K0overK*K0overK);
}
	 
// Solve the stress equation using a fourth-order Runge-Kutta scheme
void
ViscoScramForBinder::doRungeKuttaForStress(void (ViscoScramForBinder::*fptr)
                                                (Matrix3*, double, double*, 
                                                 double*, Matrix3&, double, 
                                                 Matrix3*), 
                                           Matrix3* y_n,
					   double h, 
					   double* rkc, 
                                           double c,
					   double* G_n, 
					   double* Tau_n, 
                                           Matrix3& eDot,
					   double cDot,
                                           Matrix3* y_rk)
{
  int n = d_initialData.numMaxwellElements;
  double c_rk; 

  for (int ii = 0; ii < n; ++ii) y_rk[ii] = y_n[ii];
  c_rk = c; 
  Matrix3 k1[n];
  ((this->*fptr)(y_rk, c_rk, G_n, Tau_n, eDot, cDot, k1));

  for (int ii = 0; ii < n; ++ii) y_rk[ii] = y_n[ii]+k1[ii]*(0.5*h);
  c_rk = c + 0.5*h*rkc[0];
  Matrix3 k2[n];
  ((this->*fptr)(y_rk, c_rk, G_n, Tau_n, eDot, cDot, k2));

  for (int ii = 0; ii < n; ++ii) y_rk[ii] = y_n[ii]+k2[ii]*(0.5*h);
  c_rk = c + 0.5*h*rkc[1];
  Matrix3 k3[n];
  ((this->*fptr)(y_rk, c_rk, G_n, Tau_n, eDot, cDot, k3));

  for (int ii = 0; ii < n; ++ii) y_rk[ii] = y_n[ii]+k3[ii]*h;
  c_rk = c + 0.5*h*rkc[2];
  Matrix3 k4[n];
  ((this->*fptr)(y_rk, c_rk, G_n, Tau_n, eDot, cDot, k4));

  for (int ii = 0; ii < n; ++ii) 
     y_rk[ii] = y_n[ii] + ((k1[ii]+k4[ii])+(k2[ii]+k3[ii])*2.0)*(h/6.0);
}

void
ViscoScramForBinder::stressEqnWithCrack(Matrix3* S_n,
                                        double c,
                                        double* G_n,
                                        double* Tau_n,
                                        Matrix3& eDot,
                                        double cDot,
                                        Matrix3* k_n)
{
  double a = d_initialData.initialSize_Crack;
  double c1 = c/a;  double c1Sq = c1*c1; double c1Cub = c1Sq*c1;
  double c2 = cDot/a; double c1Sqc2 = c1Sq*c2;
  int n = d_initialData.numMaxwellElements;
  double G = 0.0;
  Matrix3 sumS_nOverTau_n(0.0);
  Matrix3 S(0.0);
  for (int ii = 0; ii < n; ++ii) {
    G += G_n[ii];
    sumS_nOverTau_n += S_n[ii]/Tau_n[ii];
    S += S_n[ii];
  }
  Matrix3 SDot = (eDot*(2.0*G) - S*(3.0*c1Cub*c2) - sumS_nOverTau_n)/(1.0+c1Cub);
  for (int ii = 0; ii < n; ++ii) {
    k_n[ii] = eDot*(2.0*G_n[ii]) - S_n[ii]/Tau_n[ii] 
                 - (S*(3.0*c1Sqc2) + SDot*c1Cub)*(G_n[ii]/G);
  }
}
 
void
ViscoScramForBinder::stressEqnWithoutCrack(Matrix3* S_n,
					   double ,
					   double* G_n,
					   double* Tau_n,
					   Matrix3& eDot,
					   double ,
					   Matrix3* k_n)
{
  int n = d_initialData.numMaxwellElements;
  for (int ii = 0; ii < n; ++ii) {
    k_n[ii] = eDot*(2.0*G_n[ii]) - S_n[ii]/Tau_n[ii] ;
  }
}
 
void 
ViscoScramForBinder::addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pStatedataLabel,    matlset);
  if (d_doCrack) 
    task->computes(lb->pCrackRadiusLabel,matlset);
}

void 
ViscoScramForBinder::addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet*) const
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,Ghost::None);
  task->requires(Task::OldDW, pStatedataLabel,             matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel,       matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
  if(d_8or27==27){
    task->requires(Task::OldDW, lb->pSizeLabel,            matlset,Ghost::None);
  }

  task->requires(Task::NewDW, lb->gVelocityLabel,          matlset, gac, NGN);

#ifdef FRACTURE
  // for Fracture -------------------------------------------------------------
  task->requires(Task::NewDW, lb->pgCodeLabel,            matlset,Ghost::None);
  task->requires(Task::NewDW, lb->GVelocityLabel,         matlset, gac, NGN);
  // --------------------------------------------------------------------------
#endif

  task->computes(lb->pStressLabel_preReloc,               matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc,   matlset);
  task->computes(pStatedataLabel_preReloc,                matlset);
  task->computes(lb->pVolumeDeformedLabel,                matlset);

  if (d_doCrack) {
    task->requires(Task::OldDW, lb->pCrackRadiusLabel, matlset, Ghost::None);
    task->computes(lb->pCrackRadiusLabel_preReloc,     matlset);
  }
}

void 
ViscoScramForBinder::addComputesAndRequires(Task* ,
					const MPMMaterial* ,
					const PatchSet*,
					const bool ) const
{
}
	 
double 
ViscoScramForBinder::computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.bulkModulus;
  
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;       // Modified EOS
    double n = p_ref/bulk;
    rho_cur  = rho_orig*pow(pressure/A,n);
  }else{                    // Standard EOS
    rho_cur = rho_orig/(1-p_gauge/bulk);
  }
  return rho_cur;

}

void 
ViscoScramForBinder::computePressEOSCM(double rho_cur,double& pressure,
                                   double p_ref,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial* matl)
{
  double bulk = d_initialData.bulkModulus;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;         // MODIFIED EOS
    double n = bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;       // speed of sound squared
  }else{                       // STANDARD EOS
    double p_g = bulk*(1.0 - rho_orig/rho_cur);
    pressure   = p_ref + p_g;  
    dp_drho    = bulk*rho_orig/(rho_cur*rho_cur);
    tmp        = dp_drho;       // speed of sound squared
  }
}

double 
ViscoScramForBinder::getCompressibility()
{
  double bulk = d_initialData.bulkModulus;
  return 1.0/bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

static
MPI_Datatype
makeMPI_CMData()
{
   //ASSERTEQ(sizeof(ViscoScramForBinder::Statedata), sizeof(double)*49);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, sizeof(ViscoScramForBinder::Statedata), 
                      sizeof(ViscoScramForBinder::Statedata), 
                      MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const Uintah::TypeDescription*
fun_getTypeDescription(ViscoScramForBinder::Statedata*)
{
   static Uintah::TypeDescription* td = 0;
   if(!td){
      td = scinew Uintah::TypeDescription(TypeDescription::Other,
	"ViscoScramForBinder::Statedata", true, &makeMPI_CMData);
   }
   return td;
}

} // End namespace Uintah

namespace SCIRun {
void swapbytes(Uintah::ViscoScramForBinder::Statedata& d)
{
  for (int i = 0; i < (int) d.numElements; i++) 
    swapbytes(d.sigDev[i]);
  swapbytes(d.numElements);
}
  
} // namespace SCIRun
