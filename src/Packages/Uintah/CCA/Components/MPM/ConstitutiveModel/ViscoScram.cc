#include "ConstitutiveModelFactory.h"
#include "ViscoScram.h"
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <Core/Math/MinMax.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

ViscoScram::ViscoScram(ProblemSpecP& ps, MPMLabel* Mlb)
{
  lb = Mlb;

  ps->require("PR",d_initialData.PR);
  ps->require("CrackParameterA",d_initialData.CrackParameterA);
  ps->require("CrackPowerValue",d_initialData.CrackPowerValue);
  ps->require("CrackMaxGrowthRate",d_initialData.CrackMaxGrowthRate);
  ps->require("StressIntensityF",d_initialData.StressIntensityF);
  ps->require("CrackFriction",d_initialData.CrackFriction);
  ps->require("InitialCrackRadius",d_initialData.InitialCrackRadius);
  ps->require("CrackGrowthRate",d_initialData.CrackGrowthRate);
  ps->require("G1",d_initialData.G[0]);
  ps->require("G2",d_initialData.G[1]);
  ps->require("G3",d_initialData.G[2]);
  ps->require("G4",d_initialData.G[3]);
  ps->require("G5",d_initialData.G[4]);
  ps->require("RTau1",d_initialData.RTau[0]);
  ps->require("RTau2",d_initialData.RTau[1]);
  ps->require("RTau3",d_initialData.RTau[2]);
  ps->require("RTau4",d_initialData.RTau[3]);
  ps->require("RTau5",d_initialData.RTau[4]);
  ps->require("Beta",d_initialData.Beta);
  ps->require("Gamma",d_initialData.Gamma);
  ps->require("DCp_DTemperature",d_initialData.DCp_DTemperature);
  d_se=0;

  p_statedata_label          = VarLabel::create("p.statedata_vs",
                            ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = VarLabel::create("p.statedata_vs+",
                            ParticleVariable<StateData>::getTypeDescription());
  pRandLabel                 = VarLabel::create( "p.rand",
                            ParticleVariable<double>::getTypeDescription() );
  pRandLabel_preReloc        = VarLabel::create( "p.rand+",
                            ParticleVariable<double>::getTypeDescription() );
}

ViscoScram::~ViscoScram()
{
  // Destructor

  delete p_statedata_label;
  delete p_statedata_label_preReloc;
  delete pRandLabel;
  delete pRandLabel_preReloc;
}

void ViscoScram::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   d_se=0;

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

   ParticleVariable<StateData> statedata;
   ParticleVariable<Matrix3> deformationGradient, pstress;
   ParticleVariable<double> pCrackRadius;
   ParticleVariable<double> pRand;
   new_dw->allocate(statedata,           p_statedata_label,            pset);
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   new_dw->allocate(pstress,             lb->pStressLabel,             pset);
   new_dw->allocate(pCrackRadius,        lb->pCrackRadiusLabel,        pset);
   new_dw->allocate(pRand,               pRandLabel,                   pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
      statedata[*iter].VolumeChangeHeating = 0.0;
      statedata[*iter].ViscousHeating = 0.0;
      statedata[*iter].CrackHeating = 0.0;
      statedata[*iter].CrackRadius = d_initialData.InitialCrackRadius;
      for(int imaxwell=0; imaxwell<5; imaxwell++){
	statedata[*iter].DevStress[imaxwell] = zero;
      }

      deformationGradient[*iter] = Identity;
      pstress[*iter] = zero;
      pCrackRadius[*iter] = 0.0;
//      pRand[*iter] = drand48();
      pRand[*iter] = .5;
   }
   new_dw->put(statedata,           p_statedata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress,             lb->pStressLabel);
   new_dw->put(pCrackRadius,        lb->pCrackRadiusLabel);
   new_dw->put(pRand,               pRandLabel);

   computeStableTimestep(patch, matl, new_dw);
}

void ViscoScram::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(p_statedata_label);
   from.push_back(pRandLabel);
   from.push_back(lb->pDeformationMeasureLabel);
   from.push_back(lb->pStressLabel);
   from.push_back(lb->pCrackRadiusLabel);

   to.push_back(p_statedata_label_preReloc);
   to.push_back(pRandLabel_preReloc);
   to.push_back(lb->pDeformationMeasureLabel_preReloc);
   to.push_back(lb->pStressLabel_preReloc);
   to.push_back(lb->pCrackRadiusLabel_preReloc);
}

void ViscoScram::computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  constParticleVariable<StateData> statedata;
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(statedata, p_statedata_label,  pset);
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G[0] + d_initialData.G[1] +
 	     d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;

    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void ViscoScram::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //
    //  FIX  To do:  Obtain and modify particle temperature (deg K)
    //
    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    // Unused variable - Steve
    // double J;
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    double onesixth = (1.0/6.0);
    double sqrtopf=sqrt(1.5);
    double PI = 3.141592654;

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int matlindex = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> deformationGradient_new;
    ParticleVariable<Matrix3> pstressnew;
    ParticleVariable<double> pCrackRadius;
    ParticleVariable<StateData> statedata;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<double> pvolume_deformed;    
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pRand;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    new_dw->allocate(pvolume_deformed, lb->pVolumeDeformedLabel, pset);
    new_dw->allocate(deformationGradient_new,
		     lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocate(pstressnew,     lb->pStressLabel_afterStrainRate,  pset);
    new_dw->allocate(pCrackRadius,   lb->pCrackRadiusLabel_preReloc,    pset);
    new_dw->allocate(statedata, p_statedata_label_preReloc, pset);
    old_dw->copyOut(statedata, p_statedata_label, pset);    
    new_dw->allocate(pRand, pRandLabel_preReloc, pset);
    old_dw->copyOut(pRand, pRandLabel, pset);    

    ASSERTEQ(pset, statedata.getParticleSubset());

    constNCVariable<Vector> gvelocity;

    new_dw->get(gvelocity, lb->gVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);
    double Gmw[5];
    Gmw[0]=d_initialData.G[0];
    Gmw[1]=d_initialData.G[1];
    Gmw[2]=d_initialData.G[2];
    Gmw[3]=d_initialData.G[3];
    Gmw[4]=d_initialData.G[4];

    double G = Gmw[0] + Gmw[1] +
 	       Gmw[2] + Gmw[3] + Gmw[4];
    double cf = d_initialData.CrackFriction;
    double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

    for(ParticleSubset::iterator iter = pset->begin();
					     iter != pset->end(); iter++){
      particleIndex idx = *iter;

      Gmw[0]=d_initialData.G[0]*(1.+.4*(pRand[idx]-.5));
      Gmw[1]=d_initialData.G[1]*(1.+.4*(pRand[idx]-.5));
      Gmw[2]=d_initialData.G[2]*(1.+.4*(pRand[idx]-.5));
      Gmw[3]=d_initialData.G[3]*(1.+.4*(pRand[idx]-.5));
      Gmw[4]=d_initialData.G[4]*(1.+.4*(pRand[idx]-.5));
      double G = Gmw[0] + Gmw[1] +
                 Gmw[2] + Gmw[3] + Gmw[4];
      double bulk = (2.*G*(1.+ d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));
//    double beta = 3.*d_initialData.CoefThermExp*bulk*(1.+.4*(pRand[idx]-.5));

      velGrad.set(0.0);
      // Get the node indices that surround the cell
      IntVector ni[8];
      Vector d_S[8];
      patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

       for(int k = 0; k < 8; k++) {
           const Vector& gvel = gvelocity[ni[k]];
           for (int j = 0; j<3; j++){
             for (int i = 0; i<3; i++) {
                 velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
             }
           }
       }

       // Calculate rate of deformation D, and deviatoric rate DPrime
    
       Matrix3 D = (velGrad + velGrad.Transpose())*.5;
       Matrix3 DPrime = D - Identity*onethird*D.Trace();

       // Effective deviatoric strain rate

       double EDeff = sqrtopf*DPrime.Norm();

       // Sum of old deviatoric stresses

      Matrix3 DevStress =statedata[idx].DevStress[0]+statedata[idx].DevStress[1]
	                +statedata[idx].DevStress[2]+statedata[idx].DevStress[3]
			+statedata[idx].DevStress[4];

       // old total stress norm

       double EffStress = sqrtopf*pstress[idx].Norm();

       //old deviatoric stress norm

       double DevStressNorm = DevStress.Norm();

       // old effective deviatoric stress

       double EffDevStress = sqrtopf*DevStressNorm;

       // Baseline
       double vres_a = 0.90564746;
       double vres_b =-2.90178468;
       // Aged
       //      double vres_a = 0.90863805;
       //      double vres_b =-2.5061966;

       double vres = 0.0;
       if(EDeff > 1.e-8){
	 vres = exp(vres_a*log(EDeff) + vres_b);
       }

       double p = -onethird * pstress[idx].Trace();

       int compflag = 0;
       if(p < 0.0){
	 compflag = -1;
       }

       EffStress    = (1+compflag)*EffDevStress - compflag*EffStress;
       vres        *= ((1 + compflag) - d_initialData.CrackGrowthRate*compflag);
       double sigmae = sqrt(DevStressNorm*DevStressNorm - compflag*(3*p*p));

       // Stress intensity factor
       double sif    = sqrt(1.5*PI*statedata[idx].CrackRadius)*sigmae;

       // Modification to include friction on crack faces
       double xmup   = (1 + compflag)*sqrt(45./(2.*(3. - 2.*cf*cf)))*cf;
       double a      = xmup*p*sqrt(statedata[idx].CrackRadius);
       double b      = 1. + a/d_initialData.StressIntensityF;
       double termm  = 1. + PI*a*b/d_initialData.StressIntensityF;
       double rko    = d_initialData.StressIntensityF*sqrt(termm);
       double skp    = rko*sqrt(1. + (2./d_initialData.CrackPowerValue));
       double sk1    = skp*pow((1. + (d_initialData.CrackPowerValue/2.)),
			1./d_initialData.CrackPowerValue);

       if(vres > d_initialData.CrackMaxGrowthRate){
	 vres = d_initialData.CrackMaxGrowthRate;
       }

       double c    = statedata[idx].CrackRadius;
       double cdot,cc,rk1c,rk2c,rk3c,rk4c;

       // cdot is crack speed
       // Use fourth order Runge Kutta integration to find new crack radius

       if(sif < skp ){
	 cdot = vres*pow((sif/sk1),d_initialData.CrackPowerValue);
	 cc   = vres*delT;
	 rk1c = cc*pow(sqrt(PI*c)*(EffStress/sk1),
				d_initialData.CrackPowerValue);
	 rk2c = cc*pow(sqrt(PI*(c+.5*rk1c))*(EffStress/sk1),
				d_initialData.CrackPowerValue);
	 rk3c = cc*pow(sqrt(PI*(c+.5*rk2c))*(EffStress/sk1),
				d_initialData.CrackPowerValue);
	 rk4c = cc*pow(sqrt(PI*(c+rk3c))*(EffStress/sk1),
				d_initialData.CrackPowerValue);
       }
       else{
	 cdot = vres*(1. - rko*rko/(sif*sif));
	 cc   = vres*delT;
	 rk1c = cc*(1. - rko*rko/(PI*c*EffStress*EffStress));
	 rk2c = cc*(1. - rko*rko/(PI*(c+.5*rk1c)*EffStress*EffStress));
	 rk3c = cc*(1. - rko*rko/(PI*(c+.5*rk2c)*EffStress*EffStress));
	 rk4c = cc*(1. - rko*rko/(PI*(c+rk3c)*EffStress*EffStress));
       }

       // Deviatoric stress integration

       a = d_initialData.CrackParameterA;
       for(int imw=0;imw<5;imw++){
	 // First Runga-Kutta Term
	 double con1 = 3.*c*c*cdot/(a*a*a);
	 double con3 = (c*c*c)/(a*a*a);
	 double con2 = 1. + con3;
	 Matrix3 DevStressOld = statedata[idx].DevStress[imw];
	 Matrix3 DevStressS = zero; 
	 Matrix3 DevStressT = zero;
	 for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
	   DevStressS += statedata[idx].DevStress[jmaxwell];
	   DevStressT += statedata[idx].DevStress[jmaxwell]*
				d_initialData.RTau[jmaxwell];
	 }
	 Matrix3 rk1 = (DPrime*2.*Gmw[imw] -
		       DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(Gmw[imw]/G))*delT;

	 // Second Runga-Kutta Term
	 con1 = 3.*(c+.5*rk1c)*(c+.5*rk1c)*cdot/(a*a*a);
	 con3 = (c + .5*rk1c)*(c + .5*rk1c)*(c + .5*rk1c)/(a*a*a);
	 con2 = 1. + con3;
	 DevStressOld = statedata[idx].DevStress[imw] + rk1*.5;
	 DevStressS = zero; 
	 DevStressT = zero;
	 for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
	   DevStressS += (statedata[idx].DevStress[jmaxwell] + rk1*.5);
	   DevStressT += (statedata[idx].DevStress[jmaxwell] + rk1*.5)*
						d_initialData.RTau[jmaxwell];
	 }
	 Matrix3 rk2 = (DPrime*2.*Gmw[imw] - 
		       DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(Gmw[imw]/G))*delT;

	 // Third Runga-Kutta Term
	 con1 = 3.*(c+.5*rk2c)*(c+.5*rk2c)*cdot/(a*a*a);
	 con3 = (c + .5*rk2c)*(c + .5*rk2c)*(c + .5*rk2c)/(a*a*a);
	 con2 = 1. + con3;
	 DevStressOld = statedata[idx].DevStress[imw] + rk2*.5;
	 DevStressS = zero; 
	 DevStressT = zero;
	 for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
	   DevStressS += (statedata[idx].DevStress[jmaxwell] + rk2*.5);
	   DevStressT += (statedata[idx].DevStress[jmaxwell] + rk2*.5)*
						d_initialData.RTau[jmaxwell];
	 }
	 Matrix3 rk3 = (DPrime*2.*Gmw[imw] -
			DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(Gmw[imw]/G))*delT;

	 // Fourth Runga-Kutta Term
	 con1 = 3.*(c+.5*rk3c)*(c+.5*rk3c)*cdot/(a*a*a);
	 con3 = (c + .5*rk3c)*(c + .5*rk3c)*(c + .5*rk3c)/(a*a*a);
	 con2 = 1. + con3;
	 DevStressOld = statedata[idx].DevStress[imw] + rk3;
	 DevStressS = zero; 
	 DevStressT = zero;
	 for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
	   DevStressS += (statedata[idx].DevStress[jmaxwell] + rk3);
	   DevStressT += (statedata[idx].DevStress[jmaxwell] + rk3)*
					d_initialData.RTau[jmaxwell];
	 }
	 Matrix3 rk4 = (DPrime*2.*Gmw[imw] -
			DevStressOld*d_initialData.RTau[imw] -
		        (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(Gmw[imw]/G))*delT;

        // Update Maxwell element Deviatoric Stresses

        statedata[idx].DevStress[imw] +=
		 (rk1 + rk4)*onesixth + (rk2 + rk3)*onethird;
       }

       // Update crack radius
       statedata[idx].CrackRadius +=
		onesixth*(rk1c + rk4c) + onethird*(rk2c + rk3c);

       DevStress = statedata[idx].DevStress[0]+statedata[idx].DevStress[1]
	         + statedata[idx].DevStress[2]+statedata[idx].DevStress[3]
		 + statedata[idx].DevStress[4];

       c = statedata[idx].CrackRadius;
       pCrackRadius[idx] = c;
       // Unused variable - Steve
       // double coa3   = (c*c*c)/(a*a*a);
       // Unused variable - Steve
       // double topc   = 3.*(coa3/c)*cdot;
       // Unused variable - Steve
       // double odt    = 1./delT;
       // Unused variable - Steve
       //Matrix3 SRate = DevStress*odt;

       // This is the cracking work rate
       // Unused variable - Steve
       //double scrdot =(DevStress.Norm()*DevStress.Norm()*topc
       //             + (DevStress(1,1)*SRate(1,1) + DevStress(2,2)*SRate(2,2) +
       //		         DevStress(3,3)*SRate(3,3) + 
       //			 2.*(DevStress(2,3)*SRate(2,3) +
       //		             DevStress(1,2)*SRate(1,2) + 
       //			     DevStress(1,3)*SRate(1,3))
       //			) * coa3
       //		     )/(2*G);

       double ekkdot = D.Trace();
       p = onethird*(pstress[idx].Trace()) + ekkdot*bulk*delT;

       // This is the (updated) Cauchy stress

       Matrix3 OldStress = pstress[idx];
       pstressnew[idx] = DevStress + Identity*p;

       // Viscoelastic work rate

       double svedot = 0.;
       for(int imw=0;imw<5;imw++){
	svedot += (statedata[idx].DevStress[imw].Norm()*
		   statedata[idx].DevStress[imw].Norm())
	           /(2.*Gmw[imw])*d_initialData.RTau[imw] ;
       }

       // FIX  Need to access particle temperature, thermal constants.

       // Unused variable - Steve
       // double cpnew = Cp0 + d_initialData.DCp_DTemperature*ptemperature[idx];
       // Unused variable - Steve
       //double Cv = cpnew/(1+d_initialData.Beta*ptemperature[idx]);

       // Compute the deformation gradient increment using the time_step
       // velocity gradient
       // F_n^np1 = dudx * dt + Identity
       deformationGradientInc = velGrad * delT + Identity;

       Jinc = deformationGradientInc.Determinant();

       // Update the deformation gradient tensor to its time n+1 value.
       deformationGradient_new[idx] = deformationGradientInc *
                             deformationGradient[idx];

       // get the volumetric part of the deformation
       // Unused variable - steve
       // J = deformationGradient[idx].Determinant();

       pvolume_deformed[idx]=Jinc*pvolume[idx];

       // FIX when have the particle temperatures, thermal properties

       //      double rhocv = (pvolume[idx]/pmass[idx])*Cv;

       //      statedata[idx].VolumeChangeHeating -=
       //		 d_intitalData.Gamma*ptemperature[idx]*ekkdot*delT;

       // FIX Need access to thermal constants
       // Increments to the particle temperature
       //      statedata[idx].ViscousHeating += svedot/rhocv*delT;
       //      statedata[idx].CrackHeating   += scrdot/rhocv*delT;

       //      ptemperature[idx] += (svedot + scrdot)/rhocv*delT;

       // Compute the strain energy for all the particles
       OldStress = (pstressnew[idx] + OldStress)*.5;
       d_se += (D(1,1)*OldStress(1,1) +
	        D(2,2)*OldStress(2,2) +
	        D(3,3)*OldStress(3,3) +
	        2.*(D(1,2)*OldStress(1,2) +
		    D(1,3)*OldStress(1,3) +
		    D(2,3)*OldStress(2,3))) * pvolume_deformed[idx]*delT;

       // Compute wave speed at each particle, store the maximum
       Vector pvelocity_idx = pvelocity[idx];
       if(pmass[idx] > 0){
         c_dil = sqrt((bulk + 4.*G/3.)*pvolume_deformed[idx]/pmass[idx]);
       }
       else{
         c_dil = 0.0;
         pvelocity_idx = Vector(0.0,0.0,0.0);
       }
       WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
		        Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
		        Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    delt_vartype doMech;
    old_dw->get(doMech, lb->doMechLabel);

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(doMech < 0.){
	delT_new = WaveSpeed.minComponent();
    }
    else{
	delT_new = 100.0;
    }
    new_dw->put(delt_vartype(delT_new),lb->delTLabel);
    new_dw->put(pstressnew,            lb->pStressLabel_afterStrainRate);
    new_dw->put(pCrackRadius,          lb->pCrackRadiusLabel_preReloc);
    new_dw->put(deformationGradient_new,lb->pDeformationMeasureLabel_preReloc);
    new_dw->put(sum_vartype(d_se),     lb->StrainEnergyLabel);
    new_dw->put(statedata,             p_statedata_label_preReloc);
    new_dw->put(pRand,                 pRandLabel_preReloc);
    new_dw->put(pvolume_deformed,      lb->pVolumeDeformedLabel);
  }
}

void ViscoScram::addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->doMechLabel);
  task->requires(Task::OldDW, lb->pXLabel,           matlset, Ghost::None);
  task->requires(Task::OldDW, p_statedata_label,     matlset, Ghost::None);
  task->requires(Task::OldDW, pRandLabel,            matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,      matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);

  task->requires(Task::NewDW, lb->gVelocityLabel,    matlset,
                  Ghost::AroundCells, 1);

  task->computes(lb->pStressLabel_afterStrainRate,        matlset);
  task->computes(lb->pCrackRadiusLabel_preReloc,          matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc,   matlset);
  task->computes(pRandLabel_preReloc,                     matlset);
  task->computes(p_statedata_label_preReloc,              matlset);
  task->computes(lb->pVolumeDeformedLabel,                matlset);
}

double ViscoScram::computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double G = d_initialData.G[0] + d_initialData.G[1] +
 	     d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

  rho_cur = rho_orig*exp(p_gauge/bulk);

//  if(p_gauge < .5*bulk){
//     rho_cur = rho_orig/(1-p_gauge/bulk);
//  }
//  else{
//     rho_cur = 4.*rho_orig*p_gauge/bulk;
//  }

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR ViscoScram"
       << endl;
#endif
}

void ViscoScram::computePressEOSCM(const double rho_cur,double& pressure,
                                   const double p_ref,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial* matl)
{
  double G = d_initialData.G[0] + d_initialData.G[1] +
 	     d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*log(rho_cur/rho_orig);
  dp_drho    = bulk/rho_cur;

//  if(rho_cur/rho_orig < 2.0){
//    p_g = bulk*(1.0 - rho_orig/rho_cur);
//    dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
//  }
//  else {
//    p_g = (bulk/(4*rho_orig))*rho_cur;
//    dp_drho  = bulk/(4*rho_orig);
//  }

  pressure = p_ref + p_g;
  tmp = sqrt((bulk + 4.*G/3.)/rho_cur);  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR ViscoScram" << endl;
#endif
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(ViscoScram::StateData), sizeof(double)*49);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 49, 49, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(ViscoScram::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			       "ViscoScram::StateData", true, &makeMPI_CMData);
   }
   return td;
}

} // End namespace Uintah
