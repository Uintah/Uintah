#include "ConstitutiveModelFactory.h"
#include "ViscoScram.h"
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

ViscoScram::ViscoScram(ProblemSpecP& ps)
{
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

  p_statedata_label = scinew VarLabel("p.statedata_vs",
                                ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = scinew VarLabel("p.statedata_vs+",
                                ParticleVariable<StateData>::getTypeDescription());
}

ViscoScram::~ViscoScram()
{
  // Destructor

  delete p_statedata_label;
  delete p_statedata_label_preReloc;
 
}

void ViscoScram::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   d_se=0;

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

   ParticleVariable<StateData> statedata;
   new_dw->allocate(statedata, p_statedata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);
   ParticleVariable<double> pCrackRadius;
   new_dw->allocate(pCrackRadius, lb->pCrackRadiusLabel, pset);

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
   }
   new_dw->put(statedata, p_statedata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);
   new_dw->put(pCrackRadius, lb->pCrackRadiusLabel);

   computeStableTimestep(patch, matl, new_dw);

}

void ViscoScram::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(p_statedata_label);
   to.push_back(p_statedata_label_preReloc);
}

void ViscoScram::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<StateData> statedata;
  new_dw->get(statedata, p_statedata_label, pset);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G[0] + d_initialData.G[1] +
 	     d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
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

void ViscoScram::computeStressTensor(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw)
{
  //
  //  FIX  To do:  Obtain and modify particle temperature (deg K)
  //
  Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
  double J;
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
  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, pset);

  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  old_dw->get(pstress, lb->pStressLabel, pset);

  // Retrieve the array of constitutive parameters
  ParticleVariable<double> pCrackRadius;
  old_dw->get(pCrackRadius, lb->pCrackRadiusLabel, pset);
  ParticleVariable<StateData> statedata;
  old_dw->get(statedata, p_statedata_label, pset);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  old_dw->get(pvelocity, lb->pVelocityLabel, pset);
  ParticleVariable<double> ptemperature;
  old_dw->get(ptemperature, lb->pTemperatureLabel, pset);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  double G = d_initialData.G[0] + d_initialData.G[1] +
 	     d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double cf = d_initialData.CrackFriction;
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

  double Cp0 = matl->getSpecificHeat();
  //  cout << "Cp0 = " << Cp0 << endl;

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];
     patch->findCellAndShapeDerivatives(px[idx], ni, d_S);

      for(int k = 0; k < 8; k++) {
          Vector& gvel = gvelocity[ni[k]];
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

      EffStress     = (1+compflag)*EffDevStress - compflag*EffStress;
      vres         *= ((1 + compflag) - d_initialData.CrackGrowthRate*compflag);
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
	Matrix3 rk1 = (DPrime*2.*d_initialData.G[imw] -
		       DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(d_initialData.G[imw]/G))*delT;

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
	Matrix3 rk2 = (DPrime*2.*d_initialData.G[imw] - 
		       DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(d_initialData.G[imw]/G))*delT;

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
	Matrix3 rk3 = (DPrime*2.*d_initialData.G[imw] -
			DevStressOld*d_initialData.RTau[imw] -
		       (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(d_initialData.G[imw]/G))*delT;

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
	Matrix3 rk4 = (DPrime*2.*d_initialData.G[imw] -
			DevStressOld*d_initialData.RTau[imw] -
		        (DevStressS*con1 +
			(DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
		          *(d_initialData.G[imw]/G))*delT;

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
      double coa3   = (c*c*c)/(a*a*a);
      double topc   = 3.*(coa3/c)*cdot;
      double odt    = 1./delT;
      Matrix3 SRate = DevStress*odt;

      // This is the cracking work rate

      double scrdot =(DevStress.Norm()*DevStress.Norm()*topc
                      + (DevStress(1,1)*SRate(1,1) + DevStress(2,2)*SRate(2,2) +
		         DevStress(3,3)*SRate(3,3) + 
			 2.*(DevStress(2,3)*SRate(2,3) +
		             DevStress(1,2)*SRate(1,2) + 
			     DevStress(1,3)*SRate(1,3))
			) * coa3
		     )/(2*G);

      double ekkdot = D.Trace();
      p = onethird*(pstress[idx].Trace()) + ekkdot*bulk*delT;

      // This is the (updated) Cauchy stress

      Matrix3 OldStress = pstress[idx];
      pstress[idx] = DevStress + Identity*p;

      // Viscoelastic work rate

      double svedot = 0.;
      for(int imw=0;imw<5;imw++){
	svedot += (statedata[idx].DevStress[imw].Norm()*
		   statedata[idx].DevStress[imw].Norm())
	           /(2.*d_initialData.G[imw])*d_initialData.RTau[imw] ;
      }

      // FIX  Need to access particle temperature, thermal constants.

      double cpnew = Cp0 + d_initialData.DCp_DTemperature*ptemperature[idx];
      double Cv = cpnew/(1+d_initialData.Beta*ptemperature[idx]);

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

      // get the volumetric part of the deformation
      J = deformationGradient[idx].Determinant();

      pvolume[idx]=Jinc*pvolume[idx];

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
      OldStress = (pstress[idx] + OldStress)*.5;
      d_se += (D(1,1)*OldStress(1,1) +
	       D(2,2)*OldStress(2,2) +
	       D(3,3)*OldStress(3,3) +
	       2.*(D(1,2)*OldStress(1,2) +
		   D(1,3)*OldStress(1,3) +
		   D(2,3)*OldStress(2,3))
	       )*pvolume[idx]*delT;

      // Compute wave speed at each particle, store the maximum

      if(pmass[idx] > 0){
        c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
      }
      else{
        c_dil = 0.0;
        pvelocity[idx] = Vector(0.0,0.0,0.0);
      }
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new),lb->delTLabel);
  new_dw->put(pstress, lb->pStressAfterStrainRateLabel);
  new_dw->put(pCrackRadius, lb->pCrackRadiusLabel);
  new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);

  // Put the strain energy in the data warehouse
  new_dw->put(sum_vartype(d_se), lb->StrainEnergyLabel);

  // This is updated
  new_dw->put(statedata, p_statedata_label_preReloc);
  // Store deformed volume
  new_dw->put(pvolume,lb->pVolumeDeformedLabel);
}

//double ViscoScram::computeStrainEnergy(const Patch* patch,
//                                        const MPMMaterial* matl,
//                                        DataWarehouseP& new_dw)
//{
//  double se=0;
//
//  return se;
//}

void ViscoScram::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) const
{
   task->requires(old_dw, lb->pXLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, p_statedata_label, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pTemperatureLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matl->getDWIndex(), patch,
                  Ghost::AroundCells, 1);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressAfterStrainRateLabel, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pCrackRadiusLabel, matl->getDWIndex(), patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, matl->getDWIndex(), patch);
   task->computes(new_dw, p_statedata_label_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(), patch);
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(ViscoScram::StateData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
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
