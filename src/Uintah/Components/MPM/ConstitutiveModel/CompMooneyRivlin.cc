//  CompMooneyRivlin.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Mooney Rivlin materials
//     

#include "CompMooneyRivlin.h"

CompMooneyRivlin::CompMooneyRivlin(region, matl, old_dw, new_dw)
{
  // Constructor
  // Create storage in datawarehouse for data fields
  // needed for model parameters

  ParticleVariable<CMData> cmdata;
  dw->get(cmdata,"p.cmdata", matl, region, 0);
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
}

CompMooneyRivlin::intializeCMData(region, matl, new_dw);
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure

}

void CompMooneyRivlin::computeStressTensor(region, matl, old_dw, new_dw)
{

  Matrix3 Identity,deformationGradientInc,B,velGrad;
  double invar1,invar3,J,w1,w2,w3,i3w3,w1pi1w2;
  Identity.Identity();
  double WaveSpeed = 0.0,c_dil = 0.0,c_rot = 0.0;

  Vector dx = region->dCell();
  double oodx[3] { 1.0/dx.x(),1.0/dx.y(),1.0/dx.z() };

  // Create array for the particle position
  ParticleVariable<Vector> px;
  old_dw->get(px, "p.x", matl, region, 0);
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, "p.deformationMeasure", matl, region, 0);
  // Create array for the particle stress
  ParticleVariable<Matrix3> stress;
  new_dw->get(stress, "p.stress", matl, region, 0);
  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, "p.cmdata", matl, region, 0);
  ParticleVariable<Matrix3> pmass;
  old_dw->get(pmass, "p.mass", matl, region, 0);
  ParticleVariable<Matrix3> pvolume;
  old_dw->get(pvolume, "p.volume", matl, region, 0);

  NCVariable<Vector> gvelocity;
  new_dw->get(gvelocity, "g.velocity", matl,region, 0);
  SoleVariable<double> delt;
  old_dw->get(delt, "delt");

  ParticleSubset* pset = px.getParticleSubset();
  ASSERT(pset == pstress.getParticleSubset());
  ASSERT(pset == pdeformationMeasure.getParticleSubset());
  ASSERT(pset == pmass.getParticleSubset());
  ASSERT(pset == pvolume.getParticleSubset());

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     ParticleSet::index idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     Array3Index ni[8];
     Vector d_S[8];
     if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
         continue;

      for(int k = 0; k < 8; k++) {
          Vector& gvel = gvelocity[ni[k]];
          for (int j = 0; j<3; j++){
            for (int i = 0; i<3; i++) {
                velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
            }
          }
      }


      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delt + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient[idx] = deformationGradientInc * deformationGradient;

      // Actually calculate the stress from the n+1 deformation gradient.

      // Compute the left Cauchy-Green deformation tensor
      B = deformationGradient[idx] * deformationGradient[idx].Transpose();

      // Compute the invariants
      invar1 = B.Trace();
      J = deformationGradient.Determinant();
      invar3 = J*J;

      double C1 = cmdata[idx].C1;
      double C2 = cmdata[idx].C2;
      double C3 = cmdata[idx].C3;
      double C4 = cmdata[idx].C4;

      w1 = C1;
      w2 = C2;
      w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);

      // Compute T = 2/sqrt(I3)*(I3*W3*Identity + (W1+I1*W2)*B - W2*B^2)
      w1pi1w2 = w1 + invar1*w2;
      i3w3 = invar3*w3;

      stress[idx]=(B*w1pi1w2 - (B*B)*w2 + Identity*i3w3)*2.0/J;

      // Compute wave speed at each particle, store the maximum
      double mu,PR,lambda;
      mu = 2.*(C1[idx] + C2[idx]);
      PR = (2.*C1[idx] + 5.*C2[idx] + 2.*C4[idx])/
		(4.*C4[idx] + 5.*C1[idx] + 11.*C2[idx]);
      lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;
      c_dil = Max(c_dil,(lambda + 2.*mu)*pvolume[idx]/pmass[idx]);
      c_rot = Max(c_rot, mu*pvolume[idx]/pmass[idx]);
    }
    WaveSpeed = sqrt(Max(c_rot,c_dil));
    new_dw->put(SoleVariable<double>(MaxWaveSpeed),
				"WaveSpeed", DataWarehouse::Max);

    new_dw->put(stress, "p.stress", matl, region, 0);
    new_dw->put(deformationGradient, "p.deformationMeasure",
						matl, region, 0);
}

double CompMooneyRivlin::computeStrainEnergy(region, matl, new_dw)
{
  double invar1,invar2,invar3,J,se=0.0;
  Matrix3 B,BSQ;

  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  new->get(deformationGradient, "p.deformationMeasure", matl, region, 0);
  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, "p.cmdata", matl, region, 0);
  ParticleVariable<Matrix3> pvolume;
  old_dw->get(pvolume, "p.volume", matl, region, 0);

  ParticleSubset* pset = pvolume.getParticleSubset();
  ASSERT(pset == pdeformationMeasure.getParticleSubset());
  ASSERT(pset == pcmdata.getParticleSubset());

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     ParticleSet::index idx = *iter;

     double C1 = cmdata[idx].C1;
     double C2 = cmdata[idx].C2;
     double C3 = cmdata[idx].C3;
     double C4 = cmdata[idx].C4;
 
     B = deformationGradient[idx] * deformationGradient[idx].Transpose();
     // Compute the invariants
     invar1 = B.Trace();
     invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
     J = deformationGradient.Determinant();
     invar3 = J*J;
  
     se += C1*(invar1-3.0) + C2*(invar2-3.0) +
           C3*(1.0/(invar3*invar3) - 1.0) +
           C4*(invar3-1.0)*(invar3-1.0);
  }

  return se;
}

// $Log$
// Revision 1.2  2000/03/15 20:05:57  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a region of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
