//  CompMooneyRivlin.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Mooney Rivlin materials
//     

#include "CompMooneyRivlin.h"
#include <iostream>
using std::cerr;
using namespace Uintah::Components;
using SCICore::Geometry::Vector;

CompMooneyRivlin::CompMooneyRivlin(const Region* region,
				   const MPMMaterial* matl,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw)
{
  // Constructor
  // Create storage in datawarehouse for data fields
  // needed for model parameters

#ifdef WONT_COMPILE_YET
  int matlindex = matl->getDWIndex();

  ParticleVariable<CMData> cmdata;
  dw->get(cmdata,"p.cmdata", matlindex, region, 0);
#endif
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
}

void CompMooneyRivlin::initializeCMData(const Region* region,
					const MPMMaterial* matl,
					DataWarehouseP& new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure

}

void CompMooneyRivlin::computeStressTensor(const Region* region,
                                           const MPMMaterial* matl,
                                           const DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  Matrix3 Identity,deformationGradientInc,B,velGrad;
  double invar1,invar3,J,w1,w2,w3,i3w3,w1pi1w2;
  Identity.Identity();
  double WaveSpeed = 0.0,c_dil = 0.0,c_rot = 0.0;

  Vector dx = region->dCell();
  double oodx[3] { 1.0/dx.x(),1.0/dx.y(),1.0/dx.z() };

  int matlindex = matl->getDWIndex();

  // Create array for the particle position
  ParticleVariable<Vector> px;
  old_dw->get(px, "p.x", matlindex, region, 0);
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, "p.deformationMeasure", matlindex, region, 0);

  // Create array for the particle stress
  ParticleVariable<Matrix3> stress;
  new_dw->get(stress, "p.stress", matlindex, region, 0);

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, "p.cmdata", matlindex, region, 0);
  ParticleVariable<Matrix3> pmass;
  old_dw->get(pmass, "p.mass", matlindex, region, 0);
  ParticleVariable<Matrix3> pvolume;
  old_dw->get(pvolume, "p.volume", matlindex, region, 0);

  NCVariable<Vector> gvelocity;
  new_dw->get(gvelocity, "g.velocity", matlindex,region, 0);
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

    new_dw->put(stress, "p.stress", matlindex, region, 0);
    new_dw->put(deformationGradient, "p.deformationMeasure",
						matlindex, region, 0);
#else
    cerr << "CompMooneyRivlin::computeStressTensor not finished\n";
#endif
}

double CompMooneyRivlin::computeStrainEnergy(const Region* region,
                                             const MPMMaterial* matl,
                                             const DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  double invar1,invar2,invar3,J,se=0.0;
  Matrix3 B,BSQ;

  matlindex = matl->getDWIndex();

  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  new->get(deformationGradient, "p.deformationMeasure", matlindex, region, 0);
  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, "p.cmdata", matlindex, region, 0);
  ParticleVariable<Matrix3> pvolume;
  old_dw->get(pvolume, "p.volume", matlindex, region, 0);

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
#else
  cerr << "CompMooneyRivlin::computeStrainEnergy not finished\n";
  return 0;
#endif

}

void CompMooneyRivlin::readParameters(ProblemSpecP ps, double *p_array)
{
  ps->require("he_constant_1",p_array[0]);
  ps->require("he_constant_2",p_array[1]);
  ps->require("he_constant_3",p_array[2]);
  ps->require("he_constant_4",p_array[3]);
 
}

#ifdef WONT_COMPILE_YET
ConstitutiveModel* CompMooneyRivlin::readParametersAndCreate(ProblemSpecP ps)
{

  double p_array[4];
  readParameters(ps, p_array);
  return(create(p_array));
  
}

ConstitutiveModel* CompMooneyRivlin::readRestartParametersAndCreate(
                                             ProblemSpecP ps)
{
#if 0
  Matrix3 st(0.0);
  ConstitutiveModel *cm = readParametersAndCreate(ps);
  
  in >> st(1,1) >> st(1,2) >> st(1,3)
     >> st(2,2) >> st(2,3) >> st(3,3);
  st(2,1)=st(1,2);
  st(3,1)=st(1,3);
  st(3,2)=st(2,3);
  cm->setStressTensor(st);
  
  return(cm);


#endif
}
#endif

ConstitutiveModel* CompMooneyRivlin::create(double *p_array)
{
#ifdef WONT_COMPILE_YET
  return(new CompMooneyRivlin(p_array[0], p_array[1], p_array[2], p_array[3]));
#else
  return 0;
#endif
}

// $Log$
// Revision 1.7  2000/04/19 05:26:03  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.6  2000/04/14 17:34:41  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.5  2000/03/24 00:44:33  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//
// Revision 1.4  2000/03/20 17:17:07  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.3  2000/03/16 00:49:31  guilkey
// Fixed the parameter lists in the .cc files
//
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
