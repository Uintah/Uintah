//  CompMooneyRivlin.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Mooney Rivlin materials
//     

#include "CompMooneyRivlin.h"
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <iostream>
using std::cerr;
using namespace Uintah::MPM;
using SCICore::Math::Min;
using SCICore::Math::Max;
using SCICore::Geometry::Vector;

CompMooneyRivlin::CompMooneyRivlin(ProblemSpecP& ps)
{
  ps->require("he_constant_1",d_initialData.C1);
  ps->require("he_constant_2",d_initialData.C2);
  ps->require("he_constant_3",d_initialData.C3);
  ps->require("he_constant_4",d_initialData.C4);
  p_cmdata_label = new VarLabel("p.cmdata",
				ParticleVariable<CMData>::getTypeDescription());
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
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   ParticleVariable<CMData> cmdata;
   new_dw->allocate(cmdata, p_cmdata_label, matl->getDWIndex(), region);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, pDeformationMeasureLabel, matl->getDWIndex(), region);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, pStressLabel, matl->getDWIndex(), region);
   ParticleSubset* pset = cmdata.getParticleSubset();
   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
	 cmdata[*iter] = d_initialData;
         deformationGradient[*iter] = Identity;
         pstress[*iter] = zero;
   }
   new_dw->put(cmdata, p_cmdata_label, matl->getDWIndex(), region);
   new_dw->put(deformationGradient, pDeformationMeasureLabel, matl->getDWIndex(), region);
   new_dw->put(pstress, pStressLabel, matl->getDWIndex(), region);

   computeStableTimestep(region, matl, new_dw);
}

void CompMooneyRivlin::computeStableTimestep(const Region* region,
					     const MPMMaterial* matl,
					     DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = region->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, matlindex, region, 0);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, pMassLabel, matlindex, region, 0);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, pVolumeLabel, matlindex, region, 0);

  ParticleSubset* pset = pmass.getParticleSubset();
  ASSERT(pset == pvolume.getParticleSubset());

  double c_dil = 0.0,c_rot = 0.0;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     double C1 = cmdata[idx].C1;
     double C2 = cmdata[idx].C2;
     double C4 = cmdata[idx].C4;

     // Compute wave speed at each particle, store the maximum
     double mu = 2.*(C1 + C2);
     double PR = (2.*C1 + 5.*C2 + 2.*C4) / (4.*C4 + 5.*C1 + 11.*C2);
     double lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;
     c_dil = Max(c_dil,(lambda + 2.*mu)*pvolume[idx]/pmass[idx]);
     c_rot = Max(c_rot, mu*pvolume[idx]/pmass[idx]);
    }
    double WaveSpeed = sqrt(Max(c_rot,c_dil));
    double delt_new = Min(dx.x(), dx.y(), dx.z())/WaveSpeed;
    new_dw->put(delt_vartype(delt_new), deltLabel);
}

void CompMooneyRivlin::computeStressTensor(const Region* region,
                                           const MPMMaterial* matl,
                                           const DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw)
{
  Matrix3 Identity,deformationGradientInc,B,velGrad;
  double invar1,invar3,J,w1,w2,w3,i3w3,w1pi1w2;
  Identity.Identity();
  double WaveSpeed = 0.0,c_dil = 0.0,c_rot = 0.0;

  Vector dx = region->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  int matlindex = matl->getDWIndex();

  // Create array for the particle position
  ParticleVariable<Point> px;
  old_dw->get(px, pXLabel, matlindex, region, 0);
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, pDeformationMeasureLabel, matlindex, region, 0);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  new_dw->allocate(pstress, pStressLabel, matlindex, region);

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, p_cmdata_label, matlindex, region, 0);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, pMassLabel, matlindex, region, 0);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, pVolumeLabel, matlindex, region, 0);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, gInterpVelocityLabel, matlindex,region, 0);
  delt_vartype delt;
  old_dw->get(delt, deltLabel);

  ParticleSubset* pset = px.getParticleSubset();
  ASSERT(pset == pstress.getParticleSubset());
  ASSERT(pset == deformationGradient.getParticleSubset());
  ASSERT(pset == pmass.getParticleSubset());
  ASSERT(pset == pvolume.getParticleSubset());

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     IntVector ni[8];
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
      deformationGradient[idx] = deformationGradientInc * deformationGradient[idx];

      // Actually calculate the stress from the n+1 deformation gradient.

      // Compute the left Cauchy-Green deformation tensor
      B = deformationGradient[idx] * deformationGradient[idx].Transpose();

      // Compute the invariants
      invar1 = B.Trace();
      J = deformationGradient[idx].Determinant();
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

      pstress[idx]=(B*w1pi1w2 - (B*B)*w2 + Identity*i3w3)*2.0/J;

      // Compute wave speed at each particle, store the maximum
      double mu = 2.*(C1 + C2);
      double PR = (2.*C1 + 5.*C2 + 2.*C4)/
		(4.*C4 + 5.*C1 + 11.*C2);
      double lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;
      c_dil = Max(c_dil,(lambda + 2.*mu)*pvolume[idx]/pmass[idx]);
      c_rot = Max(c_rot, mu*pvolume[idx]/pmass[idx]);
    }
    WaveSpeed = sqrt(Max(c_rot,c_dil));
    double delt_new = Min(dx.x(), dx.y(), dx.z())/WaveSpeed;
    new_dw->put(delt_vartype(delt_new), deltLabel);
    new_dw->put(pstress, pStressLabel, matlindex, region);
    new_dw->put(deformationGradient, pDeformationMeasureLabel,
		matlindex, region);
}

double CompMooneyRivlin::computeStrainEnergy(const Region* /*region*/,
                                             const MPMMaterial* /*matl*/,
                                             const DataWarehouseP& /*new_dw*/)
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
     particleIndex idx = *iter;

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

// $Log$
// Revision 1.17  2000/05/02 17:54:24  sparker
// Implemented more of SerialMPM
//
// Revision 1.16  2000/05/02 06:07:11  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.15  2000/05/01 17:25:00  jas
// Changed the var labels to be consistent with SerialMPM.
//
// Revision 1.14  2000/05/01 17:10:27  jas
// Added allocations for mass and volume.
//
// Revision 1.13  2000/04/28 07:35:27  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.12  2000/04/27 23:18:43  sparker
// Added problem initialization for MPM
//
// Revision 1.11  2000/04/26 06:48:14  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/25 18:42:33  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.9  2000/04/21 01:22:55  guilkey
// Put the VarLabels which are common to all constitutive models in the
// base class.  The only one which isn't common is the one for the CMData.
//
// Revision 1.8  2000/04/20 18:56:18  sparker
// Updates to MPM
//
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
