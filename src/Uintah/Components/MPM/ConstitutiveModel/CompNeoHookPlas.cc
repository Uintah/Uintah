
#include "CompNeoHookPlas.h"
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

CompNeoHookPlas::CompNeoHookPlas(ProblemSpecP& ps)
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->require("yield_stress",d_initialData.FlowStress);
  ps->require("hardening_modulus",d_initialData.K);
  ps->require("alpha",d_initialData.Alpha);

  p_cmdata_label = new VarLabel("p.cmdata",
                                ParticleVariable<CMData>::getTypeDescription());
 
  bElBarLabel = new VarLabel("p.bElBar",
		ParticleVariable<Point>::getTypeDescription(),
                                VarLabel::PositionVariable);
}

CompNeoHookPlas::~CompNeoHookPlas()
{
  // Destructor 
}

void CompNeoHookPlas::initializeCMData(const Region* region,
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
   new_dw->allocate(deformationGradient, 
		pDeformationMeasureLabel, matl->getDWIndex(), region);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, pStressLabel, matl->getDWIndex(), region);
   ParticleVariable<Matrix3> bElBar;
   new_dw->allocate(bElBar,  bElBarLabel, matl->getDWIndex(), region);

   ParticleSubset* pset = cmdata.getParticleSubset();
   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
	    cmdata[*iter] = d_initialData;
          deformationGradient[*iter] = Identity;
          bElBar[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(cmdata, p_cmdata_label, matl->getDWIndex(), region);
   new_dw->put(deformationGradient, pDeformationMeasureLabel,
				 matl->getDWIndex(), region);
   new_dw->put(pstress, pStressLabel, matl->getDWIndex(), region);
   new_dw->put(bElBar, bElBarLabel, matl->getDWIndex(), region);

   computeStableTimestep(region, matl, new_dw);

}

void CompNeoHookPlas::computeStableTimestep(const Region* region,
					     const MPMMaterial* matl,
					     DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = region->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, matlindex, region, Ghost::None);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, pMassLabel, matlindex, region, Ghost::None);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, pVolumeLabel, matlindex, region, Ghost::None);

  ParticleSubset* pset = pmass.getParticleSubset();
  ASSERT(pset == pvolume.getParticleSubset());

  double c_dil = 0.0,c_rot = 0.0;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     double mu = cmdata[idx].Shear;
     double lambda = cmdata[idx].Bulk -.6666666667*cmdata[idx].Shear;
     c_dil = Max(c_dil,(lambda + 2.*mu)*pvolume[idx]/pmass[idx]);
     c_rot = Max(c_rot, mu*pvolume[idx]/pmass[idx]);
    }
    double WaveSpeed = sqrt(Max(c_rot,c_dil));
    // Fudge factor of .8 added, just in case
    double delt_new = .8*(Min(dx.x(), dx.y(), dx.z())/WaveSpeed);
    new_dw->put(delt_vartype(delt_new), deltLabel);
}

void CompNeoHookPlas::computeStressTensor(const Region* region,
					  const MPMMaterial* matl,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{

  Matrix3 bElBarTrial,deformationGradientInc;
  Matrix3 shearTrial,Shear,normal;
  Matrix3 fbar,velGrad;
  double J,p,fTrial,IEl,muBar,delgamma,sTnorm;
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  double WaveSpeed = 0.0,c_dil = 0.0,c_rot = 0.0;

  Identity.Identity();

  Vector dx = region->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  int matlindex = matl->getDWIndex();

  // Create array for the particle position
  ParticleVariable<Point> px;
  old_dw->get(px, pXLabel, matlindex, region, Ghost::None);
  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, pDeformationMeasureLabel, 
	      matlindex, region, Ghost::None);
  ParticleVariable<Matrix3> bElBar;
  old_dw->get(bElBar, bElBarLabel, matlindex, region, Ghost::None);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  new_dw->allocate(pstress, pStressLabel, matlindex, region);

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, p_cmdata_label, matlindex, region, Ghost::None);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, pMassLabel, matlindex, region, Ghost::None);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, pVolumeLabel, matlindex, region, Ghost::None);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, gMomExedVelocityLabel, matlindex,region,
	      Ghost::None);
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

    // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
    // time step and the velocity gradient and the material constants
    double shear = cmdata[idx].Shear;
    double bulk  = cmdata[idx].Bulk;
    double flow  = cmdata[idx].FlowStress;
    double K     = cmdata[idx].K;
    double alpha = cmdata[idx].Alpha;

    // Compute the deformation gradient increment using the time_step
    // velocity gradient
    // F_n^np1 = dudx * dt + Identity
    deformationGradientInc = velGrad * delt + Identity;

    // Update the deformation gradient tensor to its time n+1 value.
    deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

    // get the volume preserving part of the deformation gradient increment
    fbar = deformationGradientInc *
			pow(deformationGradientInc.Determinant(),-onethird);

    // predict the elastic part of the volume preserving part of the left
    // Cauchy-Green deformation tensor
    bElBarTrial = fbar*bElBar[idx]*fbar.Transpose();

    // shearTrial is equal to the shear modulus times dev(bElBar)
    shearTrial = (bElBarTrial - Identity*onethird*bElBarTrial.Trace())*shear;

    // get the volumetric part of the deformation
    J = deformationGradient[idx].Determinant();

    // get the hydrostatic part of the stress
    p = 0.5*bulk*(J - 1.0/J);

    // Compute ||shearTrial||
    sTnorm = shearTrial.Norm();

    // Check for plastic loading
    fTrial = sTnorm - sqtwthds*(K*alpha + flow);

    if(fTrial > 0.0){
	// plastic

	IEl = onethird*bElBarTrial.Trace();

	muBar = IEl * shear;

	delgamma = (fTrial/(2.0*muBar)) / (1.0 + (K/(3.0*muBar)));

	normal = shearTrial/sTnorm;

        // The actual elastic shear stress
	Shear = shearTrial - normal*2.0*muBar*delgamma;

        // Deal with history variables
      cmdata[idx].Alpha = alpha + sqtwthds*delgamma;
      bElBar[idx] = Shear/shear + Identity*IEl;
    }
    else {
	// not plastic

      bElBar[idx] = bElBarTrial;
	Shear = shearTrial;
    }

    // compute the total stress (volumetric + deviatoric)
    pstress[idx] = Identity*J*p + Shear;

    // Compute wave speed at each particle, store the maximum
    double mu = cmdata[idx].Shear;
    double lambda = cmdata[idx].Bulk -.6666666667*cmdata[idx].Shear;

    c_dil = Max(c_dil,(lambda + 2.*mu)*pvolume[idx]/pmass[idx]);
    c_rot = Max(c_rot, mu*pvolume[idx]/pmass[idx]);
  }
  WaveSpeed = sqrt(Max(c_rot,c_dil));
  // Fudge factor of .8 added, just in case
  double delt_new = .8*Min(dx.x(), dx.y(), dx.z())/WaveSpeed;
  new_dw->put(delt_vartype(delt_new), deltLabel);
  new_dw->put(pstress, pStressLabel, matlindex, region);
  new_dw->put(deformationGradient, pDeformationMeasureLabel,
		matlindex, region);
  new_dw->put(bElBar, bElBarLabel, matlindex, region);

  // This is just carried forward with the updated alpha
  new_dw->put(cmdata, p_cmdata_label, matlindex, region);

}

double CompNeoHookPlas::computeStrainEnergy(const Region* region,
					    const MPMMaterial* matl,
					    const DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  double se,J,U,W;

  J = deformationGradient.Determinant();
  U = .5*d_Bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
  W = .5*d_Shear*(bElBar.Trace() - 3.0);

  se = U + W;

  return se;
#endif
  return 0;
}

void CompNeoHookPlas::addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Region* region,
					     const DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const
{
   cerr << "CompNeoHookPlas::addComputesAndRequires needs to be filled in\n";
}

void CompNeoHookPlas::readParameters(ProblemSpecP ps, double *p_array)
{
  ps->require("bulk_modulus",p_array[0]);
  ps->require("shear_modulus",p_array[1]);
  ps->require("yield_stress",p_array[2]);
  ps->require("hardening_modulus",p_array[3]);
  ps->require("alpha",p_array[4]);

}

#ifdef WONT_COMPILE_YET
ConstitutiveModel* CompNeoHookPlas::readParametersAndCreate(ProblemSpecP ps)
{
  double p_array[4];
  readParameters(ps, p_array);
  return(create(p_array));
}
   
void CompNeoHookPlas::writeRestartParameters(ofstream& out) const
{
  out << getType() << " ";
  out << d_Bulk << " " << d_Shear << " "
      << d_FlowStress << " " << d_K << " " << d_Alpha << " ";
  out << (getDeformationMeasure())(1,1) << " "
      << (getDeformationMeasure())(1,2) << " "
      << (getDeformationMeasure())(1,3) << " "
      << (getDeformationMeasure())(2,1) << " "
      << (getDeformationMeasure())(2,2) << " "
      << (getDeformationMeasure())(2,3) << " "
      << (getDeformationMeasure())(3,1) << " "
      << (getDeformationMeasure())(3,2) << " "
      << (getDeformationMeasure())(3,3) << endl;
}

ConstitutiveModel* CompNeoHookPlas::readRestartParametersAndCreate
							(ProblemSpecP ps)
{
#if 0
  Matrix3 dg(0.0);
  double p_array[5];
  
  readParameters(in, p_array);
  in >> p_array[4];
  
  ConstitutiveModel *cm = new CompNeoHookPlas(p_array[0], p_array[1], 
p_array[2],
					      p_array[3], p_array[4]);

  in >> dg(1,1) >> dg(1,2) >> dg(1,3)
     >> dg(2,1) >> dg(2,2) >> dg(2,3)
     >> dg(3,1) >> dg(3,2) >> dg(3,3);
  cm->setDeformationMeasure(dg);
  
  return(cm);
#endif
}
#endif

// $Log$
// Revision 1.9  2000/05/10 20:02:46  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.8  2000/05/07 06:02:03  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.7  2000/05/04 16:37:30  guilkey
// Got the CompNeoHookPlas constitutive model up to speed.  It seems
// to work but hasn't had a rigorous test yet.
//
// Revision 1.6  2000/04/26 06:48:15  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/25 18:42:34  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.4  2000/04/19 21:15:55  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.3  2000/04/14 17:34:42  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.2  2000/03/20 17:17:07  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:48  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:54  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:48  sparker
// Stuff may actually work someday...
//

