
#include "CompMooneyRivlin.h"
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <values.h>
#include <iostream>
using std::cerr;
using namespace Uintah::MPM;
using SCICore::Math::Min;
using SCICore::Math::Max;
using SCICore::Geometry::Vector;

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

CompMooneyRivlin::CompMooneyRivlin(ProblemSpecP& ps)
{
  ps->require("he_constant_1",d_initialData.C1);
  ps->require("he_constant_2",d_initialData.C2);
  ps->require("he_PR",d_initialData.PR);
  p_cmdata_label = scinew VarLabel("p.cmdata",
			ParticleVariable<CMData>::getTypeDescription());
  p_cmdata_label_preReloc = scinew VarLabel("p.cmdata+",
			ParticleVariable<CMData>::getTypeDescription());

}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
  delete p_cmdata_label;
  delete p_cmdata_label_preReloc;
  
}

void CompMooneyRivlin::initializeCMData(const Patch* patch,
					const MPMMaterial* matl,
					DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();
   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<CMData> cmdata;
   new_dw->allocate(cmdata, p_cmdata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
	 cmdata[*iter] = d_initialData;
         deformationGradient[*iter] = Identity;
         pstress[*iter] = zero;
   }
   new_dw->put(cmdata, p_cmdata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);

   computeStableTimestep(patch, matl, new_dw);
}

void CompMooneyRivlin::computeStableTimestep(const Patch* patch,
					     const MPMMaterial* matl,
					     DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     double C1 = cmdata[idx].C1;
     double C2 = cmdata[idx].C2;
     double PR = cmdata[idx].PR;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     double mu = 2.*(C1 + C2);
     double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);
     c_dil = sqrt(2.*mu*(1.- PR)*pvolume[idx]/((1.-2.*PR)*pmass[idx]));
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void CompMooneyRivlin::computeStressTensor(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw)
{
  Matrix3 Identity,deformationGradientInc,B,velGrad;
  double invar1,invar2,invar3,J,w1,w2,w3,i3w3,w1pi1w2;
  Identity.Identity();
  double c_dil = 0.0,se=0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

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
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  old_dw->get(pvelocity, lb->pVelocityLabel, pset);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
	      Ghost::AroundCells, 1);
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];
     if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S)){
	cerr << "p=" << px[idx] << '\n';
	cerr << "patch=" << patch << '\n';
	throw InternalError("Particle not in this patch?");
     }

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
      deformationGradientInc = velGrad * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient[idx] = deformationGradientInc * deformationGradient[idx];

      // Actually calculate the stress from the n+1 deformation gradient.

      // Compute the left Cauchy-Green deformation tensor
      B = deformationGradient[idx] * deformationGradient[idx].Transpose();

      // Compute the invariants
      invar1 = B.Trace();
      invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
      J = deformationGradient[idx].Determinant();
      invar3 = J*J;

      double C1 = cmdata[idx].C1;
      double C2 = cmdata[idx].C2;
      double C3 = .5*C1 + C2;
      double PR = cmdata[idx].PR;
      double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

      w1 = C1;
      w2 = C2;
      w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);

      // Compute T = 2/sqrt(I3)*(I3*W3*Identity + (W1+I1*W2)*B - W2*B^2)
      w1pi1w2 = w1 + invar1*w2;
      i3w3 = invar3*w3;

      pstress[idx]=(B*w1pi1w2 - (B*B)*w2 + Identity*i3w3)*2.0/J;

      // Compute wave speed + particle velocity at each particle, 
      // store the maximum
      c_dil = sqrt((4.*(C1+C2*invar2)/J
		    +8.*(2.*C3/(invar3*invar3*invar3)+C4*(2.*invar3-1.))
		    -Min((pstress[idx])(1,1),(pstress[idx])(2,2)
			 ,(pstress[idx])(3,3))/J)
		   *pvolume[idx]/pmass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute the strain energy for all the particles
      se += (C1*(invar1-3.0) + C2*(invar2-3.0) +
            C3*(1.0/(invar3*invar3) - 1.0) +
            C4*(invar3-1.0)*(invar3-1.0))*pvolume[idx]/J;
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(pstress, lb->pStressLabel_preReloc);
    new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);

    new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

    // This is just carried forward.
    new_dw->put(cmdata, p_cmdata_label_preReloc);
    // Volume is currently just carried forward, but will be updated.
    new_dw->put(pvolume, lb->pVolumeDeformedLabel);
}

void CompMooneyRivlin::addParticleState(std::vector<const VarLabel*>& from,
					std::vector<const VarLabel*>& to)
{
   from.push_back(p_cmdata_label);
   to.push_back(p_cmdata_label_preReloc);
}

void CompMooneyRivlin::addComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw) const
{
  //  const MPMLabel* lb = MPMLabel::getLabels();
   task->requires(old_dw, lb->pXLabel, matl->getDWIndex(), patch,
		  Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matl->getDWIndex(), patch,
		  Ghost::None);
   task->requires(old_dw, p_cmdata_label, matl->getDWIndex(),  patch,
		  Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),  patch,
		  Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matl->getDWIndex(),  patch,
		  Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matl->getDWIndex(), patch,
		  Ghost::AroundCells, 1);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressLabel_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, matl->getDWIndex(), patch);
   task->computes(new_dw, p_cmdata_label_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(), patch);
}

double CompMooneyRivlin::computeStrainEnergy(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouseP& new_dw)
{
  double se=0.0;
  return se;

}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace MPM {
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompMooneyRivlin::CMData), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(CompMooneyRivlin::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other, "CompMooneyRivlin::CMData", true, &makeMPI_CMData);
   }
   return td;   
}
   }
}

// $Log$
// Revision 1.53  2000/08/21 19:01:36  guilkey
// Removed some garbage from the constitutive models.
//
// Revision 1.52  2000/08/14 22:38:10  bard
// Corrected strain energy calculation.
//
// Revision 1.51  2000/08/08 01:32:42  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.50  2000/07/27 22:39:44  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.49  2000/07/07 23:52:08  guilkey
// Removed some inefficiences in the way the deformed volume was allocated
// and stored, and also added changing particle volume to CompNeoHookPlas.
//
// Revision 1.48  2000/07/05 23:43:33  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.47  2000/06/23 22:11:07  guilkey
// Added hack to the wavespeed to avoid floating point exception in case of no particles
// on a patch.
//
// Revision 1.46  2000/06/21 00:35:16  bard
// Added timestep control.  Changed constitutive constant number (only 3 are
// independent) and format.
//
// Revision 1.45  2000/06/19 21:22:33  bard
// Moved computes for reduction variables outside of loops over materials.
//
// Revision 1.44  2000/06/16 23:23:39  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.43  2000/06/16 05:03:03  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.42  2000/06/15 21:57:03  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.41  2000/06/09 23:52:36  bard
// Added fudge factors to time step calculations.
//
// Revision 1.40  2000/06/09 21:02:39  jas
// Added code to get the fudge factor directly into the constitutive model
// inititialization.
//
// Revision 1.39  2000/06/08 16:50:51  guilkey
// Changed some of the dependencies to account for what goes on in
// the burn models.
//
// Revision 1.38  2000/06/01 23:12:06  guilkey
// Code to store integrated quantities in the DW and save them in
// an archive of sorts.  Also added the "computes" in the right tasks.
//
// Revision 1.37  2000/05/31 22:37:09  guilkey
// Put computation of strain energy inside the computeStressTensor functions,
// and store it in a reduction variable in the datawarehouse.
//
// Revision 1.36  2000/05/30 21:07:02  dav
// delt to delT
//
// Revision 1.35  2000/05/30 20:19:01  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.34  2000/05/30 17:08:26  dav
// Changed delt to delT
//
// Revision 1.33  2000/05/26 21:37:33  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.32  2000/05/26 18:15:11  guilkey
// Brought the CompNeoHook constitutive model up to functionality
// with the UCF.  Also, cleaned up all of the working models to
// rid them of the SAMRAI crap.
//
// Revision 1.31  2000/05/20 08:09:06  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.30  2000/05/18 19:45:57  guilkey
// Fixed (really this time) the statements inside the ASSERTS.
//
// Revision 1.29  2000/05/18 19:32:05  guilkey
// Fixed an error inside of an ASSERT statement that I wasn't getting
// at compile time.
//
// Revision 1.28  2000/05/18 17:03:21  guilkey
// Fixed computeStrainEnergy.
//
// Revision 1.27  2000/05/18 16:06:24  guilkey
// Implemented computeStrainEnergy for CompNeoHookPlas.  In both working
// constitutive models, moved the carry forward of the particle volume to
// computeStressTensor.  This "carry forward" will be replaced by a real
// update eventually.  Removed the carry forward in the SerialMPM and
// then replaced where the particle volume was being required from the old_dw
// with requires from the new_dw.  Don't update these files until I've
// checked in a new SerialMPM.cc, which should be in a few minutes.
//
// Revision 1.26  2000/05/11 20:10:13  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.25  2000/05/10 20:02:45  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.24  2000/05/07 06:02:02  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.23  2000/05/04 16:37:30  guilkey
// Got the CompNeoHookPlas constitutive model up to speed.  It seems
// to work but hasn't had a rigorous test yet.
//
// Revision 1.22  2000/05/03 20:35:20  guilkey
// Added fudge factor to the other place where delt is calculated.
//
// Revision 1.21  2000/05/02 22:57:50  guilkey
// Added fudge factor to timestep calculation
//
// Revision 1.20  2000/05/02 20:13:00  sparker
// Implemented findCellAndWeights
//
// Revision 1.19  2000/05/02 19:31:23  guilkey
// Added a put for cmdata.
//
// Revision 1.18  2000/05/02 18:41:16  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
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
// class to operate on all particles in a patch of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
