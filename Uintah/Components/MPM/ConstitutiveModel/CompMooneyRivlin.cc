
#include "CompMooneyRivlin.h"
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/NodeIterator.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <values.h>
#include <iostream>

#include <Uintah/Components/MPM/Fracture/Visibility.h>

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
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
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
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
         deformationGradient[*iter] = Identity;
         pstress[*iter] = zero;
   }
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
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double C1 = d_initialData.C1;
  double C2 = d_initialData.C2;
  double PR = d_initialData.PR;

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     double mu = 2.*(C1 + C2);
     //double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);
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

  ParticleVariable<int> pVisibility;
  ParticleVariable<Vector> pRotationRate;
  ParticleVariable<double> pStrainEnergy;
  if(matl->getFractureModel()) {
    new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
    new_dw->allocate(pRotationRate, lb->pRotationRateLabel, pset);
    new_dw->allocate(pStrainEnergy, lb->pStrainEnergyLabel, pset);
  }

  double C1 = d_initialData.C1;
  double C2 = d_initialData.C2;
  double C3 = .5*C1 + C2;
  double PR = d_initialData.PR;
  double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];

     patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
     
     Visibility vis;
     if(matl->getFractureModel()) {
  	vis = pVisibility[idx];
      	vis.modifyShapeDerivatives(d_S);
     }

     //ratation rate: (omega1,omega2,omega3)
     double omega1 = 0;
     double omega2 = 0;
     double omega3 = 0;
     
     for(int k = 0; k < 8; k++) {
	 if(vis.visible(k)) {	    
	    Vector& gvel = gvelocity[ni[k]];
	    for (int j = 0; j<3; j++){
	       for (int i = 0; i<3; i++) {
	          velGrad(i+1,j+1) += gvel(i) * d_S[k](j) * oodx[j];		  
	       }
	    }

	    //rotation rate computation, required for fracture
            if(matl->getFractureModel()) {
	      //NOTE!!! gvel(0) = gvel.x() !!!
	      omega1 += gvel(2) * d_S[k](1) * oodx[1] - gvel(1) * d_S[k](2) * oodx[2];
	      omega2 += gvel(0) * d_S[k](2) * oodx[2] - gvel(2) * d_S[k](0) * oodx[0];
	      omega3 += gvel(1) * d_S[k](0) * oodx[0] - gvel(0) * d_S[k](1) * oodx[1];
	    }
	    
	 }
     }

     if( matl->getFractureModel() ) {
        pRotationRate[idx] = Vector(omega1/2,omega2/2,omega3/2);
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
      double e = (C1*(invar1-3.0) + C2*(invar2-3.0) +
            C3*(1.0/(invar3*invar3) - 1.0) +
            C4*(invar3-1.0)*(invar3-1.0))*pvolume[idx]/J;

      if(matl->getFractureModel()) pStrainEnergy[idx] = e;
      
      se += e;
    }
        
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    
    if(delT_new < 1.e-12) delT_new = MAXDOUBLE;
    new_dw->put(delt_vartype(delT_new), lb->delTAfterConstitutiveModelLabel);    
    new_dw->put(pstress, lb->pStressAfterStrainRateLabel);
    new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);

    //
    if( matl->getFractureModel() ) {
      new_dw->put(pRotationRate, lb->pRotationRateLabel);
      new_dw->put(pStrainEnergy, lb->pStrainEnergyLabel);
    }

    new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

    // Volume is currently just carried forward, but will be updated.
    new_dw->put(pvolume, lb->pVolumeDeformedLabel);
}

void CompMooneyRivlin::addParticleState(std::vector<const VarLabel*>& from,
					std::vector<const VarLabel*>& to)
{
}

void CompMooneyRivlin::addComputesAndRequires(Task* task,
					      const MPMMaterial* matl,
					      const Patch* patch,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw) const
{
   int idx = matl->getDWIndex();
   
   task->requires(old_dw, lb->pXLabel, idx, patch, Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, idx, patch, Ghost::None);
   task->requires(old_dw, lb->pMassLabel,   idx,  patch, Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, idx,  patch, Ghost::None);
   task->requires(old_dw, lb->pStressLabel, idx,  patch, Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, idx, patch,
						Ghost::AroundCells, 1);
   task->requires(old_dw, lb->delTLabel);

   task->computes(new_dw, lb->pStressAfterStrainRateLabel,       idx, patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, idx, patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel,              idx, patch);
   
   if(matl->getFractureModel()) {
      task->requires(new_dw, lb->pVisibilityLabel,   idx, patch, Ghost::None);
      task->computes(new_dw, lb->pRotationRateLabel, idx, patch);
      task->computes(new_dw, lb->pStrainEnergyLabel, idx,  patch);
   }
}

void CompMooneyRivlin::computeCrackSurfaceContactForce(const Patch* patch,
                                           const MPMMaterial* mpm_matl,
                                           DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw)
{
  Matrix3 Identity;
  Identity.Identity();

  int matlindex = mpm_matl->getDWIndex();

  // Create arrays for the particle data
  ParticleVariable<Point>  pX_patchAndGhost;
  ParticleVariable<int>    pIsBroken;
  ParticleVariable<Vector> pCrackSurfaceNormal;
  ParticleVariable<double> pVolume;

  ParticleSubset* pset_patchAndGhost = old_dw->getParticleSubset(
     matlindex, patch, Ghost::AroundCells, 1, lb->pXLabel);

  old_dw->get(pX_patchAndGhost, lb->pXLabel, pset_patchAndGhost);
  new_dw->get(pIsBroken, lb->pIsBrokenLabel_preReloc, pset_patchAndGhost);
  new_dw->get(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel_preReloc, 
    pset_patchAndGhost);
  new_dw->get(pVolume, lb->pVolumeLabel_preReloc, pset_patchAndGhost);

  Lattice lattice(pX_patchAndGhost);
  ParticlesNeighbor particles;

  ParticleSubset* pset_patchOnly = old_dw->getParticleSubset(
     matlindex, patch);

  ParticleVariable<Vector> pCrackSurfaceContactForce;
  new_dw->allocate(pCrackSurfaceContactForce,
     lb->pCrackSurfaceContactForceLabel, pset_patchOnly);

  ParticleVariable<Point>  pX_patchOnly;
  old_dw->get(pX_patchOnly, lb->pXLabel, pset_patchOnly);

  for(ParticleSubset::iterator iter = pset_patchOnly->begin();
          iter != pset_patchOnly->end(); iter++)
  {
    pCrackSurfaceContactForce[*iter] = Vector(0.,0.,0.);
  }

  IntVector cellIdx;
  double C1 = ( d_initialData.C1 + d_initialData.C1 )/2;
  double C2 = ( d_initialData.C2 + d_initialData.C2 )/2;
  double C3 = .5*C1 + C2;
  double PR = ( d_initialData.PR + d_initialData.PR )/2;
  double C4 = .5*(C1*(5.*PR-2) + C2*(11.*PR-5)) / (1. - 2.*PR);

  for(ParticleSubset::iterator iter = pset_patchOnly->begin();
          iter != pset_patchOnly->end(); iter++)
  {
    particleIndex pIdx = *iter;

    const Point& X1 = pX_patchOnly[pIdx];

    patch->findCell(X1,cellIdx);
    particles.clear();
    particles.buildIn(cellIdx,lattice);

    double size1 = pow(pVolume[pIdx],0.3333);

    //crack surface contact force
    for(int pNeighbor=0; pNeighbor<(int)particles.size(); ++pNeighbor)
    {
      particleIndex pContact = particles[pNeighbor];
      if(pContact == pIdx) continue;

      if(!particles.visible( pContact,
                            X1,
		            pX_patchAndGhost,
		            pIsBroken,
		            pCrackSurfaceNormal,
		            pVolume) ) 
      {
        const Point& X2 = pX_patchAndGhost[pContact];

        double size2 = pow(pVolume[pContact],0.3333);
        Vector N = X2-X1;
        double distance = N.length();
        double l = (size1+size2) /2 * 0.8;
        double delta = distance /l - 1;
	
        if( delta < 0 ) {
          N /= distance;
          Matrix3 deformationGradient = Identity;
	  
          for(int i=1;i<=3;++i)
	  for(int j=1;j<=3;++j)
	  deformationGradient(i,j) += N(i-1) * N(j-1) * delta;

          // Compute the left Cauchy-Green deformation tensor
          Matrix3 B = deformationGradient * deformationGradient.Transpose();

          // Compute the invariants
          double invar1 = B.Trace();
          double invar2 = 0.5*((invar1*invar1) - (B*B).Trace());
          double J = deformationGradient.Determinant();
          double invar3 = J*J;

          double w1 = C1;
          double w2 = C2;
          double w3 = -2.0*C3/(invar3*invar3*invar3) + 2.0*C4*(invar3 -1.0);

          double w1pi1w2 = w1 + invar1*w2;
          double i3w3 = invar3*w3;

          Matrix3 stress = (B*w1pi1w2 - (B*B)*w2 + Identity*i3w3)*2.0/J;
	  double area = M_PI* l * l * fabs(delta) /2;
          Vector F = stress * N * area;
          pCrackSurfaceContactForce[pIdx] += F;
	}
      }
    }
  }

  new_dw->put(pCrackSurfaceContactForce, lb->pCrackSurfaceContactForceLabel_preReloc);


  //time step requirement
  delt_vartype delT;
  new_dw->get(delT, lb->delTAfterFractureLabel);

  double delT_new = delT;

  ParticleVariable<double> pMass;
  new_dw->get(pMass, lb->pMassLabel_preReloc, pset_patchOnly);

  double tolerance = 0.001;
  
  Vector dx = patch->dCell();
  double dxLength = dx.length() * tolerance;

  for(ParticleSubset::iterator iter = pset_patchOnly->begin();
          iter != pset_patchOnly->end(); iter++)
  {
    double force = pCrackSurfaceContactForce[*iter].length();
    if(force > 0) {
      delT_new = Min(delT_new,sqrt(2*dxLength*pMass[*iter]/force));
    }
  }

  new_dw->put(delt_vartype(delT_new), lb->delTAfterCrackSurfaceContactLabel);
}

void CompMooneyRivlin::addComputesAndRequiresForCrackSurfaceContact(
	                                     Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  
  task->requires(old_dw, lb->pXLabel, idx,  patch,
			Ghost::AroundCells, 1 );
  task->requires(new_dw, lb->pVolumeLabel_preReloc, idx, patch,
			Ghost::AroundCells, 1 );
  task->requires(new_dw, lb->pIsBrokenLabel_preReloc, idx, patch,
			Ghost::AroundCells, 1 );
  task->requires(new_dw, lb->pCrackSurfaceNormalLabel_preReloc, idx, patch,
			Ghost::AroundCells, 1 );
  task->requires(new_dw, lb->delTAfterFractureLabel );
  task->requires(new_dw, lb->pMassLabel_preReloc, idx, patch, Ghost::None);
		  
  task->computes(new_dw, lb->pCrackSurfaceContactForceLabel_preReloc, idx, patch );
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace MPM {
#if 0
static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompMooneyRivlin::CMData), sizeof(double)*3);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 3, 3, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
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
#endif
   }
}

// $Log$
// Revision 1.77  2001/01/04 00:18:04  jas
// Remove g++ warnings.
//
// Revision 1.76  2000/12/30 05:08:09  tan
// Fixed a problem concerning patch and ghost in fracture computations.
//
// Revision 1.75  2000/12/10 06:42:29  tan
// Modifications on fracture contact computations.
//
// Revision 1.74  2000/11/30 22:59:19  guilkey
// Got rid of the if(! in front of all of the patch->findCellAnd...
// since this is no longer needed.
//
// Revision 1.73  2000/11/21 20:51:04  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
// Revision 1.72  2000/11/15 18:37:23  guilkey
// Reduced warnings in constitutive models.
//
// Revision 1.71  2000/10/13 19:49:19  sparker
// Fixed a few warnings
//
// Revision 1.70  2000/10/11 01:30:28  guilkey
// Made CMData no longer a per particle variable for these models.
// None of them currently have anything worthy of being called StateData,
// so no such struct was created.
//
// Revision 1.69  2000/09/26 17:08:35  sparker
// Need to commit MPI types
//
// Revision 1.68  2000/09/25 20:23:19  sparker
// Quiet g++ warnings
//
// Revision 1.67  2000/09/22 07:10:57  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.66  2000/09/16 04:18:29  tan
// Modifications to make fracture works well.
//
// Revision 1.65  2000/09/12 17:18:21  tan
// Modified ParticlesNeighbor initialize.
//
// Revision 1.64  2000/09/12 16:52:10  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.63  2000/09/11 20:23:25  tan
// Fixed a mistake in crack surface contact force algorithm.
//
// Revision 1.62  2000/09/11 01:08:43  tan
// Modified time step calculation (in constitutive model computeStressTensor(...))
// when fracture cracking speed involved.
//
// Revision 1.61  2000/09/10 22:51:12  tan
// Added particle rotationRate computation in computeStressTensor functions
// in each constitutive model classes.  The particle rotationRate will be used
// for fracture.
//
// Revision 1.60  2000/09/09 20:23:20  tan
// Replace BrokenCellShapeFunction with particle visibility information in shape
// function computation.
//
// Revision 1.59  2000/09/08 18:23:19  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.58  2000/09/08 01:45:28  tan
// Added pDilationalWaveSpeedLabel for fracture and is saved as a
// side-effect of computeStressTensor in each constitutive model class.
//
// Revision 1.57  2000/09/07 21:17:49  tan
// Removed a debugging output.
//
// Revision 1.56  2000/09/07 21:11:09  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.55  2000/09/06 19:45:09  jas
// Changed new to scinew in constitutive models related to crack stuff.
//
// Revision 1.54  2000/09/05 07:45:49  tan
// Applied BrokenCellShapeFunction to constitutive models where fracture
// is involved.
//
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
