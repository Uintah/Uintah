
#include "FrictionContact.h"
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::string;

using namespace std;

FrictionContact::FrictionContact(ProblemSpecP& ps,
				 SimulationStateP& d_sS, MPMLabel* Mlb)
{
  // Constructor
  IntVector v_f;

  ps->require("vel_fields",v_f);
  ps->require("mu",d_mu);

  d_sharedState = d_sS;
  lb = Mlb;
}

FrictionContact::~FrictionContact()
{
  // Destructor
}

void FrictionContact::initializeContact(const Patch* patch,
                                        int matlindex,
                                        DataWarehouse* new_dw)
{
  NCVariable<double> normtraction;
  NCVariable<Vector> surfnorm;

  new_dw->allocate(normtraction,lb->gNormTractionLabel, matlindex, patch);
  new_dw->allocate(surfnorm,    lb->gSurfNormLabel,     matlindex, patch);

  normtraction.initialize(0.0);
  surfnorm.initialize(Vector(0.0,0.0,0.0));

  new_dw->put(normtraction,lb->gNormTractionLabel,matlindex, patch);
  new_dw->put(surfnorm,    lb->gSurfNormLabel,    matlindex, patch);

}

void FrictionContact::exMomInterpolated(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    // Need access to all velocity fields at once
    StaticArray<NCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity(numMatls);
    StaticArray<NCVariable<double> > normtraction(numMatls);
    StaticArray<NCVariable<Vector> > surfnorm(numMatls);
    StaticArray<NCVariable<double> > frictionalWork(numMatls);
  
    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
        new_dw->get(gmass[m],       lb->gMassLabel,         dwindex, patch,
		  Ghost::None, 0);
        new_dw->get(gvelocity[m],   lb->gVelocityLabel,     dwindex, patch,
		  Ghost::None, 0);
        old_dw->get(normtraction[m],lb->gNormTractionLabel, dwindex, patch,
		Ghost::None, 0);
        old_dw->get(surfnorm[m],    lb->gSurfNormLabel,     dwindex, patch,
		  Ghost::None, 0);
        new_dw->allocate(frictionalWork[m], lb->frictionalWorkLabel,
							    dwindex, patch);
	frictionalWork[m].initialize(0.);
    }
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      for(int n = 0; n < numMatls; n++){
        centerOfMassMom+=gvelocity[n][*iter] * gmass[n][*iter];
        centerOfMassMass+=gmass[n][*iter]; 
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
        Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

        // Loop over velocity fields.  Only proceed if velocity field mass
        // is nonzero (not numerical noise) and the difference from
        // the centerOfMassVelocity is nonzero (More than one velocity
        // field is contributing to grid vertex).
        for(int n = 0; n < numMatls; n++){
          Vector deltaVelocity=gvelocity[n][*iter]-centerOfMassVelocity;
          if(!compare(gmass[n][*iter]/centerOfMassMass,0.0)
	     //           && !compare(deltaVelocity.length(),0.0)){
             && !compare(gmass[n][*iter]-centerOfMassMass,0.0)){

            // Apply frictional contact if the surface is in compression
            // or the surface is stress free and surface is approaching.
            // Otherwise apply free surface conditions (do nothing).
            double normalDeltaVel=Dot(deltaVelocity,surfnorm[n][*iter]);
	    Vector Dvdt(0.,0.,0.);
            if((normtraction[n][*iter] < 0.0) ||
               (compare(fabs(normtraction[n][*iter]),0.0) &&
                normalDeltaVel>0.0)){

                // Specialize algorithm in case where approach velocity
                // is in direction of surface normal.
                if(compare( (deltaVelocity
                        -surfnorm[n][*iter]*normalDeltaVel).length(),0.0)){
                  Dvdt=-surfnorm[n][*iter]*normalDeltaVel;
		  frictionalWork[n][*iter] = 0.;
                }
	        else if(!compare(fabs(normalDeltaVel),0.0)){
                  Vector surfaceTangent=
		  (deltaVelocity-surfnorm[n][*iter]*normalDeltaVel)/
                  (deltaVelocity-surfnorm[n][*iter]*normalDeltaVel).length();
                  double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                  double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));
                  Dvdt=
                    -surfnorm[n][*iter]*normalDeltaVel
		    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

		  frictionalWork[n][*iter] = gmass[n][*iter]*
					frictionCoefficient*normalDeltaVel*
		    (tangentDeltaVelocity - normalDeltaVel*frictionCoefficient);

		  frictionalWork[n][*iter] = gmass[n][*iter]*frictionCoefficient
					  * (normalDeltaVel*normalDeltaVel) *
		                 (tangentDeltaVelocity/fabs(normalDeltaVel)-
							   frictionCoefficient);

	        }
	        Vector epsilon=(Dvdt/dx)*delT;
	        double epsilon_max=
		  Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	        if(!compare(epsilon_max,0.0)){
		  epsilon_max=epsilon_max*Max(1.0,
			    gmass[n][*iter]/(centerOfMassMass-gmass[n][*iter]));
		  double ff=Min(epsilon_max,.5)/epsilon_max;
		  Dvdt=Dvdt*ff;
	        }
	        gvelocity[n][*iter]+=Dvdt;
            }
	  }
        }
      }
    }

    // Store new velocities in DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
      new_dw->modify(gvelocity[m],   lb->gVelocityLabel,      dwindex, patch);
      new_dw->put(frictionalWork[m], lb->frictionalWorkLabel, dwindex, patch);
    }
  }
}

void FrictionContact::exMomIntegrated(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  IntVector onex(1,0,0), oney(0,1,0), onez(0,0,1);
  typedef IntVector IV;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    // This model requires getting the normal component of the
    // surface traction.  The first step is to calculate the
    // surface normals of each object.  Next, interpolate the
    // stress to the grid.  The quantity we want is n^T*stress*n
    // at each node.

    // Need access to all velocity fields at once, so store in
    // vectors of NCVariables
    StaticArray<NCVariable<double> > gmass(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gacceleration(numMatls);
    StaticArray<NCVariable<double> > normtraction(numMatls);
    StaticArray<NCVariable<Vector> > gsurfnorm(numMatls);
    StaticArray<NCVariable<double> > frictionalWork(numMatls);

    Vector surnor;

    // First, calculate the gradient of the mass everywhere
    // normalize it, and stick it in surfNorm
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],lb->gMassLabel, dwi, patch, Ghost::AroundNodes, 1);
      new_dw->allocate(gsurfnorm[m],lb->gSurfNormLabel, dwi, patch);

      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));

      IntVector low(gsurfnorm[m].getLowIndex());
      IntVector high(gsurfnorm[m].getHighIndex());

      int ILOW,IHIGH,JLOW,JHIGH,KLOW,KHIGH;
      // First, figure out some ranges for for loops
      for(Patch::FaceType face = Patch::startFace;
      		  face <= Patch::endFace; face=Patch::nextFace(face)){
	Patch::BCType bc_type = patch->getBCType(face);

	if(face==Patch::xminus){
	  if(bc_type == Patch::Neighbor) { ILOW = low.x(); }
	  else if(bc_type == Patch::None){ ILOW = low.x()+1; }
	}
	if(face==Patch::xplus){
	  if(bc_type == Patch::Neighbor) { IHIGH = high.x(); }
	  else if(bc_type == Patch::None){ IHIGH = high.x()-1; }
	}
	if(face==Patch::yminus){
	  if(bc_type == Patch::Neighbor) { JLOW = low.y(); }
	  else if(bc_type == Patch::None){ JLOW = low.y()+1; }
	}
	if(face==Patch::yplus){
	  if(bc_type == Patch::Neighbor) { JHIGH = high.y(); }
	  else if(bc_type == Patch::None){ JHIGH = high.y()-1; }
	}
	if(face==Patch::zminus){
	  if(bc_type == Patch::Neighbor) { KLOW = low.z(); }
	  else if(bc_type == Patch::None){ KLOW = low.z()+1; }
	}
	if(face==Patch::zplus){
	  if(bc_type == Patch::Neighbor) { KHIGH = high.z(); }
	  else if(bc_type == Patch::None){ KHIGH = high.z()-1; }
	}
      }

      // Compute the normals for all of the interior nodes
      for(int i = ILOW; i < IHIGH; i++){
        for(int j = JLOW; j < JHIGH; j++){
          for(int k = KLOW; k < KHIGH; k++){
	     surnor = Vector(
	        -(gmass[m][IV(i+1,j,k)] - gmass[m][IV(i-1,j,k)])/dx.x(),
         	-(gmass[m][IV(i,j+1,k)] - gmass[m][IV(i,j-1,k)])/dx.y(), 
	        -(gmass[m][IV(i,j,k+1)] - gmass[m][IV(i,j,k-1)])/dx.z()); 
	     double length = surnor.length();
	     if(length>0.0){
	    	 gsurfnorm[m][IntVector(i,j,k)] = surnor/length;;
	     }
          }
        }
      }

      // Fix the normals on the surface nodes
      for(Patch::FaceType face = Patch::startFace;
      		  face <= Patch::endFace; face=Patch::nextFace(face)){
	Patch::BCType bc_type = patch->getBCType(face);
	// First the nodes which border another patch
	if (bc_type == Patch::Neighbor){
		// Do nothing, we already got those
	}
	// Next the nodes which make up the problem domain
	else if (bc_type == Patch::None) {
	  if(face==Patch::xplus || face==Patch::xminus){
            int I;
            if(face==Patch::xminus){ I=low.x(); }
            if(face==Patch::xplus) { I=high.x()-1; }
            for (int j = JLOW; j<JHIGH; j++) {
              for (int k = KLOW; k<KHIGH; k++) {
           	surnor = Vector( 0.0,
                       -(gmass[m][IV(I,j+1,k)] - gmass[m][IV(I,j-1,k)])/dx.y(),
                       -(gmass[m][IV(I,j,k+1)] - gmass[m][IV(I,j,k-1)])/dx.z());
                double length = surnor.length();
                if(length>0.0){
                   gsurfnorm[m][IntVector(I,j,k)] = surnor/length;;
                }
              }
            }
          }

          if(face==Patch::yplus || face==Patch::yminus){
            int J;
            if(face==Patch::yminus){ J=low.y(); }
            if(face==Patch::yplus) { J=high.y()-1; }
            for (int i = ILOW; i<IHIGH; i++) {
              for (int k = KLOW; k<KHIGH; k++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,J,k)] - gmass[m][IV(i-1,J,k)])/dx.x(),
			 0.0,
		       -(gmass[m][IV(i,J,k+1)] - gmass[m][IV(i,J,k-1)])/dx.z());
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,J,k)] = surnor/length;;
		}
              }
            }
          }

          if(face==Patch::zplus || face==Patch::zminus){
            int K;
            if(face==Patch::zminus){ K=low.z(); }
            if(face==Patch::zplus) { K=high.z()-1; }
            for (int i = ILOW; i<IHIGH; i++) {
              for (int j = JLOW; j<JHIGH; j++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,j,K)] - gmass[m][IV(i-1,j,K)])/dx.x(),
	               -(gmass[m][IV(i,j+1,K)] - gmass[m][IV(i,j-1,K)])/dx.y(), 
	               0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,j,K)] = surnor/length;;
		}
              }
            }
          }

	}
      }

      new_dw->put(gsurfnorm[m],lb->gSurfNormLabel, dwi, patch);
    }

    // Next, interpolate the stress to the grid
    for(int m=0;m<matls->size();m++){
      int matlindex = matls->get(m);
      // Create arrays for the particle stress and grid stress
      ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch,
                                               Ghost::AroundNodes, 1,
                                               lb->pXLabel);
      ParticleVariable<Matrix3> pstress;
      NCVariable<Matrix3>       gstress;
      new_dw->get(pstress, lb->pStressLabel_afterStrainRate, pset);
      new_dw->allocate(gstress, lb->gStressLabel, matlindex, patch);
      gstress.initialize(Matrix3(0.0));
      
      ParticleVariable<Point> px;
      old_dw->get(px, lb->pXLabel, pset);

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
         IntVector ni[8];
	 double S[8];
         patch->findCellAndWeights(px[idx], ni, S);
         // Add each particles contribution to the local mass & velocity
         // Must use the node indices
         for(int k = 0; k < 8; k++) {
	   if (patch->containsNode(ni[k]))
	     gstress[ni[k]] += pstress[idx] * S[k];
         }
      }
      new_dw->put(gstress, lb->gStressLabel, matlindex, patch);
    }

    // Compute the normal component of the traction
    for(int m=0;m<matls->size();m++){
      int matlindex = matls->get(m);
      NCVariable<Matrix3>      gstress;
      NCVariable<double>       gnormtraction;
      NCVariable<Vector>       surfnorm;
      new_dw->get(gstress, lb->gStressLabel,  matlindex, patch, Ghost::None, 0);
      new_dw->get(surfnorm,lb->gSurfNormLabel,matlindex, patch, Ghost::None, 0);
      new_dw->allocate(gnormtraction,lb->gNormTractionLabel, matlindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	gnormtraction[*iter]=
			Dot(surfnorm[*iter]*gstress[*iter],surfnorm[*iter]);
      }

      new_dw->put(gnormtraction,lb->gNormTractionLabel, matlindex, patch);
    }


    // FINALLY, we have all the pieces in place, compute the proper interaction

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int matlindex = matls->get(m);
      new_dw->get(gmass[m],          lb->gMassLabel,matlindex,
					patch,Ghost::None, 0);
      new_dw->get(gvelocity_star[m], lb->gVelocityStarLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(gacceleration[m],  lb->gAccelerationLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(normtraction[m],   lb->gNormTractionLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(gsurfnorm[m],      lb->gSurfNormLabel,matlindex,
					patch, Ghost::None, 0);
      new_dw->get(frictionalWork[m], lb->frictionalWorkLabel,matlindex,
					patch, Ghost::None, 0);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);
    double epsilon_max_max=0.0;

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
         centerOfMassMom+=gvelocity_star[n][*iter] * gmass[n][*iter];
         centerOfMassMass+=gmass[n][*iter]; 
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
        Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

        // Loop over velocity fields.  Only proceed if velocity field mass
        // is nonzero (not numerical noise) and the difference from
        // the centerOfMassVelocity is nonzero (More than one velocity
        // field is contributing to grid vertex).
        for(int n = 0; n < numMatls; n++){
          Vector deltaVelocity=gvelocity_star[n][*iter]-centerOfMassVelocity;
          if(!compare(gmass[n][*iter]/centerOfMassMass,0.0)
	     //           && !compare(deltaVelocity.length(),0.0)){
             && !compare(gmass[n][*iter]-centerOfMassMass,0.0)){

            // Apply frictional contact if the surface is in compression
            // or the surface is stress free and surface is approaching.
            // Otherwise apply free surface conditions (do nothing).
            double normalDeltaVel=Dot(deltaVelocity,gsurfnorm[n][*iter]);
	    Vector Dvdt(0.,0.,0.);
            if((normtraction[n][*iter] < 0.0) ||
	       (compare(fabs(normtraction[n][*iter]),0.0) &&
                normalDeltaVel>0.0)){

	      // Specialize algorithm in case where approach velocity
	      // is in direction of surface normal.
	      if(compare( (deltaVelocity
		   -gsurfnorm[n][*iter]*normalDeltaVel).length(),0.0)){
	        Dvdt=-gsurfnorm[n][*iter]*normalDeltaVel;
		frictionalWork[n][*iter] += 0.;
	      }
	      else if(!compare(fabs(normalDeltaVel),0.0)){
	        Vector surfaceTangent=
	         (deltaVelocity-gsurfnorm[n][*iter]*normalDeltaVel)/
                 (deltaVelocity-gsurfnorm[n][*iter]*normalDeltaVel).length();
	        double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
	        double frictionCoefficient=
		  Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));
	        Dvdt=
		  -gsurfnorm[n][*iter]*normalDeltaVel
		  -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

		frictionalWork[n][*iter] += gmass[n][*iter]*frictionCoefficient
					  * (normalDeltaVel*normalDeltaVel) *
		(tangentDeltaVelocity/fabs(normalDeltaVel)-frictionCoefficient);

	      }
	      Vector epsilon=(Dvdt/dx)*delT;
	      double epsilon_max=
	        Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	      epsilon_max_max=max(epsilon_max,epsilon_max_max);
	      if(!compare(epsilon_max,0.0)){
	        epsilon_max=epsilon_max*Max(1.0,
			  gmass[n][*iter]/(centerOfMassMass-gmass[n][*iter]));
	        double ff=Min(epsilon_max,.5)/epsilon_max;
	        Dvdt=Dvdt*ff;
	      }
	      gvelocity_star[n][*iter]+=Dvdt;
	      Dvdt=Dvdt/delT;
	      gacceleration[n][*iter]+=Dvdt;
            }
	  }
        }
      }
    }

    //  print out epsilon_max_max
    //  static int ts=0;
    //  static ofstream tmpout("max_strain.dat");

    //  tmpout << ts << " " << epsilon_max_max << endl;
    //  ts++;

    // This converts frictional work into a temperature rate
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      double c_v = mpm_matl->getSpecificHeat();
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	frictionalWork[m][*iter] /= (c_v * gmass[m][*iter] * delT);
        if(frictionalWork[m][*iter]<0.0){
	  cout << "FW is negative: " << frictionalWork[m][*iter] << endl;
	}
      }
    }

    // Store new velocities and accelerations in DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwindex = matls->get(m);
      new_dw->modify(gvelocity_star[m], lb->gVelocityStarLabel, dwindex, patch);
      new_dw->modify(gacceleration[m],  lb->gAccelerationLabel, dwindex, patch);
      new_dw->modify(frictionalWork[m], lb->frictionalWorkLabel,dwindex, patch);
    }
  }
}

void FrictionContact::addComputesAndRequiresInterpolated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->gNormTractionLabel,  Ghost::None);
  t->requires(Task::OldDW, lb->gSurfNormLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->computes(             lb->frictionalWorkLabel);
  t->modifies(             lb->gVelocityLabel, mss);
}

void FrictionContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::NewDW,lb->pStressLabel_afterStrainRate,
	      Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->gMassLabel,              Ghost::AroundNodes, 1);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->gAccelerationLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

  t->computes(lb->gNormTractionLabel);
  t->computes(lb->gSurfNormLabel);
  t->computes(lb->gStressLabel);
}
