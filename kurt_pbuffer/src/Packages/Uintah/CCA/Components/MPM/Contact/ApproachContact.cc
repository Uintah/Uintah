#include <Packages/Uintah/CCA/Components/MPM/Contact/ApproachContact.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using std::vector;
using std::string;

using namespace std;


ApproachContact::ApproachContact(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps,SimulationStateP& d_sS,
                                 MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_vol_const=0.;
  
  ps->require("mu",d_mu);
  ps->get("volume_constraint",d_vol_const);

  d_sharedState = d_sS;

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(flag->d_8or27==27){
    NGP=2;
    NGN=2;
  }
}

ApproachContact::~ApproachContact()
{
  // Destructor
}

void ApproachContact::exMomInterpolated(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset* matls,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw)
{ 
  typedef IntVector IV;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once
  StaticArray<constNCVariable<double> >  gmass(numMatls);
  StaticArray<constNCVariable<double> >  gvolume(numMatls);
  StaticArray<constNCVariable<double> >  numnearparticles(numMatls);
  StaticArray<NCVariable<Vector> >       gvelocity(numMatls);
  StaticArray<NCVariable<Vector> >       gsurfnorm(numMatls);
  StaticArray<NCVariable<double> >       frictionWork(numMatls);
  StaticArray<NCVariable<Matrix3> >      gstress(numMatls);
  StaticArray<NCVariable<double> >       gnormtraction(numMatls);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
  
    Vector surnor;

    // First, calculate the gradient of the mass everywhere
    // normalize it, and stick it in surfNorm
    for(int m=0;m<matls->size();m++){
      if(!d_matls.requested(m)) continue;
      
      int dwi = matls->get(m);

      new_dw->get(gmass[m],           lb->gMassLabel,  dwi, patch, gan,   1);
      new_dw->get(gvolume[m],         lb->gVolumeLabel,dwi, patch, gnone, 0);
      new_dw->get(numnearparticles[m],lb->gNumNearParticlesLabel, 
                                                       dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity[m],  lb->gVelocityLabel,  dwi, patch);
      new_dw->allocateAndPut(gsurfnorm[m], lb->gSurfNormLabel,  dwi, patch);
      new_dw->getModifiable(frictionWork[m],lb->frictionalWorkLabel,dwi,
                            patch);
      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));

      IntVector low(patch->getInteriorNodeLowIndex());
      IntVector high(patch->getInteriorNodeHighIndex());

      int ILOW=0,IHIGH=0,JLOW=0,JHIGH=0,KLOW=0,KHIGH=0;
      // First, figure out some ranges for for loops
      for(Patch::FaceType face = Patch::startFace;
      		  face <= Patch::endFace; face=Patch::nextFace(face)){
	Patch::BCType bc_type = patch->getBCType(face);

        switch(face) {
         case Patch::xminus:
          if(bc_type == Patch::Neighbor) { ILOW = low.x(); }
          else if(bc_type == Patch::None){ ILOW = low.x()+1; }
          break;
         case Patch::xplus:
          if(bc_type == Patch::Neighbor) { IHIGH = high.x(); }
          else if(bc_type == Patch::None){ IHIGH = high.x()-1; }
          break;
         case Patch::yminus:
          if(bc_type == Patch::Neighbor) { JLOW = low.y(); }
          else if(bc_type == Patch::None){ JLOW = low.y()+1; }
          break;
         case Patch::yplus:
          if(bc_type == Patch::Neighbor) { JHIGH = high.y(); }
          else if(bc_type == Patch::None){ JHIGH = high.y()-1; }
          break;
         case Patch::zminus:
          if(bc_type == Patch::Neighbor) { KLOW = low.z(); }
          else if(bc_type == Patch::None){ KLOW = low.z()+1; }
          break;
         case Patch::zplus:
          if(bc_type == Patch::Neighbor) { KHIGH = high.z(); }
          else if(bc_type == Patch::None){ KHIGH = high.z()-1; }
          break;
         default:
          break;
        }
      }

      // Compute the normals for all of the interior nodes
      for(int i = ILOW; i < IHIGH; i++){
        int ip = i+1; int im = i-1;
        for(int j = JLOW; j < JHIGH; j++){
          int jp = j+1; int jm = j-1;
          for(int k = KLOW; k < KHIGH; k++){
	     surnor = Vector(
                    -(gmass[m][IV(ip,j,k)] - gmass[m][IV(im,j,k)])/dx.x(),
                    -(gmass[m][IV(i,jp,k)] - gmass[m][IV(i,jm,k)])/dx.y(), 
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

	if (bc_type == Patch::None) {
          int i=0,j=0,k=0;
	  if(face==Patch::xplus || face==Patch::xminus){
            int I=0;
            if(face==Patch::xminus){ I=low.x(); }
            if(face==Patch::xplus) { I=high.x()-1; }
            // Faces
            for (j = JLOW; j<JHIGH; j++) {
              int jp = j+1; int jm = j-1;
              for (k = KLOW; k<KHIGH; k++) {
           	surnor = Vector( 0.0,
                       -(gmass[m][IV(I,jp,k)] - gmass[m][IV(I,jm,k)])/dx.y(),
                       -(gmass[m][IV(I,j,k+1)] - gmass[m][IV(I,j,k-1)])/dx.z());
                double length = surnor.length();
                if(length>0.0){
                   gsurfnorm[m][IntVector(I,j,k)] = surnor/length;;
                }
              }
            }
            // Edges
            if(patch->getBCType(Patch::yminus)==Patch::None){
              j=JLOW-1;
              for (k = KLOW; k<KHIGH; k++) {
                surnor = Vector( 0.0,0.0,
                     -(gmass[m][IV(I,j,k+1)] - gmass[m][IV(I,j,k-1)])/dx.z());
                double length = surnor.length();
                if(length>0.0){
                  gsurfnorm[m][IntVector(I,j,k)] = surnor/length;;
                }
              }
            }
            if(patch->getBCType(Patch::yplus)==Patch::None){
              j=JHIGH;
              for (k = KLOW; k<KHIGH; k++) {
                surnor = Vector( 0.0,0.0,
                     -(gmass[m][IV(I,j,k+1)] - gmass[m][IV(I,j,k-1)])/dx.z());
                double length = surnor.length();
                if(length>0.0){
                  gsurfnorm[m][IntVector(I,j,k)] = surnor/length;;
                }
              }
            }
          }

          if(face==Patch::yplus || face==Patch::yminus){
            int J=0;
            if(face==Patch::yminus){ J=low.y(); }
            if(face==Patch::yplus) { J=high.y()-1; }
            // Faces
            for (i = ILOW; i<IHIGH; i++) {
              int ip = i+1; int im = i-1;
              for (k = KLOW; k<KHIGH; k++) {
		surnor = Vector(
	               -(gmass[m][IV(ip,J,k)] - gmass[m][IV(im,J,k)])/dx.x(),
			 0.0,
		       -(gmass[m][IV(i,J,k+1)] - gmass[m][IV(i,J,k-1)])/dx.z());
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,J,k)] = surnor/length;;
		}
              }
            }
            // Edges
            if(patch->getBCType(Patch::zminus)==Patch::None){
              k=KLOW-1;
              for (i = ILOW; i<IHIGH; i++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,J,k)] - gmass[m][IV(i-1,J,k)])/dx.x(),
			 0.0,0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,J,k)] = surnor/length;;
		}
              }
            }
            if(patch->getBCType(Patch::zplus)==Patch::None){
              k=KHIGH;
              for (i = ILOW; i<IHIGH; i++) {
		surnor = Vector(
	               -(gmass[m][IV(i+1,J,k)] - gmass[m][IV(i-1,J,k)])/dx.x(),
			 0.0,0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,J,k)] = surnor/length;;
		}
              }
            }
          }

          if(face==Patch::zplus || face==Patch::zminus){
            int K=0;
            if(face==Patch::zminus){ K=low.z(); }
            if(face==Patch::zplus) { K=high.z()-1; }
            // Faces
            for (i = ILOW; i<IHIGH; i++) {
              int ip = i+1; int im = i-1;
              for (j = JLOW; j<JHIGH; j++) {
		surnor = Vector(
	               -(gmass[m][IV(ip,j,K)] - gmass[m][IV(im,j,K)])/dx.x(),
	               -(gmass[m][IV(i,j+1,K)] - gmass[m][IV(i,j-1,K)])/dx.y(), 
	               0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,j,K)] = surnor/length;;
		}
              }
            }
            // Edges
            if(patch->getBCType(Patch::xminus)==Patch::None){
              i=ILOW-1;
              for (j = JLOW; j<JHIGH; j++) {
		surnor = Vector(0.0,
	               -(gmass[m][IV(i,j+1,K)] - gmass[m][IV(i,j-1,K)])/dx.y(), 
	               0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,j,K)] = surnor/length;;
		}
              }
            }
            if(patch->getBCType(Patch::xplus)==Patch::None){
              i=IHIGH;
              for (j = JLOW; j<JHIGH; j++) {
		surnor = Vector(0.0,
	               -(gmass[m][IV(i,j+1,K)] - gmass[m][IV(i,j-1,K)])/dx.y(), 
	               0.0);
		double length = surnor.length();
		if(length>0.0){
		  gsurfnorm[m][IntVector(i,j,K)] = surnor/length;;
		}
              }
            }
          } // if zsomething
	} // else if (bc_type == Patch::None) {
      }

      // Create arrays for the particle stress and grid stress
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                               Ghost::AroundNodes, NGP,
                                               lb->pXLabel);
      constParticleVariable<Matrix3> pstress;
      constParticleVariable<Point> px;
      old_dw->get(pstress, lb->pStressLabel, pset);
      old_dw->get(px,      lb->pXLabel,      pset);
      new_dw->allocateAndPut(gstress[m],      lb->gStressLabel,      dwi,patch);
      new_dw->allocateAndPut(gnormtraction[m],lb->gNormTractionLabel,dwi,patch);
      gstress[m].initialize(Matrix3(0.0));
      
      // Next, interpolate the stress to the grid
      constParticleVariable<Vector> psize;
      old_dw->get(psize, lb->pSizeLabel, pset);
      

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;

         // Get the node indices that surround the cell
	 interpolator->findCellAndWeights(px[idx], ni, S, psize[idx]);
         
         // Add each particles contribution to the local mass & velocity
         // Must use the node indices
         for(int k = 0; k < flag->d_8or27; k++) {
	   if (patch->containsNode(ni[k]))
	     gstress[m][ni[k]] += pstress[idx] * S[k];
         }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
	gnormtraction[m][c]=
		Dot(gsurfnorm[m][c]*gstress[m][c],gsurfnorm[m][c]);
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      if(!d_matls.present(gmass, c)) continue;
      
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      double totalNearParticles=0.0; 
      for(int n = 0; n < numMatls; n++){
        centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
        centerOfMassMass+=gmass[n][c]; 
        totalNodalVol+=gvolume[n][c];
        totalNearParticles+=numnearparticles[n][c];
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
       Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

       // Only apply contact if the node is nearly "full".  There are
       // two options:

       // 1. This option uses particle counting
       if((totalNodalVol/cell_vol)*(64./totalNearParticles) > d_vol_const){
	 double scale_factor=1.0;

       // 2. This option uses only cell volumes.  The idea is that a cell is full
       // if totalNodalVol/cell_vol .ge. 1.0, and the contraint should only be
       // applied when cells are full.  This logic is used when d_vol_const=0.
       // For d_vol_const > 0 the contact forces are ramped up linearly from 0
       // for totalNodalVol/cell_vol .le. 1.0-d_vol_const to 1.0 for
       // totalNodalVol/cell_vol = 1.  Ramping the contact influence seems to 
       // help remove a "switching" instability.  A good value seems to be
       // d_vol_const=.05

	 //	 double scale_factor=0.0;
	 //	 if(d_vol_const > 0.0){
	 //	   scale_factor=(totalNodalVol/cell_vol-1.+d_vol_const)/d_vol_const;
	 //	   scale_factor=Max(0.0,scale_factor);
	 //	 }
	 //	 else if(totalNodalVol/cell_vol > 1.0){
	 //	   scale_factor=1.0;
	 //	 }

	 //	 if(scale_factor > 0.0){
	 //	   scale_factor=Min(1.0,scale_factor);

        // Loop over velocity fields.  Only proceed if velocity field mass
        // is nonzero (not numerical noise) and the difference from
        // the centerOfMassVelocity is nonzero (More than one velocity
        // field is contributing to grid vertex).
        for(int n = 0; n < numMatls; n++){
          Vector deltaVelocity=gvelocity[n][c]-centerOfMassVelocity;
          if(!compare(gmass[n][c]/centerOfMassMass,0.0)
             && !compare(gmass[n][c]-centerOfMassMass,0.0)){

            // Apply frictional contact IF the surface is in compression
            // OR the surface is stress free and approaching.
            // Otherwise apply free surface conditions (do nothing).
            double normalDeltaVel=Dot(deltaVelocity,gsurfnorm[n][c]);
	    Vector Dv(0.,0.,0.);
            if(normalDeltaVel>0.0){

                // Simplify algorithm in case where approach velocity
                // is in direction of surface normal (no slip).
                if(compare( (deltaVelocity
                        -gsurfnorm[n][c]*normalDeltaVel).length(),0.0)){

		  // Calculate velocity change needed to enforce contact
                  Dv=-gsurfnorm[n][c]*normalDeltaVel;
		  
		  // Calculate work done by frictional force
		  if (flag->d_fracture)
		    frictionWork[n][c] += 0.;
		  else
		    frictionWork[n][c] = 0.;
		}

                // General algorithm, including frictional slip.  The
		// contact velocity change and frictional work are both
		// zero if normalDeltaVel is zero.
	        else if(!compare(fabs(normalDeltaVel),0.0)){
                  Vector surfaceTangent=
		  (deltaVelocity-gsurfnorm[n][c]*normalDeltaVel)/
                  (deltaVelocity-gsurfnorm[n][c]*normalDeltaVel).length();
                  double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
                  double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));

		  // Calculate velocity change needed to enforce contact
                  Dv=
                    -gsurfnorm[n][c]*normalDeltaVel
		    -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

                  // Calculate work done by the frictional force (only) if
                  // contact slips.  Because the frictional force opposes motion
                  // it is dissipative and should always be negative per the
                  // conventional definition.  However, here it is calculated
                  // as positive (Work=-force*distance).
		  if(compare(frictionCoefficient,d_mu)){
		    if (flag->d_fracture)
		      frictionWork[n][c] += gmass[n][c]*frictionCoefficient
			* (normalDeltaVel*normalDeltaVel) *
			(tangentDeltaVelocity/fabs(normalDeltaVel)-
			 frictionCoefficient);
		    else
		      frictionWork[n][c] = gmass[n][c]*frictionCoefficient
			* (normalDeltaVel*normalDeltaVel) *
			(tangentDeltaVelocity/fabs(normalDeltaVel)-
			 frictionCoefficient);
		  }
	        }

		// Define contact algorithm imposed strain, find maximum
	        Vector epsilon=(Dv/dx)*delT;
	        double epsilon_max=
		  Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	        if(!compare(epsilon_max,0.0)){
		  epsilon_max=epsilon_max*Max(1.0,
			    gmass[n][c]/(centerOfMassMass-gmass[n][c]));

		  // Scale velocity change if contact algorithm
                  // imposed strain is too large.
		  double ff=Min(epsilon_max,.5)/epsilon_max;
		  Dv=Dv*ff;
	        }
		Dv=scale_factor*Dv;
	        gvelocity[n][c]+=Dv;
            }  // if traction
	  }    // if !compare && !compare
        }      // matls
       }       // if (volume constraint)
      }        // if(!compare(centerOfMassMass,0.0))
    }          // NodeIterator
    delete interpolator;
  }  // patches
  
}

void ApproachContact::exMomIntegrated(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  IntVector onex(1,0,0), oney(0,1,0), onez(0,0,1);
  typedef IntVector IV;
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  StaticArray<constNCVariable<double> > gmass(numMatls);
  StaticArray<constNCVariable<double> > gvolume(numMatls);
  StaticArray<constNCVariable<double> > numnearparticles(numMatls);
  StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
  StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
  StaticArray<constNCVariable<double> > normtraction(numMatls);
  StaticArray<NCVariable<double> >      frictionWork(numMatls);
  StaticArray<constNCVariable<Vector> > gsurfnorm(numMatls);    

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],       lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->get(normtraction[m],lb->gNormTractionLabel,dwi, patch, gnone, 0);
      new_dw->get(gsurfnorm[m],   lb->gSurfNormLabel,    dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],     lb->gVolumeLabel,      dwi, patch, gnone, 0);
      new_dw->get(numnearparticles[m],lb->gNumNearParticlesLabel, 
                                                         dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(gacceleration[m],lb->gAccelerationLabel,
                                                         dwi, patch);
      new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel,
                                                         dwi, patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    double epsilon_max_max=0.0;

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      Vector centerOfMassMom(0.,0.,0.);
      double centerOfMassMass=0.0; 
      double totalNodalVol=0.0; 
      double totalNearParticles=0.0; 
      for(int  n = 0; n < numMatls; n++){
        centerOfMassMom+=gvelocity_star[n][c] * gmass[n][c];
        centerOfMassMass+=gmass[n][c]; 
        totalNodalVol+=gvolume[n][c];
        totalNearParticles+=numnearparticles[n][c];
      }

      // Apply Coulomb friction contact
      // For grid points with mass calculate velocity
      if(!compare(centerOfMassMass,0.0)){
       Vector centerOfMassVelocity=centerOfMassMom/centerOfMassMass;

       // Only apply contact if the node is nearly "full".  There are
       // two options:

       // 1. This option uses particle counting
       if((totalNodalVol/cell_vol)*(64./totalNearParticles) > d_vol_const){
	 double scale_factor=1.0;

       // 2. This option uses only cell volumes.  The idea is that a cell is full
       // if totalNodalVol/cell_vol .ge. 1.0, and the contraint should only be
       // applied when cells are full.  This logic is used when d_vol_const=0.
       // For d_vol_const > 0 the contact forces are ramped up linearly from 0
       // for totalNodalVol/cell_vol .le. 1.0-d_vol_const to 1.0 for
       // totalNodalVol/cell_vol = 1.  Ramping the contact influence seems to 
       // help remove a "switching" instability.  A good value seems to be
       // d_vol_const=.05

	 //	 double scale_factor=0.0;
	 //	 if(d_vol_const > 0.0){
	 //	   scale_factor=(totalNodalVol/cell_vol-1.+d_vol_const)/d_vol_const;
	 //	   scale_factor=Max(0.0,scale_factor);
	 //	 }
	 //	 else if(totalNodalVol/cell_vol > 1.0){
	 //	   scale_factor=1.0;
	 //	 }

	 //	 if(scale_factor > 0.0){
	 //	   scale_factor=Min(1.0,scale_factor);

        // Loop over velocity fields.  Only proceed if velocity field mass
        // is nonzero (not numerical noise) and the difference from
        // the centerOfMassVelocity is nonzero (More than one velocity
        // field is contributing to grid vertex).
        for(int n = 0; n < numMatls; n++){
          Vector deltaVelocity=gvelocity_star[n][c]-centerOfMassVelocity;
          if(!compare(gmass[n][c]/centerOfMassMass,0.0)
             && !compare(gmass[n][c]-centerOfMassMass,0.0)){

            // Apply frictional contact IF the surface is in compression
            // OR the surface is stress free and approaching.
            // Otherwise apply free surface conditions (do nothing).
            double normalDeltaVel=Dot(deltaVelocity,gsurfnorm[n][c]);

	    Vector Dv(0.,0.,0.);
            if(normalDeltaVel>0.0){

	      // Simplify algorithm in case where approach velocity
	      // is in direction of surface normal (no slip).
	      if(compare( (deltaVelocity
		   -gsurfnorm[n][c]*normalDeltaVel).length(),0.0)){

		// Calculate velocity change needed to enforce contact
	        Dv=-gsurfnorm[n][c]*normalDeltaVel;
		  
		// Calculate work done by frictional force
		frictionWork[n][c] += 0.;
	      }

	      // General algorithm, including frictional slip.  The
	      // contact velocity change and frictional work are both
	      // zero if normalDeltaVel is zero.
	      else if(!compare(fabs(normalDeltaVel),0.0)){
	        Vector surfaceTangent=
	         (deltaVelocity-gsurfnorm[n][c]*normalDeltaVel)/
                 (deltaVelocity-gsurfnorm[n][c]*normalDeltaVel).length();
	        double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
	        double frictionCoefficient=
		  Min(d_mu,tangentDeltaVelocity/fabs(normalDeltaVel));

		// Calculate velocity change needed to enforce contact
	        Dv=
		  -gsurfnorm[n][c]*normalDeltaVel
		  -surfaceTangent*frictionCoefficient*fabs(normalDeltaVel);

                // Calculate work done by the frictional force (only) if
                // contact slips.  Because the frictional force opposes motion
                // it is dissipative and should always be negative per the
                // conventional definition.  However, here it is calculated
                // as positive (Work=-force*distance).
		if(compare(frictionCoefficient,d_mu)){
		  frictionWork[n][c] += gmass[n][c]*frictionCoefficient
					  * (normalDeltaVel*normalDeltaVel) *
		(tangentDeltaVelocity/fabs(normalDeltaVel)-frictionCoefficient);
		}
	      }

	      // Define contact algorithm imposed strain, find maximum
	      Vector epsilon=(Dv/dx)*delT;
	      double epsilon_max=
	        Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
	      epsilon_max_max=max(epsilon_max,epsilon_max_max);
	      if(!compare(epsilon_max,0.0)){
	        epsilon_max=epsilon_max*Max(1.0,
			  gmass[n][c]/(centerOfMassMass-gmass[n][c]));

		// Scale velocity change if contact algorithm imposed strain
                // is too large.
	        double ff=Min(epsilon_max,.5)/epsilon_max;
	        Dv=Dv*ff;
	      }
	      Dv=scale_factor*Dv;
	      gvelocity_star[n][c]+=Dv;
	      Dv=Dv/delT;
	      gacceleration[n][c]+=Dv;
            } // traction
	  }   // if !compare && !compare
        }     // for numMatls
       }      // volume constraint
      }       // if centerofmass > 0
    }         // nodeiterator

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
        IntVector c = *iter;
       	frictionWork[m][c] /= (c_v * gmass[m][c] * delT);
        if(frictionWork[m][c]<0.0){
	  cout << "dT/dt is negative: " << frictionWork[m][c] << endl;
	}
      }
    }

  }
}

void ApproachContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("ApproachContact::exMomInterpolated", 
                      this, &ApproachContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW,   lb->delTLabel);
  t->requires(Task::OldDW,   lb->pXLabel,           Ghost::AroundNodes, NGP);
  t->requires(Task::OldDW,   lb->pStressLabel,      Ghost::AroundNodes, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,        Ghost::AroundNodes, NGP);
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::AroundNodes, 1);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gNumNearParticlesLabel, Ghost::None);
  t->computes(lb->gNormTractionLabel);
  t->computes(lb->gSurfNormLabel);
  t->computes(lb->gStressLabel);
  t->modifies(lb->frictionalWorkLabel, mss);
  t->modifies(lb->gVelocityLabel, mss);
  
  sched->addTask(t, patches, ms);
}

void ApproachContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = new Task("ApproachContact::exMomIntegrated", 
                      this, &ApproachContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gNormTractionLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gSurfNormLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gNumNearParticlesLabel, Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->gAccelerationLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);
  
  sched->addTask(t, patches, ms);
}
