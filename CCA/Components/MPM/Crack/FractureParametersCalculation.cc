/********************************************************************************
    Crack.cc
    PART FOUR: FRACTURE PARAMETERS CALCULATION

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;

#define MAX_BASIS 27

void Crack::addComputesAndRequiresGetNodalSolutions(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  
  t->requires(Task::NewDW,lb->pMassLabel_preReloc,                gan,NGP);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,              gan,NGP);
  t->requires(Task::NewDW,lb->pDispGradsLabel_preReloc,           gan,NGP);
  t->requires(Task::NewDW,lb->pStrainEnergyDensityLabel_preReloc, gan,NGP);
  t->requires(Task::NewDW,lb->pgCodeLabel,                        gan,NGP);
  t->requires(Task::NewDW,lb->pKineticEnergyDensityLabel,         gan,NGP);
  t->requires(Task::NewDW,lb->pVelGradsLabel,                     gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                            gan,NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,                        gan,NGP);
  t->requires(Task::NewDW,lb->gMassLabel,                         gnone);
  t->requires(Task::NewDW,lb->GMassLabel,                         gnone);

  t->computes(lb->gGridStressLabel);
  t->computes(lb->GGridStressLabel);
  t->computes(lb->gStrainEnergyDensityLabel);
  t->computes(lb->GStrainEnergyDensityLabel);
  t->computes(lb->gKineticEnergyDensityLabel);
  t->computes(lb->GKineticEnergyDensityLabel);
  t->computes(lb->gDispGradsLabel);
  t->computes(lb->GDispGradsLabel);
  t->computes(lb->gVelGradsLabel);
  t->computes(lb->GVelGradsLabel);
}

void Crack::GetNodalSolutions(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  // Compute nodal solutions of stresses, displacement gradients,
  // strain energy density and  kinetic energy density by interpolating
  // particle's solutions to grid. Those variables will be used to calculate
  // J-integral

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    double time = d_sharedState->getElapsedTime();

    // Detect if calculating fracture parameters or
    // doing crack propagation at this time step
    DetectIfDoingFractureAnalysisAtThisTimeStep(time);
    
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get particle's solutions
      constParticleVariable<Short27> pgCode;
      constParticleVariable<Point>   px;
      constParticleVariable<Vector>  psize;
      constParticleVariable<double>  pmass;
      constParticleVariable<double>  pstrainenergydensity;
      constParticleVariable<double>  pkineticenergydensity;
      constParticleVariable<Matrix3> pstress,pdispgrads,pvelgrads;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                             Ghost::AroundNodes, NGP,lb->pXLabel);

      new_dw->get(pmass,                lb->pMassLabel_preReloc,       pset);
      new_dw->get(pstress,              lb->pStressLabel_preReloc,     pset);
      new_dw->get(pdispgrads,           lb->pDispGradsLabel_preReloc,  pset);
      new_dw->get(pstrainenergydensity,
                               lb->pStrainEnergyDensityLabel_preReloc, pset);

      new_dw->get(pgCode,               lb->pgCodeLabel,               pset);
      new_dw->get(pvelgrads,            lb->pVelGradsLabel,            pset);
      new_dw->get(pkineticenergydensity,lb->pKineticEnergyDensityLabel,pset);

      old_dw->get(px,                   lb->pXLabel,                   pset);
      old_dw->get(psize,lb->pSizeLabel,                pset);

      // Get nodal mass
      constNCVariable<double> gmass, Gmass;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, Ghost::None, 0);

      // Declare the nodal variables to be calculated
      NCVariable<Matrix3> ggridstress,Ggridstress;
      NCVariable<Matrix3> gdispgrads,Gdispgrads;
      NCVariable<Matrix3> gvelgrads,Gvelgrads;
      NCVariable<double>  gstrainenergydensity,Gstrainenergydensity;
      NCVariable<double>  gkineticenergydensity,Gkineticenergydensity;

      new_dw->allocateAndPut(ggridstress, lb->gGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(Ggridstress, lb->GGridStressLabel,dwi,patch);
      new_dw->allocateAndPut(gdispgrads,  lb->gDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(Gdispgrads,  lb->GDispGradsLabel, dwi,patch);
      new_dw->allocateAndPut(gvelgrads,   lb->gVelGradsLabel,  dwi,patch);
      new_dw->allocateAndPut(Gvelgrads,   lb->GVelGradsLabel,  dwi,patch);
      new_dw->allocateAndPut(gstrainenergydensity,
                                lb->gStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(Gstrainenergydensity,
                                lb->GStrainEnergyDensityLabel, dwi,patch);
      new_dw->allocateAndPut(gkineticenergydensity,
                                lb->gKineticEnergyDensityLabel,dwi,patch);
      new_dw->allocateAndPut(Gkineticenergydensity,
                                lb->GKineticEnergyDensityLabel,dwi,patch);

      ggridstress.initialize(Matrix3(0.));
      Ggridstress.initialize(Matrix3(0.));
      gdispgrads.initialize(Matrix3(0.));
      Gdispgrads.initialize(Matrix3(0.));
      gvelgrads.initialize(Matrix3(0.));
      Gvelgrads.initialize(Matrix3(0.));
      gstrainenergydensity.initialize(0.);
      Gstrainenergydensity.initialize(0.);
      gkineticenergydensity.initialize(0.);
      Gkineticenergydensity.initialize(0.);

      if(calFractParameters || doCrackPropagation) {
        for (ParticleSubset::iterator iter = pset->begin();
                             iter != pset->end(); iter++) {
          particleIndex idx = *iter;
	  interpolator->findCellAndWeights(px[idx], ni, S, psize[idx]);

          for (int k = 0; k < n8or27; k++){
            if(patch->containsNode(ni[k])){
              double pmassTimesS=pmass[idx]*S[k];
              if(pgCode[idx][k]==1) {
                ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                gvelgrads[ni[k]]   += pvelgrads[idx]  * pmassTimesS;
                gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
              else if(pgCode[idx][k]==2) {
                Ggridstress[ni[k]] += pstress[idx]    * pmassTimesS;
                Gdispgrads[ni[k]]  += pdispgrads[idx] * pmassTimesS;
                Gvelgrads[ni[k]]   += pvelgrads[idx]  * pmassTimesS;
                Gstrainenergydensity[ni[k]]  += pstrainenergydensity[idx] *
                                                        pmassTimesS;
                Gkineticenergydensity[ni[k]] += pkineticenergydensity[idx] *
                                                        pmassTimesS;
              }
            }
          } // End of loop over k
        } // End of loop over particles

        for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
          IntVector c = *iter;
          // above crack
          ggridstress[c]           /= gmass[c];
          gdispgrads[c]            /= gmass[c];
          gvelgrads[c]             /= gmass[c];
          gstrainenergydensity[c]  /= gmass[c];
          gkineticenergydensity[c] /= gmass[c];
          // below crack
          Ggridstress[c]           /= Gmass[c];
          Gdispgrads[c]            /= Gmass[c];
          Gvelgrads[c]             /= Gmass[c];
          Gstrainenergydensity[c]  /= Gmass[c];
          Gkineticenergydensity[c] /= Gmass[c];
        }
      } // End if(calFractParameters || doCrackPropagation)
    }
    delete interpolator;
  }
}

void Crack::addComputesAndRequiresCalculateFractureParameters(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Required for contour integral
  int NGC=NJ+NGN+1;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gMassLabel,                gac,NGC);
  t->requires(Task::NewDW, lb->GMassLabel,                gac,NGC);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->gDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->GDisplacementLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->gGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->GGridStressLabel,          gac,NGC);
  t->requires(Task::NewDW, lb->gDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->GDispGradsLabel,           gac,NGC);
  t->requires(Task::NewDW, lb->gStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->GStrainEnergyDensityLabel, gac,NGC);
  t->requires(Task::NewDW, lb->gKineticEnergyDensityLabel,gac,NGC);
  t->requires(Task::NewDW, lb->GKineticEnergyDensityLabel,gac,NGC);

  // Required for area integral
  t->requires(Task::NewDW, lb->gAccelerationLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->GAccelerationLabel,        gac,NGC);
  t->requires(Task::NewDW, lb->gVelGradsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->GVelGradsLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->gVelocityLabel,            gac,NGC);
  t->requires(Task::NewDW, lb->GVelocityLabel,            gac,NGC);
  t->requires(Task::OldDW, lb->pSizeLabel,            Ghost::None);
}

void Crack::CalculateFractureParameters(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_max=Max(dx.x(),dx.y(),dx.z());

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    int pid,patch_size;
    MPI_Comm_size(mpi_crack_comm,&patch_size);
    MPI_Comm_rank(mpi_crack_comm,&pid);
    MPI_Datatype MPI_VECTOR=fun_getTypeDescription((Vector*)0)->getMPIType();

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m=0;m<numMatls;m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      int dwi = matls->get(m);
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<double> gmass,Gmass;
      constNCVariable<int>     GnumPatls;
      constNCVariable<Vector>  gdisp,Gdisp;
      constNCVariable<Matrix3> ggridStress,GgridStress;
      constNCVariable<Matrix3> gdispGrads,GdispGrads;
      constNCVariable<double>  gW,GW;
      constNCVariable<double>  gK,GK;
      constNCVariable<Vector>  gacc,Gacc;
      constNCVariable<Vector>  gvel,Gvel;
      constNCVariable<Matrix3> gvelGrads,GvelGrads;

      // Get nodal solutions
      int NGC=NJ+NGN+1;
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gmass,      lb->gMassLabel,         dwi,patch,gac,NGC);
      new_dw->get(Gmass,      lb->GMassLabel,         dwi,patch,gac,NGC);
      new_dw->get(GnumPatls,  lb->GNumPatlsLabel,     dwi,patch,gac,NGC);
      new_dw->get(gdisp,      lb->gDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(Gdisp,      lb->GDisplacementLabel, dwi,patch,gac,NGC);
      new_dw->get(ggridStress,lb->gGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(GgridStress,lb->GGridStressLabel,   dwi,patch,gac,NGC);
      new_dw->get(gdispGrads, lb->gDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(GdispGrads, lb->GDispGradsLabel,    dwi,patch,gac,NGC);
      new_dw->get(gW,lb->gStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(GW,lb->GStrainEnergyDensityLabel,   dwi,patch,gac,NGC);
      new_dw->get(gK,lb->gKineticEnergyDensityLabel,  dwi,patch,gac,NGC);
      new_dw->get(GK,lb->GKineticEnergyDensityLabel,  dwi,patch,gac,NGC);
      new_dw->get(gacc,       lb->gAccelerationLabel, dwi,patch,gac,NGC);
      new_dw->get(Gacc,       lb->GAccelerationLabel, dwi,patch,gac,NGC);
      new_dw->get(gvel,       lb->gVelocityLabel,     dwi,patch,gac,NGC);
      new_dw->get(Gvel,       lb->GVelocityLabel,     dwi,patch,gac,NGC);
      new_dw->get(gvelGrads,  lb->gVelGradsLabel,     dwi,patch,gac,NGC);
      new_dw->get(GvelGrads,  lb->GVelGradsLabel,     dwi,patch,gac,NGC);

      constParticleVariable<Vector> psize;
      old_dw->get(psize, lb->pSizeLabel, pset);

      // Allocate memories for cfSegJ and cfSegK
      int cfNodeSize=(int)cfSegNodes[m].size();
      cfSegJ[m].resize(cfNodeSize);
      cfSegK[m].resize(cfNodeSize);
      if(calFractParameters || doCrackPropagation) {
        for(int i=0; i<patch_size; i++) { // Loop over all patches
          // number of crack-front nodes in patch i		
          int num= (int) cfnset[m][i].size(); 
          if(num>0) { // If there is crack-front node(s) in patch i
            Vector* cfJ=new Vector[num];
            Vector* cfK=new Vector[num];

            if(pid==i) { // Calculte J & K by processor i
              for(int l=0; l<num; l++) {
                int idx=cfnset[m][i][l];     // index of this node
                int node=cfSegNodes[m][idx]; // node number
                
                int preIdx=cfSegPreIdx[m][idx];
                for(int ij=l-1; ij>=0; ij--) {
                  if(preIdx==cfnset[m][i][ij]) {
                    preIdx=ij;
                    break;
                  } 
                } 
                if(preIdx<0) { // duplicate node, not operated
			
                  // Step 1: Define crack-front local coordinates with
                  // origin located at the point at which J&K is calculated.
                  // v1,v2,v3: direction consies of new axes X',Y' and Z'
			 
                  // Two segments connected by the node
                  int segs[2];
                  FindSegsFromNode(m,node,segs);
                  
		  // Detect if this is a single segment crack, and the neighbor node
		  short singleSeg=NO;
		  int neighbor=-1,segsNeighbor[2]={-1,-1};
		  if(segs[R]<0) {
	            neighbor=cfSegNodes[m][2*segs[L]+1];
		    FindSegsFromNode(m,neighbor,segsNeighbor);
	            if(segsNeighbor[L]<0) singleSeg=YES;
                  }
                  if(segs[L]<0) {
                    neighbor=cfSegNodes[m][2*segs[R]];
                    FindSegsFromNode(m,neighbor,segsNeighbor);
                    if(segsNeighbor[R]<0) singleSeg=YES;		    
                  }		    
		    
                  // Position where to calculate J & K
                  Point origin;
                  double x0,y0,z0;
		  if(singleSeg) { // single segment
                    Point pt1=cx[m][node];
		    Point pt2=cx[m][neighbor];
		    origin=pt1+(pt2-pt1)/2.;
		  }
		  else { // multiple segments
                    if(segs[R]<0) { 
		      // For the right edge node, shift the position to the neighbor
	              origin=cx[m][neighbor];
	            }
                    else if(segs[L]<0) {
		      // For the left edge node, shift the position to the neighbor
                      origin=cx[m][neighbor];
	            }
                    else { // middle nodes
	              origin=cx[m][node];
	            }		    
		  }	  
		  // Coordinates of origin
                  x0=origin.x();  y0=origin.y();  z0=origin.z();

                  // Direction cosines of local coordinates at the node
                  Vector v1=cfSegV1[m][idx];
                  Vector v2=cfSegV2[m][idx];
                  Vector v3=cfSegV3[m][idx];
                  double l1,m1,n1,l2,m2,n2,l3,m3,n3;
                  l1=v1.x(); m1=v1.y(); n1=v1.z();
                  l2=v2.x(); m2=v2.y(); n2=v2.z();
                  l3=v3.x(); m3=v3.y(); n3=v3.z();

                  // Coordinate transformation matrix from global to local(T)
                  // and the one from local to global (TT)
                  Matrix3 T =Matrix3(l1,m1,n1,l2,m2,n2,l3,m3,n3);
                  Matrix3 TT=Matrix3(l1,l2,l3,m1,m2,m3,n1,n2,n3);


                  // Step 2: Find parameters A[14] of J-integral contour
		  //   with the equation:
                  //   A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0 and
                  //   A10x+A11y+A12z+A13=0
		     
                  double A[14];
                  FindJIntegralPath(origin,v1,v2,v3,A);

		  
                  // Step 3: Find intersection (crossPt) between J-integral ptah 
		  // and crack plane
                 
                  double d_rJ=rJ; // Radius of J-contour
                  Point crossPt;
                  while(!FindIntersectionJPathAndCrackPlane(m,d_rJ,A,crossPt)) {
                    d_rJ*=0.7;
		    if(d_rJ/rJ<0.01) {
		      cout << "Error: J-integral radius (d_rJ) has been decreassed 100 times "
			   << " before finding the intersection between J-contour and crack plane."
	                   << " Program terminated." << endl;
                      exit(1);
                    }	
                  }		    

                  // Get coordinates of intersection in local system 
                  double xc,yc,zc,xcprime,ycprime,scprime;
                  xc=crossPt.x(); yc=crossPt.y(); zc=crossPt.z();
                  xcprime=l1*(xc-x0)+m1*(yc-y0)+n1*(zc-z0);
                  ycprime=l2*(xc-x0)+m2*(yc-y0)+n2*(zc-z0);
                  scprime=sqrt(xcprime*xcprime+ycprime*ycprime);

		  
                  // Step 4: Set integral points in J-integral contour
                 
                  int nSegs=16;
                  double xprime,yprime,x,y,z;
                  double PI=3.141592654;
                  Point*   X  = new Point[nSegs+1];   // Integral points
                  double*  W  = new double[nSegs+1];  // Strain energy density
                  double*  K  = new double[nSegs+1];  // Kinetic energy density
                  Matrix3* ST = new Matrix3[nSegs+1]; // Stresses in global coordinates
                  Matrix3* DG = new Matrix3[nSegs+1]; // Disp grads in global coordinates
                  Matrix3* st = new Matrix3[nSegs+1]; // Stresses in local coordinates
                  Matrix3* dg = new Matrix3[nSegs+1]; // Disp grads in local coordinates

                  for(int j=0; j<=nSegs; j++) {       // Loop over points on the circle
                    double angle,cosTheta,sinTheta;
                    angle=2*PI*(float)j/(float)nSegs;
                    cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
                    sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
                    // Coordinates of integral points in local coordinates
                    xprime=d_rJ*cosTheta;
                    yprime=d_rJ*sinTheta;
                    // Coordinates of integral points in global coordinates
                    x=l1*xprime+l2*yprime+x0;
                    y=m1*xprime+m2*yprime+y0;
                    z=n1*xprime+n2*yprime+z0;
                    X[j] = Point(x,y,z);
                    W[j]  = 0.0;
                    K[j]  = 0.0;
                    ST[j] = Matrix3(0.);
                    DG[j] = Matrix3(0.);
                    st[j] = Matrix3(0.);
                    dg[j] = Matrix3(0.);
                  }

		  
                  // Step 5: Evaluate solutions at integral points in global coordinates
                 
                  for(int j=0; j<=nSegs; j++) {
		    interpolator->findCellAndWeights(X[j],ni,S,psize[j]);
                    for(int k=0; k<n8or27; k++) {
                      if(GnumPatls[ni[k]]!=0 && j<nSegs/2) { // below crack
                        W[j]  += GW[ni[k]]          * S[k];
                        K[j]  += GK[ni[k]]          * S[k];
                        ST[j] += GgridStress[ni[k]] * S[k];
                        DG[j] += GdispGrads[ni[k]]  * S[k];
                      }
                      else { // above crack
                        W[j]  += gW[ni[k]]          * S[k];
                        K[j]  += gK[ni[k]]          * S[k];
                        ST[j] += ggridStress[ni[k]] * S[k];
                        DG[j] += gdispGrads[ni[k]]  * S[k];
                      }
                    } 
                  } 

		  
                  // Step 6: Transform the solutions to crack-front local coordinates
                  
                  for(int j=0; j<=nSegs; j++) {
                    for(int i1=0; i1<3; i1++) {
                      for(int j1=0; j1<3; j1++) {
                        for(int i2=0; i2<3; i2++) {
                          for(int j2=0; j2<3; j2++) {
                            st[j](i1,j1) += T(i1,i2)*T(j1,j2)*ST[j](i2,j2);
                            dg[j](i1,j1) += T(i1,i2)*T(j1,j2)*DG[j](i2,j2);
                          }
                        }
                      } 
                    } 
                  } 

		  
                  // Step 7: Compute integrand values at integral points
                 
                  double* f1ForJx = new double[nSegs+1];
                  double* f1ForJy = new double[nSegs+1];
                  for(int j=0; j<=nSegs; j++) {
                    double angle,cosTheta,sinTheta;

                    angle=2*PI*(float)j/(float)nSegs;
                    cosTheta=(xcprime*cos(angle)-ycprime*sin(angle))/scprime;
                    sinTheta=(ycprime*cos(angle)+xcprime*sin(angle))/scprime;
                    double t1=st[j](0,0)*cosTheta+st[j](0,1)*sinTheta;
                    double t2=st[j](1,0)*cosTheta+st[j](1,1)*sinTheta;
                    //double t3=st[j](2,0)*cosTheta+st[j](2,1)*sinTheta;

                    Vector t123=Vector(t1,t2,0./*t3*/); // plane state
                    Vector dgx=Vector(dg[j](0,0),dg[j](1,0),dg[j](2,0));
                    Vector dgy=Vector(dg[j](0,1),dg[j](1,1),dg[j](2,1));

                    f1ForJx[j]=(W[j]+K[j])*cosTheta-Dot(t123,dgx);
                    f1ForJy[j]=(W[j]+K[j])*sinTheta-Dot(t123,dgy);
                  }

		  
                  // Step 8: Calculate contour integral (primary part of J integral)
                 
                  double Jx1=0.,Jy1=0.;
                  for(int j=0; j<nSegs; j++) { 
                    Jx1 += f1ForJx[j] + f1ForJx[j+1];
                    Jy1 += f1ForJx[j] + f1ForJy[j+1];
                  } 
                  Jx1 *= d_rJ*PI/nSegs;
                  Jy1 *= d_rJ*PI/nSegs;

		  
                  // Step 9: Release dynamic arries for this crack front segment
                 
                  delete [] X;
                  delete [] W;        delete [] K;
                  delete [] ST;       delete [] DG;
                  delete [] st;       delete [] dg;
                  delete [] f1ForJx;  delete [] f1ForJy;

		  
                  // Step 10: Area integral (secondary part of J-integral)
		  // Area integral calculation is optional. 
		  // It can be forced in input file by setting useVolumeIntegral=YES.
		  // If Jx1 less than zero, it will be activated automatically.  
			 
                  double Jx2=0.,Jy2=0.;
                  if(useVolumeIntegral || Jx1<0.) {
                    // Define integral points in the area enclosed by J-integral contour
                    int nc=(int)(d_rJ/dx_max);
                    if(d_rJ/dx_max-nc>=0.5) nc++;
                    if(nc<2) nc=2; // Cell number J-path away from the origin
                    double* c=new double[4*nc];
                    for(int j=0; j<4*nc; j++)
                      c[j]=(float)(-4*nc+1+2*j)/4/nc*d_rJ;

                    int count=0;  // number of integral points
                    Point* x=new Point[16*nc*nc];
                    Point* X=new Point[16*nc*nc];
                    for(int i1=0; i1<4*nc; i1++) {
                      for(int j1=0; j1<4*nc; j1++) {
                        Point pq=Point(c[i1],c[j1],0.);
                        if(pq.asVector().length()<d_rJ) {
                          x[count]=pq;
                          X[count]=origin+TT*pq.asVector();
                          count++;
                        }
                      }
                    }
                    delete [] c;

                    // Get the solution at the integral points in local system
                    Vector* acc=new Vector[count];      // accelerations
                    Vector* vel=new Vector[count];      // velocities
                    Matrix3* dg = new Matrix3[count];   // displacement gradients
                    Matrix3* vg = new Matrix3[count];   // velocity gradients

                    for(int j=0; j<count; j++) {
                      // Get the solutions in global system
                      Vector ACC=Vector(0.,0.,0.);
                      Vector VEL=Vector(0.,0.,0.);
                      Matrix3 DG=Matrix3(0.0);
                      Matrix3 VG=Matrix3(0.0);

		      interpolator->findCellAndWeights(X[j],ni,S,psize[j]);

                      for(int k=0; k<n8or27; k++) {
                        if(GnumPatls[ni[k]]!=0 && x[j].y()<0.) { // below crack
                          // Valid only for stright crack within J-path, usually true
                          ACC += Gacc[ni[k]]       * S[k];
                          VEL += Gvel[ni[k]]       * S[k];
                          DG  += GdispGrads[ni[k]] * S[k];
                          VG  += GvelGrads[ni[k]]  * S[k];
                        }
                        else {  // above crack
                          ACC += gacc[ni[k]]       * S[k];
                          VEL += gvel[ni[k]]       * S[k];
                          DG  += gdispGrads[ni[k]] * S[k];
                          VG  += gvelGrads[ni[k]]  * S[k];
                        }
                      } 

                      // Transfer into the local system
                      acc[j] = T*ACC;
                      vel[j] = T*VEL;

                      dg[j]=Matrix3(0.0);
                      vg[j]=Matrix3(0.0);
                      for(int i1=0; i1<3; i1++) {
                        for(int j1=0; j1<3; j1++) {
                          for(int i2=0; i2<3; i2++) {
                            for(int j2=0; j2<3; j2++) {
                              dg[j](i1,j1) += T(i1,i2)*T(j1,j2)*DG(i2,j2);
                              vg[j](i1,j1) += T(i1,i2)*T(j1,j2)*VG(i2,j2);
                            }
                          }
                        }
                      } 
                    } // End of loop over j 

                    double f2ForJx=0.,f2ForJy=0.;
                    double rho=mpm_matl->getInitialDensity();
                    for(int j=0; j<count;j++) {
                      // Zero components in z direction for plane state
                      Vector dgx=Vector(dg[j](0,0),dg[j](1,0),0./*dg[j](2,0)*/);
                      Vector dgy=Vector(dg[j](0,1),dg[j](1,1),0./*dg[j](2,1)*/);
                      Vector vgx=Vector(vg[j](0,0),vg[j](1,0),0./*vg[j](2,0)*/);
                      Vector vgy=Vector(vg[j](0,1),vg[j](1,1),0./*vg[j](2,1)*/);
                      f2ForJx+=rho*(Dot(acc[j],dgx)-Dot(vel[j],vgx));
                      f2ForJy+=rho*(Dot(acc[j],dgy)-Dot(vel[j],vgy));
                    }

                    double Jarea=PI*d_rJ*d_rJ;
                    Jx2=f2ForJx/count*Jarea;
                    Jy2=f2ForJy/count*Jarea;

                    delete [] x;    delete [] X;
                    delete [] acc;  delete [] vel;
                    delete [] dg;   delete [] vg;

                  } // End of if(useVoluemIntegral)

                  // J-integral components
		  cfJ[l]=Vector(Jx1+Jx2,Jy1+Jy2,0.);

		   
                  // Step 11: Convert J-integral to stress intensity (K)
                 
                  // Task 11a: Find COD near crack tip (point(-d,0,0) in local coordinates)
                  double d;
                  if(d_doCrackPropagation!="false")  // For crack propagation
                    d=(rdadx<1.? 1.:rdadx)*dx_max;
                  else  // For calculation of pure fracture parameters
                    d=d_rJ/2.;

                  double x_d=l1*(-d)+x0;
                  double y_d=m1*(-d)+y0;
                  double z_d=n1*(-d)+z0;
                  Point  p_d=Point(x_d,y_d,z_d);

                  // Calculate displacements at point p_d
                  Vector disp_a=Vector(0.);
                  Vector disp_b=Vector(0.);
		  interpolator->findCellAndWeights(p_d,ni,S,psize[0]);
                  for(int k=0; k<n8or27; k++) {
                    disp_a += gdisp[ni[k]] * S[k];
                    disp_b += Gdisp[ni[k]] * S[k];
                  }

                  // Tranform to local system
                  Vector disp_a_prime=T*disp_a;
                  Vector disp_b_prime=T*disp_b;

                  // Crack opening displacements
                  Vector D = disp_a_prime - disp_b_prime;

                  // Task 11b: Get crack propagating velocity
                  double C=cfSegVel[m][idx];

                  // Convert J-integral into stress intensity factors
		  // cfJ is the components of J-integral in the crack-axis coordinates,
		  // and its first component is the total energy release rate. 
                  Vector SIF;
                  cm->ConvertJToK(mpm_matl,cfJ[l],C,D,SIF);
                  cfK[l]=SIF;
                } // End if not operated
                else { // if operated
                  cfJ[l]=cfJ[preIdx];
                  cfK[l]=cfK[preIdx];
                }
              } // End of loop over nodes (l)
            } // End if(pid==i)

            // Broadcast the results calculated by rank i to all the ranks
            MPI_Bcast(cfJ,num,MPI_VECTOR,i,mpi_crack_comm);
            MPI_Bcast(cfK,num,MPI_VECTOR,i,mpi_crack_comm);

            // Save data in cfsegJ and cfSegK
            for(int l=0; l<num; l++) {
              int idx=cfnset[m][i][l];
              cfSegJ[m][idx]=cfJ[l];
              cfSegK[m][idx]=cfK[l];
            }

            // Release dynamic arries
            delete [] cfJ;
            delete [] cfK;
          } // End if(num>0)
        } // End of loop over ranks (i)
      } // End if(calFractParameters || doCrackPropagation)
   
      if(pid==0) OutputCrackFrontResults(m);
      
    } // End of loop over matls
    delete interpolator;
  } 
}

void Crack::DetectIfDoingFractureAnalysisAtThisTimeStep(double time)
{
  static double timeforcalculateJK=-1.e-200;
  static double timeforpropagation=-1.e-200;

  // For fracture parameters calculation
  if(d_calFractParameters=="true") {
    if(time>=timeforcalculateJK) {
      calFractParameters=YES;
      timeforcalculateJK=time+calFractParasInterval;
    }
    else {
     calFractParameters=NO;
    }
  }
  else if(d_calFractParameters=="false") {
   calFractParameters=NO;
  }
  else if(d_calFractParameters=="every_time_step"){
    calFractParameters=YES;
  }

  // For crack propagation
  if(d_doCrackPropagation=="true") {
    if(time>=timeforpropagation){
      doCrackPropagation=YES;
      timeforpropagation=time+crackPropInterval;
    }
    else {
      doCrackPropagation=NO;
    }
  }
  else if(d_doCrackPropagation=="false") {
    doCrackPropagation=NO;
  }
  else if(d_doCrackPropagation=="every_time_step"){
    doCrackPropagation=YES;
  }
}

// Determine parameters (A[14]) of J-integral contour
void Crack::FindJIntegralPath(const Point& origin, const Vector& v1,
                              const Vector& v2,const Vector& v3, double A[])
{
   // J-integral is a spatial circle with the equation system:
   //   A0x^2+A1y^2+A2z^2+A3xy+A4xz+A5yz+A6x+A7y+A8z+A9-r^2=0
   //   A10x+A11y+A12z+A13=0
   // where r is radius of the circle

   double x0,y0,z0;
   double l1,m1,n1,l2,m2,n2,l3,m3,n3;

   x0=origin.x(); y0=origin.y(); z0=origin.z();

   l1=v1.x(); m1=v1.y(); n1=v1.z();
   l2=v2.x(); m2=v2.y(); n2=v2.z();
   l3=v3.x(); m3=v3.y(); n3=v3.z();

   double term1,term2;
   term1=l1*x0+m1*y0+n1*z0;
   term2=l2*x0+m2*y0+n2*z0;

   A[0]=l1*l1+l2*l2;
   A[1]=m1*m1+m2*m2;
   A[2]=n1*n1+n2*n2;
   A[3]=2*(l1*m1+l2*m2);
   A[4]=2*(l1*n1+l2*n2);
   A[5]=2*(m1*n1+m2*n2);
   A[6]=-2*(l1*term1+l2*term2);
   A[7]=-2*(m1*term1+m2*term2);
   A[8]=-2*(n1*term1+n2*term2);
   A[9]=A[0]*x0*x0+A[1]*y0*y0+A[2]*z0*z0+A[3]*x0*y0+A[4]*x0*z0+A[5]*y0*z0;

   A[10]=l3;
   A[11]=m3;
   A[12]=n3;
   A[13]=-(l3*x0+m3*y0+n3*z0);
}

// Find the intersection between J-integral contour and crack plane
bool Crack::FindIntersectionJPathAndCrackPlane(const int& m,
              const double& radius, const double M[],Point& crossPt)
{  
   // J-integral contour's equations:
   //   Ax^2+By^2+Cz^2+Dxy+Exz+Fyz+Gx+Hy+Iz+J-r^2=0 and a1x+b1y+c1z+d1=0
   // crack plane equation:
   //   a2x+b2y+c2z+d2=0
   // where r -- J-integral contour's radius. Parameters are stroed in M.
   
   double A,B,C,D,E,F,G,H,I,J,a1,b1,c1,d1;
   A=M[0];      a1=M[10];
   B=M[1];      b1=M[11];
   C=M[2];      c1=M[12];
   D=M[3];      d1=M[13];
   E=M[4];
   F=M[5];
   G=M[6];
   H=M[7];
   I=M[8];
   J=M[9];

   int numCross=0;
   crossPt=Point(-9e32,-9e32,-9e32);
   for(int i=0; i<(int)ce[m].size(); i++) { 
     // Find the equation of the plane of the crack elem: a2x+b2y+c2z+d2=0
     double a2,b2,c2,d2; 
     Point pt1,pt2,pt3; 
     pt1=cx[m][ce[m][i].x()];
     pt2=cx[m][ce[m][i].y()];
     pt3=cx[m][ce[m][i].z()];
     FindPlaneEquation(pt1,pt2,pt3,a2,b2,c2,d2);

     // Define crack-front local coordinates (X',Y',Z')
     // with the origin located at p1, and X'=p1->p2
     // v1,v2,v3 -- dirction cosines of new axes X',Y' and Z'
     
     Vector v1,v2,v3;
     double term1 = sqrt(a2*a2+b2*b2+c2*c2);
     v2=Vector(a2/term1,b2/term1,c2/term1);
     v1=TwoPtsDirCos(pt1,pt2);
     Vector v12=Cross(v1,v2);
     v3=v12/v12.length(); 
     // Transform matrix from global to local
     Matrix3 T=Matrix3(v1.x(),v1.y(),v1.z(),v2.x(),v2.y(),v2.z(),
                       v3.x(),v3.y(),v3.z());

     // Find intersection between J-path circle and crack plane
     //   first combine a1x+b1y+c1z+d1=0 And a2x+b2y+c2z+d2=0, get
     //   x=p1*z+q1 & y=p2*z+q2 (CASE 1) or
     //   x=p1*y+q1 & z=p2*y+q2 (CASE 2) or
     //   y=p1*x+q1 & z=p2*y+q2 (CASE 3), depending on the equations
     //  then combine with equation of the circle, getting the intersection
     
     int CASE=0;
     double delt1,delt2,delt3,p1,q1,p2,q2;
     double abar,bbar,cbar,abc;
     Point crossPt1,crossPt2;

     delt1=a1*b2-a2*b1;
     delt2=a1*c2-a2*c1;
     delt3=b1*c2-b2*c1;
     if(fabs(delt1)>=fabs(delt2) && fabs(delt1)>=fabs(delt3)) CASE=1;
     if(fabs(delt2)>=fabs(delt1) && fabs(delt2)>=fabs(delt3)) CASE=2;
     if(fabs(delt3)>=fabs(delt1) && fabs(delt3)>=fabs(delt2)) CASE=3;

     double x1=0.,y1=0.,z1=0.,x2=0.,y2=0.,z2=0.;
     switch(CASE) {
       case 1:
         p1=(b1*c2-b2*c1)/delt1;
         q1=(b1*d2-b2*d1)/delt1;
         p2=(a2*c1-a1*c2)/delt1;
         q2=(a2*d1-a1*d2)/delt1;
         abar=p1*p1*A+p2*p2*B+C+p1*p2*D+p1*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*B+(p1*q2+p2*q1)*D+q1*E+q2*F+p1*G+p2*H+I;
         cbar=q1*q1*A+q2*q2*B+q1*q2*D+q1*G+q2*H+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no solution, skip to the next segment
         // the first solution
         z1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*z1+q1;
         y1=p2*z1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         z2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*z2+q1;
         y2=p2*z2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 2:
         p1=(b2*c1-b1*c2)/delt2;
         q1=(c1*d2-c2*d1)/delt2;
         p2=(a2*b1-a1*b2)/delt2;
         q2=(a2*d1-a1*d2)/delt2;
         abar=p1*p1*A+B+p2*p2*C+p1*D+p1*p2*E+p2*F;
         bbar=2*p1*q1*A+2*p2*q2*C+q1*D+(p1*q2+p2*q1)*E+q2*F+p1*G+H+p2*I;
         cbar=q1*q1*A+q2*q2*C+q1*q2*E+q1*G+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no intersection, skip to the next elem
         // the first solution
         y1=0.5*(-bbar+sqrt(abc))/abar;
         x1=p1*y1+q1;
         z1=p2*y1+q2;
         crossPt1=Point(x1,y1,z1);
         //the second solution
         y2=0.5*(-bbar-sqrt(abc))/abar;
         x2=p1*y2+q1;
         z2=p2*y2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
       case 3:
         p1=(a2*c1-a1*c2)/delt3;
         q1=(c1*d2-c2*d1)/delt3;
         p2=(a1*b2-a2*b1)/delt3;
         q2=(b2*d1-b1*d2)/delt3;
         abar=A+p1*p1*B+p2*p2*C+p1*D+p2*E+p1*p2*F;
         bbar=2*p1*q1*B+2*p2*q2*C+q1*D+q2*E+(p1*q2+p2*q1)*F+G+p1*H+p2*I;
         cbar=q1*q1*B+q2*q2*C+q1*q2*F+q1*H+q2*I+J-radius*radius;
         abc=bbar*bbar-4*abar*cbar;
         if(abc<0.0) continue;  // no intersection, skip to the next elem
         // the first solution
         x1=0.5*(-bbar+sqrt(abc))/abar;
         y1=p1*x1+q1;
         z1=p2*x1+q2;
         crossPt1=Point(x1,y1,z1);
         // the second solution
         x2=0.5*(-bbar-sqrt(abc))/abar;
         y2=p1*x2+q1;
         z2=p2*x2+q2;
         crossPt2=Point(x2,y2,z2);
         break;
     }

     // Detect if crossPt1 & crossPt2 are in the triangular element.
     //   Transform and rotate the coordinates of crossPt1 and crossPt2 into
     //   crack-elem coordinates (X', Y' and Z')
     
     Point p1p,p2p,p3p,crossPt1p,crossPt2p;
     p1p     =Point(0.,0.,0.)+T*(pt1-pt1);
     p2p     =Point(0.,0.,0.)+T*(pt2-pt1);
     p3p     =Point(0.,0.,0.)+T*(pt3-pt1);
     crossPt1p=Point(0.,0.,0.)+T*(crossPt1-pt1);
     crossPt2p=Point(0.,0.,0.)+T*(crossPt2-pt1);
     if(PointInTriangle(crossPt1p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt1;
     }
     if(PointInTriangle(crossPt2p,p1p,p2p,p3p)) {
       numCross++;
       crossPt=crossPt2;
     }
   } // End of loop over crack segments

   if(numCross==0)
     return NO;
   else
     return YES;
}

// Find the equation of a plane defined by three points
void Crack::FindPlaneEquation(const Point& p1,const Point& p2,
            const Point& p3, double& a,double& b,double& c,double& d)
{
  // plane equation: ax+by+cz+d=0

  double x21,x31,y21,y31,z21,z31;

  x21=p2.x()-p1.x();
  y21=p2.y()-p1.y();
  z21=p2.z()-p1.z();

  x31=p3.x()-p1.x();
  y31=p3.y()-p1.y();
  z31=p3.z()-p1.z();

  a=y21*z31-z21*y31;
  b=x31*z21-z31*x21;
  c=x21*y31-y21*x31;
  d=-p1.x()*a-p1.y()*b-p1.z()*c;
}

// Detect if a point falls in a triangle (2D case)
short Crack::PointInTriangle(const Point& p,const Point& pt1,
                           const Point& pt2,const Point& pt3)
{
  // y=0 for all points

  double x1,z1,x2,z2,x3,z3,x,z;
  double area_p1p2p,area_p2p3p,area_p3p1p,area_p123;

  x1=pt1.x(); z1=pt1.z();
  x2=pt2.x(); z2=pt2.z();
  x3=pt3.x(); z3=pt3.z();
  x =p.x();   z =p.z();

  area_p1p2p=x1*z2+x2*z+x*z1-x1*z-x2*z1-x*z2;
  area_p2p3p=x2*z3+x3*z+x*z2-x2*z-x3*z2-x*z3;
  area_p3p1p=x3*z1+x1*z+x*z3-x3*z-x1*z3-x*z1;

  area_p123=fabs(x1*z2+x2*z3+x3*z1-x1*z3-x2*z1-x3*z2);

  // Set the area zero if relatively error less than 0.1%
  if(fabs(area_p1p2p)/area_p123<1.e-3) area_p1p2p=0.;
  if(fabs(area_p2p3p)/area_p123<1.e-3) area_p2p3p=0.;
  if(fabs(area_p3p1p)/area_p123<1.e-3) area_p3p1p=0.;

  return (area_p1p2p<=0. && area_p2p3p<=0. && area_p3p1p<=0.);
}


void Crack::OutputCrackFrontResults(const int& m)
{
  if(cfSegNodes[m].size()>0) {
    // Create output file name in format: CrackFrontResults.matXXX
    char outFileName[200]="";
    strcat(outFileName,(udaDir+"/CrackFrontResults.mat").c_str());
      
    char* matbuf=new char[10];
    sprintf(matbuf,"%d",m);
    if(m<10) strcat(outFileName,"00");
    else if(m<100) strcat(outFileName,"0");
    strcat(outFileName,matbuf);
    
    ofstream outCrkFrt(outFileName, ios::app);
	
    /*
    char outFileName0[200];
    char outFileName1[200];
    char outFileName2[200];
    strcpy(outFileName0,outFileName);
    strcpy(outFileName1,outFileName);
    strcpy(outFileName2,outFileName);
    strcat(outFileName0,".0");
    strcat(outFileName1,".1");
    strcat(outFileName2,".2");
    ofstream outCrkFrt0(outFileName0, ios::app);
    ofstream outCrkFrt1(outFileName1, ios::app);
    ofstream outCrkFrt2(outFileName2, ios::app);
    */
    
    double time=d_sharedState->getElapsedTime();
    int timestep=d_sharedState->getCurrentTopLevelTimeStep();

    int num=(int)cfSegNodes[m].size();
    int numSubCracks=0;
    for(int i=0;i<num;i++) {
      if(i==0 || i==num-1 || cfSegPreIdx[m][i]<0) {
        if(i==cfSegMinIdx[m][i]) numSubCracks++;
        int node=cfSegNodes[m][i];
        Point cp=cx[m][node];
	Vector cfPara = Vector(0.,0.,0.);
	if(outputJ==true) cfPara=cfSegJ[m][i];
	else              cfPara=cfSegK[m][i];
        outCrkFrt << setw(5) << timestep
                  << setw(15) << time
                  << setw(5)  << (i-1+2*numSubCracks)/2
                  << setw(10)  << node
                  << setw(15) << cp.x()
                  << setw(15) << cp.y()
                  << setw(15) << cp.z()
                  << setw(15) << cfPara.x()
                  << setw(15) << cfPara.y()
                  << setw(5) << cfPara.z();
        if(cfPara.x()!=0.) 
          outCrkFrt << setw(15) << cfPara.y()/cfPara.x() << endl;
        else 
          outCrkFrt << setw(15) << "inf" << endl;

        if(i==cfSegMaxIdx[m][i]) outCrkFrt << endl;
        /*
	if(i==0) {
          outCrkFrt0 << setw(5) << timestep
	            << setw(15) << time
	            << setw(5)  << (i-1+2*numSubCracks)/2
	            << setw(10)  << node
	            << setw(15) << cp.x()
	            << setw(15) << cp.y()
	            << setw(15) << cp.z()
	            << setw(15) << cfPara.x()
	            << setw(15) << cfPara.y()
	            << setw(5) << cfPara.z();
          if(cfPara.x()!=0.)
	    outCrkFrt0 << setw(15) << cfPara.y()/cfPara.x() << endl;
          else
	    outCrkFrt0 << setw(15) << "inf" << endl;	
	}

	if(i==2) {
	  outCrkFrt1 << setw(5) << timestep
	             << setw(15) << time
	             << setw(5)  << (i-1+2*numSubCracks)/2
	             << setw(10)  << node
	             << setw(15) << cp.x()
	             << setw(15) << cp.y()
 		     << setw(15) << cp.z()
	             << setw(15) << cfPara.x()
                     << setw(15) << cfPara.y()
                     << setw(5) << cfPara.z();
          if(cfPara.x()!=0.)
            outCrkFrt1 << setw(15) << cfPara.y()/cfPara.x() << endl;
          else
            outCrkFrt1 << setw(15) << "inf" << endl;      
        }

        if(i==4) {
	  outCrkFrt2 << setw(5) << timestep
	             << setw(15) << time
	             << setw(5)  << (i-1+2*numSubCracks)/2
	             << setw(10)  << node
	             << setw(15) << cp.x()
	             << setw(15) << cp.y()
		     << setw(15) << cp.z()
	             << setw(15) << cfPara.x()
	             << setw(15) << cfPara.y()
                     << setw(5) << cfPara.z();
          if(cfPara.x()!=0.)
            outCrkFrt2 << setw(15) << cfPara.y()/cfPara.x() << endl;
          else 
            outCrkFrt2 << setw(15) << "inf" << endl;           
        }	  
        */  	
      }
    } // End of loop over i 
  } 
}

