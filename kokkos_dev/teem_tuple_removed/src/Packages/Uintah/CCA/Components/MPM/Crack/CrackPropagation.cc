/********************************************************************************
    Crack.cc
    PART FIVE: CRACK PROPAGATION SIMULATION

    Created by Yajun Guo in 2002-2004.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
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
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/NotFinished.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;

#define MAX_BASIS 27

void Crack::addComputesAndRequiresPropagateCrackFrontPoints(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  int NGC=2*NGN;
  t->requires(Task::NewDW, lb->gMassLabel, gac, NGC);
  t->requires(Task::NewDW, lb->GMassLabel, gac, NGC);

  if(d_8or27==27)
   t->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
}

void Crack::PropagateCrackFrontPoints(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_bar=(dx.x()+dx.y()+dx.z())/3.;

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);
    
    MPI_Datatype MPI_POINT=fun_getTypeDescription((Point*)0)->getMPIType();

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

      // Cell mass of the material
      double d_cell_mass=mpm_matl->getInitialDensity()*dx.x()*dx.y()*dx.z(); 
      
      // Get nodal mass information
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      Ghost::GhostType  gac = Ghost::AroundCells;
      constNCVariable<double> gmass,Gmass;
      int NGC=2*NGN;
      new_dw->get(gmass, lb->gMassLabel, dwi, patch, gac, NGC);
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, gac, NGC);

      constParticleVariable<Vector> psize;
      if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);

      if(doCrackPropagation) {
        /* Step 1: Detect if crack front nodes propagate (cp)
           and propagate them virtually (da) for the active nodes
        */
        // Clear up cfSegPtsT -- crack-front points after propagation
        cfSegPtsT[m].clear();

        int cfNodeSize=cfSegNodes[m].size();
        short*  cp=new  short[cfNodeSize];
        Vector* da=new Vector[cfNodeSize];

        for(int i=0; i<cfNodeSize; i++) {
          int preIdx=cfSegPreIdx[m][i];
          if(preIdx<0) { // Not operated
            // Direction-cosines at the node
            Vector v1=cfSegV1[m][i];
            Vector v2=cfSegV2[m][i];
            Vector v3=cfSegV3[m][i];

            // Coordinates transformation matrix from local to global
            Matrix3 T=Matrix3(v1.x(), v2.x(), v3.x(),
                              v1.y(), v2.y(), v3.y(),
                              v1.z(), v2.z(), v3.z());

            // Determine if the node propagates
            cp[i]=NO;
            double KI  = cfSegK[m][i].x();
            double KII = cfSegK[m][i].y();
            if(cm->CrackSegmentPropagates(KI,KII)) cp[i]=YES;

            // Propagate the node virtually
            double theta=cm->GetPropagationDirection(KI,KII);
            double dl=rdadx*dx_bar;
            Vector da_local=Vector(dl*cos(theta),dl*sin(theta),0.);
            da[i]=T*da_local;
          } // End of if(!operated)
          else { // if(operated)
            cp[i]=cp[preIdx];
            da[i]=da[preIdx];
          }  // End of if(!operated) {} else {}
        } // End of loop over cfNodeSize

        /* Step 2: Propagate crack-front nodes
        */
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];

          // Maximum and minimum indexes of the sub-crack of the node
          int maxIdx=cfSegMaxIdx[m][i];
          int minIdx=cfSegMinIdx[m][i];

          int preIdx=cfSegPreIdx[m][i];
          if(preIdx<0) { // Not operated
            // Count the nodes which propagate among (2ns+1) nodes around pt
  	    int nsegs=(maxIdx-minIdx+1)/2; 
            int ns=(nsegs+8)/8;
            // ns=1 for 1-7 segs; ns=2 for 8-15 segs; ...
            int np=0;
            for(int j=-ns; j<=ns; j++) {
              int cIdx=i+2*j;
              if(cIdx<minIdx && cp[minIdx]) np++;
              if(cIdx>maxIdx && cp[maxIdx]) np++;
              if(cIdx>=minIdx && cIdx<=maxIdx && cp[cIdx]) np++;
            }

            // New position of pt after virtual propagation
            double fraction=(double)np/(2*ns+1);
            Point new_pt=pt+fraction*da[i];
            cfSegPtsT[m].push_back(new_pt);
          } // End if(!operated)
          else {
            Point prePt=cfSegPtsT[m][preIdx];
            cfSegPtsT[m].push_back(prePt);
          }
        } // End of loop cfSegNodes

        // Release dynamic arraies
        delete [] cp;
        delete [] da;

        /* Step 3: Deal with propagating edge nodes which is prpagating,
                   extendinga new_pt out to the boundary 
        */
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];
          Point new_pt=cfSegPtsT[m][i];

          // segments connected by the node
          int segs[2];
          FindSegsFromNode(m,node,segs);

          if((segs[R]<0||segs[L]<0) &&  // Edge nodes
             (new_pt-pt).length()/dx_bar>0.01) {  // It propagates
	     
	    // Find the direction of the edge crack-front line segemnt 
            Point ptp=cfSegPtsT[m][i];
            Point pt2p;      
	    if(segs[R]<0) { // right edge nodes
   	      pt2p=cfSegPtsT[m][i+1];
	    }
	    else if(segs[L]<0) { // left edge nodes
              pt2p=cfSegPtsT[m][i-1];
	    }
	    Vector v=TwoPtsDirCos(pt2p,ptp);
				
            IntVector ni[MAX_BASIS];
	    
            // Task 3a: Extend new_pt to the outside of the material
            short newPtInMat=YES;
            int extTimes=0;

	    while(newPtInMat) {
	      MPI_Status status;
	      int tag=node+extTimes;
	      // See if new_pt is inside		    	  
	      if(patch->containsPoint(new_pt)) {
                if(d_8or27==8)
                  patch->findCellNodes(new_pt, ni);
                else if(d_8or27==27)
                  patch->findCellNodes27(new_pt, ni);
		  
                for(int k = 0; k < d_8or27; k++) {
                  double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                  if(totalMass<d_cell_mass/64.) {
                    newPtInMat=NO;
                    break;
                  }
                }

	        // Send newPtInMat to all other processors
		for(int k=0; k<patch_size; k++) {
		  if(k!=pid) MPI_Send(&newPtInMat,1,MPI_SHORT,k,tag,mpi_crack_comm);
		}  
	      } 
	      else { // Processors which do not include new_pt, receive newPtInMat
	        MPI_Recv(&newPtInMat,1,MPI_SHORT,MPI_ANY_SOURCE,tag,
	                   mpi_crack_comm, &status);
	      }  
		
	      // If new_pt is inside, extend it out by dx_bar/3 each time
              if(newPtInMat) { 
                new_pt+=v*(dx_bar/3.);
	        extTimes++;
     	      } // end of if(newPtInMat)
            } // End of while(newPtInMat)
		
            // If new_pt is outside the global grid, trim it
            TrimLineSegmentWithBox(pt2p,new_pt,GLP,GHP);

            // Task 3b: If new_pt is outside, trim it back to the material boundary
	    if(!newPtInMat) {
	      short* newPtInPatch=new short[patch_size];
	      for(int k=0; k<patch_size; k++) newPtInPatch[k]=NO;
	      if(patch->containsPoint(new_pt)) newPtInPatch[pid]=YES;
	      
	      MPI_Barrier(mpi_crack_comm);	      
             
	      for(int k=0; k<patch_size;k++) {
	        if(newPtInPatch[k]) {
                  // Get cell nodes where new_pt is
	          if(d_8or27==8)
	            patch->findCellNodes(new_pt, ni);
	          else if(d_8or27==27)
                    patch->findCellNodes27(new_pt, ni);
                  // Get the lowest and highest points of the cell
	          Point LLP=Point( 9e99, 9e99, 9e99);
	          Point LHP=Point(-9e99,-9e99,-9e99);		  
	          for(int j=0; j<d_8or27; j++) {
	            Point pj=patch->nodePosition(ni[j]);
	            LLP=Min(LLP,pj);
	            LHP=Max(LHP,pj);
	          }

                  // Trim ptp(or pt2p)->new_pt by the cell
		  Point cross_pt=pt2p; 
		  TrimLineSegmentWithBox(new_pt,cross_pt,LLP,LHP); 

		  // Extend cross_pt a little bit (dx_bar*10%) outside 
		  new_pt=cross_pt+v*(dx_bar*0.1);
		} 
		MPI_Bcast(&new_pt,1,MPI_POINT,k,mpi_crack_comm);
	      } // End of loop over k         
	      delete [] newPtInPatch;
	    } // End of if(!newPtInMat)

	    // Save the eventual position of the edge node after propagation
            cfSegPtsT[m][i]=new_pt;
          } // End of if this is a edge node
        } // End of loop cfSegNodes
	 
        /* Step 4: Prune crack-front points if the direction of the two segments 
	           connected by any point is larger than a critical angle (ca),
		   moving it to the mass-center of the three points
	*/
	double ca=3*csa[m]+15.;
        PruneCrackFrontAfterPropagation(m,ca);	
	
        /* Step 5: Apply symmetric BCs to new crack-front points
        */
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          Point pt=cx[m][cfSegNodes[m][i]];
          ApplySymmetricBCsToCrackPoints(dx,pt,cfSegPtsT[m][i]);
        }

      } // End of if(doCrackPropagation)
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresConstructNewCrackFrontElems(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Nothing to do currently
}

void Crack::ConstructNewCrackFrontElems(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{
  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_bar=(dx.x()+dx.y()+dx.z())/3.;
    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m=0; m<numMPMMatls; m++) {
      if(doCrackPropagation) {
        /*
        // Step 1: Combine crack front nodes if they propagates
        // a little (<10%) or in self-similar way (angle<10 degree)
        for(int i=0; i<(int)cfSegNodes[m].size(); i++) {
           // Crack-front node and normal
           int node=cfSegNodes[m][i];
           Vector v2=cfSegV2[m][i];

           // Crack propagation increment(dis) and direction(vp)
           double dis=(cfSegPtsT[m][i]-cx[m][node]).length();
           Vector vp=TwoPtsDirCos(cx[m][node],cfSegPtsT[m][i]);
           // Crack propa angle(in degree) measured from crack plane
           double angle=90-acos(Dot(vp,v2))*180/3.141592654;
           if(dis<0.1*(rdadx*dx_bar) || fabs(angle)<5)
             cx[m][node]=cfSegPtsT[m][i];
        }
        */
        // Temporary crack-front segment nodes
        vector<int> cfSegNodesT;
        cfSegNodesT.clear();

        int ncfSegs=cfSegNodes[m].size()/2;
        int preIdx1T=-1;
        for(int i=0; i<ncfSegs; i++) { // Loop over front segs
          // Relations of this seg with the left and right segs
          int preIdx1=cfSegPreIdx[m][2*i];
          int preIdx2=cfSegPreIdx[m][2*i+1];
          if(preIdx1<0) { // right edge nodes
            preIdx1T=cfSegNodesT.size();
          }

          // crack front nodes and points
          int n1,n2,n1p,n2p,nc;
          Point p1,p2,p1p,p2p,pc;
          n1=cfSegNodes[m][2*i];
          n2=cfSegNodes[m][2*i+1];
          p1=cx[m][n1];
          p2=cx[m][n2];

          p1p=cfSegPtsT[m][2*i];
          p2p=cfSegPtsT[m][2*i+1];
          pc =p1p+(p2p-p1p)/2.;

          // length of crack front segment after propagation
          double l12=(p2p-p1p).length();

          // Step 2: Determine ten cases of propagation
          short sp=YES, ep=YES;
          if((p1p-p1).length()/dx_bar<0.01) sp=NO; // p1 no propagating
          if((p2p-p2).length()/dx_bar<0.01) ep=NO; // p2 no propagating

          short CASE=0;        // no propagation
          if(l12/css[m]<0.5) { // Adjust or combine the seg
            if( !sp && ep) CASE=1;  // p2 propagates, p1 doesn't
            if( sp && !ep) CASE=2;  // p1 propagates, p2 doesn't
            if( sp &&  ep) CASE=3;  // Both of p1 and p2 propagate
          }
          else if(l12/css[m]<2.) {// Keep the seg
            if( sp && !ep) CASE=4;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=5;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=6;  // Both of p1 and p2 propagate
          }
          else { // Break the seg into two
            if( sp && !ep) CASE=7;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=8;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=9;  // Both of p1 and p2 propagate
          }

          //if(CASE>=1 && CASE<=3 && patch->getID()==0)
          //  cout << "   ! Crack-front seg " << i << "(mat " << m
          //      << ") is combined into the next segs (CASE "
          //      << CASE << ")" << endl;
          //if(CASE>=7 && patch->getID()==0)
          //  cout << "   ! Crack-front seg " << i << "(mat " << m
          //      << ") is split into two segs (CASE "
          //      << CASE << ")" << endl;

          // Step 3: Construct new crack elems and crack-front segs
          // Detect if the seg is the first seg of a crack
          switch(CASE) { 
            case 0:
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2);
              break;
            case 1:
              n2p=(int)cx[m].size();
              // Modify p2p to the center of p1p and p2p of next seg
              if(i<ncfSegs-1)  p2p=cfSegPtsT[m][2*i]+
                  (cfSegPtsT[m][2*(i+1)+1]-cfSegPtsT[m][2*i])/2.;
              cx[m].push_back(p2p);
              cfSegPtsT[m][2*(i+1)]=p2p;
              // the new crack element
              ce[m].push_back(IntVector(n1,n2p,n2));
              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2p);
              break;

            case 2:
            case 3:
              // new crack points
              if(preIdx1<0) {
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // operated
                n1p=(int)cx[m].size()-1;
              }

              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,n2));

              // Move cfSegPtsT[m][2*(i+1)]
              if(i<ncfSegs-1) cfSegPtsT[m][2*(i+1)]=cfSegPtsT[m][2*i];
              break;

            case 4:
              // new crack points
              if(preIdx1<0) { // not operated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // operated
                n1p=(int)cx[m].size()-1;
              }

              // the new crack element
              ce[m].push_back(IntVector(n1,n1p,n2));

              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2);
              break;

            case 5:
              // new crack points
              if(preIdx2<0) { // not operated
                n2p=(int)cx[m].size();
                cx[m].push_back(p2p);
              }
              else { // operated
                n2p=cfSegNodesT[preIdx1T];
              }

              // the new crack element
              ce[m].push_back(IntVector(n1,n2p,n2));

              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2p);
              break;

            case 6:
              // new crack points
              if(preIdx1<0) { // not operated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // operated
                n1p=(int)cx[m].size()-1;
              }

              if(preIdx2<0) { // not operated
                n2p=n1p+1;
                cx[m].push_back(p2p);
              }
              else { // operated
                n2p=cfSegNodesT[preIdx1T];
              }

              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,n2));
              ce[m].push_back(IntVector(n1p,n2p,n2));
              // the new crack front-seg nodes

              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2p);
              break;

            case 7:
              // new crack points
              if(preIdx1<0) { // not operated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // operated
                n1p=(int)cx[m].size()-1;
              }

              nc=n1p+1;
              cx[m].push_back(pc);

              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));

              // crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2);
              break;

            case 8:
              // new crack points
              nc=(int)cx[m].size();
              cx[m].push_back(pc);

              if(preIdx2<0) { // not operated
                n2p=nc+1;
                cx[m].push_back(p2p);
              }
              else { // operated
                n2p=cfSegNodesT[preIdx1T];
              }

              // the new crack elements
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));

              // crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);
              break;

            case 9:
              // new crack points
              if(preIdx1<0) { // not operated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // operated
                n1p=(int)cx[m].size()-1;
              }

              nc=n1p+1;
              cx[m].push_back(pc);

              if(preIdx2<0) { // not operated
                n2p=n1p+2;
                cx[m].push_back(p2p);
              }
              else { // operated
                n2p=cfSegNodesT[preIdx1T];
              }

              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));

              // crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);
              break;
          }
        } // End of loop over crack front segs

        MPI_Barrier(mpi_crack_comm);

        // Reset crack front segment nodes after crack propagation
        cfSegNodes[m].clear();
        for(int i=0; i<(int)cfSegNodesT.size(); i++) {
          cfSegNodes[m].push_back(cfSegNodesT[i]);
        }
        cfSegNodesT.clear();

      } // End of if(doCrackPropagation)
    } // End of loop over matls
  } // End of loop over patches
}

