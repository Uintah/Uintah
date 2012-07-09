/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


/********************************************************************************
    Crack.cc
    PART FIVE: CRACK PROPAGATION SIMULATION

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace Uintah;
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

  if(n8or27==27)
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

      constParticleVariable<Matrix3> psize;
      if(n8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);

      if(doCrackPropagation) {
              
        // Step 1: Detect if crack front nodes propagate (cp)
        //         and propagate them virtually (da) for the active nodes
       
        // Clear up cfSegPtsT, which stores temporarily the coordinates 
        // of crack-front nodes after propagation
        cfSegPtsT[m].clear();

        int cfNodeSize= (int) cfSegNodes[m].size();
        short*  cp=new  short[cfNodeSize];
        Vector* da=new Vector[cfNodeSize];

        for(int i=0; i<cfNodeSize; i++) {
          int preIdx=cfSegPreIdx[m][i];
          if(preIdx<0) { // a duplicate node, not operated
            // Direction cosines at the node
            Vector v1=cfSegV1[m][i];
            Vector v2=cfSegV2[m][i];
            Vector v3=cfSegV3[m][i];

            // Coordinate transformation matrix from local to global coordinates
            Matrix3 T=Matrix3(v1.x(), v2.x(), v3.x(),
                              v1.y(), v2.y(), v3.y(),
                              v1.z(), v2.z(), v3.z());

            // Get fracture parameters at crack-front nodes
            double KI  = cfSegK[m][i].x();
            double KII = cfSegK[m][i].y();
            double Vc  = cfSegVel[m][i];
            // If KI less than zero, set it to zero.
            // if(KI<0.) KI=0.;             
             
            // Determine if the node propagates and the propagation direction
            cp[i]=NO;
            double theta;
            if(cm->CrackPropagates(Vc,KI,KII,theta)) cp[i]=YES;

            // Propagate the node virtually
            double dl=rdadx*dx_bar;
            Vector da_local=Vector(dl*cos(theta),dl*sin(theta),0.);
            da[i]=T*da_local;
          } // End of if(!operated)
          else { // if(operated)
            cp[i]=cp[preIdx];
            da[i]=da[preIdx];
          } 
        } // End of loop over cfNodeSize

        
        // Step 2: Propagate crack-front nodes
       
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];

          // Maximum and minimum indexes of the sub-crack which the node resides
          int maxIdx=cfSegMaxIdx[m][i];
          int minIdx=cfSegMinIdx[m][i];

          int preIdx=cfSegPreIdx[m][i];
          if(preIdx<0) { // a duplicate node, not operated
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

            // New position of the node after virtual propagation
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

        
        // Step 3: Deal with the propagating edge nodes,
        //         extending new_pt out to material boundary
       
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];
          Point new_pt=cfSegPtsT[m][i];

          // segments connected by the node
          int segs[2];
          FindSegsFromNode(m,node,segs);

          if((segs[R]<0||segs[L]<0) &&  // Edge nodes
             (new_pt-pt).length()/dx_bar>0.01) {  // It propagates
             
            // Find the direction of the edge crack-front segment 
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
            while(newPtInMat) {
              // Detect which patch new_pt resides in
              short* newPtInPatch=new short[patch_size];
              for(int k=0; k<patch_size; k++) newPtInPatch[k]=NO;
              if(patch->containsPointInExtraCells(new_pt)) newPtInPatch[pid]=YES;
              
              MPI_Barrier(mpi_crack_comm);

              // Detect if new_pt is inside material
              for(int k=0; k<patch_size; k++) {
                if(newPtInPatch[k]) {
                  if(n8or27==8)
                    patch->findCellNodes(new_pt, ni);
                  else if(n8or27==27)
                    patch->findCellNodes27(new_pt, ni);
                  
                  for(int j=0; j<n8or27; j++) {
                    double totalMass=gmass[ni[j]]+Gmass[ni[j]];
                    if(totalMass<d_cell_mass/64.) {
                      newPtInMat=NO;
                      break;
                    }
                  } // End of loop over j
                } // End if(newPtInPatch[k])  
                
                MPI_Bcast(&newPtInMat,1,MPI_SHORT,k,mpi_crack_comm);
              } // End of loop over k   

              delete [] newPtInPatch;
              
              // If new_pt is inside, extend it out by dx_bar/3 each time
              if(newPtInMat) new_pt+=v*(dx_bar/3.);
              
            } // End of while(newPtInMat)
                
            // If new_pt is outside the global grid, trim it
            TrimLineSegmentWithBox(pt2p,new_pt,GLP,GHP);

            // Task 3b: If new_pt is outside, trim it back to MPM material boundary
            if(!newPtInMat) {
              short* newPtInPatch=new short[patch_size];
              for(int k=0; k<patch_size; k++) newPtInPatch[k]=NO;
              if(patch->containsPointInExtraCells(new_pt)) newPtInPatch[pid]=YES;
              
              MPI_Barrier(mpi_crack_comm);            
             
              for(int k=0; k<patch_size; k++) {
                if(newPtInPatch[k]) {
                  // Get cell nodes where new_pt resides
                  if(n8or27==8)
                    patch->findCellNodes(new_pt, ni);
                  else if(n8or27==27)
                    patch->findCellNodes27(new_pt, ni);
                  // Get the lowest and highest points of the cell
                  Point LLP=Point( 9e99, 9e99, 9e99);
                  Point LHP=Point(-9e99,-9e99,-9e99);             
                  for(int j=0; j<n8or27; j++) {
                    Point pj=patch->nodePosition(ni[j]);
                    LLP=Min(LLP,pj);
                    LHP=Max(LHP,pj);
                  }

                  // Trim ptp(or pt2p)->new_pt by the cell
                  Point cross_pt=pt2p; 
                  TrimLineSegmentWithBox(new_pt,cross_pt,LLP,LHP); 

                  // Extend cross_pt a little bit (dx_bar*10%) outside, 
                  // which is the need of crack MPM calculation
                  new_pt=cross_pt+v*(dx_bar*0.1);
                } // End of if(newPtInPatch[k])
                
                MPI_Bcast(&new_pt,1,MPI_POINT,k,mpi_crack_comm);
                
              } // End of loop over k         
              delete [] newPtInPatch;
            } // End of if(!newPtInMat)

            // Save the eventual position of the edge node after propagation
            cfSegPtsT[m][i]=new_pt;
          } 
        } // End of loop cfSegNodes
         
        
        // Step 4: Prune crack-front points if the angle of any two adjacent
        //         segments is larger than a critical angle (ca),
        //         moving it to the mass-center of the three points
        
        double ca=2*csa[m]+15.;
        PruneCrackFrontAfterPropagation(m,ca);  
        
        
        // Step 5: Apply symmetric BCs to new crack-front points
       
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          Point pt=cx[m][cfSegNodes[m][i]];
          ApplySymmetricBCsToCrackPoints(dx,pt,cfSegPtsT[m][i]);
        }

      } // End of if(doCrackPropagation)
    } // End of loop over matls
  } 
}

void Crack::addComputesAndRequiresConstructNewCrackFrontElems(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // delT will be used to calculate crack propagation velocity  
   t->requires(Task::OldDW, d_sharedState->get_delt_label());
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
              
        // Temporary crack-front segment nodes and velocity
        vector<int> cfSegNodesT;
        cfSegNodesT.clear();
        cfSegVel[m].clear();
        
        int ncfSegs= (int) cfSegNodes[m].size()/2;
        int preIdxAtMin=-1;     
        for(int i=0; i<ncfSegs; i++) { 
          // Relations of this seg with the left and right segs
          int preIdx1=cfSegPreIdx[m][2*i];
          int preIdx2=cfSegPreIdx[m][2*i+1];

          // crack-front nodes and coordinates before propagation
          int n1,n2,n1p,n2p,nc,nmc;
          Point p1,p2,p1p,p2p,pc,pmc;
          n1=cfSegNodes[m][2*i];
          n2=cfSegNodes[m][2*i+1];
          p1=cx[m][n1];
          p2=cx[m][n2];

          // crack-front node coordinates after propagaion
          p1p=cfSegPtsT[m][2*i];
          p2p=cfSegPtsT[m][2*i+1];
          pc =p1p+(p2p-p1p)/2.;

          // Detect if it is the first segment of an enclosed crack-front 
          short firstSegOfEnclosedCrack=NO;
          int minIdx=cfSegMinIdx[m][2*i];
          int maxIdx=cfSegMaxIdx[m][2*i+1];
         if(cfSegNodes[m][minIdx]==cfSegNodes[m][maxIdx] && 
            minIdx==(2*i)) firstSegOfEnclosedCrack=YES;

          // Detect if this is the last segment of the crack
          short lastSegment=NO;
          if(maxIdx==(2*i+1)) lastSegment=YES;    
                    
          // length of crack front segment after propagation
          double l12=(p2p-p1p).length();

          
          // Step 2: Determine ten cases of propagation of a segment
          
          short sp=YES, ep=YES;
          if((p1p-p1).length()/dx_bar<0.01) sp=NO; // p1 no propagating
          if((p2p-p2).length()/dx_bar<0.01) ep=NO; // p2 no propagating

          // Calculate crack propagation velocity
          double vc1=0.,vc2=0.,vcc=0.;
          /*      
          double time=d_sharedState->getElapsedTime();
          delt_vartype delT;
          old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
          if(sp) { // Record crack incremental and time instant 
            cfSegDis[m][2*i]=(p1p-p1).length();
            cfSegTime[m][2*i]=time-delT;            
          }
                  
          if(ep) { // Record crack incremental and time instant
            cfSegDis[m][2*i+1]=(p2p-p2).length();
            cfSegTime[m][2*i+1]=time-delT;          
          }
          
          if(time>0.) {
            vc1=cfSegDis[m][2*i]/(time-cfSegTime[m][2*i]);
            vc2=cfSegDis[m][2*i+1]/(time-cfSegTime[m][2*i+1]);
            vcc=(vc1+vc2)/2.;
          }  
          */      
          short CASE=0;             // No propagation
          if(l12/css[m]<0.25) {
            CASE=1;                 // Too short segment, drop it
          }
          else if(l12/css[m]>2.) {  // Too long segment, break it into two
            if( sp && !ep) CASE=5;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=6;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=7;  // Both p1 and p2 propagate
          }
          else {                    // Normal propagation
            if( sp && !ep) CASE=2;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=3;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=4;  // Both p1 and p2 propagate
          }
                    

          // Step 3: Construct new crack elems and crack-front segments
           
          // Detect if the segment is the first segment of a crack
          switch(CASE) { 
            case 0:  // Both ends of the segment do not propagate
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();    
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2);
              
              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vc2);
              
              break;

            case 1: // The segment becomes too short (<25%) after propagation
              // Set the new position of both ends after propagation to pc
              if(preIdx1<0) { // Not generated
                n1p=(int)cx[m].size();
                cx[m].push_back(pc);
              }
              else { // Change p1p to pc 
                n1p=(int)cx[m].size()-1;
                cx[m][n1p]=pc;
              }

              // Accordingly, chnage p1p of the next seg to pc if it is not 
              // the last segment of a crack 
              if(!lastSegment) cfSegPtsT[m][2*(i+1)]=pc;
                                          
              // A new crack elem generated, but no new crack-front segment
              ce[m].push_back(IntVector(n1,n1p,n2));
              
              break;
              
            case 2: // The first end propagates, but the second does not
              // The first end after propagation
              if(preIdx1<0) { // Not generated
                n1p=(int)cx[m].size(); 
                cx[m].push_back(p1p);
              }
              else { // Just generated
                n1p=(int)cx[m].size()-1;
              }

              // The new crack element
              ce[m].push_back(IntVector(n1,n1p,n2));

              // The new crack-front segment
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size(); 
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2);

              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vc2); 
              
              break;

            case 3: // The second end propagates, but the first does not
              // The second end after propagation
              if(preIdx2<0) { // Not generated
                n2p=(int)cx[m].size();
                cx[m].push_back(p2p);
              }
              else { // The last segment of an enclosed crack, p2p has been generated
                n2p=cfSegNodesT[preIdxAtMin];
              }

              // The new crack element
              ce[m].push_back(IntVector(n1,n2p,n2));

              // The new crack-front segment
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();  
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2p);

              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vc2);
              
              break;

            case 4: // Both ends of the segment propagate
              // Three new crack points
              // 1. The first end of the segment
              if(preIdx1<0) { // Not generated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // Just generated
                n1p=(int)cx[m].size()-1;
              }
              
              // 2. The mass center of the quad
              nmc=n1p+1;
              pmc=p1+(p1p-p1)/4.+(p2-p1)/4.+(p2p-p1)/4.;
              cx[m].push_back(pmc);
              
              // 3. The second end of the segment
              if(preIdx2<0) { // Not generated
                n2p=n1p+2;
                cx[m].push_back(p2p);
              }
              else { // The last segment of an enclosed crack, p2p has been generated
                n2p=cfSegNodesT[preIdxAtMin];
              }

              // Four new crack elements
              ce[m].push_back(IntVector(nmc,n2,n1));
              ce[m].push_back(IntVector(nmc,n2p,n2));
              ce[m].push_back(IntVector(nmc,n1,n1p));
              ce[m].push_back(IntVector(nmc,n1p,n2p));
              
              // The new crack-front segment
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();             
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2p);
              
              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vc2);
              
              break;

            case 5: // Too long segment with only the first end propagating   
              // New crack points
              // 1. The first end after propagation
              if(preIdx1<0) { // Not generated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // Just generated
                n1p=(int)cx[m].size()-1;
              }

              // 2. The center of the segment after propagation
              nc=n1p+1;
              cx[m].push_back(pc);

              // Two new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));

              // Two new crack-front segment
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();  
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2);

              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vc2);
              
              break;

            case 6: // Too long segment with only the second end propagating 
              // Two new crack points
              // 1. The center of the sgement after propagation
              nc=(int)cx[m].size();
              cx[m].push_back(pc);

              // 2. The second end after propagation
              if(preIdx2<0) { // Not generated
                n2p=nc+1;
                cx[m].push_back(p2p);
              }
              else { // The last seg of an enclosed crack, p2p has been generated
                n2p=cfSegNodesT[preIdxAtMin];
              }

              // Two new crack elements
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));

              // Two new crack-front segments
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();  
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);
 
              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vc2);
              
              break;

            case 7: // Too long segment with both ends propagating
              // Four new crack points
              // 1. The first end of the segment after propagation
              if(preIdx1<0) { // Not generated
                n1p=(int)cx[m].size();
                cx[m].push_back(p1p);
              }
              else { // Just generated
                n1p=(int)cx[m].size()-1;
              }

              // 2. The center of segment after propagation
              nc=n1p+1;
              cx[m].push_back(pc);
              
              // 3. The mass center of the quad
              nmc=n1p+2;
              pmc=p1+(p1p-p1)/4.+(p2-p1)/4.+(p2p-p1)/4.;
              cx[m].push_back(pmc);
                            
              // 4. The second end of the segment after propagation
              if(preIdx2<0) { // Not generated
                n2p=n1p+3;
                cx[m].push_back(p2p);
              }
              else { // The last segment of an enclosed crack, p2p has been generated
                n2p=cfSegNodesT[preIdxAtMin];
              }
              
              // Five new crack elements
              ce[m].push_back(IntVector(nmc,n2,n1));
              ce[m].push_back(IntVector(nmc,n2p,n2));
              ce[m].push_back(IntVector(nmc,n1,n1p));
              ce[m].push_back(IntVector(nmc,nc,n2p));
              ce[m].push_back(IntVector(nmc,n1p,nc));
                            
              // Two new crack-front segments
              if(firstSegOfEnclosedCrack) preIdxAtMin= (int) cfSegNodesT.size();  
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);

              // Velocity of crack-front nodes
              cfSegVel[m].push_back(vc1);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vcc);
              cfSegVel[m].push_back(vc2);
              
              break;
          }
        } // End of loop over crack-front segs

        MPI_Barrier(mpi_crack_comm);

        // Reset crack-front segment nodes after crack propagation
        cfSegNodes[m].clear();
        for(int i=0; i<(int)cfSegNodesT.size(); i++) {
          cfSegNodes[m].push_back(cfSegNodesT[i]);
        }
        cfSegNodesT.clear();

      } // End of if(doCrackPropagation)
    } // End of loop over matls
  } // End of loop over patches
}

// Find the intersection between a line-segment (p1->p2) and a box
void Crack::TrimLineSegmentWithBox(const Point& p1, Point& p2,
                                const Point& lp, const Point& hp)
{
  // For a box with the lowest and highest points (lp & hp) and
  // a line-seement (p1->p2), p1 is inside the box. If p2 is outside,
  // find the intersection between the line-segment (p1->p2) and the box,
  // and store the intersection in p2.

  Vector v;
  double l,m,n;
  
  // Make sure p1!=p2
  if(p1==p2) {
    cout << "Error: p1=p2=" << p1 << " in Crack::TrimLineSegmentWithBox(...)."
         << " Program is terminated." << endl;
    exit(1);
  }
  else {
    v=TwoPtsDirCos(p1,p2);
    l=v.x(); m=v.y(); n=v.z();
  }

  double xl=lp.x(), yl=lp.y(), zl=lp.z();
  double xh=hp.x(), yh=hp.y(), zh=hp.z();

  double x1=p1.x(), y1=p1.y(), z1=p1.z();
  double x2=p2.x(), y2=p2.y(), z2=p2.z();

  // one-millionth of the diagonal length  of the box
  double d=(hp-lp).length()*1.e-6;

  // Detect if p1 is inside the box
  short p1Outside=YES;
  if(x1>xl-d && x1<xh+d && y1>yl-d && y1<yh+d && z1>zl-d && z1<zh+d) p1Outside=NO;

  if(p1Outside) {
    cout << "Error: p1=" << p1
         << " is outside of the box in Crack::TrimLineSegmentWithBox(): "
         << lp << "-->" << hp << ", where p2=" << p2 << endl;
    cout << " Program terminated." << endl;
    exit(1);
  }

  // If p2 is outside the box, find the intersection
  short p2Outside=YES;
  if(x2>xl-d && x2<xh+d && y2>yl-d && y2<yh+d && z2>zl-d && z2<zh+d) p2Outside=NO;

  while(p2Outside) {
    if(x2>xh || x2<xl) {
      if(x2>xh) x2=xh;
      if(x2<xl) x2=xl;
      if(l>1.e-6) {
        y2=y1+m*(x2-x1)/l;
        z2=z1+n*(x2-x1)/l;
      }
    }
    else if(y2>yh || y2<yl) {
      if(y2>yh) y2=yh;
      if(y2<yl) y2=yl;
      if(m>1.e-6) {
        x2=x1+l*(y2-y1)/m;
        z2=z1+n*(y2-y1)/m;
      }
    }
    else if(z2>zh || z2<zl) {
      if(z2>zh) z2=zh;
      if(z2<zl) z2=zl;
      if(n>1.e-6) {
        x2=x1+l*(z2-z1)/n;
        y2=y1+m*(z2-z1)/n;
      }
    }

    if(x2>xl-d && x2<xh+d && y2>yl-d && y2<yh+d && z2>zl-d && z2<zh+d) p2Outside=NO;

  } // End of while(!p2Inside)

  p2=Point(x2,y2,z2);
}

void Crack::PruneCrackFrontAfterPropagation(const int& m, const double& ca)
{
  // If the angle between two line-segments connected by
  // a point is larger than a certain value (ca), move the point to
  // the mass center of the triangle

  int num=(int)cfSegNodes[m].size();
  vector<Point> cfSegPtsPruned;
  cfSegPtsPruned.resize(num);

  for(int i=0; i<num; i++) {
    cfSegPtsPruned[i]=cfSegPtsT[m][i];
  }

  for(int i=0; i<(int)cfSegNodes[m].size(); i++) {
    int preIdx=cfSegPreIdx[m][i];
    if(preIdx<0) { // not operated
      if(i>cfSegMinIdx[m][i] && i<cfSegMaxIdx[m][i]) {
        Point p =cfSegPtsT[m][i];
        Point p1=cfSegPtsT[m][i-1];
        Point p2=cfSegPtsT[m][i+2];
        Vector v1=TwoPtsDirCos(p1,p);
        Vector v2=TwoPtsDirCos(p,p2);
        double theta=acos(Dot(v1,v2))*180/3.141592654;
        if(fabs(theta)>ca) {
          cfSegPtsPruned[i]=p+(p1-p)/3.+(p2-p)/3.;
        }
      } // End of if(i>minIdx && i<maxIdx)
    }
    else { // operated
      cfSegPtsPruned[i]=cfSegPtsPruned[preIdx];
    }
  } // End of loop over i

  for(int i=0; i<num; i++) {
    cfSegPtsT[m][i]=cfSegPtsPruned[i];
  }
  cfSegPtsPruned.clear();
}
