/********************************************************************************
    Crack.cc
    PART FIVE: CRACK PROPAGATION SIMULATION

    Created by Yajun Guo in 2002-2004.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
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

void Crack::addComputesAndRequiresRecollectCrackFrontNodes(Task* t,
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

void Crack::RecollectCrackFrontNodes(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

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
        // cfSegPtsT -- crack front points after propagation
        // cfSegPtsT[m].clear();

        IntVector ni[MAX_BASIS];
        double S[MAX_BASIS];

        /* Task 1: Detect if crack-front nodes are inside of materials
        */
        vector<short> cfSegNodesInMat;
        cfSegNodesInMat.resize(cfSegNodes[m].size());
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          cfSegNodesInMat[i]=YES;
        }

        for(int i=0; i<patch_size; i++) {
          int num=cfnset[m][i].size();
          short* inMat=new short[num];

          if(pid==i) { // Rank i does it
            for(int j=0; j<num; j++) {
              int idx=cfnset[m][i][j];
              int node=cfSegNodes[m][idx];
              Point pt=cx[m][node];
              inMat[j]=YES;

              // Get the node indices that surround the cell
              if(d_8or27==8)
                patch->findCellAndWeights(pt, ni, S);
              else if(d_8or27==27)
                patch->findCellAndWeights27(pt, ni, S, psize[j]);

              for(int k = 0; k < d_8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<5*d_SMALL_NUM_MPM) {
                  inMat[j]=NO;
                  break;
                }
              }
            } // End of loop over j
          } // End of if(pid==i)

          MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

          for(int j=0; j<num; j++) {
            int idx=cfnset[m][i][j];
            cfSegNodesInMat[idx]=inMat[j];
          }
          delete [] inMat;
        } // End of loop over patches

        MPI_Barrier(mpi_crack_comm);

        /* Task 1a: Detect if crack-front-segment center is inside of
                   materials for single crack-front-seg problems
        */
        int ncfSegs=(int)cfSegNodes[m].size()/2;
        short cfSegCenterInMat=YES;
        if(ncfSegs==1) {
          int workID=-1;
          for(int i=0; i<patch_size; i++) {
            if(cfsset[m][i].size()==1) {workID=i; break;}
          }
          
          if(pid==workID) {
            int nd1=cfSegNodes[m][0];
            int nd2=cfSegNodes[m][1];
            Point cent=cx[m][nd1]+(cx[m][nd2]-cx[m][nd1])/2.;

            // Get the node indices that surround the cell
            if(d_8or27==8)
              patch->findCellAndWeights(cent, ni, S);
            else if(d_8or27==27)
              patch->findCellAndWeights27(cent, ni, S, psize[0]);

            for(int k = 0; k < d_8or27; k++) {
              double totalMass=gmass[ni[k]]+Gmass[ni[k]];
              if(totalMass<5*d_SMALL_NUM_MPM) {
                cfSegCenterInMat=NO;
                break;
              }
            }
          } // End of if(pid==workID) 

          MPI_Bcast(&cfSegCenterInMat,1,MPI_SHORT,workID,mpi_crack_comm);
        } // End of if(ncfSegs==1)

        MPI_Barrier(mpi_crack_comm);

        /* Task 2: Recollect crack-front segs, discarding the dead ones
        */ 
        // Store crack-front parameters in temporary arraies
        int old_size=(int)cfSegNodes[m].size();
        int*  cfSegNodeT = new int[old_size];
        Vector*  cfSegJT = new Vector[old_size];
        Vector*  cfSegKT = new Vector[old_size];
        Vector* cfSegV1T = new Vector[old_size];
        Vector* cfSegV2T = new Vector[old_size];
        Vector* cfSegV3T = new Vector[old_size];
        
        for(int i=0; i<old_size; i++) {
          cfSegNodeT[i]=cfSegNodes[m][i];
          cfSegJT[i]=cfSegJ[m][i];
          cfSegKT[i]=cfSegK[m][i];
          cfSegV1T[i]=cfSegV1[m][i];
          cfSegV2T[i]=cfSegV2[m][i];
          cfSegV3T[i]=cfSegV3[m][i];
        }
        cfSegNodes[m].clear();
        cfSegJ[m].clear();
        cfSegK[m].clear();
        cfSegV1[m].clear();
        cfSegV2[m].clear();
        cfSegV3[m].clear();

        // Collect the active crack-front segs
        for(int i=0; i<old_size/2; i++) { // Loop over crack-front segs
          short thisSegActive=NO;
          int nd1=cfSegNodeT[2*i];
          int nd2=cfSegNodeT[2*i+1];
          if(old_size/2==1) { // for single seg problems
            // Remain active if any of two ends and center are inside
            if(cfSegNodesInMat[2*i] || cfSegNodesInMat[2*i+1] ||
               cfSegCenterInMat) thisSegActive=YES;
          }
          else { // for multiple seg problems
            // Remain active if any of two ends is inside
            if(cfSegNodesInMat[2*i] || cfSegNodesInMat[2*i+1])
              thisSegActive=YES;
          }
          if(thisSegActive) { // Collect parameters of the node
            cfSegNodes[m].push_back(nd1);
            cfSegNodes[m].push_back(nd2);
            cfSegJ[m].push_back(cfSegJT[2*i]);
            cfSegJ[m].push_back(cfSegJT[2*i+1]);
            cfSegK[m].push_back(cfSegKT[2*i]);
            cfSegK[m].push_back(cfSegKT[2*i+1]);
            cfSegV1[m].push_back(cfSegV1T[2*i]);
            cfSegV1[m].push_back(cfSegV1T[2*i+1]);
            cfSegV2[m].push_back(cfSegV2T[2*i]);
            cfSegV2[m].push_back(cfSegV2T[2*i+1]);
            cfSegV3[m].push_back(cfSegV3T[2*i]);
            cfSegV3[m].push_back(cfSegV3T[2*i+1]);
          }
          else { // The segment is dead
            if(pid==0) {
              cout << "   ! Crack-front seg " << i << "(" << nd1
                   << cx[m][nd1] << "-->" << nd2 << cx[m][nd2]
                   << ") of Mat " << m << " is dead." << endl;
            }
          }
        } // End of loop over crack-front segs
        delete [] cfSegNodeT;
        delete [] cfSegJT;
        delete [] cfSegKT;
        delete [] cfSegV1T;
        delete [] cfSegV2T;
        delete [] cfSegV3T;

        // If all crack-front segs dead, the material is broken.
        if(cfSegNodes[m].size()/2<=0 && pid==0) {
         cout << "   !!! Material " << m
              << " is broken. Program terminated." << endl;
         exit(1);
        } 

      } // End of if(doCrackPropagation)

    } // End of loop over matls

  } // End of loop over patches
}

void Crack::addComputesAndRequiresPropagateCrackFrontNodes(Task* t,
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

void Crack::PropagateCrackFrontNodes(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dx_max=Max(dx.x(),dx.y(),dx.z());

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();

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

        // Get crack-front-node previous index
        FindCrackFrontNodePreIdx(m);

        int cfNodeSize=cfSegNodes[m].size();
        short*  cp=new  short[cfNodeSize];
        Vector* da=new Vector[cfNodeSize];

        for(int i=0; i<cfNodeSize; i++) {
          int pre_idx=cfSegPreIdx[m][i];
          if(pre_idx<0) { // Not operated
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
            double dl=rdadx*dx_max;
            Vector da_local=Vector(dl*cos(theta),dl*sin(theta),0.);
            da[i]=T*da_local;
          } // End of if(!operated)
          else { // if(operated)
            cp[i]=cp[pre_idx];
            da[i]=da[pre_idx];
          }  // End of if(!operated) {} else {}
        } // End of loop over cfNodeSize

        /* Step 2: Determine the propagation extent for each node
        */
        int min_idx=0,max_idx=-1;
        for(int i=0; i<cfNodeSize; i++) {
          int node=cfSegNodes[m][i];
          Point pt=cx[m][node];

          int pre_idx=cfSegPreIdx[m][i];
          if(pre_idx<0) { // Not operated
            // Find the segments coonected by the node
            int segs[2];
            FindSegsFromNode(m,node,segs);
            //  Find the minimum and maximum indices of this crack
            if(i>max_idx) {
              int segsT[2];
              min_idx=i;
              segsT[R]=segs[R];
              while(segsT[R]>=0 && min_idx>0)
                FindSegsFromNode(m,cfSegNodes[m][--min_idx],segsT);
              max_idx=i;
              segsT[L]=segs[L];
              while(segsT[L]>=0 && max_idx<cfNodeSize-1)
                FindSegsFromNode(m,cfSegNodes[m][++max_idx],segsT);
            }

            // Count the nodes which propagate among (2ns+1) nodes around pt
            int ns=(max_idx-min_idx+1)/10+2;
            int np=0;
            for(int j=-ns; j<=ns; j++) {
              int cidx=i+2*j;
              if(cidx<min_idx && cp[min_idx]) np++;
              if(cidx>max_idx && cp[max_idx]) np++;
              if(cidx>=min_idx && cidx<=max_idx && cp[cidx]) np++;
            }

            // New position of pt after virtual propagation
            double fraction=(double)np/(2*ns+1);
            Point new_pt=pt+fraction*da[i];

            /* Step 3: Deal with the boundary nodes: extending new_pt
                       out to the boundary by (fraction*rdadx*dx_max)
                       if it is inside of material
            */
            if((segs[R]<0||segs[L]<0) &&  // Edge nodes
               (new_pt-pt).length()/dx_max>0.01) {  // propagate
              // Check if new_pt is inside of the material
              short newPtInMat=YES;

              // Processor ID (workID) of the node before propagation
              int workID=-1;
              for(int i1=0; i1<patch_size; i1++) {
                for(int j1=0; j1<(int)cfnset[m][i1].size(); j1++){
                  int nodeij=cfSegNodes[m][cfnset[m][i1][j1]];
                  if(node==nodeij) { workID=i1; break;}
                }
              }

              // Detect if new_pt is inside of material
              if(pid==workID) {
                IntVector ni[MAX_BASIS];
                double S[MAX_BASIS];
                if(d_8or27==8)
                  patch->findCellAndWeights(new_pt, ni, S);
                else if(d_8or27==27)
                  patch->findCellAndWeights27(new_pt, ni, S, psize[0]);
                for(int k = 0; k < d_8or27; k++) {
                  double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                  if(totalMass<5*d_SMALL_NUM_MPM) {
                    newPtInMat=NO;
                    break;
                  }
                }
              }
              MPI_Bcast(&newPtInMat,1,MPI_SHORT,workID,mpi_crack_comm);

              if(newPtInMat) { // If it's inside of material, extend it out
                Point tmp_pt=new_pt;
                int n1=-1,n2=-1;
                if(segs[R]<0) { // right edge nodes
                  n1=cfSegNodes[m][2*segs[L]+1];
                  n2=cfSegNodes[m][2*segs[L]];
                }
                else if(segs[L]<0) { // left edge nodes
                  n1=cfSegNodes[m][2*segs[R]];
                  n2=cfSegNodes[m][2*segs[R]+1];
                }

                Vector v=TwoPtsDirCos(cx[m][n1],cx[m][n2]);
                new_pt=tmp_pt+v*(fraction*rdadx*dx_max);

                // Check if it beyond the global gird
                FindIntersectionLineAndGridBoundary(tmp_pt,new_pt);
              }
            }

            // Push back the new_pt
            cfSegPtsT[m].push_back(new_pt);
          } // End if(!operated)
          else {
            Point pre_pt=cfSegPtsT[m][pre_idx];
            cfSegPtsT[m].push_back(pre_pt);
          }
        } // End of loop cfSegNodes

        /* Step 4: Apply symmetric BCs to new crack-front points
        */
        for(int i=0; i<(int)cfSegNodes[m].size();i++) {
          Point pt=cx[m][cfSegNodes[m][i]];
          ApplySymmetricBCsToCrackPoints(dx,pt,cfSegPtsT[m][i]);
        }

        // Release dynamic arraies
        delete [] cp;
        delete [] da;
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
    double dx_max=Max(dx.x(),dx.y(),dx.z());

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
           if(dis<0.1*(rdadx*dx_max) || fabs(angle)<5)
             cx[m][node]=cfSegPtsT[m][i];
        }
*/
        // Clear up the temporary crack front segment nodes
        vector<int> cfSegNodesT;
        cfSegNodesT.clear();

        int ncfSegs=cfSegNodes[m].size()/2;
        for(int i=0; i<ncfSegs; i++) { // Loop over front segs
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
          if((p1p-p1).length()/dx_max<0.01) sp=NO; // p1 no propagating
          if((p2p-p2).length()/dx_max<0.01) ep=NO; // p2 no propagating

          short CASE=0;        // no propagation
          if(l12/cs0[m]<0.5) { // Adjust or combine the seg
            if( !sp && ep) CASE=1;  // p2 propagates, p1 doesn't
            if( sp && !ep) CASE=2;  // p1 propagates, p2 doesn't
            if( sp &&  ep) CASE=3;  // Both of p1 and p2 propagate
          }
          else if(l12/cs0[m]<2.) {// Keep the seg
            if( sp && !ep) CASE=4;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=5;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=6;  // Both of p1 and p2 propagate
          }
          else { // Break the seg into two
            if( sp && !ep) CASE=7;  // p1 propagates, p2 doesn't
            if(!sp &&  ep) CASE=8;  // p2 propagates, p1 doesn't
            if( sp &&  ep) CASE=9;  // Both of p1 and p2 propagate
          }

          if(CASE>=1 && CASE<=3 && patch->getID()==0)
            cout << "   ! Crack-front seg " << i << "(mat " << m
                << ") is combined into the next segs (CASE "
                << CASE << ")" << endl;
          if(CASE>=7 && patch->getID()==0)
            cout << "   ! Crack-front seg " << i << "(mat " << m
                << ") is split into two segs (CASE "
                << CASE << ")" << endl;

          // Step 3: Construct new crack elems and crack-front segs
          // Detect if the seg is the first seg of a crack
          short firstSeg=YES;
          if(i>0 && n1==cfSegNodes[m][2*(i-1)+1]) firstSeg=NO;

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
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,n2));
              // Move cfSegPtsT[m][2*(i+1)]
              if(i<ncfSegs-1) cfSegPtsT[m][2*(i+1)]=cfSegPtsT[m][2*i];
              break;
            case 4:
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              // the new crack element
              ce[m].push_back(IntVector(n1,n1p,n2));
              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2);
              break;
            case 5:
              n2p=(int)cx[m].size();
              // the new crack point
              cx[m].push_back(p2p);
              // the new crack element
              ce[m].push_back(IntVector(n1,n2p,n2));
              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(n2p);
              break;
            case 6:
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              n2p=n1p+1;
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,n2));
              ce[m].push_back(IntVector(n1p,n2p,n2));
              // the new crack front-seg nodes
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(n2p);
              break;
            case 7:
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              nc=n1p+1;
              cx[m].push_back(pc);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));
              // the new crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2);
              break;
            case 8:
              nc=(int)cx[m].size();
              n2p=nc+1;
              // the new crack points
              cx[m].push_back(pc);
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));
              // the new crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);
              break;
            case 9:
              // the new crack point
              if(!firstSeg) n1p=(int)cx[m].size()-1;
              else {n1p=(int)cx[m].size(); cx[m].push_back(p1p);}
              nc =n1p+1;
              n2p=n1p+2;
              cx[m].push_back(pc);
              cx[m].push_back(p2p);
              // the new crack elements
              ce[m].push_back(IntVector(n1,n1p,nc));
              ce[m].push_back(IntVector(n1,nc,n2));
              ce[m].push_back(IntVector(n2,nc,n2p));
              // the new crack front-seg nodes, a new seg generated
              cfSegNodesT.push_back(n1p);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(nc);
              cfSegNodesT.push_back(n2p);
              break;
          }
        } // End of loop over crack front segments

        MPI_Barrier(mpi_crack_comm);

        // Reset crack front segment nodes after crack propagation
        cfSegNodes[m].clear();
        for(int i=0; i<(int)cfSegNodesT.size(); i++) {
          cfSegNodes[m].push_back(cfSegNodesT[i]);
        }

        // Get carck-front-node previous index
        FindCrackFrontNodePreIdx(m);

      } // End of if(doCrackpropagation)
    } // End of loop over matls

  } // End of loop over patches
}

