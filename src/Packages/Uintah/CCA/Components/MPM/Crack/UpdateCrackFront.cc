/********************************************************************************
    Crack.cc
    PART SEVEN: UPDATE CRACK FRONT AND CALCULATE CRACK-FRONT NORMALS

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

void Crack::addComputesAndRequiresCrackFrontNodeSubset(Task* /*t*/,
                                const PatchSet* /*patches*/,
                                const MaterialSet* /*matls*/) const
{
  // Currently do nothing
}

void Crack::CrackFrontNodeSubset(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* /*matls*/,
                      DataWarehouse* /*old_dw*/,
                      DataWarehouse* /*new_dw*/)
{

  for(int p=0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      if(d_calFractParameters!="false"||d_doCrackPropagation!="false") {
        // cfnset -- subset of crack-front nodes for each patch
        cfnset[m][pid].clear();
        for(int j=0; j<(int)cfSegNodes[m].size(); j++) {
          int node=cfSegNodes[m][j];
          if(patch->containsPoint(cx[m][node]))
            cfnset[m][pid].push_back(j);
        }
        MPI_Barrier(mpi_crack_comm);

        // Broadcast cfnset of each patch to all ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front nodes in patch i
          if(i==pid) num=cfnset[m][i].size();
          MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
          cfnset[m][i].resize(num);
          MPI_Bcast(&cfnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
        }
        MPI_Barrier(mpi_crack_comm);

        // cfsset -- subset of crack-front seg center for each patch
        cfsset[m][pid].clear();
        int ncfSegs=(int)cfSegNodes[m].size()/2;
        for(int j=0; j<ncfSegs; j++) {
          int n1=cfSegNodes[m][2*j]; 
          int n2=cfSegNodes[m][2*j+1];
          Point cent=cx[m][n1]+(cx[m][n2]-cx[m][n1])/2.;
          if(patch->containsPoint(cent)) {
            cfsset[m][pid].push_back(j); 
          } 
        } // End of loop over j
        MPI_Barrier(mpi_crack_comm);
        
        // Broadcast cfsset of each patch to all ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front segs in patch i
          if(i==pid) num=cfsset[m][i].size();
          MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
          cfsset[m][i].resize(num);
          MPI_Bcast(&cfsset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
        }
      } // End if(...)
    } // End of loop over matls
  } // End of loop over patches
}

void Crack::addComputesAndRequiresRecollectCrackFrontSegments(Task* t,
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

void Crack::RecollectCrackFrontSegments(const ProcessorGroup*,
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

      if(d_doCrackPropagation!="false") {
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
        int*  copyData = new int[old_size];

        for(int i=0; i<old_size; i++) copyData[i]=cfSegNodes[m][i];
        cfSegNodes[m].clear();

        // Collect the active crack-front segs
        for(int i=0; i<old_size/2; i++) { // Loop over crack-front segs
          int nd1=copyData[2*i];
          int nd2=copyData[2*i+1];

          short thisSegActive=NO;
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
          }
          //else { // The segment is dead
          //  if(pid==0) {
          //    cout << "   ! Crack-front seg " << i << "(" << nd1
          //         << cx[m][nd1] << "-->" << nd2 << cx[m][nd2]
          //         << ") of Mat " << m << " is dead." << endl;
          //  }
          //}
        } // End of loop over crack-front segs
        delete [] copyData;

        // If all crack-front segs dead, the material is broken.
        if(cfSegNodes[m].size()/2<=0 && pid==0) {
         cout << "   !!! Material " << m
              << " is broken. Program terminated." << endl;
         exit(1);
        }
  
        // Seek the start crack point (sIdx), re-arrange crack-front nodes 
        int node0=cfSegNodes[m][0];
        int segs[2];
        FindSegsFromNode(m,node0,segs);

        if(segs[R]>=0) {
          int numNodes=(int)cfSegNodes[m].size();
          int sIdx=0;
          for(int i=0; i<numNodes; i++) {
            int segsT[2];
            FindSegsFromNode(m,cfSegNodes[m][i],segsT);
            if(segsT[R]<0) {sIdx=i; break;}
          }

          int* copyData =new int[numNodes];
          int rIdx=2*segs[R]+1;
          for(int i=0; i<numNodes; i++) {
            int oldIdx=sIdx+i;
            if(oldIdx>rIdx) oldIdx-=(rIdx+1);
            copyData[i]=cfSegNodes[m][oldIdx];
          }

          for(int i=0; i<numNodes; i++) {
            cfSegNodes[m][i]=copyData[i];
          }      

          delete [] copyData;
        }

        /* Task 3: Get previous index, and minimum & maximum indexes 
           for crack-front nodes
        */
        FindCrackFrontNodeIndexes(m);
    
        /* Task 4: Calculate normals, tangential normals and binormals
        */         
        if(smoothCrackFront) { // Smooth crack front with cubic-spline
          short smoothSuccessfully=SmoothCrackFrontAndCalculateNormals(m);
          if(!smoothSuccessfully) {
          //  if(pid==0)
          //    cout << " ! Crack-front normals are obtained "
          //         << "by raw crack-front points." << endl;
          }
        }
        else { // Calculate crack-front normals directly
          CalculateCrackFrontNormals(m);
        }
   
      } // End of if(d_doCrackPropagation!="false")

      // Output crack elems, crack points and crack-front nodes
      // visualization
      if(doCrackVisualization) {
        int curTimeStep=d_sharedState->getCurrentTopLevelTimeStep();
        if(pid==0) OutputCrackGeometry(m,curTimeStep);
      }

    } // End of loop over matls
  } // End of loop over patches
}

