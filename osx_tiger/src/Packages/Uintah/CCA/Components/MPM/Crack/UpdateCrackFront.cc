/********************************************************************************
    Crack.cc
    PART SEVEN: UPDATE CRACK FRONT AND CALCULATE CRACK-FRONT NORMALS

    Created by Yajun Guo in 2002-2004.
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
        // cfnset - subset of crack-front nodes for each patch and each mat
        cfnset[m][pid].clear();
        for(int j=0; j<(int)cfSegNodes[m].size(); j++) {
          int node=cfSegNodes[m][j];
          if(patch->containsPoint(cx[m][node]))
            cfnset[m][pid].push_back(j);
        }
        MPI_Barrier(mpi_crack_comm);

        // Broadcast cfnset to all ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front nodes in patch i
          if(i==pid) num=cfnset[m][i].size();
          MPI_Bcast(&num,1,MPI_INT,i,mpi_crack_comm);
          cfnset[m][i].resize(num);
          MPI_Bcast(&cfnset[m][i][0],num,MPI_INT,i,mpi_crack_comm);
        }
        MPI_Barrier(mpi_crack_comm);

        // cfsset - subset of crack-front segment center
	//  for each patch and each mat
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
        
        // Broadcast cfsset to all ranks
        for(int i=0; i<patch_size; i++) {
          int num; // number of crack-front segments in patch i
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

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());

    Vector dx = patch->dCell();

    int pid,patch_size;
    MPI_Comm_rank(mpi_crack_comm, &pid);
    MPI_Comm_size(mpi_crack_comm, &patch_size);

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m=0; m<numMPMMatls; m++) {
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);

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
      old_dw->get(psize, lb->pSizeLabel, pset);

      if(d_doCrackPropagation!="false") {

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

          if(pid==i) { // Processor i does it
            for(int j=0; j<num; j++) {
              int idx=cfnset[m][i][j];
              int node=cfSegNodes[m][idx];
              Point pt=cx[m][node];
              inMat[j]=YES;

              // Get the node indices that surround the cell
	      interpolator->findCellAndWeights(pt, ni, S, psize[j]);

              for(int k = 0; k < n8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<d_cell_mass/32.) {
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

        /* Task 2: Detect if the centers of crack-front segments
	           are inside of materials
        */
        vector<short> cfSegCenterInMat;
        cfSegCenterInMat.resize(cfSegNodes[m].size()/2);
        for(int i=0; i<(int)cfSegNodes[m].size()/2;i++) {
          cfSegCenterInMat[i]=YES;
        }

        for(int i=0; i<patch_size; i++) {
          int num=cfsset[m][i].size();
          short* inMat=new short[num];

          if(pid==i) { // Processor i does it
            for(int j=0; j<num; j++) {
              int idx=cfsset[m][i][j];
              int node1=cfSegNodes[m][2*idx];
	      int node2=cfSegNodes[m][2*idx+1];
              Point cent=cx[m][node1]+(cx[m][node2]-cx[m][node1])/2.;
              inMat[j]=YES;

              // Get the node indices that surround the cell
	      interpolator->findCellAndWeights(cent, ni, S, psize[j]);

              for(int k = 0; k < n8or27; k++) {
                double totalMass=gmass[ni[k]]+Gmass[ni[k]];
                if(totalMass<d_cell_mass/32.) {
                  inMat[j]=NO;
                  break;
                }
              }
            } // End of loop over j
          } // End of if(pid==i)

          MPI_Bcast(&inMat[0],num,MPI_SHORT,i,mpi_crack_comm);

          for(int j=0; j<num; j++) {
            int idx=cfsset[m][i][j];
            cfSegCenterInMat[idx]=inMat[j];
          }
          delete [] inMat;
        } // End of loop over patches

        MPI_Barrier(mpi_crack_comm);

        /* Task 3: Recollect crack-front segments, discarding the 
	           dead ones. A segment is regarded dead if both
		   ends of it are outside of the material 
        */
        // Store crack-front parameters in temporary arraies
        int old_size=(int)cfSegNodes[m].size();
        int*  copyData = new int[old_size];

        for(int i=0; i<old_size; i++) copyData[i]=cfSegNodes[m][i];
        cfSegNodes[m].clear();

        // Collect the active crack-front segments
        for(int i=0; i<old_size/2; i++) { // Loop over crack-front segs
          int nd1=copyData[2*i];
          int nd2=copyData[2*i+1];

          short thisSegActive=NO;
	  if(cfSegNodesInMat[2*i] || cfSegNodesInMat[2*i+1] || cfSegCenterInMat[i])
		  thisSegActive=YES;
	  
          if(thisSegActive) { // Collect parameters of the node
            cfSegNodes[m].push_back(nd1);
            cfSegNodes[m].push_back(nd2);
          }

        } // End of loop over crack-front segs
        delete [] copyData;

	if(cfSegNodes[m].size()>0) { // New crack front is still in material
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
       
          /* Task 4: Get previous index, and minimum & maximum indexes 
             for crack-front nodes
          */
          FindCrackFrontNodeIndexes(m);
    
          /* Task 5: Calculate outer normals, tangential normals and binormals
	             of crack plane at crack-front nodes  
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
        } // End if(cfSegNodes[m].size()>0)
	
	else { // Crack has penetrated the material
          // If all crack-front segments dead, the material is broken.
          if(ce[m].size()>0) { // The material has crack(s)		
	    if(pid==0) cout << "!!! Material " << m << " is broken." << endl;
	  }  
	}
	
      } // End of if(d_doCrackPropagation!="false")

      // Save crack elems, crack points and crack-front nodes
      // for crack geometry visualization
      if(saveCrackGeometry) {
        int curTimeStep=d_sharedState->getCurrentTopLevelTimeStep();
        if(pid==0) OutputCrackGeometry(m,curTimeStep);
      }

    } // End of loop over matls
    delete interpolator;
  } // End of loop over patches
}

